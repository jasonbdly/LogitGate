from __future__ import annotations

from logitgate.gates.ast_gate import ASTFailureCode, ASTGate
from logitgate.models import DecisionStatus


def test_syntax_check():
    gate = ASTGate(syntax=True)

    valid_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    v_pass = gate.evaluate(valid_code)
    assert v_pass.passed is True
    assert v_pass.status == DecisionStatus.PASSED
    assert bool(v_pass) is True

    invalid_code = "def add(a: int, b: int) -> int\n    return a + b\n"
    v_fail = gate.evaluate(invalid_code)
    assert v_fail.passed is False
    assert v_fail.status == DecisionStatus.FAILED
    assert "Syntax error" in v_fail.reason


def test_forbidden_imports():
    gate = ASTGate(forbidden_imports=["subprocess", "os.system"])

    clean_code = "import math\nfrom typing import List\nx = math.sqrt(4)\n"
    assert gate.evaluate(clean_code).passed is True

    dirty_code = "import subprocess\nsubprocess.run(['ls'])\n"
    v_fail = gate.evaluate(dirty_code)
    assert v_fail.passed is False
    assert "Forbidden import detected: 'subprocess'" in v_fail.reason

    # Test regex fallback for non-parsable or partial snippet
    partial_code = "from subprocess import Popen"
    v_fail_from = gate.evaluate(partial_code)
    assert v_fail_from.passed is False
    assert "subprocess" in v_fail_from.reason


def test_forbidden_patterns_and_required_sections():
    gate = ASTGate(
        forbidden_patterns=[r"eval\(.*?\)", r"exec\(.*?\)", r"TODO"],
        required_sections=["## Architecture", "## Security"],
    )

    clean_doc = "## Architecture\nMicroservices\n\n## Security\nTLS 1.3 everywhere."
    assert gate.evaluate(clean_doc).passed is True

    # Missing required section
    missing_doc = "## Architecture\nMicroservices"
    v_missing = gate.evaluate(missing_doc)
    assert v_missing.passed is False
    assert "Missing required section: '## Security'" in v_missing.reason

    # Forbidden pattern present
    forbidden_doc = "## Architecture\nMicroservices\n\n## Security\nTLS.\n# TODO: fix later"
    v_patt = gate.evaluate(forbidden_doc)
    assert v_patt.passed is False
    assert "Forbidden pattern matched" in v_patt.reason


def test_acyclic_dag_check():
    gate = ASTGate(acyclic_dag=True)

    # Valid DAG
    valid_tasks = [
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t1", "t2"]},
    ]
    assert gate.evaluate(valid_tasks).passed is True

    # Missing dependency
    missing_dep_tasks = [
        {"id": "t1", "dependencies": ["t_nonexistent"]},
    ]
    v_missing = gate.evaluate(missing_dep_tasks)
    assert v_missing.passed is False
    assert "Missing dependency" in v_missing.reason

    # Cycle in DAG
    cyclic_tasks = [
        {"id": "t1", "dependencies": ["t2"]},
        {"id": "t2", "dependencies": ["t1"]},
    ]
    v_cycle = gate.evaluate(cyclic_tasks)
    assert v_cycle.passed is False
    assert "Cycle detected" in v_cycle.reason


def test_custom_validators():
    def must_be_even(n: int):
        if n % 2 != 0:
            return False, "Number must be even"
        return True, ""

    def must_be_positive(n: int):
        return n > 0

    gate = ASTGate(validators=[must_be_even, must_be_positive])

    assert gate.evaluate(4).passed is True

    v_odd = gate.evaluate(3)
    assert v_odd.passed is False
    assert "Number must be even" in v_odd.reason

    v_neg = gate.evaluate(-2)
    assert v_neg.passed is False


def test_ast_failure_codes():
    gate = ASTGate(syntax=True, name="TestAST")
    verdict = gate.evaluate("def invalid_syntax(:")
    assert verdict.passed is False
    assert len(verdict.traces) == 1
    trace = verdict.traces[0]
    assert trace.failure_code == ASTFailureCode.SYNTAX_ERROR
    assert trace.metadata.get("failure_code") == ASTFailureCode.SYNTAX_ERROR


def test_ast_forbidden_import_code():
    gate = ASTGate(forbidden_imports=["os", "sys"])
    verdict = gate.evaluate("import os\nprint(1)")
    assert verdict.passed is False
    assert verdict.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT


def test_ast_json_schema_failure_code():
    gate = ASTGate(json_schema={"required": ["name", "age"]})
    verdict = gate.evaluate('{"name": "Alice"}')
    assert verdict.passed is False
    assert verdict.traces[0].failure_code == ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR


def test_all_ast_failure_codes():
    # 1. Max length
    g_len = ASTGate(max_length=5)
    v_len = g_len.evaluate("123456")
    assert v_len.traces[0].failure_code == ASTFailureCode.MAX_LENGTH_VIOLATION

    # 2. Forbidden pattern
    g_pat = ASTGate(forbidden_patterns=[r"\bDROP\s+TABLE\b"])
    v_pat = g_pat.evaluate("DROP TABLE users;")
    assert v_pat.traces[0].failure_code == ASTFailureCode.FORBIDDEN_PATTERN

    # 3. Missing section
    g_sec = ASTGate(required_sections=["## Installation"])
    v_sec = g_sec.evaluate("# Readme with no install section")
    assert v_sec.traces[0].failure_code == ASTFailureCode.MISSING_SECTION

    # 4. DAG cycle
    g_dag = ASTGate(acyclic_dag=True)
    v_dag = g_dag.evaluate([
        {"id": "A", "deps": ["B"]},
        {"id": "B", "deps": ["A"]},
    ])
    assert v_dag.traces[0].failure_code == ASTFailureCode.DAG_CYCLE_DETECTED

    # 5. Custom validator
    def reject_all(x):
        return False, "Custom rejected"

    g_val = ASTGate(validators=[reject_all])
    v_val = g_val.evaluate("any text")
    assert v_val.traces[0].failure_code == ASTFailureCode.CUSTOM_VALIDATOR_FAILED


def test_byo_positional_validators_and_normalization():
    import ast

    from pydantic import BaseModel, TypeAdapter

    from logitgate import normalize_candidate

    class PayloadModel(BaseModel):
        query: str
        limit: int

    # 1. Positional ast.parse (returns ast.Module, raises SyntaxError on bad input)
    py_gate = ASTGate(ast.parse, normalize=True)
    fenced_py = "<think>Let me write python</think>\n```python\ndef greet() -> str:\n    return 'hi'\n```"
    assert py_gate.evaluate(fenced_py).passed is True
    assert py_gate.evaluate("```python\ndef broken(:\n```").passed is False

    # 2. Positional Pydantic BaseModel subclass + prose-wrapped JSON extraction
    pyd_gate = ASTGate(PayloadModel, normalize=True)
    prose_json = (
        "<think>Thinking about schema...</think>\n"
        "Sure! Here is the payload you requested:\n"
        '{"query": "SELECT 1", "limit": 10}\n'
        "Let me know if you need anything else."
    )
    v_pyd = pyd_gate.evaluate(prose_json)
    assert v_pyd.passed is True

    v_pyd_fail = pyd_gate.evaluate('Here is the JSON: {"query": "SELECT 1", "limit": "not_an_int"}')
    assert v_pyd_fail.passed is False
    assert v_pyd_fail.traces[0].failure_code == ASTFailureCode.CUSTOM_VALIDATOR_FAILED

    # 3. Positional Pydantic TypeAdapter
    list_adapter = TypeAdapter(list[int])
    ta_gate = ASTGate(list_adapter, normalize=True)
    assert ta_gate.evaluate("Result: [1, 2, 3]").passed is True
    assert ta_gate.evaluate("Result: ['a', 'b']").passed is False

    # 4. Validator returning None (like jsonschema.validate) or empty dict/list passes
    def returns_none(val: str) -> None:
        if "bad" in val:
            raise ValueError("Contains bad token")

    none_gate = ASTGate(returns_none, lambda _: {})
    assert none_gate.evaluate("good input").passed is True
    v_exc = none_gate.evaluate("bad input")
    assert v_exc.passed is False
    assert "ValueError: Contains bad token" in v_exc.reason

    # 5. Custom normalize callable
    custom_norm_gate = ASTGate(ast.parse, normalize=lambda s: str(s).removeprefix("CODE:"))
    assert custom_norm_gate.evaluate("CODE:x = 1").passed is True
    assert normalize_candidate("<think>t</think>```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_ast_edge_cases_relative_imports_empty_dag_and_shorthand_schema():
    # 1. Relative import `from . import subprocess` caught
    imp_gate = ASTGate(forbidden_imports=["subprocess"])
    assert imp_gate.evaluate("from . import subprocess\n").passed is False

    # 2. Empty tasks list in dict `{"tasks": []}` is a valid empty DAG
    dag_gate = ASTGate(acyclic_dag=True)
    assert dag_gate.evaluate({"tasks": []}).passed is True
    assert dag_gate.evaluate('{"tasks": []}').passed is True

    # 3. Shorthand string type names in json_schema
    shorthand_gate = ASTGate(json_schema={"code": "integer", "flag": "boolean", "name": "string"})
    assert shorthand_gate.evaluate({"code": 200, "flag": True, "name": "ok"}).passed is True
    assert shorthand_gate.evaluate({"code": True, "flag": True, "name": "ok"}).passed is False

    # 4. Positional dict json_schema + citation [1] in preamble before JSON object
    pos_dict_gate = ASTGate({"required": ["status"], "properties": {"status": {"type": "string"}}}, normalize=True)
    citation_prose = 'According to source [1], the payload is: {"status": "healthy"}'
    assert pos_dict_gate.evaluate(citation_prose).passed is True

    # 5. Fence info strings with special characters (e.g. ```c# or ```sql postgres)
    from logitgate import normalize_candidate
    assert normalize_candidate("```c#\nint x = 1;\n```") == "int x = 1;"
    assert normalize_candidate("```sql postgres\nSELECT 1;\n```") == "SELECT 1;"

    # 6. Prose-wrapped JSON array of objects preserves outer list (not inner dict)
    dag_norm_gate = ASTGate(acyclic_dag=True, normalize=True)
    prose_dag = 'Here is the task list:\n[{"id": "1", "deps": []}, {"id": "2", "deps": ["1"]}]\nDone!'
    assert dag_norm_gate.evaluate(prose_dag).passed is True

    # 7. Markdown with required_sections + embedded code fence + normalize=True
    md_gate = ASTGate(required_sections=["## Architecture"], syntax=True, normalize=True)
    md_doc = "## Architecture\nHere is the implementation:\n```python\ndef run():\n    return 1\n```"
    assert md_gate.evaluate(md_doc).passed is True

    # 8. Semicolon-separated and `from pkg import sub` fallback when syntax is broken
    fallback_imp_gate = ASTGate(forbidden_imports=["subprocess", "os.path"])
    assert fallback_imp_gate.evaluate("x = 1; import subprocess\ndef broken(:").passed is False
    assert fallback_imp_gate.evaluate("from os import path\ndef broken(:").passed is False
    assert fallback_imp_gate.evaluate("from . import subprocess\ndef broken(:").passed is False

    # 9. Top-level JSON array must not be peeled into an inner dict when schema requires a JSON object
    obj_gate = ASTGate(json_schema={"required": ["id"]}, normalize=True)
    assert obj_gate.evaluate('[{"id": "1"}]').passed is False
    assert obj_gate.evaluate('[{"id": "1"}]').traces[0].failure_code == ASTFailureCode.JSON_TYPE_ERROR

    # 10. Citation [1] before JSON array of task dicts does not shadow the DAG task list
    citation_dag = 'According to [1], the tasks are: [{"id": "1", "deps": []}, {"id": "2", "deps": ["1"]}]'
    assert dag_norm_gate.evaluate(citation_dag).passed is True

    # 11. Union types ["string", "null"] and minimal {"type": "object"} in dict json_schema
    union_gate = ASTGate(
        json_schema={
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": ["string", "null"]}},
        }
    )
    assert union_gate.evaluate({"name": "Alice"}).passed is True
    assert union_gate.evaluate({"name": None}).passed is True
    assert union_gate.evaluate({"name": 123}).passed is False
    assert ASTGate(json_schema={"type": "object"}).evaluate({"any": "key"}).passed is True

    # 12. Cascade forwards normalized candidate from ASTGate(normalize=True) to downstream gates
    from logitgate import LogitGate, StubBackend

    seen_prompts: list[str] = []

    class _RecordingBackend(StubBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens=None, **kwargs):
            seen_prompts.append(prompt)
            return super().query_top_logprobs(prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs)

    cascade = ASTGate(syntax=True, normalize=True) >> LogitGate("Clean code", backend=_RecordingBackend())
    raw_input = "<think>Secret reasoning</think>\n```python\ndef add(a, b):\n    return a + b\n```"
    verdict = cascade.evaluate(raw_input)
    assert verdict.passed is True
    assert seen_prompts
    assert "<think>" not in seen_prompts[0]
    assert "```python" not in seen_prompts[0]
    assert "def add(a, b):" in seen_prompts[0]





