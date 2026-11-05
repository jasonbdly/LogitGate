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

