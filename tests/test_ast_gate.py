from __future__ import annotations

from logitgate.gates.ast_gate import ASTGate
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
