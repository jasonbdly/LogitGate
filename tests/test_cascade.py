from __future__ import annotations

import pytest
from logitgate.backends.base import StubBackend
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.models import DecisionStatus


def test_cascade_short_circuit_layer_0():
    """Layer 0 failure short-circuits before any LLM calls."""
    ast_gate = ASTGate(syntax=True)
    logit_backend = StubBackend(default_prob=0.99)
    logit_gate = LogitGate(rubric="Docstring required", backend=logit_backend)
    judge_backend = StubBackend(diagnostic_text="Should never be called")
    judge_gate = JudgeGate(backend=judge_backend)

    cascade = ast_gate >> logit_gate >> judge_gate

    # Invalid Python syntax
    verdict = cascade.evaluate("def broken_func(")

    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.triggered_layer == "ASTGate"
    assert verdict.total_tokens_generated == 0
    assert len(logit_backend.prompt_log) == 0  # No LLM calls
    assert len(judge_backend.prompt_log) == 0  # No LLM calls
    assert len(verdict.traces) == 1


def test_cascade_short_circuit_layer_1_pass():
    """Layer 1 pass short-circuits before JudgeGate (Layer 2)."""
    ast_gate = ASTGate(syntax=True)
    logit_backend = StubBackend(default_prob=0.95)
    logit_gate = LogitGate(rubric="Safe code", backend=logit_backend, pass_threshold=0.85)
    judge_backend = StubBackend(diagnostic_text="Should never be called", tokens_generated=100)
    judge_gate = JudgeGate(backend=judge_backend)

    cascade = ast_gate >> logit_gate >> judge_gate

    valid_code = "def hello():\n    return 'world'\n"
    verdict = cascade.evaluate(valid_code)

    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED
    assert verdict.triggered_layer == "LogitGate"
    assert verdict.total_tokens_generated == 0  # Zero tokens generated!
    assert len(judge_backend.prompt_log) == 0  # JudgeGate never invoked
    assert len(verdict.traces) == 2  # ASTGate + LogitGate traces


def test_cascade_decisive_fail_on_ambiguous_only():
    """
    When JudgeGate has trigger='on_ambiguous_only', a decisive failure at
    Layer 1 (prob <= fail_threshold) short-circuits without calling JudgeGate.
    """
    logit_backend = StubBackend(default_prob=0.05)  # <= 0.20
    logit_gate = LogitGate(rubric="Clean code", backend=logit_backend, pass_threshold=0.85, fail_threshold=0.20)
    judge_backend = StubBackend(diagnostic_text="Should not be called")
    judge_gate = JudgeGate(backend=judge_backend, trigger="on_ambiguous_only")

    cascade = logit_gate >> judge_gate

    verdict = cascade.evaluate("Some code")
    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.triggered_layer == "LogitGate"
    assert len(judge_backend.prompt_log) == 0  # Short-circuited!


def test_cascade_escalate_to_layer_2():
    """
    When Layer 1 is ambiguous (prob=0.50), it escalates to Layer 2 JudgeGate
    which provides diagnostic critique.
    """
    logit_backend = StubBackend(default_prob=0.50)  # Ambiguous band
    logit_gate = LogitGate(rubric="Must have docstring", backend=logit_backend, pass_threshold=0.85, fail_threshold=0.20)
    judge_backend = StubBackend(
        diagnostic_text="[VERDICT: FAIL] Missing function docstring explaining parameters.",
        tokens_generated=42,
    )
    judge_gate = JudgeGate(backend=judge_backend, trigger="on_ambiguous_or_fail")

    cascade = logit_gate >> judge_gate

    verdict = cascade.evaluate("def compute(x): return x * 2")
    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.triggered_layer == "JudgeGate"
    assert verdict.total_tokens_generated == 42
    assert "Missing function docstring" in verdict.reason
    assert len(judge_backend.prompt_log) == 1
    # Check that LogitGate's rubric was passed to JudgeGate
    assert "Must have docstring" in judge_backend.prompt_log[0]
    assert len(verdict.traces) == 2


@pytest.mark.asyncio
async def test_cascade_async_evaluation():
    """Tests async aevaluate() through all cascade conditions."""
    ast_gate = ASTGate(syntax=True)
    logit_backend = StubBackend(default_prob=0.92)
    logit_gate = LogitGate(rubric="Valid Python code", backend=logit_backend)
    judge_gate = JudgeGate(backend=StubBackend())

    cascade = ast_gate >> logit_gate >> judge_gate

    verdict = await cascade.aevaluate("x = 10\ny = 20\n")
    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED
    assert verdict.triggered_layer == "LogitGate"
