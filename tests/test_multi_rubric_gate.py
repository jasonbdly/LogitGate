from __future__ import annotations

import math
from typing import Any

import pytest

from logitgate.backends.base import LogitBackend, StubBackend
from logitgate.cascade import Cascade
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import TIER_LOGIT
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import MultiRubricGate
from logitgate.models import DecisionStatus
from logitgate.snapshot import VerdictSnapshot


class RubricMockBackend(LogitBackend):
    """Deterministic backend returning specific probabilities per rubric substring."""

    def __init__(self, rubric_probs: dict[str, float], default_prob: float = 0.90) -> None:
        self.rubric_probs = rubric_probs
        self.default_prob = default_prob
        self.prompts: list[str] = []

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.prompts.append(prompt)
        p = self.default_prob
        for r_sub, prob in self.rubric_probs.items():
            if r_sub in prompt:
                p = prob
                break

        is_reverse_pass = (
            "A: The candidate artifact violates" in prompt
            or "A: The content violates" in prompt
            or "A: The candidate does not" in prompt
        )
        p_clamped = max(min(p, 0.9999), 0.0001)
        if is_reverse_pass:
            return [
                {"token": "B", "logprob": math.log(p_clamped)},
                {"token": "A", "logprob": math.log(1.0 - p_clamped)},
            ]
        return [
            {"token": "A", "logprob": math.log(p_clamped)},
            {"token": "B", "logprob": math.log(1.0 - p_clamped)},
        ]


def test_multi_rubric_gate_init_validation() -> None:
    # Empty rubrics
    with pytest.raises(ConfigurationError, match="one or more non-empty rubric"):
        MultiRubricGate([])

    # Empty string or whitespace in rubrics
    with pytest.raises(ConfigurationError, match="one or more non-empty rubric"):
        MultiRubricGate(["Valid rubric", "   "])

    # Non-string item
    with pytest.raises(ConfigurationError, match="one or more non-empty rubric"):
        MultiRubricGate(["Valid rubric", 123])  # type: ignore[list-item]

    # Valid initialization
    rubrics = ["Is it typed?", "Is it documented?", "Is it safe?"]
    gate = MultiRubricGate(rubrics, backend=StubBackend())
    assert gate.rubrics == rubrics
    assert gate.rubric == "- Is it typed?\n- Is it documented?\n- Is it safe?"
    assert "MultiRubricGate(rubrics=3" in repr(gate)
    assert gate.layer_tier == TIER_LOGIT
    assert gate.can_fast_path is True

    replay_cfg = gate.replay_config()
    assert replay_cfg["gate_type"] == "MultiRubricGate"
    assert replay_cfg["rubrics"] == rubrics


def test_multi_rubric_gate_all_pass() -> None:
    backend = RubricMockBackend(
        rubric_probs={
            "Is it typed?": 0.95,
            "Is it documented?": 0.88,
            "Is it safe?": 0.92,
        }
    )
    gate = MultiRubricGate(
        ["Is it typed?", "Is it documented?", "Is it safe?"],
        backend=backend,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    candidate = "def add(x: int, y: int) -> int:\n    '''Add two ints.'''\n    return x + y"
    verdict = gate.evaluate(candidate)

    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED
    assert verdict.probability is not None
    # Weakest-link probability is min(0.95, 0.88, 0.92) = 0.88
    assert math.isclose(verdict.probability, 0.88, abs_tol=1e-3)
    assert verdict.total_tokens_generated == 0
    assert verdict.reason is not None and "3/3 rubrics passed" in verdict.reason

    assert len(verdict.traces) == 3
    assert verdict.traces[0].layer_name == "MultiRubricGate[0]"
    assert verdict.traces[0].metadata["rubric"] == "Is it typed?"
    assert verdict.traces[1].layer_name == "MultiRubricGate[1]"
    assert verdict.traces[1].metadata["rubric"] == "Is it documented?"
    assert verdict.traces[2].layer_name == "MultiRubricGate[2]"
    assert verdict.traces[2].metadata["rubric"] == "Is it safe?"


def test_multi_rubric_gate_weakest_link_failed() -> None:
    backend = RubricMockBackend(
        rubric_probs={
            "Is it typed?": 0.95,  # Pass
            "Is it documented?": 0.50,  # Ambiguous
            "Is it safe?": 0.05,  # Failed
        }
    )
    gate = MultiRubricGate(
        ["Is it typed?", "Is it documented?", "Is it safe?"],
        backend=backend,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    verdict = gate.evaluate("def f(): pass")

    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.probability is not None
    assert math.isclose(verdict.probability, 0.05, abs_tol=1e-3)
    assert verdict.reason is not None
    assert "1/3 rubrics passed" in verdict.reason
    assert "FAILED 'Is it safe?'" in verdict.reason
    assert "AMBIGUOUS 'Is it documented?'" in verdict.reason


def test_multi_rubric_gate_weakest_link_ambiguous() -> None:
    backend = RubricMockBackend(
        rubric_probs={
            "Is it typed?": 0.95,  # Pass
            "Is it documented?": 0.50,  # Ambiguous
        }
    )
    gate = MultiRubricGate(
        ["Is it typed?", "Is it documented?"],
        backend=backend,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    verdict = gate.evaluate("def f(): pass")

    assert verdict.passed is False
    assert verdict.status == DecisionStatus.AMBIGUOUS
    assert verdict.probability is not None
    assert math.isclose(verdict.probability, 0.50, abs_tol=1e-3)
    assert verdict.reason is not None
    assert "1/2 rubrics passed" in verdict.reason
    assert "AMBIGUOUS 'Is it documented?'" in verdict.reason


@pytest.mark.asyncio
async def test_multi_rubric_gate_sync_async_parity() -> None:
    backend_sync = RubricMockBackend(
        rubric_probs={"Is it typed?": 0.92, "Is it safe?": 0.15}
    )
    backend_async = RubricMockBackend(
        rubric_probs={"Is it typed?": 0.92, "Is it safe?": 0.15}
    )

    gate_sync = MultiRubricGate(
        ["Is it typed?", "Is it safe?"],
        backend=backend_sync,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    gate_async = MultiRubricGate(
        ["Is it typed?", "Is it safe?"],
        backend=backend_async,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )

    candidate = "def f(): pass"
    v_sync = gate_sync.evaluate(candidate)
    v_async = await gate_async.aevaluate(candidate)

    assert v_sync.passed == v_async.passed
    assert v_sync.status == v_async.status
    assert v_sync.probability == v_async.probability
    assert v_sync.reason == v_async.reason
    assert len(v_sync.traces) == len(v_async.traces)

    # evaluate_each vs aevaluate_each
    each_sync = gate_sync.evaluate_each(candidate)
    each_async = await gate_async.aevaluate_each(candidate)
    assert len(each_sync) == len(each_async) == 2
    assert each_sync[0].passed == each_async[0].passed
    assert each_sync[1].passed == each_async[1].passed


def test_multi_rubric_gate_cascade_fast_path() -> None:
    judge_backend = StubBackend(tokens_generated=50, diagnostic_text="[VERDICT: PASS] All good.")
    gate = MultiRubricGate(
        ["Is it typed?", "Is it safe?"],
        backend=StubBackend(default_prob=0.95),
        pass_threshold=0.85,
    )
    judge = JudgeGate(backend=judge_backend)
    pipeline = gate >> judge

    verdict = pipeline.evaluate("def add(x: int, y: int) -> int: return x + y")
    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED
    assert verdict.total_tokens_generated == 0  # JudgeGate was skipped!
    assert verdict.reason is not None and "Fast-path pass at MultiRubricGate" in verdict.reason
    assert len(judge_backend.prompt_log) == 0


def test_multi_rubric_gate_cascade_escalation() -> None:
    judge_backend = StubBackend(
        tokens_generated=42,
        diagnostic_text="[VERDICT: FAIL] The function does not handle negative values.",
    )
    gate = MultiRubricGate(
        ["Is it typed?", "Does it handle negative input?"],
        backend=RubricMockBackend({"Is it typed?": 0.95, "Does it handle negative input?": 0.50}),
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    judge = JudgeGate(backend=judge_backend)
    pipeline = Cascade([gate, judge])

    verdict = pipeline.evaluate("def f(x: int) -> int: return x")
    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.total_tokens_generated == 42
    # Verify the judge received the bulleted rubrics
    assert len(judge_backend.prompt_log) == 1
    judge_prompt = judge_backend.prompt_log[0]
    assert "- Is it typed?" in judge_prompt
    assert "- Does it handle negative input?" in judge_prompt


def test_multi_rubric_gate_snapshot_and_replay(tmp_path: Any) -> None:
    backend = RubricMockBackend(
        rubric_probs={"Is it typed?": 0.95, "Is it safe?": 0.10}
    )
    gate = MultiRubricGate(
        ["Is it typed?", "Is it safe?"],
        backend=backend,
        pass_threshold=0.85,
        fail_threshold=0.20,
        save_on_fail=True,
    )
    candidate = "import evil; evil.attack()"
    verdict = gate.evaluate(candidate, snapshot_dir=str(tmp_path), test_name="test_multi_fail")

    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED

    # Check that snapshot was saved
    snapshot_files = list(tmp_path.glob("fail_*.json"))
    assert len(snapshot_files) >= 1

    snapshot = VerdictSnapshot.load(tmp_path)
    assert "- Is it typed?" in snapshot.rubric
    assert "- Is it safe?" in snapshot.rubric
    assert snapshot.gate_config.get("gate_type") == "MultiRubricGate"
    assert snapshot.gate_config.get("rubrics") == ["Is it typed?", "Is it safe?"]

    # Replay snapshot with same backend
    replay_verdict = snapshot.replay(backend=backend)
    assert replay_verdict.passed is False
    assert replay_verdict.status == DecisionStatus.FAILED
    assert math.isclose(replay_verdict.probability or 0.0, 0.10, abs_tol=1e-3)
    assert len(replay_verdict.traces) == 2
