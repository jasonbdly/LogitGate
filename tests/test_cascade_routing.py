"""Cascade routing via gate traits (no isinstance), and sync/async pipeline parity."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from logitgate import (
    ASTGate,
    BaseGate,
    Cascade,
    ChoiceGate,
    DecisionStatus,
    JudgeGate,
    LayerTrace,
    LogitGate,
    ScoreGate,
    StubBackend,
    Verdict,
)
from logitgate.gates.base import TIER_DETERMINISTIC, TIER_GENERATIVE, TIER_LOGIT


def _verdict(name: str, status: DecisionStatus, prob: float | None = None, tokens: int = 0) -> Verdict:
    return Verdict(
        passed=status == DecisionStatus.PASSED,
        status=status,
        probability=prob,
        triggered_layer=name,
        total_latency_ms=0.1,
        total_tokens_generated=tokens,
        reason=f"{name} said {status.value}",
        traces=[LayerTrace(layer_name=name, status=status, latency_ms=0.1, calibrated_prob=prob, tokens_generated=tokens)],
    )


class ScriptedGate(BaseGate):
    """A third-party gate that is *not* a subclass of any built-in gate."""

    def __init__(
        self,
        status: DecisionStatus,
        *,
        name: str = "Scripted",
        tier: int = TIER_LOGIT,
        fast_path: bool = True,
        policy: str = "on_ambiguous_or_fail",
        prob: float | None = 0.5,
        tokens: int = 0,
        rubric: str | None = None,
    ) -> None:
        self.status = status
        self.name = name
        self.layer_tier = tier
        self.can_fast_path = fast_path
        self.escalation_policy = policy  # type: ignore[assignment]
        self.prob = prob
        self.tokens = tokens
        if rubric is not None:
            self.rubric = rubric
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        self.calls.append(kwargs)
        return _verdict(self.name, self.status, self.prob, self.tokens)


class TracingWrapper(BaseGate):
    """Decorator-style gate wrapping a LogitGate. The old isinstance routing treated this as 'generic'."""

    def __init__(self, inner: LogitGate) -> None:
        self.inner = inner
        self.name = f"Traced[{inner.name}]"
        self.rubric = inner.rubric

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return self.inner.evaluate(candidate, **kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return await self.inner.aevaluate(candidate, **kwargs)


def _judge(text: str = "[VERDICT: PASS] fine", trigger: str = "on_ambiguous_or_fail", tokens: int = 40) -> tuple[JudgeGate, StubBackend]:
    backend = StubBackend(diagnostic_text=text, tokens_generated=tokens)
    return JudgeGate(backend=backend, trigger=trigger), backend  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------
# Built-in trait declarations
# ----------------------------------------------------------------------------------------------


def test_builtin_gates_declare_routing_traits() -> None:
    stub = StubBackend()
    assert (ASTGate.layer_tier, ASTGate.can_fast_path) == (TIER_DETERMINISTIC, False)
    for gate in (
        LogitGate("r", backend=stub),
        ScoreGate("r", backend=stub),
        ChoiceGate("q", {"A": "x", "B": "y"}, backend=stub),
    ):
        assert gate.layer_tier == TIER_LOGIT
        assert gate.can_fast_path is True
    judge = JudgeGate(backend=stub, trigger="on_ambiguous_only")
    assert judge.layer_tier == TIER_GENERATIVE
    assert judge.escalation_policy == "on_ambiguous_only"


def test_judge_trigger_is_an_alias_for_escalation_policy() -> None:
    judge, _ = _judge()
    judge.trigger = "always"
    assert judge.escalation_policy == "always"
    judge.escalation_policy = "on_ambiguous_only"
    assert judge.trigger == "on_ambiguous_only"


# ----------------------------------------------------------------------------------------------
# Custom gates participate in fast-path routing
# ----------------------------------------------------------------------------------------------


def test_custom_fast_path_gate_short_circuits_judge() -> None:
    judge, judge_backend = _judge()
    verdict = (ScriptedGate(DecisionStatus.PASSED, prob=0.97) >> judge).evaluate("x")

    assert verdict.passed is True
    assert verdict.triggered_layer == "Scripted"
    assert verdict.total_tokens_generated == 0
    assert judge_backend.prompt_log == []
    assert "Fast-path pass at Scripted" in (verdict.reason or "")


def test_custom_gate_without_fast_path_still_escalates_on_pass() -> None:
    judge, judge_backend = _judge()
    verdict = (ScriptedGate(DecisionStatus.PASSED, fast_path=False) >> judge).evaluate("x")

    assert verdict.passed is True
    assert verdict.triggered_layer == "JudgeGate"
    assert len(judge_backend.prompt_log) == 1


def test_custom_generative_gate_is_skipped_and_receives_inherited_rubric() -> None:
    critic_pass = ScriptedGate(DecisionStatus.FAILED, name="Critic", tier=TIER_GENERATIVE, tokens=99)
    upstream_pass = LogitGate("Upstream rubric", backend=StubBackend(default_prob=0.99))
    assert (upstream_pass >> critic_pass).evaluate("x").passed is True
    assert critic_pass.calls == []

    critic = ScriptedGate(DecisionStatus.PASSED, name="Critic", tier=TIER_GENERATIVE, tokens=12)
    upstream_ambiguous = LogitGate("Upstream rubric", backend=StubBackend(default_prob=0.5))
    verdict = (upstream_ambiguous >> critic).evaluate("x")
    assert verdict.passed is True
    assert verdict.triggered_layer == "Critic"
    assert verdict.probability == pytest.approx(0.5)
    assert critic.calls[0]["rubric_override"] == "Upstream rubric"


def test_custom_deterministic_failure_is_final() -> None:
    judge, judge_backend = _judge()
    verdict = (ScriptedGate(DecisionStatus.FAILED, tier=TIER_DETERMINISTIC, fast_path=False) >> judge).evaluate("x")

    assert verdict.status == DecisionStatus.FAILED
    assert verdict.triggered_layer == "Scripted"
    assert judge_backend.prompt_log == []


def test_wrapped_logit_gate_escalates_ambiguity_under_on_ambiguous_only() -> None:
    # Before the trait refactor, wrappers fell into the "generic" branch, which treated
    # AMBIGUOUS as a decisive failure under on_ambiguous_only and never called the judge.
    wrapped = TracingWrapper(LogitGate("Must be polite", backend=StubBackend(default_prob=0.5)))
    judge, judge_backend = _judge("[VERDICT: PASS] polite", trigger="on_ambiguous_only")

    verdict = (wrapped >> judge).evaluate("Thanks!")
    assert verdict.passed is True
    assert len(judge_backend.prompt_log) == 1
    assert "Must be polite" in judge_backend.prompt_log[0]


# ----------------------------------------------------------------------------------------------
# Escalation policies
# ----------------------------------------------------------------------------------------------


def test_decisive_failure_skips_judge_under_on_ambiguous_only() -> None:
    judge, judge_backend = _judge(trigger="on_ambiguous_only")
    verdict = (ScriptedGate(DecisionStatus.FAILED, prob=0.03) >> judge).evaluate("x")

    assert verdict.status == DecisionStatus.FAILED
    assert verdict.triggered_layer == "Scripted"
    assert verdict.probability == pytest.approx(0.03)
    assert judge_backend.prompt_log == []
    assert "Decisive failure at Scripted" in (verdict.reason or "")


def test_decisive_failure_is_preserved_when_judge_is_lenient() -> None:
    judge, judge_backend = _judge("[VERDICT: PASS] looks fine to me", trigger="on_ambiguous_or_fail")
    verdict = (ScriptedGate(DecisionStatus.FAILED, prob=0.04) >> judge).evaluate("x")

    assert len(judge_backend.prompt_log) == 1  # judge ran for the diagnostic...
    assert verdict.passed is False  # ...but cannot overturn a decisive upstream failure
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.total_tokens_generated == 40


def test_always_policy_runs_judge_even_after_confident_pass() -> None:
    judge, judge_backend = _judge("[VERDICT: FAIL] subtle defect", trigger="always")
    verdict = (LogitGate("r", backend=StubBackend(default_prob=0.99)) >> judge).evaluate("x")

    assert len(judge_backend.prompt_log) == 1
    assert verdict.passed is False
    assert verdict.triggered_layer == "JudgeGate"


def test_judge_inherits_rubric_from_score_gate() -> None:
    score = ScoreGate("Rate the docstring quality", backend=StubBackend(logprob_responses=[[{"token": "1", "logprob": 0.0}]]))
    judge, judge_backend = _judge("[VERDICT: FAIL] no docstring")
    verdict = (score >> judge).evaluate("def f(): pass")

    assert verdict.passed is False
    assert "Rate the docstring quality" in judge_backend.prompt_log[0]


def test_choice_gate_metadata_survives_fast_path() -> None:
    router = ChoiceGate(
        "Route",
        {"A": "billing", "B": "tech"},
        backend=StubBackend(logprob_responses=[[{"token": "A", "logprob": -0.01}, {"token": "B", "logprob": -5.0}]]),
        symmetric_debiasing=False,
    )
    judge, judge_backend = _judge()
    verdict = (router >> judge).evaluate("charged twice")
    assert verdict.choice == "A"
    assert verdict.choice_probabilities is not None
    assert judge_backend.prompt_log == []


def test_judge_skipped_mid_pipeline_does_not_stop_later_gates() -> None:
    judge, judge_backend = _judge()
    tail = ScriptedGate(DecisionStatus.FAILED, name="Tail", prob=0.1)
    verdict = Cascade([ScriptedGate(DecisionStatus.PASSED, prob=0.95), judge, tail]).evaluate("x")

    assert judge_backend.prompt_log == []
    assert len(tail.calls) == 1
    assert verdict.triggered_layer == "Tail"
    assert verdict.passed is False


# ----------------------------------------------------------------------------------------------
# Sync / async parity: one pipeline, two drivers
# ----------------------------------------------------------------------------------------------


def _scenarios() -> dict[str, Any]:
    def build(prob: float, trigger: str, judge_text: str, ast: bool = False) -> tuple[Cascade, StubBackend]:
        judge_backend = StubBackend(diagnostic_text=judge_text, tokens_generated=33)
        gates: list[BaseGate] = [ASTGate(syntax=True)] if ast else []
        gates += [
            LogitGate("Has docstring", backend=StubBackend(default_prob=prob)),
            JudgeGate(backend=judge_backend, trigger=trigger),  # type: ignore[arg-type]
        ]
        return Cascade(gates), judge_backend

    return {
        "fast_pass": lambda: build(0.97, "on_ambiguous_or_fail", "[VERDICT: FAIL]"),
        "ambiguous_judge_pass": lambda: build(0.5, "on_ambiguous_only", "[VERDICT: PASS]"),
        "ambiguous_judge_fail": lambda: build(0.5, "on_ambiguous_or_fail", "[VERDICT: FAIL]"),
        "decisive_fail_skip": lambda: build(0.02, "on_ambiguous_only", "[VERDICT: PASS]"),
        "decisive_fail_preserved": lambda: build(0.02, "on_ambiguous_or_fail", "[VERDICT: PASS]"),
        "always_after_pass": lambda: build(0.97, "always", "[VERDICT: FAIL]"),
        "ast_fail": lambda: build(0.97, "always", "[VERDICT: PASS]", ast=True),
    }


@pytest.mark.parametrize("scenario", sorted(_scenarios()))
def test_sync_and_async_pipelines_agree(scenario: str) -> None:
    candidate = "def broken(" if scenario == "ast_fail" else "def f():\n    return 1\n"
    sync_cascade, sync_judge = _scenarios()[scenario]()
    async_cascade, async_judge = _scenarios()[scenario]()

    v_sync = sync_cascade.evaluate(candidate)
    v_async = asyncio.run(async_cascade.aevaluate(candidate))

    def shape(v: Verdict) -> tuple[Any, ...]:
        return (v.passed, v.status, v.triggered_layer, v.probability, v.total_tokens_generated, [t.layer_name for t in v.traces])

    assert shape(v_sync) == shape(v_async)
    assert len(sync_judge.prompt_log) == len(async_judge.prompt_log)


def test_cascade_writes_single_snapshot_with_full_trace_and_replay_config(tmp_path: Any) -> None:
    from logitgate.snapshot import VerdictSnapshot

    logit = LogitGate("Must cite sources", backend=StubBackend(default_prob=0.5), save_on_fail=True)
    judge, _ = _judge("[VERDICT: FAIL] no citations")
    verdict = (logit >> judge).evaluate("Trust me.", save_on_fail=True, snapshot_dir=str(tmp_path))

    files = sorted(p.name for p in tmp_path.glob("fail_*.json"))
    assert len(files) == 1, "inner gates must not write their own snapshots"
    snap = VerdictSnapshot.load(tmp_path)
    assert [t.layer_name for t in snap.verdict.traces] == ["LogitGate", "JudgeGate"]
    assert snap.gate_config["rubric"] == "Must cite sources"
    assert snap.gate_config["cascade_gates"] == ["LogitGate", "JudgeGate"]
    assert verdict.traces[-1].metadata["snapshot_path"].endswith(files[0])
