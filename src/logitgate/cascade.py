from __future__ import annotations

import math
import time
from collections.abc import Generator, Sequence
from typing import Any

from logitgate.debiasing.permutation import compute_shannon_entropy
from logitgate.debiasing.softmax import clamped_sigmoid
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import TIER_GENERATIVE, TIER_LOGIT, BaseGate, EscalationPolicy
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

# The pipeline is a generator that yields "please run this gate on this candidate with these kwargs"
# and is sent back the resulting Verdict. All routing lives in `Cascade._pipeline`; the sync and async
# drivers differ only in whether they call `gate.evaluate` or `await gate.aevaluate`.
_GateCall = tuple[BaseGate, Any, dict[str, Any]]
_Pipeline = Generator[_GateCall, Verdict, Verdict]


def _should_escalate(
    policy: EscalationPolicy,
    upstream_gate: BaseGate | None,
    upstream: Verdict | None,
) -> bool:
    """Decides whether a generative gate runs, given the most recent non-generative verdict."""
    if policy == "always" or upstream is None or upstream_gate is None:
        return True
    if upstream.passed:
        # A trusted pass makes generation pure cost. A deterministic pass (schema OK) carries
        # no semantic signal, so those gates opt out of fast-pathing via `can_fast_path=False`.
        return not upstream_gate.can_fast_path
    if upstream.status == DecisionStatus.FAILED:
        return policy in ("on_ambiguous_or_fail", "adjudicate")
    return True  # AMBIGUOUS: this is exactly what the generative layer exists for.


def _verdict_fields(source: Verdict, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "passed": source.passed,
        "status": source.status,
        "probability": source.probability,
        "score": source.score,
        "choice": source.choice,
        "choice_probabilities": source.choice_probabilities,
        "triggered_layer": source.triggered_layer,
        "reason": source.reason,
    }
    fields.update(overrides)
    return fields


def _same_nonzero_sign(a: float, b: float) -> bool:
    """Underflow-safe check that two floats share the same strictly non-zero sign."""
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def _select_upstream_logit_trace(upstream: Verdict | None) -> LayerTrace | None:
    """Selects the weakest non-passing Tier-1 trace carrying full log-odds evidence."""
    if upstream is None or not upstream.traces:
        return None
    with_logits = [
        tr
        for tr in upstream.traces
        if tr.log_odds_pass is not None
        and math.isfinite(tr.log_odds_pass)
        and tr.forward_log_odds is not None
        and math.isfinite(tr.forward_log_odds)
        and tr.reverse_log_odds is not None
        and math.isfinite(tr.reverse_log_odds)
    ]
    if not with_logits:
        return upstream.traces[-1]
    non_passing = [tr for tr in with_logits if tr.status != DecisionStatus.PASSED]
    pool = non_passing or with_logits
    return min(pool, key=lambda tr: float(tr.log_odds_pass))  # type: ignore[arg-type]


def fuse_escalated_verdict(
    upstream: Verdict | None,
    verdict: Verdict,
    *,
    prior_weight: float = 0.30,
) -> tuple[bool, DecisionStatus, float | None, float | None]:
    """Fuses calibrated Layer-1 prior log-odds with permutation-split Layer-2 post-critique log-odds.

    When an ambiguous Tier-1 logit gate escalates to a Tier-2 judge that emits calibrated
    post-critique log-odds, both layers produce log-odds on the ``logit(P)`` scale. Whenever
    Tier 2's forward and reverse passes agree on polarity (same nonzero sign) or Tier 2 is
    decisive (``abs(z2) >= 1.0``), Tier 2's reasoned critique governs directly. When Tier 2 is
    permutation-split and uncertain (``abs(z2) < 1.0``) while Tier 1 was permutation-consistent
    and non-trivial (``abs(z1) >= 1.0``), blending ``(1 - w) * z2 + w * clamp(z1, -2, 2)``
    breaks the tie using Tier 1's debiased semantic signal without overriding decisive critiques.
    """
    if (
        isinstance(prior_weight, bool)
        or not isinstance(prior_weight, (int, float))
        or not math.isfinite(float(prior_weight))
        or not (0.0 <= float(prior_weight) <= 1.0)
    ):
        raise ConfigurationError(f"prior_weight must be between 0.0 and 1.0, got {prior_weight!r}")
    w = float(prior_weight)
    fallback_prob = (
        verdict.probability
        if verdict.probability is not None
        else (upstream.probability if upstream is not None else None)
    )
    if (
        upstream is None
        or upstream.status != DecisionStatus.AMBIGUOUS
        or w <= 0.0
        or not upstream.traces
        or not verdict.traces
    ):
        return verdict.passed, verdict.status, fallback_prob, None

    tr1 = _select_upstream_logit_trace(upstream)
    tr2 = verdict.traces[-1]
    if tr1 is None or (tr1.metadata or {}).get("abstained_reason") or (tr2.metadata or {}).get("abstained_reason"):
        return verdict.passed, verdict.status, fallback_prob, None

    z1 = tr1.log_odds_pass
    z2 = tr2.log_odds_pass
    f1, r1 = tr1.forward_log_odds, tr1.reverse_log_odds
    f2, r2 = tr2.forward_log_odds, tr2.reverse_log_odds
    if (
        z1 is not None
        and z2 is not None
        and f1 is not None
        and r1 is not None
        and f2 is not None
        and r2 is not None
        and math.isfinite(z1)
        and math.isfinite(z2)
        and math.isfinite(f1)
        and math.isfinite(r1)
        and math.isfinite(f2)
        and math.isfinite(r2)
        and abs(z1) >= 1.0
        and abs(z2) < 1.0
        and _same_nonzero_sign(f1, r1)
        and not _same_nonzero_sign(f2, r2)
    ):
        z1_clamped = max(-2.0, min(2.0, float(z1)))
        z_fused = (1.0 - w) * float(z2) + w * z1_clamped
        passed = z_fused > 0.0
        status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
        return passed, status, round(clamped_sigmoid(z_fused), 4), round(z_fused, 4)

    return verdict.passed, verdict.status, fallback_prob, None


class Cascade(BaseGate):
    """Sequential gate pipeline that only pays for expensive layers when cheaper ones are unsure.

    Gates run in order with AND semantics: the first gate that fails decides. Routing comes
    from each gate's traits (`layer_tier`, `can_fast_path`, `escalation_policy`, see
    `BaseGate`) rather than from its class, so custom gates take part in short-circuiting too:

    - A tier-0 (deterministic) failure is final. No LLM is consulted.
    - A tier-1 failure or ambiguity is final unless the *next* gate is generative (tier 2).
      In that case the generative gate's `escalation_policy` decides whether it runs.
    - A tier-1 pass from a gate with ``can_fast_path=True`` skips the following generative
      gate (unless its policy is ``"always"``). This is the path that lets a typical cascade
      finish in one or two forward passes with zero generated tokens.
    - When a decisive upstream FAILED escalates under ``"on_ambiguous_or_fail"``, the final
      verdict stays FAILED unless the generative gate uses ``escalation_policy="adjudicate"``
      (or sets ``can_override_fail=True``).

    Inner gates run with ``save_on_fail=False``. On failure the cascade writes one snapshot
    containing every layer's trace.

    Args:
        gates: Ordered sequence of gate instances to execute sequentially.
        name: Identifier for this cascade pipeline.
        fuse_logits: Whether to fuse permutation-consistent Tier-1 log-odds with an
            uncertain/permutation-split Tier-2 critique. Defaults to ``True``.
        prior_weight: Weight in ``[0.0, 1.0]`` assigned to the clamped Tier-1 prior when
            log-odds fusion triggers. Defaults to ``0.30``.

    Raises:
        ConfigurationError: If gates is empty, contains non-``BaseGate`` elements, or
            parameters are invalid.
    """

    def __init__(
        self,
        gates: Sequence[BaseGate],
        name: str = "Cascade",
        *,
        fuse_logits: bool = True,
        prior_weight: float = 0.30,
    ) -> None:
        if not gates:
            raise ConfigurationError("Cascade requires at least one gate")
        if any(not isinstance(g, BaseGate) for g in gates):
            raise ConfigurationError("Cascade gates must all inherit from BaseGate")
        if not isinstance(fuse_logits, bool):
            raise ConfigurationError(f"fuse_logits must be a bool, got {type(fuse_logits).__name__}")
        if isinstance(prior_weight, bool):
            raise ConfigurationError(f"prior_weight must be between 0.0 and 1.0, got {prior_weight!r}")
        try:
            pw = float(prior_weight)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(f"prior_weight must be a float between 0.0 and 1.0, got {prior_weight!r}") from e
        if not math.isfinite(pw) or not (0.0 <= pw <= 1.0):
            raise ConfigurationError(f"prior_weight must be between 0.0 and 1.0, got {prior_weight!r}")
        self.gates = list(gates)
        self.name = name
        self.fuse_logits = fuse_logits
        self.prior_weight = pw

    def __repr__(self) -> str:
        pipeline_repr = " >> ".join(repr(g) for g in self.gates)
        return f"Cascade([{pipeline_repr}])"

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        pipeline = self._pipeline(candidate, kwargs)
        try:
            request = next(pipeline)
        except StopIteration as done:  # pragma: no cover - the first gate always runs
            return done.value
        while True:
            gate, current_candidate, gate_kwargs = request
            # Evaluate outside the try so a StopIteration leaking from a gate is never
            # mistaken for pipeline completion.
            verdict = gate.evaluate(current_candidate, **gate_kwargs)
            try:
                request = pipeline.send(verdict)
            except StopIteration as done:
                return done.value

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        pipeline = self._pipeline(candidate, kwargs)
        try:
            request = next(pipeline)
        except StopIteration as done:  # pragma: no cover - the first gate always runs
            return done.value
        while True:
            gate, current_candidate, gate_kwargs = request
            verdict = await gate.aevaluate(current_candidate, **gate_kwargs)
            try:
                request = pipeline.send(verdict)
            except StopIteration as done:
                return done.value

    def _pipeline(self, candidate: Any, kwargs: dict[str, Any]) -> _Pipeline:
        t0 = time.perf_counter()
        gate_kwargs = {**kwargs, "save_on_fail": False}
        traces: list[LayerTrace] = []
        tokens = 0
        rubric: str | None = None  # inherited by generative gates without their own rubric
        replay_cfg: dict[str, Any] | None = None  # most recent replay_config from a gate that ran
        upstream_gate: BaseGate | None = None  # most recent non-generative gate that ran
        upstream: Verdict | None = None
        outcome: dict[str, Any] | None = None  # what the cascade returns if nothing else runs

        def conclude(fields: dict[str, Any]) -> Verdict:
            verdict = Verdict(
                **fields,
                total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                total_tokens_generated=tokens,
                traces=traces,
            )
            return self._persist_failure(verdict, candidate, rubric, kwargs, replay_cfg=replay_cfg)

        for i, gate in enumerate(self.gates):
            if gate.layer_tier >= TIER_GENERATIVE:
                gate_rubric = getattr(gate, "rubric", None)
                if rubric is None and isinstance(gate_rubric, str) and gate_rubric:
                    rubric = gate_rubric
                policy = gate.escalation_policy
                if not _should_escalate(policy, upstream_gate, upstream):
                    assert upstream is not None and upstream_gate is not None
                    if upstream.passed:
                        outcome = _verdict_fields(
                            upstream,
                            triggered_layer=upstream_gate.name,
                            reason=f"Fast-path pass at {upstream_gate.name} "
                            f"({gate.name} skipped): {upstream.reason}",
                        )
                        continue
                    return conclude(_verdict_fields(
                        upstream,
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=upstream_gate.name,
                        reason=f"Decisive failure at {upstream_gate.name} ({gate.name} skipped, "
                        f"escalation_policy='{policy}'): {upstream.reason}",
                    ))

                verdict = yield gate, candidate, {
                    **gate_kwargs,
                    "rubric_override": rubric if rubric is not None else gate_kwargs.get("rubric_override"),
                }
                traces.extend(verdict.traces)
                tokens += verdict.total_tokens_generated

                passed, status = verdict.passed, verdict.status
                overrode_judge_pass = False
                can_override = getattr(gate, "can_override_fail", False) or policy == "adjudicate"
                locked_upstream_fail = (
                    upstream is not None
                    and upstream.status == DecisionStatus.FAILED
                    and policy == "on_ambiguous_or_fail"
                    and not can_override
                )
                fused_lo: float | None = None
                tr1_prior_lo: float | None = None
                judge_lo_orig: float | None = None
                if locked_upstream_fail:
                    if verdict.passed:
                        overrode_judge_pass = True
                    passed, status = False, DecisionStatus.FAILED
                    effective_prob = (
                        upstream.probability
                        if (upstream is not None and upstream.probability is not None)
                        else verdict.probability
                    )
                elif self.fuse_logits:
                    passed, status, effective_prob, fused_lo = fuse_escalated_verdict(
                        upstream,
                        verdict,
                        prior_weight=self.prior_weight,
                    )
                    if fused_lo is not None and traces and upstream is not None and upstream.traces:
                        tr1 = _select_upstream_logit_trace(upstream)
                        tr1_prior_lo = tr1.log_odds_pass if tr1 is not None else upstream.traces[-1].log_odds_pass
                        last_tr = traces[-1]
                        judge_lo_orig = last_tr.log_odds_pass
                        fused_entropy = (
                            round(compute_shannon_entropy(effective_prob), 4)
                            if effective_prob is not None
                            else last_tr.shannon_entropy
                        )
                        traces[-1] = last_tr.model_copy(
                            update={
                                "status": status,
                                "calibrated_prob": effective_prob,
                                "log_odds_pass": fused_lo,
                                "shannon_entropy": fused_entropy,
                                "metadata": {
                                    **last_tr.metadata,
                                    "judge_log_odds": judge_lo_orig,
                                    "fused_log_odds": fused_lo,
                                    "prior_log_odds": tr1_prior_lo,
                                    "prior_weight": self.prior_weight,
                                },
                            }
                        )
                else:
                    effective_prob = (
                        verdict.probability
                        if verdict.probability is not None
                        else (upstream.probability if upstream is not None else None)
                    )
                fused_score = (
                    (upstream.score if (upstream is not None and upstream.score is not None) else 0.0)
                    if overrode_judge_pass
                    else (
                        upstream.score
                        if (upstream is not None and upstream.score is not None)
                        else (
                            (effective_prob if effective_prob is not None else (1.0 if passed else 0.0))
                            if (fused_lo is not None and passed != verdict.passed)
                            else verdict.score
                        )
                    )
                )
                fused_reason = verdict.reason
                if fused_lo is not None and passed != verdict.passed:
                    prior_desc = f"{tr1_prior_lo:+.2f}" if tr1_prior_lo is not None else "n/a"
                    judge_desc = f"{judge_lo_orig:+.2f}" if judge_lo_orig is not None else "n/a"
                    fused_reason = (
                        f"Fused Tier-1 prior (z1={prior_desc}) with split Tier-2 critique "
                        f"(z2={judge_desc}, z_fused={fused_lo:+.2f}): {verdict.reason}"
                    )
                fields = _verdict_fields(
                    verdict,
                    passed=passed,
                    status=status,
                    probability=effective_prob,
                    score=fused_score,
                    reason=fused_reason,
                    choice=(
                        verdict.choice
                        if verdict.choice is not None
                        else (upstream.choice if upstream is not None else None)
                    ),
                    choice_probabilities=(
                        verdict.choice_probabilities
                        if verdict.choice_probabilities is not None
                        else (upstream.choice_probabilities if upstream is not None else None)
                    ),
                    triggered_layer=gate.name,
                )
                if not passed:
                    return conclude(fields)
                outcome = fields
                continue

            norm_box: list[Any] = []
            verdict = yield gate, candidate, {**gate_kwargs, "_normalized_out": norm_box}
            if verdict.passed:
                if norm_box:
                    candidate = norm_box[-1]
                elif hasattr(gate, "normalize_input"):
                    candidate = gate.normalize_input(candidate)
            traces.extend(verdict.traces)
            tokens += verdict.total_tokens_generated
            upstream_gate, upstream = gate, verdict
            if (cfg := gate.replay_config()) is not None:
                replay_cfg = cfg
            gate_rubric = getattr(gate, "rubric", None) or getattr(gate, "question", None)
            if isinstance(gate_rubric, str) and gate_rubric:
                rubric = gate_rubric

            fields = _verdict_fields(verdict, triggered_layer=gate.name)
            if verdict.passed:
                outcome = fields
                continue

            next_gate = self.gates[i + 1] if i + 1 < len(self.gates) else None
            defers_to_generative = (
                gate.layer_tier >= TIER_LOGIT
                and next_gate is not None
                and next_gate.layer_tier >= TIER_GENERATIVE
            )
            if not defers_to_generative:
                return conclude(fields)

        if outcome is None:  # pragma: no cover - at least one gate always runs
            raise RuntimeError("Cascade finished without evaluating any gate")
        return conclude(outcome)

    def _persist_failure(
        self,
        verdict: Verdict,
        candidate: Any,
        rubric: str | None,
        kwargs: dict[str, Any],
        replay_cfg: dict[str, Any] | None = None,
    ) -> Verdict:
        if verdict.passed:
            return verdict
        from logitgate.backends.base import infer_backend_spec

        gate_config: dict[str, Any] = {"cascade_gates": [g.name for g in self.gates]}
        replay = replay_cfg or next((cfg for g in self.gates if (cfg := g.replay_config()) is not None), None)
        if replay is not None:
            gate_config.update(replay)
        else:
            for g in reversed(self.gates):
                g_backend = getattr(g, "backend", None)
                if g_backend is not None:
                    gate_config["backend"] = infer_backend_spec(g_backend)
                    g_model = (
                        getattr(g, "model", None)
                        or getattr(g_backend, "model", None)
                        or getattr(g_backend, "model_name", None)
                    )
                    if g_model:
                        gate_config["model"] = g_model
                    break
        for g in self.gates:
            if g.layer_tier >= TIER_GENERATIVE:
                jg_backend = getattr(g, "backend", None)
                if jg_backend is not None:
                    gate_config["judge_backend"] = infer_backend_spec(jg_backend)
                    jg_model = (
                        getattr(g, "model", None)
                        or getattr(jg_backend, "model", None)
                        or getattr(jg_backend, "model_name", None)
                    )
                    if jg_model:
                        gate_config["judge_model"] = jg_model
                break
        effective_save = kwargs.get("save_on_fail")
        if effective_save is None:
            for g in self.gates:
                g_save = getattr(g, "save_on_fail", None)
                if g_save is not None:
                    effective_save = g_save
                    break
        fallback_rubric = next(
            (
                r
                for g in self.gates
                if isinstance(r := (getattr(g, "rubric", None) or getattr(g, "question", None)), str) and r
            ),
            "Cascade evaluation",
        )
        return persist_failure_snapshot(
            verdict,
            candidate=candidate,
            rubric=rubric or fallback_rubric,
            gate_config=gate_config,
            save_on_fail=effective_save,
            test_name=kwargs.get("test_name"),
            snapshot_dir=kwargs.get("snapshot_dir"),
        )
