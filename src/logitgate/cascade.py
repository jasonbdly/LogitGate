from __future__ import annotations

import time
from collections.abc import Generator, Sequence
from typing import Any

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
      verdict stays FAILED. The judge adds the diagnostic but can't overturn the failure.

    Inner gates run with ``save_on_fail=False``. On failure the cascade writes one snapshot
    containing every layer's trace.

    Args:
        gates: Ordered sequence of gate instances to execute sequentially.
        name: Identifier for this cascade pipeline.

    Raises:
        ConfigurationError: If gates sequence is empty.
    """

    def __init__(self, gates: Sequence[BaseGate], name: str = "Cascade") -> None:
        if not gates:
            raise ConfigurationError("Cascade requires at least one gate")
        self.gates = list(gates)
        self.name = name

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
                if (
                    upstream is not None
                    and upstream.status == DecisionStatus.FAILED
                    and policy == "on_ambiguous_or_fail"
                    and not can_override
                ):
                    if verdict.passed:
                        overrode_judge_pass = True
                    passed, status = False, DecisionStatus.FAILED
                fields = _verdict_fields(
                    verdict,
                    passed=passed,
                    status=status,
                    probability=(
                        upstream.probability
                        if (upstream is not None and upstream.probability is not None)
                        else verdict.probability
                    ),
                    score=(
                        (upstream.score if (upstream is not None and upstream.score is not None) else 0.0)
                        if overrode_judge_pass
                        else (
                            upstream.score
                            if (upstream is not None and upstream.score is not None)
                            else verdict.score
                        )
                    ),
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
        gate_config: dict[str, Any] = {"cascade_gates": [g.name for g in self.gates]}
        replay = replay_cfg or next((cfg for g in self.gates if (cfg := g.replay_config()) is not None), None)
        if replay is not None:
            gate_config.update(replay)
        else:
            from logitgate.backends.base import infer_backend_spec

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
