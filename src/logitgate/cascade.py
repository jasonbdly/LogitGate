from __future__ import annotations

import math
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import Self

from logitgate.backends.base import LogitBackend, infer_backend_spec
from logitgate.calibration.lifecycle import _same_nonzero_sign
from logitgate.debiasing.permutation import compute_shannon_entropy
from logitgate.debiasing.softmax import clamped_sigmoid
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import TIER_GENERATIVE, TIER_LOGIT, BaseGate, EscalationPolicy
from logitgate.gates.logit_gate import (
    UnsupportedLogprobsPolicy,
    _call_resolve_logit_backend,
    _validate_rag_context_and_query,
)
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


def _has_active_cascade_audit(gates: Sequence[BaseGate]) -> bool:
    for g in gates:
        aci = getattr(g, "adaptive_controller", None)
        if aci is not None and getattr(aci, "feedback_mode", None) == "cascade_audit":
            return True
    return False


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
        gates: Sequence[BaseGate] | None = None,
        name: str = "Cascade",
        *,
        rubric: str | None = None,
        backend: Any = None,
        model: str | None = None,
        judge_backend: Any = None,
        judge_model: str | None = None,
        distillation_bundle: Any = None,
        fuse_logits: bool = True,
        prior_weight: float = 0.30,
        telemetry_sink: Any = None,
        cache: Any = None,
        save_failures: bool | None = None,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
        on_unsupported_logprobs: UnsupportedLogprobsPolicy = "raise",
        sidecar_backend: str | LogitBackend | None = None,
        sidecar_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        from logitgate.cache import resolve_verdict_cache
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        _validate_rag_context_and_query(context, query)
        if on_unsupported_logprobs not in ("raise", "escalate", "sidecar"):
            raise ConfigurationError(
                f"Invalid on_unsupported_logprobs '{on_unsupported_logprobs}'. "
                f"Must be one of ['raise', 'escalate', 'sidecar']"
            )
        if on_unsupported_logprobs == "sidecar" and sidecar_backend is None:
            raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'")

        owns_gates = False
        if gates is None:
            if rubric is not None:
                from logitgate.gates.judge_gate import JudgeGate
                from logitgate.gates.logit_gate import LogitGate

                l1_kw: dict[str, Any] = {
                    "backend": backend,
                    "model": model,
                    "distillation_bundle": distillation_bundle,
                    "context": context,
                    "query": query,
                    "on_unsupported_logprobs": on_unsupported_logprobs,
                    "sidecar_backend": sidecar_backend,
                    "sidecar_model": sidecar_model,
                }
                for k in ("pass_threshold", "fail_threshold", "logit_temperature", "profile"):
                    if k in kwargs:
                        l1_kw[k] = kwargs[k]
                l1 = LogitGate(rubric=rubric, **l1_kw)
                l2_kw: dict[str, Any] = {
                    "backend": judge_backend,
                    "model": judge_model,
                    "context": context,
                    "query": query,
                }
                if "trigger" in kwargs:
                    l2_kw["trigger"] = kwargs["trigger"]
                l2 = JudgeGate(rubric=rubric, **l2_kw)
                resolved_gates: list[BaseGate] = [l1, l2]
                owns_gates = True
            else:
                raise ConfigurationError("Cascade requires at least one gate")
        else:
            resolved_gates = list(gates)

        if not resolved_gates:
            raise ConfigurationError("Cascade requires at least one gate")
        if any(not isinstance(g, BaseGate) for g in resolved_gates):
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
        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        owns_sink = isinstance(telemetry_sink, str) or hasattr(telemetry_sink, "__fspath__")
        owns_sidecar = False
        resolved_sidecar: LogitBackend | None = None
        if sidecar_backend is not None:
            owns_sidecar = not isinstance(sidecar_backend, LogitBackend)
            resolved_sidecar = _call_resolve_logit_backend(sidecar_backend, model=sidecar_model)
        try:
            resolved_cache = resolve_verdict_cache(cache)
        except BaseException:
            if owns_sink and resolved_sink not in (None, False):
                resolved_sink.close()  # type: ignore[union-attr]
            if owns_sidecar and resolved_sidecar is not None:
                resolved_sidecar.close()
            raise
        self.gates = resolved_gates
        self.name = name
        self.fuse_logits = fuse_logits
        self.prior_weight = pw
        self.telemetry_sink = resolved_sink
        self.cache = resolved_cache
        self.save_failures = save_failures
        self.context = context
        self.query = query
        self.on_unsupported_logprobs = on_unsupported_logprobs
        self.sidecar_backend = resolved_sidecar
        self.sidecar_model = sidecar_model
        self._owns_sidecar_backend = owns_sidecar
        self._owns_telemetry_sink = owns_sink
        self._owns_cache = isinstance(cache, (str, Path)) or hasattr(cache, "__fspath__")
        self._owns_gates = owns_gates
        self._closed = False
        self._replay_extra_backends: list[Any] = []
        self.rubric = rubric or next(
            (
                r
                for g in self.gates
                if isinstance(r := (getattr(g, "rubric", None) or getattr(g, "question", None)), str) and r
            ),
            None,
        )

    def close(self) -> None:
        """Closes owned resources including telemetry sink, cache, and child gates."""
        if self._closed:
            return
        self._closed = True
        if (
            self._owns_cache
            and self.cache is not None
            and not isinstance(self.cache, bool)
            and hasattr(self.cache, "close")
            and callable(self.cache.close)
        ):
            self.cache.close()
        if (
            self._owns_telemetry_sink
            and self.telemetry_sink is not None
            and not isinstance(self.telemetry_sink, bool)
            and hasattr(self.telemetry_sink, "close")
            and callable(self.telemetry_sink.close)
        ):
            self.telemetry_sink.close()
        for g in self.gates:
            if hasattr(g, "close") and callable(g.close):
                g.close()
        for b in getattr(self, "_replay_extra_backends", ()):
            if hasattr(b, "close") and callable(b.close):
                b.close()
        if getattr(self, "_owns_sidecar_backend", False):
            sb = getattr(self, "sidecar_backend", None)
            if sb is not None and hasattr(sb, "close") and callable(sb.close):
                sb.close()

    async def aclose(self) -> None:
        """Asynchronously closes owned resources including telemetry sink, cache, and child gates."""
        if self._closed:
            return
        self._closed = True
        if self._owns_cache and self.cache is not None and not isinstance(self.cache, bool):
            if hasattr(self.cache, "aclose") and callable(self.cache.aclose):
                await self.cache.aclose()
            elif hasattr(self.cache, "close") and callable(self.cache.close):
                self.cache.close()
        if self._owns_telemetry_sink and self.telemetry_sink is not None and not isinstance(self.telemetry_sink, bool):
            if hasattr(self.telemetry_sink, "aclose") and callable(self.telemetry_sink.aclose):
                await self.telemetry_sink.aclose()
            elif hasattr(self.telemetry_sink, "close") and callable(self.telemetry_sink.close):
                self.telemetry_sink.close()
        for g in self.gates:
            if hasattr(g, "aclose") and callable(g.aclose):
                await g.aclose()
            elif hasattr(g, "close") and callable(g.close):
                g.close()
        for b in getattr(self, "_replay_extra_backends", ()):
            if hasattr(b, "aclose") and callable(b.aclose):
                await b.aclose()
            elif hasattr(b, "close") and callable(b.close):
                b.close()
        if getattr(self, "_owns_sidecar_backend", False):
            sb = getattr(self, "sidecar_backend", None)
            if sb is not None:
                if hasattr(sb, "aclose") and callable(sb.aclose):
                    await sb.aclose()
                elif hasattr(sb, "close") and callable(sb.close):
                    sb.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.aclose()

    def evaluate_batch(
        self,
        candidates: Sequence[Any],
        *,
        on_error: Literal["raise", "abstain", "fail"] = "raise",
        **kwargs: Any,
    ) -> list[Verdict]:
        if on_error not in ("raise", "abstain", "fail"):
            raise ConfigurationError(f"Invalid on_error={on_error!r}: must be 'raise', 'abstain', or 'fail'")
        results: list[Verdict] = []
        for c in candidates:
            try:
                results.append(self.evaluate(c, **kwargs))
            except BaseException as exc:
                if isinstance(exc, Exception) and on_error in ("abstain", "fail"):
                    status = (
                        getattr(DecisionStatus, "ABSTAINED", DecisionStatus.AMBIGUOUS)
                        if on_error == "abstain"
                        else DecisionStatus.FAILED
                    )
                    reason_str = f"{type(exc).__name__}: {exc}"
                    trace = LayerTrace(
                        layer_name="Cascade",
                        status=status,
                        latency_ms=0.0,
                        diagnostic_report=reason_str,
                        metadata={"abstained_reason": reason_str, "error": reason_str},
                    )
                    v = Verdict(
                        passed=False,
                        status=status,
                        triggered_layer="Cascade",
                        total_latency_ms=0.0,
                        reason=reason_str,
                        traces=[trace],
                    )
                    object.__setattr__(v, "abstained_reason", reason_str)
                    results.append(v)
                else:
                    raise
        return results

    async def aevaluate_batch(
        self,
        candidates: Sequence[Any],
        *,
        max_concurrency: int | None = None,
        on_error: Literal["raise", "abstain", "fail"] = "raise",
        **kwargs: Any,
    ) -> list[Verdict]:
        import asyncio

        if on_error not in ("raise", "abstain", "fail"):
            raise ConfigurationError(f"Invalid on_error={on_error!r}: must be 'raise', 'abstain', or 'fail'")
        if max_concurrency is not None and (
            isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1
        ):
            raise ConfigurationError(f"max_concurrency must be an integer >= 1, got {max_concurrency!r}")

        sem = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

        async def _eval_one(c: Any) -> Verdict:
            async def _run() -> Verdict:
                return await self.aevaluate(c, **kwargs)

            try:
                if sem is not None:
                    async with sem:
                        return await _run()
                return await _run()
            except BaseException as exc:
                if isinstance(exc, Exception) and on_error in ("abstain", "fail"):
                    status = (
                        getattr(DecisionStatus, "ABSTAINED", DecisionStatus.AMBIGUOUS)
                        if on_error == "abstain"
                        else DecisionStatus.FAILED
                    )
                    reason_str = f"{type(exc).__name__}: {exc}"
                    trace = LayerTrace(
                        layer_name="Cascade",
                        status=status,
                        latency_ms=0.0,
                        diagnostic_report=reason_str,
                        metadata={"abstained_reason": reason_str, "error": reason_str},
                    )
                    v = Verdict(
                        passed=False,
                        status=status,
                        triggered_layer="Cascade",
                        total_latency_ms=0.0,
                        reason=reason_str,
                        traces=[trace],
                    )
                    object.__setattr__(v, "abstained_reason", reason_str)
                    return v
                raise

        return list(await asyncio.gather(*[_eval_one(c) for c in candidates]))

    def replay_config(self) -> dict[str, Any]:
        layers = [cfg for g in self.gates if (cfg := g.replay_config()) is not None]
        config: dict[str, Any] = {
            "gate_type": "Cascade",
            "rubric": self.rubric,
            "layers": layers,
        }
        if self.context is not None:
            config["context"] = self.context
        if self.query is not None:
            config["query"] = self.query
        if self.on_unsupported_logprobs != "raise":
            config["on_unsupported_logprobs"] = self.on_unsupported_logprobs
        if self.sidecar_backend is not None:
            config["sidecar_backend"] = infer_backend_spec(self.sidecar_backend)
        if self.sidecar_model is not None:
            config["sidecar_model"] = self.sidecar_model
        return config

    def _effective_cache(self) -> Any:
        if self.cache is not None:
            return self.cache
        from logitgate.cache import VerdictCache

        all_false = True
        for g in self.gates:
            c = getattr(g, "cache", None)
            if isinstance(c, VerdictCache):
                return c
            if c is not False:
                all_false = False
        if all_false:
            return False
        return None

    def _effective_telemetry_sink(self) -> Any:
        if self.telemetry_sink is not None:
            return self.telemetry_sink
        from logitgate.telemetry.sinks import CompositeTelemetrySink, TelemetrySink

        inner_sinks: list[TelemetrySink] = []
        seen_ids: set[int] = set()
        all_false = True
        for g in self.gates:
            s = getattr(g, "telemetry_sink", None)
            if isinstance(s, TelemetrySink):
                all_false = False
                if id(s) not in seen_ids:
                    seen_ids.add(id(s))
                    inner_sinks.append(s)
            elif s is not False:
                all_false = False
        if len(inner_sinks) == 1:
            return inner_sinks[0]
        if len(inner_sinks) > 1:
            return CompositeTelemetrySink(inner_sinks)
        if all_false:
            return False
        return None

    def __repr__(self) -> str:
        pipeline_repr = " >> ".join(repr(g) for g in self.gates)
        return f"Cascade([{pipeline_repr}])"

    def _ensure_children_calibrated_sync(self) -> None:
        for g in self.gates:
            ensure_fn = getattr(g, "_ensure_calibrated", None)
            if callable(ensure_fn):
                ensure_fn()
            inner = getattr(g, "_gate", None)
            if inner is not None and callable(getattr(inner, "_ensure_calibrated", None)):
                inner._ensure_calibrated()

    async def _aensure_children_calibrated_async(self) -> None:
        for g in self.gates:
            aensure_fn = getattr(g, "_aensure_calibrated", None)
            if callable(aensure_fn):
                await aensure_fn()
            inner = getattr(g, "_gate", None)
            if inner is not None and callable(getattr(inner, "_aensure_calibrated", None)):
                await inner._aensure_calibrated()

    def _finalize_cache_hit(
        self, hit: Verdict, orig_candidate: Any, kwargs: dict[str, Any]
    ) -> Verdict:
        from logitgate.telemetry.sinks import emit_verdict_telemetry

        eff_rubric = next(
            (
                r
                for g in self.gates
                if isinstance(r := (getattr(g, "rubric", None) or getattr(g, "question", None)), str)
                and r
            ),
            None,
        )
        primary_backend: Any = None
        primary_model: str | None = None
        for g in self.gates:
            g_backend = getattr(g, "backend", None)
            if g.layer_tier >= TIER_LOGIT and g_backend is not None:
                primary_backend = g_backend
                primary_model = getattr(g, "model", None)
                break
        raw_tags = kwargs.get("telemetry_tags")
        merged_tags = dict(raw_tags) if isinstance(raw_tags, dict) else {}
        merged_tags["cache_hit"] = True
        hit_kwargs = {**kwargs, "telemetry_tags": merged_tags}
        emit_verdict_telemetry(
            hit,
            candidate=orig_candidate,
            gate_name=self.name,
            gate_type="Cascade",
            rubric=eff_rubric,
            model=primary_model,
            backend=primary_backend,
            gate_sink=self._effective_telemetry_sink(),
            call_kwargs=hit_kwargs,
        )
        return self._persist_failure(hit, orig_candidate, eff_rubric, kwargs)

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import _resolve_active_cache, lookup_cached_verdict

        eff_context = kwargs.get("context", self.context) if "context" in kwargs else self.context
        eff_query = kwargs.get("query", self.query) if "query" in kwargs else self.query
        _validate_rag_context_and_query(eff_context, eff_query)
        call_kwargs = {**kwargs, "context": eff_context, "query": eff_query}

        eff_cache = self._effective_cache()
        active_cache, owns_temp = _resolve_active_cache(eff_cache, call_kwargs)
        if active_cache is not None:
            if owns_temp:
                active_cache.close()
            if not _has_active_cascade_audit(self.gates):
                self._ensure_children_calibrated_sync()
                hit = lookup_cached_verdict(
                    self, candidate, gate_cache=eff_cache, call_kwargs=call_kwargs
                )
                if hit is not None:
                    return self._finalize_cache_hit(hit, candidate, call_kwargs)

        pipeline = self._pipeline(candidate, call_kwargs)
        try:
            request = next(pipeline)
        except StopIteration as done:  # pragma: no cover - the first gate always runs
            return cast(Verdict, done.value)
        while True:
            gate, current_candidate, gate_kwargs = request
            # Evaluate outside the try so a StopIteration leaking from a gate is never
            # mistaken for pipeline completion.
            verdict = gate.evaluate(current_candidate, **gate_kwargs)
            try:
                request = pipeline.send(verdict)
            except StopIteration as done:
                return cast(Verdict, done.value)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import _resolve_active_cache, lookup_cached_verdict

        eff_context = kwargs.get("context", self.context) if "context" in kwargs else self.context
        eff_query = kwargs.get("query", self.query) if "query" in kwargs else self.query
        _validate_rag_context_and_query(eff_context, eff_query)
        call_kwargs = {**kwargs, "context": eff_context, "query": eff_query}

        eff_cache = self._effective_cache()
        active_cache, owns_temp = _resolve_active_cache(eff_cache, call_kwargs)
        if active_cache is not None:
            if owns_temp:
                active_cache.close()
            if not _has_active_cascade_audit(self.gates):
                await self._aensure_children_calibrated_async()
                hit = lookup_cached_verdict(
                    self, candidate, gate_cache=eff_cache, call_kwargs=call_kwargs
                )
                if hit is not None:
                    return self._finalize_cache_hit(hit, candidate, call_kwargs)

        pipeline = self._pipeline(candidate, call_kwargs)
        try:
            request = next(pipeline)
        except StopIteration as done:  # pragma: no cover - the first gate always runs
            return cast(Verdict, done.value)
        while True:
            gate, current_candidate, gate_kwargs = request
            verdict = await gate.aevaluate(current_candidate, **gate_kwargs)
            try:
                request = pipeline.send(verdict)
            except StopIteration as done:
                return cast(Verdict, done.value)

    def _pipeline(self, candidate: Any, kwargs: dict[str, Any]) -> _Pipeline:
        from logitgate.cache import _resolve_active_cache, store_cached_verdict

        t0 = time.perf_counter()
        orig_candidate = candidate
        eff_cache = self._effective_cache()
        active_top_cache, owns_top_temp = _resolve_active_cache(eff_cache, kwargs)
        if owns_top_temp and active_top_cache is not None:
            active_top_cache.close()
        has_top_cache = active_top_cache is not None

        child_base_kwargs = {k: v for k, v in kwargs.items() if k != "cache"}
        gate_kwargs = {**child_base_kwargs, "save_on_fail": False, "_suppress_telemetry": True}
        traces: list[LayerTrace] = []
        tokens = 0
        rubric: str | None = None  # inherited by generative gates without their own rubric
        replay_cfg: dict[str, Any] | None = None  # most recent replay_config from a gate that ran
        upstream_gate: BaseGate | None = None  # most recent non-generative gate that ran
        upstream: Verdict | None = None
        outcome: dict[str, Any] | None = None  # what the cascade returns if nothing else runs
        escalated = False
        fast_pathed = False
        primary_model: str | None = None
        primary_backend: Any = None
        teacher_model: str | None = None
        raw_teacher_passed: bool | None = None

        def conclude(fields: dict[str, Any]) -> Verdict:
            verdict = Verdict(
                **fields,
                total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                total_tokens_generated=tokens,
                traces=traces,
            )
            has_norm = candidate is not orig_candidate
            caller_norm_box = kwargs.get("_normalized_out")
            if has_norm and isinstance(caller_norm_box, list):
                caller_norm_box.append(candidate)
            if not _has_active_cascade_audit(self.gates):
                store_cached_verdict(
                    self,
                    orig_candidate,
                    verdict,
                    gate_cache=eff_cache,
                    call_kwargs=kwargs,
                    has_normalized=has_norm,
                    normalized_artifact=candidate if has_norm else None,
                )
            from logitgate.telemetry.sinks import emit_verdict_telemetry

            eff_rubric = rubric or next(
                (
                    r
                    for g in self.gates
                    if isinstance(r := (getattr(g, "rubric", None) or getattr(g, "question", None)), str)
                    and r
                ),
                None,
            )
            emit_verdict_telemetry(
                verdict,
                candidate=orig_candidate,
                gate_name=self.name,
                gate_type="Cascade",
                rubric=eff_rubric,
                model=primary_model,
                backend=primary_backend,
                gate_sink=self._effective_telemetry_sink(),
                call_kwargs=kwargs,
                fast_pathed=fast_pathed and not escalated,
                escalated=escalated,
                teacher_model=teacher_model,
                teacher_passed=raw_teacher_passed,
            )
            return self._persist_failure(verdict, orig_candidate, rubric, kwargs, replay_cfg=replay_cfg)

        for i, gate in enumerate(self.gates):
            if gate.layer_tier >= TIER_GENERATIVE:
                gate_rubric = getattr(gate, "rubric", None)
                if rubric is None and isinstance(gate_rubric, str) and gate_rubric:
                    rubric = gate_rubric
                policy = gate.escalation_policy
                upstream_aci = (
                    getattr(upstream_gate, "adaptive_controller", None)
                    if upstream_gate is not None
                    else None
                )
                is_cascade_audit = (
                    upstream_aci is not None
                    and upstream is not None
                    and getattr(upstream_aci, "feedback_mode", None) == "cascade_audit"
                )
                normal_escalate = _should_escalate(policy, upstream_gate, upstream)
                shadow_audited = False
                audit_sampling_prob = 1.0
                if not normal_escalate:
                    if upstream is None or upstream_gate is None:
                        raise RuntimeError(
                            "Internal cascade state invariant violated: expected upstream verdict and gate"
                        )
                    if is_cascade_audit and upstream_aci is not None and upstream_aci.should_audit_fast_path():
                        shadow_audited = True
                        audit_sampling_prob = float(upstream_aci.audit_sample_rate)
                    else:
                        if is_cascade_audit and upstream_aci is not None:
                            upstream_aci.record_cascade_outcome(
                                l1_status=upstream.status,
                                teacher_passed=None,
                                sampling_prob=float(upstream_aci.audit_sample_rate),
                                audited_fast_path=False,
                            )
                        fast_pathed = True
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

                gen_step_kwargs = {
                    **gate_kwargs,
                    "rubric_override": rubric if rubric is not None else gate_kwargs.get("rubric_override"),
                }
                if (has_top_cache or is_cascade_audit) and (
                    getattr(gate, "cache", None) is None
                    or getattr(gate, "cache", None) is active_top_cache
                ):
                    gen_step_kwargs["_bypass_cache"] = True
                verdict = yield gate, candidate, gen_step_kwargs
                escalated = True
                raw_teacher_passed = verdict.passed
                if is_cascade_audit and upstream is not None and upstream_aci is not None:
                    upstream_aci.record_cascade_outcome(
                        l1_status=upstream.status,
                        teacher_passed=verdict.passed,
                        sampling_prob=audit_sampling_prob,
                        audited_fast_path=shadow_audited,
                    )
                jg_backend = getattr(gate, "backend", None)
                teacher_model = (
                    getattr(gate, "model", None)
                    or getattr(jg_backend, "model", None)
                    or getattr(jg_backend, "model_name", None)
                )
                if primary_backend is None and jg_backend is not None:
                    primary_backend = jg_backend
                if primary_model is None and teacher_model is not None:
                    primary_model = teacher_model
                traces.extend(verdict.traces)
                if shadow_audited and traces:
                    last_meta = dict(traces[-1].metadata or {})
                    last_meta["shadow_audit"] = True
                    traces[-1] = traces[-1].model_copy(update={"metadata": last_meta})
                tokens += verdict.total_tokens_generated

                passed, status = verdict.passed, verdict.status
                overrode_judge_pass = False
                aci_allows_override = bool(
                    is_cascade_audit and getattr(upstream_aci, "override_on_audit", False)
                )
                if shadow_audited and not aci_allows_override and upstream is not None:
                    passed, status = upstream.passed, upstream.status
                    effective_prob = upstream.probability
                    fused_lo: float | None = None
                    tr1_prior_lo: float | None = None
                    judge_lo_orig: float | None = None
                else:
                    can_override = (
                        getattr(gate, "can_override_fail", False)
                        or policy == "adjudicate"
                        or aci_allows_override
                    )
                    locked_upstream_fail = (
                        upstream is not None
                        and upstream.status == DecisionStatus.FAILED
                        and policy == "on_ambiguous_or_fail"
                        and not can_override
                    )
                    fused_lo = None
                    tr1_prior_lo = None
                    judge_lo_orig = None
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
            step_kwargs = {**gate_kwargs, "_normalized_out": norm_box}
            if has_top_cache and (
                getattr(gate, "cache", None) is None
                or getattr(gate, "cache", None) is active_top_cache
            ):
                step_kwargs["_bypass_cache"] = True
            verdict = yield gate, candidate, step_kwargs
            if gate.layer_tier >= TIER_LOGIT:
                g_backend = getattr(gate, "backend", None)
                g_model = (
                    getattr(gate, "model", None)
                    or getattr(g_backend, "model", None)
                    or getattr(g_backend, "model_name", None)
                )
                if g_backend is not None:
                    primary_backend = g_backend
                if g_model is not None:
                    primary_model = g_model
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
            gate_rubric = (
                getattr(gate, "rubric", None)
                or getattr(gate, "intent_rubric", None)
                or getattr(gate, "question", None)
            )
            if isinstance(gate_rubric, str) and gate_rubric:
                rubric = gate_rubric

            fields = _verdict_fields(verdict, triggered_layer=gate.name)
            if verdict.passed:
                outcome = fields
                continue

            effective_fail_tier = (
                verdict.traces[-1].metadata.get("layer_tier", gate.layer_tier)
                if verdict.traces and isinstance(verdict.traces[-1].metadata, dict)
                else gate.layer_tier
            )
            next_gate = self.gates[i + 1] if i + 1 < len(self.gates) else None
            defers_to_generative = (
                effective_fail_tier >= TIER_LOGIT
                and next_gate is not None
                and next_gate.layer_tier >= TIER_GENERATIVE
            )
            if not defers_to_generative:
                if any(g.layer_tier > gate.layer_tier for g in self.gates[i + 1 :]):
                    fast_pathed = True
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
        if effective_save is None and getattr(self, "save_failures", None) is not None:
            effective_save = self.save_failures
        fallback_rubric = next(
            (
                r
                for g in self.gates
                if isinstance(r := (getattr(g, "rubric", None) or getattr(g, "question", None)), str) and r
            ),
            "Cascade evaluation",
        )
        gate_type = None
        replayable_layers = [cfg for g in self.gates if (cfg := g.replay_config()) is not None]
        if len(replayable_layers) >= 2:
            gate_type = "Cascade"
            cascade_cfg = self.replay_config()
            gate_config["gate_type"] = "Cascade"
            gate_config["layers"] = cascade_cfg.get("layers", [])
        return persist_failure_snapshot(
            verdict,
            candidate=candidate,
            rubric=rubric or fallback_rubric,
            gate_type=gate_type,
            gate_config=gate_config,
            save_on_fail=effective_save,
            test_name=kwargs.get("test_name"),
            snapshot_dir=kwargs.get("snapshot_dir"),
        )
