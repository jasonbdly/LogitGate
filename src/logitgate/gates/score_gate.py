from __future__ import annotations

import asyncio
import math
import secrets
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logitgate.backends.base import LogitBackend
from logitgate.calibration.lifecycle import (
    TopLogprobs,
    _ensure_logprobs_present,
    _resolve_gate_temperature,
    _resolve_warm_prefix,
    _validate_optional_bool,
    _warn_if_profile_model_mismatch,
)
from logitgate.config import (
    LogitGateDefaults,
    get_default_model,
    is_temperature_configured,
)
from logitgate.debiasing.inversion import (
    compile_score_prompt,
)
from logitgate.debiasing.softmax import (
    extract_token_logprob,
    has_any_target_token,
)
from logitgate.debiasing.softmax import (
    softmax_probabilities as _softmax,
)
from logitgate.exceptions import ConfigurationError, UnsupportedLogprobsError
from logitgate.gates.base import BaseGate
from logitgate.gates.logit_gate import UnsupportedLogprobsPolicy, _validate_rag_context_and_query
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

if TYPE_CHECKING:
    from logitgate.calibration.profile import CalibrationProfile


def _get_lg_mod() -> Any:
    return sys.modules.get("logitgate.gates.logit_gate")


def _resolve_logit_backend(*args: Any, **kwargs: Any) -> Any:
    mod = _get_lg_mod()
    fn = getattr(mod, "resolve_logit_backend", None) if mod else None
    if fn is not None:
        return fn(*args, **kwargs)
    from logitgate.backends.base import resolve_logit_backend as canonical
    return canonical(*args, **kwargs)


def _lazy_calibrate_sync(*args: Any, **kwargs: Any) -> Any:
    mod = _get_lg_mod()
    fn = getattr(mod, "_lazy_calibrate_sync", None) if mod else None
    if fn is not None:
        return fn(*args, **kwargs)
    from logitgate.calibration.lifecycle import _lazy_calibrate_sync as canonical
    return canonical(*args, **kwargs)


async def _lazy_calibrate_async(*args: Any, **kwargs: Any) -> Any:
    mod = _get_lg_mod()
    fn = getattr(mod, "_lazy_calibrate_async", None) if mod else None
    if fn is not None:
        return await fn(*args, **kwargs)
    from logitgate.calibration.lifecycle import _lazy_calibrate_async as canonical
    return await canonical(*args, **kwargs)


class ScoreGate(BaseGate):
    """Continuous probability-weighted rubric scoring gate.

    Computes expected continuous quality score:
        E[S] = sum_{s=min_score}^{max_score} s * P(s)
    from first-token logits as formalized by Liu et al. (2023, G-Eval).
    Supports temperature scaling and optional symmetric scale reversal debiasing
    (evaluating both standard 1..5 and inverted 5..1 scale mappings).

    Args:
        rubric: Evaluation rubric describing scoring expectations.
        min_score: Minimum integer score rating (default: 1).
        max_score: Maximum integer score rating (default: 5).
        threshold: Score threshold required for a PASSED verdict (default: 3.5, or
            profile.score_threshold when a CalibrationProfile provides one).
        backend: Backend instance or specifier URI.
        model: Optional model identifier string for the backend.
        symmetric_debiasing: When True, queries both normal and inverted scale prompts
            and averages logprobs to eliminate scale-end anchoring bias.
        query_first: When True, compiles prompt with rubric and candidate first.
        logit_temperature: Temperature scaling factor, 'auto' to calibrate via
            micro-probes, or None to use global/model-calibrated defaults (lazy-calibrating
            unknown BYO models when auto_calibrate=True).
        save_on_fail: Override failure snapshot persistence for this gate.
        name: Identifier for this gate instance.
        fail_threshold: Optional lower cutoff below or at which scores return FAILED; scores in
            (fail_threshold, threshold) abstain as AMBIGUOUS.
        profile: Optional CalibrationProfile (or path to JSON) supplying calibrated
            multiclass temperature and score cutoffs.

    Raises:
        ConfigurationError: If rubric is empty, min_score >= max_score, threshold is out of bounds,
            or logit_temperature <= 0.
    """

    def __init__(
        self,
        rubric: str,
        min_score: int = 1,
        max_score: int = 5,
        threshold: float | None = None,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        symmetric_debiasing: bool | None = True,
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        save_on_fail: bool | None = None,
        name: str = "ScoreGate",
        *,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
        on_unsupported_logprobs: UnsupportedLogprobsPolicy = "raise",
        sidecar_backend: str | LogitBackend | None = None,
        sidecar_model: str | None = None,
        fail_threshold: float | None = None,
        max_score_std: float | None = None,
        warm_prefix: bool | None = None,
        profile: CalibrationProfile | str | Path | None = None,
        telemetry_sink: Any = None,
        cache: Any = None,
    ) -> None:
        from logitgate.calibration.profile import resolve_calibration_profile
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        loaded_profile = resolve_calibration_profile(profile)
        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        if not isinstance(rubric, str) or not rubric.strip():
            raise ConfigurationError("rubric cannot be empty or whitespace")
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, int)
            or isinstance(max_score, bool)
            or not isinstance(max_score, int)
        ):
            raise ConfigurationError(
                f"min_score and max_score must be integers, got min_score={min_score!r}, max_score={max_score!r}"
            )
        if min_score >= max_score:
            raise ConfigurationError(f"min_score ({min_score}) must be strictly less than max_score ({max_score})")
        if min_score < 0 or max_score > 9:
            raise ConfigurationError(
                f"ScoreGate requires single-digit integer bounds (0 <= min_score < max_score <= 9), "
                f"got min_score={min_score}, max_score={max_score}"
            )

        raw_threshold: Any = (
            threshold
            if threshold is not None
            else (
                loaded_profile.score_threshold
                if loaded_profile is not None and loaded_profile.score_threshold is not None
                else 3.5
            )
        )
        if (
            isinstance(raw_threshold, bool)
            or not isinstance(raw_threshold, (int, float))
            or not math.isfinite(float(raw_threshold))
            or not (min_score <= float(raw_threshold) <= max_score)
        ):
            raise ConfigurationError(
                f"threshold ({raw_threshold}) must be between min_score ({min_score}) and max_score ({max_score})"
            )
        raw_fail_threshold: Any = (
            fail_threshold
            if fail_threshold is not None
            else (loaded_profile.score_fail_threshold if loaded_profile is not None else None)
        )
        effective_fail_thresh: float | None = None
        if raw_fail_threshold is not None:
            if (
                isinstance(raw_fail_threshold, bool)
                or not isinstance(raw_fail_threshold, (int, float))
                or not math.isfinite(float(raw_fail_threshold))
                or not (min_score <= float(raw_fail_threshold) <= float(raw_threshold))
            ):
                raise ConfigurationError(
                    f"fail_threshold ({raw_fail_threshold!r}) must be between min_score ({min_score}) and threshold ({raw_threshold})"
                )
            effective_fail_thresh = float(raw_fail_threshold)
        effective_max_std: float | None = None
        if max_score_std is not None:
            if (
                isinstance(max_score_std, bool)
                or not isinstance(max_score_std, (int, float))
                or not math.isfinite(float(max_score_std))
                or float(max_score_std) <= 0.0
            ):
                raise ConfigurationError(
                    f"max_score_std must be a positive float, got {max_score_std!r}"
                )
            effective_max_std = float(max_score_std)
        effective_save_on_fail = _validate_optional_bool("save_on_fail", save_on_fail)
        effective_warm_prefix = _validate_optional_bool("warm_prefix", warm_prefix)

        _validate_rag_context_and_query(context, query)
        if on_unsupported_logprobs not in ("raise", "escalate", "sidecar"):
            raise ConfigurationError(
                f"Invalid on_unsupported_logprobs '{on_unsupported_logprobs}'. "
                f"Must be one of ['raise', 'escalate', 'sidecar']"
            )
        if on_unsupported_logprobs == "sidecar" and sidecar_backend is None:
            raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'")

        owns_backend = not isinstance(backend, LogitBackend)
        resolved_backend = _resolve_logit_backend(backend, model=model)
        owns_sidecar = False
        resolved_sidecar: LogitBackend | None = None
        eff_sidecar_model: str | None = None
        sidecar_temp = 1.0
        try:
            if sidecar_backend is not None:
                owns_sidecar = not isinstance(sidecar_backend, LogitBackend)
                resolved_sidecar = _resolve_logit_backend(sidecar_backend, model=sidecar_model)
                eff_sidecar_model = (
                    sidecar_model
                    or getattr(resolved_sidecar, "model", None)
                    or getattr(resolved_sidecar, "model_name", None)
                    or get_default_model()
                )
                sidecar_temp, _ = _resolve_gate_temperature(
                    None,
                    eff_sidecar_model,
                    resolved_sidecar,
                    loaded_profile,
                    binary=False,
                )

            effective_model = (
                model
                or getattr(resolved_backend, "model", None)
                or getattr(resolved_backend, "model_name", None)
                or get_default_model()
            )
            effective_temp, needs_lazy = _resolve_gate_temperature(
                logit_temperature,
                effective_model,
                resolved_backend,
                loaded_profile,
                binary=False,
            )
            from logitgate.cache import resolve_verdict_cache

            resolved_cache = resolve_verdict_cache(cache)
        except BaseException:
            if owns_backend:
                resolved_backend.close()
            if owns_sidecar and resolved_sidecar is not None:
                resolved_sidecar.close()
            raise

        _warn_if_profile_model_mismatch(loaded_profile, effective_model, name)
        self.backend = resolved_backend
        self.model = effective_model
        self.profile = loaded_profile
        self.rubric = rubric.strip()
        self.min_score = min_score
        self.max_score = max_score
        self.threshold = float(raw_threshold)
        self.symmetric_debiasing = True if symmetric_debiasing is None else bool(symmetric_debiasing)
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.context = context
        self.query = query
        self.on_unsupported_logprobs = on_unsupported_logprobs
        self.sidecar_backend = resolved_sidecar
        self.sidecar_model = eff_sidecar_model
        self.sidecar_temperature = sidecar_temp
        self._owns_sidecar_backend = owns_sidecar
        self._needs_lazy_calibration = needs_lazy
        self._calibration_lock = threading.Lock()
        self.fail_threshold = effective_fail_thresh
        self.max_score_std = effective_max_std
        self.warm_prefix = effective_warm_prefix
        self.save_on_fail = effective_save_on_fail
        self.telemetry_sink = resolved_sink
        self.cache = resolved_cache
        self.name = name
        self._owns_backend = owns_backend
        self._owns_cache = isinstance(cache, (str, Path))
        self._owns_sink = isinstance(telemetry_sink, str) or hasattr(telemetry_sink, "__fspath__")

    def close(self) -> None:
        """Safely close any owned resources (backend, cache, sidecar). Idempotent."""
        super().close()
        if getattr(self, "_owns_sidecar_backend", False):
            sb = getattr(self, "sidecar_backend", None)
            if sb is not None and hasattr(sb, "close"):
                sb.close()

    async def aclose(self) -> None:
        """Safely and asynchronously close any owned resources. Idempotent."""
        await super().aclose()
        if getattr(self, "_owns_sidecar_backend", False):
            sb = getattr(self, "sidecar_backend", None)
            if sb is not None:
                if hasattr(sb, "aclose"):
                    await sb.aclose()
                elif hasattr(sb, "close"):
                    sb.close()

    @property
    def calibration_profile(self) -> CalibrationProfile | None:
        return self.profile

    def replay_config(self) -> dict[str, Any]:
        from logitgate.backends.base import infer_backend_spec

        config: dict[str, Any] = {
            "gate_type": "ScoreGate",
            "rubric": self.rubric,
            "backend": infer_backend_spec(self.backend),
            "min_score": self.min_score,
            "max_score": self.max_score,
            "threshold": self.threshold,
            "symmetric_debiasing": self.symmetric_debiasing,
            "query_first": self.query_first,
            "logit_temperature": self.logit_temperature,
            "name": self.name,
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
        if self.fail_threshold is not None:
            config["fail_threshold"] = self.fail_threshold
        if self.max_score_std is not None:
            config["max_score_std"] = self.max_score_std
        if self.warm_prefix is not None:
            config["warm_prefix"] = self.warm_prefix
        model = (
            self.model
            or getattr(self.backend, "model", None)
            or getattr(self.backend, "model_name", None)
        )
        if model:
            config["model"] = model
        if self.calibration_profile is not None:
            config["calibration_profile"] = self.calibration_profile.model_dump(mode="json")
        dist_bundle = getattr(self, "distillation_bundle", None)
        if dist_bundle is not None:
            config["distillation_bundle"] = dist_bundle.model_dump(mode="json")
        return config

    def _handle_failure_snapshot(
        self, candidate: Any, verdict: Verdict, *, _is_cache_hit: bool = False, **kwargs: Any
    ) -> Verdict:
        from logitgate.cache import store_cached_verdict
        from logitgate.telemetry.sinks import emit_verdict_telemetry

        call_kw = kwargs
        if _is_cache_hit:
            raw_tags = kwargs.get("telemetry_tags")
            merged_tags = dict(raw_tags) if isinstance(raw_tags, dict) else {}
            merged_tags["cache_hit"] = True
            call_kw = {**kwargs, "telemetry_tags": merged_tags}
        else:
            store_cached_verdict(
                self, candidate, verdict, gate_cache=self.cache, call_kwargs=kwargs
            )

        emit_verdict_telemetry(
            verdict,
            candidate=candidate,
            gate_name=self.name,
            gate_type="ScoreGate",
            rubric=self.rubric,
            model=self.model,
            backend=self.backend,
            gate_sink=self.telemetry_sink,
            call_kwargs=call_kw,
        )
        if verdict.passed:
            return verdict
        return persist_failure_snapshot(
            verdict,
            candidate=candidate,
            rubric=self.rubric,
            gate_config=self.replay_config(),
            save_on_fail=kwargs.get("save_on_fail", self.save_on_fail),
            test_name=kwargs.get("test_name"),
            snapshot_dir=kwargs.get("snapshot_dir"),
        )

    def __repr__(self) -> str:
        rubric_repr = repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
        return (
            f"ScoreGate(rubric={rubric_repr}, "
            f"range=[{self.min_score}..{self.max_score}], "
            f"threshold={self.threshold}, "
            f"backend={self.backend!r})"
        )

    def _score_prompts(
        self,
        candidate: Any,
        *,
        boundary_nonce: str | None = None,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
    ) -> tuple[str, str | None]:
        def build(inverted: bool) -> str:
            return compile_score_prompt(
                rubric=self.rubric,
                candidate=candidate,
                min_score=self.min_score,
                max_score=self.max_score,
                query_first=self.query_first,
                inverted=inverted,
                boundary_nonce=boundary_nonce,
                context=context,
                query=query,
            )

        return build(False), (build(True) if self.symmetric_debiasing else None)

    def _escalate_unsupported(
        self,
        candidate: Any,
        exc_msg: str,
        t0: float,
        kwargs: dict[str, Any],
    ) -> Verdict:
        latency = (time.perf_counter() - t0) * 1000.0
        reason = f"[{self.name}] Backend does not support logprobs; escalating ({exc_msg})"
        trace = LayerTrace(
            layer_name=self.name,
            status=DecisionStatus.AMBIGUOUS,
            latency_ms=latency,
            tokens_generated=0,
            metadata={
                "abstained_reason": "unsupported_logprobs",
                "fallback_policy": "escalate",
            },
        )
        verdict = Verdict(
            passed=False,
            status=DecisionStatus.AMBIGUOUS,
            probability=None,
            score=None,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )
        return self._handle_failure_snapshot(candidate, verdict, **kwargs)

    def _execute_queries(
        self, forward: str, reverse: str | None, tokens: list[str], top_k: int
    ) -> tuple[TopLogprobs, TopLogprobs | None, bool, str | None, float]:
        from logitgate.backends.base import infer_backend_spec

        backend = self.backend
        supports = getattr(backend, "supports_logprobs", True) is not False
        if not supports:
            if self.on_unsupported_logprobs == "raise":
                raise UnsupportedLogprobsError(
                    f"Backend '{getattr(backend, 'model', 'unknown')}' does not support logprobs."
                )
            if self.on_unsupported_logprobs == "escalate":
                raise UnsupportedLogprobsError("Backend does not support logprobs")
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'")
            top_fwd = self.sidecar_backend.query_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
            top_rev = (
                self.sidecar_backend.query_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens)
                if reverse is not None
                else None
            )
            spec = infer_backend_spec(self.sidecar_backend)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

        try:
            top_fwd = backend.query_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
            top_rev = (
                backend.query_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens)
                if reverse is not None
                else None
            )
            return top_fwd, top_rev, False, None, self.logit_temperature
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs in ("raise", "escalate"):
                raise
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'") from exc
            top_fwd = self.sidecar_backend.query_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
            top_rev = (
                self.sidecar_backend.query_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens)
                if reverse is not None
                else None
            )
            spec = infer_backend_spec(self.sidecar_backend)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

    async def _aexecute_queries(
        self, forward: str, reverse: str | None, tokens: list[str], top_k: int
    ) -> tuple[TopLogprobs, TopLogprobs | None, bool, str | None, float]:
        from logitgate.backends.base import infer_backend_spec

        backend = self.backend
        supports = getattr(backend, "supports_logprobs", True) is not False
        if not supports:
            if self.on_unsupported_logprobs == "raise":
                raise UnsupportedLogprobsError(
                    f"Backend '{getattr(backend, 'model', 'unknown')}' does not support logprobs."
                )
            if self.on_unsupported_logprobs == "escalate":
                raise UnsupportedLogprobsError("Backend does not support logprobs")
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'")
            target_backend = self.sidecar_backend
            used_temp = self.sidecar_temperature
            spec = infer_backend_spec(target_backend)
            sidecar_used = True
        else:
            target_backend = backend
            used_temp = self.logit_temperature
            spec = None
            sidecar_used = False

        try:
            top_rev: TopLogprobs | None = None
            if reverse is None:
                top_fwd = await target_backend.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
            elif _resolve_warm_prefix(self.warm_prefix, target_backend):
                top_fwd = await target_backend.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
                top_rev = await target_backend.aquery_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens)
            else:
                top_fwd, top_rev = await asyncio.gather(
                    target_backend.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens),
                    target_backend.aquery_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens),
                )
            return top_fwd, top_rev, sidecar_used, spec, used_temp
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs in ("raise", "escalate"):
                raise
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'") from exc
            sb = self.sidecar_backend
            top_rev = None
            if reverse is None:
                top_fwd = await sb.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
            elif _resolve_warm_prefix(self.warm_prefix, sb):
                top_fwd = await sb.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
                top_rev = await sb.aquery_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens)
            else:
                top_fwd, top_rev = await asyncio.gather(
                    sb.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens),
                    sb.aquery_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens),
                )
            spec = infer_backend_spec(sb)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

    def _verdict_from_logprobs(
        self,
        top_fwd: TopLogprobs,
        top_rev: TopLogprobs | None,
        t0: float,
        *,
        sidecar_used: bool = False,
        sidecar_spec: str | None = None,
        effective_temp: float | None = None,
    ) -> Verdict:
        _ensure_logprobs_present(top_fwd, top_rev)
        scores = range(self.min_score, self.max_score + 1)
        fwd_raw = {s: extract_token_logprob(top_fwd, str(s)) for s in scores}
        m_fwd = max(fwd_raw.values())
        lse_fwd = m_fwd + math.log(sum(math.exp(v - m_fwd) for v in fwd_raw.values()))
        combined = {s: v - lse_fwd for s, v in fwd_raw.items()}
        if top_rev is not None:
            # On the inverted scale, quality s is written as token (max + min - s).
            mirror = self.max_score + self.min_score
            rev_raw = {s: extract_token_logprob(top_rev, str(mirror - s)) for s in scores}
            m_rev = max(rev_raw.values())
            lse_rev = m_rev + math.log(sum(math.exp(v - m_rev) for v in rev_raw.values()))
            combined = {s: (combined[s] + (rev_raw[s] - lse_rev)) / 2.0 for s in scores}

        used_temp = max(effective_temp if effective_temp is not None else self.logit_temperature, 1e-6)
        score_probs = _softmax(combined, used_temp)
        expected_score_raw = sum(s * prob for s, prob in score_probs.items())
        expected_score = round(expected_score_raw, 4)
        score_variance = max(0.0, sum(((s - expected_score_raw) ** 2) * prob for s, prob in score_probs.items()))
        score_std = round(math.sqrt(score_variance), 4)
        score_tokens = [str(s) for s in scores]
        fwd_has_target = has_any_target_token(top_fwd, score_tokens)
        rev_has_target = True if top_rev is None else has_any_target_token(top_rev, score_tokens)

        abstained_reason: str | None = None
        if not fwd_has_target or not rev_has_target:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            abstained_reason = "unobserved_target_tokens"
        elif self.max_score_std is not None and score_std > self.max_score_std:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            abstained_reason = f"score_std_exceeded ({score_std:.4f} > {self.max_score_std:.4f})"
        elif expected_score >= self.threshold:
            status = DecisionStatus.PASSED
            passed = True
        elif self.fail_threshold is not None and expected_score > self.fail_threshold:
            status = DecisionStatus.AMBIGUOUS
            passed = False
        else:
            status = DecisionStatus.FAILED
            passed = False

        latency = (time.perf_counter() - t0) * 1000.0
        meta: dict[str, Any] = {
            "expected_score": expected_score,
            "score_std": score_std,
            "token_probabilities": {str(k): round(v, 4) for k, v in score_probs.items()},
        }
        if sidecar_used:
            meta["sidecar_used"] = True
            if sidecar_spec is not None:
                meta["sidecar_backend"] = sidecar_spec
        if abstained_reason is not None:
            meta["abstained_reason"] = abstained_reason
        thresh_desc = (
            f"threshold={self.threshold}, fail_thresh={self.fail_threshold}"
            if self.fail_threshold is not None
            else f"threshold={self.threshold}"
        )
        reason = f"ScoreGate E[S]={expected_score:.2f} ({thresh_desc})"
        if abstained_reason is not None:
            reason += f"; abstained={abstained_reason}"

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            metadata=meta,
        )
        return Verdict(
            passed=passed,
            status=status,
            score=None if abstained_reason == "unobserved_target_tokens" else expected_score,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )

    def _ensure_calibrated(self) -> None:
        if not self._needs_lazy_calibration or not self.model:
            return
        with self._calibration_lock:
            if not self._needs_lazy_calibration:
                return
            if not LogitGateDefaults.auto_calibrate or is_temperature_configured():
                self._needs_lazy_calibration = False
                return
            self.logit_temperature = _lazy_calibrate_sync(self.model, self.backend, binary=False)
            self._needs_lazy_calibration = False

    async def _aensure_calibrated(self) -> None:
        if not self._needs_lazy_calibration or not self.model:
            return
        if not LogitGateDefaults.auto_calibrate or is_temperature_configured():
            self._needs_lazy_calibration = False
            return
        temp = await _lazy_calibrate_async(self.model, self.backend, binary=False)
        with self._calibration_lock:
            self.logit_temperature = temp
            self._needs_lazy_calibration = False

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        eff_context = kwargs.get("context", self.context) if "context" in kwargs else self.context
        eff_query = kwargs.get("query", self.query) if "query" in kwargs else self.query
        _validate_rag_context_and_query(eff_context, eff_query)

        self._ensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._handle_failure_snapshot(
                candidate, hit, _is_cache_hit=True, **kwargs
            )
        t0 = time.perf_counter()
        tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]
        top_k = max(10, 2 * len(tokens))
        nonce = secrets.token_hex(8)
        forward, reverse = self._score_prompts(
            candidate, boundary_nonce=nonce, context=eff_context, query=eff_query
        )
        try:
            top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = self._execute_queries(forward, reverse, tokens, top_k)
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs == "escalate":
                return self._escalate_unsupported(candidate, str(exc), t0, kwargs)
            raise
        return self._handle_failure_snapshot(
            candidate,
            self._verdict_from_logprobs(
                top_fwd,
                top_rev,
                t0,
                sidecar_used=sidecar_used,
                sidecar_spec=sidecar_spec,
                effective_temp=eff_temp,
            ),
            **kwargs,
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        eff_context = kwargs.get("context", self.context) if "context" in kwargs else self.context
        eff_query = kwargs.get("query", self.query) if "query" in kwargs else self.query
        _validate_rag_context_and_query(eff_context, eff_query)

        await self._aensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._handle_failure_snapshot(
                candidate, hit, _is_cache_hit=True, **kwargs
            )
        t0 = time.perf_counter()
        tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]
        top_k = max(10, 2 * len(tokens))
        nonce = secrets.token_hex(8)
        forward, reverse = self._score_prompts(
            candidate, boundary_nonce=nonce, context=eff_context, query=eff_query
        )
        try:
            top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = await self._aexecute_queries(forward, reverse, tokens, top_k)
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs == "escalate":
                return self._escalate_unsupported(candidate, str(exc), t0, kwargs)
            raise
        return self._handle_failure_snapshot(
            candidate,
            self._verdict_from_logprobs(
                top_fwd,
                top_rev,
                t0,
                sidecar_used=sidecar_used,
                sidecar_spec=sidecar_spec,
                effective_temp=eff_temp,
            ),
            **kwargs,
        )


__all__ = ["ScoreGate"]
