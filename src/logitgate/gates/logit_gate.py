from __future__ import annotations

import asyncio
import math
import secrets
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from logitgate.backends.base import LogitBackend, resolve_logit_backend
from logitgate.calibration.lifecycle import (
    TopLogprobs,
    _cal_lock_key,
    _ensure_logprobs_present,
    _get_async_cal_lock,
    _get_sync_cal_lock,
    _lazy_calibrate_async,
    _lazy_calibrate_sync,
    _permuted_key_indices,
    _resolve_gate_temperature,
    _resolve_logit_temperature,
    _resolve_warm_prefix,
    _same_nonzero_sign,
    _should_lazy_auto_calibrate,
    _validate_optional_bool,
    _validate_optional_unit_float,
    _validate_prob_threshold,
    _warn_if_profile_model_mismatch,
)
from logitgate.config import (
    LogitGateDefaults,
    get_default_model,
    is_temperature_configured,
)
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_binary_prompt,
)
from logitgate.debiasing.permutation import (
    combine_symmetric_log_odds,
    compute_shannon_entropy,
)
from logitgate.debiasing.softmax import (
    clamped_sigmoid as _clamped_sigmoid,
)
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    has_any_target_token,
)
from logitgate.exceptions import ConfigurationError, UnsupportedLogprobsError
from logitgate.gates.base import BaseGate, BatchErrorPolicy, _make_batch_error_verdict
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

UnsupportedLogprobsPolicy = Literal["raise", "escalate", "sidecar"]


def _validate_rag_context_and_query(
    context: Any, query: Any
) -> None:
    if context is not None:
        if isinstance(context, bool):
            raise ConfigurationError("context cannot be a boolean")
        if not isinstance(context, (str, Sequence)):
            raise ConfigurationError(
                f"context must be a string, sequence of strings, or None, got {type(context).__name__}"
            )
        if isinstance(context, Sequence) and not isinstance(context, str):
            for idx, c in enumerate(context):
                if isinstance(c, bool):
                    raise ConfigurationError(
                        f"context chunk at index {idx} cannot be a boolean"
                    )
                if not isinstance(c, (str, Mapping)) and not hasattr(c, "__dict__"):
                    raise ConfigurationError(
                        f"context chunk at index {idx} must be a string or structured object, got {type(c).__name__}"
                    )
    if query is not None and (isinstance(query, bool) or not isinstance(query, str)):
        raise ConfigurationError(
            f"query must be a string or None, got {type(query).__name__}"
        )

if TYPE_CHECKING:
    from logitgate.calibration.adaptive import AdaptiveConformalController
    from logitgate.calibration.profile import CalibrationProfile
    from logitgate.distill.anchor_bank import DistilledAnchorBank
    from logitgate.distill.bundle import DistillationBundle
    from logitgate.distill.meta_calibrator import LogitMetaCalibrator
    from logitgate.gates.choice_gate import ChoiceGate
    from logitgate.gates.multi_rubric_gate import MultiRubricGate
    from logitgate.gates.score_gate import ScoreGate


def _get_lg_mod() -> Any:
    return sys.modules.get("logitgate.gates.logit_gate")


def _call_lazy_calibrate_sync(model_name: str, backend: LogitBackend, *, binary: bool) -> float:
    mod = _get_lg_mod()
    fn = getattr(mod, "_lazy_calibrate_sync", _lazy_calibrate_sync) if mod else _lazy_calibrate_sync
    return fn(model_name, backend, binary=binary)


async def _call_lazy_calibrate_async(model_name: str, backend: LogitBackend, *, binary: bool) -> float:
    mod = _get_lg_mod()
    fn = getattr(mod, "_lazy_calibrate_async", _lazy_calibrate_async) if mod else _lazy_calibrate_async
    return await fn(model_name, backend, binary=binary)


def _call_resolve_logit_backend(backend_spec: Any, **kwargs: Any) -> LogitBackend:
    mod = _get_lg_mod()
    fn = getattr(mod, "resolve_logit_backend", resolve_logit_backend) if mod else resolve_logit_backend
    return cast(LogitBackend, fn(backend_spec, **kwargs))


class LogitGate(BaseGate):
    """First-token Logit Gate (Layer 1).

    Computes calibrated binary classification confidence scores directly from the
    language model's first-token log-probabilities (`max_tokens=1`) without generating
    multi-token Chain-of-Thought critiques. Uses symmetric log-odds debiasing across
    forward and reverse label mappings to cancel prior token biases (Zhao et al., 2021)
    and positional biases.

    Args:
        rubric: Clear evaluation criterion or question to test against candidate.
        backend: Backend instance or specifier URI (e.g. 'ollama://', 'openai://', 'stub://').
        model: Optional model identifier string for the backend.
        pass_threshold: Probability threshold required to award a PASSED verdict (default: 0.85).
        fail_threshold: Probability threshold below which a decisive FAILED verdict is awarded (default: 0.20).
        symmetric_debiasing: When True, runs forward (A=Pass, B=Fail) and reverse (B=Pass, A=Fail)
            passes and averages them in log-odds space.
        query_first: When True, compiles prompt with rubric and candidate first, ensuring
            identical KV-cache prefixes across both passes.
        logit_temperature: Post-hoc temperature scaling parameter, 'auto' to calibrate via
            micro-probes, or None to use global/model-calibrated defaults (lazy-calibrating
            unknown BYO models when auto_calibrate=True).
        preset: Optional threshold preset ('strict', 'balanced', 'fast_triage').
        save_on_fail: Override failure snapshot persistence for this gate.
        max_positional_bias: Optional maximum allowed positional log-odds divergence across
            opposite-polarity passes before abstaining to AMBIGUOUS.
        name: Identifier for this gate instance.
        warm_prefix: Override whether async symmetric passes run Pass 1 before Pass 2 to warm KV prefix.
        profile: Optional CalibrationProfile (or path to JSON) supplying calibrated
            temperature and conformal cutoffs; explicit keyword arguments take precedence.

    Raises:
        ConfigurationError: If rubric is empty, thresholds are outside [0, 1],
            pass_threshold <= fail_threshold, max_positional_bias <= 0, or logit_temperature <= 0.
    """

    def __init__(
        self,
        rubric: str,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        pass_threshold: float | None = None,
        fail_threshold: float | None = None,
        symmetric_debiasing: bool | None = None,
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        preset: Literal["strict", "balanced", "fast_triage"] | None = None,
        save_on_fail: bool | None = None,
        max_positional_bias: float | None = None,
        name: str = "LogitGate",
        *,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
        on_unsupported_logprobs: UnsupportedLogprobsPolicy = "raise",
        sidecar_backend: str | LogitBackend | None = None,
        sidecar_model: str | None = None,
        warm_prefix: bool | None = None,
        profile: CalibrationProfile | str | Path | None = None,
        adaptive_controller: AdaptiveConformalController | None = None,
        anchor_bank: DistilledAnchorBank | str | Path | None = None,
        meta_calibrator: LogitMetaCalibrator | str | Path | None = None,
        distillation_bundle: DistillationBundle | str | Path | None = None,
        telemetry_sink: Any = None,
        cache: Any = None,
    ) -> None:
        from logitgate.calibration.adaptive import AdaptiveConformalController as _ACI
        from logitgate.calibration.profile import resolve_calibration_profile
        from logitgate.distill.anchor_bank import resolve_anchor_bank
        from logitgate.distill.bundle import resolve_distillation_bundle
        from logitgate.distill.meta_calibrator import resolve_meta_calibrator
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        if adaptive_controller is not None and not isinstance(adaptive_controller, _ACI):
            raise ConfigurationError(
                f"adaptive_controller must be an AdaptiveConformalController or None, "
                f"got {type(adaptive_controller).__name__}"
            )
        resolved_bundle = resolve_distillation_bundle(distillation_bundle)
        resolved_anchors = resolve_anchor_bank(anchor_bank)
        if resolved_anchors is None and resolved_bundle is not None:
            resolved_anchors = resolved_bundle.anchor_bank
        resolved_meta = resolve_meta_calibrator(meta_calibrator)
        if resolved_meta is None and resolved_bundle is not None:
            resolved_meta = resolved_bundle.meta_calibrator
        resolved_unanchored_meta = (
            resolved_bundle.unanchored_meta_calibrator
            if meta_calibrator is None and resolved_bundle is not None
            else None
        )
        explicit_profile = resolve_calibration_profile(profile)
        loaded_profile = explicit_profile
        if loaded_profile is None and resolved_bundle is not None:
            loaded_profile = resolved_bundle.profile
        if loaded_profile is None and adaptive_controller is not None:
            loaded_profile = adaptive_controller.profile
        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        if preset is not None:
            if preset == "strict":
                default_pass = 0.90
                default_fail = 0.25
                default_sym = True
            elif preset == "balanced":
                default_pass = 0.85
                default_fail = 0.20
                default_sym = True
            elif preset == "fast_triage":
                default_pass = 0.80
                default_fail = 0.20
                default_sym = False
            else:
                raise ConfigurationError(
                    f"Invalid preset '{preset}'. Must be one of ['strict', 'balanced', 'fast_triage']"
                )
        else:
            default_pass = LogitGateDefaults.pass_threshold
            default_fail = LogitGateDefaults.fail_threshold
            default_sym = True

        raw_pass_thresh = (
            pass_threshold
            if pass_threshold is not None
            else (
                explicit_profile.pass_threshold
                if explicit_profile is not None and explicit_profile.pass_threshold is not None
                else (
                    resolved_meta.pass_threshold
                    if resolved_meta is not None and resolved_meta.pass_threshold is not None
                    else (
                        loaded_profile.pass_threshold
                        if loaded_profile is not None and loaded_profile.pass_threshold is not None
                        else (
                            adaptive_controller.base_pass_threshold
                            if adaptive_controller is not None
                            else default_pass
                        )
                    )
                )
            )
        )
        raw_fail_thresh = (
            fail_threshold
            if fail_threshold is not None
            else (
                explicit_profile.fail_threshold
                if explicit_profile is not None and explicit_profile.fail_threshold is not None
                else (
                    resolved_meta.fail_threshold
                    if resolved_meta is not None and resolved_meta.fail_threshold is not None
                    else (
                        loaded_profile.fail_threshold
                        if loaded_profile is not None and loaded_profile.fail_threshold is not None
                        else (
                            adaptive_controller.base_fail_threshold
                            if adaptive_controller is not None
                            else default_fail
                        )
                    )
                )
            )
        )
        effective_sym = (
            default_sym if symmetric_debiasing is None else bool(symmetric_debiasing)
        )

        if not isinstance(rubric, str) or not rubric.strip():
            raise ConfigurationError("rubric cannot be empty or whitespace")
        effective_pass_thresh = _validate_prob_threshold("pass_threshold", raw_pass_thresh)
        effective_fail_thresh = _validate_prob_threshold("fail_threshold", raw_fail_thresh)
        if effective_pass_thresh <= effective_fail_thresh:
            raise ConfigurationError(
                f"pass_threshold ({effective_pass_thresh}) must be strictly greater than fail_threshold ({effective_fail_thresh})"
            )

        if max_positional_bias is not None:
            if isinstance(max_positional_bias, bool) or not isinstance(max_positional_bias, (int, float)):
                raise ConfigurationError(
                    f"max_positional_bias must be a positive float, got {max_positional_bias!r}"
                )
            bias_val = float(max_positional_bias)
            if not math.isfinite(bias_val) or bias_val <= 0.0:
                raise ConfigurationError(
                    f"max_positional_bias must be a positive float, got {max_positional_bias!r}"
                )
        else:
            bias_val = None
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
        resolved_backend = _call_resolve_logit_backend(backend, model=model)
        owns_sidecar = False
        resolved_sidecar: LogitBackend | None = None
        eff_sidecar_model: str | None = None
        sidecar_temp = 1.0
        try:
            if sidecar_backend is not None:
                owns_sidecar = not isinstance(sidecar_backend, LogitBackend)
                resolved_sidecar = _call_resolve_logit_backend(sidecar_backend, model=sidecar_model)
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
                    binary=True,
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
                binary=True,
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
        self.preset = preset
        self.save_on_fail = save_on_fail
        self.pass_threshold = effective_pass_thresh
        self.fail_threshold = effective_fail_thresh
        self.symmetric_debiasing = effective_sym
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.context = context
        self.query = query
        self.on_unsupported_logprobs = on_unsupported_logprobs
        self.sidecar_backend = resolved_sidecar
        self.sidecar_model = eff_sidecar_model
        self.sidecar_temperature = sidecar_temp
        self._owns_sidecar_backend = owns_sidecar
        self._needs_lazy_calibration = (
            False
            if (resolved_meta is not None or resolved_unanchored_meta is not None)
            else needs_lazy
        )
        self._explicit_thresholds = (
            pass_threshold is not None
            or fail_threshold is not None
            or explicit_profile is not None
            or preset is not None
        )
        self._calibration_lock = threading.Lock()
        self.max_positional_bias = bias_val
        self.warm_prefix = effective_warm_prefix
        self.adaptive_controller = adaptive_controller
        self.anchor_bank = resolved_anchors
        self.meta_calibrator = resolved_meta
        self.unanchored_meta_calibrator = resolved_unanchored_meta
        self.distillation_bundle = resolved_bundle
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
            "rubric": self.rubric,
            "backend": infer_backend_spec(self.backend),
            "pass_threshold": self.pass_threshold,
            "fail_threshold": self.fail_threshold,
            "symmetric_debiasing": self.symmetric_debiasing,
            "query_first": self.query_first,
            "logit_temperature": self.logit_temperature,
            "preset": self.preset,
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
        if self.max_positional_bias is not None:
            config["max_positional_bias"] = self.max_positional_bias
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
            gate_type="LogitGate",
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
            f"LogitGate(rubric={rubric_repr}, "
            f"pass_threshold={self.pass_threshold}, "
            f"fail_threshold={self.fail_threshold}, "
            f"backend={self.backend!r})"
        )

    def _binary_prompts_with_anchor_flag(
        self,
        candidate: Any,
        rubrics: Sequence[str] | None = None,
        rubric_index: int = 0,
        *,
        boundary_nonce: str | None = None,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
    ) -> tuple[str, str | None, bool]:
        target_r = self.rubric if rubrics is None else rubrics[rubric_index]
        anchor_str = (
            self.anchor_bank.format_anchor_block(
                target_r,
                candidate,
                boundary_nonce=boundary_nonce,
            )
            if self.anchor_bank is not None
            else None
        )
        was_anchored = bool(anchor_str and anchor_str.strip())

        def build(pass_token: str, fail_token: str) -> str:
            if rubrics is None:
                return compile_binary_prompt(
                    rubric=self.rubric,
                    candidate=candidate,
                    pass_token=pass_token,
                    fail_token=fail_token,
                    query_first=self.query_first,
                    boundary_nonce=boundary_nonce,
                    anchor_block=anchor_str,
                    context=context,
                    query=query,
                )
            return compile_batch_binary_prompt(
                rubrics=rubrics,
                candidate=candidate,
                rubric_index=rubric_index,
                pass_token=pass_token,
                fail_token=fail_token,
                query_first=False,
                boundary_nonce=boundary_nonce,
                anchor_block=anchor_str,
                context=context,
                query=query,
            )

        return build("A", "B"), (build("B", "A") if self.symmetric_debiasing else None), was_anchored

    def _binary_prompts(
        self,
        candidate: Any,
        rubrics: Sequence[str] | None = None,
        rubric_index: int = 0,
        *,
        boundary_nonce: str | None = None,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
    ) -> tuple[str, str | None]:
        fwd, rev, _ = self._binary_prompts_with_anchor_flag(
            candidate,
            rubrics,
            rubric_index,
            boundary_nonce=boundary_nonce,
            context=context,
            query=query,
        )
        return fwd, rev

    def _query(self, prompt: str) -> list[dict[str, Any]]:
        return self.backend.query_top_logprobs(prompt, top_k=10, allowed_tokens=["A", "B"])

    async def _aquery_pair(
        self, forward: str, reverse: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        if reverse is None:
            return await self.backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"]), None
        if _resolve_warm_prefix(self.warm_prefix, self.backend):
            top_fwd = await self.backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"])
            top_rev = await self.backend.aquery_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"])
            return top_fwd, top_rev
        top_fwd, top_rev = await asyncio.gather(
            self.backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"]),
            self.backend.aquery_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"]),
        )
        return top_fwd, top_rev

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
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )
        return self._handle_failure_snapshot(candidate, verdict, **kwargs)

    def _execute_queries(
        self, forward: str, reverse: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, bool, str | None, float]:
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
            top_fwd = self.sidecar_backend.query_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"])
            top_rev = (
                self.sidecar_backend.query_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"])
                if reverse is not None
                else None
            )
            spec = infer_backend_spec(self.sidecar_backend)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

        try:
            top_fwd = self._query(forward)
            top_rev = self._query(reverse) if reverse is not None else None
            return top_fwd, top_rev, False, None, self.logit_temperature
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs in ("raise", "escalate"):
                raise
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'") from exc
            top_fwd = self.sidecar_backend.query_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"])
            top_rev = (
                self.sidecar_backend.query_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"])
                if reverse is not None
                else None
            )
            spec = infer_backend_spec(self.sidecar_backend)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

    async def _aquery_sidecar(
        self, forward: str, reverse: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        if self.sidecar_backend is None:
            raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'")
        if reverse is None:
            return await self.sidecar_backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"]), None
        if _resolve_warm_prefix(self.warm_prefix, self.sidecar_backend):
            top_fwd = await self.sidecar_backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"])
            top_rev = await self.sidecar_backend.aquery_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"])
            return top_fwd, top_rev
        top_fwd, top_rev = await asyncio.gather(
            self.sidecar_backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"]),
            self.sidecar_backend.aquery_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"]),
        )
        return top_fwd, top_rev

    async def _aexecute_queries(
        self, forward: str, reverse: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, bool, str | None, float]:
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
            top_fwd, top_rev = await self._aquery_sidecar(forward, reverse)
            spec = infer_backend_spec(self.sidecar_backend)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

        try:
            top_fwd, top_rev = await self._aquery_pair(forward, reverse)
            return top_fwd, top_rev, False, None, self.logit_temperature
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs in ("raise", "escalate"):
                raise
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'") from exc
            top_fwd, top_rev = await self._aquery_sidecar(forward, reverse)
            spec = infer_backend_spec(self.sidecar_backend)
            return top_fwd, top_rev, True, spec, self.sidecar_temperature

    def _verdict_from_logprobs(
        self,
        top_fwd: list[dict[str, Any]],
        top_rev: list[dict[str, Any]] | None,
        t0: float,
        *,
        anchored: bool = False,
        sidecar_used: bool = False,
        sidecar_spec: str | None = None,
        effective_temp: float | None = None,
    ) -> Verdict:
        _ensure_logprobs_present(top_fwd, top_rev)
        fwd_has_target = has_any_target_token(top_fwd, ("A", "B"))
        rev_has_target = True if top_rev is None else has_any_target_token(top_rev, ("A", "B"))
        l1, p1 = compute_binary_log_odds(top_fwd, pass_token="A", fail_token="B")
        temp = max(effective_temp if effective_temp is not None else self.logit_temperature, 1e-6)
        l2: float | None = None
        p2: float | None = None
        pos_bias: float | None = None
        active_mc = (
            self.meta_calibrator
            if anchored
            else (self.unanchored_meta_calibrator or self.meta_calibrator)
        )
        if top_rev is not None:
            l2, p2 = compute_binary_log_odds(top_rev, pass_token="B", fail_token="A")
            pos_bias = 0.5 * abs(l1 - l2)
            if active_mc is not None:
                from logitgate.distill.meta_calibrator import extract_logit_features

                feat_vec = extract_logit_features(top_fwd, top_rev)
                log_odds_pass = active_mc.predict_log_odds(feat_vec)
                calibrated_prob = _clamped_sigmoid(log_odds_pass)
            else:
                calibrated_prob = combine_symmetric_log_odds(
                    forward_log_odds=l1,
                    reverse_log_odds=l2,
                    logit_temperature=temp,
                )
                log_odds_pass = (l1 + l2) / (2.0 * temp)
        else:
            if active_mc is not None:
                from logitgate.distill.meta_calibrator import extract_logit_features

                feat_vec = extract_logit_features(top_fwd, None)
                log_odds_pass = active_mc.predict_log_odds(feat_vec)
                calibrated_prob = _clamped_sigmoid(log_odds_pass)
            else:
                log_odds_pass = l1 / temp
                calibrated_prob = _clamped_sigmoid(log_odds_pass)

        shannon_entropy = compute_shannon_entropy(calibrated_prob)
        latency = (time.perf_counter() - t0) * 1000.0
        prob_rounded = round(calibrated_prob, 4)

        aci = self.adaptive_controller
        aci_theta: float | None = None
        eff_fail_thresh = self.fail_threshold
        eff_pass_thresh = self.pass_threshold
        if (
            aci is None
            and not self._explicit_thresholds
            and active_mc is not None
            and active_mc.pass_threshold is not None
            and active_mc.fail_threshold is not None
            and active_mc.pass_threshold > active_mc.fail_threshold
        ):
            eff_pass_thresh = active_mc.pass_threshold
            eff_fail_thresh = active_mc.fail_threshold
        eff_fail_lo = 0.0
        eff_pass_lo = 0.0
        force_ambig = False
        if aci is not None:
            (
                aci_theta,
                eff_fail_thresh,
                eff_pass_thresh,
                eff_fail_lo,
                eff_pass_lo,
                force_ambig,
            ) = aci.current_decision_state()

        abstained_reason: str | None = None
        if not fwd_has_target or not rev_has_target:
            status = DecisionStatus.AMBIGUOUS
            abstained_reason = "unobserved_target_tokens"
        elif (
            self.max_positional_bias is not None
            and pos_bias is not None
            and l2 is not None
            and ((l1 > 0.0 and l2 < 0.0) or (l1 < 0.0 and l2 > 0.0))
            and pos_bias > self.max_positional_bias
        ):
            status = DecisionStatus.AMBIGUOUS
            abstained_reason = f"positional_bias_exceeded ({pos_bias:.4f} > {self.max_positional_bias:.4f})"
        elif force_ambig:
            status = DecisionStatus.AMBIGUOUS
            abstained_reason = "adaptive_conformal_band_saturated"
        elif aci is not None and aci_theta is not None and aci_theta > 0.0:
            if prob_rounded >= eff_pass_thresh and log_odds_pass >= eff_pass_lo:
                status = DecisionStatus.PASSED
            elif prob_rounded <= eff_fail_thresh and log_odds_pass <= eff_fail_lo:
                status = DecisionStatus.FAILED
            else:
                status = DecisionStatus.AMBIGUOUS
                if prob_rounded >= eff_pass_thresh or prob_rounded <= eff_fail_thresh:
                    abstained_reason = "adaptive_log_odds_margin_unmet"
        elif prob_rounded >= eff_pass_thresh:
            status = DecisionStatus.PASSED
        elif prob_rounded <= eff_fail_thresh:
            status = DecisionStatus.FAILED
        else:
            status = DecisionStatus.AMBIGUOUS

        bias_rounded = round(pos_bias, 4) if pos_bias is not None else None
        meta: dict[str, Any] = {
            "positional_bias_magnitude": bias_rounded,
            "log_odds_pass": round(log_odds_pass, 4),
            "shannon_entropy": round(shannon_entropy, 4),
        }
        if sidecar_used:
            meta["sidecar_used"] = True
            if sidecar_spec is not None:
                meta["sidecar_backend"] = sidecar_spec
        if active_mc is not None:
            meta["meta_calibrated"] = True
        if self.anchor_bank is not None:
            meta["anchored"] = anchored
        if aci is not None and aci_theta is not None:
            meta["adaptive_theta"] = round(aci_theta, 4)
            meta["adaptive_fail_threshold"] = eff_fail_thresh
            meta["adaptive_pass_threshold"] = eff_pass_thresh
            meta["raw_l1_status"] = status.value
        if abstained_reason is not None:
            meta["abstained_reason"] = abstained_reason
        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            calibrated_prob=prob_rounded,
            forward_prob=round(p1, 4),
            reverse_prob=round(p2, 4) if p2 is not None else None,
            forward_log_odds=round(l1, 4),
            reverse_log_odds=round(l2, 4) if l2 is not None else None,
            positional_bias_magnitude=bias_rounded,
            log_odds_pass=round(log_odds_pass, 4),
            shannon_entropy=round(shannon_entropy, 4),
            tokens_generated=0,
            metadata=meta,
        )
        reason_suffix = f" [{abstained_reason}]" if abstained_reason is not None else ""
        return Verdict(
            passed=status == DecisionStatus.PASSED,
            status=status,
            probability=prob_rounded,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=(
                f"LogitGate prob={prob_rounded:.4f} "
                f"(pass_thresh={eff_pass_thresh}, fail_thresh={eff_fail_thresh}){reason_suffix}"
            ),
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
            self.logit_temperature = _call_lazy_calibrate_sync(self.model, self.backend, binary=True)
            self._needs_lazy_calibration = False

    async def _aensure_calibrated(self) -> None:
        if not self._needs_lazy_calibration or not self.model:
            return
        if not LogitGateDefaults.auto_calibrate or is_temperature_configured():
            self._needs_lazy_calibration = False
            return
        temp = await _call_lazy_calibrate_async(self.model, self.backend, binary=True)
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
        nonce = secrets.token_hex(8)
        forward, reverse, was_anchored = self._binary_prompts_with_anchor_flag(
            candidate, boundary_nonce=nonce, context=eff_context, query=eff_query
        )
        try:
            top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = self._execute_queries(forward, reverse)
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
                anchored=was_anchored,
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
        nonce = secrets.token_hex(8)
        forward, reverse, was_anchored = self._binary_prompts_with_anchor_flag(
            candidate, boundary_nonce=nonce, context=eff_context, query=eff_query
        )
        try:
            top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = await self._aexecute_queries(forward, reverse)
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
                anchored=was_anchored,
                sidecar_used=sidecar_used,
                sidecar_spec=sidecar_spec,
                effective_temp=eff_temp,
            ),
            **kwargs,
        )

    def evaluate_batch(  # type: ignore[override]
        self,
        candidate: Any = None,
        rubrics: Sequence[str] | None = None,
        *,
        candidates: Sequence[Any] | None = None,
        fail_fast: bool = False,
        on_error: BatchErrorPolicy = "raise",
        **kwargs: Any,
    ) -> list[Verdict]:
        """Evaluates several atomic rubrics against one candidate, or a batch of candidates.

        When rubrics is provided, evaluates the single candidate against rubrics sharing
        one cached prefix. When rubrics is None, evaluates the batch of candidates.
        """
        if on_error not in ("raise", "abstain", "fail"):
            raise ConfigurationError(
                f"on_error must be one of 'raise', 'abstain', or 'fail', got {on_error!r}"
            )
        # Candidate-batch mode (rubrics is None)
        if rubrics is None:
            target_candidates = candidates if candidates is not None else candidate
            if target_candidates is None:
                return []
            if not isinstance(target_candidates, Sequence) or isinstance(target_candidates, (str, bytes, dict)):
                target_candidates = [target_candidates]
            return super().evaluate_batch(target_candidates, on_error=on_error, **kwargs)

        # Rubric-batch mode: evaluating one candidate against multiple rubrics
        if not isinstance(fail_fast, bool):
            raise ConfigurationError(f"fail_fast must be a bool, got {fail_fast!r}")
        if isinstance(rubrics, (str, bytes)):
            raise ConfigurationError("rubrics must be a sequence of rubric strings, not a single string")
        for r in rubrics:
            if not isinstance(r, str) or not r.strip():
                raise ConfigurationError("Each rubric in rubrics must be a non-empty string")
        if not rubrics:
            return []
        eff_context = kwargs.get("context", self.context) if "context" in kwargs else self.context
        eff_query = kwargs.get("query", self.query) if "query" in kwargs else self.query
        _validate_rag_context_and_query(eff_context, eff_query)

        self._ensure_calibrated()
        nonce = secrets.token_hex(8)
        results = []
        for idx in range(len(rubrics)):
            t0 = time.perf_counter()
            if on_error == "raise":
                forward, reverse, was_anchored = self._binary_prompts_with_anchor_flag(
                    candidate, rubrics, rubric_index=idx, boundary_nonce=nonce, context=eff_context, query=eff_query
                )
                try:
                    top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = self._execute_queries(forward, reverse)
                    verdict = self._verdict_from_logprobs(
                        top_fwd,
                        top_rev,
                        t0,
                        anchored=was_anchored,
                        sidecar_used=sidecar_used,
                        sidecar_spec=sidecar_spec,
                        effective_temp=eff_temp,
                    )
                except UnsupportedLogprobsError as exc:
                    if self.on_unsupported_logprobs == "escalate":
                        verdict = self._escalate_unsupported(candidate, str(exc), t0, kwargs)
                    else:
                        raise
            else:
                try:
                    forward, reverse, was_anchored = self._binary_prompts_with_anchor_flag(
                        candidate, rubrics, rubric_index=idx, boundary_nonce=nonce, context=eff_context, query=eff_query
                    )
                    top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = self._execute_queries(forward, reverse)
                    verdict = self._verdict_from_logprobs(
                        top_fwd,
                        top_rev,
                        t0,
                        anchored=was_anchored,
                        sidecar_used=sidecar_used,
                        sidecar_spec=sidecar_spec,
                        effective_temp=eff_temp,
                    )
                except UnsupportedLogprobsError as exc:
                    if self.on_unsupported_logprobs == "escalate":
                        verdict = self._escalate_unsupported(candidate, str(exc), t0, kwargs)
                    else:
                        verdict = _make_batch_error_verdict(self, exc, on_error, t0)
                except Exception as exc:
                    verdict = _make_batch_error_verdict(self, exc, on_error, t0)
            results.append(verdict)
            if fail_fast and verdict.status == DecisionStatus.FAILED:
                break
        return results

    async def aevaluate_batch(  # type: ignore[override]
        self,
        candidate: Any = None,
        rubrics: Sequence[str] | None = None,
        *,
        candidates: Sequence[Any] | None = None,
        fail_fast: bool = False,
        max_concurrency: int | None = None,
        on_error: BatchErrorPolicy = "raise",
        **kwargs: Any,
    ) -> list[Verdict]:
        """Async evaluate_batch.

        When rubrics is provided, evaluates the single candidate against rubrics warming
        the shared prefix first. When rubrics is None, evaluates the batch of candidates.
        """
        if on_error not in ("raise", "abstain", "fail"):
            raise ConfigurationError(
                f"on_error must be one of 'raise', 'abstain', or 'fail', got {on_error!r}"
            )
        if (
            max_concurrency is not None
            and (
                isinstance(max_concurrency, bool)
                or not isinstance(max_concurrency, int)
                or max_concurrency < 1
            )
        ):
            raise ConfigurationError(
                f"max_concurrency must be an integer >= 1 or None, got {max_concurrency!r}"
            )
        # Candidate-batch mode (rubrics is None)
        if rubrics is None:
            target_candidates = candidates if candidates is not None else candidate
            if target_candidates is None:
                return []
            if not isinstance(target_candidates, Sequence) or isinstance(target_candidates, (str, bytes, dict)):
                target_candidates = [target_candidates]
            return await super().aevaluate_batch(
                target_candidates, max_concurrency=max_concurrency, on_error=on_error, **kwargs
            )

        # Rubric-batch mode:
        if not isinstance(fail_fast, bool):
            raise ConfigurationError(f"fail_fast must be a bool, got {fail_fast!r}")
        if isinstance(rubrics, (str, bytes)):
            raise ConfigurationError("rubrics must be a sequence of rubric strings, not a single string")
        for r in rubrics:
            if not isinstance(r, str) or not r.strip():
                raise ConfigurationError("Each rubric in rubrics must be a non-empty string")
        if not rubrics:
            return []
        eff_context = kwargs.get("context", self.context) if "context" in kwargs else self.context
        eff_query = kwargs.get("query", self.query) if "query" in kwargs else self.query
        _validate_rag_context_and_query(eff_context, eff_query)

        await self._aensure_calibrated()
        nonce = secrets.token_hex(8)

        sem_limit = max_concurrency if max_concurrency is not None else 32
        sem = asyncio.Semaphore(sem_limit)

        async def _eval_single(idx: int) -> Verdict:
            async with sem:
                t0 = time.perf_counter()
                if on_error == "raise":
                    forward, reverse, was_anchored = self._binary_prompts_with_anchor_flag(
                        candidate, rubrics, rubric_index=idx, boundary_nonce=nonce, context=eff_context, query=eff_query
                    )
                    try:
                        top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = await self._aexecute_queries(forward, reverse)
                        return self._verdict_from_logprobs(
                            top_fwd,
                            top_rev,
                            t0,
                            anchored=was_anchored,
                            sidecar_used=sidecar_used,
                            sidecar_spec=sidecar_spec,
                            effective_temp=eff_temp,
                        )
                    except UnsupportedLogprobsError as exc:
                        if self.on_unsupported_logprobs == "escalate":
                            return self._escalate_unsupported(candidate, str(exc), t0, kwargs)
                        raise
                try:
                    forward, reverse, was_anchored = self._binary_prompts_with_anchor_flag(
                        candidate, rubrics, rubric_index=idx, boundary_nonce=nonce, context=eff_context, query=eff_query
                    )
                    top_fwd, top_rev, sidecar_used, sidecar_spec, eff_temp = await self._aexecute_queries(forward, reverse)
                    return self._verdict_from_logprobs(
                        top_fwd,
                        top_rev,
                        t0,
                        anchored=was_anchored,
                        sidecar_used=sidecar_used,
                        sidecar_spec=sidecar_spec,
                        effective_temp=eff_temp,
                    )
                except UnsupportedLogprobsError as exc:
                    if self.on_unsupported_logprobs == "escalate":
                        return self._escalate_unsupported(candidate, str(exc), t0, kwargs)
                    return _make_batch_error_verdict(self, exc, on_error, t0)
                except Exception as exc:
                    return _make_batch_error_verdict(self, exc, on_error, t0)

        first_verdict = await _eval_single(0)
        if fail_fast:
            if first_verdict.status == DecisionStatus.FAILED:
                return [first_verdict]
            results = [first_verdict]
            for idx in range(1, len(rubrics)):
                verdict = await _eval_single(idx)
                results.append(verdict)
                if verdict.status == DecisionStatus.FAILED:
                    break
            return results
        rest_verdicts = await asyncio.gather(*[_eval_single(idx) for idx in range(1, len(rubrics))])
        return [first_verdict, *rest_verdicts]


def __getattr__(name: str) -> Any:
    if name == "ChoiceGate":
        from logitgate.gates.choice_gate import ChoiceGate

        globals()["ChoiceGate"] = ChoiceGate
        return ChoiceGate
    if name == "ScoreGate":
        from logitgate.gates.score_gate import ScoreGate

        globals()["ScoreGate"] = ScoreGate
        return ScoreGate
    if name == "MultiRubricGate":
        from logitgate.gates.multi_rubric_gate import MultiRubricGate

        globals()["MultiRubricGate"] = MultiRubricGate
        return MultiRubricGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ChoiceGate",
    "LogitGate",
    "MultiRubricGate",
    "ScoreGate",
    "TopLogprobs",
    "UnsupportedLogprobsPolicy",
    "_cal_lock_key",
    "_ensure_logprobs_present",
    "_get_async_cal_lock",
    "_get_sync_cal_lock",
    "_lazy_calibrate_async",
    "_lazy_calibrate_sync",
    "_permuted_key_indices",
    "_resolve_gate_temperature",
    "_resolve_logit_temperature",
    "_resolve_warm_prefix",
    "_same_nonzero_sign",
    "_should_lazy_auto_calibrate",
    "_validate_optional_bool",
    "_validate_optional_unit_float",
    "_validate_prob_threshold",
    "_validate_rag_context_and_query",
    "_warn_if_profile_model_mismatch",
    "resolve_logit_backend",
]
