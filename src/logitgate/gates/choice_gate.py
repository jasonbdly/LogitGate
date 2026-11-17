from __future__ import annotations

import asyncio
import math
import secrets
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logitgate.backends.base import LogitBackend
from logitgate.calibration.lifecycle import (
    TopLogprobs,
    _permuted_key_indices,
    _resolve_gate_temperature,
    _resolve_warm_prefix,
    _validate_optional_bool,
    _validate_optional_unit_float,
    _warn_if_profile_model_mismatch,
)
from logitgate.config import (
    LogitGateDefaults,
    get_default_model,
    is_temperature_configured,
)
from logitgate.debiasing.inversion import (
    compile_choice_prompt,
)
from logitgate.debiasing.softmax import (
    extract_token_logprob,
    has_any_target_token,
    has_finite_logprobs,
)
from logitgate.debiasing.softmax import (
    softmax_probabilities as _softmax,
)
from logitgate.exceptions import BackendError, ConfigurationError, UnsupportedLogprobsError
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


class ChoiceGate(BaseGate):
    """Discrete K-ary categorical decision and routing gate.

    Evaluates K discrete options from first-token logits, cancelling label order
    and position bias via symmetric permutation debiasing (Zheng et al., 2023).

    Args:
        question: Decision question or routing prompt.
        options: Mapping of option tokens (e.g., 'A', 'B', 'C') to their descriptions.
        backend: Backend instance or specifier URI.
        model: Optional model identifier string for the backend.
        symmetric_debiasing: When True, queries options in multiple label permutations
            and averages logprobs before softmax normalization.
        permutation_mode: Multi-option debiasing schedule ('derangement',
            'cyclic', or 'reverse').
        query_first: When True, compiles prompt with question and candidate first.
        logit_temperature: Temperature scaling factor, 'auto' to calibrate via
            micro-probes, or None to use global/model-calibrated defaults (lazy-calibrating
            unknown BYO models when auto_calibrate=True).
        name: Identifier for this gate instance.
        min_confidence: Optional minimum top-choice probability required to avoid abstaining (AMBIGUOUS).
        min_margin: Optional minimum top-2 probability margin required to avoid abstaining (AMBIGUOUS).
        profile: Optional CalibrationProfile (or path to JSON) supplying calibrated
            multiclass temperature and selective confidence/margin cutoffs.

    Raises:
        ConfigurationError: If question is empty, options has fewer than 2 items, or logit_temperature <= 0.
    """

    def __init__(
        self,
        question: str,
        options: Mapping[str, str],
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        symmetric_debiasing: bool | None = True,
        permutation_mode: Literal["derangement", "cyclic", "reverse"] = "derangement",
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        name: str = "ChoiceGate",
        *,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
        on_unsupported_logprobs: UnsupportedLogprobsPolicy = "raise",
        sidecar_backend: str | LogitBackend | None = None,
        sidecar_model: str | None = None,
        min_confidence: float | None = None,
        min_margin: float | None = None,
        save_on_fail: bool | None = None,
        warm_prefix: bool | None = None,
        profile: CalibrationProfile | str | Path | None = None,
        telemetry_sink: Any = None,
        cache: Any = None,
    ) -> None:
        from logitgate.calibration.profile import resolve_calibration_profile
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        loaded_profile = resolve_calibration_profile(profile)
        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        if not isinstance(question, str) or not question.strip():
            raise ConfigurationError("question cannot be empty or whitespace")
        if permutation_mode not in ("derangement", "cyclic", "reverse"):
            raise ConfigurationError(
                f"Invalid permutation_mode '{permutation_mode}'. Must be one of ['derangement', 'cyclic', 'reverse']"
            )
        if not isinstance(options, Mapping) or len(options) < 2:
            raise ConfigurationError(
                f"ChoiceGate requires at least 2 options, got {len(options) if isinstance(options, Mapping) else type(options).__name__}"
            )
        invalid_keys = [k for k in options if not isinstance(k, str) or len(k.strip()) != 1]
        if invalid_keys:
            raise ConfigurationError(
                f"ChoiceGate option keys must be single characters (e.g. 'A', 'B', 'C'), got: {invalid_keys}"
            )
        raw_min_conf = (
            min_confidence
            if min_confidence is not None
            else (loaded_profile.choice_min_confidence if loaded_profile is not None else None)
        )
        raw_min_margin = (
            min_margin
            if min_margin is not None
            else (loaded_profile.choice_min_margin if loaded_profile is not None else None)
        )
        effective_min_conf = _validate_optional_unit_float("min_confidence", raw_min_conf)
        effective_min_margin = _validate_optional_unit_float("min_margin", raw_min_margin)
        effective_save_on_fail = _validate_optional_bool("save_on_fail", save_on_fail)
        effective_warm_prefix = _validate_optional_bool("warm_prefix", warm_prefix)
        normalized_options: dict[str, str] = {}
        seen_upper: set[str] = set()
        for k, v in options.items():
            if not isinstance(v, str) or not v.strip():
                raise ConfigurationError(f"ChoiceGate option description for '{k}' must be a non-empty string")
            k_clean = k.strip()
            k_upper = k_clean.upper()
            if k_upper in seen_upper:
                raise ConfigurationError(
                    f"ChoiceGate option keys must be case-insensitively unique, got duplicate for '{k_clean}'"
                )
            seen_upper.add(k_upper)
            normalized_options[k_clean] = v

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
        self.question = question.strip()
        self.options = normalized_options
        self.symmetric_debiasing = True if symmetric_debiasing is None else bool(symmetric_debiasing)
        self.permutation_mode = permutation_mode
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
        self.min_confidence = effective_min_conf
        self.min_margin = effective_min_margin
        self.save_on_fail = effective_save_on_fail
        self.warm_prefix = effective_warm_prefix
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
            "gate_type": "ChoiceGate",
            "question": self.question,
            "options": dict(self.options),
            "backend": infer_backend_spec(self.backend),
            "symmetric_debiasing": self.symmetric_debiasing,
            "permutation_mode": self.permutation_mode,
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
        if self.min_confidence is not None:
            config["min_confidence"] = self.min_confidence
        if self.min_margin is not None:
            config["min_margin"] = self.min_margin
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
            gate_type="ChoiceGate",
            rubric=self.question,
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
            rubric=self.question,
            gate_config=self.replay_config(),
            save_on_fail=kwargs.get("save_on_fail", self.save_on_fail),
            test_name=kwargs.get("test_name"),
            snapshot_dir=kwargs.get("snapshot_dir"),
        )

    def __repr__(self) -> str:
        q_repr = repr(self.question[:40] + "..." if len(self.question) > 40 else self.question)
        return f"ChoiceGate(question={q_repr}, options={list(self.options.keys())!r}, backend={self.backend!r})"

    def _permuted_key_indices(self) -> list[list[int]]:
        """Returns the list of pass permutations where perm[i] is the key index assigned to option i."""
        return _permuted_key_indices(
            len(self.options),
            symmetric_debiasing=self.symmetric_debiasing,
            permutation_mode=self.permutation_mode,
        )

    def _choice_prompts_list(
        self,
        candidate: Any,
        *,
        boundary_nonce: str | None = None,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
    ) -> list[str]:
        keys = list(self.options.keys())
        values = list(self.options.values())
        n = len(keys)
        prompts: list[str] = []
        for perm in self._permuted_key_indices():
            # perm[i] is the key index where option i's description is placed
            slot_to_desc: dict[str, str] = {}
            for key_idx in range(n):
                opt_idx = perm.index(key_idx)
                slot_to_desc[keys[key_idx]] = values[opt_idx]
            prompts.append(
                compile_choice_prompt(
                    question=self.question,
                    candidate=candidate,
                    options=slot_to_desc,
                    query_first=self.query_first,
                    boundary_nonce=boundary_nonce,
                    context=context,
                    query=query,
                )
            )
        return prompts

    def _choice_prompts(
        self,
        candidate: Any,
        *,
        context: str | Sequence[str] | None = None,
        query: str | None = None,
    ) -> tuple[str, str | None]:
        prompts = self._choice_prompts_list(candidate, context=context, query=query)
        return prompts[0], (prompts[1] if len(prompts) > 1 else None)

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
            choice=None,
            choice_probabilities=None,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )
        return self._handle_failure_snapshot(candidate, verdict, **kwargs)

    def _execute_queries(
        self, prompts: list[str], keys: list[str], top_k: int
    ) -> tuple[list[TopLogprobs], bool, str | None, float]:
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
            top_list = [
                self.sidecar_backend.query_top_logprobs(p, top_k=top_k, allowed_tokens=keys)
                for p in prompts
            ]
            spec = infer_backend_spec(self.sidecar_backend)
            return top_list, True, spec, self.sidecar_temperature

        try:
            top_list = [
                backend.query_top_logprobs(p, top_k=top_k, allowed_tokens=keys)
                for p in prompts
            ]
            return top_list, False, None, self.logit_temperature
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs in ("raise", "escalate"):
                raise
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'") from exc
            top_list = [
                self.sidecar_backend.query_top_logprobs(p, top_k=top_k, allowed_tokens=keys)
                for p in prompts
            ]
            spec = infer_backend_spec(self.sidecar_backend)
            return top_list, True, spec, self.sidecar_temperature

    async def _aexecute_queries(
        self, prompts: list[str], keys: list[str], top_k: int
    ) -> tuple[list[TopLogprobs], bool, str | None, float]:
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
            if len(prompts) == 1:
                top_list = [await target_backend.aquery_top_logprobs(prompts[0], top_k=top_k, allowed_tokens=keys)]
            elif _resolve_warm_prefix(self.warm_prefix, target_backend):
                first = await target_backend.aquery_top_logprobs(prompts[0], top_k=top_k, allowed_tokens=keys)
                rest = list(
                    await asyncio.gather(
                        *[target_backend.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=keys) for p in prompts[1:]]
                    )
                )
                top_list = [first, *rest]
            else:
                top_list = list(
                    await asyncio.gather(
                        *[target_backend.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=keys) for p in prompts]
                    )
                )
            return top_list, sidecar_used, spec, used_temp
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs in ("raise", "escalate"):
                raise
            if self.sidecar_backend is None:
                raise ConfigurationError("sidecar_backend is required when on_unsupported_logprobs='sidecar'") from exc
            sb = self.sidecar_backend
            if len(prompts) == 1:
                top_list = [await sb.aquery_top_logprobs(prompts[0], top_k=top_k, allowed_tokens=keys)]
            elif _resolve_warm_prefix(self.warm_prefix, sb):
                first = await sb.aquery_top_logprobs(prompts[0], top_k=top_k, allowed_tokens=keys)
                rest = list(
                    await asyncio.gather(
                        *[sb.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=keys) for p in prompts[1:]]
                    )
                )
                top_list = [first, *rest]
            else:
                top_list = list(
                    await asyncio.gather(
                        *[sb.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=keys) for p in prompts]
                    )
                )
            spec = infer_backend_spec(sb)
            return top_list, True, spec, self.sidecar_temperature

    def _verdict_from_logprobs_list(
        self,
        top_list: list[TopLogprobs],
        t0: float,
        *,
        sidecar_used: bool = False,
        sidecar_spec: str | None = None,
        effective_temp: float | None = None,
    ) -> Verdict:
        if not top_list or any(not has_finite_logprobs(top) for top in top_list):
            raise BackendError(
                "Backend returned no token log-probabilities (top_logprobs is empty). "
                "Ensure the model and inference server support logprobs."
            )
        keys = list(self.options)
        perms = self._permuted_key_indices()[: len(top_list)]
        accum = {k: 0.0 for k in keys}
        for top, perm in zip(top_list, perms, strict=False):
            # In this pass, option i sits under keys[perm[i]]
            pass_raw = {keys[i]: extract_token_logprob(top, keys[perm[i]]) for i in range(len(keys))}
            m_pass = max(pass_raw.values())
            lse_pass = m_pass + math.log(sum(math.exp(v - m_pass) for v in pass_raw.values()))
            for k in keys:
                accum[k] += (pass_raw[k] - lse_pass) / float(len(perms))

        used_temp = max(effective_temp if effective_temp is not None else self.logit_temperature, 1e-6)
        choice_probs = _softmax(accum, used_temp)
        best_choice, max_prob = max(choice_probs.items(), key=lambda kv: kv[1])
        prob_rounded = round(max_prob, 4)
        rounded = {k: round(v, 4) for k, v in choice_probs.items()}
        sorted_probs = sorted(choice_probs.values(), reverse=True)
        margin = round(sorted_probs[0] - sorted_probs[1], 4) if len(sorted_probs) >= 2 else 1.0
        latency = (time.perf_counter() - t0) * 1000.0

        all_have_targets = all(has_any_target_token(top, keys) for top in top_list)
        abstained_reason: str | None = None
        if not all_have_targets:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice: str | None = None
            abstained_reason = "unobserved_target_tokens"
            reason = "ChoiceGate abstained (AMBIGUOUS): unobserved_target_tokens"
        elif self.min_confidence is not None and prob_rounded < self.min_confidence:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice = best_choice
            abstained_reason = f"below_min_confidence ({prob_rounded:.4f} < {self.min_confidence:.4f})"
            reason = (
                f"ChoiceGate abstained on '{best_choice}' "
                f"(P={prob_rounded:.4f} < min_confidence={self.min_confidence:.4f})"
            )
        elif self.min_margin is not None and margin < self.min_margin:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice = best_choice
            abstained_reason = f"below_min_margin ({margin:.4f} < {self.min_margin:.4f})"
            reason = (
                f"ChoiceGate abstained on '{best_choice}' "
                f"(margin={margin:.4f} < min_margin={self.min_margin:.4f})"
            )
        else:
            status = DecisionStatus.PASSED
            passed = True
            selected_choice = best_choice
            reason = f"ChoiceGate selected '{best_choice}' with P={max_prob:.4f}"

        meta: dict[str, Any] = {
            "choice_probabilities": rounded,
            "permutation_mode": self.permutation_mode,
            "top2_margin": margin,
        }
        if sidecar_used:
            meta["sidecar_used"] = True
            if sidecar_spec is not None:
                meta["sidecar_backend"] = sidecar_spec
        if abstained_reason is not None:
            meta["abstained_reason"] = abstained_reason

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            calibrated_prob=prob_rounded,
            metadata=meta,
        )
        return Verdict(
            passed=passed,
            status=status,
            probability=prob_rounded,
            choice=selected_choice,
            choice_probabilities=rounded,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )

    def _verdict_from_logprobs(self, top_fwd: TopLogprobs, top_rev: TopLogprobs | None, t0: float) -> Verdict:
        top_list = [top_fwd] if top_rev is None else [top_fwd, top_rev]
        return self._verdict_from_logprobs_list(top_list, t0)

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
        keys = list(self.options)
        top_k = max(10, 2 * len(keys))
        nonce = secrets.token_hex(8)
        prompts = self._choice_prompts_list(
            candidate, boundary_nonce=nonce, context=eff_context, query=eff_query
        )
        try:
            top_list, sidecar_used, sidecar_spec, eff_temp = self._execute_queries(prompts, keys, top_k)
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs == "escalate":
                return self._escalate_unsupported(candidate, str(exc), t0, kwargs)
            raise
        return self._handle_failure_snapshot(
            candidate,
            self._verdict_from_logprobs_list(
                top_list,
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
        keys = list(self.options)
        top_k = max(10, 2 * len(keys))
        nonce = secrets.token_hex(8)
        prompts = self._choice_prompts_list(
            candidate, boundary_nonce=nonce, context=eff_context, query=eff_query
        )
        try:
            top_list, sidecar_used, sidecar_spec, eff_temp = await self._aexecute_queries(prompts, keys, top_k)
        except UnsupportedLogprobsError as exc:
            if self.on_unsupported_logprobs == "escalate":
                return self._escalate_unsupported(candidate, str(exc), t0, kwargs)
            raise
        return self._handle_failure_snapshot(
            candidate,
            self._verdict_from_logprobs_list(
                top_list,
                t0,
                sidecar_used=sidecar_used,
                sidecar_spec=sidecar_spec,
                effective_temp=eff_temp,
            ),
            **kwargs,
        )


__all__ = ["ChoiceGate"]
