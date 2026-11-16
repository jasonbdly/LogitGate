from __future__ import annotations

import asyncio
import math
import threading
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Literal

from logitgate.backends.base import LogitBackend, StubBackend
from logitgate.config import (
    LogitGateDefaults,
    is_temperature_configured,
)
from logitgate.debiasing.softmax import (
    has_finite_logprobs,
)
from logitgate.exceptions import BackendError, CalibrationWarning, ConfigurationError

if TYPE_CHECKING:
    from logitgate.calibration.profile import CalibrationProfile

TopLogprobs = list[dict[str, Any]]


def _ensure_logprobs_present(top_fwd: TopLogprobs, top_rev: TopLogprobs | None = None) -> None:
    if not has_finite_logprobs(top_fwd) or (top_rev is not None and not has_finite_logprobs(top_rev)):
        raise BackendError(
            "Backend returned no token log-probabilities (top_logprobs is empty). "
            "Ensure the model and inference server support logprobs."
        )


def _same_nonzero_sign(a: float, b: float) -> bool:
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def _validate_prob_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be between 0.0 and 1.0, got {value!r}")
    vf = float(value)
    if not math.isfinite(vf) or not (0.0 <= vf <= 1.0):
        raise ConfigurationError(f"{name} must be between 0.0 and 1.0, got {value!r}")
    return vf


def _validate_optional_bool(name: str, value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a bool or None, got {value!r}")
    return value


def _validate_optional_unit_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a float in (0.0, 1.0], got {value!r}")
    vf = float(value)
    if not math.isfinite(vf) or not (0.0 < vf <= 1.0):
        raise ConfigurationError(f"{name} must be a float in (0.0, 1.0], got {value!r}")
    return vf


def _resolve_warm_prefix(warm_prefix: bool | None, backend: Any) -> bool:
    if warm_prefix is not None:
        return warm_prefix
    return bool(getattr(backend, "prefers_sequential_prefix_warmup", False))


_SYNC_CAL_META_LOCK = threading.Lock()
_SYNC_CAL_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_ASYNC_CAL_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str], asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _cal_lock_key(model_name: str) -> tuple[str, str]:
    from logitgate.debiasing.permutation import (
        _cache_key_for_path,
        get_calibration_cache_path,
    )

    return _cache_key_for_path(get_calibration_cache_path()), model_name.strip().lower()


def _get_sync_cal_lock(key: tuple[str, str]) -> threading.Lock:
    with _SYNC_CAL_META_LOCK:
        lock = _SYNC_CAL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SYNC_CAL_LOCKS[key] = lock
        return lock


def _get_async_cal_lock(key: tuple[str, str]) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _SYNC_CAL_META_LOCK:
        loop_locks = _ASYNC_CAL_LOCKS.get(loop)
        if loop_locks is None:
            loop_locks = {}
            _ASYNC_CAL_LOCKS[loop] = loop_locks
        lock = loop_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[key] = lock
        return lock


def _resolve_logit_temperature(
    logit_temperature: float | Literal["auto"] | None,
    effective_model: str | None,
    backend: Any,
    *,
    binary: bool = False,
) -> float:
    """Resolves logit temperature from explicit value, 'auto', global config, or model calibration."""
    if logit_temperature == "auto":
        from logitgate.debiasing.permutation import (
            auto_calibrate_backend,
            get_model_temperature,
            has_known_or_cached_temperature,
        )

        if has_known_or_cached_temperature(effective_model, binary=binary):
            temp = get_model_temperature(effective_model, binary=binary)
        elif effective_model not in ("default", "stub", None, ""):
            if not isinstance(effective_model, str):
                raise TypeError(f"expected effective_model to be a str, got {type(effective_model)}")
            cal_lock = _get_sync_cal_lock(_cal_lock_key(effective_model))
            with cal_lock:
                if has_known_or_cached_temperature(effective_model, binary=binary):
                    temp = get_model_temperature(effective_model, binary=binary)
                else:
                    cal_res = auto_calibrate_backend(
                        backend,
                        model_key=effective_model,
                        strict_cache_io=False,
                    )
                    temp = (
                        cal_res.optimal_temperature
                        if binary
                        else (cal_res.multiclass_temperature or cal_res.optimal_temperature)
                    )
        else:
            temp = 1.0
        effective_temp = temp
    elif logit_temperature is not None:
        if isinstance(logit_temperature, bool) or not isinstance(logit_temperature, (int, float)):
            raise ConfigurationError(f"logit_temperature must be a positive float, got {logit_temperature!r}")
        try:
            effective_temp = float(logit_temperature)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"logit_temperature must be a positive float, got {logit_temperature!r}") from exc
    else:
        from logitgate.debiasing.permutation import get_model_temperature

        if is_temperature_configured():
            effective_temp = LogitGateDefaults.logit_temperature
        else:
            effective_temp = get_model_temperature(effective_model, binary=binary)

    if not math.isfinite(effective_temp) or effective_temp <= 0.0:
        raise ConfigurationError(f"logit_temperature must be positive, got {effective_temp}")
    return effective_temp


def _should_lazy_auto_calibrate(
    logit_temperature: float | Literal["auto"] | None,
    effective_model: str | None,
    backend: Any,
    *,
    binary: bool = False,
) -> bool:
    """Returns True when an unknown BYO model should lazily calibrate on first live evaluation."""
    if logit_temperature is not None:
        return False
    if is_temperature_configured() or not LogitGateDefaults.auto_calibrate:
        return False
    if not effective_model or effective_model.strip().lower() in ("", "default", "stub"):
        return False
    if not isinstance(backend, LogitBackend):
        return False
    inner = getattr(backend, "inner", backend)
    if isinstance(inner, StubBackend):
        return False
    from logitgate.debiasing.permutation import has_known_or_cached_temperature

    return not has_known_or_cached_temperature(effective_model, binary=binary)


def _lazy_calibrate_sync(model_name: str, backend: LogitBackend, *, binary: bool) -> float:
    from logitgate.debiasing.permutation import (
        auto_calibrate_backend,
        get_model_temperature,
        has_known_or_cached_temperature,
    )

    if has_known_or_cached_temperature(model_name, binary=binary):
        return get_model_temperature(model_name, binary=binary)
    with _get_sync_cal_lock(_cal_lock_key(model_name)):
        if has_known_or_cached_temperature(model_name, binary=binary):
            return get_model_temperature(model_name, binary=binary)
        cal = auto_calibrate_backend(backend, model_key=model_name, strict_cache_io=False)
        return cal.optimal_temperature if binary else (cal.multiclass_temperature or cal.optimal_temperature)


async def _lazy_calibrate_async(model_name: str, backend: LogitBackend, *, binary: bool) -> float:
    from logitgate.debiasing.permutation import (
        aauto_calibrate_backend,
        get_model_temperature,
        has_known_or_cached_temperature,
    )

    if has_known_or_cached_temperature(model_name, binary=binary):
        return get_model_temperature(model_name, binary=binary)
    async with _get_async_cal_lock(_cal_lock_key(model_name)):
        if has_known_or_cached_temperature(model_name, binary=binary):
            return get_model_temperature(model_name, binary=binary)
        cal = await aauto_calibrate_backend(backend, model_key=model_name, strict_cache_io=False)
        return cal.optimal_temperature if binary else (cal.multiclass_temperature or cal.optimal_temperature)


def _warn_if_profile_model_mismatch(
    profile: CalibrationProfile | None,
    effective_model: str | None,
    gate_name: str,
) -> None:
    if profile is None or not profile.model or not effective_model:
        return
    prof_m = profile.model.strip().lower()
    eff_m = effective_model.strip().lower()
    if not prof_m or prof_m in ("default", "stub") or not eff_m or eff_m in ("default", "stub"):
        return
    if prof_m != eff_m:
        warnings.warn(
            f"{gate_name} was initialized for model '{effective_model}', but CalibrationProfile "
            f"was fitted on '{profile.model}'. Thresholds and temperatures may not transfer faithfully.",
            CalibrationWarning,
            stacklevel=3,
        )


def _resolve_gate_temperature(
    logit_temperature: float | Literal["auto"] | None,
    effective_model: str | None,
    backend: Any,
    profile: CalibrationProfile | None,
    *,
    binary: bool,
) -> tuple[float, bool]:
    """Resolves gate temperature and lazy-calibration flag respecting explicit > profile > defaults."""
    if logit_temperature is None and profile is not None:
        temp = profile.binary_temperature if binary else profile.effective_multiclass_temperature
        return temp, False
    temp = _resolve_logit_temperature(
        logit_temperature,
        effective_model,
        backend,
        binary=binary,
    )
    needs_lazy = _should_lazy_auto_calibrate(
        logit_temperature,
        effective_model,
        backend,
        binary=binary,
    )
    return temp, needs_lazy


def _permuted_key_indices(
    n: int,
    *,
    symmetric_debiasing: bool = True,
    permutation_mode: str = "derangement",
) -> list[list[int]]:
    """Returns the list of pass permutations where perm[i] is the key index assigned to option i."""
    identity = list(range(n))
    if not symmetric_debiasing:
        return [identity]
    if permutation_mode == "cyclic":
        return [[(i + s) % n for i in range(n)] for s in range(n)]
    if permutation_mode == "reverse":
        return [identity, [n - 1 - i for i in range(n)]]
    shift = max(1, n // 2)
    return [identity, [(i + shift) % n for i in range(n)]]


__all__ = [
    "TopLogprobs",
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
    "_warn_if_profile_model_mismatch",
]
