from __future__ import annotations

import math
import os
import threading
from typing import Any


class _ConfigState:
    def __init__(self) -> None:
        self.backend: str | None = None
        self.model: str | None = None
        self.pass_threshold: float | None = None
        self.fail_threshold: float | None = None
        self.logit_temperature: float | None = None
        self.auto_calibrate: bool | None = None
        self.num_ctx: int | None = None
        self.save_failures: bool | None = None
        self.snapshot_dir: str | None = None

    def reset(self) -> None:
        self.backend = None
        self.model = None
        self.pass_threshold = None
        self.fail_threshold = None
        self.logit_temperature = None
        self.auto_calibrate = None
        self.num_ctx = None
        self.save_failures = None
        self.snapshot_dir = None


_config_lock = threading.RLock()
_state = _ConfigState()


def _validate_threshold(name: str, val: Any) -> float:
    from logitgate.exceptions import ConfigurationError

    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ConfigurationError(f"Invalid {name}={val!r}: must be a float in [0.0, 1.0]")
    vf = float(val)
    if not math.isfinite(vf) or not (0.0 <= vf <= 1.0):
        raise ConfigurationError(f"Invalid {name}={val!r}: must be between 0.0 and 1.0")
    return vf


def _validate_temperature(name: str, val: Any) -> float:
    from logitgate.exceptions import ConfigurationError

    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ConfigurationError(f"Invalid {name}={val!r}: must be a positive finite float")
    vf = float(val)
    if not math.isfinite(vf) or vf <= 0.0:
        raise ConfigurationError(f"Invalid {name}={val!r}: must be a positive finite float")
    return vf


def _validate_num_ctx(name: str, val: Any) -> int:
    from logitgate.exceptions import ConfigurationError

    if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise ConfigurationError(f"Invalid {name}={val!r}: must be an integer >= 1")
    return val


def _validate_bool(name: str, val: Any) -> bool:
    from logitgate.exceptions import ConfigurationError

    if not isinstance(val, bool):
        raise ConfigurationError(f"Invalid {name}={val!r}: must be a boolean")
    return val


class _DefaultsMeta(type):
    @property
    def backend(cls) -> str:
        with _config_lock:
            if _state.backend is not None:
                return _state.backend
            return os.environ.get("LOGITGATE_BACKEND", "stub://default")

    @backend.setter
    def backend(cls, val: str | None) -> None:
        with _config_lock:
            _state.backend = val

    @property
    def model(cls) -> str | None:
        with _config_lock:
            if _state.model is not None:
                return _state.model
            return os.environ.get("LOGITGATE_MODEL", None)

    @model.setter
    def model(cls, val: str | None) -> None:
        with _config_lock:
            _state.model = val

    @property
    def pass_threshold(cls) -> float:
        with _config_lock:
            if _state.pass_threshold is not None:
                return _state.pass_threshold
            val = os.environ.get("LOGITGATE_PASS_THRESHOLD")
        if val is None:
            return 0.85
        try:
            vf = float(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_PASS_THRESHOLD={val!r}: must be a float") from err
        return _validate_threshold("LOGITGATE_PASS_THRESHOLD", vf)

    @pass_threshold.setter
    def pass_threshold(cls, val: float | None) -> None:
        validated = None if val is None else _validate_threshold("pass_threshold", val)
        with _config_lock:
            _state.pass_threshold = validated

    @property
    def fail_threshold(cls) -> float:
        with _config_lock:
            if _state.fail_threshold is not None:
                return _state.fail_threshold
            val = os.environ.get("LOGITGATE_FAIL_THRESHOLD")
        if val is None:
            return 0.20
        try:
            vf = float(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_FAIL_THRESHOLD={val!r}: must be a float") from err
        return _validate_threshold("LOGITGATE_FAIL_THRESHOLD", vf)

    @fail_threshold.setter
    def fail_threshold(cls, val: float | None) -> None:
        validated = None if val is None else _validate_threshold("fail_threshold", val)
        with _config_lock:
            _state.fail_threshold = validated

    @property
    def logit_temperature(cls) -> float:
        with _config_lock:
            if _state.logit_temperature is not None:
                return _state.logit_temperature
            val = os.environ.get("LOGITGATE_TEMPERATURE")
        if val is None:
            return 1.0
        try:
            vf = float(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_TEMPERATURE={val!r}: must be a float") from err
        return _validate_temperature("LOGITGATE_TEMPERATURE", vf)

    @logit_temperature.setter
    def logit_temperature(cls, val: float | None) -> None:
        validated = None if val is None else _validate_temperature("logit_temperature", val)
        with _config_lock:
            _state.logit_temperature = validated

    @property
    def auto_calibrate(cls) -> bool:
        with _config_lock:
            if _state.auto_calibrate is not None:
                return _state.auto_calibrate
            raw = os.environ.get("LOGITGATE_AUTO_CALIBRATE", "1")
        return raw.strip().lower() not in ("0", "false", "no", "off")

    @auto_calibrate.setter
    def auto_calibrate(cls, val: bool | None) -> None:
        validated = None if val is None else _validate_bool("auto_calibrate", val)
        with _config_lock:
            _state.auto_calibrate = validated

    @property
    def num_ctx(cls) -> int | None:
        with _config_lock:
            if _state.num_ctx is not None:
                return _state.num_ctx
            val = os.environ.get("LOGITGATE_NUM_CTX")
        if val is None:
            return None
        try:
            parsed = int(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_NUM_CTX={val!r}: must be an integer") from err
        return _validate_num_ctx("LOGITGATE_NUM_CTX", parsed)

    @num_ctx.setter
    def num_ctx(cls, val: int | None) -> None:
        validated = None if val is None else _validate_num_ctx("num_ctx", val)
        with _config_lock:
            _state.num_ctx = validated

    @property
    def save_failures(cls) -> bool:
        with _config_lock:
            if _state.save_failures is not None:
                return _state.save_failures
            return os.environ.get("LOGITGATE_SAVE_FAILURES", "0").lower() in ("1", "true", "yes")

    @save_failures.setter
    def save_failures(cls, val: bool | None) -> None:
        validated = None if val is None else _validate_bool("save_failures", val)
        with _config_lock:
            _state.save_failures = validated

    @property
    def snapshot_dir(cls) -> str:
        with _config_lock:
            if _state.snapshot_dir is not None:
                return _state.snapshot_dir
            return os.environ.get("LOGITGATE_SNAPSHOT_DIR", ".logitgate/failures")

    @snapshot_dir.setter
    def snapshot_dir(cls, val: str | None) -> None:
        with _config_lock:
            _state.snapshot_dir = val


class LogitGateDefaults(metaclass=_DefaultsMeta):
    """Global configuration defaults for LogitGate."""


def configure(
    backend: str | None = None,
    model: str | None = None,
    pass_threshold: float | None = None,
    fail_threshold: float | None = None,
    logit_temperature: float | None = None,
    auto_calibrate: bool | None = None,
    num_ctx: int | None = None,
    save_failures: bool | None = None,
    snapshot_dir: str | None = None,
) -> None:
    """Configures process-wide defaults for LogitGate gates and backends.

    Args:
        backend: Default backend URI specifier (e.g. ``"ollama://localhost:11434"``).
        model: Default model identifier passed to gates when omitted per gate.
        pass_threshold: Default pass probability threshold in ``[0.0, 1.0]``.
        fail_threshold: Default fail probability threshold in ``[0.0, 1.0]``.
        logit_temperature: Explicit global temperature override (``> 0``). Setting this
            overrides built-in model temperatures and suppresses lazy BYO auto-calibration.
        auto_calibrate: Enables lazy BYO model micro-calibration on unknown models.
        num_ctx: Optional context-window size (``>= 1``) forwarded to local backends.
        save_failures: Automatically persist replayable snapshots on failed verdicts.
        snapshot_dir: Directory path where failure snapshots are written.

    Raises:
        ConfigurationError: If any supplied threshold, temperature, boolean flag, or
            context-window value is invalid or inconsistent (``pass_threshold <= fail_threshold``).
    """
    from logitgate.exceptions import ConfigurationError

    val_pass = None if pass_threshold is None else _validate_threshold("pass_threshold", pass_threshold)
    val_fail = None if fail_threshold is None else _validate_threshold("fail_threshold", fail_threshold)
    val_temp = None if logit_temperature is None else _validate_temperature("logit_temperature", logit_temperature)
    val_auto = None if auto_calibrate is None else _validate_bool("auto_calibrate", auto_calibrate)
    val_ctx = None if num_ctx is None else _validate_num_ctx("num_ctx", num_ctx)
    val_save = None if save_failures is None else _validate_bool("save_failures", save_failures)

    with _config_lock:
        eff_pass = val_pass if val_pass is not None else (_state.pass_threshold if _state.pass_threshold is not None else 0.85)
        eff_fail = val_fail if val_fail is not None else (_state.fail_threshold if _state.fail_threshold is not None else 0.20)
        if (val_pass is not None or val_fail is not None) and eff_pass <= eff_fail:
            raise ConfigurationError(
                f"pass_threshold ({eff_pass}) must be strictly greater than fail_threshold ({eff_fail})"
            )
        if backend is not None:
            _state.backend = backend
        if model is not None:
            _state.model = model
        if val_pass is not None:
            _state.pass_threshold = val_pass
        if val_fail is not None:
            _state.fail_threshold = val_fail
        if val_temp is not None:
            _state.logit_temperature = val_temp
        if val_auto is not None:
            _state.auto_calibrate = val_auto
        if val_ctx is not None:
            _state.num_ctx = val_ctx
        if val_save is not None:
            _state.save_failures = val_save
        if snapshot_dir is not None:
            _state.snapshot_dir = snapshot_dir


def reset_config() -> None:
    """Resets global configuration to environment variables and defaults."""
    global _warned_implicit_stub
    with _config_lock:
        _state.reset()
        _warned_implicit_stub = False
    from logitgate.debiasing.permutation import reset_runtime_calibration_cache
    from logitgate.snapshot import reset_snapshot_notices

    reset_runtime_calibration_cache()
    reset_snapshot_notices()


_IMPLICIT_STUB = "stub://default"
_warned_implicit_stub = False


def is_temperature_configured() -> bool:
    """Returns True if ``logit_temperature`` was explicitly set via `configure()` or ``LOGITGATE_TEMPERATURE``."""
    with _config_lock:
        if _state.logit_temperature is not None:
            return True
        return "LOGITGATE_TEMPERATURE" in os.environ


def is_backend_implicit() -> bool:
    """True when neither `configure(backend=...)` nor ``LOGITGATE_BACKEND`` chose a backend."""
    with _config_lock:
        return _state.backend is None and "LOGITGATE_BACKEND" not in os.environ


def warn_if_implicit_stub(spec: str) -> None:
    """Warns once if ``spec`` is the stub and nobody explicitly asked for it.

    Without this, ``LogitGate("Is this factual?")("anything")`` returns P=0.95 and looks like
    it works. That is the worst first impression a first-time user can get.
    """
    global _warned_implicit_stub
    with _config_lock:
        if _warned_implicit_stub or spec != _IMPLICIT_STUB or not is_backend_implicit():
            return
        _warned_implicit_stub = True
    import warnings

    from logitgate.exceptions import StubBackendWarning

    warnings.warn(
        "No LogitGate backend configured; using the offline stub, which returns a fixed "
        "P(pass)=0.95 for every input. For real evaluations set LOGITGATE_BACKEND "
        "(e.g. 'ollama://localhost:11434/qwen2.5:7b', 'vllm://localhost:8000/v1') or call "
        "logitgate.configure(backend=...). Set LOGITGATE_BACKEND=stub:// to use the stub "
        "intentionally and silence this warning.",
        StubBackendWarning,
        stacklevel=4,
    )


def get_default_backend() -> str:
    """Resolves default backend string via configure() or LOGITGATE_BACKEND."""
    backend = LogitGateDefaults.backend
    warn_if_implicit_stub(backend)
    return backend


def get_default_model() -> str | None:
    """Resolves default model string via configure() or LOGITGATE_MODEL."""
    return LogitGateDefaults.model


def get_default_save_failures() -> bool:
    """Resolves default failure saving flag via configure() or LOGITGATE_SAVE_FAILURES."""
    return LogitGateDefaults.save_failures


def get_default_snapshot_dir() -> str:
    """Resolves default snapshot directory via configure() or LOGITGATE_SNAPSHOT_DIR."""
    return LogitGateDefaults.snapshot_dir


def get_default_auto_calibrate() -> bool:
    """Resolves default lazy auto-calibration flag via configure() or LOGITGATE_AUTO_CALIBRATE."""
    return LogitGateDefaults.auto_calibrate
