from __future__ import annotations

import os


class _ConfigState:
    def __init__(self) -> None:
        self.backend: str | None = None
        self.model: str | None = None
        self.pass_threshold: float | None = None
        self.fail_threshold: float | None = None
        self.logit_temperature: float | None = None
        self.num_ctx: int | None = None
        self.save_failures: bool | None = None
        self.snapshot_dir: str | None = None

    def reset(self) -> None:
        self.backend = None
        self.model = None
        self.pass_threshold = None
        self.fail_threshold = None
        self.logit_temperature = None
        self.num_ctx = None
        self.save_failures = None
        self.snapshot_dir = None


_state = _ConfigState()


class _DefaultsMeta(type):
    @property
    def backend(cls) -> str:
        if _state.backend is not None:
            return _state.backend
        return os.environ.get("LOGITGATE_BACKEND", "stub://default")

    @backend.setter
    def backend(cls, val: str | None) -> None:
        _state.backend = val

    @property
    def model(cls) -> str | None:
        if _state.model is not None:
            return _state.model
        return os.environ.get("LOGITGATE_MODEL", None)

    @model.setter
    def model(cls, val: str | None) -> None:
        _state.model = val

    @property
    def pass_threshold(cls) -> float:
        if _state.pass_threshold is not None:
            return _state.pass_threshold
        val = os.environ.get("LOGITGATE_PASS_THRESHOLD")
        return float(val) if val is not None else 0.85

    @pass_threshold.setter
    def pass_threshold(cls, val: float | None) -> None:
        _state.pass_threshold = val

    @property
    def fail_threshold(cls) -> float:
        if _state.fail_threshold is not None:
            return _state.fail_threshold
        val = os.environ.get("LOGITGATE_FAIL_THRESHOLD")
        return float(val) if val is not None else 0.20

    @fail_threshold.setter
    def fail_threshold(cls, val: float | None) -> None:
        _state.fail_threshold = val

    @property
    def logit_temperature(cls) -> float:
        if _state.logit_temperature is not None:
            return _state.logit_temperature
        val = os.environ.get("LOGITGATE_TEMPERATURE")
        return float(val) if val is not None else 1.0

    @logit_temperature.setter
    def logit_temperature(cls, val: float | None) -> None:
        _state.logit_temperature = val

    @property
    def num_ctx(cls) -> int | None:
        if _state.num_ctx is not None:
            return _state.num_ctx
        val = os.environ.get("LOGITGATE_NUM_CTX")
        return int(val) if val is not None else None

    @num_ctx.setter
    def num_ctx(cls, val: int | None) -> None:
        _state.num_ctx = val

    @property
    def save_failures(cls) -> bool:
        if _state.save_failures is not None:
            return _state.save_failures
        return os.environ.get("LOGITGATE_SAVE_FAILURES", "0").lower() in ("1", "true", "yes")

    @save_failures.setter
    def save_failures(cls, val: bool | None) -> None:
        _state.save_failures = val

    @property
    def snapshot_dir(cls) -> str:
        if _state.snapshot_dir is not None:
            return _state.snapshot_dir
        return os.environ.get("LOGITGATE_SNAPSHOT_DIR", ".logitgate/failures")

    @snapshot_dir.setter
    def snapshot_dir(cls, val: str | None) -> None:
        _state.snapshot_dir = val


class LogitGateDefaults(metaclass=_DefaultsMeta):
    """Global configuration defaults for LogitGate."""


def configure(
    backend: str | None = None,
    model: str | None = None,
    pass_threshold: float | None = None,
    fail_threshold: float | None = None,
    logit_temperature: float | None = None,
    num_ctx: int | None = None,
    save_failures: bool | None = None,
    snapshot_dir: str | None = None,
) -> None:
    """Configures global defaults for LogitGate."""
    if backend is not None:
        _state.backend = backend
    if model is not None:
        _state.model = model
    if pass_threshold is not None:
        _state.pass_threshold = pass_threshold
    if fail_threshold is not None:
        _state.fail_threshold = fail_threshold
    if logit_temperature is not None:
        _state.logit_temperature = logit_temperature
    if num_ctx is not None:
        _state.num_ctx = num_ctx
    if save_failures is not None:
        _state.save_failures = save_failures
    if snapshot_dir is not None:
        _state.snapshot_dir = snapshot_dir


def reset_config() -> None:
    """Resets global configuration to environment variables and defaults."""
    _state.reset()


def get_default_backend() -> str:
    """Resolves default backend string via configure() or LOGITGATE_BACKEND."""
    return LogitGateDefaults.backend


def get_default_model() -> str | None:
    """Resolves default model string via configure() or LOGITGATE_MODEL."""
    return LogitGateDefaults.model


def get_default_save_failures() -> bool:
    """Resolves default failure saving flag via configure() or LOGITGATE_SAVE_FAILURES."""
    return LogitGateDefaults.save_failures


def get_default_snapshot_dir() -> str:
    """Resolves default snapshot directory via configure() or LOGITGATE_SNAPSHOT_DIR."""
    return LogitGateDefaults.snapshot_dir
