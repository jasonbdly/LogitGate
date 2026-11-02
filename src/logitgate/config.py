from __future__ import annotations

import os
from typing import Any, Optional


class _ConfigState:
    def __init__(self) -> None:
        self.backend: Optional[str] = None
        self.model: Optional[str] = None
        self.pass_threshold: Optional[float] = None
        self.fail_threshold: Optional[float] = None
        self.logit_temperature: Optional[float] = None

    def reset(self) -> None:
        self.backend = None
        self.model = None
        self.pass_threshold = None
        self.fail_threshold = None
        self.logit_temperature = None


_state = _ConfigState()


class _DefaultsMeta(type):
    @property
    def backend(cls) -> str:
        if _state.backend is not None:
            return _state.backend
        return os.environ.get("LOGITGATE_BACKEND", "stub://default")

    @backend.setter
    def backend(cls, val: Optional[str]) -> None:
        _state.backend = val

    @property
    def model(cls) -> Optional[str]:
        if _state.model is not None:
            return _state.model
        return os.environ.get("LOGITGATE_MODEL", None)

    @model.setter
    def model(cls, val: Optional[str]) -> None:
        _state.model = val

    @property
    def pass_threshold(cls) -> float:
        if _state.pass_threshold is not None:
            return _state.pass_threshold
        val = os.environ.get("LOGITGATE_PASS_THRESHOLD")
        return float(val) if val is not None else 0.85

    @pass_threshold.setter
    def pass_threshold(cls, val: Optional[float]) -> None:
        _state.pass_threshold = val

    @property
    def fail_threshold(cls) -> float:
        if _state.fail_threshold is not None:
            return _state.fail_threshold
        val = os.environ.get("LOGITGATE_FAIL_THRESHOLD")
        return float(val) if val is not None else 0.20

    @fail_threshold.setter
    def fail_threshold(cls, val: Optional[float]) -> None:
        _state.fail_threshold = val

    @property
    def logit_temperature(cls) -> float:
        if _state.logit_temperature is not None:
            return _state.logit_temperature
        val = os.environ.get("LOGITGATE_TEMPERATURE")
        return float(val) if val is not None else 1.0

    @logit_temperature.setter
    def logit_temperature(cls, val: Optional[float]) -> None:
        _state.logit_temperature = val


class LogitGateDefaults(metaclass=_DefaultsMeta):
    """Global configuration defaults for LogitGate."""
    pass


def configure(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    pass_threshold: Optional[float] = None,
    fail_threshold: Optional[float] = None,
    logit_temperature: Optional[float] = None,
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


def reset_config() -> None:
    """Resets global configuration to environment variables and defaults."""
    _state.reset()


def get_default_backend() -> str:
    """Resolves default backend string via configure() or LOGITGATE_BACKEND."""
    return LogitGateDefaults.backend


def get_default_model() -> Optional[str]:
    """Resolves default model string via configure() or LOGITGATE_MODEL."""
    return LogitGateDefaults.model
