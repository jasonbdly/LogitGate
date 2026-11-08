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
        if val is None:
            return 0.85
        try:
            return float(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_PASS_THRESHOLD={val!r}: must be a float") from err

    @pass_threshold.setter
    def pass_threshold(cls, val: float | None) -> None:
        _state.pass_threshold = val

    @property
    def fail_threshold(cls) -> float:
        if _state.fail_threshold is not None:
            return _state.fail_threshold
        val = os.environ.get("LOGITGATE_FAIL_THRESHOLD")
        if val is None:
            return 0.20
        try:
            return float(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_FAIL_THRESHOLD={val!r}: must be a float") from err

    @fail_threshold.setter
    def fail_threshold(cls, val: float | None) -> None:
        _state.fail_threshold = val

    @property
    def logit_temperature(cls) -> float:
        if _state.logit_temperature is not None:
            return _state.logit_temperature
        val = os.environ.get("LOGITGATE_TEMPERATURE")
        if val is None:
            return 1.0
        try:
            return float(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_TEMPERATURE={val!r}: must be a float") from err

    @logit_temperature.setter
    def logit_temperature(cls, val: float | None) -> None:
        _state.logit_temperature = val

    @property
    def num_ctx(cls) -> int | None:
        if _state.num_ctx is not None:
            return _state.num_ctx
        val = os.environ.get("LOGITGATE_NUM_CTX")
        if val is None:
            return None
        try:
            return int(val)
        except ValueError as err:
            from logitgate.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid LOGITGATE_NUM_CTX={val!r}: must be an integer") from err

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
    global _warned_implicit_stub
    _state.reset()
    _warned_implicit_stub = False
    from logitgate.snapshot import reset_snapshot_notices

    reset_snapshot_notices()


_IMPLICIT_STUB = "stub://default"
_warned_implicit_stub = False


def is_backend_implicit() -> bool:
    """True when neither `configure(backend=...)` nor ``LOGITGATE_BACKEND`` chose a backend."""
    return _state.backend is None and "LOGITGATE_BACKEND" not in os.environ


def warn_if_implicit_stub(spec: str) -> None:
    """Warns once if ``spec`` is the stub and nobody explicitly asked for it.

    Without this, ``LogitGate("Is this factual?")("anything")`` returns P=0.95 and looks like
    it works. That is the worst first impression a first-time user can get.
    """
    global _warned_implicit_stub
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
