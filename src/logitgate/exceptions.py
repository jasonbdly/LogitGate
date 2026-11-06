"""
Exception hierarchy for LogitGate.

All custom exceptions inherit from `LogitGateError`. Where applicable,
exceptions also subclass standard Python built-ins (such as `ValueError` or
`RuntimeError`) to maintain 100% backward compatibility with callers catching standard exceptions.
"""

from __future__ import annotations


class LogitGateError(Exception):
    """Base exception for all LogitGate errors."""


class ConfigurationError(LogitGateError, ValueError):
    """Raised when gate, cascade, or global configuration parameters are invalid."""


class BackendError(LogitGateError, RuntimeError):
    """Raised when an inference backend fails during execution, connection, or response parsing."""


class BackendUnavailableError(BackendError, ConnectionError):
    """The inference server could not be reached (connection refused, DNS failure, wrong port).

    Also a `ConnectionError`, so generic ``except ConnectionError`` retry logic keeps working.
    The original ``httpx`` exception is preserved as ``__cause__``.
    """


class BackendTimeoutError(BackendError, TimeoutError):
    """The server accepted the connection but didn't answer in time (often a cold model load)."""


class StubBackendWarning(UserWarning):
    """Emitted once per process when no backend is configured and the offline stub is used.

    The stub returns a fixed P(pass) for every input, which is right for CI plumbing and wrong
    for anything else. Configure a real backend, or set ``LOGITGATE_BACKEND=stub://`` to say
    the stub is intentional, which silences this warning.
    """


class CalibrationError(LogitGateError, ValueError):
    """Raised when post-hoc temperature calibration fails or receives degenerate data."""


class CalibrationWarning(UserWarning):
    """A calibration result is usable but does not carry its advertised statistical guarantee."""


class ContractViolationError(LogitGateError, ValueError):
    """Raised when an AST or ContractGate encounters an unrecoverable structural violation."""
