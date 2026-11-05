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


class CalibrationError(LogitGateError, ValueError):
    """Raised when post-hoc temperature calibration fails or receives degenerate data."""


class ContractViolationError(LogitGateError, ValueError):
    """Raised when an AST or ContractGate encounters an unrecoverable structural violation."""
