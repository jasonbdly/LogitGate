"""
Exception hierarchy for LogitGate.

All custom exceptions inherit from `LogitGateError`. Where applicable,
exceptions also subclass standard Python built-ins (such as `ValueError` or
`RuntimeError`) to maintain 100% backward compatibility with callers catching standard exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logitgate.models import Verdict


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


class TelemetryWarning(UserWarning):
    """Emitted when a telemetry or distillation sink fails in non-fatal ``on_error='warn'`` mode."""


class ContractViolationError(LogitGateError, ValueError):
    """Raised when an AST or ContractGate encounters an unrecoverable structural violation."""


class CacheWarning(UserWarning):
    """Emitted when a persistent VerdictCache encounters a non-fatal storage warning."""


class StreamAbortedError(LogitGateError, RuntimeError):
    """Raised when `StreamGuard` aborts a token stream due to a failing guardrail check.

    Attributes:
        verdict: The `Verdict` that triggered the stream termination.
        partial_text: Full accumulated stream text observed up to the abort point.
        yielded_text: Text already delivered downstream before termination (empty or shorter
            than ``partial_text`` in ``delivery='buffered'`` mode).
        chunk_index: Zero-based index of the chunk at which the abort occurred.
        is_final: ``True`` if aborted during the end-of-stream check rather than mid-stream.
    """

    def __init__(
        self,
        verdict: Verdict,
        *,
        partial_text: str,
        yielded_text: str = "",
        chunk_index: int = 0,
        is_final: bool = False,
        message: str | None = None,
    ) -> None:
        stage = "final" if is_final else "mid-stream"
        detail = verdict.reason or verdict.status.value
        msg = message or (
            f"Stream aborted at chunk {chunk_index} ({stage}, {len(partial_text)} chars): {detail}"
        )
        super().__init__(msg)
        self.verdict = verdict
        self.partial_text = partial_text
        self.yielded_text = yielded_text
        self.chunk_index = chunk_index
        self.is_final = is_final
