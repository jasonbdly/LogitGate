from __future__ import annotations

import abc
import json
import math
import os
import stat
import statistics
import threading
import warnings
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal

from typing_extensions import Self

from logitgate.debiasing.inversion import render_candidate
from logitgate.exceptions import ConfigurationError, TelemetryWarning
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.telemetry.models import (
    VALID_ERROR_POLICIES,
    VALID_PRIVACY_MODES,
    ErrorPolicy,
    PrivacyMode,
    TelemetryRecord,
    TelemetrySummary,
    hash_candidate,
)

_DEFAULT_MAX_READ_BYTES = 64 * 1024 * 1024


def _validate_privacy_and_bounds(
    privacy: PrivacyMode,
    max_candidate_chars: int | None,
    on_error: ErrorPolicy = "warn",
) -> tuple[PrivacyMode, int | None, ErrorPolicy]:
    if not isinstance(privacy, str) or privacy not in VALID_PRIVACY_MODES:
        raise ConfigurationError(
            f"Invalid privacy={privacy!r}: must be one of {sorted(VALID_PRIVACY_MODES)}"
        )
    if max_candidate_chars is not None and (
        isinstance(max_candidate_chars, bool)
        or not isinstance(max_candidate_chars, int)
        or max_candidate_chars < 1
    ):
        raise ConfigurationError(
            f"max_candidate_chars must be an integer >= 1 or None, got {max_candidate_chars!r}"
        )
    if not isinstance(on_error, str) or on_error not in VALID_ERROR_POLICIES:
        raise ConfigurationError(
            f"Invalid on_error={on_error!r}: must be one of {sorted(VALID_ERROR_POLICIES)}"
        )
    return privacy, max_candidate_chars, on_error


def _handle_sink_error(exc: Exception, on_error: ErrorPolicy, sink_name: str) -> None:
    if on_error == "raise":
        raise exc
    if on_error == "warn":
        warnings.warn(
            f"{sink_name} failed to emit telemetry record: {exc}",
            TelemetryWarning,
            stacklevel=3,
        )


def _validate_target_file_path(path: str | Path, label: str) -> Path:
    if isinstance(path, str):
        if not path.strip():
            raise ConfigurationError(f"{label} path cannot be empty or whitespace")
        target = Path(path)
    elif isinstance(path, Path):
        target = path
    else:
        raise ConfigurationError(
            f"{label} path must be a str or Path, got {type(path).__name__}"
        )
    if target.is_symlink():
        raise ConfigurationError(f"Refusing to access {label} through symlink: '{target}'")
    if target.exists() and target.is_dir():
        raise ConfigurationError(f"{label} path cannot be a directory: '{target}'")
    return target


def open_append_nofollow(target: Path) -> IO[str]:
    """Opens ``target`` for UTF-8 append without following symlinks on POSIX."""
    if target.is_symlink():
        raise OSError(f"Refusing to open symlink telemetry target: '{target}'")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise OSError(f"Refusing to write inside symlink parent directory: '{target.parent}'")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(target, flags, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"Telemetry path is not a regular file: '{target}'")
        return os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def read_regular_text_nofollow(target: Path, *, max_bytes: int, label: str) -> str:
    """Reads a regular UTF-8 file up to ``max_bytes`` without following symlinks."""
    if target.is_symlink():
        raise ConfigurationError(f"Refusing to read {label} from symlink: '{target}'")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise ConfigurationError(f"Cannot open {label} '{target}': {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ConfigurationError(f"{label} '{target}' is not a regular file.")
        if st.st_size > max_bytes:
            raise ConfigurationError(
                f"{label} '{target}' ({st.st_size} bytes) exceeds maximum size ({max_bytes} bytes)."
            )
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return handle.read(max_bytes + 1)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def rotate_jsonl_files(target: Path, backup_count: int) -> None:
    """Rotates ``target -> target.1 -> ... -> target.<backup_count>``, rejecting symlinks."""
    if target.is_symlink():
        raise OSError(f"Refusing to rotate symlink telemetry file: '{target}'")
    if backup_count <= 0:
        target.unlink(missing_ok=True)
        return
    for idx in range(backup_count, 0, -1):
        dst = target.with_name(f"{target.name}.{idx}")
        src = target if idx == 1 else target.with_name(f"{target.name}.{idx - 1}")
        if dst.is_symlink() or src.is_symlink():
            raise OSError(f"Refusing to rotate over symlink backup target: '{dst}'")
        if idx == backup_count and dst.exists():
            dst.unlink(missing_ok=True)
        if src.exists():
            os.replace(src, dst)


class TelemetrySink(abc.ABC):
    """Abstract base class for structured `LogitGate` telemetry sinks."""

    @abc.abstractmethod
    def emit(self, record: TelemetryRecord) -> None:
        """Persists or processes a single `TelemetryRecord`."""

    async def aemit(self, record: TelemetryRecord) -> None:
        """Async hook; defaults to synchronous `emit` for fast non-blocking sinks."""
        self.emit(record)

    def flush(self) -> None:
        """Flushes any buffered records."""

    def close(self) -> None:
        """Flushes and releases underlying resources."""
        self.flush()

    async def aclose(self) -> None:
        """Async close hook; defaults to synchronous `close()`."""
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class JsonlTelemetrySink(TelemetrySink):
    """Thread-safe, symlink-hardened JSONL telemetry sink with optional size rotation."""

    def __init__(
        self,
        path: str | Path,
        *,
        privacy: PrivacyMode = "full",
        max_candidate_chars: int | None = 16384,
        max_bytes: int | None = None,
        backup_count: int = 0,
        filter_fn: Callable[[TelemetryRecord], bool] | None = None,
        fsync: bool = False,
        on_error: ErrorPolicy = "warn",
    ) -> None:
        target = _validate_target_file_path(path, "JsonlTelemetrySink")
        val_priv, val_chars, val_err = _validate_privacy_and_bounds(
            privacy, max_candidate_chars, on_error
        )
        if max_bytes is not None and (
            isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 256
        ):
            raise ConfigurationError(
                f"max_bytes must be an integer >= 256 or None, got {max_bytes!r}"
            )
        if isinstance(backup_count, bool) or not isinstance(backup_count, int) or backup_count < 0:
            raise ConfigurationError(
                f"backup_count must be an integer >= 0, got {backup_count!r}"
            )
        if not isinstance(fsync, bool):
            raise ConfigurationError(f"fsync must be a bool, got {fsync!r}")
        if filter_fn is not None and not callable(filter_fn):
            raise ConfigurationError("filter_fn must be callable or None")

        self.path: Path = target
        self.privacy: PrivacyMode = val_priv
        self.max_candidate_chars: int | None = val_chars
        self.max_bytes: int | None = max_bytes
        self.backup_count: int = backup_count
        self.filter_fn = filter_fn
        self.fsync: bool = fsync
        self.on_error: ErrorPolicy = val_err
        self._lock = threading.Lock()
        self._handle: IO[str] | None = None
        self._closed = False
        self._records_written = 0

    @property
    def records_written(self) -> int:
        return self._records_written

    def _ensure_open_locked(self) -> IO[str]:
        if self._handle is None:
            self._handle = open_append_nofollow(self.path)
        return self._handle

    def _rotate_if_needed_locked(self, incoming_bytes: int) -> None:
        if self.max_bytes is None:
            return
        current_size = 0
        if self._handle is not None:
            self._handle.flush()
            try:
                current_size = os.fstat(self._handle.fileno()).st_size
            except OSError:
                current_size = 0
        elif self.path.exists() and not self.path.is_symlink():
            try:
                current_size = self.path.stat().st_size
            except OSError:
                current_size = 0
        if current_size > 0 and current_size + incoming_bytes > self.max_bytes:
            if self._handle is not None:
                try:
                    self._handle.close()
                finally:
                    self._handle = None
            rotate_jsonl_files(self.path, self.backup_count)

    def emit(self, record: TelemetryRecord) -> None:
        try:
            if self.filter_fn is not None and not self.filter_fn(record):
                return
            prepared = record.with_privacy(
                privacy=self.privacy,
                max_candidate_chars=self.max_candidate_chars,
            )
            line = json.dumps(prepared.model_dump(mode="json"), allow_nan=False) + "\n"
            encoded_len = len(line.encode("utf-8"))
            with self._lock:
                if self._closed:
                    return
                self._rotate_if_needed_locked(encoded_len)
                handle = self._ensure_open_locked()
                handle.write(line)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
                self._records_written += 1
        except Exception as exc:  # noqa: BLE001 - user filter_fn or sink I/O routed via on_error
            _handle_sink_error(exc, self.on_error, "JsonlTelemetrySink")

    def flush(self) -> None:
        with self._lock:
            if self._handle is not None and not self._closed:
                try:
                    self._handle.flush()
                    if self.fsync:
                        os.fsync(self._handle.fileno())
                except OSError as exc:
                    _handle_sink_error(exc, self.on_error, "JsonlTelemetrySink")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except OSError:
                    pass
                finally:
                    self._handle = None


class MemoryTelemetrySink(TelemetrySink):
    """Bounded in-memory ring-buffer telemetry sink for tests, notebooks, and live inspection."""

    def __init__(
        self,
        *,
        max_records: int = 10000,
        privacy: PrivacyMode = "full",
        max_candidate_chars: int | None = 16384,
        filter_fn: Callable[[TelemetryRecord], bool] | None = None,
    ) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ConfigurationError(
                f"max_records must be an integer >= 1, got {max_records!r}"
            )
        val_priv, val_chars, _ = _validate_privacy_and_bounds(privacy, max_candidate_chars)
        if filter_fn is not None and not callable(filter_fn):
            raise ConfigurationError("filter_fn must be callable or None")
        self.max_records = max_records
        self.privacy = val_priv
        self.max_candidate_chars = val_chars
        self.filter_fn = filter_fn
        self._buffer: deque[TelemetryRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    @property
    def records(self) -> list[TelemetryRecord]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def emit(self, record: TelemetryRecord) -> None:
        if self.filter_fn is not None and not self.filter_fn(record):
            return
        prepared = record.with_privacy(
            privacy=self.privacy,
            max_candidate_chars=self.max_candidate_chars,
        )
        with self._lock:
            self._buffer.append(prepared)

    def summary(self) -> TelemetrySummary:
        return summarize_telemetry(self.records)


class CallbackTelemetrySink(TelemetrySink):
    """Forwards `TelemetryRecord` events to a user-provided Python callable (e.g. OpenTelemetry/Prometheus)."""

    def __init__(
        self,
        callback: Callable[[TelemetryRecord], Any],
        *,
        privacy: PrivacyMode = "full",
        max_candidate_chars: int | None = 16384,
        filter_fn: Callable[[TelemetryRecord], bool] | None = None,
        on_error: ErrorPolicy = "warn",
    ) -> None:
        if not callable(callback):
            raise ConfigurationError("CallbackTelemetrySink callback must be callable")
        val_priv, val_chars, val_err = _validate_privacy_and_bounds(
            privacy, max_candidate_chars, on_error
        )
        if filter_fn is not None and not callable(filter_fn):
            raise ConfigurationError("filter_fn must be callable or None")
        self.callback = callback
        self.privacy = val_priv
        self.max_candidate_chars = val_chars
        self.filter_fn = filter_fn
        self.on_error = val_err

    def emit(self, record: TelemetryRecord) -> None:
        try:
            if self.filter_fn is not None and not self.filter_fn(record):
                return
            prepared = record.with_privacy(
                privacy=self.privacy,
                max_candidate_chars=self.max_candidate_chars,
            )
            self.callback(prepared)
        except Exception as exc:  # noqa: BLE001 - arbitrary user callback routed via on_error
            _handle_sink_error(exc, self.on_error, "CallbackTelemetrySink")


class CompositeTelemetrySink(TelemetrySink):
    """Multiplexes telemetry events across multiple sinks with per-sink error isolation."""

    def __init__(self, sinks: Sequence[TelemetrySink | str | Path]) -> None:
        if isinstance(sinks, (str, bytes)) or not isinstance(sinks, Sequence) or not sinks:
            raise ConfigurationError("CompositeTelemetrySink requires a non-empty sequence of sinks")
        resolved: list[TelemetrySink] = []
        for item in sinks:
            sink = resolve_telemetry_sink(item)
            if not isinstance(sink, TelemetrySink):
                raise ConfigurationError(
                    f"CompositeTelemetrySink entries must resolve to a TelemetrySink, got {item!r}"
                )
            resolved.append(sink)
        self.sinks: tuple[TelemetrySink, ...] = tuple(resolved)

    def emit(self, record: TelemetryRecord) -> None:
        first_err: Exception | None = None
        for sink in self.sinks:
            try:
                sink.emit(record)
            except Exception as exc:  # noqa: BLE001 - isolate sibling sinks before re-raising
                if first_err is None:
                    first_err = exc
        if first_err is not None:
            raise first_err

    async def aemit(self, record: TelemetryRecord) -> None:
        first_err: Exception | None = None
        for sink in self.sinks:
            try:
                await sink.aemit(record)
            except Exception as exc:  # noqa: BLE001 - isolate sibling sinks before re-raising
                if first_err is None:
                    first_err = exc
        if first_err is not None:
            raise first_err

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    async def aclose(self) -> None:
        for sink in self.sinks:
            await sink.aclose()


def resolve_telemetry_sink(
    sink: TelemetrySink | str | Path | bool | None,
) -> TelemetrySink | Literal[False] | None:
    """Resolves a gate/call/global telemetry sink specification.

    - ``None``: inherit from outer scope (gate -> global config -> disabled).
    - ``False``: explicitly disable telemetry for this gate or call.
    - ``str`` or ``Path``: construct a `JsonlTelemetrySink(path)`.
    - `TelemetrySink`: use directly.
    """
    if sink is None:
        return None
    if sink is False:
        return False
    if sink is True:
        raise ConfigurationError(
            "telemetry_sink cannot be True; pass a TelemetrySink instance, a file path, None, or False."
        )
    if isinstance(sink, TelemetrySink):
        return sink
    if isinstance(sink, (str, Path)):
        return JsonlTelemetrySink(sink)
    raise ConfigurationError(
        f"telemetry_sink must be a TelemetrySink, file path, False, or None, got {type(sink).__name__}"
    )


def _find_student_and_teacher_traces(
    traces: Sequence[LayerTrace],
) -> tuple[LayerTrace | None, LayerTrace | None]:
    """Identifies the last binary Tier-1 student trace and last Tier-2 generative teacher trace."""
    teacher_idx = -1
    for idx in range(len(traces) - 1, -1, -1):
        tr = traces[idx]
        if tr.tokens_generated > 0 or (
            tr.diagnostic_report is not None and tr.failure_code is None
        ):
            teacher_idx = idx
            break
    if teacher_idx <= 0:
        return None, None
    teacher_tr = traces[teacher_idx]
    for idx in range(teacher_idx - 1, -1, -1):
        tr = traces[idx]
        if (
            tr.tokens_generated == 0
            and tr.failure_code is None
            and tr.log_odds_pass is not None
            and "ranking" not in tr.metadata
            and "choice_probabilities" not in tr.metadata
        ):
            return tr, teacher_tr
    return None, None


def _has_student_teacher_discrepancy(record: TelemetryRecord) -> bool:
    student_tr, teacher_tr = _find_student_and_teacher_traces(record.traces)
    if student_tr is None or teacher_tr is None:
        return False
    if record.teacher_passed is not None:
        t_pass = record.teacher_passed
    elif "judge_log_odds" in teacher_tr.metadata and isinstance(
        teacher_tr.metadata["judge_log_odds"], (int, float)
    ):
        t_pass = float(teacher_tr.metadata["judge_log_odds"]) >= 0.0
    else:
        t_pass = teacher_tr.status == DecisionStatus.PASSED

    if student_tr.status == DecisionStatus.PASSED and not t_pass:
        return True
    if student_tr.status == DecisionStatus.FAILED and t_pass:
        return True
    return bool(
        student_tr.status == DecisionStatus.AMBIGUOUS
        and student_tr.log_odds_pass is not None
        and ((student_tr.log_odds_pass > 0.0) != t_pass)
    )


def load_telemetry_records(
    path: str | Path,
    *,
    max_bytes: int = _DEFAULT_MAX_READ_BYTES,
) -> list[TelemetryRecord]:
    """Loads structured `TelemetryRecord` entries from a JSONL file (symlink-protected)."""
    target = _validate_target_file_path(path, "Telemetry JSONL")
    raw_text = read_regular_text_nofollow(target, max_bytes=max_bytes, label="Telemetry JSONL")
    records: list[TelemetryRecord] = []
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise TypeError("JSONL line must decode to an object")
            records.append(TelemetryRecord.model_validate(parsed))
        except (ValueError, TypeError, RecursionError) as exc:
            raise ConfigurationError(
                f"Malformed TelemetryRecord at '{target}' line {lineno}: {exc}"
            ) from exc
    return records


def summarize_telemetry(
    source: str | Path | Sequence[TelemetryRecord],
) -> TelemetrySummary:
    """Computes operational summary metrics across a telemetry JSONL file or record sequence."""
    records = (
        load_telemetry_records(source)
        if isinstance(source, (str, Path))
        else list(source)
    )
    n = len(records)
    if n == 0:
        return TelemetrySummary(
            total_records=0,
            passed_count=0,
            failed_count=0,
            ambiguous_count=0,
            fast_path_count=0,
            fast_path_rate=0.0,
            escalated_count=0,
            escalation_rate=0.0,
            discrepancy_count=0,
            discrepancy_rate=0.0,
            mean_latency_ms=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_tokens_generated=0,
            by_gate_type={},
        )

    passed_count = sum(1 for r in records if r.status == DecisionStatus.PASSED)
    failed_count = sum(1 for r in records if r.status == DecisionStatus.FAILED)
    ambiguous_count = sum(1 for r in records if r.status == DecisionStatus.AMBIGUOUS)
    fast_path_count = sum(1 for r in records if r.fast_pathed)
    escalated_count = sum(1 for r in records if r.escalated)
    discrepancy_count = sum(1 for r in records if _has_student_teacher_discrepancy(r))
    latencies = sorted(max(0.0, float(r.total_latency_ms)) for r in records)
    p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    by_gate: dict[str, int] = {}
    for r in records:
        by_gate[r.gate_type] = by_gate.get(r.gate_type, 0) + 1

    return TelemetrySummary(
        total_records=n,
        passed_count=passed_count,
        failed_count=failed_count,
        ambiguous_count=ambiguous_count,
        fast_path_count=fast_path_count,
        fast_path_rate=round(fast_path_count / float(n), 4),
        escalated_count=escalated_count,
        escalation_rate=round(escalated_count / float(n), 4),
        discrepancy_count=discrepancy_count,
        discrepancy_rate=(
            round(discrepancy_count / float(escalated_count), 4)
            if escalated_count > 0
            else 0.0
        ),
        mean_latency_ms=round(statistics.fmean(latencies), 4),
        median_latency_ms=round(statistics.median(latencies), 4),
        p95_latency_ms=round(latencies[p95_idx], 4),
        total_tokens_generated=sum(max(0, int(r.total_tokens_generated)) for r in records),
        by_gate_type=by_gate,
    )


def emit_verdict_telemetry(
    verdict: Verdict,
    *,
    candidate: Any,
    gate_name: str,
    gate_type: str,
    rubric: str | None = None,
    model: str | None = None,
    backend: Any = None,
    gate_sink: TelemetrySink | Literal[False] | None = None,
    call_kwargs: Mapping[str, Any] | None = None,
    fast_pathed: bool = False,
    escalated: bool = False,
    teacher_model: str | None = None,
    teacher_passed: bool | None = None,
) -> Verdict:
    """Zero-overhead telemetry dispatch helper called on gate/cascade completion.

    Fast-returns ``verdict`` untouched in sub-microsecond time when ``_suppress_telemetry``
    is active or no call/gate/global telemetry sink is configured.
    """
    if call_kwargs is not None and call_kwargs.get("_suppress_telemetry"):
        return verdict

    call_sink_raw = call_kwargs.get("telemetry_sink") if call_kwargs is not None else None
    owns_temp_sink = False
    active_sink: TelemetrySink | None = None

    if call_sink_raw is not None:
        if call_sink_raw is False:
            return verdict
        if isinstance(call_sink_raw, TelemetrySink):
            active_sink = call_sink_raw
        else:
            resolved_call = resolve_telemetry_sink(call_sink_raw)
            if resolved_call is False or resolved_call is None:
                return verdict
            active_sink = resolved_call
            owns_temp_sink = True
    elif gate_sink is False:
        return verdict
    elif isinstance(gate_sink, TelemetrySink):
        active_sink = gate_sink
    else:
        from logitgate.config import get_default_telemetry_sink

        active_sink = get_default_telemetry_sink()

    if active_sink is None:
        return verdict

    try:
        from logitgate.backends.base import infer_backend_spec

        backend_spec: str | None
        if isinstance(backend, str):
            backend_spec = backend
        elif backend is not None:
            backend_spec = infer_backend_spec(backend)
        else:
            backend_spec = None

        raw_tags = call_kwargs.get("telemetry_tags") if call_kwargs is not None else None
        tags: dict[str, Any] = {}
        if isinstance(raw_tags, Mapping):
            tags = {str(k): v for k, v in raw_tags.items()}

        rendered = render_candidate(candidate)
        record = TelemetryRecord(
            gate_name=gate_name,
            gate_type=gate_type,
            model=model,
            backend=backend_spec,
            teacher_model=teacher_model,
            teacher_passed=teacher_passed,
            rubric=rubric,
            candidate=rendered,
            candidate_sha256=hash_candidate(rendered),
            candidate_length=len(rendered),
            passed=verdict.passed,
            status=verdict.status,
            probability=verdict.probability,
            score=verdict.score,
            choice=verdict.choice,
            choice_probabilities=(
                dict(verdict.choice_probabilities)
                if verdict.choice_probabilities is not None
                else None
            ),
            triggered_layer=verdict.triggered_layer,
            escalated=escalated,
            fast_pathed=fast_pathed,
            total_latency_ms=round(verdict.total_latency_ms, 4),
            total_tokens_generated=verdict.total_tokens_generated,
            reason=verdict.reason,
            traces=list(verdict.traces),
            tags=tags,
        )
        active_sink.emit(record)
    finally:
        if owns_temp_sink and active_sink is not None:
            active_sink.close()

    return verdict
