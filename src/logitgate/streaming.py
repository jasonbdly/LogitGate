"""Mid-stream guardrail wrapper (`StreamGuard`, `guard_stream`, `aguard_stream`) for sync and async LLM token streams."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from logitgate.exceptions import ConfigurationError, StreamAbortedError
from logitgate.models import DecisionStatus, LayerTrace, Verdict

if TYPE_CHECKING:
    from logitgate.gates.base import BaseGate

T_Chunk = TypeVar("T_Chunk")
StreamDeliveryMode = Literal["optimistic", "buffered"]

_CLAUSE_TERMINATORS: frozenset[str] = frozenset({".", "!", "?", ";", ":", "\n", "}", "]"})
_TRAILING_SPACE_CHARS: frozenset[str] = frozenset({" ", "\t", "\r"})


@dataclass(frozen=True, slots=True)
class StreamCheckRecord:
    """Telemetry record for a single mid-stream or end-of-stream checkpoint."""

    chunk_index: int
    prefix_chars: int
    is_final: bool
    verdict: Verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "prefix_chars": self.prefix_chars,
            "is_final": self.is_final,
            "passed": self.verdict.passed,
            "status": self.verdict.status.value,
            "probability": self.verdict.probability,
            "latency_ms": round(self.verdict.total_latency_ms, 2),
        }


@dataclass(frozen=True, slots=True)
class StreamGuardSummary:
    """Aggregated outcome and latency metrics for a completed or aborted stream."""

    passed: bool
    aborted: bool
    aborted_mid_stream: bool
    abort_chunk_index: int | None
    total_chunks_seen: int
    chunks_yielded: int
    chars_seen: int
    chars_yielded: int
    checks_executed: int
    total_gate_latency_ms: float
    final_verdict: Verdict | None
    checks: tuple[StreamCheckRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "aborted": self.aborted,
            "aborted_mid_stream": self.aborted_mid_stream,
            "abort_chunk_index": self.abort_chunk_index,
            "total_chunks_seen": self.total_chunks_seen,
            "chunks_yielded": self.chunks_yielded,
            "chars_seen": self.chars_seen,
            "chars_yielded": self.chars_yielded,
            "checks_executed": self.checks_executed,
            "total_gate_latency_ms": round(self.total_gate_latency_ms, 2),
            "final_status": self.final_verdict.status.value if self.final_verdict else None,
            "final_probability": self.final_verdict.probability if self.final_verdict else None,
            "checks": [c.to_dict() for c in self.checks],
        }


def _extract_chunk_text(
    chunk: Any,
    chunk_extractor: Callable[[Any], str] | None = None,
) -> str:
    if chunk_extractor is not None:
        extracted = chunk_extractor(chunk)
        if not isinstance(extracted, str):
            raise TypeError(
                f"chunk_extractor must return a str, got {type(extracted).__name__}"
            )
        return extracted
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        if "response" in chunk and (chunk["response"] is None or isinstance(chunk["response"], str)):
            return chunk["response"] or ""
        msg = chunk.get("message")
        if isinstance(msg, Mapping) and ("content" in msg):
            c = msg["content"]
            if c is None or isinstance(c, str):
                return c or ""
        delta = chunk.get("delta")
        if isinstance(delta, Mapping) and ("content" in delta):
            c = delta["content"]
            if c is None or isinstance(c, str):
                return c or ""
        if "content" in chunk and (chunk["content"] is None or isinstance(chunk["content"], str)):
            return chunk["content"] or ""
        raise TypeError(
            "Mapping stream chunk does not contain 'response', 'message.content', "
            "'delta.content', or 'content'; pass chunk_extractor=..."
        )
    choices = getattr(chunk, "choices", None)
    if isinstance(choices, (list, tuple)):
        if not choices:
            return ""
        delta_obj = getattr(choices[0], "delta", None)
        content = getattr(delta_obj, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
    raise TypeError(
        f"Unsupported stream chunk type {type(chunk).__name__}; yield strings or pass chunk_extractor=..."
    )


def _ends_at_clause_boundary(accumulated: list[str]) -> bool:
    for piece in reversed(accumulated):
        for idx in range(len(piece) - 1, -1, -1):
            ch = piece[idx]
            if ch in _TRAILING_SPACE_CHARS:
                continue
            return ch in _CLAUSE_TERMINATORS
    return False


def _overflow_verdict(gate_name: str, max_buffer_chars: int) -> Verdict:
    reason = f"Stream buffer exceeded max_buffer_chars={max_buffer_chars}"
    trace = LayerTrace(
        layer_name=gate_name,
        status=DecisionStatus.FAILED,
        latency_ms=0.0,
        failure_code="STREAM_BUFFER_OVERFLOW",
        diagnostic_report=reason,
    )
    return Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.0,
        triggered_layer=gate_name,
        total_latency_ms=0.0,
        total_tokens_generated=0,
        reason=reason,
        traces=[trace],
    )


def _emit_stream_telemetry(
    gate: BaseGate,
    candidate_text: str,
    verdict: Verdict,
    *,
    guard_sink: Any = None,
    aborted_mid_stream: bool,
    abort_chunk_index: int | None,
    checks_executed: int,
) -> None:
    from logitgate.telemetry.sinks import emit_verdict_telemetry

    gate_sink = (
        guard_sink
        if guard_sink is not None
        else getattr(gate, "_effective_telemetry_sink", lambda: getattr(gate, "telemetry_sink", None))()
    )
    rubric = getattr(gate, "rubric", None) or getattr(gate, "question", None)
    backend = getattr(gate, "backend", None)
    model = getattr(gate, "model", None)
    if backend is None and hasattr(gate, "gates"):
        for g in getattr(gate, "gates", ()):
            if getattr(g, "backend", None) is not None:
                backend = g.backend
                model = getattr(g, "model", None)
                break
    emit_verdict_telemetry(
        verdict,
        candidate=candidate_text,
        gate_name=getattr(gate, "name", type(gate).__name__),
        gate_type=type(gate).__name__,
        rubric=rubric if isinstance(rubric, str) else None,
        model=model if isinstance(model, str) else None,
        backend=backend,
        gate_sink=gate_sink,
        call_kwargs={
            "telemetry_tags": {
                "stream_guard": True,
                "aborted_mid_stream": aborted_mid_stream,
                "abort_chunk_index": abort_chunk_index,
                "stream_checks": checks_executed,
            }
        },
    )


class GuardedStream(Iterator[T_Chunk], Generic[T_Chunk]):
    """Synchronous guarded stream iterator carrying `.summary` and `.final_verdict`."""

    def __init__(self, guard: StreamGuard, source: Iterable[T_Chunk]) -> None:
        self._guard = guard
        self._gen = guard._iter_sync(source, self)
        self.summary: StreamGuardSummary | None = None
        self.final_verdict: Verdict | None = None

    def __iter__(self) -> GuardedStream[T_Chunk]:
        return self

    def __next__(self) -> T_Chunk:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()


class AsyncGuardedStream(AsyncIterator[T_Chunk], Generic[T_Chunk]):
    """Asynchronous guarded stream iterator carrying `.summary` and `.final_verdict`."""

    def __init__(self, guard: StreamGuard, source: AsyncIterable[T_Chunk]) -> None:
        self._guard = guard
        self._agen = guard._iter_async(source, self)
        self.summary: StreamGuardSummary | None = None
        self.final_verdict: Verdict | None = None

    def __aiter__(self) -> AsyncGuardedStream[T_Chunk]:
        return self

    async def __anext__(self) -> T_Chunk:
        return await self._agen.__anext__()

    async def aclose(self) -> None:
        await self._agen.aclose()


class StreamGuard:
    """Incremental mid-stream guardrail evaluator for synchronous and asynchronous LLM streams.

    Schedules periodic prefix checks at configurable character/chunk strides (optionally aligned
    to clause/sentence boundaries to avoid mid-phrase false alarms), supports consecutive-failure
    hysteresis and ``"optimistic"`` vs. ``"buffered"`` chunk delivery, cancels the upstream
    generator immediately upon abort, and deduplicates the final end-of-stream verification.
    """

    def __init__(
        self,
        gate: BaseGate,
        *,
        min_chars: int = 64,
        stride_chars: int = 128,
        stride_chunks: int | None = None,
        clause_boundary: bool = True,
        max_wait_chars: int | None = None,
        consecutive_failures: int = 1,
        abort_on_ambiguous: bool = False,
        abort_prob_threshold: float | None = None,
        final_check: bool = True,
        fail_on_final_ambiguous: bool = True,
        delivery: StreamDeliveryMode = "optimistic",
        raise_on_abort: bool = True,
        close_upstream_on_abort: bool = True,
        max_buffer_chars: int = 1_048_576,
        chunk_extractor: Callable[[Any], str] | None = None,
        telemetry_sink: Any = None,
    ) -> None:
        from logitgate.gates.base import BaseGate as _BaseGate
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        if not isinstance(gate, _BaseGate):
            raise ConfigurationError(f"gate must be a BaseGate instance, got {type(gate).__name__}")
        if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1:
            raise ConfigurationError(f"min_chars must be an integer >= 1, got {min_chars!r}")
        if isinstance(stride_chars, bool) or not isinstance(stride_chars, int) or stride_chars < 1:
            raise ConfigurationError(f"stride_chars must be an integer >= 1, got {stride_chars!r}")
        if stride_chunks is not None and (
            isinstance(stride_chunks, bool) or not isinstance(stride_chunks, int) or stride_chunks < 1
        ):
            raise ConfigurationError(
                f"stride_chunks must be an integer >= 1 or None, got {stride_chunks!r}"
            )
        if not isinstance(clause_boundary, bool):
            raise ConfigurationError(f"clause_boundary must be a bool, got {clause_boundary!r}")
        if max_wait_chars is not None and (
            isinstance(max_wait_chars, bool) or not isinstance(max_wait_chars, int) or max_wait_chars < 1
        ):
            raise ConfigurationError(
                f"max_wait_chars must be an integer >= 1 or None, got {max_wait_chars!r}"
            )
        if (
            isinstance(consecutive_failures, bool)
            or not isinstance(consecutive_failures, int)
            or consecutive_failures < 1
        ):
            raise ConfigurationError(
                f"consecutive_failures must be an integer >= 1, got {consecutive_failures!r}"
            )
        if not isinstance(abort_on_ambiguous, bool):
            raise ConfigurationError(
                f"abort_on_ambiguous must be a bool, got {abort_on_ambiguous!r}"
            )
        if abort_prob_threshold is not None:
            if isinstance(abort_prob_threshold, bool) or not isinstance(
                abort_prob_threshold, (int, float)
            ):
                raise ConfigurationError(
                    f"abort_prob_threshold must be a float in [0.0, 1.0] or None, got {abort_prob_threshold!r}"
                )
            apt = float(abort_prob_threshold)
            if not (0.0 <= apt <= 1.0):
                raise ConfigurationError(
                    f"abort_prob_threshold must be in [0.0, 1.0], got {abort_prob_threshold!r}"
                )
            self.abort_prob_threshold: float | None = apt
        else:
            self.abort_prob_threshold = None

        if not isinstance(final_check, bool):
            raise ConfigurationError(f"final_check must be a bool, got {final_check!r}")
        if not isinstance(fail_on_final_ambiguous, bool):
            raise ConfigurationError(
                f"fail_on_final_ambiguous must be a bool, got {fail_on_final_ambiguous!r}"
            )
        if delivery not in ("optimistic", "buffered"):
            raise ConfigurationError(
                f"delivery must be 'optimistic' or 'buffered', got {delivery!r}"
            )
        if not isinstance(raise_on_abort, bool):
            raise ConfigurationError(f"raise_on_abort must be a bool, got {raise_on_abort!r}")
        if not isinstance(close_upstream_on_abort, bool):
            raise ConfigurationError(
                f"close_upstream_on_abort must be a bool, got {close_upstream_on_abort!r}"
            )
        if (
            isinstance(max_buffer_chars, bool)
            or not isinstance(max_buffer_chars, int)
            or max_buffer_chars < 1
        ):
            raise ConfigurationError(
                f"max_buffer_chars must be an integer >= 1, got {max_buffer_chars!r}"
            )
        if chunk_extractor is not None and not callable(chunk_extractor):
            raise ConfigurationError("chunk_extractor must be callable or None")
        resolved_sink = resolve_telemetry_sink(telemetry_sink)

        self.gate = gate
        self.min_chars = min_chars
        self.stride_chars = stride_chars
        self.stride_chunks = stride_chunks
        self.clause_boundary = clause_boundary
        self.max_wait_chars = max_wait_chars if max_wait_chars is not None else stride_chars * 2
        self.consecutive_failures = consecutive_failures
        self.abort_on_ambiguous = abort_on_ambiguous
        self.final_check = final_check
        self.fail_on_final_ambiguous = fail_on_final_ambiguous
        self.delivery: StreamDeliveryMode = delivery
        self.raise_on_abort = raise_on_abort
        self.close_upstream_on_abort = close_upstream_on_abort
        self.max_buffer_chars = max_buffer_chars
        self.chunk_extractor = chunk_extractor
        self.telemetry_sink = resolved_sink

        self._summary_lock = threading.Lock()
        self._last_summary: StreamGuardSummary | None = None

    @property
    def last_summary(self) -> StreamGuardSummary | None:
        with self._summary_lock:
            return self._last_summary

    def _record_summary(self, summary: StreamGuardSummary) -> None:
        with self._summary_lock:
            self._last_summary = summary

    def _should_check_now(
        self,
        *,
        chars_seen: int,
        chars_at_last_check: int,
        chunk_count: int,
        chunks_at_last_check: int,
        accumulated: list[str],
    ) -> bool:
        if chars_seen < self.min_chars or chars_seen == chars_at_last_check:
            return False
        chars_delta = chars_seen - chars_at_last_check
        chunks_delta = chunk_count - chunks_at_last_check
        cadence_ready = chars_delta >= self.stride_chars or (
            self.stride_chunks is not None and chunks_delta >= self.stride_chunks
        )
        if not cadence_ready:
            return False
        if not self.clause_boundary:
            return True
        if chars_delta >= self.max_wait_chars:
            return True
        return _ends_at_clause_boundary(accumulated)

    def _is_midstream_strike(self, verdict: Verdict) -> bool:
        strike = verdict.status == DecisionStatus.FAILED or (
            self.abort_on_ambiguous and verdict.status == DecisionStatus.AMBIGUOUS
        )
        if not strike:
            return False
        if self.abort_prob_threshold is not None and verdict.probability is not None:
            return verdict.probability <= self.abort_prob_threshold
        return True

    def _is_final_failure(self, verdict: Verdict) -> bool:
        if verdict.status == DecisionStatus.FAILED:
            return True
        if verdict.status == DecisionStatus.AMBIGUOUS:
            return self.fail_on_final_ambiguous or self.abort_on_ambiguous
        return not verdict.passed

    def stream(self, source: Iterable[T_Chunk]) -> GuardedStream[T_Chunk]:
        """Wraps a synchronous chunk iterable with mid-stream verification."""
        if isinstance(source, (str, bytes)):
            raise ConfigurationError(
                "source must be an iterable of stream chunks, not a single str/bytes"
            )
        return GuardedStream(self, source)

    def astream(self, source: AsyncIterable[T_Chunk]) -> AsyncGuardedStream[T_Chunk]:
        """Wraps an asynchronous chunk iterable with mid-stream verification."""
        return AsyncGuardedStream(self, source)

    def __call__(self, source: Iterable[T_Chunk]) -> GuardedStream[T_Chunk]:
        return self.stream(source)

    def _iter_sync(
        self,
        source: Iterable[T_Chunk],
        handle: GuardedStream[T_Chunk],
    ) -> Generator[T_Chunk, None, None]:
        iterator = iter(source)
        accumulated: list[str] = []
        yielded_pieces: list[str] = []
        pending_chunks: list[tuple[T_Chunk, str]] = []
        checks: list[StreamCheckRecord] = []

        chars_seen = 0
        chars_yielded = 0
        chunks_seen = 0
        chunks_yielded = 0
        chars_at_last_check = 0
        chunks_at_last_check = 0
        fail_streak = 0
        gate_latency_ms = 0.0

        def _close_upstream() -> None:
            if self.close_upstream_on_abort:
                close_fn = getattr(iterator, "close", None)
                if callable(close_fn):
                    with contextlib.suppress(Exception):
                        close_fn()

        try:
            for chunk_idx, item in enumerate(iterator):
                text = _extract_chunk_text(item, self.chunk_extractor)
                chunks_seen += 1
                if text:
                    accumulated.append(text)
                    chars_seen += len(text)

                if chars_seen > self.max_buffer_chars:
                    _close_upstream()
                    partial = "".join(accumulated)
                    yielded_str = "".join(yielded_pieces)
                    ov_verdict = _overflow_verdict(self.gate.name, self.max_buffer_chars)
                    summary = StreamGuardSummary(
                        passed=False,
                        aborted=True,
                        aborted_mid_stream=True,
                        abort_chunk_index=chunk_idx,
                        total_chunks_seen=chunks_seen,
                        chunks_yielded=chunks_yielded,
                        chars_seen=chars_seen,
                        chars_yielded=chars_yielded,
                        checks_executed=len(checks),
                        total_gate_latency_ms=gate_latency_ms,
                        final_verdict=ov_verdict,
                        checks=tuple(checks),
                    )
                    handle.summary = summary
                    handle.final_verdict = ov_verdict
                    self._record_summary(summary)
                    if self.raise_on_abort:
                        raise StreamAbortedError(
                            ov_verdict,
                            partial_text=partial,
                            yielded_text=yielded_str,
                            chunk_index=chunk_idx,
                            is_final=False,
                        )
                    return

                if self.delivery == "optimistic":
                    yield item
                    chunks_yielded += 1
                    if text:
                        yielded_pieces.append(text)
                        chars_yielded += len(text)
                else:
                    pending_chunks.append((item, text))

                if self._should_check_now(
                    chars_seen=chars_seen,
                    chars_at_last_check=chars_at_last_check,
                    chunk_count=chunks_seen,
                    chunks_at_last_check=chunks_at_last_check,
                    accumulated=accumulated,
                ):
                    prefix = "".join(accumulated)
                    verdict = self.gate.evaluate(
                        prefix, save_on_fail=False, _suppress_telemetry=True
                    )
                    gate_latency_ms += verdict.total_latency_ms
                    chars_at_last_check = chars_seen
                    chunks_at_last_check = chunks_seen
                    checks.append(
                        StreamCheckRecord(
                            chunk_index=chunk_idx,
                            prefix_chars=chars_seen,
                            is_final=False,
                            verdict=verdict,
                        )
                    )

                    if self._is_midstream_strike(verdict):
                        fail_streak += 1
                    else:
                        fail_streak = 0

                    if fail_streak >= self.consecutive_failures:
                        _close_upstream()
                        _emit_stream_telemetry(
                            self.gate,
                            prefix,
                            verdict,
                            guard_sink=self.telemetry_sink,
                            aborted_mid_stream=True,
                            abort_chunk_index=chunk_idx,
                            checks_executed=len(checks),
                        )
                        yielded_str = "".join(yielded_pieces)
                        summary = StreamGuardSummary(
                            passed=False,
                            aborted=True,
                            aborted_mid_stream=True,
                            abort_chunk_index=chunk_idx,
                            total_chunks_seen=chunks_seen,
                            chunks_yielded=chunks_yielded,
                            chars_seen=chars_seen,
                            chars_yielded=chars_yielded,
                            checks_executed=len(checks),
                            total_gate_latency_ms=gate_latency_ms,
                            final_verdict=verdict,
                            checks=tuple(checks),
                        )
                        handle.summary = summary
                        handle.final_verdict = verdict
                        self._record_summary(summary)
                        if self.raise_on_abort:
                            raise StreamAbortedError(
                                verdict,
                                partial_text=prefix,
                                yielded_text=yielded_str,
                                chunk_index=chunk_idx,
                                is_final=False,
                            )
                        return

                    if (
                        self.delivery == "buffered"
                        and verdict.passed
                        and fail_streak == 0
                        and pending_chunks
                    ):
                        for pend_item, pend_text in pending_chunks:
                            yield pend_item
                            chunks_yielded += 1
                            if pend_text:
                                yielded_pieces.append(pend_text)
                                chars_yielded += len(pend_text)
                        pending_chunks.clear()

            # End of stream reached without mid-stream abort
            full_text = "".join(accumulated)
            final_verdict: Verdict | None = None
            if self.final_check:
                if checks and chars_at_last_check == chars_seen:
                    final_verdict = checks[-1].verdict
                else:
                    final_verdict = self.gate.evaluate(
                        full_text, save_on_fail=False, _suppress_telemetry=True
                    )
                    gate_latency_ms += final_verdict.total_latency_ms
                    checks.append(
                        StreamCheckRecord(
                            chunk_index=max(0, chunks_seen - 1),
                            prefix_chars=chars_seen,
                            is_final=True,
                            verdict=final_verdict,
                        )
                    )

                _emit_stream_telemetry(
                    self.gate,
                    full_text,
                    final_verdict,
                    guard_sink=self.telemetry_sink,
                    aborted_mid_stream=False,
                    abort_chunk_index=max(0, chunks_seen - 1)
                    if self._is_final_failure(final_verdict)
                    else None,
                    checks_executed=len(checks),
                )

                if self._is_final_failure(final_verdict):
                    yielded_str = "".join(yielded_pieces)
                    last_idx = max(0, chunks_seen - 1)
                    summary = StreamGuardSummary(
                        passed=False,
                        aborted=True,
                        aborted_mid_stream=False,
                        abort_chunk_index=last_idx,
                        total_chunks_seen=chunks_seen,
                        chunks_yielded=chunks_yielded,
                        chars_seen=chars_seen,
                        chars_yielded=chars_yielded,
                        checks_executed=len(checks),
                        total_gate_latency_ms=gate_latency_ms,
                        final_verdict=final_verdict,
                        checks=tuple(checks),
                    )
                    handle.summary = summary
                    handle.final_verdict = final_verdict
                    self._record_summary(summary)
                    if self.raise_on_abort:
                        raise StreamAbortedError(
                            final_verdict,
                            partial_text=full_text,
                            yielded_text=yielded_str,
                            chunk_index=last_idx,
                            is_final=True,
                        )
                    return
            elif checks:
                final_verdict = checks[-1].verdict

            if self.delivery == "buffered" and pending_chunks:
                for pend_item, pend_text in pending_chunks:
                    yield pend_item
                    chunks_yielded += 1
                    if pend_text:
                        yielded_pieces.append(pend_text)
                        chars_yielded += len(pend_text)
                pending_chunks.clear()

            summary = StreamGuardSummary(
                passed=True,
                aborted=False,
                aborted_mid_stream=False,
                abort_chunk_index=None,
                total_chunks_seen=chunks_seen,
                chunks_yielded=chunks_yielded,
                chars_seen=chars_seen,
                chars_yielded=chars_yielded,
                checks_executed=len(checks),
                total_gate_latency_ms=gate_latency_ms,
                final_verdict=final_verdict,
                checks=tuple(checks),
            )
            handle.summary = summary
            handle.final_verdict = final_verdict
            self._record_summary(summary)
        finally:
            pending_chunks.clear()

    async def _iter_async(
        self,
        source: AsyncIterable[T_Chunk],
        handle: AsyncGuardedStream[T_Chunk],
    ) -> AsyncGenerator[T_Chunk, None]:
        aiterator = source.__aiter__()
        accumulated: list[str] = []
        yielded_pieces: list[str] = []
        pending_chunks: list[tuple[T_Chunk, str]] = []
        checks: list[StreamCheckRecord] = []

        chars_seen = 0
        chars_yielded = 0
        chunks_seen = 0
        chunks_yielded = 0
        chars_at_last_check = 0
        chunks_at_last_check = 0
        fail_streak = 0
        gate_latency_ms = 0.0

        async def _aclose_upstream() -> None:
            if self.close_upstream_on_abort:
                aclose_fn = getattr(aiterator, "aclose", None)
                if callable(aclose_fn):
                    with contextlib.suppress(Exception):
                        await aclose_fn()

        try:
            chunk_idx = -1
            while True:
                try:
                    item = await aiterator.__anext__()
                except StopAsyncIteration:
                    break
                chunk_idx += 1
                text = _extract_chunk_text(item, self.chunk_extractor)
                chunks_seen += 1
                if text:
                    accumulated.append(text)
                    chars_seen += len(text)

                if chars_seen > self.max_buffer_chars:
                    await _aclose_upstream()
                    partial = "".join(accumulated)
                    yielded_str = "".join(yielded_pieces)
                    ov_verdict = _overflow_verdict(self.gate.name, self.max_buffer_chars)
                    summary = StreamGuardSummary(
                        passed=False,
                        aborted=True,
                        aborted_mid_stream=True,
                        abort_chunk_index=chunk_idx,
                        total_chunks_seen=chunks_seen,
                        chunks_yielded=chunks_yielded,
                        chars_seen=chars_seen,
                        chars_yielded=chars_yielded,
                        checks_executed=len(checks),
                        total_gate_latency_ms=gate_latency_ms,
                        final_verdict=ov_verdict,
                        checks=tuple(checks),
                    )
                    handle.summary = summary
                    handle.final_verdict = ov_verdict
                    self._record_summary(summary)
                    if self.raise_on_abort:
                        raise StreamAbortedError(
                            ov_verdict,
                            partial_text=partial,
                            yielded_text=yielded_str,
                            chunk_index=chunk_idx,
                            is_final=False,
                        )
                    return

                if self.delivery == "optimistic":
                    yield item
                    chunks_yielded += 1
                    if text:
                        yielded_pieces.append(text)
                        chars_yielded += len(text)
                else:
                    pending_chunks.append((item, text))

                if self._should_check_now(
                    chars_seen=chars_seen,
                    chars_at_last_check=chars_at_last_check,
                    chunk_count=chunks_seen,
                    chunks_at_last_check=chunks_at_last_check,
                    accumulated=accumulated,
                ):
                    prefix = "".join(accumulated)
                    verdict = await self.gate.aevaluate(
                        prefix, save_on_fail=False, _suppress_telemetry=True
                    )
                    gate_latency_ms += verdict.total_latency_ms
                    chars_at_last_check = chars_seen
                    chunks_at_last_check = chunks_seen
                    checks.append(
                        StreamCheckRecord(
                            chunk_index=chunk_idx,
                            prefix_chars=chars_seen,
                            is_final=False,
                            verdict=verdict,
                        )
                    )

                    if self._is_midstream_strike(verdict):
                        fail_streak += 1
                    else:
                        fail_streak = 0

                    if fail_streak >= self.consecutive_failures:
                        await _aclose_upstream()
                        _emit_stream_telemetry(
                            self.gate,
                            prefix,
                            verdict,
                            guard_sink=self.telemetry_sink,
                            aborted_mid_stream=True,
                            abort_chunk_index=chunk_idx,
                            checks_executed=len(checks),
                        )
                        yielded_str = "".join(yielded_pieces)
                        summary = StreamGuardSummary(
                            passed=False,
                            aborted=True,
                            aborted_mid_stream=True,
                            abort_chunk_index=chunk_idx,
                            total_chunks_seen=chunks_seen,
                            chunks_yielded=chunks_yielded,
                            chars_seen=chars_seen,
                            chars_yielded=chars_yielded,
                            checks_executed=len(checks),
                            total_gate_latency_ms=gate_latency_ms,
                            final_verdict=verdict,
                            checks=tuple(checks),
                        )
                        handle.summary = summary
                        handle.final_verdict = verdict
                        self._record_summary(summary)
                        if self.raise_on_abort:
                            raise StreamAbortedError(
                                verdict,
                                partial_text=prefix,
                                yielded_text=yielded_str,
                                chunk_index=chunk_idx,
                                is_final=False,
                            )
                        return

                    if (
                        self.delivery == "buffered"
                        and verdict.passed
                        and fail_streak == 0
                        and pending_chunks
                    ):
                        for pend_item, pend_text in pending_chunks:
                            yield pend_item
                            chunks_yielded += 1
                            if pend_text:
                                yielded_pieces.append(pend_text)
                                chars_yielded += len(pend_text)
                        pending_chunks.clear()

            full_text = "".join(accumulated)
            final_verdict: Verdict | None = None
            if self.final_check:
                if checks and chars_at_last_check == chars_seen:
                    final_verdict = checks[-1].verdict
                else:
                    final_verdict = await self.gate.aevaluate(
                        full_text, save_on_fail=False, _suppress_telemetry=True
                    )
                    gate_latency_ms += final_verdict.total_latency_ms
                    checks.append(
                        StreamCheckRecord(
                            chunk_index=max(0, chunks_seen - 1),
                            prefix_chars=chars_seen,
                            is_final=True,
                            verdict=final_verdict,
                        )
                    )

                _emit_stream_telemetry(
                    self.gate,
                    full_text,
                    final_verdict,
                    guard_sink=self.telemetry_sink,
                    aborted_mid_stream=False,
                    abort_chunk_index=max(0, chunks_seen - 1)
                    if self._is_final_failure(final_verdict)
                    else None,
                    checks_executed=len(checks),
                )

                if self._is_final_failure(final_verdict):
                    yielded_str = "".join(yielded_pieces)
                    last_idx = max(0, chunks_seen - 1)
                    summary = StreamGuardSummary(
                        passed=False,
                        aborted=True,
                        aborted_mid_stream=False,
                        abort_chunk_index=last_idx,
                        total_chunks_seen=chunks_seen,
                        chunks_yielded=chunks_yielded,
                        chars_seen=chars_seen,
                        chars_yielded=chars_yielded,
                        checks_executed=len(checks),
                        total_gate_latency_ms=gate_latency_ms,
                        final_verdict=final_verdict,
                        checks=tuple(checks),
                    )
                    handle.summary = summary
                    handle.final_verdict = final_verdict
                    self._record_summary(summary)
                    if self.raise_on_abort:
                        raise StreamAbortedError(
                            final_verdict,
                            partial_text=full_text,
                            yielded_text=yielded_str,
                            chunk_index=last_idx,
                            is_final=True,
                        )
                    return
            elif checks:
                final_verdict = checks[-1].verdict

            if self.delivery == "buffered" and pending_chunks:
                for pend_item, pend_text in pending_chunks:
                    yield pend_item
                    chunks_yielded += 1
                    if pend_text:
                        yielded_pieces.append(pend_text)
                        chars_yielded += len(pend_text)
                pending_chunks.clear()

            summary = StreamGuardSummary(
                passed=True,
                aborted=False,
                aborted_mid_stream=False,
                abort_chunk_index=None,
                total_chunks_seen=chunks_seen,
                chunks_yielded=chunks_yielded,
                chars_seen=chars_seen,
                chars_yielded=chars_yielded,
                checks_executed=len(checks),
                total_gate_latency_ms=gate_latency_ms,
                final_verdict=final_verdict,
                checks=tuple(checks),
            )
            handle.summary = summary
            handle.final_verdict = final_verdict
            self._record_summary(summary)
        finally:
            pending_chunks.clear()


def guard_stream(
    source: Iterable[T_Chunk],
    gate: BaseGate,
    **guard_kwargs: Any,
) -> GuardedStream[T_Chunk]:
    """Convenience wrapper returning a synchronous `GuardedStream` over ``source``."""
    return StreamGuard(gate, **guard_kwargs).stream(source)


def aguard_stream(
    source: AsyncIterable[T_Chunk],
    gate: BaseGate,
    **guard_kwargs: Any,
) -> AsyncGuardedStream[T_Chunk]:
    """Convenience wrapper returning an asynchronous `AsyncGuardedStream` over ``source``."""
    return StreamGuard(gate, **guard_kwargs).astream(source)
