from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from logitgate import (
    BaseGate,
    CacheWarning,
    Cascade,
    ConfigurationError,
    DecisionStatus,
    LayerTrace,
    LogitGate,
    StreamAbortedError,
    StreamGuard,
    StubBackend,
    Verdict,
    aguard_stream,
    guard_stream,
)
from logitgate.cache import (
    VerdictCache,
    _backend_fingerprint,
    _canonicalize_value,
    _prepare_secure_sqlite_path,
    _profile_fingerprint,
    gate_cache_fingerprint,
    lookup_cached_verdict,
    resolve_verdict_cache,
    store_cached_verdict,
)
from logitgate.calibration.profile import CalibrationError, CalibrationProfile
from logitgate.streaming import (
    AsyncGuardedStream,
    GuardedStream,
    StreamCheckRecord,
    StreamGuardSummary,
    _ends_at_clause_boundary,
    _extract_chunk_text,
)
from logitgate.telemetry.distill import (
    DistillationSink,
    export_distillation_dataset,
    extract_distillation_record,
    load_distillation_records,
    recalibrate_from_telemetry,
)
from logitgate.telemetry.drift import (
    RollingDriftMonitor,
    _extract_signals,
    _kolmogorov_survival,
    compute_psi,
    detect_telemetry_drift,
)
from logitgate.telemetry.models import (
    TelemetryRecord,
    _luhn_valid,
    apply_privacy,
    hash_candidate,
)
from logitgate.telemetry.otel import (
    OtelSpanSink,
    _infer_gen_ai_system,
    _sanitize_otel_value,
)
from logitgate.telemetry.sinks import (
    CallbackTelemetrySink,
    CompositeTelemetrySink,
    JsonlTelemetrySink,
    MemoryTelemetrySink,
    TelemetrySink,
    TelemetryWarning,
    _find_student_and_teacher_traces,
    _has_student_teacher_discrepancy,
    _validate_privacy_and_bounds,
    _validate_target_file_path,
    emit_verdict_telemetry,
    load_telemetry_records,
    open_append_nofollow,
    read_regular_text_nofollow,
    resolve_telemetry_sink,
    rotate_jsonl_files,
    summarize_telemetry,
)


class DummyGate(BaseGate):
    """Test gate with deterministic, scripted, or configurable verdicts."""

    name: str = "DummyGate"

    def __init__(
        self,
        name: str = "DummyGate",
        verdicts: Sequence[Verdict] | None = None,
        default_prob: float = 0.95,
        default_passed: bool = True,
        backend: Any = None,
        model: str = "test-model",
    ) -> None:
        self.name = name
        self._verdicts = list(verdicts) if verdicts is not None else []
        self._verdict_idx = 0
        self.default_prob = default_prob
        self.default_passed = default_passed
        self.backend = backend
        self.model = model

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        if self._verdict_idx < len(self._verdicts):
            v = self._verdicts[self._verdict_idx]
            self._verdict_idx += 1
            return v
        status = DecisionStatus.PASSED if self.default_passed else DecisionStatus.FAILED
        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=1.5,
            calibrated_prob=self.default_prob,
            log_odds_pass=2.5 if self.default_passed else -2.5,
        )
        return Verdict(
            passed=self.default_passed,
            status=status,
            probability=self.default_prob,
            triggered_layer=self.name,
            total_latency_ms=1.5,
            total_tokens_generated=0,
            reason="dummy ok" if self.default_passed else "dummy fail",
            traces=[trace],
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return self.evaluate(candidate, **kwargs)


def _make_telemetry_record(
    passed: bool = True,
    status: DecisionStatus = DecisionStatus.PASSED,
    probability: float = 0.90,
    student_lo: float | None = 2.0,
    teacher_passed: bool | None = True,
    with_teacher: bool = True,
    candidate: str = "test candidate",
    rubric: str = "test rubric",
    tags: dict[str, Any] | None = None,
) -> TelemetryRecord:
    traces: list[LayerTrace] = [
        LayerTrace(
            layer_name="tier1_student",
            status=status,
            latency_ms=2.0,
            tokens_generated=0,
            calibrated_prob=probability,
            log_odds_pass=student_lo,
            forward_log_odds=student_lo,
            reverse_log_odds=student_lo,
            positional_bias_magnitude=0.05,
        )
    ]
    if with_teacher:
        traces.append(
            LayerTrace(
                layer_name="tier2_teacher",
                status=DecisionStatus.PASSED if teacher_passed else DecisionStatus.FAILED,
                latency_ms=15.0,
                tokens_generated=25,
                calibrated_prob=0.99 if teacher_passed else 0.01,
                log_odds_pass=4.5 if teacher_passed else -4.5,
                diagnostic_report="Teacher evaluation reasoning",
            )
        )
    return TelemetryRecord(
        gate_name="CascadeGate",
        gate_type="Cascade",
        model="student-model-7b",
        backend="stub://default",
        teacher_model="teacher-model-70b",
        teacher_passed=teacher_passed,
        rubric=rubric,
        candidate=candidate,
        candidate_sha256=hash_candidate(candidate),
        candidate_length=len(candidate),
        passed=passed,
        status=status,
        probability=probability,
        triggered_layer="tier2_teacher" if with_teacher else "tier1_student",
        escalated=with_teacher,
        fast_pathed=False,
        total_latency_ms=17.0,
        total_tokens_generated=25 if with_teacher else 0,
        reason="evaluated",
        traces=traces,
        tags=tags or {},
    )


# ==============================================================================
# SECTION 1: STREAMING UNIT & MUTANT-KILLING TESTS (src/logitgate/streaming.py)
# ==============================================================================


def test_streaming_extract_chunk_text_edge_cases() -> None:
    # Custom chunk_extractor returning non-string -> TypeError
    with pytest.raises(TypeError, match="chunk_extractor must return a str, got int"):
        _extract_chunk_text(123, chunk_extractor=lambda x: 42)  # type: ignore[arg-type,return-value]

    # Valid string chunk
    assert _extract_chunk_text("hello world") == "hello world"

    # Mapping with response
    assert _extract_chunk_text({"response": "chunk-resp"}) == "chunk-resp"
    assert _extract_chunk_text({"response": None}) == ""

    # Mapping with message.content
    assert _extract_chunk_text({"message": {"content": "msg-chunk"}}) == "msg-chunk"
    assert _extract_chunk_text({"message": {"content": None}}) == ""

    # Mapping with delta.content
    assert _extract_chunk_text({"delta": {"content": "delta-chunk"}}) == "delta-chunk"
    assert _extract_chunk_text({"delta": {"content": None}}) == ""

    # Mapping with content
    assert _extract_chunk_text({"content": "content-chunk"}) == "content-chunk"
    assert _extract_chunk_text({"content": None}) == ""

    # Mapping without valid keys -> TypeError
    with pytest.raises(TypeError, match="Mapping stream chunk does not contain"):
        _extract_chunk_text({"unknown_key": "val"})

    # OpenAI-style object with choices
    # Empty choices
    assert _extract_chunk_text(SimpleNamespace(choices=[])) == ""

    # delta.content is None
    delta_none = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
    assert _extract_chunk_text(delta_none) == ""

    # delta.content is string
    delta_str = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="tok"))])
    assert _extract_chunk_text(delta_str) == "tok"

    # Unsupported type -> TypeError
    with pytest.raises(TypeError, match="Unsupported stream chunk type int"):
        _extract_chunk_text(9999)


def test_streaming_clause_boundary_detection() -> None:
    # Empty list
    assert not _ends_at_clause_boundary([])

    # Terminators with trailing spaces
    assert _ends_at_clause_boundary(["Hello world.", "   "])
    assert _ends_at_clause_boundary(["Sentence finished!\t\n"])
    assert _ends_at_clause_boundary(["Question asked? "])
    assert _ends_at_clause_boundary(["Clause; "])
    assert _ends_at_clause_boundary(["Header:\n"])
    assert _ends_at_clause_boundary(["JSON object: {\"key\": \"value\"}"])
    assert _ends_at_clause_boundary(["JSON array: [1, 2, 3]"])

    # Not ending at clause boundary
    assert not _ends_at_clause_boundary(["Hello world", " still talking"])
    assert not _ends_at_clause_boundary(["Trailing whitespace only    "])


def test_stream_guard_parameter_validation_errors() -> None:
    gate = DummyGate()

    # gate not BaseGate
    with pytest.raises(ConfigurationError, match="gate must be a BaseGate instance"):
        StreamGuard(gate="not a gate")  # type: ignore[arg-type]

    # min_chars validations
    with pytest.raises(ConfigurationError, match="min_chars must be an integer >= 1"):
        StreamGuard(gate, min_chars=True)
    with pytest.raises(ConfigurationError, match="min_chars must be an integer >= 1"):
        StreamGuard(gate, min_chars=0)
    with pytest.raises(ConfigurationError, match="min_chars must be an integer >= 1"):
        StreamGuard(gate, min_chars=-5)

    # stride_chars validations
    with pytest.raises(ConfigurationError, match="stride_chars must be an integer >= 1"):
        StreamGuard(gate, stride_chars=True)
    with pytest.raises(ConfigurationError, match="stride_chars must be an integer >= 1"):
        StreamGuard(gate, stride_chars=0)

    # stride_chunks validations
    with pytest.raises(ConfigurationError, match="stride_chunks must be an integer >= 1"):
        StreamGuard(gate, stride_chunks=True)
    with pytest.raises(ConfigurationError, match="stride_chunks must be an integer >= 1"):
        StreamGuard(gate, stride_chunks=0)

    # clause_boundary validation
    with pytest.raises(ConfigurationError, match="clause_boundary must be a bool"):
        StreamGuard(gate, clause_boundary="yes")  # type: ignore[arg-type]

    # max_wait_chars validations
    with pytest.raises(ConfigurationError, match="max_wait_chars must be an integer >= 1"):
        StreamGuard(gate, max_wait_chars=True)
    with pytest.raises(ConfigurationError, match="max_wait_chars must be an integer >= 1"):
        StreamGuard(gate, max_wait_chars=0)

    # consecutive_failures validations
    with pytest.raises(ConfigurationError, match="consecutive_failures must be an integer >= 1"):
        StreamGuard(gate, consecutive_failures=True)
    with pytest.raises(ConfigurationError, match="consecutive_failures must be an integer >= 1"):
        StreamGuard(gate, consecutive_failures=0)

    # abort_on_ambiguous validation
    with pytest.raises(ConfigurationError, match="abort_on_ambiguous must be a bool"):
        StreamGuard(gate, abort_on_ambiguous="true")  # type: ignore[arg-type]

    # abort_prob_threshold validations
    with pytest.raises(ConfigurationError, match="abort_prob_threshold must be a float"):
        StreamGuard(gate, abort_prob_threshold=True)
    with pytest.raises(ConfigurationError, match="abort_prob_threshold must be in \\[0.0, 1.0\\]"):
        StreamGuard(gate, abort_prob_threshold=-0.1)
    with pytest.raises(ConfigurationError, match="abort_prob_threshold must be in \\[0.0, 1.0\\]"):
        StreamGuard(gate, abort_prob_threshold=1.5)

    # final_check validation
    with pytest.raises(ConfigurationError, match="final_check must be a bool"):
        StreamGuard(gate, final_check="no")  # type: ignore[arg-type]

    # fail_on_final_ambiguous validation
    with pytest.raises(ConfigurationError, match="fail_on_final_ambiguous must be a bool"):
        StreamGuard(gate, fail_on_final_ambiguous="maybe")  # type: ignore[arg-type]

    # delivery validation
    with pytest.raises(ConfigurationError, match="delivery must be 'optimistic' or 'buffered'"):
        StreamGuard(gate, delivery="instant")  # type: ignore[arg-type]

    # raise_on_abort validation
    with pytest.raises(ConfigurationError, match="raise_on_abort must be a bool"):
        StreamGuard(gate, raise_on_abort=1)  # type: ignore[arg-type]

    # close_upstream_on_abort validation
    with pytest.raises(ConfigurationError, match="close_upstream_on_abort must be a bool"):
        StreamGuard(gate, close_upstream_on_abort="yes")  # type: ignore[arg-type]

    # max_buffer_chars validations
    with pytest.raises(ConfigurationError, match="max_buffer_chars must be an integer >= 1"):
        StreamGuard(gate, max_buffer_chars=True)
    with pytest.raises(ConfigurationError, match="max_buffer_chars must be an integer >= 1"):
        StreamGuard(gate, max_buffer_chars=0)

    # chunk_extractor validation
    with pytest.raises(ConfigurationError, match="chunk_extractor must be callable or None"):
        StreamGuard(gate, chunk_extractor="extract")  # type: ignore[arg-type]

    # stream(source) passing str or bytes -> ConfigurationError
    guard = StreamGuard(gate)
    with pytest.raises(ConfigurationError, match="source must be an iterable of stream chunks"):
        guard.stream("raw string is invalid")
    with pytest.raises(ConfigurationError, match="source must be an iterable of stream chunks"):
        guard.stream(b"raw bytes are invalid")


def test_stream_guard_call_and_close() -> None:
    gate = DummyGate()
    guard = StreamGuard(gate)

    chunks = ["chunk 1 ", "chunk 2 ", "chunk 3."]
    # __call__ delegates to stream
    stream_iter = guard(chunks)
    assert isinstance(stream_iter, GuardedStream)
    assert iter(stream_iter) is stream_iter
    first = next(stream_iter)
    assert first == "chunk 1 "
    # close stream early
    stream_iter.close()


@pytest.mark.asyncio
async def test_async_guarded_stream_lifecycle_and_aclose() -> None:
    gate = DummyGate()
    guard = StreamGuard(gate)

    async def _source() -> AsyncIterator[str]:
        yield "part 1 "
        yield "part 2 "
        yield "part 3."

    astream = guard.astream(_source())
    assert isinstance(astream, AsyncGuardedStream)
    assert astream.__aiter__() is astream
    item1 = await astream.__anext__()
    assert item1 == "part 1 "
    await astream.aclose()


def test_stream_buffer_overflow_sync_and_async() -> None:
    gate = DummyGate()
    # Tiny buffer: 10 chars
    guard_raising = StreamGuard(gate, max_buffer_chars=10, raise_on_abort=True)

    chunks = ["short", " overflow string that exceeds ten characters."]
    with pytest.raises(StreamAbortedError) as exc_info:
        list(guard_raising.stream(chunks))
    assert exc_info.value.verdict.traces[0].failure_code == "STREAM_BUFFER_OVERFLOW"
    assert not exc_info.value.is_final

    # Sync non-raising overflow
    guard_quiet = StreamGuard(gate, max_buffer_chars=10, raise_on_abort=False)
    stream_obj = guard_quiet.stream(chunks)
    yielded = list(stream_obj)
    assert len(yielded) >= 1
    assert stream_obj.summary is not None
    assert stream_obj.summary.aborted
    assert stream_obj.final_verdict is not None
    assert stream_obj.final_verdict.traces[0].failure_code == "STREAM_BUFFER_OVERFLOW"

    # Async overflow raising
    async def _async_chunks() -> AsyncIterator[str]:
        for c in chunks:
            yield c

    async def _run_async_overflow_raising() -> None:
        astream_obj = guard_raising.astream(_async_chunks())
        with pytest.raises(StreamAbortedError) as a_exc:
            async for _ in astream_obj:
                pass
        assert a_exc.value.verdict.traces[0].failure_code == "STREAM_BUFFER_OVERFLOW"

    asyncio.run(_run_async_overflow_raising())

    # Async overflow quiet
    async def _run_async_overflow_quiet() -> None:
        astream_quiet = guard_quiet.astream(_async_chunks())
        out = [item async for item in astream_quiet]
        assert len(out) >= 1
        assert astream_quiet.summary is not None
        assert astream_quiet.summary.aborted
        assert astream_quiet.final_verdict is not None
        assert astream_quiet.final_verdict.traces[0].failure_code == "STREAM_BUFFER_OVERFLOW"

    asyncio.run(_run_async_overflow_quiet())


def test_stream_guard_optimistic_vs_buffered_delivery() -> None:
    # Fail on first check
    fail_verdict = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.1,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="violation detected",
    )
    fail_gate = DummyGate(verdicts=[fail_verdict])

    chunks = ["Dangerous prefix that triggers early check immediately.", " Leaked data."]

    # Optimistic yields before checking
    opt_guard = StreamGuard(
        fail_gate,
        min_chars=10,
        stride_chars=10,
        clause_boundary=False,
        delivery="optimistic",
        raise_on_abort=False,
    )
    opt_stream = opt_guard.stream(chunks)
    opt_yielded = list(opt_stream)
    assert len(opt_yielded) > 0  # Optimistic leaked chunk before abort
    assert opt_stream.summary is not None
    assert opt_stream.summary.aborted

    # Buffered does NOT yield chunks that fail
    fail_gate_buf = DummyGate(verdicts=[fail_verdict])
    buf_guard = StreamGuard(
        fail_gate_buf,
        min_chars=10,
        stride_chars=10,
        clause_boundary=False,
        delivery="buffered",
        raise_on_abort=False,
    )
    buf_stream = buf_guard.stream(chunks)
    buf_yielded = list(buf_stream)
    assert buf_yielded == []  # ZERO chunks leaked!
    assert buf_stream.summary is not None
    assert buf_stream.summary.aborted


def test_stream_guard_abort_on_ambiguous_and_prob_threshold() -> None:
    ambig_verdict = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        probability=0.45,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="uncertain content",
    )

    # 1. abort_on_ambiguous=False -> does not trigger strike
    gate1 = DummyGate(verdicts=[ambig_verdict])
    guard1 = StreamGuard(
        gate1,
        min_chars=10,
        stride_chars=10,
        clause_boundary=False,
        abort_on_ambiguous=False,
        raise_on_abort=False,
    )
    s1 = guard1.stream(["Some ambiguous sentence.", " Ending peacefully."])
    list(s1)
    assert s1.summary is not None
    assert not s1.summary.aborted_mid_stream

    # 2. abort_on_ambiguous=True with abort_prob_threshold=0.50 (0.45 <= 0.50 -> strikes and aborts)
    gate2 = DummyGate(verdicts=[ambig_verdict])
    guard2 = StreamGuard(
        gate2,
        min_chars=10,
        stride_chars=10,
        clause_boundary=False,
        abort_on_ambiguous=True,
        abort_prob_threshold=0.50,
        raise_on_abort=False,
    )
    s2 = guard2.stream(["Some ambiguous sentence.", " Ending peacefully."])
    list(s2)
    assert s2.summary is not None
    assert s2.summary.aborted_mid_stream

    # 3. abort_on_ambiguous=True with abort_prob_threshold=0.30 (0.45 > 0.30 -> not a strike)
    gate3 = DummyGate(verdicts=[ambig_verdict])
    guard3 = StreamGuard(
        gate3,
        min_chars=10,
        stride_chars=10,
        clause_boundary=False,
        abort_on_ambiguous=True,
        abort_prob_threshold=0.30,
        raise_on_abort=False,
    )
    s3 = guard3.stream(["Some ambiguous sentence.", " Ending peacefully."])
    list(s3)
    assert s3.summary is not None
    assert not s3.summary.aborted_mid_stream


def test_stream_guard_final_check_variations() -> None:
    pass_verdict = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.95,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="clean",
    )
    gate = DummyGate(verdicts=[pass_verdict])

    # final_check=False with mid-stream check -> reuses last check verdict
    guard_no_final = StreamGuard(
        gate,
        min_chars=10,
        stride_chars=10,
        clause_boundary=False,
        final_check=False,
    )
    s = guard_no_final.stream(["This chunk triggers mid-stream check.", "."])
    list(s)
    assert s.final_verdict == pass_verdict

    # final_check=False with no mid-stream checks -> final_verdict is None
    gate_short = DummyGate()
    guard_short = StreamGuard(
        gate_short,
        min_chars=1000,
        final_check=False,
    )
    s_short = guard_short.stream(["Tiny."])
    list(s_short)
    assert s_short.final_verdict is None

    # fail_on_final_ambiguous=False and abort_on_ambiguous=False with final ambiguous -> passes
    ambig_final = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        probability=0.5,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="ambiguous end",
    )
    gate_ambig = DummyGate(verdicts=[ambig_final])
    guard_allow_ambig = StreamGuard(
        gate_ambig,
        min_chars=500,  # No mid-stream check
        final_check=True,
        fail_on_final_ambiguous=False,
        abort_on_ambiguous=False,
    )
    s_allow = guard_allow_ambig.stream(["Single chunk that ends ambiguously."])
    list(s_allow)
    assert s_allow.summary is not None
    assert s_allow.summary.passed
    assert not s_allow.summary.aborted


def test_stream_guard_cascade_backend_telemetry_extraction() -> None:
    subgate = DummyGate(name="SubLogit", backend="stub://test", model="sub-model")
    cascade = Cascade([subgate])
    # Cascade itself has backend=None, but child subgate has backend!
    assert getattr(cascade, "backend", None) is None

    sink = MemoryTelemetrySink()
    guard = StreamGuard(cascade, telemetry_sink=sink, min_chars=5, stride_chars=5)
    list(guard.stream(["Valid chunk 1.", " Valid chunk 2."]))
    assert len(sink.records) >= 1
    assert sink.records[0].backend == "stub://test"


def test_stream_guard_summary_serialization() -> None:
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=2.5,
        total_tokens_generated=0,
        reason="ok",
    )
    rec = StreamCheckRecord(chunk_index=0, prefix_chars=12, is_final=False, verdict=v)
    rec_d = rec.to_dict()
    assert rec_d["chunk_index"] == 0
    assert rec_d["prefix_chars"] == 12
    assert rec_d["passed"] is True

    summary = StreamGuardSummary(
        passed=True,
        aborted=False,
        aborted_mid_stream=False,
        abort_chunk_index=None,
        total_chunks_seen=2,
        chunks_yielded=2,
        chars_seen=30,
        chars_yielded=30,
        checks_executed=1,
        total_gate_latency_ms=2.5,
        final_verdict=v,
        checks=(rec,),
    )
    sum_d = summary.to_dict()
    assert sum_d["passed"] is True
    assert sum_d["total_chunks_seen"] == 2
    assert len(sum_d["checks"]) == 1


# ==============================================================================
# SECTION 2: CACHE UNIT & MUTANT-KILLING TESTS (src/logitgate/cache.py)
# ==============================================================================


class DummyPydanticModel(BaseModel):
    name: str
    count: int


class CustomUnserializableObject:
    def __init__(self, val: str) -> None:
        self.val = val

    def __repr__(self) -> str:
        return f"<CustomObject {self.val}>"


def test_canonicalize_value_deep_types_and_overflow() -> None:
    # 1. Depth overflow (> 16)
    nested: Any = "leaf"
    for _ in range(20):
        nested = [nested]
    canon = _canonicalize_value(nested)
    assert canon["t"] == "list"
    # Traversing down to depth 17
    curr = canon
    while isinstance(curr, dict) and curr.get("t") == "list":
        curr = curr["v"][0]
    assert curr["t"] == "depth_capped"

    # 2. NaNs and Infs
    assert _canonicalize_value(float("nan")) == {"t": "float", "v": "NaN"}
    assert _canonicalize_value(float("inf")) == {"t": "float", "v": "Inf"}
    assert _canonicalize_value(float("-inf")) == {"t": "float", "v": "-Inf"}

    # 3. Bytes and bytearrays
    assert _canonicalize_value(b"hello") == {"t": "bytes", "v": b"hello".hex()}
    assert _canonicalize_value(bytearray(b"world")) == {"t": "bytes", "v": b"world".hex()}

    # 4. Pydantic BaseModel instance and class
    m = DummyPydanticModel(name="test", count=42)
    canon_m = _canonicalize_value(m)
    assert "pydantic:" in canon_m["t"]
    assert canon_m["v"]["t"] == "map"

    canon_cls = _canonicalize_value(DummyPydanticModel)
    assert canon_cls["t"] == "pydantic_schema"

    # 5. Tuple, Sets, Frozensets
    tup = (1, "a", True)
    assert _canonicalize_value(tup)["t"] == "tuple"

    s = {"banana", "apple", "cherry"}
    canon_s = _canonicalize_value(s)
    assert canon_s["t"] == "set"
    # Ensure set elements are stably sorted
    elems = [item["v"] for item in canon_s["v"]]
    assert elems == sorted(elems)

    # 6. Callable with code vs without code (built-in len)
    def my_fn(x: int) -> int:
        return x + 1

    fn_canon = _canonicalize_value(my_fn)
    assert fn_canon["t"] == "callable"
    assert "code" in fn_canon["v"]

    builtin_canon = _canonicalize_value(len)
    assert builtin_canon["t"] == "callable"
    assert "repr" in builtin_canon["v"]

    # 7. Arbitrary object fallback
    custom_obj = CustomUnserializableObject("test123")
    obj_canon = _canonicalize_value(custom_obj)
    assert "obj:" in obj_canon["t"]
    assert "<CustomObject test123>" in obj_canon["v"]


def test_backend_and_profile_fingerprints() -> None:
    # backend is None
    fp_none = _backend_fingerprint(None, "model-x")
    assert fp_none == {"spec": None, "model": "model-x"}

    # Custom backend with options, lora, and cache_id
    class CustomBackendWithId:
        def __init__(self) -> None:
            self.cache_id = "custom-id-99"
            self.options = {"temperature": 0.2}
            self.lora_path = "/path/to/adapter"
            self.lora_scale = 0.75

    cust = CustomBackendWithId()
    cust_fp = _backend_fingerprint(cust, "custom-model")
    assert cust_fp["spec"].startswith("custom://")
    assert cust_fp["cache_id"]["v"] == "custom-id-99"
    assert cust_fp["lora_path"] == "/path/to/adapter"
    assert cust_fp["lora_scale"] == 0.75

    # Profile fingerprint
    assert _profile_fingerprint(None) is None

    # Custom profile without model_dump
    class CustomProfile:
        def __repr__(self) -> str:
            return "Profile(alpha=0.05)"

    p_fp = _profile_fingerprint(CustomProfile())
    assert p_fp == {"t": "str", "v": "Profile(alpha=0.05)"}


def test_gate_cache_fingerprint_advanced_components() -> None:
    class MockComponent:
        def cache_fingerprint(self) -> dict[str, Any]:
            return {"component": "mock_val"}

    class FullFeaturedGate(BaseGate):
        def __init__(self) -> None:
            self.adaptive_controller = MockComponent()
            self.anchor_bank = MockComponent()
            self.meta_calibrator = MockComponent()
            self.unanchored_meta_calibrator = MockComponent()

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(
                passed=True,
                status=DecisionStatus.PASSED,
                probability=1.0,
                triggered_layer="FullFeaturedGate",
                total_latency_ms=0.0,
                total_tokens_generated=0,
                reason="mock",
            )

    fg = FullFeaturedGate()
    fp = gate_cache_fingerprint(fg)
    assert "adaptive_controller" in fp
    assert "anchor_bank" in fp
    assert "meta_calibrator" in fp
    assert "unanchored_meta_calibrator" in fp

    # Custom gate without replay or backend records public attributes
    class SimpleAttrGate(BaseGate):
        def __init__(self) -> None:
            self.custom_flag = 42
            self.custom_str = "flag_val"

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(
                passed=True,
                status=DecisionStatus.PASSED,
                probability=1.0,
                triggered_layer="SimpleAttrGate",
                total_latency_ms=0.0,
                total_tokens_generated=0,
                reason="mock",
            )

    sag = SimpleAttrGate()
    sag_fp = gate_cache_fingerprint(sag)
    assert "custom_attrs" in sag_fp
    assert sag_fp["custom_attrs"]["custom_flag"]["v"] == 42
    assert sag_fp["custom_attrs"]["custom_str"]["v"] == "flag_val"


def test_sqlite_security_path_hardening(tmp_path: Path) -> None:
    # 1. Reject ':memory:'
    with pytest.raises(ConfigurationError, match="Pass path=None for in-memory VerdictCache"):
        _prepare_secure_sqlite_path(":memory:")

    # 2. Reject symlink parent
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    sym_dir = tmp_path / "sym_dir"
    sym_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="Refusing to open VerdictCache inside symlinked directory"):
        _prepare_secure_sqlite_path(sym_dir / "cache.db")

    # 3. Reject symlink file
    real_file = tmp_path / "real_cache.db"
    real_file.touch()
    sym_file = tmp_path / "sym_cache.db"
    sym_file.symlink_to(real_file)
    with pytest.raises(ConfigurationError, match="Refusing to open symlinked VerdictCache file"):
        _prepare_secure_sqlite_path(sym_file)


def test_verdict_cache_validation_errors() -> None:
    # max_entries
    with pytest.raises(ConfigurationError, match="max_entries must be an integer >= 1"):
        VerdictCache(max_entries=True)
    with pytest.raises(ConfigurationError, match="max_entries must be an integer >= 1"):
        VerdictCache(max_entries=0)

    # max_disk_entries
    with pytest.raises(ConfigurationError, match="max_disk_entries must be an integer >= 1"):
        VerdictCache(max_disk_entries=True)
    with pytest.raises(ConfigurationError, match="max_disk_entries must be an integer >= 1"):
        VerdictCache(max_disk_entries=0)

    # ttl_seconds
    with pytest.raises(ConfigurationError, match="ttl_seconds must be a positive finite float"):
        VerdictCache(ttl_seconds=True)
    with pytest.raises(ConfigurationError, match="ttl_seconds must be a positive finite float"):
        VerdictCache(ttl_seconds=-1.0)
    with pytest.raises(ConfigurationError, match="ttl_seconds must be a positive finite float"):
        VerdictCache(ttl_seconds=float("inf"))

    # cache_ambiguous
    with pytest.raises(ConfigurationError, match="cache_ambiguous must be a bool"):
        VerdictCache(cache_ambiguous="yes")  # type: ignore[arg-type]

    # namespace
    with pytest.raises(ConfigurationError, match="namespace must be a str"):
        VerdictCache(namespace=123)  # type: ignore[arg-type]


def test_verdict_cache_sqlite_corrupt_row_and_eviction(tmp_path: Path) -> None:
    db_path = tmp_path / "corrupt_test.db"
    gate = DummyGate()
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.5,
        total_tokens_generated=0,
        reason="ok",
    )

    # Populate valid row
    with VerdictCache(db_path) as cache:
        cache.put(gate, "candidate-1", v)

    # Manually corrupt verdict_json in sqlite
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE verdict_cache SET verdict_json = 'NOT VALID JSON'")
    conn.commit()
    conn.close()

    # Reopen cache: fetch should warn about corrupt row and evict it
    with VerdictCache(db_path) as cache:
        with pytest.warns(CacheWarning, match="Evicting corrupt VerdictCache row"):
            hit = cache.get(gate, "candidate-1")
        assert hit is None

        # Verify it was deleted from DB
        stats = cache.stats()
        assert stats.disk_entries == 0

    # Max disk entries eviction test
    small_disk_db = tmp_path / "small_disk.db"
    with VerdictCache(small_disk_db, max_disk_entries=5) as disk_cache:
        for idx in range(10):
            disk_cache.put(gate, f"cand-{idx}", v)
        stats = disk_cache.stats()
        assert stats.disk_entries <= 5
        assert stats.evictions > 0


def test_verdict_cache_evaluate_and_aevaluate() -> None:
    gate = LogitGate(rubric="is helpful", backend=StubBackend(default_prob=0.99))
    with VerdictCache() as cache:
        v_sync = cache.evaluate(gate, "test sync cand")
        assert v_sync.passed
        # Second call should be a cache hit
        v_hit = cache.evaluate(gate, "test sync cand")
        assert v_hit.traces[0].metadata.get("cache_hit") is True

        # Async evaluate
        async def _test_async() -> None:
            v_async = await cache.aevaluate(gate, "test async cand")
            assert v_async.passed
            v_async_hit = await cache.aevaluate(gate, "test async cand")
            assert v_async_hit.traces[0].metadata.get("cache_hit") is True

        asyncio.run(_test_async())


def test_resolve_verdict_cache_and_call_kwargs_scenarios(tmp_path: Path) -> None:
    # Bad cache paths
    with pytest.raises(ConfigurationError, match="cache path cannot be empty"):
        resolve_verdict_cache("   ")
    with pytest.raises(ConfigurationError, match="cache must be a VerdictCache, file path, False, or None"):
        resolve_verdict_cache(12345)  # type: ignore[arg-type]

    # Temp cache path via call_kwargs
    gate = DummyGate()
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.5,
        total_tokens_generated=0,
        reason="ok",
    )

    temp_db = str(tmp_path / "temp_call_cache.db")
    store_cached_verdict(
        gate,
        "cand-temp",
        v,
        call_kwargs={"cache": temp_db},
    )
    hit = lookup_cached_verdict(
        gate,
        "cand-temp",
        call_kwargs={"cache": temp_db},
    )
    assert hit is not None
    assert hit.passed

    # Bypass cache via call_kwargs
    assert lookup_cached_verdict(gate, "cand-temp", call_kwargs={"_bypass_cache": True}) is None
    assert lookup_cached_verdict(gate, "cand-temp", call_kwargs={"cache": False}) is None


# ==============================================================================
# SECTION 3: DISTILL UNIT & MUTANT-KILLING TESTS (src/logitgate/telemetry/distill.py)
# ==============================================================================


def test_extract_distillation_record_mode_and_eligibility() -> None:
    # 1. Invalid mode -> ConfigurationError
    tel = _make_telemetry_record()
    with pytest.raises(ConfigurationError, match="Invalid distillation mode"):
        extract_distillation_record(tel, mode="invalid_mode")  # type: ignore[arg-type]

    # 2. Missing/empty rubric or None candidate
    no_rubric = _make_telemetry_record(rubric="  ")
    assert extract_distillation_record(no_rubric) is None
    no_cand = _make_telemetry_record()
    no_cand = no_cand.model_copy(update={"candidate": None})
    assert extract_distillation_record(no_cand) is None

    # 3. Missing student or teacher trace
    student_only = _make_telemetry_record(with_teacher=False)
    assert extract_distillation_record(student_only) is None

    # 4. MultiRubricGate sub-trace ("rubric" in student trace metadata)
    multirubric_rec = _make_telemetry_record()
    multirubric_rec.traces[0].metadata["rubric"] = "sub_criterion"
    assert extract_distillation_record(multirubric_rec) is None

    # 5. Teacher judge_log_odds in metadata
    meta_judge_pos = _make_telemetry_record(teacher_passed=None)
    meta_judge_pos.traces[1].metadata["judge_log_odds"] = 2.5
    d_pos = extract_distillation_record(meta_judge_pos)
    assert d_pos is not None
    assert d_pos.teacher_passed is True

    meta_judge_neg = _make_telemetry_record(teacher_passed=None)
    meta_judge_neg.traces[1].metadata["judge_log_odds"] = -2.5
    d_neg = extract_distillation_record(meta_judge_neg)
    assert d_neg is not None
    assert d_neg.teacher_passed is False

    # 6. Mode discrepancies_only and ambiguous_resolved -> returns None
    ambig_resolved = _make_telemetry_record(
        status=DecisionStatus.AMBIGUOUS,
        student_lo=1.5,
        teacher_passed=True,
    )
    assert (
        extract_distillation_record(ambig_resolved, mode="discrepancies_only") is None
    )


def test_extract_distillation_record_discrepancy_types() -> None:
    # 1. false_accept_caught: student PASSED, teacher False
    fa = _make_telemetry_record(
        status=DecisionStatus.PASSED,
        student_lo=2.0,
        teacher_passed=False,
    )
    d_fa = extract_distillation_record(fa)
    assert d_fa is not None
    assert d_fa.discrepancy_type == "false_accept_caught"

    # 2. false_reject_overturned: student FAILED, teacher True
    fr = _make_telemetry_record(
        status=DecisionStatus.FAILED,
        student_lo=-2.0,
        teacher_passed=True,
    )
    d_fr = extract_distillation_record(fr)
    assert d_fr is not None
    assert d_fr.discrepancy_type == "false_reject_overturned"

    # 3. polarity_flip: student AMBIGUOUS, student_lo > 0 != teacher False
    pf = _make_telemetry_record(
        status=DecisionStatus.AMBIGUOUS,
        student_lo=1.0,
        teacher_passed=False,
    )
    d_pf = extract_distillation_record(pf)
    assert d_pf is not None
    assert d_pf.discrepancy_type == "polarity_flip"


def test_distillation_sink_parameter_validation_and_dedupe(tmp_path: Path) -> None:
    sink_path = tmp_path / "distill.jsonl"

    # Invalid mode
    with pytest.raises(ConfigurationError, match="Invalid mode"):
        DistillationSink(sink_path, mode="bad_mode")  # type: ignore[arg-type]

    # dedupe_by_hash
    with pytest.raises(ConfigurationError, match="dedupe_by_hash must be a bool"):
        DistillationSink(sink_path, dedupe_by_hash="yes")  # type: ignore[arg-type]

    # max_dedupe_keys
    with pytest.raises(ConfigurationError, match="max_dedupe_keys must be an integer >= 1"):
        DistillationSink(sink_path, max_dedupe_keys=0)

    # max_bytes
    with pytest.raises(ConfigurationError, match="max_bytes must be an integer >= 256"):
        DistillationSink(sink_path, max_bytes=100)

    # backup_count
    with pytest.raises(ConfigurationError, match="backup_count must be an integer >= 0"):
        DistillationSink(sink_path, backup_count=-1)

    # fsync
    with pytest.raises(ConfigurationError, match="fsync must be a bool"):
        DistillationSink(sink_path, fsync="no")  # type: ignore[arg-type]

    # Functional emit, rotation, dedupe eviction, and close
    rec = _make_telemetry_record(candidate="cand-1")
    rec2 = _make_telemetry_record(candidate="cand-2")
    rec3 = _make_telemetry_record(candidate="cand-3")

    sink = DistillationSink(
        sink_path,
        max_dedupe_keys=2,
        fsync=True,
    )
    sink.emit(rec)
    sink.emit(rec)  # Duplicate skipped
    assert sink.records_written == 1

    sink.emit(rec2)
    sink.emit(rec3)  # Evicts rec from dedupe keys
    assert sink.records_written == 3

    sink.flush()
    sink.close()
    # Emit after close does nothing
    sink.emit(rec)
    assert sink.records_written == 3


def test_load_distillation_records_corrupt_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "distill_corrupt.jsonl"

    # Line not a JSON dict
    file_path.write_text("12345\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="JSONL line must decode to an object"):
        load_distillation_records(file_path)

    # Malformed JSON
    file_path.write_text("NOT JSON\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Malformed JSONL record"):
        load_distillation_records(file_path)


def test_recalibrate_from_telemetry_edge_cases() -> None:
    # 1. dedupe not bool
    with pytest.raises(ConfigurationError, match="dedupe must be a bool"):
        recalibrate_from_telemetry([], dedupe="yes")  # type: ignore[arg-type]

    # 2. Sequence with invalid type
    with pytest.raises(ConfigurationError, match="Expected DistillationRecord or TelemetryRecord"):
        recalibrate_from_telemetry([123])  # type: ignore[list-item]

    # 3. Empty records -> CalibrationError
    with pytest.raises(CalibrationError, match="No teacher-adjudicated binary escalation records"):
        recalibrate_from_telemetry([])

    # 4. Fewer than 4 samples
    r1 = _make_telemetry_record(student_lo=2.0, teacher_passed=True)
    d1 = extract_distillation_record(r1)
    assert d1 is not None
    with pytest.raises(CalibrationError, match="Telemetry recalibration requires at least 4"):
        recalibrate_from_telemetry([d1])

    # 5. All same label
    d_list = [
        extract_distillation_record(_make_telemetry_record(candidate=f"c{i}", student_lo=2.0, teacher_passed=True))
        for i in range(5)
    ]
    valid_d = [x for x in d_list if x is not None]
    with pytest.raises(CalibrationError, match="requires both positive .* and negative"):
        recalibrate_from_telemetry(valid_d)


def test_export_distillation_dataset_validation_and_symlinks(tmp_path: Path) -> None:
    out_file = tmp_path / "export.jsonl"

    # Format validation
    with pytest.raises(ConfigurationError, match="Invalid export format"):
        export_distillation_dataset([], out_file, format="xml")  # type: ignore[arg-type]

    # bool validations
    with pytest.raises(ConfigurationError, match="include_symmetric_swaps must be a bool"):
        export_distillation_dataset([], out_file, include_symmetric_swaps="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="dedupe must be a bool"):
        export_distillation_dataset([], out_file, dedupe="no")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="discrepancies_only must be a bool"):
        export_distillation_dataset([], out_file, discrepancies_only="maybe")  # type: ignore[arg-type]

    # Symlink protection
    real_out = tmp_path / "real_out.jsonl"
    real_out.touch()
    sym_out = tmp_path / "sym_out.jsonl"
    sym_out.symlink_to(real_out)
    with pytest.raises(ConfigurationError, match="Refusing to .* through symlink"):
        export_distillation_dataset([], sym_out)


# ==============================================================================
# SECTION 4: SINKS UNIT & MUTANT-KILLING TESTS (src/logitgate/telemetry/sinks.py)
# ==============================================================================


def test_sinks_validation_helpers() -> None:
    # Privacy validation
    with pytest.raises(ConfigurationError, match="Invalid privacy"):
        _validate_privacy_and_bounds("super_private", None)  # type: ignore[arg-type]

    # max_candidate_chars validation
    with pytest.raises(ConfigurationError, match="max_candidate_chars must be an integer >= 1"):
        _validate_privacy_and_bounds("full", max_candidate_chars=True)
    with pytest.raises(ConfigurationError, match="max_candidate_chars must be an integer >= 1"):
        _validate_privacy_and_bounds("full", max_candidate_chars=0)

    # on_error validation
    with pytest.raises(ConfigurationError, match="Invalid on_error"):
        _validate_privacy_and_bounds("full", None, on_error="crash")  # type: ignore[arg-type]

    # Target path validation
    with pytest.raises(ConfigurationError, match="path cannot be empty or whitespace"):
        _validate_target_file_path("   ", "TestSink")
    with pytest.raises(ConfigurationError, match="path must be a str or Path"):
        _validate_target_file_path(12345, "TestSink")  # type: ignore[arg-type]


def test_sinks_symlink_and_nonregular_protections(tmp_path: Path) -> None:
    # Symlink target
    real_target = tmp_path / "real_target.jsonl"
    real_target.touch()
    sym_target = tmp_path / "sym_target.jsonl"
    sym_target.symlink_to(real_target)

    with pytest.raises(ConfigurationError, match="Refusing to access TestSink through symlink"):
        _validate_target_file_path(sym_target, "TestSink")

    with pytest.raises(OSError, match="Refusing to open symlink telemetry target"):
        open_append_nofollow(sym_target)

    with pytest.raises(ConfigurationError, match="Refusing to read TestSink from symlink"):
        read_regular_text_nofollow(sym_target, max_bytes=1000, label="TestSink")

    with pytest.raises(OSError, match="Refusing to rotate symlink telemetry file"):
        rotate_jsonl_files(sym_target, backup_count=2)

    # Backup count <= 0 unlinks target
    to_unlink = tmp_path / "unlink_me.jsonl"
    to_unlink.touch()
    rotate_jsonl_files(to_unlink, backup_count=0)
    assert not to_unlink.exists()


def test_jsonl_telemetry_sink_validation_and_filtering(tmp_path: Path) -> None:
    sink_path = tmp_path / "jsonl_sink.jsonl"

    # Validation errors
    with pytest.raises(ConfigurationError, match="max_bytes must be an integer >= 256"):
        JsonlTelemetrySink(sink_path, max_bytes=50)
    with pytest.raises(ConfigurationError, match="backup_count must be an integer >= 0"):
        JsonlTelemetrySink(sink_path, backup_count=-1)
    with pytest.raises(ConfigurationError, match="fsync must be a bool"):
        JsonlTelemetrySink(sink_path, fsync="true")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="filter_fn must be callable or None"):
        JsonlTelemetrySink(sink_path, filter_fn="filter")  # type: ignore[arg-type]

    # filter_fn skipping record
    sink = JsonlTelemetrySink(
        sink_path,
        filter_fn=lambda r: r.passed is True,
        fsync=True,
    )
    fail_rec = _make_telemetry_record(passed=False)
    pass_rec = _make_telemetry_record(passed=True)

    sink.emit(fail_rec)
    assert sink.records_written == 0
    sink.emit(pass_rec)
    assert sink.records_written == 1

    sink.flush()
    sink.close()
    # emit after close does nothing
    sink.emit(pass_rec)
    assert sink.records_written == 1


def test_memory_and_callback_telemetry_sinks() -> None:
    # MemoryTelemetrySink validation
    with pytest.raises(ConfigurationError, match="max_records must be an integer >= 1"):
        MemoryTelemetrySink(max_records=0)
    with pytest.raises(ConfigurationError, match="filter_fn must be callable or None"):
        MemoryTelemetrySink(filter_fn="fn")  # type: ignore[arg-type]

    mem_sink = MemoryTelemetrySink(filter_fn=lambda r: r.passed is True)
    mem_sink.emit(_make_telemetry_record(passed=False))
    assert len(mem_sink.records) == 0
    mem_sink.emit(_make_telemetry_record(passed=True))
    assert len(mem_sink.records) == 1
    assert mem_sink.summary().total_records == 1
    mem_sink.clear()
    assert len(mem_sink.records) == 0

    # CallbackTelemetrySink validation
    with pytest.raises(ConfigurationError, match="CallbackTelemetrySink callback must be callable"):
        CallbackTelemetrySink("not_callable")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="filter_fn must be callable or None"):
        CallbackTelemetrySink(lambda r: None, filter_fn="fn")  # type: ignore[arg-type]

    # Callback execution with filter_fn
    called: list[TelemetryRecord] = []
    cb_sink = CallbackTelemetrySink(
        callback=called.append,
        filter_fn=lambda r: r.passed is True,
    )
    cb_sink.emit(_make_telemetry_record(passed=False))
    assert len(called) == 0
    cb_sink.emit(_make_telemetry_record(passed=True))
    assert len(called) == 1

    # Callback error handling
    def _exploding_callback(rec: TelemetryRecord) -> None:
        raise RuntimeError("Callback explosion")

    cb_warn = CallbackTelemetrySink(_exploding_callback, on_error="warn")
    with pytest.warns(TelemetryWarning, match="CallbackTelemetrySink failed to emit"):
        cb_warn.emit(_make_telemetry_record())

    cb_raise = CallbackTelemetrySink(_exploding_callback, on_error="raise")
    with pytest.raises(RuntimeError, match="Callback explosion"):
        cb_raise.emit(_make_telemetry_record())


def test_composite_telemetry_sink_isolation_and_async() -> None:
    # Validation
    with pytest.raises(ConfigurationError, match="CompositeTelemetrySink requires a non-empty sequence"):
        CompositeTelemetrySink([])
    with pytest.raises(ConfigurationError, match="CompositeTelemetrySink requires a non-empty sequence"):
        CompositeTelemetrySink("not a list")
    with pytest.raises(ConfigurationError, match="telemetry_sink must be a TelemetrySink"):
        CompositeTelemetrySink([123])  # type: ignore[list-item]

    class ExplodingSink(TelemetrySink):
        def emit(self, record: TelemetryRecord) -> None:
            raise ValueError("Sink exploded")

        async def aemit(self, record: TelemetryRecord) -> None:
            raise ValueError("Async sink exploded")

    sink1 = ExplodingSink()
    sink2 = MemoryTelemetrySink()
    comp = CompositeTelemetrySink([sink1, sink2])

    rec = _make_telemetry_record()

    # Sync emit: sink2 must still receive the record before first error is re-raised
    with pytest.raises(ValueError, match="Sink exploded"):
        comp.emit(rec)
    assert len(sink2.records) == 1

    # Async aemit: sink2 must still receive the record before first error is re-raised
    async def _test_async_comp() -> None:
        with pytest.raises(ValueError, match="Async sink exploded"):
            await comp.aemit(rec)
        assert len(sink2.records) == 2

    asyncio.run(_test_async_comp())

    # flush, close, aclose
    comp.flush()
    comp.close()
    asyncio.run(comp.aclose())


def test_resolve_telemetry_sink_branches() -> None:
    assert resolve_telemetry_sink(None) is None
    assert resolve_telemetry_sink(False) is False
    with pytest.raises(ConfigurationError, match="telemetry_sink cannot be True"):
        resolve_telemetry_sink(True)
    with pytest.raises(ConfigurationError, match="telemetry_sink must be a TelemetrySink"):
        resolve_telemetry_sink(12345)  # type: ignore[arg-type]


def test_trace_extraction_and_discrepancy_helpers() -> None:
    # Trace with ranking or choice_probabilities is skipped for student trace
    tr_ranking = LayerTrace(
        layer_name="ranker",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        log_odds_pass=1.5,
        metadata={"ranking": ["opt1", "opt2"]},
    )
    tr_teacher = LayerTrace(
        layer_name="teacher",
        status=DecisionStatus.PASSED,
        latency_ms=10.0,
        tokens_generated=50,
    )
    st, te = _find_student_and_teacher_traces([tr_ranking, tr_teacher])
    assert st is None
    assert te is None

    # Discrepancy checks: judge_log_odds in teacher metadata
    rec = _make_telemetry_record(
        status=DecisionStatus.PASSED,
        student_lo=2.0,
        teacher_passed=None,
    )
    rec.traces[1].metadata["judge_log_odds"] = -3.0
    assert _has_student_teacher_discrepancy(rec) is True


def test_load_telemetry_records_corrupt(tmp_path: Path) -> None:
    p = tmp_path / "corrupt_tel.jsonl"
    p.write_text("NOT A DICT JSON\n12345\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_telemetry_records(p)


def test_emit_verdict_telemetry_temp_sink(tmp_path: Path) -> None:
    temp_jsonl = str(tmp_path / "temp_emit.jsonl")
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="ok",
    )
    emit_verdict_telemetry(
        v,
        candidate="test candidate",
        gate_name="DummyGate",
        gate_type="DummyGate",
        backend="custom://test_spec",
        call_kwargs={
            "telemetry_sink": temp_jsonl,
            "telemetry_tags": {"env": "unit-test"},
        },
    )
    recs = load_telemetry_records(temp_jsonl)
    assert len(recs) == 1
    assert recs[0].tags.get("env") == "unit-test"
    assert recs[0].backend == "custom://test_spec"


# ==============================================================================
# SECTION 5: DRIFT & OTEL TESTS (drift.py & otel.py)
# ==============================================================================


def test_compute_psi_edge_cases() -> None:
    # num_bins validation
    with pytest.raises(ConfigurationError, match="num_bins must be an integer >= 2"):
        compute_psi([1.0, 2.0], [1.0, 2.0], num_bins=True)
    with pytest.raises(ConfigurationError, match="num_bins must be an integer >= 2"):
        compute_psi([1.0, 2.0], [1.0, 2.0], num_bins=1)

    # eps validation
    with pytest.raises(ConfigurationError, match="eps must be a positive finite float"):
        compute_psi([1.0, 2.0], [1.0, 2.0], eps=True)
    with pytest.raises(ConfigurationError, match="eps must be a positive finite float"):
        compute_psi([1.0, 2.0], [1.0, 2.0], eps=-0.1)

    # Reference with 0 variance (all values identical)
    psi_zero_var = compute_psi([5.0] * 20, [5.0] * 20)
    assert psi_zero_var == 0.0

    # Current differing from constant reference
    psi_shifted = compute_psi([5.0] * 20, [10.0] * 20)
    assert psi_shifted > 0.0


def test_kolmogorov_survival_bounds() -> None:
    # d_stat <= 0 -> 1.0
    assert _kolmogorov_survival(0.0, 10, 10) == 1.0
    assert _kolmogorov_survival(-0.5, 10, 10) == 1.0

    # lam <= 1e-6 -> 1.0
    assert _kolmogorov_survival(1e-9, 10, 10) == 1.0

    # lam >= 6.0 -> 0.0
    assert _kolmogorov_survival(1.0, 1000, 1000) == 0.0


def test_extract_signals_validation_and_fallbacks() -> None:
    # Empty source
    with pytest.raises(ConfigurationError, match="reference cannot be empty"):
        _extract_signals([], "reference")

    # Mixed types in sequence
    tel = _make_telemetry_record()
    with pytest.raises(ConfigurationError, match="reference cannot mix TelemetryRecord"):
        _extract_signals([tel, 42.0], "reference")  # type: ignore[arg-type]

    # Non-path non-sequence type
    with pytest.raises(ConfigurationError, match="reference must be a path, sequence of TelemetryRecord"):
        _extract_signals(12345, "reference")  # type: ignore[arg-type]

    # Float series extraction
    sig_floats = _extract_signals([0.5, 1.2, -0.8], "floats")
    assert sig_floats.count == 3
    assert len(sig_floats.probs) == 3

    # Records with probability fallback and no traces
    tel_no_traces = _make_telemetry_record()
    tel_no_traces.traces.clear()
    sig_fb = _extract_signals([tel_no_traces], "no_traces")
    assert len(sig_fb.log_odds) == 1
    assert len(sig_fb.probs) == 1


def test_detect_telemetry_drift_validation_and_escalation() -> None:
    # Validation errors
    with pytest.raises(ConfigurationError, match="psi_moderate must be a positive finite float"):
        detect_telemetry_drift([1.0, 2.0], [1.0, 2.0], psi_moderate=-0.1)

    with pytest.raises(ConfigurationError, match="psi_severe .* must be >= psi_moderate"):
        detect_telemetry_drift([1.0, 2.0], [1.0, 2.0], psi_moderate=0.3, psi_severe=0.1)

    with pytest.raises(ConfigurationError, match="ks_alpha must be in \\(0.0, 1.0\\)"):
        detect_telemetry_drift([1.0, 2.0], [1.0, 2.0], ks_alpha=1.5)

    with pytest.raises(ConfigurationError, match="min_teacher_samples must be an integer >= 1"):
        detect_telemetry_drift([1.0, 2.0], [1.0, 2.0], min_teacher_samples=0)

    # Moderate severity single reason -> action='monitor'
    # Reference centered at 0.0, current slightly shifted to 0.5
    ref_vals = [float(i) for i in range(-15, 16)]
    cur_vals = [float(i) + 2.0 for i in range(-15, 16)]
    rep = detect_telemetry_drift(
        ref_vals,
        cur_vals,
        psi_moderate=0.01,
        psi_severe=2.0,
        ks_alpha=0.0001,
        pos_bias_delta_threshold=10.0,
        escalation_delta_threshold=10.0,
    )
    assert rep.severity in ("moderate", "severe")


def test_rolling_drift_monitor_validation_and_lifecycle() -> None:
    # Parameter validations
    with pytest.raises(ConfigurationError, match="baseline_window must be an integer >= 5"):
        RollingDriftMonitor(baseline_window=2)
    with pytest.raises(ConfigurationError, match="current_window must be an integer >= 5"):
        RollingDriftMonitor(current_window=2)
    with pytest.raises(ConfigurationError, match="check_every must be an integer >= 1"):
        RollingDriftMonitor(check_every=0)
    with pytest.raises(ConfigurationError, match="min_current_samples must be an integer >= 5"):
        RollingDriftMonitor(min_current_samples=2)
    with pytest.raises(ConfigurationError, match="boost_cooldown_checks must be an integer >= 1"):
        RollingDriftMonitor(boost_cooldown_checks=0)
    with pytest.raises(ConfigurationError, match="drift_boost_theta must be a finite float >= 0.0"):
        RollingDriftMonitor(drift_boost_theta=-1.0)
    with pytest.raises(ConfigurationError, match="on_drift must be callable or None"):
        RollingDriftMonitor(on_drift="callback")  # type: ignore[arg-type]

    # Reference sequence validation
    with pytest.raises(ConfigurationError, match="reference sequence must contain TelemetryRecord"):
        RollingDriftMonitor(reference=[123])  # type: ignore[list-item]
    with pytest.raises(ConfigurationError, match="Provided reference baseline cannot be empty"):
        RollingDriftMonitor(reference=[])
    with pytest.raises(ConfigurationError, match="reference must be a path"):
        RollingDriftMonitor(reference=12345)  # type: ignore[arg-type]

    # Lifecycle and methods
    monitor = RollingDriftMonitor(baseline_window=5, current_window=5, check_every=2, min_current_samples=5)
    assert not monitor.baseline_ready
    assert monitor.evaluate_now() is None

    # Ingest baseline
    for i in range(5):
        monitor.emit(_make_telemetry_record(candidate=f"base-{i}", student_lo=float(i)))
    assert monitor.baseline_ready

    # Ingest current
    for i in range(5):
        monitor.emit(_make_telemetry_record(candidate=f"cur-{i}", student_lo=float(i)))

    rep = monitor.evaluate_now()
    assert rep is not None
    assert monitor.latest_report is rep
    assert len(monitor.reports) >= 1

    monitor.reset_current()
    assert monitor.evaluate_now() is None  # Current is empty again


def test_otel_sanitization_and_sink_validation() -> None:
    # Sanitization edge cases
    assert _sanitize_otel_value([]) == []
    assert _sanitize_otel_value([True, False]) == [True, False]
    assert _sanitize_otel_value([1, 2, 3]) == [1, 2, 3]
    assert _sanitize_otel_value([1.5, 2.5]) == [1.5, 2.5]
    assert _sanitize_otel_value(["a", "b"]) == ["a", "b"]
    assert _sanitize_otel_value(float("nan")) is None

    # Backend gen_ai system inference
    assert _infer_gen_ai_system(None) is None
    assert _infer_gen_ai_system("ollama://llama3") == "ollama"
    assert _infer_gen_ai_system("openai_compat") == "openai_compat"

    # OtelSpanSink validation
    with pytest.raises(ConfigurationError, match="instrumentation_name must be a non-empty string"):
        OtelSpanSink(instrumentation_name="   ")
    with pytest.raises(ConfigurationError, match="instrumentation_version must be a string or None"):
        OtelSpanSink(instrumentation_version=123)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="span_prefix must be a non-empty string"):
        OtelSpanSink(span_prefix="   ")
    with pytest.raises(ConfigurationError, match="record_layer_events must be a bool"):
        OtelSpanSink(record_layer_events="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="error_on_fail must be a bool"):
        OtelSpanSink(error_on_fail="no")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="require_otel must be a bool"):
        OtelSpanSink(require_otel=1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="filter_fn must be callable or None"):
        OtelSpanSink(filter_fn="fn")  # type: ignore[arg-type]

    # Duck-typed tracer validation
    with pytest.raises(ConfigurationError, match="OtelSpanSink tracer must implement"):
        OtelSpanSink(tracer=object())


def test_otel_sink_start_span_fallback_and_filtering() -> None:
    class MockSpan:
        def __init__(self) -> None:
            self.events: list[tuple[str, Any]] = []
            self.status: Any = None
            self.ended = False

        def add_event(self, name: str, attributes: Any = None) -> None:
            self.events.append((name, attributes))

        def set_status(self, status: Any, desc: str | None = None) -> None:
            self.status = (status, desc)

        def end(self) -> None:
            self.ended = True

    class MockTracerStartSpanOnly:
        def __init__(self) -> None:
            self.spans: list[MockSpan] = []

        def start_span(self, name: str, attributes: Any = None) -> MockSpan:
            s = MockSpan()
            self.spans.append(s)
            return s

    tracer = MockTracerStartSpanOnly()
    sink = OtelSpanSink(
        tracer=tracer,
        filter_fn=lambda r: r.passed is False,
        error_on_fail=True,
    )

    pass_rec = _make_telemetry_record(passed=True)
    fail_rec = _make_telemetry_record(
        passed=False,
        status=DecisionStatus.FAILED,
        tags={"": "empty_key_skipped", "env": "test"},
    )

    # pass_rec filtered out
    sink.emit(pass_rec)
    assert len(tracer.spans) == 0

    # fail_rec emitted and uses start_span fallback
    sink.emit(fail_rec)
    assert len(tracer.spans) == 1
    assert tracer.spans[0].ended is True
    assert len(tracer.spans[0].events) >= 1

    sink.close()


def test_telemetry_models_luhn_and_privacy_coverage() -> None:
    # Luhn PAN checks
    assert not _luhn_valid("123")  # Too short
    assert not _luhn_valid("12345678901234567890")  # Too long
    assert not _luhn_valid("1111111111111111")  # All identical digits

    # Privacy modes in apply_privacy
    c, sha, length = apply_privacy("test candidate text", privacy="hash_only")
    assert c is None
    assert sha == hash_candidate("test candidate text")
    assert length == 19

    c_omit, _, _ = apply_privacy("test candidate text", privacy="omit_candidate")
    assert c_omit is None

    c_redact, _, _ = apply_privacy("my secret token is Bearer 12345abcdef", privacy="redact_pii")
    assert "[REDACTED_TOKEN]" in c_redact  # type: ignore[operator]

    c_full, _, _ = apply_privacy("test candidate", privacy="full", max_candidate_chars=4)
    assert c_full == "test...[truncated]"

    # TelemetryRecord.with_privacy("full")
    rec = _make_telemetry_record(candidate="long candidate text")
    rec_full = rec.with_privacy(privacy="full", max_candidate_chars=4)
    assert rec_full.candidate == "long...[truncated]"



# ==============================================================================
# SECTION 6: HIGH-LEVERAGE MUTANT-KILLING & EXHAUSTIVE BRANCH TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_streaming_async_full_lifecycle_and_branches() -> None:
    # 1. Custom chunk extractor returning str
    extracted_chunks: list[str] = []

    def custom_ext(c: Any) -> str:
        s = str(c).upper()
        extracted_chunks.append(s)
        return s

    gate_ok = DummyGate()
    guard_ok = StreamGuard(gate_ok, chunk_extractor=custom_ext, min_chars=5, stride_chars=5)

    async def _src_ok() -> AsyncIterator[str]:
        yield "chunk1 "
        yield ""  # empty text chunk
        yield "chunk2."

    out_chunks = [c async for c in guard_ok.astream(_src_ok())]
    assert len(out_chunks) == 3
    assert len(extracted_chunks) == 3

    # 2. Async clean stream with final_check=False
    guard_no_final = StreamGuard(gate_ok, min_chars=5, stride_chars=5, final_check=False)
    s_no_final = guard_no_final.astream(_src_ok())
    async for _ in s_no_final:
        pass
    assert s_no_final.summary is not None
    assert s_no_final.summary.passed

    # 3. Async final check failure with raise_on_abort=True
    fail_verdict = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.1,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="final failure",
    )
    fail_gate = DummyGate(verdicts=[Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="mid ok",
    ), fail_verdict])

    fail_gate = DummyGate(default_passed=False, verdicts=[fail_verdict])
    guard_final_fail = StreamGuard(
        fail_gate,
        min_chars=500,
        final_check=True,
        raise_on_abort=True,
    )
    with pytest.raises(StreamAbortedError) as exc_info:
        async for _ in guard_final_fail.astream(_src_ok()):
            pass
    assert exc_info.value.is_final is True

    # 4. Async final check failure with raise_on_abort=False
    fail_gate_quiet = DummyGate(default_passed=False, verdicts=[fail_verdict])
    guard_quiet = StreamGuard(
        fail_gate_quiet,
        min_chars=500,
        final_check=True,
        raise_on_abort=False,
    )
    s_quiet = guard_quiet.astream(_src_ok())
    async for _ in s_quiet:
        pass
    assert s_quiet.summary is not None
    assert s_quiet.summary.aborted is True

    # 5. Async delivery="buffered" clean completion flushes pending chunks
    gate_buf = DummyGate()
    guard_buf = StreamGuard(gate_buf, delivery="buffered", min_chars=10, stride_chars=10)
    buf_out = [c async for c in guard_buf.astream(_src_ok())]
    assert len(buf_out) == 3

    # 6. Async midstream abort with raise_on_abort=False
    fail_mid = DummyGate(verdicts=[fail_verdict])
    guard_mid_quiet = StreamGuard(
        fail_mid,
        min_chars=5,
        stride_chars=5,
        clause_boundary=False,
        raise_on_abort=False,
    )
    s_mid = guard_mid_quiet.astream(_src_ok())
    async for _ in s_mid:
        pass
    assert s_mid.summary is not None
    assert s_mid.summary.aborted_mid_stream is True

    # 7. Helper functions guard_stream and aguard_stream
    g_sync = guard_stream(["a", "b."], gate_ok, min_chars=5)
    assert list(g_sync) == ["a", "b."]

    async def _simple_src() -> AsyncIterator[str]:
        yield "async1"
        yield "async2."

    g_async = aguard_stream(_simple_src(), gate_ok, min_chars=5)
    res_async = [item async for item in g_async]
    assert res_async == ["async1", "async2."]

    # 8. Sync final check failure with raise_on_abort=True and False
    gate_sync_final = DummyGate(default_passed=False, verdicts=[fail_verdict])
    guard_sync_final = StreamGuard(gate_sync_final, min_chars=500, final_check=True, raise_on_abort=True)
    with pytest.raises(StreamAbortedError) as sync_exc:
        list(guard_sync_final.stream(["item1 ", "item2."]))
    assert sync_exc.value.is_final is True

    gate_sync_quiet = DummyGate(default_passed=False, verdicts=[fail_verdict])
    guard_sync_quiet = StreamGuard(gate_sync_quiet, min_chars=500, final_check=True, raise_on_abort=False)
    s_sq = guard_sync_quiet.stream(["item1 ", "item2."])
    list(s_sq)
    assert s_sq.summary is not None
    assert s_sq.summary.aborted is True

    # 9. Sync midstream abort with raise_on_abort=False
    gate_sync_mid = DummyGate(verdicts=[fail_verdict])
    guard_sync_mid = StreamGuard(gate_sync_mid, min_chars=5, stride_chars=5, clause_boundary=False, raise_on_abort=False)
    s_sm = guard_sync_mid.stream(["item1 ", "item2."])
    list(s_sm)
    assert s_sm.summary is not None
    assert s_sm.summary.aborted_mid_stream is True

    # 10. Sync buffered clean completion flushes pending chunks
    guard_sync_buf = StreamGuard(gate_ok, delivery="buffered", min_chars=10, stride_chars=10)
    assert list(guard_sync_buf.stream(["buf1 ", "buf2."])) == ["buf1 ", "buf2."]


def test_distillation_rotation_and_export_branches(tmp_path: Path) -> None:
    sink_path = tmp_path / "rotating_distill.jsonl"
    sink = DistillationSink(sink_path, max_bytes=350, backup_count=2, fsync=True)

    # Emit multiple large records to trigger rotation
    r1 = _make_telemetry_record(candidate="A" * 150)
    r2 = _make_telemetry_record(candidate="B" * 150)
    r3 = _make_telemetry_record(candidate="C" * 150)

    sink.emit(r1)
    sink.emit(r2)  # Should trigger rotation
    sink.emit(r3)
    sink.flush()
    sink.close()

    assert sink.records_written == 3
    # Check that backup files exist
    assert sink_path.exists()

    # Load distillation records from file
    loaded = load_distillation_records(sink_path)
    assert len(loaded) >= 1

    # Load from raw TelemetryRecord sequence
    from logitgate.telemetry.distill import _normalize_distillation_source
    norm_from_tel = _normalize_distillation_source([r1, r2])
    assert len(norm_from_tel) == 2

    # Recalibrate from telemetry with dedupe hit and non-finite student log-odds
    r_pos1 = _make_telemetry_record(candidate="cand-p1", student_lo=2.0, teacher_passed=True)
    r_pos2 = _make_telemetry_record(candidate="cand-p2", student_lo=1.5, teacher_passed=True)
    r_neg1 = _make_telemetry_record(candidate="cand-n1", student_lo=-2.0, teacher_passed=False)
    r_neg2 = _make_telemetry_record(candidate="cand-n2", student_lo=-1.5, teacher_passed=False)
    r_dup = _make_telemetry_record(candidate="cand-p1", student_lo=2.0, teacher_passed=True)
    r_nan = _make_telemetry_record(candidate="cand-nan", student_lo=float("nan"), teacher_passed=True)

    prof = recalibrate_from_telemetry([r_pos1, r_pos2, r_neg1, r_neg2, r_dup, r_nan], target_risk_alpha=None, dedupe=True)
    assert prof is not None

    # Export formats
    export_sft = tmp_path / "exp_sft.jsonl"
    rows_sft = export_distillation_dataset(
        [r_pos1, r_pos2, r_neg1, r_neg2, r_dup],
        export_sft,
        format="sft_jsonl",
        include_symmetric_swaps=False,
        dedupe=True,
    )
    assert rows_sft == 4

    export_dpo = tmp_path / "exp_dpo.jsonl"
    rows_dpo = export_distillation_dataset(
        [r_pos1, r_pos2, r_neg1, r_neg2],
        export_dpo,
        format="dpo_jsonl",
        include_symmetric_swaps=True,
        dedupe=True,
    )
    assert rows_dpo == 8

    export_cal = tmp_path / "exp_cal.jsonl"
    rows_cal = export_distillation_dataset(
        [r_pos1, r_pos2, r_neg1, r_neg2],
        export_cal,
        format="calibration_jsonl",
        dedupe=True,
    )
    assert rows_cal == 4


def test_cache_sqlite_and_fingerprint_advanced(tmp_path: Path) -> None:
    # 1. Gate with CalibrationProfile
    samples = [(2.0, 2.0, 1), (-2.0, -2.0, 0), (1.5, 1.5, 1), (-1.5, -1.5, 0)]
    prof = CalibrationProfile.from_log_odds_samples(samples, target_risk_alpha=None)
    gate_with_prof = LogitGate(rubric="is accurate", profile=prof, backend=StubBackend(default_prob=0.95))
    fp = gate_cache_fingerprint(gate_with_prof)
    assert "profile" in fp

    # 2. SQLite cache operations: invalidate, clear, stats
    db_path = tmp_path / "ops_cache.db"
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="LogitGate",
        total_latency_ms=2.0,
        total_tokens_generated=0,
        reason="ok",
    )
    with VerdictCache(db_path, ttl_seconds=100.0) as cache:
        cache.put(gate_with_prof, "cand-op", v, has_normalized=True, normalized_artifact="norm_val")
        stats1 = cache.stats()
        assert stats1.puts == 1

        # Invalidate key
        existed = cache.invalidate(gate_with_prof, "cand-op")
        assert existed is True
        assert cache.get(gate_with_prof, "cand-op") is None

        # Repopulate and clear
        cache.put(gate_with_prof, "cand-op2", v)
        cache.clear()
        assert cache.stats().memory_entries == 0
        assert cache.stats().disk_entries == 0

    # 3. Non-serializable normalized_artifact with has_normalized=True
    with VerdictCache(db_path) as cache2:
        cache2.put(
            gate_with_prof,
            "cand-unserializable",
            v,
            has_normalized=True,
            normalized_artifact=object(),  # TypeError inside json.dumps handled gracefully
        )
        hit = cache2.get(gate_with_prof, "cand-unserializable")
        assert hit is not None

    # 4. _resolve_active_cache with call_kwargs["cache"] = False
    from logitgate.cache import _resolve_active_cache
    active, owns = _resolve_active_cache(None, {"cache": False})
    assert active is None
    assert owns is False

    active_gate_f, owns_gate_f = _resolve_active_cache(False, None)
    assert active_gate_f is None
    assert owns_gate_f is False


def test_sinks_rotation_and_security_branches(tmp_path: Path) -> None:
    # 1. Directory path -> ConfigurationError
    dir_target = tmp_path / "dir_target"
    dir_target.mkdir()
    with pytest.raises(ConfigurationError, match="path cannot be a directory"):
        _validate_target_file_path(dir_target, "TestSink")

    # 2. File size > max_bytes in read_regular_text_nofollow
    oversized = tmp_path / "oversized.txt"
    oversized.write_text("A" * 100, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="exceeds maximum size"):
        read_regular_text_nofollow(oversized, max_bytes=10, label="OversizedFile")

    # 3. JsonlTelemetrySink rotation
    sink_path = tmp_path / "rotating_sink.jsonl"
    sink = JsonlTelemetrySink(sink_path, max_bytes=350, backup_count=2, fsync=True)

    r1 = _make_telemetry_record(candidate="A" * 150)
    r2 = _make_telemetry_record(candidate="B" * 150)
    r3 = _make_telemetry_record(candidate="C" * 150)

    sink.emit(r1)
    sink.emit(r2)  # rotation triggered
    sink.emit(r3)
    sink.flush()
    sink.close()
    assert sink.records_written == 3

    # 4. CompositeTelemetrySink with multiple failing child sinks
    class FailingSinkA(TelemetrySink):
        def emit(self, record: TelemetryRecord) -> None:
            raise RuntimeError("Error A")

        async def aemit(self, record: TelemetryRecord) -> None:
            raise RuntimeError("Async Error A")

    class FailingSinkB(TelemetrySink):
        def emit(self, record: TelemetryRecord) -> None:
            raise ValueError("Error B")

        async def aemit(self, record: TelemetryRecord) -> None:
            raise ValueError("Async Error B")

    comp_fails = CompositeTelemetrySink([FailingSinkA(), FailingSinkB()])
    with pytest.raises(RuntimeError, match="Error A"):
        comp_fails.emit(r1)

    async def _test_async_comp_fails() -> None:
        with pytest.raises(RuntimeError, match="Async Error A"):
            await comp_fails.aemit(r1)

    asyncio.run(_test_async_comp_fails())

    # 5. _has_student_teacher_discrepancy without traces
    rec_empty = _make_telemetry_record()
    rec_empty.traces.clear()
    assert _has_student_teacher_discrepancy(rec_empty) is False

    # 6. summarize_telemetry([])
    empty_summary = summarize_telemetry([])
    assert empty_summary.total_records == 0

    # 7. emit_verdict_telemetry with call_kwargs={"telemetry_sink": False}
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="ok",
    )
    res_v = emit_verdict_telemetry(
        v,
        candidate="c",
        gate_name="g",
        gate_type="gt",
        call_kwargs={"telemetry_sink": False},
    )
    assert res_v is v


def test_drift_and_otel_additional_branches(tmp_path: Path) -> None:
    # 1. target_tr has calibrated_prob but log_odds_pass is None
    rec_prob_only = _make_telemetry_record()
    rec_prob_only.traces[0].calibrated_prob = 0.85
    rec_prob_only.traces[0].log_odds_pass = None
    sig1 = _extract_signals([rec_prob_only], "prob_only")
    assert len(sig1.log_odds) == 1
    assert sig1.log_odds[0] > 0.0

    # 2. target_tr has log_odds_pass but calibrated_prob is None
    rec_lo_only = _make_telemetry_record()
    rec_lo_only.traces[0].calibrated_prob = None
    rec_lo_only.traces[0].log_odds_pass = 1.8
    sig2 = _extract_signals([rec_lo_only], "lo_only")
    assert len(sig2.probs) == 1
    assert sig2.probs[0] > 0.5

    # 3. RollingDriftMonitor reference from JSONL file
    ref_file = tmp_path / "ref_records.jsonl"
    sink = JsonlTelemetrySink(ref_file)
    for i in range(5):
        sink.emit(_make_telemetry_record(candidate=f"ref-{i}", student_lo=float(i)))
    sink.close()

    monitor = RollingDriftMonitor(reference=str(ref_file), current_window=5, min_current_samples=5)
    assert monitor.baseline_ready is True

    # 4. OtelSpanSink default instrumentation_version
    otel_sink = OtelSpanSink(require_otel=False)
    assert otel_sink.instrumentation_version is not None
    # Emit after close does nothing
    otel_sink.close()
    otel_sink.emit(_make_telemetry_record())



# ==============================================================================
# SECTION 7: SURGICAL BRANCH TESTS FOR 95%+ PER-MODULE COVERAGE
# ==============================================================================


def test_distillation_surgical_branches(tmp_path: Path) -> None:
    # 1. teacher_passed=None and no judge_log_odds in teacher metadata -> line 82
    rec = _make_telemetry_record(teacher_passed=None)
    # Ensure no judge_log_odds in teacher trace metadata
    rec.traces[1].metadata.pop("judge_log_odds", None)
    rec.traces[1].status = DecisionStatus.PASSED
    d = extract_distillation_record(rec)
    assert d is not None
    assert d.teacher_passed is True

    rec_fail = _make_telemetry_record(teacher_passed=None)
    rec_fail.traces[1].metadata.pop("judge_log_odds", None)
    rec_fail.traces[1].status = DecisionStatus.FAILED
    d_fail = extract_distillation_record(rec_fail)
    assert d_fail is not None
    assert d_fail.teacher_passed is False

    # 2. DistillationSink.emit with ineligible record -> line 256
    sink_path = tmp_path / "ineligible.jsonl"
    sink = DistillationSink(sink_path)
    ineligible_rec = _make_telemetry_record(rubric="")
    sink.emit(ineligible_rec)
    assert sink.records_written == 0
    sink.close()

    # 3. load_distillation_records with empty lines and raw TelemetryRecord line -> lines 316, 326
    raw_tel_path = tmp_path / "raw_tel.jsonl"
    rec_json = rec.model_dump_json()
    raw_tel_path.write_text(f"\n   \n{rec_json}\n\n", encoding="utf-8")
    loaded = load_distillation_records(raw_tel_path)
    assert len(loaded) == 1
    assert loaded[0].candidate == rec.candidate

    # 4. _normalize_distillation_source passing [TelemetryRecord]
    from logitgate.telemetry.distill import _normalize_distillation_source
    norm_res = _normalize_distillation_source([rec])
    assert len(norm_res) == 1

    # 5. export_distillation_dataset with dedupe hit -> line 478
    export_out = tmp_path / "dedupe_exp.jsonl"
    d_pos = extract_distillation_record(rec)
    assert d_pos is not None
    # Pass duplicate records with dedupe=True
    rows = export_distillation_dataset([d_pos, d_pos], export_out, dedupe=True)
    assert rows == 2  # 1 record x 2 symmetric directions


def test_sinks_surgical_branches(tmp_path: Path) -> None:
    # 1. CompositeTelemetrySink([False]) -> line 424
    with pytest.raises(ConfigurationError, match="must resolve to a TelemetrySink"):
        CompositeTelemetrySink([False])  # type: ignore[list-item]

    # 2. _has_student_teacher_discrepancy with teacher_passed=None and no judge_log_odds -> line 531
    rec = _make_telemetry_record(status=DecisionStatus.PASSED, teacher_passed=None)
    rec.traces[1].metadata.pop("judge_log_odds", None)
    rec.traces[1].status = DecisionStatus.FAILED
    assert _has_student_teacher_discrepancy(rec) is True

    # 3. load_telemetry_records with empty lines and non-dict line -> lines 556, 560
    tel_path = tmp_path / "blank_lines.jsonl"
    valid_json = rec.model_dump_json()
    tel_path.write_text(f"\n   \n{valid_json}\n\n", encoding="utf-8")
    loaded = load_telemetry_records(tel_path)
    assert len(loaded) == 1

    bad_type_path = tmp_path / "non_dict.jsonl"
    bad_type_path.write_text('"a string not an object"\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="JSONL line must decode to an object"):
        load_telemetry_records(bad_type_path)

    # 4. emit_verdict_telemetry with gate_sink=False -> line 673
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="ok",
    )
    res = emit_verdict_telemetry(v, candidate="c", gate_name="g", gate_type="gt", gate_sink=False)
    assert res is v


def test_cache_surgical_branches(tmp_path: Path) -> None:
    db_path = tmp_path / "expiry_cache.db"
    gate = DummyGate()
    v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="DummyGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="ok",
    )

    # 1. SQLite expired entry deleted on read -> lines 559-560
    with VerdictCache(db_path, ttl_seconds=0.005) as cache:
        cache.put(gate, "cand-exp", v)
        # Clear memory cache so get() must query SQLite
        cache._mem.clear()
        import time
        time.sleep(0.02)
        # Entry in SQLite is now expired
        hit = cache.get(gate, "cand-exp")
        assert hit is None
        # Verify it was deleted from SQLite
        assert cache.stats().disk_entries == 0

    # 2. cache.invalidate on key in SQLite after memory cleared -> line 738
    with VerdictCache(db_path, ttl_seconds=100.0) as cache:
        cache.put(gate, "cand-inval", v)
        cache._mem.clear()
        assert cache.stats().disk_entries == 1
        existed = cache.invalidate(gate, "cand-inval")
        assert existed is True
        assert cache.stats().disk_entries == 0

    # 3. cache.stats() handling sqlite3.Error -> line 760
    with VerdictCache(db_path) as cache:
        # Drop the table to trigger sqlite3.Error during stats query
        if cache._conn is not None:
            cache._conn.execute("DROP TABLE verdict_cache")
        stats = cache.stats()
        assert stats.disk_entries == 0


@pytest.mark.asyncio
async def test_streaming_surgical_branches() -> None:
    # 1. Chunk extraction edge branches (lines 109, 114, 130)
    assert _extract_chunk_text({"message": {"content": None}}) == ""
    assert _extract_chunk_text({"delta": {"content": None}}) == ""
    choice_tok = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="tok"))])
    assert _extract_chunk_text(choice_tok) == "tok"

    # 2. Async buffered delivery flush pending chunks when no midstream check ran (lines 922-928)
    gate = DummyGate()
    guard_async_buf = StreamGuard(gate, delivery="buffered", min_chars=1000)

    async def _short_src() -> AsyncIterator[str]:
        yield "chunk-a "
        yield "chunk-b."

    collected_async = [c async for c in guard_async_buf.astream(_short_src())]
    assert collected_async == ["chunk-a ", "chunk-b."]

    # 3. Sync buffered delivery flush pending chunks when no midstream check ran (lines 667-674)
    guard_sync_buf = StreamGuard(gate, delivery="buffered", min_chars=1000)
    assert list(guard_sync_buf.stream(["sync-a ", "sync-b."])) == ["sync-a ", "sync-b."]



def test_mocked_sink_and_export_io_errors(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    rec = _make_telemetry_record()

    # 1. DistillationSink handle I/O exceptions
    d_path = tmp_path / "d_err.jsonl"
    d_sink = DistillationSink(d_path, on_error="warn", fsync=True)
    # Open handle first
    d_sink.emit(rec)
    mock_handle = MagicMock()
    mock_handle.write.side_effect = OSError("Mock write error")
    mock_handle.flush.side_effect = OSError("Mock flush error")
    mock_handle.close.side_effect = OSError("Mock close error")
    mock_handle.fileno.return_value = 1
    d_sink._handle = mock_handle

    # emit warning with fresh non-deduped record
    rec2 = _make_telemetry_record(candidate="fresh candidate for d_sink error")
    with pytest.warns(TelemetryWarning, match="DistillationSink failed to emit"):
        d_sink.emit(rec2)

    # flush warning
    with pytest.warns(TelemetryWarning, match="DistillationSink failed to emit"):
        d_sink.flush()

    # close suppresses OSError
    d_sink.close()

    # 2. JsonlTelemetrySink handle I/O exceptions
    j_path = tmp_path / "j_err.jsonl"
    j_sink = JsonlTelemetrySink(j_path, on_error="warn", fsync=True)
    j_sink.emit(rec)
    j_mock = MagicMock()
    j_mock.write.side_effect = OSError("Mock jsonl write error")
    j_mock.flush.side_effect = OSError("Mock jsonl flush error")
    j_mock.close.side_effect = OSError("Mock jsonl close error")
    j_mock.fileno.return_value = 1
    j_sink._handle = j_mock

    rec3 = _make_telemetry_record(candidate="fresh candidate for j_sink error")
    with pytest.warns(TelemetryWarning, match="JsonlTelemetrySink failed to emit"):
        j_sink.emit(rec3)

    with pytest.warns(TelemetryWarning, match="JsonlTelemetrySink failed to emit"):
        j_sink.flush()

    j_sink.close()

    # 3. export_distillation_dataset with symlinked parent directory -> line 572
    real_dir = tmp_path / "exp_real_dir"
    real_dir.mkdir()
    sym_dir = tmp_path / "exp_sym_dir"
    sym_dir.symlink_to(real_dir, target_is_directory=True)
    d_rec = extract_distillation_record(rec)
    assert d_rec is not None
    with pytest.raises(ConfigurationError, match="Refusing to write export dataset through symlink"):
        export_distillation_dataset([d_rec], sym_dir / "out.jsonl")

    # 4. export_distillation_dataset exception during write -> cleanup tempfile (lines 590-600)
    out_dest = tmp_path / "fail_replace.jsonl"
    with (
        patch("os.replace", side_effect=OSError("Mock replace error")),
        pytest.raises(OSError, match="Mock replace error"),
    ):
        export_distillation_dataset([d_rec], out_dest)
