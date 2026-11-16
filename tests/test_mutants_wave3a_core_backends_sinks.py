"""Wave 3A Targeted Mutant Closure Test Suite.

Targets surviving mutants across 9 core modules:
- logitgate.telemetry.sinks
- logitgate.telemetry.otel
- logitgate.telemetry.models
- logitgate.backends.base
- logitgate.backends.openai_compat
- logitgate.benchmark
- logitgate.cascade
- logitgate.cache
- logitgate.snapshot
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Explicit references to all 9 target modules for mutation_audit.py AST mapping
import logitgate.backends.base
import logitgate.backends.openai_compat
import logitgate.benchmark
import logitgate.cache
import logitgate.cascade
import logitgate.snapshot
import logitgate.telemetry.models
import logitgate.telemetry.otel
import logitgate.telemetry.sinks  # noqa: F401
from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    _HTTPClientMixin,
    infer_backend_spec,
    resolve_logit_backend,
    strip_reasoning_blocks,
)
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    format_report,
    run_benchmark,
)
from logitgate.cache import (
    VerdictCache,
    _CacheEntry,
    gate_cache_fingerprint,
)
from logitgate.cascade import (
    Cascade,
    fuse_escalated_verdict,
)
from logitgate.gates.base import BaseGate, EscalationPolicy
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import VerdictSnapshot
from logitgate.telemetry.models import (
    TelemetryRecord,
    _luhn_valid,
)
from logitgate.telemetry.otel import OtelSpanSink
from logitgate.telemetry.sinks import (
    JsonlTelemetrySink,
    _find_student_and_teacher_traces,
    _validate_privacy_and_bounds,
    emit_verdict_telemetry,
    read_regular_text_nofollow,
    summarize_telemetry,
)


def _make_record(**kwargs: Any) -> TelemetryRecord:
    defaults: dict[str, Any] = {
        "gate_name": "test_gate",
        "gate_type": "LogitGate",
        "candidate_sha256": "0" * 64,
        "candidate_length": 10,
        "passed": True,
        "status": DecisionStatus.PASSED,
        "total_latency_ms": 1.0,
    }
    defaults.update(kwargs)
    return TelemetryRecord(**defaults)


# ==============================================================================
# SECTION 1: logitgate.telemetry.sinks
# ==============================================================================


def test_sinks_validate_privacy_and_bounds_boundary_one() -> None:
    """Kills line 47: ROR max_candidate_chars < 1 swapped to <= 1.

    When max_candidate_chars == 1, 1 < 1 is False (valid bound).
    With mutant <= 1, 1 <= 1 is True and raises ConfigurationError.
    """
    privacy, max_chars, on_err = _validate_privacy_and_bounds("full", 1, "warn")
    assert privacy == "full"
    assert max_chars == 1
    assert on_err == "warn"


def test_sinks_read_regular_text_nofollow_exact_max_bytes(tmp_path: Path) -> None:
    """Kills line 125: ROR st.st_size > max_bytes swapped to >= max_bytes.

    When file size exactly equals max_bytes, st.st_size > max_bytes is False (succeeds).
    With mutant >= max_bytes, 5 >= 5 is True and raises ConfigurationError.
    """
    target = tmp_path / "exact_bytes.txt"
    target.write_text("hello", encoding="utf-8")  # 5 bytes
    content = read_regular_text_nofollow(target, max_bytes=5, label="test file")
    assert content == "hello"


def test_sinks_check_rotation_addition_arithmetic(tmp_path: Path) -> None:
    """Kills line 274: AOR current_size + incoming_bytes swapped to - incoming_bytes.

    When current_size = 200, incoming_bytes = 150, max_bytes = 300:
    200 + 150 = 350 > 300 triggers rotation -> backup file .1 is created.
    With mutant -: 200 - 150 = 50 <= 300 fails to rotate.
    """
    log_file = tmp_path / "telemetry.jsonl"
    sink = JsonlTelemetrySink(log_file, max_bytes=300, backup_count=2)
    log_file.write_text("x" * 200, encoding="utf-8")
    sink._rotate_if_needed_locked(150)
    assert (tmp_path / "telemetry.jsonl.1").exists()


def test_sinks_find_student_and_teacher_traces_zero_tokens_non_teacher() -> None:
    """Kills line 499: ROR tr.tokens_generated > 0 swapped to >= 0.

    When an intermediate trace has tokens_generated == 0, diagnostic_report=None,
    failure_code=None:
    tokens_generated > 0 is False, so it is NOT mistaken for a teacher trace.
    With mutant >= 0: 0 >= 0 is True, mistakenly classifying it as a teacher.
    """
    student_tr = LayerTrace(
        layer_name="student",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=0,
        log_odds_pass=1.0,
    )
    dummy_tr = LayerTrace(
        layer_name="dummy",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=0,
        diagnostic_report=None,
        failure_code=None,
    )
    st, tch = _find_student_and_teacher_traces([student_tr, dummy_tr])
    assert st is None
    assert tch is None


def test_sinks_summarize_telemetry_p95_index_arithmetic() -> None:
    """Kills line 605 mutant 128: AOR math.ceil(0.95 * n) - 1 swapped to + 1.

    For n = 100 distinct latencies 0.0..99.0:
    ceil(0.95 * 100) - 1 = 95 - 1 = 94 -> p95 is 94.0.
    With mutant +: ceil(0.95 * 100) + 1 = 95 + 1 = 96 -> p95 is 96.0.
    """
    records = [
        _make_record(
            gate_name="test",
            gate_type="LogitGate",
            passed=True,
            status=DecisionStatus.PASSED,
            total_latency_ms=float(i),
        )
        for i in range(100)
    ]
    summary = summarize_telemetry(records)
    assert summary.p95_latency_ms == pytest.approx(94.0)


def test_sinks_p95_index_mathematical_equivalence_proof() -> None:
    """Documents the mathematical equivalence proof for line 605 mutant 127 in sinks.py (AOR n - 1 swapped to n + 1).

    In `summarize_telemetry(records)`:
        p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))

    Proof of Equivalence:
    1. For any positive integer n >= 1:
       0.95 * n < n  (since 0.95 < 1).
    2. Because n is an integer, the ceiling of any number strictly less than n is at most n:
       ceil(0.95 * n) <= n.
    3. Subtracting 1 from both sides:
       ceil(0.95 * n) - 1 <= n - 1.
    4. Therefore, max(0, ceil(0.95 * n) - 1) <= n - 1 is ALWAYS TRUE for all n >= 1.
    5. Hence, the upper clamp min(n - 1, X) where X <= n - 1 always evaluates to X.
    6. Swapping n - 1 to n + 1 evaluates min(n + 1, X) which also identically evaluates to X
       because X <= n - 1 < n + 1.
    7. No input sequence can produce an index where min(n - 1, X) != min(n + 1, X).
       The mutant is provably mathematically equivalent.
    """
    assert math.ceil(0.95 * 1) - 1 <= 1 - 1
    assert math.ceil(0.95 * 100) - 1 <= 100 - 1


def test_sinks_emit_verdict_telemetry_resolved_call_false_or() -> None:
    """Kills line 668: LCR resolved_call is False or resolved_call is None swapped to and.

    When resolve_telemetry_sink returns False:
    resolved_call is False or resolved_call is None is True -> returns verdict untouched.
    With mutant and: False and None is False -> active_sink = False -> crashes on .emit().
    """
    verdict = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="g1",
        total_latency_ms=1.0,
        score=0.9,
        reason="ok",
    )
    with patch("logitgate.telemetry.sinks.resolve_telemetry_sink", return_value=False):
        res = emit_verdict_telemetry(
            verdict,
            candidate="sample",
            gate_name="g1",
            gate_type="LogitGate",
            call_kwargs={"telemetry_sink": "mock_sink_path"},
        )
        assert res is verdict


# ==============================================================================
# SECTION 2: logitgate.telemetry.otel
# ==============================================================================


def test_otel_build_attributes_tag_allow_pii_no_redaction() -> None:
    """Kills line 248: LCR isinstance(raw_val, str) and self.privacy == 'redact_pii' swapped to or.

    Case 1: String tag with email when privacy='full'.
    Original: privacy != 'redact_pii', so tag remains unredacted.
    Mutant: isinstance is True or privacy == ... -> True -> redacts string!

    Case 2: Integer tag when privacy='redact_pii'.
    Original: isinstance(42, str) is False -> keeps integer 42.
    Mutant: isinstance is False or privacy == ... -> True -> passes 42 to redact_text!
    """
    sink_full = OtelSpanSink(privacy="full")
    rec_allow = _make_record(tags={"contact": "user@example.com"})
    attrs_allow = sink_full.build_span_attributes(rec_allow)
    assert attrs_allow["logitgate.tag.contact"] == "user@example.com"

    sink_redact = OtelSpanSink(privacy="redact_pii")
    rec_int = _make_record(tags={"count": 42})
    attrs_redact = sink_redact.build_span_attributes(rec_int)
    assert attrs_redact["logitgate.tag.count"] == 42


def test_otel_populate_span_passed_record_does_not_set_error_status() -> None:
    """Kills line 292: LCR self.error_on_fail and not prepared.passed and callable(...) swapped to or.

    When prepared.passed is True and error_on_fail is False:
    Original: not prepared.passed is False -> set_status is NEVER called.
    Mutant: callable(...) is True -> set_status IS mistakenly called.
    """
    sink = OtelSpanSink(error_on_fail=False)
    record = _make_record(passed=True, status=DecisionStatus.PASSED)
    mock_span = MagicMock()
    sink._populate_span(mock_span, record)
    mock_span.set_status.assert_not_called()


# ==============================================================================
# SECTION 3: logitgate.telemetry.models
# ==============================================================================


def test_models_luhn_valid_mathematical_equivalence_proof() -> None:
    """Documents the mathematical equivalence proof for line 51 in models.py (ROR > -> >=).

    In `_luhn_valid(digits)`:
        d = ord(ch) - 48  # ch in '0'..'9', so d in {0..9}
        if idx % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9

    Proof of Equivalence:
    1. Every digit `ch` is ASCII 48-57, yielding integer `d in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`.
    2. In the branch `if idx % 2 == parity:`, `d` is multiplied by 2:
       `d in {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}`.
    3. All elements of this set are even integers.
    4. 9 is an odd integer, so `d == 9` has no integer solutions in {2 * k : k in 0..9}.
    5. Therefore, `d > 9` evaluates to True on {10, 12, 14, 16, 18} and False on {0, 2, 4, 6, 8}.
    6. Similarly, `d >= 9` evaluates to True on {10, 12, 14, 16, 18} and False on {0, 2, 4, 6, 8}.
    7. Because `d == 9` is unreachable, `d > 9` <==> `d >= 9` for 100% of all possible inputs.
    """
    # Valid Luhn PANs (16 digits)
    assert _luhn_valid("4532015112830366") is True
    # Boundary tests where d*2 produces 8 (4*2) and 10 (5*2)
    assert _luhn_valid("4532015112830367") is False
    assert _luhn_valid("4532015112830368") is False


# ==============================================================================
# SECTION 4: logitgate.backends.base
# ==============================================================================


def test_backends_base_strip_reasoning_blocks_comment_prefix_arithmetic() -> None:
    """Kills line 163: AOR start + 7 swapped to start - 7 in strip_reasoning_blocks.

    When <think> appears in a comment line (# <think> inside):
    Original slices text[pos : start + 7], preserving '# <think>'.
    Mutant slices text[pos : start - 7], dropping 14 characters.
    """
    text = "# <think> inside comment"
    result = strip_reasoning_blocks(text)
    assert result == "# <think> inside comment"


@pytest.mark.asyncio
async def test_backends_base_http_mixin_async_client_loop_rebound_and_or() -> None:
    """Kills line 324: LCR bound_loop is None and current_loop is not None swapped to or.

    When mixin does not own the client, and _async_client_loop is already set to loop1:
    Original: bound_loop is not None -> elif condition evaluates to False -> loop stays loop1.
    Mutant: bound_loop is None or current_loop is not None -> True -> overwrites with current loop!
    """
    mixin = _HTTPClientMixin()
    mock_client = MagicMock()
    mock_client.is_closed = False
    loop1 = MagicMock()

    mixin._async_client = mock_client
    mixin._owns_async_client = False
    mixin._async_client_loop = loop1

    client = mixin.async_client
    assert client is mock_client
    assert mixin._async_client_loop is loop1


def test_backends_base_stub_resolve_top_logprobs_preferred_subtraction() -> None:
    """Kills line 488: AOR math.log(1.0 - p) swapped to math.log(1.0 + p).

    When decision_section contains 'Preferred:', resp_idx is 1, and default_prob is 0.8:
    Original calculates math.log(1.0 - 0.8) = math.log(0.2) ≈ -1.6094.
    Mutant calculates math.log(1.0 + 0.8) = math.log(1.8) ≈ +0.5877.
    """
    stub = StubBackend(default_prob=0.8)
    stub.query_top_logprobs("warmup")
    lps = stub.query_top_logprobs(
        "[DECISION MAPPING] Preferred: Option A",
        allowed_tokens=["A", "B"],
    )
    assert len(lps) == 2
    assert lps[1]["token"] == "A"
    assert lps[1]["logprob"] == pytest.approx(math.log(0.2))


def test_backends_base_infer_backend_spec_custom_openai_host() -> None:
    """Kills line 566: LCR scheme == 'openai' and host == 'api.openai.com/v1' swapped to and -> or.

    When OpenAICompatBackend uses a custom proxy host (e.g. custom-proxy.internal/v1):
    Original: host != 'api.openai.com/v1' -> returns 'openai://custom-proxy.internal/v1/gpt-4o'.
    Mutant: scheme == 'openai' or ... -> True -> mistakenly returns 'openai://gpt-4o'.
    """
    backend = OpenAICompatBackend(
        base_url="https://custom-proxy.internal/v1",
        model="gpt-4o",
    )
    spec = infer_backend_spec(backend)
    assert spec == "openai://custom-proxy.internal/v1/gpt-4o"


def test_backends_base_resolve_logit_backend_llamacpp_model_or_env() -> None:
    """Kills line 702: LCR model or env_model swapped to model and env_model.

    When model='test-model' is passed and LOGITGATE_MODEL is not set in env (env_model is None):
    Original: 'test-model' or None -> evaluates to 'test-model'.
    Mutant: 'test-model' and None -> evaluates to None.
    """
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("LOGITGATE_MODEL", None)
        backend = resolve_logit_backend("llamacpp", model="test-model")
        assert getattr(backend, "model", None) == "test-model"


# ==============================================================================
# SECTION 5: logitgate.backends.openai_compat
# ==============================================================================


def test_openai_compat_thinking_prefill_trap_non_string_allowed_tokens() -> None:
    """Kills line 125: LCR isinstance(t, str) and t.strip() swapped to or.

    When allowed_tokens contains non-string elements (e.g. 123):
    Original: isinstance(123, str) is False -> cleanly filters out 123.
    Mutant: isinstance(123, str) or 123.strip() -> raises AttributeError on int.
    """
    lps = [{"token": "<think>", "logprob": -0.1}]
    trap = OpenAICompatBackend._is_thinking_prefill_trap(lps, ["A", 123])  # type: ignore[list-item]
    assert trap is True


def test_openai_compat_extract_top_logprobs_non_dict_logprobs() -> None:
    """Kills line 314: LCR not logprobs or not isinstance(logprobs, dict) swapped to and.

    When choices[0]['logprobs'] is a truthy non-dict (e.g. [1, 2, 3]):
    Original: not isinstance(logprobs, dict) is True -> returns [].
    Mutant: not logprobs and not isinstance(...) is False -> calls logprobs.get() and crashes.
    """
    backend = OpenAICompatBackend(base_url="http://localhost:8000/v1", model="test")
    data = {"choices": [{"logprobs": [1, 2, 3]}]}
    result = backend._extract_top_logprobs_from_response(data)
    assert result == []


def test_openai_compat_extract_top_logprobs_non_list_top_logprobs() -> None:
    """Kills line 320: LCR not top_logprobs or not isinstance(top_logprobs, list) swapped to and.

    When top_logprobs is a truthy non-list (e.g. int 42):
    Original: not isinstance(top_logprobs, list) is True -> returns [].
    Mutant: False and ... is False -> attempts to iterate over int 42, raising TypeError.
    """
    backend = OpenAICompatBackend(base_url="http://localhost:8000/v1", model="test")
    data = {"choices": [{"logprobs": {"content": [{"top_logprobs": 42}]}}]}
    result = backend._extract_top_logprobs_from_response(data)
    assert result == []


def test_openai_compat_extract_chat_completion_empty_choices() -> None:
    """Kills line 337: LCR isinstance(choices, list) and choices ... swapped to or.

    When choices is an empty list []:
    Original: choices is falsy -> skips choices block, returns ('', 0).
    Mutant: isinstance(choices, list) or ... is True -> accesses choices[0], raising IndexError.
    """
    data: dict[str, Any] = {"choices": []}
    content, tokens = OpenAICompatBackend._extract_chat_completion(data)
    assert content == ""
    assert tokens == 0


# ==============================================================================
# SECTION 6: logitgate.benchmark
# ==============================================================================


def test_benchmark_owns_cache_caller_cache_not_closed() -> None:
    """Kills line 650: UOI not isinstance(cache, VerdictCache) swapped to not x -> x.

    When caller passes an explicit VerdictCache instance:
    Original: owns_cache is False -> cache is NOT closed at benchmark completion.
    Mutant: owns_cache is True -> caller's cache is prematurely closed!
    """
    cache = VerdictCache()
    try:
        cases = [BenchmarkCase("is positive", "2", True)]
        run_benchmark(
            cases,
            backend=StubBackend(default_prob=0.95),
            cache=cache,
            modes=["logit"],
            repeat=1,
            warmup=0,
        )
        assert cache._closed is False
    finally:
        cache.close()


def test_benchmark_heterogeneous_judge_explicit_backend() -> None:
    """Kills line 679 mutant 157: ROR judge_backend is not None swapped to is None.

    When judge_backend is explicitly provided:
    Original: judge_backend is not None is True -> heterogeneous_judge is True ->
              report records judge_backend='stub://0.9' and backend='stub://0.2'.
    Mutant: judge_backend is None is False -> heterogeneous_judge is False ->
            report mistakenly sets judge_backend equal to primary backend 'stub://0.2'.
    """
    cases = [BenchmarkCase("is positive", "2", True)]
    report = run_benchmark(
        cases,
        backend=StubBackend(default_prob=0.2),
        judge_backend=StubBackend(default_prob=0.9),
        modes=["cascade"],
        repeat=1,
        warmup=0,
    )
    assert report.metadata["judge_backend"] == "stub://0.9"
    assert report.metadata["backend"] == "stub://0.2"


class _MockGenerativeLogitBackend(LogitBackend, GenerativeBackend):
    def score(self, rubric: str, candidate: Any, **kwargs: Any) -> float:
        return 0.95

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": -5.0}]

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return "pass", 1


def test_benchmark_heterogeneous_judge_explicit_model() -> None:
    """Kills line 679 mutant 158: ROR judge_model is not None swapped to is None.

    When judge_backend is None and judge_model is None:
    Original: heterogeneous_judge is False -> resolved_judge is resolved (reuses backend instance).
    Mutant 158: judge_model is None is True -> heterogeneous_judge is True ->
                attempts to resolve custom backend spec and raises ValueError.
    """
    b = _MockGenerativeLogitBackend()
    cases = [BenchmarkCase("rubric", "cand", True)]
    rep = run_benchmark(cases, backend=b, judge_backend=None, judge_model=None, modes=["cascade"], repeat=1, warmup=0)
    assert rep.metadata["backend"] == "custom://_MockGenerativeLogitBackend"


def test_benchmark_format_report_homogeneous_judge_no_suffix() -> None:
    """Kills line 894: ROR judge_model_meta is not None swapped to is None.

    When judge_backend matches primary backend and judge_model is None:
    Original: judge_model_meta is not None is False -> no judge_suffix is added.
    Mutant: judge_model_meta is None is True and != 'stub-model' -> appends '| judge_backend=... judge_model=None'.
    """
    meta = {
        "logitgate_version": "0.12.0",
        "python": "3.12",
        "platform": "linux",
        "backend": "stub://default",
        "model": "stub-model",
        "judge_backend": "stub://default",
        "judge_model": None,
        "is_stub": True,
        "timestamp": "2026-11-16T12:00:00Z",
        "dataset": {"source": "test", "cases": 1, "sha256_12": "abc123def456"},
        "repeat": 1,
        "warmup": 0,
        "judge_think": "strip",
        "judge_max_tokens": None,
        "cascade_trigger": "ambiguous",
    }
    report = BenchmarkReport(metadata=meta, modes={})
    table = format_report(report)
    first_line = table.splitlines()[0]
    assert "judge_backend=" not in first_line
    assert "judge_model=" not in first_line


# ==============================================================================
# SECTION 7: logitgate.cascade
# ==============================================================================


def test_cascade_fuse_escalated_verdict_abstain_logic() -> None:
    """Kills line 128: LCR tr1 is None or ... or ... swapped to and.

    Case 1: tr1 is None (upstream has no logit trace).
    Original: tr1 is None is True -> fast returns None without accessing tr1.metadata.
    Mutant: tr1 is None and tr1.metadata -> crashes with AttributeError on None.

    Case 2: Fusion condition (lines 148-151) is satisfied, but tr1 has abstained_reason.
    Original: detects abstained_reason and returns None for fused log odds.
    Mutant: and condition evaluates to False -> proceeds to fuse logits and returns 1.0!
    """
    tr_teacher = LayerTrace(
        layer_name="teacher",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=10,
        log_odds_pass=0.5,
        forward_log_odds=0.8,
        reverse_log_odds=-0.3,
    )
    v_teacher = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="teacher",
        total_latency_ms=1.0,
        score=0.9,
        traces=[tr_teacher],
    )

    # Case 1: tr1 is None
    tr_non_logit = LayerTrace(
        layer_name="non_logit",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=5,
        log_odds_pass=None,
    )
    up_no_logit = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        triggered_layer="non_logit",
        total_latency_ms=1.0,
        traces=[tr_non_logit],
    )
    _, _, _, lo_none = fuse_escalated_verdict(up_no_logit, v_teacher, prior_weight=0.5)
    assert lo_none is None

    # Case 2: tr1 satisfies fusion math (z1=1.5, f1=1.5, r1=1.5) but abstains
    tr_abstain = LayerTrace(
        layer_name="student",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=0,
        log_odds_pass=1.5,
        forward_log_odds=1.5,
        reverse_log_odds=1.5,
        metadata={"abstained_reason": "low_confidence"},
    )
    up_abstain = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        triggered_layer="student",
        total_latency_ms=1.0,
        traces=[tr_abstain],
    )
    _, _, _, lo_abstain = fuse_escalated_verdict(up_abstain, v_teacher, prior_weight=0.5)
    assert lo_abstain is None


class _MockStudentGate(BaseGate):
    layer_tier = 1

    def __init__(self, traces: list[LayerTrace]) -> None:
        super().__init__()
        self.name = "student"
        self.rubric = "Mock student"
        self._traces = traces

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            passed=False,
            status=DecisionStatus.AMBIGUOUS,
            triggered_layer="student",
            total_latency_ms=1.0,
            score=0.5,
            reason="Ambiguous student",
            traces=self._traces,
        )

    def replay_config(self) -> dict[str, Any] | None:
        return None


class _MockTeacherGate(BaseGate):
    layer_tier = 2
    escalation_policy: EscalationPolicy = "adjudicate"

    def __init__(self, trace: LayerTrace) -> None:
        super().__init__()
        self.name = "teacher"
        self.rubric = "Mock teacher"
        self._trace = trace

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer="teacher",
            total_latency_ms=1.0,
            score=0.9,
            reason="Teacher passed",
            traces=[self._trace],
        )

    def replay_config(self) -> dict[str, Any] | None:
        return None


def test_cascade_fuse_logits_prior_lo_selected_logit_trace() -> None:
    """Kills line 604: ROR tr1 is not None swapped to is None in CascadeGate.evaluate.

    When upstream traces contain multiple traces where tr_logit has log_odds_pass=1.5
    and tr_extra has log_odds_pass=4.56:
    Original: tr1 is not None is True -> prior_log_odds records tr1.log_odds_pass (1.5).
    Mutant: tr1 is None is False -> prior_log_odds records upstream.traces[-1].log_odds_pass (4.56).
    """
    tr_logit = LayerTrace(
        layer_name="student",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=0,
        log_odds_pass=1.5,
        forward_log_odds=1.5,
        reverse_log_odds=1.5,
    )
    tr_extra = LayerTrace(
        layer_name="extra",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=5,
        log_odds_pass=4.56,
        forward_log_odds=4.56,
        reverse_log_odds=4.56,
    )
    tr_teacher = LayerTrace(
        layer_name="teacher",
        status=DecisionStatus.PASSED,
        latency_ms=1.0,
        tokens_generated=10,
        log_odds_pass=0.5,
        forward_log_odds=0.8,
        reverse_log_odds=-0.3,
    )

    g_student = _MockStudentGate([tr_logit, tr_extra])
    g_teacher = _MockTeacherGate(tr_teacher)
    casc = Cascade([g_student, g_teacher], fuse_logits=True, prior_weight=0.5)

    v = casc.evaluate("candidate text")
    assert v.traces[-1].metadata["prior_log_odds"] == pytest.approx(1.5)


class _TierGate(BaseGate):
    def __init__(self, name: str, tier: int, should_pass: bool) -> None:
        super().__init__()
        self.name = name
        self.rubric = name
        self.layer_tier = tier
        self.should_pass = should_pass

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            passed=self.should_pass,
            status=DecisionStatus.PASSED if self.should_pass else DecisionStatus.FAILED,
            triggered_layer=self.name,
            total_latency_ms=1.0,
            score=1.0 if self.should_pass else 0.0,
            reason=f"{self.name} result",
            traces=[
                LayerTrace(
                    layer_name=self.name,
                    status=DecisionStatus.PASSED if self.should_pass else DecisionStatus.FAILED,
                    latency_ms=1.0,
                )
            ],
        )

    def replay_config(self) -> dict[str, Any] | None:
        return None


def test_cascade_fast_pathed_slice_addition() -> None:
    """Kills line 723: AOR self.gates[i + 1 :] swapped to [i - 1 :].

    Setup: 3-gate pipeline [g0 (tier 1, pass), g1 (tier 0, fail), g2 (tier 0)].
    At i = 1 (g1 fails, no generative deferral):
    Original: self.gates[i + 1 :] is [g2]. g2.layer_tier (0) > g1.layer_tier (0) is False -> fast_pathed is False.
    Mutant: self.gates[i - 1 :] is [g0, g1, g2]. g0.layer_tier (1) > g1.layer_tier (0) is True -> fast_pathed is True!
    """
    g0 = _TierGate("g0", tier=1, should_pass=True)
    g1 = _TierGate("g1", tier=0, should_pass=False)
    g2 = _TierGate("g2", tier=0, should_pass=True)

    casc = Cascade([g0, g1, g2])
    with patch("logitgate.telemetry.sinks.emit_verdict_telemetry") as mock_emit:
        casc.evaluate("candidate text")
        assert mock_emit.call_args[1]["fast_pathed"] is False


# ==============================================================================
# SECTION 8: logitgate.cache
# ==============================================================================


def test_cache_gate_fingerprint_replay_config_not_none() -> None:
    """Kills line 266: ROR if replay is not None swapped to is None in gate_cache_fingerprint.

    When gate.replay_config() returns a non-None dict:
    Original: replay is not None is True -> fp contains 'replay'.
    Mutant: replay is None is False -> 'replay' key is omitted from fp.
    """
    mock_gate = MagicMock()
    mock_gate.__class__.__name__ = "LogitGate"
    mock_gate.replay_config.return_value = {"mode": "deterministic"}
    mock_gate.backend = None
    mock_gate.profile = None
    mock_gate.adaptive_controller = None
    mock_gate.anchor_bank = None
    mock_gate.meta_calibrator = None
    mock_gate.unanchored_meta_calibrator = None
    for attr in ("rubric", "question", "options", "context", "pass_threshold", "fail_threshold"):
        setattr(mock_gate, attr, None)

    fp = gate_cache_fingerprint(mock_gate)
    assert "replay" in fp
    assert fp["replay"] is not None


def test_cache_get_entry_locked_exact_expiry_boundary() -> None:
    """Kills line 535: ROR now >= mem_entry.expires_at swapped to >.

    When now == expires_at (exact boundary):
    Original: now >= expires_at is True -> entry is expired -> pops entry and returns None.
    Mutant: now > expires_at is False -> entry is mistakenly treated as valid.
    """
    cache = VerdictCache()
    entry = _CacheEntry(
        verdict=Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer="test",
            total_latency_ms=1.0,
        ),
        created_at=100.0,
        expires_at=200.0,
        original_latency_ms=1.0,
        original_tokens=0,
    )
    cache._mem["k1"] = entry
    res = cache._get_entry_locked("k1", now=200.0)
    assert res is None


# ==============================================================================
# SECTION 9: logitgate.snapshot
# ==============================================================================


class _PureLogitBackend(LogitBackend):
    """LogitBackend that does NOT inherit from GenerativeBackend."""

    def score(
        self,
        rubric: str,
        candidate: Any,
        *,
        system_prompt: str | None = None,
        context: Mapping[str, Any] | None = None,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> float:
        return 0.95

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [{"token": "A", "logprob": 0.0}]


class _DummyChildGate(BaseGate):
    def __init__(self, backend: Any) -> None:
        super().__init__()
        self.backend = backend
        self.rubric = "dummy"

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer="dummy",
            total_latency_ms=1.0,
        )

    def replay_config(self) -> dict[str, Any] | None:
        return None


def test_snapshot_rebuild_child_exact_gate_type_check() -> None:
    """Kills line 515: ROR cfg.get('gate_type') == 'JudgeGate' swapped to !=.

    When rebuilding a BestOfNGate with a child LogitGate using a pure LogitBackend:
    Original: c_is_judge is False ('LogitGate' == 'JudgeGate' is False).
              isinstance(backend, LogitBackend) is True -> reuses pure backend as child.backend.
    Mutant: c_is_judge is True ('LogitGate' != 'JudgeGate' is True).
            isinstance(backend, GenerativeBackend) is False -> falls back to resolving new backend.
    """
    pure_backend = _PureLogitBackend()
    snapshot = VerdictSnapshot(
        rubric="test rubric",
        candidate="test candidate",
        verdict=Verdict(
            passed=False,
            status=DecisionStatus.FAILED,
            triggered_layer="test",
            total_latency_ms=1.0,
            score=0.0,
        ),
    )
    gate_kwargs = {
        "gate_type": "BestOfNGate",
        "pre_gate_config": {
            "gate_type": "LogitGate",
            "rubric": "Child rubric",
            "gate_kwargs": {"pass_threshold": 0.5},
        },
    }
    gate, _ = snapshot._build_replay_gate(gate_kwargs, effective_backend=pure_backend, effective_model="model")
    assert gate.pre_gate.backend is pure_backend


def test_snapshot_rebuild_extra_backends_excludes_best_gate_backend() -> None:
    """Kills line 557: LCR c_b is not None and c_b is not best_gate.backend and ... swapped to or.

    Tests two critical properties of BestOfNGate extra_backends collection:
    1. Duplicate child backends (pre_gate and post_gate sharing shared_b) are deduplicated
       so shared_b appears exactly once (len == 1).
       Under mutant 93 (and -> or), len == 2 (duplicated).
       Under mutants 94, 95, 96, len == 0 (prematurely omitted).
    2. Child sharing best_gate.backend is excluded from _replay_extra_backends.
    """
    snapshot = VerdictSnapshot(
        rubric="test rubric",
        candidate="test candidate",
        verdict=Verdict(
            passed=False,
            status=DecisionStatus.FAILED,
            triggered_layer="test",
            total_latency_ms=1.0,
            score=0.0,
        ),
    )
    shared_b = StubBackend()
    g1 = _DummyChildGate(shared_b)
    g2 = _DummyChildGate(shared_b)

    orig_build = snapshot._build_replay_gate

    def fake_build(cfg: Any, b: Any, m: Any) -> tuple[Any, bool]:
        if isinstance(cfg, dict):
            if cfg.get("name") == "child1":
                return g1, True
            if cfg.get("name") == "child2":
                return g2, True
        return orig_build(cfg, b, m)

    object.__setattr__(snapshot, "_build_replay_gate", fake_build)
    gate_kwargs = {
        "gate_type": "BestOfNGate",
        "pre_gate_config": {"name": "child1"},
        "post_gate_config": {"name": "child2"},
    }
    best_gate, _ = orig_build(gate_kwargs, effective_backend=StubBackend(), effective_model=None)
    assert best_gate._replay_extra_backends == [shared_b]

    # Child sharing best_gate.backend is excluded
    best_backend = StubBackend()
    g_same = _DummyChildGate(best_backend)

    def fake_build_same(cfg: Any, b: Any, m: Any) -> tuple[Any, bool]:
        if isinstance(cfg, dict) and cfg.get("name") == "same_child":
            return g_same, True
        return orig_build(cfg, b, m)

    object.__setattr__(snapshot, "_build_replay_gate", fake_build_same)
    gate_kwargs_same = {
        "gate_type": "BestOfNGate",
        "pre_gate_config": {"name": "same_child"},
    }
    best_gate_same, _ = orig_build(gate_kwargs_same, effective_backend=best_backend, effective_model=None)
    assert best_backend not in best_gate_same._replay_extra_backends
    assert len(best_gate_same._replay_extra_backends) == 0
