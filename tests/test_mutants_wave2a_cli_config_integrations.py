"""Wave 2A Mutant-Killing and Branch-Closure test suite.

Explicitly targets and closes all AST mutant survival paths across:
- logitgate.exceptions
- logitgate.models
- logitgate.cli
- logitgate.config
- logitgate.integrations.deepeval
- logitgate.integrations.pytest_plugin
- logitgate.integrations.langgraph
- logitgate.telemetry.models
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any, TextIO, cast
from unittest.mock import MagicMock, patch

import pytest

# Explicit top-level module imports for mutation engine mapping:
# Explicit symbol imports:
from logitgate.backends.base import LogitBackend, StubBackend
from logitgate.cache import VerdictCache
from logitgate.cli import (
    EXIT_BACKEND_ERROR,
    EXIT_FAILED,
    EXIT_OK,
    _bias_hint,
    _cmd_benchmark,
    _cmd_calibrate,
    _cmd_eval,
    _cmd_explain,
    _cmd_rank,
    _cmd_stream,
    _cmd_telemetry,
    _entropy_hint,
    _Style,
    _use_color,
    render_snapshot_report,
)
from logitgate.config import (
    LogitGateDefaults,
    _ConfigState,
    _set_telemetry_sink,
    _set_verdict_cache,
    _validate_num_ctx,
    _validate_threshold,
    configure,
    get_default_telemetry_sink,
    get_default_verdict_cache,
)
from logitgate.exceptions import (
    BackendError,
    ConfigurationError,
    StreamAbortedError,
)
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import ChoiceGate, LogitGate
from logitgate.integrations.deepeval import (
    ChoiceQuestion,
    LogitGateMetric,
    LogitGateSystemOneModel,
    NoulQuestion,
    _extract_test_case_candidate,
)
from logitgate.integrations.langgraph import (
    _extract,
    _router,
)
from logitgate.integrations.pytest_plugin import (
    check,
)
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import VerdictSnapshot
from logitgate.telemetry.models import TelemetryRecord, _luhn_valid
from logitgate.telemetry.sinks import TelemetrySink

# =============================================================================================
# 1. logitgate.exceptions MUTANT KILLERS
# =============================================================================================


def test_exceptions_stream_aborted_error_mutants() -> None:
    """Kills mutants at lines 93 and 94 (or -> and) in exceptions.py."""
    v_empty = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="TestLayer",
        total_latency_ms=1.5,
        reason="",  # empty string: line 93 detail = reason or status.value
    )
    # detail should fall back to status.value ("failed")
    # If mutated (or -> and), "" and "failed" evaluates to ""
    err1 = StreamAbortedError(v_empty, partial_text="abc", is_final=False)
    assert ": failed" in str(err1)
    assert str(err1) == "Stream aborted at chunk 0 (mid-stream, 3 chars): failed"

    v_reason = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="TestLayer",
        total_latency_ms=2.0,
        reason="token policy violated",
    )
    # detail should be reason. If mutated (or -> and), "token policy violated" and "failed" evaluates to "failed"
    err2 = StreamAbortedError(v_reason, partial_text="abcdef", is_final=True)
    assert "token policy violated" in str(err2)
    assert str(err2) == "Stream aborted at chunk 0 (final, 6 chars): token policy violated"

    # Line 94: msg = message or (...)
    # When message is None, if mutated (or -> and), None and (...) evaluates to None!
    assert str(err2) != "None"
    assert len(str(err2)) > 10

    # When message is explicitly provided, if mutated (or -> and), message and (...) evaluates to (...)!
    err3 = StreamAbortedError(v_reason, partial_text="abcdef", message="custom termination message")
    assert str(err3) == "custom termination message"


# =============================================================================================
# 2. logitgate.models MUTANT KILLERS & 100% COVERAGE
# =============================================================================================


def test_models_layer_trace_repr_and_branches() -> None:
    """Kills mutants and closes lines 83, 85, 87, 89 in LayerTrace.__repr__."""
    # 1. Base minimal
    lt_min = LayerTrace(
        layer_name="ASTGate",
        status=DecisionStatus.PASSED,
        latency_ms=1.234,
    )
    assert repr(lt_min) == "LayerTrace(layer='ASTGate', status=passed, latency=1.23ms)"
    assert "prob=" not in repr(lt_min)
    assert "bias=" not in repr(lt_min)
    assert "tokens=" not in repr(lt_min)
    assert "failure_code=" not in repr(lt_min)

    # 2. tokens_generated == 0 should NOT be included
    lt_zero = LayerTrace(
        layer_name="ASTGate",
        status=DecisionStatus.PASSED,
        latency_ms=0.5,
        tokens_generated=0,
        failure_code=None,
    )
    assert "tokens=" not in repr(lt_zero)

    # 3. Full LayerTrace with all branches active:
    # line 83: calibrated_prob is not None
    # line 85: positional_bias_magnitude is not None
    # line 87: tokens_generated > 0
    # line 89: failure_code is not None and non-empty
    lt_full = LayerTrace(
        layer_name="LogitGate",
        status=DecisionStatus.FAILED,
        latency_ms=5.678,
        calibrated_prob=0.87654,
        positional_bias_magnitude=0.12345,
        tokens_generated=42,
        failure_code="RECURRENT_TIMEOUT",
    )
    expected_full = (
        "LayerTrace(layer='LogitGate', status=failed, latency=5.68ms, "
        "prob=0.8765, bias=0.1235, tokens=42, failure_code='RECURRENT_TIMEOUT')"
    )
    assert repr(lt_full) == expected_full


def test_models_verdict_repr_and_bool() -> None:
    """Kills mutants in Verdict.__repr__ and Verdict.__bool__."""
    v_pass = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.91234,
        score=4.567,
        choice="A",
        choice_probabilities={"A": 0.9, "B": 0.1},
        triggered_layer="LogitGate",
        total_latency_ms=10.5,
        total_tokens_generated=15,
        reason="met criteria",
    )
    assert bool(v_pass) is True
    assert repr(v_pass) == "Verdict(status='passed', prob=0.9123, score=4.57, choice='A', layer='LogitGate', latency=10.50ms, tokens=15)"

    v_fail = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="ASTGate",
        total_latency_ms=2.0,
        total_tokens_generated=0,
    )
    assert bool(v_fail) is False
    assert repr(v_fail) == "Verdict(status='failed', layer='ASTGate', latency=2.00ms, tokens=0)"


# =============================================================================================
# 3. logitgate.cli MUTANT KILLERS
# =============================================================================================


def test_cli_use_color_mode_always_and_never() -> None:
    """Kills line 839 (== -> !=) in cli._use_color."""
    class DummyStream:
        def isatty(self) -> bool:
            return False

    s = cast(TextIO, DummyStream())
    with patch.dict(os.environ, {}, clear=True):
        # mode == "always" must return True regardless of tty
        assert _use_color("always", s) is True
        # mode == "never" must return False
        assert _use_color("never", s) is False
        # mode == "auto" on non-tty must return False
        assert _use_color("auto", s) is False


def test_cli_bias_hint_exact_threshold_ties() -> None:
    """Kills line 871 (< -> <=) and surrounding comparisons in cli._bias_hint."""
    # Boundary 1: bias = 0.5
    # If bias < 0.5: green. When bias == 0.5, strictly False -> yellow!
    msg_05, col_05 = _bias_hint(0.5)
    assert col_05 == "yellow"
    assert "moderate label-order prior" in msg_05

    msg_049, col_049 = _bias_hint(0.499)
    assert col_049 == "green"
    assert "forward and reverse passes agree" in msg_049

    # Boundary 2: bias = 1.5
    # If bias < 1.5: yellow. When bias == 1.5, strictly False -> red!
    msg_15, col_15 = _bias_hint(1.5)
    assert col_15 == "red"
    assert "strong label-order prior" in msg_15

    msg_149, col_149 = _bias_hint(1.499)
    assert col_149 == "yellow"
    assert "moderate label-order prior" in msg_149


def test_cli_entropy_hint_exact_threshold_ties() -> None:
    """Kills threshold mutants in cli._entropy_hint."""
    msg_09, col_09 = _entropy_hint(0.9)
    assert col_09 == "red"
    assert "near coin-flip" in msg_09

    msg_089, col_089 = _entropy_hint(0.899)
    assert col_089 == "yellow"
    assert "uncertain" in msg_089

    _msg_06, col_06 = _entropy_hint(0.6)
    assert col_06 == "yellow"

    msg_059, col_059 = _entropy_hint(0.599)
    assert col_059 == "green"
    assert "confident" in msg_059


def test_cli_render_snapshot_report_mutants() -> None:
    """Kills mutants at lines 928 (- -> +), 967 (or -> and), 989 (+ -> -), 1011 (is -> is not)."""
    style = _Style(color=False, stream=sys.stdout)

    trace = LayerTrace(
        layer_name="LogitGate",
        status=DecisionStatus.FAILED,
        latency_ms=12.34,
        forward_prob=0.8500,
        reverse_prob=0.7500,
        log_odds_pass=1.234,
        shannon_entropy=0.550,
        positional_bias_magnitude=0.10,
        tokens_generated=5,
        diagnostic_report="rubric check failed",
    )

    verdict = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.4500,
        triggered_layer="LogitGate",
        total_latency_ms=12.34,
        total_tokens_generated=5,
        traces=[trace],
    )

    # 65 chars rubric: with width = 72 - 17 = 55, it wraps into 2 lines!
    # If line 928 mutated (- -> +), width would be 89, so 65 chars would not wrap!
    long_rubric = "This is a sixty-five character test rubric intended for line wrapping"
    snapshot = VerdictSnapshot(
        snapshot_id="snap_123",
        timestamp="2026-11-16T12:00:00Z",
        rubric=long_rubric,
        candidate="short candidate text",
        verdict=verdict,
        gate_config={"pass_threshold": 0.85, "fail_threshold": 0.20},
    )

    # 1. Test replay is None (line 1011: is -> is not)
    rep_none = render_snapshot_report(snapshot, style, replay=None)
    assert "Replay skipped." in rep_none
    assert "Replay Verdict:" not in rep_none
    assert "Recorded critique (LogitGate, 5 tokens):" in rep_none

    # Line 967: (FAILED at LogitGate). If line 967 mutated (or -> and), "LogitGate" and "" evaluates to ""!
    assert "(FAILED at LogitGate)" in rep_none

    # Line 989 (+ -> -): string concatenation for forward and reverse pass
    assert "P=0.8500 (A=pass) / P=0.7500 (B=pass)" in rep_none

    # Line 928 (- -> +): rubric wrapping check under width 55
    assert "intended for\n                 line wrapping" in rep_none

    # 2. Test replay is not None
    replay_verdict = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="ReplayJudge",
        total_latency_ms=25.0,
        total_tokens_generated=50,
        reason="all issues clarified upon review",
    )
    rep_with_replay = render_snapshot_report(snapshot, style, replay=replay_verdict)
    assert "Replay Verdict:" in rep_with_replay
    assert "PASSED" in rep_with_replay
    assert "Diagnostic Critique:" in rep_with_replay
    assert "all issues clarified upon review" in rep_with_replay

    # 3. Line 967: verdict.triggered_layer is empty / None
    verdict_no_layer = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.10,
        triggered_layer="",
        total_latency_ms=1.0,
    )
    snap_no_layer = VerdictSnapshot(
        snapshot_id="snap_456",
        timestamp="2026-11-16T12:00:00Z",
        rubric="simple rubric",
        candidate="cand",
        verdict=verdict_no_layer,
    )
    rep_empty_layer = render_snapshot_report(snap_no_layer, style)
    assert "Triggered Layer:" in rep_empty_layer


def test_cli_cmd_calibrate_sample_size_mutant() -> None:
    """Kills line 1068 (< -> <=) in _cmd_calibrate."""
    args = argparse.Namespace(
        dataset="fake_dataset.jsonl",
        backend="stub://",
        model="default",
        alpha=0.05,
        output=None,
        cache_path=None,
        json=False,
    )
    # Exactly 10 samples (5 positive, 5 negative):
    # If len(samples) < 10: passes. If mutated to <= 10, it fails with "requires at least 10 samples"!
    samples_10 = [("cand", "rubric", i % 2 == 0) for i in range(10)]
    mock_prof = MagicMock()
    mock_prof.fail_threshold = 0.20
    mock_prof.pass_threshold = 0.85
    mock_prof.target_risk_alpha = 0.05
    mock_prof.empirical_escalation_rate = 0.10
    mock_prof.binary_temperature = 1.0
    mock_prof.calibration_set_size = 10

    with (
        patch("logitgate.calibration.conformal.load_calibration_dataset", return_value=samples_10),
        patch("logitgate.calibration.profile.fit_calibration_profile", return_value=mock_prof),
    ):
        rc = _cmd_calibrate(args)
        assert rc == EXIT_OK


def test_cli_cmd_benchmark_judge_max_tokens_mutant() -> None:
    """Kills line 1278 (< -> <=) in _cmd_benchmark."""
    args = argparse.Namespace(
        audit_only=False,
        dataset=None,
        backend="stub://",
        model="default",
        judge_backend=None,
        judge_model=None,
        judge_max_tokens=1,  # Exactly 1! If < 1 mutated to <= 1, it raises error!
        repeat=1,
        warmup=0,
        output=None,
        quiet=True,
        limit=1,  # line 1291: limit < 1
        modes="logitgate",
        judge_think=False,
        cascade_trigger="on_ambiguous_only",
        profile=None,
        bundle=None,
        cache=None,
        json=False,
        reference_dataset=None,
    )
    mock_rep = MagicMock()
    mock_rep.format_summary.return_value = "Summary"
    with (
        patch("logitgate.benchmark.load_cases", return_value=([("cand", "rubric", True)], "label")),
        patch("logitgate.benchmark.run_benchmark", return_value=mock_rep),
        patch("logitgate.benchmark.format_report", return_value="Report summary"),
    ):
        rc = _cmd_benchmark(args)
        assert rc == EXIT_OK


def test_cli_cmd_explain_backend_error_not_json_mutant() -> None:
    """Kills line 1213 (not x -> x) in _cmd_explain."""
    args = argparse.Namespace(
        snapshot_path="test_snap.json",
        backend="stub://",
        model="default",
        max_tokens=50,
        no_think=True,
        json=False,  # If not args.json mutated to args.json, report is NOT printed!
        offline=False,
        color="never",
    )
    v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="LogitGate",
        total_latency_ms=5.0,
    )
    real_snap = VerdictSnapshot(
        snapshot_id="s1",
        timestamp="2026-11-16T12:00:00Z",
        rubric="rubric",
        candidate="candidate",
        verdict=v,
        gate_config={},
    )

    with (
        patch.object(VerdictSnapshot, "load", return_value=real_snap),
        patch.object(
            VerdictSnapshot,
            "explain",
            side_effect=BackendError("model inference dropped connection"),
        ),
        patch("sys.stdout", new_callable=io.StringIO) as mock_out,
    ):
        rc = _cmd_explain(args)
        assert rc == EXIT_BACKEND_ERROR
        output = mock_out.getvalue()
        # Must have printed snapshot report!
        assert "LOGITGATE FAILURE REPLAY & EXPLANATION" in output


def test_cli_cmd_eval_cache_arg_mutant() -> None:
    """Kills line 1370 (== -> !=) in _cmd_eval."""
    args = argparse.Namespace(
        rubric="test rubric",
        candidate="sample candidate",
        file=None,
        threshold=0.85,
        fail_threshold=0.20,
        backend="stub://",
        model="default",
        profile=None,
        bundle=None,
        telemetry_out=None,
        cache="memory",  # Exactly "memory"!
        ast_syntax=False,
        normalize=False,
        json=False,
    )
    created_caches: list[Any] = []
    orig_init = LogitGate.__init__

    def capture_init(self: Any, *a: Any, **kw: Any) -> None:
        created_caches.append(kw.get("cache"))
        orig_init(self, *a, **kw)

    with patch.object(LogitGate, "__init__", capture_init):
        rc = _cmd_eval(args)
        assert rc in (EXIT_OK, EXIT_FAILED)
        assert len(created_caches) == 1
        # When cache == "memory", VerdictCache() has path is None!
        # If line 1370 is mutated (== -> !=), VerdictCache(path="memory") has path == "memory"!
        assert created_caches[0].path is None


def test_cli_cmd_rank_candidates_json_mutant(tmp_path: Path) -> None:
    """Kills line 1448 (is not -> is) in _cmd_rank."""
    cand_file = tmp_path / "candidates.json"
    cand_file.write_text('["Option 1", "Option 2"]', encoding="utf-8")

    # 1. candidates_json is provided: must load from json
    args1 = argparse.Namespace(
        rubric="pick best",
        candidate=[],
        file=[],
        candidates_json=str(cand_file),
        backend="stub://",
        model="default",
        output=None,
        json=False,
        n=2,
        ast_syntax=False,
        normalize=False,
        threshold=None,
        fail_threshold=None,
        profile=None,
        cache=None,
        telemetry_out=None,
        temperature=None,
        symmetric=True,
        verify_winner=False,
        mode="tournament",
        context=None,
        min_confidence=None,
        min_margin=None,
        max_positional_bias=None,
    )
    rc = _cmd_rank(args1)
    assert rc == EXIT_OK

    # 2. candidates_json is None: must use candidates list
    args2 = argparse.Namespace(
        rubric="pick best",
        candidate=["Candidate A", "Candidate B"],
        file=[],
        candidates_json=None,
        backend="stub://",
        model="default",
        output=None,
        json=False,
        n=2,
        ast_syntax=False,
        normalize=False,
        threshold=None,
        fail_threshold=None,
        profile=None,
        cache=None,
        telemetry_out=None,
        temperature=None,
        symmetric=True,
        verify_winner=False,
        mode="tournament",
        context=None,
        min_confidence=None,
        min_margin=None,
        max_positional_bias=None,
    )
    rc2 = _cmd_rank(args2)
    assert rc2 == EXIT_OK


def test_cli_cmd_telemetry_recalibrate_mutant() -> None:
    """Kills line 1639 (== -> !=) in _cmd_telemetry."""
    args = argparse.Namespace(
        telemetry_command="recalibrate",
        input_path="telemetry.jsonl",
        alpha=0.05,
        model="default",
        output=None,
        no_dedupe=False,
        json=False,
    )
    mock_prof = MagicMock()
    mock_prof.calibration_set_size = 50
    mock_prof.binary_temperature = 1.05
    mock_prof.fail_threshold = 0.20
    mock_prof.pass_threshold = 0.85
    mock_prof.empirical_escalation_rate = 0.12

    with patch("logitgate.telemetry.recalibrate_from_telemetry", return_value=mock_prof):
        rc = _cmd_telemetry(args)
        assert rc == EXIT_OK


def test_cli_cmd_stream_chunk_indexing_and_quiet_stream_mutants() -> None:
    """Kills lines 1738 (+ -> -) and 1813 (and -> or) in _cmd_stream."""
    # 1. Line 1738: chunks = raw_text[i : i + args.chunk_size]
    # If mutated to i - chunk_size, chunks would be empty or inverted!
    args_chunks = argparse.Namespace(
        chunk_size=5,
        file=None,
        candidate="0123456789",  # exactly 10 chars
        rubric="safety check",
        threshold=0.85,
        fail_threshold=0.20,
        backend="stub://",
        model="default",
        judge_backend=None,
        judge_model=None,
        profile=None,
        bundle=None,
        min_chars=1,
        stride_chars=1,
        no_clause_boundary=True,
        consecutive_failures=1,
        delivery="optimistic",
        telemetry_out=None,
        cache=None,
        json=False,
        quiet_stream=False,
    )
    with patch("sys.stdout", new_callable=io.StringIO) as mock_out1:
        rc1 = _cmd_stream(args_chunks)
        assert rc1 in (EXIT_OK, EXIT_FAILED)
        out1 = mock_out1.getvalue()
        # Chunks: 2/2 yielded (10/10 chars)
        assert "10/10 chars" in out1

    # 2. Line 1813: if not args.quiet_stream and yielded_text:
    # When candidate="" (so yielded_text=""), not args.quiet_stream is True.
    # With and: True and "" is False -> does NOT print "------------------------------------------------------------"
    # With or: True or "" is True -> PRINTS "------------------------------------------------------------"
    args_empty = argparse.Namespace(
        chunk_size=10,
        file=None,
        candidate="",  # empty text -> yielded_text is ""
        rubric="safety check",
        threshold=0.85,
        fail_threshold=0.20,
        backend="stub://",
        model="default",
        judge_backend=None,
        judge_model=None,
        profile=None,
        bundle=None,
        min_chars=1,
        stride_chars=1,
        no_clause_boundary=True,
        consecutive_failures=1,
        delivery="optimistic",
        telemetry_out=None,
        cache=None,
        json=False,
        quiet_stream=False,
    )
    with patch("sys.stdout", new_callable=io.StringIO) as mock_out2:
        rc2 = _cmd_stream(args_empty)
        assert rc2 in (EXIT_OK, EXIT_FAILED)
        out2 = mock_out2.getvalue()
        # Must NOT contain separator line from line 1815!
        assert "-" * 60 not in out2


# =============================================================================================
# 4. logitgate.config MUTANT KILLERS
# =============================================================================================


def test_config_close_owned_telemetry_mutant() -> None:
    """Kills line 35 (is not -> is) in _ConfigState.close_owned_telemetry."""
    state = _ConfigState()
    mock_sink = MagicMock()
    state.owned_telemetry_sink = mock_sink
    state.close_owned_telemetry()
    assert mock_sink.close.called
    assert state.owned_telemetry_sink is None


def test_config_validate_threshold_mutants() -> None:
    """Kills line 78 (not x -> x) in _validate_threshold."""
    assert _validate_threshold("p", 0.0) == 0.0
    assert _validate_threshold("p", 0.5) == 0.5
    assert _validate_threshold("p", 1.0) == 1.0

    with pytest.raises(ConfigurationError):
        _validate_threshold("p", -0.01)
    with pytest.raises(ConfigurationError):
        _validate_threshold("p", 1.01)
    with pytest.raises(ConfigurationError):
        _validate_threshold("p", "not-a-number")


def test_config_validate_num_ctx_mutants() -> None:
    """Kills line 97 (< -> <=) in _validate_num_ctx."""
    assert _validate_num_ctx("ctx", 1) == 1  # exactly 1: if < 1 mutated to <= 1, it raises!
    assert _validate_num_ctx("ctx", 2048) == 2048

    with pytest.raises(ConfigurationError):
        _validate_num_ctx("ctx", 0)
    with pytest.raises(ConfigurationError):
        _validate_num_ctx("ctx", -1)


def test_config_defaults_property_setters_none_handling() -> None:
    """Kills line 175 and 211 (is -> is not) in LogitGateDefaults setters."""
    # fail_threshold setter
    LogitGateDefaults.fail_threshold = None
    assert LogitGateDefaults.fail_threshold == 0.20
    LogitGateDefaults.fail_threshold = 0.30
    assert LogitGateDefaults.fail_threshold == 0.30
    LogitGateDefaults.fail_threshold = None

    # auto_calibrate setter
    LogitGateDefaults.auto_calibrate = None
    assert LogitGateDefaults.auto_calibrate is True
    LogitGateDefaults.auto_calibrate = False
    assert LogitGateDefaults.auto_calibrate is False
    LogitGateDefaults.auto_calibrate = None
    assert LogitGateDefaults.auto_calibrate is True


def test_config_set_verdict_cache_and_sink_ownership_mutants(tmp_path: Path) -> None:
    """Kills line 292 (is not -> is) and line 317 (and -> or)."""
    # Line 292: replacing owned cache with a new cache must close old owned cache
    old_cache = MagicMock(spec=VerdictCache)
    new_cache = MagicMock(spec=VerdictCache)

    with patch("logitgate.config._state") as mock_state:
        mock_state.owned_cache = old_cache
        mock_state.cache = old_cache

        with patch("logitgate.cache.resolve_verdict_cache", return_value=new_cache):
            _set_verdict_cache(str(tmp_path / "cache2.db"))
            assert old_cache.close.called

    # Line 317: owns_new and isinstance(resolved, TelemetrySink)
    # When val is an existing sink instance (owns_new is False), owned_telemetry_sink must NOT be set!
    existing_sink = MagicMock(spec=TelemetrySink)
    with patch("logitgate.config._state") as mock_state2:
        mock_state2.owned_telemetry_sink = None
        with patch("logitgate.telemetry.sinks.resolve_telemetry_sink", return_value=existing_sink):
            _set_telemetry_sink(existing_sink)
            # If line 317 mutated (and -> or), owned_telemetry_sink would incorrectly be assigned existing_sink!
            assert mock_state2.owned_telemetry_sink is None


def test_config_configure_validation_and_threshold_mutants(tmp_path: Path) -> None:
    """Kills lines 386 (and -> or), 392, 394 (or -> and), 408 (is not -> is), 431 (is not -> is)."""
    # Line 394: (val_pass is not None or val_fail is not None) and eff_pass <= eff_fail
    # When pass_threshold=0.10 and fail_threshold is default 0.20:
    # 0.10 <= 0.20 -> must raise ConfigurationError!
    with pytest.raises(ConfigurationError, match="strictly greater"):
        configure(pass_threshold=0.10)

    # Line 408: configure(auto_calibrate=False)
    configure(auto_calibrate=False)
    assert LogitGateDefaults.auto_calibrate is False
    configure(auto_calibrate=True)
    assert LogitGateDefaults.auto_calibrate is True

    # Line 431: owns_new_cache and resolved_cache is not False and resolved_cache is not None
    # If line 431 mutated (is not False -> is False), _state.owned_cache is NOT set!
    db_path = str(tmp_path / "test_owned_cache.db")
    configure(cache=db_path)
    from logitgate.config import _state

    assert _state.owned_cache is not None
    assert isinstance(_state.owned_cache, VerdictCache)
    configure(cache=False)
    assert _state.owned_cache is None

    # Line 386: if owns_new_sink and resolved_sink is not False and resolved_sink is not None:
    # When owns_new_sink is False (unowned sink object), if cache resolution raises an error:
    # With and: unowned sink is NOT closed!
    # With or: False or resolved_sink is not False -> calls close()!
    unowned_sink = MagicMock(spec=TelemetrySink)
    with patch("logitgate.cache.resolve_verdict_cache", side_effect=ValueError("cache error")):
        with pytest.raises(ValueError):
            configure(telemetry_sink=unowned_sink, cache="trigger_error")
        assert not unowned_sink.close.called


def test_config_get_default_cache_and_sink_mutants() -> None:
    """Kills lines 530 (is not -> is) and 553 (is -> is not)."""
    # Line 530: if _state.telemetry_sink is not None: return _state.telemetry_sink
    # If mutated (is not -> is), returns None instead of the configured sink!
    mock_sink = MagicMock(spec=TelemetrySink)
    configure(telemetry_sink=mock_sink)
    assert get_default_telemetry_sink() is mock_sink
    configure(telemetry_sink=False)
    assert get_default_telemetry_sink() is None

    # Line 553: if _state.cache is False: return None
    configure(cache=False)
    assert get_default_verdict_cache() is None

    mock_cache = MagicMock(spec=VerdictCache)
    configure(cache=mock_cache)
    assert get_default_verdict_cache() is mock_cache

    configure(cache=False)
    assert get_default_verdict_cache() is None


# =============================================================================================
# 5. logitgate.integrations.deepeval MUTANT KILLERS
# =============================================================================================


def test_deepeval_fallback_classes_or_mutants() -> None:
    """Kills lines 48, 70, 85 (or -> and) in deepeval fallback classes."""
    code_obj = None
    for finder in sys.meta_path:
        if getattr(finder, "target_module", None) == "logitgate.integrations.deepeval":
            loader = getattr(finder, "loader", None)
            code_obj = getattr(loader, "_code_obj", None)
            break

    if code_obj is None:
        p = Path(__file__).resolve().parent.parent / "src" / "logitgate" / "integrations" / "deepeval.py"
        code_obj = compile(p.read_text(encoding="utf-8"), str(p), "exec")

    ns: dict[str, Any] = {"__name__": "logitgate.integrations.deepeval"}
    with patch.dict(
        sys.modules,
        {
            "deepeval": None,
            "deepeval.metrics": None,
            "deepeval.system_one": None,
            "deepeval.test_case": None,
        },
    ):
        exec(code_obj, ns)  # noqa: S102

    ChoiceAns = ns["ChoiceAnswer"]
    ChoiceQ = ns["ChoiceQuestion"]
    Sys1Ans = ns["SystemOneAnswers"]

    # Line 48: probabilities or {}
    ca_none = ChoiceAns("A", probabilities=None)
    assert ca_none.probabilities == {}
    ca_dict = ChoiceAns("A", probabilities={"A": 0.8, "B": 0.2})
    assert ca_dict.probabilities == {"A": 0.8, "B": 0.2}

    # Line 70: options or {}
    cq_none = ChoiceQ("Choose", options=None)
    assert cq_none.options == {}
    cq_dict = ChoiceQ("Choose", options={"A": "Option A"})
    assert cq_dict.options == {"A": "Option A"}

    # Line 85: nouls or {}
    s1_none = Sys1Ans(nouls=None)
    assert s1_none.nouls == {}
    s1_dict = Sys1Ans(nouls={"q1": ns["NoulAnswer"](0.9)})
    assert len(s1_dict.nouls) == 1


def test_deepeval_system_one_model_init_or_mutants() -> None:
    """Kills lines 114 (or -> and) and 134 (or -> and) in LogitGateSystemOneModel.__init__."""
    # Line 114: self.api_base = api_base or "http://localhost:11434"
    # When api_base is "http://custom:11434", if mutated (or -> and),
    # "http://custom:11434" and "http://localhost:11434" evaluates to "http://localhost:11434"!
    m1 = LogitGateSystemOneModel(api_base="http://custom:11434", backend="stub://")
    assert m1.api_base == "http://custom:11434"

    m2 = LogitGateSystemOneModel(api_base=None, backend="stub://")
    assert m2.api_base == "http://localhost:11434"

    # Line 134: self.model_name = model_name or get_default_model() or "qwen2.5:7b"
    # When api_base is not None, model_name is "my_special_model":
    m3 = LogitGateSystemOneModel(api_base="http://test:11434", model_name="my_special_model")
    assert m3.model_name == "my_special_model"


def test_deepeval_extract_options_exact_boundary() -> None:
    """Kills line 187 (>= -> >) in _extract_options."""
    # Exactly 2 options: len(opts) == 2.
    # If >= 2 mutated to > 2, it would return None!
    opts_2 = {"A": "Choice A", "B": "Choice B"}
    extracted = LogitGateSystemOneModel._extract_options({"options": opts_2})
    assert extracted == opts_2

    opts_1 = {"A": "Single"}
    assert LogitGateSystemOneModel._extract_options({"options": opts_1}) is None


def test_deepeval_extract_statement_fallbacks() -> None:
    """Kills line 171 (or -> and) in _extract_statement."""
    assert LogitGateSystemOneModel._extract_statement({"rubric": "rubric only"}) == "rubric only"
    assert LogitGateSystemOneModel._extract_statement({"question": "question only"}) == "question only"
    assert LogitGateSystemOneModel._extract_statement({"instructions": "inst only"}) == "inst only"
    assert LogitGateSystemOneModel._extract_statement({"statement": "stmt only"}) == "stmt only"


def test_deepeval_extract_test_case_candidate() -> None:
    """Kills line 412 (is not -> is) in _extract_test_case_candidate."""
    class CaseInputOnly:
        def __init__(self) -> None:
            self.actual_output = None
            self.input = "input candidate string"

    res = _extract_test_case_candidate(CaseInputOnly())
    assert res == "input candidate string"


def test_deepeval_metric_is_successful_mutants() -> None:
    """Kills line 478 (or -> and) in LogitGateMetric.is_successful."""
    metric = LogitGateMetric("rubric", threshold=0.85, backend="stub://")
    metric.success = None
    metric.score = None
    # If line 478 mutated (or -> and), self.score >= self.threshold would execute with None and raise TypeError!
    assert metric.is_successful() is False

    metric.threshold = None
    metric.score = 0.90
    assert metric.is_successful() is False

    metric.threshold = 0.85
    metric.score = 0.85  # exact tie
    assert metric.is_successful() is True


def test_deepeval_system_one_decide_probability_and_choice_mutants() -> None:
    """Kills lines 226 (or -> and) and 266 (is not -> is) in deepeval.py."""
    metric = LogitGateSystemOneModel(api_base="http://localhost:11434", backend="stub://")

    # 1. Line 226: choices[q_id] = ChoiceAnswer(..., probabilities=v.choice_probabilities or {}, ...)
    # If or -> and, probabilities becomes {}!
    v_choice = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        choice="A",
        choice_probabilities={"A": 0.85, "B": 0.15},
        triggered_layer="ChoiceGate",
        total_latency_ms=2.0,
    )
    with patch.object(ChoiceGate, "evaluate", return_value=v_choice):
        ans_c, _ = metric.decide(
            "candidate",
            {"qc": ChoiceQuestion(instructions="pick", options={"A": "1", "B": "2"})},
        )
        assert ans_c.choices["qc"].probabilities == {"A": 0.85, "B": 0.15}

    # 2. Line 266: len(binary_qids) > 1 batch path!
    # p = verdict.probability if verdict.probability is not None else ...
    # If is not -> is: p would be 1.0 instead of 0.7432!
    v_b1 = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.7432,
        triggered_layer="LogitGate",
        total_latency_ms=2.0,
    )
    v_b2 = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.6210,
        triggered_layer="LogitGate",
        total_latency_ms=2.0,
    )
    with patch.object(LogitGate, "evaluate_batch", return_value=[v_b1, v_b2]):
        ans, _ = metric.decide(
            "candidate",
            {
                "q1": NoulQuestion(instructions="check 1"),
                "q2": NoulQuestion(instructions="check 2"),
            },
        )
        assert ans.nouls["q1"].probability == 0.7432
        assert ans.nouls["q2"].probability == 0.6210


@pytest.mark.asyncio
async def test_deepeval_system_one_a_decide_mutants() -> None:
    """Kills lines 309 (and -> or), 312 (not x -> x), 339 (or -> and), 379 (is not -> is) in deepeval.py."""
    # Line 309: isinstance(backend, LogitBackend) and not isinstance(inner, StubBackend)
    # When backend is StubBackend, needs_precal should be False!
    # If line 309 mutated (and -> or), needs_precal would be True!
    metric_stub = LogitGateSystemOneModel(
        api_base="http://localhost:11434",
        backend=StubBackend(),
        model_name="custom_m",
        logit_temperature="auto",
    )
    with patch("logitgate.gates.logit_gate._lazy_calibrate_async") as mock_lazy:
        await metric_stub.a_decide("cand", {"q1": NoulQuestion(instructions="c")})
        assert not mock_lazy.called

    # Line 312: and not has_known_or_cached_temperature(self.model_name, binary=True)
    # When temperature is known (returns True), needs_precal must be False!
    # If mutated (not x -> x), needs_precal becomes True and calls _lazy_calibrate_async!
    real_mock_backend = MagicMock(spec=LogitBackend)
    real_mock_backend.inner = real_mock_backend
    metric_known_temp = LogitGateSystemOneModel(
        api_base="http://localhost:11434",
        backend=real_mock_backend,
        model_name="known_temp_model",
        logit_temperature="auto",
    )
    v_dummy = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.88,
        triggered_layer="LogitGate",
        total_latency_ms=1.0,
    )
    with (
        patch("logitgate.debiasing.permutation.has_known_or_cached_temperature", return_value=True),
        patch("logitgate.gates.logit_gate._lazy_calibrate_async") as mock_lazy2,
        patch.object(LogitGate, "aevaluate", return_value=v_dummy),
    ):
        await metric_known_temp.a_decide("cand", {"q1": NoulQuestion(instructions="c")})
        assert not mock_lazy2.called

    # Line 339: choices[q_id] = ChoiceAnswer(..., probabilities=v.choice_probabilities or {}, ...)
    # If or -> and: choice_probabilities is wiped to {}!
    v_choice = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        choice="A",
        choice_probabilities={"A": 0.85, "B": 0.15},
        triggered_layer="ChoiceGate",
        total_latency_ms=2.0,
    )
    with patch.object(ChoiceGate, "aevaluate", return_value=v_choice):
        metric = LogitGateSystemOneModel(api_base="http://localhost:11434", backend="stub://")
        ans_c, _ = await metric.a_decide(
            "candidate",
            {"qc": ChoiceQuestion(instructions="pick", options={"A": "1", "B": "2"})},
        )
        assert ans_c.choices["qc"].probabilities == {"A": 0.85, "B": 0.15}

    # Line 379: len(binary_qids) > 1 async batch path!
    v_b1 = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.7432,
        triggered_layer="LogitGate",
        total_latency_ms=2.0,
    )
    v_b2 = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.6210,
        triggered_layer="LogitGate",
        total_latency_ms=2.0,
    )
    with patch.object(LogitGate, "aevaluate_batch", return_value=[v_b1, v_b2]):
        ans_b, _ = await metric.a_decide(
            "candidate",
            {
                "q1": NoulQuestion(instructions="c1"),
                "q2": NoulQuestion(instructions="c2"),
            },
        )
        assert ans_b.nouls["q1"].probability == 0.7432
        assert ans_b.nouls["q2"].probability == 0.6210


# =============================================================================================
# 6. logitgate.integrations.pytest_plugin MUTANT KILLERS
# =============================================================================================


def _get_check_gate(fn: Any) -> Any:
    for c in fn.__closure__:
        fn_inner = c.cell_contents
        if hasattr(fn_inner, "__closure__") and fn_inner.__closure__:
            for c2 in fn_inner.__closure__:
                if isinstance(c2.cell_contents, dict) and c2.cell_contents:
                    return next(iter(c2.cell_contents.values()))
    return None


def test_pytest_plugin_check_ast_normalize_cloning_mutants() -> None:
    """Kills line 58 (and -> or, != -> ==) in pytest_plugin.py."""
    # 1. normalize=True and ast_check.normalize=False:
    # ast_check.normalize != normalize is True.
    # If line 58 mutated (!= -> ==), it wouldn't clone, keeping normalize=False!
    ast_gate_unnorm = ASTGate(syntax=True, normalize=False)
    dec1 = check("rubric", ast_check=ast_gate_unnorm, normalize=True, backend="stub://")
    fn1 = dec1(lambda: "code")
    fn1()
    g1 = _get_check_gate(fn1)
    assert g1 is not None
    assert g1.gates[0].normalize is True

    # 2. normalize=False and ast_check.normalize=True:
    # If line 58 mutated (and -> or), False or True != False is True -> it would clone with normalize=False!
    ast_gate_norm = ASTGate(syntax=True, normalize=True)
    dec2 = check("rubric", ast_check=ast_gate_norm, normalize=False, backend="stub://")
    fn2 = dec2(lambda: "code")
    fn2()
    g2 = _get_check_gate(fn2)
    assert g2 is not None
    assert g2.gates[0].normalize is True


def test_pytest_plugin_check_base_gate_normalize_mutants() -> None:
    """Kills line 76 (and -> or, not x -> x) in pytest_plugin.py."""
    base_gate = LogitGate(rubric="inner gate", backend="stub://")

    # 1. normalize=True and not getattr(ast_check, "normalize", False):
    # Inserts ASTGate(normalize=True) >> ast_check >> gate -> len is 3!
    # If mutated (not x -> x), len would be 2!
    dec1 = check("outer", ast_check=base_gate, normalize=True, backend="stub://")
    fn1 = dec1(lambda: "text")
    fn1()
    g1 = _get_check_gate(fn1)
    assert g1 is not None
    assert len(g1.gates) == 3
    assert isinstance(g1.gates[0], ASTGate)

    # 2. normalize=False:
    # If mutated (and -> or), False or not False is True -> len would be 3!
    dec2 = check("outer", ast_check=base_gate, normalize=False, backend="stub://")
    fn2 = dec2(lambda: "text")
    fn2()
    g2 = _get_check_gate(fn2)
    assert g2 is not None
    assert len(g2.gates) == 2


def test_pytest_plugin_raise_if_failed_none_probability_mutant() -> None:
    """Kills line 93 (is not -> is) in pytest_plugin._raise_if_failed."""
    v_none_prob = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=None,
        triggered_layer="ASTGate",
        total_latency_ms=1.0,
        reason="syntax error",
    )

    fn = check("syntax valid")(lambda: "bad code")
    with patch.object(LogitGate, "evaluate", return_value=v_none_prob):
        with pytest.raises(AssertionError) as exc_info:
            fn()
        assert "P=n/a" in str(exc_info.value)
        assert "syntax error" in str(exc_info.value)

    v_prob = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.1234,
        triggered_layer="LogitGate",
        total_latency_ms=2.0,
        reason="low confidence",
    )
    with patch.object(LogitGate, "evaluate", return_value=v_prob):
        with pytest.raises(AssertionError) as exc_info2:
            fn()
        assert "P=0.1234" in str(exc_info2.value)


def test_pytest_plugin_cache_key_invalidation_mutants() -> None:
    """Kills lines 120, 121, 122 (is -> is not) in @check decorator."""
    v_ok = Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="LogitGate", total_latency_ms=1.0)

    # 1. Line 120: backend is None
    # When LogitGateDefaults.backend changes, a new gate must be created!
    fn_backend = check("rubric", backend=None, model="default", threshold=0.85)(lambda: "output")
    created_backends: list[Any] = []
    orig_init = LogitGate.__init__

    def capture_init_b(self: Any, *a: Any, **kw: Any) -> None:
        created_backends.append(kw.get("backend"))
        orig_init(self, *a, **kw)

    with (
        patch.object(LogitGate, "evaluate", return_value=v_ok),
        patch.object(LogitGate, "__init__", capture_init_b),
    ):
        LogitGateDefaults.backend = "stub://first"
        fn_backend()
        LogitGateDefaults.backend = "stub://second"
        fn_backend()
        # If line 120 mutated (is -> is not), cfg_key didn't change so second call reused first gate!
        assert len(created_backends) == 2
        LogitGateDefaults.backend = None

    # 2. Line 121: model is None
    fn_model = check("rubric", backend="stub://", model=None, threshold=0.85)(lambda: "output")
    created_models: list[Any] = []

    def capture_init_m(self: Any, *a: Any, **kw: Any) -> None:
        created_models.append(kw.get("model"))
        orig_init(self, *a, **kw)

    with (
        patch.object(LogitGate, "evaluate", return_value=v_ok),
        patch.object(LogitGate, "__init__", capture_init_m),
    ):
        LogitGateDefaults.model = "model_alpha"
        fn_model()
        LogitGateDefaults.model = "model_beta"
        fn_model()
        assert len(created_models) == 2
        LogitGateDefaults.model = None

    # 3. Line 122: threshold is None
    fn_thresh = check("rubric", backend="stub://", model="default", threshold=None)(lambda: "output")
    created_thresh: list[Any] = []

    def capture_init_t(self: Any, *a: Any, **kw: Any) -> None:
        created_thresh.append(kw.get("pass_threshold"))
        orig_init(self, *a, **kw)

    with (
        patch.object(LogitGate, "evaluate", return_value=v_ok),
        patch.object(LogitGate, "__init__", capture_init_t),
    ):
        LogitGateDefaults.pass_threshold = 0.81
        fn_thresh()
        LogitGateDefaults.pass_threshold = 0.89
        fn_thresh()
        assert len(created_thresh) == 2
        LogitGateDefaults.pass_threshold = None


# =============================================================================================
# 7. logitgate.integrations.langgraph MUTANT KILLERS
# =============================================================================================


def test_langgraph_extract_private_attribute_mutant() -> None:
    """Kills line 24 (not x -> x) in langgraph.py."""
    class StateWithPrivate:
        public_state_key = "valid candidate"
        _hidden_key = "secret"

    # When state_key is missing, available keys must contain 'public_state_key' and NOT '_hidden_key'
    with pytest.raises(KeyError) as exc_info:
        _extract(StateWithPrivate(), "missing_key")

    err = str(exc_info.value)
    assert "'public_state_key'" in err
    assert "'_hidden_key'" not in err
    assert "'__class__'" not in err


def test_langgraph_as_conditional_edge_ambiguous_and_mutant() -> None:
    """Kills line 44 (and -> or) in langgraph.py."""
    # Line 44: if ambiguous_node is not None and verdict.status == DecisionStatus.AMBIGUOUS:
    # If mutated (and -> or), when ambiguous_node is not None, it returns ambiguous_node
    # even when verdict.status is FAILED!
    router = _router(
        pass_node=None,
        fail_node="fail_handler",
        ambiguous_node="ambiguous_handler",
        node_mapping={"A": "node_a", "B": "node_b"},
    )

    v_failed_unmapped = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        choice="UNKNOWN",
        triggered_layer="ChoiceGate",
        total_latency_ms=2.0,
    )
    # Must return fail_handler, NOT ambiguous_handler!
    assert router(v_failed_unmapped) == "fail_handler"

    v_ambig = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        choice="UNKNOWN",
        triggered_layer="ChoiceGate",
        total_latency_ms=2.0,
    )
    assert router(v_ambig) == "ambiguous_handler"


# =============================================================================================
# 8. logitgate.telemetry.models MUTANT KILLERS
# =============================================================================================


def test_telemetry_models_luhn_and_privacy_mutants() -> None:
    """Kills lines 43 (or -> and, > -> >=), 51 (> -> >=), 148, 150, 160, 161 in telemetry/models.py."""
    # Line 43: len(digits) < 13 or len(digits) > 19 or len(set(digits)) == 1
    # 1. Invalid size or all same digits:
    assert _luhn_valid("0" * 16) is False
    assert _luhn_valid("0" * 10) is False
    assert _luhn_valid("0" * 20) is False
    assert _luhn_valid("4000000000000") is False

    # 2. Exactly 19 digits valid PAN: line 43 (> -> >=)
    # If mutated to >= 19, 19 digits returns False!
    assert _luhn_valid("4532123456789012345") is True

    # Line 148 & 150: with_privacy(privacy="full", ...)
    rec = TelemetryRecord(
        timestamp="2026-11-16T12:00:00Z",
        gate_name="test_gate",
        gate_type="LogitGate",
        candidate_sha256="abc",
        candidate_length=16,
        passed=True,
        status=DecisionStatus.PASSED,
        total_latency_ms=5.0,
        candidate="exact_ten_chars!",  # 16 chars
    )
    # Line 150: len(candidate) <= max_candidate_chars (mutant <= -> <)
    # When len == max_chars:
    assert rec.with_privacy(privacy="full", max_candidate_chars=16) is rec

    # Line 148: max_candidate_chars is None or candidate is None
    assert rec.with_privacy(privacy="full", max_candidate_chars=None) is rec
    rec_none_cand = TelemetryRecord(
        timestamp="2026-11-16T12:00:00Z",
        gate_name="test_gate",
        gate_type="LogitGate",
        candidate_sha256="abc",
        candidate_length=0,
        passed=True,
        status=DecisionStatus.PASSED,
        total_latency_ms=5.0,
        candidate=None,
    )
    assert rec_none_cand.with_privacy(privacy="full", max_candidate_chars=10).candidate is None

    # Line 160: has_generative_critique = self.gate_type == "JudgeGate" or ...
    # When gate_type == "JudgeGate", in omit_candidate mode, reason is scrubbed to None!
    # If line 160 mutated (or -> and), reason would NOT be scrubbed!
    rec_judge = TelemetryRecord(
        timestamp="2026-11-16T12:00:00Z",
        gate_name="judge",
        gate_type="JudgeGate",
        candidate_sha256="abc",
        candidate_length=10,
        passed=False,
        status=DecisionStatus.FAILED,
        total_latency_ms=5.0,
        candidate="text",
        reason="critique of candidate",
        traces=[],
    )
    assert rec_judge.with_privacy(privacy="omit_candidate", max_candidate_chars=None).reason is None

    # Line 161: tr.diagnostic_report is not None or tr.tokens_generated > 0
    # 1. When diagnostic_report is present and tokens_generated == 0:
    # If line 161 mutated (or -> and), True and False is False -> reason would NOT be scrubbed!
    rec_report_only = TelemetryRecord(
        timestamp="2026-11-16T12:00:00Z",
        gate_name="logit",
        gate_type="LogitGate",
        candidate_sha256="abc",
        candidate_length=10,
        passed=False,
        status=DecisionStatus.FAILED,
        total_latency_ms=5.0,
        candidate="text",
        reason="static violation",
        traces=[
            LayerTrace(
                layer_name="ASTGate",
                status=DecisionStatus.FAILED,
                latency_ms=1.0,
                tokens_generated=0,
                diagnostic_report="syntax error",
            )
        ],
    )
    assert rec_report_only.with_privacy(privacy="omit_candidate", max_candidate_chars=None).reason is None

    # 2. When tokens_generated == 0 and diagnostic_report is None: line 161 (> -> >=)
    # If mutated to >= 0, 0 >= 0 is True -> reason would be incorrectly scrubbed to None!
    rec_no_gen = TelemetryRecord(
        timestamp="2026-11-16T12:00:00Z",
        gate_name="logit",
        gate_type="LogitGate",
        candidate_sha256="abc",
        candidate_length=10,
        passed=False,
        status=DecisionStatus.FAILED,
        total_latency_ms=5.0,
        candidate="text",
        reason="unscrubbed reason",
        traces=[
            LayerTrace(
                layer_name="ASTGate",
                status=DecisionStatus.FAILED,
                latency_ms=1.0,
                tokens_generated=0,
                diagnostic_report=None,
            )
        ],
    )
    assert rec_no_gen.with_privacy(privacy="omit_candidate", max_candidate_chars=None).reason == "unscrubbed reason"
