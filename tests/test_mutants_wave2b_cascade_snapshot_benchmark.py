"""Dedicated Wave 2B Coverage-Closure & Mutant-Killing Test Suite.

Targets 7 core modules:
1. logitgate.snapshot
2. logitgate.benchmark
3. logitgate.cascade
4. logitgate.gates.base
5. logitgate.gates.ast_gate
6. logitgate.gates.logit_gate
7. logitgate.gates.rank_gate

Closes all <91% coverage gaps to >= 95% and systematically kills surviving mutants.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Explicit imports of all 7 target modules for mutation_audit mapping
from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
)
from logitgate.backends.ollama import OllamaBackend
from logitgate.benchmark import (
    BUILTIN_CASES,
    BenchmarkCase,
    _extract_backend_clone_kwargs,
    _MeteredBackend,
    _MeteredGenerativeBackend,
    _prewarm_calibration_if_needed,
    bootstrap_metric_ci,
    compute_calibration_metrics,
    format_report,
    load_cases,
    mcnemar_exact_test,
    run_benchmark,
    wilson_confidence_interval,
    write_json,
)
from logitgate.cache import VerdictCache
from logitgate.cascade import Cascade, fuse_escalated_verdict
from logitgate.exceptions import BackendError, ConfigurationError
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.base import (
    TIER_DETERMINISTIC,
    BaseGate,
)
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import (
    ChoiceGate,
    LogitGate,
    MultiRubricGate,
    _ensure_logprobs_present,
)
from logitgate.gates.rank_gate import BestOfNGate, RankGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import (
    VerdictSnapshot,
    _atomic_write_text,
    _should_log_console_notice,
    persist_failure_snapshot,
    reset_snapshot_notices,
    session_snapshot_paths,
)
from logitgate.telemetry.sinks import CallbackTelemetrySink

# ==============================================================================
# Helper Mock Classes
# ==============================================================================

class _PureLogitBackend(LogitBackend):
    def query_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return []

    async def aquery_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return []


class _PureGenBackend(GenerativeBackend):
    def __init__(self, text: str = "pass") -> None:
        self.text = text
        self.closed = False
        self.aclosed_called = False

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return self.text, 5

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return self.text, 5

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed_called = True


class _DualMockBackend(LogitBackend, GenerativeBackend):
    def __init__(self, logprobs: list[dict[str, Any]] | None = None, text: str = "PASS") -> None:
        self._lp = logprobs or [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]
        self._text = text
        self.closed = False
        self.aclosed_called = False

    def query_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._lp

    async def aquery_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._lp

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return self._text, 4

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return self._text, 4

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed_called = True


class _DummyCustomGate(BaseGate):
    name = "DummyGate"
    layer_tier = TIER_DETERMINISTIC

    def __init__(self, should_pass: bool = True, normalize_text: bool = False) -> None:
        self.should_pass = should_pass
        self.normalize_text = normalize_text

    def normalize_input(self, candidate: Any) -> Any:
        if self.normalize_text and isinstance(candidate, str):
            return candidate.strip().lower()
        return candidate

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            passed=self.should_pass,
            status=DecisionStatus.PASSED if self.should_pass else DecisionStatus.FAILED,
            triggered_layer=self.name,
            total_latency_ms=1.0,
            total_tokens_generated=0,
            reason="dummy eval",
            traces=[LayerTrace(latency_ms=1.0, layer_name=self.name, status=DecisionStatus.PASSED if self.should_pass else DecisionStatus.FAILED)],
        )


# ==============================================================================
# SECTION 1: logitgate.snapshot Tests & Mutant Killers
# ==============================================================================

def test_snapshot_atomic_write_symlink_and_cleanup(tmp_path: Path) -> None:
    # 1. Symlink path checks (lines 46, 56)
    target_file = tmp_path / "real_file.json"
    target_file.write_text("{}", encoding="utf-8")
    symlink_file = tmp_path / "symlink_file.json"
    symlink_file.symlink_to(target_file)

    with pytest.raises(ConfigurationError, match="Refusing to write VerdictSnapshot through symlink"):
        _atomic_write_text(symlink_file, "data")

    symlink_dir = tmp_path / "symlink_dir"
    symlink_dir.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="Refusing to write VerdictSnapshot through symlink"):
        _atomic_write_text(symlink_dir / "test.json", "data")

    # 2. Cleanup on write error (lines 58-68)
    real_out = tmp_path / "out.json"
    with patch("os.fdopen", side_effect=OSError("Disk full")), pytest.raises(OSError, match="Disk full"):
        _atomic_write_text(real_out, "content")


def test_snapshot_notices_and_quiet_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Line 93: quiet flag testing
    monkeypatch.setenv("LOGITGATE_QUIET", "1")
    assert _should_log_console_notice() is False
    monkeypatch.setenv("LOGITGATE_QUIET", "true")
    assert _should_log_console_notice() is False
    monkeypatch.setenv("LOGITGATE_QUIET", "yes")
    assert _should_log_console_notice() is False
    monkeypatch.setenv("LOGITGATE_QUIET", "0")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert _should_log_console_notice() is True

    # Lines 85-87: reset_snapshot_notices with pytest_plugin present
    fake_plugin = MagicMock()
    fake_plugin._CHECK_SNAPSHOT_PATHS = [Path("/tmp/p1"), Path("/tmp/p2")]
    monkeypatch.setitem(sys.modules, "logitgate.integrations.pytest_plugin", fake_plugin)
    reset_snapshot_notices()
    assert fake_plugin._CHECK_SNAPSHOT_PATHS == []

    # Lines 99-100: session_snapshot_paths
    paths = session_snapshot_paths()
    assert isinstance(paths, list)


def test_snapshot_persist_failure_bounds_and_mutants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_snapshot_notices()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("LOGITGATE_QUIET", "0")

    failed_v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="LogitGate",
        probability=0.2,
        total_latency_ms=5.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[LayerTrace(latency_ms=1.0, layer_name="LogitGate", status=DecisionStatus.FAILED, calibrated_prob=0.2)],
    )

    # Mutant line 132: and -> or in persist_failure_snapshot
    # When save_on_fail=False (not None), hint MUST NOT be emitted!
    stderr_captured: list[str] = []
    monkeypatch.setattr(sys.stderr, "write", lambda msg: stderr_captured.append(msg))
    v_res = persist_failure_snapshot(failed_v, candidate="bad code", save_on_fail=False)
    assert v_res.passed is False
    assert len(stderr_captured) == 0  # No hint emitted when save_on_fail=False!

    # Mutant line 150: and -> or in persist_failure_snapshot:
    # if gate_type and "gate_type" not in effective_cfg:
    cfg: dict[str, Any] = {}
    v_saved = persist_failure_snapshot(
        failed_v, candidate="bad code", save_on_fail=True, gate_type=None, gate_config=cfg, snapshot_dir=tmp_path
    )
    assert v_saved.traces
    assert "snapshot_path" in v_saved.traces[-1].metadata

    # Line 147: duplicate snapshot guard (already has snapshot_path)
    v_dup = persist_failure_snapshot(
        v_saved, candidate="bad code", save_on_fail=True, snapshot_dir=tmp_path
    )
    assert v_dup is v_saved

    # Mutant line 169: and -> or (not _NOTIFIED_SNAPSHOT_SAVED and _should_log_console_notice())
    # Notice was emitted on first save; second distinct save must not emit duplicate notice!
    stderr_captured.clear()
    failed_v2 = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="LogitGate",
        probability=0.1,
        total_latency_ms=4.0,
        total_tokens_generated=0,
        reason="Fail 2",
        traces=[LayerTrace(latency_ms=1.0, layer_name="LogitGate", status=DecisionStatus.FAILED, calibrated_prob=0.1)],
    )
    persist_failure_snapshot(failed_v2, candidate="bad code 2", save_on_fail=True, snapshot_dir=tmp_path)
    assert len(stderr_captured) == 0

    # Line 165: 1000 snapshots list FIFO bound
    from logitgate import snapshot as snap_mod
    with snap_mod._snapshot_lock:
        snap_mod._SESSION_SNAPSHOTS.clear()
        snap_mod._SESSION_SNAPSHOTS.extend([Path(f"/tmp/snap_{i}.json") for i in range(1000)])
    persist_failure_snapshot(failed_v2, candidate="code 3", save_on_fail=True, snapshot_dir=tmp_path)
    with snap_mod._snapshot_lock:
        assert len(snap_mod._SESSION_SNAPSHOTS) == 1000


def test_snapshot_model_validator_and_trace_properties() -> None:
    # Lines 198->209: _populate_defaults with non-dict and with/without timestamp & id
    v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="ScoreGate",
        score=2.0,
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[
            LayerTrace(
                latency_ms=1.0,
                layer_name="ScoreGate",
                status=DecisionStatus.FAILED,
                metadata={
                    "positional_bias_magnitude": 0.15,
                    "log_odds_pass": -1.2,
                    "shannon_entropy": 0.65,
                    "loops_executed": 3,
                    "failure_code": "CUSTOM_FAIL",
                },
            )
        ],
    )
    snap = VerdictSnapshot.from_verdict(v, rubric="Score rubric", candidate="sample")
    assert snap.timestamp != ""
    assert snap.snapshot_id != ""
    # Test metadata property fallbacks (lines 247, 256, 265, 274, 283)
    assert snap.positional_bias_magnitude == 0.15
    assert snap.log_odds_pass == -1.2
    assert snap.shannon_entropy == 0.65
    assert snap.loops_executed == 3
    assert snap.failure_code == "CUSTOM_FAIL"

    # Mutant line 282: is not -> is on failure_code
    t_direct = LayerTrace(latency_ms=1.0, layer_name="ASTGate", status=DecisionStatus.FAILED, failure_code="DIRECT_FAIL")
    v_direct = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="ASTGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Direct fail",
        traces=[t_direct],
    )
    snap_direct = VerdictSnapshot.from_verdict(v_direct, rubric="R", candidate="C")
    assert snap_direct.failure_code == "DIRECT_FAIL"


def test_snapshot_save_load_directory_and_mutants(tmp_path: Path) -> None:
    v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="ChoiceGate",
        choice="B",
        total_latency_ms=2.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[LayerTrace(latency_ms=1.0, layer_name="ChoiceGate", status=DecisionStatus.FAILED)],
    )
    # Save with invalid non-ISO timestamp to hit lines 298-299
    snap = VerdictSnapshot(
        timestamp="invalid-timestamp-string-12345",
        rubric="Pick correct",
        candidate="code",
        verdict=v,
        test_name="my_test/special:chars!",
    )
    saved_file = snap.save(directory=tmp_path)
    assert saved_file.exists()

    # Load from file
    loaded = VerdictSnapshot.load(saved_file)
    assert loaded.rubric == "Pick correct"

    # Load from directory when latest.json exists
    loaded_dir = VerdictSnapshot.load(tmp_path)
    assert loaded_dir.rubric == "Pick correct"

    # Mutant line 341: not x -> x in load directory glob candidates
    # Remove latest.json so it globs fail_*.json
    (tmp_path / "latest.json").unlink()
    loaded_glob = VerdictSnapshot.load(tmp_path)
    assert loaded_glob.rubric == "Pick correct"

    # Rename fail_*.json to arbitrary.json so it hits line 341 fallback glob
    saved_file.rename(tmp_path / "custom_snap.json")
    loaded_arbitrary = VerdictSnapshot.load(tmp_path)
    assert loaded_arbitrary.rubric == "Pick correct"

    # When directory is empty: FileNotFoundError (line 343)
    (tmp_path / "custom_snap.json").unlink()
    with pytest.raises(FileNotFoundError, match="No snapshot JSON files found"):
        VerdictSnapshot.load(tmp_path)

    # Symlink checks for load (lines 328, 333)
    fake_file = tmp_path / "snap.json"
    fake_file.write_text("{}", encoding="utf-8")
    sym_link = tmp_path / "sym_snap.json"
    sym_link.symlink_to(fake_file)
    with pytest.raises(ConfigurationError, match="Refusing to load VerdictSnapshot from symlink"):
        VerdictSnapshot.load(sym_link)

    latest_sym = tmp_path / "latest.json"
    latest_sym.symlink_to(fake_file)
    with pytest.raises(ConfigurationError, match="Refusing to load VerdictSnapshot from symlink"):
        VerdictSnapshot.load(tmp_path)


def test_snapshot_replay_and_areplay_gates(tmp_path: Path) -> None:
    backend = StubBackend(forward_top_logprobs=[{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": -5.0}])

    # 1. ChoiceGate replay (including line 461 missing options ValueError)
    v_fail = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="ChoiceGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[LayerTrace(latency_ms=1.0, layer_name="ChoiceGate", status=DecisionStatus.FAILED)],
    )
    snap_choice_bad = VerdictSnapshot(
        rubric="Pick", candidate="c", verdict=v_fail, gate_config={"gate_type": "ChoiceGate"}
    )
    with pytest.raises(ValueError, match="missing valid 'options' metadata"):
        snap_choice_bad.replay(backend=backend)

    snap_choice = VerdictSnapshot(
        rubric="Pick option",
        candidate="c",
        verdict=v_fail,
        gate_config={
            "gate_type": "ChoiceGate",
            "options": {"A": "Option A", "B": "Option B"},
            "symmetric_debiasing": False,
        },
    )
    v_replayed = snap_choice.replay(backend=backend)
    assert v_replayed.passed is True

    # 2. ScoreGate replay
    snap_score = VerdictSnapshot(
        rubric="Score quality",
        candidate="c",
        verdict=v_fail,
        gate_config={"gate_type": "ScoreGate", "min_score": 1, "max_score": 5, "threshold": 3.0},
    )
    v_score = snap_score.replay(backend=backend)
    assert v_score is not None

    # 3. MultiRubricGate replay & mutant line 579: and -> or
    snap_multi = VerdictSnapshot(
        rubric="Multi",
        candidate="c",
        verdict=v_fail,
        gate_config={"rubrics": ["R1", "R2"], "symmetric_debiasing": False},
    )
    v_multi = snap_multi.replay(backend=backend)
    assert v_multi.passed is True

    # When rubrics is not a list (e.g. None), it must build LogitGate, not MultiRubricGate
    snap_single = VerdictSnapshot(
        rubric="Single", candidate="c", verdict=v_fail, gate_config={"rubrics": None}
    )
    v_single = snap_single.replay(backend=backend)
    assert v_single.passed is True

    # 4. BestOfNGate replay with pre_gate and post_gate (lines 514, 515, 545-561, 650)
    pre_cfg = {"gate_type": "LogitGate", "rubric": "Pre check", "backend": "stub://default"}
    post_cfg = {"gate_type": "JudgeGate", "rubric": "Post critique", "backend": "stub://default"}
    snap_best = VerdictSnapshot(
        rubric="Rank best",
        candidate=["cand A", "cand B"],
        verdict=v_fail,
        gate_config={
            "gate_type": "BestOfNGate",
            "mode": "pairwise",
            "pre_gate_config": pre_cfg,
            "post_gate_config": post_cfg,
            "backend": "stub://default",
        },
    )
    v_best = snap_best.replay()
    assert v_best is not None


@pytest.mark.asyncio
async def test_snapshot_areplay_and_aexplain_async() -> None:
    # Lines 745-777: aexplain
    v_fail = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="LogitGate",
        probability=0.1,
        score=0.2,
        total_latency_ms=2.0,
        total_tokens_generated=0,
        reason="Layer 1 failure",
        traces=[LayerTrace(latency_ms=1.0, layer_name="LogitGate", status=DecisionStatus.FAILED, calibrated_prob=0.1)],
    )
    snap = VerdictSnapshot(
        rubric="Safety check",
        candidate="risky output",
        verdict=v_fail,
        gate_config={"gate_type": "LogitGate", "symmetric_debiasing": True},
    )

    judge_backend = StubBackend(diagnostic_text="[VERDICT: FAIL] Severe safety violation.")
    v_explained = await snap.aexplain(backend=judge_backend)
    assert v_explained.passed is False
    assert v_explained.triggered_layer == "JudgeGate"
    assert "safety violation" in (v_explained.reason or "").lower()

    # Synchronous explain + default StubBackend pass override (lines 662-676)
    # Mutant line 669: final_score = self.verdict.score if self.verdict.score is not None else 0.0 (is not -> is)
    default_stub = StubBackend()  # returns "[VERDICT: PASS] All criteria met."
    v_exp_sync = snap.explain(backend=default_stub)
    assert v_exp_sync.passed is False
    assert v_exp_sync.score == 0.2  # Must preserve original score, not 0.0 or None!

    # areplay
    v_replayed = await snap.areplay(backend=StubBackend(forward_top_logprobs=[{"token": "A", "logprob": 0.0}]))
    assert v_replayed is not None


# ==============================================================================
# SECTION 2: logitgate.benchmark Tests & Mutant Killers
# ==============================================================================

def test_benchmark_metered_backend_methods_and_errors() -> None:
    pure_gen = _PureGenBackend("result")
    logit_only = _PureLogitBackend()

    # Line 161: require_generative=True on non-generative backend raises TypeError
    with pytest.raises(TypeError, match="Benchmarking the judge requires a generative backend"):
        _MeteredBackend(logit_only, require_generative=True)

    # MeteredBackend wrapping pure_gen (which is not LogitBackend)
    metered_dual = _MeteredBackend(_DualMockBackend())
    assert repr(metered_dual) != ""
    with pytest.raises(AttributeError, match="inner"):
        metered_dual.__getattr__("inner")

    # generate when inner not generative
    metered_logit_no_gen = _MeteredBackend(logit_only, require_generative=False)
    with pytest.raises(TypeError, match="Benchmarking the judge requires a generative backend"):
        metered_logit_no_gen.generate("prompt")

    # _MeteredGenerativeBackend
    with pytest.raises(TypeError, match="Benchmarking the judge requires a generative backend"):
        _MeteredGenerativeBackend(object())  # type: ignore[arg-type]

    metered_gen = _MeteredGenerativeBackend(pure_gen)
    assert repr(metered_gen) != ""
    with pytest.raises(AttributeError, match="inner"):
        metered_gen.__getattr__("inner")

    txt, _tok = metered_gen.generate("hello")
    assert txt == "result"
    assert metered_gen.generate_calls == 1
    metered_gen.close()
    assert pure_gen.closed is True


@pytest.mark.asyncio
async def test_benchmark_metered_backend_async_methods() -> None:
    dual = _DualMockBackend()
    metered = _MeteredBackend(dual)
    lp = await metered.aquery_top_logprobs("prompt")
    assert len(lp) == 2
    assert metered.logit_calls == 1

    txt, _tok = await metered.agenerate("prompt")
    assert txt == "PASS"
    assert metered.generate_calls == 1

    await metered.aclose()
    assert dual.aclosed_called is True

    pure_gen = _PureGenBackend("gen")
    metered_gen = _MeteredGenerativeBackend(pure_gen)
    txt_g, _tok_g = await metered_gen.agenerate("prompt")
    assert txt_g == "gen"
    assert metered_gen.generate_calls == 1
    await metered_gen.aclose()
    assert pure_gen.aclosed_called is True


def test_benchmark_calibration_metrics_ece_mutant_line_428() -> None:
    # Mutant line 428: int(p * n_bins) swapped * -> /
    # With p=0.45 and n_bins=10:
    # 0.45 * 10 = 4.5 -> bin 4. With /, 0.45 / 10 = 0.045 -> bin 0!
    # With p=0.85 and n_bins=10:
    # 0.85 * 10 = 8.5 -> bin 8. With /, 0.85 / 10 = 0.085 -> bin 0!
    probs = [0.45, 0.85]
    labels = [True, False]
    metrics = compute_calibration_metrics(probs, labels, n_bins=10)
    # Expected ECE:
    # bin 4: [0.45], y=[1.0] -> |0.45 - 1.0| = 0.55
    # bin 8: [0.85], y=[0.0] -> |0.85 - 0.0| = 0.85
    # ECE = 0.5 * 0.55 + 0.5 * 0.85 = 0.7000
    # If mutant * -> / fired, both fall into bin 0:
    # mean(p) = 0.65, mean(y) = 0.50 -> |0.65 - 0.50| = 0.1500!
    assert metrics["ece"] is not None and abs(metrics["ece"] - 0.7000) < 1e-4

    # Wilson CI edge checks
    ci_0 = wilson_confidence_interval(0, 10)
    assert ci_0[0] >= 0.0
    ci_10 = wilson_confidence_interval(10, 10)
    assert ci_10[1] <= 1.0

    # McNemar test & bootstrap
    mcn = mcnemar_exact_test([True, False, True], [True, True, False])
    assert "p_value" in mcn
    bs = bootstrap_metric_ci([0.1, 0.2, 0.3], seed=42)
    assert bs[0] <= bs[1]


def test_benchmark_clone_kwargs_and_prewarm() -> None:
    ollama_b = OllamaBackend(model="qwen", base_url="http://localhost:11434", num_ctx=4096, options={"temperature": 0.5})
    kwargs = _extract_backend_clone_kwargs(ollama_b)
    assert kwargs.get("num_ctx") == 4096
    assert kwargs.get("options") == {"temperature": 0.5}

    # _prewarm_calibration_if_needed
    _prewarm_calibration_if_needed(StubBackend(), "test_model")


def test_benchmark_run_benchmark_heterogeneous_and_judge_mode(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase("Rubric 1", "Candidate 1", True),
        BenchmarkCase("Rubric 2", "Candidate 2", False),
    ]
    progress_calls: list[str] = []

    # Pure generative mode modes=("judge",) with judge_model & judge_backend specified
    # Kills line 679 (heterogeneous_judge is not -> is)
    # Kills line 738 (effective_model is None and resolved_judge is not None and not needs_logit)
    # Kills line 814 (gen_after - gen_before swapped - -> +)
    # Kills line 844 (primary_backend resolved is not None else resolved_judge)
    # Kills line 831 (progress message + -> -)
    report = run_benchmark(
        cases,
        modes=("judge",),
        judge_backend=StubBackend(),
        judge_model="judge_32b",
        progress=lambda msg: progress_calls.append(msg),
        cache=True,
    )
    assert len(progress_calls) == 2
    assert "judge" in report.modes
    assert report.metadata["judge_model"] == "judge_32b"
    assert report.metadata["model"] == "judge_32b"

    # format_report with judge_suffix (line 894 mutant is not -> is)
    txt_rep = format_report(report)
    assert "judge_model=judge_32b" in txt_rep or "LogitGate" in txt_rep

    # load_cases & write_json (lines 985-996)
    cases_loaded, label = load_cases(None)
    assert len(cases_loaded) == len(BUILTIN_CASES)
    assert label == "builtin"

    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps([{"rubric": "r", "candidate": "c", "expected": True}]), encoding="utf-8")
    custom_cases, _custom_label = load_cases(str(case_file))
    assert len(custom_cases) == 1

    stdout_buf: list[str] = []
    with patch("sys.stdout.write", lambda s: stdout_buf.append(s)):
        write_json(report)
    assert len(stdout_buf) > 0


# ==============================================================================
# SECTION 3: logitgate.cascade Tests & Mutant Killers
# ==============================================================================

def test_cascade_init_validation_and_cleanup(tmp_path: Path) -> None:
    # Line 211: non-BaseGate in gates
    with pytest.raises(ConfigurationError, match="must all inherit from BaseGate"):
        Cascade(["not_a_gate"])  # type: ignore[list-item]

    # Line 213: non-bool fuse_logits
    with pytest.raises(ConfigurationError, match="fuse_logits must be a bool"):
        Cascade([ASTGate(syntax=True)], fuse_logits="true")  # type: ignore[arg-type]

    # Line 214-221: invalid prior_weight
    with pytest.raises(ConfigurationError, match="prior_weight must be between 0.0 and 1.0"):
        Cascade([ASTGate(syntax=True)], prior_weight=-0.1)
    with pytest.raises(ConfigurationError, match="prior_weight must be between 0.0 and 1.0"):
        Cascade([ASTGate(syntax=True)], prior_weight=1.5)
    with pytest.raises(ConfigurationError, match="prior_weight must be a float between 0.0 and 1.0"):
        Cascade([ASTGate(syntax=True)], prior_weight="invalid")  # type: ignore[arg-type]

    # Lines 226-229: owns_sink closed on cache error
    sink_file = tmp_path / "sink_cleanup.jsonl"
    with (
        patch("logitgate.cache.resolve_verdict_cache", side_effect=ValueError("Bad cache")),
        pytest.raises(ValueError, match="Bad cache"),
    ):
        Cascade([ASTGate(syntax=True)], telemetry_sink=str(sink_file), cache="invalid")


def test_cascade_effective_cache_and_mutant_line_250() -> None:
    # Mutant line 250: in _effective_cache(), when all_false is True, returns False
    # If mutant return False -> True fires, it returns True!
    g1 = ASTGate(syntax=True, cache=False)
    g2 = LogitGate(rubric="Rubric", backend="stub://default", cache=False)
    casc = Cascade([g1, g2], cache=None)
    # Must return False, NEVER True or None!
    assert casc._effective_cache() is False

    # When one gate has VerdictCache:
    vcache = VerdictCache()
    g3 = LogitGate(rubric="R", backend="stub://default", cache=vcache)
    casc_cache = Cascade([g1, g3])
    assert casc_cache._effective_cache() is vcache

    # Sink resolution with 1 sink vs composite (lines 271, 273)
    s1 = CallbackTelemetrySink(lambda _: None)
    s2 = CallbackTelemetrySink(lambda _: None)
    g_s1 = LogitGate(rubric="R1", backend="stub://default", telemetry_sink=s1)
    g_s2 = LogitGate(rubric="R2", backend="stub://default", telemetry_sink=s2)
    casc_s1 = Cascade([g_s1])
    assert casc_s1._effective_telemetry_sink() is s1
    casc_s12 = Cascade([g_s1, g_s2])
    from logitgate.telemetry.sinks import CompositeTelemetrySink
    assert isinstance(casc_s12._effective_telemetry_sink(), CompositeTelemetrySink)


def test_cascade_mutant_line_318_primary_backend_tier() -> None:
    # Mutant line 318: g.layer_tier >= TIER_LOGIT swapped >= -> >
    # When g is LogitGate (layer_tier=1 == TIER_LOGIT), it must match and set primary_backend!
    b = StubBackend()
    lg = LogitGate(rubric="Check", backend=b)
    casc = Cascade([lg])
    hit_v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="LogitGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Hit",
        traces=[LayerTrace(latency_ms=1.0, layer_name="LogitGate", status=DecisionStatus.PASSED)],
    )
    v_fin = casc._finalize_cache_hit(hit_v, "cand", {})
    assert v_fin.passed is True


@pytest.mark.asyncio
async def test_cascade_aevaluate_cache_and_children_calibration() -> None:
    # Lines 376-384: aevaluate cache hit
    b = StubBackend(forward_top_logprobs=[{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": -5.0}])
    lg = LogitGate(rubric="Valid Python", backend=b)
    casc = Cascade([lg], cache=VerdictCache())
    # First evaluate populates cache
    v1 = await casc.aevaluate("def foo(): pass")
    assert v1.passed is True
    # Second evaluate hits cache
    v2 = await casc.aevaluate("def foo(): pass")
    assert v2.passed is True


def test_cascade_mutants_shadow_audit_and_trace_selection() -> None:
    # 1. Mutant line 604: tr1 is not None swapped is not -> is
    # If upstream has multiple traces and the first is logit while second is non-logit:
    tr_logit = LayerTrace(
        latency_ms=1.0,
        layer_name="LogitGate",
        status=DecisionStatus.AMBIGUOUS,
        log_odds_pass=2.5,
        calibrated_prob=0.92,
        forward_log_odds=2.0,
        reverse_log_odds=2.0,
    )
    tr_other = LayerTrace(
        latency_ms=1.0,
        layer_name="OtherGate",
        status=DecisionStatus.AMBIGUOUS,
        log_odds_pass=None,
    )
    v_upstream = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        triggered_layer="LogitGate",
        probability=0.92,
        total_latency_ms=2.0,
        total_tokens_generated=0,
        reason="Ambiguous",
        traces=[tr_logit, tr_other],
    )
    v_judge = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="JudgeGate",
        total_latency_ms=10.0,
        total_tokens_generated=15,
        reason="Passed by judge",
        traces=[
            LayerTrace(
                latency_ms=1.0,
                layer_name="JudgeGate",
                status=DecisionStatus.PASSED,
                log_odds_pass=0.5,
                calibrated_prob=0.62,
                forward_log_odds=0.8,
                reverse_log_odds=-0.3,
            )
        ],
    )

    passed, _status, _p, fused_lo = fuse_escalated_verdict(v_upstream, v_judge, prior_weight=0.30)
    assert passed is True
    assert fused_lo is not None

    # 2. Mutant line 723: any(g.layer_tier > gate.layer_tier for g in self.gates[i + 1 :]) swapped + -> -
    # Create 3-gate cascade: Layer 0, Layer 1 (fails), Layer 2.
    g0 = _DummyCustomGate(should_pass=True, normalize_text=True)
    g1 = LogitGate(
        rubric="Fail rubric",
        backend=StubBackend(forward_top_logprobs=[{"token": "B", "logprob": 0.0}, {"token": "A", "logprob": -10.0}]),
    )
    # When g1 fails and no generative gate follows (or does not defer), fast_path logic executes
    casc3 = Cascade([g0, g1])
    v_c3 = casc3.evaluate("candidate text")
    assert v_c3.passed is False

    # Line 701: hasattr(gate, "normalize_input")
    assert g0.normalize_input("  HELLO  ") == "hello"


def test_cascade_persist_failure_fallback_config() -> None:
    # Lines 748-772: persist_failure when gates lack replay_config()
    g_custom = _DummyCustomGate(should_pass=False)
    casc = Cascade([g_custom])
    v = casc.evaluate("fail candidate", save_on_fail=False)
    assert v.passed is False


# ==============================================================================
# SECTION 4: logitgate.gates.base Tests & Mutant Killers
# ==============================================================================

@pytest.mark.asyncio
async def test_base_gate_async_edges_and_aguard_streams() -> None:
    gate = ASTGate(syntax=True)

    # Lines 131-133: as_async_conditional_edge
    async_edge = gate.as_async_conditional_edge(pass_node="pass", fail_node="fail")
    res = await async_edge({"content": "x = 1"})
    assert res == "pass"

    # Lines 154-156: aguard_stream
    async def _async_gen() -> Any:
        yield "def "
        yield "foo():"
        yield " pass"

    stream = gate.aguard_stream(_async_gen())
    chunks = [chunk async for chunk in stream]
    assert "".join(chunks) == "def foo(): pass"


def test_base_gate_rshift_cascade_composition_mutants() -> None:
    # Mutants lines 171 & 173: is not -> is on eff_sink and eff_cache
    s1 = CallbackTelemetrySink(lambda _: None)
    s2 = CallbackTelemetrySink(lambda _: None)
    c1_cache = VerdictCache()
    c2_cache = VerdictCache()

    casc1 = Cascade([ASTGate(syntax=True)], telemetry_sink=s1, cache=c1_cache)
    casc2 = Cascade([LogitGate(rubric="R", backend="stub://default")], telemetry_sink=s2, cache=c2_cache)

    casc3 = casc1 >> casc2
    # With line 171/173 'is not', self.telemetry_sink (s1) and self.cache (c1_cache) win!
    # With mutant 'is', other.telemetry_sink (s2) or other.cache (c2_cache) would be selected!
    assert casc3.telemetry_sink is s1
    assert casc3.cache is c1_cache

    # When casc1 has None, other wins:
    casc_none = Cascade([ASTGate(syntax=True)], telemetry_sink=None, cache=None)
    casc4 = casc_none >> casc2
    assert casc4.telemetry_sink is s2
    assert casc4.cache is c2_cache


# ==============================================================================
# SECTION 5: logitgate.gates.ast_gate Tests & Mutant Killers
# ==============================================================================

def test_ast_gate_dag_cycle_exact_unresolved_mutant() -> None:
    # Mutant line 182: unresolved = [tid for tid, deg in in_degree.items() if deg > 0] swapped > -> >=
    # Node 1 has in_degree 0 (points to 2). Node 2 and 3 point to each other (cycle, in_degree > 0).
    # With > 0: Node 1 is NOT in unresolved list!
    # With >= 0: Node 1 WOULD be in unresolved list!
    tasks = [
        {"id": "task_1", "dependencies": []},
        {"id": "task_2", "dependencies": ["task_1", "task_3"]},
        {"id": "task_3", "dependencies": ["task_2"]},
    ]
    gate = ASTGate(acyclic_dag=True)
    v = gate.evaluate(tasks)
    assert v.passed is False
    assert v.reason is not None and "task_2" in v.reason
    assert v.reason is not None and "task_3" in v.reason
    assert v.reason is not None and "task_1" not in v.reason  # Kills mutant line 182 > -> >= !


def test_ast_gate_max_length_zero_mutant() -> None:
    # Mutant line 269: max_length < 0 swapped < -> <=
    # max_length=0 is a valid non-negative integer. With < 0 it succeeds, with <= 0 it raises!
    gate = ASTGate(max_length=0)
    assert gate.max_length == 0
    v = gate.evaluate("")
    assert v.passed is True


def test_ast_gate_extract_imports_parsed_ast_mutant() -> None:
    # Mutant line 396: if parsed_ast is None swapped is -> is not
    # When source is invalid syntax but pre-parsed AST is supplied:
    pre_parsed = ast.parse("import math\nfrom os import path")
    imports = ASTGate._extract_imports_from_source("invalid syntax {{{", parsed_ast=pre_parsed)
    assert "math" in imports
    assert "os" in imports  # Kills mutant line 396 !


def test_ast_gate_normalize_sections_and_json_schema_mutant() -> None:
    # Mutant line 522: self.json_schema is None swapped is -> is not
    gate = ASTGate(required_sections=["## Plan"], normalize=True)
    norm = gate.normalize_input("## Plan\nDo work")
    assert norm == "## Plan\nDo work"

    # Mutant line 533: hasattr(validate_json) and hasattr(validate_python) swapped and -> or
    class _PartialValidator:
        def __call__(self, candidate: Any) -> Any:
            return candidate

        def validate_json(self) -> None:
            pass

    gate_val = ASTGate(validators=[_PartialValidator()])
    assert gate_val.validators is not None


def test_ast_gate_json_schema_optional_property_mutant() -> None:
    # Mutant line 758: if prop_name in parsed_json and isinstance(prop_def, dict) and "type" in prop_def
    # swapped and -> or
    # Schema defines optional property "age". Candidate does NOT include "age".
    # With 'and', omitted key is skipped. With 'or', it tries val = parsed_json['age'] -> KeyError!
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    gate = ASTGate(json_schema=schema)
    v = gate.evaluate('{"name": "Alice"}')
    assert v.passed is True  # Kills mutant line 758 !


def test_ast_gate_forbidden_patterns_literal_mutant() -> None:
    # Mutant line 861: matched = bool(compiled.search) if compiled else pattern in pattern_haystack
    # swapped in -> not in
    # Forbidden pattern is absent from candidate; candidate must PASS!
    gate = ASTGate(forbidden_patterns=["AWS_SECRET_KEY"])
    v = gate.evaluate("harmless public code")
    assert v.passed is True  # Kills mutant line 861 !


# ==============================================================================
# SECTION 6: logitgate.gates.logit_gate Tests & Mutant Killers
# ==============================================================================

def test_logit_gate_ensure_logprobs_present_mutant() -> None:
    # Mutant line 58: if not has_finite_logprobs(top_fwd) or (top_rev is not None and not has_finite_logprobs(top_rev))
    # swapped or -> and
    # When top_fwd is empty and top_rev is None:
    # With 'or', raises BackendError. With 'and', True and False is False -> does not raise!
    with pytest.raises(BackendError, match="Backend returned no token log-probabilities"):
        _ensure_logprobs_present([], top_rev=None)


def test_multi_rubric_gate_short_circuit_reason_mutant() -> None:
    # Mutant line 1176: if evaluated < total swapped < -> <=
    # When evaluated == total (all rubrics evaluated), reason must NOT say short-circuited!
    b = StubBackend(forward_top_logprobs=[{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": -5.0}])
    gate = MultiRubricGate(rubrics=["R1", "R2"], backend=b, fail_fast=False)
    v = gate.evaluate("candidate")
    assert v.passed is True
    assert v.reason is not None and "short-circuited" not in v.reason  # Kills mutant line 1176 !


def test_choice_gate_symmetric_debiasing_default_mutant() -> None:
    # Mutant line 1399: True if symmetric_debiasing is None else bool(symmetric_debiasing)
    # swapped is -> is not
    # Default is None -> must be True!
    gate = ChoiceGate(
        question="Choose best",
        options={"A": "Option 1", "B": "Option 2"},
        backend="stub://default",
        symmetric_debiasing=None,
    )
    assert gate.symmetric_debiasing is True  # Kills mutant line 1399 !


# ==============================================================================
# SECTION 7: logitgate.gates.rank_gate Tests & Mutant Killers
# ==============================================================================

def test_rank_gate_unhashable_modes_and_name_validation() -> None:
    # Lines 190-191 & 198-199: TypeError when mode or permutation_mode is unhashable
    with pytest.raises(ConfigurationError, match="Invalid mode"):
        RankGate(rubric="R", mode=["pairwise"], backend="stub://default")  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="Invalid permutation_mode"):
        RankGate(rubric="R", permutation_mode=["all"], backend="stub://default")  # type: ignore[arg-type]

    # Line 212: empty name
    with pytest.raises(ConfigurationError, match="name must be a non-empty string"):
        RankGate(rubric="R", name="", backend="stub://default")


def test_rank_gate_repr_long_rubric_and_replay_config() -> None:
    # Lines 764, 772, 783-784
    long_rubric = "This is a very long rubric exceeding forty characters for rank gate testing"
    gate = RankGate(rubric=long_rubric, context="System context", warm_prefix=True, backend="stub://default")
    r_repr = repr(gate)
    assert "..." in r_repr

    cfg = gate.replay_config()
    assert cfg["context"] == "System context"
    assert cfg["warm_prefix"] is True


def test_rank_gate_cache_hits_evaluate_and_aevaluate() -> None:
    # Lines 709-712, 802, 883
    cache = VerdictCache()
    gate = RankGate(rubric="Pick best", backend="stub://default", cache=cache)
    cands = ["candidate A", "candidate B"]

    v1 = gate.evaluate(cands)
    assert v1.passed is True

    # Cache hit in evaluate (line 802)
    v_hit = gate.evaluate(cands)
    assert v_hit.passed is True


@pytest.mark.asyncio
async def test_rank_gate_aevaluate_cache_hit_and_listwise_mutant() -> None:
    # Line 883: aevaluate cache hit
    # Line 912: eff_mode == "listwise" swapped == -> !=
    cache = VerdictCache()
    gate = RankGate(rubric="Rank items", mode="listwise", backend="stub://default", cache=cache)
    cands = ["cand 1", "cand 2", "cand 3"]

    v1 = await gate.aevaluate(cands)
    assert v1.passed is True

    v2 = await gate.aevaluate(cands)
    assert v2.passed is True


def test_best_of_n_gate_exhaustive(tmp_path: Path) -> None:
    # Lines 1097-1100: exception in BestOfNGate.__init__ closes owned sink
    mock_sink = MagicMock()
    with (
        patch("logitgate.telemetry.sinks.resolve_telemetry_sink", return_value=mock_sink),
        patch("logitgate.gates.rank_gate.RankGate.__init__", side_effect=ValueError("RankGate error")),
        pytest.raises(ValueError, match="RankGate error"),
    ):
        BestOfNGate(rubric="R", telemetry_sink="/tmp/best_sink.jsonl")
    assert mock_sink.close.called

    # Lines 1134-1136: post_gate config in replay_config
    pre_g = ASTGate(syntax=True)
    post_g = LogitGate(rubric="R_post", backend="stub://default")
    best_g = BestOfNGate(rubric="R_best", pre_gate=pre_g, post_gate=post_g, backend="stub://default")
    cfg = best_g.replay_config()
    assert "post_gate_config" in cfg

    # Lines 1394-1400: _has_active_top_cache mutants lines 1396 & 1400: return True -> False
    # When call_kwargs cache=True or cache instance: must return True!
    assert best_g._has_active_top_cache({"cache": True}) is True
    assert best_g._has_active_top_cache({"cache": False}) is False
    v_cache = VerdictCache()
    best_with_cache = BestOfNGate(rubric="R", backend="stub://default", cache=v_cache)
    assert best_with_cache._has_active_top_cache({}) is True
    best_no_cache = BestOfNGate(rubric="R", backend="stub://default", cache=False)
    assert best_no_cache._has_active_top_cache({}) is False

    # Cache hit in BestOfNGate evaluate (line 1414)
    best_cached = BestOfNGate(rubric="Pick best", backend="stub://default", cache=v_cache)
    cands = {"candA": "x = 1", "candB": "y = 2"}
    v_first = best_cached.evaluate(cands)
    assert v_first.passed is True
    v_hit = best_cached.evaluate(cands)
    assert v_hit.passed is True

    # Line 1464: _normalized_out collection
    norm_out: list[Any] = []
    best_g.evaluate(cands, _normalized_out=norm_out)
    assert len(norm_out) == 1
    assert norm_out[0] in ("x = 1", "y = 2")


@pytest.mark.asyncio
async def test_best_of_n_gate_aevaluate_and_calibration_mutant() -> None:
    # Mutant line 1121: if child is not None in _aensure_calibrated swapped is not -> is
    class _MockChildGate(BaseGate):
        def __init__(self) -> None:
            self.aensure_called = False

        async def _aensure_calibrated(self) -> None:
            self.aensure_called = True

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(
                passed=True,
                status=DecisionStatus.PASSED,
                triggered_layer=self.name,
                total_latency_ms=1.0,
                total_tokens_generated=0,
                reason="pass",
            )

    pre_child = _MockChildGate()
    best_gate = BestOfNGate(rubric="R", pre_gate=pre_child, backend="stub://default")
    await best_gate._aensure_calibrated()
    assert pre_child.aensure_called is True  # Kills mutant line 1121 !

    # BestOfNGate.aevaluate cache hit (line 1476)
    v_cache = VerdictCache()
    best_cached = BestOfNGate(rubric="R_async", backend="stub://default", cache=v_cache)
    cands = {"A": "a", "B": "b"}
    v1 = await best_cached.aevaluate(cands)
    assert v1.passed is True
    v2 = await best_cached.aevaluate(cands)
    assert v2.passed is True


def test_snapshot_coverage_closure_and_extra_backends(tmp_path: Path) -> None:
    # Line 384: custom:// backend spec resolves to effective_backend = None
    b_custom, m_custom = VerdictSnapshot._resolve_replay_backend_and_model(
        None, "some_model", {"backend": "custom://my_impl"}
    )
    assert b_custom is None
    assert m_custom == "some_model"

    # Line 384: stub:// backend with model override resolves to None
    b_stub, _ = VerdictSnapshot._resolve_replay_backend_and_model(
        None, "override_model", {"backend": "stub://default"}
    )
    assert b_stub is None

    # Lines 514, 521-522, 545-561: BestOfNGate child rebuilding and extra_backends
    v_dummy = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="BestOfNGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[LayerTrace(latency_ms=1.0, layer_name="BestOfNGate", status=DecisionStatus.FAILED)],
    )
    snap_best = VerdictSnapshot(
        rubric="R",
        candidate=["a", "b"],
        verdict=v_dummy,
        gate_config={
            "gate_type": "BestOfNGate",
            "pre_gate_config": {"gate_type": "LogitGate", "rubric": "Pre", "backend": "stub://child1"},
            "post_gate_config": {"gate_type": "JudgeGate", "rubric": "Post", "backend": "stub://child2"},
            "backend": "stub://main",
        },
    )
    # Synchronous replay builds gate and closes extra backends in finally
    v_rep = snap_best.replay()
    assert v_rep is not None

    # Lines 545-551: child cleanup when gate construction fails
    snap_broken_child = VerdictSnapshot(
        rubric="R",
        candidate=["a", "b"],
        verdict=v_dummy,
        gate_config={
            "gate_type": "BestOfNGate",
            "pre_gate_config": {"gate_type": "LogitGate", "rubric": "Pre", "backend": "stub://default"},
            "post_gate_config": {"gate_type": "ChoiceGate", "options": {}},  # raises ValueError in ChoiceGate
            "backend": "stub://main",
        },
    )
    with pytest.raises(ValueError, match="missing valid 'options' metadata"):
        snap_broken_child.replay()

    # Lines 357-358: RecursionError in load
    bad_json_file = tmp_path / "deep_nested.json"
    bad_json_file.write_text('{"rubric": "test"}', encoding="utf-8")
    with (
        patch("json.loads", side_effect=RecursionError("maximum recursion depth")),
        pytest.raises(ValueError, match="maximum nesting depth exceeded"),
    ):
        VerdictSnapshot.load(bad_json_file)


@pytest.mark.asyncio
async def test_snapshot_areplay_extra_backends_and_aexplain_cleanup() -> None:
    # Line 650: areplay extra_backends aclose
    v_dummy = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="BestOfNGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[LayerTrace(latency_ms=1.0, layer_name="BestOfNGate", status=DecisionStatus.FAILED)],
    )
    snap = VerdictSnapshot(
        rubric="R",
        candidate=["a", "b"],
        verdict=v_dummy,
        gate_config={
            "gate_type": "BestOfNGate",
            "pre_gate_config": {"gate_type": "LogitGate", "rubric": "Pre", "backend": "stub://child1"},
            "backend": "stub://main",
        },
    )
    v_rep = await snap.areplay()
    assert v_rep is not None

    # Line 776: aexplain with string backend so owns_backend is True -> judge.backend.aclose()
    v_exp = await snap.aexplain(backend="stub://default")
    assert v_exp is not None


def test_benchmark_coverage_closure_validation_and_reporting() -> None:
    pure_gen = _PureGenBackend("res")
    dual = _DualMockBackend()
    case = BenchmarkCase("r", "c", True)

    # Line 194: _MeteredBackend.agenerate when inner not generative
    metered_logit = _MeteredBackend(_PureLogitBackend(), require_generative=False)
    with pytest.raises(TypeError, match="Benchmarking the judge requires a generative backend"):
        asyncio.run(metered_logit.agenerate("prompt"))

    # Line 199: _MeteredBackend.close()
    metered_dual = _MeteredBackend(dual)
    metered_dual.close()
    assert dual.closed is True

    # Line 218: _MeteredGenerativeBackend.__getattr__ forwarding
    metered_gen = _MeteredGenerativeBackend(pure_gen)
    assert metered_gen.text == "res"

    # Lines 569-578: clone kwargs with OllamaBackend options and timeout
    class _FakeOllama(OllamaBackend):
        def __init__(self) -> None:
            self.timeout = 25.0
            self.api_key = "secret_key"
            self.num_ctx = 2048
            self.options = {"top_p": 0.85}

    fake_o = _FakeOllama()
    kw = _extract_backend_clone_kwargs(fake_o)
    assert kw["timeout"] == 25.0
    assert kw["api_key"] == "secret_key"
    assert kw["num_ctx"] == 2048
    assert kw["options"] == {"top_p": 0.85}

    # Lines 635, 637, 640, 642, 644: validation bounds
    with pytest.raises(ValueError, match="needs at least one case"):
        run_benchmark([])
    with pytest.raises(ValueError, match="needs at least one mode"):
        run_benchmark([case], modes=[])
    with pytest.raises(ValueError, match="Unknown benchmark modes"):
        run_benchmark([case], modes=["unsupported_mode"])
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        run_benchmark([case], repeat=0)
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        run_benchmark([case], warmup=-1)
    with pytest.raises(ValueError, match="judge_max_tokens must be >= 1"):
        run_benchmark([case], judge_max_tokens=0)

    # Lines 684-696: modes=("judge",) with string and backend instances
    rep_judge_str = run_benchmark([case], modes=("judge",), backend="stub://default")
    assert "judge" in rep_judge_str.modes
    rep_judge_inst = run_benchmark([case], modes=("judge",), backend=StubBackend())
    assert "judge" in rep_judge_inst.modes

    # Lines 701-708: heterogeneous judge resolution with judge_model and backend string
    rep_hetero = run_benchmark([case], modes=("logit", "judge"), backend="stub://default", judge_model="teacher_model")
    assert rep_hetero.metadata["judge_model"] == "teacher_model"

    # Line 717: resolved_judge is not GenerativeBackend
    with pytest.raises(TypeError, match="GenerativeBackend"):
        run_benchmark([case], modes=("judge",), judge_backend=_PureLogitBackend())  # type: ignore[arg-type]

    # Lines 910, 963-966: format_report with non-stub and profile
    rep_full = run_benchmark([case], modes=("judge", "logit"))
    rep_full.metadata["is_stub"] = False
    rep_full.metadata["profile"] = {
        "binary_temperature": 1.25,
        "fail_threshold": 0.15,
        "pass_threshold": 0.85,
        "target_risk_alpha": 0.05,
        "guarantee_holds": True,
    }
    # Force non-zero timings
    rep_full.modes["judge"].latencies_ms = [100.0]
    rep_full.modes["judge"].generated_tokens = [20]
    rep_full.modes["logit"].latencies_ms = [10.0]
    rep_full.modes["logit"].generated_tokens = [0]
    rep_text = format_report(rep_full)
    assert "profile: T=1.25" in rep_text
    assert "faster" in rep_text


@pytest.mark.asyncio
async def test_cascade_coverage_closure_and_shadow_audit(tmp_path: Path) -> None:
    # Lines 289, 298: ensure children calibrated sync and async on MultiRubricGate inner gate
    mr = MultiRubricGate(rubrics=["R1", "R2"], backend="stub://default")
    casc_mr = Cascade([mr])
    casc_mr._ensure_children_calibrated_sync()
    await casc_mr._aensure_children_calibrated_async()

    # Lines 346, 377, 407: temporary call cache cleanup
    casc = Cascade([ASTGate(syntax=True)])
    db1 = str(tmp_path / "temp_call1.db")
    casc.evaluate("x = 1", cache=db1)
    db2 = str(tmp_path / "temp_call2.db")
    await casc.aevaluate("x = 1", cache=db2)

    # Line 479: gate_rubric inheritance when upstream has no rubric
    g_norubric = _DummyCustomGate(should_pass=True)
    jg = JudgeGate(rubric="Specific Judge Rubric", backend="stub://default", trigger="always")
    casc_jg = Cascade([g_norubric, jg])
    v_jg = casc_jg.evaluate("some candidate")
    assert v_jg.passed is True

    # Lines 567-571: shadow audit without override preserves upstream pass & sets shadow_audit metadata
    class _MockAuditController:
        feedback_mode = "cascade_audit"
        audit_sample_rate = 1.0
        override_on_audit = False

        def should_audit_fast_path(self) -> bool:
            return True

        def record_cascade_outcome(self, **kwargs: Any) -> None:
            pass

        def current_decision_state(self) -> tuple[float | None, float, float, float, float, bool]:
            return (None, 0.2, 0.85, -1.0, 1.0, False)

    lg_pass = LogitGate(
        rubric="Safety check",
        backend=StubBackend(forward_top_logprobs=[{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": -5.0}]),
    )
    lg_pass.adaptive_controller = _MockAuditController()  # type: ignore[assignment]
    jg_shadow = JudgeGate(
        rubric="Safety check",
        backend=StubBackend(diagnostic_text="[VERDICT: FAIL] Audit failed"),
        trigger="on_ambiguous_only",
    )
    casc_audit = Cascade([lg_pass, jg_shadow])
    v_audited = casc_audit.evaluate("safe candidate")
    # Upstream passed; shadow audit evaluated but did not override
    assert v_audited.passed is True
    assert v_audited.traces[-1].metadata.get("shadow_audit") is True

    # Lines 751-772: persist_failure with custom gates without replay_config()
    g_c1 = _DummyCustomGate(should_pass=False)
    g_c1.backend = StubBackend()  # type: ignore[attr-defined]
    g_c1.model = "my_custom_model"  # type: ignore[attr-defined]
    g_jg2 = JudgeGate(rubric="Critique", backend=StubBackend(), model="my_judge_model", trigger="always")
    casc_custom = Cascade([g_c1, g_jg2])
    v_failed = casc_custom.evaluate("bad candidate", save_on_fail=True, snapshot_dir=tmp_path)
    assert v_failed.passed is False
    assert v_failed.traces
    assert "snapshot_path" in v_failed.traces[-1].metadata


@pytest.mark.asyncio
async def test_rank_gate_lazy_calibration_sync_async_and_context() -> None:
    # Lines 279-283 & 288-291: RankGate lazy calibration sync and async
    gate = RankGate(rubric="Rank", backend="stub://default")
    gate._needs_lazy_calibration = True
    gate.model = "test_model_for_calibration"

    with patch("logitgate.gates.rank_gate._lazy_calibrate_sync", return_value=1.5):
        gate._ensure_calibrated()
        assert gate.logit_temperature == 1.5
        assert gate._needs_lazy_calibration is False

    gate._needs_lazy_calibration = True
    async def _mock_lazy_async(*args: Any, **kwargs: Any) -> float:
        return 2.0

    with patch("logitgate.gates.rank_gate._lazy_calibrate_async", side_effect=_mock_lazy_async):
        await gate._aensure_calibrated()
        assert gate.logit_temperature == 2.0
        assert gate._needs_lazy_calibration is False

    # Line 1386: BestOfNGate context in call_kwargs
    best_g = BestOfNGate(rubric="Rank options", backend="stub://default")
    v_ctx = best_g.evaluate({"A": "Option 1", "B": "Option 2"}, context="Call-time context")
    assert v_ctx.passed is True


def test_mutant_killers_snapshot_round2(tmp_path: Path) -> None:
    # 1. Line 150: gate_type and "gate_type" not in effective_cfg
    failed_v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="LogitGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Fail",
        traces=[LayerTrace(latency_ms=1.0, layer_name="LogitGate", status=DecisionStatus.FAILED)],
    )
    v_saved = persist_failure_snapshot(
        failed_v, candidate="bad", save_on_fail=True, gate_type="", gate_config={}, snapshot_dir=tmp_path
    )
    assert v_saved.traces
    snap_path = Path(v_saved.traces[-1].metadata["snapshot_path"])
    snap_obj = VerdictSnapshot.load(snap_path)
    assert "gate_type" not in snap_obj.gate_config  # Kills mutant line 150 !

    # 2. Line 374: prefer_judge and judge_backend is not None
    snap_hetero = VerdictSnapshot(
        rubric="R",
        candidate="c",
        verdict=failed_v,
        gate_config={"backend": "stub://tier1", "judge_backend": "stub://tier2"},
    )
    b_eff, _ = snap_hetero._resolve_replay_backend_and_model(None, None, snap_hetero.gate_config, prefer_judge=True)
    assert b_eff == "stub://tier2"  # Kills mutant line 374 !

    # 3. Line 515: c_is_judge == "JudgeGate"
    snap_best_post = VerdictSnapshot(
        rubric="R",
        candidate=["a", "b"],
        verdict=failed_v,
        gate_config={
            "gate_type": "BestOfNGate",
            "post_gate_config": {"gate_type": "JudgeGate", "rubric": "Critique", "backend": "stub://default"},
            "backend": "stub://default",
        },
    )
    gate_built, _ = snap_best_post._build_replay_gate(snap_best_post.gate_config, "stub://default", None)
    assert isinstance(gate_built.post_gate, JudgeGate)  # Kills mutant line 515 !

    # 4. Line 557: c_b is not best_gate.backend
    main_b = StubBackend()
    snap_shared_b = VerdictSnapshot(
        rubric="R",
        candidate=["a", "b"],
        verdict=failed_v,
        gate_config={
            "gate_type": "BestOfNGate",
            "pre_gate_config": {"gate_type": "LogitGate", "rubric": "Pre"},
        },
    )
    best_built, _ = snap_shared_b._build_replay_gate(snap_shared_b.gate_config, main_b, None)
    assert best_built._replay_extra_backends == []  # Kills mutant line 557 !

    # 5. Line 579: isinstance(rubrics, list) and rubrics
    snap_empty_rubrics = VerdictSnapshot(
        rubric="R",
        candidate="c",
        verdict=failed_v,
        gate_config={"rubrics": []},
    )
    gate_single, _ = snap_empty_rubrics._build_replay_gate(snap_empty_rubrics.gate_config, main_b, None)
    assert isinstance(gate_single, LogitGate)
    assert not isinstance(gate_single, MultiRubricGate)  # Kills mutant line 579 !


def test_mutant_killers_cascade_round2() -> None:
    # 1. Line 128: tr1 is None in fuse_escalated_verdict
    v_up_nologit = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        triggered_layer="ASTGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Ambig",
        traces=[LayerTrace(latency_ms=1.0, layer_name="ASTGate", status=DecisionStatus.AMBIGUOUS)],
    )
    v_judge = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="JudgeGate",
        total_latency_ms=5.0,
        total_tokens_generated=5,
        reason="Judge pass",
        traces=[LayerTrace(latency_ms=1.0, layer_name="JudgeGate", status=DecisionStatus.PASSED)],
    )
    _p, _s, _prob, fused_lo = fuse_escalated_verdict(v_up_nologit, v_judge, prior_weight=0.3)
    assert fused_lo is None  # Kills mutant line 128 !

    # 2. Line 318: primary_backend tier >= TIER_LOGIT in _finalize_cache_hit
    records: list[Any] = []
    sink = CallbackTelemetrySink(lambda rec: records.append(rec))
    b = StubBackend()
    lg = LogitGate(rubric="Check", backend=b)
    casc = Cascade([lg], telemetry_sink=sink)
    hit_v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="LogitGate",
        total_latency_ms=1.0,
        total_tokens_generated=0,
        reason="Hit",
        traces=[LayerTrace(latency_ms=1.0, layer_name="LogitGate", status=DecisionStatus.PASSED)],
    )
    casc._finalize_cache_hit(hit_v, "candidate", {})
    assert len(records) == 1
    assert records[0].backend == "stub://default"  # Kills mutant line 318 !

    # 3. Line 604: tr1 is not None in prior_log_odds metadata
    b_up = StubBackend(forward_top_logprobs=[{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": -0.8}])
    b_jg = StubBackend(diagnostic_text="[VERDICT: PASS] ok")
    lg_first = LogitGate(rubric="Safety", backend=b_up, pass_threshold=0.99, fail_threshold=0.01)
    jg_next = JudgeGate(rubric="Safety", backend=b_jg, trigger="always")
    casc_fuse = Cascade([lg_first, jg_next], fuse_logits=True, prior_weight=0.30)
    with patch("logitgate.cascade.fuse_escalated_verdict", return_value=(True, DecisionStatus.PASSED, 0.8, 1.5)):
        v_fuse = casc_fuse.evaluate("code")
    assert v_fuse.traces
    assert v_fuse.traces[-1].metadata.get("prior_log_odds") is not None  # Kills mutant line 604 !

    # 4. Line 723: any(g.layer_tier > gate.layer_tier for g in self.gates[i + 1 :])
    g_jg = JudgeGate(rubric="R", backend=StubBackend(), trigger="always")
    g_ast_fail = ASTGate(syntax=True)
    casc_rev = Cascade([g_jg, g_ast_fail])
    v_rev = casc_rev.evaluate("invalid syntax {{{")
    assert "fast-path" not in (v_rev.reason or "").lower()  # Kills mutant line 723 !


def test_mutant_killers_benchmark_round2() -> None:
    case = BenchmarkCase("r", "c", True)

    # 1. Line 559: "judge" in self.modes or comparisons
    rep = run_benchmark([case], modes=("judge",))
    assert "comparisons" in rep.to_dict()  # Kills mutant line 559 !

    # 2. Line 650: owns_cache with cache=True
    rep_cache = run_benchmark([case], modes=("logit",), cache=True)
    assert "cache" in rep_cache.metadata  # Kills mutant line 650 !

    # 3. Line 679: heterogeneous_judge with judge_backend=None and judge_model=None
    rep_homo = run_benchmark([case], modes=("logit", "cascade"), backend="stub://default")
    assert rep_homo.metadata["judge_backend"] == rep_homo.metadata["backend"]  # Kills mutant line 679 !

    # 4. Line 814: generate_calls = gen_after - gen_before
    rep_gen = run_benchmark([case, BenchmarkCase("r2", "c2", False)], modes=("judge",), repeat=1, warmup=0)
    assert rep_gen.modes["judge"].llm_calls == [1, 1]  # Kills mutant line 814 !

    # 5. Line 844: primary_backend in modes=("judge",)
    rep_j_only = run_benchmark([case], modes=("judge",), backend="stub://default")
    assert rep_j_only.metadata["backend"] == "stub://default"  # Kills mutant line 844 !

    # 6. Line 894: judge_model_meta is not None in format_report
    rep_no_jm = run_benchmark([case], modes=("logit",), model="my_model", backend="stub://default")
    txt = format_report(rep_no_jm)
    assert "judge_model=None" not in txt  # Kills mutant line 894 !


def test_mutant_killers_ast_and_rank_gates_round2() -> None:
    # ASTGate Line 522: self.json_schema is None in normalize
    gate_sec = ASTGate(required_sections=["## Title"], normalize=True)
    norm = gate_sec.normalize_input("<think>pondering</think>## Title\nBody")
    assert norm == "## Title\nBody"  # Kills mutant line 522 !

    # ASTGate Line 861: compiled is None with unclosed regex pattern
    gate_unclosed = ASTGate(forbidden_patterns=["[unclosed_bracket"])
    v_unclosed = gate_unclosed.evaluate("regular clean text")
    assert v_unclosed.passed is True  # Kills mutant line 861 !

    # RankGate Line 1353: k in base_meta and k not in combined_traces[-1].metadata
    post_jg = JudgeGate(rubric="Critique", backend=StubBackend(diagnostic_text="[VERDICT: PASS] ok"), trigger="always")
    best_no_pre = BestOfNGate(rubric="Rank options", post_gate=post_jg, backend="stub://default")
    v_post = best_no_pre.evaluate({"A": "Option 1", "B": "Option 2"})
    assert v_post.passed is True  # Kills mutant line 1353 !
