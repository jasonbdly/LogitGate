from __future__ import annotations

import asyncio
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import logitgate
from logitgate import (
    ASTGate,
    BestOfNGate,
    CalibrationError,
    CalibrationProfile,
    CallbackTelemetrySink,
    Cascade,
    ChoiceGate,
    CompositeTelemetrySink,
    ConfigurationError,
    DistillationSink,
    JsonlTelemetrySink,
    JudgeGate,
    LogitGate,
    LogitGateDefaults,
    MemoryTelemetrySink,
    MultiRubricGate,
    RankGate,
    ScoreGate,
    StubBackend,
    TelemetryRecord,
    TelemetryWarning,
    VerdictSnapshot,
    configure,
    export_distillation_dataset,
    extract_distillation_record,
    get_default_telemetry_sink,
    load_calibration_dataset,
    load_distillation_records,
    load_telemetry_records,
    recalibrate_from_telemetry,
    redact_text,
    reset_config,
    summarize_telemetry,
)
from logitgate.backends.base import EXCLUDED_INTERNAL_KWARGS
from logitgate.cli import main as cli_main


@pytest.fixture(autouse=True)
def _clean_global_config(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("LOGITGATE_TELEMETRY_PATH", raising=False)
    monkeypatch.setenv("LOGITGATE_BACKEND", "stub://default")
    reset_config()
    yield
    reset_config()


def _make_binary_backend(fwd_a: float, fwd_b: float, rev_b: float, rev_a: float) -> StubBackend:
    """Builds a 2-pass StubBackend returning specified logprobs on Pass 1 (A=Pass) and Pass 2 (B=Pass)."""
    return StubBackend(
        logprob_responses=[
            [{"token": "A", "logprob": fwd_a}, {"token": "B", "logprob": fwd_b}],
            [{"token": "B", "logprob": rev_b}, {"token": "A", "logprob": rev_a}],
        ]
    )


def _make_logodds_backend(f_lo: float, r_lo: float) -> StubBackend:
    """Builds a 2-pass StubBackend yielding exact forward/reverse log-odds (with logprobs <= 0)."""
    fwd_a, fwd_b = (0.0, -f_lo) if f_lo >= 0.0 else (f_lo, 0.0)
    rev_b, rev_a = (0.0, -r_lo) if r_lo >= 0.0 else (r_lo, 0.0)
    return _make_binary_backend(fwd_a, fwd_b, rev_b, rev_a)


def _make_judge_backend(
    verdict_pass: bool,
    *,
    cot_a: float | None = None,
    cot_b: float | None = None,
) -> StubBackend:
    tag = "[VERDICT: PASS]" if verdict_pass else "[VERDICT: FAIL]"
    text = f"Detailed step-by-step verification notes on the artifact. {tag}"
    if cot_a is None or cot_b is None:
        if verdict_pass:
            cot_lp = [
                [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.20}],
                [{"token": "B", "logprob": -0.05}, {"token": "A", "logprob": -3.20}],
            ]
        else:
            cot_lp = [
                [{"token": "A", "logprob": -3.20}, {"token": "B", "logprob": -0.05}],
                [{"token": "B", "logprob": -3.20}, {"token": "A", "logprob": -0.05}],
            ]
    else:
        # Custom post-critique logprobs (e.g. split / uncertain judge for fuse_logits testing)
        cot_lp = [
            [{"token": "A", "logprob": cot_a}, {"token": "B", "logprob": cot_b}],
            [{"token": "B", "logprob": cot_b}, {"token": "A", "logprob": cot_a}],
        ]
    return StubBackend(diagnostic_text=text, logprob_responses=cot_lp)


# ---------------------------------------------------------------------------
# 1. Configuration, Scope Precedence, & Internal Kwargs Isolation
# ---------------------------------------------------------------------------


def test_default_noop_and_excluded_internal_kwargs() -> None:
    assert get_default_telemetry_sink() is None
    for kw in ("_suppress_telemetry", "telemetry_sink", "telemetry_tags"):
        assert kw in EXCLUDED_INTERNAL_KWARGS

    gate = LogitGate("Is this safe?", backend="stub://default")
    v = gate("hello world")
    assert v.passed is True


def test_global_configure_env_var_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "env_telemetry.jsonl"
    monkeypatch.setenv("LOGITGATE_TELEMETRY_PATH", str(env_file))

    gate = LogitGate("Is valid?", backend="stub://default")
    gate("from env", telemetry_tags={"stage": "env"})
    assert len(load_telemetry_records(env_file)) == 1
    assert load_telemetry_records(env_file)[0].tags == {"stage": "env"}

    # configure(telemetry_sink=False) suppresses LOGITGATE_TELEMETRY_PATH
    configure(telemetry_sink=False)
    assert get_default_telemetry_sink() is None
    gate("suppressed")
    assert len(load_telemetry_records(env_file)) == 1

    # Global sink < gate sink < per-call sink; False opts out at gate or call level
    global_sink = MemoryTelemetrySink()
    gate_sink = MemoryTelemetrySink()
    call_sink = MemoryTelemetrySink()
    configure(telemetry_sink=global_sink)
    assert LogitGateDefaults.telemetry_sink is global_sink

    g_inherit = LogitGate("Rubric A", backend="stub://default")
    g_override = LogitGate("Rubric B", backend="stub://default", telemetry_sink=gate_sink)
    g_optout = LogitGate("Rubric C", backend="stub://default", telemetry_sink=False)

    g_inherit("c1")
    assert len(global_sink.records) == 1
    assert len(gate_sink.records) == 0

    g_override("c2")
    assert len(global_sink.records) == 1
    assert len(gate_sink.records) == 1

    g_override("c3", telemetry_sink=call_sink)
    assert len(gate_sink.records) == 1
    assert len(call_sink.records) == 1

    g_override("c4", telemetry_sink=False)
    assert len(gate_sink.records) == 1
    assert len(call_sink.records) == 1

    g_optout("c5")
    assert len(global_sink.records) == 1

    # Configure via path string closes owned handle cleanly on reset_config()
    cfg_path = tmp_path / "configured.jsonl"
    configure(telemetry_sink=cfg_path)
    g_inherit("c6")
    reset_config()
    assert len(load_telemetry_records(cfg_path)) == 1


# ---------------------------------------------------------------------------
# 2. Single-Record Emission Across All Gates & Nested Pipelines (Sync + Async)
# ---------------------------------------------------------------------------


def test_all_gates_emit_exactly_one_record_sync_and_async(tmp_path: Path) -> None:
    sink = MemoryTelemetrySink()
    configure(telemetry_sink=sink)

    # 1. ASTGate pass + fail
    ast_gate = ASTGate(syntax=True)
    assert ast_gate("def ok():\n    return 1\n").passed is True
    assert ast_gate("def broken(:\n").passed is False
    assert [r.gate_type for r in sink.records] == ["ASTGate", "ASTGate"]
    assert [r.passed for r in sink.records] == [True, False]
    sink.clear()

    # 2. MultiRubricGate (3 rubrics -> 1 aggregated record, not 3)
    mr_gate = MultiRubricGate(["R1?", "R2?", "R3?"], backend="stub://default")
    assert mr_gate("artifact").passed is True
    assert asyncio.run(mr_gate.aevaluate("artifact")).passed is True
    assert len(sink.records) == 2
    assert all(r.gate_type == "MultiRubricGate" and len(r.traces) == 3 for r in sink.records)
    sink.clear()

    # 3. ChoiceGate & ScoreGate
    cg = ChoiceGate("Route?", {"A": " Billing", "B": "Support"}, backend="stub://default")
    sg = ScoreGate("Quality?", backend="stub://default")
    cg("invoice issue")
    asyncio.run(sg.aevaluate("clean answer"))
    assert [r.gate_type for r in sink.records] == ["ChoiceGate", "ScoreGate"]
    sink.clear()

    # 4. RankGate & BestOfNGate (pre_gate + rank_gate + post_gate -> 1 record per call!)
    rg = RankGate("Better code?", backend="stub://default")
    rg.rank(["def a(): return 1", "def b(): return 2"])
    assert len(sink.records) == 1 and sink.records[0].gate_type == "RankGate"
    sink.clear()

    bon = BestOfNGate(
        "Cleanest Python?",
        backend="stub://default",
        pre_gate=ASTGate(syntax=True),
        post_gate=LogitGate("Cleanest Python?", backend="stub://default"),
    )
    # 2 valid + 1 syntax error -> pre_gate runs 3x, rank_gate runs 1x, post_gate runs 1x -> 1 record!
    res = bon.select(["def a(): return 1", "def bad(:", "def c(): return 3"])
    assert res.verdict.passed is True
    assert len(sink.records) == 1
    assert sink.records[0].gate_type == "BestOfNGate"
    sink.clear()

    # Sole-survivor without post_gate (regression check for pass-without-post_gate exit path)
    bon_no_post = BestOfNGate(
        "Cleanest Python?",
        backend="stub://default",
        pre_gate=ASTGate(syntax=True),
    )
    res_sole = asyncio.run(bon_no_post.aselect(["def a(): return 1", "def bad(:"]))
    assert res_sole.verdict.passed is True
    res_two = bon_no_post.select(["def a(): return 1", "def b(): return 2"])
    assert res_two.verdict.passed is True
    res_none = bon_no_post.select(["def bad1(:", "def bad2(:"])
    assert res_none.verdict.passed is False
    assert len(sink.records) == 3
    assert all(r.gate_type == "BestOfNGate" for r in sink.records)
    sink.clear()

    # 5. JudgeGate with missing tags triggering internal _probe_gate -> still 1 record!
    untagged_backend = StubBackend(
        diagnostic_text="The code satisfies every rubric requirement cleanly.",
        logprob_responses=[
            [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.0}],
            [{"token": "B", "logprob": -0.05}, {"token": "A", "logprob": -3.0}],
        ],
    )
    jg = JudgeGate("Is code clean?", backend=untagged_backend, symmetric_debiasing=False)
    jv = jg("def f(): return 1")
    assert jv.passed is True
    assert len(sink.records) == 1
    assert sink.records[0].gate_type == "JudgeGate"
    sink.clear()

    # 6. Snapshot replay/explain suppresses telemetry by default
    snap_dir = tmp_path / "snaps"
    fail_gate = ASTGate(syntax=True, save_on_fail=True)
    fail_gate("def broken(:", snapshot_dir=str(snap_dir))
    assert len(sink.records) == 1
    snap = VerdictSnapshot.load(snap_dir)
    snap.explain(backend="stub://default")
    assert len(sink.records) == 1  # explain did not pollute production telemetry!


def test_cascade_fast_path_escalation_and_inner_sink_inheritance() -> None:
    s1 = MemoryTelemetrySink()
    s2 = MemoryTelemetrySink()

    # Inner sinks attached to individual gates chained via `>>` are automatically multiplexed!
    ambig_backend = _make_binary_backend(-0.69, -0.69, -0.69, -0.69)  # P = 0.50 -> AMBIGUOUS
    pass_backend = _make_binary_backend(-0.05, -3.20, -0.05, -3.20)  # P > 0.95 -> PASSED
    judge_backend = _make_judge_backend(True)

    esc_pipe = (
        ASTGate(syntax=True)
        >> LogitGate("Is safe?", backend=ambig_backend, model="qwen2.5-coder:3b", telemetry_sink=s1)
        >> JudgeGate(backend=judge_backend, model="gemma4:26b", telemetry_sink=s2)
    )
    v_esc = esc_pipe("def ok():\n    return 42\n")
    assert v_esc.passed is True
    assert len(s1.records) == 1 and len(s2.records) == 1
    rec = s1.records[0]
    assert rec.gate_type == "Cascade"
    assert rec.escalated is True
    assert rec.fast_pathed is False
    assert rec.model == "qwen2.5-coder:3b"
    assert rec.teacher_model == "gemma4:26b"
    assert rec.teacher_passed is True
    assert len(rec.traces) == 3

    # Fast-path at Layer 0 (AST fail skips LogitGate and JudgeGate)
    v_ast_fail = esc_pipe("def broken(:\n")
    assert v_ast_fail.passed is False
    assert len(s1.records) == 2
    assert s1.records[1].fast_pathed is True
    assert s1.records[1].escalated is False

    # Fast-path pass at Layer 1 (LogitGate passes, JudgeGate skipped)
    fp_sink = MemoryTelemetrySink()
    fp_pipe = Cascade(
        [
            LogitGate("Is safe?", backend=pass_backend, model="qwen2.5-coder:3b"),
            JudgeGate(backend=judge_backend, model="gemma4:26b"),
        ],
        telemetry_sink=fp_sink,
    )
    v_fp = asyncio.run(fp_pipe.aevaluate("def ok(): return 1"))
    assert v_fp.passed is True
    assert len(fp_sink.records) == 1
    assert fp_sink.records[0].fast_pathed is True
    assert fp_sink.records[0].escalated is False


# ---------------------------------------------------------------------------
# 3. Privacy Modes, ReDoS Linearity, Symlink Security, Rotation, & Concurrency
# ---------------------------------------------------------------------------


def test_privacy_modes_pii_redaction_and_redos_linearity() -> None:
    secret_payload = (
        "Contact alice.smith+ops@corp.example.co.uk with SSN 123-45-6789, "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload, "
        "OpenAI key sk-proj_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456, "
        "card 4111 1111 1111 1111, and benign order ID 1234567890123456."
    )
    scrubbed = redact_text(secret_payload)
    assert "alice.smith" not in scrubbed and "[REDACTED_EMAIL]" in scrubbed
    assert "123-45-6789" not in scrubbed and "[REDACTED_SSN]" in scrubbed
    assert "eyJhbGci" not in scrubbed and "Bearer [REDACTED_TOKEN]" in scrubbed
    assert "sk-proj_" not in scrubbed and "[REDACTED_SECRET]" in scrubbed
    assert "4111 1111 1111 1111" not in scrubbed and "[REDACTED_PAN]" in scrubbed
    # Non-Luhn 16-digit order ID is preserved!
    assert "1234567890123456" in scrubbed

    # Adversarial ReDoS linearity check (200 KB pathological near-miss input)
    adversarial = ("a" * 60 + "@" + "b" * 60 + " ") * 1500 + ("4" * 16 + " ") * 1000
    t0 = time.perf_counter()
    _ = redact_text(adversarial, max_chars=65536)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 100.0

    # Sink privacy modes: hash_only, omit_candidate, redact_pii, max_candidate_chars
    s_hash = MemoryTelemetrySink(privacy="hash_only")
    s_omit = MemoryTelemetrySink(privacy="omit_candidate")
    s_pii = MemoryTelemetrySink(privacy="redact_pii", max_candidate_chars=120)

    jg = JudgeGate(
        "Check compliance",
        backend=StubBackend(diagnostic_text="[VERDICT: PASS] Contact bob@example.com"),
    )
    comp = CompositeTelemetrySink([s_hash, s_omit, s_pii])
    jg(secret_payload, telemetry_sink=comp)

    assert s_hash.records[0].candidate is None
    assert len(s_hash.records[0].candidate_sha256) == 64
    assert s_hash.records[0].traces[0].diagnostic_report is not None

    assert s_omit.records[0].candidate is None
    assert s_omit.records[0].traces[0].diagnostic_report is None

    assert s_pii.records[0].candidate is not None
    assert "alice.smith" not in s_pii.records[0].candidate
    assert s_pii.records[0].traces[0].diagnostic_report is not None
    assert "bob@example.com" not in s_pii.records[0].traces[0].diagnostic_report


def test_symlink_rejection_rotation_concurrency_and_error_isolation(tmp_path: Path) -> None:
    real_file = tmp_path / "real.jsonl"
    real_file.write_text("", encoding="utf-8")
    sym_file = tmp_path / "symlink.jsonl"
    sym_file.symlink_to(real_file)

    with pytest.raises(ConfigurationError, match="symlink"):
        JsonlTelemetrySink(sym_file)
    with pytest.raises(ConfigurationError, match="symlink"):
        DistillationSink(sym_file)
    with pytest.raises(ConfigurationError, match="symlink"):
        load_telemetry_records(sym_file)
    with pytest.raises(ConfigurationError, match="symlink"):
        load_distillation_records(sym_file)
    with pytest.raises(ConfigurationError, match="symlink"):
        export_distillation_dataset([], sym_file)

    # Size-based rotation
    rot_path = tmp_path / "rotated.jsonl"
    with JsonlTelemetrySink(rot_path, max_bytes=800, backup_count=2) as rot_sink:
        gate = LogitGate("Is clean?", backend="stub://default", telemetry_sink=rot_sink)
        for idx in range(12):
            gate(f"candidate payload {idx} " + ("x" * 80))

    assert rot_path.exists()
    assert rot_path.with_name("rotated.jsonl.1").exists()
    assert rot_path.with_name("rotated.jsonl.2").exists()
    assert not rot_path.with_name("rotated.jsonl.3").exists()
    # Every rotated file contains valid TelemetryRecord JSONL lines
    for p in (rot_path, rot_path.with_name("rotated.jsonl.1"), rot_path.with_name("rotated.jsonl.2")):
        recs = load_telemetry_records(p)
        assert len(recs) >= 1

    # Concurrent multi-threaded writes (16 workers, 160 records)
    conc_path = tmp_path / "concurrent.jsonl"
    with JsonlTelemetrySink(conc_path) as conc_sink:
        c_gate = LogitGate("Thread safe?", backend="stub://default", telemetry_sink=conc_sink)

        def worker(i: int) -> bool:
            return c_gate(f"item-{i}").passed

        with ThreadPoolExecutor(max_workers=16) as pool:
            outcomes = list(pool.map(worker, range(160)))
    assert all(outcomes)
    loaded_conc = load_telemetry_records(conc_path)
    assert len(loaded_conc) == 160

    # Error policies & CompositeTelemetrySink isolation
    def boom(_rec: TelemetryRecord) -> None:
        raise RuntimeError("downstream collector timeout")

    warn_cb = CallbackTelemetrySink(boom, on_error="warn")
    raise_cb = CallbackTelemetrySink(boom, on_error="raise")
    ignore_cb = CallbackTelemetrySink(boom, on_error="ignore")
    healthy_mem = MemoryTelemetrySink()

    g = LogitGate("Test?", backend="stub://default")
    with pytest.warns(TelemetryWarning, match="CallbackTelemetrySink"):
        g("x", telemetry_sink=warn_cb)

    g("x", telemetry_sink=ignore_cb)  # silent

    comp_isolated = CompositeTelemetrySink([raise_cb, healthy_mem])
    with pytest.raises(RuntimeError, match="downstream collector timeout"):
        g("x", telemetry_sink=comp_isolated)
    # Sibling sink STILL received the record despite raise_cb throwing first!
    assert len(healthy_mem.records) == 1


# ---------------------------------------------------------------------------
# 4. DistillationSink, Discrepancy Taxonomy, Zero-Call Recalibration, & Export
# ---------------------------------------------------------------------------


def _run_synthetic_escalation_stream(
    tel_path: Path,
    dist_path: Path,
) -> tuple[list[TelemetryRecord], list[Any]]:
    """Streams 24 synthetic student/teacher escalations covering all 4 discrepancy types."""
    comp = CompositeTelemetrySink([JsonlTelemetrySink(tel_path), DistillationSink(dist_path)])
    rubric = "Does the implementation safely parameterize SQL queries without string interpolation?"

    # 12 true-PASS cases and 12 true-FAIL cases where an overconfident/miscalibrated student
    # (T=1.0, wide band [0.05, 0.95]) escalates to teacher JudgeGate.
    cases: list[tuple[str, float, float, bool, str]] = []
    # Positive ground-truth items (teacher=True):
    # - 2 false_reject_overturned (student P <= 0.20 FAILED, teacher PASS via adjudicate)
    cases.append(("cursor.execute('SELECT * FROM t WHERE id = ?', (uid,)) # v1", -1.8, -1.6, True, "adjudicate"))
    cases.append(("cursor.execute('SELECT * FROM t WHERE id = %s', (uid,)) # v2", -1.9, -1.7, True, "adjudicate"))
    # - 2 polarity_flip (student AMBIGUOUS with negative log-odds -0.4, teacher PASS)
    cases.append(("stmt = select(User).where(User.id == uid) # v3", -0.5, -0.3, True, "adjudicate"))
    cases.append(("db.query( SQL_CONST, params=[uid] ) # v4", -0.4, -0.2, True, "adjudicate"))
    # - 8 ambiguous_resolved positive (student AMBIGUOUS with positive log-odds 0.6..1.5, teacher PASS)
    for idx in range(8):
        lo = 0.55 + 0.12 * idx
        cases.append((f"safe_query_{idx}(conn, uid)", lo, lo + 0.1, True, "adjudicate"))

    # Negative ground-truth items (teacher=False):
    # - 2 false_accept_caught (student PASSED, teacher FAIL via trigger='always')
    cases.append((f"cursor.execute(f'SELECT * FROM t WHERE id={1}') # sneaky1", 2.2, 2.1, False, "always"))
    cases.append(("cursor.execute('SELECT * FROM t WHERE name=' + name) # sneaky2", 2.0, 2.1, False, "always"))
    # - 2 polarity_flip (student AMBIGUOUS with positive log-odds +0.35, teacher FAIL)
    cases.append(("sql = 'SELECT * FROM ' + tbl; cur.execute(sql, (x,)) # p1", 0.4, 0.3, False, "adjudicate"))
    cases.append(("cur.execute( raw_template % uid ) # p2", 0.35, 0.25, False, "adjudicate"))
    # - 8 ambiguous_resolved negative (student AMBIGUOUS with negative log-odds -0.6..-1.5, teacher FAIL)
    for idx in range(8):
        lo = -(0.55 + 0.12 * idx)
        cases.append((f"unsafe_interp_{idx}(conn, raw_input)", lo, lo - 0.1, False, "adjudicate"))

    with comp:
        for cand, f_lo, r_lo, t_pass, trig in cases:
            s_backend = _make_logodds_backend(f_lo, r_lo)
            t_backend = _make_judge_backend(t_pass)
            student = LogitGate(
                rubric,
                backend=s_backend,
                model="qwen2.5-coder:3b",
                logit_temperature=1.0,
                pass_threshold=0.85,
                fail_threshold=0.20,
            )
            teacher = JudgeGate(
                backend=t_backend,
                model="gemma4:26b",
                trigger=trig,  # type: ignore[arg-type]
            )
            pipe = Cascade([student, teacher], telemetry_sink=comp)
            pipe(cand, telemetry_tags={"suite": "sql_safety"})

    return load_telemetry_records(tel_path), load_distillation_records(dist_path)


def test_distillation_taxonomy_zero_call_recalibration_and_exporters(tmp_path: Path) -> None:
    tel_path = tmp_path / "prod_telemetry.jsonl"
    dist_path = tmp_path / "distill.jsonl"
    tel_recs, dist_recs = _run_synthetic_escalation_stream(tel_path, dist_path)

    assert len(tel_recs) == 24
    assert len(dist_recs) == 24

    kinds = {r.discrepancy_type for r in dist_recs}
    assert kinds == {
        "false_accept_caught",
        "false_reject_overturned",
        "polarity_flip",
        "ambiguous_resolved",
    }

    # Summary detects discrepancies accurately
    summary = summarize_telemetry(tel_path)
    assert summary.total_records == 24
    assert summary.escalated_count == 24
    assert summary.discrepancy_count == 8  # 2 false_accept + 2 false_reject + 4 polarity_flip
    assert summary.discrepancy_rate == round(8 / 24, 4)

    # Native compatibility: DistillationSink JSONL loads directly in load_calibration_dataset()!
    cal_tuples = load_calibration_dataset(dist_path)
    assert len(cal_tuples) == 24
    assert sum(1 for _, _, y in cal_tuples if y) == 12
    assert sum(1 for _, _, y in cal_tuples if not y) == 12

    # Zero-LLM-call profile recalibration from EITHER distill.jsonl OR raw prod_telemetry.jsonl!
    prof_path = tmp_path / "recalibrated_profile.json"
    prof_from_dist = recalibrate_from_telemetry(
        dist_path,
        target_risk_alpha=0.15,
        output_path=prof_path,
    )
    prof_from_tel = recalibrate_from_telemetry(tel_path, target_risk_alpha=0.15)
    assert prof_path.exists()
    assert prof_from_dist.model == "qwen2.5-coder:3b"
    assert prof_from_dist.calibration_set_size == 24
    assert prof_from_dist.pass_threshold is not None
    assert prof_from_dist.fail_threshold is not None
    assert prof_from_dist.pass_threshold > prof_from_dist.fail_threshold
    assert math.isclose(prof_from_dist.binary_temperature, prof_from_tel.binary_temperature)
    assert prof_from_dist.pass_threshold == prof_from_tel.pass_threshold

    # Plugging the zero-call recalibrated profile back into LogitGate works seamlessly
    loaded_prof = CalibrationProfile.load(prof_path)
    recal_gate = LogitGate("SQL safety?", backend="stub://default", profile=loaded_prof)
    assert recal_gate.pass_threshold == loaded_prof.pass_threshold
    assert recal_gate.fail_threshold == loaded_prof.fail_threshold

    # Exporters: sft_jsonl (2x symmetric swaps = 48 rows), dpo_jsonl (48 rows), calibration_jsonl (24 rows)
    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    cal_out = tmp_path / "cal_export.jsonl"
    disc_sft_path = tmp_path / "disc_only_sft.jsonl"

    assert export_distillation_dataset(dist_path, sft_path, format="sft_jsonl") == 48
    assert export_distillation_dataset(tel_path, dpo_path, format="dpo_jsonl") == 48
    assert export_distillation_dataset(dist_path, cal_out, format="calibration_jsonl") == 24
    assert (
        export_distillation_dataset(
            dist_path,
            disc_sft_path,
            format="sft_jsonl",
            include_symmetric_swaps=False,
            discrepancies_only=True,
        )
        == 8
    )

    sft_rows = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines()]
    assert sft_rows[0]["orientation"] == "forward"
    assert sft_rows[1]["orientation"] == "reverse"
    # Forward PASS -> "A", Reverse PASS -> "B" (exact positional-bias cancellation in SFT!)
    assert sft_rows[0]["label"] is True
    assert sft_rows[0]["messages"][-1]["content"] == "A"
    assert sft_rows[1]["messages"][-1]["content"] == "B"

    dpo_rows = [json.loads(line) for line in dpo_path.read_text(encoding="utf-8").splitlines()]
    assert dpo_rows[0]["chosen"] == "A" and dpo_rows[0]["rejected"] == "B"
    assert dpo_rows[1]["chosen"] == "B" and dpo_rows[1]["rejected"] == "A"


def test_locked_upstream_fail_and_fuse_logits_preserve_true_teacher_verdict(tmp_path: Path) -> None:
    dist_path = tmp_path / "edge_distill.jsonl"
    with DistillationSink(dist_path, mode="discrepancies_only", dedupe_by_hash=False) as dsink:
        # Case A: locked_upstream_fail (trigger="on_ambiguous_or_fail", allow_fail_override=False).
        # Student says FAILED (P < 0.20), Teacher says PASS, Cascade stays FAILED,
        # yet DistillationSink MUST capture teacher_passed=True ("false_reject_overturned")!
        s_fail = _make_binary_backend(-2.5, 0.0, -2.5, 0.0)
        j_pass = _make_judge_backend(True)
        p_locked = Cascade(
            [
                LogitGate("Valid?", backend=s_fail, logit_temperature=1.0),
                JudgeGate(backend=j_pass, trigger="on_ambiguous_or_fail", allow_fail_override=False),
            ],
            telemetry_sink=dsink,
        )
        v_locked = p_locked("artifact_locked")
        assert v_locked.passed is False  # Cascade honored monotonic failure lock

        # Case B: fuse_logits=True where a strong positive Tier-1 prior (+1.2, AMBIGUOUS under pass_threshold=0.90)
        # overrides a razor-thin negative Tier-2 judge (-0.05 -> raw JudgeGate FAIL, fused > 0 -> Cascade PASS).
        # DistillationSink MUST record raw teacher_passed=False!
        s_high_ambig = _make_logodds_backend(1.2, 1.2)
        j_split_neg = StubBackend(
            diagnostic_text="Borderline critique notes examining edge cases thoroughly. [VERDICT: FAIL]",
            forward_top_logprobs=[{"token": "A", "logprob": -0.60}, {"token": "B", "logprob": -0.80}],  # fwd +0.20
            reverse_top_logprobs=[{"token": "B", "logprob": -0.85}, {"token": "A", "logprob": -0.55}],  # rev -0.30 -> avg -0.05
        )
        p_fused = Cascade(
            [
                LogitGate(
                    "Valid?",
                    backend=s_high_ambig,
                    logit_temperature=1.0,
                    pass_threshold=0.90,
                    fail_threshold=0.10,
                ),
                JudgeGate(backend=j_split_neg, logit_temperature=1.0, trigger="on_ambiguous_only"),
            ],
            fuse_logits=True,
            prior_weight=0.50,
            telemetry_sink=dsink,
        )
        v_fused = p_fused("artifact_fused")
        assert v_fused.passed is True  # Prior tipped split critique to PASS

    recs = load_distillation_records(dist_path)
    assert len(recs) == 2
    assert recs[0].cascade_passed is False
    assert recs[0].teacher_passed is True
    assert recs[0].label is True
    assert recs[0].discrepancy_type == "false_reject_overturned"

    assert recs[1].cascade_passed is True
    assert recs[1].teacher_passed is False
    assert recs[1].label is False
    assert recs[1].discrepancy_type == "polarity_flip"


def test_recalibrate_validation_errors_and_deduplication(tmp_path: Path) -> None:
    dist_path = tmp_path / "dedupe.jsonl"
    with DistillationSink(dist_path, dedupe_by_hash=True) as dsink:
        pipe = Cascade(
            [
                LogitGate(
                    "Check?",
                    backend=_make_binary_backend(-0.69, -0.69, -0.69, -0.69),
                    logit_temperature=1.0,
                ),
                JudgeGate(backend=_make_judge_backend(True)),
            ],
            telemetry_sink=dsink,
        )
        pipe("same candidate")
        pipe("same candidate")  # duplicate (rubric, candidate) skipped!
    assert len(load_distillation_records(dist_path)) == 1

    with pytest.raises(CalibrationError, match="at least 4"):
        recalibrate_from_telemetry(dist_path)

    # Privacy-omitted records return None cleanly from extract_distillation_record
    mem_hash = MemoryTelemetrySink(privacy="hash_only")
    pipe_hash = Cascade(
        [
            LogitGate(
                "Check?",
                backend=_make_binary_backend(-0.69, -0.69, -0.69, -0.69),
                logit_temperature=1.0,
            ),
            JudgeGate(backend=_make_judge_backend(True)),
        ],
        telemetry_sink=mem_hash,
    )
    pipe_hash("secret text")
    assert extract_distillation_record(mem_hash.records[0]) is None


# ---------------------------------------------------------------------------
# 5. CLI Integration (`eval --telemetry-out`, `rank --telemetry-out`, `telemetry ...`)
# ---------------------------------------------------------------------------


def test_cli_telemetry_subcommands_and_eval_rank_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_tel = tmp_path / "cli_eval.jsonl"
    rc_eval = cli_main(
        [
            "eval",
            "--rubric",
            "Is valid Python?",
            "--candidate",
            "def f(): return 1",
            "--ast-syntax",
            "--backend",
            "stub://default",
            "--telemetry-out",
            str(eval_tel),
            "--json",
        ]
    )
    assert rc_eval == 0
    capsys.readouterr()
    assert len(load_telemetry_records(eval_tel)) == 1
    assert load_telemetry_records(eval_tel)[0].gate_type == "Cascade"

    rank_tel = tmp_path / "cli_rank.jsonl"
    rc_rank = cli_main(
        [
            "rank",
            "--rubric",
            "Better implementation?",
            "--candidate",
            "def a(): return 1",
            "--candidate",
            "def b(): return 2",
            "--verify-winner",
            "--backend",
            "stub://default",
            "--telemetry-out",
            str(rank_tel),
            "--json",
        ]
    )
    assert rc_rank == 0
    capsys.readouterr()
    assert len(load_telemetry_records(rank_tel)) == 1
    assert load_telemetry_records(rank_tel)[0].gate_type == "BestOfNGate"

    # Populate synthetic escalation telemetry and run CLI summary / export / recalibrate
    tel_path = tmp_path / "stream_tel.jsonl"
    dist_path = tmp_path / "stream_dist.jsonl"
    _run_synthetic_escalation_stream(tel_path, dist_path)

    rc_sum = cli_main(["telemetry", "summary", str(tel_path), "--json"])
    assert rc_sum == 0
    sum_json = json.loads(capsys.readouterr().out)
    assert sum_json["total_records"] == 24
    assert sum_json["discrepancy_count"] == 8

    # Human-readable summary also succeeds cleanly
    assert cli_main(["telemetry", "summary", str(tel_path)]) == 0
    assert "LOGITGATE TELEMETRY SUMMARY" in capsys.readouterr().out

    sft_out = tmp_path / "cli_sft.jsonl"
    rc_exp = cli_main(
        [
            "telemetry",
            "export",
            str(tel_path),
            "-o",
            str(sft_out),
            "--format",
            "sft_jsonl",
            "--json",
        ]
    )
    assert rc_exp == 0
    exp_json = json.loads(capsys.readouterr().out)
    assert exp_json["rows_written"] == 48

    prof_out = tmp_path / "cli_profile.json"
    rc_recal = cli_main(
        [
            "telemetry",
            "recalibrate",
            str(dist_path),
            "-o",
            str(prof_out),
            "--alpha",
            "0.15",
            "--json",
        ]
    )
    assert rc_recal == 0
    recal_json = json.loads(capsys.readouterr().out)
    assert recal_json["calibration_set_size"] == 24
    assert prof_out.exists()

    # Verify top-level public exports in logitgate.__all__
    for sym in (
        "TelemetryRecord",
        "DistillationRecord",
        "TelemetrySummary",
        "TelemetrySink",
        "JsonlTelemetrySink",
        "MemoryTelemetrySink",
        "CallbackTelemetrySink",
        "CompositeTelemetrySink",
        "DistillationSink",
        "TelemetryWarning",
        "recalibrate_from_telemetry",
        "export_distillation_dataset",
        "summarize_telemetry",
    ):
        assert sym in logitgate.__all__
        assert hasattr(logitgate, sym)
