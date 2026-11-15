from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from logitgate import (
    AdaptiveConformalController,
    CalibrationProfile,
    Cascade,
    ConfigurationError,
    DecisionStatus,
    JudgeGate,
    LogitBackend,
    LogitGate,
    MultiRubricGate,
    RollingDriftMonitor,
    StubBackend,
    VerdictCache,
    compute_ks_two_sample,
    compute_psi,
    compute_verdict_cache_key,
    detect_telemetry_drift,
)
from logitgate.cli import main as cli_main
from logitgate.models import LayerTrace
from logitgate.telemetry import JsonlTelemetrySink, TelemetryRecord, hash_candidate


class _ControlledLogitBackend(LogitBackend):
    """Deterministic backend mapping candidate strings to exact symmetric log-odds."""

    def __init__(self, mapping: dict[str, float], default_lo: float = 0.0) -> None:
        self.mapping = mapping
        self.default_lo = default_lo
        self.calls = 0

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.calls += 1
        lo = self.default_lo
        for key, val in self.mapping.items():
            if key in prompt:
                lo = val
                break
        mapping_idx = prompt.rfind("[DECISION MAPPING]")
        tail = prompt[mapping_idx:] if mapping_idx != -1 else prompt[-300:]
        is_reverse = False
        for line in tail.splitlines():
            s = line.strip()
            if s.startswith("B:") and ("complies" in s.lower() or "satisfies" in s.lower()):
                is_reverse = True
                break

        p_pass = 1.0 / (1.0 + math.exp(-lo))
        p_fail = 1.0 - p_pass
        lp_pass = math.log(max(p_pass, 1e-12))
        lp_fail = math.log(max(p_fail, 1e-12))
        if is_reverse:
            return [{"token": "B", "logprob": lp_pass}, {"token": "A", "logprob": lp_fail}]
        return [{"token": "A", "logprob": lp_pass}, {"token": "B", "logprob": lp_fail}]


def _make_record(
    *,
    log_odds: float,
    pos_bias: float = 0.05,
    escalated: bool = False,
    l1_status: DecisionStatus = DecisionStatus.PASSED,
    teacher_passed: bool | None = None,
    candidate_text: str | None = "secret candidate payload",
) -> TelemetryRecord:
    prob = round(1.0 / (1.0 + math.exp(-log_odds)), 4)
    traces = [
        LayerTrace(
            layer_name="LogitGate",
            status=l1_status,
            latency_ms=12.0,
            calibrated_prob=prob,
            forward_log_odds=round(log_odds + pos_bias, 4),
            reverse_log_odds=round(log_odds - pos_bias, 4),
            positional_bias_magnitude=pos_bias,
            log_odds_pass=log_odds,
            tokens_generated=0,
        )
    ]
    if teacher_passed is not None:
        traces.append(
            LayerTrace(
                layer_name="JudgeGate",
                status=DecisionStatus.PASSED if teacher_passed else DecisionStatus.FAILED,
                latency_ms=150.0,
                tokens_generated=42,
                diagnostic_report="[VERDICT: PASS]" if teacher_passed else "[VERDICT: FAIL]",
            )
        )
    final_passed = teacher_passed if teacher_passed is not None else (l1_status == DecisionStatus.PASSED)
    final_status = (
        (DecisionStatus.PASSED if teacher_passed else DecisionStatus.FAILED)
        if teacher_passed is not None
        else l1_status
    )
    raw_cand = candidate_text or ""
    return TelemetryRecord(
        gate_name="Cascade",
        gate_type="Cascade",
        model="qwen2.5-coder:3b",
        backend="ollama://localhost:11434",
        teacher_model="gemma4:26b" if teacher_passed is not None else None,
        teacher_passed=teacher_passed,
        rubric="Is the code safe?",
        candidate=candidate_text,
        candidate_sha256=hash_candidate(raw_cand),
        candidate_length=len(raw_cand),
        passed=final_passed,
        status=final_status,
        probability=prob,
        triggered_layer="JudgeGate" if teacher_passed is not None else "LogitGate",
        escalated=escalated,
        fast_pathed=not escalated,
        total_latency_ms=162.0 if teacher_passed is not None else 12.0,
        total_tokens_generated=42 if teacher_passed is not None else 0,
        traces=traces,
    )


def test_psi_and_ks_mathematical_invariants() -> None:
    ref = [-4.0 + 0.1 * i for i in range(81)]
    assert compute_psi(ref, list(ref)) == 0.0
    d_same, p_same = compute_ks_two_sample(ref, list(ref))
    assert d_same == 0.0
    assert p_same == 1.0

    # Mild shift vs severe shift
    mild = [x + 0.6 for x in ref]
    severe = [x + 4.5 for x in ref]
    psi_mild = compute_psi(ref, mild)
    psi_sev = compute_psi(ref, severe)
    assert 0.0 < psi_mild < psi_sev
    assert psi_sev > 0.25

    d_mild, p_mild = compute_ks_two_sample(ref, mild)
    d_sev, p_sev = compute_ks_two_sample(ref, severe)
    assert 0.0 < d_mild < d_sev <= 1.0
    assert 0.0 <= p_sev < p_mild <= 1.0
    assert p_sev < 1e-4

    # Constant reference array edge case (zero variance)
    const_ref = [2.0] * 30
    assert compute_psi(const_ref, [2.0] * 30) == 0.0
    assert compute_psi(const_ref, [-2.0] * 30) > 0.25
    d_const, p_const = compute_ks_two_sample(const_ref, [-2.0] * 30)
    assert d_const == 1.0
    assert p_const == 0.0

    # Validation errors
    with pytest.raises(ConfigurationError):
        compute_psi([], [1.0])
    with pytest.raises(ConfigurationError):
        compute_psi([True, False], [1.0, 0.0])
    with pytest.raises(ConfigurationError):
        compute_psi([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(ConfigurationError):
        compute_psi([1.0, 2.0], [1.0, 2.0], num_bins=1)
    with pytest.raises(ConfigurationError):
        compute_ks_two_sample("bad", [1.0])  # type: ignore[arg-type]


def test_detect_telemetry_drift_signals_and_teacher_disagreement() -> None:
    ref = [
        _make_record(
            log_odds=(3.0 if i % 2 == 0 else -3.0),
            pos_bias=0.08,
            escalated=(i % 10 == 0),
            l1_status=DecisionStatus.AMBIGUOUS if i % 10 == 0 else (
                DecisionStatus.PASSED if i % 2 == 0 else DecisionStatus.FAILED
            ),
            teacher_passed=(i % 2 == 0) if i % 10 == 0 else None,
        )
        for i in range(40)
    ]
    same_report = detect_telemetry_drift(ref, list(ref))
    assert not same_report.drift_detected
    assert same_report.severity == "none"
    assert same_report.recommended_action == "ok"

    # Shifted covariate regime with compressed logits, higher bias, and teacher disagreements
    drifted = [
        _make_record(
            log_odds=0.25 + 0.02 * (i % 5),
            pos_bias=0.85,
            escalated=True,
            l1_status=DecisionStatus.PASSED,
            teacher_passed=False,
        )
        for i in range(40)
    ]
    rep = detect_telemetry_drift(ref, drifted)
    assert rep.drift_detected
    assert rep.severity == "severe"
    assert rep.recommended_action == "widen_band_or_recalibrate"
    assert rep.psi_log_odds >= 0.25
    assert rep.ks_log_odds_p_value < 0.01
    assert rep.pos_bias_delta > 0.5
    assert rep.teacher_disagreement_rate == 1.0
    assert len(rep.reasons) >= 3


def test_rolling_drift_monitor_auto_baseline_privacy_and_aci_boost() -> None:
    aci = AdaptiveConformalController(
        pass_threshold=0.80,
        fail_threshold=0.20,
        target_risk_alpha=0.05,
    )
    fired: list[Any] = []
    monitor = RollingDriftMonitor(
        baseline_window=15,
        current_window=15,
        check_every=5,
        min_current_samples=10,
        on_drift=fired.append,
        adaptive_controller=aci,
        drift_boost_theta=1.25,
        boost_cooldown_checks=2,
    )
    assert not monitor.baseline_ready

    # Emit 15 stationary records -> freezes baseline, scrubs candidate text
    for i in range(15):
        monitor.emit(_make_record(log_odds=3.5 if i % 2 == 0 else -3.5, candidate_text="PII 123-45-6789"))
    assert monitor.baseline_ready
    assert monitor.latest_report is None
    assert aci.theta == 0.0

    # Emit 15 severely shifted records -> triggers checks at #10 and #15
    for _ in range(15):
        monitor.emit(_make_record(log_odds=0.05, pos_bias=0.9, escalated=True, l1_status=DecisionStatus.AMBIGUOUS))

    assert monitor.latest_report is not None
    assert monitor.latest_report.drift_detected
    assert len(fired) >= 1
    assert aci.theta == pytest.approx(1.25)
    assert aci.summary().drift_boosts_applied == 1


def test_adaptive_conformal_adversarial_finite_sample_risk_bound() -> None:
    """Deterministic adversarial sequence where static thresholds suffer 100% selective error."""
    profile = CalibrationProfile(
        model="stub",
        binary_temperature=1.0,
        pass_threshold=0.80,
        fail_threshold=0.20,
        target_risk_alpha=0.05,
    )
    backend = _ControlledLogitBackend(
        {
            "adv_false_pass": 2.2,   # sigmoid(2.2) = 0.9002 >= 0.80, but ground_truth = False!
            "adv_false_fail": -2.2,  # sigmoid(-2.2) = 0.0998 <= 0.20, but ground_truth = True!
            "clean_pass": 4.0,       # sigmoid(4.0) = 0.9820, ground_truth = True
        }
    )

    static_gate = LogitGate("Is valid?", backend=backend, profile=profile, logit_temperature=1.0)
    aci = AdaptiveConformalController(
        profile=profile,
        target_risk_alpha=0.05,
        step_size=1.5,
        abstain_theta=6.0,
    )
    adaptive_gate = LogitGate(
        "Is valid?",
        backend=backend,
        adaptive_controller=aci,
        logit_temperature=1.0,
    )

    # Exact parity at theta = 0.0 before any observations
    assert aci.current_thresholds() == (0.20, 0.80, False)

    # 300 adversarial inputs designed to fool base thresholds 100% of the time
    total_adv = 300
    static_errors = 0
    adaptive_errors = 0
    for t in range(total_adv):
        cand = "adv_false_pass" if t % 2 == 0 else "adv_false_fail"
        truth = t % 2 != 0

        sv = static_gate.evaluate(cand)
        if sv.status != DecisionStatus.AMBIGUOUS and sv.passed != truth:
            static_errors += 1

        av = adaptive_gate.evaluate(cand)
        if av.status != DecisionStatus.AMBIGUOUS and av.passed != truth:
            adaptive_errors += 1
        aci.observe(av, ground_truth=truth)

    static_risk = static_errors / total_adv
    adaptive_risk = adaptive_errors / total_adv
    # Theoretical finite-sample bound: alpha + (abstain_theta + step_size) / (step_size * T)
    theoretical_bound = 0.05 + (6.0 + 1.5) / (1.5 * total_adv)
    assert static_risk == 1.0
    assert adaptive_risk <= theoretical_bound
    assert aci.summary().empirical_selective_risk_overall == pytest.approx(adaptive_risk, abs=1e-4)

    # Now feed 120 clean inputs -> theta decays smoothly and resumes fast-pathing!
    fast_paths_in_recovery = 0
    for _ in range(120):
        av = adaptive_gate.evaluate("clean_pass")
        if av.status == DecisionStatus.PASSED:
            fast_paths_in_recovery += 1
        aci.observe(av, ground_truth=True)
    assert fast_paths_in_recovery > 40
    assert aci.theta < 2.2


@pytest.mark.asyncio
async def test_cascade_audit_horvitz_thompson_zero_label_adaptation() -> None:
    """Cascade shadow-audits fast-path decisions and adapts L1 thresholds with zero human labels."""
    backend = _ControlledLogitBackend({"tricky_item": 2.0, "easy_item": 5.5})
    aci = AdaptiveConformalController(
        pass_threshold=0.80,
        fail_threshold=0.20,
        target_risk_alpha=0.05,
        step_size=1.5,
        audit_sample_rate=0.50,
        override_on_audit=True,
        seed=7,
    )
    l1 = LogitGate(
        "Does candidate satisfy policy?",
        backend=backend,
        adaptive_controller=aci,
        logit_temperature=1.0,
    )
    # Teacher rejects tricky_item!
    teacher_stub = StubBackend(diagnostic_text="Violation found.\n[VERDICT: FAIL]")
    l2 = JudgeGate(backend=teacher_stub, trigger="on_ambiguous_only", logit_temperature=1.0)
    cascade = Cascade([l1, l2])

    # Run tricky_item repeatedly: first shadow-audit catches L1 false-pass, overrides verdict to FAIL,
    # and jumps theta up by 1.5 * (1/0.5 - 0.05) = +2.925 so subsequent tricky_items escalate cleanly!
    verdicts = [cascade.evaluate("tricky_item") for _ in range(8)]
    assert any(not v.passed for v in verdicts)
    assert aci.summary().audited_fast_path_count >= 1
    assert aci.summary().observed_l1_errors >= 1
    assert aci.theta > 2.0
    # Once theta > 2.0, tricky_item (log_odds=2.0) is escalated as AMBIGUOUS and caught 100% of the time!
    for _ in range(4):
        v_async = await cascade.aevaluate("tricky_item")
        assert not v_async.passed
        assert v_async.status == DecisionStatus.FAILED

    # MultiRubricGate forwards adaptive_controller cleanly as well
    aci_multi = AdaptiveConformalController(
        pass_threshold=0.80,
        fail_threshold=0.20,
        audit_sample_rate=1.0,
        seed=1,
    )
    mrg = MultiRubricGate(
        ["Rule 1?", "Rule 2?"],
        backend=backend,
        adaptive_controller=aci_multi,
        logit_temperature=1.0,
    )
    assert mrg.adaptive_controller is aci_multi
    m_cascade = Cascade([mrg, l2])
    mv = m_cascade.evaluate("tricky_item")
    assert not mv.passed
    assert aci_multi.summary().audited_fast_path_count == 1


def test_verdict_cache_dynamic_adaptive_threshold_isolation() -> None:
    backend = _ControlledLogitBackend({"borderline_pass": 1.8})  # P = 0.8581
    cache = VerdictCache()
    aci = AdaptiveConformalController(
        pass_threshold=0.80,
        fail_threshold=0.20,
        target_risk_alpha=0.05,
        step_size=2.0,
    )
    gate = LogitGate(
        "Check safety",
        backend=backend,
        adaptive_controller=aci,
        logit_temperature=1.0,
        cache=cache,
    )

    k_before = compute_verdict_cache_key(gate, "borderline_pass")
    v1 = gate.evaluate("borderline_pass")
    assert v1.status == DecisionStatus.PASSED
    calls_after_first = backend.calls

    # Cache hit when theta unchanged
    v1_cached = gate.evaluate("borderline_pass")
    assert v1_cached.status == DecisionStatus.PASSED
    assert backend.calls == calls_after_first

    # Observe a false pass -> theta increases by 2.0 * 0.95 = +1.90 -> pass_lo = 1.386 + 1.90 = 3.286 > 1.8!
    aci.observe(v1, ground_truth=False)
    k_after = compute_verdict_cache_key(gate, "borderline_pass")
    assert k_after != k_before

    # Must NOT serve stale cached PASSED verdict; re-evaluates and abstains to AMBIGUOUS!
    v2 = gate.evaluate("borderline_pass")
    assert v2.status == DecisionStatus.AMBIGUOUS
    assert backend.calls > calls_after_first

    exported = aci.to_profile()
    assert exported.pass_threshold is not None and exported.pass_threshold > 0.80
    aci.reset()
    assert aci.theta == 0.0
    assert compute_verdict_cache_key(gate, "borderline_pass") == k_before


def test_concurrent_aci_and_drift_monitor_thread_safety() -> None:
    backend = _ControlledLogitBackend({"ok": 3.0, "bad": 2.0})
    aci = AdaptiveConformalController(
        pass_threshold=0.80,
        fail_threshold=0.20,
        target_risk_alpha=0.05,
        step_size=0.5,
    )
    monitor = RollingDriftMonitor(
        baseline_window=20,
        current_window=20,
        check_every=10,
        adaptive_controller=aci,
        drift_boost_theta=0.25,
    )
    gate = LogitGate(
        "Thread safety?",
        backend=backend,
        adaptive_controller=aci,
        telemetry_sink=monitor,
        logit_temperature=1.0,
    )

    def _worker(idx: int) -> DecisionStatus:
        cand = "ok" if idx % 3 != 0 else "bad"
        truth = idx % 3 != 0
        v = gate.evaluate(cand)
        aci.observe(v, ground_truth=truth)
        return v.status

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(_worker, range(120)))

    assert len(statuses) == 120
    assert aci.summary().total_steps == 120
    assert monitor.baseline_ready


def test_cli_telemetry_drift_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref_path = tmp_path / "ref.jsonl"
    cur_same_path = tmp_path / "cur_same.jsonl"
    cur_drift_path = tmp_path / "cur_drift.jsonl"

    with JsonlTelemetrySink(ref_path) as s_ref, JsonlTelemetrySink(cur_same_path) as s_same:
        for i in range(25):
            rec = _make_record(log_odds=3.2 if i % 2 == 0 else -3.2)
            s_ref.emit(rec)
            s_same.emit(rec)

    with JsonlTelemetrySink(cur_drift_path) as s_drift:
        for _ in range(25):
            s_drift.emit(
                _make_record(
                    log_odds=0.1,
                    pos_bias=0.95,
                    escalated=True,
                    l1_status=DecisionStatus.PASSED,
                    teacher_passed=False,
                )
            )

    rc_ok = cli_main(
        [
            "telemetry",
            "drift",
            str(cur_same_path),
            "--reference",
            str(ref_path),
            "--fail-on-drift",
            "--json",
        ]
    )
    assert rc_ok == 0
    payload_ok = json.loads(capsys.readouterr().out)
    assert payload_ok["drift_detected"] is False

    rc_drift = cli_main(
        [
            "telemetry",
            "drift",
            str(cur_drift_path),
            "--reference",
            str(ref_path),
            "--fail-on-drift",
        ]
    )
    assert rc_drift == 1
    human_out = capsys.readouterr().out
    assert "LOGITGATE TELEMETRY DRIFT REPORT" in human_out
    assert "SEVERE" in human_out
