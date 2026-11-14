from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest

from logitgate.backends.base import GenerativeBackend, LogitBackend, StubBackend
from logitgate.calibration.profile import (
    CalibrationProfile,
    ChoiceConformalThresholds,
    ScoreConformalThresholds,
    afit_calibration_profile,
    calibrate_choice_thresholds,
    calibrate_score_thresholds,
    fit_calibration_profile,
    resolve_calibration_profile,
)
from logitgate.cli import main
from logitgate.config import configure, reset_config
from logitgate.debiasing.permutation import combine_symmetric_log_odds
from logitgate.exceptions import (
    CalibrationError,
    CalibrationWarning,
    ConfigurationError,
)
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import (
    ChoiceGate,
    LogitGate,
    MultiRubricGate,
    ScoreGate,
)
from logitgate.models import DecisionStatus
from logitgate.snapshot import VerdictSnapshot


@pytest.fixture(autouse=True)
def _clean_global_config() -> Any:
    reset_config()
    yield
    reset_config()


class _PromptKeyedLogitBackend(LogitBackend, GenerativeBackend):
    """Deterministic or per-prompt log-odds backend with concurrency and call-count tracking."""

    def __init__(
        self,
        *,
        model: str = "custom-byo-model",
        prefers_warmup: bool = False,
        delay: float = 0.0,
        responder: Any | None = None,
    ) -> None:
        self.model = model
        self.prefers_sequential_prefix_warmup = prefers_warmup
        self.delay = delay
        self.responder = responder
        self.active_calls = 0
        self.max_concurrent_calls = 0
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.aclosed = False

    def _default_logprobs(
        self,
        prompt: str,
        allowed_tokens: list[str] | None,
    ) -> list[dict[str, Any]]:
        if self.responder is not None:
            return list(self.responder(prompt, allowed_tokens))
        if allowed_tokens and any(tok.isdigit() for tok in allowed_tokens):
            return [{"token": "5", "logprob": -0.1}, {"token": "1", "logprob": -3.5}]
        if allowed_tokens and len(allowed_tokens) > 2:
            return [
                {"token": allowed_tokens[0], "logprob": -0.1},
                {"token": allowed_tokens[1], "logprob": -3.5},
            ]
        # Positive log-odds toward semantic PASS: forward prefers 'A', reverse prefers 'B'
        is_reverse = "A: The candidate artifact violates" in prompt
        if "invalid" in prompt.lower() or "fail_case" in prompt.lower():
            win, lose = ("A", "B") if is_reverse else ("B", "A")
        else:
            win, lose = ("B", "A") if is_reverse else ("A", "B")
        return [{"token": win, "logprob": -0.15}, {"token": lose, "logprob": -2.4}]

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append({"prompt": prompt, "allowed_tokens": allowed_tokens, "kwargs": kwargs})
        return self._default_logprobs(prompt, allowed_tokens)

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        try:
            if self.delay > 0.0:
                await asyncio.sleep(self.delay)
            self.calls.append({"prompt": prompt, "allowed_tokens": allowed_tokens, "kwargs": kwargs})
            return self._default_logprobs(prompt, allowed_tokens)
        finally:
            self.active_calls -= 1

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, int]:
        return "- Criterion 1: The candidate satisfies all rubric requirements.\n[VERDICT: PASS]", 14

    async def agenerate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, int]:
        return "- Criterion 1: The candidate satisfies all rubric requirements.\n[VERDICT: PASS]", 14

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed = True


def _synthetic_log_odds_draw(
    rng: random.Random,
    n: int,
    *,
    scale: float,
    noise_rate: float = 0.08,
) -> list[tuple[float, float, int]]:
    """Generates balanced synthetic (forward_lo, reverse_lo, y) samples at logit scale `scale`."""
    half = n // 2
    out: list[tuple[float, float, int]] = []
    for y in (0, 1):
        for _ in range(half):
            sign = 1.0 if y == 1 else -1.0
            if rng.random() < noise_rate:
                sign = -sign
            z_true = sign * rng.uniform(0.3, 1.8) * scale
            token_bias = rng.uniform(-0.4, 0.4)
            f_lo = z_true + token_bias
            r_lo = z_true - token_bias
            out.append((f_lo, r_lo, y))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# 1. Joint Temperature + Conformal Invariance & T_old -> T_new Bug Fix Proof
# ---------------------------------------------------------------------------


def test_joint_profile_conformal_empirical_risk_attenuated_and_overconfident() -> None:
    rng = random.Random(42)

    # Regime A: Attenuated compact model (small raw logits -> T_opt < 1.0)
    cal_att = _synthetic_log_odds_draw(rng, 100, scale=0.25, noise_rate=0.06)
    test_att = _synthetic_log_odds_draw(rng, 400, scale=0.25, noise_rate=0.06)

    profile_att = CalibrationProfile.from_log_odds_samples(
        cal_att, target_risk_alpha=0.10, model="attenuated-3b"
    )
    assert profile_att.guarantee_holds is True
    assert profile_att.binary_temperature < 1.0
    assert profile_att.pass_threshold is not None
    assert profile_att.fail_threshold is not None
    assert profile_att.pass_threshold > profile_att.fail_threshold

    # Verify out-of-sample Layer-1 selective error stays within alpha (+ tiny finite-sample variance margin)
    errors = 0
    for f_lo, r_lo, y in test_att:
        p = round(
            combine_symmetric_log_odds(f_lo, r_lo, logit_temperature=profile_att.binary_temperature),
            4,
        )
        if (p >= profile_att.pass_threshold and y == 0) or (p <= profile_att.fail_threshold and y == 1):
            errors += 1
    assert errors / len(test_att) <= 0.12

    # Prove stale T_old=1.0 thresholds applied on top of T_new (< 1.0) inflate Layer-1 errors
    # Pick a sample set where wrong items cluster near z=+0.15 so P(z; T=1.0) is ~0.537 but P(z; T=0.1) is ~0.818
    stale_samples = [
        (0.15, 0.15, 0) if i % 5 == 0 else (-0.45, -0.45, 0)
        for i in range(40)
    ] + [
        (0.45, 0.45, 1) for _ in range(40)
    ]
    fixed_prof = CalibrationProfile.from_log_odds_samples(stale_samples, target_risk_alpha=0.10)
    stale_p_neg_max = max(combine_symmetric_log_odds(f, r, logit_temperature=1.0) for f, r, y in stale_samples if y == 0)
    stale_tau_accept = round(stale_p_neg_max + 0.0001, 4)  # ~0.5376 at T=1.0
    # Under T_new, z=0.05 (ambiguous!) sits below stale_tau_accept at T=1.0 (0.5125 < 0.5376)
    # yet exceeds stale_tau_accept at T_new (contradicting conformal calibration!)
    ambig_at_t1 = round(combine_symmetric_log_odds(0.05, 0.05, logit_temperature=1.0), 4)
    inflated_at_tnew = round(
        combine_symmetric_log_odds(0.05, 0.05, logit_temperature=fixed_prof.binary_temperature),
        4,
    )
    assert ambig_at_t1 < stale_tau_accept
    assert inflated_at_tnew > stale_tau_accept
    assert fixed_prof.pass_threshold is not None
    # Whereas unified profile matches scale and keeps z=0.05 safely in AMBIGUOUS band!
    assert inflated_at_tnew < fixed_prof.pass_threshold

    # Regime B: Overconfident reasoning/high-logit model (large logits -> T_opt > 1.0)
    cal_over = _synthetic_log_odds_draw(rng, 100, scale=3.5, noise_rate=0.06)
    test_over = _synthetic_log_odds_draw(rng, 400, scale=3.5, noise_rate=0.06)
    profile_over = CalibrationProfile.from_log_odds_samples(
        cal_over, target_risk_alpha=0.10, model="overconfident-model"
    )
    assert profile_over.guarantee_holds is True
    assert profile_over.binary_temperature > 1.0
    assert profile_over.pass_threshold is not None
    assert profile_over.fail_threshold is not None

    errors_over = 0
    for f_lo, r_lo, y in test_over:
        p = round(
            combine_symmetric_log_odds(f_lo, r_lo, logit_temperature=profile_over.binary_temperature),
            4,
        )
        if (p >= profile_over.pass_threshold and y == 0) or (p <= profile_over.fail_threshold and y == 1):
            errors_over += 1
    assert errors_over / len(test_over) <= 0.12


def test_from_log_odds_samples_small_and_invalid_inputs() -> None:
    rng = random.Random(7)
    small_samples = _synthetic_log_odds_draw(rng, 16, scale=1.0, noise_rate=0.0)
    with pytest.warns(CalibrationWarning, match="Conformal guarantee does not hold"):
        prof = CalibrationProfile.from_log_odds_samples(small_samples, target_risk_alpha=0.05)
    assert prof.guarantee_holds is False
    assert prof.calibration_set_size == 16
    assert prof.pass_threshold is not None and prof.fail_threshold is not None

    temp_only = CalibrationProfile.from_log_odds_samples(
        small_samples[:6], target_risk_alpha=None, multiclass_temperature=1.2
    )
    assert temp_only.pass_threshold is None
    assert temp_only.fail_threshold is None
    assert temp_only.effective_multiclass_temperature == pytest.approx(1.2)

    with pytest.raises(CalibrationError):
        CalibrationProfile.from_log_odds_samples(small_samples[:8], target_risk_alpha=0.05)
    with pytest.raises(CalibrationError):
        CalibrationProfile.from_log_odds_samples([(1.0, 1.0, 1)] * 12, target_risk_alpha=0.05)


# ---------------------------------------------------------------------------
# 2. CalibrationProfile Validation, Alias Compatibility, & Atomic I/O
# ---------------------------------------------------------------------------


def test_calibration_profile_save_load_aliases_and_validation(tmp_path: Path) -> None:
    prof = CalibrationProfile(
        model="qwen2.5:7b",
        backend="ollama://localhost:11434",
        binary_temperature=0.45,
        multiclass_temperature=0.95,
        pass_threshold=0.88,
        fail_threshold=0.14,
        score_threshold=4.1,
        score_fail_threshold=2.3,
        choice_min_confidence=0.72,
        choice_min_margin=0.25,
        target_risk_alpha=0.05,
        empirical_escalation_rate=0.18,
        calibration_set_size=80,
        guarantee_holds=True,
        min_samples_per_class=39,
        num_positive=40,
        num_negative=40,
    )
    assert prof.optimal_temperature == pytest.approx(0.45)
    assert prof.tau_accept == pytest.approx(0.88)
    assert prof.tau_reject == pytest.approx(0.14)
    assert prof.effective_multiclass_temperature == pytest.approx(0.95)

    out_path = tmp_path / "nested" / "profile.json"
    saved = prof.save(out_path)
    assert saved == out_path
    loaded = CalibrationProfile.load(str(out_path))
    assert loaded == prof

    # Overwrite cleanly
    prof2 = CalibrationProfile(binary_temperature=0.5, pass_threshold=0.9, fail_threshold=0.1)
    prof2.save(out_path)
    assert CalibrationProfile.load(out_path).binary_temperature == pytest.approx(0.5)

    # Legacy JSON keys compatibility (even with pass_threshold=null + tau_accept alias)
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps({
            "optimal_temperature": 0.42,
            "pass_threshold": None,
            "tau_accept": 0.87,
            "tau_reject": 0.19,
            "target_risk_alpha": 0.1,
            "calibration_set_size": 40,
        }),
        encoding="utf-8",
    )
    legacy = CalibrationProfile.load(legacy_path)
    assert legacy.binary_temperature == pytest.approx(0.42)
    assert legacy.pass_threshold == pytest.approx(0.87)
    assert legacy.fail_threshold == pytest.approx(0.19)

    # Newer schema version warns cleanly and ignores unknown fields
    future_path = tmp_path / "future.json"
    future_path.write_text(
        json.dumps({"schema_version": 99, "binary_temperature": 1.1, "unknown_future_field": [1, 2]}),
        encoding="utf-8",
    )
    with pytest.warns(CalibrationWarning, match="schema_version=99"):
        future = CalibrationProfile.load(future_path)
    assert future.binary_temperature == pytest.approx(1.1)


def test_calibration_profile_rejects_corrupt_invalid_and_symlinks(tmp_path: Path) -> None:
    # Reject bools on numeric fields
    with pytest.raises(CalibrationError):
        CalibrationProfile(binary_temperature=True)
    with pytest.raises(CalibrationError):
        CalibrationProfile(pass_threshold=False)
    with pytest.raises(CalibrationError):
        CalibrationProfile(calibration_set_size=True)
    with pytest.raises(CalibrationError):
        CalibrationProfile.model_validate({"binary_temperature": True})

    # Reject invalid numeric ranges / NaN / Inf / inverted cutoffs
    for bad in (
        {"binary_temperature": 0.0},
        {"binary_temperature": -0.5},
        {"binary_temperature": float("nan")},
        {"multiclass_temperature": float("inf")},
        {"pass_threshold": 0.5, "fail_threshold": 0.5},
        {"pass_threshold": 0.3, "fail_threshold": 0.7},
        {"score_threshold": 2.0, "score_fail_threshold": 3.0},
        {"choice_min_confidence": 0.0},
        {"choice_min_confidence": 1.2},
        {"target_risk_alpha": 1.0},
        {"calibration_set_size": -1},
        {"schema_version": 0},
    ):
        with pytest.raises(CalibrationError):
            CalibrationProfile(**bad)

    # Reject symlinks, missing files, wrong JSON types, NaN JSON, oversized files
    real_file = tmp_path / "real.json"
    CalibrationProfile(binary_temperature=1.0).save(real_file)
    link_file = tmp_path / "link.json"
    link_file.symlink_to(real_file)

    with pytest.raises(CalibrationError, match="symlink"):
        CalibrationProfile.load(link_file)
    with pytest.raises(CalibrationError, match="symlink"):
        CalibrationProfile(binary_temperature=1.0).save(link_file)
    with pytest.raises(CalibrationError):
        CalibrationProfile.load(tmp_path / "missing.json")
    with pytest.raises(CalibrationError):
        CalibrationProfile.load(tmp_path)  # directory, not regular file
    with pytest.raises(CalibrationError):
        CalibrationProfile.load("")
    with pytest.raises(CalibrationError):
        CalibrationProfile(binary_temperature=1.0).save("   ")

    bad_json = tmp_path / "list.json"
    bad_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CalibrationError):
        CalibrationProfile.load(bad_json)

    nan_json = tmp_path / "nan.json"
    nan_json.write_text('{"binary_temperature": NaN}', encoding="utf-8")
    with pytest.raises(CalibrationError):
        CalibrationProfile.load(nan_json)

    huge_json = tmp_path / "huge.json"
    huge_json.write_text('{"binary_temperature": 1.0, "pad": "' + ("x" * 1_100_000) + '"}', encoding="utf-8")
    with pytest.raises(CalibrationError, match="exceeds maximum size"):
        CalibrationProfile.load(huge_json)

    assert resolve_calibration_profile(None) is None
    with pytest.raises(ConfigurationError):
        resolve_calibration_profile("   ")
    with pytest.raises(ConfigurationError):
        resolve_calibration_profile(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Selective Conformal Calibrators for ChoiceGate & ScoreGate
# ---------------------------------------------------------------------------


def test_calibrate_choice_thresholds_guarantees_and_gate_wiring() -> None:
    # 1. Small sample (10..18) at alpha=0.05 -> guarantee_holds=False with warning, uniform-tie floor respected
    small_probs = [{"A": 0.90, "B": 0.05, "C": 0.05} for _ in range(12)]
    small_gt = ["A"] * 12
    with pytest.warns(CalibrationWarning, match="need >= 19"):
        ct_small = calibrate_choice_thresholds(small_probs, small_gt, target_risk_alpha=0.05)
    assert isinstance(ct_small, ChoiceConformalThresholds)
    assert ct_small.guarantee_holds is False
    assert ct_small.min_confidence == pytest.approx(0.3334)  # strictly > 1/3

    # 2. Sufficient samples (30) at alpha=0.10 -> guarantee_holds=True and controls out-of-sample error
    rng = random.Random(101)
    def _sample_choice(n: int) -> tuple[list[dict[str, float]], list[str]]:
        plist: list[dict[str, float]] = []
        glist: list[str] = []
        for _ in range(n):
            true_k = rng.choice(("A", "B", "C", "D"))
            if rng.random() < 0.85:
                top_p = rng.uniform(0.75, 0.95)
                pred_k = true_k
            else:
                top_p = rng.uniform(0.35, 0.65)
                pred_k = rng.choice([k for k in ("A", "B", "C", "D") if k != true_k])
            rem = (1.0 - top_p) / 3.0
            d = {k: rem for k in ("A", "B", "C", "D")}
            d[pred_k] = top_p
            plist.append(d)
            glist.append(true_k)
        return plist, glist

    cal_p, cal_gt = _sample_choice(50)
    test_p, test_gt = _sample_choice(200)
    ct = calibrate_choice_thresholds(cal_p, cal_gt, target_risk_alpha=0.10, calibrate_margin=True)
    assert ct.guarantee_holds is True
    assert ct.min_margin is not None
    assert ct.min_confidence <= 0.70

    selective_errors = 0
    for probs, gt in zip(test_p, test_gt):
        best_k, max_p = max(probs.items(), key=lambda kv: kv[1])
        sp = sorted(probs.values(), reverse=True)
        marg = round(sp[0] - sp[1], 4)
        if round(max_p, 4) >= ct.min_confidence and marg >= ct.min_margin and best_k != gt:
            selective_errors += 1
    assert selective_errors / len(test_p) <= 0.12

    # 3. Wire into Profile & ChoiceGate
    base_prof = CalibrationProfile(model="stub", binary_temperature=0.5, multiclass_temperature=0.9)
    wired_prof = base_prof.with_choice_thresholds(ct)
    assert wired_prof.choice_min_confidence == ct.min_confidence
    assert wired_prof.choice_min_margin == ct.min_margin

    weak_backend = StubBackend(
        forward_top_logprobs=[{"token": "A", "logprob": -0.7}, {"token": "B", "logprob": -0.85}],
        reverse_top_logprobs=[{"token": "B", "logprob": -0.7}, {"token": "A", "logprob": -0.85}],
    )
    gate = ChoiceGate(
        "Route ticket",
        {"A": "Billing", "B": "Support", "C": "Security", "D": "Infra"},
        backend=weak_backend,
        profile=wired_prof,
    )
    assert gate.min_confidence == ct.min_confidence
    assert gate.min_margin == ct.min_margin
    v_amb = gate.evaluate("Ambiguous ticket")
    assert v_amb.status == DecisionStatus.AMBIGUOUS

    # 4. Saturation & invalid inputs
    sat_p = [{"A": 1.0, "B": 0.0}] + [{"A": 0.9, "B": 0.1} for _ in range(19)]
    sat_gt = ["B"] + ["A"] * 19
    with pytest.warns(CalibrationWarning, match="P=1.0"):
        ct_sat = calibrate_choice_thresholds(sat_p, sat_gt, target_risk_alpha=0.05)
    assert ct_sat.guarantee_holds is False

    for bad_args in (
        (cal_p[:9], cal_gt[:9]),
        ([{"A": 0.2, "B": 0.2}] * 10, ["A"] * 10),
        ([{"A": 0.6, "B": 0.4}] * 10, ["Z"] * 10),
        ([{"A": True, "B": 0.4}] * 10, ["A"] * 10),
    ):
        with pytest.raises(CalibrationError):
            calibrate_choice_thresholds(bad_args[0], bad_args[1])


def test_calibrate_score_thresholds_guarantees_and_gate_wiring() -> None:
    # 1. Small sample warning
    with pytest.warns(CalibrationWarning, match="need >= 19"):
        st_small = calibrate_score_thresholds(
            [1.4] * 6 + [4.6] * 6,
            [False] * 6 + [True] * 6,
            target_risk_alpha=0.10,
        )
    assert isinstance(st_small, ScoreConformalThresholds)
    assert st_small.guarantee_holds is False
    assert 1.4 < st_small.score_fail_threshold < st_small.score_threshold < 4.6

    # 2. Sufficient samples (>= 19 per class at alpha=0.10) with crossing tail outlier
    neg = [1.5 + 0.02 * i for i in range(19)] + [3.4]  # rank 20 quantile catches crossing
    pos = [4.5 - 0.02 * i for i in range(19)] + [2.6]
    st = calibrate_score_thresholds(
        neg + pos,
        [False] * 20 + [True] * 20,
        target_risk_alpha=0.10,
    )
    assert st.guarantee_holds is True
    assert st.score_fail_threshold < 2.6
    assert st.score_threshold > 3.4
    assert st.score_fail_threshold < st.score_threshold

    prof = CalibrationProfile(binary_temperature=0.5, multiclass_temperature=1.0).with_score_thresholds(st)

    # Wire into ScoreGate and verify 3-way routing: FAILED / AMBIGUOUS / PASSED
    def _mid_score_responder(prompt: str, allowed_tokens: list[str] | None) -> list[dict[str, Any]]:
        return [{"token": "3", "logprob": -0.01}, {"token": "4", "logprob": -5.0}]

    backend = _PromptKeyedLogitBackend(responder=_mid_score_responder)
    gate = ScoreGate("Rate code quality", backend=backend, profile=prof)
    assert gate.threshold == st.score_threshold
    assert gate.fail_threshold == st.score_fail_threshold
    v = gate.evaluate("some code")
    assert v.status == DecisionStatus.AMBIGUOUS
    assert not v.passed

    # Invalid inputs
    with pytest.raises(CalibrationError):
        calibrate_score_thresholds([3.0] * 10, [True] * 10)
    with pytest.raises(CalibrationError):
        calibrate_score_thresholds([6.5] * 10, [True] * 5 + [False] * 5)
    with pytest.raises(CalibrationError):
        calibrate_score_thresholds([3.0] * 10, [True] * 5 + [False] * 5, min_score=5, max_score=1)


# ---------------------------------------------------------------------------
# 4. Live fit_calibration_profile & afit_calibration_profile (Sync + Async)
# ---------------------------------------------------------------------------


def test_fit_calibration_profile_micro_and_holdout_and_cli_integration(tmp_path: Path) -> None:
    backend = _PromptKeyedLogitBackend(model="custom-ci-model")
    out_micro = tmp_path / "micro_profile.json"
    cache_file = tmp_path / "cache.json"

    # A. Micro-calibration mode (no dataset/samples)
    micro_prof = fit_calibration_profile(
        backend,
        model="custom-ci-model",
        output_path=out_micro,
        save_cache=True,
        cache_path=cache_file,
    )
    assert out_micro.exists()
    assert cache_file.exists()
    assert micro_prof.pass_threshold is None
    assert micro_prof.fail_threshold is None
    assert micro_prof.binary_temperature > 0.0
    assert backend.closed is False  # caller owns passed backend

    # B. Holdout samples with mixed unobserved target tokens + calibrate_multiclass=True
    holdout: list[tuple[str, str, bool]] = []
    for i in range(12):
        holdout.append(("Is candidate valid?", f"valid sample {i}", True))
        holdout.append(("Is candidate valid?", f"invalid sample {i}", False))

    def _mixed_responder(prompt: str, allowed_tokens: list[str] | None) -> list[dict[str, Any]]:
        if allowed_tokens and len(allowed_tokens) > 2:
            return [{"token": allowed_tokens[0], "logprob": -0.05}, {"token": allowed_tokens[1], "logprob": -3.5}]
        if "valid sample 0" in prompt:
            # Unobserved target tokens on one item
            return [{"token": "Z", "logprob": -0.1}]
        is_rev = "A: The candidate artifact violates" in prompt
        is_bad = "invalid sample" in prompt
        if is_bad:
            win, lose = ("A", "B") if is_rev else ("B", "A")
        else:
            win, lose = ("B", "A") if is_rev else ("A", "B")
        return [{"token": win, "logprob": -0.08}, {"token": lose, "logprob": -3.0}]

    mixed_backend = _PromptKeyedLogitBackend(model="mixed-model", responder=_mixed_responder)
    out_holdout = tmp_path / "holdout_profile.json"
    with pytest.warns(CalibrationWarning):
        holdout_prof = fit_calibration_profile(
            mixed_backend,
            samples=holdout,
            target_risk_alpha=0.10,
            calibrate_multiclass=True,
            output_path=out_holdout,
        )
    assert out_holdout.exists()
    assert holdout_prof.calibration_set_size == 24
    assert holdout_prof.pass_threshold is not None and holdout_prof.fail_threshold is not None
    assert holdout_prof.pass_threshold > 0.5 and holdout_prof.fail_threshold < 0.5
    assert holdout_prof.multiclass_temperature is not None

    # C. Temperature-only with < 10 custom samples and target_risk_alpha=None
    temp_only = fit_calibration_profile(
        backend,
        samples=holdout[:6],
        target_risk_alpha=None,
    )
    assert temp_only.pass_threshold is None
    assert temp_only.fail_threshold is None

    # D. Fail-fast validation before touching backend
    calls_before = len(backend.calls)
    with pytest.raises(ConfigurationError):
        fit_calibration_profile(backend, dataset="foo.json", samples=holdout)
    with pytest.raises(CalibrationError):
        fit_calibration_profile(backend, samples=holdout[:6], target_risk_alpha=0.05)
    with pytest.raises(CalibrationError):
        fit_calibration_profile(backend, samples=holdout[:12:2], target_risk_alpha=0.05)  # single class
    assert len(backend.calls) == calls_before


@pytest.mark.asyncio
async def test_afit_calibration_profile_warm_prefix_and_sync_parity(tmp_path: Path) -> None:
    holdout = [
        ("Is it valid?", f"valid_{i}" if i % 2 == 0 else f"invalid_{i}", i % 2 == 0)
        for i in range(12)
    ]

    # Sync
    sync_backend = _PromptKeyedLogitBackend(model="parity-model", prefers_warmup=True)
    with pytest.warns(CalibrationWarning):
        prof_sync = fit_calibration_profile(
            sync_backend,
            samples=holdout,
            target_risk_alpha=0.10,
            calibrate_multiclass=True,
        )

    # Async warm_prefix=True -> sequential Pass 1 -> Pass 2 (max_concurrent_calls == 1)
    warm_backend = _PromptKeyedLogitBackend(model="parity-model", prefers_warmup=False, delay=0.01)
    with pytest.warns(CalibrationWarning):
        prof_async_warm = await afit_calibration_profile(
            warm_backend,
            samples=holdout,
            target_risk_alpha=0.10,
            calibrate_multiclass=True,
            warm_prefix=True,
            output_path=tmp_path / "async_profile.json",
        )
    assert warm_backend.max_concurrent_calls == 1
    assert warm_backend.aclosed is False

    assert prof_async_warm.binary_temperature == pytest.approx(prof_sync.binary_temperature)
    assert prof_async_warm.multiclass_temperature == pytest.approx(prof_sync.multiclass_temperature)
    assert prof_async_warm.pass_threshold == pytest.approx(prof_sync.pass_threshold)
    assert prof_async_warm.fail_threshold == pytest.approx(prof_sync.fail_threshold)

    # Async warm_prefix=False -> concurrent Pass 1 + Pass 2 (max_concurrent_calls == 2)
    par_backend = _PromptKeyedLogitBackend(model="parity-model", prefers_warmup=True, delay=0.015)
    with pytest.warns(CalibrationWarning):
        await afit_calibration_profile(
            par_backend,
            samples=holdout,
            target_risk_alpha=0.10,
            warm_prefix=False,
        )
    assert par_backend.max_concurrent_calls == 2


# ---------------------------------------------------------------------------
# 5. Gate Precedence, Model-Mismatch Warning, No-Lazy-Probes, & Replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_profile_precedence_no_lazy_probes_and_snapshot_replay(tmp_path: Path) -> None:
    prof_path = tmp_path / "team_profile.json"
    prof = CalibrationProfile(
        model="unseen-byo-model",
        binary_temperature=0.06,
        multiclass_temperature=None,  # effective multiclass temp must clamp to 0.65!
        pass_threshold=0.91,
        fail_threshold=0.12,
        score_threshold=4.2,
        score_fail_threshold=2.4,
        choice_min_confidence=0.77,
        choice_min_margin=0.22,
    )
    prof.save(prof_path)

    # 1. Unknown BYO model with profile=path -> zero lazy micro-probe calls!
    configure(auto_calibrate=True)
    byo = _PromptKeyedLogitBackend(model="unseen-byo-model")

    lg = LogitGate("Is it clean?", backend=byo, profile=prof_path, save_on_fail=False)
    assert lg._needs_lazy_calibration is False
    assert lg.pass_threshold == pytest.approx(0.91)
    assert lg.fail_threshold == pytest.approx(0.12)
    assert lg.logit_temperature == pytest.approx(0.06)

    # MultiRubricGate inherits profile cleanly via **options
    mg = MultiRubricGate(["Is it safe?", "Is it clean?"], backend=byo, profile=prof_path, save_on_fail=False)
    assert mg.profile == prof
    assert mg.logit_temperature == pytest.approx(0.06)
    assert mg.pass_threshold == pytest.approx(0.91)

    # Binary-only profile (multiclass_temperature=None) floors multiclass T at 0.65 on ChoiceGate/ScoreGate/JudgeGate
    cg = ChoiceGate(
        "Route?",
        {"A": "Alpha", "B": "Beta"},
        backend=byo,
        profile=prof_path,
        save_on_fail=False,
    )
    assert cg._needs_lazy_calibration is False
    assert cg.logit_temperature == pytest.approx(0.65)
    assert cg.min_confidence == pytest.approx(0.77)
    assert cg.min_margin == pytest.approx(0.22)

    sg = ScoreGate("Rate?", backend=byo, profile=prof_path, save_on_fail=False)
    assert sg._needs_lazy_calibration is False
    assert sg.logit_temperature == pytest.approx(0.65)
    assert sg.threshold == pytest.approx(4.2)
    assert sg.fail_threshold == pytest.approx(2.4)

    jg = JudgeGate("Judge?", backend=byo, profile=prof_path)
    assert jg._needs_lazy_calibration is False
    assert jg.logit_temperature == pytest.approx(0.65)

    # Evaluate sync + async and verify ZERO lazy calibration probe calls happened
    lg.evaluate("invalid sample")
    assert len(byo.calls) == 2  # only 2 symmetric calls, not 2 + 24 micro-probes!
    await lg.aevaluate("valid sample")
    assert len(byo.calls) == 4

    cg.evaluate("some input")
    sg.evaluate("some input")
    jg.evaluate("some input")
    # Choice(2) + Score(2) + Judge CoT proberead(2) == 6 calls -> total 10 calls
    assert len(byo.calls) == 10

    # 2. Explicit kwargs beat profile, and profile beats preset
    lg_preset = LogitGate(
        "Is it clean?",
        backend=byo,
        preset="fast_triage",
        profile=prof,
        save_on_fail=False,
    )
    assert lg_preset.pass_threshold == pytest.approx(0.91)
    assert lg_preset.fail_threshold == pytest.approx(0.12)
    assert lg_preset.symmetric_debiasing is False  # preset still supplies unconfigured symmetric flag

    lg_override = LogitGate(
        "Is it clean?",
        backend=byo,
        pass_threshold=0.95,
        logit_temperature=1.4,
        profile=prof,
        save_on_fail=False,
    )
    assert lg_override.pass_threshold == pytest.approx(0.95)
    assert lg_override.fail_threshold == pytest.approx(0.12)  # unoverridden came from profile
    assert lg_override.logit_temperature == pytest.approx(1.4)

    # 3. Model mismatch emits CalibrationWarning cleanly
    other_byo = _PromptKeyedLogitBackend(model="llama3.2:3b")
    with pytest.warns(CalibrationWarning, match="unseen-byo-model"):
        LogitGate("Check", backend=other_byo, profile=prof, save_on_fail=False)

    # 4. Snapshot saved with profile replays identically sync & async EVEN AFTER deleting profile.json!
    snap_dir = tmp_path / "snaps"
    lg_snap_gate = LogitGate(
        "Is it clean?",
        backend="stub://0.05",
        profile=prof_path,
        save_on_fail=True,
    )
    v_fail = lg_snap_gate.evaluate("invalid_artifact", snapshot_dir=snap_dir, test_name="profile_replay")
    assert v_fail.passed is False

    prof_path.unlink()  # Delete external profile JSON!
    snap = VerdictSnapshot.load(snap_dir)
    assert snap.gate_config["pass_threshold"] == pytest.approx(0.91)
    assert snap.gate_config["fail_threshold"] == pytest.approx(0.12)
    assert snap.gate_config["logit_temperature"] == pytest.approx(0.06)

    replayed = snap.replay(backend="stub://0.99")
    assert replayed.passed is True
    areplayed = await snap.areplay(backend="stub://0.99")
    assert areplayed.passed is True


# ---------------------------------------------------------------------------
# 6. CLI (`calibrate --output`, `eval --profile`, `benchmark --profile`)
# ---------------------------------------------------------------------------


def test_cli_calibrate_output_eval_and_benchmark_with_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGITGATE_BACKEND", "stub://0.92")
    profile_path = tmp_path / "ci_profile.json"
    dataset_path = tmp_path / "holdout.json"
    rows = [
        {"rubric": "Is it valid?", "candidate": f"good_{i}", "ground_truth": True}
        for i in range(10)
    ] + [
        {"rubric": "Is it valid?", "candidate": f"bad_{i}", "ground_truth": False}
        for i in range(10)
    ]
    dataset_path.write_text(json.dumps(rows), encoding="utf-8")

    # 1. `logitgate calibrate --dataset ... --output ...` saves portable CalibrationProfile
    with pytest.warns(CalibrationWarning):
        rc = main([
            "calibrate",
            "--dataset",
            str(dataset_path),
            "--alpha",
            "0.10",
            "--output",
            str(profile_path),
            "--json",
        ])
    assert rc == 0
    cli_json = capsys.readouterr().out
    assert profile_path.exists()
    loaded = CalibrationProfile.load(profile_path)
    # CLI --json output BOTH satisfies CalibrationProfile schema AND legacy keys
    from_stdout = CalibrationProfile.model_validate(json.loads(cli_json))
    assert from_stdout.pass_threshold == loaded.pass_threshold
    assert from_stdout.fail_threshold == loaded.fail_threshold
    parsed_stdout = json.loads(cli_json)
    assert parsed_stdout["tau_accept"] == loaded.pass_threshold
    assert parsed_stdout["tau_reject"] == loaded.fail_threshold

    # 2. `logitgate calibrate --output ...` without --dataset also saves temperature profile
    temp_prof_path = tmp_path / "temp_profile.json"
    rc_temp = main([
        "calibrate",
        "--backend",
        "stub://0.95",
        "--model",
        "custom-temp-model",
        "--cache-path",
        str(tmp_path / "cache.json"),
        "--output",
        str(temp_prof_path),
    ])
    assert rc_temp == 0
    temp_out = capsys.readouterr().out
    assert "Calibration profile saved to" in temp_out
    assert CalibrationProfile.load(temp_prof_path).binary_temperature > 0.0

    # 3. `logitgate eval --profile ...` applies profile and allows --threshold override
    known_profile = tmp_path / "eval_profile.json"
    CalibrationProfile(
        binary_temperature=0.8,
        pass_threshold=0.99,
        fail_threshold=0.05,
    ).save(known_profile)

    # stub://0.92 gives P < 0.99 -> AMBIGUOUS -> exit code 1
    rc_eval_ambig = main([
        "eval",
        "--rubric",
        "Is it ok?",
        "--candidate",
        "ok",
        "--backend",
        "stub://0.92",
        "--profile",
        str(known_profile),
        "--json",
    ])
    assert rc_eval_ambig == 1
    eval_data = json.loads(capsys.readouterr().out)
    assert eval_data["status"] == "ambiguous"

    # Explicit --threshold 0.80 overrides profile.pass_threshold=0.99 -> PASSED -> exit code 0
    rc_eval_pass = main([
        "eval",
        "--rubric",
        "Is it ok?",
        "--candidate",
        "ok",
        "--backend",
        "stub://0.92",
        "--profile",
        str(known_profile),
        "--threshold",
        "0.80",
        "--json",
    ])
    assert rc_eval_pass == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

    # 4. `logitgate benchmark --profile ...` threads profile into metadata & skips BYO lazy probes
    rc_bench = main([
        "benchmark",
        "--backend",
        "stub://0.92",
        "--profile",
        str(known_profile),
        "--modes",
        "logit,cascade",
        "--limit",
        "4",
        "--warmup",
        "0",
        "--json",
    ])
    assert rc_bench == 0
    bench_data = json.loads(capsys.readouterr().out)
    assert bench_data["metadata"]["profile"]["pass_threshold"] == pytest.approx(0.99)
    assert bench_data["metadata"]["profile"]["fail_threshold"] == pytest.approx(0.05)

    # Unseen BYO model with run_benchmark(..., profile=...) incurs zero extra prewarm probes
    from logitgate.benchmark import BenchmarkCase, run_benchmark

    byo = _PromptKeyedLogitBackend(model="unseen-bench-model")
    byo_prof = CalibrationProfile(
        model="unseen-bench-model",
        binary_temperature=0.5,
        pass_threshold=0.88,
        fail_threshold=0.12,
    )
    report = run_benchmark(
        [BenchmarkCase("Valid?", "good sample", True)],
        backend=byo,
        model="unseen-bench-model",
        modes=["logit"],
        warmup=0,
        profile=byo_prof,
    )
    assert report.modes["logit"].summary()["llm_calls_per_case"] == pytest.approx(2.0)
    assert len(byo.calls) == 2
