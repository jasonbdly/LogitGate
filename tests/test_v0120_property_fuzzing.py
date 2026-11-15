from __future__ import annotations

import math
import time
import warnings
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from logitgate import (
    ASTGate,
    LogitFeatureVector,
    LogitMetaCalibrator,
    calibrate_cascade_thresholds,
    combine_symmetric_log_odds,
    compute_ks_two_sample,
    compute_psi,
    compute_shannon_entropy,
    extract_logit_features,
    extract_token_logprob,
    normalize_candidate,
    redact_text,
)
from logitgate.debiasing.softmax import clamped_sigmoid, softmax_probabilities

FINITE_LOGITS = st.floats(min_value=-40.0, max_value=40.0, allow_nan=False, allow_infinity=False)
TEMPERATURES = st.floats(min_value=0.05, max_value=10.0, allow_nan=False, allow_infinity=False)
PROBS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=120, deadline=None)
@given(
    x=st.floats(allow_nan=True, allow_infinity=True),
    y=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_clamped_sigmoid_bounds_monotonicity_and_antisymmetry(x: float, y: float) -> None:
    p = clamped_sigmoid(x)
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
    if math.isnan(x):
        assert p == 0.5
    elif math.isfinite(x):
        p_neg = clamped_sigmoid(-x)
        assert p + p_neg == pytest.approx(1.0, abs=1e-12)
        if x <= y:
            assert p <= clamped_sigmoid(y) + 1e-12


@settings(max_examples=100, deadline=None)
@given(
    logprobs=st.dictionaries(
        keys=st.sampled_from(["A", "B", "C", "D", "E"]),
        values=st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    shift=st.floats(min_value=-25.0, max_value=25.0, allow_nan=False, allow_infinity=False),
    temp=TEMPERATURES,
)
def test_softmax_simplex_and_shift_invariance(
    logprobs: dict[str, float],
    shift: float,
    temp: float,
) -> None:
    probs = softmax_probabilities(logprobs, temperature=temp)
    assert set(probs.keys()) == set(logprobs.keys())
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    shifted = {k: v + shift for k, v in logprobs.items()}
    probs_shifted = softmax_probabilities(shifted, temperature=temp)
    for k in logprobs:
        assert probs[k] == pytest.approx(probs_shifted[k], abs=1e-7)


@settings(max_examples=100, deadline=None)
@given(
    entries=st.lists(
        st.fixed_dictionaries(
            {
                "token": st.sampled_from(["A", " A", "a\n", "B", " B", "Yes", "", "???"]),
                "logprob": st.floats(
                    min_value=-50.0, max_value=0.0, allow_nan=False, allow_infinity=False
                ),
            }
        ),
        min_size=0,
        max_size=12,
    ),
    target=st.sampled_from(["A", "B", "C"]),
)
def test_extract_token_logprob_finite_and_non_positive(
    entries: list[dict[str, Any]],
    target: str,
) -> None:
    lp = extract_token_logprob(entries, target)
    assert math.isfinite(lp)
    assert lp <= 0.0


@settings(max_examples=120, deadline=None)
@given(
    signal=st.floats(min_value=-15.0, max_value=15.0, allow_nan=False, allow_infinity=False),
    slot_bias=st.floats(min_value=-15.0, max_value=15.0, allow_nan=False, allow_infinity=False),
    temp=st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_symmetric_debiasing_cancels_slot_bias_and_preserves_antisymmetry(
    signal: float,
    slot_bias: float,
    temp: float,
) -> None:
    # Additive positional/label bias +b on pass 1 and -b on pass 2 cancels algebraically.
    p_biased = combine_symmetric_log_odds(signal + slot_bias, signal - slot_bias, temp)
    p_clean = combine_symmetric_log_odds(signal, signal, temp)
    assert p_biased == pytest.approx(p_clean, abs=1e-10)

    # Exact antisymmetry under pass/fail negation
    p_fwd = combine_symmetric_log_odds(signal + slot_bias, signal - 0.5 * slot_bias, temp)
    p_inv = combine_symmetric_log_odds(-(signal - 0.5 * slot_bias), -(signal + slot_bias), temp)
    assert p_fwd + p_inv == pytest.approx(1.0, abs=1e-10)

    # Shannon entropy symmetry and unit bounds
    h1 = compute_shannon_entropy(p_fwd)
    h2 = compute_shannon_entropy(1.0 - p_fwd)
    assert 0.0 <= h1 <= 1.0
    assert h1 == pytest.approx(h2, abs=1e-9)


@settings(max_examples=80, deadline=None)
@given(
    pos_probs=st.lists(PROBS, min_size=5, max_size=45),
    neg_probs=st.lists(PROBS, min_size=5, max_size=45),
    alpha=st.floats(min_value=0.02, max_value=0.45, allow_nan=False, allow_infinity=False),
)
def test_conformal_threshold_ordering_and_escalation_bounds(
    pos_probs: list[float],
    neg_probs: list[float],
    alpha: float,
) -> None:
    probs = pos_probs + neg_probs
    labels = [True] * len(pos_probs) + [False] * len(neg_probs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = calibrate_cascade_thresholds(probs, labels, target_risk_alpha=alpha)
    assert 0.0 <= res.tau_reject <= res.tau_accept <= 1.0
    assert 0.0 <= res.empirical_escalation_rate <= 1.0
    assert res.calibration_set_size == len(probs)
    assert res.num_positive == len(pos_probs)
    assert res.num_negative == len(neg_probs)


@settings(max_examples=100, deadline=None)
@given(
    lo_low=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.05, max_value=10.0, allow_nan=False, allow_infinity=False),
    bias=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    entropy=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    residual=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_meta_calibrator_strict_monotonicity_in_symmetric_log_odds(
    lo_low: float,
    delta: float,
    bias: float,
    entropy: float,
    residual: float,
) -> None:
    cal = LogitMetaCalibrator(
        weights=[0.85, -0.10, -0.20, -0.15, -0.05],
        intercept=-0.10,
    )
    # Verify extract_logit_features always emits finite 5-vectors on top_logprobs dicts
    top_fwd = [{"token": "A", "logprob": -0.3}, {"token": "B", "logprob": -1.5}]
    top_rev = [{"token": "B", "logprob": -0.4}, {"token": "A", "logprob": -1.2}]
    extracted = extract_logit_features(top_fwd, top_rev)
    assert all(math.isfinite(v) for v in extracted.as_tuple())

    # Isolate partial derivative dP / d(symmetric_log_odds) holding uncertainty features fixed
    f1 = LogitFeatureVector(
        symmetric_log_odds=lo_low,
        signed_slot_bias=bias,
        positional_bias_magnitude=abs(bias),
        top_k_entropy=entropy,
        unobserved_residual_mass=residual,
    )
    f2 = LogitFeatureVector(
        symmetric_log_odds=lo_low + delta,
        signed_slot_bias=bias,
        positional_bias_magnitude=abs(bias),
        top_k_entropy=entropy,
        unobserved_residual_mass=residual,
    )
    p1 = cal.predict_proba(f1)
    p2 = cal.predict_proba(f2)
    assert 0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0
    assert p2 >= p1 - 1e-12


@settings(max_examples=80, deadline=None)
@given(
    ref=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=60,
    ),
    cur=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=60,
    ),
    shift=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_drift_psi_and_ks_mathematical_invariants(
    ref: list[float],
    cur: list[float],
    shift: float,
) -> None:
    assert compute_psi(ref, ref) == 0.0
    psi_xy = compute_psi(ref, cur)
    assert math.isfinite(psi_xy) and psi_xy >= 0.0

    d_self, p_self = compute_ks_two_sample(ref, ref)
    assert d_self == 0.0
    assert p_self == 1.0

    d_xy, p_xy = compute_ks_two_sample(ref, cur)
    d_yx, p_yx = compute_ks_two_sample(cur, ref)
    assert 0.0 <= d_xy <= 1.0
    assert 0.0 <= p_xy <= 1.0
    assert d_xy == pytest.approx(d_yx, abs=1e-9)
    assert p_xy == pytest.approx(p_yx, abs=1e-9)

    # KS statistic is invariant under strictly increasing order-preserving transforms
    # (2x scaling has zero float64 mantissa absorption; +shift holds when float64 order is preserved).
    d_scaled, _ = compute_ks_two_sample([2.0 * x for x in ref], [2.0 * x for x in cur])
    assert d_scaled == pytest.approx(d_xy, abs=1e-9)
    if len({x + shift for x in ref + cur}) == len(set(ref + cur)):
        d_shifted, _ = compute_ks_two_sample([x + shift for x in ref], [x + shift for x in cur])
        assert d_shifted == pytest.approx(d_xy, abs=1e-6)


@settings(max_examples=100, deadline=None)
@given(
    payload=st.lists(
        st.sampled_from(
            [
                "a",
                "1",
                "4",
                "@",
                ".",
                "-",
                "_",
                " ",
                "\n",
                "`",
                "{",
                "}",
                "(",
                ")",
                ":",
                "Bearer ",
                "sk-proj_",
                "```python\n",
                "```\n",
                "\u2603",
                "\x00",
            ]
        ),
        min_size=0,
        max_size=80,
    ).map("".join),
    max_chars=st.one_of(st.none(), st.integers(min_value=16, max_value=256)),
)
def test_adversarial_redaction_normalization_and_ast_gate_never_crash_or_hang(
    payload: str,
    max_chars: int | None,
) -> None:
    t0 = time.perf_counter()
    scrubbed = redact_text(payload, max_chars=max_chars)
    norm = normalize_candidate(payload)
    py_gate = ASTGate(syntax=True, normalize=True)
    json_gate = ASTGate(json_schema={"type": "object"}, normalize=True)
    v_py = py_gate(payload)
    v_json = json_gate(payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(scrubbed, str)
    assert isinstance(norm, str)
    assert isinstance(v_py.passed, bool)
    assert isinstance(v_json.passed, bool)
    assert elapsed_ms < 250.0
