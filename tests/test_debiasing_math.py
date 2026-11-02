from __future__ import annotations

import math
import pytest
from logitgate.debiasing.permutation import combine_symmetric_log_odds, fit_calibration
from logitgate.debiasing.softmax import extract_token_logprob, compute_binary_log_odds
from logitgate.debiasing.inversion import compile_binary_prompt


def test_symmetric_log_odds_cancels_extreme_bias():
    """
    Proves that combine_symmetric_log_odds cancels out an extreme +5.0 logit bias
    for token 'A' when true signal is +2.0, returning sigma(2.0) ~= 0.8808,
    whereas naive probability averaging fails at 0.523.
    """
    true_signal = 2.0
    bias_a = 5.0

    # Pass 1: A is Pass, B is Fail -> logit(A) has bias +5.0
    forward_log_odds = true_signal + bias_a  # 7.0
    # Pass 2: B is Pass, A is Fail -> logit(A) has bias +5.0, so logit(B) - logit(A) = -3.0
    reverse_log_odds = true_signal - bias_a  # -3.0

    # 1. LogitGate combined symmetric log-odds
    calibrated_prob = combine_symmetric_log_odds(forward_log_odds, reverse_log_odds)
    expected_prob = 1.0 / (1.0 + math.exp(-true_signal))  # sigma(2.0) ~= 0.880797
    assert pytest.approx(calibrated_prob, abs=1e-4) == expected_prob
    assert pytest.approx(calibrated_prob, abs=1e-3) == 0.8808

    # 2. Naive probability averaging failure
    p1 = 1.0 / (1.0 + math.exp(-forward_log_odds))  # sigma(7.0) ~= 0.9991
    p2 = 1.0 / (1.0 + math.exp(-reverse_log_odds))  # sigma(-3.0) ~= 0.0474
    naive_avg = (p1 + p2) / 2.0
    assert pytest.approx(naive_avg, abs=1e-3) == 0.523
    assert naive_avg < 0.60  # Severely degraded to coin flip


def test_bounded_floor_missing_token():
    """
    Tests When target token is absent from top_logprobs,
    extract_token_logprob returns a bounded floor based on residual probability
    instead of an arbitrary -99.0 cliff.
    """
    top_logprobs = [
        {"token": "A", "logprob": -0.05},  # p ~= 0.9512
        {"token": "C", "logprob": -3.5},   # p ~= 0.0302
    ]
    # Target 'B' is absent
    floor_lp = extract_token_logprob(top_logprobs, "B")

    # Must NOT be -99.0 cliff
    assert floor_lp > -15.0
    # Must be bounded by min observed or residual logprob
    sum_probs = math.exp(-0.05) + math.exp(-3.5)
    residual = max(1.0 - sum_probs, 1e-6)
    expected_floor = min(-3.5, math.log(residual))
    assert pytest.approx(floor_lp, rel=1e-3) == expected_floor

    # Empty list fallback
    assert extract_token_logprob([], "A") == -10.0


def test_fit_calibration_lowers_brier_score():
    """
    Tests fit_calibration optimizes temperature scaling, lowering
    Brier score on synthetic overconfident predictions.
    """
    # Create synthetic overconfident logits where logits are too large (+- 6.0)
    # causing overconfident probabilities (0.9975 vs true accuracy 0.70)
    samples = []
    # Class 1 samples
    for _ in range(70):
        samples.append((6.0, 6.0, 1))
    # Class 0 samples (overconfident mistakes)
    for _ in range(30):
        samples.append((6.0, 6.0, 0))

    calib = fit_calibration(samples)
    assert calib.optimal_temperature > 1.0  # Temperature should increase to soften logits
    assert calib.brier_score_after < calib.brier_score_before


def test_compile_binary_prompt_identical_static_prefix():
    """
    Tests compile_binary_prompt keeps the [EVALUATION RUBRIC] + [CANDIDATE ARTIFACT]
    prefix 100% byte-identical between Pass 1 and Pass 2 for KV-cache reuse.
    """
    rubric = "Functions must have docstrings and type annotations."
    candidate = "def compute(x: int) -> int:\n    '''Doubles x.'''\n    return x * 2"

    pass1 = compile_binary_prompt(rubric, candidate, pass_token="A", fail_token="B", query_first=True)
    pass2 = compile_binary_prompt(rubric, candidate, pass_token="B", fail_token="A", query_first=True)

    marker = "[DECISION MAPPING]"
    assert marker in pass1
    assert marker in pass2

    prefix1 = pass1.split(marker)[0]
    prefix2 = pass2.split(marker)[0]

    # Must be 100% byte-identical
    assert prefix1 == prefix2
    assert "[EVALUATION RUBRIC]" in prefix1
    assert "[CANDIDATE ARTIFACT]" in prefix1
    assert rubric in prefix1
    assert candidate in prefix1
