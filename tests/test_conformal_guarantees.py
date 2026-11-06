"""Statistical properties of the conformal thresholds, checked by simulation (seeded, ~0.5 s)."""

from __future__ import annotations

import random
import warnings

import pytest

from logitgate.calibration.conformal import (
    calibrate_cascade_thresholds,
    min_samples_per_class,
)
from logitgate.exceptions import CalibrationWarning


def _draw(rng: random.Random, n: int, a: float, b: float, decimals: int) -> list[float]:
    return [round(rng.betavariate(a, b), decimals) for _ in range(n)]


@pytest.mark.parametrize("decimals", [6, 2], ids=["continuous", "heavy-ties"])
def test_false_accept_rate_respects_alpha_over_calibration_draws(decimals: int) -> None:
    # The bound is in expectation over calibration sets, so average over many draws.
    # Rounding to 2 decimals creates ties with the quantile: with `>=` acceptance, a threshold
    # equal to the quantile exceeds alpha/2. The pre-fix implementation measured 0.0254 here.
    rng = random.Random(7)
    alpha, per_class, trials = 0.05, 40, 150
    rates = []
    for _ in range(trials):
        neg = _draw(rng, per_class, 2, 5, decimals)
        pos = _draw(rng, per_class, 5, 2, decimals)
        th = calibrate_cascade_thresholds(neg + pos, [False] * per_class + [True] * per_class, alpha)
        assert th.guarantee_holds
        fresh = _draw(rng, 1000, 2, 5, decimals)
        rates.append(sum(p >= th.tau_accept for p in fresh) / len(fresh))
    assert sum(rates) / trials <= alpha / 2


def test_undersized_calibration_set_is_flagged_not_passed_off_as_a_bound() -> None:
    assert min_samples_per_class(0.05) == 39
    assert min_samples_per_class(0.10) == 19
    probs = [0.1 * i for i in range(10)] * 2
    labels = [False] * 10 + [True] * 10
    with pytest.warns(CalibrationWarning, match=r"need >= 39 positives and >= 39 negatives \(have 10 / 10\)"):
        th = calibrate_cascade_thresholds(probs, labels, 0.05)
    assert th.guarantee_holds is False
    assert (th.num_positive, th.num_negative) == (10, 10)


def test_thresholds_are_strict_and_never_clamped_into_unsafe_territory() -> None:
    rng = random.Random(3)
    neg = [round(rng.uniform(0.0, 0.3), 2) for _ in range(60)] + [0.97]  # one confident mistake
    pos = [round(rng.uniform(0.7, 1.0), 2) for _ in range(60)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalibrationWarning)
        th = calibrate_cascade_thresholds(neg + pos, [False] * 61 + [True] * 60, 0.10)
    # A threshold equal to a calibration score would accept that score under `>=`.
    assert th.tau_accept not in neg
    assert th.tau_reject not in [1 - p for p in pos] and th.tau_reject not in pos
    # The pre-fix version clamped tau_accept to <= 0.99 and tau_reject to >= 0.01, which
    # moved both cutoffs *toward* the decision regions.
    assert 0.0 <= th.tau_reject < th.tau_accept <= 1.0
    escalated = sum(th.tau_reject < p < th.tau_accept for p in neg + pos) / len(neg + pos)
    assert th.empirical_escalation_rate == round(escalated, 4)
