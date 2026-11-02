from __future__ import annotations

import math
from typing import Sequence, Tuple
from pydantic import BaseModel


class CalibrationResult(BaseModel):
    optimal_temperature: float
    brier_score_before: float
    brier_score_after: float
    ece_before: float
    ece_after: float


def combine_symmetric_log_odds(
    forward_log_odds: float,
    reverse_log_odds: float,
    logit_temperature: float = 1.0,
) -> float:
    """
    Combines forward and reverse log-odds to cancel token and positional bias.
    mean_log_odds = (forward_log_odds + reverse_log_odds) / (2.0 * max(logit_temperature, 1e-6))
    Returns 1.0 / (1.0 + exp(-mean_log_odds)).
    """
    t = max(logit_temperature, 1e-6)
    mean_log_odds = (forward_log_odds + reverse_log_odds) / (2.0 * t)

    if mean_log_odds > 50.0:
        return 1.0
    if mean_log_odds < -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-mean_log_odds))


def fit_calibration(
    samples: Sequence[Tuple[float, float, int]],
    min_temp: float = 0.1,
    max_temp: float = 10.0,
    num_bins: int = 10,
) -> CalibrationResult:
    """
    Fits optimal logit temperature scaling on (forward_log_odds, reverse_log_odds, y)
    to minimize Brier score and computes Expected Calibration Error (ECE).
    """
    if not samples:
        return CalibrationResult(
            optimal_temperature=1.0,
            brier_score_before=0.0,
            brier_score_after=0.0,
            ece_before=0.0,
            ece_after=0.0,
        )

    def calc_prob(f_lo: float, r_lo: float, temp: float) -> float:
        return combine_symmetric_log_odds(f_lo, r_lo, logit_temperature=temp)

    def brier_score(temp: float) -> float:
        total = sum((calc_prob(f, r, temp) - float(y)) ** 2 for f, r, y in samples)
        return total / len(samples)

    def compute_ece(temp: float) -> float:
        n = len(samples)
        bins: list[list[Tuple[float, float]]] = [[] for _ in range(num_bins)]
        for f, r, y in samples:
            p = calc_prob(f, r, temp)
            idx = min(int(p * num_bins), num_bins - 1)
            bins[idx].append((p, float(y)))

        ece = 0.0
        for b in bins:
            if not b:
                continue
            b_size = len(b)
            avg_conf = sum(p for p, _ in b) / b_size
            avg_acc = sum(y for _, y in b) / b_size
            ece += (b_size / n) * abs(avg_acc - avg_conf)
        return ece

    # 1. Grid search for coarse minimum
    best_temp = 1.0
    best_brier = float("inf")
    steps = 200
    for step in range(steps + 1):
        temp = min_temp + step * (max_temp - min_temp) / steps
        bs = brier_score(temp)
        if bs < best_brier:
            best_brier = bs
            best_temp = temp

    # 2. Golden-section refinement around coarse minimum
    a = max(min_temp, best_temp - (max_temp - min_temp) / steps)
    b = min(max_temp, best_temp + (max_temp - min_temp) / steps)
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = brier_score(c)
    fd = brier_score(d)

    for _ in range(30):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = brier_score(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = brier_score(d)

    optimal_temp = (a + b) / 2.0
    brier_before = brier_score(1.0)
    brier_after = brier_score(optimal_temp)
    ece_before = compute_ece(1.0)
    ece_after = compute_ece(optimal_temp)

    return CalibrationResult(
        optimal_temperature=round(optimal_temp, 4),
        brier_score_before=round(brier_before, 6),
        brier_score_after=round(brier_after, 6),
        ece_before=round(ece_before, 6),
        ece_after=round(ece_after, 6),
    )
