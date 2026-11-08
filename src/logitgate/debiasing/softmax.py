from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def clamped_sigmoid(x: float) -> float:
    """Logistic function saturated at |x| > 50 to avoid `math.exp` overflow on degenerate logits."""
    if x > 50.0:
        return 1.0
    if x < -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def softmax_probabilities(logprobs: Mapping[Any, float], temperature: float = 1.0) -> dict[Any, float]:
    """Temperature-scaled softmax renormalized over the given label set only."""
    if not logprobs:
        return {}
    temp = max(temperature, 1e-6)
    scaled = {k: lp / temp for k, lp in logprobs.items()}
    max_lp = max(scaled.values())
    exp_vals = {k: math.exp(lp - max_lp) for k, lp in scaled.items()}
    total = sum(exp_vals.values())
    if total <= 0:
        return {k: 1.0 / len(scaled) for k in scaled}
    return {k: v / total for k, v in exp_vals.items()}


def has_finite_logprobs(top_logprobs: list[dict[str, Any]]) -> bool:
    """Returns True if top_logprobs contains at least one entry with a finite numeric logprob."""
    for item in top_logprobs:
        if not isinstance(item, dict):
            continue
        lp = item.get("logprob")
        if lp is None:
            continue
        try:
            if math.isfinite(float(lp)):
                return True
        except (ValueError, TypeError):
            continue
    return False


def logsumexp(logprobs: list[float]) -> float:
    """Numerically stable logsumexp."""
    if not logprobs:
        return -10.0
    max_lp = max(logprobs)
    if math.isinf(max_lp):
        return max_lp
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in logprobs))


def extract_token_logprob(top_logprobs: list[dict[str, Any]], target_token: str) -> float:
    """
    Extracts the log probability for target_token from top_logprobs.
    - Normalizes token strings by stripping whitespace and uppercase.
    - Combines multiple distinct surface variants (e.g. 'A' and ' A') using logsumexp.
    - Bounded Missing-Token Floor: If target_token is absent, computes
      min(min_observed_logprob, ln(max(1.0 - sum_observed_probs, 1e-6))),
      preventing -99.0 cliff artifacts in symmetric log-odds debiasing.
    """
    if not top_logprobs:
        return -10.0

    target_norm = target_token.strip().upper()
    matching_by_raw_token: dict[str, float] = {}
    observed_logprobs: list[float] = []

    for item in top_logprobs:
        raw_tok = str(item.get("token", ""))
        tok = raw_tok.strip().upper()
        lp = item.get("logprob")
        if lp is None:
            continue
        try:
            lp_float = float(lp)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(lp_float):
            continue
        lp_clamped = min(lp_float, 0.0)
        observed_logprobs.append(lp_clamped)
        if tok == target_norm and (
            raw_tok not in matching_by_raw_token or lp_clamped > matching_by_raw_token[raw_tok]
        ):
            matching_by_raw_token[raw_tok] = lp_clamped

    if matching_by_raw_token:
        return min(logsumexp(list(matching_by_raw_token.values())), 0.0)

    # Bounded missing-token floor based on residual probability mass
    if not observed_logprobs:
        return -10.0

    min_observed_logprob = min(observed_logprobs)
    sum_observed_probs = sum(math.exp(lp) for lp in observed_logprobs)
    residual_prob = max(1.0 - sum_observed_probs, 1e-6)
    residual_logprob = math.log(residual_prob)
    return min(min_observed_logprob, residual_logprob)


def compute_binary_log_odds(
    top_logprobs: list[dict[str, Any]],
    pass_token: str = "A",
    fail_token: str = "B",
) -> tuple[float, float]:
    """
    Computes binary log-odds and pass probability from top_logprobs.
    log_odds = lp_pass - lp_fail
    prob_pass = 1.0 / (1.0 + exp(-log_odds))
    """
    lp_pass = extract_token_logprob(top_logprobs, pass_token)
    lp_fail = extract_token_logprob(top_logprobs, fail_token)
    log_odds = lp_pass - lp_fail
    return log_odds, clamped_sigmoid(log_odds)


def compute_normalized_probabilities(
    top_logprobs: list[dict[str, Any]],
    tokens: list[str],
    temperature: float = 1.0,
) -> dict[str, float]:
    """
    Extracts logprobs for candidate tokens and returns normalized softmax probabilities.
    """
    logprobs = {t: extract_token_logprob(top_logprobs, t) for t in tokens}
    return softmax_probabilities(logprobs, temperature=temperature)
