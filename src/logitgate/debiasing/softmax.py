from __future__ import annotations

import math
from typing import Any


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
    - Combines multiple token variants (e.g. 'A' and ' A') using logsumexp.
    - Bounded Missing-Token Floor: If target_token is absent, computes
      min(min_observed_logprob, ln(max(1.0 - sum_observed_probs, 1e-6))),
      preventing -99.0 cliff artifacts in symmetric log-odds debiasing.
    """
    if not top_logprobs:
        return -10.0

    target_norm = target_token.strip().upper()
    matching_logprobs: list[float] = []
    observed_logprobs: list[float] = []

    for item in top_logprobs:
        tok = str(item.get("token", "")).strip().upper()
        lp = item.get("logprob")
        if lp is None:
            continue
        try:
            lp_float = float(lp)
        except (ValueError, TypeError):
            continue
        observed_logprobs.append(lp_float)
        if tok == target_norm:
            matching_logprobs.append(lp_float)

    if matching_logprobs:
        return logsumexp(matching_logprobs)

    # Bounded missing-token floor based on residual probability mass
    if not observed_logprobs:
        return -10.0

    min_observed_logprob = min(observed_logprobs)
    sum_observed_probs = sum(math.exp(min(lp, 0.0)) for lp in observed_logprobs)
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

    if log_odds > 50.0:
        prob_pass = 1.0
    elif log_odds < -50.0:
        prob_pass = 0.0
    else:
        prob_pass = 1.0 / (1.0 + math.exp(-log_odds))

    return log_odds, prob_pass


def compute_normalized_probabilities(
    top_logprobs: list[dict[str, Any]],
    tokens: list[str],
) -> dict[str, float]:
    """
    Extracts logprobs for candidate tokens and returns normalized softmax probabilities.
    """
    logprobs = {t: extract_token_logprob(top_logprobs, t) for t in tokens}
    max_lp = max(logprobs.values()) if logprobs else 0.0
    exp_vals = {t: math.exp(lp - max_lp) for t, lp in logprobs.items()}
    total = sum(exp_vals.values())
    if total <= 0:
        uniform = 1.0 / len(tokens) if tokens else 0.0
        return {t: uniform for t in tokens}
    return {t: exp_vals[t] / total for t in tokens}
