from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CalibrationResult(BaseModel):
    optimal_temperature: float
    brier_score_before: float
    brier_score_after: float
    ece_before: float
    ece_after: float


KNOWN_MODEL_TEMPERATURES: dict[str, float] = {
    "qwen2.5-coder:14b": 1.05,
    "qwen2.5:14b": 1.05,
    "gemma4:e4b": 0.95,
    "gpt-4o-mini": 1.15,
    "llama3": 1.10,
}

UNIVERSAL_CALIBRATION_DATASET: list[tuple[str, str, int]] = [
    # 1. QA / Factual Pass
    (
        "Does the text state that Paris is the capital of France?",
        "Paris is the capital and largest city of France.",
        1,
    ),
    # 2. QA / Factual Fail
    (
        "Does the text state that Paris is the capital of France?",
        "Berlin is the capital of Germany and Rome is the capital of Italy.",
        0,
    ),
    # 3. Policy Pass
    (
        "Does the assistant refuse to disclose sensitive credentials without authorization?",
        "I cannot provide master API keys or passwords without administrative verification.",
        1,
    ),
    # 4. Policy Fail
    (
        "Does the assistant refuse to disclose sensitive credentials without authorization?",
        "Sure, here is the secret admin key: sk-secret-12345.",
        0,
    ),
    # 5. JSON Validity Pass
    (
        "Is the artifact valid JSON containing required 'status' and 'code' fields?",
        '{"status": "ok", "code": 200, "message": "Success"}',
        1,
    ),
    # 6. JSON Validity Fail
    (
        "Is the artifact valid JSON containing required 'status' and 'code' fields?",
        "This is arbitrary plain text with no structured format or status code.",
        0,
    ),
    # 7. Code Correctness Pass
    (
        "Does the Python function compute the square of an integer x?",
        "def square(x: int) -> int:\n    return x * x",
        1,
    ),
    # 8. Code Correctness Fail
    (
        "Does the Python function compute the square of an integer x?",
        "def square(x: int) -> int:\n    return x + 100",
        0,
    ),
]


def get_calibration_cache_path() -> Path:
    cache_dir = os.environ.get("LOGITGATE_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "calibration.json"
    return Path.home() / ".cache" / "logitgate" / "calibration.json"


def get_model_temperature(
    model_name: str | None,
    cache_path: Path | None = None,
) -> float:
    """Retrieves calibrated temperature for a model checking cache, known models, then default 1.0."""
    if not model_name:
        return 1.0

    m_lower = model_name.strip().lower()

    # 1. Local calibration cache file
    target_cache = cache_path or get_calibration_cache_path()
    if target_cache.exists():
        try:
            with open(target_cache, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                for k, v in cached.items():
                    if k.lower() == m_lower or k.lower() in m_lower:
                        return float(v)
        except (OSError, ValueError, TypeError):
            cached = None

    # 2. KNOWN_MODEL_TEMPERATURES
    for k, v in KNOWN_MODEL_TEMPERATURES.items():
        if k.lower() == m_lower or k.lower() in m_lower:
            return v

    # 3. Default
    return 1.0


def auto_calibrate_backend(
    backend: Any,
    model_key: str | None = None,
    save_cache: bool = True,
    cache_path: Path | None = None,
) -> CalibrationResult:
    """Runs built-in 8-item micro-calibration dataset, fits optimal temperature, and caches it."""
    from logitgate.debiasing.inversion import compile_binary_prompt
    from logitgate.debiasing.softmax import compute_binary_log_odds

    effective_model = model_key or getattr(backend, "model", None) or "default"
    samples: list[tuple[float, float, int]] = []

    for rubric, candidate, y in UNIVERSAL_CALIBRATION_DATASET:
        p1 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="A",
            fail_token="B",
            query_first=True,
        )
        top1 = backend.query_top_logprobs(p1, top_k=10, allowed_tokens=["A", "B"])
        f_lo, _ = compute_binary_log_odds(top1, pass_token="A", fail_token="B")

        p2 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="B",
            fail_token="A",
            query_first=True,
        )
        top2 = backend.query_top_logprobs(p2, top_k=10, allowed_tokens=["A", "B"])
        r_lo, _ = compute_binary_log_odds(top2, pass_token="B", fail_token="A")

        samples.append((f_lo, r_lo, y))

    result = fit_calibration(samples)

    if save_cache:
        target_path = cache_path or get_calibration_cache_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data: dict[str, float] = {}
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    cache_data = loaded
            except (OSError, ValueError, TypeError):
                cache_data = {}
        cache_data[effective_model] = result.optimal_temperature
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    return result


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
    samples: Sequence[tuple[float, float, int]],
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
        bins: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]
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
