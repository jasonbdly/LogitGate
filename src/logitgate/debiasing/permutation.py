from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
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
    "qwen2.5:7b": 1.05,
    "qwen2.5": 1.05,
    "gemma4:e4b": 0.95,
    "gemma4": 0.95,
    "gpt-4o-mini": 1.15,
    "llama3": 1.10,
}

UNIVERSAL_CALIBRATION_DATASET: list[tuple[str, str, int]] = [
    # 1. Grounded RAG Pass (paraphrased factual alignment)
    (
        "Does the answer rely strictly on the provided context without extrapolating unsupported dates or regions?",
        "Context: The v2.4 migration applies to EU-West clusters starting Q3.\nAnswer: Only EU-West clusters are scheduled for the v2.4 migration in Q3.",
        1,
    ),
    # 2. Grounded RAG Fail (subtle regional extrapolation)
    (
        "Does the answer rely strictly on the provided context without extrapolating unsupported dates or regions?",
        "Context: The v2.4 migration applies to EU-West clusters starting Q3.\nAnswer: The v2.4 migration rolls out globally to all US and EU clusters in Q3.",
        0,
    ),
    # 3. Security Policy Pass (parameterized SQL)
    (
        "Does the database query use parameterized placeholders for all user inputs and avoid string interpolation?",
        'cursor.execute("SELECT id, role FROM accounts WHERE email = ? AND active = 1", (user_email,))',
        1,
    ),
    # 4. Security Policy Fail (partial SQL string interpolation)
    (
        "Does the database query use parameterized placeholders for all user inputs and avoid string interpolation?",
        'cursor.execute(f"SELECT id, role FROM accounts WHERE org = \'{org_id}\' AND email = ?", (user_email,))',
        0,
    ),
    # 5. Multi-Constraint Code Contract Pass
    (
        "Does the function include parameter and return type annotations, handle empty sequences without raising ValueError, and avoid mutating its input?",
        "def head_or_none(items: list[int]) -> int | None:\n    return items[0] if items else None",
        1,
    ),
    # 6. Multi-Constraint Code Contract Fail (satisfies 2 of 3 rules, fails mutation rule)
    (
        "Does the function include parameter and return type annotations, handle empty sequences without raising ValueError, and avoid mutating its input?",
        "def head_or_none(items: list[int]) -> int | None:\n    return items.pop(0) if items else None",
        0,
    ),
    # 7. Support Policy Pass (apology + refund confirmation + no sensitive data request)
    (
        "Does the support reply apologize for the duplicate charge, confirm a refund is issued, and avoid asking for full card numbers or passwords?",
        "I apologize for the duplicate charge on invoice #402. I have issued a full refund to your original payment method, which will appear in 3-5 business days.",
        1,
    ),
    # 8. Support Policy Fail (apologizes, but asks for CVV/password before refunding)
    (
        "Does the support reply apologize for the duplicate charge, confirm a refund is issued, and avoid asking for full card numbers or passwords?",
        "I apologize for the duplicate charge on invoice #402. Please reply with your account password and 3-digit CVV so I can issue your refund.",
        0,
    ),
]


def get_calibration_cache_path() -> Path:
    cache_dir = os.environ.get("LOGITGATE_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "calibration.json"
    return Path.home() / ".cache" / "logitgate" / "calibration.json"


def _lookup_temperature_in_mapping(m_lower: str, mapping: dict[str, Any]) -> float | None:
    """Finds a temperature by exact case-insensitive match first, then longest-prefix substring match."""
    for k, v in mapping.items():
        if not isinstance(k, str) or isinstance(v, bool):
            continue
        if k.lower() == m_lower:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(vf) and vf > 0.0:
                return vf
    for k, v in sorted(mapping.items(), key=lambda kv: len(str(kv[0])), reverse=True):
        if not isinstance(k, str) or isinstance(v, bool):
            continue
        k_low = k.strip().lower()
        if k_low and k_low != "default" and k_low in m_lower:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(vf) and vf > 0.0:
                return vf
    return None


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
                found = _lookup_temperature_in_mapping(m_lower, cached)
                if found is not None:
                    return found
        except (OSError, ValueError, TypeError):
            cached = None

    # 2. KNOWN_MODEL_TEMPERATURES
    found_known = _lookup_temperature_in_mapping(m_lower, KNOWN_MODEL_TEMPERATURES)
    if found_known is not None:
        return found_known

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
    from logitgate.debiasing.softmax import compute_binary_log_odds, has_finite_logprobs
    from logitgate.exceptions import BackendError

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
        if not has_finite_logprobs(top1):
            raise BackendError(
                f"Backend {type(backend).__name__} returned no token log-probabilities during calibration."
            )
        f_lo, _ = compute_binary_log_odds(top1, pass_token="A", fail_token="B")

        p2 = compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token="B",
            fail_token="A",
            query_first=True,
        )
        top2 = backend.query_top_logprobs(p2, top_k=10, allowed_tokens=["A", "B"])
        if not has_finite_logprobs(top2):
            raise BackendError(
                f"Backend {type(backend).__name__} returned no token log-probabilities during calibration."
            )
        r_lo, _ = compute_binary_log_odds(top2, pass_token="B", fail_token="A")

        samples.append((f_lo, r_lo, y))

    result = fit_calibration(samples)

    if save_cache:
        import tempfile

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
        fd, tmp_path = tempfile.mkstemp(dir=target_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            os.replace(tmp_path, target_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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
    from logitgate.debiasing.softmax import clamped_sigmoid

    t = max(logit_temperature, 1e-6)
    mean_log_odds = (forward_log_odds + reverse_log_odds) / (2.0 * t)
    return clamped_sigmoid(mean_log_odds)


def compute_shannon_entropy(p_pass: float, eps: float = 1e-12) -> float:
    """Computes normalized binary Shannon entropy in [0.0, 1.0] (bits)."""
    p_c = min(max(p_pass, eps), 1.0 - eps)
    return -(p_c * math.log2(p_c) + (1.0 - p_c) * math.log2(1.0 - p_c))


@dataclass(frozen=True, slots=True)
class SurfaceTokenMap:
    """Maps canonical binary poles to all single-token surface form IDs."""

    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for seq_name, seq_vals in (
            ("positive_token_ids", self.positive_token_ids),
            ("negative_token_ids", self.negative_token_ids),
        ):
            for x in seq_vals:
                if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                    raise ValueError(f"{seq_name} must contain non-negative integers, got {x!r}")
        pos_norm = tuple(sorted({int(x) for x in self.positive_token_ids}))
        neg_norm = tuple(sorted({int(x) for x in self.negative_token_ids}))
        object.__setattr__(self, "positive_token_ids", pos_norm)
        object.__setattr__(self, "negative_token_ids", neg_norm)
        if not pos_norm or not neg_norm:
            raise ValueError("positive_token_ids and negative_token_ids must not be empty.")
        overlap = set(pos_norm) & set(neg_norm)
        if overlap:
            raise ValueError(
                f"Positive and negative token IDs must be disjoint, got overlapping IDs: {sorted(overlap)}"
            )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        pos_label: str = "A",
        neg_label: str = "B",
    ) -> SurfaceTokenMap:
        def _collect_ids(label: str) -> tuple[int, ...]:
            variants = {
                label,
                label.lower(),
                label.upper(),
                label.capitalize(),
                f" {label}",
                f" {label.lower()}",
                f" {label.upper()}",
                f" {label.capitalize()}",
            }
            token_ids: set[int] = set()
            for v in variants:
                try:
                    encoded = tokenizer.encode(v, add_special_tokens=False)
                except TypeError:
                    encoded = tokenizer.encode(v)
                if len(encoded) == 1:
                    token_ids.add(int(encoded[0]))
            if not token_ids:
                raise ValueError(f"Label '{label}' does not map to any single-token ID.")
            return tuple(sorted(token_ids))

        return cls(
            positive_token_ids=_collect_ids(pos_label),
            negative_token_ids=_collect_ids(neg_label),
        )

    def to_logit_bias(self, bias: int | float = 100) -> dict[str, int | float]:
        """Builds an OpenAI/vLLM/llama.cpp `logit_bias` dictionary for all surface token IDs."""
        all_ids = sorted(set(self.positive_token_ids) | set(self.negative_token_ids))
        return {str(tid): bias for tid in all_ids}


def marginalize_pole_logprob(
    logits: Any,
    pos_ids: Sequence[int],
    neg_ids: Sequence[int],
    temperature: float = 1.0,
) -> tuple[float, float]:
    """Computes logsumexp over all surface-form tokens for each pole, normalized

    strictly over the union of valid verdict tokens (eliminating leakage to irrelevant vocab).

    Args:
        logits: 1D Tensor or sequence of shape [vocab_size] from the final prefill token.
        pos_ids: Token IDs corresponding to the first pole (e.g. token 'A').
        neg_ids: Token IDs corresponding to the second pole (e.g. token 'B').
        temperature: Calibration temperature T > 0.

    Returns:
        Tuple of (log_prob_pos, log_prob_neg)
    """
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be a finite strictly positive float.")
    if not pos_ids or not neg_ids:
        raise ValueError("pos_ids and neg_ids must not be empty.")
    for seq_name, seq_vals in (("pos_ids", pos_ids), ("neg_ids", neg_ids)):
        for idx_val in seq_vals:
            if isinstance(idx_val, bool) or not isinstance(idx_val, int) or idx_val < 0:
                raise ValueError(f"{seq_name} must contain non-negative integer token IDs, got {idx_val!r}")

    lse_pos_raw: float
    lse_neg_raw: float
    used_torch = False
    try:
        import torch

        if isinstance(logits, torch.Tensor):
            pos_tensor = logits[list(pos_ids)].float()
            neg_tensor = logits[list(neg_ids)].float()
            lse_pos_raw = float(torch.logsumexp(pos_tensor, dim=0).item())
            lse_neg_raw = float(torch.logsumexp(neg_tensor, dim=0).item())
            used_torch = True
    except ImportError:
        pass

    if not used_torch:
        # Sequence fallback (marginalize surface-form tokens first, then apply temperature scaling)
        pos_vals = [float(logits[i]) for i in pos_ids]
        neg_vals = [float(logits[i]) for i in neg_ids]

        def _lse(vals: Sequence[float]) -> float:
            if any(math.isnan(v) for v in vals):
                return float("nan")
            m = max(vals)
            if math.isinf(m):
                return m
            return m + math.log(sum(math.exp(v - m) for v in vals))

        lse_pos_raw = _lse(pos_vals)
        lse_neg_raw = _lse(neg_vals)

    lse_pos_f = lse_pos_raw / temperature
    lse_neg_f = lse_neg_raw / temperature

    if math.isnan(lse_pos_f) or math.isnan(lse_neg_f):
        return -math.log(2.0), -math.log(2.0)
    if math.isinf(lse_pos_f) and math.isinf(lse_neg_f):
        if (lse_pos_f > 0) == (lse_neg_f > 0):
            return -math.log(2.0), -math.log(2.0)
        return (0.0, -float("inf")) if lse_pos_f > 0 else (-float("inf"), 0.0)
    if math.isinf(lse_pos_f):
        return (0.0, -float("inf")) if lse_pos_f > 0 else (-float("inf"), 0.0)
    if math.isinf(lse_neg_f):
        return (-float("inf"), 0.0) if lse_neg_f > 0 else (0.0, -float("inf"))

    m_tot = max(lse_pos_f, lse_neg_f)
    lse_total_f = m_tot + math.log(math.exp(lse_pos_f - m_tot) + math.exp(lse_neg_f - m_tot))
    return lse_pos_f - lse_total_f, lse_neg_f - lse_total_f


@dataclass(frozen=True, slots=True)
class DebiasedLogitResult:
    """Output of one symmetric (forward + permuted) logit read; see `compute_symmetric_debiased_verdict`."""

    calibrated_prob_pass: float
    shannon_entropy: float
    log_odds_pass: float
    positional_bias_magnitude: float
    pass_forward_prob: float
    pass_permuted_prob: float


def compute_symmetric_debiased_verdict(
    logits_pass_first: Any,
    logits_fail_first: Any,
    token_map: SurfaceTokenMap,
    temperature: float = 1.0,
) -> DebiasedLogitResult:
    """Executes Symmetric Label Permutation debiasing in log-odds space.

    Pass 1 (Forward):  Token 'A' = PASS, Token 'B' = FAIL
    Pass 2 (Permuted): Token 'A' = FAIL, Token 'B' = PASS

    By binding the physical tokens ('A', 'B') to swapped semantic meanings across
    Pass 1 and Pass 2, subtracting the log-odds cancels both token-identity prior
    (preference for 'A' over 'B') and slot-order bias.
    """
    # Pass 1: A -> Pass, B -> Fail
    lp_A1, lp_B1 = marginalize_pole_logprob(
        logits_pass_first,
        token_map.positive_token_ids,
        token_map.negative_token_ids,
        temperature=temperature,
    )
    raw_delta_1 = lp_A1 - lp_B1
    delta_1 = max(-100.0, min(100.0, raw_delta_1)) if math.isinf(raw_delta_1) else raw_delta_1

    # Pass 2: A -> Fail, B -> Pass
    lp_A2, lp_B2 = marginalize_pole_logprob(
        logits_fail_first,
        token_map.positive_token_ids,
        token_map.negative_token_ids,
        temperature=temperature,
    )
    raw_delta_2 = lp_B2 - lp_A2
    delta_2 = max(-100.0, min(100.0, raw_delta_2)) if math.isinf(raw_delta_2) else raw_delta_2

    from logitgate.debiasing.softmax import clamped_sigmoid

    unbiased_log_odds = max(-100.0, min(100.0, 0.5 * (delta_1 + delta_2)))
    positional_bias = min(100.0, 0.5 * abs(delta_1 - delta_2))
    p_pass = clamped_sigmoid(unbiased_log_odds)

    entropy = compute_shannon_entropy(p_pass)

    return DebiasedLogitResult(
        calibrated_prob_pass=p_pass,
        shannon_entropy=entropy,
        log_odds_pass=unbiased_log_odds,
        positional_bias_magnitude=positional_bias,
        pass_forward_prob=math.exp(lp_A1),
        pass_permuted_prob=math.exp(lp_B2),
    )


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
    import warnings

    from logitgate.exceptions import CalibrationError, CalibrationWarning

    if (
        isinstance(min_temp, bool)
        or isinstance(max_temp, bool)
        or not math.isfinite(min_temp)
        or not math.isfinite(max_temp)
        or min_temp <= 0.0
        or max_temp <= min_temp
    ):
        raise CalibrationError(
            f"Expected finite 0 < min_temp < max_temp, got min_temp={min_temp}, max_temp={max_temp}"
        )
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
        raise CalibrationError(f"num_bins must be an integer >= 1, got {num_bins}")

    if not samples:
        warnings.warn(
            "fit_calibration received an empty sample sequence; returning default optimal_temperature=1.0.",
            CalibrationWarning,
            stacklevel=2,
        )
        return CalibrationResult(
            optimal_temperature=1.0,
            brier_score_before=0.0,
            brier_score_after=0.0,
            ece_before=0.0,
            ece_after=0.0,
        )

    labels_seen: set[int] = set()
    for f_lo, r_lo, y in samples:
        if isinstance(f_lo, bool) or isinstance(r_lo, bool):
            raise CalibrationError(
                f"Calibration samples must contain finite float log-odds, got ({f_lo!r}, {r_lo!r})"
            )
        try:
            f_val = float(f_lo)
            r_val = float(r_lo)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                f"Calibration samples must contain finite log-odds, got ({f_lo!r}, {r_lo!r})"
            ) from exc
        if not math.isfinite(f_val) or not math.isfinite(r_val):
            raise CalibrationError(
                f"Calibration samples must contain finite log-odds, got ({f_lo}, {r_lo})"
            )
        if isinstance(y, bool) or y not in (0, 1):
            raise CalibrationError(f"Calibration sample label must be 0 or 1, got {y!r}")
        labels_seen.add(int(y))

    if len(labels_seen) < 2:
        warnings.warn(
            "fit_calibration received samples from only a single class; temperature scaling is degenerate.",
            CalibrationWarning,
            stacklevel=2,
        )

    def calc_prob(f_lo: float, r_lo: float, temp: float) -> float:
        return combine_symmetric_log_odds(f_lo, r_lo, logit_temperature=temp)

    def brier_score(temp: float) -> float:
        total = sum((calc_prob(f, r, temp) - float(y)) ** 2 for f, r, y in samples)
        return total / len(samples)

    # Mild log-temperature regularization prevents T -> min_temp collapse on small,
    # 100%-separable micro-calibration datasets (where raw Brier decreases monotonically as T -> 0).
    reg_weight = 1e-2 / max(1.0, math.sqrt(len(samples) / 8.0))

    def objective(temp: float) -> float:
        return brier_score(temp) + reg_weight * (math.log(max(temp, 1e-6)) ** 2)

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
    brier_before = brier_score(1.0)
    best_temp = 1.0
    best_obj = objective(1.0) if min_temp <= 1.0 <= max_temp else float("inf")
    steps = 200
    for step in range(steps + 1):
        temp = min_temp + step * (max_temp - min_temp) / steps
        obj_val = objective(temp)
        if obj_val < best_obj - 1e-12:
            best_obj = obj_val
            best_temp = temp

    # 2. Golden-section refinement around coarse minimum
    a = max(min_temp, best_temp - (max_temp - min_temp) / steps)
    b = min(max_temp, best_temp + (max_temp - min_temp) / steps)
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = objective(c)
    fd = objective(d)

    for _ in range(30):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = objective(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = objective(d)

    optimal_temp = (a + b) / 2.0
    brier_after = brier_score(optimal_temp)
    if brier_after >= brier_before - 1e-12 and min_temp <= 1.0 <= max_temp:
        optimal_temp = 1.0
        brier_after = brier_before
    ece_before = compute_ece(1.0)
    ece_after = compute_ece(optimal_temp)

    return CalibrationResult(
        optimal_temperature=round(optimal_temp, 4),
        brier_score_before=round(brier_before, 6),
        brier_score_after=round(brier_after, 6),
        ece_before=round(ece_before, 6),
        ece_after=round(ece_after, 6),
    )
