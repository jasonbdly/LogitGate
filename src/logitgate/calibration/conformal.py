from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logitgate.exceptions import CalibrationError

_GRID = 1e4  # thresholds are reported at 4 decimals, like LogitGate probabilities
_EPS = 1e-7  # absorbs float noise such as 0.45 * 1e4 == 4500.000000000001


def _ceil4(x: float) -> float:
    return round(math.ceil(x * _GRID - _EPS) / _GRID, 4)


def _floor4(x: float) -> float:
    return round(math.floor(x * _GRID + _EPS) / _GRID, 4)


def _strictly_above4(x: float) -> float:
    """Smallest 4-decimal value strictly greater than ``x``."""
    return round((math.floor(x * _GRID + _EPS) + 1) / _GRID, 4)


def _strictly_below4(x: float) -> float:
    """Largest 4-decimal value strictly less than ``x``."""
    return round((math.ceil(x * _GRID - _EPS) - 1) / _GRID, 4)


def min_samples_per_class(target_risk_alpha: float) -> int:
    """Smallest per-class calibration size for which the conformal bound is attainable.

    The split-conformal quantile is the ⌈(m+1)(1-a)⌉-th smallest of m scores, with
    a = alpha / 2 per class. Once that rank exceeds m, the valid threshold is +infinity
    ("never decide at Layer 1"), so we need m >= ⌈(1-a)/a⌉. That's 39 per class at
    alpha=0.05 and 19 at alpha=0.10.
    """
    if isinstance(target_risk_alpha, bool) or not math.isfinite(target_risk_alpha) or not (0.0 < target_risk_alpha < 1.0):
        raise CalibrationError(
            f"target_risk_alpha must be strictly between 0.0 and 1.0, got {target_risk_alpha}"
        )
    a = target_risk_alpha / 2.0
    return math.ceil((1.0 - a) / a - 1e-9)


@dataclass(frozen=True, slots=True)
class ConformalCascadeThresholds:
    """Layer-1 accept/reject cutoffs fitted on a labeled holdout set.

    Scores strictly between ``tau_reject`` and ``tau_accept`` are escalated to Layer 2. Use
    ``tau_accept`` as `LogitGate(pass_threshold=...)` and ``tau_reject`` as ``fail_threshold``.

    When ``guarantee_holds`` is True: the false-accept rate on negatives and the false-reject
    rate on positives are each <= alpha/2, so the overall Layer-1 error rate is <= alpha.
    This is a marginal guarantee, in expectation over the calibration draw, and holds only for
    inputs exchangeable with the calibration set. Refit whenever the model, prompt template,
    or input distribution changes. When False, the thresholds are a best-effort heuristic.
    """

    tau_reject: float  # Reject immediately at Layer 1 if P(Pass) <= tau_reject
    tau_accept: float  # Accept immediately at Layer 1 if P(Pass) >= tau_accept
    target_risk_alpha: float  # Maximum tolerated error rate at Layer 1
    empirical_escalation_rate: float
    calibration_set_size: int
    guarantee_holds: bool = True
    min_samples_per_class: int = 0
    num_positive: int = 0
    num_negative: int = 0


def calibrate_cascade_thresholds(
    calibrated_probs: Sequence[float],
    ground_truth_labels: Sequence[bool],
    target_risk_alpha: float = 0.05,
) -> ConformalCascadeThresholds:
    """Computes split-conformal selective-classification thresholds for the Layer 1 -> 2 handoff.

    Negatives' P(pass) bounds how high a fail can score (``tau_accept``); positives' 1 - P(pass)
    bounds how low a pass can score (``tau_reject``). Each side gets alpha/2.

    The thresholds are deliberately conservative in three ways:

    - **Strictness.** The conformal bound covers ``P > q``, but `LogitGate` accepts on
      ``P >= pass_threshold``. Thresholds are moved one 4-decimal step past the quantile so
      that ties with calibration scores (common, since probabilities are rounded) can't
      leak through.
    - **Rounding** always moves away from the decision region: accept rounds up, reject
      rounds down.
    - **Crossing.** If the raw cutoffs meet or cross (heavily overlapping classes), both move
      outward around their midpoint. That only widens the escalation band.

    If a class has fewer than `min_samples_per_class` samples, no finite threshold carries
    the guarantee. The result then has ``guarantee_holds=False`` and a `CalibrationWarning`
    is emitted, instead of a number being quietly passed off as a bound.

    Parameters:
        calibrated_probs: Model predicted pass probabilities P(Pass) in [0.0, 1.0].
        ground_truth_labels: Boolean ground truth labels (True = Pass, False = Fail).
        target_risk_alpha: Maximum tolerated error rate at Layer 1 (default: 0.05).

    Returns:
        ConformalCascadeThresholds with conservative tau_reject and tau_accept cutoffs.
    """
    n = len(calibrated_probs)
    if n != len(ground_truth_labels) or n < 10:
        raise CalibrationError("Calibration requires matching sequences of at least 10 samples.")

    if isinstance(target_risk_alpha, bool) or not math.isfinite(target_risk_alpha) or not (0.0 < target_risk_alpha < 1.0):
        raise CalibrationError(
            f"target_risk_alpha must be strictly between 0.0 and 1.0, got {target_risk_alpha}"
        )

    for p in calibrated_probs:
        if isinstance(p, bool):
            raise CalibrationError(
                f"calibrated_probs must contain finite probabilities in [0.0, 1.0], got {p!r}"
            )
        try:
            pf = float(p)
        except (TypeError, ValueError) as err:
            raise CalibrationError(
                f"calibrated_probs must contain finite probabilities in [0.0, 1.0], got {p!r}"
            ) from err
        if not math.isfinite(pf) or not (0.0 <= pf <= 1.0):
            raise CalibrationError(
                f"calibrated_probs must contain finite probabilities in [0.0, 1.0], got {p}"
            )

    for y in ground_truth_labels:
        if not isinstance(y, bool) and not (isinstance(y, int) and y in (0, 1)):
            raise CalibrationError(
                f"ground_truth_labels must contain booleans or 0/1 integers, got {y!r}"
            )

    # For ground truth False (0), high P(Pass) is a false-acceptance risk
    neg_scores = sorted(float(p) for p, y in zip(calibrated_probs, ground_truth_labels) if not y)
    # For ground truth True (1), low P(Pass) is a false-rejection risk
    pos_scores = sorted(1.0 - float(p) for p, y in zip(calibrated_probs, ground_truth_labels) if y)

    if not neg_scores or not pos_scores:
        raise CalibrationError(
            "Calibration set must contain both positive and negative ground truth labels."
        )

    half_alpha = target_risk_alpha / 2.0
    required = min_samples_per_class(target_risk_alpha)

    def _conformal_quantile(scores: list[float]) -> tuple[float, bool]:
        """Returns the ⌈(m+1)(1-a)⌉-th order statistic, and whether that rank exists."""
        m = len(scores)
        rank = math.ceil((m + 1) * (1.0 - half_alpha) - 1e-9)
        return scores[max(min(rank, m), 1) - 1], rank <= m

    q_accept, accept_ok = _conformal_quantile(neg_scores)
    q_reject, reject_ok = _conformal_quantile(pos_scores)
    raw_accept, raw_reject = q_accept, 1.0 - q_reject

    tau_accept = _strictly_above4(raw_accept)
    tau_reject = _strictly_below4(raw_reject)

    # Thresholds outside [0, 1] mean "never decide on this side". LogitGate's range is [0, 1];
    # pinning there admits only exact saturation (P == 1.0 / P == 0.0, |log-odds| > 50), and
    # if a calibration point itself sits at saturation, the guarantee can't be expressed.
    saturated = raw_accept >= 1.0 or raw_reject <= 0.0 or tau_accept > 1.0 or tau_reject < 0.0

    if raw_reject >= raw_accept:
        midpoint = 0.5 * (raw_reject + raw_accept)
        tau_reject = min(tau_reject, _floor4(midpoint - 0.05))
        tau_accept = max(tau_accept, _ceil4(midpoint + 0.05))

    tau_accept = min(tau_accept, 1.0)
    tau_reject = max(tau_reject, 0.0)
    if tau_reject >= tau_accept:
        tau_reject = max(0.0, _strictly_below4(tau_accept))

    guarantee_holds = accept_ok and reject_ok and not saturated
    if not guarantee_holds:
        import warnings

        from logitgate.exceptions import CalibrationWarning

        reasons: list[str] = []
        if not (accept_ok and reject_ok):
            reasons.append(
                f"need >= {required} positives and >= {required} negatives "
                f"(have {len(pos_scores)} / {len(neg_scores)})"
            )
        if saturated:
            reasons.append("a calibration score sits at P=0 or P=1")
        warnings.warn(
            f"Conformal guarantee does not hold at alpha={target_risk_alpha}: "
            + "; ".join(reasons)
            + ". Thresholds are best-effort only.",
            CalibrationWarning,
            stacklevel=2,
        )

    escalated = sum(1 for p in calibrated_probs if tau_reject < p < tau_accept)
    return ConformalCascadeThresholds(
        tau_reject=tau_reject,
        tau_accept=tau_accept,
        target_risk_alpha=target_risk_alpha,
        empirical_escalation_rate=round(escalated / float(n), 4),
        calibration_set_size=n,
        guarantee_holds=guarantee_holds,
        min_samples_per_class=required,
        num_positive=len(pos_scores),
        num_negative=len(neg_scores),
    )


def load_calibration_dataset(
    path: str | Path,
) -> list[tuple[str, str, bool]]:
    """Loads calibration samples (rubric, candidate, label) from JSON, JSONL, or Markdown files."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Calibration dataset file not found: {path}")

    content = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()

    # 1. JSON
    if suffix == ".json":
        data = json.loads(content)
        raw_items: list[dict[str, Any]]
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = (
                data.get("test_cases")
                or data.get("samples")
                or data.get("cases")
                or data.get("data")
                or []
            )
        else:
            raise CalibrationError(f"Unrecognized JSON structure in {path}")
        return _extract_samples_from_dicts(raw_items)

    # 2. JSONL
    if suffix == ".jsonl":
        raw_items = []
        for line in content.splitlines():
            line_str = line.strip()
            if line_str:
                raw_items.append(json.loads(line_str))
        return _extract_samples_from_dicts(raw_items)

    # 3. Markdown
    if suffix in (".md", ".markdown"):
        return _parse_markdown_dataset(content)

    # Fallback attempt JSON then Markdown
    try:
        data = json.loads(content)
        if isinstance(data, (list, dict)):
            raw_items = (
                data
                if isinstance(data, list)
                else (
                    data.get("test_cases")
                    or data.get("samples")
                    or data.get("cases")
                    or data.get("data")
                    or []
                )
            )
            return _extract_samples_from_dicts(raw_items)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return _parse_markdown_dataset(content)


def _extract_samples_from_dicts(
    items: Sequence[dict[str, Any]],
) -> list[tuple[str, str, bool]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise CalibrationError(f"Expected sequence of sample objects, got {type(items).__name__}")
    samples = []
    for item in items:
        if not isinstance(item, Mapping):
            raise CalibrationError(f"Expected JSON object for calibration sample, got {type(item).__name__}")
        rubric = str(
            item.get("rubric") or item.get("question") or item.get("instruction") or ""
        ).strip()
        raw_cand: Any = ""
        for cand_key in ("candidate", "actual_output", "output", "response", "text"):
            if item.get(cand_key) is not None:
                raw_cand = item[cand_key]
                break
        candidate = (
            json.dumps(raw_cand) if isinstance(raw_cand, (dict, list)) else str(raw_cand)
        ).strip()
        raw_label = item.get("label", item.get("ground_truth", item.get("expected", None)))
        if raw_label is None:
            raw_label = item.get("passed", item.get("pass", None))
        if raw_label is None:
            continue

        if isinstance(raw_label, bool):
            label = raw_label
        elif isinstance(raw_label, (int, float)):
            label = bool(raw_label >= 0.5)
        elif isinstance(raw_label, str):
            label = raw_label.strip().lower() in ("pass", "passed", "true", "1", "yes")
        else:
            label = bool(raw_label)

        if rubric and candidate:
            samples.append((rubric, candidate, label))
    return samples


def _parse_markdown_dataset(
    text: str,
) -> list[tuple[str, str, bool]]:
    """Parses markdown tables or sections formatted with Rubric, Candidate, Label."""
    samples: list[tuple[str, str, bool]] = []

    # Markdown table parsing
    table_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    if len(table_lines) >= 3:
        headers = [h.strip().lower() for h in table_lines[0].strip("|").split("|")]
        rubric_idx = next(
            (i for i, h in enumerate(headers) if "rubric" in h or "question" in h), -1
        )
        cand_idx = next(
            (
                i
                for i, h in enumerate(headers)
                if "candidate" in h or "output" in h or "text" in h or "input" in h
            ),
            -1,
        )
        label_idx = next(
            (
                i
                for i, h in enumerate(headers)
                if "label" in h or "truth" in h or "verdict" in h or "expected" in h
            ),
            -1,
        )

        if rubric_idx != -1 and cand_idx != -1 and label_idx != -1:
            for line in table_lines[2:]:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if (
                    len(cells) > max(rubric_idx, cand_idx, label_idx)
                    and not set(cells[0]) <= {"-", ":"}
                ):
                    rub = cells[rubric_idx]
                    cand = cells[cand_idx]
                    lab_str = cells[label_idx].lower()
                    lab = lab_str in ("pass", "passed", "true", "1", "yes")
                    if rub and cand:
                        samples.append((rub, cand, lab))
            if samples:
                return samples

    # Section / regex parsing
    case_blocks = re.split(r"\n#{2,4}\s+", text)
    for block in case_blocks:
        rub_match = re.search(r"\*\*(?:Rubric|Question)\*\*:\s*(.+)", block, re.IGNORECASE)
        cand_match = re.search(
            r"\*\*(?:Candidate|Output|Artifact)\*\*:\s*(.+)", block, re.IGNORECASE
        )
        lab_match = re.search(
            r"\*\*(?:Label|Ground Truth|Verdict)\*\*:\s*(.+)", block, re.IGNORECASE
        )

        if rub_match and cand_match and lab_match:
            rub = rub_match.group(1).strip()
            cand = cand_match.group(1).strip()
            lab_str = lab_match.group(1).strip().lower()
            lab = lab_str in ("pass", "passed", "true", "1", "yes")
            samples.append((rub, cand, lab))

    return samples
