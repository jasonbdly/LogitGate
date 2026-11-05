from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConformalCascadeThresholds:
    tau_reject: float  # Reject immediately at Layer 1 if P(Pass) <= tau_reject
    tau_accept: float  # Accept immediately at Layer 1 if P(Pass) >= tau_accept
    target_risk_alpha: float  # Maximum tolerated error rate at Layer 1
    empirical_escalation_rate: float
    calibration_set_size: int


def calibrate_cascade_thresholds(
    calibrated_probs: Sequence[float],
    ground_truth_labels: Sequence[bool],
    target_risk_alpha: float = 0.05,
) -> ConformalCascadeThresholds:
    """Computes finite-sample Conformal Selective Classification thresholds

    (tau_reject, tau_accept) for the Layer 1 -> Layer 2 Cascade handoff.

    Any sample with tau_reject < P(Pass) < tau_accept is escalated to JudgeGate (Layer 2).

    Parameters:
        calibrated_probs: Model predicted pass probabilities P(Pass) in [0.0, 1.0].
        ground_truth_labels: Boolean ground truth labels (True = Pass, False = Fail).
        target_risk_alpha: Maximum tolerated error rate at Layer 1 (default: 0.05).

    Returns:
        ConformalCascadeThresholds with bounded tau_reject and tau_accept cutoffs.
    """
    n = len(calibrated_probs)
    if n != len(ground_truth_labels) or n < 10:
        raise ValueError("Calibration requires matching sequences of at least 10 samples.")

    if not (0.0 < target_risk_alpha < 1.0):
        raise ValueError(
            f"target_risk_alpha must be strictly between 0.0 and 1.0, got {target_risk_alpha}"
        )

    # Separate non-conformity scores for negative and positive samples
    # For ground truth False (0), high P(Pass) is a false-acceptance risk
    neg_scores = sorted(
        [float(p) for p, y in zip(calibrated_probs, ground_truth_labels) if not y]
    )
    # For ground truth True (1), low P(Pass) is a false-rejection risk
    pos_scores = sorted(
        [1.0 - float(p) for p, y in zip(calibrated_probs, ground_truth_labels) if y]
    )

    if not neg_scores or not pos_scores:
        raise ValueError(
            "Calibration set must contain both positive and negative ground truth labels."
        )

    # Finite-sample conformal quantile index: ceil((m + 1) * (1 - alpha)) - 1
    def _conformal_quantile(scores: list[float], alpha: float) -> float:
        m = len(scores)
        q_idx = min(math.ceil((m + 1) * (1.0 - alpha)) - 1, m - 1)
        q_idx = max(q_idx, 0)
        return scores[q_idx]

    half_alpha = target_risk_alpha / 2.0
    tau_accept = _conformal_quantile(neg_scores, half_alpha)
    one_minus_tau_reject = _conformal_quantile(pos_scores, half_alpha)
    tau_reject = 1.0 - one_minus_tau_reject

    # Ensure non-crossing interval
    if tau_reject >= tau_accept:
        midpoint = 0.5 * (tau_reject + tau_accept)
        tau_reject = min(tau_reject, midpoint - 0.05)
        tau_accept = max(tau_accept, midpoint + 0.05)

    escalated = sum(1 for p in calibrated_probs if tau_reject < p < tau_accept)
    escalation_rate = escalated / float(n)

    return ConformalCascadeThresholds(
        tau_reject=round(max(0.01, tau_reject), 4),
        tau_accept=round(min(0.99, tau_accept), 4),
        target_risk_alpha=target_risk_alpha,
        empirical_escalation_rate=round(escalation_rate, 4),
        calibration_set_size=n,
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
            raise ValueError(f"Unrecognized JSON structure in {path}")
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
            raw_items = data if isinstance(data, list) else data.get("test_cases", [])
            return _extract_samples_from_dicts(raw_items)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    return _parse_markdown_dataset(content)


def _extract_samples_from_dicts(
    items: Sequence[dict[str, Any]],
) -> list[tuple[str, str, bool]]:
    samples = []
    for item in items:
        rubric = str(
            item.get("rubric") or item.get("question") or item.get("instruction") or ""
        ).strip()
        candidate = str(
            item.get("candidate")
            or item.get("actual_output")
            or item.get("output")
            or item.get("response")
            or item.get("text")
            or ""
        ).strip()
        raw_label = item.get("label", item.get("ground_truth", item.get("expected", None)))
        if raw_label is None:
            raw_label = item.get("passed", item.get("pass", True))

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
