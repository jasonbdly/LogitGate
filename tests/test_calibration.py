from __future__ import annotations

import json
from pathlib import Path

import pytest

from logitgate.calibration.conformal import (
    ConformalCascadeThresholds,
    calibrate_cascade_thresholds,
    load_calibration_dataset,
)


def test_calibrate_cascade_thresholds_guarantees():
    # 20 samples: 10 positives, 10 negatives
    calibrated_probs = [
        0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,  # Negatives (y=False)
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98,  # Positives (y=True)
    ]
    labels = [False] * 10 + [True] * 10

    thresholds = calibrate_cascade_thresholds(
        calibrated_probs=calibrated_probs,
        ground_truth_labels=labels,
        target_risk_alpha=0.10,
    )

    assert isinstance(thresholds, ConformalCascadeThresholds)
    assert thresholds.calibration_set_size == 20
    assert thresholds.target_risk_alpha == 0.10
    assert 0.0 < thresholds.tau_reject < thresholds.tau_accept < 1.0
    assert 0.0 <= thresholds.empirical_escalation_rate <= 1.0


def test_calibrate_cascade_thresholds_validation_errors():
    with pytest.raises(ValueError, match="at least 10 samples"):
        calibrate_cascade_thresholds([0.5] * 5, [True] * 5)

    with pytest.raises(ValueError, match="target_risk_alpha must be strictly between"):
        calibrate_cascade_thresholds([0.5] * 10, [True] * 5 + [False] * 5, target_risk_alpha=1.5)

    with pytest.raises(ValueError, match="both positive and negative"):
        calibrate_cascade_thresholds([0.5] * 12, [True] * 12)


def test_conformal_non_crossing_correction():
    # Ambiguous model where raw conformal thresholds cross (tau_reject >= tau_accept)
    calibrated_probs = [0.4] * 10 + [0.4] * 10
    labels = [False] * 10 + [True] * 10

    thresholds = calibrate_cascade_thresholds(
        calibrated_probs=calibrated_probs,
        ground_truth_labels=labels,
        target_risk_alpha=0.05,
    )
    # Midpoint non-crossing logic must ensure tau_reject < tau_accept
    assert thresholds.tau_reject < thresholds.tau_accept
    assert thresholds.tau_reject == 0.35
    assert thresholds.tau_accept == 0.45
    assert thresholds.empirical_escalation_rate == 1.0


def test_load_calibration_dataset_markdown(tmp_path: Path):
    md_content = """# Test Dataset
| Rubric | Candidate | Ground Truth |
| :--- | :--- | :--- |
| Capital of France? | Paris | PASS |
| Capital of France? | Berlin | FAIL |
| Square of 3 is 9? | 3 * 3 = 9 | PASS |
"""
    p = tmp_path / "test_data.md"
    p.write_text(md_content, encoding="utf-8")

    samples = load_calibration_dataset(p)
    assert len(samples) == 3
    assert samples[0] == ("Capital of France?", "Paris", True)
    assert samples[1] == ("Capital of France?", "Berlin", False)
    assert samples[2] == ("Square of 3 is 9?", "3 * 3 = 9", True)


def test_load_calibration_dataset_markdown_sections(tmp_path: Path):
    md_path = tmp_path / "sections.md"
    md_path.write_text("""# Epistemic Test Cases

### Case 1: Simple Math
**Rubric**: Is 2 + 2 = 4?
**Candidate**: 2 + 2 = 4
**Label**: PASS

### Case 2: Broken Math
**Rubric**: Is 2 + 2 = 4?
**Candidate**: 2 + 2 = 5
**Label**: FAIL
""")
    samples = load_calibration_dataset(md_path)
    assert len(samples) == 2
    assert samples[0] == ("Is 2 + 2 = 4?", "2 + 2 = 4", True)
    assert samples[1] == ("Is 2 + 2 = 4?", "2 + 2 = 5", False)


def test_load_calibration_dataset_json_and_jsonl(tmp_path: Path):
    # 1. JSON List
    json_list_path = tmp_path / "data_list.json"
    json_list_path.write_text(json.dumps([
        {"rubric": "Is Python?", "candidate": "x = 1", "label": "PASS"},
        {"rubric": "Is Python?", "candidate": "def (: 1", "label": "FAIL"},
    ]))
    samples_list = load_calibration_dataset(json_list_path)
    assert len(samples_list) == 2
    assert samples_list[0] == ("Is Python?", "x = 1", True)
    assert samples_list[1] == ("Is Python?", "def (: 1", False)

    # 2. JSON Dict with 'test_cases'
    json_dict_path = tmp_path / "data_dict.json"
    json_dict_path.write_text(json.dumps({
        "test_cases": [
            {"rubric": "R1", "candidate": "C1", "ground_truth": 1},
            {"rubric": "R2", "candidate": "C2", "ground_truth": 0},
        ]
    }))
    samples_dict = load_calibration_dataset(json_dict_path)
    assert len(samples_dict) == 2
    assert samples_dict[0] == ("R1", "C1", True)
    assert samples_dict[1] == ("R2", "C2", False)

    # 3. JSONL
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        '{"rubric": "R1", "actual_output": "Out1", "passed": true}\n'
        '{"rubric": "R2", "actual_output": "Out2", "passed": false}\n'
    )
    samples_jsonl = load_calibration_dataset(jsonl_path)
    assert len(samples_jsonl) == 2
    assert samples_jsonl[0] == ("R1", "Out1", True)
    assert samples_jsonl[1] == ("R2", "Out2", False)


def test_load_calibration_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        load_calibration_dataset("non_existent_file.xyz")


def test_load_epistemic_reasoning_test_cases_md():
    samples = load_calibration_dataset("EPISTEMIC_REASONING_TEST_CASES.md")
    assert len(samples) >= 15
    # Ensure balanced positive and negative ground truths
    pos_count = sum(1 for _, _, y in samples if y)
    neg_count = sum(1 for _, _, y in samples if not y)
    assert pos_count >= 5
    assert neg_count >= 5
