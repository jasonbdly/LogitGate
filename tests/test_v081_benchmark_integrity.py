"""CI benchmark integrity, anti-shortcut, conformal power, and cross-split leakage tests (v0.8.1)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from logitgate import (
    CalibrationError,
    DatasetAuditReport,
    SplitLeakageReport,
    audit_benchmark_dataset,
    check_split_leakage,
)
from logitgate.benchmark import BUILTIN_CASES
from logitgate.cli import main
from logitgate.debiasing.permutation import UNIVERSAL_CALIBRATION_DATASET

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "benchmarks" / "data"


def test_cal_distill_80_quad_structure_length_parity_and_conformal_power() -> None:
    path = DATA_DIR / "cal_distill_80.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw) == 80

    # Verify exact 2x2 contrastive quad completeness across 20 quads and 4 Split-A domains
    quads: dict[str, set[str]] = {}
    for item in raw:
        quads.setdefault(item["quad_id"], set()).add(item["variant"])
    assert len(quads) == 20
    expected_variants = {"clean_pass", "hard_positive", "direct_fail", "hard_negative"}
    for qid, variants in quads.items():
        assert variants == expected_variants, f"Incomplete quad {qid}: {variants}"

    rep = audit_benchmark_dataset(path)
    assert isinstance(rep, DatasetAuditReport)
    assert rep.total_cases == 80
    assert rep.num_positive == 40
    assert rep.num_negative == 40
    assert rep.holds_at_alpha_010 is True
    assert rep.holds_at_alpha_005 is True
    assert rep.char_length_rel_diff <= 0.08
    assert abs(rep.length_auroc - 0.50) <= 0.08
    assert rep.lexical_loo_accuracy <= 0.55
    assert rep.duplicate_candidates == 0
    assert rep.warnings == ()


def test_ood_generalization_96_deconfounded_and_strictly_disjoint() -> None:
    ood_path = DATA_DIR / "ood_generalization_96.json"
    raw = json.loads(ood_path.read_text(encoding="utf-8"))
    assert len(raw) == 96

    quads: dict[str, set[str]] = {}
    for item in raw:
        quads.setdefault(item["quad_id"], set()).add(item["variant"])
    assert len(quads) == 24
    expected_variants = {"clean_pass", "hard_positive", "direct_fail", "hard_negative"}
    for qid, variants in quads.items():
        assert variants == expected_variants, f"Incomplete OOD quad {qid}: {variants}"

    rep = audit_benchmark_dataset(ood_path)
    assert rep.total_cases == 96
    assert rep.num_positive == 48
    assert rep.num_negative == 48
    assert len(rep.domains) == 6
    assert all(count == 16 for count in rep.domains.values())
    assert rep.holds_at_alpha_010 is True
    assert rep.holds_at_alpha_005 is True
    assert rep.char_length_rel_diff <= 0.10
    assert abs(rep.length_auroc - 0.50) <= 0.12
    assert rep.lexical_loo_accuracy <= 0.55
    assert rep.duplicate_candidates == 0
    assert rep.warnings == ()

    # Zero leakage against every calibration, built-in, and legacy benchmark dataset
    for ref_name, ref_src in (
        ("cal_distill_80.json", DATA_DIR / "cal_distill_80.json"),
        ("EPISTEMIC_REASONING_TEST_CASES.md", REPO_ROOT / "EPISTEMIC_REASONING_TEST_CASES.md"),
        ("UNIVERSAL_CALIBRATION_DATASET", UNIVERSAL_CALIBRATION_DATASET),
        ("BUILTIN_CASES", BUILTIN_CASES),
        ("multi_domain_24.json", DATA_DIR / "multi_domain_24.json"),
    ):
        leak = check_split_leakage(ref_src, ood_path, reference_label=ref_name)
        assert isinstance(leak, SplitLeakageReport)
        assert leak.is_clean is True, f"Leakage from {ref_name}: {leak.warnings}"
        assert leak.exact_candidate_overlaps == 0
        assert leak.exact_rubric_overlaps == 0
        assert leak.near_duplicate_candidates == 0
        assert leak.shared_domains == ()
        assert leak.max_candidate_jaccard < 0.15


def test_epistemic_markdown_80_and_streaming_and_drift_fixtures() -> None:
    md_rep = audit_benchmark_dataset(REPO_ROOT / "EPISTEMIC_REASONING_TEST_CASES.md")
    assert md_rep.total_cases == 80
    assert md_rep.num_positive == 40
    assert md_rep.num_negative == 40
    assert md_rep.holds_at_alpha_005 is True
    assert md_rep.char_length_rel_diff <= 0.08
    assert md_rep.warnings == ()

    stream_path = DATA_DIR / "streaming_trajectories_32.json"
    stream_raw = json.loads(stream_path.read_text(encoding="utf-8"))
    assert len(stream_raw) == 32
    traj_counts = Counter(item["trajectory_type"] for item in stream_raw)
    assert traj_counts == {
        "clean_throughout": 8,
        "false_alarm_prefix_resolved_safely": 8,
        "early_onset_violation": 8,
        "late_onset_pivot_violation": 8,
    }
    for item in stream_raw:
        assert isinstance(item["chunks"], list) and len(item["chunks"]) >= 4
        assert "".join(item["chunks"]) == item["candidate"]
        if item["expected"]:
            assert item["first_violation_chunk_index"] is None
        else:
            idx = item["first_violation_chunk_index"]
            assert isinstance(idx, int) and 0 <= idx < len(item["chunks"])

    stream_rep = audit_benchmark_dataset(stream_path)
    assert stream_rep.char_length_rel_diff <= 0.10
    assert abs(stream_rep.length_auroc - 0.50) <= 0.10

    drift_path = DATA_DIR / "drift_regimes_120.json"
    drift_raw = json.loads(drift_path.read_text(encoding="utf-8"))
    assert len(drift_raw) == 120
    assert [item["step"] for item in drift_raw] == list(range(120))
    regime_counts = Counter(item["regime"] for item in drift_raw)
    assert regime_counts == {
        "stationary_in_domain": 30,
        "covariate_domain_shift": 30,
        "adversarial_prevalence_shift": 30,
        "stationary_recovery": 30,
    }
    drift_rep = audit_benchmark_dataset(drift_path)
    assert drift_rep.duplicate_candidates == 0
    assert drift_rep.warnings == ()


def test_audit_detects_legacy_length_confounder_and_synthetic_shortcuts(tmp_path: Path) -> None:
    # 1. Verify detector flags the real +30.6% / 0.7465 AUROC length confounder in legacy MultiDomain-24
    legacy_rep = audit_benchmark_dataset(DATA_DIR / "multi_domain_24.json")
    assert legacy_rep.char_length_rel_diff > 0.25
    assert legacy_rep.length_auroc > 0.70
    assert any("Length confounder" in w for w in legacy_rep.warnings)
    assert any("Conformal guarantee" in w for w in legacy_rep.warnings)

    # 2. Verify detector catches pure lexical unigram shortcuts even when lengths match identically
    shortcut_items = [
        {
            "rubric": "Check compliance",
            "candidate": f"verified compliant clean safe module {i:02d}",
            "expected": True,
        }
        for i in range(15)
    ] + [
        {
            "rubric": "Check compliance",
            "candidate": f"broken unsafe exploit danger defect {i:02d}",
            "expected": False,
        }
        for i in range(15)
    ]
    sc_rep = audit_benchmark_dataset(shortcut_items)
    assert sc_rep.char_length_rel_diff < 0.05
    assert sc_rep.lexical_loo_accuracy >= 0.95
    assert any("Surface lexical shortcut" in w for w in sc_rep.warnings)

    # 3. Verify symlink rejection and invalid jaccard_threshold validation
    real_file = tmp_path / "real.json"
    real_file.write_text(json.dumps(shortcut_items), encoding="utf-8")
    symlink_file = tmp_path / "link.json"
    symlink_file.symlink_to(real_file)
    with pytest.raises(CalibrationError, match="symlink"):
        audit_benchmark_dataset(symlink_file)
    with pytest.raises(CalibrationError, match="jaccard_threshold"):
        check_split_leakage(shortcut_items, shortcut_items, jaccard_threshold=0.0)
    with pytest.raises(CalibrationError, match="jaccard_threshold"):
        check_split_leakage(shortcut_items, shortcut_items, jaccard_threshold=True)

    # 4. Verify near-duplicate shingle leakage detection
    ref_split = [
        {
            "rubric": "R1",
            "candidate": "alpha beta gamma delta epsilon zeta eta theta",
            "expected": True,
            "domain": "dom_a",
        }
    ]
    eval_contaminated = [
        {
            "rubric": "R2",
            "candidate": "alpha beta gamma delta epsilon zeta eta omega",
            "expected": True,
            "domain": "dom_a",
        }
    ]
    leak = check_split_leakage(ref_split, eval_contaminated, jaccard_threshold=0.60)
    assert leak.is_clean is False
    assert leak.near_duplicate_candidates == 1
    assert leak.shared_domains == ("dom_a",)


def test_cli_benchmark_audit_only_and_strict_mode(capsys: pytest.CaptureFixture[str]) -> None:
    # Clean OOD dataset vs Split-A calibration reference passes --strict-audit with exit 0
    rc_clean = main(
        [
            "benchmark",
            "--audit-only",
            "--strict-audit",
            "--dataset",
            str(DATA_DIR / "ood_generalization_96.json"),
            "--reference-dataset",
            str(DATA_DIR / "cal_distill_80.json"),
            "--json",
        ]
    )
    assert rc_clean == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit"]["total_cases"] == 96
    assert payload["audit"]["warnings"] == []
    assert payload["leakage"]["is_clean"] is True

    # Legacy confounded dataset fails --strict-audit with exit 1, passes non-strict with exit 0
    rc_legacy_strict = main(
        [
            "benchmark",
            "--audit-only",
            "--strict-audit",
            "--dataset",
            str(DATA_DIR / "multi_domain_24.json"),
        ]
    )
    assert rc_legacy_strict == 1
    human_out = capsys.readouterr().out
    assert "Length confounder detected" in human_out

    rc_legacy_info = main(
        [
            "benchmark",
            "--audit-only",
            "--dataset",
            str(DATA_DIR / "multi_domain_24.json"),
        ]
    )
    assert rc_legacy_info == 0
    capsys.readouterr()
