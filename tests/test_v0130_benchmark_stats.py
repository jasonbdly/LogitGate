"""Unit and statistical tests for LogitGate v0.13.0 Benchmark Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from logitgate import (
    DatasetAuditReport,
    audit_benchmark_dataset,
    bootstrap_metric_ci,
    compute_calibration_metrics,
    mcnemar_exact_test,
    wilson_confidence_interval,
)
from logitgate.backends.base import StubBackend
from logitgate.benchmark import (
    BUILTIN_CASES,
    BenchmarkReport,
    ModeStats,
    format_report,
    run_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# 1. Wilson Confidence Interval
# -----------------------------------------------------------------------------


def test_wilson_confidence_interval_bounds_and_values() -> None:
    # Empty set returns uninformative [0.0, 1.0]
    assert wilson_confidence_interval(0, 0) == (0.0, 1.0)

    # All failures / all successes bounded in [0.0, 1.0]
    lo_0, hi_0 = wilson_confidence_interval(0, 100, 0.95)
    assert lo_0 == 0.0
    assert 0.0 < hi_0 < 0.05

    lo_100, hi_100 = wilson_confidence_interval(100, 100, 0.95)
    assert 0.95 < lo_100 < 1.0
    assert hi_100 == 1.0

    # Symmetric case: 50 / 100 at 95% confidence
    lo_50, hi_50 = wilson_confidence_interval(50, 100, 0.95)
    assert lo_50 == pytest.approx(0.4038, abs=1e-3)
    assert hi_50 == pytest.approx(0.5962, abs=1e-3)
    assert (lo_50 + hi_50) / 2.0 == pytest.approx(0.5, abs=1e-6)

    # Monotonicity with confidence level
    lo_90, hi_90 = wilson_confidence_interval(50, 100, 0.90)
    lo_99, hi_99 = wilson_confidence_interval(50, 100, 0.99)
    assert lo_99 < lo_50 < lo_90
    assert hi_90 < hi_50 < hi_99


def test_wilson_confidence_interval_input_validation() -> None:
    # Rejects bools as successes, total, confidence
    with pytest.raises(TypeError, match="successes must be an integer"):
        wilson_confidence_interval(cast(Any, True), 10)
    with pytest.raises(TypeError, match="total must be an integer"):
        wilson_confidence_interval(5, cast(Any, True))
    with pytest.raises(TypeError, match="confidence must be a float"):
        wilson_confidence_interval(5, 10, cast(Any, False))

    # Rejects negative successes or successes > total
    with pytest.raises(ValueError, match="successes must be between 0 and total"):
        wilson_confidence_interval(-1, 10)
    with pytest.raises(ValueError, match="successes must be between 0 and total"):
        wilson_confidence_interval(11, 10)

    # Rejects invalid confidence levels
    with pytest.raises(ValueError, match="confidence must be a finite float"):
        wilson_confidence_interval(5, 10, 0.0)
    with pytest.raises(ValueError, match="confidence must be a finite float"):
        wilson_confidence_interval(5, 10, 1.0)
    with pytest.raises(ValueError, match="confidence must be a finite float"):
        wilson_confidence_interval(5, 10, float("nan"))


# -----------------------------------------------------------------------------
# 2. Bootstrap Metric Confidence Interval
# -----------------------------------------------------------------------------


def test_bootstrap_metric_ci_deterministic_and_exact() -> None:
    # Constant input sequence produces identical constant CI
    lo, hi = bootstrap_metric_ci([1.5] * 50, n_resamples=200, confidence=0.95, seed=123)
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(1.5)

    # Seed reproducibility
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ci1 = bootstrap_metric_ci(vals, n_resamples=500, seed=42)
    ci2 = bootstrap_metric_ci(vals, n_resamples=500, seed=42)
    assert ci1 == ci2

    # CI brackets the sample mean
    mean = sum(vals) / len(vals)
    assert ci1[0] <= mean <= ci1[1]


def test_bootstrap_metric_ci_input_validation() -> None:
    with pytest.raises(ValueError, match="values cannot be empty"):
        bootstrap_metric_ci([])
    with pytest.raises(TypeError, match="n_resamples must be an integer"):
        bootstrap_metric_ci([1.0, 2.0], n_resamples=cast(Any, True))
    with pytest.raises(ValueError, match="n_resamples must be >= 10"):
        bootstrap_metric_ci([1.0, 2.0], n_resamples=5)
    with pytest.raises(TypeError, match="confidence must be a float"):
        bootstrap_metric_ci([1.0, 2.0], confidence=cast(Any, True))
    with pytest.raises(ValueError, match="confidence must be a finite float"):
        bootstrap_metric_ci([1.0, 2.0], confidence=1.5)
    with pytest.raises(TypeError, match="values must contain finite numeric floats"):
        bootstrap_metric_ci([1.0, cast(Any, "two")])
    with pytest.raises(ValueError, match="values must contain finite numeric floats"):
        bootstrap_metric_ci([1.0, float("nan")])


# -----------------------------------------------------------------------------
# 3. McNemar Exact Test
# -----------------------------------------------------------------------------


def test_mcnemar_exact_test_concordant_and_asymmetric() -> None:
    # 1. Perfectly concordant vectors: discordant = 0 -> p_value = 1.0
    vec_a = [True, False, True, True, False]
    res_ident = mcnemar_exact_test(vec_a, vec_a)
    assert res_ident == {
        "n_pairs": 5,
        "a_only_correct": 0,
        "b_only_correct": 0,
        "discordant": 0,
        "p_value": 1.0,
    }

    # 2. Symmetric discordant: a_only = 5, b_only = 5 -> p_value = 1.0
    a_sym = [True] * 5 + [False] * 5
    b_sym = [False] * 5 + [True] * 5
    res_sym = mcnemar_exact_test(a_sym, b_sym)
    assert res_sym["a_only_correct"] == 5
    assert res_sym["b_only_correct"] == 5
    assert res_sym["discordant"] == 10
    assert res_sym["p_value"] == 1.0

    # 3. Completely asymmetric discordant: a_only = 10, b_only = 0
    # Two-sided exact binomial: 2 * (1 / 2^10) = 2 / 1024 = 0.001953
    a_asym = [True] * 10
    b_asym = [False] * 10
    res_asym = mcnemar_exact_test(a_asym, b_asym)
    assert res_asym["a_only_correct"] == 10
    assert res_asym["b_only_correct"] == 0
    assert res_asym["discordant"] == 10
    assert res_asym["p_value"] == round(2.0 / 1024.0, 6)

    # 4. Swapped order gives identical p_value
    res_swapped = mcnemar_exact_test(b_asym, a_asym)
    assert res_swapped["p_value"] == res_asym["p_value"]


def test_mcnemar_exact_test_input_validation() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        mcnemar_exact_test([], [])
    with pytest.raises(ValueError, match="must have equal length"):
        mcnemar_exact_test([True], [True, False])


# -----------------------------------------------------------------------------
# 4. Calibration Metrics: Brier, ECE, AUROC, AURC
# -----------------------------------------------------------------------------


def test_compute_calibration_metrics_perfect_and_inverted() -> None:
    # 1. Perfect calibration & separation
    probs_perf = [0.0, 0.0, 1.0, 1.0]
    labels_perf = [False, False, True, True]
    met_perf = compute_calibration_metrics(probs_perf, labels_perf, n_bins=10)
    assert met_perf["brier_score"] == 0.0
    assert met_perf["ece"] == 0.0
    assert met_perf["auroc"] == 1.0
    assert met_perf["aurc"] == 0.0

    # 2. Inverted predictions (worst AUROC)
    probs_inv = [1.0, 1.0, 0.0, 0.0]
    labels_inv = [False, False, True, True]
    met_inv = compute_calibration_metrics(probs_inv, labels_inv, n_bins=10)
    assert met_inv["brier_score"] == 1.0
    assert met_inv["auroc"] == 0.0
    assert met_inv["aurc"] == 1.0

    # 3. Single-class labels returns AUROC = None
    met_single = compute_calibration_metrics([0.2, 0.8], [True, True])
    assert met_single["auroc"] is None


def test_compute_calibration_metrics_aurc_abstention_ordering() -> None:
    # Both Model A and Model B make 1 error out of 4 cases (75% accuracy).
    # Model A makes its error with LOW confidence (p=0.51, true=False).
    # Model B makes its error with HIGH confidence (p=0.99, true=False).
    # Model A must have strictly lower AURC than Model B because its selective
    # risk profile is superior when filtering low-confidence predictions.
    labels = [True, True, True, False]
    probs_a = [0.95, 0.90, 0.85, 0.51]  # error at lowest confidence margin 0.01
    probs_b = [0.51, 0.60, 0.70, 0.99]  # error at highest confidence margin 0.49

    met_a = compute_calibration_metrics(probs_a, labels)
    met_b = compute_calibration_metrics(probs_b, labels)
    assert met_a["aurc"] is not None and met_b["aurc"] is not None
    assert met_a["aurc"] < met_b["aurc"]


def test_compute_calibration_metrics_input_validation() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        compute_calibration_metrics([], [])
    with pytest.raises(ValueError, match="must have equal length"):
        compute_calibration_metrics([0.5], [True, False])
    with pytest.raises(TypeError, match="n_bins must be an integer"):
        compute_calibration_metrics([0.5], [True], n_bins=cast(Any, True))
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        compute_calibration_metrics([0.5], [True], n_bins=1)
    with pytest.raises(TypeError, match="probabilities must contain finite floats"):
        compute_calibration_metrics([cast(Any, True)], [True])
    with pytest.raises(ValueError, match="probabilities must be in"):
        compute_calibration_metrics([1.5], [True])


# -----------------------------------------------------------------------------
# 5. ModeStats, run_benchmark, format_report, and to_dict Upgrades
# -----------------------------------------------------------------------------


def test_mode_stats_summary_extended_metrics() -> None:
    stats = ModeStats(
        mode="logit",
        latencies_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
        generated_tokens=[0, 0, 0, 0, 0],
        llm_calls=[2, 2, 2, 2, 2],
        correct=[True, True, True, False, True],
        ambiguous=[False, False, True, False, False],
        judge_skipped=[True, True, True, True, True],
        probabilities=[0.9, 0.8, 0.5, 0.3, 0.85],
        labels=[True, True, False, True, True],
        cold_latencies_ms=[10.0, 20.0],
        warm_latencies_ms=[30.0, 40.0, 50.0],
    )
    s = stats.summary()

    # Base keys
    assert s["mode"] == "logit"
    assert s["runs"] == 5
    assert s["accuracy"] == 0.8

    # Extended statistical keys
    assert s["p99_ms"] == pytest.approx(50.0)
    assert s["cold_prefix_p50_ms"] == pytest.approx(10.0)  # nearest-rank percentile of [10.0, 20.0]
    assert s["warm_prefix_p50_ms"] == pytest.approx(40.0)  # nearest-rank percentile of [30.0, 40.0, 50.0]
    assert len(s["accuracy_ci95"]) == 2
    assert s["accuracy_ci95"][0] <= 0.8 <= s["accuracy_ci95"][1]
    assert s["coverage"] == 0.8  # 4/5 decisive
    assert s["selective_risk"] == 0.25  # 1 error among 4 decisive runs
    assert isinstance(s["calibration"], dict)
    assert "brier_score" in s["calibration"]
    assert "ece" in s["calibration"]
    assert "auroc" in s["calibration"]
    assert "aurc" in s["calibration"]


def test_run_benchmark_harness_integration_and_comparisons() -> None:
    # Use distinct rubrics so at least one rubric is not warmed up by warmup=1
    cases = [BUILTIN_CASES[0], BUILTIN_CASES[4], BUILTIN_CASES[6]]
    backend = StubBackend(default_prob=0.88, diagnostic_text="[VERDICT: PASS]", tokens_generated=15)

    report = run_benchmark(
        cases,
        backend=backend,
        modes=["judge", "cascade"],
        repeat=2,
        warmup=1,
    )
    assert isinstance(report, BenchmarkReport)

    # 1. Verify cold vs warm prefix tracking
    casc = report.modes["cascade"]
    assert len(casc.cold_latencies_ms) > 0
    assert len(casc.warm_latencies_ms) > 0
    summary = casc.summary()
    assert summary["p99_ms"] is not None
    assert summary["cold_prefix_p50_ms"] is not None
    assert summary["warm_prefix_p50_ms"] is not None

    # 2. Verify BenchmarkReport.to_dict comparisons
    rep_dict = report.to_dict()
    assert "metadata" in rep_dict
    assert "modes" in rep_dict
    assert "comparisons" in rep_dict
    assert "cascade_vs_judge" in rep_dict["comparisons"]
    comp = rep_dict["comparisons"]["cascade_vs_judge"]
    assert "n_pairs" in comp
    assert comp["n_pairs"] == len(casc.correct)
    assert "p_value" in comp
    assert "accuracy_delta" in comp
    assert "accuracy_delta_ci95" in comp

    # 3. Verify format_report output
    table_text = format_report(report)
    assert "p99 ms" in table_text
    assert "95% CI" in table_text
    assert "cascade vs judge:" in table_text
    assert "McNemar p=" in table_text
    # Enforce repo style rule: no em-dashes
    assert "—" not in table_text


# -----------------------------------------------------------------------------
# 6. audit_benchmark_dataset over pairwise and reasoning-tier schemas
# -----------------------------------------------------------------------------


def test_audit_benchmark_dataset_pairwise_items_schema(tmp_path: Path) -> None:
    pairwise_data = {
        "pairwise_items": [
            {
                "id": "pair_1",
                "rubric": "Is the code secure against SQL injection?",
                "candidate_a": "cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))",
                "candidate_b": "cursor.execute(f'SELECT * FROM users WHERE id = {uid}')",
                "winner": "a",
                "domain": "security",
            },
            {
                "id": "pair_2",
                "rubric": "Does the function handle empty inputs gracefully?",
                "candidate_a": "def avg(xs): return sum(xs) / len(xs)",
                "candidate_b": "def avg(xs): return sum(xs) / len(xs) if xs else 0.0",
                "winner": 1,
                "domain": "correctness",
            },
        ]
    }
    p = tmp_path / "pairwise.json"
    p.write_text(json.dumps(pairwise_data), encoding="utf-8")

    rep = audit_benchmark_dataset(p)
    assert isinstance(rep, DatasetAuditReport)
    # 2 pairs produce 4 normalized cases (2 winners=True, 2 losers=False)
    assert rep.total_cases == 4
    assert rep.num_positive == 2
    assert rep.num_negative == 2
    assert rep.positive_rate == 0.5
    # Pairwise extraction is exact, not a synthetic multi-class projection
    assert not any("Multi-class/ordinal schema projected" in w for w in rep.warnings)


def test_audit_benchmark_dataset_reasoning_tiers_and_choice_score_schemas() -> None:
    # 1. Reasoning tiers file (reasoning_tiers_32.json)
    tiers_path = REPO_ROOT / "benchmarks" / "data" / "reasoning_tiers_32.json"
    rep_tiers = audit_benchmark_dataset(tiers_path)
    assert rep_tiers.total_cases == 32
    assert rep_tiers.num_positive == 16
    assert rep_tiers.num_negative == 16
    assert not any("Multi-class/ordinal schema projected" in w for w in rep_tiers.warnings)

    # 2. Multi-class / ordinal choice and score file (choice_and_score_12.json)
    choice_score_path = REPO_ROOT / "benchmarks" / "data" / "choice_and_score_12.json"
    rep_cs = audit_benchmark_dataset(choice_score_path)
    assert rep_cs.total_cases == 12
    # Choice/score items without explicit boolean labels are projected with a warning
    assert any("Multi-class/ordinal schema projected to binary halves" in w for w in rep_cs.warnings)


# -----------------------------------------------------------------------------
# 7. Public API Contract Re-exports
# -----------------------------------------------------------------------------


def test_public_reexports() -> None:
    import logitgate

    for fn_name in (
        "bootstrap_metric_ci",
        "compute_calibration_metrics",
        "mcnemar_exact_test",
        "wilson_confidence_interval",
    ):
        assert hasattr(logitgate, fn_name)
        assert fn_name in logitgate.__all__


def test_audit_detects_artificial_padding_artifacts() -> None:
    padded = [
        {"rubric": "Check", "candidate": f"Valid clinical guidance statement {i}.", "expected": True}
        for i in range(20)
    ] + [
        {"rubric": "Check", "candidate": f"Invalid guidance {i}               .", "expected": False}
        for i in range(20)
    ]
    rep = audit_benchmark_dataset(padded)
    assert any("Artificial length-padding artifact" in w for w in rep.warnings)

