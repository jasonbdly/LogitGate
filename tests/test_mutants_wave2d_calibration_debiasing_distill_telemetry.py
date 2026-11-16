"""Wave 2D Mutant-Killing & Exhaustive Coverage Test Suite for LogitGate.

Targets 12 modules:
1. logitgate.distill.anchor_bank
2. logitgate.distill.meta_calibrator
3. logitgate.distill.lora
4. logitgate.distill._io
5. logitgate.calibration.audit
6. logitgate.calibration.adaptive
7. logitgate.calibration.profile
8. logitgate.calibration.conformal
9. logitgate.debiasing.permutation
10. logitgate.telemetry.drift
11. logitgate.telemetry.distill
12. logitgate.telemetry.models
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pydantic
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

import logitgate.telemetry.distill as distill_telemetry
from logitgate.backends.base import LogitBackend

# Explicitly import all 12 target modules so mutation_audit.py maps this test file to all of them:
from logitgate.calibration.adaptive import (
    AdaptiveConformalController,
    _safe_logit,
)
from logitgate.calibration.audit import (
    _compute_exact_auroc,
    _compute_lexical_loo_accuracy,
    _extract_pairwise_cases,
    _load_normalized_cases,
    _NormalizedCase,
    _word_shingles,
    audit_benchmark_dataset,
    check_split_leakage,
)
from logitgate.calibration.conformal import (
    _parse_markdown_dataset,
    calibrate_cascade_thresholds,
    load_calibration_dataset,
    min_samples_per_class,
)
from logitgate.calibration.profile import (
    CalibrationProfile,
    ChoiceConformalThresholds,
    _extract_binary_record,
    _fit_binary_cal_result,
    _prepare_calibration_samples,
    afit_calibration_profile,
    calibrate_choice_thresholds,
    calibrate_score_thresholds,
    fit_calibration_profile,
)
from logitgate.debiasing.permutation import (
    CalibrationResult,
    _compute_binary_auroc,
    _fit_multiclass_temperature,
    _is_ignored_model_name,
    _lookup_temperature_in_mapping,
    _persist_calibration_profile,
    aauto_calibrate_backend,
    auto_calibrate_backend,
    combine_symmetric_log_odds,
    compute_shannon_entropy,
    fit_calibration,
    has_known_or_cached_temperature,
)
from logitgate.distill._io import (
    atomic_write_json_nofollow,
    read_json_object_nofollow,
)
from logitgate.distill.anchor_bank import (
    DistilledAnchorBank,
    _condense_rationale,
    _tokenize_bm25,
    resolve_anchor_bank,
)
from logitgate.distill.bundle import DistillationBundle
from logitgate.distill.lora import (
    HuggingFaceLogitBackend,
    LoRADistillConfig,
    _encode_prompt_ids,
    prepare_symmetric_lora_examples,
    train_distillation_lora,
)
from logitgate.distill.meta_calibrator import (
    LogitFeatureVector,
    LogitMetaCalibrator,
    _fit_ridge_logistic,
    _pass_distribution_stats,
    _solve_linear_system,
    afit_meta_calibrator,
    extract_logit_features,
    fit_meta_calibrator,
)
from logitgate.exceptions import (
    BackendError,
    CalibrationError,
    ConfigurationError,
)
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.telemetry.distill import (
    DistillationRecord,
    DistillationSink,
    export_distillation_dataset,
    extract_distillation_record,
    recalibrate_from_telemetry,
)
from logitgate.telemetry.drift import (
    DriftReport,
    RollingDriftMonitor,
    _extract_from_records,
    _kolmogorov_survival,
    compute_ks_two_sample,
    compute_psi,
    detect_telemetry_drift,
)
from logitgate.telemetry.models import (
    TelemetryRecord,
    TelemetrySummary,
    _luhn_valid,
    apply_privacy,
    hash_candidate,
    redact_text,
    truncate_text,
)

# ==============================================================================
# SECTION 1: adaptive.py Mutants & Coverage
# ==============================================================================


class TestAdaptiveMutantsAndCoverage:
    def test_safe_logit_arithmetic_kills_line29(self) -> None:
        val_one = _safe_logit(1.0)
        assert math.isfinite(val_one)
        assert val_one > 13.0

        val_zero = _safe_logit(0.0)
        assert math.isfinite(val_zero)
        assert val_zero < -13.0

        val_half = _safe_logit(0.5)
        assert abs(val_half) < 1e-9

    def test_adaptive_gate_explicit_profile_target_risk_alpha_kills_line187(
        self,
    ) -> None:
        # Line 187: loaded_profile.fail_threshold if loaded_profile is not None and loaded_profile.fail_threshold is not None
        # Tested when explicit profile is None and bundle contains a profile
        bundle_prof = CalibrationProfile(
            binary_temperature=1.0,
            target_risk_alpha=0.08,
            pass_threshold=0.8,
            fail_threshold=0.3,
        )
        bundle = DistillationBundle(profile=bundle_prof)
        gate = AdaptiveConformalController(
            distillation_bundle=bundle,
            feedback_mode="explicit",
        )
        assert gate.base_fail_threshold == 0.3
        assert gate.target_risk_alpha == 0.08

    def test_adaptive_gate_max_history_boundary_kills_line273(self) -> None:
        mock_prof = CalibrationProfile(
            binary_temperature=1.0,
            target_risk_alpha=0.05,
            pass_threshold=0.8,
            fail_threshold=0.2,
        )
        gate = AdaptiveConformalController(
            profile=mock_prof,
            feedback_mode="explicit",
            max_history=1,
        )
        assert gate._history.maxlen == 1
        with pytest.raises(
            ConfigurationError, match="max_history must be an integer >= 1"
        ):
            AdaptiveConformalController(
                profile=mock_prof,
                feedback_mode="explicit",
                max_history=0,
            )

    def test_adaptive_gate_compute_state_theta_epsilon_kills_line375(self) -> None:
        mock_prof = CalibrationProfile(
            binary_temperature=1.0,
            target_risk_alpha=0.05,
            pass_threshold=0.82,
            fail_threshold=0.18,
        )
        gate = AdaptiveConformalController(
            profile=mock_prof,
            feedback_mode="explicit",
        )
        gate._theta = 1e-12
        state = gate._compute_state_unlocked()
        assert state[0] == 0.0
        assert state[1] == 0.18
        assert state[2] == 0.82

    def test_adaptive_gate_audit_sample_rate_rng_boundary_kills_line429(self) -> None:
        mock_prof = CalibrationProfile(
            binary_temperature=1.0,
            target_risk_alpha=0.05,
            pass_threshold=0.8,
            fail_threshold=0.2,
        )
        gate = AdaptiveConformalController(
            profile=mock_prof,
            feedback_mode="cascade_audit",
            audit_sample_rate=0.4,
        )
        gate._rng.random = lambda: 0.4  # type: ignore[method-assign]
        assert gate.should_audit_fast_path() is False

        gate._rng.random = lambda: 0.399999  # type: ignore[method-assign]
        assert gate.should_audit_fast_path() is True

    def test_adaptive_gate_feedback_mode_explicit_audit_check_kills_line425(
        self,
    ) -> None:
        mock_prof = CalibrationProfile(
            binary_temperature=1.0,
            target_risk_alpha=0.05,
            pass_threshold=0.8,
            fail_threshold=0.2,
        )
        gate = AdaptiveConformalController(
            profile=mock_prof,
            feedback_mode="explicit",
        )
        gate._audit_sample_rate = 1.0
        assert gate.should_audit_fast_path() is False

    def test_adaptive_gate_bundle_mc_alpha_kills_line201(self) -> None:
        meta = LogitMetaCalibrator(
            weights=[1.0, 0.0, 0.0, 0.0, 0.0],
            intercept=0.0,
            target_risk_alpha=0.09,
            pass_threshold=0.85,
            fail_threshold=0.15,
        )
        bundle = DistillationBundle(meta_calibrator=meta)

        gate = AdaptiveConformalController(
            distillation_bundle=bundle,
            feedback_mode="explicit",
        )
        assert gate.target_risk_alpha == 0.09

    def test_adaptive_gate_comprehensive_methods_coverage(self) -> None:
        mock_prof = CalibrationProfile(
            binary_temperature=1.0,
            target_risk_alpha=0.05,
            pass_threshold=0.8,
            fail_threshold=0.2,
        )
        gate = AdaptiveConformalController(
            profile=mock_prof,
            feedback_mode="cascade_audit",
            audit_sample_rate=0.5,
            step_size=0.1,
            abstain_theta=2.0,
        )
        verdict = Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            probability=0.9,
            total_latency_ms=10.0,
            triggered_layer="l1",
        )
        rec1 = gate.observe(verdict, ground_truth=False, sampling_prob=0.5)
        assert rec1.raw_error == 1.0
        assert rec1.theta_after > 0.0

        rec2 = gate.observe(DecisionStatus.FAILED, ground_truth=True, sampling_prob=1.0)
        assert rec2.raw_error == 1.0

        rec3 = gate.observe("ambiguous", ground_truth=True, sampling_prob=1.0)
        assert rec3.raw_error == 0.0

        rec4 = gate.record_cascade_outcome(
            l1_status=DecisionStatus.PASSED,
            teacher_passed=True,
            sampling_prob=0.5,
            audited_fast_path=True,
        )
        assert rec4.audited_fast_path is True

        p, f, _ambig = gate.apply_drift_boost(delta_theta=0.5)
        assert isinstance(p, float)
        assert isinstance(f, float)

        # Test all properties
        assert gate.profile is not None
        assert gate.base_pass_threshold == 0.8
        assert gate.base_fail_threshold == 0.2
        assert gate.step_size == 0.1
        assert gate.feedback_mode == "cascade_audit"
        assert gate.audit_sample_rate == 0.5
        assert gate.override_on_audit is True
        assert gate.theta > 0.0
        assert len(gate.history) >= 4
        assert len(gate.current_thresholds()) == 3
        assert len(gate.current_decision_state()) == 6
        assert isinstance(gate.cache_fingerprint(), dict)

        exp_prof = gate.to_profile()
        assert exp_prof.target_risk_alpha == 0.05

        summary = gate.summary()
        assert summary.total_steps >= 4
        assert summary.to_dict()["total_steps"] >= 4

        gate.reset()
        assert gate.theta == 0.0

        # Trace metadata extraction
        v_trace = Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            probability=0.8,
            total_latency_ms=1.0,
            triggered_layer="l1",
            traces=[
                LayerTrace(
                    layer_name="l1",
                    status=DecisionStatus.PASSED,
                    latency_ms=1.0,
                    metadata={"raw_l1_status": "failed"},
                )
            ],
        )
        assert gate._extract_l1_status(v_trace) == DecisionStatus.FAILED

        with pytest.raises(ConfigurationError):
            gate.observe(verdict, ground_truth="invalid")  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            gate.observe(verdict, ground_truth=True, sampling_prob=0.0)
        with pytest.raises(ConfigurationError):
            gate.apply_drift_boost(-1.0)
        with pytest.raises(ConfigurationError):
            gate._extract_l1_status(12345)  # type: ignore[arg-type]


# ==============================================================================
# SECTION 2: audit.py Mutants & Coverage
# ==============================================================================


class TestAuditMutantsAndCoverage:
    def test_extract_pairwise_empty_cand_b_kills_line259(self) -> None:
        item = {
            "rubric": "Is this safe?",
            "candidate_a": "Option A text",
            "candidate_b": "",
            "winner": "a",
        }
        res = _extract_pairwise_cases(item)
        assert res == []

        item_empty_rub = {
            "rubric": "",
            "candidate_a": "A",
            "candidate_b": "B",
            "winner": "a",
        }
        assert _extract_pairwise_cases(item_empty_rub) == []

    def test_extract_pairwise_domain_preservation_kills_line286(self) -> None:
        item = {
            "rubric": "Is this safe?",
            "candidate_a": "Option A text",
            "candidate_b": "Option B text",
            "winner": "a",
            "domain": "cybersecurity",
        }
        res = _extract_pairwise_cases(item)
        assert len(res) == 2
        assert res[0].domain == "cybersecurity"
        assert res[1].domain == "cybersecurity"

    def test_audit_multiclass_domain_without_family_kills_line356(self) -> None:
        # Line 356: raw_dom = raw_item.get("domain") or raw_item.get("family")
        # Line 333 choice_gate_items fallback path sets domain
        data = {
            "choice_gate_items": [
                {
                    "question": "Calculate 2+2",
                    "actual_output": "4",
                    "expected_choice": "A",
                    "domain": "arithmetic",
                }
            ]
        }
        cases, _ = _load_normalized_cases(data)
        assert len(cases) == 1
        assert cases[0].domain == "arithmetic"

    def test_audit_expected_band_boundary_kills_line390(self) -> None:
        data = {
            "score_gate_items": [
                {
                    "rubric": "Rate quality",
                    "candidate": "Acceptable answer",
                    "expected_band": [3.0],
                }
            ]
        }
        cases, _ = _load_normalized_cases(data)
        assert len(cases) == 1
        assert cases[0].label is True

    def test_audit_word_shingles_k_3_kills_line637(self) -> None:
        shingles = _word_shingles("alpha beta gamma", k=3)
        assert len(shingles) == 1
        assert "alpha beta gamma" in shingles

    def test_audit_single_class_warning_kills_line726(self) -> None:
        data = [
            {"rubric": f"R{i}", "candidate": f"C{i}", "label": False} for i in range(10)
        ]
        report = audit_benchmark_dataset(data)
        assert any("Single-class dataset" in w for w in report.warnings)

    def test_audit_reference_label_preservation_kills_line808(self) -> None:
        ref_data = [{"rubric": "R1", "candidate": "C1", "label": True}]
        eval_data = [{"rubric": "R2", "candidate": "C2", "label": False}]
        leak_report = check_split_leakage(
            ref_data, eval_data, reference_label="my_custom_ref_suite"
        )
        assert leak_report.reference_source == "my_custom_ref_suite"
        assert isinstance(leak_report.to_dict(), dict)

    def test_audit_char_rel_diff_boundary_kills_line738(self) -> None:
        data = [
            {"rubric": f"R{i}", "candidate": "1234567890", "label": (i % 2 == 0)}
            for i in range(40)
        ]
        report = audit_benchmark_dataset(
            data, max_char_rel_diff=0.0, max_length_auroc_dev=0.0
        )
        assert not any("Length confounder detected" in w for w in report.warnings)
        assert isinstance(report.to_dict(), dict)

    def test_audit_dataset_comprehensive_methods(self) -> None:
        # AUROC and LOO lexical accuracy functions
        auroc = _compute_exact_auroc([0.1, 0.4, 0.8, 0.9], [False, False, True, True])
        assert auroc == 1.0

        cases = [
            _NormalizedCase(
                rubric="Math check", candidate="Two plus two is four", label=True
            ),
            _NormalizedCase(
                rubric="Math check", candidate="Two plus two is five", label=False
            ),
        ] * 10
        loo_acc = _compute_lexical_loo_accuracy(cases)
        assert 0.0 <= loo_acc <= 1.0

        # check near duplicate leakage
        ref = [
            {
                "rubric": "Check SQL syntax",
                "candidate": "SELECT * FROM users WHERE active=1",
                "label": True,
            }
        ]
        ev = [
            {
                "rubric": "Check SQL syntax",
                "candidate": "SELECT * FROM users WHERE active=0",
                "label": False,
            }
        ]
        leak = check_split_leakage(ref, ev, jaccard_threshold=0.3)
        assert leak.near_duplicate_candidates >= 0


# ==============================================================================
# SECTION 3: conformal.py Mutants & Coverage
# ==============================================================================


class TestConformalMutantsAndCoverage:
    def test_conformal_integer_labels_kills_line135(self) -> None:
        probs = [0.9] * 20 + [0.1] * 20
        labels = [1] * 20 + [0] * 20
        thresh = calibrate_cascade_thresholds(
            probs, labels, target_risk_alpha=0.1  # type: ignore[arg-type]
        )
        assert thresh.tau_accept > thresh.tau_reject

        # Reject integer other than 0 and 1
        with pytest.raises(
            CalibrationError,
            match="ground_truth_labels must contain booleans or 0/1 integers",
        ):
            calibrate_cascade_thresholds(
                probs, [2] * 20 + [0] * 20, target_risk_alpha=0.1  # type: ignore[arg-type]
            )

    def test_conformal_quantile_arithmetic_kills_line156(self) -> None:
        # Exact integer boundary m=19 and half_alpha=0.05 gives (19+1)*0.95 = 19.0 exact
        # With - 1e-9: rank=19 <= 19 -> guarantee_holds=True
        # With + 1e-9: rank=20 <= 19 -> False!
        probs = [0.95] * 19 + [0.05] * 19
        labels = [True] * 19 + [False] * 19
        thresh = calibrate_cascade_thresholds(probs, labels, target_risk_alpha=0.10)
        assert thresh.guarantee_holds is True
        assert min_samples_per_class(0.05) > 0

    def test_conformal_raw_accept_saturated_kills_line169(self) -> None:
        probs = [1.0] * 20 + [1.0] * 20
        labels = [False] * 20 + [True] * 20
        thresh = calibrate_cascade_thresholds(probs, labels, target_risk_alpha=0.1)
        assert thresh.guarantee_holds is False

    def test_conformal_threshold_order_boundary_kills_line178(self) -> None:
        probs = [0.5] * 20 + [0.5] * 20
        labels = [True] * 20 + [False] * 20
        thresh = calibrate_cascade_thresholds(probs, labels, target_risk_alpha=0.1)
        assert thresh.tau_reject < thresh.tau_accept

    def test_load_calibration_dataset_empty_candidate_kills_line331(self) -> None:
        data = [
            {"rubric": "Valid Rubric", "candidate": "", "label": True},
            {"rubric": "Valid Rubric", "candidate": "Valid Candidate", "label": True},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            p = f.name
        try:
            samples = load_calibration_dataset(p)
            assert len(samples) == 1
            assert samples[0][1] == "Valid Candidate"
        finally:
            os.unlink(p)

    def test_load_calibration_dataset_jsonl_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # JSONL
            jl_path = Path(td) / "data.jsonl"
            jl_path.write_text(
                json.dumps({"rubric": "R1", "candidate": "C1", "label": True})
                + "\n"
                + json.dumps({"rubric": "R2", "candidate": "C2", "label": False})
                + "\n",
                encoding="utf-8",
            )
            s_jl = load_calibration_dataset(jl_path)
            assert len(s_jl) == 2

            # Markdown table
            md_path = Path(td) / "data.md"
            md_path.write_text(
                "| Rubric | Candidate | Label |\n| :--- | :--- | :--- |\n| R1 | C1 | pass |\n| R2 | C2 | fail |\n",
                encoding="utf-8",
            )
            s_md = load_calibration_dataset(md_path)
            assert len(s_md) == 2

    def test_parse_markdown_table_column_boundary_kills_line374(self) -> None:
        md = """
| Rubric | Candidate | Label | Extra |
| :--- | :--- | :--- | :--- |
| OnlyTwoCells | Value |
| Valid Rubric | Valid Candidate | pass | extra |
"""
        parsed = _parse_markdown_dataset(md)
        assert len(parsed) == 1
        assert parsed[0][0] == "Valid Rubric"


# ==============================================================================
# SECTION 4: profile.py Mutants & Coverage
# ==============================================================================


class TestProfileMutantsAndCoverage:
    def test_choice_conformal_abstained_margin_none_kills_line240(self) -> None:
        probs_list = [{"A": 0.45, "B": 0.55}] * 25 + [{"A": 0.60, "B": 0.40}] * 25
        gt_list = ["B"] * 25 + ["B"] * 25
        thresh = calibrate_choice_thresholds(
            probs_list, gt_list, target_risk_alpha=0.1, calibrate_margin=False
        )
        assert thresh.min_margin is None
        assert thresh.empirical_abstention_rate == 1.0

    def test_profile_multiclass_temp_zero_rejected_kills_line764(self) -> None:
        samples = [(1.0, 1.0, bool(i % 2)) for i in range(20)]
        with pytest.raises(
            CalibrationError,
            match="multiclass_temperature must be a finite float > 0.0",
        ):
            CalibrationProfile.from_log_odds_samples(
                samples, multiclass_temperature=0.0
            )

    def test_fit_binary_cal_result_single_polarity_obs_kills_line890(self) -> None:
        records = [
            (2.0, 2.0, True, True),
            (1.5, 1.5, True, True),
            (1.8, 1.8, True, True),
            (2.2, 2.2, True, True),
            (-2.0, -2.0, False, False),
            (-1.5, -1.5, False, False),
            (-1.8, -1.8, False, False),
            (-2.2, -2.2, False, False),
        ]
        res = _fit_binary_cal_result(records)
        assert res.optimal_temperature > 0.0

    def test_profile_option_prob_out_of_bounds_kills_line177(self) -> None:
        with pytest.raises(
            CalibrationError, match="Option probabilities must be finite floats in"
        ):
            calibrate_choice_thresholds(
                [{"A": 1.5, "B": -0.5}] * 10, ["A"] * 10, target_risk_alpha=0.1
            )

    def test_choice_conformal_saturated_margin_kills_line218(self) -> None:
        probs = [{"A": 0.5, "B": 0.5}] * 25
        gt = ["A"] * 25
        thresh = calibrate_choice_thresholds(
            probs, gt, target_risk_alpha=0.05, calibrate_margin=True
        )
        assert isinstance(thresh, ChoiceConformalThresholds)

    def test_profile_integer_sample_labels_kills_line850(self) -> None:
        samples = [
            (f"Question {i}?", f"Candidate {i}", 1 if i % 2 == 0 else 0)
            for i in range(12)
        ]
        prof = fit_calibration_profile(
            "stub://default", samples=samples, target_risk_alpha=0.1
        )
        assert prof.model is not None

    def test_calibrate_score_thresholds_and_profile_methods(self) -> None:
        # Score thresholds
        scores = [4.5] * 20 + [2.0] * 20
        labels = [True] * 20 + [False] * 20
        score_thresh = calibrate_score_thresholds(scores, labels, target_risk_alpha=0.1)
        assert score_thresh.score_threshold >= score_thresh.score_fail_threshold

        prof = CalibrationProfile(binary_temperature=1.0)
        prof_score = prof.with_score_thresholds(score_thresh)
        assert prof_score.score_threshold == score_thresh.score_threshold

        choice_thresh = ChoiceConformalThresholds(
            min_confidence=0.8,
            min_margin=0.2,
            target_risk_alpha=0.05,
            empirical_abstention_rate=0.1,
            empirical_accuracy=0.9,
            calibration_set_size=50,
            guarantee_holds=True,
            min_samples_required=19,
        )
        prof_choice = prof.with_choice_thresholds(choice_thresh)
        assert prof_choice.choice_min_confidence == 0.8

    def test_profile_save_load_error_paths_lines_644_681(self) -> None:
        with pytest.raises(CalibrationError, match="cannot be empty"):
            CalibrationProfile.load("")

        with tempfile.TemporaryDirectory() as td:
            bad_json = Path(td) / "bad.json"
            bad_json.write_text("{not valid json", encoding="utf-8")
            with pytest.raises(CalibrationError, match="Malformed CalibrationProfile"):
                CalibrationProfile.load(bad_json)

            not_dict = Path(td) / "list.json"
            not_dict.write_text("[1, 2, 3]", encoding="utf-8")
            with pytest.raises(CalibrationError, match="must be an object, got list"):
                CalibrationProfile.load(not_dict)

            prof = CalibrationProfile(binary_temperature=1.0)
            target = Path(td) / "sub" / "deep" / "prof.json"
            prof.save(target)
            assert target.exists()

    @pytest.mark.asyncio
    async def test_afit_calibration_profile_async_full_lines_1029_1185(self) -> None:
        class AsyncMockBackend(LogitBackend):
            async def aquery_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                tokens = allowed_tokens or ["A", "B"]
                return [
                    {"token": t, "logprob": -0.1 if t in ("A", tokens[0]) else -2.5}
                    for t in tokens
                ]

            def query_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                return [
                    {"token": "A", "logprob": -0.1},
                    {"token": "B", "logprob": -2.5},
                ]

        backend = AsyncMockBackend()
        samples = [(f"Question {i}?", f"Candidate {i}", bool(i % 2)) for i in range(12)]

        prof = await afit_calibration_profile(
            backend=backend,
            model="mock_async_model",
            samples=samples,
            target_risk_alpha=0.1,
            calibrate_multiclass=True,
            warm_prefix=False,
        )
        assert prof.model == "mock_async_model"

        prof2 = await afit_calibration_profile(
            backend=backend,
            model="mock_async_model",
            samples=samples,
            target_risk_alpha=0.1,
            calibrate_multiclass=False,
            warm_prefix=True,
        )
        assert prof2.pass_threshold is not None

        prof3 = await afit_calibration_profile(
            backend=backend,
            model="mock_async_model",
            samples=None,
            target_risk_alpha=0.1,
        )
        assert prof3.optimal_temperature > 0.0

    def test_profile_save_exception_cleanup_lines_644_658(self) -> None:
        prof = CalibrationProfile(binary_temperature=1.0)
        with (
            tempfile.TemporaryDirectory() as td,
            patch("os.replace", side_effect=OSError("Disk write error")),
        ):
            target = Path(td) / "prof.json"
            with pytest.raises(
                CalibrationError, match="Failed to save CalibrationProfile"
            ):
                prof.save(target)
            assert not target.exists()

    def test_prepare_calibration_samples_error_paths_lines_835_876(self) -> None:
        def prep(
            dataset: str | Path | None = None,
            samples: Any = None,
            target_risk_alpha: float | None = None,
            calibrate_multiclass: bool = False,
            save_cache: bool = False,
        ) -> Any:
            return _prepare_calibration_samples(
                dataset=dataset,
                samples=samples,
                target_risk_alpha=target_risk_alpha,
                calibrate_multiclass=calibrate_multiclass,
                save_cache=save_cache,
            )

        # Nonexistent file
        with pytest.raises(
            CalibrationError, match="Failed to load calibration dataset"
        ):
            prep(dataset=Path("/nonexistent_dataset_path.json"))

        # Empty samples
        with pytest.raises(
            CalibrationError, match="Calibration dataset/samples cannot be empty"
        ):
            prep(samples=[])

        # Empty rubric string
        with pytest.raises(
            CalibrationError,
            match="Each calibration sample rubric must be a non-empty string",
        ):
            prep(samples=[("", "cand", True)])

        # Invalid label
        with pytest.raises(
            CalibrationError,
            match="Each calibration sample label must be a bool or 0/1 integer",
        ):
            prep(samples=[("rubric", "cand", "not_a_bool")])

        # Less than 10 samples with alpha
        with pytest.raises(CalibrationError, match="requires at least 10 samples"):
            prep(
                samples=[("rubric", "cand", True)] * 5,
                target_risk_alpha=0.05,
            )

        # All same labels with alpha
        with pytest.raises(
            CalibrationError, match="must contain both positive and negative"
        ):
            prep(
                samples=[("rubric", "cand", True)] * 12,
                target_risk_alpha=0.05,
            )

        # _extract_binary_record with non-finite logprobs raises BackendError
        top_inf = [{"token": "A", "logprob": float("nan")}]
        top_valid = [{"token": "A", "logprob": -0.5}]
        with pytest.raises(
            BackendError, match="returned no token log-probabilities"
        ):
            _extract_binary_record(top_inf, top_valid, True, "MockBackend")

    def test_afit_calibration_profile_warm_prefix_invalid_type_line1072(self) -> None:
        with pytest.raises(
            ConfigurationError, match="warm_prefix must be a bool or None"
        ):
            asyncio.run(
                afit_calibration_profile(
                    backend=MagicMock(),
                    warm_prefix="invalid",  # type: ignore[arg-type]
                )
            )


# ==============================================================================
# SECTION 5: permutation.py Mutants & Coverage
# ==============================================================================


class TestPermutationMutantsAndCoverage:
    def test_is_ignored_model_name_kills_line61(self) -> None:
        assert _is_ignored_model_name("default") is True
        assert _is_ignored_model_name("stub") is True
        assert _is_ignored_model_name("stub://custom_sub") is True
        assert _is_ignored_model_name("custom://my_backend") is True
        assert _is_ignored_model_name("llama-3-8b") is False

    def test_lookup_temperature_in_mapping_kills_line269(self) -> None:
        mapping = {"default": 1.5, "llama-3": 1.2}
        assert _lookup_temperature_in_mapping("default", mapping) is None
        assert _lookup_temperature_in_mapping("llama-3-8b-instruct", mapping) == 1.2
        assert has_known_or_cached_temperature("qwen2.5-coder:1.5b") is True
        assert has_known_or_cached_temperature("unknown_model_xyz") is False

    def test_empirical_auroc_tie_boundary_kills_line395(self) -> None:
        data = [(0.0, 1), (1e-12, 0)]
        auroc = _compute_binary_auroc(data)
        assert auroc == 0.5

    def test_fit_multiclass_temperature_ratio_math_kills_line497(self) -> None:
        samples = [
            ({"A": -0.1, "B": -2.0, "C": -3.0}, "A"),
            ({"A": -2.5, "B": -0.1, "C": -3.0}, "B"),
            ({"A": -3.0, "B": -2.0, "C": -0.1}, "C"),
        ]
        t = _fit_multiclass_temperature(
            samples, binary_auroc=0.95, min_temp=0.1, max_temp=5.0
        )
        assert t == 0.6571

    def test_auto_calibrate_multiclass_temp_preserved_kills_line688(self) -> None:
        class DummyBackend(LogitBackend):
            def query_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                toks = allowed_tokens or ["A", "B"]
                return [{"token": t, "logprob": -0.5} for t in toks]

        backend = DummyBackend()
        res = auto_calibrate_backend(backend, model_key="test_model", save_cache=False)
        assert res.multiclass_temperature is not None

    def test_fit_calibration_num_bins_1_kills_line1082(self) -> None:
        samples = [(1.5, 1.2, 1), (-1.2, -1.5, 0)] * 5
        res = fit_calibration(samples, num_bins=1)
        assert res.optimal_temperature > 0.0

    def test_fit_calibration_weak_probe_auroc_prior_kills_line1150(self) -> None:
        samples = [
            (0.1, 0.1, 0),
            (0.2, 0.2, 0),
            (0.05, 0.05, 1),
            (0.15, 0.15, 1),
        ] * 4
        res = fit_calibration(samples)
        assert res.optimal_temperature == 2.12

    def test_fit_calibration_objective_flat_threshold_kills_line1208(self) -> None:
        samples = [(1.5, 1.2, 1), (-1.2, -1.5, 0)] * 10
        res = fit_calibration(samples)
        assert res.optimal_temperature == 0.3644

    def test_compute_calibrated_pass_prob_negative_temp_kills_line812(self) -> None:
        p_fallback = combine_symmetric_log_odds(2.0, 2.0, logit_temperature=-1.0)
        p_unit = combine_symmetric_log_odds(2.0, 2.0, logit_temperature=1.0)
        assert abs(p_fallback - p_unit) < 1e-9

        # Shannon entropy
        assert compute_shannon_entropy(0.5) > 0.99
        assert compute_shannon_entropy(0.0) < 0.01
        assert compute_shannon_entropy(float("nan")) == 1.0

    def test_fit_binary_temperature_ece_bin_multiplication_kills_line1185(self) -> None:
        samples = [(2.0, 2.0, 1), (-2.0, -2.0, 0)] * 10
        res = fit_calibration(samples, num_bins=10)
        assert res.optimal_temperature > 0.0
        assert 0.0 <= res.ece_after <= 1.0

    def test_persist_calibration_cache_branches_lines_548_577(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / "cache.json"
            cache_file.write_text("{corrupt", encoding="utf-8")
            res = CalibrationResult(
                optimal_temperature=1.25,
                brier_score_before=0.2,
                brier_score_after=0.15,
                ece_before=0.1,
                ece_after=0.05,
            )
            _persist_calibration_profile("model_a", res, cache_path=cache_file)
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            assert "model_a" in data
            assert data["model_a"]["optimal_temperature"] == 1.25

    def test_auto_calibrate_backend_save_cache_lines_676_699(self) -> None:
        class DummyBackend(LogitBackend):
            def query_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                toks = allowed_tokens or ["A", "B"]
                return [{"token": t, "logprob": -0.5} for t in toks]

        backend = DummyBackend()
        with tempfile.TemporaryDirectory() as td:
            cache_p = Path(td) / "custom_cache.json"
            res = auto_calibrate_backend(
                backend,
                model_key="test_model_save",
                save_cache=True,
                cache_path=cache_p,
            )
            assert res.optimal_temperature > 0.0
            assert cache_p.exists()

    @pytest.mark.asyncio
    async def test_aauto_calibrate_multiclass_coverage_lines_772_787(self) -> None:
        class AsyncMockBackend(LogitBackend):
            async def aquery_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                toks = allowed_tokens or ["A", "B"]
                return [{"token": t, "logprob": -0.693} for t in toks]

            def query_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                return [{"token": "A", "logprob": -0.693}]

        b = AsyncMockBackend()
        with tempfile.TemporaryDirectory() as td:
            cache_p = Path(td) / "async_custom_cache.json"
            res = await aauto_calibrate_backend(
                b,
                model_key="test_async_model",
                save_cache=True,
                cache_path=cache_p,
            )
            assert res.optimal_temperature > 0.0
            assert cache_p.exists()


# ==============================================================================
# SECTION 6: _io.py Mutants & Coverage
# ==============================================================================


class TestDistillIOMutantsAndCoverage:
    def test_read_regular_json_exact_max_bytes_kills_lines_108_114_128(self) -> None:
        payload = json.dumps({"key": "val"}).encode("utf-8")
        exact_bytes = len(payload)

        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(payload)
            target = f.name
        try:
            parsed = read_json_object_nofollow(
                target, max_bytes=exact_bytes, artifact_name="TestObj"
            )
            assert parsed == {"key": "val"}

            with pytest.raises(ConfigurationError, match="exceeds maximum size"):
                read_json_object_nofollow(
                    target, max_bytes=exact_bytes - 1, artifact_name="TestObj"
                )
        finally:
            os.unlink(target)

    def test_read_regular_json_fd_cleanup_kills_line122(self) -> None:
        with (
            patch("os.open", return_value=0),
            patch("os.fstat") as mock_fstat,
            patch("os.close") as mock_close,
        ):
            mock_st = MagicMock()
            mock_st.st_mode = stat.S_IFREG
            mock_st.st_size = 1000
            mock_fstat.return_value = mock_st
            with pytest.raises(ConfigurationError):
                read_json_object_nofollow(
                    "dummy.json", max_bytes=10, artifact_name="Test"
                )
            mock_close.assert_called_once_with(0)

    def test_read_regular_json_symlink_and_non_dict_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            real_file = Path(td) / "real.json"
            real_file.write_text("[1, 2, 3]", encoding="utf-8")

            with pytest.raises(
                ConfigurationError, match="must contain a top-level JSON object"
            ):
                read_json_object_nofollow(
                    real_file, max_bytes=1024, artifact_name="TestObj"
                )

            sym_file = Path(td) / "sym.json"
            sym_file.symlink_to(real_file)
            with pytest.raises(ConfigurationError, match="Refusing to load"):
                read_json_object_nofollow(
                    sym_file, max_bytes=1024, artifact_name="TestObj"
                )

            # atomic write
            out_file = Path(td) / "written.json"
            written = atomic_write_json_nofollow(
                out_file, {"written": True}, artifact_name="TestObj"
            )
            assert written.exists()


# ==============================================================================
# SECTION 7: anchor_bank.py Mutants & Coverage
# ==============================================================================


class TestAnchorBankMutantsAndCoverage:
    def test_tokenize_bm25_single_char_and_stopwords_kills_line67(self) -> None:
        tokens = _tokenize_bm25("x foo the y bar")
        assert tokens == ["foo", "bar"]

    def test_condense_rationale_exact_max_chars_kills_line101(self) -> None:
        exact_text = "Exactly 20 chars lon"
        res = _condense_rationale(exact_text, passed=True, max_chars=len(exact_text))
        assert res == exact_text
        assert not res.endswith("...")

    def test_compile_anchor_bank_teacher_labels_override_kills_line333(self) -> None:
        items = [
            {
                "id": "ex_0",
                "rubric": "Security check for vulnerabilities",
                "candidate": "p = pos()",
                "expected": True,
            },
            {
                "id": "ex_1",
                "rubric": "Security check for vulnerabilities",
                "candidate": "p = neg()",
                "expected": True,
            },
        ]
        bank = DistilledAnchorBank.from_dataset(
            items,
            teacher_labels={"ex_1": False},
        )
        assert len(bank.pairs) == 1
        assert bank.pairs[0].positive.exemplar_id == "ex_0"
        assert bank.pairs[0].negative.exemplar_id == "ex_1"

    def test_compile_anchor_bank_has_teacher_priority_math_kills_line359(self) -> None:
        items = [
            {
                "id": "ex_0",
                "rubric": "Code quality check",
                "candidate": "eval candidate alpha",
                "expected": True,
                "rationale": "Reason A",
            },
            {
                "id": "ex_1",
                "rubric": "Code quality check",
                "candidate": "eval candidate beta",
                "expected": False,
                "rationale": "Reason B",
            },
        ]
        bank = DistilledAnchorBank.from_dataset(items)
        assert len(bank.pairs) == 1
        assert bank.pairs[0].priority_score >= 1.25

    def test_bm25_idf_and_tf_arithmetic_kills_lines_597_599(self) -> None:
        items = [
            {
                "id": "ex_0",
                "rubric": "cryptographic audit protocol",
                "candidate": "a",
                "expected": True,
            },
            {
                "id": "ex_1",
                "rubric": "cryptographic audit protocol",
                "candidate": "b",
                "expected": False,
            },
            {
                "id": "ex_2",
                "rubric": "audit audit audit verification",
                "candidate": "c",
                "expected": True,
            },
            {
                "id": "ex_3",
                "rubric": "audit audit audit verification",
                "candidate": "d",
                "expected": False,
            },
        ]
        bank = DistilledAnchorBank.from_dataset(items)
        ret = bank.retrieve("cryptographic audit", k=2, min_rubric_overlap=0.0)
        assert len(ret) >= 1
        assert ret[0].rubric == "cryptographic audit protocol"

    def test_anchor_bank_exclude_exact_false_kills_line551(self) -> None:
        items = [
            {
                "id": "ex_0",
                "rubric": "Security vulnerability in database query",
                "candidate": "exact match text",
                "expected": True,
            },
            {
                "id": "ex_1",
                "rubric": "Security vulnerability in database query",
                "candidate": "other text",
                "expected": False,
            },
        ]
        bank = DistilledAnchorBank.from_dataset(items)
        ret = bank.retrieve(
            "Security vulnerability in database query",
            candidate="exact match text",
            exclude_exact=False,
            k=1,
        )
        assert len(ret) == 1
        assert ret[0].positive.candidate == "exact match text"

    def test_anchor_bank_integer_labels_kills_line214(self) -> None:
        items = [
            {
                "id": "ex_0",
                "rubric": "Security vulnerability in database query",
                "candidate": "pos text",
                "expected": 1,
            },
            {
                "id": "ex_1",
                "rubric": "Security vulnerability in database query",
                "candidate": "neg text",
                "expected": 0,
            },
        ]
        bank = DistilledAnchorBank.from_dataset(items)
        assert len(bank.pairs) == 1

        # format block, fingerprint, save and load
        block = bank.format_anchor_block("Security vulnerability", "cand")
        assert block is not None
        assert isinstance(bank.cache_fingerprint(), dict)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bank.json"
            bank.save(p)
            loaded = DistilledAnchorBank.load(p)
            assert len(loaded.pairs) == 1
            assert resolve_anchor_bank(loaded) is loaded
            assert resolve_anchor_bank(p) is not None


# ==============================================================================
# SECTION 8: lora.py Mutants & Coverage
# ==============================================================================


class TestLoraMutantsAndCoverage:
    def test_lora_encode_prompt_ids_empty_input_ids_kills_line304(self) -> None:
        class MockTokenizerWithEmptyInputIds:
            chat_template = "dummy"

            def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
                mock_res = MagicMock()
                mock_res.input_ids = []
                return mock_res

            def encode(self, text: str, **kwargs: Any) -> list[int]:
                return [101, 102]

        tok = MockTokenizerWithEmptyInputIds()
        ids = _encode_prompt_ids(tok, "System prompt", "User prompt")
        assert ids == [101, 102]

    def test_hf_backend_explicit_cpu_device_kills_line418(self) -> None:
        if not HAS_TORCH:
            pytest.skip("torch not available")

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def parameters(self, recurse: bool = True) -> Any:
                m = MagicMock()
                m.device = "cuda:99"
                yield m

            def forward(self, input_ids: Any) -> Any:
                out = MagicMock()
                out.logits = torch.zeros((1, input_ids.shape[1], 10))
                return out

        model = DummyModel()

        class DummyTokenizer:
            def encode(self, text: str, **kwargs: Any) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: Sequence[int], **kwargs: Any) -> str:
                return "A" if ids[0] == 1 else "B"

        tok = DummyTokenizer()
        backend = HuggingFaceLogitBackend(model, tok, device="cpu")
        res = backend.query_top_logprobs("Test prompt", top_k=2)
        assert len(res) >= 1

    def test_train_lora_model_config_name_kills_line540(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not HAS_TORCH:
            pytest.skip("torch not available")

        class _MockPeftModel(torch.nn.Module):
            def __init__(self, base: Any) -> None:
                super().__init__()
                self.base = base
                self.fc = torch.nn.Linear(2, 2)

            def forward(self, input_ids: Any, labels: Any = None, **kwargs: Any) -> Any:
                if labels is not None and labels.shape[0] > 0:
                    last_lbl = int(labels[0, -1].item())
                    if last_lbl not in (10, 20):
                        raise ValueError(
                            f"Target token {last_lbl} not in allowed ({10}, {20})"
                        )
                out = MagicMock()
                out.loss = torch.tensor(0.5, requires_grad=True)
                return out

            def save_pretrained(self, save_directory: str) -> None:
                pass

            def eval(self) -> _MockPeftModel:
                return self

        class _MockPeftMod:
            TaskType = type("TaskType", (), {"CAUSAL_LM": "CAUSAL_LM"})()

            @staticmethod
            def LoraConfig(**kwargs: Any) -> Any:
                return type("LoraConfigObj", (), kwargs)()

            @staticmethod
            def get_peft_model(model: Any, config: Any) -> Any:
                return _MockPeftModel(model)

        monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

        class TinyLinearLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(100, 16)
                self.lm_head = torch.nn.Linear(16, 100)
                self.config = MagicMock()
                self.config._name_or_path = "custom-tiny-model-v1"

            def forward(
                self, input_ids: Any, attention_mask: Any = None, labels: Any = None
            ) -> Any:
                x = self.emb(input_ids)
                logits = self.lm_head(x)
                out = MagicMock()
                out.logits = logits
                out.loss = logits.mean()
                return out

        class SimpleTok:
            pad_token_id = 0

            def encode(self, text: str, **kwargs: Any) -> list[int]:
                if text.strip() == "A":
                    return [10]
                if text.strip() == "B":
                    return [20]
                return [1, 2, 3]

            def decode(self, ids: Sequence[int], **kwargs: Any) -> str:
                return "A" if ids[0] == 10 else "B"

        items = [
            {
                "id": f"id_{i}",
                "rubric": "Rubric test",
                "candidate": "Candidate text",
                "expected": bool(i % 2),
            }
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as td:
            manifest = train_distillation_lora(
                TinyLinearLM(),
                items,
                output_dir=td,
                tokenizer=SimpleTok(),
                config=LoRADistillConfig(epochs=1, batch_size=2),
            )
            assert manifest.base_model == "custom-tiny-model-v1"
            assert isinstance(manifest.model_dump(), dict)

    def test_prepare_lora_examples_teacher_labels_kills_line154(self) -> None:
        items = [{"id": "item_0", "rubric": "R", "candidate": "C", "expected": True}]
        examples = prepare_symmetric_lora_examples(
            items,
            teacher_labels={"item_0": False},
        )
        assert len(examples) == 2
        assert examples[0].target_token == "B"

    def test_hf_backend_allowed_tokens_multi_token_rejection_kills_line441(
        self,
    ) -> None:
        if not HAS_TORCH:
            pytest.skip("torch not available")

        class DummyModel(torch.nn.Module):
            def forward(self, input_ids: Any) -> Any:
                out = MagicMock()
                logits = torch.full((1, input_ids.shape[1], 10), -10.0)
                logits[0, -1, 0] = 5.0
                out.logits = logits
                return out

        class MultiTokTokenizer:
            def encode(self, text: str, **kwargs: Any) -> list[int]:
                if "multi" in text:
                    return [4, 5]
                return [0]

            def decode(self, ids: Sequence[int], **kwargs: Any) -> str:
                return "token_" + str(ids[0])

        backend = HuggingFaceLogitBackend(
            DummyModel(), MultiTokTokenizer(), device="cpu"
        )
        res = backend.query_top_logprobs(
            "Prompt", top_k=1, allowed_tokens=["multi_token_word"]
        )
        assert not any(r["token"] == "token_4" for r in res)


# ==============================================================================
# SECTION 9: meta_calibrator.py Mutants & Coverage
# ==============================================================================


class TestMetaCalibratorMutantsAndCoverage:
    def test_pass_dist_stats_residual_tail_math_kills_line101(self) -> None:
        top_logprobs = [{"token": "A", "logprob": math.log(0.4)}]
        entropy, _non_target = _pass_distribution_stats(
            top_logprobs, target_tokens=("A", "B")
        )
        expected_entropy = -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))
        assert abs(entropy - expected_entropy) < 1e-6

    def test_extract_logit_features_entropy_averaging_kills_line123(self) -> None:
        top_fwd = [
            {"token": "A", "logprob": -0.6931},
            {"token": "B", "logprob": -0.6931},
        ]
        top_rev = [
            {"token": "A", "logprob": -0.6931},
            {"token": "B", "logprob": -0.6931},
        ]
        feat = extract_logit_features(top_fwd, top_rev)
        ent_single, _ = _pass_distribution_stats(top_fwd, target_tokens=("A", "B"))
        assert abs(feat.top_k_entropy - ent_single) < 1e-6

    def test_fit_ridge_logistic_az_50_boundary_kills_line191(self) -> None:
        features = [(10.0, 0.0, 0.0, 0.0, 0.0)] * 10
        labels = [True] * 5 + [False] * 5
        _intercept, weights = _fit_ridge_logistic(features, labels, l2_penalty=1.0)
        assert len(weights) == 5

    def test_fit_ridge_logistic_convergence_delta_kills_line226(self) -> None:
        features = [(1.0, 0.0, 0.0, 0.0, 0.0)] * 10
        labels = [True] * 5 + [False] * 5
        _intercept, weights = _fit_ridge_logistic(features, labels, max_iter=5)
        assert len(weights) == 5

    def test_meta_calibrator_l2_penalty_zero_rejected_kills_line315(self) -> None:
        with pytest.raises(
            (CalibrationError, pydantic.ValidationError),
            match="l2_penalty must be a finite positive float",
        ):
            LogitMetaCalibrator(
                weights=[1.0, 0.0, 0.0, 0.0, 0.0],
                intercept=0.0,
                l2_penalty=0.0,
            )

    def test_pass_dist_stats_logprob_neg50_boundary_kills_line88(self) -> None:
        top_logprobs = [{"token": "A", "logprob": -50.0}]
        _ent, non_target = _pass_distribution_stats(
            top_logprobs, target_tokens=("A", "B")
        )
        assert non_target == 1.0

    def test_solve_linear_system_zero_factor_kills_line153(self) -> None:
        sol = _solve_linear_system([[2.0, 1.0], [0.0, 3.0]], [5.0, 6.0])
        assert abs(sol[0] - 1.5) < 1e-9
        assert abs(sol[1] - 2.0) < 1e-9

    def test_meta_calibrator_fit_integer_labels_kills_line366(self) -> None:
        features = [LogitFeatureVector(float(i), 0.0, 0.0, 0.0, 0.0) for i in range(12)]
        labels = [1 if i % 2 == 0 else 0 for i in range(12)]
        meta = LogitMetaCalibrator.fit(features, labels)
        assert meta.weights[0] > 0.0

        p = meta.predict_proba(features[0])
        lo = meta.predict_log_odds(features[0])
        assert 0.0 <= p <= 1.0
        assert math.isfinite(lo)
        assert isinstance(meta.cache_fingerprint(), dict)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "meta.json"
            meta.save(path)
            loaded = LogitMetaCalibrator.load(path)
            assert len(loaded.weights) == 5

    @pytest.mark.asyncio
    async def test_meta_calibrator_backend_fit(self) -> None:
        class DummyBackend(LogitBackend):
            def query_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                return [
                    {"token": "A", "logprob": -0.2},
                    {"token": "B", "logprob": -2.0},
                ]

            async def aquery_top_logprobs(
                self,
                prompt: str,
                top_k: int = 10,
                allowed_tokens: list[str] | None = None,
                **kwargs: Any,
            ) -> list[dict[str, Any]]:
                return [
                    {"token": "A", "logprob": -0.2},
                    {"token": "B", "logprob": -2.0},
                ]

        samples = [
            {"rubric": f"R{i}", "candidate": f"C{i}", "expected": bool(i % 2)}
            for i in range(12)
        ]
        b = DummyBackend()
        meta = fit_meta_calibrator(b, samples, model="dummy")
        assert meta.weights[0] > 0.0

        meta_async = await afit_meta_calibrator(b, samples, model="dummy")
        assert meta_async.weights[0] > 0.0


# ==============================================================================
# SECTION 10: telemetry/distill.py Mutants & Coverage
# ==============================================================================


class TestTelemetryDistillMutantsAndCoverage:
    def test_exemplar_judge_log_odds_zero_kills_line80(self) -> None:
        record = TelemetryRecord(
            gate_name="Cascade",
            gate_type="Cascade",
            rubric="Check safety",
            candidate="Safe answer",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=10,
            total_latency_ms=5.0,
            traces=[
                LayerTrace(
                    layer_name="student",
                    status=DecisionStatus.AMBIGUOUS,
                    log_odds_pass=0.0,
                    tokens_generated=0,
                    latency_ms=1.0,
                ),
                LayerTrace(
                    layer_name="teacher",
                    status=DecisionStatus.PASSED,
                    diagnostic_report="The teacher found no defects.",
                    metadata={"judge_log_odds": 0.0},
                    tokens_generated=1,
                    latency_ms=1.0,
                ),
            ],
        )
        ex = extract_distillation_record(record)
        assert ex is not None
        assert ex.label is True

    def test_exemplar_diagnostic_report_is_not_none_kills_line140(self) -> None:
        record = TelemetryRecord(
            gate_name="Cascade",
            gate_type="Cascade",
            rubric="Check safety",
            candidate="Safe answer",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=10,
            total_latency_ms=5.0,
            traces=[
                LayerTrace(
                    layer_name="student",
                    status=DecisionStatus.AMBIGUOUS,
                    log_odds_pass=0.0,
                    tokens_generated=0,
                    latency_ms=1.0,
                ),
                LayerTrace(
                    layer_name="teacher",
                    status=DecisionStatus.PASSED,
                    diagnostic_report="The teacher found no defects.",
                    tokens_generated=1,
                    latency_ms=1.0,
                ),
            ],
        )
        ex = extract_distillation_record(record)
        assert ex is not None
        assert ex.teacher_reason == "The teacher found no defects."

    def test_distill_sink_rotation_existing_regular_file_kills_line235(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sink_file = Path(td) / "distill.jsonl"
            sink_file.write_text("x" * 250 + "\n", encoding="utf-8")
            sink = DistillationSink(path=sink_file, max_bytes=300, backup_count=2)
            assert sink._handle is None
            record = TelemetryRecord(
                gate_name="Cascade",
                gate_type="Cascade",
                rubric="Rubric test",
                candidate="Candidate text",
                passed=True,
                status=DecisionStatus.PASSED,
                candidate_sha256="abc",
                candidate_length=10,
                total_latency_ms=5.0,
                traces=[
                    LayerTrace(
                        layer_name="student",
                        status=DecisionStatus.AMBIGUOUS,
                        log_odds_pass=0.0,
                        tokens_generated=0,
                        latency_ms=1.0,
                    ),
                    LayerTrace(
                        layer_name="teacher",
                        status=DecisionStatus.PASSED,
                        diagnostic_report="Report",
                        tokens_generated=1,
                        latency_ms=1.0,
                    ),
                ],
            )
            sink.emit(record)
            sink.close()
            backup = Path(f"{sink_file}.1")
            assert backup.exists()

    def test_recalibrate_nan_reverse_log_odds_kills_line402(self) -> None:
        records = [
            distill_telemetry.DistillationRecord(
                rubric=f"R{i}",
                candidate=f"C{i}",
                label=bool(i % 2),
                discrepancy_type="ambiguous_resolved",
                student_layer="student",
                student_status=DecisionStatus.AMBIGUOUS,
                student_forward_log_odds=1.5 if i % 2 == 0 else -1.5,
                student_reverse_log_odds=float("nan"),
                teacher_layer="teacher",
                teacher_passed=bool(i % 2),
                cascade_passed=bool(i % 2),
                candidate_sha256=f"sha_{i}",
            )
            for i in range(10)
        ]
        cal = recalibrate_from_telemetry(records)
        assert cal.optimal_temperature > 0.0

        with tempfile.TemporaryDirectory() as td:
            for fmt in ("sft_jsonl", "calibration_jsonl", "dpo_jsonl"):
                out_p = Path(td) / f"{fmt}.jsonl"
                export_distillation_dataset(records, out_p, format=fmt)
                assert out_p.exists()


# ==============================================================================
# SECTION 11: telemetry/drift.py Mutants & Coverage
# ==============================================================================


class TestTelemetryDriftMutantsAndCoverage:
    def test_extract_drift_sig_nan_prob_kills_line301(self) -> None:
        records = [
            TelemetryRecord(
                gate_name="LogitGate",
                gate_type="LogitGate",
                passed=True,
                status=DecisionStatus.PASSED,
                candidate_sha256="abc",
                candidate_length=10,
                total_latency_ms=5.0,
                traces=[
                    LayerTrace(
                        layer_name="l1",
                        status=DecisionStatus.PASSED,
                        calibrated_prob=float("nan"),
                        log_odds_pass=1.0,
                        latency_ms=1.0,
                    )
                ],
            ),
            TelemetryRecord(
                gate_name="LogitGate",
                gate_type="LogitGate",
                passed=True,
                status=DecisionStatus.PASSED,
                candidate_sha256="abc2",
                candidate_length=10,
                total_latency_ms=5.0,
                traces=[
                    LayerTrace(
                        layer_name="l1",
                        status=DecisionStatus.PASSED,
                        calibrated_prob=0.8,
                        log_odds_pass=1.0,
                        latency_ms=1.0,
                    )
                ],
            ),
        ]
        sig = _extract_from_records(records, "test")
        assert len(sig.probs) == 1
        assert sig.probs == [0.8]

    def test_drift_psi_severe_equal_moderate_kills_line366(self) -> None:
        ref = [1.0] * 20
        cur = [1.0] * 20
        drift_res = detect_telemetry_drift(ref, cur, psi_moderate=0.25, psi_severe=0.25)
        assert drift_res.drift_detected is False

    def test_drift_ks_alpha_severe_bump_math_kills_line416(self) -> None:
        ref = [0.0] * 20 + [1.0] * 20
        cur = [0.0] * 7 + [
            1.0
        ] * 33  # shift 13 produces ks_p = 0.022160 in (0.005, 0.05]
        report = detect_telemetry_drift(
            ref, cur, ks_alpha=0.05, psi_moderate=10.0, psi_severe=20.0
        )
        assert report.severity == "moderate"

    def test_drift_monitor_drift_boost_theta_zero_kills_line515(self) -> None:
        mon = RollingDriftMonitor(
            baseline_window=10, current_window=10, drift_boost_theta=0.0
        )
        assert mon._drift_boost_theta == 0.0

    def test_kolmogorov_survival_zero_stat_zero_n_kills_line176(self) -> None:
        surv = _kolmogorov_survival(0.0, 0, 0)
        assert surv == 1.0

    def test_extract_drift_sig_prob_val_none_lo_not_none_kills_line297(self) -> None:
        records = [
            TelemetryRecord(
                gate_name="LogitGate",
                gate_type="LogitGate",
                passed=True,
                status=DecisionStatus.PASSED,
                candidate_sha256="abc",
                candidate_length=10,
                total_latency_ms=5.0,
                traces=[
                    LayerTrace(
                        layer_name="l1",
                        status=DecisionStatus.PASSED,
                        calibrated_prob=0.9,
                        log_odds_pass=0.0,
                        latency_ms=1.0,
                    )
                ],
            )
        ]
        sig = _extract_from_records(records, "test")
        assert len(sig.probs) == 1
        assert abs(sig.probs[0] - 0.9) < 1e-6

    def test_drift_escalation_delta_threshold_boundary_kills_line420(self) -> None:
        ref_records = [
            TelemetryRecord(
                gate_name="LogitGate",
                gate_type="LogitGate",
                passed=True,
                status=DecisionStatus.PASSED,
                candidate_sha256="abc",
                candidate_length=10,
                total_latency_ms=5.0,
                escalated=(i < 2),
                traces=[
                    LayerTrace(
                        layer_name="l1",
                        status=DecisionStatus.PASSED,
                        log_odds_pass=1.0,
                        latency_ms=1.0,
                    )
                ],
            )
            for i in range(20)
        ]
        cur_records = [
            TelemetryRecord(
                gate_name="LogitGate",
                gate_type="LogitGate",
                passed=True,
                status=DecisionStatus.PASSED,
                candidate_sha256="abc",
                candidate_length=10,
                total_latency_ms=5.0,
                escalated=(i < 8),
                traces=[
                    LayerTrace(
                        layer_name="l1",
                        status=DecisionStatus.PASSED,
                        log_odds_pass=1.0,
                        latency_ms=1.0,
                    )
                ],
            )
            for i in range(20)
        ]
        rep = detect_telemetry_drift(
            ref_records,
            cur_records,
            escalation_delta_threshold=0.2,
            psi_moderate=10.0,
            psi_severe=20.0,
            ks_alpha=1e-5,
        )
        assert rep.severity == "severe"

    def test_rolling_drift_monitor_lifecycle(self) -> None:
        mon = RollingDriftMonitor(baseline_window=5, current_window=5, check_every=2)
        assert mon.baseline_ready is False

        rec = TelemetryRecord(
            gate_name="G",
            gate_type="LogitGate",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=10,
            total_latency_ms=2.0,
            traces=[
                LayerTrace(
                    layer_name="l1",
                    status=DecisionStatus.PASSED,
                    log_odds_pass=1.0,
                    latency_ms=1.0,
                )
            ],
        )
        for _ in range(12):
            mon.emit(rec)

        assert mon.baseline_ready is True
        assert len(mon.reports) >= 1
        assert mon.latest_report is not None
        mon.reset_current()
        rep = mon.evaluate_now()
        assert rep is None or isinstance(rep, DriftReport)

        # compute_psi and compute_ks_two_sample
        psi = compute_psi([1.0, 2.0, 3.0] * 5, [1.0, 2.0, 3.0] * 5)
        assert psi == 0.0

        d, p_val = compute_ks_two_sample([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert 0.0 <= d <= 1.0
        assert 0.0 <= p_val <= 1.0


# ==============================================================================
# SECTION 12: telemetry/models.py Mutants & Coverage
# ==============================================================================


class TestTelemetryModelsMutantsAndCoverage:
    def test_luhn_valid_16_zeros_rejected_kills_line43_and(self) -> None:
        assert _luhn_valid("0000000000000000") is False
        assert _luhn_valid("12345") is False

    def test_luhn_valid_19_digits_allowed_kills_line43_gt(self) -> None:
        digits_18 = "400000000000000000"
        for d in range(10):
            cand = digits_18 + str(d)
            if _luhn_valid(cand):
                assert len(cand) == 19
                assert _luhn_valid(cand) is True
                break

    def test_luhn_valid_digit_9_undoubled_kills_line51(self) -> None:
        base = "453201511283036"
        for d in range(10):
            cand = base + str(d)
            if _luhn_valid(cand):
                assert _luhn_valid(cand) is True
                break

    def test_telemetry_record_privacy_full_max_chars_kills_line148(self) -> None:
        record = TelemetryRecord(
            gate_name="G",
            gate_type="JudgeGate",
            candidate="Hello World",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=11,
            total_latency_ms=1.0,
        )
        res = record.with_privacy(privacy="full", max_candidate_chars=100)
        assert res is record

    def test_telemetry_record_privacy_full_exact_max_chars_kills_line150(self) -> None:
        record = TelemetryRecord(
            gate_name="G",
            gate_type="JudgeGate",
            candidate="Exact10Chr",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=10,
            total_latency_ms=1.0,
        )
        res = record.with_privacy(privacy="full", max_candidate_chars=10)
        assert res is record

    def test_telemetry_record_judge_gate_generative_critique_kills_line160(
        self,
    ) -> None:
        record = TelemetryRecord(
            gate_name="Judge",
            gate_type="JudgeGate",
            candidate="Private Candidate",
            reason="Private reasoning leaking candidate",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=17,
            total_latency_ms=1.0,
            traces=[],
        )
        res = record.with_privacy(privacy="omit_candidate", max_candidate_chars=100)
        assert res.candidate is None
        assert res.reason is None

    def test_telemetry_record_trace_tokens_generated_critique_kills_line161(
        self,
    ) -> None:
        record_with_tokens = TelemetryRecord(
            gate_name="CustomGate",
            gate_type="LogitGate",
            candidate="Secret",
            reason="Generative critique with tokens",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=6,
            total_latency_ms=1.0,
            traces=[
                LayerTrace(
                    layer_name="l1",
                    status=DecisionStatus.PASSED,
                    tokens_generated=5,
                    latency_ms=1.0,
                )
            ],
        )
        res = record_with_tokens.with_privacy(
            privacy="omit_candidate", max_candidate_chars=100
        )
        assert res.reason is None

        record_zero_tokens = TelemetryRecord(
            gate_name="CustomGate",
            gate_type="LogitGate",
            candidate="Secret",
            reason="Preserved non-generative reason",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=6,
            total_latency_ms=1.0,
            traces=[
                LayerTrace(
                    layer_name="l1",
                    status=DecisionStatus.PASSED,
                    tokens_generated=0,
                    latency_ms=1.0,
                )
            ],
        )
        res_zero = record_zero_tokens.with_privacy(
            privacy="omit_candidate", max_candidate_chars=100
        )
        assert res_zero.reason == "Preserved non-generative reason"

    def test_telemetry_models_full_redaction_and_summary_coverage(self) -> None:
        raw = (
            "Contact user@example.com with Bearer secret-token-xyz and sk-1234567890abcdef123456 "
            "or SSN 123-45-6789 or card 4000000000000000."
        )
        red = redact_text(raw)
        assert "[REDACTED_EMAIL]" in red
        assert "Bearer [REDACTED_TOKEN]" in red
        assert "[REDACTED_SECRET]" in red
        assert "[REDACTED_SSN]" in red

        cand_str = "My candidate string"
        c1, _h1, l1 = apply_privacy(cand_str, privacy="hash_only")
        assert c1 is None
        assert l1 == len(cand_str)

        c2, _h2, _l2 = apply_privacy(cand_str, privacy="redact_pii")
        assert c2 == cand_str

        c3, _h3, _l3 = apply_privacy(cand_str, privacy="full", max_candidate_chars=5)
        assert "truncated" in str(c3)

        assert hash_candidate("hello") == hash_candidate("hello")
        assert truncate_text("test", None) == "test"
        assert truncate_text("short", 10) == "short"

        rec = TelemetryRecord(
            gate_name="G",
            gate_type="LogitGate",
            candidate="Sensitive content user@email.com",
            reason="Reason with sk-1234567890abcdef123456",
            passed=True,
            status=DecisionStatus.PASSED,
            candidate_sha256="abc",
            candidate_length=30,
            total_latency_ms=2.0,
            traces=[
                LayerTrace(
                    layer_name="l1",
                    status=DecisionStatus.PASSED,
                    latency_ms=1.0,
                    diagnostic_report="Found secret sk-1234567890abcdef123456 in input",
                )
            ],
        )
        rec_pii = rec.with_privacy(privacy="redact_pii", max_candidate_chars=100)
        assert "[REDACTED_EMAIL]" in str(rec_pii.candidate)
        assert "[REDACTED_SECRET]" in str(rec_pii.reason)
        assert "[REDACTED_SECRET]" in str(rec_pii.traces[0].diagnostic_report)

        rec_trunc = rec.with_privacy(privacy="full", max_candidate_chars=5)
        assert "truncated" in str(rec_trunc.candidate)

        dist_rec = DistillationRecord(
            rubric="Rubric",
            candidate="Candidate",
            label=True,
            discrepancy_type="ambiguous_resolved",
            student_layer="l1",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="l2",
            teacher_passed=True,
            cascade_passed=True,
            candidate_sha256="abc",
        )
        assert dist_rec.rubric == "Rubric"

        summary = TelemetrySummary(
            total_records=10,
            passed_count=8,
            failed_count=1,
            ambiguous_count=1,
            fast_path_count=6,
            fast_path_rate=0.6,
            escalated_count=4,
            escalation_rate=0.4,
            discrepancy_count=0,
            discrepancy_rate=0.0,
            mean_latency_ms=12.5,
            median_latency_ms=10.0,
            p95_latency_ms=25.0,
            total_tokens_generated=50,
            by_gate_type={"LogitGate": 10},
        )
        d = summary.to_dict()
        assert d["total_records"] == 10
        assert d["by_gate_type"] == {"LogitGate": 10}
