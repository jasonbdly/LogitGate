from __future__ import annotations

import math
import tempfile
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

# Explicitly import all 11 target modules at module level for mutation_audit.py test-mapping discovery.
import logitgate.calibration.profile
import logitgate.debiasing.inversion
import logitgate.debiasing.permutation
import logitgate.debiasing.softmax
import logitgate.distill.anchor_bank
import logitgate.distill.bundle
import logitgate.distill.lora
import logitgate.distill.meta_calibrator
import logitgate.gates.ast_gate
import logitgate.gates.judge_gate
import logitgate.gates.rank_gate
from logitgate.calibration.profile import CalibrationProfile, _fit_binary_cal_result
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import fit_calibration
from logitgate.debiasing.softmax import clamped_sigmoid, softmax_probabilities
from logitgate.distill.anchor_bank import (
    AnchorExemplar,
    ContrastiveAnchorPair,
    DistilledAnchorBank,
)
from logitgate.distill.bundle import compile_distillation_bundle
from logitgate.distill.lora import LoRADistillConfig, train_distillation_lora
from logitgate.distill.meta_calibrator import _fit_ridge_logistic, _solve_linear_system
from logitgate.exceptions import CalibrationError
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.rank_gate import BestOfNGate, RankGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.telemetry.distill import DistillationRecord


# =============================================================================
# 1. logitgate.debiasing.inversion (Lines 221, 339, 345)
# =============================================================================
def test_inversion_line221_out_of_bounds_rubric_index() -> None:
    """ROR >= -> > at line 221: rubric_index == len(rubrics) must raise out of bounds IndexError."""
    with pytest.raises(IndexError, match=r"rubric_index 2 out of bounds for rubrics list of size 2"):
        compile_batch_binary_prompt(
            rubrics=["rubric 0", "rubric 1"],
            candidate="candidate code",
            rubric_index=2,
        )


def test_inversion_lines339_345_integer_scoring_thresholds() -> None:
    """ROR >= -> > at line 339 (q >= 0.99) and <= -> < at line 345 (q <= 0.35)."""
    # min_score=0, max_score=100 gives span=100.
    # tok_val=99 -> q = 99/100 = 0.99. q >= 0.99 is True -> Highest quality.
    # tok_val=35 -> q = 35/100 = 0.35. q <= 0.35 is True -> Poor quality.
    prompt = compile_score_prompt(
        rubric="valid rubric",
        candidate="candidate code",
        min_score=0,
        max_score=100,
        inverted=False,
    )
    assert "99: Highest quality (exemplary; satisfies every rubric criterion with no defects)" in prompt
    assert "35: Poor quality (notable defects, omissions, or reliability issues)" in prompt


# =============================================================================
# 2. logitgate.debiasing.softmax (Lines 12, 35, 47)
# =============================================================================
def test_softmax_line12_clamped_sigmoid_upper_boundary() -> None:
    """ROR > -> >= at line 12: at x == 50.0, x > 50.0 is False so math.exp is evaluated."""
    with patch("logitgate.debiasing.softmax.math.exp", return_value=0.5):
        val = clamped_sigmoid(50.0)
        # 1.0 / (1.0 + 0.5) = 2/3
        assert abs(val - (2.0 / 3.0)) < 1e-9


def test_softmax_line35_pos_inf_keys_strict_gt_zero() -> None:
    """ROR > -> >= at line 35: math.isinf(v) and v > 0 excludes v == 0.0."""
    real_isinf = math.isinf
    with patch(
        "logitgate.debiasing.softmax.math.isinf",
        side_effect=lambda v: True if v == 0.0 else real_isinf(v),
    ):
        res = softmax_probabilities({"a": 0.0, "b": -1.0})
        # If line 35 mutant triggers, pos_inf_keys contains 'a' and returns {'a': 1.0, 'b': 0.0}.
        # In original, pos_inf_keys is empty and softmax over finite values runs.
        assert "a" in res and res["a"] < 0.9
        assert abs(res["a"] - 0.73105857863) < 1e-4


def test_softmax_line47_total_zero_or_non_finite_fallback() -> None:
    """LCR or -> and at line 47: total <= 0.0 triggers uniform fallback even if math.isfinite(total)."""
    with patch("logitgate.debiasing.softmax.math.exp", return_value=0.0):
        res = softmax_probabilities({"a": 1.0, "b": 2.0})
        # Total is 0.0. Original enters if because not (total > 0.0) is True.
        # Mutant with and requires not math.isfinite(0.0) which is False, causing ZeroDivisionError.
        assert res == {"a": 0.5, "b": 0.5}


# =============================================================================
# 3. logitgate.debiasing.permutation (Line 1208)
# =============================================================================
def test_permutation_line1208_flat_objective_best_temp_stability() -> None:
    """AOR - -> + at line 1208: obj_val < best_obj - 1e-12 prevents updating best_temp on flat objective."""
    with (
        patch("logitgate.debiasing.permutation.math.log", return_value=0.0),
        patch("logitgate.debiasing.permutation.combine_symmetric_log_odds", return_value=0.5),
    ):
        samples = [(2.0, 2.0, 1), (-2.0, -2.0, 0)] * 5
        res = fit_calibration(samples, min_temp=0.1, max_temp=10.0)
        # Original keeps best_step as None because obj_val is flat, refining around 1.0 (~1.0233).
        # Mutant with + 1e-12 updates best_step at every step and marches to max_temp=10.0.
        assert abs(res.optimal_temperature - 1.0233) < 0.05


# =============================================================================
# 4. logitgate.distill.bundle (Lines 174, 215, 246)
# =============================================================================
def _make_dummy_records(n: int) -> list[DistillationRecord]:
    return [
        DistillationRecord(
            event_id=f"ev_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric=f"rubric {i % 3}",
            candidate=f"candidate {i}",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_model="student-3b",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_model="teacher-26b",
            teacher_passed=(i % 2 == 0),
            teacher_reason="Safe" if i % 2 == 0 else "Unsafe",
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"hash_{i}",
        )
        for i in range(n)
    ]


def test_bundle_line174_fit_meta_boundary_at_10_records() -> None:
    """ROR >= -> > at line 174: len(recs) == 10 with fit_meta=True must fit meta_calibrator."""
    recs = _make_dummy_records(10)
    bundle = compile_distillation_bundle(
        telemetry=recs,
        backend="stub://pass",
        fit_meta=True,
    )
    assert bundle.meta_calibrator is not None


def test_bundle_line215_effective_student_model_preserved_when_meta_cal_none() -> None:
    """LCR or -> and at line 215: eff_student = student_model or (...) must preserve student_model."""
    recs = _make_dummy_records(5)
    bundle = compile_distillation_bundle(
        telemetry=recs,
        student_model="student-3b",
        fit_meta=False,
        include_anchors=True,
    )
    assert bundle.student_model == "student-3b"


def test_bundle_line246_zero_rubric_overlap_skips_unanchored_meta() -> None:
    """ROR > -> >= at line 246: min_rubric_overlap == 0.0 must NOT fit unanchored_meta_calibrator."""
    dataset = [
        {"id": f"{i}", "rubric": "test rubric", "candidate": f"cand {i}", "expected": (i % 2 == 0)}
        for i in range(10)
    ]
    bundle = compile_distillation_bundle(
        dataset=dataset,
        backend="stub://pass",
        min_rubric_overlap=0.0,
        fit_meta=True,
    )
    assert bundle.unanchored_meta_calibrator is None


# =============================================================================
# 5. logitgate.distill.anchor_bank (Lines 597, 599)
# =============================================================================
def _make_exemplar(ex_id: str, rubric: str, passed: bool = True) -> AnchorExemplar:
    return AnchorExemplar(
        exemplar_id=ex_id,
        rubric=rubric,
        candidate="candidate text",
        passed=passed,
    )


def test_anchor_bank_line597_bm25_idf_numerator_rare_vs_common() -> None:
    """AOR - -> + at line 597: n_docs - n_q + 0.5 must reward rare words over high-frequency common words."""
    # Rare term in 1 doc with filler words (dl=11)
    p_rare = ContrastiveAnchorPair(
        pair_id="rare_pair",
        rubric="rare zebra topic filler alpha beta gamma delta epsilon zeta eta",
        positive=_make_exemplar("p1", "rare zebra topic", True),
        negative=_make_exemplar("n1", "rare zebra topic", False),
        domain="d_rare",
        priority_score=0.0,
    )
    # Common term in 15 docs with 4 occurrences in rubric (dl=4)
    p_common = ContrastiveAnchorPair(
        pair_id="common_pair",
        rubric="common common common common",
        positive=_make_exemplar("p2", "common", True),
        negative=_make_exemplar("n2", "common", False),
        domain="d_common",
        priority_score=0.0,
    )
    bg = [
        ContrastiveAnchorPair(
            pair_id=f"bg_{i}",
            rubric=f"common background rubric item {i}",
            positive=_make_exemplar(f"pbg_{i}", f"common background {i}", True),
            negative=_make_exemplar(f"nbg_{i}", f"common background {i}", False),
            domain=f"bg_{i}",
            priority_score=0.0,
        )
        for i in range(14)
    ]
    bank = DistilledAnchorBank(pairs=[p_rare, p_common, *bg])
    retrieved = bank.retrieve("rare common", k=1, min_rubric_overlap=0.0)
    # Original IDF numerator (16 - 1 + 0.5) gives rare BM25 ~1.76 vs common BM25 ~0.31 -> rare wins.
    # Mutant with + (16 + 14 + 0.5) inflates common IDF so common BM25 ~2.21 > rare BM25 ~1.84 -> common wins.
    assert retrieved[0].pair_id == "rare_pair"


def test_anchor_bank_line599_bm25_saturation_multiplier_vs_priority() -> None:
    """AOR + -> - at line 599: k1 + 1.0 (2.5) vs k1 - 1.0 (0.5) BM25 scaling against priority tiebreak."""
    p_bm25 = ContrastiveAnchorPair(
        pair_id="bm25_winner",
        rubric="target topic evaluation",
        positive=_make_exemplar("p1", "target topic", True),
        negative=_make_exemplar("n1", "target topic", False),
        domain="d1",
        priority_score=0.0,
    )
    p_prio = ContrastiveAnchorPair(
        pair_id="prio_winner",
        rubric="unrelated topic assessment",
        positive=_make_exemplar("p2", "unrelated", True),
        negative=_make_exemplar("n2", "unrelated", False),
        domain="d2",
        priority_score=10.0,
    )
    bank = DistilledAnchorBank(pairs=[p_bm25, p_prio])
    retrieved = bank.retrieve("target", k=1, min_rubric_overlap=0.0)
    # In original: p_bm25 has BM25 ~1.0 > p_prio tie-break (0.36) -> bm25_winner.
    # In mutant: k1 - 1.0 divides BM25 by 5 (0.20) < 0.36 -> prio_winner wins.
    assert retrieved[0].pair_id == "bm25_winner"


# =============================================================================
# 6. logitgate.distill.meta_calibrator (Lines 191, 226)
# =============================================================================
def test_meta_calibrator_line191_abs_z_saturation_boundary_50() -> None:
    """ROR < -> <= at line 191: az < 50.0 is False when az == 50.0, skipping math.log1p."""
    real_log1p = math.log1p
    mock_log1p = MagicMock(side_effect=real_log1p)
    with patch("logitgate.distill.meta_calibrator.math.log1p", mock_log1p):
        _fit_ridge_logistic([(50.0, 0.0, 0.0, 0.0, 0.0)], [True], max_iter=1)
        # Original: _loss(theta) has az=50.0 (skipped). Then _loss(cand) has az < 50.0 (1 call).
        # Mutant with az <= 50.0 calls log1p during _loss(theta) as well (2 calls).
        assert mock_log1p.call_count == 1


def test_meta_calibrator_line226_convergence_step_subtraction_vs_addition() -> None:
    """AOR - -> + at line 226: delta = max(abs(cand - theta)) enables early stopping when delta < 1e-7."""
    features = [(1.0, 0.1, 0.2, 0.3, 0.4), (-1.0, 0.5, 0.1, 0.2, 0.3)] * 10
    labels = [True, False] * 10
    mock_solve = MagicMock(side_effect=_solve_linear_system)
    with patch("logitgate.distill.meta_calibrator._solve_linear_system", mock_solve):
        _fit_ridge_logistic(features, labels, max_iter=40)
        # Original converges in ~4 iterations and exits early at line 230.
        # Mutant computes delta = abs(cand + theta) >= 0.1, never reaching 1e-7, running all 40 iterations.
        assert mock_solve.call_count < 10


# =============================================================================
# 7. logitgate.distill.lora (Line 571)
# =============================================================================
def test_lora_line571_token_id_mapping_target_a_vs_b() -> None:
    """ROR == -> != at line 571: target_id = id_a if ex.target_token == 'A' else id_b."""
    mock_torch = MagicMock()
    mock_torch.randperm.side_effect = lambda n, **kw: MagicMock(tolist=lambda: list(range(n)))
    mock_torch.tensor.side_effect = lambda x, **kw: x
    mock_transformers = MagicMock()
    mock_peft = MagicMock()

    mock_peft_model = MagicMock()
    param = MagicMock()
    param.requires_grad = True
    mock_peft_model.parameters.return_value = [param]
    mock_peft.get_peft_model.return_value = mock_peft_model

    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.eos_token_id = 1

    dataset = [{"id": "1", "rubric": "rubric", "candidate": "cand", "expected": True}]
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("logitgate.distill.lora._require_hf_deps", return_value=(mock_torch, mock_transformers, mock_peft)),
        patch("logitgate.distill.lora._resolve_torch_device", return_value="cpu"),
        patch("logitgate.distill.lora._resolve_torch_dtype", return_value="float32"),
        patch(
            "logitgate.distill.lora._resolve_single_token_id",
            side_effect=lambda tok, s: 101 if s == "A" else 102,
        ),
        patch("logitgate.distill.lora._encode_prompt_ids", return_value=[1, 2, 3]),
    ):
        cfg = LoRADistillConfig(epochs=1, batch_size=1)
        train_distillation_lora(
            base_model=MagicMock(),
            dataset=dataset,
            output_dir=tmpdir,
            tokenizer=mock_tok,
            config=cfg,
        )

    # First symmetric pass maps positive to 'A' (target token id 101).
    # Mutant with ex.target_token != 'A' assigns id 102 to pass 1 and 101 to pass 2.
    first_call_labels = mock_peft_model.call_args_list[0].kwargs.get("labels")
    assert first_call_labels is not None
    assert first_call_labels[0][-1] == 101


# =============================================================================
# 8. logitgate.calibration.profile (Lines 764, 890)
# =============================================================================
def test_profile_line764_multiclass_temperature_zero_rejection() -> None:
    """ROR <= -> < at line 764: multiclass_temperature <= 0.0 must raise CalibrationError directly."""
    samples = [(1.0, 1.0, True), (-1.0, -1.0, False)] * 5
    with pytest.raises(CalibrationError, match=r"^multiclass_temperature must be a finite float > 0\.0, got 0\.0$"):
        CalibrationProfile.from_log_odds_samples(samples, multiclass_temperature=0.0)


def test_profile_line890_min_observed_samples_conjunction() -> None:
    """LCR and -> or at line 890: len(obs_lo) >= 4 and len(classes) >= 2 must both hold to use obs_lo."""
    # 2 observed samples (1 pos, 1 neg), 8 unobserved samples with large log-odds (10.0)
    records: Sequence[tuple[float, float, bool, bool]] = [
        (1.0, 1.0, True, True),
        (-1.0, -1.0, False, True),
    ] + [(10.0, 10.0, True, False)] * 4 + [(-10.0, -10.0, False, False)] * 4

    res = _fit_binary_cal_result(records)
    # In original: len(obs_lo)=2 < 4, so condition is False and all_lo is used (median = 10.0).
    # In mutant with or: len(classes)=2 >= 2 is True, so obs_lo is used (median = 1.0).
    assert res.median_abs_log_odds == 10.0


# =============================================================================
# 9. logitgate.gates.rank_gate (Lines 586, 912, 1353)
# =============================================================================
def test_rank_gate_line586_cycle_residual_guard_zero_division() -> None:
    """LCR and -> or at line 586: n >= 3 and pair_diffs prevents division by zero when pair_diffs is empty."""
    rg = RankGate(rubric="test rubric", backend=MagicMock())
    entries: list[tuple[str, str]] = [("a", "cand a"), ("b", "cand b"), ("c", "cand c")]
    # Empty pairs with n=3 candidates: original sets cycle_res = 0.0.
    # Mutant with or enters computation and divides by len(pair_diffs)==0, raising ZeroDivisionError.
    v = rg._compute_round_robin_verdict(entries, pairs=[], top_list=[], t0=0.0, call_kwargs={})
    assert v.traces[-1].metadata.get("cycle_residual") == 0.0


@pytest.mark.asyncio
async def test_rank_gate_line912_aevaluate_mode_listwise_branch() -> None:
    """ROR == -> != at line 912: if eff_mode == 'listwise' executes listwise ranking branch."""
    gate = RankGate(rubric="test rubric", backend="stub://pass", mode="listwise")
    verdict = await gate.aevaluate(["cand1", "cand2", "cand3"], mode="listwise")
    # Original executes listwise branch with effective_mode='listwise'.
    # Mutant with != skips listwise branch and executes tournament branch with effective_mode='tournament'.
    assert verdict.traces[-1].metadata.get("effective_mode") == "listwise"


def test_rank_gate_line1353_post_rank_trace_carry_key_error() -> None:
    """LCR and -> or at line 1353: k in base_meta and k not in metadata prevents KeyError on missing keys."""
    base_trace = LayerTrace(layer_name="base", status=DecisionStatus.PASSED, latency_ms=1.0, metadata={})
    base_v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=1.0,
        triggered_layer="base",
        total_latency_ms=1.0,
        traces=[base_trace],
    )
    post_v = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=1.0,
        triggered_layer="post",
        total_latency_ms=1.0,
        traces=[base_trace],
    )
    gate = BestOfNGate(rubric="rubric", backend=MagicMock())
    # In original: 'surviving_keys' in base_meta is False -> skipped safely.
    # In mutant with or: 'surviving_keys' not in combined_traces[-1].metadata is True -> KeyError: 'surviving_keys'.
    v = gate._merge_post_verdict([("a", "cand")], "a", 0, base_v, post_v, {}, 0.0)
    assert v.passed is True


# =============================================================================
# 10. logitgate.gates.judge_gate (Lines 253, 646)
# =============================================================================
def test_judge_gate_line253_keyword_signal_strict_tiebreak_lt() -> None:
    """ROR < -> <= at line 253: pass_match.start() < fail_match.start() tiebreak favors FAIL on equal start."""
    m_pass = MagicMock()
    m_pass.start.return_value = 5
    m_fail = MagicMock()
    m_fail.start.return_value = 5

    mock_pass_re = MagicMock()
    mock_pass_re.search.return_value = m_pass
    mock_fail_re = MagicMock()
    mock_fail_re.search.return_value = m_fail

    with (
        patch.object(logitgate.gates.judge_gate, "_PASS_WORD_RE", mock_pass_re),
        patch.object(logitgate.gates.judge_gate, "_FAIL_WORD_RE", mock_fail_re),
    ):
        res = logitgate.gates.judge_gate._keyword_signal("ambiguous text", tie_break=True)
        # Original: 5 < 5 is False -> returns _FAIL (False, DecisionStatus.FAILED, 0.0).
        # Mutant with <=: 5 <= 5 is True -> returns _PASS (True, DecisionStatus.PASSED, 1.0).
        assert res == (False, DecisionStatus.FAILED, 0.0)


def test_judge_gate_line646_whitespace_cleaned_disjunction_no_probe() -> None:
    """LCR or -> and at line 646: not cleaned or not cleaned.strip() returns _FAIL without calling probe."""
    mock_probe = MagicMock()
    mock_probe.evaluate.return_value = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.99,
        triggered_layer="probe",
        total_latency_ms=1.0,
    )
    jg = JudgeGate(rubric="test rubric", backend=MagicMock())
    with (
        patch("logitgate.gates.judge_gate.strip_reasoning_blocks", return_value="   "),
        patch.object(jg, "_probe", return_value=mock_probe),
    ):
        res = jg._parse_verdict("raw critique")
        # Original: not cleaned (False) or not cleaned.strip() (True) -> returns _FAIL immediately.
        # Mutant with and: False and True is False -> skips to probe evaluation and returns PASSED.
        assert res == (False, DecisionStatus.FAILED, 0.0)
        assert not mock_probe.evaluate.called


# =============================================================================
# 11. logitgate.gates.ast_gate (Line 522)
# =============================================================================
def test_ast_gate_line522_normalize_input_json_schema_is_none() -> None:
    """ROR is -> is not at line 522: self.json_schema is None keeps fenced markdown verbatim."""
    gate = ASTGate(required_sections=["Summary"], normalize=True)
    candidate = "```markdown\n# Summary\nHere is text.\n```"
    res = gate.normalize_input(candidate)
    # Original: json_schema is None is True -> returns strip_reasoning_blocks(candidate).strip()
    # keeping the code fences.
    # Mutant with is not None is False -> falls through to normalize_candidate which strips the code fence.
    assert res == "```markdown\n# Summary\nHere is text.\n```"
