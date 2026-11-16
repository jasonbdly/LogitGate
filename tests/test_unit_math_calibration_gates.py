from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

# Ensure coverage tracks single-file modules under --source=src/logitgate/cascade.py etc.
try:
    import coverage  # type: ignore[import-not-found]

    _cov = coverage.Coverage.current()
    if _cov and hasattr(_cov, "_inorout"):
        for _mod in ("cascade", "snapshot", "benchmark"):
            _spec_name = f"src/logitgate/{_mod}.py"
            _mod_name = f"logitgate.{_mod}"
            if _mod_name not in _cov._inorout.source_pkgs:
                _cov._inorout.source_pkgs.append(_mod_name)
            if _cov._inorout.source_pkgs_match and _mod_name not in _cov._inorout.source_pkgs_match.modules:
                _cov._inorout.source_pkgs_match.modules.append(_mod_name)
            if _spec_name in _cov._inorout.source_pkgs_unmatched:
                _cov._inorout.source_pkgs_unmatched.remove(_spec_name)
except (ImportError, AttributeError, ValueError, KeyError):
    _cov = None

from logitgate.backends.base import GenerativeBackend, StubBackend
from logitgate.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    _percentile,
    bootstrap_metric_ci,
    compute_calibration_metrics,
    format_report,
    load_cases,
    mcnemar_exact_test,
    run_benchmark,
    wilson_confidence_interval,
    write_json,
)
from logitgate.calibration.adaptive import (
    AdaptiveConformalController,
    AdaptiveConformalSummary,
    AdaptiveStepRecord,
)
from logitgate.calibration.audit import (
    DatasetAuditReport,
    SplitLeakageReport,
    _extract_case_from_mapping,
    _extract_cases_from_mapping_root,
    _extract_pairwise_cases,
    _load_normalized_cases,
    _parse_bool_label,
    audit_benchmark_dataset,
    check_split_leakage,
)
from logitgate.calibration.conformal import (
    calibrate_cascade_thresholds,
    load_calibration_dataset,
    min_samples_per_class,
)
from logitgate.calibration.profile import (
    CalibrationProfile,
    afit_calibration_profile,
    fit_calibration_profile,
)
from logitgate.cascade import (
    Cascade,
    _has_active_cascade_audit,
    _same_nonzero_sign,
    _select_upstream_logit_trace,
    _should_escalate,
    fuse_escalated_verdict,
)
from logitgate.debiasing.inversion import (
    _validate_binary_tokens,
    compile_batch_binary_prompt,
    compile_binary_prompt,
    compile_choice_prompt,
    compile_cot_binary_prompt,
    compile_listwise_rank_prompt,
    compile_pairwise_prompt,
    compile_score_prompt,
    render_candidate,
)
from logitgate.debiasing.permutation import (
    CalibrationResult,
    DebiasedLogitResult,
    SurfaceTokenMap,
    _compute_binary_auroc,
    _debiased_multiclass_accum,
    _extract_temp_from_entry,
    _fit_multiclass_temperature,
    _load_cached_mapping,
    _lookup_temperature_in_mapping,
    _median,
    _persist_calibration_profile,
    aauto_calibrate_backend,
    auto_calibrate_backend,
    compute_symmetric_debiased_verdict,
    fit_calibration,
    get_calibration_cache_path,
    get_model_temperature,
    has_known_or_cached_temperature,
    marginalize_pole_logprob,
    reset_runtime_calibration_cache,
)
from logitgate.debiasing.softmax import (
    _compute_missing_token_floor,
    clamped_sigmoid,
    compute_normalized_probabilities,
    extract_token_logprob,
    has_any_target_token,
    has_finite_logprobs,
    logsumexp,
    softmax_probabilities,
)
from logitgate.exceptions import (
    BackendError,
    CalibrationError,
    CalibrationWarning,
    ConfigurationError,
)
from logitgate.gates.ast_gate import ASTFailureCode, ASTGate
from logitgate.gates.judge_gate import (
    JudgeGate,
    _extract_keyword_verdict,
    _extract_structured_verdict,
    _extract_verification_notes,
    _iter_json_objects,
    _keyword_signal,
)
from logitgate.gates.logit_gate import (
    ChoiceGate,
    LogitGate,
    MultiRubricGate,
    ScoreGate,
)
from logitgate.gates.rank_gate import BestOfNGate, RankGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import (
    VerdictSnapshot,
)

# ---------------------------------------------------------------------------
# Section 1: Debiasing Math & Softmax Unit Tests
# ---------------------------------------------------------------------------


def test_clamped_sigmoid_boundaries_and_nan() -> None:
    assert clamped_sigmoid(float("nan")) == 0.5
    assert clamped_sigmoid(51.0) == 1.0
    assert clamped_sigmoid(100.0) == 1.0
    assert clamped_sigmoid(-51.0) == 0.0
    assert clamped_sigmoid(-100.0) == 0.0
    assert math.isclose(clamped_sigmoid(0.0), 0.5, abs_tol=1e-7)
    assert 0.0 < clamped_sigmoid(2.0) < 1.0


def test_softmax_probabilities_edge_cases() -> None:
    assert softmax_probabilities({}) == {}

    with pytest.raises(ValueError, match="temperature must be a finite positive float"):
        softmax_probabilities({"A": 0.0}, temperature=True)

    with pytest.raises(ValueError, match="temperature must be a finite positive float"):
        softmax_probabilities({"A": 0.0}, temperature=-1.0)

    with pytest.raises(ValueError, match="temperature must be a finite positive float"):
        softmax_probabilities({"A": 0.0}, temperature=float("nan"))

    with pytest.raises(ValueError, match="temperature must be a finite positive float"):
        softmax_probabilities({"A": 0.0}, temperature=float("inf"))

    with pytest.raises(TypeError, match="must be numeric, got bool"):
        softmax_probabilities({"A": True})

    res_nan = softmax_probabilities({"A": float("nan"), "B": -1.0})
    assert res_nan == {"A": 0.5, "B": 0.5}

    res_inf = softmax_probabilities({"A": float("inf"), "B": -1.0})
    assert res_inf == {"A": 1.0, "B": 0.0}

    res_two_inf = softmax_probabilities({"A": float("inf"), "B": float("inf"), "C": 0.0})
    assert res_two_inf == {"A": 0.5, "B": 0.5, "C": 0.0}

    res_all_neg_inf = softmax_probabilities({"A": float("-inf"), "B": float("-inf")})
    assert res_all_neg_inf == {"A": 0.5, "B": 0.5}


def test_has_finite_logprobs_and_target_tokens() -> None:
    assert not has_finite_logprobs(None)
    assert not has_finite_logprobs([])
    assert not has_finite_logprobs(["not a dict"])  # type: ignore[list-item]
    assert not has_finite_logprobs([{"token": None, "logprob": -0.5}])
    assert not has_finite_logprobs([{"token": "A", "logprob": None}])
    assert not has_finite_logprobs([{"token": "A", "logprob": True}])
    assert not has_finite_logprobs([{"token": "A", "logprob": "invalid_float"}])
    assert not has_finite_logprobs([{"token": "A", "logprob": float("nan")}])
    assert has_finite_logprobs([{"token": "A", "logprob": -0.2}])

    assert not has_any_target_token([], ["A"])
    assert not has_any_target_token([{"token": "A", "logprob": -0.1}], [])
    assert not has_any_target_token([{"token": "A", "logprob": -0.1}], ["  ", 123])  # type: ignore[list-item]
    assert not has_any_target_token(["invalid"], ["A"])  # type: ignore[list-item]
    assert not has_any_target_token([{"token": None, "logprob": -0.1}], ["A"])
    assert not has_any_target_token([{"token": "B", "logprob": -0.1}], ["A"])
    assert not has_any_target_token([{"token": "A", "logprob": None}], ["A"])
    assert not has_any_target_token([{"token": "A", "logprob": True}], ["A"])
    assert not has_any_target_token([{"token": "A", "logprob": "not_float"}], ["A"])
    assert not has_any_target_token([{"token": "A", "logprob": float("inf")}], ["A"])
    assert has_any_target_token([{"token": " a ", "logprob": -0.5}], ["A"])


def test_logsumexp_and_missing_token_floor() -> None:
    assert logsumexp([]) == -10.0
    assert logsumexp([True, False]) == -10.0
    assert logsumexp([float("inf")]) == float("inf")
    assert logsumexp([float("-inf"), float("-inf")]) == -10.0
    assert math.isclose(logsumexp([0.0, 0.0]), math.log(2.0), abs_tol=1e-7)

    assert _compute_missing_token_floor([]) == -10.0
    floor_val = _compute_missing_token_floor([-0.1, -2.0])
    assert math.isfinite(floor_val)
    assert floor_val <= -2.0


def test_extract_token_logprob_and_normalized_probs() -> None:
    assert extract_token_logprob([], "A") == -10.0

    raw_items: list[Any] = [
        "not-a-dict",
        {"token": None, "logprob": -0.1},
        {"token": "A", "logprob": None},
        {"token": "A", "logprob": True},
        {"token": "A", "logprob": "bad"},
        {"token": "A", "logprob": float("nan")},
        {"token": "A", "logprob": 0.5},  # clamped to 0.0
        {"token": "A", "logprob": -0.2},  # duplicate lower than 0.0, ignored
        {"token": " A", "logprob": -0.6},
        {"token": "B", "logprob": -1.2},
    ]
    lp_a = extract_token_logprob(raw_items, "A")
    assert lp_a <= 0.0

    lp_missing = extract_token_logprob([{"token": "B", "logprob": -0.1}], "Z")
    assert lp_missing <= -0.1

    norm_probs = compute_normalized_probabilities(
        [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.3}],
        tokens=["A", "B"],
        temperature=1.0,
    )
    assert "A" in norm_probs and "B" in norm_probs
    assert math.isclose(sum(norm_probs.values()), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Section 1B: Debiasing Inversion & Prompt Compiler Unit Tests
# ---------------------------------------------------------------------------


class _SamplePydanticCandidate(BaseModel):
    title: str = "Test Candidate"
    score: int = 100


@dataclass
class _SampleDataclassCandidate:
    name: str = "DC Candidate"
    value: float = 3.14


def test_render_candidate_formats() -> None:
    assert render_candidate("  hello world  ") == "hello world"

    pyd_cand = _SamplePydanticCandidate()
    rendered_pyd = render_candidate(pyd_cand)
    assert "Test Candidate" in rendered_pyd

    assert render_candidate(_SamplePydanticCandidate) == str(_SamplePydanticCandidate)

    dc_cand = _SampleDataclassCandidate()
    rendered_dc = render_candidate(dc_cand)
    assert "DC Candidate" in rendered_dc

    list_cand = ["alpha", 42]
    assert render_candidate(list_cand) == '[\n  "alpha",\n  42\n]'

    class _UnserializableKeyDict(dict[Any, Any]):
        pass

    bad_dict = _UnserializableKeyDict()
    bad_dict[object()] = "unserializable"
    rendered_fallback = render_candidate(bad_dict)
    assert isinstance(rendered_fallback, str)


def test_inversion_prompt_compilation_branches() -> None:
    with pytest.raises(ValueError, match="must be distinct non-empty"):
        _validate_binary_tokens("", "B")
    with pytest.raises(ValueError, match="must be distinct non-empty"):
        _validate_binary_tokens("A", "a")

    p_fwd = compile_binary_prompt(
        "Check code", "def foo(): pass", query_first=False, boundary_nonce="nonce-1", anchor_block="Sample anchor"
    )
    assert "[CALIBRATION ANCHORS]" in p_fwd
    assert "<candidate_nonce-1>" in p_fwd

    p_cot_short = compile_cot_binary_prompt("Rubric", "Cand", verification_notes="short")
    assert "[VERIFICATION NOTES]" not in p_cot_short

    p_cot_long = compile_cot_binary_prompt(
        "Rubric",
        "Cand",
        verification_notes="Detailed analysis shows compliance with all constraints.",
        query_first=False,
        boundary_nonce="nonce_99",
    )
    assert "[VERIFICATION NOTES]" in p_cot_long
    assert "<notes_nonce_99>" in p_cot_long

    with pytest.raises(IndexError, match="rubric_index"):
        compile_batch_binary_prompt(["R1"], "Cand", rubric_index=5)

    p_batch_qf = compile_batch_binary_prompt(
        ["R1", "R2"], "Cand", rubric_index=0, query_first=True, anchor_block="Anchors", boundary_nonce="b1"
    )
    assert "[EVALUATION RUBRIC]" in p_batch_qf
    assert "[CALIBRATION ANCHORS]" in p_batch_qf

    p_choice = compile_choice_prompt("Which?", "Cand", {"A": "First", "B": "Second"}, query_first=False)
    assert "[QUESTION]" in p_choice

    with pytest.raises(ValueError, match="Expected integers with min_score < max_score"):
        compile_score_prompt("Rubric", "Cand", min_score=True)
    with pytest.raises(ValueError, match="Expected integers with min_score < max_score"):
        compile_score_prompt("Rubric", "Cand", min_score=5, max_score=1)

    p_score_inv = compile_score_prompt(
        "Rubric", "Cand", min_score=1, max_score=5, query_first=False, inverted=True, boundary_nonce="s1"
    )
    assert "Rate the candidate artifact on an inverted scale" in p_score_inv
    assert "Lowest quality" in p_score_inv
    assert "Highest quality" in p_score_inv
    assert "Good quality" in p_score_inv
    assert "Poor quality" in p_score_inv
    assert "Moderate quality" in p_score_inv

    p_pair = compile_pairwise_prompt("Rubric", "Cand A", "Cand B", context="Context", query_first=False, boundary_nonce="n1")
    assert "[TASK CONTEXT]" in p_pair
    assert "<context_n1>" in p_pair

    with pytest.raises(ValueError, match="mapping with at least 2 slot entries"):
        compile_listwise_rank_prompt("Rubric", {"A": "Only one"})

    with pytest.raises(ValueError, match="Slot keys must be single characters"):
        compile_listwise_rank_prompt("Rubric", {"AA": "C1", "B": "C2"})

    with pytest.raises(ValueError, match="Slot keys must be case-insensitively unique"):
        compile_listwise_rank_prompt("Rubric", {"a": "C1", "A": "C2"})

    p_list = compile_listwise_rank_prompt("Rubric", {"A": "C1", "B": "C2"}, query_first=False, boundary_nonce="ln1")
    assert "[CANDIDATE A]" in p_list
    assert "[CANDIDATE B]" in p_list


# ---------------------------------------------------------------------------
# Section 1C: Debiasing Permutation Math & Calibration Unit Tests
# ---------------------------------------------------------------------------


def test_calibration_cache_path_and_helpers(tmp_path: Path) -> None:
    reset_runtime_calibration_cache()
    old_env = os.environ.get("LOGITGATE_CACHE_DIR")
    try:
        os.environ["LOGITGATE_CACHE_DIR"] = str(tmp_path)
        assert get_calibration_cache_path() == tmp_path / "calibration.json"
    finally:
        if old_env is not None:
            os.environ["LOGITGATE_CACHE_DIR"] = old_env
        else:
            os.environ.pop("LOGITGATE_CACHE_DIR", None)

    cache_file = tmp_path / "custom_cal.json"
    assert _load_cached_mapping(cache_file) is None

    cache_file.write_text(json.dumps({"test-model": {"optimal_temperature": 1.45}}), encoding="utf-8")
    loaded = _load_cached_mapping(cache_file)
    assert loaded is not None and "test-model" in loaded

    assert _load_cached_mapping(cache_file) == loaded

    cache_file.write_text("not json", encoding="utf-8")
    reset_runtime_calibration_cache()
    assert _load_cached_mapping(cache_file) is None


def test_temperature_entry_extraction_and_lookup() -> None:
    assert _extract_temp_from_entry(True) is None
    assert _extract_temp_from_entry(float("nan")) is None
    assert _extract_temp_from_entry(0.0) is None
    assert _extract_temp_from_entry(2.0, binary=False) == 2.0
    assert _extract_temp_from_entry(0.2, binary=False) == 0.65

    entry_dict = {"optimal_temperature": 1.25, "binary_temperature": 1.1, "multiclass_temperature": 1.3}
    assert _extract_temp_from_entry(entry_dict, binary=True) == 1.1
    assert _extract_temp_from_entry(entry_dict, binary=False) == 1.3
    assert _extract_temp_from_entry(entry_dict, binary=None) == 1.25
    assert _extract_temp_from_entry({"bad": "val"}) is None

    mapping: dict[str, Any] = {
        "gpt-4": 1.0,
        "qwen2.5-coder-32b": {"optimal_temperature": 1.5},
        "qwen2.5": 1.2,
        "default": 1.0,
    }
    assert _lookup_temperature_in_mapping("default", mapping) is None
    assert _lookup_temperature_in_mapping("qwen2.5-coder-32b", mapping) == 1.5
    assert _lookup_temperature_in_mapping("qwen2.5-coder-7b-instruct", mapping) == 1.2
    assert _lookup_temperature_in_mapping("qwen2.5-coder-7b-instruct", mapping, allow_family_fallback=False) is None
    assert _lookup_temperature_in_mapping("unknown-model", mapping) is None

    assert not has_known_or_cached_temperature(None)
    assert not has_known_or_cached_temperature("default")
    assert get_model_temperature(None) == 1.0
    assert get_model_temperature("default") == 1.0


def test_median_and_binary_auroc() -> None:
    assert _median([]) == 0.0
    assert _median([5.0]) == 5.0
    assert _median([1.0, 3.0, 2.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    assert _compute_binary_auroc([]) is None
    assert _compute_binary_auroc([(1.0, 1), (2.0, 1)]) is None
    assert _compute_binary_auroc([(1.0, 0), (2.0, 0)]) is None
    assert _compute_binary_auroc([(2.0, 1), (-1.0, 0)]) == 1.0
    assert _compute_binary_auroc([(0.0, 1), (0.0, 0)]) == 0.5


def test_multiclass_accum_and_fit_temperature() -> None:
    assert _debiased_multiclass_accum([], [], []) == {}

    top_fwd = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.3}]
    top_rev = [{"token": "A", "logprob": -1.9}, {"token": "B", "logprob": -0.2}]
    accum = _debiased_multiclass_accum(top_fwd, top_rev, ["A", "B"])
    assert "A" in accum and "B" in accum

    accum_empty = _debiased_multiclass_accum(
        [{"token": "Z", "logprob": float("-inf")}],
        [{"token": "Z", "logprob": float("-inf")}],
        ["A", "B"],
    )
    assert "A" in accum_empty and "B" in accum_empty

    assert _fit_multiclass_temperature([], min_temp=True) == 1.0
    assert _fit_multiclass_temperature([]) == 1.0

    mc_samples = [
        ({"A": 0.0, "B": 0.0}, "A"),
    ]
    assert _fit_multiclass_temperature(mc_samples) == 1.0

    mc_low_acc = [
        ({"A": 5.0, "B": 0.0}, "B"),
        ({"A": 4.0, "B": 0.0}, "B"),
    ]
    assert _fit_multiclass_temperature(mc_low_acc) == 1.0

    mc_good = [
        ({"A": 2.0, "B": -1.0}, "A"),
        ({"A": -1.5, "B": 2.5}, "B"),
        ({"A": 1.8, "B": -0.5}, "A"),
        ({"A": -0.8, "B": 1.2}, "B"),
    ]
    temp_fit = _fit_multiclass_temperature(mc_good, binary_auroc=0.9)
    assert 0.65 <= temp_fit <= 10.0


def test_persist_calibration_profile_and_symlinks(tmp_path: Path) -> None:
    reset_runtime_calibration_cache()
    target_cache = tmp_path / "cache_dir" / "cal.json"
    cal_res = CalibrationResult(
        optimal_temperature=1.35,
        brier_score_before=0.2,
        brier_score_after=0.15,
        ece_before=0.08,
        ece_after=0.03,
        multiclass_temperature=1.2,
        probe_auroc=0.92,
        median_abs_log_odds=2.4,
    )
    _persist_calibration_profile("test-model", cal_res, cache_path=target_cache, strict_cache_io=True)
    assert target_cache.exists()

    sym_cache = tmp_path / "sym_cal.json"
    sym_cache.symlink_to(target_cache)
    with pytest.raises(OSError, match="Refusing to write calibration cache through symlink"):
        _persist_calibration_profile("test-model", cal_res, cache_path=sym_cache, strict_cache_io=True)

    _persist_calibration_profile("test-model-ephemeral", cal_res, cache_path=sym_cache, strict_cache_io=False)
    cached_ephem = _load_cached_mapping(sym_cache)
    assert cached_ephem is not None


def test_surface_token_map_validation_and_tokenizer() -> None:
    with pytest.raises(ValueError, match="must contain non-negative integers"):
        SurfaceTokenMap(positive_token_ids=(True,), negative_token_ids=(1,))
    with pytest.raises(ValueError, match="must contain non-negative integers"):
        SurfaceTokenMap(positive_token_ids=(-1,), negative_token_ids=(1,))
    with pytest.raises(ValueError, match="must not be empty"):
        SurfaceTokenMap(positive_token_ids=(), negative_token_ids=(1,))
    with pytest.raises(ValueError, match="must be disjoint"):
        SurfaceTokenMap(positive_token_ids=(1, 2), negative_token_ids=(2, 3))

    smap = SurfaceTokenMap(positive_token_ids=(10, 11), negative_token_ids=(20, 21))
    bias = smap.to_logit_bias(bias=50.0)
    assert bias["10"] == 50.0
    assert bias["20"] == 50.0

    class _MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            clean = text.strip().upper()
            if clean == "A":
                return [101]
            if clean == "B":
                return [102]
            return [999, 998]

    smap_tok = SurfaceTokenMap.from_tokenizer(_MockTokenizer(), pos_label="A", neg_label="B")
    assert 101 in smap_tok.positive_token_ids
    assert 102 in smap_tok.negative_token_ids

    class _FailTokenizer:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

    with pytest.raises(ValueError, match="does not map to any single-token ID"):
        SurfaceTokenMap.from_tokenizer(_FailTokenizer(), pos_label="A", neg_label="B")


def test_marginalize_pole_logprob_and_debiased_verdict() -> None:
    with pytest.raises(ValueError, match="Temperature must be a finite strictly positive"):
        marginalize_pole_logprob([0.0, 0.0], [0], [1], temperature=-1.0)
    with pytest.raises(ValueError, match="Temperature must be a finite strictly positive"):
        marginalize_pole_logprob([0.0, 0.0], [0], [1], temperature=True)
    with pytest.raises(ValueError, match="pos_ids and neg_ids must not be empty"):
        marginalize_pole_logprob([0.0, 0.0], [], [1])
    with pytest.raises(ValueError, match="must contain non-negative integer token IDs"):
        marginalize_pole_logprob([0.0, 0.0], [-1], [1])

    logits = [0.0, 2.0, -1.0, 3.0]
    lp_pos, lp_neg = marginalize_pole_logprob(logits, [1], [3], temperature=1.0)
    assert lp_pos < lp_neg
    assert math.isclose(math.exp(lp_pos) + math.exp(lp_neg), 1.0, abs_tol=1e-5)

    lp_nan1, _lp_nan2 = marginalize_pole_logprob([float("nan"), 1.0], [0], [1])
    assert math.isclose(lp_nan1, -math.log(2.0), abs_tol=1e-5)

    lp_inf1, _lp_inf2 = marginalize_pole_logprob([float("inf"), float("inf")], [0], [1])
    assert math.isclose(lp_inf1, -math.log(2.0), abs_tol=1e-5)

    lp_inf_diff1, lp_inf_diff2 = marginalize_pole_logprob([float("inf"), float("-inf")], [0], [1])
    assert lp_inf_diff1 == 0.0 and lp_inf_diff2 == -float("inf")

    smap = SurfaceTokenMap(positive_token_ids=(1,), negative_token_ids=(2,))
    debiased_res = compute_symmetric_debiased_verdict(
        logits_pass_first=[0.0, 2.0, -2.0],
        logits_fail_first=[0.0, -1.8, 1.9],
        token_map=smap,
        temperature=1.0,
    )
    assert isinstance(debiased_res, DebiasedLogitResult)
    assert 0.0 <= debiased_res.calibrated_prob_pass <= 1.0
    assert 0.0 <= debiased_res.shannon_entropy <= 1.0


def test_fit_calibration_validation_and_branches() -> None:
    with pytest.raises(CalibrationError, match="Expected finite 0 < min_temp < max_temp"):
        fit_calibration([], min_temp=-1.0)
    with pytest.raises(CalibrationError, match="Expected finite 0 < min_temp < max_temp"):
        fit_calibration([], min_temp=5.0, max_temp=2.0)
    with pytest.raises(CalibrationError, match="num_bins must be an integer >= 1"):
        fit_calibration([], num_bins=0)

    with pytest.warns(CalibrationWarning, match="received an empty sample sequence"):
        empty_res = fit_calibration([])
    assert empty_res.optimal_temperature == 1.0

    with pytest.raises(CalibrationError, match="Calibration samples must contain finite"):
        fit_calibration([(True, 1.0, 1)])
    with pytest.raises(CalibrationError, match="Calibration samples must contain finite"):
        fit_calibration([(float("nan"), 1.0, 1)])
    with pytest.raises(CalibrationError, match="Calibration sample label must be 0 or 1"):
        fit_calibration([(1.0, 1.0, 2)])

    with pytest.warns(CalibrationWarning, match="samples from only a single class"):
        res_single = fit_calibration([(1.0, 1.0, 1), (2.0, 2.0, 1)])
    assert res_single.optimal_temperature > 0.0

    with pytest.warns(CalibrationWarning, match="fewer than 4 samples"):
        res_few = fit_calibration([(1.0, 1.0, 1), (-1.0, -1.0, 0)])
    assert res_few.optimal_temperature is not None

    samples_probe_weak = [
        (0.1, -0.1, 1),
        (0.2, -0.2, 0),
        (-0.1, 0.1, 1),
        (-0.2, 0.2, 0),
    ]
    res_weak = fit_calibration(samples_probe_weak, min_temp=0.1, max_temp=10.0)
    assert res_weak.optimal_temperature >= 1.0


def test_auto_calibrate_backend_sync_and_async(tmp_path: Path) -> None:
    backend = StubBackend(default_prob=0.85)
    cal_res = auto_calibrate_backend(
        backend,
        model_key="test-stub",
        save_cache=True,
        cache_path=tmp_path / "cal.json",
        strict_cache_io=False,
    )
    assert isinstance(cal_res, CalibrationResult)
    assert cal_res.optimal_temperature > 0.0

    class _BrokenBackend(StubBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    with pytest.raises(BackendError, match="returned no token log-probabilities"):
        auto_calibrate_backend(_BrokenBackend(), save_cache=False)


@pytest.mark.asyncio
async def test_aauto_calibrate_backend_sync_and_async(tmp_path: Path) -> None:
    backend = StubBackend(default_prob=0.85)
    cal_res = await aauto_calibrate_backend(
        backend,
        model_key="test-async-stub",
        save_cache=True,
        cache_path=tmp_path / "async_cal.json",
        strict_cache_io=False,
    )
    assert isinstance(cal_res, CalibrationResult)

    class _BrokenAsyncBackend(StubBackend):
        async def aquery_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    with pytest.raises(BackendError, match="returned no token log-probabilities"):
        await aauto_calibrate_backend(_BrokenAsyncBackend(), save_cache=False)


# ---------------------------------------------------------------------------
# Section 2: Conformal & Adaptive Calibration Unit Tests
# ---------------------------------------------------------------------------


def test_conformal_min_samples_and_thresholds() -> None:
    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        min_samples_per_class(0.0)
    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        min_samples_per_class(1.0)
    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        min_samples_per_class(True)

    assert min_samples_per_class(0.05) == 39
    assert min_samples_per_class(0.10) == 19

    probs = [0.9] * 10
    labels = [True] * 5
    with pytest.raises(CalibrationError, match="matching sequences of at least 10 samples"):
        calibrate_cascade_thresholds(probs, labels)

    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        calibrate_cascade_thresholds([0.5] * 10, [True] * 5 + [False] * 5, target_risk_alpha=1.5)

    with pytest.raises(CalibrationError, match="must contain finite probabilities"):
        calibrate_cascade_thresholds([True] * 10, [True] * 5 + [False] * 5)

    with pytest.raises(CalibrationError, match="must contain booleans or 0/1 integers"):
        calibrate_cascade_thresholds([0.5] * 10, ["bad"] * 10)  # type: ignore[list-item]

    with pytest.raises(CalibrationError, match="must contain both positive and negative"):
        calibrate_cascade_thresholds([0.5] * 10, [True] * 10)

    with pytest.warns(CalibrationWarning, match="Conformal guarantee does not hold"):
        thresh = calibrate_cascade_thresholds([0.9] * 5 + [0.1] * 5, [True] * 5 + [False] * 5, target_risk_alpha=0.05)
    assert not thresh.guarantee_holds
    assert thresh.tau_reject <= thresh.tau_accept


def test_load_calibration_dataset_formats(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_calibration_dataset(non_existent)

    json_cases = [
        {"rubric": "R1", "candidate": "C1", "label": True},
        {"rubric": "R2", "candidate": "C2", "passed": False},
        {"instruction": "R3", "response": "C3", "expected": 1},
        {"rubric": "R4", "candidate": "C4"},
    ]
    p_json = tmp_path / "cases.json"
    p_json.write_text(json.dumps({"test_cases": json_cases}), encoding="utf-8")
    loaded_json = load_calibration_dataset(p_json)
    assert len(loaded_json) == 3

    p_json_raw = tmp_path / "cases_raw.json"
    p_json_raw.write_text(json.dumps(json_cases[:2]), encoding="utf-8")
    assert len(load_calibration_dataset(p_json_raw)) == 2

    p_json_bad = tmp_path / "bad.json"
    p_json_bad.write_text('"string only"', encoding="utf-8")
    with pytest.raises(CalibrationError, match="Unrecognized JSON structure"):
        load_calibration_dataset(p_json_bad)

    p_jsonl = tmp_path / "cases.jsonl"
    p_jsonl.write_text(
        '{"rubric": "R1", "candidate": "C1", "label": "pass"}\n\n{"rubric": "R2", "candidate": "C2", "label": 0}\n',
        encoding="utf-8",
    )
    loaded_jsonl = load_calibration_dataset(p_jsonl)
    assert len(loaded_jsonl) == 2

    p_md_table = tmp_path / "cases_table.md"
    p_md_table.write_text(
        "| Rubric | Candidate | Label |\n|---|---|---|\n| R1 | C1 | pass |\n| R2 | C2 | fail |\n",
        encoding="utf-8",
    )
    loaded_md = load_calibration_dataset(p_md_table)
    assert len(loaded_md) == 2

    p_md_sec = tmp_path / "cases_sec.md"
    p_md_sec.write_text(
        "### Case 1\n**Rubric**: R1\n**Candidate**: C1\n**Label**: pass\n\n### Case 2\n**Rubric**: R2\n**Candidate**: C2\n**Label**: fail\n",
        encoding="utf-8",
    )
    loaded_md_sec = load_calibration_dataset(p_md_sec)
    assert len(loaded_md_sec) == 2


def test_calibration_profile_validation_and_bundle(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError):
        CalibrationProfile(binary_temperature=-1.0)
    with pytest.raises(CalibrationError):
        CalibrationProfile(multiclass_temperature=0.0)
    with pytest.raises(CalibrationError):
        CalibrationProfile(fail_threshold=1.5)
    with pytest.raises(CalibrationError):
        CalibrationProfile(fail_threshold=0.8, pass_threshold=0.5)

    prof = CalibrationProfile(model="test-prof", binary_temperature=1.2, fail_threshold=0.3, pass_threshold=0.7)
    prof_path = tmp_path / "profile.json"
    prof.save(prof_path)
    loaded = CalibrationProfile.load(prof_path)
    assert loaded.model == "test-prof"

    sym_prof = tmp_path / "sym_profile.json"
    sym_prof.symlink_to(prof_path)
    with pytest.raises(CalibrationError, match="Refusing to read CalibrationProfile from symlink"):
        CalibrationProfile.load(sym_prof)

    cal_res = CalibrationResult(optimal_temperature=1.25, brier_score_before=0.2, brier_score_after=0.15, ece_before=0.1, ece_after=0.05)
    prof_from_res = CalibrationProfile.from_calibration_result(cal_res, model="from-res")
    assert prof_from_res.binary_temperature == 1.25

    prof_from_samples = CalibrationProfile.from_log_odds_samples([(1.0, 1.0, 1), (-1.0, -1.0, 0)] * 10, model="from-samples")
    assert prof_from_samples.model == "from-samples"


def test_fit_and_afit_calibration_profile(tmp_path: Path) -> None:
    samples: list[tuple[str, str, bool | int]] = [
        ("Rubric 1", "def foo(): pass", True),
        ("Rubric 2", "syntax error", False),
    ] * 6
    backend = StubBackend(default_prob=0.8)

    prof = fit_calibration_profile(
        backend,
        model="stub-fit",
        samples=samples,
        target_risk_alpha=0.1,
        save_cache=True,
        cache_path=tmp_path / "fit_cal.json",
    )
    assert prof.model == "stub-fit"
    assert prof.fail_threshold is not None and prof.pass_threshold is not None
    assert prof.fail_threshold < prof.pass_threshold


@pytest.mark.asyncio
async def test_afit_calibration_profile_async(tmp_path: Path) -> None:
    samples: list[tuple[str, str, bool | int]] = [
        ("Rubric 1", "def foo(): pass", True),
        ("Rubric 2", "syntax error", False),
    ] * 6
    backend = StubBackend(default_prob=0.8)

    prof = await afit_calibration_profile(
        backend,
        model="stub-afit",
        samples=samples,
        target_risk_alpha=0.1,
        output_path=tmp_path / "afit_profile.json",
    )
    assert prof.model == "stub-afit"
    assert (tmp_path / "afit_profile.json").exists()

    prof_auto = await afit_calibration_profile(
        backend,
        model="stub-auto-afit",
        samples=None,
    )
    assert prof_auto.model == "stub-auto-afit"


def test_adaptive_conformal_controller_comprehensive() -> None:
    with pytest.raises(ConfigurationError, match="pass_threshold must be a finite float"):
        AdaptiveConformalController(pass_threshold=True)
    with pytest.raises(ConfigurationError, match="pass_threshold must be in"):
        AdaptiveConformalController(pass_threshold=1.5)
    with pytest.raises(ConfigurationError, match="must be strictly greater than fail_threshold"):
        AdaptiveConformalController(pass_threshold=0.4, fail_threshold=0.6)
    with pytest.raises(ConfigurationError, match="target_risk_alpha must be strictly in"):
        AdaptiveConformalController(target_risk_alpha=1.0)
    with pytest.raises(ConfigurationError, match="step_size must be a positive"):
        AdaptiveConformalController(step_size=-0.1)
    with pytest.raises(ConfigurationError, match="min_theta must be a finite float <= 0.0"):
        AdaptiveConformalController(min_theta=1.0)
    with pytest.raises(ConfigurationError, match="abstain_theta must be a positive"):
        AdaptiveConformalController(abstain_theta=-0.5)
    with pytest.raises(ConfigurationError, match="min_band_gap must be in"):
        AdaptiveConformalController(min_band_gap=0.6)
    with pytest.raises(ConfigurationError, match="audit_sample_rate must be in"):
        AdaptiveConformalController(audit_sample_rate=1.5)
    with pytest.raises(ConfigurationError, match="override_on_audit must be a bool"):
        AdaptiveConformalController(override_on_audit="not_bool")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="max_history must be an integer >= 1"):
        AdaptiveConformalController(max_history=0)
    with pytest.raises(ConfigurationError, match="seed must be an int or None"):
        AdaptiveConformalController(seed="seed")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="feedback_mode must be"):
        AdaptiveConformalController(feedback_mode="bad_mode")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="audit_sample_rate must be > 0.0"):
        AdaptiveConformalController(feedback_mode="cascade_audit", audit_sample_rate=0.0)
    with pytest.raises(ConfigurationError, match="audit_sample_rate must be 0.0"):
        AdaptiveConformalController(feedback_mode="explicit", audit_sample_rate=0.2)

    ctrl = AdaptiveConformalController(
        pass_threshold=0.7,
        fail_threshold=0.3,
        target_risk_alpha=0.05,
        step_size=0.1,
        abstain_theta=2.0,
        feedback_mode="cascade_audit",
        audit_sample_rate=1.0,
        seed=42,
    )
    assert ctrl.target_risk_alpha == 0.05
    assert ctrl.step_size == 0.1
    assert ctrl.feedback_mode == "cascade_audit"
    assert ctrl.audit_sample_rate == 1.0
    assert ctrl.override_on_audit is True
    assert ctrl.theta == 0.0
    assert len(ctrl.history) == 0

    assert ctrl.should_audit_fast_path() is True
    fail_t, pass_t, force_ambig = ctrl.current_thresholds()
    assert fail_t == 0.3 and pass_t == 0.7 and force_ambig is False

    fp = ctrl.cache_fingerprint()
    assert fp["fail_threshold"] == 0.3
    assert fp["pass_threshold"] == 0.7

    with pytest.raises(ConfigurationError, match="ground_truth must be a bool"):
        ctrl.observe(DecisionStatus.PASSED, "not_bool")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="sampling_prob must be in"):
        ctrl.observe(DecisionStatus.PASSED, True, sampling_prob=1.5)

    with pytest.raises(ConfigurationError, match="Invalid DecisionStatus"):
        ctrl.observe("INVALID_STATUS", True)

    rec_pass = ctrl.observe(DecisionStatus.PASSED, True)
    assert isinstance(rec_pass, AdaptiveStepRecord)
    assert rec_pass.raw_error == 0.0
    assert ctrl.theta <= 0.0

    rec_err = ctrl.observe(DecisionStatus.PASSED, False)
    assert rec_err.raw_error == 1.0
    assert ctrl.theta > rec_pass.theta_after

    with pytest.raises(ConfigurationError, match="delta_theta must be a finite float >= 0.0"):
        ctrl.apply_drift_boost(-1.0)
    ctrl.apply_drift_boost(0.0)
    ctrl.apply_drift_boost(0.5)
    assert ctrl.theta > 0.0

    summ = ctrl.summary()
    assert isinstance(summ, AdaptiveConformalSummary)
    summ_dict = summ.to_dict()
    assert "total_steps" in summ_dict
    assert summ_dict["total_steps"] == 2

    prof_exported = ctrl.to_profile()
    assert prof_exported.fail_threshold is not None and prof_exported.pass_threshold is not None and prof_exported.fail_threshold < prof_exported.pass_threshold

    ctrl.reset()
    assert ctrl.theta == 0.0
    assert len(ctrl.history) == 0


# ---------------------------------------------------------------------------
# Section 3: Calibration Auditing & Split Leakage Unit Tests
# ---------------------------------------------------------------------------


def test_audit_bool_parsing_and_extraction() -> None:
    assert _parse_bool_label(True) is True
    assert _parse_bool_label(False) is False
    assert _parse_bool_label(1) is True
    assert _parse_bool_label(0) is False
    assert _parse_bool_label(2) is True
    assert _parse_bool_label(0.9) is True
    assert _parse_bool_label(0.1) is False
    assert _parse_bool_label("pass") is True
    assert _parse_bool_label("fail") is False
    assert _parse_bool_label(None) is None

    assert _extract_case_from_mapping({}) is None
    assert _extract_case_from_mapping({"rubric": "R", "candidate": "C"}) is None
    case = _extract_case_from_mapping({"rubric": "R", "candidate": "C", "label": True})
    assert case is not None and case.rubric == "R"

    pair_cases = _extract_pairwise_cases({
        "rubric": "R",
        "candidate_a": "CA",
        "candidate_b": "CB",
        "winner": "A",
    })
    assert len(pair_cases) == 2

    assert _extract_cases_from_mapping_root({}, "path") == []
    assert len(_extract_cases_from_mapping_root({"test_cases": [{"rubric": "R", "candidate": "C", "label": True}]}, "path")) == 1


def test_audit_dataset_and_split_leakage(tmp_path: Path) -> None:
    single_class_cases = [
        {"rubric": f"R{i}", "candidate": f"def func_{i}(): return {i}", "label": True}
        for i in range(12)
    ]
    rep_single = audit_benchmark_dataset(single_class_cases)
    assert isinstance(rep_single, DatasetAuditReport)
    assert len(rep_single.warnings) > 0
    formatted = rep_single.format_summary()
    assert "Dataset Integrity Audit:" in formatted

    balanced_cases = [
        {"rubric": f"R{i}", "candidate": f"def func_{i}(): return {i}", "label": (i % 2 == 0)}
        for i in range(20)
    ]
    rep_balanced = audit_benchmark_dataset(balanced_cases)
    assert rep_balanced.total_cases == 20

    train_cases = balanced_cases[:10]
    test_cases_leaked = balanced_cases[8:15]
    leak_rep = check_split_leakage(train_cases, test_cases_leaked)
    assert isinstance(leak_rep, SplitLeakageReport)
    assert leak_rep.exact_candidate_overlaps >= 2
    assert leak_rep.is_clean is False
    leak_formatted = leak_rep.format_summary()
    assert "Cross-Split Leakage Check:" in leak_formatted

    test_cases_clean = [
        {"rubric": f"R_clean_{i}", "candidate": f"def clean_unique_code_{i}(): pass", "label": (i % 2 == 0)}
        for i in range(10)
    ]
    leak_clean = check_split_leakage(train_cases, test_cases_clean)
    assert leak_clean.exact_candidate_overlaps == 0
    assert leak_clean.is_clean is True


# ---------------------------------------------------------------------------
# Section 4: AST Gate & Rule Validation Unit Tests
# ---------------------------------------------------------------------------


class _PydanticSchemaForAST(BaseModel):
    name: str
    count: int = Field(ge=1)


def test_ast_gate_syntax_schema_complexity_and_dag() -> None:
    ast_syntax = ASTGate(syntax=True)
    v_pass = ast_syntax.evaluate("def valid_python(): return 42")
    assert v_pass.passed is True

    v_fail = ast_syntax.evaluate("def broken_python(:")
    assert v_fail.passed is False
    assert v_fail.traces[0].failure_code == ASTFailureCode.SYNTAX_ERROR

    ast_schema = ASTGate(json_schema=_PydanticSchemaForAST)
    v_sch_pass = ast_schema.evaluate('{"name": "test", "count": 5}')
    assert v_sch_pass.passed is True

    v_sch_fail = ast_schema.evaluate('{"name": "test", "count": 0}')
    assert v_sch_fail.passed is False
    assert v_sch_fail.traces[0].failure_code == ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR

    ast_len = ASTGate(min_length=5, max_length=15)
    assert ast_len.evaluate("too short").passed is True
    assert ast_len.evaluate("a").passed is False
    assert ast_len.evaluate("this is way too long for the maximum limit").passed is False

    ast_dag = ASTGate(acyclic_dag=True)
    valid_dag = [
        {"id": "A", "dependencies": []},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["B"]},
    ]
    assert ast_dag.evaluate(valid_dag).passed is True

    cycle_dag = [
        {"id": "A", "dependencies": ["B"]},
        {"id": "B", "dependencies": ["A"]},
    ]
    v_dag_cycle = ast_dag.evaluate(cycle_dag)
    assert v_dag_cycle.passed is False
    assert v_dag_cycle.traces[0].failure_code == ASTFailureCode.DAG_CYCLE_DETECTED

    missing_dep_dag = [
        {"id": "A", "dependencies": ["NON_EXISTENT"]},
    ]
    v_dep_missing = ast_dag.evaluate(missing_dep_dag)
    assert v_dep_missing.passed is False

    v_invalid_dag_input = ast_dag.evaluate("not a valid dag representation")
    assert v_invalid_dag_input.passed is False


def test_ast_gate_security_and_regex() -> None:
    ast_sec = ASTGate(forbidden_imports=["os", "subprocess"], forbidden_patterns=[r"eval\(", r"exec\("])
    v_sec_import = ast_sec.evaluate("import os\nprint('hello')")
    assert v_sec_import.passed is False
    assert v_sec_import.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT

    v_sec_call = ast_sec.evaluate("eval('2 + 2')")
    assert v_sec_call.passed is False
    assert v_sec_call.traces[0].failure_code == ASTFailureCode.FORBIDDEN_PATTERN

    ast_req = ASTGate(required_sections=["# Section 1"])
    v_req_fail = ast_req.evaluate("def execute_task(): pass")
    assert v_req_fail.passed is False
    assert v_req_fail.traces[0].failure_code == ASTFailureCode.MISSING_SECTION

    v_req_pass = ast_req.evaluate("# Section 1\ndef execute_task(): pass")
    assert v_req_pass.passed is True


@pytest.mark.asyncio
async def test_ast_gate_async_evaluate() -> None:
    ast_gate = ASTGate(syntax=True)
    v_pass = await ast_gate.aevaluate("def valid_async(): return True")
    assert v_pass.passed is True
    v_fail = await ast_gate.aevaluate("broken syntax(")
    assert v_fail.passed is False


def test_base_gate_dunder_operators() -> None:
    g1 = ASTGate(syntax=True)
    g2 = LogitGate(rubric="Rubric 1", backend=StubBackend(default_prob=0.9))
    g3 = JudgeGate(backend=StubBackend(diagnostic_text="All good"))

    c1 = g1 >> g2
    assert isinstance(c1, Cascade)
    assert len(c1.gates) == 2

    c2 = c1 >> g3
    assert isinstance(c2, Cascade)
    assert len(c2.gates) == 3

    c3 = g1 >> c1
    assert isinstance(c3, Cascade)
    assert len(c3.gates) == 3

    c4 = c1 >> c2
    assert isinstance(c4, Cascade)
    assert len(c4.gates) == 5

    c_or = g1 | g2
    assert isinstance(c_or, Cascade)

    verdict = g1("def func(): pass")
    assert verdict.passed is True


# ---------------------------------------------------------------------------
# Section 5: Generative Judge Gate & Self-Healing Probes Unit Tests
# ---------------------------------------------------------------------------


def test_judge_gate_validation_and_prompt_modes() -> None:
    with pytest.raises(ConfigurationError, match="Invalid trigger"):
        JudgeGate(trigger="unsupported_trigger")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="rubric cannot be empty or whitespace"):
        JudgeGate(rubric="   ")
    with pytest.raises(ConfigurationError, match="max_tokens must be a positive integer"):
        JudgeGate(max_tokens=0)

    jg_standard = JudgeGate(rubric="Quality check", backend=StubBackend(diagnostic_text="[VERDICT: PASS]"))
    p_std = jg_standard._compile_prompt("candidate code")
    assert "[EVALUATION RUBRIC]" in p_std
    assert "Quality check" in p_std

    jg_short = JudgeGate(rubric="Quality check", backend=StubBackend(diagnostic_text="PASS"), max_tokens=20)
    p_short = jg_short._compile_prompt("candidate code")
    assert "[VERDICT: PASS] or [VERDICT: FAIL]" in p_short


def test_judge_gate_verdict_extraction_heuristics() -> None:
    assert _extract_verification_notes("") == ""
    assert _extract_verification_notes("I cannot fulfill this request.") == ""
    cleaned = _extract_verification_notes("Here are findings:\n[VERDICT: PASS]\nGood code.")
    assert "Good code." in cleaned

    objs = list(_iter_json_objects('Some text {"pass": true, "score": 0.95} and {"pass": false}'))
    assert len(objs) == 2

    v_tag = _extract_structured_verdict("Analysis complete.\n[VERDICT: PASS]")
    assert v_tag is not None and v_tag[0] is True

    v_tag_fail = _extract_structured_verdict("Critique:\nVerdict: FAIL")
    assert v_tag_fail is not None and v_tag_fail[0] is False

    v_quoted = _extract_structured_verdict('Rubric says "[VERDICT: PASS]" but the candidate fails.')
    assert v_quoted is None

    v_json = _extract_structured_verdict('Critique:\n```json\n{"verdict": "pass", "score": 0.88}\n```')
    assert v_json is not None and v_json[0] is True and v_json[2] == 0.88

    v_kw_lead = _extract_keyword_verdict("PASS - all criteria met with no major issues.")
    assert v_kw_lead is not None and v_kw_lead[0] is True

    v_kw_neg = _extract_keyword_verdict("The candidate did not pass validation.")
    assert v_kw_neg is not None and v_kw_neg[0] is False


def test_judge_gate_triggers_and_evaluation() -> None:
    jg_ambig_only = JudgeGate(trigger="on_ambiguous_only", backend=StubBackend(diagnostic_text="[VERDICT: PASS]"))
    assert jg_ambig_only.trigger == "on_ambiguous_only"

    lg_upstream = LogitGate(rubric="R", backend=StubBackend(default_prob=0.9))
    v_passed = Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="L", total_latency_ms=1.0)
    v_failed = Verdict(passed=False, status=DecisionStatus.FAILED, triggered_layer="L", total_latency_ms=1.0)
    v_ambig = Verdict(passed=False, status=DecisionStatus.AMBIGUOUS, triggered_layer="L", total_latency_ms=1.0)

    assert _should_escalate("on_ambiguous_only", lg_upstream, v_passed) is False
    assert _should_escalate("on_ambiguous_only", lg_upstream, v_failed) is False
    assert _should_escalate("on_ambiguous_only", lg_upstream, v_ambig) is True
    assert _should_escalate("adjudicate", lg_upstream, v_failed) is True
    assert _should_escalate("always", lg_upstream, v_passed) is True

    jg_always = JudgeGate(trigger="always", backend=StubBackend(diagnostic_text="[VERDICT: PASS]"))
    verdict_standalone = jg_always.evaluate("candidate text")
    assert verdict_standalone.passed is True
    assert jg_always.layer_tier == 2


@pytest.mark.asyncio
async def test_judge_gate_async_evaluate() -> None:
    jg_async = JudgeGate(
        rubric="Async rubric",
        trigger="always",
        backend=StubBackend(diagnostic_text="[VERDICT: PASS]\nClean async code."),
    )
    verdict = await jg_async.aevaluate("async candidate")
    assert verdict.passed is True


# ---------------------------------------------------------------------------
# Section 6: Logit, Choice, Score, & Multi-Rubric Gates Unit Tests
# ---------------------------------------------------------------------------


def test_choice_gate_permutation_modes_and_boundaries() -> None:
    with pytest.raises(ConfigurationError, match="ChoiceGate requires at least 2 options"):
        ChoiceGate(question="Which?", options={"A": "One"})

    with pytest.raises(ConfigurationError, match="Invalid permutation_mode"):
        ChoiceGate(question="Which?", options={"A": "1", "B": "2"}, permutation_mode="invalid_mode")  # type: ignore[arg-type]

    cg_derange_odd = ChoiceGate(
        question="Which implementation is best?",
        options={"A": "Opt A", "B": "Opt B", "C": "Opt C"},
        permutation_mode="derangement",
        backend=StubBackend(default_prob=0.9),
    )
    perms_odd = cg_derange_odd._permuted_key_indices()
    assert len(perms_odd) == 2
    for i in range(3):
        assert perms_odd[0][i] != perms_odd[1][i]

    cg_derange_even = ChoiceGate(
        question="Which?",
        options={"A": "1", "B": "2", "C": "3", "D": "4"},
        permutation_mode="derangement",
        backend=StubBackend(default_prob=0.9),
    )
    perms_even = cg_derange_even._permuted_key_indices()
    assert len(perms_even) == 2

    cg_cyclic = ChoiceGate(
        question="Which?",
        options={"A": "1", "B": "2", "C": "3"},
        permutation_mode="cyclic",
        backend=StubBackend(default_prob=0.9),
    )
    assert len(cg_cyclic._permuted_key_indices()) == 3

    cg_reverse = ChoiceGate(
        question="Which?",
        options={"A": "1", "B": "2", "C": "3"},
        permutation_mode="reverse",
        backend=StubBackend(default_prob=0.9),
    )
    assert len(cg_reverse._permuted_key_indices()) == 2

    v_choice = cg_derange_odd.evaluate("Candidate choice")
    assert v_choice.passed is True
    assert v_choice.choice in ("A", "B", "C")


@pytest.mark.asyncio
async def test_choice_gate_async_evaluate() -> None:
    cg = ChoiceGate(
        question="Which?",
        options={"A": "First", "B": "Second"},
        backend=StubBackend(default_prob=0.8),
    )
    v = await cg.aevaluate("Async choice candidate")
    assert v.passed is True


def test_score_gate_bounds_and_inverted_scale() -> None:
    with pytest.raises(ConfigurationError, match="min_score.*must be strictly less than max_score"):
        ScoreGate(rubric="R", min_score=5, max_score=1)

    sg_normal = ScoreGate(
        rubric="Rate 1 to 5",
        min_score=1,
        max_score=5,
        backend=StubBackend(default_prob=0.8),
    )
    v_norm = sg_normal.evaluate("Good candidate")
    assert v_norm.passed is True
    assert v_norm.score is not None

    _p_fwd, p_rev = sg_normal._score_prompts("Good candidate")
    assert p_rev is not None
    assert "inverted scale" in p_rev


@pytest.mark.asyncio
async def test_score_gate_async_evaluate() -> None:
    sg = ScoreGate(rubric="Async rate", min_score=1, max_score=5, backend=StubBackend(default_prob=0.7))
    v = await sg.aevaluate("Candidate")
    assert v.passed is True


def test_multi_rubric_gate_modes_and_weights() -> None:
    with pytest.raises(ConfigurationError, match="MultiRubricGate rubrics must be a sequence of strings"):
        MultiRubricGate(rubrics="not a sequence")

    with pytest.raises(ConfigurationError, match="MultiRubricGate needs one or more non-empty"):
        MultiRubricGate(rubrics=[])

    mrg_all = MultiRubricGate(
        rubrics=["R1: Syntax", "R2: Types"],
        pass_threshold=0.6,
        backend=StubBackend(default_prob=0.9),
    )
    v_all = mrg_all.evaluate("Candidate code")
    assert v_all.passed is True

    verdicts_each = mrg_all.evaluate_each("Candidate code")
    assert len(verdicts_each) == 2

    mrg_ff = MultiRubricGate(
        rubrics=["R1", "R2"],
        fail_fast=True,
        pass_threshold=0.8,
        backend=StubBackend(default_prob=0.1),
    )
    v_ff = mrg_ff.evaluate("Candidate")
    assert v_ff.passed is False


@pytest.mark.asyncio
async def test_multi_rubric_gate_async_evaluate() -> None:
    mrg = MultiRubricGate(rubrics=["R1", "R2"], backend=StubBackend(default_prob=0.85))
    v = await mrg.aevaluate("Async candidate")
    assert v.passed is True
    v_each = await mrg.aevaluate_each("Async candidate")
    assert len(v_each) == 2


def test_logit_gate_threshold_boundaries_and_async() -> None:
    backend_ambig = StubBackend(default_prob=0.5)
    lg_ambig = LogitGate(rubric="Test", pass_threshold=0.8, fail_threshold=0.2, backend=backend_ambig)
    v_ambig = lg_ambig.evaluate("Candidate")
    assert v_ambig.status == DecisionStatus.AMBIGUOUS
    assert v_ambig.passed is False

    backend_pass = StubBackend(default_prob=0.95)
    lg_pass = LogitGate(rubric="Test", pass_threshold=0.8, backend=backend_pass)
    assert lg_pass.evaluate("Candidate").passed is True

    backend_fail = StubBackend(default_prob=0.05)
    lg_fail = LogitGate(rubric="Test", pass_threshold=0.8, fail_threshold=0.2, backend=backend_fail)
    assert lg_fail.evaluate("Candidate").status == DecisionStatus.FAILED


@pytest.mark.asyncio
async def test_logit_gate_async_evaluate() -> None:
    lg = LogitGate(rubric="Async rubric", backend=StubBackend(default_prob=0.95))
    v = await lg.aevaluate("Candidate")
    assert v.passed is True


# ---------------------------------------------------------------------------
# Section 7: Rank Gate & Best-Of-N Ranking Unit Tests
# ---------------------------------------------------------------------------


def test_rank_gate_modes_and_hodge_curl() -> None:
    with pytest.raises(ConfigurationError, match="mode='round_robin' supports 2..16"):
        RankGate(rubric="R", mode="round_robin")._resolve_effective_mode(20)

    with pytest.raises(ConfigurationError, match="mode='tournament' supports 2..32"):
        RankGate(rubric="R", mode="tournament")._resolve_effective_mode(50)

    rg_auto = RankGate(rubric="R", mode="auto")
    assert rg_auto._resolve_effective_mode(2) == "pairwise"
    assert rg_auto._resolve_effective_mode(5) == "listwise"
    assert rg_auto._resolve_effective_mode(12) == "tournament"
    assert rg_auto._resolve_effective_mode(24) == "tournament"
    assert RankGate(rubric="R", mode="round_robin")._resolve_effective_mode(8) == "round_robin"

    candidates = {
        "A": "def clean_impl(): return True",
        "B": "def messy_impl(): pass",
        "C": "def terrible_impl(): syntax error",
    }
    rg_rr = RankGate(
        rubric="Code quality",
        mode="round_robin",
        backend=StubBackend(default_prob=0.9),
    )
    v_rr = rg_rr.evaluate(candidates)
    assert v_rr.passed is True
    assert v_rr.choice is not None
    assert "cycle_residual" in v_rr.traces[0].metadata

    rg_list = RankGate(
        rubric="Code quality",
        mode="listwise",
        permutation_mode="cyclic",
        backend=StubBackend(default_prob=0.9),
    )
    v_list = rg_list.evaluate(candidates)
    assert v_list.passed is True

    rg_tourn = RankGate(
        rubric="Code quality",
        mode="tournament",
        backend=StubBackend(default_prob=0.9),
    )
    v_tourn = rg_tourn.evaluate(candidates)
    assert v_tourn.passed is True


@pytest.mark.asyncio
async def test_rank_gate_async_evaluate() -> None:
    rg = RankGate(rubric="Quality", mode="round_robin", backend=StubBackend(default_prob=0.85))
    candidates = {"A": "cand A", "B": "cand B", "C": "cand C"}
    v = await rg.aevaluate(candidates)
    assert v.passed is True


def test_best_of_n_gate_stages_and_filtering() -> None:
    bon_gate = BestOfNGate(
        rubric="Code quality",
        pre_gate=ASTGate(syntax=True),
        backend=StubBackend(default_prob=0.9),
    )
    candidates = {
        "A": "def broken(:",
        "B": "def valid_one(): return 1",
        "C": "def valid_two(): return 2",
    }
    v_bon = bon_gate.evaluate(candidates)
    assert v_bon.passed is True
    assert v_bon.choice in ("B", "C")


@pytest.mark.asyncio
async def test_best_of_n_gate_async_evaluate() -> None:
    bon = BestOfNGate(rubric="Quality", backend=StubBackend(default_prob=0.9))
    v = await bon.aevaluate({"A": "cand 1", "B": "cand 2"})
    assert v.passed is True


# ---------------------------------------------------------------------------
# Section 8: Cascaded Routing Pipelines Unit Tests
# ---------------------------------------------------------------------------


def test_cascade_short_circuit_and_escalation() -> None:
    ast_gate = ASTGate(syntax=True)
    lg_gate = LogitGate(rubric="Clean code", pass_threshold=0.8, fail_threshold=0.2, backend=StubBackend(default_prob=0.5))
    jg_gate = JudgeGate(trigger="on_ambiguous_only", backend=StubBackend(diagnostic_text="[VERDICT: PASS]\nFixed."))

    pipeline = ast_gate >> lg_gate >> jg_gate

    assert len(pipeline.gates) == 3
    assert list(pipeline.gates) == [ast_gate, lg_gate, jg_gate]

    v_l0_fail = pipeline.evaluate("def broken(")
    assert v_l0_fail.passed is False
    assert len(v_l0_fail.traces) == 1
    assert v_l0_fail.traces[0].layer_name == "ASTGate"

    lg_pass_gate = LogitGate(rubric="Clean", backend=StubBackend(default_prob=0.95))
    pipe_pass = ast_gate >> lg_pass_gate >> jg_gate
    v_l1_pass = pipe_pass.evaluate("def good(): pass")
    assert v_l1_pass.passed is True
    assert len(v_l1_pass.traces) == 2
    assert v_l1_pass.traces[1].layer_name == "LogitGate"

    v_escalated = pipeline.evaluate("def good(): pass")
    assert v_escalated.passed is True
    assert len(v_escalated.traces) == 3
    assert v_escalated.traces[2].layer_name == "JudgeGate"


@pytest.mark.asyncio
async def test_cascade_async_aevaluate() -> None:
    ast_gate = ASTGate(syntax=True)
    lg_gate = LogitGate(rubric="Clean code", pass_threshold=0.8, fail_threshold=0.2, backend=StubBackend(default_prob=0.5))
    jg_gate = JudgeGate(trigger="on_ambiguous_only", backend=StubBackend(diagnostic_text="[VERDICT: PASS]"))

    pipeline = ast_gate >> lg_gate >> jg_gate
    v_async = await pipeline.aevaluate("def valid_async(): return True")
    assert v_async.passed is True
    assert len(v_async.traces) == 3


def test_cascade_fused_logits_and_helpers() -> None:
    assert _same_nonzero_sign(1.0, 2.0) is True
    assert _same_nonzero_sign(-1.0, -2.0) is True
    assert _same_nonzero_sign(1.0, -1.0) is False
    assert _same_nonzero_sign(0.0, 1.0) is False

    tr1 = LayerTrace(
        layer_name="LogitGate",
        latency_ms=1.0,
        status=DecisionStatus.AMBIGUOUS,
        forward_log_odds=1.5,
        reverse_log_odds=1.2,
        log_odds_pass=1.35,
        calibrated_prob=0.6,
    )
    tr2 = LayerTrace(
        layer_name="JudgeGate",
        latency_ms=1.0,
        status=DecisionStatus.PASSED,
        forward_log_odds=0.5,
        reverse_log_odds=-0.3,
        log_odds_pass=0.1,
        calibrated_prob=0.52,
    )
    v_up = Verdict(passed=False, status=DecisionStatus.AMBIGUOUS, traces=[tr1], probability=0.6, triggered_layer="LogitGate", total_latency_ms=1.0)
    v_judge = Verdict(passed=True, status=DecisionStatus.PASSED, traces=[tr2], probability=0.52, triggered_layer="JudgeGate", total_latency_ms=1.0)
    passed_f, _status_f, prob_f, _lo_f = fuse_escalated_verdict(v_up, v_judge, prior_weight=0.35)
    assert passed_f is True
    assert prob_f is not None


# ---------------------------------------------------------------------------
# Section 9: Verdict Snapshots & Replay Unit Tests
# ---------------------------------------------------------------------------


def test_verdict_snapshot_save_load_replay_explain(tmp_path: Path) -> None:
    verdict = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="LogitGate",
        total_latency_ms=1.0,
        traces=[
            LayerTrace(
                layer_name="LogitGate",
                latency_ms=1.0,
                status=DecisionStatus.FAILED,
                calibrated_prob=0.1,
                forward_log_odds=-2.0,
                reverse_log_odds=-2.2,
                log_odds_pass=-2.1,
                shannon_entropy=0.45,
                positional_bias_magnitude=0.1,
                failure_code=ASTFailureCode.SYNTAX_ERROR,
            )
        ],
    )
    snap = VerdictSnapshot.from_verdict(verdict, candidate="def broken(", rubric="Check syntax")
    assert snap.positional_bias_magnitude == 0.1
    assert snap.log_odds_pass == -2.1
    assert snap.shannon_entropy == 0.45
    assert snap.failure_code == ASTFailureCode.SYNTAX_ERROR

    snap_file = tmp_path / "snap.json"
    snap.save(snap_file)
    assert snap_file.exists()

    loaded_snap = VerdictSnapshot.load(snap_file)
    assert loaded_snap.verdict.status == DecisionStatus.FAILED

    explanation = snap.explain()
    assert isinstance(explanation, Verdict)

    replay_v = snap.replay(backend=StubBackend(default_prob=0.05))
    assert replay_v.passed is False


# ---------------------------------------------------------------------------
# Section 10: Benchmark Statistics & McNemar Significance Unit Tests
# ---------------------------------------------------------------------------


def test_wilson_ci_and_mcnemar_exact_test() -> None:
    low, high = wilson_confidence_interval(0, 100)
    assert low == 0.0 and 0.0 < high < 0.1

    low_full, high_full = wilson_confidence_interval(100, 100)
    assert high_full == 1.0 and 0.9 < low_full < 1.0

    low_mid, high_mid = wilson_confidence_interval(50, 100)
    assert 0.35 < low_mid < 0.5 < high_mid < 0.65

    p_equal = mcnemar_exact_test([True, False], [True, False])
    assert p_equal["p_value"] == 1.0

    p_sig = mcnemar_exact_test([True] * 20 + [False] * 5, [False] * 20 + [False] * 5)
    assert p_sig["p_value"] < 0.01


def test_bootstrap_ci_and_percentile() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == 2.0
    assert _percentile([], 50.0) == 0.0

    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = bootstrap_metric_ci(data, n_resamples=100)
    assert 1.0 <= low <= high <= 5.0

    with pytest.raises(ValueError, match="values cannot be empty"):
        bootstrap_metric_ci([])


def test_calibration_metrics_calculation() -> None:
    probs = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]
    metrics = compute_calibration_metrics(probs, labels, n_bins=5)
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert "auroc" in metrics
    assert metrics["brier_score"] is not None and metrics["brier_score"] < 0.1
    assert metrics["auroc"] == 1.0


def test_run_benchmark_and_report_formatting(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase("Check 1", "def f1(): pass", True),
        BenchmarkCase("Check 2", "broken(", False),
    ]
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([{"rubric": c.rubric, "candidate": c.candidate, "expected": c.expected} for c in cases]), encoding="utf-8")
    loaded_cases, _ = load_cases(str(cases_file))
    assert len(loaded_cases) == 2

    backend = StubBackend(default_prob=0.9)
    report = run_benchmark(
        cases,
        backend=backend,
        model="test-bench-model",
    )
    assert isinstance(report, BenchmarkReport)
    assert report.metadata["dataset"]["cases"] == 2

    write_json(report)
    formatted = format_report(report)
    assert "benchmark | backend=" in formatted
    rep_dict = report.to_dict()
    assert "modes" in rep_dict


# ---------------------------------------------------------------------------
# Section 11: Deep Branch & Boundary Mutant-Killing Tests
# ---------------------------------------------------------------------------


def test_adaptive_controller_min_band_gap_and_profile_branches() -> None:
    prof = CalibrationProfile(model="test-prof-adaptive", fail_threshold=0.2, pass_threshold=0.8)
    ctrl = AdaptiveConformalController(
        profile=prof,
        pass_threshold=0.8,
        fail_threshold=0.2,
        min_band_gap=0.35,
        target_risk_alpha=0.05,
        feedback_mode="cascade_audit",
        audit_sample_rate=0.5,
    )
    prof_out = ctrl.to_profile()
    assert prof_out.model == "test-prof-adaptive"

    ctrl_explicit = AdaptiveConformalController(
        pass_threshold=0.8,
        fail_threshold=0.2,
        min_band_gap=0.45,
        feedback_mode="explicit",
    )
    assert ctrl_explicit.should_audit_fast_path() is False

    with ctrl_explicit._lock:
        ctrl_explicit._theta = -1.5
        _, fail_t, pass_t, _, _, _ = ctrl_explicit._compute_state_unlocked()
        assert math.isclose(pass_t - fail_t, 0.45, abs_tol=1e-4) or pass_t - fail_t >= 0.45

    v_trace_raw = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="L1",
        total_latency_ms=1.0,
        traces=[
            LayerTrace(
                layer_name="L1",
                latency_ms=1.0,
                status=DecisionStatus.PASSED,
                metadata={"raw_l1_status": "passed"},
            )
        ],
    )
    assert ctrl.observe(v_trace_raw, True).raw_error == 0.0

    v_trace_fallback = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        triggered_layer="L1",
        total_latency_ms=1.0,
        traces=[
            LayerTrace(
                layer_name="L1",
                latency_ms=1.0,
                status=DecisionStatus.PASSED,
            )
        ],
    )
    assert ctrl.observe(v_trace_fallback, True).raw_error == 0.0

    v_no_traces = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        triggered_layer="L1",
        total_latency_ms=1.0,
    )
    assert ctrl.observe(v_no_traces, False).raw_error == 0.0


def test_conformal_and_profile_choice_score_calibration() -> None:
    from logitgate.calibration.profile import (
        ChoiceConformalThresholds,
        ScoreConformalThresholds,
        calibrate_choice_thresholds,
        calibrate_score_thresholds,
    )

    with pytest.raises(CalibrationError, match="Option keys must be non-empty strings"):
        calibrate_choice_thresholds([{"": 0.5, "B": 0.5}] * 10, ["B"] * 10)

    with pytest.raises(CalibrationError, match="Option keys must be case-insensitively unique"):
        calibrate_choice_thresholds([{"a": 0.5, "A": 0.5}] * 10, ["A"] * 10)

    with pytest.raises(CalibrationError, match="Option probabilities must be finite floats"):
        calibrate_choice_thresholds([{"A": True, "B": False}] * 10, ["A"] * 10)

    with pytest.raises(CalibrationError, match="Option probabilities must be finite floats"):
        calibrate_choice_thresholds([{"A": float("nan"), "B": 1.0}] * 10, ["A"] * 10)

    with pytest.raises(CalibrationError, match="Option probabilities must sum to ~1.0"):
        calibrate_choice_thresholds([{"A": 0.1, "B": 0.1}] * 10, ["A"] * 10)

    with pytest.raises(CalibrationError, match="Ground-truth choice.*is missing"):
        calibrate_choice_thresholds([{"A": 0.6, "B": 0.4}] * 10, ["Z"] * 10)

    choice_samples = [{"A": 0.7, "B": 0.3}, {"A": 0.2, "B": 0.8}] * 20
    gt_choices = ["A", "B"] * 20
    choice_thresh = calibrate_choice_thresholds(choice_samples, gt_choices, target_risk_alpha=0.05)
    assert isinstance(choice_thresh, ChoiceConformalThresholds)
    assert choice_thresh.min_confidence > 0.5

    with pytest.raises(CalibrationError, match="Calibration requires matching sequences of at least 10 samples"):
        calibrate_score_thresholds([3.0] * 5, [True] * 5)

    with pytest.raises(CalibrationError, match="expected_scores must contain finite floats"):
        calibrate_score_thresholds([True] * 10, [True] * 10)

    with pytest.raises(CalibrationError, match="expected_scores must contain finite floats"):
        calibrate_score_thresholds([10.0] * 10, [True] * 10)

    with pytest.raises(CalibrationError, match="ground_truth_labels must contain booleans"):
        calibrate_score_thresholds([3.0] * 10, ["bad"] * 10)  # type: ignore[list-item]

    scores = [4.5, 4.0, 3.8, 4.2, 4.6, 1.2, 1.5, 2.0, 1.1, 1.8] * 4
    labels = [True, True, True, True, True, False, False, False, False, False] * 4
    score_thresh = calibrate_score_thresholds(scores, labels, min_score=1, max_score=5, target_risk_alpha=0.05)
    assert isinstance(score_thresh, ScoreConformalThresholds)
    assert score_thresh.score_fail_threshold < score_thresh.score_threshold

    neg_high = [0.85] * 10
    pos_low = [0.15] * 10
    cross_thresh = calibrate_cascade_thresholds(pos_low + neg_high, [True] * 10 + [False] * 10, target_risk_alpha=0.1)
    assert cross_thresh.tau_reject <= cross_thresh.tau_accept


@pytest.mark.asyncio
async def test_afit_calibration_profile_extra_branches(tmp_path: Path) -> None:
    backend = StubBackend(default_prob=0.85)

    with pytest.raises(ConfigurationError, match="warm_prefix must be a bool or None"):
        await afit_calibration_profile(backend, warm_prefix=123)  # type: ignore[arg-type]

    prof_warm = await afit_calibration_profile(
        backend,
        model="stub-warm-test",
        samples=[("R1", "C1", True), ("R2", "C2", False)] * 6,
        warm_prefix=True,
        save_cache=True,
        cache_path=tmp_path / "warm_cal.json",
    )
    assert prof_warm.model == "stub-warm-test"


def test_audit_deep_branches(tmp_path: Path) -> None:
    cases = [
        {"rubric": f"R{i}", "candidate": f"short_{i}", "label": True} for i in range(10)
    ] + [
        {"rubric": f"R{i}", "candidate": f"very_long_candidate_content_that_inflates_character_count_{i}", "label": False} for i in range(10)
    ]
    rep = audit_benchmark_dataset(cases)
    assert rep.length_auroc is not None
    rep_d = rep.to_dict()
    assert "source" in rep_d
    assert "warnings" in rep_d

    with pytest.raises(CalibrationError, match="Non-finite dataset label"):
        _parse_bool_label(float("nan"))
    with pytest.raises(CalibrationError, match="Non-finite dataset label"):
        _parse_bool_label(float("inf"))

    assert _parse_bool_label("other_unrecognized") is True

    assert _extract_cases_from_mapping_root({"data": [{"rubric": "R", "candidate": "C", "label": True}]}, "path") != []
    assert _extract_cases_from_mapping_root({"samples": [{"rubric": "R", "candidate": "C", "label": True}]}, "path") != []
    assert _extract_cases_from_mapping_root({"cases": [{"rubric": "R", "candidate": "C", "label": True}]}, "path") != []
    assert _extract_cases_from_mapping_root({"pairwise_items": [{"rubric": "R", "candidate_a": "CA", "candidate_b": "CB", "winner": "A"}]}, "path") != []
    assert _extract_cases_from_mapping_root({"tier_a": [{"rubric": "R", "candidate": "C", "label": True}]}, "path") != []
    assert _extract_cases_from_mapping_root({"tier_b": [{"candidate": "C", "label": True}], "tier_b_compound_rubric": "Fallback R"}, "path") != []
    assert _extract_cases_from_mapping_root({"choice_gate_items": [{"question": "Q", "options": {"A": "1", "B": "2"}, "candidate": "C", "expected_choice": "A"}]}, "path") != []
    assert _extract_cases_from_mapping_root({"score_gate_items": [{"rubric": "R", "candidate": "C", "expected_score": 4}]}, "path") != []

    non_sym = tmp_path / "real.json"
    non_sym.write_text("[]", encoding="utf-8")
    sym = tmp_path / "sym.json"
    sym.symlink_to(non_sym)
    with pytest.raises(CalibrationError, match="Refusing to audit dataset through symlink"):
        _load_normalized_cases(sym)

    non_existent = tmp_path / "missing_data.json"
    with pytest.raises(FileNotFoundError):
        _load_normalized_cases(non_existent)


def test_ast_gate_deep_branches() -> None:
    from logitgate.gates.ast_gate import (
        _detect_dag_cycle,
        _extract_json_with_span,
        normalize_candidate,
    )

    assert normalize_candidate(123) == 123
    assert normalize_candidate("<think>some reasoning</think>\ndef func(): pass") == "def func(): pass"
    assert normalize_candidate("```python\ndef func(): pass\n```") == "def func(): pass"
    assert normalize_candidate("```\ninline block\n```") == "inline block"

    obj, span = _extract_json_with_span('Prefix text {"key": "val"} trailing', prefer_object=True)
    assert obj == {"key": "val"}
    assert '{"key": "val"}' in span

    assert "Expected list of tasks" in (_detect_dag_cycle("not a list") or "")  # type: ignore[arg-type]
    assert "Expected task node to be dict" in (_detect_dag_cycle(["not dict"]) or "")  # type: ignore[list-item]
    assert "missing required 'id'" in (_detect_dag_cycle([{"name": "task"}]) or "")
    assert "Duplicate task ID" in (_detect_dag_cycle([{"id": "A"}, {"id": "A"}]) or "")
    assert "dependencies must be a list" in (_detect_dag_cycle([{"id": "A", "dependencies": "not-list"}]) or "")

    ast_custom_fail = ASTGate(validators=[lambda cand: False])
    v_cf = ast_custom_fail.evaluate("candidate")
    assert v_cf.passed is False
    assert v_cf.traces[0].failure_code == ASTFailureCode.CUSTOM_VALIDATOR_FAILED

    ast_custom_tuple = ASTGate(validators=[lambda cand: (False, "Specific custom failure reason")])
    v_ct = ast_custom_tuple.evaluate("candidate")
    assert v_ct.passed is False
    assert "Specific custom failure reason" in (v_ct.traces[0].diagnostic_report or "")

    def _broken_validator(cand: Any) -> Any:
        raise ValueError("Boom")

    ast_custom_err = ASTGate(validators=[_broken_validator])
    v_ce = ast_custom_err.evaluate("candidate")
    assert v_ce.passed is False

    ast_norm_callable = ASTGate(normalize=lambda c: str(c).strip().lower(), required_sections=["lowercase"])
    assert ast_norm_callable.evaluate("  LOWERCASE  ").passed is True

    dict_schema = {
        "type": "object",
        "properties": {"val": {"type": "integer"}},
        "required": ["val"],
    }
    ast_dict_sch = ASTGate(json_schema=dict_schema)
    assert ast_dict_sch.evaluate('{"val": 42}').passed is True
    assert ast_dict_sch.evaluate('{"val": "not_int"}').passed is False


@pytest.mark.asyncio
async def test_ast_gate_async_custom_validator() -> None:
    def _sync_val(cand: Any) -> bool:
        return "pass" in str(cand)

    ast_async = ASTGate(validators=[_sync_val])
    v_ok = await ast_async.aevaluate("pass this")
    assert v_ok.passed is True
    v_no = await ast_async.aevaluate("fail this")
    assert v_no.passed is False


def test_judge_gate_deep_branches() -> None:
    from logitgate.gates.judge_gate import (
        _coerce_pass,
        _is_valid_verdict_value,
        _keyword_signal,
    )

    assert _is_valid_verdict_value(True) is True
    assert _is_valid_verdict_value(False) is True
    assert _is_valid_verdict_value(1) is True
    assert _is_valid_verdict_value(0) is True
    assert _is_valid_verdict_value(2) is False
    assert _is_valid_verdict_value("pass") is True
    assert _is_valid_verdict_value("invalid_word") is False

    assert _coerce_pass("pass") is True
    assert _coerce_pass("fail") is False
    assert _coerce_pass(1) is True
    assert _coerce_pass(0) is False

    assert _keyword_signal("There are no defects and no failure found.", tie_break=False) is None

    jg_no_debias = JudgeGate(rubric="R", symmetric_debiasing=False, backend=StubBackend(diagnostic_text="PASS"))
    assert jg_no_debias._can_debias_cot() is False

    jg_short_tokens = JudgeGate(rubric="R", max_tokens=10, backend=StubBackend(diagnostic_text="PASS"))
    assert jg_short_tokens._can_debias_cot() is False

    class _GenerativeOnlyBackend(GenerativeBackend):
        def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
            return "critique", 10
        async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
            return "critique", 10

    jg_gen_only = JudgeGate(rubric="R", backend=_GenerativeOnlyBackend())
    assert jg_gen_only._can_debias_cot() is False

    fwd_lps = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.3}]
    rev_lps = [{"token": "A", "logprob": -2.1}, {"token": "B", "logprob": -0.2}]
    jg_cot = JudgeGate(
        rubric="R",
        backend=StubBackend(
            diagnostic_text="Analysis shows clean code.\nVerdict: PASS",
            forward_top_logprobs=fwd_lps,
            reverse_top_logprobs=rev_lps,
        ),
    )
    assert jg_cot._can_debias_cot() is True
    cot_res = jg_cot._cot_verdict_from_logprobs(fwd_lps, rev_lps)
    assert cot_res is not None
    parsed_v, telemetry = cot_res
    assert parsed_v[0] is True
    assert "calibrated_prob" in telemetry

    assert jg_cot._cot_verdict_from_logprobs([{"token": "Z", "logprob": -0.1}], rev_lps) is None

    class _FailingGenBackend(GenerativeBackend):
        def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
            raise BackendError("Generative timeout")
        async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
            raise BackendError("Generative timeout")

    jg_failing = JudgeGate(rubric="R", backend=_FailingGenBackend())
    with pytest.raises(BackendError, match="Generative timeout"):
        jg_failing.evaluate("candidate")

    jg_override = JudgeGate(rubric="R", allow_fail_override=True, backend=StubBackend(diagnostic_text="[VERDICT: PASS]"))
    v_ovr = jg_override.evaluate("candidate")
    assert v_ovr.passed is True


@pytest.mark.asyncio
async def test_judge_gate_async_deep_branches() -> None:
    fwd_lps = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.3}]
    rev_lps = [{"token": "A", "logprob": -2.1}, {"token": "B", "logprob": -0.2}]
    jg_async = JudgeGate(
        rubric="R",
        allow_fail_override=True,
        backend=StubBackend(
            diagnostic_text="Step 1: Check constraints.\nStep 2: Meets all requirements.\n[VERDICT: PASS]",
            forward_top_logprobs=fwd_lps,
            reverse_top_logprobs=rev_lps,
        ),
    )
    v = await jg_async.aevaluate("candidate")
    assert v.passed is True


def test_logit_gate_deep_branches() -> None:
    lg_single = LogitGate(
        rubric="Single pass check",
        symmetric_debiasing=False,
        pass_threshold=0.7,
        backend=StubBackend(default_prob=0.85),
    )
    assert lg_single.symmetric_debiasing is False
    v_single = lg_single.evaluate("candidate")
    assert v_single.passed is True

    lg_warm = LogitGate(
        rubric="Warm check",
        warm_prefix=True,
        backend=StubBackend(default_prob=0.9),
    )
    assert lg_warm.warm_prefix is True
    assert lg_warm.evaluate("candidate").passed is True

    ctrl = AdaptiveConformalController(pass_threshold=0.8, fail_threshold=0.2)
    lg_adaptive = LogitGate(
        rubric="Adaptive check",
        adaptive_controller=ctrl,
        backend=StubBackend(default_prob=0.9),
    )
    assert lg_adaptive.adaptive_controller is ctrl
    v_adapt = lg_adaptive.evaluate("candidate")
    assert v_adapt.passed is True

    cg_rev = ChoiceGate(
        question="Which?",
        options={"A": "Alpha", "B": "Beta"},
        permutation_mode="reverse",
        backend=StubBackend(default_prob=0.9),
    )
    p_cg_rev = cg_rev._choice_prompts_list("cand", boundary_nonce="cg_nonce")
    assert len(p_cg_rev) == 2

    sg_std = ScoreGate(
        rubric="Rate quality",
        min_score=1,
        max_score=5,
        fail_threshold=2.0,
        max_score_std=0.01,
        backend=StubBackend(default_prob=0.7),
    )
    v_std = sg_std.evaluate("cand")
    assert v_std.status in (DecisionStatus.AMBIGUOUS, DecisionStatus.PASSED)


def test_rank_gate_deep_branches() -> None:
    rg_pairwise_single = RankGate(
        rubric="Pairwise single pass",
        mode="pairwise",
        symmetric_debiasing=False,
        backend=StubBackend(default_prob=0.9),
    )
    v_pw = rg_pairwise_single.evaluate({"A": "cand 1", "B": "cand 2"})
    assert v_pw.passed is True

    candidates_odd = {
        "A": "cand 1",
        "B": "cand 2",
        "C": "cand 3",
        "D": "cand 4",
        "E": "cand 5",
    }
    rg_tourn_odd = RankGate(
        rubric="Tournament odd",
        mode="tournament",
        backend=StubBackend(default_prob=0.9),
    )
    v_tourn_odd = rg_tourn_odd.evaluate(candidates_odd)
    assert v_tourn_odd.passed is True

    rg_pos_bias = RankGate(
        rubric="Bias check",
        mode="round_robin",
        max_positional_bias=0.0001,
        backend=StubBackend(default_prob=0.9),
    )
    v_bias = rg_pos_bias.evaluate({"A": "cand 1", "B": "cand 2", "C": "cand 3"})
    assert v_bias.status in (DecisionStatus.AMBIGUOUS, DecisionStatus.PASSED)

    rg_min_conf = RankGate(
        rubric="Conf check",
        mode="round_robin",
        min_confidence=0.9999,
        backend=StubBackend(default_prob=0.9),
    )
    v_conf = rg_min_conf.evaluate({"A": "cand 1", "B": "cand 2", "C": "cand 3"})
    assert v_conf.status == DecisionStatus.AMBIGUOUS

    bon_gates = BestOfNGate(
        rubric="Quality",
        pre_gate=ASTGate(syntax=True),
        post_gate=JudgeGate(backend=StubBackend(diagnostic_text="[VERDICT: PASS]")),
        backend=StubBackend(default_prob=0.9),
    )
    v_bon_full = bon_gates.evaluate({"A": "def f1(): pass", "B": "def f2(): pass"})
    assert v_bon_full.passed is True

    v_bon_all_fail = bon_gates.evaluate({"A": "broken(:", "B": "broken(:"})
    assert v_bon_all_fail.passed is False


@pytest.mark.asyncio
async def test_rank_gate_async_deep_branches() -> None:
    rg_listwise = RankGate(rubric="R", mode="listwise", backend=StubBackend(default_prob=0.9))
    v = await rg_listwise.aevaluate({"A": "c1", "B": "c2", "C": "c3"})
    assert v.passed is True

    rg_tourn = RankGate(rubric="R", mode="tournament", backend=StubBackend(default_prob=0.9))
    v_tourn = await rg_tourn.aevaluate({"A": "c1", "B": "c2", "C": "c3", "D": "c4"})
    assert v_tourn.passed is True


def test_cascade_deep_branches() -> None:
    from logitgate.cascade import _verdict_fields

    with pytest.raises(ConfigurationError, match="prior_weight must be between 0.0 and 1.0"):
        fuse_escalated_verdict(None, Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="L", total_latency_ms=1.0), prior_weight=1.5)

    v_base = Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="TestLayer", total_latency_ms=5.0)
    fields = _verdict_fields(v_base, extra="override")
    assert fields["passed"] is True
    assert fields["extra"] == "override"

    assert _select_upstream_logit_trace(None) is None
    assert _select_upstream_logit_trace(Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="L", total_latency_ms=1.0, traces=[])) is None

    tr_no_odds = LayerTrace(layer_name="L1", latency_ms=1.0, status=DecisionStatus.PASSED)
    v_no_odds = Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="L", total_latency_ms=1.0, traces=[tr_no_odds])
    assert _select_upstream_logit_trace(v_no_odds) is tr_no_odds

    ctrl = AdaptiveConformalController(pass_threshold=0.8, fail_threshold=0.2, feedback_mode="cascade_audit", audit_sample_rate=0.5)
    lg = LogitGate(rubric="R", adaptive_controller=ctrl, backend=StubBackend(default_prob=0.9))
    assert _has_active_cascade_audit([lg]) is True
    assert _has_active_cascade_audit([ASTGate(syntax=True)]) is False


def test_base_gate_stream_and_conditional_edges() -> None:
    g = ASTGate(syntax=True)
    edge = g.as_conditional_edge(pass_node="pass_node", fail_node="fail_node")
    assert edge({"content": "def good(): pass"}) == "pass_node"
    assert edge({"content": "broken("}) == "fail_node"

    guarded = g.guard_stream(["def ", "func():\n", "    return 42"])
    chunks = list(guarded)
    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# Section 12: Comprehensive Boundary Coverage Push
# ---------------------------------------------------------------------------


def test_logit_gate_and_choice_score_validation_and_features() -> None:
    from logitgate.gates.logit_gate import (
        _cal_lock_key,
        _get_sync_cal_lock,
        _resolve_logit_temperature,
    )

    backend = StubBackend(default_prob=0.8)
    key = _cal_lock_key("test-model")
    assert key[1] == "test-model"
    assert _get_sync_cal_lock(key) is not None

    with pytest.raises(ConfigurationError, match="logit_temperature must be a positive float"):
        _resolve_logit_temperature(True, "model", backend)

    with pytest.raises(ConfigurationError, match="logit_temperature must be positive"):
        _resolve_logit_temperature(-2.0, "model", backend)

    t_auto_default = _resolve_logit_temperature("auto", "default", backend)
    assert t_auto_default == 1.0

    t_auto_known = _resolve_logit_temperature("auto", "qwen2.5-coder-32b", backend, binary=True)
    assert t_auto_known > 0.0

    # Test presets on LogitGate
    lg_strict = LogitGate(rubric="R", preset="strict", backend=backend)
    assert lg_strict.pass_threshold == 0.90
    assert lg_strict.fail_threshold == 0.25

    lg_balanced = LogitGate(rubric="R", preset="balanced", backend=backend)
    assert lg_balanced.pass_threshold == 0.85
    assert lg_balanced.fail_threshold == 0.20

    lg_fast = LogitGate(rubric="R", preset="fast_triage", backend=backend)
    assert lg_fast.symmetric_debiasing is False

    with pytest.raises(ConfigurationError, match="Invalid preset"):
        LogitGate(rubric="R", preset="unknown", backend=backend)  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="max_positional_bias must be a positive float"):
        LogitGate(rubric="R", max_positional_bias=True, backend=backend)

    with pytest.raises(ConfigurationError, match="max_positional_bias must be a positive float"):
        LogitGate(rubric="R", max_positional_bias=-1.0, backend=backend)

    # ChoiceGate validation
    with pytest.raises(ConfigurationError, match="question cannot be empty or whitespace"):
        ChoiceGate(question="   ", options={"A": "1", "B": "2"}, backend=backend)

    with pytest.raises(ConfigurationError, match="option keys must be single characters"):
        ChoiceGate(question="Q", options={"AB": "1", "C": "2"}, backend=backend)

    with pytest.raises(ConfigurationError, match="option description for 'A' must be a non-empty string"):
        ChoiceGate(question="Q", options={"A": "", "B": "2"}, backend=backend)

    with pytest.raises(ConfigurationError, match="option keys must be case-insensitively unique"):
        ChoiceGate(question="Q", options={"a": "1", "A": "2"}, backend=backend)

    # ScoreGate validation
    with pytest.raises(ConfigurationError, match="rubric cannot be empty or whitespace"):
        ScoreGate(rubric="   ", backend=backend)

    with pytest.raises(ConfigurationError, match="min_score and max_score must be integers"):
        ScoreGate(rubric="R", min_score=True, backend=backend)

    with pytest.raises(ConfigurationError, match="ScoreGate requires single-digit integer bounds"):
        ScoreGate(rubric="R", min_score=-1, max_score=5, backend=backend)

    with pytest.raises(ConfigurationError, match=r"threshold \(6\.0\) must be between min_score"):
        ScoreGate(rubric="R", min_score=1, max_score=5, threshold=6.0, backend=backend)

    with pytest.raises(ConfigurationError, match="fail_threshold .* must be between min_score"):
        ScoreGate(rubric="R", min_score=1, max_score=5, fail_threshold=0.5, backend=backend)

    with pytest.raises(ConfigurationError, match="fail_threshold .* must be between min_score"):
        ScoreGate(rubric="R", min_score=1, max_score=5, fail_threshold=4.0, backend=backend)

    # MultiRubricGate aevaluate_batch error branches
    lg_batch = LogitGate(rubric="R", backend=backend)
    with pytest.raises(ConfigurationError, match="fail_fast must be a bool"):
        asyncio.run(lg_batch.aevaluate_batch("cand", ["R1"], fail_fast="not_bool"))  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="rubrics must be a sequence of rubric strings"):
        asyncio.run(lg_batch.aevaluate_batch("cand", "not_seq"))

    with pytest.raises(ConfigurationError, match="Each rubric in rubrics must be a non-empty string"):
        asyncio.run(lg_batch.aevaluate_batch("cand", [""]))

    assert asyncio.run(lg_batch.aevaluate_batch("cand", [])) == []


@pytest.mark.asyncio
async def test_logit_gate_aevaluate_batch_modes() -> None:
    backend_fail = StubBackend(default_prob=0.1)
    lg = LogitGate(rubric="R", pass_threshold=0.8, backend=backend_fail)

    # fail_fast with first failed
    res_ff = await lg.aevaluate_batch("cand", ["R1", "R2", "R3"], fail_fast=True)
    assert len(res_ff) == 1
    assert res_ff[0].status == DecisionStatus.FAILED

    backend_pass = StubBackend(default_prob=0.9)
    lg_pass = LogitGate(rubric="R", pass_threshold=0.8, backend=backend_pass)
    res_all = await lg_pass.aevaluate_batch("cand", ["R1", "R2", "R3"], fail_fast=False)
    assert len(res_all) == 3


@pytest.mark.asyncio
async def test_score_gate_aevaluate_and_cache(tmp_path: Path) -> None:
    from logitgate.cache import VerdictCache

    cache = VerdictCache()
    sg = ScoreGate(
        rubric="Rate 1 to 5",
        min_score=1,
        max_score=5,
        cache=cache,
        backend=StubBackend(default_prob=0.85),
    )
    v1 = await sg.aevaluate("cand")
    assert v1.passed is True
    # Cache hit
    v2 = await sg.aevaluate("cand")
    assert v2.passed is True

    # Synchronous cache hit
    v_sync = sg.evaluate("cand")
    assert v_sync.passed is True


@pytest.mark.asyncio
async def test_multirubric_gate_cache_and_properties() -> None:
    from logitgate.cache import VerdictCache

    cache = VerdictCache()
    mrg = MultiRubricGate(
        rubrics=["R1", "R2"],
        pass_threshold=0.7,
        fail_threshold=0.2,
        cache=cache,
        backend=StubBackend(default_prob=0.9),
    )
    assert mrg.rubric == "- R1\n- R2"
    assert mrg.pass_threshold == 0.7
    assert mrg.fail_threshold == 0.2
    assert mrg.symmetric_debiasing is True
    assert mrg.backend is not None
    assert mrg.logit_temperature > 0.0

    v1 = mrg.evaluate("cand")
    assert v1.passed is True
    v_hit = mrg.evaluate("cand")
    assert v_hit.passed is True

    v_async1 = await mrg.aevaluate("cand_async")
    assert v_async1.passed is True
    v_async_hit = await mrg.aevaluate("cand_async")
    assert v_async_hit.passed is True


def test_judge_gate_max_attempts_and_structured_verdict() -> None:
    # 130 invalid JSON braces to hit attempts >= _MAX_JSON_DECODE_ATTEMPTS
    text_bad_braces = "{" * 135
    assert list(_iter_json_objects(text_bad_braces)) == []

    # Negated unbracketed tags
    v_neg_unbracketed = _extract_structured_verdict("This is not Verdict: PASS")
    assert v_neg_unbracketed is None

    # Quoted unbracketed tags
    v_quoted_unbracketed = _extract_structured_verdict('He said "Verdict: PASS" in quote.')
    assert v_quoted_unbracketed is None

    # JSON with non-finite score falls back to 1.0 or 0.0
    v_json_nan = _extract_structured_verdict('```json\n{"verdict": "pass", "score": "not_finite"}\n```')
    assert v_json_nan is not None and v_json_nan[0] is True and v_json_nan[2] == 1.0


def test_ast_gate_additional_validators_and_options() -> None:
    with pytest.raises(ConfigurationError, match="min_length must be a non-negative integer"):
        ASTGate(min_length=-1)

    with pytest.raises(ConfigurationError, match="min_length .* cannot exceed max_length"):
        ASTGate(min_length=10, max_length=5)

    ast_invalid_regex = ASTGate(forbidden_patterns=["[invalid_regex*"])
    assert ast_invalid_regex._forbidden_res[0][1] is None

    ast_len_ok = ASTGate(min_length=3, max_length=20)
    assert ast_len_ok.evaluate("hello world").passed is True

    # 3-node cycle in detect_dag_cycle
    from logitgate.gates.ast_gate import _detect_dag_cycle

    cycle_3 = [
        {"id": "A", "dependencies": ["C"]},
        {"id": "B", "dependencies": ["A"]},
        {"id": "C", "dependencies": ["B"]},
    ]
    assert "Cycle detected" in (_detect_dag_cycle(cycle_3) or "")


def test_rank_gate_validation_and_tournament_modes() -> None:
    with pytest.raises(ConfigurationError, match="Invalid mode 'invalid_mode'"):
        RankGate(rubric="R", mode="invalid_mode")  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="rubric cannot be empty or whitespace"):
        RankGate(rubric="   ")

    with pytest.raises(ConfigurationError, match="context must be a string or None"):
        RankGate(rubric="R", context=123)  # type: ignore[arg-type]

    rg = RankGate(rubric="R", backend=StubBackend(default_prob=0.9))

    with pytest.raises(ConfigurationError, match="candidates must be a sequence or mapping of >= 2"):
        rg.evaluate("single string")

    with pytest.raises(ConfigurationError, match="Ranking requires at least 2 candidates"):
        rg.evaluate(["only one"])

    with pytest.raises(ConfigurationError, match="Mapping candidate keys must be unique after stripping"):
        rg.evaluate({"A": "c1", " A ": "c2"})

    # Tournament with 7 candidates (odd tree size)
    cands_7 = {str(i): f"candidate {i}" for i in range(7)}
    v_tourn_7 = rg.evaluate(cands_7)
    assert v_tourn_7.passed is True


def test_permutation_deep_edge_cases() -> None:
    from logitgate.debiasing.permutation import _cache_key_for_path

    class _FailingResolvePath:
        def resolve(self) -> Path:
            raise OSError("Access denied")
        def __str__(self) -> str:
            return "/fallback/path"

    assert _cache_key_for_path(_FailingResolvePath()) == "/fallback/path"  # type: ignore[arg-type]

    # _extract_temp_from_entry with binary=False and key != "multiclass_temperature"
    entry_bin_false = {"binary_temperature": 0.3}
    assert _extract_temp_from_entry(entry_bin_false, binary=False) == 0.65

    # _lookup_temperature_in_mapping prefix match
    mapping = {"qwen2.5-coder-long-prefix-name": 1.75}
    assert _lookup_temperature_in_mapping("qwen2.5-coder-long-prefix-name-v2", mapping) == 1.75

    # fit_calibration with max_abs_lo <= 1e-9
    with pytest.warns(CalibrationWarning, match="received all zero symmetric log-odds"):
        res_zero = fit_calibration([(0.0, 0.0, 1), (0.0, 0.0, 0)] * 3)
    assert res_zero.optimal_temperature == 1.0


# ---------------------------------------------------------------------------
# Section 13: Final Frontier Full-Module Coverage Push
# ---------------------------------------------------------------------------


def test_ast_gate_schema_and_dynamic_imports_full(tmp_path: Path) -> None:
    from logitgate.cache import VerdictCache

    # 1. Dynamic imports in ASTGate
    code_dyn = """
import os
__import__('subprocess')
from importlib import import_module
import_module('sys')
import_module(name='math')
"""
    ast_dyn = ASTGate(forbidden_imports=["sys", "math"])
    v_dyn = ast_dyn.evaluate(code_dyn)
    assert v_dyn.passed is False
    assert v_dyn.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT

    # Import statements syntax regex
    code_mult_imp = "import a, b.c, d\nfrom pkg.sub import x, y, *"
    found = ast_dyn._extract_imports_from_source(code_mult_imp)
    assert "a" in found and "b.c" in found and "pkg.sub" in found

    # 2. Syntax error with null bytes (ValueError in ast.parse when syntax=True)
    ast_syn = ASTGate(syntax=True)
    v_null = ast_syn.evaluate("def f():\x00 pass")
    assert v_null.passed is False
    assert v_null.traces[0].failure_code == ASTFailureCode.SYNTAX_ERROR

    # 3. JSON schema array checking
    ast_arr = ASTGate(json_schema={"type": "array", "items": {"type": "integer"}})
    assert ast_arr.evaluate([1, 2, 3]).passed is True
    v_arr_bad = ast_arr.evaluate([1, "bad", 3])
    assert v_arr_bad.passed is False
    assert v_arr_bad.traces[0].failure_code == ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR

    v_arr_type = ast_arr.evaluate({"not": "array"})
    assert v_arr_type.passed is False
    assert v_arr_type.traces[0].failure_code == ASTFailureCode.JSON_TYPE_ERROR

    # 4. JSON schema property checking & required keys
    ast_props = ASTGate(json_schema={
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name"],
    })
    assert ast_props.evaluate('{"name": "test", "count": 10}').passed is True
    v_miss_req = ast_props.evaluate('{"count": 10}')
    assert v_miss_req.passed is False
    assert v_miss_req.traces[0].failure_code == ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR

    v_bad_prop = ast_props.evaluate('{"name": "test", "count": "not_int"}')
    assert v_bad_prop.passed is False
    assert v_bad_prop.traces[0].failure_code == ASTFailureCode.JSON_SCHEMA_VALIDATION_ERROR

    # 5. Non-standard schema dictionary (flat mapping)
    ast_flat = ASTGate(json_schema={"age": int, "active": bool})
    assert ast_flat.evaluate('{"age": 30, "active": true}').passed is True
    v_flat_miss = ast_flat.evaluate('{"age": 30}')
    assert v_flat_miss.passed is False
    v_flat_type = ast_flat.evaluate('{"age": "thirty", "active": true}')
    assert v_flat_type.passed is False

    # 6. ASTGate cache hit
    cache = VerdictCache()
    ast_cached = ASTGate(syntax=True, cache=cache)
    v_c1 = ast_cached.evaluate("def ok(): pass")
    assert v_c1.passed is True
    v_c2 = ast_cached.evaluate("def ok(): pass")
    assert v_c2.passed is True

    v_c_fail1 = ast_cached.evaluate("def broken(:")
    assert v_c_fail1.passed is False
    v_c_fail2 = ast_cached.evaluate("def broken(:")
    assert v_c_fail2.passed is False

    # 7. ASTGate normalize callable exception
    def _exploding_norm(c: Any) -> Any:
        raise RuntimeError("Normalize explosion")

    ast_exp_norm = ASTGate(normalize=_exploding_norm)
    v_norm_err = ast_exp_norm.evaluate("cand")
    assert v_norm_err.passed is False


def test_judge_gate_self_healing_probe_and_cache() -> None:
    from logitgate.cache import VerdictCache

    # 1. Self-healing probe on free-form critique (no [VERDICT] tag)
    fwd_lps = [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.5}]
    rev_lps = [{"token": "A", "logprob": -3.2}, {"token": "B", "logprob": -0.1}]
    jg_probe = JudgeGate(
        rubric="Quality check",
        trigger="always",
        backend=StubBackend(
            diagnostic_text="The code satisfies every single requirement in the rubric with high quality.",
            logprob_responses=[fwd_lps, rev_lps],
        ),
    )
    v_probed = jg_probe.evaluate("candidate")
    assert v_probed.passed is True

    # 2. JudgeGate cache hit in evaluate and aevaluate
    cache = VerdictCache()
    jg_cached = JudgeGate(
        rubric="Quality check",
        trigger="always",
        cache=cache,
        backend=StubBackend(diagnostic_text="[VERDICT: PASS]"),
    )
    v1 = jg_cached.evaluate("candidate text")
    assert v1.passed is True
    v_hit = jg_cached.evaluate("candidate text")
    assert v_hit.passed is True


@pytest.mark.asyncio
async def test_judge_gate_async_probe_and_cache() -> None:
    from logitgate.cache import VerdictCache

    fwd_lps = [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.5}]
    rev_lps = [{"token": "A", "logprob": -3.2}, {"token": "B", "logprob": -0.1}]
    jg_async_probe = JudgeGate(
        rubric="Quality check",
        trigger="always",
        backend=StubBackend(
            diagnostic_text="Clean code with zero issues.",
            logprob_responses=[fwd_lps, rev_lps],
        ),
    )
    v_async_probed = await jg_async_probe.aevaluate("async candidate")
    assert v_async_probed.passed is True

    cache = VerdictCache()
    jg_async_cache = JudgeGate(
        rubric="Quality check",
        trigger="always",
        cache=cache,
        backend=StubBackend(diagnostic_text="[VERDICT: PASS]"),
    )
    v1 = await jg_async_cache.aevaluate("async candidate text")
    assert v1.passed is True
    v_hit = await jg_async_cache.aevaluate("async candidate text")
    assert v_hit.passed is True


def test_audit_dataset_and_split_leakage_full(tmp_path: Path) -> None:
    # 1. DatasetAuditReport format_summary when pos_mean is 0
    all_fail_cases = [
        {"rubric": f"R{i}", "candidate": f"broken_code_{i}", "label": False}
        for i in range(12)
    ]
    rep_all_fail = audit_benchmark_dataset(all_fail_cases)
    formatted_fail = rep_all_fail.format_summary()
    assert "0.0 chars" in formatted_fail

    # 2. Dataset with duplicate candidates, quad variants, and domains
    cases_rich = [
        {"rubric": f"R{i}", "candidate": "identical_candidate", "label": True, "domain": "math"}
        for i in range(10)
    ] + [
        {"rubric": f"R{i}", "candidate": "different_candidate", "label": False, "domain": "code"}
        for i in range(10)
    ]
    rep_rich = audit_benchmark_dataset(cases_rich)
    assert rep_rich.duplicate_candidates > 0
    assert "math" in rep_rich.domains
    assert "code" in rep_rich.domains

    # 3. SplitLeakageReport with exact rubric overlap and shared domains
    train_cases = [
        {"rubric": "Shared Rubric", "candidate": "Candidate A", "label": True, "domain": "finance"},
        {"rubric": "Unique Train", "candidate": "Candidate B", "label": False, "domain": "medical"},
    ]
    test_cases = [
        {"rubric": "Shared Rubric", "candidate": "Candidate A slightly modified", "label": True, "domain": "finance"},
        {"rubric": "Unique Test", "candidate": "Candidate C", "label": False, "domain": "code"},
    ]
    rep_leak = check_split_leakage(train_cases, test_cases)
    assert rep_leak.exact_rubric_overlaps == 1
    assert "finance" in rep_leak.shared_domains
    leak_summ = rep_leak.format_summary()
    assert "share domain tag(s): finance" in leak_summ

    leak_d = rep_leak.to_dict()
    assert "exact_rubric_overlaps" in leak_d
    assert "shared_domains" in leak_d

    # 4. JSONL file parsing with empty lines
    p_jsonl = tmp_path / "valid_audit.jsonl"
    p_jsonl.write_text(
        '{"rubric": "R1", "candidate": "C1", "label": true}\n\n{"rubric": "R2", "candidate": "C2", "label": false}\n',
        encoding="utf-8",
    )
    cases_jsonl, _ = _load_normalized_cases(p_jsonl)
    assert len(cases_jsonl) == 2

    # 5. Invalid JSON file
    p_bad_json = tmp_path / "bad_syntax.json"
    p_bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _load_normalized_cases(p_bad_json)


def test_conformal_probs_string_conversion_and_files(tmp_path: Path) -> None:
    # 1. calibrated_probs with numeric floats
    thresh = calibrate_cascade_thresholds([0.85] * 5 + [0.15] * 5, [True] * 5 + [False] * 5, target_risk_alpha=0.1)
    assert thresh.tau_reject <= thresh.tau_accept

    # 2. String probabilities conversion succeeds and matches numeric floats
    thresh_str = calibrate_cascade_thresholds(["0.85"] * 5 + ["0.15"] * 5, [True] * 5 + [False] * 5, target_risk_alpha=0.1)  # type: ignore[arg-type]
    assert thresh_str == thresh

    # 2. calibrated_probs with invalid string
    with pytest.raises(CalibrationError, match="must contain finite probabilities"):
        calibrate_cascade_thresholds(["invalid_float"] * 10, [True] * 5 + [False] * 5)  # type: ignore[list-item]

    # 3. File without extension containing JSON
    p_no_ext = tmp_path / "no_extension_dataset"
    p_no_ext.write_text(json.dumps([{"rubric": "R", "candidate": "C", "label": True}] * 10), encoding="utf-8")
    loaded = load_calibration_dataset(p_no_ext)
    assert len(loaded) == 10


def test_rank_gate_tournament_and_best_of_n_deep() -> None:
    # 1. Tournament mode with 4 candidates (power of 2)
    rg_4 = RankGate(rubric="Code quality", mode="tournament", backend=StubBackend(default_prob=0.9))
    v_4 = rg_4.evaluate({"0": "c0", "1": "c1", "2": "c2", "3": "c3"})
    assert v_4.passed is True

    # 2. BestOfNGate with post_gate failing
    jg_failing = JudgeGate(rubric="Strict check", backend=StubBackend(diagnostic_text="[VERDICT: FAIL]"))
    bon_post_fail = BestOfNGate(
        rubric="Quality",
        post_gate=jg_failing,
        backend=StubBackend(default_prob=0.9),
    )
    v_post_fail = bon_post_fail.evaluate({"A": "cand 1", "B": "cand 2"})
    assert v_post_fail.passed is False


@pytest.mark.asyncio
async def test_rank_gate_async_tournament_power_of_two() -> None:
    rg_4 = RankGate(rubric="Code quality", mode="tournament", backend=StubBackend(default_prob=0.9))
    v = await rg_4.aevaluate({"0": "c0", "1": "c1", "2": "c2", "3": "c3"})
    assert v.passed is True


# ---------------------------------------------------------------------------
# Section 14: Final 93%+ Coverage Surge Tests
# ---------------------------------------------------------------------------


def test_logit_gate_lazy_and_meta_calibrator_branches() -> None:
    from logitgate.gates.logit_gate import (
        _should_lazy_auto_calibrate,
    )

    backend = StubBackend(default_prob=0.8)
    assert _should_lazy_auto_calibrate(1.0, "any-model", backend) is False
    assert _should_lazy_auto_calibrate(None, "default", backend) is False
    assert _should_lazy_auto_calibrate(None, "custom-model", "not-a-backend") is False
    assert _should_lazy_auto_calibrate(None, "custom-model", backend) is False

    # ChoiceGate min_confidence and min_margin AMBIGUOUS branches
    cg_strict = ChoiceGate(
        question="Which?",
        options={"A": "Alpha", "B": "Beta"},
        min_confidence=0.9999,
        backend=StubBackend(default_prob=0.7),
    )
    v_cg_conf = cg_strict.evaluate("cand")
    assert v_cg_conf.status == DecisionStatus.AMBIGUOUS
    assert "min_confidence" in (v_cg_conf.reason or "")

    cg_margin = ChoiceGate(
        question="Which?",
        options={"A": "Alpha", "B": "Beta"},
        min_margin=0.9999,
        backend=StubBackend(default_prob=0.7),
    )
    v_cg_marg = cg_margin.evaluate("cand")
    assert v_cg_marg.status == DecisionStatus.AMBIGUOUS
    assert "min_margin" in (v_cg_marg.reason or "")

    # ChoiceGate missing tokens
    class _EmptyChoiceBackend(StubBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"token": "Z", "logprob": -0.1}]
        async def aquery_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"token": "Z", "logprob": -0.1}]

    cg_empty = ChoiceGate(
        question="Which?",
        options={"A": "Alpha", "B": "Beta"},
        backend=_EmptyChoiceBackend(),
    )
    v_cg_empty = cg_empty.evaluate("cand")
    assert v_cg_empty.status == DecisionStatus.AMBIGUOUS

    # ScoreGate missing tokens
    sg_empty = ScoreGate(
        rubric="R",
        min_score=1,
        max_score=5,
        backend=_EmptyChoiceBackend(),
    )
    v_sg_empty = sg_empty.evaluate("cand")
    assert v_sg_empty.status == DecisionStatus.AMBIGUOUS
    assert "unobserved_target_tokens" in (v_sg_empty.reason or "")

    # MultiRubricGate fail_fast validation
    with pytest.raises(ConfigurationError, match="fail_fast must be a bool"):
        MultiRubricGate(rubrics=["R1"], fail_fast="not_bool")  # type: ignore[arg-type]

    # MultiRubricGate aggregate ambiguous branch
    LogitGate(rubric="R1", pass_threshold=0.8, fail_threshold=0.2, backend=StubBackend(default_prob=0.5))
    mrg_ambig = MultiRubricGate(rubrics=["R1", "R2"], pass_threshold=0.8, fail_threshold=0.2, backend=StubBackend(default_prob=0.5))
    v_mrg_ambig = mrg_ambig.evaluate("cand")
    assert v_mrg_ambig.status == DecisionStatus.AMBIGUOUS

    # LogitGate adaptive margin unmet
    ctrl_step = AdaptiveConformalController(pass_threshold=0.8, fail_threshold=0.2, step_size=0.5)
    ctrl_step.observe(DecisionStatus.PASSED, False)
    lg_step = LogitGate(rubric="R", pass_threshold=0.8, fail_threshold=0.2, adaptive_controller=ctrl_step, backend=StubBackend(default_prob=0.82))
    v_step = lg_step.evaluate("cand")
    assert v_step.status in (DecisionStatus.PASSED, DecisionStatus.AMBIGUOUS)


def test_ast_gate_positional_validators_and_repr() -> None:
    from pydantic import TypeAdapter

    # Positional validators in ASTGate: bool, dict, sequence, callable
    class _PydModel(BaseModel):
        num: int

    adapter = TypeAdapter(_PydModel)
    ast_multi = ASTGate(
        True,
        {"type": "object", "properties": {"val": {"type": "integer"}}},
        [lambda x: True],
        min_length=2,
        max_length=100,
        forbidden_imports=["os"],
        forbidden_patterns=[r"eval\("],
        required_sections=["# Section"],
        acyclic_dag=True,
    )
    repr_str = repr(ast_multi)
    assert "ASTGate" in repr_str
    assert "syntax=True" in repr_str
    assert "acyclic_dag=True" in repr_str

    # TypeAdapter in _run_validator
    ok, _, _ = ASTGate._run_validator(adapter, '{"num": 10}', normalized=False)
    assert ok is True
    bad, _, _ = ASTGate._run_validator(adapter, '{"num": "bad"}', normalized=False)
    assert bad is False

    # Validator returning a Verdict
    v_return = Verdict(passed=True, status=DecisionStatus.PASSED, triggered_layer="V", total_latency_ms=1.0)
    ok_v, _, _ = ASTGate._run_validator(lambda c: v_return, "cand")
    assert ok_v is True

    # ASTGate normalize with required_sections only (strips thinking blocks)
    ast_sec_only = ASTGate(required_sections=["Target Section"], normalize=True)
    v_sec = ast_sec_only.evaluate("<think>thinking</think>\nTarget Section: details")
    assert v_sec.passed is True


def test_judge_gate_heuristics_and_refusal() -> None:
    # 1. Refusal regex in _parse_verdict returns _FAIL
    jg = JudgeGate(rubric="R", backend=StubBackend(diagnostic_text="I cannot fulfill this request because it violates safety guidelines."))
    v_refusal = jg.evaluate("cand")
    assert v_refusal.passed is False

    # 2. _keyword_signal with tie-break: pass before fail vs fail before pass
    assert _keyword_signal("PASS - but contains a small failure", tie_break=True) == (True, DecisionStatus.PASSED, 1.0)
    assert _keyword_signal("FAIL - but passed earlier tests", tie_break=True) == (False, DecisionStatus.FAILED, 0.0)

    # 3. _extract_keyword_verdict on single line vs multi-line
    assert _extract_keyword_verdict("PASS") == (True, DecisionStatus.PASSED, 1.0)
    assert _extract_keyword_verdict("Critique:\nCode fails constraints.") == (False, DecisionStatus.FAILED, 0.0)

    # 4. _can_debias_cot with single logprob response (use_sym=False)
    jg_single_resp = JudgeGate(
        rubric="R",
        backend=StubBackend(diagnostic_text="Looks good.", logprob_responses=[[{"token": "A", "logprob": -0.1}]]),
    )
    probe_gate = jg_single_resp._probe()
    assert probe_gate is not None
    assert probe_gate.symmetric_debiasing is False


def test_audit_dataset_and_leakage_additional_branches(tmp_path: Path) -> None:
    # 1. _extract_pairwise_cases with various invalid / missing structures
    assert _extract_pairwise_cases({}) == []
    assert _extract_pairwise_cases({"candidate_a": "CA", "candidate_b": "CB"}) == []
    assert _extract_pairwise_cases({"candidate_a": "CA", "candidate_b": "CB", "winner": "tie"}) == []

    # Valid pairwise cases with turn_1 / turn_2
    pw_turns = _extract_pairwise_cases({
        "rubric": "User query",
        "candidate_a": "CA",
        "candidate_b": "CB",
        "winner": "candidate_a",
    })
    assert len(pw_turns) == 2

    # 2. _extract_cases_from_mapping_root with tier_c and tier_d
    root_tiers = _extract_cases_from_mapping_root({
        "tier_c": [{"rubric": "R", "candidate": "C", "label": True}],
        "tier_d": [{"rubric": "R", "candidate": "C", "label": False}],
    }, "path")
    assert len(root_tiers) == 2

    # 3. Invalid items in choice_gate_items / score_gate_items raises CalibrationError
    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"choice_gate_items": ["not-a-dict"]}, "path")

    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"score_gate_items": ["not-a-dict"]}, "path")

    # 4. Severe class imbalance and shortcut warnings in audit_benchmark_dataset
    imbalanced_cases = [
        {"rubric": f"R{i}", "candidate": f"def func(): pass # marker_{i}", "label": True}
        for i in range(25)
    ] + [
        {"rubric": "R_neg", "candidate": "broken(:", "label": False}
    ]
    rep_imbal = audit_benchmark_dataset(imbalanced_cases)
    assert any("Class imbalance" in w for w in rep_imbal.warnings)

    # 5. Markdown table and sections in load_calibration_dataset on file without extension
    md_content = "# Test Cases\n### Case 1\n**Rubric**: Clean\n**Candidate**: pass\n**Label**: yes\n"
    p_no_ext_md = tmp_path / "raw_md_dataset"
    p_no_ext_md.write_text(md_content, encoding="utf-8")
    loaded_md = load_calibration_dataset(p_no_ext_md)
    assert len(loaded_md) == 1


def test_permutation_cache_persistence_branches(tmp_path: Path) -> None:

    # Non-dict in loaded mapping
    p_list_json = tmp_path / "list_cache.json"
    p_list_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert _load_cached_mapping(p_list_json) is None

    # float conversion error in _extract_temp_from_entry
    assert _extract_temp_from_entry("not_a_float") is None

    # non-str keys in mapping for _lookup_temperature_in_mapping
    bad_key_mapping: dict[Any, Any] = {123: 1.5, "model": 1.2}
    assert _lookup_temperature_in_mapping("model", bad_key_mapping) == 1.2


def test_rank_gate_full_coverage_branches() -> None:
    # 1. Candidates validation
    rg = RankGate(rubric="R", backend=StubBackend(default_prob=0.9))
    with pytest.raises(ConfigurationError, match="candidates must be a Sequence or Mapping"):
        rg.evaluate(12345)

    with pytest.raises(ConfigurationError, match="Mapping candidate keys must be non-empty strings"):
        rg.evaluate({"": "c1", "B": "c2"})

    with pytest.raises(ConfigurationError, match="Mapping candidate keys must be non-empty strings"):
        rg.evaluate({123: "c1", "B": "c2"})

    # 2. Replay config on RankGate
    cfg = rg.replay_config()
    assert cfg["name"] == rg.name
    assert cfg["mode"] == "auto"

    # 3. BestOfNGate tie_break and pass_threshold
    bon = BestOfNGate(
        rubric="R",
        min_confidence=0.9999,
        backend=StubBackend(default_prob=0.7),
    )
    v_bon_strict = bon.evaluate({"A": "cand 1", "B": "cand 2"})
    assert v_bon_strict.passed is False


# ---------------------------------------------------------------------------
# Section 15: 93%+ High-Yield Surge Across All Targets
# ---------------------------------------------------------------------------


def test_logit_gate_lazy_calibration_and_adaptive_saturation() -> None:
    from logitgate.gates.logit_gate import (
        _get_async_cal_lock,
        _lazy_calibrate_async,
        _lazy_calibrate_sync,
        _resolve_logit_temperature,
    )

    backend = StubBackend(default_prob=0.85)

    # Trigger unknown model sync and async auto-calibration
    t_sync_bin = _resolve_logit_temperature("auto", "unknown-byo-model-sync-1", backend, binary=True)
    assert t_sync_bin > 0.0

    t_sync_mc = _resolve_logit_temperature("auto", "unknown-byo-model-sync-2", backend, binary=False)
    assert t_sync_mc > 0.0

    t_lazy_sync = _lazy_calibrate_sync("unknown-byo-model-sync-3", backend, binary=True)
    assert t_lazy_sync > 0.0

    # Async calibration lock and lazy async calibration
    loop = asyncio.new_event_loop()
    try:
        async def _test_async_cal() -> None:
            lock = _get_async_cal_lock(("cache_path", "model_async_1"))
            assert lock is not None
            t_async = await _lazy_calibrate_async("unknown-byo-model-async-1", backend, binary=True)
            assert t_async > 0.0

        loop.run_until_complete(_test_async_cal())
    finally:
        loop.close()

    # LogitGate _ensure_calibrated and _aensure_calibrated
    lg_lazy = LogitGate(rubric="R", backend=backend, model="lazy-model-1")
    lg_lazy._needs_lazy_calibration = True
    lg_lazy._ensure_calibrated()
    assert lg_lazy._needs_lazy_calibration is False

    # Adaptive conformal force_ambiguous branch
    ctrl_sat = AdaptiveConformalController(pass_threshold=0.8, fail_threshold=0.2, abstain_theta=1.0)
    with ctrl_sat._lock:
        ctrl_sat._theta = 1.5  # theta >= abstain_theta forces ambiguous
    lg_sat = LogitGate(rubric="R", pass_threshold=0.8, fail_threshold=0.2, adaptive_controller=ctrl_sat, backend=backend)
    v_sat = lg_sat.evaluate("cand")
    assert v_sat.status == DecisionStatus.AMBIGUOUS
    assert "adaptive_conformal_band_saturated" in (v_sat.traces[0].metadata.get("abstained_reason") or "")

    # MultiRubricGate properties
    mrg = MultiRubricGate(rubrics=["R1", "R2"], backend=backend)
    assert mrg.distillation_bundle is None
    assert mrg.meta_calibrator is None
    assert mrg.telemetry_sink is None
    mrg.cache = None
    assert mrg.cache is None


@pytest.mark.asyncio
async def test_logit_gate_async_ensure_calibrated_and_choice_score_prompts() -> None:
    backend = StubBackend(default_prob=0.85)
    lg = LogitGate(rubric="R", backend=backend, model="lazy-async-model-2")
    lg._needs_lazy_calibration = True
    await lg._aensure_calibrated()
    assert lg._needs_lazy_calibration is False

    # ChoiceGate _ensure_calibrated and _aensure_calibrated
    cg = ChoiceGate(question="Q", options={"A": "1", "B": "2"}, backend=backend, model="cg-model-1")
    cg._needs_lazy_calibration = True
    cg._ensure_calibrated()
    assert cg._needs_lazy_calibration is False

    cg._needs_lazy_calibration = True
    await cg._aensure_calibrated()
    assert cg._needs_lazy_calibration is False

    # ScoreGate _ensure_calibrated and _aensure_calibrated
    sg = ScoreGate(rubric="R", backend=backend, model="sg-model-1")
    sg._needs_lazy_calibration = True
    sg._ensure_calibrated()
    assert sg._needs_lazy_calibration is False

    sg._needs_lazy_calibration = True
    await sg._aensure_calibrated()
    assert sg._needs_lazy_calibration is False

    # ChoiceGate symmetric_debiasing=False and warm_prefix=True in aevaluate
    cg_single = ChoiceGate(question="Q", options={"A": "1", "B": "2"}, symmetric_debiasing=False, backend=backend)
    v_cg_single = await cg_single.aevaluate("cand")
    assert v_cg_single.passed is True

    cg_warm = ChoiceGate(question="Q", options={"A": "1", "B": "2"}, warm_prefix=True, backend=backend)
    v_cg_warm = await cg_warm.aevaluate("cand")
    assert v_cg_warm.passed is True

    # ScoreGate symmetric_debiasing=False and warm_prefix=True in aevaluate
    sg_single = ScoreGate(rubric="R", symmetric_debiasing=False, backend=backend)
    v_sg_single = await sg_single.aevaluate("cand")
    assert v_sg_single.passed is True

    sg_warm = ScoreGate(rubric="R", warm_prefix=True, backend=backend)
    v_sg_warm = await sg_warm.aevaluate("cand")
    assert v_sg_warm.passed is True


def test_judge_gate_ensure_calibrated_probe_failures_and_cot() -> None:
    backend = StubBackend(default_prob=0.85)

    # 1. _ensure_calibrated and _aensure_calibrated in JudgeGate
    jg = JudgeGate(rubric="R", backend=backend, model="jg-model-1")
    jg._needs_lazy_calibration = True
    jg._ensure_calibrated()
    assert jg._needs_lazy_calibration is False

    # 2. Probe when probe evaluation raises BackendError falls back to None
    class _FailingProbeBackend(StubBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            raise BackendError("Probe timeout")
        async def aquery_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            raise BackendError("Probe timeout")

    jg_probe_fail = JudgeGate(
        rubric="R",
        backend=_FailingProbeBackend(
            diagnostic_text="Ambiguous critique without verdict tag.",
            logprob_responses=[[{"token": "A", "logprob": -0.1}]],
        ),
    )
    v_probe_fail = jg_probe_fail.evaluate("cand")
    assert v_probe_fail.passed is False

    # 3. _adjudicate_cot short notes branch (< 12 characters)
    jg_short_notes = JudgeGate(
        rubric="R",
        backend=StubBackend(
            diagnostic_text="Short.",
            forward_top_logprobs=[{"token": "A", "logprob": -0.1}],
            reverse_top_logprobs=[{"token": "B", "logprob": -0.1}],
        ),
    )
    _parsed_short, meta_short = jg_short_notes._adjudicate_cot("cand", None, "Short.")
    assert meta_short is None


@pytest.mark.asyncio
async def test_judge_gate_async_ensure_calibrated_and_cot_failure() -> None:
    backend = StubBackend(default_prob=0.85)
    jg = JudgeGate(rubric="R", backend=backend, model="jg-async-model-1")
    jg._needs_lazy_calibration = True
    await jg._aensure_calibrated()
    assert jg._needs_lazy_calibration is False

    # _aadjudicate_cot when backend raises BackendError
    class _FailingAsyncCotBackend(StubBackend):
        async def aquery_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            raise BackendError("Async CoT error")

    jg_async_cot = JudgeGate(
        rubric="R",
        backend=_FailingAsyncCotBackend(
            diagnostic_text="Detailed verification notes: checking criteria 1, 2, and 3 carefully.",
            forward_top_logprobs=[{"token": "A", "logprob": -0.1}],
            reverse_top_logprobs=[{"token": "B", "logprob": -0.1}],
        ),
    )
    _parsed_cot, meta_cot = await jg_async_cot._aadjudicate_cot(
        "cand", None, "Detailed verification notes: checking criteria 1, 2, and 3 carefully."
    )
    assert meta_cot is None


def test_audit_pairwise_and_root_structures_deep() -> None:
    # 1. _extract_pairwise_cases with winner variations
    cand_a = {"code": "def a(): pass"}
    cand_b = {"code": "def b(): pass"}
    item_win_0 = {"rubric": "R", "candidate_a": cand_a, "candidate_b": cand_b, "winner": 0}
    cases_0 = _extract_pairwise_cases(item_win_0)
    assert len(cases_0) == 2
    assert cases_0[0].label is True
    assert cases_0[1].label is False

    item_win_1 = {"rubric": "R", "candidate_a": cand_a, "candidate_b": cand_b, "winner": 1}
    cases_1 = _extract_pairwise_cases(item_win_1)
    assert len(cases_1) == 2
    assert cases_1[0].label is True
    assert "def b" in cases_1[0].candidate

    item_win_b = {"rubric": "R", "candidate_a": "CA", "candidate_b": "CB", "winner": "b"}
    assert len(_extract_pairwise_cases(item_win_b)) == 2

    # Winner invalid integer
    assert _extract_pairwise_cases({"rubric": "R", "candidate_a": "CA", "candidate_b": "CB", "winner": 5}) == []

    # 2. _extract_cases_from_mapping_root with choice_gate_items and score_gate_items
    cg_items = [
        {"question": "Q1", "options": {"A": "1", "B": "2"}, "candidate": "C1", "expected_choice": "A"},
        {"question": "Q2", "options": {"A": "1", "B": "2"}, "candidate": "C2", "expected_choice": "B"},
    ]
    sg_items = [
        {"rubric": "R1", "candidate": "C1", "expected_score": 5},
        {"rubric": "R2", "candidate": "C2", "expected_score": 1},
    ]
    root_extracted = _extract_cases_from_mapping_root(
        {"choice_gate_items": cg_items, "score_gate_items": sg_items}, "test_path"
    )
    assert len(root_extracted) == 4

    # 3. DatasetAuditReport format_summary without warnings
    balanced_good = [
        {"rubric": f"R{i}", "candidate": f"candidate_{i}", "label": (i % 2 == 0)}
        for i in range(80)
    ]
    rep_good = audit_benchmark_dataset(balanced_good)
    summ_good = rep_good.format_summary()
    assert "Warnings: none" in summ_good


def test_profile_fit_from_dataset_file(tmp_path: Path) -> None:
    # fit_calibration_profile with dataset path
    dataset_file = tmp_path / "profile_dataset.json"
    cases = [
        {"rubric": "R1", "candidate": "def foo(): pass", "label": True},
        {"rubric": "R2", "candidate": "syntax error", "label": False},
    ] * 6
    dataset_file.write_text(json.dumps(cases), encoding="utf-8")

    backend = StubBackend(default_prob=0.85)
    prof = fit_calibration_profile(
        backend,
        dataset=dataset_file,
        model="from-file-model",
        target_risk_alpha=0.1,
    )
    assert prof.model == "from-file-model"


def test_rank_gate_tournament_odd_trees_and_async() -> None:
    backend = StubBackend(default_prob=0.85)

    # Tournament mode with 3, 5, 6 candidates
    for count in (3, 5, 6):
        cands = {str(i): f"candidate {i}" for i in range(count)}
        rg = RankGate(rubric="R", mode="tournament", backend=backend)
        v = rg.evaluate(cands)
        assert v.passed is True

    # BestOfNGate tie_break="first"
    bon_first = BestOfNGate(rubric="R", backend=backend)
    v_first = bon_first.evaluate({"A": "cand 1", "B": "cand 2"})
    assert v_first.passed is True


@pytest.mark.asyncio
async def test_rank_gate_async_tournament_odd_trees() -> None:
    backend = StubBackend(default_prob=0.85)
    cands_5 = {str(i): f"candidate {i}" for i in range(5)}
    rg = RankGate(rubric="R", mode="tournament", backend=backend)
    v = await rg.aevaluate(cands_5)
    assert v.passed is True


def test_conformal_more_boundary_branches(tmp_path: Path) -> None:
    # 1. Non-finite float in calibrated_probs
    with pytest.raises(CalibrationError, match="must contain finite probabilities"):
        calibrate_cascade_thresholds([float("nan")] * 10, [True] * 5 + [False] * 5)

    # 2. Non-standard label types in load_calibration_dataset
    raw_dict_cases = [
        {"rubric": "R1", "candidate": "C1", "label": 1.0},
        {"rubric": "R2", "candidate": "C2", "label": 0.0},
        {"rubric": "R3", "candidate": "C3", "label": "true"},
        {"rubric": "R4", "candidate": "C4", "label": "false"},
        {"rubric": "R5", "candidate": "C5", "label": "yes"},
        {"rubric": "R6", "candidate": "C6", "label": "no"},
    ]
    p_labels = tmp_path / "labels.json"
    p_labels.write_text(json.dumps(raw_dict_cases), encoding="utf-8")
    loaded = load_calibration_dataset(p_labels)
    assert len(loaded) == 6


def test_permutation_temperatures_and_multiclass_branches() -> None:
    # KNOWN_BINARY_MODEL_TEMPERATURES lookup
    assert has_known_or_cached_temperature("qwen2.5-coder-32b", binary=True) is True
    assert get_model_temperature("qwen2.5-coder-32b", binary=True) > 0.0
    assert get_model_temperature("qwen2.5-coder-32b", binary=False) > 0.0

    # _debiased_multiclass_accum with all non-finite logprobs
    non_finite_top = [{"token": "A", "logprob": float("-inf")}, {"token": "B", "logprob": float("-inf")}]
    accum_nf = _debiased_multiclass_accum(non_finite_top, non_finite_top, ["A", "B"])
    assert "A" in accum_nf and "B" in accum_nf


# ---------------------------------------------------------------------------
# Section 16: Ultra High-Yield Target Precision Sweep
# ---------------------------------------------------------------------------


def test_audit_load_normalized_cases_exhaustive(tmp_path: Path) -> None:
    # 1. JSON list file (lines 466-476)
    p_json_list = tmp_path / "cases_list.json"
    p_json_list.write_text(json.dumps([
        {"rubric": "R1", "candidate": "C1", "label": True},
        {"rubric": "R2", "candidate_a": "CA", "candidate_b": "CB", "winner": "candidate_a"},
        {"rubric": "R3"},  # no candidate or label, skipped
    ]), encoding="utf-8")
    cases_from_json, _ = _load_normalized_cases(p_json_list)
    assert len(cases_from_json) == 3

    # Invalid item in JSON list
    p_bad_item = tmp_path / "bad_item.json"
    p_bad_item.write_text(json.dumps(["not-a-mapping"]), encoding="utf-8")
    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _load_normalized_cases(p_bad_item)

    # Unsupported JSON root type
    p_bad_root = tmp_path / "bad_root.json"
    p_bad_root.write_text("12345", encoding="utf-8")
    with pytest.raises(CalibrationError, match="Unsupported JSON root type"):
        _load_normalized_cases(p_bad_root)

    # Empty cases in file
    p_empty_cases = tmp_path / "empty_cases.json"
    p_empty_cases.write_text(json.dumps([{"rubric": "only_rubric"}]), encoding="utf-8")
    with pytest.raises(CalibrationError, match="No valid labeled cases found"):
        _load_normalized_cases(p_empty_cases)

    # 2. Markdown file loading via _load_normalized_cases (lines 486-492)
    p_md = tmp_path / "audit_cases.md"
    p_md.write_text("| Rubric | Candidate | Label |\n|---|---|---|\n| R | C | pass |\n", encoding="utf-8")
    cases_from_md, _ = _load_normalized_cases(p_md)
    assert len(cases_from_md) == 1

    # Empty markdown file
    p_empty_md = tmp_path / "empty_audit.md"
    p_empty_md.write_text("No cases here", encoding="utf-8")
    with pytest.raises(CalibrationError, match="No valid labeled cases found"):
        _load_normalized_cases(p_empty_md)

    # 3. Source as Mapping (lines 494-498)
    cases_from_map, _ = _load_normalized_cases({"data": [{"rubric": "R", "candidate": "C", "label": True}]})
    assert len(cases_from_map) == 1

    with pytest.raises(CalibrationError, match="No valid labeled cases found"):
        _load_normalized_cases({"data": []})

    # 4. Source not Sequence (line 501)
    with pytest.raises(CalibrationError, match="Unsupported dataset source type"):
        _load_normalized_cases(12345)

    # 5. Source containing tuples (lines 513-519)
    cases_from_tuples, _ = _load_normalized_cases([("Rubric T", "Candidate T", True)])
    assert len(cases_from_tuples) == 1
    assert cases_from_tuples[0].rubric == "Rubric T"

    # 6. Source containing objects with rubric/candidate/expected (lines 520-529)
    cases_from_objs, _ = _load_normalized_cases([BenchmarkCase("Rubric B", "Candidate B", False)])
    assert len(cases_from_objs) == 1
    assert cases_from_objs[0].label is False

    # 7. Unsupported item in source sequence (line 531)
    with pytest.raises(CalibrationError, match="Unsupported dataset item type"):
        _load_normalized_cases([object()])

    # 8. Empty dataset sequence (line 534)
    with pytest.raises(CalibrationError, match="Dataset contains no valid labeled cases"):
        _load_normalized_cases([])


def test_ast_gate_full_normalization_and_validation_branches(tmp_path: Path) -> None:
    # 1. normalize_input with json_schema and markdown fences
    ast_json_norm = ASTGate(
        json_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
        normalize=True,
    )
    unwrapped_span = ast_json_norm.normalize_input("```json\n{\"a\": 10}\n```")
    assert '{"a": 10}' in unwrapped_span

    # normalize_input with array json_schema
    ast_arr_norm = ASTGate(json_schema={"type": "array"}, normalize=True)
    assert "[1, 2]" in ast_arr_norm.normalize_input("```json\n[1, 2]\n```")

    # 2. evaluate with min_length and max_length violations (line 643)
    ast_bounds = ASTGate(min_length=5, max_length=10)
    v_too_short = ast_bounds.evaluate("hi")
    assert v_too_short.passed is False
    assert v_too_short.traces[0].failure_code == ASTFailureCode.MIN_LENGTH_VIOLATION

    v_too_long = ast_bounds.evaluate("this is way too long")
    assert v_too_long.passed is False
    assert v_too_long.traces[0].failure_code == ASTFailureCode.MAX_LENGTH_VIOLATION

    # 3. evaluate with JSON_SYNTAX_ERROR on broken JSON (lines 660-671)
    ast_schema = ASTGate(json_schema={"type": "object"})
    v_syntax_err = ast_schema.evaluate("{broken json")
    assert v_syntax_err.passed is False
    assert v_syntax_err.traces[0].failure_code == ASTFailureCode.JSON_SYNTAX_ERROR

    # evaluate with JSON_TYPE_ERROR on integer when schema expects JSON (lines 673-674)
    v_type_err = ast_schema.evaluate(12345)
    assert v_type_err.passed is False
    assert v_type_err.traces[0].failure_code == ASTFailureCode.JSON_TYPE_ERROR

    # 4. Imports extraction from fenced blocks when normalize=True (lines 837-840)
    ast_fenced_imp = ASTGate(forbidden_imports=["sys"], normalize=True)
    v_fence_imp = ast_fenced_imp.evaluate("<think>thought</think>\n```python\nimport sys\n```")
    assert v_fence_imp.passed is False
    assert v_fence_imp.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT

    # 5. acyclic_dag with JSON strings and dicts (lines 898-913)
    ast_dag = ASTGate(acyclic_dag=True)
    v_dag_json_str = ast_dag.evaluate('[{"id": "T1", "dependencies": []}]')
    assert v_dag_json_str.passed is True

    v_dag_tasks_dict = ast_dag.evaluate({"tasks": [{"id": "T1", "dependencies": []}]})
    assert v_dag_tasks_dict.passed is True

    v_dag_nodes_dict = ast_dag.evaluate({"nodes": [{"id": "T1", "dependencies": []}]})
    assert v_dag_nodes_dict.passed is True

    v_dag_invalid_str = ast_dag.evaluate("not json at all")
    assert v_dag_invalid_str.passed is False


def test_judge_gate_structured_heuristics_and_probe_deeper() -> None:
    # 1. _extract_structured_verdict with JSON having 'passed' and 'verdict'
    v_json_passed = _extract_structured_verdict('{"passed": true}')
    assert v_json_passed is not None and v_json_passed[0] is True

    v_json_verdict = _extract_structured_verdict('{"verdict": "pass", "score": 1e999}')
    assert v_json_verdict is not None and v_json_verdict[0] is True and v_json_verdict[2] == 1.0

    v_json_bad_score = _extract_structured_verdict('{"verdict": "pass", "score": "not_numeric"}')
    assert v_json_bad_score is not None and v_json_bad_score[2] == 1.0

    # 2. _extract_keyword_verdict on opening line and empty text
    assert _extract_keyword_verdict("") is None
    assert _extract_keyword_verdict("PASS on first line\nMore critique text.") == (True, DecisionStatus.PASSED, 1.0)
    assert _extract_keyword_verdict("Critique line 1.\nCritique line 2.\nFAIL") == (False, DecisionStatus.FAILED, 0.0)

    # 3. JudgeGate with unhashable trigger raises ConfigurationError
    with pytest.raises(ConfigurationError, match="Invalid trigger"):
        JudgeGate(trigger=[])  # type: ignore[arg-type]

    # 4. _compile_prompt with short max_tokens
    jg = JudgeGate(rubric=None, backend=StubBackend(default_prob=0.9), max_tokens=30)
    prompt_short = jg._compile_prompt("candidate artifact", rubric_override="Specific task rubric")
    assert "Specific task rubric" in prompt_short
    assert "candidate artifact" in prompt_short
    assert "Begin your response on the very first line with your verdict tag" in prompt_short

    # 5. _parse_verdict empty text
    assert jg._parse_verdict("") == (False, DecisionStatus.FAILED, 0.0)


def test_logit_gate_prompts_and_snapshots_exhaustion(tmp_path: Path) -> None:
    backend = StubBackend(default_prob=0.85)

    # 1. LogitGate failure snapshot with save_on_fail=True
    lg_fail = LogitGate(
        rubric="R",
        pass_threshold=0.95,
        fail_threshold=0.5,
        save_on_fail=True,
        backend=StubBackend(default_prob=0.1),
    )
    v_fail = lg_fail.evaluate("failing candidate", snapshot_dir=tmp_path)
    assert v_fail.passed is False

    # 2. LogitGate _binary_prompts with rubric_index
    lg = LogitGate(rubric="Default Rubric", backend=backend)
    p_fwd, p_rev = lg._binary_prompts("cand", rubrics=["Rubric A", "Rubric B"], rubric_index=1, boundary_nonce="nonce_test")
    assert "Rubric B" in p_fwd
    assert p_rev is not None and "Rubric B" in p_rev

    # 3. LogitGate evaluate_batch empty rubrics
    assert lg.evaluate_batch("cand", []) == []

    # 4. LogitGate evaluate_batch fail_fast=True with failure
    lg_batch_ff = LogitGate(rubric="R", pass_threshold=0.8, backend=StubBackend(default_prob=0.1))
    res_batch_ff = lg_batch_ff.evaluate_batch("cand", ["R1", "R2"], fail_fast=True)
    assert len(res_batch_ff) == 1

    # 5. ChoiceGate save_on_fail in evaluate and aevaluate
    cg_fail = ChoiceGate(
        question="Q",
        options={"A": "1", "B": "2"},
        min_confidence=0.9999,
        save_on_fail=True,
        backend=StubBackend(default_prob=0.1),
    )
    v_cg_fail = cg_fail.evaluate("cand", snapshot_dir=tmp_path)
    assert v_cg_fail.passed is False

    # 6. ChoiceGate _choice_prompts_list with boundary_nonce
    p_cg_anchors = cg_fail._choice_prompts_list("cand", boundary_nonce="nonce_test")
    assert len(p_cg_anchors) >= 2
    assert "<candidate_nonce_test>" in p_cg_anchors[0]

    # 7. ScoreGate replay_config and repr
    sg = ScoreGate(rubric="Rate 1 to 5", fail_threshold=2.0, min_score=1, max_score=5, backend=backend)
    assert "Rate 1 to 5" in repr(sg)
    cfg_sg = sg.replay_config()
    assert cfg_sg["threshold"] == 3.5

    v_sg_fail = ScoreGate(
        rubric="R",
        fail_threshold=2.0,
        save_on_fail=True,
        backend=StubBackend(default_prob=0.1),
    ).evaluate("cand", snapshot_dir=tmp_path)
    assert v_sg_fail.passed is False


@pytest.mark.asyncio
async def test_choice_gate_async_failure_snapshot(tmp_path: Path) -> None:
    cg_async_fail = ChoiceGate(
        question="Q",
        options={"A": "1", "B": "2"},
        min_confidence=0.9999,
        save_on_fail=True,
        backend=StubBackend(default_prob=0.1),
    )
    v = await cg_async_fail.aevaluate("cand", snapshot_dir=tmp_path)
    assert v.passed is False


def test_rank_gate_deep_validation_and_listwise_error() -> None:
    # 1. _normalize_context with invalid type
    with pytest.raises(ConfigurationError, match="context must be a string or None"):
        RankGate(rubric="R", context=object())  # type: ignore[arg-type]

    # 2. _normalize_candidates with non-mapping, non-sequence
    with pytest.raises(ConfigurationError, match="candidates must be a Sequence or Mapping"):
        RankGate(rubric="R").evaluate(12345)

    # 3. _normalize_candidates with single item mapping
    with pytest.raises(ConfigurationError, match="Ranking requires at least 2 candidates"):
        RankGate(rubric="R").evaluate({"A": "cand"})

    # 4. _normalize_candidates with empty key
    with pytest.raises(ConfigurationError, match="Mapping candidate keys must be non-empty strings"):
        RankGate(rubric="R").evaluate({" ": "cand 1", "B": "cand 2"})

    # 5. Invalid permutation_mode on RankGate
    with pytest.raises(ConfigurationError, match="Invalid permutation_mode"):
        RankGate(rubric="R", permutation_mode="invalid")  # type: ignore[arg-type]

    # 6. Invalid min_confidence on RankGate
    with pytest.raises(ConfigurationError, match=r"min_confidence must be a float in \(0.0, 1.0\]"):
        RankGate(rubric="R", min_confidence=1.5)

    # 7. Invalid max_positional_bias on RankGate
    with pytest.raises(ConfigurationError, match="max_positional_bias must be a positive float"):
        RankGate(rubric="R", max_positional_bias=-0.5)

    # 8. _compute_listwise_verdict with empty top_list raises BackendError
    rg = RankGate(rubric="R", mode="listwise")
    with pytest.raises(BackendError, match="Backend returned no token log-probabilities"):
        rg._compute_listwise_verdict([("A", "c1"), ("B", "c2")], ["A", "B"], [[0, 1]], [], 0.0, {})


def test_profile_and_permutation_deep_branches() -> None:
    from logitgate.calibration.profile import (
        _validate_alpha,
        calibrate_choice_thresholds,
    )

    # 1. _validate_alpha invalid values
    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        _validate_alpha(0.0)
    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        _validate_alpha(1.0)
    with pytest.raises(CalibrationError, match="target_risk_alpha must be strictly"):
        _validate_alpha(True)

    # 2. calibrate_choice_thresholds input validations
    with pytest.raises(CalibrationError, match="Calibration requires matching sequences of at least 10 samples"):
        calibrate_choice_thresholds("not_seq", ["A"])  # type: ignore[arg-type]

    with pytest.raises(CalibrationError, match="Calibration requires matching sequences of at least 10 samples"):
        calibrate_choice_thresholds([{"A": 0.5, "B": 0.5}] * 5, ["A"] * 5)

    with pytest.raises(CalibrationError, match="Calibration requires matching sequences of at least 10 samples"):
        calibrate_choice_thresholds([{"A": 0.5, "B": 0.5}] * 10, ["A"] * 8)

    with pytest.raises(CalibrationError, match="Each entry in choice_probs_list must be a mapping"):
        calibrate_choice_thresholds(["not_mapping"] * 10, ["A"] * 10)  # type: ignore[list-item]

    with pytest.raises(CalibrationError, match="ground_truth_choices must contain non-empty strings"):
        calibrate_choice_thresholds([{"A": 0.5, "B": 0.5}] * 10, [""] * 10)

    # 3. CalibrationProfile schema_version < 1
    with pytest.raises(CalibrationError):
        CalibrationProfile(schema_version=0)

    # 4. auto_calibrate_backend top2 non-finite error
    class _FailingTop2Backend(StubBackend):
        def __init__(self) -> None:
            super().__init__()
            self._call_count = 0
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            self._call_count += 1
            if self._call_count == 2:
                return []
            return [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}]

    with pytest.raises(BackendError, match="returned no token log-probabilities"):
        auto_calibrate_backend(_FailingTop2Backend(), save_cache=False)


@pytest.mark.asyncio
async def test_aauto_calibrate_backend_top2_failure() -> None:
    class _AsyncFailingTop2Backend(StubBackend):
        def __init__(self) -> None:
            super().__init__()
            self._call_count = 0
        async def aquery_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            self._call_count += 1
            if self._call_count == 2:
                return []
            return [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}]

    with pytest.raises(BackendError, match="returned no token log-probabilities"):
        await aauto_calibrate_backend(_AsyncFailingTop2Backend(), save_cache=False)



# ==============================================================================
# SECTION 17: DEEP EXHAUSTIVE BRANCH KILLING (LOGIT_GATE, AST_GATE, PROFILE, AUDIT, RANK_GATE, JUDGE_GATE)
# ==============================================================================


def test_logit_gate_lazy_calibration_and_resolution() -> None:
    from logitgate.gates.logit_gate import (
        _lazy_calibrate_sync,
        _resolve_gate_temperature,
        _resolve_logit_temperature,
    )

    backend = StubBackend(default_prob=0.8)

    # 1. _resolve_logit_temperature with "auto" on custom model
    t_res = _resolve_logit_temperature("auto", "custom_model_x", backend, binary=True)
    assert t_res > 0.0

    # 2. _lazy_calibrate_sync
    t_lazy = _lazy_calibrate_sync("custom_model_y", backend, binary=True)
    assert t_lazy > 0.0

    # 3. _resolve_gate_temperature
    eff_t, needs_cal = _resolve_gate_temperature(1.25, "custom_model_z", backend, None, binary=False)
    assert eff_t == 1.25
    assert needs_cal is False


@pytest.mark.asyncio
async def test_logit_gate_async_lazy_and_aquery_branches() -> None:
    from logitgate.gates.logit_gate import _lazy_calibrate_async

    backend = StubBackend(default_prob=0.8)
    t_async = await _lazy_calibrate_async("custom_model_async", backend, binary=False)
    assert t_async > 0.0

    # _aquery_pair with reverse=None
    lg = LogitGate(rubric="Rubric", backend=backend)
    fwd_top, rev_top = await lg._aquery_pair("forward prompt", None)
    assert fwd_top is not None
    assert rev_top is None

    # _aquery_pair with warm_prefix=True
    lg_warm = LogitGate(rubric="Rubric", warm_prefix=True, backend=backend)
    fwd_w, rev_w = await lg_warm._aquery_pair("forward prompt", "reverse prompt")
    assert fwd_w is not None
    assert rev_w is not None


def test_logit_gate_meta_calibrator_and_eval_batch_validation() -> None:
    import time
    from unittest.mock import MagicMock

    backend = StubBackend(default_prob=0.85)

    # 1. Meta-calibrator branch in _verdict_from_logprobs
    mock_mc = MagicMock()
    mock_mc.predict_log_odds.return_value = 1.2
    mock_mc.pass_threshold = None
    mock_mc.fail_threshold = None
    lg_mc = LogitGate(rubric="R", pass_threshold=0.75, backend=backend)
    lg_mc.meta_calibrator = mock_mc

    top_fwd = [{"token": "A", "logprob": -0.2}, {"token": "B", "logprob": -1.8}]
    top_rev = [{"token": "B", "logprob": -0.3}, {"token": "A", "logprob": -1.5}]

    # Dual pass with meta_calibrator
    v_dual = lg_mc._verdict_from_logprobs(top_fwd, top_rev, time.perf_counter())
    assert v_dual.passed is True
    assert mock_mc.predict_log_odds.called

    # Single pass with meta_calibrator
    v_single = lg_mc._verdict_from_logprobs(top_fwd, None, time.perf_counter())
    assert v_single.passed is True

    # 2. evaluate_batch validation errors
    lg = LogitGate(rubric="R", backend=backend)
    with pytest.raises(ConfigurationError, match="fail_fast must be a bool"):
        lg.evaluate_batch("cand", ["R1"], fail_fast="invalid")  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="rubrics must be a sequence of rubric strings"):
        lg.evaluate_batch("cand", "not_a_sequence")

    with pytest.raises(ConfigurationError, match="Each rubric in rubrics must be a non-empty string"):
        lg.evaluate_batch("cand", ["   "])


@pytest.mark.asyncio
async def test_logit_gate_aevaluate_batch_fail_fast_multi_item() -> None:
    backend = StubBackend(default_prob=0.9)
    lg = LogitGate(rubric="R", pass_threshold=0.8, fail_threshold=0.3, backend=backend)

    call_idx = 0

    async def _mock_aquery(p: str, **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:  # rubric 0 fwd: pass token A high
            return [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.5}]
        if call_idx == 2:  # rubric 0 rev: pass token B high
            return [{"token": "B", "logprob": -0.05}, {"token": "A", "logprob": -3.5}]
        if call_idx == 3:  # rubric 1 fwd: fail token B high
            return [{"token": "A", "logprob": -3.5}, {"token": "B", "logprob": -0.05}]
        # rubric 1 rev: fail token A high
        return [{"token": "B", "logprob": -3.5}, {"token": "A", "logprob": -0.05}]

    lg.backend.aquery_top_logprobs = _mock_aquery  # type: ignore[assignment,method-assign]
    results = await lg.aevaluate_batch("cand", ["Rubric 1", "Rubric 2", "Rubric 3"], fail_fast=True)
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False


def test_choice_gate_and_score_gate_deep_branches(tmp_path: Path) -> None:
    import time

    backend = StubBackend(default_prob=0.85)

    # 1. ChoiceGate repr with long question
    cg_long = ChoiceGate(
        question="This is a very long question that exceeds forty characters in total length",
        options={"A": "1", "B": "2"},
        backend=backend,
        warm_prefix=True,
    )
    assert "..." in repr(cg_long)
    cfg_cg = cg_long.replay_config()
    assert cfg_cg["warm_prefix"] is True

    # 2. ChoiceGate _choice_prompts
    p1, p2 = cg_long._choice_prompts("cand")
    assert p1 is not None and p2 is not None

    # 3. ChoiceGate _verdict_from_logprobs_list empty error
    with pytest.raises(BackendError, match="Backend returned no token log-probabilities"):
        cg_long._verdict_from_logprobs_list([], time.perf_counter())

    # 4. ChoiceGate _verdict_from_logprobs helper
    top_fwd = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}]
    v_cg = cg_long._verdict_from_logprobs(top_fwd, None, time.perf_counter())
    assert v_cg.choice == "A"

    # 5. ChoiceGate cache hit
    cg_cached = ChoiceGate(question="Q?", options={"A": "1", "B": "2"}, cache=tmp_path / "cg_cache.db", backend=backend)
    v1 = cg_cached.evaluate("cand_cache")
    v2 = cg_cached.evaluate("cand_cache")
    assert v2.passed == v1.passed

    # 6. ScoreGate invalid max_score_std
    with pytest.raises(ConfigurationError, match="max_score_std must be a positive float"):
        ScoreGate(rubric="R", max_score_std=-1.0, backend=backend)

    # 7. ScoreGate replay_config with warm_prefix
    sg_warm = ScoreGate(rubric="R", warm_prefix=True, backend=backend)
    assert sg_warm.replay_config()["warm_prefix"] is True

    # 8. ScoreGate ensure_calibrated branches
    sg_warm._ensure_calibrated()
    cg_cached._ensure_calibrated()


@pytest.mark.asyncio
async def test_choice_and_score_async_calibrated_and_cache(tmp_path: Path) -> None:
    backend = StubBackend(default_prob=0.85)

    cg = ChoiceGate(question="Q?", options={"A": "1", "B": "2"}, cache=tmp_path / "cg_async_cache.db", backend=backend)
    v1 = await cg.aevaluate("cand_async_cache")
    v2 = await cg.aevaluate("cand_async_cache")
    assert v2.choice == v1.choice

    await cg._aensure_calibrated()

    sg = ScoreGate(rubric="R", backend=backend)
    await sg._aensure_calibrated()


def test_ast_gate_full_repr_and_config_validation() -> None:
    # 1. ASTGate repr with all features
    gate_all = ASTGate(
        name="full_gate",
        syntax=True,
        forbidden_imports=["os", "sys"],
        forbidden_patterns=["eval", "exec"],
        required_sections=["## Title", "## Body"],
        acyclic_dag=True,
        json_schema={"type": "object"},
        min_length=10,
        max_length=200,
        validators=[lambda c: None],
        normalize=True,
    )
    repr_str = repr(gate_all)
    assert "syntax=True" in repr_str
    assert "forbidden_imports=" in repr_str
    assert "forbidden_patterns=" in repr_str
    assert "required_sections=" in repr_str
    assert "acyclic_dag=True" in repr_str
    assert "json_schema=" in repr_str
    assert "min_length=10" in repr_str
    assert "max_length=200" in repr_str
    assert "validators=" in repr_str

    # 2. Config validation for min_length and max_length
    with pytest.raises(ConfigurationError, match="max_length must be a non-negative integer"):
        ASTGate(max_length=-5)

    with pytest.raises(ConfigurationError, match=r"min_length \(50\) cannot exceed max_length \(10\)"):
        ASTGate(min_length=50, max_length=10)


def test_ast_gate_single_validator_and_normalize_branches() -> None:
    class _SampleModel(BaseModel):
        val: int

    class _OtherModel(BaseModel):
        val: int = 42

    # 1. Pydantic validator on un-normalized string
    g_pyd = ASTGate(validators=[_SampleModel], normalize=False)  # type: ignore[list-item]
    assert g_pyd.evaluate('{"val": 99}').passed is True
    assert g_pyd.evaluate(_OtherModel()).passed is True
    assert g_pyd.evaluate({"val": 7}).passed is True

    # 2. Validator with validate_python
    class _CustomTypeValidator:
        def validate_python(self, obj: Any) -> Any:
            if obj != 42:
                raise ValueError("Must be 42")
            return obj
        def validate_json(self, s: str) -> Any:
            return self.validate_python(json.loads(s))

    g_type = ASTGate(validators=[_CustomTypeValidator()], normalize=False)  # type: ignore[list-item]
    assert g_type.evaluate(42).passed is True
    assert g_type.evaluate(99).passed is False

    # 3. Non-callable object with .validate
    class _DuckValidator:
        def validate(self, obj: Any) -> None:
            if not isinstance(obj, dict):
                raise TypeError("Must be dict")

    g_duck = ASTGate(validators=[_DuckValidator()], normalize=False)  # type: ignore[list-item]
    assert g_duck.evaluate('{"key": 1}').passed is True

    # 4. normalize_input branches
    g_custom_norm = ASTGate(normalize=lambda c: str(c).upper())
    assert g_custom_norm.normalize_input("lower") == "LOWER"

    g_sec_norm = ASTGate(required_sections=["Target"], syntax=False, normalize=True)
    assert g_sec_norm.normalize_input("<think>thoughts</think>Target text") == "Target text"

    g_invalid_json_norm = ASTGate(json_schema={"type": "object"}, normalize=True)
    assert g_invalid_json_norm.normalize_input("not { valid json") == "not { valid json"


def test_ast_gate_json_recursion_type_and_dag_branches() -> None:
    from unittest import mock

    g_json = ASTGate(json_schema={"type": "object"}, normalize=False)

    # 1. RecursionError in json.loads
    with mock.patch("json.loads", side_effect=RecursionError("Depth exceeded")):
        v_rec = g_json.evaluate('{"a": 1}')
        assert v_rec.passed is False
        assert "Maximum JSON nesting depth exceeded" in (v_rec.reason or "")

    # 2. MemoryError in ast.parse
    g_syntax = ASTGate(syntax=True)
    with mock.patch("ast.parse", side_effect=MemoryError("OOM")):
        v_oom = g_syntax.evaluate("x = 1")
        assert v_oom.passed is False
        assert "Syntax error: MemoryError" in (v_oom.reason or "")

    # 3. Acyclic DAG with string input having 'tasks' or 'nodes'
    g_dag = ASTGate(acyclic_dag=True, normalize=False)
    v_tasks = g_dag.evaluate('{"tasks": [{"id": "1", "deps": []}]}')
    assert v_tasks.passed is True

    v_nodes = g_dag.evaluate('{"nodes": [{"id": "A", "deps": []}]}')
    assert v_nodes.passed is True

    v_bad_dag = g_dag.evaluate('{"invalid_key": 123}')
    assert v_bad_dag.passed is False
    assert "Candidate does not contain a valid task list" in (v_bad_dag.reason or "")


def test_profile_deep_calibration_thresholds() -> None:
    from logitgate.calibration.profile import (
        calibrate_choice_thresholds,
        calibrate_score_thresholds,
    )

    # 1. calibrate_choice_thresholds calibrate_margin non-bool
    with pytest.raises(CalibrationError, match="calibrate_margin must be a bool"):
        calibrate_choice_thresholds([{"A": 0.5, "B": 0.5}] * 10, ["A"] * 10, calibrate_margin="not_bool")  # type: ignore[arg-type]

    # 2. calibrate_choice_thresholds raw_val non-float
    with pytest.raises(CalibrationError, match="Option probabilities must be finite floats"):
        calibrate_choice_thresholds([{"A": "bad", "B": 0.5}] * 10, ["A"] * 10)  # type: ignore[dict-item]

    # 3. calibrate_choice_thresholds ground truth missing
    with pytest.raises(CalibrationError, match="Ground-truth choice 'C' is missing"):
        calibrate_choice_thresholds([{"A": 0.5, "B": 0.5}] * 10, ["C"] * 10)

    # 4. calibrate_choice_thresholds sum not near 1.0
    with pytest.raises(CalibrationError, match=r"Option probabilities must sum to ~1\.0"):
        calibrate_choice_thresholds([{"A": 0.1, "B": 0.1}] * 10, ["A"] * 10)

    # 5. calibrate_score_thresholds raw_s non-float
    with pytest.raises(CalibrationError, match="expected_scores must contain finite floats"):
        calibrate_score_thresholds(["invalid"] * 10, [True] * 5 + [False] * 5, min_score=1, max_score=5)  # type: ignore[list-item]

    # 6. calibrate_score_thresholds saturated score warning
    with pytest.warns(CalibrationWarning, match="ScoreGate conformal guarantee does not hold"):
        calibrate_score_thresholds([1.0] * 5 + [5.0] * 5, [False] * 5 + [True] * 5, min_score=1, max_score=5, target_risk_alpha=0.1)


def test_audit_pairwise_and_extract_mapping_roots() -> None:
    from logitgate.calibration.audit import (
        _extract_cases_from_mapping_root,
        _extract_pairwise_cases,
    )

    # 1. _extract_pairwise_cases with winner=True / False
    pw_item_true = {"candidate_a": "good", "candidate_b": "bad", "winner": True, "rubric": "R"}
    cases_t = _extract_pairwise_cases(pw_item_true)
    assert len(cases_t) == 2
    assert cases_t[0].candidate == "good" and cases_t[0].label is True

    pw_item_false = {"candidate_a": "bad", "candidate_b": "good", "winner": False, "rubric": "R"}
    cases_f = _extract_pairwise_cases(pw_item_false)
    assert len(cases_f) == 2
    assert cases_f[0].candidate == "good" and cases_f[0].label is True

    # 2. _extract_cases_from_mapping_root validation errors
    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"pairwise_items": [123]}, "test_root")

    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"tier_a": ["not_a_dict"]}, "test_root")

    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"choice_gate_items": [None]}, "test_root")

    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"score_gate_items": [False]}, "test_root")

    with pytest.raises(CalibrationError, match="Expected JSON object"):
        _extract_cases_from_mapping_root({"test_cases": [456]}, "test_root")

    # 3. _extract_cases_from_mapping_root projected cases
    cg_root = {
        "choice_gate_items": [
            {"question": "Q?", "candidate": "cand1", "expected_choice": "A"},
            {"question": "Q?", "candidate": "cand2", "expected_choice": "unknown_opt"},
        ]
    }
    cases_cg = _extract_cases_from_mapping_root(cg_root, "cg_root")
    assert len(cases_cg) == 2
    assert cases_cg[0].label is True
    assert cases_cg[1].label is False

    sg_root = {
        "score_gate_items": [
            {"rubric": "R", "candidate": "cand1", "expected_band": [4.0]},
            {"rubric": "R", "candidate": "cand2", "expected_band": "other"},
        ]
    }
    cases_sg = _extract_cases_from_mapping_root(sg_root, "sg_root")
    assert len(cases_sg) == 2
    assert cases_sg[0].label is True
    assert cases_sg[1].label is False


def test_rank_gate_calibration_mode_and_select() -> None:
    from logitgate.gates.rank_gate import BestOfNGate

    backend = StubBackend(default_prob=0.85)
    rg = RankGate(rubric="R", mode="auto", backend=backend)

    # 1. _resolve_effective_mode > 32 in auto mode
    with pytest.raises(ConfigurationError, match="RankGate supports at most 32 candidates"):
        rg._resolve_effective_mode(35)

    # 2. _ensure_calibrated
    rg._ensure_calibrated()

    # 3. BestOfNGate select
    bon = BestOfNGate(rubric="R", backend=backend)
    candidates = ["Candidate A is superior", "Candidate B is inferior"]
    res_sel = bon.select(candidates)
    assert res_sel.winner_index in (0, 1)
    assert res_sel.winner in candidates


@pytest.mark.asyncio
async def test_rank_gate_async_select_and_post_gate() -> None:
    from logitgate.gates.rank_gate import BestOfNGate

    backend = StubBackend(default_prob=0.85)
    post_g = LogitGate(rubric="Verify Winner", backend=backend)
    bon = BestOfNGate(rubric="R", post_gate=post_g, backend=backend)

    await bon._aensure_calibrated()

    candidates = ["Candidate 1", "Candidate 2"]
    res_asel = await bon.aselect(candidates)
    assert res_asel.winner_index in (0, 1)
    assert res_asel.winner in candidates

    v_post = await bon.aevaluate(candidates)
    assert v_post.passed is True


def test_judge_gate_trigger_setter_repr_and_probes() -> None:
    backend = StubBackend(default_prob=0.85)
    jg = JudgeGate(backend=backend)

    # 1. trigger property and setter
    assert jg.trigger == "on_ambiguous_only"
    jg.trigger = "adjudicate"
    assert jg.can_override_fail is True

    with pytest.raises(ConfigurationError, match="Invalid trigger 'unsupported'"):
        jg.trigger = "unsupported"  # type: ignore[assignment]

    # 2. repr with rubric=None and max_tokens
    jg_tok = JudgeGate(rubric=None, max_tokens=150, backend=backend)
    r_tok = repr(jg_tok)
    assert "None (inherited)" in r_tok
    assert "max_tokens=150" in r_tok

    # 3. _from_probe branches
    v_none = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=None,
        total_latency_ms=1.0,
        triggered_layer="l",
    )
    assert JudgeGate._from_probe(v_none) is None

    v_ambig = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        probability=0.5,
        total_latency_ms=1.0,
        triggered_layer="l",
    )
    parsed_kw = JudgeGate._from_probe(v_ambig, text="PASS on top line\nCriticism.")
    assert parsed_kw is not None
    assert parsed_kw[0] is True


@pytest.mark.asyncio
async def test_judge_gate_aparse_verdict_refusal_and_probe_error() -> None:
    backend = StubBackend(default_prob=0.85)
    jg = JudgeGate(backend=backend)

    # Refusal string
    res_refusal = await jg._aparse_verdict("I am sorry, but as an AI I cannot evaluate this.")
    assert res_refusal == (False, DecisionStatus.FAILED, 0.0)

    # Probe failure handling
    class _FailingProbeBackend(StubBackend):
        async def aquery_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            raise BackendError("Probe query failed")

    jg_failing = JudgeGate(backend=_FailingProbeBackend())
    res_fail = await jg_failing._aparse_verdict("Ambiguous text without keywords.")
    assert res_fail[0] is False


def test_permutation_save_model_profile_existing_and_ephemeral(tmp_path: Path) -> None:
    from logitgate.debiasing.permutation import (
        _EPHEMERAL_CALIBRATION_CACHE,
        CalibrationResult,
        _persist_calibration_profile,
    )

    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"existing_model": {"optimal_temperature": 1.0}}), encoding="utf-8")

    path_key = str(cache_file.resolve())
    _EPHEMERAL_CALIBRATION_CACHE[path_key] = {"ephemeral_model": {"optimal_temperature": 1.5}}

    res = CalibrationResult(
        optimal_temperature=2.0,
        brier_score_before=0.2,
        brier_score_after=0.1,
        ece_before=0.15,
        ece_after=0.05,
    )
    _persist_calibration_profile("new_model", res, cache_path=cache_file)

    loaded = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "existing_model" in loaded
    assert "ephemeral_model" in loaded
    assert "new_model" in loaded
