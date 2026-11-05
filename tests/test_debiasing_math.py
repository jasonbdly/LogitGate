from __future__ import annotations

import math

import pytest

from logitgate.backends.base import StubBackend
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_binary_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import (
    SurfaceTokenMap,
    combine_symmetric_log_odds,
    compute_symmetric_debiased_verdict,
    fit_calibration,
    marginalize_pole_logprob,
)
from logitgate.debiasing.softmax import extract_token_logprob
from logitgate.gates.logit_gate import LogitGate


def test_symmetric_log_odds_cancels_extreme_bias():
    """
    Proves that combine_symmetric_log_odds cancels out an extreme +5.0 logit bias
    for token 'A' when true signal is +2.0, returning sigma(2.0) ~= 0.8808,
    whereas naive probability averaging fails at 0.523.
    """
    true_signal = 2.0
    bias_a = 5.0

    # Pass 1: A is Pass, B is Fail -> logit(A) has bias +5.0
    forward_log_odds = true_signal + bias_a  # 7.0
    # Pass 2: B is Pass, A is Fail -> logit(A) has bias +5.0, so logit(B) - logit(A) = -3.0
    reverse_log_odds = true_signal - bias_a  # -3.0

    # 1. LogitGate combined symmetric log-odds
    calibrated_prob = combine_symmetric_log_odds(forward_log_odds, reverse_log_odds)
    expected_prob = 1.0 / (1.0 + math.exp(-true_signal))  # sigma(2.0) ~= 0.880797
    assert pytest.approx(calibrated_prob, abs=1e-4) == expected_prob
    assert pytest.approx(calibrated_prob, abs=1e-3) == 0.8808

    # 2. Naive probability averaging failure
    p1 = 1.0 / (1.0 + math.exp(-forward_log_odds))  # sigma(7.0) ~= 0.9991
    p2 = 1.0 / (1.0 + math.exp(-reverse_log_odds))  # sigma(-3.0) ~= 0.0474
    naive_avg = (p1 + p2) / 2.0
    assert pytest.approx(naive_avg, abs=1e-3) == 0.523
    assert naive_avg < 0.60  # Severely degraded to coin flip


def test_bounded_floor_missing_token():
    """
    When target token is absent from top_logprobs,
    extract_token_logprob returns a bounded floor based on residual probability
    instead of an arbitrary -99.0 cliff.
    """
    top_logprobs = [
        {"token": "A", "logprob": -0.05},  # p ~= 0.9512
        {"token": "C", "logprob": -3.5},   # p ~= 0.0302
    ]
    # Target 'B' is absent
    floor_lp = extract_token_logprob(top_logprobs, "B")

    # Must NOT be -99.0 cliff
    assert floor_lp > -15.0
    # Must be bounded by min observed or residual logprob
    sum_probs = math.exp(-0.05) + math.exp(-3.5)
    residual = max(1.0 - sum_probs, 1e-6)
    expected_floor = min(-3.5, math.log(residual))
    assert pytest.approx(floor_lp, rel=1e-3) == expected_floor

    # Empty list fallback
    assert extract_token_logprob([], "A") == -10.0


def test_fit_calibration_lowers_brier_score():
    """
    fit_calibration optimizes temperature scaling, lowering
    Brier score on synthetic overconfident predictions.
    """
    # Create synthetic overconfident logits where logits are too large (+- 6.0)
    # causing overconfident probabilities (0.9975 vs true accuracy 0.70)
    samples = []
    # Class 1 samples
    for _ in range(70):
        samples.append((6.0, 6.0, 1))
    # Class 0 samples (overconfident mistakes)
    for _ in range(30):
        samples.append((6.0, 6.0, 0))

    calib = fit_calibration(samples)
    assert calib.optimal_temperature > 1.0  # Temperature should increase to soften logits
    assert calib.brier_score_after < calib.brier_score_before


def test_compile_binary_prompt_identical_static_prefix():
    """
    compile_binary_prompt keeps the [EVALUATION RUBRIC] + [CANDIDATE ARTIFACT]
    prefix 100% byte-identical between Pass 1 and Pass 2 for KV-cache reuse.
    """
    rubric = "Functions must have docstrings and type annotations."
    candidate = "def compute(x: int) -> int:\n    '''Doubles x.'''\n    return x * 2"

    pass1 = compile_binary_prompt(rubric, candidate, pass_token="A", fail_token="B", query_first=True)
    pass2 = compile_binary_prompt(rubric, candidate, pass_token="B", fail_token="A", query_first=True)

    marker = "[DECISION MAPPING]"
    assert marker in pass1
    assert marker in pass2

    prefix1 = pass1.split(marker)[0]
    prefix2 = pass2.split(marker)[0]

    # Must be 100% byte-identical
    assert prefix1 == prefix2
    assert "[EVALUATION RUBRIC]" in prefix1
    assert "[CANDIDATE ARTIFACT]" in prefix1
    assert rubric in prefix1
    assert candidate in prefix1


def test_compile_batch_binary_prompt_identical_static_prefix():
    """
    Verifies that compile_batch_binary_prompt preserves an isolated 100% byte-identical
    [CANDIDATE ARTIFACT] prefix across all 2K passes while maintaining strict rubric isolation.
    """
    rubrics = [
        "Rubric 1: Functions must have docstrings.",
        "Rubric 2: Type annotations required.",
        "Rubric 3: Variable names must be snake_case.",
    ]
    candidate = "def my_func(a: int) -> int:\n    '''Doc.'''\n    return a * 2"

    prompts = []
    for idx in range(len(rubrics)):
        # Pass 1: A=Pass, B=Fail
        p1 = compile_batch_binary_prompt(rubrics, candidate, rubric_index=idx, pass_token="A", fail_token="B")
        # Pass 2: B=Pass, A=Fail
        p2 = compile_batch_binary_prompt(rubrics, candidate, rubric_index=idx, pass_token="B", fail_token="A")

        # Rubric Isolation: Prompt must contain only its target rubric, never other rubrics
        for other_idx, other_r in enumerate(rubrics):
            if other_idx != idx:
                assert other_r not in p1, f"Rubric {other_idx} leaked into prompt for rubric {idx}"
                assert other_r not in p2, f"Rubric {other_idx} leaked into prompt for rubric {idx}"

        prompts.extend([p1, p2])

    assert len(prompts) == 6
    marker = "[EVALUATION RUBRIC]"
    prefixes = [p.split(marker)[0] for p in prompts]

    # All 6 prefixes across all 3 rubrics and both passes must be 100% byte-identical
    for pref in prefixes[1:]:
        assert pref == prefixes[0]

    assert "[CANDIDATE ARTIFACT]" in prefixes[0]
    assert candidate in prefixes[0]

    # Out-of-bounds raises IndexError
    with pytest.raises(IndexError):
        compile_batch_binary_prompt(rubrics, candidate, rubric_index=99)


def test_batch_evaluate_prompts_byte_identical_prefix_math():
    """
    Verifies that LogitGate.evaluate_batch records 6 prompts in StubBackend
    with exact same byte-identical prefix up through [CANDIDATE ARTIFACT].
    """
    backend = StubBackend(default_prob=0.90)
    gate = LogitGate(rubric="Batch evaluation", backend=backend)
    rubrics = [
        "Rubric 1: Documentation complete.",
        "Rubric 2: No side effects.",
        "Rubric 3: Bounded memory usage.",
    ]
    candidate = "def execute(cmd: str) -> None:\n    pass"

    verdicts = gate.evaluate_batch(candidate, rubrics)
    assert len(verdicts) == 3
    assert len(backend.prompt_log) == 6

    marker = "[EVALUATION RUBRIC]"
    prefixes = [p.split(marker)[0] for p in backend.prompt_log]
    for pref in prefixes[1:]:
        assert pref == prefixes[0]
    assert "[CANDIDATE ARTIFACT]" in prefixes[0]
    assert candidate in prefixes[0]


@pytest.mark.asyncio
async def test_batch_aevaluate_prompts_byte_identical_prefix_math():
    """
    Verifies that LogitGate.aevaluate_batch records 6 prompts in StubBackend
    with exact same byte-identical prefix up through [CANDIDATE ARTIFACT].
    """
    backend = StubBackend(default_prob=0.90)
    gate = LogitGate(rubric="Batch evaluation", backend=backend)
    rubrics = [
        "Rubric 1: Documentation complete.",
        "Rubric 2: No side effects.",
        "Rubric 3: Bounded memory usage.",
    ]
    candidate = "def execute(cmd: str) -> None:\n    pass"

    verdicts = await gate.aevaluate_batch(candidate, rubrics)
    assert len(verdicts) == 3
    assert len(backend.prompt_log) == 6

    marker = "[EVALUATION RUBRIC]"
    prefixes = [p.split(marker)[0] for p in backend.prompt_log]
    for pref in prefixes[1:]:
        assert pref == prefixes[0]
    assert "[CANDIDATE ARTIFACT]" in prefixes[0]
    assert candidate in prefixes[0]


def test_compile_score_prompt_identical_static_prefix():
    """
    Verifies that compile_score_prompt retains identical prefix between
    normal and inverted scoring passes for KV-cache reuse.
    """
    rubric = "Rate code modularity"
    candidate = "class Service:\n    pass"

    p_norm = compile_score_prompt(rubric, candidate, min_score=1, max_score=5, inverted=False)
    p_inv = compile_score_prompt(rubric, candidate, min_score=1, max_score=5, inverted=True)

    marker = "[SCORING SCALE]"
    pref_norm = p_norm.split(marker)[0]
    pref_inv = p_inv.split(marker)[0]

    assert pref_norm == pref_inv
    assert "[EVALUATION RUBRIC]" in pref_norm
    assert "[CANDIDATE ARTIFACT]" in pref_norm


class _DummyTokenizer:
    def __init__(self, vocab_map: dict[str, list[int]]) -> None:
        self.vocab_map = vocab_map

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.vocab_map.get(text, [99999, 99998])


def test_surface_token_map_from_tokenizer():
    vocab = {
        "A": [10],
        "a": [11],
        " A": [12],
        " a": [13],
        "B": [20],
        "b": [21],
        " B": [22],
        " b": [23],
    }
    tok = _DummyTokenizer(vocab)
    token_map = SurfaceTokenMap.from_tokenizer(tok, pos_label="A", neg_label="B")

    assert set(token_map.positive_token_ids) == {10, 11, 12, 13}
    assert set(token_map.negative_token_ids) == {20, 21, 22, 23}


def test_surface_token_map_empty_raises():
    tok = _DummyTokenizer({})
    with pytest.raises(ValueError, match="does not map to any single-token ID"):
        SurfaceTokenMap.from_tokenizer(tok, pos_label="XYZ")


def test_marginalize_pole_logprob_tensor_and_fallback():
    import torch

    logits = torch.zeros(30)
    logits[10] = 2.0   # A
    logits[12] = 2.5   # " A"
    logits[20] = 1.0   # B
    logits[22] = 1.2   # " B"

    pos_ids = [10, 12]
    neg_ids = [20, 22]

    lp_pos, lp_neg = marginalize_pole_logprob(logits, pos_ids, neg_ids, temperature=1.0)
    p_pos = math.exp(lp_pos)
    p_neg = math.exp(lp_neg)
    assert pytest.approx(p_pos + p_neg, rel=1e-5) == 1.0
    assert p_pos > p_neg

    logits_list = logits.tolist()
    lp_pos_seq, lp_neg_seq = marginalize_pole_logprob(
        logits_list, pos_ids, neg_ids, temperature=1.0
    )
    assert pytest.approx(lp_pos_seq, rel=1e-5) == lp_pos
    assert pytest.approx(lp_neg_seq, rel=1e-5) == lp_neg


def test_compute_symmetric_debiased_verdict_cancels_bias():
    import torch

    token_map = SurfaceTokenMap(positive_token_ids=(0,), negative_token_ids=(1,))

    logits_pass_first = torch.tensor([5.0, 0.0])
    logits_fail_first = torch.tensor([3.0, 2.0])

    res = compute_symmetric_debiased_verdict(
        logits_pass_first, logits_fail_first, token_map=token_map, temperature=1.0
    )

    expected_prob = 1.0 / (1.0 + math.exp(-2.0))
    assert pytest.approx(res.log_odds_pass, rel=1e-4) == 2.0
    assert pytest.approx(res.positional_bias_magnitude, rel=1e-4) == 3.0
    assert pytest.approx(res.calibrated_prob_pass, rel=1e-4) == expected_prob
    assert 0.0 <= res.shannon_entropy <= 1.0

