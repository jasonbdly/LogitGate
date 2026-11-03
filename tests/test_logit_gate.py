from __future__ import annotations

import math

import pytest

from logitgate.backends.base import StubBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import ChoiceGate, LogitGate, ScoreGate
from logitgate.integrations.deepeval import LogitGateSystemOneModel, NoulQuestion
from logitgate.models import DecisionStatus


def test_logit_gate_threshold_bands():
    # 1. High confidence Pass (prob = 0.95 >= 0.85)
    backend_pass = StubBackend(default_prob=0.95)
    gate_pass = LogitGate(
        rubric="Must be safe.",
        backend=backend_pass,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    v_pass = gate_pass.evaluate("Safe code")
    assert v_pass.passed is True
    assert v_pass.status == DecisionStatus.PASSED
    assert v_pass.probability >= 0.85
    assert len(backend_pass.prompt_log) == 2  # Symmetric debiasing (Pass 1 and Pass 2)

    # 2. Decisive Failure (prob = 0.05 <= 0.20)
    backend_fail = StubBackend(default_prob=0.05)
    gate_fail = LogitGate(
        rubric="Must be safe.",
        backend=backend_fail,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    v_fail = gate_fail.evaluate("Unsafe code")
    assert v_fail.passed is False
    assert v_fail.status == DecisionStatus.FAILED
    assert v_fail.probability <= 0.20

    # 3. Ambiguous band (fail_thresh < prob < pass_thresh, e.g. prob = 0.50)
    backend_ambiguous = StubBackend(default_prob=0.50)
    gate_ambiguous = LogitGate(
        rubric="Must be safe.",
        backend=backend_ambiguous,
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    v_ambiguous = gate_ambiguous.evaluate("Borderline code")
    assert v_ambiguous.passed is False
    assert v_ambiguous.status == DecisionStatus.AMBIGUOUS
    assert 0.20 < v_ambiguous.probability < 0.85


def test_choice_gate():
    # StubBackend returning high prob for token 'B'
    forward_lps = [
        {"token": "A", "logprob": -2.5},
        {"token": "B", "logprob": -0.1},
        {"token": "C", "logprob": -3.0},
    ]
    # In reverse mapping: keys are A, B, C; option B is still in middle (or reversed)
    reverse_lps = [
        {"token": "A", "logprob": -3.0},
        {"token": "B", "logprob": -0.1},
        {"token": "C", "logprob": -2.5},
    ]
    backend = StubBackend(
        forward_top_logprobs=forward_lps,
        reverse_top_logprobs=reverse_lps,
    )
    gate = ChoiceGate(
        question="Select the best architecture:",
        options={
            "A": "Monolith",
            "B": "Modular Microservices",
            "C": "Serverless without state",
        },
        backend=backend,
    )
    verdict = gate.evaluate("Candidate Architecture Doc")
    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED
    assert verdict.choice == "B"
    assert "choice_probabilities" in verdict.__dict__ or verdict.choice_probabilities is not None
    assert verdict.choice_probabilities["B"] > verdict.choice_probabilities["A"]
    assert verdict.choice_probabilities["B"] > verdict.choice_probabilities["C"]


def test_score_gate_expected_value():
    # Discrete distribution: 80% weight on score '4', 20% weight on score '5'
    # E[S] = 4 * 0.8 + 5 * 0.2 = 3.2 + 1.0 = 4.2
    top_lps = [
        {"token": "1", "logprob": -10.0},
        {"token": "2", "logprob": -10.0},
        {"token": "3", "logprob": -10.0},
        {"token": "4", "logprob": math.log(0.80)},
        {"token": "5", "logprob": math.log(0.20)},
    ]
    backend = StubBackend(forward_top_logprobs=top_lps)
    gate = ScoreGate(
        rubric="Rate overall quality 1-5",
        min_score=1,
        max_score=5,
        threshold=4.0,
        backend=backend,
    )
    verdict = gate.evaluate("Good artifact")
    assert verdict.passed is True
    assert pytest.approx(verdict.score, abs=0.05) == 4.2


def test_ollama_backend_payload_think_suppression():
    """
    Verifies OllamaBackend constructs query payloads with
    think=False to suppress reasoning tokens on classification calls.
    """
    backend = OllamaBackend(
        base_url="http://localhost:11434",
        model="deepseek-r1:32b",
    )
    payload = backend.build_query_payload("Test prompt", top_k=10)

    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 10
    assert payload["options"]["num_predict"] == 1
    assert payload["options"]["temperature"] == 0.0
    assert "Do not think or explain" in payload["messages"][0]["content"]


def test_logit_gate_batch_evaluate_prefix_cache_sharing():
    """
    Verifies that evaluate_batch records 6 prompts in StubBackend for 3 rubrics,
    all sharing the exact same byte-identical prefix up through [CANDIDATE ARTIFACT].
    """
    stub = StubBackend(default_prob=0.92)
    gate = LogitGate(rubric="Batch evaluation", backend=stub)
    rubrics = [
        "1. Docstrings required.",
        "2. Strict typing required.",
        "3. Zero warnings.",
    ]
    candidate = "def process_data(payload: dict) -> dict:\n    '''Process data docstring.'''\n    return payload"

    verdicts = gate.evaluate_batch(candidate, rubrics)
    assert len(verdicts) == 3
    assert len(stub.prompt_log) == 6

    # Verify byte-identical prefix across all 6 prompts
    marker = "[DECISION MAPPING — RUBRIC #"
    prefixes = [p.split(marker)[0] for p in stub.prompt_log]
    for pref in prefixes[1:]:
        assert pref == prefixes[0]
    assert "[EVALUATION RUBRICS]" in prefixes[0]
    assert "[CANDIDATE ARTIFACT]" in prefixes[0]
    assert candidate in prefixes[0]
    for r in rubrics:
        assert r in prefixes[0]


@pytest.mark.asyncio
async def test_logit_gate_batch_aevaluate_prefix_cache_sharing():
    """
    Verifies that aevaluate_batch records 6 prompts in StubBackend for 3 rubrics,
    all sharing the exact same byte-identical prefix up through [CANDIDATE ARTIFACT].
    """
    stub = StubBackend(default_prob=0.88)
    gate = LogitGate(rubric="Batch evaluation", backend=stub)
    rubrics = [
        "Rubric A: Clean architecture.",
        "Rubric B: Explicit contracts.",
        "Rubric C: Proper error handling.",
    ]
    candidate = "def run() -> None:\n    '''Run system.'''\n    pass"

    verdicts = await gate.aevaluate_batch(candidate, rubrics)
    assert len(verdicts) == 3
    assert len(stub.prompt_log) == 6

    marker = "[DECISION MAPPING — RUBRIC #"
    prefixes = [p.split(marker)[0] for p in stub.prompt_log]
    for pref in prefixes[1:]:
        assert pref == prefixes[0]
    assert "[CANDIDATE ARTIFACT]" in prefixes[0]
    assert candidate in prefixes[0]


@pytest.mark.asyncio
async def test_choice_gate_aevaluate_and_temperature():
    forward_lps = [
        {"token": "A", "logprob": -2.0},
        {"token": "B", "logprob": -0.2},
        {"token": "C", "logprob": -3.0},
    ]
    reverse_lps = [
        {"token": "A", "logprob": -3.0},
        {"token": "B", "logprob": -0.2},
        {"token": "C", "logprob": -2.0},
    ]
    backend = StubBackend(
        forward_top_logprobs=forward_lps,
        reverse_top_logprobs=reverse_lps,
    )
    # Default temperature = 1.0
    gate = ChoiceGate(
        question="Select optimal design:",
        options={"A": "Option 1", "B": "Option 2", "C": "Option 3"},
        backend=backend,
        logit_temperature=1.0,
    )
    verdict = await gate.aevaluate("Candidate design doc")
    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED
    assert verdict.choice == "B"
    assert verdict.choice_probabilities["B"] > 0.60

    # Low temperature = 0.1 (sharpens probabilities towards 1.0)
    backend_sharp = StubBackend(
        forward_top_logprobs=forward_lps,
        reverse_top_logprobs=reverse_lps,
    )
    gate_sharp = ChoiceGate(
        question="Select optimal design:",
        options={"A": "Option 1", "B": "Option 2", "C": "Option 3"},
        backend=backend_sharp,
        logit_temperature=0.1,
    )
    verdict_sharp = await gate_sharp.aevaluate("Candidate design doc")
    assert verdict_sharp.choice == "B"
    assert verdict_sharp.choice_probabilities["B"] > verdict.choice_probabilities["B"]
    assert verdict_sharp.choice_probabilities["B"] > 0.99


@pytest.mark.asyncio
async def test_score_gate_aevaluate():
    top_lps = [
        {"token": "1", "logprob": -10.0},
        {"token": "2", "logprob": -10.0},
        {"token": "3", "logprob": -10.0},
        {"token": "4", "logprob": math.log(0.70)},
        {"token": "5", "logprob": math.log(0.30)},
    ]
    backend = StubBackend(forward_top_logprobs=top_lps)
    gate = ScoreGate(
        rubric="Rate overall quality 1-5",
        min_score=1,
        max_score=5,
        threshold=4.0,
        backend=backend,
    )
    verdict = await gate.aevaluate("Good artifact")
    assert verdict.passed is True
    assert pytest.approx(verdict.score, abs=0.05) == 4.3


def test_score_gate_symmetric_debiasing_and_temperature():
    # Pass 1: standard scale (1..5) where 5 is highest quality
    top1 = [
        {"token": "1", "logprob": -10.0},
        {"token": "2", "logprob": -10.0},
        {"token": "3", "logprob": -10.0},
        {"token": "4", "logprob": -1.6094},  # p ~ 0.20
        {"token": "5", "logprob": -0.2231},  # p ~ 0.80
    ]
    # Pass 2: inverted scale (1..5) where 1 is highest quality
    # Quality score 5 maps to token: 5 + 1 - 5 = 1
    # Quality score 4 maps to token: 5 + 1 - 4 = 2
    top2 = [
        {"token": "1", "logprob": -0.2231},  # quality 5
        {"token": "2", "logprob": -1.6094},  # quality 4
        {"token": "3", "logprob": -10.0},
        {"token": "4", "logprob": -10.0},
        {"token": "5", "logprob": -10.0},
    ]

    backend = StubBackend(logprob_responses=[top1, top2])
    gate = ScoreGate(
        rubric="Code quality score",
        min_score=1,
        max_score=5,
        threshold=4.5,
        backend=backend,
        symmetric_debiasing=True,
        logit_temperature=1.0,
    )
    verdict = gate.evaluate("High quality code")
    assert verdict.passed is True
    assert pytest.approx(verdict.score, abs=0.05) == 4.8
    assert len(backend.prompt_log) == 2

    # Verify both prompts share identical prefix up through candidate
    marker = "[SCORING SCALE]"
    assert backend.prompt_log[0].split(marker)[0] == backend.prompt_log[1].split(marker)[0]

    # Test temperature scaling: low temp sharpens score toward 5.0
    backend_cold = StubBackend(logprob_responses=[top1, top2])
    gate_cold = ScoreGate(
        rubric="Code quality score",
        min_score=1,
        max_score=5,
        threshold=4.5,
        backend=backend_cold,
        symmetric_debiasing=True,
        logit_temperature=0.2,
    )
    verdict_cold = gate_cold.evaluate("High quality code")
    assert verdict_cold.score > verdict.score
    assert pytest.approx(verdict_cold.score, abs=0.02) == 5.0


@pytest.mark.asyncio
async def test_score_gate_symmetric_debiasing_async():
    top1 = [
        {"token": "1", "logprob": -10.0},
        {"token": "2", "logprob": -10.0},
        {"token": "3", "logprob": -10.0},
        {"token": "4", "logprob": -1.6094},
        {"token": "5", "logprob": -0.2231},
    ]
    top2 = [
        {"token": "1", "logprob": -0.2231},
        {"token": "2", "logprob": -1.6094},
        {"token": "3", "logprob": -10.0},
        {"token": "4", "logprob": -10.0},
        {"token": "5", "logprob": -10.0},
    ]

    backend = StubBackend(logprob_responses=[top1, top2])
    gate = ScoreGate(
        rubric="Code quality score",
        min_score=1,
        max_score=5,
        threshold=4.5,
        backend=backend,
        symmetric_debiasing=True,
        logit_temperature=1.0,
    )
    verdict = await gate.aevaluate("High quality code")
    assert verdict.passed is True
    assert pytest.approx(verdict.score, abs=0.05) == 4.8
    assert len(backend.prompt_log) == 2


@pytest.mark.asyncio
async def test_ast_gate_aevaluate():
    gate = ASTGate(syntax=True, forbidden_imports=["os", "subprocess"])
    # 1. Valid code passes
    v_ok = await gate.aevaluate("def add(a: int, b: int) -> int:\n    return a + b")
    assert v_ok.passed is True
    assert v_ok.status == DecisionStatus.PASSED

    # 2. Syntax error fails
    v_syntax = await gate.aevaluate("def invalid_syntax(:")
    assert v_syntax.passed is False
    assert v_syntax.status == DecisionStatus.FAILED
    assert "Syntax error" in v_syntax.reason

    # 3. Forbidden import fails
    v_import = await gate.aevaluate("import os\nos.system('echo hi')")
    assert v_import.passed is False
    assert v_import.status == DecisionStatus.FAILED
    assert "Forbidden import" in v_import.reason


@pytest.mark.asyncio
async def test_deepeval_system_one_multi_question_batching():
    backend = StubBackend(default_prob=0.90)
    model = LogitGateSystemOneModel(
        model_name="test-model",
        backend=backend,
        symmetric=True,
    )
    questions = {
        "q1": NoulQuestion(instructions="Rubric 1: Safe code."),
        "q2": NoulQuestion(instructions="Rubric 2: Typed code."),
    }
    # Sync decide
    answers, _cost = model.decide("def foo(): pass", questions)
    assert len(answers.nouls) == 2
    assert "q1" in answers.nouls
    assert "q2" in answers.nouls
    assert len(backend.prompt_log) == 4  # 2 questions * 2 passes
    marker = "[DECISION MAPPING — RUBRIC #"
    prefixes = [p.split(marker)[0] for p in backend.prompt_log]
    for pref in prefixes[1:]:
        assert pref == prefixes[0]

    # Async a_decide
    backend.prompt_log.clear()
    async_answers, _async_cost = await model.a_decide("def foo(): pass", questions)
    assert len(async_answers.nouls) == 2
    assert len(backend.prompt_log) == 4
    prefixes = [p.split(marker)[0] for p in backend.prompt_log]
    for pref in prefixes[1:]:
        assert pref == prefixes[0]
