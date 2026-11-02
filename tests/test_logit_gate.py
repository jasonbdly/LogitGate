from __future__ import annotations

import math
import pytest
from logitgate.backends.base import StubBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.gates.logit_gate import LogitGate, ChoiceGate, ScoreGate
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
    Tests Verifies OllamaBackend constructs query payloads with
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
