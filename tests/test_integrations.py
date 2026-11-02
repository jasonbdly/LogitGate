from __future__ import annotations

import pytest

from logitgate.backends.base import StubBackend
from logitgate.gates.logit_gate import LogitGate
from logitgate.integrations.deepeval import (
    LLMTestCase,
    LogitGateMetric,
    LogitGateSystemOneModel,
    NoulQuestion,
)
from logitgate.integrations.langgraph import as_conditional_edge
from logitgate.integrations.pytest_plugin import check


def test_pytest_plugin_check_sync():
    backend = StubBackend(default_prob=0.95)

    @check(rubric="JSON must have status key", threshold=0.85, backend=backend, ast_check=False)
    def generate_response():
        return '{"status": "ok"}'

    res = generate_response()
    assert res == '{"status": "ok"}'


@pytest.mark.asyncio
async def test_pytest_plugin_check_async():
    backend = StubBackend(default_prob=0.95)

    @check(rubric="JSON must have status key", threshold=0.85, backend=backend, ast_check=False)
    async def generate_response():
        return '{"status": "ok"}'

    res = await generate_response()
    assert res == '{"status": "ok"}'


def test_langgraph_conditional_edge():
    backend_pass = StubBackend(default_prob=0.95)
    gate_pass = LogitGate(rubric="Content is safe", backend=backend_pass)
    router_pass = as_conditional_edge(gate_pass, pass_node="publish", fail_node="escalate", state_key="doc")

    state_ok = {"doc": "Safe documentation content."}
    assert router_pass(state_ok) == "publish"

    backend_fail = StubBackend(default_prob=0.10)
    gate_fail = LogitGate(rubric="Content is safe", backend=backend_fail)
    router_fail = as_conditional_edge(gate_fail, pass_node="publish", fail_node="escalate", state_key="doc")

    state_bad = {"doc": "Malicious payload."}
    assert router_fail(state_bad) == "escalate"


def test_deepeval_system_one_model():
    backend = StubBackend(default_prob=0.92)
    model = LogitGateSystemOneModel(backend=backend)

    questions = {
        "q1": NoulQuestion(instructions="Is artifact compliant?", statement="Is artifact compliant?"),
    }
    answers, _latency = model.decide("Candidate artifact", questions)
    assert "q1" in answers.nouls
    assert answers.nouls["q1"].probability == 0.92


def test_deepeval_metric():
    backend = StubBackend(default_prob=0.95)
    metric = LogitGateMetric(
        rubric="Candidate meets criteria",
        threshold=0.85,
        backend=backend,
    )
    tc = LLMTestCase(input="Query", actual_output="Compliant response")
    score = metric.measure(tc)
    assert score == 0.95
    assert metric.is_successful() is True
