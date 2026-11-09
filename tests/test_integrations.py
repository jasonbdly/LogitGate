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


@pytest.mark.asyncio
async def test_langgraph_choice_gate_node_mapping_and_base_methods():
    from logitgate import ChoiceGate
    from logitgate.integrations.langgraph import as_async_conditional_edge

    # StubBackend returns higher logprob for "A"
    backend = StubBackend(default_prob=0.90)
    choice_gate = ChoiceGate(
        question="Route intent:",
        options={"A": "Billing", "B": "Tech"},
        backend=backend,
        symmetric_debiasing=False,
    )
    edge = choice_gate.as_conditional_edge(
        node_mapping={"A": "billing_node", "B": "tech_node"},
        state_key="query",
    )
    assert edge({"query": "Refund my invoice"}) == "billing_node"

    async_edge = as_async_conditional_edge(
        choice_gate,
        node_mapping={"A": "billing_node", "B": "tech_node"},
        state_key="query",
    )
    assert await async_edge({"query": "Refund my invoice"}) == "billing_node"


def test_deepeval_metric_preset_and_empty_actual_output(monkeypatch: pytest.MonkeyPatch):
    from logitgate import configure, reset_config

    try:
        configure(backend="stub://0.91")
        metric = LogitGateMetric(
            rubric="Check output",
            preset="balanced",
        )
        # Empty actual_output="" must be preserved, not replaced with str(tc)
        tc = LLMTestCase(input="Prompt", actual_output="")
        score = metric.measure(tc)
        assert score == pytest.approx(0.91, rel=1e-2)
    finally:
        reset_config()


def test_pytest_check_with_byo_validator_and_normalize():
    from pydantic import BaseModel

    class ReplySchema(BaseModel):
        status: str

    seen_prompts: list[str] = []

    class _InspectBackend(StubBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens=None, **kwargs):
            seen_prompts.append(prompt)
            return super().query_top_logprobs(prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs)

    backend = _InspectBackend(default_prob=0.95)

    @check(
        rubric="Status must be ok",
        backend=backend,
        ast_check=ReplySchema,
        normalize=True,
    )
    def generate_fenced_reply():
        return "<think>Format JSON</think>\n```json\n{\"status\": \"ok\"}\n```"

    assert "status" in generate_fenced_reply()
    assert seen_prompts
    assert "<think>" not in seen_prompts[0]
    assert "```json" not in seen_prompts[0]

    seen_prompts.clear()

    @check(
        rubric="Must be polite",
        backend=backend,
        normalize=True,
    )
    def generate_normalized_only():
        return "<think>Internal monologue</think>\nThank you for your patience!"

    generate_normalized_only()
    assert seen_prompts
    assert "<think>" not in seen_prompts[0]
    assert "Thank you for your patience!" in seen_prompts[0]


@pytest.mark.asyncio
async def test_deepeval_presets_and_dict_questions():
    backend = StubBackend(default_prob=0.88)
    strict_metric = LogitGateMetric(rubric="Strict check", preset="strict", backend=backend)
    assert strict_metric.threshold == pytest.approx(0.90)
    tc = LLMTestCase(input="Q", actual_output="A")
    score = await strict_metric.a_measure(tc)
    assert score == pytest.approx(0.88, rel=1e-2)
    assert strict_metric.is_successful() is False

    fast_metric = LogitGateMetric(rubric="Fast check", preset="fast_triage", backend=backend)
    assert fast_metric.threshold == pytest.approx(0.80)
    fast_metric.measure(tc)
    assert fast_metric.is_successful() is True

    model = LogitGateSystemOneModel(backend=backend)
    answers, _ = await model.a_decide(
        "Candidate",
        {
            "q1": {"statement": "First rubric"},
            "q2": {"instructions": "Second rubric"},
        },
    )
    assert set(answers.nouls.keys()) == {"q1", "q2"}



