from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from logitgate.backends.base import GenerativeBackend, LogitBackend, StubBackend
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.exceptions import ConfigurationError
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import (
    ChoiceGate,
    LogitGate,
    MultiRubricGate,
    ScoreGate,
)
from logitgate.models import DecisionStatus
from logitgate.snapshot import VerdictSnapshot


def _oai_logprob_response(top_logprobs: list[dict[str, Any]], content: str = "A") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": {
                        "content": [
                            {
                                "token": top_logprobs[0]["token"] if top_logprobs else "",
                                "logprob": top_logprobs[0]["logprob"] if top_logprobs else -1.0,
                                "top_logprobs": top_logprobs,
                            }
                        ]
                    },
                }
            ],
        },
    )


def _mock_http_clients(
    handler: Any, base_url: str = "http://mock.local/v1"
) -> tuple[httpx.Client, httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    sync_client = httpx.Client(transport=transport, base_url=base_url)
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)
    return sync_client, async_client


class _TrackingAsyncLogitBackend(LogitBackend, GenerativeBackend):
    """Async/sync stub recording in-flight concurrency and call order for warmup tests."""

    def __init__(self, *, prefers_warmup: bool = False, delay: float = 0.02) -> None:
        self.model = "tracking-stub"
        self.prefers_sequential_prefix_warmup = prefers_warmup
        self.delay = delay
        self.active_calls = 0
        self.max_concurrent_calls = 0
        self.call_log: list[dict[str, Any]] = []

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.call_log.append({"prompt": prompt, "allowed_tokens": allowed_tokens})
        if allowed_tokens and any(tok.isdigit() for tok in allowed_tokens):
            return [{"token": "5", "logprob": -0.05}, {"token": "1", "logprob": -4.0}]
        if allowed_tokens and len(allowed_tokens) >= 2 and "A" not in allowed_tokens:
            return [{"token": allowed_tokens[0], "logprob": -0.05}, {"token": allowed_tokens[1], "logprob": -4.0}]
        return [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -4.0}]

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.active_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
        try:
            await asyncio.sleep(self.delay)
            return self.query_top_logprobs(prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs)
        finally:
            self.active_calls -= 1

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return "[VERDICT: PASS] ok", 8

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return "[VERDICT: PASS] ok", 8


# ----------------------------------------------------------------------------------------------
# 1. ScoreGate abstention symmetry & cascade routing
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
async def test_score_gate_unobserved_tokens_abstains_sync_and_async(is_async: bool) -> None:
    junk = [{"token": "the", "logprob": -0.1}, {"token": "score", "logprob": -2.5}]
    valid = [{"token": "5", "logprob": -0.05}, {"token": "1", "logprob": -4.0}]
    for fwd, rev in ((junk, valid), (valid, junk), (junk, junk)):
        backend = StubBackend(forward_top_logprobs=fwd, reverse_top_logprobs=rev)
        gate = ScoreGate("Rate quality", min_score=1, max_score=5, threshold=3.5, backend=backend, logit_temperature=1.0)
        v = await gate.aevaluate("candidate") if is_async else gate.evaluate("candidate")
        assert v.status == DecisionStatus.AMBIGUOUS
        assert v.passed is False
        assert v.score is None
        assert v.traces[0].metadata["abstained_reason"] == "unobserved_target_tokens"
        assert "abstained=unobserved_target_tokens" in (v.reason or "")


@pytest.mark.parametrize("is_async", [False, True])
async def test_score_gate_fail_threshold_and_max_score_std(is_async: bool) -> None:
    # 1. fail_threshold band: PASS (>= 4.0), AMBIGUOUS (2.5 < E[S] < 4.0), FAIL (<= 2.5)
    pass_backend = StubBackend(
        forward_top_logprobs=[{"token": "5", "logprob": 0.0}, {"token": "4", "logprob": -2.0}],
        reverse_top_logprobs=[{"token": "1", "logprob": 0.0}, {"token": "2", "logprob": -2.0}],
    )
    mid_backend = StubBackend(
        forward_top_logprobs=[{"token": "3", "logprob": 0.0}, {"token": "4", "logprob": -2.5}],
        reverse_top_logprobs=[{"token": "3", "logprob": 0.0}, {"token": "2", "logprob": -2.5}],
    )
    fail_backend = StubBackend(
        forward_top_logprobs=[{"token": "1", "logprob": 0.0}, {"token": "2", "logprob": -2.0}],
        reverse_top_logprobs=[{"token": "5", "logprob": 0.0}, {"token": "4", "logprob": -2.0}],
    )
    for b, expected_status, expected_pass in (
        (pass_backend, DecisionStatus.PASSED, True),
        (mid_backend, DecisionStatus.AMBIGUOUS, False),
        (fail_backend, DecisionStatus.FAILED, False),
    ):
        g = ScoreGate("Quality", min_score=1, max_score=5, threshold=4.0, fail_threshold=2.5, backend=b, logit_temperature=1.0)
        v = await g.aevaluate("x") if is_async else g.evaluate("x")
        assert v.status == expected_status
        assert v.passed is expected_pass
        assert v.score is not None

    # Boundary: fail_threshold == threshold yields zero ambiguous band on observed tokens
    tight = ScoreGate(
        "Quality", min_score=1, max_score=5, threshold=3.5, fail_threshold=3.5, backend=mid_backend, logit_temperature=1.0
    )
    vt = await tight.aevaluate("x") if is_async else tight.evaluate("x")
    assert vt.status == DecisionStatus.FAILED

    # 2. Bimodal distribution (50% on score 1, 50% on score 5 -> std = 2.0 > 1.2) -> AMBIGUOUS
    bimodal = StubBackend(
        forward_top_logprobs=[{"token": "1", "logprob": 0.0}, {"token": "5", "logprob": 0.0}],
        reverse_top_logprobs=[{"token": "1", "logprob": 0.0}, {"token": "5", "logprob": 0.0}],
    )
    std_gate = ScoreGate(
        "Quality",
        min_score=1,
        max_score=5,
        threshold=2.5,
        max_score_std=1.2,
        backend=bimodal,
        logit_temperature=1.0,
    )
    vs = await std_gate.aevaluate("x") if is_async else std_gate.evaluate("x")
    assert vs.status == DecisionStatus.AMBIGUOUS
    assert vs.passed is False
    assert vs.score is not None and vs.score == pytest.approx(3.0, abs=0.05)
    assert vs.traces[0].metadata["score_std"] > 1.2
    assert "score_std_exceeded" in vs.traces[0].metadata["abstained_reason"]


@pytest.mark.parametrize("is_async", [False, True])
async def test_score_gate_cascade_escalation_and_snapshot_replay(tmp_path: Path, is_async: bool) -> None:
    # Unobserved tokens -> AMBIGUOUS with score=None -> Cascade escalates and does not overwrite Judge score with 3.0
    junk = StubBackend(
        forward_top_logprobs=[{"token": "Sorry", "logprob": -0.01}],
        reverse_top_logprobs=[{"token": "Sorry", "logprob": -0.01}],
    )
    judge_b = StubBackend(diagnostic_text="- Complete defect breakdown.\n[VERDICT: FAIL]", tokens_generated=22)
    sg_junk = ScoreGate("Rate quality", threshold=4.0, fail_threshold=2.5, backend=junk, logit_temperature=1.0)
    cascade_junk = sg_junk >> JudgeGate(backend=judge_b, trigger="on_ambiguous_only", symmetric_debiasing=False)
    vj = await cascade_junk.aevaluate("x") if is_async else cascade_junk.evaluate("x")
    assert vj.triggered_layer == "JudgeGate"
    assert vj.passed is False
    assert len(judge_b.prompt_log) == 1
    assert vj.score != 3.0

    # Decisive FAIL (<= fail_threshold) short-circuits on_ambiguous_only JudgeGate
    low_b = StubBackend(
        forward_top_logprobs=[{"token": "1", "logprob": 0.0}],
        reverse_top_logprobs=[{"token": "5", "logprob": 0.0}],
    )
    judge_skip = StubBackend(diagnostic_text="[VERDICT: PASS]")
    c_fail = ScoreGate("Rate quality", threshold=4.0, fail_threshold=2.5, backend=low_b, logit_temperature=1.0) >> JudgeGate(
        backend=judge_skip, trigger="on_ambiguous_only"
    )
    vf = await c_fail.aevaluate("x") if is_async else c_fail.evaluate("x")
    assert vf.status == DecisionStatus.FAILED
    assert vf.triggered_layer == "ScoreGate"
    assert judge_skip.prompt_log == []

    # Decisive PASS (>= threshold) fast-paths JudgeGate
    high_b = StubBackend(
        forward_top_logprobs=[{"token": "5", "logprob": 0.0}],
        reverse_top_logprobs=[{"token": "1", "logprob": 0.0}],
    )
    c_pass = ScoreGate("Rate quality", threshold=4.0, fail_threshold=2.5, backend=high_b, logit_temperature=1.0) >> JudgeGate(
        backend=judge_skip, trigger="on_ambiguous_only"
    )
    vp = await c_pass.aevaluate("x") if is_async else c_pass.evaluate("x")
    assert vp.passed is True
    assert vp.triggered_layer == "ScoreGate"
    assert judge_skip.prompt_log == []

    # VerdictSnapshot roundtrip for fail_threshold, max_score_std, warm_prefix
    snap_dir = tmp_path / "score_snaps"
    bimodal_b = StubBackend(
        forward_top_logprobs=[{"token": "1", "logprob": 0.0}, {"token": "5", "logprob": 0.0}],
        reverse_top_logprobs=[{"token": "1", "logprob": 0.0}, {"token": "5", "logprob": 0.0}],
    )
    sg = ScoreGate(
        "Rate quality",
        threshold=4.0,
        fail_threshold=2.5,
        max_score_std=1.2,
        warm_prefix=True,
        save_on_fail=True,
        backend=bimodal_b,
        logit_temperature=1.0,
    )
    v_orig = await sg.aevaluate("candidate", snapshot_dir=snap_dir) if is_async else sg.evaluate("candidate", snapshot_dir=snap_dir)
    assert v_orig.status == DecisionStatus.AMBIGUOUS
    snap = VerdictSnapshot.load(snap_dir)
    assert snap.gate_config["fail_threshold"] == 2.5
    assert snap.gate_config["max_score_std"] == 1.2
    assert snap.gate_config["warm_prefix"] is True
    v_replayed = await snap.areplay(backend=bimodal_b) if is_async else snap.replay(backend=bimodal_b)
    assert v_replayed.status == DecisionStatus.AMBIGUOUS
    assert "score_std_exceeded" in v_replayed.traces[0].metadata["abstained_reason"]


# ----------------------------------------------------------------------------------------------
# 2. ChoiceGate abstention, custom keys, save_on_fail, replay, & cascade
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
async def test_choice_gate_abstention_custom_keys_and_snapshots(tmp_path: Path, is_async: bool) -> None:
    # Unobserved target tokens -> choice=None, AMBIGUOUS
    junk = StubBackend(
        forward_top_logprobs=[{"token": "<think>", "logprob": -0.01}],
        reverse_top_logprobs=[{"token": "<think>", "logprob": -0.01}],
    )
    cg_junk = ChoiceGate("Pick route", {"A": "Alpha", "B": "Beta"}, backend=junk, logit_temperature=1.0)
    vj = await cg_junk.aevaluate("x") if is_async else cg_junk.evaluate("x")
    assert vj.status == DecisionStatus.AMBIGUOUS
    assert vj.passed is False
    assert vj.choice is None
    assert vj.traces[0].metadata["abstained_reason"] == "unobserved_target_tokens"

    # min_confidence abstention (P = ~0.55 < 0.70)
    near_tie = StubBackend(
        forward_top_logprobs=[{"token": "A", "logprob": -0.597}, {"token": "B", "logprob": -0.797}],
        reverse_top_logprobs=[{"token": "B", "logprob": -0.597}, {"token": "A", "logprob": -0.797}],
    )
    cg_conf = ChoiceGate(
        "Pick route", {"A": "Alpha", "B": "Beta"}, backend=near_tie, logit_temperature=1.0, min_confidence=0.70
    )
    vc = await cg_conf.aevaluate("x") if is_async else cg_conf.evaluate("x")
    assert vc.status == DecisionStatus.AMBIGUOUS
    assert vc.passed is False
    assert vc.choice == "A"
    assert vc.traces[0].metadata["abstained_reason"].startswith("below_min_confidence")

    # min_margin abstention (P = ~0.55 vs 0.45 -> margin ~0.10 < 0.25)
    cg_margin = ChoiceGate(
        "Pick route", {"A": "Alpha", "B": "Beta"}, backend=near_tie, logit_temperature=1.0, min_margin=0.25
    )
    vm = await cg_margin.aevaluate("x") if is_async else cg_margin.evaluate("x")
    assert vm.status == DecisionStatus.AMBIGUOUS
    assert vm.passed is False
    assert vm.traces[0].metadata["top2_margin"] < 0.25
    assert vm.traces[0].metadata["abstained_reason"].startswith("below_min_margin")

    # Custom non-A/B keys on default StubBackend succeed without false abstention
    cg_custom = ChoiceGate("Approve?", {"X": "Approve", "Y": "Reject"}, backend=StubBackend(default_prob=0.95), logit_temperature=1.0)
    vx = await cg_custom.aevaluate("x") if is_async else cg_custom.evaluate("x")
    assert vx.status == DecisionStatus.PASSED
    assert vx.choice == "X"

    # Cascade(ChoiceGate(min_confidence=0.70) >> JudgeGate): escalates low confidence, preserves fast-path choice
    judge_b = StubBackend(diagnostic_text="- Resolved ambiguous route.\n[VERDICT: PASS]", tokens_generated=12)
    c_escalate = cg_conf >> JudgeGate(backend=judge_b, trigger="on_ambiguous_only", symmetric_debiasing=False)
    ve = await c_escalate.aevaluate("x") if is_async else c_escalate.evaluate("x")
    assert ve.triggered_layer == "JudgeGate"
    assert ve.choice == "A"
    assert len(judge_b.prompt_log) == 1

    # save_on_fail persistence + replay/areplay roundtrip
    snap_dir = tmp_path / "choice_snaps"
    cg_snap = ChoiceGate(
        "Pick route",
        {"A": "Alpha", "B": "Beta"},
        backend=near_tie,
        logit_temperature=1.0,
        min_confidence=0.70,
        min_margin=0.25,
        save_on_fail=True,
        warm_prefix=False,
    )
    vs = await cg_snap.aevaluate("item", snapshot_dir=snap_dir) if is_async else cg_snap.evaluate("item", snapshot_dir=snap_dir)
    assert vs.status == DecisionStatus.AMBIGUOUS
    snap = VerdictSnapshot.load(snap_dir)
    assert snap.gate_config["gate_type"] == "ChoiceGate"
    assert snap.gate_config["min_confidence"] == 0.70
    assert snap.gate_config["min_margin"] == 0.25
    assert snap.gate_config["warm_prefix"] is False
    vr = await snap.areplay(backend=near_tie) if is_async else snap.replay(backend=near_tie)
    assert vr.status == DecisionStatus.AMBIGUOUS
    assert vr.traces[0].metadata["abstained_reason"].startswith("below_min_confidence")


# ----------------------------------------------------------------------------------------------
# 3. OpenAICompatBackend thinking suppression & auto_logit_bias (sync + async)
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
async def test_openai_compat_reasoning_effort_sticky_400_and_suppress_off(is_async: bool) -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "Unknown parameter: reasoning_effort"}})
        return _oai_logprob_response([{"token": "A", "logprob": -0.02}, {"token": "B", "logprob": -4.0}])

    sync_client, async_client = _mock_http_clients(handler)
    backend = OpenAICompatBackend(
        base_url="http://mock.local/v1",
        model="legacy-vllm",
        client=sync_client,
        async_client=async_client,
    )
    p0 = backend.build_query_payload("hello", allowed_tokens=["A", "B"])
    assert p0["reasoning_effort"] == "none"
    assert "Do not think or explain." in p0["messages"][0]["content"]

    r1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert r1[0]["token"] == "A"
    assert backend._supports_reasoning_effort is False
    assert len(seen_payloads) == 2

    # Subsequent query goes straight through in 1 RPC without reasoning_effort
    r2 = await backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
    assert r2[0]["token"] == "A"
    assert len(seen_payloads) == 3
    assert "reasoning_effort" not in seen_payloads[-1]

    # suppress_thinking=False sends neither reasoning_effort nor chat_template_kwargs
    off_backend = OpenAICompatBackend(base_url="http://mock.local/v1", suppress_thinking=False)
    poff = off_backend.build_query_payload("hello", allowed_tokens=["A", "B"])
    assert "reasoning_effort" not in poff
    assert "chat_template_kwargs" not in poff


@pytest.mark.parametrize("trap_token", ["", "<think>", "<|thought|>", "<seed:think>", "<thinking>"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_openai_compat_prefill_trap_fallback_success_and_sticky(trap_token: str, is_async: bool) -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        if body.get("chat_template_kwargs") == {"enable_thinking": False}:
            return _oai_logprob_response([{"token": "A", "logprob": -0.01}, {"token": "B", "logprob": -5.0}])
        lps = [] if trap_token == "" else [{"token": trap_token, "logprob": -0.01}]
        return _oai_logprob_response(lps)

    sync_client, async_client = _mock_http_clients(handler)
    backend = OpenAICompatBackend(
        base_url="http://mock.local/v1",
        model="qwen3-vllm",
        client=sync_client,
        async_client=async_client,
    )
    r1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert r1[0]["token"] == "A"
    assert len(seen_payloads) == 2
    assert "reasoning_effort" not in seen_payloads[1]
    assert seen_payloads[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert backend._use_chat_template_no_think is True
    assert backend._supports_reasoning_effort is False

    # Second query uses clean sticky chat_template_kwargs directly in 1 call
    r2 = await backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
    assert r2[0]["token"] == "A"
    assert len(seen_payloads) == 3
    assert seen_payloads[2].get("chat_template_kwargs") == {"enable_thinking": False}
    assert "reasoning_effort" not in seen_payloads[2]


@pytest.mark.parametrize("fb_status", [400, 200])
@pytest.mark.parametrize("is_async", [False, True])
async def test_openai_compat_failed_chat_template_fallback_disables_sticky(fb_status: int, is_async: bool) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if "chat_template_kwargs" in body:
            if fb_status != 200:
                return httpx.Response(fb_status, json={"error": "unsupported chat_template_kwargs"})
            # Server accepts it but STILL returns prefill trap
            return _oai_logprob_response([{"token": "<think>", "logprob": 0.0}])
        return _oai_logprob_response([{"token": "<think>", "logprob": 0.0}])

    sync_client, async_client = _mock_http_clients(handler)
    backend = OpenAICompatBackend(
        base_url="http://mock.local/v1",
        model="stubborn-thinking",
        client=sync_client,
        async_client=async_client,
    )
    r1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert r1[0]["token"] == "<think>"
    assert backend._supports_chat_template_no_think is False
    assert backend._use_chat_template_no_think is False
    assert len(calls) == 2

    # Next query must NOT re-attempt chat_template_kwargs (prevents repeated 2x amplification)
    await (backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"]) if is_async else asyncio.sleep(0))
    if not is_async:
        backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
    assert len(calls) == 3


@pytest.mark.parametrize("mode", ["vllm", "llamacpp", "404"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_openai_compat_auto_logit_bias_tokenize_and_404(mode: str, is_async: bool) -> None:
    tokenize_reqs: list[tuple[str, dict[str, Any]]] = []
    chat_reqs: list[dict[str, Any]] = []
    tok_map = {"A": 32, "a": 64, " A": 362, "B": 33, "b": 65, " B": 426}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode("utf-8"))
        if path.endswith("/tokenize"):
            tokenize_reqs.append((path, body))
            if mode == "404":
                return httpx.Response(404, json={"error": "not found"})
            if mode == "vllm":
                # Strict vLLM check: reject llama-style 'content' and require add_special_tokens=False
                if "content" in body or body.get("add_special_tokens") is not False:
                    return httpx.Response(422, json={"error": "extra inputs not permitted"})
                text = body["prompt"]
                return httpx.Response(200, json={"tokens": [tok_map.get(text, 999)]})
            # llamacpp /tokenize expects root /tokenize (not /v1/tokenize) and 'content' without BOS
            if path == "/v1/tokenize":
                return httpx.Response(404, json={"error": "wrong path"})
            if "content" not in body:
                return httpx.Response(200, json={"tokens": []})
            text = body["content"]
            return httpx.Response(200, json={"tokens": [tok_map.get(text, 999)]})

        chat_reqs.append(body)
        return _oai_logprob_response([{"token": "A", "logprob": -0.01}, {"token": "B", "logprob": -5.0}])

    sync_client, async_client = _mock_http_clients(handler)
    backend = OpenAICompatBackend(
        base_url="http://mock.local/v1",
        model="test-model",
        auto_logit_bias=True,
        client=sync_client,
        async_client=async_client,
    )
    res1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert res1[0]["token"] == "A"
    tok_count_after_first = len(tokenize_reqs)
    assert tok_count_after_first > 0

    if mode == "404":
        assert backend._auto_logit_bias_supported is False
        assert "logit_bias" not in chat_reqs[-1]
    else:
        assert "logit_bias" in chat_reqs[-1]
        for tid in (32, 33, 64, 65, 362, 426):
            assert chat_reqs[-1]["logit_bias"][str(tid)] == 100
        # Calling again hits cache: zero additional /tokenize calls
        if is_async:
            await backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"])
        else:
            backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
        assert len(tokenize_reqs) == tok_count_after_first
        assert chat_reqs[-1]["logit_bias"] == chat_reqs[0]["logit_bias"]


# ----------------------------------------------------------------------------------------------
# 4. OllamaBackend legacy 400 'think' sticky fallback (sync + async)
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
async def test_ollama_legacy_400_think_fallback(is_async: bool) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "unknown field: think"})
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "A"},
                "logprobs": [{"top_logprobs": [{"token": "A", "logprob": -0.02}, {"token": "B", "logprob": -4.0}]}],
            },
        )

    sync_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.local:11434")
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.local:11434")
    backend = OllamaBackend(
        base_url="http://ollama.local:11434",
        model="qwen2.5:3b",
        client=sync_client,
        async_client=async_client,
    )
    assert backend.prefers_sequential_prefix_warmup is True
    r1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert r1[0]["token"] == "A"
    assert backend._supports_think_param is False
    assert len(calls) == 2

    # Next request goes straight through in 1 call without "think"
    r2 = await backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
    assert r2[0]["token"] == "A"
    assert len(calls) == 3
    assert "think" not in calls[-1]


# ----------------------------------------------------------------------------------------------
# 5. LlamaCppBackend Mode B chat_template_kwargs sticky 400 + Mode A auto_logit_bias
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
async def test_llamacpp_http_chat_template_sticky_400(is_async: bool) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(422, json={"error": "unsupported chat_template_kwargs"})
        return _oai_logprob_response([{"token": "A", "logprob": -0.02}, {"token": "B", "logprob": -4.0}])

    sync_client, async_client = _mock_http_clients(handler, base_url="http://llama.local:8080")
    backend = LlamaCppBackend(
        base_url="http://llama.local:8080",
        client=sync_client,
        async_client=async_client,
    )
    assert backend.prefers_sequential_prefix_warmup is True
    assert backend.build_chat_payload("p1", allowed_tokens=["A", "B"])["chat_template_kwargs"] == {"enable_thinking": False}

    r1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert r1[0]["token"] == "A"
    assert backend._supports_chat_template_kwargs is False
    assert len(calls) == 2

    r2 = await backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
    assert r2[0]["token"] == "A"
    assert len(calls) == 3
    assert "chat_template_kwargs" not in calls[-1]


@pytest.mark.parametrize("is_async", [False, True])
async def test_llamacpp_in_process_auto_logit_bias(is_async: bool) -> None:
    class _FakeLlama:
        def __init__(self) -> None:
            self.chat_calls: list[dict[str, Any]] = []
            self.tokenize_calls = 0

        def tokenize(self, text_bytes: bytes, add_bos: bool = True) -> list[int]:
            self.tokenize_calls += 1
            text = text_bytes.decode("utf-8")
            mapping = {"A": [32], "a": [64], " A": [362], " A"[1:].lower(): [64], " a": [264], "B": [33], "b": [65], " B": [426], " b": [265]}
            return mapping.get(text, [900, 901])

        def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.chat_calls.append(kwargs)
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"top_logprobs": [{"token": "A", "logprob": -0.01}, {"token": "B", "logprob": -5.0}]}
                            ]
                        }
                    }
                ]
            }

    fake_llm = _FakeLlama()
    backend = LlamaCppBackend(llm=fake_llm, auto_logit_bias=True)
    r1 = await backend.aquery_top_logprobs("p1", allowed_tokens=["A", "B"]) if is_async else backend.query_top_logprobs("p1", allowed_tokens=["A", "B"])
    assert r1[0]["token"] == "A"
    assert "logit_bias" in fake_llm.chat_calls[-1]
    assert fake_llm.chat_calls[-1]["logit_bias"]["32"] == 100
    assert fake_llm.chat_calls[-1]["logit_bias"]["33"] == 100
    tok_count = fake_llm.tokenize_calls
    assert tok_count > 0

    # Cached on second call
    if is_async:
        await backend.aquery_top_logprobs("p2", allowed_tokens=["A", "B"])
    else:
        backend.query_top_logprobs("p2", allowed_tokens=["A", "B"])
    assert fake_llm.tokenize_calls == tok_count


# ----------------------------------------------------------------------------------------------
# 6. MultiRubricGate & evaluate_batch fail_fast + snapshot roundtrip
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
async def test_multi_rubric_and_batch_fail_fast(tmp_path: Path, is_async: bool) -> None:
    rubrics = ["r0", "r1", "r2", "r3"]
    # Rubric 0 fails (p=0.05), others would pass if queried
    responses_fail_first = [
        [{"token": "B", "logprob": -0.05}, {"token": "A", "logprob": -3.0}],
        [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.0}],
    ]
    b_fast = StubBackend(logprob_responses=responses_fail_first)
    mg_fast = MultiRubricGate(rubrics, backend=b_fast, logit_temperature=1.0, fail_fast=True, save_on_fail=True)
    v_fast = await mg_fast.aevaluate("code", snapshot_dir=tmp_path / "mg_snap") if is_async else mg_fast.evaluate("code", snapshot_dir=tmp_path / "mg_snap")
    assert v_fast.status == DecisionStatus.FAILED
    assert len(b_fast.prompt_log) == 2
    assert len(v_fast.traces) == 1
    assert "short-circuited at 1/4" in (v_fast.reason or "")

    # AMBIGUOUS on rubric 0 does NOT short-circuit until an actual FAILED (rubric 1)
    responses_ambig_then_fail = [
        [{"token": "A", "logprob": -0.69}, {"token": "B", "logprob": -0.69}],
        [{"token": "A", "logprob": -0.69}, {"token": "B", "logprob": -0.69}],
        [{"token": "B", "logprob": -0.05}, {"token": "A", "logprob": -3.0}],
        [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.0}],
    ]
    b_af = StubBackend(logprob_responses=responses_ambig_then_fail)
    lg = LogitGate("base", backend=b_af, logit_temperature=1.0)
    batch_res = (
        await lg.aevaluate_batch("code", rubrics, fail_fast=True)
        if is_async
        else lg.evaluate_batch("code", rubrics, fail_fast=True)
    )
    assert [v.status for v in batch_res] == [DecisionStatus.AMBIGUOUS, DecisionStatus.FAILED]
    assert len(b_af.prompt_log) == 4

    # Snapshot replay preserves fail_fast=True
    b_replay = StubBackend(logprob_responses=responses_fail_first)
    snap = VerdictSnapshot.load(tmp_path / "mg_snap")
    assert snap.gate_config["fail_fast"] is True
    vr = await snap.areplay(backend=b_replay) if is_async else snap.replay(backend=b_replay)
    assert vr.status == DecisionStatus.FAILED
    assert len(b_replay.prompt_log) == 2


# ----------------------------------------------------------------------------------------------
# 7. warm_prefix async Pass-1-before-Pass-2 ordering across LogitGate, ChoiceGate, ScoreGate
# ----------------------------------------------------------------------------------------------


async def test_warm_prefix_async_pass1_ordering() -> None:
    for gate_factory in (
        lambda b, wp: LogitGate("Has docstring?", backend=b, logit_temperature=1.0, warm_prefix=wp),
        lambda b, wp: ScoreGate("Rate quality", backend=b, logit_temperature=1.0, warm_prefix=wp),
        lambda b, wp: ChoiceGate(
            "Pick one",
            {"A": "One", "B": "Two", "C": "Three"},
            backend=b,
            logit_temperature=1.0,
            permutation_mode="cyclic",
            warm_prefix=wp,
        ),
    ):
        # Backend default prefers_sequential_prefix_warmup=True, warm_prefix=None -> sequential Pass 1 first
        b_seq = _TrackingAsyncLogitBackend(prefers_warmup=True)
        g_seq = gate_factory(b_seq, None)
        await g_seq.aevaluate("candidate")
        # For 2-pass gates max concurrency must be 1; for 3-pass cyclic ChoiceGate Pass 1 finishes alone then 2 permutations run concurrently
        expected_max = 1 if len(b_seq.call_log) == 2 else 2
        assert b_seq.max_concurrent_calls == expected_max

        # Override warm_prefix=False runs all passes concurrently
        b_par = _TrackingAsyncLogitBackend(prefers_warmup=True)
        g_par = gate_factory(b_par, False)
        await g_par.aevaluate("candidate")
        assert b_par.max_concurrent_calls == len(b_par.call_log)

        # Force warm_prefix=True on non-warmup backend
        b_forced = _TrackingAsyncLogitBackend(prefers_warmup=False)
        g_forced = gate_factory(b_forced, True)
        await g_forced.aevaluate("candidate")
        assert b_forced.max_concurrent_calls == expected_max


# ----------------------------------------------------------------------------------------------
# 8. Input validation (ConfigurationError)
# ----------------------------------------------------------------------------------------------


def test_v051_configuration_validation() -> None:
    with pytest.raises(ConfigurationError, match="suppress_thinking"):
        OpenAICompatBackend(suppress_thinking=1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="auto_logit_bias"):
        OpenAICompatBackend(auto_logit_bias="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="suppress_thinking"):
        LlamaCppBackend(suppress_thinking=0)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="auto_logit_bias"):
        LlamaCppBackend(auto_logit_bias=1)  # type: ignore[arg-type]

    stub = StubBackend()
    with pytest.raises(ConfigurationError, match="fail_fast"):
        MultiRubricGate(["r1"], backend=stub, fail_fast="true")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="fail_fast"):
        LogitGate("r1", backend=stub).evaluate_batch("x", ["r1"], fail_fast=1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="fail_fast"):
        asyncio.run(LogitGate("r1", backend=stub).aevaluate_batch("x", ["r1"], fail_fast=1))  # type: ignore[arg-type]

    for bad_bool in ("yes", 1):
        with pytest.raises(ConfigurationError, match="warm_prefix"):
            LogitGate("r", backend=stub, warm_prefix=bad_bool)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="save_on_fail"):
            ChoiceGate("q", {"A": "a", "B": "b"}, backend=stub, save_on_fail=bad_bool)  # type: ignore[arg-type]

    for bad_unit in (0.0, -0.1, 1.01, math.nan, math.inf, True):
        with pytest.raises(ConfigurationError, match="min_confidence"):
            ChoiceGate("q", {"A": "a", "B": "b"}, backend=stub, min_confidence=bad_unit)
        with pytest.raises(ConfigurationError, match="min_margin"):
            ChoiceGate("q", {"A": "a", "B": "b"}, backend=stub, min_margin=bad_unit)

    for bad_fail in (0.5, 4.1, math.nan, math.inf, True):
        with pytest.raises(ConfigurationError, match="fail_threshold"):
            ScoreGate("r", min_score=1, max_score=5, threshold=4.0, fail_threshold=bad_fail, backend=stub)

    for bad_std in (0.0, -1.0, math.nan, math.inf, False):
        with pytest.raises(ConfigurationError, match="max_score_std"):
            ScoreGate("r", min_score=1, max_score=5, threshold=4.0, max_score_std=bad_std, backend=stub)
