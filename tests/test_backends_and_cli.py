from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from logitgate.backends.base import resolve_logit_backend, strip_reasoning_blocks
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.cli import main

# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------

def test_strip_reasoning_blocks() -> None:
    raw = "<think>Let me ponder this question deeply...</think>Final answer is A."
    assert strip_reasoning_blocks(raw) == "Final answer is A."

    multiline = "<think>\nline 1\nline 2\n</think>\nYes, approved."
    assert strip_reasoning_blocks(multiline) == "Yes, approved."

    no_think = "Standard answer without thinking."
    assert strip_reasoning_blocks(no_think) == "Standard answer without thinking."

    # Unclosed <think> truncated mid-thought
    unclosed = "<think>Pondering that gets cut off due to max_tokens"
    assert strip_reasoning_blocks(unclosed) == ""

    # Unclosed <think> with text before it
    with_pre = "Prefix answer.<think>Cut off reasoning..."
    assert strip_reasoning_blocks(with_pre) == "Prefix answer."

    # Closed block followed by unclosed block
    closed_then_unclosed = "<think>First thought</think>Middle answer<think>Second thought cut off"
    assert strip_reasoning_blocks(closed_then_unclosed) == "Middle answer"


# ---------------------------------------------------------------------------
# OllamaBackend Tests
# ---------------------------------------------------------------------------

def test_ollama_backend_sync_and_reuse() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        data = json.loads(request.content.decode("utf-8"))
        if "/api/chat" in str(request.url):
            if "options" in data and data["options"].get("num_predict") == 1:
                # query_top_logprobs
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": "A",
                            "logprobs": [
                                {
                                    "top_logprobs": [
                                        {"token": "A", "logprob": -0.05},
                                        {"token": "B", "logprob": -3.0},
                                    ]
                                }
                            ],
                        }
                    },
                )
            else:
                # generate
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": "<think>Thinking trace...</think>Verdict: PASS",
                        },
                        "eval_count": 12,
                    },
                )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    backend = OllamaBackend(client=client)

    # 1. query_top_logprobs
    lps = backend.query_top_logprobs("Is this good?")
    assert len(lps) == 2
    assert lps[0]["token"] == "A"
    assert math.isclose(lps[0]["logprob"], -0.05)

    # 2. Client reuse
    assert backend.client is client
    lps2 = backend.query_top_logprobs("Another prompt")
    assert len(lps2) == 2
    assert call_count == 2

    # 3. generate with <think> stripping
    text, tokens = backend.generate("Explain why")
    assert text == "Verdict: PASS"
    assert tokens == 12

    # 4. Context manager & close
    backend.close()
    assert client.is_closed


@pytest.mark.asyncio
async def test_ollama_backend_async_and_reuse() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        data = json.loads(request.content.decode("utf-8"))
        if "options" in data and data["options"].get("num_predict") == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "B",
                        "logprobs": [
                            {
                                "top_logprobs": [
                                    {"token": "B", "logprob": -0.1},
                                    {"token": "A", "logprob": -2.5},
                                ]
                            }
                        ],
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>Pondering...</think>Diagnostic critique.",
                },
                "eval_count": 8,
            },
        )

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    backend = OllamaBackend(async_client=async_client)

    # 1. aquery_top_logprobs
    lps = await backend.aquery_top_logprobs("Check async")
    assert len(lps) == 2
    assert lps[0]["token"] == "B"

    # 2. Reused client
    assert backend.async_client is async_client
    await backend.aquery_top_logprobs("Check async 2")
    assert call_count == 2

    # 3. agenerate
    text, tokens = await backend.agenerate("Explain")
    assert text == "Diagnostic critique."
    assert tokens == 8

    # 4. aclose
    await backend.aclose()
    assert async_client.is_closed


def test_ollama_backend_generate_auto_recovery_mid_thought_truncation() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            # First request: consumed 50 tokens inside <think>, truncated mid-thought
            assert payload.get("think") is True
            assert payload.get("options", {}).get("num_predict") == 50
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>I am analyzing whether the candidate meets the rubric and considering edge cases",
                    },
                    "eval_count": 50,
                },
            )
        else:
            # Second request: True continuation with closed </think> and think=False
            assert payload.get("think") is False
            assert len(payload.get("messages", [])) == 2
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][1]["role"] == "assistant"
            assert payload["messages"][1]["content"].endswith("</think>\n")
            assert "<think>I am analyzing" in payload["messages"][1]["content"]
            assert payload.get("options", {}).get("num_predict") == 50
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "[VERDICT: FAIL] The candidate failed the rubric requirements.",
                    },
                    "eval_count": 18,
                },
            )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    backend = OllamaBackend(client=client)

    text, tokens = backend.generate("Explain rubric", max_tokens=50, think=True)
    assert text == "[VERDICT: FAIL] The candidate failed the rubric requirements."
    assert tokens == 68  # 50 + 18 total tokens
    assert len(captured_payloads) == 2
    assert captured_payloads[0]["think"] is True
    assert captured_payloads[1]["think"] is False
    assert captured_payloads[1]["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_ollama_backend_agenerate_auto_recovery() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "<think>Async reasoning that was truncated...",
                    },
                    "eval_count": 40,
                },
            )
        else:
            assert payload.get("think") is False
            assert len(payload.get("messages", [])) == 2
            assert payload["messages"][1]["role"] == "assistant"
            assert payload["messages"][1]["content"].endswith("</think>\n")
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "[VERDICT: PASS] All criteria verified.",
                    },
                    "eval_count": 15,
                },
            )

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    backend = OllamaBackend(async_client=async_client)

    text, tokens = await backend.agenerate("Explain async", max_tokens=40, think=True)
    assert text == "[VERDICT: PASS] All criteria verified."
    assert tokens == 55
    assert len(captured_payloads) == 2
    assert captured_payloads[0]["think"] is True
    assert captured_payloads[1]["think"] is False


def test_ollama_backend_think_defaults_with_max_tokens() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Sample response"},
                "eval_count": 10,
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    backend = OllamaBackend(client=client)

    # 1. max_tokens=None -> default think=True
    backend.generate("prompt 1", max_tokens=None)
    assert captured_payloads[-1]["think"] is True

    # 2. max_tokens=100 -> default think=False
    backend.generate("prompt 2", max_tokens=100)
    assert captured_payloads[-1]["think"] is False

    # 3. max_tokens=100 with explicit think=True -> think=True
    backend.generate("prompt 3", max_tokens=100, think=True)
    assert captured_payloads[-1]["think"] is True


# ---------------------------------------------------------------------------
# OpenAICompatBackend Tests
# ---------------------------------------------------------------------------

def test_openai_compat_backend_sync_and_logit_bias() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        if payload.get("max_tokens") == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "A"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "A",
                                        "logprob": -0.02,
                                        "top_logprobs": [
                                            {"token": "A", "logprob": -0.02},
                                            {"token": "B", "logprob": -3.9},
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>DeepSeek reasoning trace</think>OpenAI PASS",
                        }
                    }
                ],
                "usage": {"completion_tokens": 15},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    backend = OpenAICompatBackend(
        base_url="https://api.together.xyz/v1",
        model="deepseek-ai/DeepSeek-R1",
        client=client,
        logit_bias={"32": 100},
    )

    # 1. query_top_logprobs includes logit_bias
    lps = backend.query_top_logprobs("Validate", allowed_tokens=["A", "B"])
    assert len(lps) == 2
    assert lps[0]["token"] == "A"
    assert captured_payloads[0]["logit_bias"] == {"32": 100}

    # 2. generate with <think> stripping
    text, tokens = backend.generate("Evaluate code")
    assert text == "OpenAI PASS"
    assert tokens == 15

    # 3. Client reuse
    assert backend.client is client
    backend.close()
    assert client.is_closed


@pytest.mark.asyncio
async def test_openai_compat_backend_async() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if payload.get("max_tokens") == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "logprobs": {
                                "content": [
                                    {
                                        "top_logprobs": [
                                            {"token": "B", "logprob": -0.15},
                                            {"token": "A", "logprob": -2.0},
                                        ]
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>QwQ trace</think>Async Passed",
                        }
                    }
                ],
                "usage": {"completion_tokens": 20},
            },
        )

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    backend = OpenAICompatBackend(async_client=async_client)

    lps = await backend.aquery_top_logprobs("Async check")
    assert len(lps) == 2
    assert lps[0]["token"] == "B"

    text, tokens = await backend.agenerate("Async review")
    assert text == "Async Passed"
    assert tokens == 20

    await backend.aclose()
    assert async_client.is_closed


# ---------------------------------------------------------------------------
# LlamaCppBackend Tests (Mode A: In-Process & Mode B: HTTP)
# ---------------------------------------------------------------------------

def test_llamacpp_backend_in_process_mode() -> None:
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        # Call 1: query_top_logprobs
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "A"},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "A", "logprob": -0.04},
                                    {"token": "B", "logprob": -3.4},
                                ]
                            }
                        ]
                    },
                }
            ]
        },
        # Call 2: generate
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>In-process think</think>Mode A Pass",
                    }
                }
            ],
            "usage": {"completion_tokens": 5},
        },
    ]

    backend = LlamaCppBackend(llm=mock_llm)

    lps = backend.query_top_logprobs("Prompt")
    assert len(lps) == 2
    assert lps[0]["token"] == "A"
    assert math.isclose(lps[0]["logprob"], -0.04)

    text, tokens = backend.generate("Prompt gen")
    assert text == "Mode A Pass"
    assert tokens == 5


@pytest.mark.asyncio
async def test_llamacpp_backend_in_process_mode_async() -> None:
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "B"},
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "B", "logprob": -0.2},
                                {"token": "A", "logprob": -1.8},
                            ]
                        }
                    ]
                },
            }
        ]
    }

    backend = LlamaCppBackend(llm=mock_llm)
    lps = await backend.aquery_top_logprobs("Prompt async")
    assert len(lps) == 2
    assert lps[0]["token"] == "B"


def test_llamacpp_backend_http_mode_v1_and_fallback() -> None:
    # 1. Test normal /v1/chat/completions endpoint
    def handler_v1(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/v1/chat/completions" in url_str:
            payload = json.loads(request.content.decode("utf-8"))
            if payload.get("max_tokens") == 1:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "logprobs": {
                                    "content": [
                                        {
                                            "top_logprobs": [
                                                {"token": "A", "logprob": -0.08},
                                                {"token": "B", "logprob": -2.8},
                                            ]
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "<think>Trace</think>Server Pass",
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 7},
                },
            )
        return httpx.Response(404)

    client_v1 = httpx.Client(transport=httpx.MockTransport(handler_v1))
    backend_v1 = LlamaCppBackend(base_url="http://localhost:8080", client=client_v1)
    lps = backend_v1.query_top_logprobs("Test v1")
    assert len(lps) == 2
    assert lps[0]["token"] == "A"
    text, tokens = backend_v1.generate("Gen v1")
    assert text == "Server Pass"
    assert tokens == 7

    # 2. Test fallback to /completion when /v1/chat/completions returns 404
    def handler_fallback(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/v1/chat/completions" in url_str:
            return httpx.Response(404)
        if "/completion" in url_str:
            payload = json.loads(request.content.decode("utf-8"))
            if payload.get("n_predict") == 1:
                return httpx.Response(
                    200,
                    json={
                        "content": "A",
                        "completion_probabilities": [
                            {
                                "content": "A",
                                "probs": [
                                    {"tok_str": "A", "prob": 0.92},
                                    {"tok_str": "B", "prob": 0.08},
                                ],
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "content": "<think>Fallback think</think>Legacy completion ok",
                    "tokens_predicted": 11,
                },
            )
        return httpx.Response(404)

    client_fb = httpx.Client(transport=httpx.MockTransport(handler_fallback))
    backend_fb = LlamaCppBackend(base_url="http://localhost:8080", client=client_fb)
    lps_fb = backend_fb.query_top_logprobs("Fallback test")
    assert len(lps_fb) == 2
    assert lps_fb[0]["token"] == "A"
    assert math.isclose(lps_fb[0]["logprob"], math.log(0.92))
    assert math.isclose(lps_fb[1]["logprob"], math.log(0.08))

    text_fb, tokens_fb = backend_fb.generate("Fallback gen")
    assert text_fb == "Legacy completion ok"
    assert tokens_fb == 11


def test_resolve_logit_backend_llamacpp() -> None:
    backend = resolve_logit_backend("llamacpp://localhost:8080/model.gguf")
    assert isinstance(backend, LlamaCppBackend)
    assert backend.base_url == "http://localhost:8080"
    assert backend.model == "model.gguf"


# ---------------------------------------------------------------------------
# CLI Entrypoint Tests
# ---------------------------------------------------------------------------

def test_cli_eval_passing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([
        "eval",
        "--rubric",
        "Is it valid?",
        "--candidate",
        "def foo(): return 1",
        "--ast-syntax",
        "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["passed"] is True
    assert data["status"] == "passed"
    assert len(data["traces"]) >= 1


def test_cli_eval_syntax_failure(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([
        "eval",
        "--rubric",
        "Is it valid Python?",
        "--candidate",
        "def foo(: syntax error",
        "--ast-syntax",
        "--json",
    ])
    assert code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["passed"] is False
    assert data["status"] == "failed"
    assert data["triggered_layer"] == "ASTGate"


def test_cli_eval_human_output_and_file(
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_file = tmp_path / "candidate.py"
    test_file.write_text("x = 42\n", encoding="utf-8")

    code = main([
        "eval",
        "--rubric",
        "Does x equal 42?",
        "--file",
        str(test_file),
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "Verdict: PASSED" in captured.out


def test_cli_explain_json_and_human(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from logitgate.models import DecisionStatus, LayerTrace, Verdict
    from logitgate.snapshot import VerdictSnapshot

    v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.12,
        triggered_layer="LogitGate",
        total_latency_ms=45.0,
        total_tokens_generated=0,
        reason="LogitGate prob=0.1200",
        traces=[
            LayerTrace(
                layer_name="LogitGate",
                status=DecisionStatus.FAILED,
                latency_ms=45.0,
                calibrated_prob=0.12,
            )
        ],
    )
    snap = VerdictSnapshot(
        test_name="test_bad_app",
        rubric="No hardcoded secrets",
        candidate="API_KEY = 'secret123'",
        verdict=v,
    )
    snap_dir = tmp_path / "failures"
    saved_path = snap.save(directory=snap_dir)

    # 1. explain with exact file and --json
    code = main([
        "explain",
        str(saved_path),
        "--backend",
        "stub://default",
        "--json",
    ])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["triggered_layer"] == "JudgeGate"
    assert len(data["traces"]) == 2  # LogitGate trace + JudgeGate trace

    # 2. explain with directory (resolves latest.json) and human output
    code2 = main([
        "explain",
        str(snap_dir),
        "--backend",
        "stub://default",
    ])
    assert code2 == 0
    captured2 = capsys.readouterr()
    assert "LOGITGATE FAILURE REPLAY & EXPLANATION" in captured2.out
    assert "Snapshot ID:" in captured2.out
    assert "test_bad_app" in captured2.out
    assert "Diagnostic Critique:" in captured2.out

    # 3. explain with missing path
    code3 = main([
        "explain",
        str(tmp_path / "nonexistent"),
    ])
    assert code3 == 1
