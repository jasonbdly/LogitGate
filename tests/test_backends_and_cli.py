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
