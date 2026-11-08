from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from logitgate.backends.base import resolve_logit_backend, strip_reasoning_blocks
from logitgate.backends.dory_recurrent import (
    execute_adaptive_recurrent_loops,
)
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.backends.pytorch_native import PrefixBroadcastLogitEvaluator
from logitgate.cli import main
from logitgate.debiasing.permutation import DebiasedLogitResult, SurfaceTokenMap
from logitgate.hardware.memory_guard import PCIeSpilloverGuard, VRAMWatermarkStatus
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import VerdictSnapshot

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


# ---------------------------------------------------------------------------
# RecurrentPyTorchBackend (Elastic Layer Looping / Dory) Tests
# ---------------------------------------------------------------------------

def test_recurrent_backend_validation_and_resolution() -> None:
    from logitgate.backends.recurrent import RecurrentPyTorchBackend

    # Invalid num_loops
    with pytest.raises(ValueError, match="num_loops must be >= 1"):
        RecurrentPyTorchBackend(num_loops=0)

    # Resolution via URI
    backend = resolve_logit_backend("recurrent://nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", num_loops=4)
    assert isinstance(backend, RecurrentPyTorchBackend)
    assert backend.model_name == "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
    assert backend.num_loops == 4
    assert "num_loops=4" in repr(backend)


def test_recurrent_backend_mock_inference() -> None:
    from logitgate.backends.recurrent import RecurrentPyTorchBackend

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock tokenizer output
    mock_inputs = {"input_ids": MagicMock()}
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.encode.side_effect = lambda tok, **kwargs: [65] if "A" in tok else [66]
    mock_tokenizer.decode.side_effect = lambda ids, **kwargs: "A" if ids == [65] else "B"

    # Mock model forward output with logits
    mock_outputs = MagicMock()
    # 100 vocab logits where index 65 has high value, 66 has lower value
    fake_logits = [0.0] * 100
    fake_logits[65] = 4.0
    fake_logits[66] = -1.0
    mock_outputs.logits = [[fake_logits]]
    mock_model.return_value = mock_outputs

    backend = RecurrentPyTorchBackend(
        model_name="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        num_loops=3,
        model=mock_model,
        tokenizer=mock_tokenizer,
        device="cpu",
    )

    logprobs = backend.query_top_logprobs("Test prompt", allowed_tokens=["A", "B"])
    assert len(logprobs) == 2
    assert logprobs[0]["token"] == "A"
    assert logprobs[1]["token"] == "B"
    assert logprobs[0]["logprob"] > logprobs[1]["logprob"]

    # Verify num_loops parameter was passed to model forward
    mock_model.assert_called_once()
    _, call_kwargs = mock_model.call_args
    assert call_kwargs.get("num_loops") == 3

    # Mock generation
    mock_gen_out = MagicMock()
    mock_gen_out.__getitem__.return_value = [65, 66]
    mock_model.generate.return_value = mock_gen_out
    mock_inputs["input_ids"].shape = [1, 5]

    text, _n_tok = backend.generate("Prompt")
    assert isinstance(text, str)
    _, gen_kwargs = mock_model.generate.call_args
    assert gen_kwargs.get("num_loops") == 3


# ---------------------------------------------------------------------------
# PrefixBroadcastLogitEvaluator Tests
# ---------------------------------------------------------------------------

def test_prefix_broadcast_evaluator_geometry():
    prefix, s_fwd, s_perm = PrefixBroadcastLogitEvaluator.build_inverted_prefix_and_suffixes(
        rubric="Candidate must be factual.",
        candidate="Paris is the capital of France.",
    )
    assert "<|system|>" in prefix
    assert "<|rubric|>" in prefix
    assert "Candidate must be factual." in prefix
    assert "<|candidate|>" in prefix
    assert "Paris is the capital of France." in prefix

    assert "Map 'A' = PASS" in s_fwd
    assert "Map 'B' = FAIL" in s_fwd
    assert "Map 'A' = FAIL" in s_perm
    assert "Map 'B' = PASS" in s_perm


def test_prefix_broadcast_evaluator_mock_execution():
    torch = pytest.importorskip("torch")

    class MockModelOutput:
        def __init__(self, logits, past_key_values=None):
            self.logits = logits
            self.past_key_values = past_key_values

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eval_call_count = 0

        def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
            self.eval_call_count += 1
            b, s = input_ids.shape
            v = 100
            dummy_logits = torch.randn(b, s, v)
            kv = tuple(
                (torch.randn(b, 2, 4, 16), torch.randn(b, 2, 4, 16)) for _ in range(2)
            )
            return MockModelOutput(dummy_logits, past_key_values=kv)

    class MockTokenizer:
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            tokens = [hash(w) % 100 for w in text.split()]
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

    mock_model = MockModel()
    mock_tok = MockTokenizer()
    evaluator = PrefixBroadcastLogitEvaluator(
        model=mock_model,
        tokenizer=mock_tok,
        device=torch.device("cpu"),
    )

    l1, l2 = evaluator.evaluate_single_pass_permuted(
        rubric="Must be safe.",
        candidate="Clean response.",
    )

    assert l1.shape == (100,)
    assert l2.shape == (100,)
    assert mock_model.eval_call_count == 2  # Call 1: prefix (B=1), Call 2: suffixes (B=2)


def test_prefix_broadcast_evaluator_dynamic_cache():
    torch = pytest.importorskip("torch")

    class DummyDynamicCache:
        def __init__(self):
            self.interleaved = False

        def batch_repeat_interleave(self, repeats):
            self.interleaved = True
            return self

    class MockModel(torch.nn.Module):
        def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
            b, s = input_ids.shape
            v = 50
            cache = past_key_values or DummyDynamicCache()
            return MagicMock(logits=torch.randn(b, s, v), past_key_values=cache)

    class MockTokenizer:
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            tokens = [hash(w) % 50 for w in text.split()]
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

    evaluator = PrefixBroadcastLogitEvaluator(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        device="cpu",
    )
    l1, l2 = evaluator.evaluate_single_pass_permuted(
        rubric="Test",
        candidate="Test candidate",
    )
    assert l1.shape == (50,)
    assert l2.shape == (50,)


def test_prefix_broadcast_evaluator_evaluate_debiased():
    torch = pytest.importorskip("torch")

    class MockModel(torch.nn.Module):
        def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
            b, s = input_ids.shape
            logits = torch.zeros(b, s, 10)
            if b == 2:
                # pass 1: pole A (id 0) high
                logits[0, -1, 0] = 3.0
                logits[0, -1, 1] = 0.0
                # pass 2: pole B (id 1) high
                logits[1, -1, 0] = 0.0
                logits[1, -1, 1] = 3.0
            return MagicMock(logits=logits, past_key_values=((torch.zeros(1, 2, 4, 16), torch.zeros(1, 2, 4, 16)),))

    class MockTokenizer:
        def encode(self, text, add_special_tokens=True, return_tensors=None):
            if return_tensors == "pt":
                return torch.tensor([[0, 1]], dtype=torch.long)
            return [0, 1]

    evaluator = PrefixBroadcastLogitEvaluator(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        device="cpu",
    )
    token_map = SurfaceTokenMap(positive_token_ids=(0,), negative_token_ids=(1,))
    res = evaluator.evaluate_debiased(
        rubric="Test",
        candidate="Candidate",
        token_map=token_map,
    )
    assert isinstance(res, DebiasedLogitResult)
    assert res.calibrated_prob_pass > 0.90


# ---------------------------------------------------------------------------
# Adaptive Recurrent Loop Early Halting Tests
# ---------------------------------------------------------------------------

def test_execute_adaptive_recurrent_loops_halts_early():
    torch = pytest.importorskip("torch")
    h0 = torch.randn(1, 5, 64)

    def recurrent_step_fn(h, loop_idx):
        return h * 0.9 + 0.1

    def lm_head(h_last):
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 0] = 8.0
        logits[0, 0, 1] = -5.0
        return logits

    token_map = SurfaceTokenMap(positive_token_ids=(0,), negative_token_ids=(1,))

    trace = execute_adaptive_recurrent_loops(
        initial_hidden_state=h0,
        recurrent_step_fn=recurrent_step_fn,
        lm_head=lm_head,
        final_norm=None,
        token_map=token_map,
        min_loops=2,
        max_loops=6,
        entropy_halt_threshold=0.20,
        kl_stability_threshold=0.01,
    )

    assert trace.halted_early is True
    assert trace.loops_executed == 2
    assert len(trace.entropy_trajectory) == 2


def test_execute_adaptive_recurrent_loops_max_loops():
    torch = pytest.importorskip("torch")
    h0 = torch.randn(1, 5, 64)

    def recurrent_step_fn(h, loop_idx):
        return h + torch.randn_like(h)

    loop_count = 0

    def lm_head(h_last):
        nonlocal loop_count
        loop_count += 1
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 0] = 0.5 if loop_count % 2 == 0 else -0.5
        logits[0, 0, 1] = 0.5 if loop_count % 2 == 1 else -0.5
        return logits

    token_map = SurfaceTokenMap(positive_token_ids=(0,), negative_token_ids=(1,))

    trace = execute_adaptive_recurrent_loops(
        initial_hidden_state=h0,
        recurrent_step_fn=recurrent_step_fn,
        lm_head=lm_head,
        final_norm=None,
        token_map=token_map,
        min_loops=2,
        max_loops=5,
        entropy_halt_threshold=0.05,
        kl_stability_threshold=0.001,
    )

    assert trace.halted_early is False
    assert trace.loops_executed == 5
    assert len(trace.entropy_trajectory) == 5


# ---------------------------------------------------------------------------
# PCIe Spillover Guard Tests
# ---------------------------------------------------------------------------

def test_pcie_spillover_guard_cpu():
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0)
    status = guard.inspect()
    assert isinstance(status, VRAMWatermarkStatus)
    assert status.spillover_risk_detected is False

    with guard.guard_forward_pass() as s:
        assert isinstance(s, VRAMWatermarkStatus)


def test_pcie_spillover_guard_mocked_cuda():
    pytest.importorskip("torch")
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0)

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(256 * 1024 * 1024, 16000 * 1024 * 1024)), \
         patch("torch.cuda.memory_allocated", return_value=(1000 * 1024 * 1024)), \
         patch("torch.cuda.memory_reserved", return_value=(2000 * 1024 * 1024)), \
         patch("torch.cuda.empty_cache") as mock_empty:

        status = guard.inspect()
        assert status.spillover_risk_detected is True
        assert status.free_physical_mb == 256.0

        with guard.guard_forward_pass(estimated_kv_mb=128.0):
            pass
        mock_empty.assert_called()


def test_pcie_spillover_guard_reserved_memory_risk():
    pytest.importorskip("torch")
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(10000 * 1024 * 1024, 100000 * 1024 * 1024)), \
         patch("torch.cuda.memory_allocated", return_value=(50000 * 1024 * 1024)), \
         patch("torch.cuda.memory_reserved", return_value=(97000 * 1024 * 1024)):
        status = guard.inspect()
        assert status.spillover_risk_detected is True


# ---------------------------------------------------------------------------
# CLI Calibration & Telemetry Tests
# ---------------------------------------------------------------------------

def test_cli_calibrate_conformal_md(capsys):
    from logitgate.exceptions import CalibrationWarning

    with pytest.warns(CalibrationWarning):
        ret = main([
            "calibrate",
            "--dataset",
            "EPISTEMIC_REASONING_TEST_CASES.md",
            "--alpha",
            "0.05",
        ])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "LOGITGATE SPLIT CONFORMAL CASCADE CALIBRATION" in captured
    assert "Layer 1 Tau Reject:" in captured
    assert "Layer 1 Tau Accept:" in captured
    assert "Guaranteed Layer 1 Error:" in captured


def test_cli_calibrate_json_dataset(tmp_path, capsys):
    from logitgate.exceptions import CalibrationWarning

    dataset_path = tmp_path / "calibration.json"
    dataset_path.write_text(json.dumps([
        {"rubric": "Is it valid?", "candidate": "valid", "ground_truth": True},
        {"rubric": "Is it valid?", "candidate": "invalid", "ground_truth": False},
    ] * 6))

    with pytest.warns(CalibrationWarning):
        ret = main([
            "calibrate",
            "--dataset",
            str(dataset_path),
            "--alpha",
            "0.10",
            "--json",
        ])
    assert ret == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["calibration_set_size"] == 12
    assert data["target_risk_alpha"] == 0.10
    assert "tau_reject" in data
    assert "tau_accept" in data


def test_cli_explain_renders_telemetry(tmp_path, capsys):
    trace = LayerTrace(
        layer_name="LogitGate",
        status=DecisionStatus.FAILED,
        latency_ms=45.0,
        calibrated_prob=0.15,
        positional_bias_magnitude=0.42,
        log_odds_pass=-1.73,
        shannon_entropy=0.61,
        loops_executed=4,
        failure_code="EPISTEMIC_UNCERTAINTY",
    )
    verdict = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.15,
        triggered_layer="LogitGate",
        total_latency_ms=45.0,
        traces=[trace],
    )
    snap = VerdictSnapshot(
        test_name="test_telemetry_render",
        rubric="Must satisfy safety",
        candidate="Uncertain candidate output",
        verdict=verdict,
    )
    snap_dir = tmp_path / "failures"
    saved_path = snap.save(directory=snap_dir)

    ret = main([
        "explain",
        str(saved_path),
        "--backend",
        "stub://default",
    ])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Positional Bias: 0.4200 log-odds" in out
    assert "Unbiased Log-Odds: -1.7300" in out
    assert "Verdict Entropy: 0.6100 bits" in out
    assert "Recurrent Loops: 4" in out
    assert "Failure Code:    EPISTEMIC_UNCERTAINTY" in out

