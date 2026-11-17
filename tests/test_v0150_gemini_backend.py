from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from logitgate.backends.base import (
    LogitBackend,
    StubBackend,
    infer_backend_spec,
    resolve_generative_backend,
    resolve_logit_backend,
)
from logitgate.backends.gemini import GeminiBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.exceptions import (
    BackendError,
    ConfigurationError,
    UnsupportedLogprobsError,
)

# ===========================================================================
# 1. GeminiBackend Instantiation, Key Resolution, and URL Safety
# ===========================================================================


def test_gemini_backend_init_validation() -> None:
    # Non-empty string requirement
    with pytest.raises(ConfigurationError, match="model must be a non-empty string"):
        GeminiBackend(model="")

    with pytest.raises(ConfigurationError, match="model must be a non-empty string"):
        GeminiBackend(model="   ")

    with pytest.raises(ConfigurationError, match="model must be a non-empty string"):
        GeminiBackend(model=123)  # type: ignore[arg-type]

    backend = GeminiBackend(model="models/gemini-2.5-flash", api_key="test-key-123")
    assert backend.clean_model == "gemini-2.5-flash"
    assert "GeminiBackend(model='models/gemini-2.5-flash'" in repr(backend)


def test_gemini_backend_api_key_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Explicit param takes precedence over env vars
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
    b_explicit = GeminiBackend(api_key="explicit-key")
    assert b_explicit.api_key == "explicit-key"

    # 2. GEMINI_API_KEY fallback
    b_env1 = GeminiBackend()
    assert b_env1.api_key == "env-gemini-key"

    # 3. GOOGLE_API_KEY fallback
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    b_env2 = GeminiBackend()
    assert b_env2.api_key == "env-google-key"

    # 4. No key and no injected client headers raises ConfigurationError at request time
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    b_none = GeminiBackend()
    assert b_none.api_key is None
    with pytest.raises(ConfigurationError, match="Gemini API key is required"):
        b_none._headers(is_async=False)
    with pytest.raises(ConfigurationError, match="Gemini API key is required"):
        b_none._headers(is_async=True)

    # 5. Injected client with authorization header bypasses ConfigurationError
    client_auth = httpx.Client(headers={"Authorization": "Bearer oauth-token"})
    b_injected = GeminiBackend(client=client_auth)
    headers = b_injected._headers(is_async=False)
    assert headers["Content-Type"] == "application/json"
    assert "x-goog-api-key" not in headers


def test_gemini_backend_security_header_never_in_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "logprobsResult": {
                            "topCandidates": [
                                {
                                    "candidates": [
                                        {"token": "A", "logProbability": -0.22},
                                        {"token": "B", "logProbability": -1.85},
                                    ]
                                }
                            ]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    secret_key = "AIzaSySecretApiKeyDoNotLeak"
    backend = GeminiBackend(
        model="models/gemini-2.5-flash",
        api_key=secret_key,
        client=client,
    )

    res = backend.query_top_logprobs("Evaluate prompt", top_k=2)
    assert len(res) == 2
    assert res[0]["token"] == "A"
    assert res[0]["logprob"] == pytest.approx(-0.22, rel=1e-4)

    assert len(captured_requests) == 1
    req = captured_requests[0]
    # Invariant: API key in header, NEVER in query string
    assert req.headers["x-goog-api-key"] == secret_key
    assert "key=" not in str(req.url)
    assert str(req.url).endswith("/models/gemini-2.5-flash:generateContent")


# ===========================================================================
# 2. Gemini Payload Construction, Thinking Budget & Clamping Ties
# ===========================================================================


def test_gemini_payload_top_k_clamping_and_clean_tokens() -> None:
    backend = GeminiBackend(model="gemini-1.5-flash", api_key="k")

    # Clamping ties:
    # 1. Strictly below lower bound: -5 -> 1
    p_neg = backend.build_query_payload("p", top_k=-5)
    assert p_neg["generationConfig"]["logprobs"] == 1

    # 2. Exact lower bound tie: 1 -> 1
    p_one = backend.build_query_payload("p", top_k=1)
    assert p_one["generationConfig"]["logprobs"] == 1

    # 3. Normal range: 7 -> 7
    p_mid = backend.build_query_payload("p", top_k=7)
    assert p_mid["generationConfig"]["logprobs"] == 7

    # 4. Exact upper bound tie: 20 -> 20
    p_twenty = backend.build_query_payload("p", top_k=20)
    assert p_twenty["generationConfig"]["logprobs"] == 20

    # 5. Strictly above upper bound: 25 -> 20
    p_above = backend.build_query_payload("p", top_k=25)
    assert p_above["generationConfig"]["logprobs"] == 20

    # 6. allowed_tokens scaling 2 * len(clean_tokens)
    p_toks = backend.build_query_payload("p", top_k=3, allowed_tokens=["A", "B", "C"])
    # 2 * 3 = 6 > 3 -> 6
    assert p_toks["generationConfig"]["logprobs"] == 6
    assert "systemInstruction" in p_toks
    assert "(A, B, C)" in p_toks["systemInstruction"]["parts"][0]["text"]

    # 7. allowed_tokens capping at 20 when 2 * len > 20
    long_tokens = [f"T{i}" for i in range(12)]
    p_long = backend.build_query_payload("p", top_k=2, allowed_tokens=long_tokens)
    assert p_long["generationConfig"]["logprobs"] == 20

    # 8. Empty/whitespace allowed_tokens filtered
    p_empty_toks = backend.build_query_payload("p", top_k=4, allowed_tokens=["", "  "])
    assert p_empty_toks["generationConfig"]["logprobs"] == 4
    assert "systemInstruction" not in p_empty_toks


def test_gemini_payload_thinking_budget_matrix() -> None:
    # 1. gemini-2.5-flash: defaults to thinkingBudget: 0 for logprobs
    b_flash_25 = GeminiBackend(model="gemini-2.5-flash", api_key="k")
    p_flash = b_flash_25.build_query_payload("p")
    assert p_flash["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    # 2. gemini-2.5-pro: does NOT set thinkingBudget: 0
    b_pro_25 = GeminiBackend(model="gemini-2.5-pro", api_key="k")
    p_pro = b_pro_25.build_query_payload("p")
    assert "thinkingConfig" not in p_pro["generationConfig"]

    # 3. gemini-1.5-flash: does NOT set thinkingConfig
    b_flash_15 = GeminiBackend(model="gemini-1.5-flash", api_key="k")
    p_15 = b_flash_15.build_query_payload("p")
    assert "thinkingConfig" not in p_15["generationConfig"]

    # 4. Caller overrides thinking_budget in kwargs
    p_custom = b_flash_25.build_query_payload("p", thinking_budget=128)
    assert p_custom["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 128}

    p_custom_none = b_flash_25.build_query_payload("p", thinking_budget=None)
    assert "thinkingConfig" not in p_custom_none["generationConfig"]

    # 5. Generate payload with think=False on 2.5 flash
    gen_p = b_flash_25.build_generate_payload("p", max_tokens=64, think=False)
    assert gen_p["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert gen_p["generationConfig"]["maxOutputTokens"] == 64

    # 6. Generate payload with think=True on 2.5 flash (does not force 0)
    gen_think_true = b_flash_25.build_generate_payload("p", think=True)
    assert "thinkingConfig" not in gen_think_true["generationConfig"]


# ===========================================================================
# 3. GeminiBackend Sync & Async Execution with Mocks
# ===========================================================================


@pytest.mark.asyncio
async def test_gemini_backend_sync_and_async_query_top_logprobs() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "logprobsResult": {
                            "topCandidates": [
                                {
                                    "candidates": [
                                        {"token": "Pass", "logProbability": -0.05},
                                        {"token": "Fail", "logProbability": -3.12},
                                    ]
                                }
                            ]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)

    backend = GeminiBackend(
        model="gemini-2.5-flash",
        api_key="mock-key",
        client=client,
        async_client=async_client,
    )

    # Sync
    res_sync = backend.query_top_logprobs("Check code", top_k=5)
    assert len(res_sync) == 2
    assert res_sync[0]["token"] == "Pass"
    assert res_sync[0]["logprob"] == pytest.approx(-0.05, rel=1e-4)

    # Async
    res_async = await backend.aquery_top_logprobs("Check code async", top_k=5)
    assert len(res_async) == 2
    assert res_async[1]["token"] == "Fail"
    assert res_async[1]["logprob"] == pytest.approx(-3.12, rel=1e-4)

    backend.close()
    await backend.aclose()


@pytest.mark.asyncio
async def test_gemini_backend_sync_and_async_generate_thought_skipping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "I should analyze this thoroughly..."},
                                {"text": "[VERDICT: PASS] All requirements satisfied."},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "candidatesTokenCount": 18,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)

    backend = GeminiBackend(
        model="gemini-2.5-flash",
        api_key="mock-key",
        client=client,
        async_client=async_client,
    )

    text, tokens = backend.generate("Evaluate", max_tokens=100)
    assert text == "[VERDICT: PASS] All requirements satisfied."
    assert "I should analyze" not in text
    assert tokens == 18

    text_async, tokens_async = await backend.agenerate("Evaluate async", max_tokens=100)
    assert text_async == "[VERDICT: PASS] All requirements satisfied."
    assert tokens_async == 18

    backend.close()
    await backend.aclose()


def test_gemini_backend_generate_usage_fallback_estimate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Missing usageMetadata
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "<think>private</think>Verdict is confirmed clear."},
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k", client=client)

    text, tokens = backend.generate("Evaluate")
    assert text == "Verdict is confirmed clear."
    # 4 words fallback estimate
    assert tokens == 4


# ===========================================================================
# 4. Error Handling: 400 Logprobs, 429/503 Retries, Safety Blocks
# ===========================================================================


def test_gemini_backend_http_400_unsupported_logprobs_sync_and_async() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid argument: responseLogprobs is not supported for this model."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = GeminiBackend(model="gemini-custom", api_key="k", client=client, async_client=async_client)

    with pytest.raises(UnsupportedLogprobsError, match="does not support logprobs: HTTP 400"):
        backend.query_top_logprobs("prompt")

    with pytest.raises(UnsupportedLogprobsError, match="does not support logprobs: HTTP 400"):
        asyncio.run(backend.aquery_top_logprobs("prompt"))


def test_gemini_backend_http_400_other_error_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Field contents[0].parts is empty."}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k", client=client)

    # Other 400 error raises generic BackendError, not UnsupportedLogprobsError
    with pytest.raises(BackendError, match="HTTP 400") as exc_info:
        backend.query_top_logprobs("prompt")
    assert not isinstance(exc_info.value, UnsupportedLogprobsError)


def test_gemini_backend_missing_logprobs_result_raises_unsupported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hello"}]},
                        # logprobsResult missing
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k", client=client)

    with pytest.raises(UnsupportedLogprobsError, match="returned response without logprobsResult"):
        backend.query_top_logprobs("prompt")


def test_gemini_backend_safety_block_raises_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k", client=client)

    with pytest.raises(BackendError, match="blocked by safety: SAFETY"):
        backend.query_top_logprobs("prompt")

    with pytest.raises(BackendError, match="blocked by safety: SAFETY"):
        backend.generate("prompt")


def test_gemini_backend_retry_on_429() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "logprobsResult": {
                            "topCandidates": [
                                {"candidates": [{"token": "A", "logProbability": -0.15}]}
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = GeminiBackend(
        model="gemini-2.5-flash",
        api_key="k",
        client=client,
        max_retries=2,
        retry_backoff_base=0.01,
        retry_max_wait=0.1,
    )

    res = backend.query_top_logprobs("prompt")
    assert len(res) == 1
    assert attempts == 2
    assert res[0]["token"] == "A"


def test_gemini_backend_transport_hints() -> None:
    backend = GeminiBackend(model="models/gemini-2.5-flash", api_key="k")
    assert "GEMINI_API_KEY" in (backend.transport_hint(401, "unauthorized") or "")
    assert "GEMINI_API_KEY" in (backend.transport_hint(403, "forbidden") or "")
    assert "gemini-2.5-flash" in (backend.transport_hint(404, "not found") or "")
    assert "responseLogprobs" in (backend.transport_hint(400, "logprobs unsupported") or "")
    assert backend.transport_hint(500, "internal error") is None


# ===========================================================================
# 5. OpenAICompatBackend UnsupportedLogprobsError Detection
# ===========================================================================


def test_openai_compat_unsupported_logprobs_400_and_422() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # Model like o1 or o3-mini returns 400 or 422 stating logprobs is not supported
        return httpx.Response(
            400,
            json={"error": {"message": "logprobs is not supported with this model"}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(
        model="o3-mini",
        client=client,
        async_client=async_client,
        max_retries=2,
    )

    # Sync: zero retries on 400!
    with pytest.raises(UnsupportedLogprobsError, match="does not support logprobs: HTTP 400"):
        backend.query_top_logprobs("prompt")
    assert call_count == 1

    # Async
    with pytest.raises(UnsupportedLogprobsError, match="does not support logprobs: HTTP 400"):
        asyncio.run(backend.aquery_top_logprobs("prompt"))
    assert call_count == 2


def test_openai_compat_unsupported_logprobs_silent_strip_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # LiteLLM/OpenRouter proxy on Claude silently returns choices without logprobs
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "A"},
                        "logprobs": None,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(model="claude-3-5-sonnet", client=client)

    with pytest.raises(UnsupportedLogprobsError, match="returned response with logprobs=None"):
        backend.query_top_logprobs("prompt")


def test_openai_compat_backend_supports_logprobs_flag() -> None:
    with pytest.raises(ConfigurationError, match="supports_logprobs must be a bool"):
        OpenAICompatBackend(supports_logprobs="false")  # type: ignore[arg-type]

    backend = OpenAICompatBackend(supports_logprobs=False)
    assert backend.supports_logprobs is False

    with pytest.raises(UnsupportedLogprobsError, match="configured with supports_logprobs=False"):
        backend.query_top_logprobs("prompt")

    with pytest.raises(UnsupportedLogprobsError, match="configured with supports_logprobs=False"):
        asyncio.run(backend.aquery_top_logprobs("prompt"))


def test_unsupported_logprobs_error_hierarchy() -> None:
    assert issubclass(UnsupportedLogprobsError, BackendError)
    err = UnsupportedLogprobsError("Logprobs unsupported")
    assert isinstance(err, BackendError)


# ===========================================================================
# 6. StubBackend supports_logprobs & URI Resolution
# ===========================================================================


def test_stub_backend_supports_logprobs_behavior() -> None:
    with pytest.raises(ConfigurationError, match="supports_logprobs must be a bool"):
        StubBackend(supports_logprobs="no")  # type: ignore[arg-type]

    stub_unsupported = StubBackend(supports_logprobs=False)
    assert stub_unsupported.supports_logprobs is False

    with pytest.raises(UnsupportedLogprobsError, match="StubBackend configured with supports_logprobs=False"):
        stub_unsupported.query_top_logprobs("prompt")

    with pytest.raises(UnsupportedLogprobsError, match="StubBackend configured with supports_logprobs=False"):
        asyncio.run(stub_unsupported.aquery_top_logprobs("prompt"))

    # Still works as GenerativeBackend
    text, tokens = stub_unsupported.generate("Critique")
    assert "PASS" in text
    assert tokens == 0


def test_resolve_stub_no_logprobs_uri() -> None:
    b1 = resolve_logit_backend("stub://no-logprobs")
    assert isinstance(b1, StubBackend)
    assert b1.supports_logprobs is False

    b2 = resolve_logit_backend("stub://unsupported")
    assert isinstance(b2, StubBackend)
    assert b2.supports_logprobs is False

    assert infer_backend_spec(b1) == "stub://no-logprobs"


def test_resolve_gemini_uri_and_spec_roundtrip() -> None:
    # 1. gemini://gemini-2.5-flash
    b1 = resolve_logit_backend("gemini://gemini-2.5-flash")
    assert isinstance(b1, GeminiBackend)
    assert b1.model == "gemini-2.5-flash"
    assert infer_backend_spec(b1) == "gemini://gemini-2.5-flash"

    # 2. gemini:// models prefix normalization
    b2 = resolve_logit_backend("gemini://models/gemini-2.5-pro")
    assert isinstance(b2, GeminiBackend)
    assert b2.clean_model == "gemini-2.5-pro"
    assert infer_backend_spec(b2) == "gemini://gemini-2.5-pro"

    # 3. Bare gemini:// defaults to gemini-2.5-flash
    b3 = resolve_logit_backend("gemini://")
    assert isinstance(b3, GeminiBackend)
    assert b3.model == "gemini-2.5-flash"

    # 4. Bare gemini defaults to gemini-2.5-flash
    b4 = resolve_logit_backend("gemini")
    assert isinstance(b4, GeminiBackend)
    assert b4.model == "gemini-2.5-flash"

    # 5. resolve_generative_backend
    gen_b = resolve_generative_backend("gemini://gemini-2.5-flash")
    assert isinstance(gen_b, GeminiBackend)


# ===========================================================================
# 7. Additional Edge-Case Coverage for Gemini & Base Resolution
# ===========================================================================


def test_gemini_payload_kwargs_and_custom_options() -> None:
    backend = GeminiBackend(model="gemini-2.5-pro", api_key="k")

    # 1. build_query_payload with generationConfig & custom kwargs
    qp = backend.build_query_payload(
        "p",
        generationConfig={"stopSequences": ["\n"]},
        generation_config={"candidateCount": 1},
        extra_param="custom_value",
    )
    assert qp["generationConfig"]["stopSequences"] == ["\n"]
    assert qp["generationConfig"]["candidateCount"] == 1
    assert qp["extra_param"] == "custom_value"

    # 2. build_generate_payload with think=False on gemini-2.5-pro (thinking budget 0 NOT set)
    gp_pro = backend.build_generate_payload("p", think=False, thinking_budget=None)
    assert "thinkingConfig" not in gp_pro["generationConfig"]
    assert "think" not in gp_pro
    assert "thinking_budget" not in gp_pro

    # 3. build_generate_payload with kwargs & generation_config
    gp_kwargs = backend.build_generate_payload(
        "p",
        generation_config={"temperature": 0.4},
        generationConfig={"topP": 0.95},
        extra_meta=True,
        think=True,
    )
    assert gp_kwargs["generationConfig"]["temperature"] == 0.4
    assert gp_kwargs["generationConfig"]["topP"] == 0.95
    assert gp_kwargs["extra_meta"] is True
    assert "think" not in gp_kwargs


def test_gemini_response_parsing_edge_cases() -> None:
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k")

    # 1. Non-dict payload
    with pytest.raises(BackendError, match="returned non-object JSON payload"):
        backend._parse_logprobs_response("not-dict")

    # 2. Empty candidates without promptFeedback
    with pytest.raises(BackendError, match="returned no candidates"):
        backend._parse_logprobs_response({"candidates": []})

    # 3. Empty candidates with empty promptFeedback
    with pytest.raises(BackendError, match="returned no candidates"):
        backend._parse_logprobs_response({"candidates": [], "promptFeedback": {}})

    # 4. Candidate is not a dict
    with pytest.raises(BackendError, match="returned non-dict candidate"):
        backend._parse_logprobs_response({"candidates": ["not-a-dict"]})

    # 5. topCandidates is empty or non-dict
    assert backend._parse_logprobs_response({"candidates": [{"logprobsResult": {"topCandidates": []}}]}) == []
    assert backend._parse_logprobs_response({"candidates": [{"logprobsResult": {"topCandidates": ["not-a-dict"]}}]}) == []
    assert backend._parse_logprobs_response({"candidates": [{"logprobsResult": {"topCandidates": "not-a-list"}}]}) == []

    # 6. inner candidates is empty or non-list
    assert backend._parse_logprobs_response({"candidates": [{"logprobsResult": {"topCandidates": [{"candidates": []}]}}]}) == []
    assert backend._parse_logprobs_response({"candidates": [{"logprobsResult": {"topCandidates": [{"candidates": "not-list"}]}}]}) == []

    # 7. Token is None / empty string
    parsed = backend._parse_logprobs_response({
        "candidates": [{
            "logprobsResult": {
                "topCandidates": [{
                    "candidates": [{"logProbability": -0.42, "token": None}]
                }]
            }
        }]
    })
    assert len(parsed) == 1
    assert parsed[0]["token"] == ""
    assert parsed[0]["logprob"] == pytest.approx(-0.42, rel=1e-4)


def test_gemini_generate_parsing_edge_cases() -> None:
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k")

    # 1. Non-dict payload
    with pytest.raises(BackendError, match="returned non-object JSON payload"):
        backend._extract_generate_response("not-dict")

    # 2. Empty candidates without promptFeedback
    with pytest.raises(BackendError, match="returned no candidates"):
        backend._extract_generate_response({"candidates": []})

    # 3. Empty candidates with empty promptFeedback
    with pytest.raises(BackendError, match="returned no candidates"):
        backend._extract_generate_response({"candidates": [], "promptFeedback": {}})

    # 4. Candidate is not a dict
    with pytest.raises(BackendError, match="returned non-dict candidate"):
        backend._extract_generate_response({"candidates": ["not-a-dict"]})

    # 5. Parts with non-dict, empty text, or empty parts
    text, tokens = backend._extract_generate_response({
        "candidates": [{
            "content": {
                "parts": ["not-dict", {"text": ""}, {}]
            }
        }]
    })
    assert text == ""
    assert tokens == 0

    # 6. Usage metadata has candidatesTokenCount = None
    text2, tokens2 = backend._extract_generate_response({
        "candidates": [{
            "content": {
                "parts": [{"text": "Alpha Beta Gamma"}]
            }
        }],
        "usageMetadata": {"candidatesTokenCount": None}
    })
    assert text2 == "Alpha Beta Gamma"
    assert tokens2 == 3


@pytest.mark.asyncio
async def test_gemini_async_errors_and_retries() -> None:
    # 1. Async 400 not mentioning logprobs raises BackendError via raise_for_status
    def handler_400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid argument: bad syntax"}})

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_400))
    backend = GeminiBackend(model="gemini-2.5-flash", api_key="k", async_client=async_client)
    with pytest.raises(BackendError, match="HTTP 400"):
        await backend.aquery_top_logprobs("p")

    # 2. Async 429 retry
    attempts = 0

    def handler_429(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "logprobsResult": {
                            "topCandidates": [
                                {"candidates": [{"token": "AsyncOK", "logProbability": -0.11}]}
                            ]
                        }
                    }
                ]
            },
        )

    async_client_429 = httpx.AsyncClient(transport=httpx.MockTransport(handler_429))
    b_retry = GeminiBackend(
        model="gemini-2.5-flash",
        api_key="k",
        async_client=async_client_429,
        max_retries=2,
        retry_backoff_base=0.01,
        retry_max_wait=0.1,
    )
    res = await b_retry.aquery_top_logprobs("p")
    assert len(res) == 1
    assert attempts == 2
    assert res[0]["token"] == "AsyncOK"


@pytest.mark.asyncio
async def test_openai_compat_unsupported_logprobs_async_matrix() -> None:
    # 1. Async 422 mentioning top_logprobs
    def handler_422(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": "top_logprobs not allowed on this model"})

    async_client_422 = httpx.AsyncClient(transport=httpx.MockTransport(handler_422))
    b_422 = OpenAICompatBackend(model="o4-mini", async_client=async_client_422)
    with pytest.raises(UnsupportedLogprobsError, match="does not support logprobs: HTTP 422"):
        await b_422.aquery_top_logprobs("prompt")

    # 2. Async 200 OK with choices[0]["logprobs"] is None
    def handler_none(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "logprobs": None}]},
        )

    async_client_none = httpx.AsyncClient(transport=httpx.MockTransport(handler_none))
    b_none = OpenAICompatBackend(model="claude-3-5", async_client=async_client_none)
    with pytest.raises(UnsupportedLogprobsError, match="returned response with logprobs=None"):
        await b_none.aquery_top_logprobs("prompt")

    # 3. Sync & Async 200 OK with choices[0]["logprobs"] == {}
    def handler_empty_dict(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "logprobs": {}}]},
        )

    client_empty = httpx.Client(transport=httpx.MockTransport(handler_empty_dict))
    async_client_empty = httpx.AsyncClient(transport=httpx.MockTransport(handler_empty_dict))
    b_empty = OpenAICompatBackend(model="claude-3-5", client=client_empty, async_client=async_client_empty)
    with pytest.raises(UnsupportedLogprobsError, match="returned response with logprobs={}"):
        b_empty.query_top_logprobs("prompt")
    with pytest.raises(UnsupportedLogprobsError, match="returned response with logprobs={}"):
        await b_empty.aquery_top_logprobs("prompt")


def test_base_spec_and_resolution_coverage_branches() -> None:
    # 1. Stub URI with invalid float part falls back to 0.95
    b_stub_inv = resolve_logit_backend("stub://not-a-number")
    assert isinstance(b_stub_inv, StubBackend)
    assert b_stub_inv.default_prob == pytest.approx(0.95, rel=1e-4)

    # 2. Stub default generative backend
    gen_stub = resolve_generative_backend("stub://default")
    assert isinstance(gen_stub, StubBackend)

    # 3. TypeError on backend that does not implement GenerativeBackend
    class DummyOnlyLogitBackend(LogitBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    with pytest.raises(TypeError, match="does not implement GenerativeBackend"):
        resolve_generative_backend(DummyOnlyLogitBackend())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_openai_compat_chat_template_fallback_and_reasoning_effort_retry() -> None:
    # 1. Sync & Async chat_template fallback when thinking prefill trap is detected
    def fallback_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if "chat_template_kwargs" in payload:
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "logprobs": {
                            "content": [{"top_logprobs": [{"token": "A", "logprob": -0.05}]}]
                        }
                    }]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "logprobs": {
                        "content": [{"top_logprobs": [{"token": "<think>", "logprob": -0.01}]}]
                    }
                }]
            },
        )

    b_sync_fb = OpenAICompatBackend(client=httpx.Client(transport=httpx.MockTransport(fallback_handler)))
    res_sync = b_sync_fb.query_top_logprobs("p", allowed_tokens=["A", "B"])
    assert len(res_sync) == 1
    assert res_sync[0]["token"] == "A"

    b_async_fb = OpenAICompatBackend(async_client=httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)))
    res_async = await b_async_fb.aquery_top_logprobs("p", allowed_tokens=["A", "B"])
    assert len(res_async) == 1
    assert res_async[0]["token"] == "A"

    # 2. Async 400 reasoning_effort retry
    call_counts = {"count": 0}

    def effort_handler(request: httpx.Request) -> httpx.Response:
        call_counts["count"] += 1
        payload = json.loads(request.content.decode("utf-8"))
        if "reasoning_effort" in payload:
            return httpx.Response(400, json={"error": "unsupported reasoning_effort"})
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "logprobs": {
                        "content": [{"top_logprobs": [{"token": "B", "logprob": -0.15}]}]
                    }
                }]
            },
        )

    b_effort = OpenAICompatBackend(async_client=httpx.AsyncClient(transport=httpx.MockTransport(effort_handler)))
    res_effort = await b_effort.aquery_top_logprobs("p", allowed_tokens=["A", "B"])
    assert len(res_effort) == 1
    assert res_effort[0]["token"] == "B"
    assert call_counts["count"] == 2

    # 3. Kwarg logit_bias in build_query_payload
    payload_bias = b_effort.build_query_payload("p", logit_bias={"10": 50})
    assert payload_bias["logit_bias"] == {"10": 50}

    # 4. _extract_chat_completion without usage dict
    content, tok_count = OpenAICompatBackend._extract_chat_completion({"choices": [{"message": {"content": "Verdict"}}]})
    assert content == "Verdict"
    assert tok_count == 0

