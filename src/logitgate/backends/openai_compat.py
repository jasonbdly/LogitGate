from __future__ import annotations

import os
from typing import Any

import httpx

from logitgate.backends.base import (
    EXCLUDED_INTERNAL_KWARGS,
    GenerativeBackend,
    LogitBackend,
    _HTTPClientMixin,
    strip_reasoning_blocks,
    translate_transport_errors,
)
from logitgate.exceptions import BackendError


class OpenAICompatBackend(_HTTPClientMixin, LogitBackend, GenerativeBackend):
    """
    OpenAI-compatible backend (vLLM, Groq, OpenRouter, Together, OpenAI)
    supporting first-token logprob extraction, persistent connection pooling,
    and generative reviews with reasoning trace suppression.
    """

    service_name = "OpenAI-compatible server"
    connect_hint = "Check base_url, and that the server is running (vLLM: `vllm serve <model>`)."

    def transport_hint(self, status_code: int, detail: str) -> str | None:
        if status_code in (401, 403):
            return "Set OPENAI_API_KEY or pass api_key= (vLLM: match its --api-key)."
        if status_code == 404:
            return f"Check that model '{self.model}' is served and base_url ends with /v1."
        if status_code == 400 and "logprobs" in detail.lower():
            return "This endpoint does not expose logprobs; LogitGate needs top_logprobs support."
        return None

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: float = 30.0,
        logit_bias: dict[str, int | float] | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-dummy")
        self.timeout = timeout
        self.logit_bias = logit_bias
        self._init_http_clients(client=client, async_client=async_client)

    def __repr__(self) -> str:
        return f"OpenAICompatBackend(model='{self.model}', base_url='{self.base_url}')"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_query_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if allowed_tokens:
            clean_tokens = [t.strip() for t in allowed_tokens if isinstance(t, str) and t.strip()]
            tokens_hint = f" ({', '.join(clean_tokens)})" if clean_tokens else ""
            effective_top_k = min(max(top_k, 2 * len(clean_tokens)), 20)
        else:
            tokens_hint = ""
            effective_top_k = min(top_k, 20)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"Respond with a single character only{tokens_hint}.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": effective_top_k,
        }
        logit_bias = kwargs.get("logit_bias", self.logit_bias)
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias
        for k, v in kwargs.items():
            if k not in payload and k != "logit_bias" and k not in EXCLUDED_INTERNAL_KWARGS:
                payload[k] = v
        return payload

    def _extract_top_logprobs_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            raise BackendError(
                f"{self.service_name} at {self.base_url} returned non-object JSON payload: {type(data).__name__}"
            )
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or not isinstance(choices[0], dict):
            return []
        logprobs = choices[0].get("logprobs")
        if not logprobs or not isinstance(logprobs, dict):
            return []
        content_logprobs = logprobs.get("content")
        if not content_logprobs or not isinstance(content_logprobs, list) or not isinstance(content_logprobs[0], dict):
            return []
        top_logprobs = content_logprobs[0].get("top_logprobs")
        if not top_logprobs or not isinstance(top_logprobs, list):
            return []
        return [
            {
                "token": str(item.get("token") or ""),
                "logprob": float(item["logprob"]),
            }
            for item in top_logprobs
            if isinstance(item, dict) and item.get("logprob") is not None
        ]

    @staticmethod
    def _extract_chat_completion(data: Any) -> tuple[str, int]:
        if not isinstance(data, dict):
            raise BackendError(f"OpenAI-compatible server returned non-object JSON payload: {type(data).__name__}")
        choices = data.get("choices")
        content = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and msg.get("content") is not None:
                content = str(msg["content"])
        usage = data.get("usage")
        tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
        return strip_reasoning_blocks(content), tokens

    @translate_transport_errors
    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = self.build_query_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = self.client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    @translate_transport_errors
    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = self.build_query_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = await self.async_client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    @translate_transport_errors
    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDED_INTERNAL_KWARGS}
        payload.update(filtered_kwargs)
        resp = self.client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return self._extract_chat_completion(resp.json())

    @translate_transport_errors
    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDED_INTERNAL_KWARGS}
        payload.update(filtered_kwargs)
        resp = await self.async_client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return self._extract_chat_completion(resp.json())
