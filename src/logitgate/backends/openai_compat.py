from __future__ import annotations

import os
from typing import Any, Optional
import httpx

from logitgate.backends.base import LogitBackend, GenerativeBackend, strip_reasoning_blocks


class OpenAICompatBackend(LogitBackend, GenerativeBackend):
    """
    OpenAI-compatible backend (vLLM, Groq, OpenRouter, Together, OpenAI)
    supporting first-token logprob extraction, persistent connection pooling,
    and generative reviews with reasoning trace suppression.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        logit_bias: Optional[dict[str, int | float]] = None,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-dummy")
        self.timeout = timeout
        self.logit_bias = logit_bias
        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def close(self) -> None:
        """Closes the synchronous HTTP client session if initialized."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    async def aclose(self) -> None:
        """Closes the asynchronous HTTP client session if initialized."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()

    def __enter__(self) -> OpenAICompatBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> OpenAICompatBackend:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_query_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Respond with a single character only.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": min(top_k, 20),
        }
        logit_bias = kwargs.get("logit_bias", self.logit_bias)
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias
        for k, v in kwargs.items():
            if k not in payload and k != "logit_bias":
                payload[k] = v
        return payload

    def _extract_top_logprobs_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        choices = data.get("choices", [])
        if not choices:
            return []
        logprobs = choices[0].get("logprobs")
        if not logprobs:
            return []
        content_logprobs = logprobs.get("content", [])
        if not content_logprobs:
            return []
        top_logprobs = content_logprobs[0].get("top_logprobs", [])
        return [
            {"token": item.get("token", ""), "logprob": item.get("logprob", -10.0)}
            for item in top_logprobs
        ]

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
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

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
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

    def generate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        payload.update(kwargs)
        resp = self.client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return strip_reasoning_blocks(content), tokens

    async def agenerate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        payload.update(kwargs)
        resp = await self.async_client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return strip_reasoning_blocks(content), tokens
