from __future__ import annotations

import re
from typing import Any, Optional
import httpx

from logitgate.backends.base import LogitBackend, GenerativeBackend, strip_reasoning_blocks


class OllamaBackend(LogitBackend, GenerativeBackend):
    """
    Ollama backend supporting first-token logprob extraction with reasoning suppression
    (think: False), persistent connection pooling, and generative diagnostic critiques.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:14b",
        timeout: float = 30.0,
        num_ctx: int = 16384,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
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

    def __enter__(self) -> OllamaBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> OllamaBackend:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def build_query_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Constructs the chat completion payload ensuring think: False."""
        messages = [
            {
                "role": "system",
                "content": "Respond with a single character only. Do not think or explain.",
            },
            {"role": "user", "content": prompt},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": {
                "num_predict": 1,
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v
        return payload

    def _extract_top_logprobs_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        lp = data.get("logprobs") or data.get("message", {}).get("logprobs")
        if not lp or not isinstance(lp, list):
            return []
        first = lp[0]
        if isinstance(first, dict):
            if "top_logprobs" in first:
                return first["top_logprobs"]
            if "token" in first:
                return lp
        return []

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        payload = self.build_query_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
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
        payload = self.build_query_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = await self.async_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    def generate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }
        payload.update(kwargs)
        resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        return strip_reasoning_blocks(content), tokens

    async def agenerate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }
        payload.update(kwargs)
        resp = await self.async_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        return strip_reasoning_blocks(content), tokens
