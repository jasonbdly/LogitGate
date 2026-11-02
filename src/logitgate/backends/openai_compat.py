from __future__ import annotations

import os
from typing import Any, Optional
import httpx

from logitgate.backends.base import LogitBackend, GenerativeBackend


class OpenAICompatBackend(LogitBackend, GenerativeBackend):
    """
    OpenAI-compatible backend (vLLM, Groq, OpenRouter, Together, OpenAI)
    supporting first-token logprob extraction and generative reviews.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-dummy")
        self.timeout = timeout

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
    ) -> dict[str, Any]:
        return {
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
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = self.build_query_payload(prompt, top_k=top_k, allowed_tokens=allowed_tokens)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        payload = self.build_query_payload(prompt, top_k=top_k, allowed_tokens=allowed_tokens)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    def generate(self, prompt: str) -> tuple[str, int]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return content.strip(), tokens

    async def agenerate(self, prompt: str) -> tuple[str, int]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return content.strip(), tokens
