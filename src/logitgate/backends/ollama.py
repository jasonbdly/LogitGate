from __future__ import annotations

import re
from typing import Any, Optional
import httpx

from logitgate.backends.base import LogitBackend, GenerativeBackend


def strip_reasoning_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning traces from model responses."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class OllamaBackend(LogitBackend, GenerativeBackend):
    """
    Ollama backend supporting first-token logprob extraction with reasoning suppression
    ("think": False) and generative diagnostic critiques.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:14b",
        timeout: float = 30.0,
        num_ctx: int = 16384,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx

    def build_query_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Constructs the chat completion payload ensuring think: False."""
        messages = [
            {
                "role": "system",
                "content": "Respond with a single character only. Do not think or explain.",
            },
            {"role": "user", "content": prompt},
        ]
        return {
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
    ) -> list[dict[str, Any]]:
        payload = self.build_query_payload(prompt, top_k=top_k, allowed_tokens=allowed_tokens)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        payload = self.build_query_payload(prompt, top_k=top_k, allowed_tokens=allowed_tokens)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    def generate(self, prompt: str) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        return strip_reasoning_blocks(content), tokens

    async def agenerate(self, prompt: str) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        return strip_reasoning_blocks(content), tokens
