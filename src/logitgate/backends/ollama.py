from __future__ import annotations

from typing import Any

import httpx
from typing_extensions import Self

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    strip_reasoning_blocks,
)
from logitgate.config import LogitGateDefaults


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
        num_ctx: int | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        eff_num_ctx = num_ctx if num_ctx is not None else (LogitGateDefaults.num_ctx or 16384)
        self.num_ctx = eff_num_ctx
        self.options = dict(options) if options else {}
        self.keep_alive = keep_alive
        self._client = client
        self._async_client = async_client

    def __repr__(self) -> str:
        return f"OllamaBackend(model='{self.model}', base_url='{self.base_url}')"

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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def build_query_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
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
        opts: dict[str, Any] = {
            "num_predict": 1,
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
        }
        if self.options:
            opts.update(self.options)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": top_k,
            "options": opts,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        for k, v in kwargs.items():
            if k == "options" and isinstance(v, dict):
                payload["options"].update(v)
            elif k not in payload:
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
        allowed_tokens: list[str] | None = None,
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
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        payload = self.build_query_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = await self.async_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        opts: dict[str, Any] = {
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
        }
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if self.options:
            opts.update(self.options)
        think_default = max_tokens is None
        think_val = kwargs.pop("think", think_default)
        if think_val is None:
            think_val = think_default
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": think_val,
            "options": opts,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        for k, v in kwargs.items():
            if k == "options" and isinstance(v, dict):
                payload["options"].update(v)
            else:
                payload[k] = v
        resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        visible_text = strip_reasoning_blocks(content)

        # Mid-Thought Truncation Safety Net: True Continuation
        # If thinking was enabled and 100% of generated tokens were consumed inside <think>,
        # visible_text is empty while raw content was non-empty. Close </think> and continue
        # with think=False so all max_tokens go to visible diagnostic output while preserving CoT.
        if not visible_text and content and think_val:
            raw_thinking = content.rstrip()
            assistant_prefill = raw_thinking + "\n</think>\n"
            continuation_opts = dict(opts)
            if max_tokens is not None:
                continuation_opts["num_predict"] = max_tokens
            continuation_payload = dict(payload)
            continuation_payload["messages"] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prefill},
            ]
            continuation_payload["think"] = False
            continuation_payload["options"] = continuation_opts
            resp_cont = self.client.post(f"{self.base_url}/api/chat", json=continuation_payload)
            resp_cont.raise_for_status()
            data_cont = resp_cont.json()
            content_cont = data_cont.get("message", {}).get("content", "")
            tokens_cont = data_cont.get("eval_count", 0)
            return strip_reasoning_blocks(content_cont), tokens + tokens_cont

        return visible_text, tokens

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        opts: dict[str, Any] = {
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
        }
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if self.options:
            opts.update(self.options)
        think_default = max_tokens is None
        think_val = kwargs.pop("think", think_default)
        if think_val is None:
            think_val = think_default
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": think_val,
            "options": opts,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        for k, v in kwargs.items():
            if k == "options" and isinstance(v, dict):
                payload["options"].update(v)
            else:
                payload[k] = v
        resp = await self.async_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        visible_text = strip_reasoning_blocks(content)

        # Mid-Thought Truncation Safety Net: True Continuation
        if not visible_text and content and think_val:
            raw_thinking = content.rstrip()
            assistant_prefill = raw_thinking + "\n</think>\n"
            continuation_opts = dict(opts)
            if max_tokens is not None:
                continuation_opts["num_predict"] = max_tokens
            continuation_payload = dict(payload)
            continuation_payload["messages"] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_prefill},
            ]
            continuation_payload["think"] = False
            continuation_payload["options"] = continuation_opts
            resp_cont = await self.async_client.post(f"{self.base_url}/api/chat", json=continuation_payload)
            resp_cont.raise_for_status()
            data_cont = resp_cont.json()
            content_cont = data_cont.get("message", {}).get("content", "")
            tokens_cont = data_cont.get("eval_count", 0)
            return strip_reasoning_blocks(content_cont), tokens + tokens_cont

        return visible_text, tokens
