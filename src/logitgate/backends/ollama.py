from __future__ import annotations

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
from logitgate.config import LogitGateDefaults
from logitgate.exceptions import BackendError


class OllamaBackend(_HTTPClientMixin, LogitBackend, GenerativeBackend):
    """
    Ollama backend supporting first-token logprob extraction with reasoning suppression
    (think: False), persistent connection pooling, and generative diagnostic critiques.
    """

    service_name = "Ollama"
    connect_hint = "Is ollama running? (start it with `ollama serve`)"
    prefers_sequential_prefix_warmup = True

    def transport_hint(self, status_code: int, detail: str) -> str | None:
        """Turns Ollama's 'model "x" not found, try pulling it first' into the exact command."""
        if status_code == 404 and "not found" in detail.lower():
            return f"Pull it first: ollama pull {self.model}"
        return None

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
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
        self.num_ctx: int | None = num_ctx if num_ctx is not None else LogitGateDefaults.num_ctx
        self.options = dict(options) if options else {}
        self.keep_alive = keep_alive
        self._supports_think_param = True
        self._init_http_clients(client=client, async_client=async_client)

    def __repr__(self) -> str:
        return f"OllamaBackend(model='{self.model}', base_url='{self.base_url}')"

    def build_query_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Constructs the chat completion payload ensuring think: False and allowed_tokens guidance."""
        if allowed_tokens:
            clean_tokens = [t.strip() for t in allowed_tokens if isinstance(t, str) and t.strip()]
            tokens_hint = f" ({', '.join(clean_tokens)})" if clean_tokens else ""
            effective_top_k = max(top_k, 2 * len(clean_tokens))
        else:
            tokens_hint = ""
            effective_top_k = top_k
        messages = [
            {
                "role": "system",
                "content": f"Respond with a single character only{tokens_hint}. Do not think or explain.",
            },
            {"role": "user", "content": prompt},
        ]
        opts: dict[str, Any] = {
            "num_predict": 1,
            "temperature": 0.0,
        }
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        if self.options:
            opts.update(self.options)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "logprobs": True,
            "top_logprobs": effective_top_k,
            "options": opts,
        }
        if "think" in kwargs:
            payload["think"] = kwargs["think"]
        elif getattr(self, "_supports_think_param", True):
            payload["think"] = False
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        for k, v in kwargs.items():
            if k == "options" and isinstance(v, dict):
                payload["options"].update(v)
            elif k not in payload and k not in EXCLUDED_INTERNAL_KWARGS:
                payload[k] = v
        return payload

    def _extract_top_logprobs_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            raise BackendError(f"Ollama at {self.base_url} returned non-object JSON payload: {type(data).__name__}")
        msg = data.get("message")
        lp = data.get("logprobs") or (msg.get("logprobs") if isinstance(msg, dict) else None)
        if not lp or not isinstance(lp, list):
            return []
        first = lp[0]
        if isinstance(first, dict):
            if "top_logprobs" in first and isinstance(first["top_logprobs"], list):
                return first["top_logprobs"]
            if "token" in first:
                return lp
        return []

    @translate_transport_errors
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
        if resp.status_code == 400 and "think" in payload and "think" not in kwargs:
            self._supports_think_param = False
            payload.pop("think", None)
            resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
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
        payload = self.build_query_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = await self.async_client.post(f"{self.base_url}/api/chat", json=payload)
        if resp.status_code == 400 and "think" in payload and "think" not in kwargs:
            self._supports_think_param = False
            payload.pop("think", None)
            resp = await self.async_client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_top_logprobs_from_response(data)

    def build_generate_payload(self, prompt: str, max_tokens: int | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Builds the /api/chat payload for a generative critique.

        Thinking defaults on only for budgets >= 256 tokens: with a smaller cap, a reasoning
        model spends the whole budget inside <think> and never produces a verdict.
        """
        opts: dict[str, Any] = {
            "temperature": 0.0,
        }
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if self.options:
            opts.update(self.options)
        kwargs = dict(kwargs)
        think_default = max_tokens is None or max_tokens >= 256
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
            elif k not in EXCLUDED_INTERNAL_KWARGS:
                payload[k] = v
        return payload

    @staticmethod
    def _continuation_payload(
        payload: dict[str, Any], prompt: str, raw_content: str, max_tokens: int | None
    ) -> dict[str, Any]:
        """Mid-thought truncation recovery: closes </think> and resumes with think=False.

        If every generated token landed inside <think>, the visible answer is empty. Pre-filling
        the assistant turn with the partial reasoning plus a closing tag keeps the chain of
        thought and spends the next budget entirely on the visible verdict.
        """
        options = dict(payload["options"])
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        continuation = dict(payload)
        continuation["messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw_content.rstrip() + "\n</think>\n"},
        ]
        continuation["think"] = False
        continuation["options"] = options
        return continuation

    @staticmethod
    def _parse_chat(data: dict[str, Any]) -> tuple[str, int]:
        if not isinstance(data, dict):
            raise BackendError(f"Ollama returned non-object JSON payload: {type(data).__name__}")
        msg = data.get("message")
        content = (msg.get("content") if isinstance(msg, dict) else None) or ""
        if not content and isinstance(msg, dict) and msg.get("thinking"):
            content = f"<think>{msg['thinking']}"
        tokens = data.get("eval_count") or 0
        return str(content), int(tokens)

    @translate_transport_errors
    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        url = f"{self.base_url}/api/chat"
        payload = self.build_generate_payload(prompt, max_tokens, kwargs)
        resp = self.client.post(url, json=payload)
        if resp.status_code == 400 and payload["think"]:
            # Non-thinking models (e.g. qwen2.5, llama3) reject think=True with HTTP 400.
            payload["think"] = False
            resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        content, tokens = self._parse_chat(resp.json())
        visible_text = strip_reasoning_blocks(content)
        if not visible_text and content and payload["think"]:
            resp_cont = self.client.post(url, json=self._continuation_payload(payload, prompt, content, max_tokens))
            resp_cont.raise_for_status()
            content_cont, tokens_cont = self._parse_chat(resp_cont.json())
            return strip_reasoning_blocks(content_cont), tokens + tokens_cont
        return visible_text, tokens

    @translate_transport_errors
    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        url = f"{self.base_url}/api/chat"
        payload = self.build_generate_payload(prompt, max_tokens, kwargs)
        resp = await self.async_client.post(url, json=payload)
        if resp.status_code == 400 and payload["think"]:
            payload["think"] = False
            resp = await self.async_client.post(url, json=payload)
        resp.raise_for_status()
        content, tokens = self._parse_chat(resp.json())
        visible_text = strip_reasoning_blocks(content)
        if not visible_text and content and payload["think"]:
            resp_cont = await self.async_client.post(
                url, json=self._continuation_payload(payload, prompt, content, max_tokens)
            )
            resp_cont.raise_for_status()
            content_cont, tokens_cont = self._parse_chat(resp_cont.json())
            return strip_reasoning_blocks(content_cont), tokens + tokens_cont
        return visible_text, tokens
