from __future__ import annotations

import asyncio
import math
from typing import Any, Optional
import httpx

from logitgate.backends.base import LogitBackend, GenerativeBackend, strip_reasoning_blocks


class LlamaCppBackend(LogitBackend, GenerativeBackend):
    """
    Backend for llama.cpp supporting:
      Mode A: In-process llama_cpp.Llama instance (via create_chat_completion)
      Mode B: llama-server HTTP endpoint (/v1/chat/completions with /completion fallback)
    """

    def __init__(
        self,
        llm: Any = None,
        base_url: str = "http://localhost:8080",
        model: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.llm = llm
        self.base_url = base_url.rstrip("/") if base_url else "http://localhost:8080"
        self.model = model
        self.timeout = timeout
        self._client = client
        self._async_client = async_client
        self._use_completion_fallback = False

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

    def __enter__(self) -> LlamaCppBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> LlamaCppBackend:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    @staticmethod
    def _extract_chat_top_logprobs(data: dict[str, Any]) -> list[dict[str, Any]]:
        choices = data.get("choices", [])
        if not choices:
            return []
        logprobs = choices[0].get("logprobs")
        if not logprobs or not isinstance(logprobs, dict):
            return []
        content = logprobs.get("content")
        if content and isinstance(content, list) and len(content) > 0:
            top_lps = content[0].get("top_logprobs", [])
            return [
                {"token": item.get("token", ""), "logprob": float(item.get("logprob", 0.0))}
                for item in top_lps
                if isinstance(item, dict) and "token" in item
            ]
        top_lps = logprobs.get("top_logprobs")
        if top_lps and isinstance(top_lps, list):
            first = top_lps[0] if isinstance(top_lps[0], list) else top_lps
            return [
                {"token": item.get("token", ""), "logprob": float(item.get("logprob", 0.0))}
                for item in first
                if isinstance(item, dict) and "token" in item
            ]
        return []

    @staticmethod
    def _extract_completion_logprobs(data: dict[str, Any]) -> list[dict[str, Any]]:
        comp_probs = data.get("completion_probabilities", [])
        if comp_probs and isinstance(comp_probs, list):
            first = comp_probs[0]
            probs = first.get("probs", []) if isinstance(first, dict) else []
            results = []
            for p in probs:
                if not isinstance(p, dict):
                    continue
                tok = p.get("tok_str", p.get("token", ""))
                if "logprob" in p:
                    lp = float(p["logprob"])
                elif "prob" in p:
                    lp = math.log(max(float(p["prob"]), 1e-12))
                else:
                    lp = 0.0
                results.append({"token": tok, "logprob": lp})
            return results
        return []

    def _in_process_query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": "Respond with a single character only. Do not think or explain.",
            },
            {"role": "user", "content": prompt},
        ]
        res = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top_k,
            **kwargs,
        )
        return self._extract_chat_top_logprobs(res)

    def _in_process_generate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        res = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            **kwargs,
        )
        choices = res.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        tokens = res.get("usage", {}).get("completion_tokens", 0)
        return strip_reasoning_blocks(content), tokens

    def build_chat_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model or "default",
            "messages": [
                {
                    "role": "system",
                    "content": "Respond with a single character only. Do not think or explain.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": top_k,
        }
        if "logit_bias" in kwargs:
            payload["logit_bias"] = kwargs["logit_bias"]
        for k, v in kwargs.items():
            if k not in payload and k != "logit_bias":
                payload[k] = v
        return payload

    def build_completion_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": top_k,
        }
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v
        return payload

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.llm is not None:
            return self._in_process_query_top_logprobs(
                prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
            )

        if not self._use_completion_fallback:
            url = f"{self.base_url}/v1/chat/completions"
            payload = self.build_chat_payload(
                prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
            )
            resp = self.client.post(url, json=payload)
            if resp.status_code == 404:
                self._use_completion_fallback = True
            else:
                resp.raise_for_status()
                return self._extract_chat_top_logprobs(resp.json())

        # Fallback to /completion
        url = f"{self.base_url}/completion"
        payload = self.build_completion_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return self._extract_completion_logprobs(resp.json())

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.llm is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._in_process_query_top_logprobs, prompt, top_k, allowed_tokens
            )

        if not self._use_completion_fallback:
            url = f"{self.base_url}/v1/chat/completions"
            payload = self.build_chat_payload(
                prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
            )
            resp = await self.async_client.post(url, json=payload)
            if resp.status_code == 404:
                self._use_completion_fallback = True
            else:
                resp.raise_for_status()
                return self._extract_chat_top_logprobs(resp.json())

        # Fallback to /completion
        url = f"{self.base_url}/completion"
        payload = self.build_completion_payload(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )
        resp = await self.async_client.post(url, json=payload)
        resp.raise_for_status()
        return self._extract_completion_logprobs(resp.json())

    def generate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        if self.llm is not None:
            return self._in_process_generate(prompt, **kwargs)

        if not self._use_completion_fallback:
            url = f"{self.base_url}/v1/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            payload.update(kwargs)
            resp = self.client.post(url, json=payload)
            if resp.status_code == 404:
                self._use_completion_fallback = True
            else:
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                return strip_reasoning_blocks(content), tokens

        # Fallback to /completion
        url = f"{self.base_url}/completion"
        payload = {"prompt": prompt, "temperature": 0.0}
        payload.update(kwargs)
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", "")
        tokens = data.get("tokens_predicted", data.get("tokens_evaluated", 0))
        return strip_reasoning_blocks(content), tokens

    async def agenerate(self, prompt: str, **kwargs: Any) -> tuple[str, int]:
        if self.llm is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._in_process_generate, prompt)

        if not self._use_completion_fallback:
            url = f"{self.base_url}/v1/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            payload.update(kwargs)
            resp = await self.async_client.post(url, json=payload)
            if resp.status_code == 404:
                self._use_completion_fallback = True
            else:
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                return strip_reasoning_blocks(content), tokens

        # Fallback to /completion
        url = f"{self.base_url}/completion"
        payload = {"prompt": prompt, "temperature": 0.0}
        payload.update(kwargs)
        resp = await self.async_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", "")
        tokens = data.get("tokens_predicted", data.get("tokens_evaluated", 0))
        return strip_reasoning_blocks(content), tokens
