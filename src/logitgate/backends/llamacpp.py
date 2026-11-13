from __future__ import annotations

import asyncio
import math
import threading
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


class LlamaCppBackend(_HTTPClientMixin, LogitBackend, GenerativeBackend):
    """
    Backend for llama.cpp supporting:
      Mode A: In-process llama_cpp.Llama instance (via create_chat_completion)
      Mode B: llama-server HTTP endpoint (/v1/chat/completions with /completion fallback)
    """

    service_name = "llama.cpp server"
    connect_hint = "Is llama-server running? (e.g. `llama-server -m model.gguf --port 8080`)"

    def __init__(
        self,
        llm: Any = None,
        base_url: str = "http://localhost:8080",
        model: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.llm = llm
        self.base_url = base_url.rstrip("/") if base_url is not None else "http://localhost:8080"
        self.model = model
        self.timeout = timeout
        self._init_http_clients(client=client, async_client=async_client)
        self._use_completion_fallback = False
        self._llm_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"LlamaCppBackend(model={self.model!r}, base_url='{self.base_url}')"

    @staticmethod
    def _extract_chat_top_logprobs(data: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            raise BackendError(f"llama.cpp server returned non-object JSON payload: {type(data).__name__}")
        choices = data.get("choices", [])
        if not choices or not isinstance(choices, list) or not isinstance(choices[0], dict):
            return []
        logprobs = choices[0].get("logprobs")
        if not logprobs or not isinstance(logprobs, dict):
            return []
        content = logprobs.get("content")
        if content and isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            top_lps = content[0].get("top_logprobs") or []
            return [
                {
                    "token": str(item.get("token") or ""),
                    "logprob": float(item["logprob"]),
                }
                for item in top_lps
                if isinstance(item, dict) and "token" in item and item.get("logprob") is not None
            ]
        top_lps = logprobs.get("top_logprobs")
        if top_lps and isinstance(top_lps, list):
            first = top_lps[0] if isinstance(top_lps[0], list) else top_lps
            return [
                {
                    "token": str(item.get("token") or ""),
                    "logprob": float(item["logprob"]),
                }
                for item in first
                if isinstance(item, dict) and "token" in item and item.get("logprob") is not None
            ]
        return []

    @staticmethod
    def _extract_completion_logprobs(data: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            raise BackendError(f"llama.cpp server returned non-object JSON payload: {type(data).__name__}")
        comp_probs = data.get("completion_probabilities", [])
        if comp_probs and isinstance(comp_probs, list):
            first = comp_probs[0]
            probs = first.get("probs", []) if isinstance(first, dict) else []
            results = []
            for p in probs:
                if not isinstance(p, dict):
                    continue
                tok = p.get("tok_str", p.get("token", ""))
                if "logprob" in p and p["logprob"] is not None:
                    lp = float(p["logprob"])
                elif "prob" in p and p["prob"] is not None:
                    lp = math.log(max(float(p["prob"]), 1e-12))
                else:
                    continue
                results.append({"token": tok, "logprob": lp})
            return results
        return []

    @staticmethod
    def _extract_chat_completion(data: Any) -> tuple[str, int]:
        if not isinstance(data, dict):
            raise BackendError(f"llama.cpp server returned non-object JSON payload: {type(data).__name__}")
        choices = data.get("choices")
        content = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and msg.get("content") is not None:
                content = str(msg["content"])
        usage = data.get("usage")
        tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
        return strip_reasoning_blocks(content), tokens

    @staticmethod
    def _extract_raw_completion(data: Any) -> tuple[str, int]:
        if not isinstance(data, dict):
            raise BackendError(f"llama.cpp server returned non-object JSON payload: {type(data).__name__}")
        content = str(data.get("content") or "")
        tokens = int(data.get("tokens_predicted") or data.get("tokens_evaluated") or 0)
        return strip_reasoning_blocks(content), tokens

    def _ensure_in_process_llm(self) -> bool:
        if self.llm is not None:
            return True
        if self.base_url:
            return False
        with self._llm_lock:
            if self.llm is not None:
                return True
            if not self.model:
                raise BackendError(
                    "Local llama.cpp backend requires a GGUF model path (e.g. 'llamacpp:///path/to/model.gguf') "
                    "or an HTTP server URL (e.g. 'llamacpp://localhost:8080')."
                )
            try:
                from llama_cpp import Llama
            except ImportError as err:
                raise BackendError(
                    "In-process llama.cpp requires 'llama-cpp-python' (pip install 'logitgate[llamacpp]') "
                    "or an HTTP server URL (e.g. 'llamacpp://localhost:8080')."
                ) from err
            try:
                self.llm = Llama(model_path=self.model, logits_all=True, verbose=False)
            except Exception as err:
                raise BackendError(f"Failed to load GGUF model '{self.model}': {err}") from err
            return True

    def _in_process_query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
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
        call_kwargs: dict[str, Any] = {
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": effective_top_k,
        }
        for k, v in kwargs.items():
            if k not in EXCLUDED_INTERNAL_KWARGS:
                call_kwargs[k] = v
        with self._llm_lock:
            res = self.llm.create_chat_completion(
                messages=messages,
                **call_kwargs,
            )
        return self._extract_chat_top_logprobs(res)

    def _in_process_generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        call_kwargs: dict[str, Any] = {"temperature": 0.0}
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        for k, v in kwargs.items():
            if k not in EXCLUDED_INTERNAL_KWARGS:
                call_kwargs[k] = v
        with self._llm_lock:
            res = self.llm.create_chat_completion(
                messages=messages,
                **call_kwargs,
            )
        return self._extract_chat_completion(res)

    def build_chat_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if allowed_tokens:
            clean_tokens = [t.strip() for t in allowed_tokens if isinstance(t, str) and t.strip()]
            tokens_hint = f" ({', '.join(clean_tokens)})" if clean_tokens else ""
            effective_top_k = max(top_k, 2 * len(clean_tokens))
        else:
            tokens_hint = ""
            effective_top_k = top_k
        payload: dict[str, Any] = {
            "model": self.model or "default",
            "messages": [
                {
                    "role": "system",
                    "content": f"Respond with a single character only{tokens_hint}. Do not think or explain.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": effective_top_k,
        }
        if "logit_bias" in kwargs:
            payload["logit_bias"] = kwargs["logit_bias"]
        for k, v in kwargs.items():
            if k not in payload and k != "logit_bias" and k not in EXCLUDED_INTERNAL_KWARGS:
                payload[k] = v
        return payload

    def build_completion_payload(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": top_k,
        }
        for k, v in kwargs.items():
            if k not in payload and k not in EXCLUDED_INTERNAL_KWARGS:
                payload[k] = v
        return payload

    @translate_transport_errors
    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self._ensure_in_process_llm():
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

    def _sync_in_process_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self._ensure_in_process_llm()
        return self._in_process_query_top_logprobs(
            prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
        )

    def _sync_in_process_generate(
        self, prompt: str, max_tokens: int | None = None, **kwargs: Any
    ) -> tuple[str, int]:
        self._ensure_in_process_llm()
        return self._in_process_generate(prompt, max_tokens=max_tokens, **kwargs)

    @translate_transport_errors
    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.llm is not None or not self.base_url:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._sync_in_process_logprobs(
                    prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs
                ),
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

    @translate_transport_errors
    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        if self._ensure_in_process_llm():
            return self._in_process_generate(prompt, max_tokens=max_tokens, **kwargs)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDED_INTERNAL_KWARGS}
        if not self._use_completion_fallback:
            url = f"{self.base_url}/v1/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            payload.update(filtered_kwargs)
            resp = self.client.post(url, json=payload)
            if resp.status_code == 404:
                self._use_completion_fallback = True
            else:
                resp.raise_for_status()
                return self._extract_chat_completion(resp.json())

        # Fallback to /completion
        url = f"{self.base_url}/completion"
        payload = {"prompt": prompt, "temperature": 0.0}
        if max_tokens is not None:
            payload["n_predict"] = max_tokens
        payload.update(filtered_kwargs)
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return self._extract_raw_completion(resp.json())

    @translate_transport_errors
    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        if self.llm is not None or not self.base_url:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._sync_in_process_generate(prompt, max_tokens=max_tokens, **kwargs)
            )

        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDED_INTERNAL_KWARGS}
        if not self._use_completion_fallback:
            url = f"{self.base_url}/v1/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            payload.update(filtered_kwargs)
            resp = await self.async_client.post(url, json=payload)
            if resp.status_code == 404:
                self._use_completion_fallback = True
            else:
                resp.raise_for_status()
                return self._extract_chat_completion(resp.json())

        # Fallback to /completion
        url = f"{self.base_url}/completion"
        payload = {"prompt": prompt, "temperature": 0.0}
        if max_tokens is not None:
            payload["n_predict"] = max_tokens
        payload.update(filtered_kwargs)
        resp = await self.async_client.post(url, json=payload)
        resp.raise_for_status()
        return self._extract_raw_completion(resp.json())
