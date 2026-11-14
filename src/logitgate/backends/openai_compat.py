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
from logitgate.debiasing.softmax import has_any_target_token
from logitgate.exceptions import BackendError, ConfigurationError


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
        *,
        suppress_thinking: bool = True,
        auto_logit_bias: bool = False,
    ) -> None:
        if not isinstance(suppress_thinking, bool):
            raise ConfigurationError(f"suppress_thinking must be a bool, got {suppress_thinking!r}")
        if not isinstance(auto_logit_bias, bool):
            raise ConfigurationError(f"auto_logit_bias must be a bool, got {auto_logit_bias!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-dummy")
        self.timeout = timeout
        self.logit_bias = logit_bias
        self.suppress_thinking = suppress_thinking
        self.auto_logit_bias = auto_logit_bias
        self._supports_reasoning_effort = True
        self._supports_chat_template_no_think = True
        self._use_chat_template_no_think = False
        self._auto_logit_bias_supported = True
        self._logit_bias_cache: dict[tuple[str, ...], dict[str, int]] = {}
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
                    "content": f"Respond with a single character only{tokens_hint}. Do not think or explain.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": effective_top_k,
        }
        if getattr(self, "suppress_thinking", False):
            if getattr(self, "_supports_reasoning_effort", True) and "reasoning_effort" not in kwargs:
                payload["reasoning_effort"] = "none"
            if getattr(self, "_use_chat_template_no_think", False) and "chat_template_kwargs" not in kwargs:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        logit_bias = kwargs.get("logit_bias", self.logit_bias)
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias
        for k, v in kwargs.items():
            if k not in payload and k != "logit_bias" and k not in EXCLUDED_INTERNAL_KWARGS:
                payload[k] = v
        return payload

    @staticmethod
    def _is_thinking_prefill_trap(lps: list[dict[str, Any]], allowed_tokens: list[str] | None) -> bool:
        if not lps:
            return True
        if allowed_tokens:
            clean_tokens = [t.strip() for t in allowed_tokens if isinstance(t, str) and t.strip()]
            if clean_tokens and not has_any_target_token(lps, clean_tokens):
                top_tok = str(lps[0].get("token") or "").strip().lower()
                if (
                    top_tok in ("", "<think>", "thought", "<|thought|>", "<seed:think>", "[think]", "<reasoning>")
                    or top_tok.startswith("<think")
                ):
                    return True
        return False

    def _tokenize_payloads(self, variant: str, mode: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        candidates: list[tuple[str, dict[str, Any]]] = [
            ("vllm", {"model": self.model, "prompt": variant, "add_special_tokens": False}),
            ("llamacpp", {"content": variant, "add_special": False}),
        ]
        if mode is not None:
            return [(m, p) for m, p in candidates if m == mode]
        return candidates

    def _tokenize_urls(self) -> list[str]:
        urls = [f"{self.base_url}/tokenize"]
        if self.base_url.endswith("/v1"):
            root = self.base_url[:-3].rstrip("/")
            if root:
                urls.append(f"{root}/tokenize")
        return urls

    @staticmethod
    def _token_variants(allowed_tokens: list[str]) -> tuple[tuple[str, ...], list[str]]:
        clean = [t.strip() for t in allowed_tokens if isinstance(t, str) and t.strip()]
        seen: set[str] = set()
        variants: list[str] = []
        for tok in clean:
            for cand in (tok, tok.lower(), tok.upper(), f" {tok}", f" {tok.lower()}", f" {tok.upper()}"):
                if cand not in seen:
                    seen.add(cand)
                    variants.append(cand)
        return tuple(clean), variants

    def _prepare_tokenize_lookup(
        self, allowed_tokens: list[str] | None
    ) -> tuple[bool, dict[str, int] | None, tuple[str, ...], list[str], list[str]]:
        """Returns (should_query, cached_bias, cache_key, variants, urls)."""
        if (
            not getattr(self, "auto_logit_bias", False)
            or not getattr(self, "_auto_logit_bias_supported", True)
            or not allowed_tokens
        ):
            return False, None, (), [], []
        cache_key, variants = self._token_variants(allowed_tokens)
        if not cache_key:
            return False, None, (), [], []
        with self._http_lock:
            cached = self._logit_bias_cache.get(cache_key)
        if cached is not None:
            return False, cached, cache_key, [], []
        return True, None, cache_key, variants, self._tokenize_urls()

    @staticmethod
    def _parse_tokenize_response(resp: httpx.Response) -> tuple[bool, int | None]:
        """Returns (matched, single_token_id_or_none) for a /tokenize HTTP response."""
        if resp.status_code != 200:
            return False, None
        try:
            body = resp.json()
        except ValueError:
            return False, None
        if not isinstance(body, dict) or not isinstance(body.get("tokens"), list):
            return False, None
        toks = body["tokens"]
        if not toks:
            return False, None
        if len(toks) == 1 and isinstance(toks[0], int) and not isinstance(toks[0], bool):
            return True, toks[0]
        return True, None

    def _finalize_logit_bias(
        self, cache_key: tuple[str, ...], token_ids: set[int]
    ) -> dict[str, int] | None:
        if not token_ids:
            self._auto_logit_bias_supported = False
            return None
        bias_map = {str(tid): 100 for tid in sorted(token_ids)}
        with self._http_lock:
            self._logit_bias_cache[cache_key] = bias_map
        return bias_map

    def _resolve_auto_logit_bias_sync(self, allowed_tokens: list[str] | None) -> dict[str, int] | None:
        should_query, cached, cache_key, variants, urls = self._prepare_tokenize_lookup(allowed_tokens)
        if not should_query:
            return cached
        token_ids: set[int] = set()
        working_url: str | None = None
        working_mode: str | None = None
        for variant in variants:
            req_urls = [working_url] if working_url is not None else urls
            req_payloads = self._tokenize_payloads(variant, mode=working_mode)
            matched = False
            for url in req_urls:
                for mode, req_payload in req_payloads:
                    try:
                        resp = self.client.post(url, json=req_payload, headers=self._headers())
                    except httpx.HTTPError:
                        continue
                    ok, tid = self._parse_tokenize_response(resp)
                    if ok:
                        working_url = url
                        working_mode = mode
                        matched = True
                        if tid is not None:
                            token_ids.add(tid)
                        break
                if matched:
                    break
            if not matched:
                self._auto_logit_bias_supported = False
                return None
        return self._finalize_logit_bias(cache_key, token_ids)

    async def _resolve_auto_logit_bias_async(self, allowed_tokens: list[str] | None) -> dict[str, int] | None:
        should_query, cached, cache_key, variants, urls = self._prepare_tokenize_lookup(allowed_tokens)
        if not should_query:
            return cached
        token_ids: set[int] = set()
        working_url: str | None = None
        working_mode: str | None = None
        for variant in variants:
            req_urls = [working_url] if working_url is not None else urls
            req_payloads = self._tokenize_payloads(variant, mode=working_mode)
            matched = False
            for url in req_urls:
                for mode, req_payload in req_payloads:
                    try:
                        resp = await self.async_client.post(url, json=req_payload, headers=self._headers())
                    except httpx.HTTPError:
                        continue
                    ok, tid = self._parse_tokenize_response(resp)
                    if ok:
                        working_url = url
                        working_mode = mode
                        matched = True
                        if tid is not None:
                            token_ids.add(tid)
                        break
                if matched:
                    break
            if not matched:
                self._auto_logit_bias_supported = False
                return None
        return self._finalize_logit_bias(cache_key, token_ids)

    def _build_chat_template_fallback_payload(
        self, payload: dict[str, Any], lps: list[dict[str, Any]], allowed_tokens: list[str] | None, kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not (
            getattr(self, "suppress_thinking", False)
            and getattr(self, "_supports_chat_template_no_think", True)
            and not getattr(self, "_use_chat_template_no_think", False)
            and "chat_template_kwargs" not in kwargs
            and self._is_thinking_prefill_trap(lps, allowed_tokens)
        ):
            return None
        fb_payload = dict(payload)
        fb_payload["chat_template_kwargs"] = {"enable_thinking": False}
        if "reasoning_effort" in fb_payload and "reasoning_effort" not in kwargs:
            fb_payload.pop("reasoning_effort", None)
        return fb_payload

    def _handle_chat_template_fallback_response(
        self, resp_fb: httpx.Response, allowed_tokens: list[str] | None
    ) -> list[dict[str, Any]] | None:
        if resp_fb.status_code == 200:
            lps_fb = self._extract_top_logprobs_from_response(resp_fb.json())
            if lps_fb and not self._is_thinking_prefill_trap(lps_fb, allowed_tokens):
                self._use_chat_template_no_think = True
                self._supports_reasoning_effort = False
                return lps_fb
        self._supports_chat_template_no_think = False
        return None

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
        if "logit_bias" not in payload:
            auto_bias = self._resolve_auto_logit_bias_sync(allowed_tokens)
            if auto_bias:
                payload["logit_bias"] = auto_bias
        resp = self.client.post(url, json=payload, headers=self._headers())
        if resp.status_code in (400, 422) and "reasoning_effort" in payload and "reasoning_effort" not in kwargs:
            self._supports_reasoning_effort = False
            payload.pop("reasoning_effort", None)
            resp = self.client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        lps = self._extract_top_logprobs_from_response(resp.json())
        fb_payload = self._build_chat_template_fallback_payload(payload, lps, allowed_tokens, kwargs)
        if fb_payload is not None:
            resp_fb = self.client.post(url, json=fb_payload, headers=self._headers())
            lps_fb = self._handle_chat_template_fallback_response(resp_fb, allowed_tokens)
            if lps_fb is not None:
                return lps_fb
        return lps

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
        if "logit_bias" not in payload:
            auto_bias = await self._resolve_auto_logit_bias_async(allowed_tokens)
            if auto_bias:
                payload["logit_bias"] = auto_bias
        resp = await self.async_client.post(url, json=payload, headers=self._headers())
        if resp.status_code in (400, 422) and "reasoning_effort" in payload and "reasoning_effort" not in kwargs:
            self._supports_reasoning_effort = False
            payload.pop("reasoning_effort", None)
            resp = await self.async_client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        lps = self._extract_top_logprobs_from_response(resp.json())
        fb_payload = self._build_chat_template_fallback_payload(payload, lps, allowed_tokens, kwargs)
        if fb_payload is not None:
            resp_fb = await self.async_client.post(url, json=fb_payload, headers=self._headers())
            lps_fb = self._handle_chat_template_fallback_response(resp_fb, allowed_tokens)
            if lps_fb is not None:
                return lps_fb
        return lps

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
