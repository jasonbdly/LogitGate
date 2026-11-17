from __future__ import annotations

import os
from typing import Any

import httpx

from logitgate.backends.base import (
    EXCLUDED_INTERNAL_KWARGS,
    GenerativeBackend,
    LogitBackend,
    _HTTPClientMixin,
    _response_detail,
    strip_reasoning_blocks,
    translate_transport_errors,
)
from logitgate.exceptions import (
    BackendError,
    ConfigurationError,
    UnsupportedLogprobsError,
)


class GeminiBackend(_HTTPClientMixin, LogitBackend, GenerativeBackend):
    """
    Native zero-dependency Google Gemini backend using direct REST endpoints
    over persistent httpx connection pools.

    Supports first-token logprob extraction (responseLogprobs) and generative
    critiques with Gemini 2.5 thinking-budget control and reasoning suppression.
    """

    supports_logprobs: bool = True
    service_name: str = "Gemini API"
    connect_hint: str = "Check GEMINI_API_KEY / GOOGLE_API_KEY and base_url."

    def transport_hint(self, status_code: int, detail: str) -> str | None:
        if status_code in (401, 403):
            return "Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass api_key= to GeminiBackend."
        if status_code == 404:
            return f"Check that model '{self.clean_model}' exists in Gemini v1beta."
        if status_code == 400 and ("logprobs" in detail.lower() or "responselogprobs" in detail.lower()):
            return "This Gemini model does not expose logprobs; LogitGate needs responseLogprobs support."
        return None

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
        retry_backoff_base: float = 0.25,
        retry_max_wait: float = 5.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError(f"model must be a non-empty string, got {model!r}")
        self.model = model.strip()
        self._explicit_api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._init_http_clients(
            client=client,
            async_client=async_client,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_max_wait=retry_max_wait,
        )

    def __repr__(self) -> str:
        return f"GeminiBackend(model='{self.model}', base_url='{self.base_url}')"

    @property
    def clean_model(self) -> str:
        if self.model.startswith("models/"):
            return self.model[len("models/"):]
        return self.model

    @property
    def api_key(self) -> str | None:
        if self._explicit_api_key:
            return self._explicit_api_key
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return None

    def _headers(self, is_async: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        resolved_key = self.api_key
        if not resolved_key:
            client_to_check = (self._async_client if is_async else self._client) or self._client or self._async_client
            has_client_auth = False
            if client_to_check is not None and hasattr(client_to_check, "headers") and client_to_check.headers is not None:
                has_client_auth = bool(
                    client_to_check.headers.get("authorization")
                    or client_to_check.headers.get("x-goog-api-key")
                )
            if not has_client_auth:
                raise ConfigurationError(
                    "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                    "or pass api_key= to GeminiBackend."
                )
        if resolved_key:
            headers["x-goog-api-key"] = resolved_key
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
            clamped_k = max(1, min(20, max(top_k, 2 * len(clean_tokens))))
        else:
            clean_tokens = []
            tokens_hint = ""
            clamped_k = max(1, min(20, top_k))

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        generation_config: dict[str, Any] = {
            "temperature": 0.0,
            "maxOutputTokens": 1,
            "responseLogprobs": True,
            "logprobs": clamped_k,
        }

        clean_model_lower = self.clean_model.lower()
        if "thinking_budget" in kwargs:
            if kwargs["thinking_budget"] is not None:
                generation_config["thinkingConfig"] = {"thinkingBudget": kwargs["thinking_budget"]}
        elif "gemini-2.5" in clean_model_lower and "pro" not in clean_model_lower:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if clean_tokens:
            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": f"Respond with a single character only{tokens_hint}. Do not think or explain."
                    }
                ]
            }

        for k, v in kwargs.items():
            if k not in EXCLUDED_INTERNAL_KWARGS and k not in ("thinking_budget", "enable_thinking"):
                if k in ("generationConfig", "generation_config") and isinstance(v, dict):
                    generation_config.update(v)
                elif k not in payload:
                    payload[k] = v
        return payload

    def build_generate_payload(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        generation_config: dict[str, Any] = {"temperature": 0.0}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        clean_model_lower = self.clean_model.lower()
        think = kwargs.get("think")
        thinking_budget = kwargs.get("thinking_budget")
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif think is False and "gemini-2.5" in clean_model_lower and "pro" not in clean_model_lower:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        for k, v in kwargs.items():
            if k not in EXCLUDED_INTERNAL_KWARGS and k not in ("think", "thinking_budget"):
                if k in ("generationConfig", "generation_config") and isinstance(v, dict):
                    generation_config.update(v)
                elif k not in payload:
                    payload[k] = v
        return payload

    def _parse_logprobs_response(self, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            raise BackendError(
                f"{self.service_name} at {self.base_url} returned non-object JSON payload: {type(data).__name__}"
            )
        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list):
            block_reason = ""
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = str(prompt_feedback.get("blockReason") or "")
            msg = f"{self.service_name} at {self.base_url} returned no candidates"
            if block_reason:
                msg = f"{msg} (blocked by safety: {block_reason})"
            raise BackendError(msg)

        first_cand = candidates[0]
        if not isinstance(first_cand, dict):
            raise BackendError(
                f"{self.service_name} at {self.base_url} returned non-dict candidate: {type(first_cand).__name__}"
            )
        logprobs_result = first_cand.get("logprobsResult")
        if logprobs_result is None or not isinstance(logprobs_result, dict):
            raise UnsupportedLogprobsError(
                f"{self.service_name} at {self.base_url} returned response without logprobsResult"
            )
        top_candidates = logprobs_result.get("topCandidates")
        if not top_candidates or not isinstance(top_candidates, list) or not isinstance(top_candidates[0], dict):
            return []
        inner_cands = top_candidates[0].get("candidates")
        if not inner_cands or not isinstance(inner_cands, list):
            return []
        return [
            {
                "token": str(item.get("token") or ""),
                "logprob": float(item["logProbability"]),
            }
            for item in inner_cands
            if isinstance(item, dict) and item.get("logProbability") is not None
        ]

    def _extract_generate_response(self, data: Any) -> tuple[str, int]:
        if not isinstance(data, dict):
            raise BackendError(
                f"{self.service_name} at {self.base_url} returned non-object JSON payload: {type(data).__name__}"
            )
        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list):
            block_reason = ""
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = str(prompt_feedback.get("blockReason") or "")
            msg = f"{self.service_name} at {self.base_url} returned no candidates"
            if block_reason:
                msg = f"{msg} (blocked by safety: {block_reason})"
            raise BackendError(msg)

        first_cand = candidates[0]
        if not isinstance(first_cand, dict):
            raise BackendError(
                f"{self.service_name} at {self.base_url} returned non-dict candidate: {type(first_cand).__name__}"
            )
        content = first_cand.get("content") or {}
        parts = content.get("parts") or []
        text_parts: list[str] = []
        for p in parts:
            if isinstance(p, dict):
                if p.get("thought") is True:
                    continue
                t = p.get("text")
                if t:
                    text_parts.append(str(t))
        raw_text = "".join(text_parts)
        final_text = strip_reasoning_blocks(raw_text)

        usage = data.get("usageMetadata")
        tokens = 0
        if isinstance(usage, dict) and "candidatesTokenCount" in usage and usage["candidatesTokenCount"] is not None:
            tokens = int(usage["candidatesTokenCount"])
        elif final_text:
            tokens = max(1, len(final_text.split()))

        return final_text, tokens

    @translate_transport_errors
    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        headers = self._headers(is_async=False)
        payload = self.build_query_payload(prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs)
        url = f"{self.base_url.rstrip('/')}/models/{self.clean_model}:generateContent"
        resp = self._post_with_retry(url, json=payload, headers=headers)
        if resp.status_code == 400:
            detail = _response_detail(resp)
            detail_lower = detail.lower()
            if "logprobs" in detail_lower or "responselogprobs" in detail_lower:
                raise UnsupportedLogprobsError(
                    f"{self.service_name} at {self.base_url} does not support logprobs: HTTP 400 {detail}"
                )
        resp.raise_for_status()
        return self._parse_logprobs_response(resp.json())

    @translate_transport_errors
    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        headers = self._headers(is_async=True)
        payload = self.build_query_payload(prompt, top_k=top_k, allowed_tokens=allowed_tokens, **kwargs)
        url = f"{self.base_url.rstrip('/')}/models/{self.clean_model}:generateContent"
        resp = await self._apost_with_retry(url, json=payload, headers=headers)
        if resp.status_code == 400:
            detail = _response_detail(resp)
            detail_lower = detail.lower()
            if "logprobs" in detail_lower or "responselogprobs" in detail_lower:
                raise UnsupportedLogprobsError(
                    f"{self.service_name} at {self.base_url} does not support logprobs: HTTP 400 {detail}"
                )
        resp.raise_for_status()
        return self._parse_logprobs_response(resp.json())

    @translate_transport_errors
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, int]:
        headers = self._headers(is_async=False)
        payload = self.build_generate_payload(prompt, max_tokens=max_tokens, **kwargs)
        url = f"{self.base_url.rstrip('/')}/models/{self.clean_model}:generateContent"
        resp = self._post_with_retry(url, json=payload, headers=headers)
        resp.raise_for_status()
        return self._extract_generate_response(resp.json())

    @translate_transport_errors
    async def agenerate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, int]:
        headers = self._headers(is_async=True)
        payload = self.build_generate_payload(prompt, max_tokens=max_tokens, **kwargs)
        url = f"{self.base_url.rstrip('/')}/models/{self.clean_model}:generateContent"
        resp = await self._apost_with_retry(url, json=payload, headers=headers)
        resp.raise_for_status()
        return self._extract_generate_response(resp.json())
