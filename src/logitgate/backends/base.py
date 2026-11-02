from __future__ import annotations

import abc
import asyncio
import math
import re
from typing import Any, Optional
from urllib.parse import urlparse


def strip_reasoning_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning traces from model responses."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class LogitBackend(abc.ABC):
    """Abstract base class for extracting first-token log-probabilities."""

    @abc.abstractmethod
    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Returns top_k token logprobs: [{'token': str, 'logprob': float}, ...]"""
        pass

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Async variant of query_top_logprobs."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.query_top_logprobs, prompt, top_k, allowed_tokens
        )


class GenerativeBackend(abc.ABC):
    """Abstract base class for generative reasoning & diagnostics."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> tuple[str, int]:
        """Returns (generated_text, tokens_generated)."""
        pass

    async def agenerate(self, prompt: str) -> tuple[str, int]:
        """Async variant of generate."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


class StubBackend(LogitBackend, GenerativeBackend):
    """
    Offline deterministic stub backend for sub-millisecond unit tests and CI.
    Records all prompted inputs in self.prompt_log.
    """

    def __init__(
        self,
        default_prob: float = 0.95,
        forward_top_logprobs: Optional[list[dict[str, Any]]] = None,
        reverse_top_logprobs: Optional[list[dict[str, Any]]] = None,
        logprob_responses: Optional[list[list[dict[str, Any]]]] = None,
        diagnostic_text: str = "[VERDICT: PASS] All criteria met.",
        tokens_generated: int = 0,
    ) -> None:
        self.default_prob = default_prob
        self.forward_top_logprobs = forward_top_logprobs
        self.reverse_top_logprobs = reverse_top_logprobs
        self.logprob_responses = list(logprob_responses) if logprob_responses else None
        self.diagnostic_text = diagnostic_text
        self.tokens_generated = tokens_generated
        self.prompt_log: list[str] = []
        self._response_idx = 0

    def __repr__(self) -> str:
        return f"StubBackend(default_prob={self.default_prob})"

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        self.prompt_log.append(prompt)

        # 1. Explicit queue response
        if self.logprob_responses:
            resp = self.logprob_responses[self._response_idx % len(self.logprob_responses)]
            self._response_idx += 1
            return resp

        # 2. Check if this is Pass 2 (reversed mapping where A is Fail / violates)
        is_reverse_pass = (
            "A: The candidate artifact violates" in prompt
            or "A: The content violates" in prompt
            or "A: The candidate does not" in prompt
        )

        if is_reverse_pass:
            if self.reverse_top_logprobs is not None:
                return self.reverse_top_logprobs
            p = max(min(self.default_prob, 0.9999), 0.0001)
            return [
                {"token": "B", "logprob": math.log(p)},
                {"token": "A", "logprob": math.log(1.0 - p)},
            ]
        else:
            if self.forward_top_logprobs is not None:
                return self.forward_top_logprobs
            p = max(min(self.default_prob, 0.9999), 0.0001)
            return [
                {"token": "A", "logprob": math.log(p)},
                {"token": "B", "logprob": math.log(1.0 - p)},
            ]

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        return self.query_top_logprobs(prompt, top_k, allowed_tokens)

    def generate(self, prompt: str) -> tuple[str, int]:
        self.prompt_log.append(prompt)
        return self.diagnostic_text, self.tokens_generated

    async def agenerate(self, prompt: str) -> tuple[str, int]:
        return self.generate(prompt)


def resolve_logit_backend(
    backend_spec: Optional[str | LogitBackend] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> LogitBackend:
    if isinstance(backend_spec, LogitBackend):
        return backend_spec

    from logitgate.config import get_default_backend, get_default_model

    if backend_spec is None or backend_spec == "default":
        backend_spec = get_default_backend()
        if model is None:
            model = get_default_model()
    elif model is None:
        model = get_default_model()

    spec = str(backend_spec).strip()
    if spec.startswith("stub://") or spec == "stub" or spec == "stub://default":
        prob = 0.95
        if "/" in spec and spec not in ("stub://", "stub://default"):
            part = spec.split("/")[-1]
            try:
                prob = float(part)
            except ValueError:
                pass
        return StubBackend(default_prob=prob, **kwargs)

    if spec.startswith("ollama://"):
        from logitgate.backends.ollama import OllamaBackend
        parsed = urlparse(spec)
        netloc = parsed.netloc or "localhost:11434"
        base_url = f"http://{netloc}"
        parsed_model = parsed.path.lstrip("/") if parsed.path else None
        effective_model = model or parsed_model or "qwen2.5-coder:14b"
        return OllamaBackend(base_url=base_url, model=effective_model, **kwargs)

    if spec.startswith("openai://") or spec.startswith("vllm://"):
        from logitgate.backends.openai_compat import OpenAICompatBackend
        parsed = urlparse(spec)
        scheme = "https" if spec.startswith("openai://") else "http"
        base_url = f"{scheme}://{parsed.netloc}{parsed.path}"
        effective_model = model or "gpt-4o-mini"
        return OpenAICompatBackend(base_url=base_url, model=effective_model, **kwargs)

    if spec.startswith("llamacpp://") or spec == "llamacpp":
        from logitgate.backends.llamacpp import LlamaCppBackend
        if spec == "llamacpp":
            return LlamaCppBackend(model=model, **kwargs)
        parsed = urlparse(spec)
        if parsed.netloc:
            base_url = f"http://{parsed.netloc}"
            parsed_model = parsed.path.lstrip("/") if parsed.path else None
            effective_model = model or parsed_model or None
            return LlamaCppBackend(base_url=base_url, model=effective_model, **kwargs)
        else:
            effective_model = model or (parsed.path.lstrip("/") if parsed.path else None)
            return LlamaCppBackend(model=effective_model, **kwargs)

    # Fallback to Ollama or Stub
    if "stub" in spec:
        return StubBackend(**kwargs)

    from logitgate.backends.ollama import OllamaBackend
    return OllamaBackend(model=model or spec, **kwargs)


def resolve_generative_backend(
    backend_spec: Optional[str | GenerativeBackend] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> GenerativeBackend:
    if isinstance(backend_spec, GenerativeBackend):
        return backend_spec

    backend = resolve_logit_backend(backend_spec, model=model, **kwargs)
    if isinstance(backend, GenerativeBackend):
        return backend
    raise TypeError(f"Backend {backend} does not implement GenerativeBackend")
