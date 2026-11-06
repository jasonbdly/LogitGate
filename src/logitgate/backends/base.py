from __future__ import annotations

import abc
import asyncio
import functools
import inspect
import math
import re
from collections.abc import Callable
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

import httpx
from typing_extensions import Self

from logitgate.exceptions import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
)

_F = TypeVar("_F", bound=Callable[..., Any])
_OFFLINE_HINT = "Or set backend='stub://' for offline testing."


def _response_detail(response: httpx.Response) -> str:
    """Extracts the server's own error message (Ollama: {"error": str}, OpenAI: {"error": {...}})."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            error = error.get("message")
        if error:
            return str(error)[:300]
    try:
        return response.text.strip()[:300] or response.reason_phrase
    except httpx.ResponseNotRead:
        return response.reason_phrase


def describe_transport_error(backend: Any, exc: httpx.HTTPError) -> BackendError:
    """Converts a raw ``httpx`` failure into a `BackendError` that says what to do next.

    Backends customize the advice through optional attributes rather than subclassing:
    ``service_name`` (display name), ``connect_hint`` (how to start the server), and
    ``transport_hint(status_code, detail) -> str | None`` (status-specific advice such as
    ``ollama pull <model>`` on a 404).
    """
    service = getattr(backend, "service_name", type(backend).__name__)
    base_url = getattr(backend, "base_url", None) or "<unknown url>"

    if isinstance(exc, httpx.TimeoutException):
        timeout = getattr(backend, "timeout", None)
        within = f" within {timeout:g}s" if isinstance(timeout, (int, float)) else ""
        return BackendTimeoutError(
            f"{service} at {base_url} did not respond{within} ({type(exc).__name__}). "
            "The first request after a model loads is often far slower than steady state: "
            "pre-warm the model, or pass a larger timeout= to the backend."
        )
    if isinstance(exc, httpx.ConnectError):
        hint = getattr(backend, "connect_hint", f"Is the {service} server running?")
        return BackendUnavailableError(f"Cannot connect to {service} at {base_url}. {hint} {_OFFLINE_HINT}")
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        detail = _response_detail(exc.response)
        hint_fn = getattr(backend, "transport_hint", None)
        hint = hint_fn(code, detail) if callable(hint_fn) else None
        message = f"{service} at {base_url} returned HTTP {code}: {detail}"
        return BackendError(f"{message}. {hint}" if hint else message)
    return BackendError(f"{service} request to {base_url} failed: {type(exc).__name__}: {exc}")


def translate_transport_errors(method: _F) -> _F:
    """Decorates backend I/O methods so callers never see a bare ``httpx`` traceback.

    The original exception is chained as ``__cause__``. Retries or fallbacks *inside* the
    method (e.g. Ollama's ``think=False`` retry on HTTP 400) still see raw ``httpx`` errors,
    because translation happens only at the method boundary.
    """
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return await method(self, *args, **kwargs)
            except httpx.HTTPError as exc:
                raise describe_transport_error(self, exc) from exc

        return cast(_F, async_wrapper)

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except httpx.HTTPError as exc:
            raise describe_transport_error(self, exc) from exc

    return cast(_F, wrapper)


def strip_reasoning_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning traces from model responses."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


class LogitBackend(abc.ABC):
    """Abstract base class for extracting first-token log-probabilities."""

    @abc.abstractmethod
    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns top_k token logprobs: [{'token': str, 'logprob': float}, ...]"""

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Async variant of query_top_logprobs."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.query_top_logprobs, prompt, top_k, allowed_tokens
        )

    def close(self) -> None:
        """Closes any underlying network connections or resources."""

    async def aclose(self) -> None:
        """Asynchronously closes any underlying network connections or resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


class GenerativeBackend(abc.ABC):
    """Abstract base class for generative reasoning & diagnostics."""

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        """Returns (generated_text, tokens_generated)."""

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        """Async variant of generate."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, max_tokens=max_tokens, **kwargs))

    def close(self) -> None:
        """Closes any underlying network connections or resources."""

    async def aclose(self) -> None:
        """Asynchronously closes any underlying network connections or resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


class StubBackend(LogitBackend, GenerativeBackend):
    """
    Offline deterministic stub backend for sub-millisecond unit tests and CI.
    Records all prompted inputs in self.prompt_log.
    """

    def __init__(
        self,
        default_prob: float = 0.95,
        forward_top_logprobs: list[dict[str, Any]] | None = None,
        reverse_top_logprobs: list[dict[str, Any]] | None = None,
        logprob_responses: list[list[dict[str, Any]]] | None = None,
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
        allowed_tokens: list[str] | None = None,
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
        allowed_tokens: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.query_top_logprobs(prompt, top_k, allowed_tokens)

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        self.prompt_log.append(prompt)
        return self.diagnostic_text, self.tokens_generated

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return self.generate(prompt, max_tokens=max_tokens, **kwargs)


def infer_backend_spec(backend: Any) -> str:
    """Infers a canonical URI specifier string from a resolved backend instance."""
    if isinstance(backend, str):
        return backend
    if isinstance(backend, StubBackend):
        return f"stub://{backend.default_prob}" if backend.default_prob != 0.95 else "stub://default"
    cls_name = backend.__class__.__name__
    base_url = getattr(backend, "base_url", None) or ""
    model = getattr(backend, "model", None) or ""
    if cls_name == "OllamaBackend":
        host = base_url.replace("http://", "").replace("https://", "").rstrip("/") or "localhost:11434"
        return f"ollama://{host}/{model}" if model else f"ollama://{host}"
    if cls_name == "OpenAICompatBackend":
        host = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        scheme = "openai" if base_url.startswith("https://") else "vllm"
        return f"{scheme}://{host}"
    if cls_name == "LlamaCppBackend":
        if base_url:
            host = base_url.replace("http://", "").replace("https://", "").rstrip("/")
            return f"llamacpp://{host}/{model}" if model else f"llamacpp://{host}"
        return f"llamacpp:///{model}" if model else "llamacpp"
    return "stub://default"


def resolve_logit_backend(
    backend_spec: str | LogitBackend | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LogitBackend:
    if isinstance(backend_spec, LogitBackend):
        return backend_spec

    from logitgate.config import (
        LogitGateDefaults,
        get_default_model,
        warn_if_implicit_stub,
    )

    # Model precedence: explicit `model=` argument > model embedded in the URI > LOGITGATE_MODEL
    # > per-backend default. The environment is a fallback; it must never override something
    # the caller wrote down.
    env_model = get_default_model()
    if backend_spec is None or backend_spec == "default":
        default_spec = LogitGateDefaults.backend
        if (
            default_spec == "stub://default"
            and model is not None
            and model not in ("default", "stub")
            and not model.startswith("stub")
        ):
            # `LogitGate(rubric, model="qwen2.5:7b")` with nothing configured means "use my
            # local Ollama model", not "silently return P=0.95".
            backend_spec = f"ollama://localhost:11434/{model}"
        else:
            warn_if_implicit_stub(default_spec)
            backend_spec = default_spec

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

    def _uri_model(parsed_path: str) -> str | None:
        return parsed_path.lstrip("/") or None

    if spec.startswith("ollama://"):
        from logitgate.backends.ollama import OllamaBackend
        parsed = urlparse(spec)
        netloc = parsed.netloc or "localhost:11434"
        effective_model = model or _uri_model(parsed.path) or env_model or "qwen2.5-coder:14b"
        return OllamaBackend(base_url=f"http://{netloc}", model=effective_model, **kwargs)

    if spec.startswith(("openai://", "vllm://")):
        from logitgate.backends.openai_compat import OpenAICompatBackend
        parsed = urlparse(spec)
        scheme = "https" if spec.startswith("openai://") else "http"
        base_url = f"{scheme}://{parsed.netloc}{parsed.path}"
        return OpenAICompatBackend(base_url=base_url, model=model or env_model or "gpt-4o-mini", **kwargs)

    if spec.startswith("llamacpp://") or spec == "llamacpp":
        from logitgate.backends.llamacpp import LlamaCppBackend
        if spec == "llamacpp":
            return LlamaCppBackend(model=model or env_model, **kwargs)
        parsed = urlparse(spec)
        effective_model_opt = model or _uri_model(parsed.path) or env_model
        if parsed.netloc:
            return LlamaCppBackend(base_url=f"http://{parsed.netloc}", model=effective_model_opt, **kwargs)
        return LlamaCppBackend(model=effective_model_opt, **kwargs)

    if spec.startswith("recurrent://") or spec == "recurrent":
        from logitgate.backends.recurrent import RecurrentPyTorchBackend
        uri_model = None
        if spec.startswith("recurrent://"):
            parsed_rec = urlparse(spec)
            uri_model = f"{parsed_rec.netloc}{parsed_rec.path}" if parsed_rec.netloc else None
        target_model = model or uri_model or env_model or "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
        return RecurrentPyTorchBackend(model_name=target_model, **kwargs)

    # Fallback to Ollama or Stub
    if "stub" in spec:
        return StubBackend(**kwargs)

    # A bare spec such as "qwen2.5:7b" is treated as an Ollama model name.
    from logitgate.backends.ollama import OllamaBackend
    return OllamaBackend(model=model or spec, **kwargs)


def resolve_generative_backend(
    backend_spec: str | GenerativeBackend | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> GenerativeBackend:
    if isinstance(backend_spec, GenerativeBackend):
        return backend_spec

    backend = resolve_logit_backend(backend_spec, model=model, **kwargs)
    if isinstance(backend, GenerativeBackend):
        return backend
    raise TypeError(f"Backend {backend} does not implement GenerativeBackend")
