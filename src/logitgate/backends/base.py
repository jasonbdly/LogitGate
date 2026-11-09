from __future__ import annotations

import abc
import asyncio
import functools
import inspect
import json
import math
import re
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast
from urllib.parse import parse_qs, urlparse

import httpx
from typing_extensions import Self

from logitgate.exceptions import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
)

_F = TypeVar("_F", bound=Callable[..., Any])
_OFFLINE_HINT = "Or set backend='stub://' for offline testing."
EXCLUDED_INTERNAL_KWARGS: frozenset[str] = frozenset(
    {"save_on_fail", "snapshot_dir", "test_name", "rubric_override", "_normalized_out"}
)


def _response_detail(response: httpx.Response) -> str:
    """Extracts the server's own error message (Ollama: {"error": str}, OpenAI: {"error": {...}})."""
    try:
        payload = response.json()
    except (ValueError, httpx.ResponseNotRead):
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
            except httpx.InvalidURL as exc:
                service = getattr(self, "service_name", type(self).__name__)
                base_url = getattr(self, "base_url", None) or "<unknown url>"
                raise BackendError(
                    f"Invalid URL '{base_url}' for {service}: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                service = getattr(self, "service_name", type(self).__name__)
                base_url = getattr(self, "base_url", None) or "<unknown url>"
                raise BackendError(
                    f"{service} at {base_url} returned malformed JSON: {exc}"
                ) from exc

        return cast(_F, async_wrapper)

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except httpx.HTTPError as exc:
            raise describe_transport_error(self, exc) from exc
        except httpx.InvalidURL as exc:
            service = getattr(self, "service_name", type(self).__name__)
            base_url = getattr(self, "base_url", None) or "<unknown url>"
            raise BackendError(
                f"Invalid URL '{base_url}' for {service}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            service = getattr(self, "service_name", type(self).__name__)
            base_url = getattr(self, "base_url", None) or "<unknown url>"
            raise BackendError(
                f"{service} at {base_url} returned malformed JSON: {exc}"
            ) from exc

    return cast(_F, wrapper)


def strip_reasoning_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning traces from model responses in O(N) linear time."""
    if not text:
        return ""
    if "<think>" not in text:
        return text.strip()
    parts: list[str] = []
    pos = 0
    n = len(text)
    comment_prefixes = ("#", "//", "--", "/*", "*", '"', "'")
    while pos < n:
        start = text.find("<think>", pos)
        if start == -1:
            parts.append(text[pos:])
            break
        line_start = text.rfind("\n", 0, start) + 1
        line_prefix = text[line_start:start].strip()
        if line_prefix.startswith(comment_prefixes):
            # Inside a source-code comment or quoted literal on the same line: do not strip
            parts.append(text[pos : start + 7])
            pos = start + 7
            continue
        parts.append(text[pos:start])
        search_from = start + 7
        closed = False
        while True:
            end = text.find("</think>", search_from)
            if end == -1:
                break
            end_line_start = text.rfind("\n", 0, end) + 1
            end_line_prefix = text[end_line_start:end].strip()
            if end_line_prefix.startswith(comment_prefixes):
                search_from = end + 8
                continue
            pos = end + 8
            closed = True
            break
        if not closed:
            # Unclosed top-level <think> tag: truncate mid-thought reasoning tail
            break
    return "".join(parts).strip()


class LogitBackend(abc.ABC):
    """Abstract base class for extracting first-token log-probabilities."""

    @abc.abstractmethod
    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Returns top_k token logprobs: [{'token': str, 'logprob': float}, ...]"""

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Async variant of query_top_logprobs."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.query_top_logprobs(prompt, top_k, allowed_tokens, **kwargs)
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


class _HTTPClientMixin:
    """Shared lazy httpx.Client / httpx.AsyncClient lifecycle with event-loop affinity tracking."""

    timeout: float
    _client: httpx.Client | None
    _async_client: httpx.AsyncClient | None
    _async_client_loop: asyncio.AbstractEventLoop | None
    _owns_async_client: bool
    _http_lock: threading.RLock

    def _init_http_clients(
        self,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client
        self._async_client = async_client
        self._async_client_loop = None
        self._owns_async_client = async_client is None
        self._http_lock = threading.RLock()

    @property
    def client(self) -> httpx.Client:
        lock = getattr(self, "_http_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._http_lock = lock
        with lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.Client(timeout=self.timeout)
            return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        lock = getattr(self, "_http_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._http_lock = lock
        with lock:
            owns_client = getattr(self, "_owns_async_client", True)
            bound_loop = getattr(self, "_async_client_loop", None)
            loop_changed = (
                owns_client
                and current_loop is not None
                and bound_loop is not None
                and (bound_loop is not current_loop or bound_loop.is_closed())
            )
            if self._async_client is None or self._async_client.is_closed or loop_changed:
                self._async_client = httpx.AsyncClient(timeout=self.timeout)
                self._async_client_loop = current_loop
                self._owns_async_client = True
            elif bound_loop is None and current_loop is not None:
                self._async_client_loop = current_loop
            return self._async_client

    def close(self) -> None:
        """Closes the synchronous HTTP client session if initialized."""
        lock = getattr(self, "_http_lock", None) or threading.RLock()
        with lock:
            if self._client is not None and not self._client.is_closed:
                self._client.close()

    async def aclose(self) -> None:
        """Closes the asynchronous HTTP client session if initialized."""
        lock = getattr(self, "_http_lock", None) or threading.RLock()
        with lock:
            client = self._async_client
            owns = getattr(self, "_owns_async_client", True)
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except RuntimeError:
                pass
            finally:
                with lock:
                    self._async_client_loop = None
                    if owns:
                        self._async_client = None


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
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"StubBackend(default_prob={self.default_prob})"

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self.prompt_log.append(prompt)
            resp_idx = self._response_idx
            if self.logprob_responses:
                self._response_idx += 1

        decision_tail_idx = max(prompt.rfind("[DECISION MAPPING]"), prompt.rfind("[SCORING SCALE]"))
        decision_section = prompt[decision_tail_idx:] if decision_tail_idx != -1 else prompt[-300:]
        is_inverted_scale = "on an inverted scale where" in decision_section

        # 1. Explicit queue response
        if self.logprob_responses:
            resp = self.logprob_responses[resp_idx % len(self.logprob_responses)]
            if (
                len(self.logprob_responses) == 1
                and is_inverted_scale
                and allowed_tokens
            ):
                int_toks = [int(t) for t in allowed_tokens if str(t).isdigit()]
                if int_toks:
                    mirror = min(int_toks) + max(int_toks)
                    return [
                        {
                            **item,
                            "token": str(mirror - int(item["token"]))
                            if str(item.get("token", "")).strip().isdigit()
                            else item["token"],
                        }
                        for item in resp
                    ]
            return resp

        if is_inverted_scale:
            if self.reverse_top_logprobs is not None:
                return self.reverse_top_logprobs
            if self.forward_top_logprobs is not None and allowed_tokens:
                int_toks = [int(t) for t in allowed_tokens if str(t).isdigit()]
                if int_toks:
                    mirror = min(int_toks) + max(int_toks)
                    return [
                        {
                            **item,
                            "token": str(mirror - int(item["token"]))
                            if str(item.get("token", "")).strip().isdigit()
                            else item["token"],
                        }
                        for item in self.forward_top_logprobs
                    ]

        # Numeric score scale (e.g. ScoreGate 1..5) when no custom logprobs were supplied
        if (
            self.forward_top_logprobs is None
            and self.reverse_top_logprobs is None
            and allowed_tokens
            and all(str(t).strip().isdigit() for t in allowed_tokens)
        ):
            int_toks = sorted(int(str(t).strip()) for t in allowed_tokens)
            low_tok, high_tok = str(int_toks[0]), str(int_toks[-1])
            p = max(min(self.default_prob, 0.9999), 0.0001)
            if is_inverted_scale:
                return [
                    {"token": low_tok, "logprob": math.log(p)},
                    {"token": high_tok, "logprob": math.log(1.0 - p)},
                ]
            return [
                {"token": high_tok, "logprob": math.log(p)},
                {"token": low_tok, "logprob": math.log(1.0 - p)},
            ]

        # 2. Check if this is Pass 2 (reversed mapping where A is Fail / violates)
        is_reverse_pass = (
            "A: The candidate artifact violates" in decision_section
            or "A: The content violates" in decision_section
            or "A: The candidate does not" in decision_section
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
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return self.query_top_logprobs(prompt, top_k, allowed_tokens, **kwargs)

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        with self._lock:
            self.prompt_log.append(prompt)
        return self.diagnostic_text, self.tokens_generated

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        return self.generate(prompt, max_tokens=max_tokens, **kwargs)


def _is_openai_host(netloc: str, path: str) -> bool:
    """Distinguishes a host (`api.openai.com`, `localhost:8000`, `10.0.0.1`) from a model ID (`gpt-4.1-mini`)."""
    if not netloc:
        return False
    if netloc == "localhost" or netloc.startswith("localhost:"):
        return True
    if path == "/v1" or path.startswith("/v1/"):
        return True
    if re.match(r"^[^:]+:\d+$", netloc):
        return True
    return bool(
        ":" not in netloc
        and (
            re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", netloc)
            or re.search(r"\.[a-zA-Z]{2,}$", netloc)
        )
    )


def infer_backend_spec(backend: Any) -> str:
    """Infers a canonical URI specifier string from a resolved backend instance."""
    if isinstance(backend, str):
        return backend
    if hasattr(backend, "inner") and isinstance(backend.inner, LogitBackend):
        backend = backend.inner
    if isinstance(backend, StubBackend):
        return f"stub://{backend.default_prob}" if backend.default_prob != 0.95 else "stub://default"
    cls_name = backend.__class__.__name__
    base_url = getattr(backend, "base_url", None) or ""
    model = getattr(backend, "model", None) or getattr(backend, "model_name", None) or ""
    if cls_name == "OllamaBackend":
        host = base_url.replace("http://", "").replace("https://", "").rstrip("/") or "localhost:11434"
        return f"ollama://{host}/{model}" if model else f"ollama://{host}"
    if cls_name == "OpenAICompatBackend":
        host = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        scheme = "openai" if base_url.startswith("https://") else "vllm"
        if scheme == "openai" and host == "api.openai.com/v1":
            return f"openai://{model}" if model else "openai://api.openai.com/v1"
        return f"{scheme}://{host}/{model}" if model else f"{scheme}://{host}"
    if cls_name == "LlamaCppBackend":
        if getattr(backend, "llm", None) is None and base_url:
            host = base_url.replace("http://", "").replace("https://", "").rstrip("/")
            return f"llamacpp://{host}/{model}" if model else f"llamacpp://{host}"
        return f"llamacpp:///{model}" if model else "llamacpp"
    if cls_name == "RecurrentPyTorchBackend":
        base_spec = f"recurrent://{model}" if model else "recurrent"
        q_parts: list[str] = []
        num_loops = getattr(backend, "num_loops", 2)
        min_loops = getattr(backend, "min_loops", None)
        max_loops = getattr(backend, "max_loops", None)
        kl_threshold = getattr(backend, "kl_threshold", None)
        device = getattr(backend, "device", "cuda")
        if num_loops != 2:
            q_parts.append(f"num_loops={num_loops}")
        if min_loops is not None:
            q_parts.append(f"min_loops={min_loops}")
        if max_loops is not None:
            q_parts.append(f"max_loops={max_loops}")
        if kl_threshold is not None:
            q_parts.append(f"kl_threshold={kl_threshold}")
        if device != "cuda":
            q_parts.append(f"device={device}")
        if q_parts and base_spec != "recurrent":
            return f"{base_spec}?{'&'.join(q_parts)}"
        return base_spec
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
        is_backend_implicit,
        warn_if_implicit_stub,
    )

    # Model precedence: explicit `model=` argument > model embedded in the URI > LOGITGATE_MODEL
    # > per-backend default. The environment is a fallback; it must never override something
    # the caller wrote down.
    env_model = get_default_model()
    if backend_spec is None or backend_spec == "default":
        default_spec = LogitGateDefaults.backend
        if (
            is_backend_implicit()
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
        if not parsed_path:
            return None
        stripped = parsed_path.removeprefix("/")
        return stripped or None

    if spec.startswith("ollama://"):
        from logitgate.backends.ollama import OllamaBackend
        parsed = urlparse(spec)
        netloc = parsed.netloc
        path = parsed.path or ""
        uri_ollama_model: str | None = None
        is_ollama_model_shorthand = bool(
            netloc
            and not path
            and not _is_openai_host(netloc, path)
        )
        if is_ollama_model_shorthand:
            # Model-only shorthand `ollama://gemma4:e4b`, `ollama://qwen2.5:7b`, `ollama://llama3.2`, or `ollama://llama3`
            uri_ollama_model = netloc
            netloc = "localhost:11434"
        else:
            netloc = netloc or "localhost:11434"
            uri_ollama_model = _uri_model(path)
        effective_model = model or uri_ollama_model or env_model or "qwen2.5:7b"
        return OllamaBackend(base_url=f"http://{netloc}", model=effective_model, **kwargs)

    if spec.startswith(("openai://", "vllm://")):
        from logitgate.backends.openai_compat import OpenAICompatBackend
        parsed = urlparse(spec)
        is_openai = spec.startswith("openai://")
        scheme = "https" if is_openai else "http"
        netloc = parsed.netloc
        path = parsed.path or ""
        uri_model_opt: str | None = None

        # Handle shorthand `openai://gpt-4o-mini` or `vllm://Qwen/Qwen2.5-7B-Instruct`
        is_host = _is_openai_host(netloc, path)
        if not is_host:
            uri_model_opt = f"{netloc}{path}".strip("/") or None
            base_url = "https://api.openai.com/v1" if is_openai else "http://localhost:8000/v1"
        else:
            if not netloc:
                netloc = "api.openai.com" if is_openai else "localhost:8000"
            if "/v1" in path:
                prefix, _, after = path.partition("/v1")
                api_path = f"{prefix}/v1"
                uri_model_opt = after.strip("/") or None
            else:
                api_path = "/v1"
                uri_model_opt = path.strip("/") or None
            base_url = f"{scheme}://{netloc}{api_path}"

        effective_model = model or uri_model_opt or env_model or "gpt-4o-mini"
        return OpenAICompatBackend(base_url=base_url, model=effective_model, **kwargs)

    if spec.startswith("llamacpp://") or spec == "llamacpp":
        from logitgate.backends.llamacpp import LlamaCppBackend
        if spec == "llamacpp":
            return LlamaCppBackend(model=model or env_model, **kwargs)
        parsed = urlparse(spec)
        if parsed.netloc and not (parsed.netloc.endswith(".gguf") and ":" not in parsed.netloc):
            effective_model_opt = model or _uri_model(parsed.path) or env_model
            return LlamaCppBackend(base_url=f"http://{parsed.netloc}", model=effective_model_opt, **kwargs)
        raw_local = f"{parsed.netloc}{parsed.path}" if parsed.netloc else (parsed.path.removeprefix("/"))
        effective_model_opt = model or (raw_local or None) or env_model
        return LlamaCppBackend(base_url="", model=effective_model_opt, **kwargs)

    if spec.startswith("recurrent://") or spec == "recurrent":
        from logitgate.backends.recurrent import RecurrentPyTorchBackend
        uri_model = None
        rec_kwargs = dict(kwargs)
        if spec.startswith("recurrent://"):
            parsed_rec = urlparse(spec)
            if parsed_rec.netloc:
                uri_model = f"{parsed_rec.netloc}{parsed_rec.path}"
            elif parsed_rec.path:
                uri_model = parsed_rec.path
            if parsed_rec.query:
                qparams = parse_qs(parsed_rec.query)
                for int_key in ("num_loops", "min_loops", "max_loops"):
                    if int_key in qparams and int_key not in rec_kwargs:
                        rec_kwargs[int_key] = int(qparams[int_key][-1])
                if "kl_threshold" in qparams and "kl_threshold" not in rec_kwargs:
                    rec_kwargs["kl_threshold"] = float(qparams["kl_threshold"][-1])
                if "device" in qparams and "device" not in rec_kwargs:
                    rec_kwargs["device"] = qparams["device"][-1]
        target_model = model or uri_model or env_model or "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
        return RecurrentPyTorchBackend(model_name=target_model, **rec_kwargs)

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
    if not isinstance(backend_spec, LogitBackend):
        backend.close()
    raise TypeError(f"Backend {backend} does not implement GenerativeBackend")
