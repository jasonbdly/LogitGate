# Backends & Sidecars (⚡ Quick Notes)

Provider configuration matrix, HTTP retry tuning, and sidecar routing for models that do not expose token log-probabilities.

---

## 1. Provider Configuration Matrix

**Problem & deployment**: Decouples your evaluation rubrics from inference infrastructure so the exact same gate code runs against local laptops (`OllamaBackend`, `LlamaCppBackend`), GPU serving clusters (`OpenAICompatBackend` with vLLM/SGLang), cloud logprob APIs (`GeminiBackend`, OpenAI), and offline CI (`StubBackend`).

| Provider | URI Scheme | Class Name | Logprobs Supported | Best For |
| :--- | :--- | :--- | :---: | :--- |
| **Ollama** | `ollama://localhost:11434/model` | `OllamaBackend` | Yes (`top_logprobs`) | Local GGUF models on developer laptops & workstations |
| **vLLM / SGLang** | `vllm://localhost:8000/v1/model` | `OpenAICompatBackend` | Yes (`top_logprobs`) | High-throughput server clusters & prefix caching |
| **OpenAI API** | `openai://gpt-4o-mini` | `OpenAICompatBackend` | Yes (`logprobs=True`) | Hosted commercial API evaluation |
| **Google Gemini** | `gemini://gemini-2.5-flash` | `GeminiBackend` | Yes (`response_logprobs`) | Fast cloud evaluation with large context windows |
| **llama.cpp** | `llamacpp://localhost:8080` | `LlamaCppBackend` | Yes (`probs`) | C++ in-process binding or standalone HTTP daemon |
| **Offline Stub** | `stub://default` | `StubBackend` | Yes (Synthetic) | Unit tests, CI plumbing checks, zero GPU required |

### Copy-Paste Configuration Recipes

```python
import os
from logitgate import LogitGate
from logitgate.backends import (
    GeminiBackend,
    LlamaCppBackend,
    OllamaBackend,
    OpenAICompatBackend,
    StubBackend,
)

# 1. Ollama (local server)
b_ollama = OllamaBackend(model="qwen2.5:7b", base_url="http://localhost:11434")

# 2. vLLM / SGLang (OpenAI compatible endpoint)
b_vllm = OpenAICompatBackend(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
)

# 3. Google Gemini
b_gemini = GeminiBackend(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# 4. OpenAI
b_openai = OpenAICompatBackend(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)

# 5. Offline Stub for CI
b_stub = StubBackend(default_prob=0.95)

# Pass backend directly to gate or set via environment variable
gate = LogitGate("Does the code follow PEP 8?", backend=b_ollama)
```

<details>
<summary>🌱 <b>ELI5 / Why this works</b></summary>

LogitGate speaks the native logprob format of each engine (Ollama's top_logprobs array, OpenAI's content logprobs, and Gemini's top_candidates). You can swap between a local 3B model on your laptop and a cluster endpoint with one URI string.
</details>

[🔬 Systems Architecture & Client Pooling](../architecture.md)

---

## 2. HTTP Lifecycle, Transient Retries & Rate Limits

**Problem & deployment**: Keeps high-throughput batch evaluation and live guardrail services resilient against transient `429`/`503` rate limits and brief network blips without retrying permanent `4xx` configuration or auth errors.

All HTTP-backed inference engines share transient retry and client-pooling lifecycle mechanics (`_HTTPClientMixin`).

### Transient Retries & Exponential Backoff

HTTP backends retry only transient failures (`{408, 429, 502, 503, 504}` and `httpx.TransportError`). Permanent client errors (`4xx` such as 400 Bad Request, 401 Unauthorized, 404 Not Found) fail immediately without wasted retry cycles.

```python
from logitgate.backends import OllamaBackend

backend = OllamaBackend(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    timeout=45.0,
    max_retries=3,            # Up to 3 retry attempts on transient network errors
    retry_backoff_base=0.5,   # Exponential backoff base (seconds)
    retry_max_wait=10.0,      # Ceiling cap on individual backoff sleep
)
```

### Server Rate Limits (`Retry-After`)

When an endpoint returns HTTP `429 Too Many Requests` or `503 Service Unavailable` with a `Retry-After` header (in integer seconds or RFC 7231 HTTP-date format), LogitGate honors the server-specified sleep duration (clamped by `retry_max_wait`) before issuing the retry.

### Async Event Loop Reuse Guarantee

When sharing a backend across sequential `asyncio.run(...)` calls, internal async HTTP sessions cleanly detect when the prior event loop has closed, instantiating fresh event-loop-bound clients automatically to avoid `RuntimeError: Event loop is closed`.

---

## 3. Solving No-Logprob Models (`sidecar_backend`, `GeminiBackend` & Escalation)

**Problem & deployment**: Lets teams running frontier models without public logprobs (Anthropic Claude, OpenAI `o1`/`o3-mini`, Groq, Amazon Bedrock) keep sub-100ms Layer 1 gating in CI suites and live apps by delegating first-token scoring to a fast logprob-capable sidecar (`OllamaBackend`, vLLM, or `GeminiBackend`).

Several popular commercial APIs do not expose token log-probabilities:
- **Anthropic Claude** (Claude 3.5 Sonnet, Claude 3.7 Sonnet)
- **OpenAI Reasoning Models** (`o1`, `o3-mini`)
- **Groq Hosted Endpoints**
- **Amazon Bedrock Converse API**

LogitGate resolves this with two explicit routing strategies:

### Strategy A: Sidecar Evaluation (`on_unsupported_logprobs="sidecar"`)

Keep the primary frontier model for candidate generation or Layer 2 escalation, and delegate Layer 1 first-token logit evaluation via `sidecar_backend` to either a local GPU model (e.g., `ollama://localhost:11434/qwen2.5:3b`) or a cloud logprob backend such as `GeminiBackend` (`gemini://gemini-2.5-flash`).

```python
from logitgate import LogitGate
from logitgate.backends import GeminiBackend

# Option 1: Local GPU sidecar (zero cloud logprob cost, ~20ms per check)
gate_local_sidecar = LogitGate(
    rubric="Is the answer concise and strictly factual?",
    backend="openai://o3-mini",  # Primary model lacks logprobs
    on_unsupported_logprobs="sidecar",
    sidecar_backend="ollama://localhost:11434/qwen2.5:3b",
)

# Option 2: Cloud GeminiBackend sidecar when no local GPU is available
gate_cloud_sidecar = LogitGate(
    rubric="Is the answer concise and strictly factual?",
    backend="openai://o3-mini",
    on_unsupported_logprobs="sidecar",
    sidecar_backend=GeminiBackend(model="gemini-2.5-flash"),
)

candidate = "The transaction completed on Tuesday."
verdict = gate_local_sidecar.evaluate(candidate)
print(f"Passed: {verdict.passed} via sidecar: {verdict.triggered_layer}")
```

### Strategy B: Escalation Routing (`on_unsupported_logprobs="escalate"`)

When no logprob sidecar is available, LogitGate skips the Layer 1 logit probe and escalates directly to Layer 2 (`JudgeGate`), prompting the primary model to generate a structured text critique and verdict.

```python
from logitgate import LogitGate

gate = LogitGate(
    rubric="Does the output strictly adhere to the requested schema?",
    backend="openai://o3-mini",
    on_unsupported_logprobs="escalate",
)

candidate = "{'data': [1, 2, 3]}"
verdict = gate.evaluate(candidate)
print(f"Passed: {verdict.passed} via escalated critique")
```

<details>
<summary>🌱 <b>ELI5 / Why this works</b></summary>

Frontier models like o3 or Claude excel at complex generation, but their APIs do not return internal token probabilities. A sidecar lets a compact 3B local model or `GeminiBackend` score the candidate's first-token logits quickly, keeping your test suite fast without needing logprobs from the closed provider.
</details>

[🔬 Production Use Cases & Sidecar Deployments](../use_cases.md)
