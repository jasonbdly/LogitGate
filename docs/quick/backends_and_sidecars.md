# Backends & Sidecars (⚡ Quick Notes)

Complete provider configuration matrix, HTTP retry tuning, and sidecar routing for models that do not return token log-probabilities.

---

## 1. Provider Configuration Matrix

LogitGate supports local, self-hosted, and cloud inference endpoints via unified URI strings or direct backend class instantiation.

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

All HTTP-backed inference engines inherit robust lifecycle and transient retry mechanics (`_HTTPClientMixin`).

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

## 3. Solving No-Logprob Models (Sidecar & Escalation Routing)

Several popular commercial models do not expose token log-probabilities via their public APIs:
- **Anthropic Claude** (Claude 3.5 Sonnet, Claude 3.7 Sonnet)
- **OpenAI Reasoning Models** (`o1`, `o3-mini`)
- **Groq Hosted Endpoints**
- **Amazon Bedrock Converse API**

LogitGate resolves this with two zero-compromise architectural strategies:

### Strategy A: Sidecar Evaluation (`on_unsupported_logprobs="sidecar"`)

Use the primary frontier model to generate the candidate response, but delegate the Layer 1 first-token logit evaluation to a fast, local sidecar model (e.g. `qwen2.5:3b` or `gemma4:e4b` on GPU).

```python
from logitgate import LogitGate

# Primary model generates candidate; sidecar evaluates first-token logits in ~20ms
gate = LogitGate(
    rubric="Is the answer concise and strictly factual?",
    backend="openai://o1-preview",  # Primary model lacks logprobs
    on_unsupported_logprobs="sidecar",
    sidecar_backend="ollama://localhost:11434/qwen2.5:3b",  # Local fast evaluator
)

candidate = "The transaction completed on Tuesday."
verdict = gate.evaluate(candidate)
print(f"Passed: {verdict.passed} via sidecar: {verdict.triggered_layer}")
```

### Strategy B: Escalation Routing (`on_unsupported_logprobs="escalate"`)

When no local sidecar model is available, LogitGate skips the Layer 1 logit probe and escalates directly to Layer 2 (`JudgeGate`), prompting the primary model to generate a structured text critique and verdict.

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

Frontier models like o1 or Claude are great at generating code, but their cloud APIs do not tell us their internal token probabilities. A sidecar pattern lets an ultra-fast 3B local model check the output in 20 ms, keeping your test suite fast without needing logprobs from the closed provider.
</details>

[🔬 Production Use Cases & Sidecar Deployments](../use_cases.md)
