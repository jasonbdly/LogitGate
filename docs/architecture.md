[🏠 README](../README.md) · [📚 Docs Hub](./README.md) · [⚡ Quick Notes](./quick/quickstart_recipes.md) · [🌱 Visual ELI5](./eli5/how_logitgate_works.md) · [🚀 Power-User](./advanced/distillation_and_lora.md) · [📊 Benchmarks](./benchmarks.md)

# Deep Notes: Systems Architecture & Execution Mechanics

This document provides an exhaustive systems-level breakdown of the **LogitGate** evaluation engine: its trait-based gate hierarchy, prefix-preserving attention mechanics, concurrency and lifecycle invariants, capability-aware backend routing, and production deployment topologies.

---

## 1. 3-Layer Cascaded Router Architecture

LogitGate organizes evaluation into three distinct physical layers to achieve maximal verification throughput at minimum cost and latency.

![LogitGate 3-Layer Cascade Flow](./assets/cascade_flow.svg)

```mermaid
flowchart TD
    artifact["Candidate Artifact (Code / JSON / Text)"] --> L0["Layer 0: ASTGate / ToolCallGate (~0.04ms CPU)"]
    
    L0 -->|"Syntax / Schema / SSRF / Path Violation"| F0["FAIL: 0 LLM calls (sub-0.1ms)"]
    L0 -->|"Passed Structural Invariants"| L1["Layer 1: LogitGate / MultiRubricGate (~40–110ms GPU)"]
    
    L1 -->|"P >= pass_threshold (>= 0.85)"| P1["PASS: 0 CoT tokens (Fast-Path)"]
    L1 -->|"P <= fail_threshold (<= 0.20)"| F1["FAIL: 0 CoT tokens (Decisive)"]
    L1 -->|"0.20 < P < 0.85 (Ambiguous / Drift)"| L2["Layer 2: JudgeGate (1,500–4,000ms GPU)"]
    
    L2 -->|"Generative Adjudication"| R2["Actionable Diagnostic Report / Reversal"]
```

### Layer Execution Breakdown

1. **Layer 0 (`ASTGate` / `ContractGate` / `ToolCallGate` Stage-0)**:
   - **Latency**: Sub-millisecond (~0.04 ms on CPU).
   - **Cost**: 0 LLM calls, 0 tokens generated, 0 GPU memory allocated.
   - **Mechanics**: Enforces deterministic, non-probabilistic invariants before invoking neural weights:
     - Python AST syntax trees (`syntax=True`) and sequence validation.
     - Forbidden imports (`forbidden_imports=["os", "sys", "subprocess"]`), inspecting both `Import` and `ImportFrom` nodes as well as dynamic `__import__` / `importlib` calls.
     - Forbidden regex patterns (`forbidden_patterns`), scanned across both raw and normalized inputs.
     - JSON Schema and Pydantic model validation (`json_schema`), strictly disallowing boolean coercions for numeric types (`not isinstance(val, bool)`).
     - Deterministic security barriers via `ToolCallGate`: path traversal (`..`), SSRF IP literal blocking (RFC-1918, link-local `169.254.169.254`, loopback), and destructive shell/SQL patterns.
   - **Short-Circuiting**: Any failure at Layer 0 halts the pipeline immediately with `passed=False` and `status=DecisionStatus.FAILED`. Layer 0 failures are final and cannot be overturned by downstream generative models.

2. **Layer 1 (`LogitGate` / `ChoiceGate` / `ScoreGate` / `MultiRubricGate` / `RankGate`)**:
   - **Latency**: ~36–45 ms (single-pass GPU), ~85–115 ms (2-pass debiased GPU), ~160–380 ms (CPU).
   - **Output Tokens**: **0 autoregressive Chain-of-Thought (CoT) tokens**. The backend requests exactly `max_tokens=1`.
   - **Mechanics**: Reads the model's raw first-token log-probabilities under greedy decoding:
     - **Binary Pass/Fail (`LogitGate`)**: Executes forward ($A=\text{Pass}, B=\text{Fail}$) and reverse ($B=\text{Pass}, A=\text{Fail}$) passes in log-odds space to cancel pretraining token priors and positional bias.
     - **Atomic Multi-Criteria (`MultiRubricGate`)**: Evaluates multiple orthogonal rubrics against a shared candidate context using weakest-link aggregation, reusing prompt KV-cache across all rubrics.
     - **K-ary Categorical Routing (`ChoiceGate`)**: Normalizes probabilities across $K$ options using fixed-point-free derangement permutations, supporting margin-based escalation.
     - **Continuous Scoring (`ScoreGate`)**: Computes probability-weighted expected scores ($\mathbb{E}[S] = \sum s \cdot P(s)$) over discrete anchors (1 to 5) with scale reversal debiasing.
     - **Zero-Token Ranking (`RankGate`)**: Ranks candidate pools using pairwise symmetric swap cancellation, listwise slot derangement, or round-robin Hodge combinatorial projections.
   - **Short-Circuiting Rules**:
     - **Fast-Path PASS**: Calibrated confidence $P \ge \text{pass\_threshold}$ (default: $0.85$) issues an immediate PASS, bypassing Layer 2.
     - **Decisive FAIL**: Calibrated confidence $P \le \text{fail\_threshold}$ (default: $0.20$) issues an immediate FAIL, short-circuiting Layer 2 when configured with `trigger="on_ambiguous_only"`.
     - **Ambiguous Escalation**: When $0.20 < P < 0.85$, or target tokens are unobserved, the gate marks the verdict `DecisionStatus.AMBIGUOUS` and escalates to Layer 2.

3. **Layer 2 (`JudgeGate`)**:
   - **Latency**: 1,500–4,500 ms (generative autoregressive decoding).
   - **Trigger Policies**:
     - `on_ambiguous_only` (default): Executes ONLY when Layer 1 confidence is genuinely ambiguous ($0.20 < P < 0.85$). Clean passes and decisive failures bypass Layer 2 completely.
     - `on_ambiguous_or_fail`: Executes if Layer 1 was ambiguous OR failed, providing rich diagnostic critiques for defective artifacts.
     - `adjudicate`: Escalates ambiguous items AND semantic Layer-1 failures, permitting Layer 2 to overturn Layer-1 false negatives while keeping Layer-0 structural failures strictly final.
     - `always`: Forces full generative critique on every artifact regardless of Layer-1 confidence.
   - **Post-Critique Logit Adjudication**: When backed by a `LogitBackend`, `JudgeGate` performs symmetric 2-pass post-critique log-odds adjudication over its generated critique notes. If tags are omitted, an internal 1-token `LogitGate` probe classifies the critique text before heuristic fallback.

---

## 2. Trait-Based Routing & Modular Gate Hierarchy (`BaseGate`)

Routing decisions in `Cascade` are strictly **trait-based**. The pipeline never checks concrete subclasses (`isinstance(gate, LogitGate)`). Instead, all gates inherit from `BaseGate` and declare routing traits that govern pipeline flow.

```mermaid
classDiagram
    class BaseGate {
        +str name
        +int layer_tier
        +bool can_fast_path
        +bool can_override_fail
        +EscalationPolicy escalation_policy
        +evaluate(candidate, **kwargs) Verdict
        +aevaluate(candidate, **kwargs) Verdict
        +evaluate_batch(candidates, **kwargs) list[Verdict]
        +close() void
        +aclose() void
    }
    class ASTGate {
        +int layer_tier = 0
        +bool can_fast_path = False
    }
    class ToolCallGate {
        +int layer_tier = 0 or 1
        +bool can_fast_path = False or True
    }
    class LogitGate {
        +int layer_tier = 1
        +bool can_fast_path = True
    }
    class JudgeGate {
        +int layer_tier = 2
        +bool can_fast_path = False
        +bool can_override_fail = True
        +EscalationPolicy escalation_policy
    }
    BaseGate <|-- ASTGate
    BaseGate <|-- ToolCallGate
    BaseGate <|-- LogitGate
    BaseGate <|-- JudgeGate
```

### Routing Traits Specification

| Routing Trait | Values | Meaning |
| :--- | :--- | :--- |
| `layer_tier` | `0` (Deterministic), `1` (Logit), `2` (Generative) | Declares the gate execution tier. Tier-0 failures are irrevocable and halt immediately. Tier-1 failures defer to an immediately following Tier-2 gate. |
| `can_fast_path` | `True`, `False` | When `True`, a decisive PASS ($P \ge \text{pass\_threshold}$) immediately bypasses subsequent gates (unless the downstream gate specifies `escalation_policy="always"`). |
| `can_override_fail` | `True`, `False` | When `True` (e.g. `JudgeGate(trigger="adjudicate")`), the gate has authority to overturn an upstream Tier-1 semantic failure if its detailed critique determines the candidate is actually correct. |
| `escalation_policy` | `"on_ambiguous_only"`, `"on_ambiguous_or_fail"`, `"adjudicate"`, `"always"` | Defines the criteria under which this gate executes when preceded by an upstream gate in a `Cascade`. |

### Hybrid Stage-0 / Stage-1 Routing in `ToolCallGate`

`ToolCallGate` exhibits dynamic trait adaptation:
- When verifying structural contracts (allowed tool names, parameter schemas, SSRF barriers, path traversal), it operates as **Tier 0** (`layer_tier = 0`, `can_fast_path = False`). Any violation fails immediately in <0.1 ms without LLM invocation.
- When `check_intent=True` or `intent_rubric` is supplied, it transitions to **Tier 1** (`layer_tier = 1`, `can_fast_path = True`). It invokes an internal `LogitGate` to assess semantic intent alignment, permitting decisive intent passes to short-circuit downstream `JudgeGate` stages.

### Custom Gate Implementation Recipe

Any developer can author custom routing gates by subclassing `BaseGate` and declaring these traits:

```python
from logitgate.gates.base import BaseGate, TIER_DETERMINISTIC
from logitgate.models import DecisionStatus, LayerTrace, Verdict

class CharacterBudgetGate(BaseGate):
    """Tier-0 gate that enforces a strict maximum character budget."""
    name = "CharacterBudgetGate"
    layer_tier = TIER_DETERMINISTIC
    can_fast_path = False

    def __init__(self, max_characters: int = 1000) -> None:
        self.max_characters = max_characters

    def evaluate(self, candidate: str, **kwargs) -> Verdict:
        length = len(candidate)
        passed = length <= self.max_characters
        status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
        reason = "Within budget" if passed else f"Exceeded {self.max_characters} chars ({length})"
        
        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=0.02,
            failure_code="OVER_BUDGET" if not passed else None,
        )
        return Verdict(
            status=status,
            passed=passed,
            triggered_layer=self.name,
            total_latency_ms=0.02,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )
```

The synchronous and asynchronous pipeline paths share one generator-based engine (`Cascade._pipeline`), ensuring identical routing behavior between `evaluate` and `aevaluate` by construction.

---

## 3. KV-Cache Prefix Engineering & Memory Layout

In causal autoregressive transformers, tokens can only attend to previous positions. When evaluating an artifact across multiple passes or rubrics, naive prompt templates cause **KV-Cache Thrashing** by altering early tokens, forcing the inference engine to recompute the entire prompt representation.

LogitGate enforces a strict **Prefix-Preserving Memory Layout** designed to keep 95%–99% of prompt tokens cached in GPU attention memory.

![Prefix-Preserving KV-Cache Layout](./assets/rag_prefix_cache.svg)

### Full Prompt Structure

```
[EVALUATION RUBRIC]
{rubric}

[REFERENCE CONTEXT]
{context}

[USER QUERY]
{query}

[CANDIDATE ARTIFACT]
<candidate_{nonce}>
{candidate}
</candidate_{nonce}>

[DECISION MAPPING]
A: The candidate complies fully with the rubric.
B: The candidate violates the rubric or has defects.
Respond with a single character (A or B).
Verdict:
```

### Prefix Sharing Across Debiasing Passes

- **Pass 1 (Forward)**: Maps $A = \text{Pass}, B = \text{Fail}$.
- **Pass 2 (Reverse)**: Maps $B = \text{Pass}, A = \text{Fail}$.
- **Memory Invariant**: Every single token from `[EVALUATION RUBRIC]` down through the end of the `</candidate_{nonce}>` block is **100% byte-identical** between Pass 1 and Pass 2.
- **Performance Impact**: Pass 2 hits the inference engine's prompt cache (Ollama, vLLM, llama.cpp) for the entire prefix. Only the short `[DECISION MAPPING]` suffix (~30 tokens) requires forward attention computation. This drops Pass-2 latency by **60%–75%**, turning a 2-pass debiased evaluation into an operation barely slower than a single un-debiased pass.

### Candidate-Prefix Isolated Batching (`MultiRubricGate` & `evaluate_batch`)

When evaluating $K$ atomic rubrics against one candidate document:
- The candidate artifact (often 1,000–16,000+ tokens) is anchored at Token 0 as the shared root prefix.
- Each rubric is appended in complete isolation at the suffix.
- Because attention cannot look forward, rubric $i$ cannot contaminate or distract attention when evaluating rubric $j$.
- Evaluating 20 rubrics reuses the cached candidate embedding 20 times, reducing per-rubric latency from ~90 ms down to ~20–35 ms on modern GPUs.

<details>
<summary>🌱 <b>ELI5: Why does prefix layout matter for GPU memory?</b></summary>
<blockquote>
Think of an LLM reading a prompt like a student studying a 50-page textbook chapter before answering a quiz.

If you ask the student: "Does this chapter mention apples? Answer A for yes, B for no", they read all 50 pages and answer "A".

Now, if you want to double-check their answer by swapping the letters: "Does this chapter mention apples? Answer B for yes, A for no", you don't want the student to re-read the 50 pages from scratch! You want them to keep their memory of the chapter fresh in their head and just apply the new rule to the very last sentence.

By putting the big document first and the A/B question at the very end, the GPU remembers the whole document in its high-speed memory (the KV-cache). The second question takes only a split second because the GPU only processes the final sentence!
</blockquote>
</details>

---

## 4. Concurrency, Lifecycles & Resilience Invariants

LogitGate is engineered for mission-critical production pipelines. It adheres to strict systems concurrency, deterministic cleanup, and failure-isolation standards.

```mermaid
sequenceDiagram
    autonumber
    participant App as Application / Caller
    participant Cas as Cascade / Gate
    participant Pool as Async Semaphore Pool
    participant Http as _HTTPClientMixin (httpx)
    participant Model as Inference Endpoint (Ollama / vLLM / Gemini)

    App->>Cas: evaluate_batch(candidates, on_error="abstain")
    Cas->>Http: Warm item 0 (cold prefix prefill)
    Http->>Model: POST /v1/chat/completions (item 0)
    Model-->>Http: 200 OK + logprobs
    Cas->>Pool: Fan-out items 1..N-1 bounded by Semaphore
    par Concurrent In-Flight Requests
        Pool->>Http: POST /v1/chat/completions (item 1)
        Pool->>Http: POST /v1/chat/completions (item 2)
    end
    Http-->>Model: Request fails (429 Rate Limit)
    Http->>Http: Read Retry-After, exponential backoff + jitter
    Http->>Model: Retry POST
    Model-->>Http: 200 OK + logprobs
    Cas->>App: list[Verdict] (fault-isolated)
    App->>Cas: close() / aclose()
    Note over Cas: Deterministic teardown: caller-owned resources preserved!
```

### Concurrency and Resilience Guarantees

1. **Ownership-Tracked Resource Teardown (`_owns_*`)**:
   - Every gate and `Cascade` tracks ownership flags (`_owns_backend`, `_owns_cache`, `_owns_sink`, `_owns_logit_gate`).
   - When a gate is closed via `close()` or `aclose()`, it tears down only resources it created automatically. Caller-injected clients, caches, or telemetry sinks remain open and healthy for reuse across the application.
   - All gates implement Python's `with` and `async with` context manager protocols for deterministic resource reclamation.

2. **Per-Model Singleflight BYO Calibration Locks (`calibration/lifecycle.py`)**:
   - Production CI runners often deploy unseen Bring-Your-Own (BYO) model checkpoints. Without explicit calibration, the first evaluation performs lazy micro-calibration probes (~16 calls).
   - If 10 worker threads simultaneously evaluate artifacts against an uncalibrated model, LogitGate uses per-model singleflight locks (`_SYNC_CAL_LOCKS` and event-loop-scoped `_ASYNC_CAL_LOCKS` stored in a `WeakKeyDictionary`).
   - The first thread executes the calibration probes; all other concurrent threads wait for the lock and immediately load the resulting `CalibrationProfile` with zero redundant calls.

3. **Bounded & Fault-Isolated Batch Concurrency**:
   - Batch operations (`evaluate_batch`, `aevaluate_batch`) accept `on_error: BatchErrorPolicy = "raise" | "abstain" | "fail"`.
   - In async execution, item 0 runs first to warm the prefix cache on single-node servers (`warm_prefix`). Subsequent items 1 through $N-1$ execute concurrently bounded by an `asyncio.Semaphore(max_concurrency)`.
   - Per-item `Exception` isolation catches evaluation errors without crashing the batch, while never swallowing `BaseException` subclasses (`asyncio.CancelledError`, `KeyboardInterrupt`).

4. **Async HTTP Client Lifecycle & Transient Retry (`_HTTPClientMixin`)**:
   - All HTTP backends (`OllamaBackend`, `OpenAICompatBackend`, `GeminiBackend`, `LlamaCppBackend`) inherit from `_HTTPClientMixin`.
   - Auto-created `httpx.AsyncClient` instances track the running event loop (`_async_client_loop`) and recreate cleanly when called across sequential `asyncio.run()` invocations.
   - Transient HTTP errors (`{408, 429, 502, 503, 504}`) and transport disconnections are retried with bounded exponential backoff, jitter, and strict honoring of the `Retry-After` header. Permanent client errors (`400`, `401`, `403`, `404`) fail immediately without retry.

5. **In-Process Backend Thread Safety**:
   - Forward and reverse debiasing passes run concurrently via `asyncio.gather`.
   - In-process backends with mutable state or native C++/CUDA bindings (`LlamaCppBackend`, `RecurrentPyTorchBackend`, `StubBackend`) serialize non-thread-safe native calls using a `threading.Lock` or `threading.RLock`.

---

## 5. Capability-Aware Backend Routing (`on_unsupported_logprobs`) & Gemini Engine

Certain modern frontier models (Anthropic Claude 3.5 Sonnet, OpenAI `o3-mini`, reasoning-tuned APIs) do not expose raw token logprobs. LogitGate handles this heterogeneous reality gracefully.

### The `on_unsupported_logprobs` Policy

All Layer-1 gates (`LogitGate`, `ChoiceGate`, `ScoreGate`, `MultiRubricGate`, `ToolCallGate`, `Cascade`) accept the parameter:
`on_unsupported_logprobs: Literal["raise", "escalate", "sidecar"] = "raise"`

```mermaid
flowchart TD
    req["Evaluation Request"] --> check{"Backend supports logprobs?"}
    check -->|"Yes (Ollama / vLLM / Gemini)"| L1["Layer 1: Raw First-Token Logprob Debiasing"]
    check -->|"No (Claude / o3-mini)"| policy{"on_unsupported_logprobs"}
    
    policy -->|"raise (Default)"| err["Raise UnsupportedLogprobsError"]
    policy -->|"escalate"| L2["Direct Escalation to Tier-2 JudgeGate (Generative Critique)"]
    policy -->|"sidecar"| sidecar["Route Layer 1 to sidecar_backend (e.g. Local 14B GPU)"]
    sidecar --> L1
```

- **`"raise"` (Default)**: Raises `UnsupportedLogprobsError` immediately with an actionable hint describing which backends support logprobs.
- **`"escalate"`**: Safely bypasses Layer 1 and escalates directly to Tier-2 `JudgeGate`, utilizing the non-logprob model for a full generative critique.
- **`"sidecar"`**: Routes the Layer-1 logprob check to an attached lightweight sidecar backend (`sidecar_backend=OllamaBackend(model="qwen2.5-coder:14b")`). This preserves sub-50ms Layer-1 filtering while allowing your primary application to remain on Claude or `o3`.

### Native Google Gemini Backend (`GeminiBackend`)

LogitGate includes a dedicated, zero-dependency REST backend for Google Gemini (`gemini-2.5-flash`, `gemini-2.5-pro`):

- **Header-Isolated Authentication**: Injects the API key into the `x-goog-api-key` HTTP header rather than appending it to the URL query string (`?key=...`), preventing credential exposure in proxy access logs and network traces.
- **Reasoning Suppression (`thinkingBudget=0`)**: Gemini 2.5 Flash employs internal reasoning tokens. For Layer-1 logprob extraction (`maxOutputTokens=1`), `GeminiBackend` automatically sets `thinkingConfig={"thinkingBudget": 0}` so thinking tokens do not consume the single emitted slot.
- **Logprob Configuration**: Automatically requests `responseLogprobs=True` with clamped logprob candidate distributions.

```python
from logitgate import LogitGate
from logitgate.backends import GeminiBackend

# Native zero-dependency Gemini backend with thinking suppression
backend = GeminiBackend(
    model="gemini-2.5-flash",
    api_key="your-api-key",  # Or read automatically from GEMINI_API_KEY
)

gate = LogitGate(
    rubric="Is the code mathematically correct and free of off-by-one errors?",
    backend=backend,
)
```

---

## 6. Deterministic Content-Addressed Verdict Caching (`VerdictCache`)

Repeated evaluations in CI suites, agent retry loops, and RAG regression runs frequently re-test identical `(gate, candidate)` pairs. `VerdictCache` provides sub-0.15ms memoization.

```mermaid
flowchart LR
    eval["Gate Evaluation"] --> hash["Compute Canonical Key lgcache:v1:sha256"]
    hash --> mem{"In RAM LRU?"}
    mem -->|"Hit (~0.11ms)"| ret["Return Cached Verdict"]
    mem -->|"Miss"| disk{"In SQLite WAL?"}
    disk -->|"Hit (~0.14ms)"| load["Promote to RAM + Return"]
    disk -->|"Miss"| run["Run Backend Inference (~45ms)"]
    run --> save["Store in RAM + SQLite WAL"]
```

### Safety and Security Invariants

- **Secret-Free Canonical Fingerprinting (`lgcache:v1:<sha256>`)**: Cache keys hash the gate class, routing traits, rubric, context, query, calibrated temperature, thresholds, backend protocol, model name, and candidate payload. API keys, authorization headers, and timestamps are stripped before hashing.
- **No Transient Ambiguity Caching (`cache_ambiguous=False`)**: Verdicts with `status=DecisionStatus.AMBIGUOUS` are never cached. If a model was uncertain or an input requires escalation, downstream gates must always have the opportunity to re-evaluate.
- **Preservation of Normalized Artifacts (`_normalized_out`)**: When `ASTGate(normalize=True)` strips markdown fences or `BestOfNGate` selects a winner, the normalized artifact is stored alongside the cached verdict so subsequent cascade stages receive the exact transformed artifact.
- **Replay Bypass**: `VerdictSnapshot.replay()` and benchmark warmups automatically pass `_bypass_cache=True` to guarantee that offline debugging replays always execute fresh inference.

---

## 7. Production Observability & Distributed Tracing (`OtelSpanSink`)

Production deployments require deep observability over fast-path short-circuit rates, Layer-1 vs. Layer-2 teacher disagreements, and latency breakdowns without leaking sensitive customer data into log files.

```mermaid
flowchart TD
    verdict["Verdict Generated"] --> filter{"Telemetry Sink Attached?"}
    filter -->|"No Sink"| fast["Zero-Allocation Fast Return (<0.001ms)"]
    filter -->|"Sink Attached"| sanitize["Sanitize PII (O(N) regex) / Omit Candidate"]
    sanitize --> otel["OtelSpanSink: Emit Span Attributes & Layer Events"]
    sanitize --> disk["JsonlTelemetrySink: Thread-Safe Log Rotation (O_NOFOLLOW)"]
    sanitize --> distill["DistillationSink: Discrepancy Mining for Auto-Tuning"]
```

- **Zero-Allocation Fast Path**: When no telemetry sink is attached, LogitGate returns immediately without allocating a `TelemetryRecord` object.
- **OpenTelemetry Integration (`OtelSpanSink`)**: Lazily bridges into `opentelemetry-api` when installed (`pip install 'logitgate[otel]'`). Emits standardized OpenTelemetry attributes (`gen_ai.system`, `gen_ai.request.model`, `logitgate.gate.*`, `logitgate.verdict.*`) and records per-layer executions as span events.
- **Symlink-Hardened Rotation & PII Redaction**: Telemetry sinks open log files under `os.O_NOFOLLOW` checks to prevent symlink hijacking. Four privacy modes (`"full"`, `"redact_pii"`, `"hash_only"`, `"omit_candidate"`) sanitize candidate text before disk persistence in linear $O(N)$ time.

---

## 8. Deployment Topologies: Preventing KV-Cache Eviction

When deploying evaluation pipelines alongside primary generation agents, a primary failure mode is **GPU Memory Contention and Cache Thrashing**: the judge model is loaded into the same VRAM as the generator, evicting the generator's multi-thousand-token context.

LogitGate supports three production topologies:

| Topology | Hardware Architecture | Generator Model | Gate Model | Latency | Memory Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Topology 1: Shared KV-Cache** | Single GPU (16–24 GB) | Same model | Same model (`MultiRubricGate`) | ~20–40 ms / rubric | Candidate cached once at Token 0 |
| **Topology 2: Dual-GPU Sidecar** | Dual GPU (e.g. 24 GB + 12 GB) | GPU 0 (32B MoE / Dense) | GPU 1 (14B or 4B Gate) | ~40–50 ms total | **Zero eviction** on GPU 0 |
| **Topology 3: CPU Sidecar** | Single GPU + Host RAM | GPU 0 (Full VRAM) | Host CPU (3B or 4B Q4) | ~160–250 ms total | **100% GPU VRAM** preserved for generator |

### Topology 2 Architecture (Heterogeneous Dual-GPU Sidecar)

```mermaid
flowchart LR
    agent["Agent Workload"] --> GPU0["GPU 0 (Primary Generator)<br/>e.g. 32B Coder (24 GB VRAM)<br/>Context: 32,000 tokens<br/><b>KV-Cache NEVER Evicted</b>"]
    
    GPU0 -->|"Generate Artifact"| gate_router["LogitGate Router"]
    
    gate_router -->|"Local IPC HTTP"| GPU1["GPU 1 (LogitGate Sidecar)<br/>e.g. 14B / 4B Gate (8 GB VRAM)<br/>Latency: ~40ms (0 CoT tokens)"]
    
    GPU1 -->|"Fast-Path PASS (P >= 0.85)"| done["Immediate Next Agent Step"]
    GPU1 -->|"Ambiguous (0.20 < P < 0.85)"| GPU0
    Note over GPU0: Escalates to GPU 0 for generative JudgeGate critique only when ambiguous!
```

---

## 9. Comprehensive Pipeline Execution Recipe

The following end-to-end Python script demonstrates the full modular architecture: `ASTGate` Layer 0 structural gating, `LogitGate` Layer 1 debiased logprob verification with content-addressed caching, and `JudgeGate` Layer 2 generative adjudication with bounded batch concurrency and deterministic context management.

```python
import asyncio
from logitgate import ASTGate, Cascade, JudgeGate, LogitGate
from logitgate.backends import OllamaBackend
from logitgate.cache import VerdictCache

async def main() -> None:
    # 1. Provision shared, ownership-tracked backend and cache
    backend = OllamaBackend(model="qwen2.5-coder:14b", keep_alive="10m")
    cache = VerdictCache(max_entries=1000)

    # 2. Assemble 3-Layer Cascade using context management
    async with Cascade(
        ASTGate(syntax=True, forbidden_imports=["subprocess", "shutil"]),
        LogitGate(
            rubric="Candidate code must be clean, correctly typed, and handle edge cases.",
            pass_threshold=0.85,
            fail_threshold=0.20,
            backend=backend,
            cache=cache,
        ),
        JudgeGate(
            max_tokens=100,
            trigger="on_ambiguous_only",  # Generative critique only when ambiguous
            backend=backend,
            cache=cache,
        ),
    ) as pipeline:
        candidates = [
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            "import subprocess\ndef exploit():\n    subprocess.run(['rm', '-rf', '/'])\n",
            "def complex_solve(matrix: list[list[float]]) -> float:\n    # Ambiguous solution\n    return 0.0\n",
        ]

        # 3. Evaluate batch with fault isolation and bounded concurrency
        verdicts = await pipeline.aevaluate_batch(
            candidates,
            on_error="abstain",
            max_concurrency=4,
        )

        for i, v in enumerate(verdicts):
            print(f"Item {i}: Passed={v.passed} at {v.triggered_layer} ({v.total_latency_ms:.1f}ms) - {v.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

[🏠 README](../README.md) · [📚 Docs Hub](./README.md) · [⚡ Quick Notes](./quick/quickstart_recipes.md) · [🌱 Visual ELI5](./eli5/how_logitgate_works.md) · [🚀 Power-User](./advanced/distillation_and_lora.md) · [📊 Benchmarks](./benchmarks.md)
