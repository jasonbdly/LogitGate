[🏠 README](../README.md) · [📚 Docs Hub](./README.md) · [⚡ Quick Notes](./quick/quickstart_recipes.md) · [🧭 Visual Guide](./eli5/how_logitgate_works.md) · [🚀 Power-User](./advanced/distillation_and_lora.md) · [📊 Benchmarks](./benchmarks.md)

# Production Use Cases & Execution Recipes

This document provides role-based architecture guides, production execution recipes, and prompting best practices for **LogitGate v0.15.0+**. For introductory tutorials, see [Quickstart Recipes](./quick/quickstart_recipes.md); for weight distillation, see [Distillation & LoRA](./advanced/distillation_and_lora.md).

---

## 1. CI/CD Final Merge Gate vs. Local Developer Inner Loop

In modern software delivery, **local iteration loops** and **CI/CD merge gates** optimize for different objective functions:
- **Local Developer Loop**: Prioritizes sub-second iteration speed and low-overhead triage (~40–90 ms on GPU, ~160 ms 1-pass on CPU).
- **CI/CD Merge Gate**: Prioritizes strict correctness, zero false passes, and reproducible failure snapshots without generating 400-token conversational LLM critiques on clear verdicts.

| Dimension | Local Developer Inner Loop | CI/CD Merge Gate (Recommended) |
| :--- | :--- | :--- |
| **Primary Goal** | Real-time agent feedback / instant triage | **High correctness & zero false passes** without stalling runners on verbose essays |
| **Preset & Debiasing** | `preset="balanced"` (GPU ~85 ms) or `preset="fast_triage"` (CPU 1-pass ~160 ms) | **`preset="balanced"` or `"strict"` (`symmetric_debiasing=True`)** - always cancel prior token bias |
| **Multi-Part Rubrics** | Single rubric or `ChoiceGate` | **Atomic rubrics via `MultiRubricGate`** (weakest-link aggregation, shared-prefix KV reuse) |
| **Layer 2 (`JudgeGate`) Trigger** | Omitted, or `trigger="on_ambiguous_only"` | **`trigger="on_ambiguous_only"` + `save_on_fail=True`** (0 CoT tokens on clear verdicts; invokes judge only when ambiguous) |
| **Caching** | In-memory `VerdictCache(max_entries=500)` | **Persistent `VerdictCache(path=".cache/logitgate.db")`** for instant re-runs |

- **Operational Impact**: Clear passes and clear fails incur **0 CoT tokens** (~40-90 ms on GPU; ~0.14 ms on SQLite WAL cache hit). Only ambiguous items (`0.20 < P < 0.85`) pay Layer-2 generative judge latency, while `save_on_fail=True` guarantees deterministic local reproduction of CI failures.

### Production CI Merge Gate Recipe

```python
from logitgate import ASTGate, Cascade, JudgeGate, LogitGate
from logitgate.backends import OllamaBackend
from logitgate.cache import VerdictCache

# 1. Configure backend for CPU CI runner (0 GPU, ~3.2 GB RAM for 3B / ~9.4 GB RAM for gemma4:e4b)
backend = OllamaBackend(
    model="qwen2.5-coder:14b",
    options={"num_ctx": 4096},
    keep_alive="5m",
)

# 2. Wire deterministic cache to eliminate redundant CI re-evaluations
cache = VerdictCache(path=".cache/logitgate.db", max_entries=5000)

# 3. Assemble 3-Layer Gate with deterministic context management
with Cascade(
    # Layer 0: Catch syntax and dangerous imports in <0.1ms without LLM calls
    ASTGate(syntax=True, forbidden_imports=["pickle", "subprocess", "shutil"]),
    # Layer 1: Full 2-pass symmetric debiasing (0 CoT tokens on clear pass OR clear fail)
    LogitGate(
        rubric="Candidate code must be clean, secure, type-hinted, and handle edge cases.",
        preset="balanced",
        save_on_fail=True,  # Freezes .logitgate/failures/*.json on failure
        backend=backend,
        cache=cache,
    ),
    # Layer 2: Fires ONLY when Layer 1 confidence is ambiguous (0.20 < P < 0.85)
    JudgeGate(
        max_tokens=100,
        trigger="on_ambiguous_only",
        backend=backend,
        cache=cache,
    ),
) as ci_pipeline:
    candidate_code = (
        "def compute_interest(principal: float, rate: float, periods: int) -> float:\n"
        "    if principal < 0 or rate < 0 or periods < 0:\n"
        "        raise ValueError('Parameters must be non-negative')\n"
        "    return principal * ((1 + rate) ** periods)\n"
    )
    verdict = ci_pipeline.evaluate(candidate_code)
    assert verdict.passed, f"CI Gate Failed at {verdict.triggered_layer}: {verdict.reason}"
```

### Reproducing CI Failures Locally (`VerdictSnapshot` & `logitgate explain`)

When non-deterministic LLM applications fail in CI, re-running them locally often produces a different completion, making bugs difficult to reproduce.

LogitGate solves this via **Deterministic Verdict Snapshots**:
1. When a test fails in CI (with `save_on_fail=True` or `LOGITGATE_SAVE_FAILURES=1`), LogitGate freezes the exact candidate text, rubric, and telemetry to `.logitgate/failures/fail_<timestamp>_<id>.json` (updating `.logitgate/failures/latest.json`).
2. Developers diagnose the exact failure locally with a single CLI command without re-running the application:

```bash
# Replay and explain the latest failure with full System-2 reasoning
logitgate explain

# Or explain a specific snapshot with custom token limit
logitgate explain .logitgate/failures/latest.json --max-tokens 150
```

Programmatic snapshot replay via Python:

```python
from logitgate import VerdictSnapshot

# Load frozen snapshot generated during CI failure
snapshot = VerdictSnapshot.load(".logitgate/failures/latest.json")

# Replay fast Layer-1 logit evaluation on the exact frozen candidate
replay_verdict = snapshot.replay()

# Or generate a deep Layer-2 diagnostic critique with reasoning enabled
explanation = snapshot.explain(model="qwen2.5-coder:14b", think=True)
print(f"Diagnostic critique: {explanation.reason}")
```

---

## 2. Grounded RAG & Hallucination Guardrails

In Retrieval-Augmented Generation (RAG), verification must ensure answers are strictly supported by retrieved context documents without hallucinations.

LogitGate separates the **evaluated output** (`candidate`, passed as the first positional argument) from the **grounding inputs** (`context=` and `query=`). Placing `context` and `query` into nonce-isolated sections ahead of `[CANDIDATE ARTIFACT]` preserves GPU KV-cache reuse across multiple checks on the same retrieved passages while ensuring unidirectional causal attention attends back to the source facts.

- **Operational Impact**: Verifies groundedness in **~45-90 ms** with **0 generated tokens**, using `preset="strict"` (`pass_threshold=0.90`) to block unverified claims before they reach end users.

```python
from logitgate import ContractGate, LogitGate
from logitgate.backends import OllamaBackend

backend = OllamaBackend(model="qwen2.5-coder:14b")

# Layer 0 verifies reasonable character bounds; Layer 1 verifies factual grounding
rag_verifier = (
    ContractGate(min_length=20, max_length=2000)
    >> LogitGate(
        rubric="Does the candidate answer strictly contain only facts asserted in the reference context?",
        preset="strict",  # 90% confidence threshold for high-stakes grounding
        backend=backend,
    )
)

retrieved_context = (
    "LogitGate v0.15.0 introduces ToolCallGate for agent sandboxing, GeminiBackend for zero-dependency "
    "Google Gemini REST integration, and capability-aware sidecar routing for models lacking logprobs."
)
user_query = "What features were added in LogitGate v0.15.0?"
generated_answer = "LogitGate v0.15.0 added ToolCallGate, GeminiBackend, and sidecar routing."

# Pass context and query directly into evaluate()
verdict = rag_verifier.evaluate(
    generated_answer,
    context=retrieved_context,
    query=user_query,
)

if not verdict.passed:
    # Trigger fallback or retrieve secondary sources
    generated_answer = "I cannot verify that answer against official documentation."
print(f"Groundedness Verified: {verdict.passed} (P = {verdict.probability:.2f})")
```

---

## 3. Autonomous AI Agent & Tool-Call Sandboxing (`ToolCallGate`)

Autonomous agents frequently execute tools across APIs, databases, and local operating systems. Indirect prompt injection attacks or hallucinations can cause agents to emit catastrophic parameters: SSRF requests against cloud metadata endpoints (`169.254.169.254`), path traversal outside the project directory, or destructive SQL commands.

`ToolCallGate` separates the **untrusted tool payload** (`candidate` passed to `evaluate()`) from the **trusted user request** (`user_intent=`), enforcing a dual-stage barrier:
- **Operational Impact**: Stage-0 structural rules (path traversal, SSRF, destructive SQL/shell) reject attacks in **<0.05 ms with 0 LLM calls**; Stage-1 semantic intent checks fire only on structurally clean calls (~45 ms GPU), catching indirect prompt injections that stay inside valid parameter syntax.

```python
from logitgate import Cascade, JudgeGate, ToolCallGate
from logitgate.backends import OllamaBackend

backend = OllamaBackend(model="qwen2.5-coder:14b")

# Configure hybrid ToolCallGate:
# - Stage 0 (<0.1ms): Blocks path traversal, SSRF against metadata/internal IPs, destructive SQL
# - Stage 1 (~45ms): Assesses semantic alignment against user intent
tool_guard = ToolCallGate(
    allowed_tools=["fetch_url", "read_file", "execute_sql"],
    allowed_roots=["/home/user/workspace/app"],
    block_path_traversal=True,
    block_ssrf=True,
    block_destructive_commands=True,
    block_destructive_sql=True,
    check_intent=True,
    user_intent="Retrieve user profile data for user_id 42",
    backend=backend,
)

# Connect to Cascade for escalation if semantic intent is ambiguous
agent_firewall = Cascade(
    tool_guard,
    JudgeGate(trigger="on_ambiguous_only", backend=backend),
)

# Safe tool call:
safe_call = {
    "name": "execute_sql",
    "arguments": {"query": "SELECT username, email FROM users WHERE id = 42;"},
}
assert agent_firewall.evaluate(safe_call).passed

# Malicious SSRF call: Blocked immediately at Stage 0 in <0.05ms (0 LLM calls)
ssrf_call = {
    "name": "fetch_url",
    "arguments": {"url": "http://169.254.169.254/latest/meta-data/"},
}
verdict_blocked = agent_firewall.evaluate(ssrf_call)
assert not verdict_blocked.passed
print(f"Blocked malicious tool call: {verdict_blocked.reason}")
```

---

## 4. Real-Time Mid-Stream Token Guardrails (`StreamGuard`)

Post-hoc verification waits for an upstream generator to finish producing 100% of its tokens before running a check. On a 500-token response, this wastes compute on toxic or violating responses, delays time-to-first-token (TTFT), and risks leaking unsafe tokens to the client.

`StreamGuard` (`logitgate.streaming`) wraps synchronous (`Iterable`) and asynchronous (`AsyncIterable`) token streams:

![StreamGuard Mid-Stream Abort](./assets/stream_guard_abort.svg)

- **Clause-Aligned Stride Scheduling**: Checkpoints trigger every `stride_chars` characters, waiting up to `max_wait_chars` for a natural clause boundary (`.`, `!`, `?`, `\n`, `}`, `;`) to avoid evaluating mid-word fragments.
- **Delivery Modes**:
  - `delivery="optimistic"`: Yields tokens immediately with zero buffering latency, aborting and canceling the upstream generator via `.close()` / `await .aclose()` as soon as a violation triggers, saving **40%–50%+** of downstream tokens.
  - `delivery="buffered"`: Holds tokens in a bounded window, flushing them to the client only when a checkpoint returns `passed=True`, guaranteeing **0% violating token leakage**.

### Asynchronous Stream Guarding Recipe

```python
import asyncio
from typing import AsyncIterable
from logitgate import LogitGate
from logitgate.backends import OllamaBackend
from logitgate.streaming import aguard_stream

async def mock_streaming_llm() -> AsyncIterable[str]:
    chunks = [
        "To configure the database, ",
        "execute the following shell command: ",
        "rm -rf / --no-preserve-root. ",
        "This will reset your system environment completely.",
    ]
    for chunk in chunks:
        await asyncio.sleep(0.05)
        yield chunk

async def run_stream() -> None:
    backend = OllamaBackend(model="qwen2.5-coder:14b")
    gate = LogitGate("Ensure response does not recommend destructive system commands.", backend=backend)

    # Wrap the generator in StreamGuard
    guarded = aguard_stream(
        mock_streaming_llm(),
        gate,
        stride_chars=30,
        delivery="buffered",  # 0% violating token leakage
    )

    try:
        async for safe_token in guarded:
            print(safe_token, end="", flush=True)
    except Exception as exc:
        print(f"\n[STREAM INTERCEPTED & HALTED]: {exc}")

if __name__ == "__main__":
    asyncio.run(run_stream())
```

---

## 5. Heterogeneous Model Routing for Non-Logprob APIs (`on_unsupported_logprobs="sidecar"`)

Frontier proprietary models (Anthropic Claude 3.5 Sonnet, OpenAI `o3-mini`) do not expose raw token logprobs. If an application uses one of those models as its primary generator or Layer-2 critic, `on_unsupported_logprobs="sidecar"` routes Layer-1 logprob checks to a compact local model (`sidecar_backend`) and invokes the expensive frontier model only on `AMBIGUOUS` cases.

- **Operational Impact**: Cuts frontier API calls and token spend by **75%-85%+** while keeping Layer-1 triage latency under ~50 ms locally.

```python
from logitgate import Cascade, JudgeGate, LogitGate
from logitgate.backends import OllamaBackend, OpenAICompatBackend

# Primary backend: High-capability generative model lacking logprobs
primary_backend = OpenAICompatBackend(
    model="o3-mini",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
)

# Sidecar backend: Local compact model providing fast first-token logprobs
local_sidecar = OllamaBackend(model="qwen2.5-coder:14b")

# Configure Cascade with transparent sidecar offloading:
with Cascade(
    LogitGate(
        rubric="Is the financial report accurate and free of arithmetic discrepancies?",
        backend=primary_backend,
        on_unsupported_logprobs="sidecar",  # Routes Layer 1 to local sidecar!
        sidecar_backend=local_sidecar,
    ),
    JudgeGate(
        trigger="on_ambiguous_only",  # Invokes o3-mini only when ambiguous
        backend=primary_backend,
    ),
) as hybrid_pipeline:
    candidate_report = "Revenue increased by 15% from $10M to $11.5M. Net margin was 20% ($2.3M)."
    verdict = hybrid_pipeline.evaluate(candidate_report)
    print(f"Passed: {verdict.passed} (Layer: {verdict.triggered_layer})")
```

---

## 6. Atomic Multi-Criteria Evaluation (`MultiRubricGate`)

When an artifact must satisfy several independent requirements (e.g. style, security, type safety, test coverage), cramming all criteria into a single compound prompt causes logprob dilution (a strong "yes" on three criteria masks a "no" on the fourth).

`MultiRubricGate` keeps each criterion isolated in `rubrics=[...]` and evaluates them against a single `candidate` using weakest-link aggregation:
- **Operational Impact**: Reuses the cached `candidate` prefix across all criteria (~20-35 ms per additional rubric on GPU), and `fail_fast=True` halts evaluation on the first failed criterion while recording per-rubric traces for `logitgate explain`.

```python
from logitgate import MultiRubricGate
from logitgate.backends import OllamaBackend

backend = OllamaBackend(model="qwen2.5-coder:14b")

rubrics = [
    "Candidate code is strictly formatted according to PEP 8 standards.",
    "Candidate code contains complete Google-style docstrings.",
    "Candidate code includes complete, non-Any Python type annotations.",
    "Candidate code does not perform unchecked file or shell operations.",
]

evaluator = MultiRubricGate(
    rubrics=rubrics,
    backend=backend,
    fail_fast=True,  # Short-circuits remaining rubrics on first failure
)

sample_code = (
    "def add_items(items: list[int]) -> int:\n"
    "    '''Sums all integers in the provided list.\n\n"
    "    Args:\n"
    "        items: List of integers.\n\n"
    "    Returns:\n"
    "        The computed integer sum.\n"
    "    '''\n"
    "    return sum(items)\n"
)

# Evaluates all rubrics, reusing the shared candidate prefix in GPU KV-cache
verdict = evaluator.evaluate(sample_code)
print(f"Overall Passed: {verdict.passed} | Reason: {verdict.reason}")
```

---

[🏠 README](../README.md) · [📚 Docs Hub](./README.md) · [⚡ Quick Notes](./quick/quickstart_recipes.md) · [🧭 Visual Guide](./eli5/how_logitgate_works.md) · [🚀 Power-User](./advanced/distillation_and_lora.md) · [📊 Benchmarks](./benchmarks.md)
