[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](../eli5/how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)

---

# RAG & Agent Tool Safety (⚡ Quick Notes)

Production recipes for prefix-cached Grounded RAG evaluation, deterministic `ToolCallGate` pre-execution shields, and mid-stream `StreamGuard` aborts.

---

## 1. Prefix-Cached Grounded RAG Evaluation (`context=` & `query=`)

**Problem & deployment**: Catches hallucinations in live Retrieval-Augmented Generation (RAG) chatbots and knowledge-base QA pipelines before ungrounded claims reach users, evaluating multi-chunk evidence in `~40-90 ms` with `0` generated tokens.

![Prefix-Cached Grounded RAG Flow](../assets/rag_prefix_cache.svg)

### Why `context=` and `query=` Exist

Passing retrieved passages separately via `context=` and the user's question via `query=` solves three problems at once:

1. **Verifying private facts absent from weights**: Evaluates claims against internal policies, account records, or recent retrieved documents the evaluator model never saw during pretraining.
2. **Nonce-isolating trusted context from untrusted drafts**: Wraps `context=` and `candidate=` in distinct randomized boundary tags so prompt injections inside an untrusted LLM draft cannot spoof retrieved evidence or override the rubric.
3. **Sharing the long document KV-cache prefix**: Single-rubric `LogitGate` (`query_first=True`) orders sections as `[EVALUATION RUBRIC] -> [REFERENCE CONTEXT] -> [USER QUERY] -> [CANDIDATE ARTIFACT]` while swapping only `[DECISION MAPPING]` across Pass 1 and Pass 2; `MultiRubricGate` (`query_first=False`) pins `[REFERENCE CONTEXT] -> [USER QUERY] -> [CANDIDATE ARTIFACT]` at the prompt root and appends each `[EVALUATION RUBRIC]` + `[DECISION MAPPING]` at the suffix so all rubrics share one cached prefix (**95% to 99% KV-cache hit rate**, up to **2.51x speedup**).

### Copy-Paste Recipe (Grounded vs. Hallucinated)

```python
from logitgate import LogitGate, MultiRubricGate

# 1. Single-rubric groundedness check
rag_gate = LogitGate(
    rubric="Is the generated answer strictly supported by the retrieved context without external claims?",
    pass_threshold=0.85,
    fail_threshold=0.20,
)

# Trusted retrieved passages (prefilled once in KV-cache)
retrieved_policy_chunks = [
    "Standard tier: 14-day refund window.",
    "Enterprise tier: 30-day refund window.",
]
user_question = "Can I get a refund on my Standard plan after 20 days?"

# Example A: Grounded reply (passes with high calibrated probability, e.g. P >= 0.90)
faithful_reply = (
    "Standard plans only qualify for refunds within 14 days; "
    "the 30-day window applies only to Enterprise."
)
v_grounded = rag_gate.evaluate(
    candidate=faithful_reply,
    context=retrieved_policy_chunks,
    query=user_question,
)
print(f"Grounded: {v_grounded.passed} (P={v_grounded.probability:.4f})")

# Example B: Hallucinated reply (contradicts the 14-day Standard policy -> fails on live backend, P <= 0.20)
hallucinated_reply = "Yes, Standard plans qualify for a 30-day refund window."
v_hallucinated = rag_gate.evaluate(
    candidate=hallucinated_reply,
    context=retrieved_policy_chunks,
    query=user_question,
)
print(
    f"Hallucinated reply: passed={v_hallucinated.passed}, status={v_hallucinated.status.value}"
)  # Live backend returns passed=False, status='failed'

# 2. Multi-rubric groundedness suite (reuses the same KV-cache prefix across all 3 rubrics)
rubrics = [
    "Is every claim in the response directly supported by the context?",
    "Does the response directly answer the user query?",
    "Does the response avoid ungrounded numeric figures or dates?",
]
multi_rag = MultiRubricGate(rubrics, fail_fast=False)
suite_verdict = multi_rag.evaluate(
    candidate=faithful_reply,
    context=retrieved_policy_chunks,
    query=user_question,
)
print(f"All RAG rubrics passed: {suite_verdict.passed}")
```

<details>
<summary>💡 <b>Why this works</b></summary>

Reference documents are often much longer than the AI's short answer. Instead of making the GPU re-read a 20-page document from scratch on every check, LogitGate keeps the retrieved context, user question, and candidate draft inside the cached prefix while only the short rubric or `A/B` label mapping at the bottom changes.
</details>

[🔬 Mathematical Debiasing & Prefix Mechanics](../math_and_debiasing.md) · [🔬 Production RAG Architectures](../use_cases.md)

---

## 2. `ToolCallGate` (Deterministic Agent Pre-Execution Shield)

**Problem & deployment**: Protects Model Context Protocol (MCP) servers, coding agents, and database tool runners by blocking path traversal, Server-Side Request Forgery (SSRF) metadata probes, and destructive shell/SQL arguments at Layer 0 (`<0.1 ms`, `0` LLM calls) before any runtime touches disk or network.

![ToolCallGate Structural Pre-Execution Shield](../assets/toolcall_shield.svg)

### Copy-Paste Recipe

```python
from logitgate import ToolCallGate

# Initialize security gate with filesystem, network, and command restrictions
tool_gate = ToolCallGate(
    allowed_tools=["read_document", "query_database"],
    allowed_roots=["/var/data/reports"],
    allowed_hosts=["api.internal.company.com"],
    block_path_traversal=True,
    block_ssrf=True,
    block_destructive_commands=True,
    block_destructive_sql=True,
    check_intent=True,
    user_intent="Fetch annual audit summary for fiscal year 2025",
)

# Case A: Valid tool call
safe_call = {
    "name": "read_document",
    "arguments": {"path": "/var/data/reports/audit_2025.pdf"},
}
assert tool_gate.evaluate(safe_call).passed

# Case B: Path traversal attack (blocked at Layer 0 in <0.08ms)
traversal_call = {
    "name": "read_document",
    "arguments": {"path": "/var/data/reports/../../../../etc/shadow"},
}
v_traversal = tool_gate.evaluate(traversal_call)
assert not v_traversal.passed
print(f"Blocked traversal: {v_traversal.reason}")

# Case C: SSRF cloud metadata IP attack (blocked at Layer 0)
ssrf_call = {
    "name": "query_database",
    "arguments": {"endpoint": "http://169.254.169.254/latest/meta-data/"},
}
v_ssrf = tool_gate.evaluate(ssrf_call)
assert not v_ssrf.passed
print(f"Blocked SSRF: {v_ssrf.reason}")
```

### Parameter Reference & Built-In Defenses

| Parameter | Type | Default | Defense Mechanism |
| :--- | :--- | :--- | :--- |
| `allowed_tools` | `Sequence[str] \| None` | `None` | Allowlist of callable tool names; unknown tools fail immediately. |
| `blocked_tools` | `Sequence[str] \| None` | `None` | Explicit denylist of dangerous tools (e.g. `bash`, `exec`). |
| `allowed_roots` | `Sequence[str] \| None` | `None` | Directory boundaries for any parameter named `path`, `file`, etc. |
| `allowed_hosts` | `Sequence[str] \| None` | `None` | Permitted network hostnames; blocks external or loopback egress. |
| `block_path_traversal` | `bool` | `True` | Resolves canonical paths and rejects `../` escapes. |
| `block_ssrf` | `bool` | `True` | Blocks private subnets, localhost, and cloud metadata (`169.254.169.254`). |
| `block_destructive_commands` | `bool` | `True` | Blocks shell tokens (`rm -rf`, `mkfs`, `sudo`, `dd`). |
| `block_destructive_sql` | `bool` | `True` | Blocks destructive schema/data SQL statements (DDL/DML: `DROP`, `TRUNCATE`, `ALTER`). |
| `check_intent` | `bool` | `False` | Evaluates Layer 1 semantic alignment between tool call and user intent. |

<details>
<summary>💡 <b>Why this works</b></summary>

Never let an AI model pass file paths, URLs, or terminal commands straight to your operating system. `ToolCallGate` inspects the proposed action in Python memory first, catching ~70% of dangerous paths, private URLs, and destructive commands in under `0.1 ms` before touching a file, network socket, or AI model.
</details>

[🔬 Security Threat Models & Adversarial Defense](../deep/security_and_adversarial.md)

---

## 3. `StreamGuard` (Mid-Stream Generation Abort)

**Problem & deployment**: Guards live Server-Sent Events (SSE) chat endpoints and streaming agent responses by checking token prefixes at sliding character strides and tearing down the upstream generator mid-flight if personally identifiable information (PII) leaks, policy violations, or jailbreaks begin to emerge.

![StreamGuard Mid-Stream Abort](../assets/stream_guard_abort.svg)

### Copy-Paste Recipe

```python
from logitgate import LogitGate, guard_stream
from logitgate.exceptions import StreamAbortedError

gate = LogitGate("Does the assistant response maintain professional security boundaries?")

def token_generator():
    yield "The system network status is healthy. "
    yield "Here is the master root password: "
    yield "hunter2_super_secret"

# Wrap any iterable token stream
stream = guard_stream(
    token_generator(),
    gate=gate,
    min_chars=32,
    stride_chars=64,
    delivery="optimistic",  # "optimistic" | "buffered"
)

try:
    for chunk in stream:
        print(chunk, end="", flush=True)
except StreamAbortedError as exc:
    print(f"\n[STREAM TERMINATED BY GUARD]: {exc}")
```

### Delivery Modes: Optimistic vs. Buffered

- **`delivery="optimistic"`**: Yields tokens immediately for zero added time-to-first-token and evaluates the **cumulative response prefix** inline at each clause/character-stride checkpoint, halting the upstream generator immediately (raising `StreamAbortedError`) if a checkpoint fails or confidence drops at or below `abort_prob_threshold`.
- **`delivery="buffered"`**: Holds chunks in a quarantine buffer until the current stride has been validated by Layer 1. Prevents any disallowed tokens from ever reaching the end-user, at the cost of a small buffer delay.

**When to skip `StreamGuard`**: For short replies under `150` characters, a single post-hoc `LogitGate.evaluate(...)` call is cheaper and avoids mid-clause ambiguity before the full sentence completes.

[🔬 Systems Concurrency & Stream Lifecycle](../architecture.md)

---

[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](../eli5/how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)
