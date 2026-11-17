# RAG & Agent Tool Safety (⚡ Quick Notes)

Zero-fluff recipes for prefix-cached Grounded RAG evaluation, deterministic `ToolCallGate` pre-execution shields, and mid-stream `StreamGuard` aborts.

---

## 1. Prefix-Cached Grounded RAG Evaluation

Evaluates whether model answers are faithful to retrieved passages and directly address user queries, without hallucinating external facts.

![Prefix-Cached Grounded RAG Flow](../assets/rag_prefix_cache.svg)

### Copy-Paste Recipe

```python
from logitgate import LogitGate, MultiRubricGate

# 1. Single-rubric groundedness check
rag_gate = LogitGate(
    rubric="Is the generated answer strictly supported by the context without external claims?",
    pass_threshold=0.85,
    fail_threshold=0.20,
)

context = [
    "Customer account #9102 registered on 2024-05-18 under the Enterprise tier.",
    "Billing contact is admin@enterprise.com; invoices settle on net-30 terms.",
]
query = "What tier is account #9102 on and when was it registered?"
candidate = "Account #9102 was registered on May 18, 2024 under the Enterprise tier."

verdict = rag_gate.evaluate(candidate, context=context, query=query)
print(f"Grounded: {verdict.passed} (P={verdict.probability:.4f})")

# 2. Multi-rubric groundedness suite (shares 95%+ of KV-cache across checks)
rubrics = [
    "Is every claim in the response directly supported by the context?",
    "Does the response directly answer the user query?",
    "Does the response avoid ungrounded numeric figures or dates?",
]
multi_rag = MultiRubricGate(rubrics, aggregation="all")
suite_verdict = multi_rag.evaluate(candidate, context=context, query=query)
print(f"All RAG rubrics passed: {suite_verdict.passed}")
```

### Why Prefix Caching Matters for RAG

In standard RAG pipelines, retrieved context can span thousands of tokens. LogitGate places the static rubric, query, and retrieved context chunks at the **root** of the prompt, swapping only the terminal answer token mappings at the very end of the sequence.

On local inference engines (`Ollama`, `vLLM`, `llama.cpp`), Pass 2 achieves a **95% to 99% KV-cache hit rate**, delivering up to **2.51x speedups** over uncached two-pass evaluation.

<details>
<summary>🌱 <b>ELI5 / Why this works</b></summary>

Retrieved context is often large. Instead of re-reading the whole document twice, LogitGate keeps the document in the GPU's memory cache and only changes the two letters at the very end. The second pass evaluates almost instantaneously.
</details>

[🔬 Mathematical Debiasing & Prefix Mechanics](../math_and_debiasing.md) · [🔬 Production RAG Architectures](../use_cases.md)

---

## 2. `ToolCallGate` (Deterministic Agent Pre-Execution Shield)

Intercepts agent tool calls before execution to block path traversal, SSRF, destructive shell/SQL commands, and schema violations at Layer 0 (<0.1 ms, 0 LLM calls).

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
| `block_destructive_sql` | `bool` | `True` | Blocks destructive DDL/DML statements (`DROP`, `TRUNCATE`, `ALTER`). |
| `check_intent` | `bool` | `False` | Evaluates Layer 1 semantic alignment between tool call and user intent. |

<details>
<summary>🌱 <b>ELI5 / Why this works</b></summary>

Never let an LLM directly execute file paths or shell commands. ToolCallGate inspects the tool call stringently in Python memory before your runtime ever touches the filesystem or database. 70% of attacks are caught deterministically in 0.08 ms without calling an LLM.
</details>

[🔬 Security Threat Models & Adversarial Defense](../deep/security_and_adversarial.md)

---

## 3. `StreamGuard` (Mid-Stream Generation Abort)

Monitors Server-Sent Event (SSE) token generation mid-flight, aborting streams immediately when toxic output, security leaks, or jailbreak confabulations appear.

![StreamGuard Mid-Stream Abort](../assets/stream_guard_abort.svg)

### Copy-Paste Recipe

```python
from logitgate import LogitGate, guard_stream

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
except Exception as exc:
    print(f"\n[STREAM TERMINATED BY GUARD]: {exc}")
```

### Delivery Modes: Optimistic vs. Buffered

- **`delivery="optimistic"`**: Yields chunks to the client immediately with zero latency penalty. Evaluates safety over a sliding character window in the background. If a violation is detected, raises `StreamAbortedError` and tears down the connection.
- **`delivery="buffered"`**: Holds chunks in a quarantine buffer until the current stride has been validated by Layer 1. Prevents any disallowed tokens from ever reaching the end-user, at the cost of a small buffer delay.

[🔬 Systems Concurrency & Stream Lifecycle](../architecture.md)
