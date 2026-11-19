[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](../eli5/how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)

---

# Framework Integrations & OpenTelemetry (⚡ Quick Notes)

Copy-paste integration recipes for `pytest` regression suites, `LangGraph` agent workflows, `DeepEval` pipelines, `OpenTelemetry` tracing, and the CLI.

---

## 1. Pytest Assertions (`@check`)

**Problem & deployment**: Turns natural-language rubrics into deterministic CI merge gates and regression tests running in `~16-90 ms` without fragile regexes or manual prompt parsing, with optional failure snapshots for offline `logitgate explain` triage.

### Copy-Paste Recipes

```python
import pytest
from logitgate import check

# 1. Standard boolean evaluation
@check("The support reply apologizes for the billing error and confirms a refund")
def test_support_refund_reply():
    return "We apologize for the duplicate charge on invoice #902. A full refund has been credited."

# 2. Categorical classification (choices)
@check(
    "Classify the sentiment of the customer feedback",
    choices={"A": "positive", "B": "neutral", "C": "negative"},
    pass_choices=["A"],
    min_confidence=0.70,
)
def test_feedback_sentiment():
    return "The product was delivered early and exceeded my expectations!"

# 3. Numeric rubric scoring (score_range)
@check(
    "Rate code quality and maintainability from 1 to 5",
    score_range=(1, 5),
    score_threshold=4.0,
)
def test_code_quality():
    return "def compute_area(radius: float) -> float:\n    import math\n    return math.pi * radius * radius\n"

# 4. Multi-rubric verification (rubrics)
@check(
    rubrics=[
        "Does the SQL query use parameterized placeholders instead of f-strings?",
        "Does the query avoid SELECT * wildcard columns?",
    ],
    aggregation="all",
)
def test_safe_sql():
    return "SELECT id, email FROM users WHERE id = :user_id"

# 5. Grounded RAG compliance (context)
@check(
    "Is the generated statement strictly supported by the context?",
    context=["Cluster maintenance starts Sunday at 02:00 UTC and lasts 30 minutes."],
    query="When is cluster maintenance?",
)
def test_maintenance_groundedness():
    return "Cluster maintenance is scheduled for Sunday at 02:00 UTC for 30 minutes."
```

### Pytest CLI Flags & Failure Triage

```bash
# Run tests against a specific backend
pytest --logitgate-backend ollama://localhost:11434/qwen2.5:7b

# Save failure snapshots for deep offline inspection
pytest --logitgate-save-failures

# Explain the latest broken test using Layer 2 step-by-step critique
logitgate explain .logitgate/failures/latest.json
```

<details>
<summary>💡 <b>Why this works</b></summary>

Instead of writing brittle string-matching rules to test AI outputs, `@check` reads the model's pass/fail confidence in milliseconds with zero explanation words written when tests pass. Whenever a test fails, you can run `logitgate explain` to get a written explanation of what went wrong.
</details>

[🔬 Framework Integration Guide](../integrations.md) · [🔬 Production CI/CD Gate Patterns](../use_cases.md)

---

## 2. LangGraph Conditional Edge Routing

**Problem & deployment**: Prevents unsafe or off-policy tool plans from reaching execution nodes in stateful LangGraph agents by routing graph edges on calibrated Layer 1 probabilities and escalating ambiguous steps to human review.

### Copy-Paste Recipe

```python
from logitgate import LogitGate
from logitgate.integrations.langgraph import as_conditional_edge

# Gate verifying that proposed agent actions do not delete production data
safety_gate = LogitGate("Does the proposed action avoid permanent data deletion?")

# Create a conditional edge router for LangGraph
edge_router = as_conditional_edge(
    safety_gate,
    pass_node="execute_action",
    fail_node="human_escalation",
    state_key="action_proposal",
    ambiguous_node="human_escalation",
)

# Simulate LangGraph state dictionary
sample_state = {"action_proposal": "SELECT id FROM users WHERE status = 'active'"}
next_node = edge_router(sample_state)
print(f"Routing target: {next_node}")  # 'execute_action'
```

### Supported State Containers

`as_conditional_edge` and `as_async_conditional_edge` automatically extract candidates from:
- Standard `dict` / `Mapping` objects
- Pydantic models (v1 and v2)
- Python `@dataclass` instances

---

## 3. DeepEval System-1 Adapter

**Problem & deployment**: Accelerates existing DeepEval evaluation suites without rewriting test cases by substituting slow generative judges with fast System-1 (single-pass, zero-generation) scoring via `LogitGateMetric` and `LogitGateSystemOneModel`.

### Copy-Paste Recipe

```python
from deepeval.test_case import LLMTestCase
from logitgate.integrations.deepeval import LogitGateMetric, LogitGateSystemOneModel

# 1. Fast Drop-In Evaluation Metric
metric = LogitGateMetric(
    rubric="The response provides clear, actionable technical instructions",
    threshold=0.85,
)

test_case = LLMTestCase(
    input="How do I reset my API key?",
    actual_output="1. Open Settings. 2. Select API Keys. 3. Click Rotate Key.",
)

metric.measure(test_case)
print(f"Metric Passed: {metric.is_successful()} (Score: {metric.score:.4f})")

# 2. System-1 Model Adapter (evaluates Yes/No, Choice, and Score without generation)
system_one_model = LogitGateSystemOneModel(
    backend="ollama://localhost:11434/qwen2.5:7b",
    symmetric=True,
)
```

[🔬 DeepEval Integration Details](../integrations.md)

---

## 4. OpenTelemetry Tracing & Telemetry Sinks

**Problem & deployment**: Gives SRE and ML platform teams per-layer latency histograms, escalation rates, and audit logs safe for PII (personally identifiable information) (`omit_candidate` / `hash_only` / `redact_pii`) in Datadog, Jaeger, Grafana Tempo, or rotating local JSONL files.

### Copy-Paste Recipe

```python
from logitgate import Cascade, LogitGate, OtelSpanSink, JsonlTelemetrySink

# 1. OpenTelemetry Distributed Tracing Bridge
otel_sink = OtelSpanSink(
    span_prefix="service.guardrail",
    privacy="omit_candidate",  # "full" | "hash_only" | "redact_pii" | "omit_candidate"
    record_layer_events=True,
)

# 2. Append-Only Local Audit Sink (JSONL)
jsonl_sink = JsonlTelemetrySink(
    path=".logitgate/telemetry.jsonl",
    privacy="hash_only",
    max_bytes=10 * 1024 * 1024,  # Auto-rotate at 10 MB
    backup_count=3,
)

# Wire sink into gate or cascade
gate = LogitGate(
    "Verify user input safety",
    telemetry_sink=otel_sink,
)
```

### Privacy Modes

- `privacy="full"`: Records candidate text (bounded by `max_candidate_chars`), rubric, verdict reason, and per-layer scalar log-odds/metadata (not raw full prompts).
- `privacy="hash_only"`: Sets `candidate=None` while retaining its SHA-256 digest (`candidate_sha256`), character length, reason, and trace metadata.
- `privacy="redact_pii"`: Scrubs emails, Bearer/API tokens, US SSNs, and checksum-valid (Luhn) payment card numbers across candidate text, verdict reasons, and diagnostic reports while keeping `candidate_sha256` and `candidate_length`.
- `privacy="omit_candidate"`: Omits candidate text and scrubs generative critique diagnostics, recording only hashes, lengths, timestamps, latencies, triggered layers, and scalar probabilities.

[🔬 Systems Architecture & Telemetry Sinks](../architecture.md)

---

## 5. CLI Quick-Reference

**Problem & deployment**: Runs ad-hoc rubric checks, failure triage, split-conformal calibration, stream guarding, and hardware sweeps directly from shell scripts and CI jobs without writing Python boilerplate.

| Command | Primary Use Case | Key Options | Example Invocations |
| :--- | :--- | :--- | :--- |
| **`eval`** | Evaluate a candidate string or file | `--rubric`, `--candidate`, `--file`, `--threshold` | `logitgate eval --rubric "Is this safe?" --candidate "input text"` |
| **`explain`** | Diagnose a failure snapshot | `snapshot_path`, `--model`, `--max-tokens`, `--offline` | `logitgate explain .logitgate/failures/latest.json` |
| **`calibrate`** | Fit conformal thresholds | `--dataset`, `--alpha`, `--backend`, `-o` | `logitgate calibrate --dataset holdout.jsonl --alpha 0.05` |
| **`rank`** | Rank competing completions | `--rubric`, `--file`, `--candidate`, `--mode` | `logitgate rank --rubric "Pick best" --file opt1.txt --file opt2.txt` |
| **`stream`** | Guard streaming generation | `--rubric`, `--delivery`, `--stride-chars` | `python gen.py \| logitgate stream --rubric "Safe?"` |
| **`benchmark`** | Run hardware benchmarks | `--dataset`, `--modes`, `--backend`, `--judge-backend` | `logitgate benchmark --dataset benchmarks/data/cal_distill_80.json` |
| **`telemetry`** | Summarize, export, or audit drift | `summary`, `export`, `recalibrate`, `drift` | `logitgate telemetry summary .logitgate/telemetry.jsonl` |
| **`distill`** | Compile student bundle or LoRA | `compile`, `train-lora`, `--telemetry`, `--model`, `-o` | `logitgate distill compile --telemetry .logitgate/telemetry.jsonl --model qwen2.5:3b -o bundle.json` |

---

[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](../eli5/how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)
