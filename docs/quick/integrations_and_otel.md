# Framework Integrations & OpenTelemetry (⚡ Quick Notes)

Copy-paste integration recipes for `pytest` testing, `LangGraph` agent workflows, `DeepEval` metrics, `OpenTelemetry` tracing, and the complete CLI reference.

---

## 1. Pytest Assertions (`@check`)

Turn any natural language rubric into an automated unit test assertion running in ~16-90 ms without writing manual regexes or parsing multi-token model outputs.

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
<summary>🌱 <b>ELI5 / Why this works</b></summary>

Instead of writing 50 lines of flaky regexes to test model outputs, @check runs a first-token logit check in milliseconds. When a test passes, it costs zero output tokens. If a test fails, you can run `logitgate explain` to generate a full paragraph explaining why it failed.
</details>

[🔬 Framework Integration Guide](../integrations.md) · [🔬 Production CI/CD Gate Patterns](../use_cases.md)

---

## 2. LangGraph Conditional Edge Routing

Routes state between agent nodes based on semantic intent, security verification, or tool safety.

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

Integrate LogitGate directly into DeepEval evaluation pipelines as a fast first-token metric or System-1 model provider.

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

Record evaluation traces, latencies, layer triggers, and confidence probabilities into OpenTelemetry distributed spans or local JSONL audit logs.

### Copy-Paste Recipe

```python
from logitgate import Cascade, LogitGate, OtelSpanSink, JsonlTelemetrySink

# 1. OpenTelemetry Distributed Tracing Bridge
otel_sink = OtelSpanSink(
    span_prefix="service.guardrail",
    privacy="omit_candidate",  # "full" | "hash_candidate" | "omit_candidate"
    record_layer_events=True,
)

# 2. Append-Only Local Audit Sink (JSONL)
jsonl_sink = JsonlTelemetrySink(
    path=".logitgate/telemetry.jsonl",
    privacy="hash_candidate",
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

- `privacy="full"`: Records full prompt, candidate text, and token logprobs in trace metadata.
- `privacy="hash_candidate"`: Records a SHA-256 hash of the candidate text and character length, omitting raw content.
- `privacy="omit_candidate"`: Omits candidate text entirely, recording only timestamps, latencies, triggered layers, and probabilities.

[🔬 Systems Architecture & Telemetry Sinks](../architecture.md)

---

## 5. Complete CLI Quick-Reference

LogitGate includes 8 dedicated CLI subcommands for offline evaluation, calibration, and analysis.

| Command | Primary Use Case | Key Options | Example Invocations |
| :--- | :--- | :--- | :--- |
| **`eval`** | Evaluate a candidate string | `--rubric`, `--backend`, `--threshold` | `logitgate eval "Is this safe?" "input text"` |
| **`explain`** | Diagnose a failure snapshot | `--model`, `--max-tokens` | `logitgate explain .logitgate/failures/latest.json` |
| **`calibrate`** | Fit conformal thresholds | `--dataset`, `--target-alpha` | `logitgate calibrate --dataset holdout.json --target-alpha 0.05` |
| **`rank`** | Rank competing completions | `--rubric`, `--backend` | `logitgate rank "Pick best" opt1.txt opt2.txt` |
| **`stream`** | Guard streaming generation | `--rubric`, `--delivery` | `python gen.py \| logitgate stream --rubric "Safe?"` |
| **`benchmark`** | Run hardware benchmarks | `--suite`, `--backend`, `--judge-backend` | `logitgate benchmark --suite multi_domain_24` |
| **`telemetry`** | Inspect telemetry logs | `--format`, `--filter` | `logitgate telemetry inspect .logitgate/telemetry.jsonl` |
| **`distill`** | Compile student bundle | `--dataset`, `--student-model` | `logitgate distill --dataset pairs.json --student-model qwen2.5:3b` |
