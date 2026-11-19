# LogitGate (`logitgate`)

[![PyPI version](https://img.shields.io/pypi/v/logitgate.svg?color=blue)](https://pypi.org/project/logitgate/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![PEP 561 Typed](https://img.shields.io/badge/typing-PEP%20561-blueviolet.svg)](src/logitgate/py.typed)
[![Typechecked: mypy](https://img.shields.io/badge/typechecked-mypy-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitLab CI](https://gitlab.com/jasonbdly/logitgate/badges/main/pipeline.svg)](https://gitlab.com/jasonbdly/logitgate/-/pipelines)

**Fast, calibrated first-token logit evaluation and 3-layer cascaded routing for LLMs.**

LogitGate evaluates LLM outputs, Retrieval-Augmented Generation (RAG) groundedness, and agent tool calls by reading **first-token log-probabilities** (`max_tokens=1`, `0` generated reasoning tokens) instead of waiting seconds for a generative Chain-of-Thought (CoT) critique. On GPU, Layer-1 gating resolves clear passes and failures in **16 to 110 ms** (**4.6x to 17.0x** faster than generative judges) and escalates only uncertain edge cases.

---

## Why First-Token Logits Instead of Text Generation?

When grading an LLM output or RAG reply against a rubric, three approaches trade off speed against calibration:

1. **Generative CoT Judge (`1.5-3.0 s`)**: Generates 150-300 reasoning tokens sequentially before emitting `PASS` or `FAIL`, making per-turn checks and large CI suites costly.
2. **Naive `"Answer YES or NO"` Prompt**: Emits 1 text token (`"YES"`), hiding whether the model was `51%` guessing or `99%` certain, skewing toward favorite tokens and positions (`"YES"` / `"A"`), splitting mass across `"Yes"` / `" Yes"` / `"yes"`, and remaining vulnerable to prompt injection inside the candidate.
3. **LogitGate (`16 to 110 ms` on GPU, `0` generated tokens)**: Runs **parallel prompt prefill only** over a nonce-isolated prompt with `<think>` disabled, reads raw first-token `top_logprobs` across `A/B` variants in two swapped passes (`A=Pass, B=Fail` and `A=Fail, B=Pass`, sharing the GPU KV-cache), and cancels position and token-prior bias algebraically to produce a calibrated `0.0-1.0` probability.

For an illustrated step-by-step walkthrough, see [docs/eli5/how_logitgate_works.md](docs/eli5/how_logitgate_works.md).

---

## How the 3-Layer Cascade Works

```text
            Candidate Artifact
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│  Layer 0: ASTGate / ToolCallGate                       ├──► [FAIL: syntax / schema / path, <0.1ms]
└──────────────────────────┬─────────────────────────────┘
                           │ Passed
                           ▼
┌────────────────────────────────────────────────────────┐
│  Layer 1: LogitGate / ChoiceGate / ScoreGate / RankGate ├──► [PASS: P >= 0.85, 16-110ms, 0 tokens]
└──────────────────────────┬─────────────────────────────┤
                           │ Ambiguous                   └──► [FAIL: P <= 0.20, 16-110ms, 0 tokens]
                           │ (0.20 < P < 0.85)
                           ▼
┌────────────────────────────────────────────────────────┐
│  Layer 2: JudgeGate                                    ├──► [Step-by-Step Diagnostic Critique]
└────────────────────────────────────────────────────────┘
```

1. **Layer 0 (`ASTGate` / `ToolCallGate`, `<0.1 ms` on CPU)**: Deterministic checks run first for Python AST syntax, forbidden imports, JSON/Pydantic schemas, regex rules, and tool-call parameters (path traversal, SSRF (internal-network probes), allowed tools), failing malformed inputs with 0 LLM calls.
2. **Layer 1 (`LogitGate` / `ChoiceGate` / `ScoreGate` / `RankGate`, `16 to 110 ms` on GPU, `0` generated tokens)**: Combines 2-pass symmetric debiasing, byte-pair encoding (BPE) token-variant summing, and split-conformal thresholds to fast-path clear passes (`P >= 0.85`) and failures (`P <= 0.20`).
3. **Layer 2 (`JudgeGate`, `1.5-5.0 s`)**: Runs generative critique only for the ambiguous band (`0.20 < P < 0.85`), preserving full diagnostics on borderline cases while skipping 70% to 90% of generative judge calls.

For full architectural details, see [docs/architecture.md](docs/architecture.md) and [docs/README.md](docs/README.md).

---

## Installation

Core package (requires only `httpx`, `pydantic`, and `typing-extensions`):

```bash
pip install logitgate
```

Optional extras:

```bash
pip install "logitgate[pytest]"      # pytest @check plugin & failure snapshots
pip install "logitgate[llamacpp]"    # In-process llama-cpp-python binding
pip install "logitgate[langgraph]"   # LangGraph conditional edge routing
pip install "logitgate[deepeval]"    # DeepEval System-1 metrics
pip install "logitgate[recurrent]"   # Recurrent layer looping (PyTorch)
pip install "logitgate[distill]"     # Student distillation and Parameter-Efficient Fine-Tuning (PEFT) LoRA
pip install "logitgate[otel]"        # OpenTelemetry span bridge (OtelSpanSink)
pip install "logitgate[all]"         # All optional integrations
```

Compatibility and dev extras: `logitgate[openai]` is a no-op alias (OpenAI-compatible HTTP support is built into core), and `logitgate[test]` adds `hypothesis` alongside `pytest`.

---

## Pick a Backend

Configure `LOGITGATE_BACKEND` (or pass `backend=...` to any gate) using a provider URI:

```bash
# Ollama local inference (default model: qwen2.5:7b)
export LOGITGATE_BACKEND="ollama://localhost:11434/qwen2.5:7b"

# vLLM / SGLang OpenAI-compatible endpoint
export LOGITGATE_BACKEND="vllm://localhost:8000/v1/Qwen/Qwen2.5-7B-Instruct"

# OpenAI API endpoint
export LOGITGATE_BACKEND="openai://gpt-4o-mini"

# Google Gemini endpoint
export LOGITGATE_BACKEND="gemini://gemini-2.5-flash"

# llama.cpp server
export LOGITGATE_BACKEND="llamacpp://localhost:8080"

# Recurrent layer-looping PyTorch backend (requires logitgate[recurrent])
export LOGITGATE_BACKEND="recurrent://Qwen/Qwen2.5-1.5B-Instruct?num_loops=2"

# Offline stub for CPU-only CI plumbing tests
export LOGITGATE_BACKEND="stub://default"
```

For authentication, timeout, retry configuration, and no-logprob sidecars, see [docs/quick/backends_and_sidecars.md](docs/quick/backends_and_sidecars.md).

---

## Quickstart

### 1. Pytest Assertion (`@check`)

```python
from logitgate import check

@check("The reply apologizes for the billing error and confirms a refund")
def test_double_charge_support_reply():
    return "We apologize for the billing error on invoice #902. A full refund has been issued to your card."
```

Run tests with optional failure snapshots:

```bash
pytest --logitgate-backend ollama://localhost:11434/qwen2.5:7b --logitgate-save-failures
```

### 2. 3-Layer Cascaded Pipeline (`ASTGate >> LogitGate >> JudgeGate`)

```python
from logitgate import ASTGate, LogitGate, JudgeGate

with (
    ASTGate(syntax=True, forbidden_imports=["os", "subprocess"])
    >> LogitGate(
        "Does the helper use parameterized SQL queries?",
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    >> JudgeGate(trigger="on_ambiguous_only")
) as cascade:
    candidate_code = """
def get_user_by_id(cursor, user_id: int):
    cursor.execute("SELECT id, email FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
"""
    verdict = cascade.evaluate(candidate_code)
    assert verdict.passed
```

### 3. Grounded RAG & Tool Call Safety

```python
from logitgate import LogitGate, ToolCallGate

# Trusted retrieved documents (prefilled once in KV-cache) vs. untrusted LLM draft
retrieved_policy_chunks = [
    "Standard tier: 14-day refund window.",
    "Enterprise tier: 30-day refund window.",
]
user_question = "Can I get a refund on my Standard plan after 20 days?"
faithful_llm_reply = (
    "Standard plans only qualify for refunds within 14 days; "
    "the 30-day window applies only to Enterprise."
)

# context= holds trusted retrieved docs (prefilled once in KV-cache); candidate= is the untrusted draft checked for hallucinations
rag_gate = LogitGate("Is the generated response faithful to the retrieved policy?")
rag_verdict = rag_gate.evaluate(
    candidate=faithful_llm_reply,
    context=retrieved_policy_chunks,
    query=user_question,
)
assert rag_verdict.passed
# Note: A hallucinated reply such as "Yes, 30 days" fails against a live backend (P <= 0.20, passed=False).

# Pre-execution agent tool-call guardrail (<0.1ms Stage-0 sink check + Stage-1 intent check)
tool_gate = ToolCallGate(
    allowed_tools=["read_invoice", "list_orders"],
    allowed_roots=["/var/data/invoices"],
    check_intent=True,
    user_intent="Read invoice #902 for the customer",
)
tool_verdict = tool_gate.evaluate(
    {"name": "read_invoice", "arguments": {"path": "/var/data/invoices/inv_902.pdf"}}
)
assert tool_verdict.passed
# Path traversal ("../../etc/passwd") or SSRF ("http://169.254.169.254") fails in <0.1ms at Stage 0.
```

---

## Common vs. Power-User Feature Map (3-Tier Docs)

LogitGate organizes documentation across three depth tiers: **Quick Notes** (copy-paste recipes), **Deep Notes** (systems architecture, mathematical proofs, security isolation), and **Visual Guide** (illustrated step-by-step walkthroughs). Explore the portal in [docs/README.md](docs/README.md).

### Table 1: Common Everyday Features (90% Path)

| Feature & Workflow | Use Case | ⚡ Quick Notes | 🔬 Deep Notes | 🧭 Visual Guide |
| :--- | :--- | :---: | :---: | :---: |
| **3-Layer Cascade** | `ASTGate >> LogitGate >> JudgeGate` routing | [Recipes](docs/quick/quickstart_recipes.md) | [Architecture](docs/architecture.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Grounded RAG** | Faithfulness verification with prefix caching | [RAG Notes](docs/quick/rag_and_agents.md) | [Math Foundations](docs/math_and_debiasing.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **ToolCallGate** | Pre-execution Model Context Protocol (MCP) and agent tool validation | [Agent Notes](docs/quick/rag_and_agents.md) | [Security & Adversarial](docs/deep/security_and_adversarial.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Choice & Score Gates** | Categorical routing, 1-to-5 ordinal (Likert) scoring, ranking | [Recipes](docs/quick/quickstart_recipes.md) | [Math Foundations](docs/math_and_debiasing.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **StreamGuard** | Mid-stream token abort for streaming Server-Sent Events (SSE) | [Streaming](docs/quick/rag_and_agents.md) | [Architecture](docs/architecture.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **No-Logprob Sidecars** | Claude / o1 fallback via `GeminiBackend` or local GPU | [Backends](docs/quick/backends_and_sidecars.md) | [Architecture](docs/architecture.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Calibration & Caching** | Conformal risk control and SQLite caching | [Recipes](docs/quick/quickstart_recipes.md) | [Math Foundations](docs/math_and_debiasing.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Frameworks & OTel** | Pytest `@check`, LangGraph, DeepEval, OpenTelemetry | [Integrations](docs/quick/integrations_and_otel.md) | [Integrations](docs/integrations.md) · [Use Cases](docs/use_cases.md) | [How It Works](docs/eli5/how_logitgate_works.md) |

### Table 2: Power-User & Research Features (`docs/advanced/`)

| Power-User Feature | Description & Capability | Research Guide |
| :--- | :--- | :---: |
| **Tier-A Distillation Bundle** | Zero-GPU student anchor calibration and meta-temperature mapping | [Distillation & LoRA](docs/advanced/distillation_and_lora.md) |
| **Tier-B PEFT LoRA Compiler** | Single-token distillation training for compact student models | [Distillation & LoRA](docs/advanced/distillation_and_lora.md) |
| **Online Drift Monitoring** | Rolling Kolmogorov-Smirnov detection with Gibbs-Candès adaptive conformal step | [Drift & Adaptive Conformal](docs/advanced/drift_and_adaptive_conformal.md) |
| **Elastic Recurrent Looping** | Layer-looping recurrent depth scaling (PyTorch backend) | [Recurrent & Hardware](docs/advanced/recurrent_and_hardware.md) |

---

## Headline Empirical Benchmarks

Summary measurements from live evaluation sweeps with 95% Wilson confidence intervals (see [docs/benchmarks.md](docs/benchmarks.md) for complete hardware configurations, datasets, and reproduction commands):

| Benchmark Suite | Evaluated Task | N | Primary Metric (95% Wilson CI) | Headline Result |
| :--- | :--- | :---: | :---: | :---: |
| **`princeton-nlp/LLMBar`** | Adversarial neighbor pairwise | 60 | 58.3% debiased acc `[45.7%, 69.9%]` | **100.0%** swap consistency (+25.0 pp vs 1-pass) |
| **`ScalerLab/JudgeBench`** | Math, coding, reasoning pairs | 60 | 61.7% debiased acc `[49.0%, 72.9%]` | **93.3%** swap consistency (+21.6 pp vs 1-pass) |
| **`rag_groundedness_80`** | RAG retrieval faithfulness | 80 | 98.8% cascade acc `[93.3%, 99.8%]` | **2.51x** speedup via shared KV-cache prefix |
| **`agent_mcp_tool_safety_80`** | Agent tool call validation | 80 | 98.8% cascade acc `[93.3%, 99.8%]` | **70%** caught at Layer 0 in **0.08 ms** |
| **`ood_generalization_96`** | Out-of-domain conformal cascade (`gemma4:e4b`) | 96 | 96.9% cascade acc `[91.2%, 98.9%]` | **1.04%** L1 risk (**1.63x** speedup) |
| **`ood_generalization_96`** | No-logprob sidecar routing (`llama3.2:3b`) | 96 | 86.5% cascade acc `[78.2%, 91.9%]` | **5.37x** speedup via sidecar routing |

---

## CLI Overview

LogitGate includes 8 dedicated CLI subcommands:

```bash
# Evaluate a candidate string against a rubric
logitgate eval --rubric "Is this SQL query safe?" --candidate "SELECT * FROM users WHERE id = :id"

# Explain a failed check using Layer 2 diagnostic critique
logitgate explain .logitgate/failures/latest.json

# Calibrate split-conformal thresholds on a holdout dataset
logitgate calibrate --dataset holdout.jsonl --alpha 0.05

# Rank multiple candidate files against a rubric
logitgate rank --rubric "Pick the best summary" --file summary_a.txt --file summary_b.txt --file summary_c.txt

# Stream an LLM completion through a mid-stream guardrail for personally identifiable information (PII)
python generate.py | logitgate stream --rubric "Does the text avoid leaking PII?"

# Measure latency, token usage, and accuracy across judge, logit, and cascade modes
logitgate benchmark --dataset benchmarks/data/cal_distill_80.json

# Summarize pass, escalation, discrepancy, and latency metrics from JSONL telemetry
logitgate telemetry summary .logitgate/telemetry.jsonl

# Compile a Tier-A zero-GPU student distillation bundle from telemetry
logitgate distill compile --telemetry .logitgate/telemetry.jsonl --model qwen2.5:3b -o bundle.json
```

---

## BibTeX Citation

```bibtex
@software{bradley2026logitgate,
  author = {Bradley, Jason},
  title = {LogitGate: Fast, Calibrated First-Token Logit Evaluation and Cascaded Routing for LLMs},
  year = {2026},
  version = {0.16.2},
  url = {https://github.com/jasonbdly/LogitGate},
  license = {Apache-2.0}
}
```

---

## Contributing & License

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local development setup and guidelines.

LogitGate is licensed under the [Apache License 2.0](LICENSE).
