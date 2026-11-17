# LogitGate (`logitgate`)

[![PyPI version](https://img.shields.io/pypi/v/logitgate.svg?color=blue)](https://pypi.org/project/logitgate/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![PEP 561 Typed](https://img.shields.io/badge/typing-PEP%20561-blueviolet.svg)](src/logitgate/py.typed)
[![Typechecked: mypy](https://img.shields.io/badge/typechecked-mypy-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitLab CI](https://gitlab.com/jasonbdly/logitgate/badges/main/pipeline.svg)](https://gitlab.com/jasonbdly/logitgate/-/pipelines)

**Fast, calibrated first-token logit evaluation and 3-layer cascaded routing for LLMs.**

Standard LLM-as-a-Judge evaluations prompt a generative model to write a 200 to 400 word critique before emitting `PASS` or `FAIL`. Generating those tokens costs 1.5 to 6 seconds per assertion on local hardware. Conversely, forcing naive `max_tokens=1` or boolean structured outputs produces severe position and affirmative token bias: off-the-shelf models strongly prefer letter "A" and positive tokens, while greedy decoding collapses a 51% coin flip and 99% certainty into the identical boolean `True`, destroying the uncertainty signal needed to escalate difficult edge cases.

LogitGate extracts debiased, calibrated probabilities from **first-token log-probabilities** (`max_tokens=1, temperature=0.0`, 0 generated critique tokens). It neutralizes label and token-order priors with an analytical 2-pass log-odds swap, and uses split-conformal calibration to escalate only borderline inputs to a generative judge.

Across multi-domain benchmarks, LogitGate evaluates inputs in **16 to 43 ms** (single-pass) and **32 to 101 ms** (debiased), delivering **4.6x to 17.0x** end-to-end speedups over generative judges while matching or exceeding evaluation accuracy.

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
│  Layer 1: LogitGate / ChoiceGate / ScoreGate / RankGate ├──► [PASS: P >= 0.85, ~16-100ms, 0 tokens]
└──────────────────────────┬─────────────────────────────┤
                           │ Ambiguous                   └──► [FAIL: P <= 0.20, ~16-100ms, 0 tokens]
                           │ (0.20 < P < 0.85)
                           ▼
┌────────────────────────────────────────────────────────┐
│  Layer 2: JudgeGate                                    ├──► [Step-by-Step Diagnostic Critique]
└────────────────────────────────────────────────────────┘
```

1. **Layer 0 (`ASTGate` / `ToolCallGate`, `<0.1 ms` on CPU)**: Deterministic validation runs first: Python AST syntax, forbidden imports, JSON/Pydantic schemas, regex security checks, and agent tool-call parameters (path traversal, SSRF, allowed tools). Malformed inputs fail immediately with 0 LLM calls.
2. **Layer 1 (`LogitGate` / `ChoiceGate` / `ScoreGate` / `RankGate`, `~16-100 ms` on GPU, `0` generated tokens)**: Evaluates the prompt prefill log-probabilities at `max_tokens=1`. A 2-pass permutation cancels position and token-prior biases analytically in log-odds space, while BPE surface variants are marginalized across token variants.
3. **Layer 2 (`JudgeGate`, `1.5-5.0 s`)**: Generative critique runs only when Layer 1 confidence lands in an ambiguous band (`0.20 < P < 0.85`), preserving full chain-of-thought diagnostics where needed while skipping 70% to 90% of generative judge latency.

For full architectural details, see [docs/architecture.md](docs/architecture.md) and [docs/README.md](docs/README.md).

---

## Installation

Core package (requires only `httpx` and `pydantic`):

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
pip install "logitgate[distill]"     # Student distillation and PEFT LoRA
pip install "logitgate[otel]"        # OpenTelemetry span bridge (OtelSpanSink)
pip install "logitgate[all]"         # All optional integrations
```

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

# Grounded RAG evaluation over retrieved context
rag_gate = LogitGate("Is the generated response faithful to the retrieved passage?")
rag_verdict = rag_gate.evaluate(
    candidate="Account #4401 was registered on June 12, 2021.",
    context=["Order record: account #4401 created 2021-06-12."],
    query="When was account #4401 created?",
)
assert rag_verdict.passed

# Pre-execution agent tool-call guardrail (<0.1ms Layer 0 check)
tool_gate = ToolCallGate(
    allowed_tools=["fetch_profile", "read_docs"],
    allowed_roots=["/var/data"],
)
tool_verdict = tool_gate.evaluate(
    {"name": "fetch_profile", "arguments": {"user_id": 4401}}
)
assert tool_verdict.passed
```

---

## Common vs. Power-User Feature Map (3-Tier Docs)

LogitGate organizes documentation across three depth tiers: **Quick Notes** (zero-fluff copy-paste recipes), **Deep Notes** (systems architecture, mathematical proofs, security isolation), and **Visual ELI5** (intuitive animated walkthroughs). Explore the portal in [docs/README.md](docs/README.md).

### Table 1: Common Everyday Features (90% Path)

| Feature & Workflow | Everyday Use Case | ⚡ Quick Notes | 🔬 Deep Notes | 🌱 Visual ELI5 |
| :--- | :--- | :---: | :---: | :---: |
| **3-Layer Cascade** | `ASTGate >> LogitGate >> JudgeGate` routing | [Recipes](docs/quick/quickstart_recipes.md) | [Architecture](docs/architecture.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Grounded RAG** | Faithfulness verification with prefix caching | [RAG Notes](docs/quick/rag_and_agents.md) | [Math Foundations](docs/math_and_debiasing.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **ToolCallGate** | Pre-execution MCP and agent tool validation | [Agent Notes](docs/quick/rag_and_agents.md) | [Security & Adversarial](docs/deep/security_and_adversarial.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Choice & Score Gates** | Categorical routing, Likert scoring, ranking | [Recipes](docs/quick/quickstart_recipes.md) | [Math Foundations](docs/math_and_debiasing.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **StreamGuard** | Mid-stream token abort for streaming SSE | [Streaming](docs/quick/rag_and_agents.md) | [Architecture](docs/architecture.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **No-Logprob Sidecars** | Claude / o1 fallback via `GeminiBackend` or local GPU | [Backends](docs/quick/backends_and_sidecars.md) | [Architecture](docs/architecture.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Calibration & Caching** | Conformal risk control and SQLite caching | [Recipes](docs/quick/quickstart_recipes.md) | [Math Foundations](docs/math_and_debiasing.md) | [How It Works](docs/eli5/how_logitgate_works.md) |
| **Frameworks & OTel** | Pytest `@check`, LangGraph, DeepEval, OpenTelemetry | [Integrations](docs/quick/integrations_and_otel.md) | [Use Cases](docs/use_cases.md) | [How It Works](docs/eli5/how_logitgate_works.md) |

### Table 2: Power-User & Research Features (`docs/advanced/`)

| Power-User Feature | Description & Capability | Research Guide |
| :--- | :--- | :---: |
| **Tier-A Distillation Bundle** | Zero-GPU student anchor calibration and meta-temperature mapping | [Distillation & LoRA](docs/advanced/distillation_and_lora.md) |
| **Tier-B PEFT LoRA Compiler** | Single-token distillation training for compact student models | [Distillation & LoRA](docs/advanced/distillation_and_lora.md) |
| **Online Drift Monitoring** | Rolling Kolmogorov-Smirnov detection with Gibbs-Candes adaptive conformal step | [Drift & Adaptive Conformal](docs/advanced/drift_and_adaptive_conformal.md) |
| **Elastic Recurrent Looping** | Layer-looping recurrent depth scaling (PyTorch backend) | [Recurrent & Hardware](docs/advanced/recurrent_and_hardware.md) |

---

## Headline Empirical Benchmarks

Summary measurements from live evaluation sweeps (see [docs/benchmarks.md](docs/benchmarks.md) for complete hardware configurations, datasets, and reproduction commands):

| Benchmark Suite | Evaluated Task | N | Primary Metric (95% Wilson CI) | Headline Result |
| :--- | :--- | :---: | :---: | :---: |
| **`princeton-nlp/LLMBar`** | Adversarial neighbor pairwise | 60 | 58.3% debiased acc `[45.7%, 69.9%]` | **100.0%** swap consistency (+25.0 pp vs 1-pass) |
| **`ScalerLab/JudgeBench`** | Math, coding, reasoning pairs | 60 | 61.7% debiased acc `[49.0%, 72.9%]` | **93.3%** swap consistency (+21.6 pp vs 1-pass) |
| **`rag_groundedness_80`** | RAG retrieval faithfulness | 80 | 98.8% cascade acc `[93.3%, 99.8%]` | **2.51x** speedup via shared KV-cache prefix |
| **`agent_mcp_tool_safety_80`** | Agent tool call validation | 80 | 98.8% cascade acc `[93.3%, 99.8%]` | **70%** caught at Layer 0 in **0.08 ms** |
| **`ood_generalization_96`** | Out-of-domain transfer | 96 | 96.9% cascade acc `[91.3%, 99.0%]` | **5.37x** speedup via sidecar routing |

---

## CLI Overview

LogitGate includes 8 dedicated CLI subcommands:

```bash
# Evaluate a candidate string against a rubric
logitgate eval "Is this SQL query safe?" "SELECT * FROM users WHERE id = :id"

# Explain a failed check using Layer 2 diagnostic critique
logitgate explain .logitgate/failures/latest.json

# Calibrate split-conformal thresholds on a holdout dataset
logitgate calibrate --dataset holdout.json --target-alpha 0.05

# Rank multiple candidates against a rubric
logitgate rank "Pick the best summary" summary_a.txt summary_b.txt summary_c.txt

# Stream an LLM completion through a mid-stream safety guardrail
python generate.py | logitgate stream --rubric "Does the text avoid leaking PII?"

# Run multi-model latency and accuracy sweeps
logitgate benchmark --suite multi_domain_24

# Inspect recorded OpenTelemetry and JSONL telemetry records
logitgate telemetry inspect .logitgate/telemetry.jsonl

# Compile a Tier-A zero-GPU student distillation bundle
logitgate distill --dataset pairs.json --student-model qwen2.5:3b --output bundle.json
```

---

## BibTeX Citation

```bibtex
@software{bradley2026logitgate,
  author = {Bradley, Jason},
  title = {LogitGate: Fast, Calibrated First-Token Logit Evaluation and Cascaded Routing for LLMs},
  year = {2026},
  version = {0.16.1},
  url = {https://github.com/jasonbdly/LogitGate},
  license = {Apache-2.0}
}
```

---

## Contributing & License

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local development setup and guidelines.

LogitGate is licensed under the [Apache License 2.0](LICENSE).
