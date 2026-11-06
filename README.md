# LogitGate (`logitgate`)

[![PyPI version](https://img.shields.io/pypi/v/logitgate.svg?color=blue)](https://pypi.org/project/logitgate/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10–3.15](https://img.shields.io/badge/python-3.10--3.15-blue.svg)](https://www.python.org/downloads/)
[![PEP 561 Typed](https://img.shields.io/badge/typing-PEP%20561-blueviolet.svg)](src/logitgate/py.typed)
[![Typechecked: mypy](https://img.shields.io/badge/typechecked-mypy-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitLab CI](https://gitlab.com/jasonbdly/logitgate/badges/main/pipeline.svg)](https://gitlab.com/jasonbdly/logitgate/-/pipelines)

**Fast, calibrated first-token logit evaluation and 3-layer cascaded routing for LLMs.**

LLM-as-a-judge asks a model to write a critique and then a verdict. That costs a full autoregressive generation per check: 1.5–6 s and a few hundred tokens on our hardware. LogitGate reads the verdict straight from the **first-token log-probabilities** (`max_tokens=1`, 0 generated tokens), cancels the model's label-order prior with a forward + reverse pass, and only falls back to a generative judge when that read is genuinely ambiguous.

Measured on the repo's 24-case multi-domain suite (RTX 4070, Ollama, [full methodology](docs/benchmarks.md)): **36–41 ms** single-pass and **90–113 ms** debiased per check. End-to-end, the cascade ran **10.6x–17x** faster than a `think=False` generative judge, at 23/24 accuracy versus the judge's 23–24/24. On pure CPU the gap shrinks to 1.4x, because prefill dominates there. Don't take our word for it: [`logitgate benchmark`](#reproduce-the-numbers-on-your-hardware) reruns the comparison on your model in about a minute.

---

## How It Works in 60 Seconds

When evaluating whether an LLM output satisfies a rubric, traditional frameworks prompt a model to generate a 200–500 word critique before outputting `PASS` or `FAIL`. That takes **seconds per check**, consumes hundreds of tokens, and is sensitive to label order and position.

`LogitGate` uses a **3-layer cascaded pipeline**:

```
            Candidate Artifact
                    │
                    ▼
┌──────────────────────────────────────┐
│  Layer 0: ASTGate / ContractGate     ├──► [FAIL: syntax/schema, <0.1ms]
└──────────────────┬───────────────────┘
                   │ Passed
                   ▼
┌──────────────────────────────────────┐
│  Layer 1: LogitGate                  ├──► [PASS: P >= 0.85, ~40–110ms]
└──────────────────┬───────────────────┤
                   │ Ambiguous         └──► [FAIL: P <= 0.20, ~40–110ms]
                   │ (0.20 < P < 0.85)
                   ▼
┌──────────────────────────────────────┐
│  Layer 2: JudgeGate                  ├──► [Actionable Diagnostic Report]
└──────────────────────────────────────┘
```

1. **Layer 0 (`ASTGate` / `ContractGate`)**: Deterministic checks in **<0.1 ms** on CPU (Python AST syntax, JSON/Pydantic schemas, regex patterns, acyclic DAG topology). Failures reject immediately with zero LLM overhead.
2. **Layer 1 (`LogitGate` / `ChoiceGate` / `ScoreGate`)**: Reads **first-token log-probabilities (`max_tokens=1`)** in **~36–41 ms on GPU** (~170 ms on CPU). To eliminate pretraining label bias, it averages forward and reverse passes in log-odds space ($\bar{L} = \frac{L_1 + L_2}{2T}$).
3. **Layer 2 (`JudgeGate`)**: Only when Layer 1 confidence is genuinely ambiguous (between 20% and 85%) or requires failure diagnostics does the pipeline escalate to generating a full text critique.

---

## Installation

```bash
# Core package: only pydantic + httpx
pip install logitgate

# With optional framework integrations:
pip install "logitgate[openai]"      # OpenAI, vLLM, and compatible endpoints
pip install "logitgate[llamacpp]"    # In-process llama.cpp (zero background daemon)
pip install "logitgate[langgraph]"   # LangGraph conditional edge routing
pip install "logitgate[deepeval]"    # DeepEval System-1 metrics
pip install "logitgate[recurrent]"   # Elastic recurrent looping (NVIDIA Dory-4B)
pip install "logitgate[all]"         # All integrations
```

---

## Pick a Backend (10 seconds)

LogitGate needs an endpoint that returns **token log-probabilities**. Anything below works; a gate's `backend=` argument overrides the global default.

```bash
export LOGITGATE_BACKEND="ollama://localhost:11434/qwen2.5:7b"   # Ollama
export LOGITGATE_BACKEND="vllm://localhost:8000/v1"              # vLLM / any OpenAI-compatible server
export LOGITGATE_BACKEND="llamacpp://localhost:8080"             # llama-server
export LOGITGATE_BACKEND="stub://"                               # offline stub for CI plumbing (fixed P=0.95)
```

With nothing configured, LogitGate falls back to the offline stub and emits a one-time `StubBackendWarning`, so a fake `P=0.95` never passes for a real result. Errors say what to do next instead of dumping an `httpx` traceback:

```text
logitgate.exceptions.BackendUnavailableError: Cannot connect to Ollama at http://localhost:11434.
Is ollama running? (start it with `ollama serve`) Or set backend='stub://' for offline testing.
```

A missing Ollama model suggests the exact `ollama pull <model>` command, an OpenAI 401 points at `OPENAI_API_KEY`, and a cold-model timeout says so. `BackendUnavailableError` is also a `ConnectionError`, so existing retry logic keeps working.

---

## 30-Second Quickstart

### 1. Single Atomic Rubric (3-Layer Cascaded Fast Path)

```python
from logitgate import ASTGate, LogitGate, JudgeGate

# Compose the 3-layer pipeline using the >> operator
cascade = (
    # Layer 0: Ensure valid Python syntax without LLM calls (<0.1ms)
    ASTGate(syntax=True)
    # Layer 1: First-token debiased logit gate (~40-90ms, 0 CoT tokens)
    >> LogitGate(
        rubric="Does the function calculate Fibonacci numbers using recursion?",
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    # Layer 2: Escalated diagnostic judge (only fires if Layer 1 is ambiguous or failing)
    >> JudgeGate(trigger="on_ambiguous_only")
)

candidate_code = """
def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

verdict = cascade.evaluate(candidate_code)

print(f"Passed: {verdict.passed}")               # True
print(f"Confidence: {verdict.probability:.4f}")   # e.g., 0.9982
print(f"Triggered Layer: {verdict.triggered_layer}")  # LogitGate (bypassed Layer 2!)
print(f"Latency: {verdict.total_latency_ms:.1f}ms")    # ~86ms
print(f"Tokens Generated: {verdict.total_tokens_generated}")  # 0 tokens!
```

### 2. Multi-Rubric Batching (Prefix-Preserved KV-Cache Reuse)

When verifying multiple criteria against a candidate artifact, **do not combine them into a single monolithic compound rubric** (which dilutes logit signals and conflates orthogonal requirements). Instead, evaluate isolated atomic rubrics with `MultiRubricGate`.

The candidate code is anchored at Token 0, sharing 95–99% of prefix tokens in KV-cache across all checks. `MultiRubricGate` enforces weakest-link aggregation: passes only if *every* atomic rubric passes, and records per-rubric traces so diagnostics show exactly which requirement failed:

```python
from logitgate import MultiRubricGate

# Best Practice: Keep rubrics atomic — evaluate multiple criteria via MultiRubricGate
rubrics = [
    "Does the function calculate Fibonacci numbers recursively?",
    "Does the function explicitly handle non-positive inputs (n <= 0)?",
    "Does the function include parameter and return type annotations?",
]

gate = MultiRubricGate(rubrics)

# Evaluates all 3 rubrics in a single batch reusing the shared KV-cache prefix
verdict = gate(candidate_code)
print(f"Passed: {verdict.passed} | {verdict.reason}")

# Or inspect individual per-rubric verdicts:
for rubric, v in zip(rubrics, gate.evaluate_each(candidate_code)):
    status = "PASS" if v.passed else "FAIL"
    print(f"[{status}] P={v.probability:.4f} | {rubric}")
```

---

## Core Gates Quick Reference

| Gate | Layer | Latency | Tokens Generated | Purpose & Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **`ASTGate`** | Layer 0 | **< 0.1 ms** | 0 | Sub-millisecond deterministic checks: Python syntax, forbidden imports, regex, acyclic DAGs, custom callables. |
| **`ContractGate`** | Layer 0 | **< 0.1 ms** | 0 | General-domain contract validation: Pydantic models, JSON Schemas, length bounds, required sections. |
| **`LogitGate`** | Layer 1 | **~40–110 ms** | 0 | Calibrated binary classification from first-token logits using symmetric log-odds debiasing ($\bar{L} = \frac{L_1 + L_2}{2T}$). |
| **`MultiRubricGate`** | Layer 1 | **~50–140 ms** | 0 | Atomic multi-rubric verification sharing KV prefix. Weakest-link aggregation (fails if any criterion fails). |
| **`ChoiceGate`** | Layer 1 | **~40–120 ms** | 0 | Discrete K-ary routing and intent classification across arbitrary string options with permutation averaging. |
| **`ScoreGate`** | Layer 1 | **~50–160 ms** | 0 | Continuous probability-weighted rubric scoring (𝔼[S] = Σ s · P(s)) with inverted scale debiasing. |
| **`JudgeGate`** | Layer 2 | **1.5–5.0 s** | 60–300 | Generative System-2 reasoning judge with reason-then-verdict parsing and self-healing 1-token logit probes. |
| **`Cascade`** | 0 $\to$ 2 | **~40–120 ms\*** | 0\* | Chained pipeline (`gate_a >> gate_b`). Fast passes short-circuit Layer 2 entirely. |

*\*Typical latency on non-ambiguous inputs.*

---

## Where LogitGate Fits (and Where It Doesn't)

| Tool | What it does | How LogitGate relates |
| :--- | :--- | :--- |
| **DeepEval / G-Eval** | Offline evaluation suites and CI scoring; G-Eval weights score tokens by probability. | Complementary. LogitGate is the *inline* path: a sub-100 ms check you can put between agent turns. It ships a DeepEval integration (`LogitGateMetric`) for suites. |
| **RouteLLM** | Picks *which model generates* a response, before generation. | Different stage. LogitGate judges the output *after* generation. `ChoiceGate` can route, but on content, not on model cost. |
| **Flow-Judge, Lynx, Prometheus 2** | Dedicated fine-tuned judge models that reason autoregressively. | LogitGate is non-autoregressive and model-agnostic. It projects a verdict from the first-token logits of a model you already serve, with no extra VRAM. A fine-tuned judge works fine as the Layer-2 `JudgeGate` backend. |

**When not to use it:**

- **No logprobs, no LogitGate.** APIs that don't expose `top_logprobs` (e.g. Anthropic) can't host Layer 1. OpenAI caps `top_logprobs` at 20, which is enough for binary and small-K gates.
- **Multi-step reasoning rubrics** ("is this 12-step derivation correct?") exceed what a single forward pass can judge. Accuracy drops on our reasoning-depth suite ([falsification boundaries](docs/benchmarks.md)). Let those escalate to `JudgeGate`, or decompose them into atomic rubrics.
- **Tiny models on CPU.** 1–1.5B models are fast but reach 58–63% cascade accuracy on our multi-domain suite. Measure before you trust them.

---

## Reproduce the Numbers on Your Hardware

```bash
logitgate benchmark --backend ollama://localhost:11434 --model qwen2.5:7b
logitgate benchmark --backend vllm://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
    --dataset benchmarks/data/multi_domain_24.json --repeat 3 --json > results.json
```

The command runs the same cases through three configurations and reports p50/p95 latency, generated tokens, LLM calls per case, accuracy, and the cascade's fast-path rate:

1. A generative `JudgeGate` alone (`think=False` by default, so the baseline is a *fast* judge)
2. A single `LogitGate`
3. `LogitGate >> JudgeGate` (`on_ambiguous_only`)

Modes run as separate passes after an untimed warm-up, so model load time isn't charged to whichever mode runs first. `--json` records the LogitGate version, backend, model, platform, and a dataset hash, so a result can be traced to exactly what produced it. With the debiased default, LogitGate makes **two** forward passes per check, and the report counts both.

---

## Custom Gates & Routing Traits

`Cascade` never checks concrete classes. It routes on three traits declared on every gate, so wrappers, decorators, and your own gates get the same short-circuiting as the built-ins:

| Trait | Meaning | Built-ins |
| :--- | :--- | :--- |
| `layer_tier` | `0` deterministic (failures are final), `1` single forward pass (failures may escalate), `2` generative | `ASTGate`=0, `LogitGate`/`ChoiceGate`/`ScoreGate`=1, `JudgeGate`=2 |
| `can_fast_path` | A PASSED verdict skips the next generative gate | `False` for `ASTGate` (a valid schema says nothing about semantics) |
| `escalation_policy` | When a tier-2 gate runs: `on_ambiguous_or_fail`, `on_ambiguous_only`, `always` | `JudgeGate(trigger=...)` |

```python
from logitgate import BaseGate, DecisionStatus, JudgeGate, LayerTrace, Verdict

class RegexPIIGate(BaseGate):
    """Deterministic: a hit is a decisive FAIL; a miss proves nothing semantic."""

    name = "RegexPII"
    layer_tier = 0
    can_fast_path = False

    def evaluate(self, candidate, **kwargs):
        leaked = "@" in str(candidate)
        status = DecisionStatus.FAILED if leaked else DecisionStatus.PASSED
        trace = LayerTrace(layer_name=self.name, status=status, latency_ms=0.01)
        return Verdict(passed=not leaked, status=status, triggered_layer=self.name,
                       total_latency_ms=0.01, traces=[trace])

pipeline = RegexPIIGate() >> JudgeGate(rubric="Does the reply avoid leaking personal data?")
print(pipeline.evaluate("Contact jane@example.com").triggered_layer)  # RegexPII; judge never called
```

---

## Benchmark Highlights

Across a 24-item multi-domain suite (RAG groundedness, customer support policy, SQL safety, and Python security), `LogitGate` resolves **95.8%–100.0% of cases at Layers 0 and 1**, avoiding multi-token generation:

| Evaluation Approach | Typical Latency | Generated Tokens | Extra GPU VRAM | Bias Cancellation | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generative LLM Judge** *(G-Eval, MT-Bench)* | 1,500ms – 8,000ms | 100 – 600 CoT tokens | 0 GB (Shared) | ❌ None | 1.0x |
| **Fine-Tuned Judge Models** *(Prometheus 2, Selene)* | 800ms – 3,500ms | 80 – 300 CoT tokens | 14 GB – 48 GB (Dedicated) | ❌ Susceptible to bias | 2.5x |
| **`LogitGate` 3-Layer Cascade** | **36ms – 113ms** | **0 CoT tokens** | **0 GB (Reuses generator)** | **✅ Symmetric Log-Odds** | **10x – 17x** |

### CPU-Only CI/CD Champion: `llama3.2:3b` in Q4_K_M
- **In-Process `llama.cpp`**: Cold-starts in **3.2 seconds** inside your Python process (zero external background daemon), allocating only **2.36 GB RAM**.
- **Performance**: Evaluates 2-pass symmetric debiased gating in **815 ms** on CPU with **$\rho = 0.8487$** continuous scoring correlation.
- **Thread Scaling Law**: CPU execution saturates at **2 threads** (112 ms forward pass) due to memory bandwidth limits.

---

## In-Depth Documentation Index

For complete research data, mathematical derivations, and production recipes, explore our modular documentation:

- 📊 **[Empirical Benchmarks & Falsification Studies](docs/benchmarks.md)**:
  - 24-item multi-domain cascade efficiency & token savings
  - Fair pure-CPU CI/CD runner benchmarks (10-item CI suite)
  - Compact models on CPU (1B–3B vs 4B baseline)
  - In-process `llama.cpp` vs Ollama daemon & thread scaling sweeps
  - Hardened 32-item adversarial benchmark suite ($N=5$)
  - Falsification boundaries: reasoning depth, context length (16k), and context window squeezing
  - Elastic recurrent layer-looping (NVIDIA Dory-4B, RTX 5090 vs RTX 4070)
- 🏗️ **[3-Layer Architecture & Deployment Topologies](docs/architecture.md)**:
  - Execution mechanics, lookahead policy, and monotonic failure preservation
  - System requirements across 0-GPU, Consumer GPU, and Enterprise hardware
  - 3 deployment topologies to prevent KV-cache eviction and thrashing
- 📐 **[Mathematical Foundations & Symmetric Debiasing](docs/math_and_debiasing.md)**:
  - Analytical proof of prior token bias cancellation in log-odds space
  - Probability averaging vs. log-odds space contrast
  - Prefix-preserving attention inversion & missing-token bounded floor
  - Temperature calibration ($T$) and Brier score minimization
- 💡 **[Production Use Cases & Best Practices](docs/use_cases.md)**:
  - CI/CD final merge gates vs local inner loop
  - Reproducing CI failures locally via `VerdictSnapshot` and `logitgate explain`
  - RAG hallucination guardrails & LangGraph agent intent routing
  - Candidate-prefix isolated batching (`evaluate_batch`) vs monolithic prompt anti-patterns
- 🔌 **[Framework Integrations](docs/integrations.md)**:
  - Pytest `@check` test decorator
  - DeepEval `LogitGateSystemOneModel` and `LogitGateMetric`
  - LangGraph conditional edge routing (`as_conditional_edge`)

---

## Common Recipes

### Multi-Rubric Batch Evaluation (`MultiRubricGate`)
Evaluates multiple rubrics against a single candidate artifact while anchoring the candidate at Token 0 to share 95–99% of prefix tokens in KV-cache:

```python
from logitgate import MultiRubricGate

rubrics = [
    "Does the code include full type annotations?",
    "Does the code avoid mutable default arguments?",
    "Does the code implement proper error handling?",
]

gate = MultiRubricGate(rubrics)
candidate_code = "def fetch(url: str, timeout: int = 10) -> str: pass"
verdicts = gate.evaluate_each(candidate_code)

for rubric, v in zip(rubrics, verdicts):
    print(f"[{'PASS' if v.passed else 'FAIL'}] P={v.probability:.2f} | {rubric}")
```

### Discrete Multi-Way Routing (`ChoiceGate`)
Routes requests across $K$ options in ~40–90 ms without generating JSON:

```python
from logitgate import ChoiceGate

router = ChoiceGate(
    question="Route this incoming user request to the appropriate specialist:",
    options={
        "A": "Database and SQL query optimizations",
        "B": "Authentication and user permissions",
        "C": "Billing, invoices, and payment processing",
    },
)

verdict = router.evaluate("My credit card was charged twice for invoice #902.")
print(f"Selected route: {verdict.choice}")  # 'C'
print(f"Probabilities: {verdict.choice_probabilities}")
```

### Continuous Rubric Scoring (`ScoreGate`)
Calculates a smooth expected score (𝔼[S] ∈ [1.0, 5.0]) using first-token softmax weights:

```python
from logitgate import ScoreGate

scorer = ScoreGate(
    rubric="Rate the architectural code quality from 1 (disastrous) to 5 (flawless):",
    min_score=1,
    max_score=5,
    threshold=3.5,
    symmetric_debiasing=True,  # Evaluates 1->5 and 5->1 to eliminate scale anchoring
)

verdict = scorer.evaluate("def add(a, b): return a + b")
print(f"Expected Score: {verdict.score:.2f} / 5.00")
```

---

## Command-Line Interface (CLI)

```bash
# Evaluate candidate text directly
logitgate eval --rubric "Must not contain shell commands" --candidate "echo 'hello'"

# Evaluate candidate from a file with Layer 0 AST check and JSON output
logitgate eval --rubric "Must be valid Python" --file script.py --ast-syntax --json

# Pipe candidate directly via standard input
cat schema.json | logitgate eval --rubric "Must be valid JSON schema"

# Replay and explain a frozen failure snapshot from CI
logitgate explain .logitgate/failures/latest.json

# Inspect margins, entropy, and label-order bias without calling any model
logitgate explain --offline

# Judge vs LogitGate vs cascade latency/cost on your backend
logitgate benchmark --backend "ollama://localhost:11434" --model "qwen2.5:7b"

# Auto-calibrate model temperature on micro-dataset
logitgate calibrate --backend "ollama://localhost:11434" --model "qwen2.5:14b"
```

Exit codes: `0` passed / ok, `1` verdict failed or bad input, `2` backend unreachable or misconfigured.

---

## BibTeX Citation

If you use LogitGate in your research or production systems, please cite it as follows:

```bibtex
@software{logitgate2026,
  author = {Jason Bradley},
  title = {LogitGate: Calibrated First-Token Logit Evaluation and Cascaded Routing for LLMs},
  year = {2026},
  version = {0.3.0},
  url = {https://github.com/jasonbdly/LogitGate},
  license = {Apache-2.0}
}
```

---

## Contributing & License

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for local development setup and testing procedures.

LogitGate is licensed under the [Apache License 2.0](LICENSE).
