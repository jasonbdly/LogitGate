# LogitGate (`logitgate`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**The sub-50ms zero-token LLM evaluation and decision engine.**

LogitGate replaces slow, expensive, autoregressive LLM-as-a-judge calls with mathematically debiased first-token logit gates and a 3-layer cascaded routing architecture.

---

## Key Features

1. **Sub-50ms Latency & Zero Autoregressive Tokens**: Gating decisions are computed from the very first token logits without generating output tokens.
2. **Mathematically Debiased Log-Odds Space**: Cancels positional and pretraining token priors (e.g., token `'A'` bias) using symmetric label inversion in log-odds space.
3. **Prefix-Cache-Friendly KV Reuse**: Evaluates Pass 1 and Pass 2 against an identical static prefix (`[EVALUATION RUBRIC]` + `[CANDIDATE ARTIFACT]`), achieving >99% KV-cache hit rates on modern inference engines (vLLM, Ollama, SGLang).
4. **Reasoning Model `<think>` Suppression**: Forces reasoning models (DeepSeek-R1, Qwen-2.5-Coder) to bypass chain-of-thought tokens on classification queries, keeping decisions sub-50ms.
5. **3-Layer Cascaded Router**:
   - **Layer 0 (ASTGate)**: ~0ms deterministic Python syntax, acyclic DAG verification, and contract checks.
   - **Layer 1 (LogitGate / ChoiceGate / ScoreGate)**: ~50ms zero-token debiased confidence gate.
   - **Layer 2 (JudgeGate)**: Generative chain-of-thought diagnostics triggered only on ambiguous or failing artifacts.

---

## Mathematical Foundations

### 1. Symmetric Log-Odds Permutation Debiasing

Standard LLMs exhibit severe token priors (e.g. favoring `'A'` over `'B'`) and positional bias. Naive probability averaging fails under strong bias.

Suppose token `'A'` has an extreme logit prior bias $+b$, and the true signal is $s$:

- **Pass 1** ($A=\text{Pass}, B=\text{Fail}$):
  $$L_1 = \text{logit}(A) - \text{logit}(B) = s + b$$

- **Pass 2** ($B=\text{Pass}, A=\text{Fail}$):
  $$L_2 = \text{logit}(B) - \text{logit}(A) = s - b$$

LogitGate combines the forward and reverse passes in **log-odds space**:
$$\bar{L} = \frac{L_1 + L_2}{2 \cdot T} = \frac{(s + b) + (s - b)}{2 \cdot T} = \frac{s}{T}$$

The prior bias $b$ cancels out completely:
$$P(\text{Pass}) = \sigma(\bar{L}) = \frac{1}{1 + e^{-\bar{L}}}$$

#### Empirical Contrast
For a true signal $s = +2.0$ ($\sigma(2.0) \approx 0.8808$) contaminated by an extreme prior bias $b = +5.0$:
- **Naive Probability Averaging**:
  $$P_1 = \sigma(7.0) \approx 0.9991, \quad P_2 = \sigma(-3.0) \approx 0.0474 \implies \bar{P} = 0.5233 \quad (\text{Failed / Coin Flip})$$
- **LogitGate Log-Odds Debiasing**:
  $$\bar{L} = \frac{7.0 + (-3.0)}{2} = 2.0 \implies \sigma(2.0) = \mathbf{0.8808} \quad (\text{Exact True Signal})$$

---

### 2. Prefix-Preserving Attention Inversion

Causal transformer attention requires the candidate artifact to attend to the evaluation rubric. LogitGate places the rubric and candidate document first, isolating the label mapping to the very end:

```
[EVALUATION RUBRIC]
{rubric}

[CANDIDATE ARTIFACT]
{candidate}

[DECISION MAPPING]
A: Complies fully with the rubric.
B: Violates the rubric or has defects.
Respond with a single character (A or B).
Verdict:
```

Because Pass 1 and Pass 2 share the identical prefix, the entire rubric and candidate context are retained in GPU KV-cache, making Pass 2 near-instantaneous.

---

### 3. Missing-Token Bounded Floor

When target tokens (e.g. `'B'`) do not appear in the top-$K$ logprobs returned by an API, assigning arbitrary $-99.0$ cliffs distorts the symmetric mean. LogitGate dynamically computes the theoretical probability ceiling:

$$\text{floor} = \min\left(\min_{i} \text{logprob}_i, \, \ln\max\left(1.0 - \sum_{i} P_i, \, 10^{-6}\right)\right)$$

This bounds missing tokens to the residual probability mass of unobserved tokens.

---

### 4. Temperature Calibration (ECE & Brier Optimization)

LogitGate provides `fit_calibration(samples)` to scale logits via temperature $T$:
$$p_i(T) = \sigma\left(\frac{L_{1, i} + L_{2, i}}{2 \cdot T}\right)$$

Optimal $T^*$ minimizes the Brier Score:
$$\text{BS}(T) = \frac{1}{N}\sum_{i=1}^N (p_i(T) - y_i)^2$$
and lowers Expected Calibration Error (ECE).

---

## 3-Layer Cascaded Router Architecture

```
 Candidate Artifact
         │
         ▼
┌──────────────────┐
│    Layer 0:      │   Syntax / DAG Cycle / Forbidden Import
│    ASTGate       ├─────────────────────────────────────────► [FAIL: 0 tokens, ~0ms]
└────────┬─────────┘
         │ Passed
         ▼
┌──────────────────┐   P >= pass_threshold (>= 0.85)
│    Layer 1:      ├─────────────────────────────────────────► [PASS: 0 tokens, ~50ms]
│   LogitGate      │
└────────┬─────────┘   P <= fail_threshold (<= 0.20)
         │ Ambiguous   └─────────────────────────────────────► [FAIL: 0 tokens, ~50ms]
         ▼
┌──────────────────┐
│    Layer 2:      │   Escalated Chain-of-Thought Critique
│   JudgeGate      ├─────────────────────────────────────────► [Actionable Diagnostic Report]
└──────────────────┘
```

Chaining gates is as simple as:
```python
from logitgate import ASTGate, LogitGate, JudgeGate

pipeline = (
    ASTGate(syntax=True, forbidden_imports=["subprocess", "os.system"])
    >> LogitGate(rubric="Code must be production-ready and fully typed", pass_threshold=0.85)
    >> JudgeGate(trigger="on_ambiguous_or_fail")
)

verdict = pipeline.evaluate(candidate_code)
if verdict:
    print("Artifact passed!")
else:
    print(f"Failed at {verdict.triggered_layer}: {verdict.reason}")
```

---

## Discrete & Continuous Gates

### `ChoiceGate`
Evaluates $K$ discrete options with forward/reverse ordering debiasing:
```python
from logitgate import ChoiceGate

gate = ChoiceGate(
    question="Select the optimal database architecture for high-throughput writes:",
    options={
        "A": "Append-only LSM-Tree with SSTables",
        "B": "In-memory B-Tree with periodic checkpoints",
        "C": "Relational table with unindexed foreign keys",
    }
)
verdict = gate.evaluate(candidate_design_doc)
print(f"Selected: {verdict.choice} (P={verdict.probability:.4f})")
```

### `ScoreGate`
Computes the probability-weighted expected score $\mathbb{E}[S] = \sum_{s=1}^5 s \cdot P(s)$:
```python
from logitgate import ScoreGate

gate = ScoreGate(
    rubric="Score the documentation clarity from 1 to 5.",
    min_score=1,
    max_score=5,
    threshold=4.0,
)
verdict = gate.evaluate(candidate_readme)
print(f"Expected Score: {verdict.score:.2f} / 5.0")
```

---

## Integrations

### 1. Pytest Plugin
```python
from logitgate import check

@check(rubric="Output must contain valid JSON with key 'status'", threshold=0.90)
def test_api_payload():
    return '{"status": "ok", "code": 200}'
```

### 2. DeepEval System-1 Model
```python
from logitgate.integrations.deepeval import LogitGateSystemOneModel
from deepeval.metrics import JevEval
from deepeval.test_case import LLMTestCase

system_one = LogitGateSystemOneModel(model_name="qwen2.5-coder:14b")
metric = JevEval(
    name="LogitGate Verification",
    questions=[{"statement": "Artifact satisfies security baseline"}],
    system_one_model=system_one,
)
```

### 3. LangGraph Conditional Router
```python
from logitgate import LogitGate
from logitgate.integrations.langgraph import as_conditional_edge

gate = LogitGate(rubric="Is the response safe and accurate?")
route_fn = as_conditional_edge(gate, pass_node="publish", fail_node="escalate_to_human")
workflow.add_conditional_edges("evaluator", route_fn)
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
