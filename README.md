# LogitGate (`logitgate`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PEP 561 Typed](https://img.shields.io/badge/typing-PEP%20561-blueviolet.svg)](src/logitgate/py.typed)
[![Offline First](https://img.shields.io/badge/offline--first-local--LLM%20ready-brightgreen.svg)](README.md)
[![CI](https://github.com/jasonbdly/logitgate/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**The 40–120ms zero-token LLM evaluation and cascaded decision engine.**

LogitGate replaces slow, expensive, autoregressive LLM-as-a-judge calls with mathematically debiased first-token logit gates and a 3-layer cascaded routing architecture (`ASTGate >> LogitGate >> JudgeGate`).

---

## How It Works in Plain English

When evaluating whether an LLM output meets a rubric, traditional LLM-as-a-judge frameworks prompt another LLM to "think step-by-step" and write a 200–500 word critique before finally outputting `PASS` or `FAIL`. This wastes **5 to 20 seconds**, consumes hundreds of unnecessary output tokens, and introduces positional and label bias.

`LogitGate` replaces this with an intelligent **3-layer cascade**:

1. **Layer 0 (`ASTGate`)**: Checks deterministic rules instantly in **<0.1 ms** using native CPU parsers (syntax validation, regex patterns, acyclic DAG topology, forbidden imports). If candidate code has a syntax error or invokes a forbidden import, it fails immediately with zero tokens and zero GPU overhead.
2. **Layer 1 (`LogitGate`)**: If Layer 0 passes, we present the rubric and candidate to the model and inspect its **first-token log-probabilities (logits)** directly. In a single forward pass (**~40–50 ms on GPU**, **~170 ms on CPU**), we extract the model's exact mathematical confidence without generating any output tokens. To eliminate prior token bias (e.g. models favoring 'A' over 'B'), we run a symmetric pass with inverted choices and combine them in log-odds space (**~85–120 ms**).
3. **Layer 2 (`JudgeGate`)**: Only if Layer 1 confidence falls into an ambiguous range (e.g., between 20% and 85%) or decisively fails does the system escalate to generating an essay critique with diagnostic rationale.

In empirical multi-domain benchmarks across 4B and 14B models, Layer 0 or Layer 1 resolved **100% of test suite requests** decisively (18.8% at Layer 0 in <0.1ms; 81.2% at Layer 1 in 43–124ms) with **zero generated tokens** and **zero additional VRAM**.

---

## Who Is LogitGate For? (3 Target Workflows)

1. **CI/CD & LLM Evaluation Suites (`pytest` / DeepEval)**:
   - Run hundreds of automated LLM evaluation checks in seconds during CI runs instead of waiting 15+ minutes for generative judges. Works completely offline with local models or deterministic stubs.
2. **Multi-Agent & LangGraph Routers**:
   - Conditional routing and decision branching in **40–85 ms** per node transition instead of waiting 2–3 seconds for an LLM supervisor agent to generate JSON routing text.
3. **Local & Resource-Constrained LLM Deployments (Laptops, CPU-Only, Single GPU)**:
   - Evaluates using your already-loaded generator model without requiring a second 14–48 GB judge model running in VRAM. Runs on 0-GPU laptops in pure CPU mode or 8 GB consumer GPUs.

---

## Why LogitGate?

Traditional LLM evaluation relies on generative judges (prompting an LLM to generate 100–500 critique tokens). This is **slow (1,500ms–8,000ms)**, **expensive**, and suffers from severe **position and label permutation biases**. Dedicated judge models (like Prometheus or Selene) require gigabytes of extra GPU VRAM and still suffer from high token generation latency.

| Evaluation Approach | Single-Pass Latency | Output Tokens | Extra GPU VRAM | Token & Positional Bias Debiasing | Ambiguity & Failure Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Generative LLM Judge** *(G-Eval, Ragas, MT-Bench)* | 1,500ms – 8,000ms | 100 – 600 tokens | 0 GB (Shared) | ❌ None (or expensive 2x generative re-query) | ✅ Verbose text critique |
| **Fine-Tuned Judge Models** *(Prometheus 2, Selene)* | 800ms – 3,500ms | 80 – 300 tokens | 14 GB – 48 GB (Dedicated weights) | ❌ Susceptible to length & positional bias | ✅ Fine-tuned score rubric critique |
| **Naive Raw First-Token Logprobs** | 80ms – 250ms | 0 tokens | 0 GB (Shared) | ❌ Severe pretraining prior bias (Zhao et al., 2021) | ❌ No diagnostic rationale |
| **`LogitGate` (3-Layer Cascade)** | **0.04ms (L0) / 40–120ms (L1)** | **0 tokens (L0 / L1)** | **0 GB (Reuses your loaded model)** | **✅ Exact mathematical log-odds cancellation** | **✅ Escalates to Layer 2 Judge on failure / ambiguity** |

---

## Real Multi-Model Live Benchmarks

Empirical measurements gathered on local inference across 4 distinct model configurations running under Ollama:

| Model Configuration | Model Family / Specialization | Compute Device | Warm Single-Pass Latency | Warm 2-Pass Symmetric Latency | 4-Rubric Batch Total | Per-Rubric Batch Latency | Classification Accuracy (4 Cases) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b`** | 4B General Model | GPU | **44.2 ms** | **118.7 ms** | **686.1 ms** | **171.5 ms** | **100% (4/4)** |
| **`gemma4:e4b`** | 4B General Model | **Pure CPU-Only (0 GPU)** | **172.9 ms** | **4,323.6 ms** | N/A | N/A | N/A |
| **`qwen2.5:14b`** | 14B General Model | GPU | **47.5 ms** | **86.2 ms** | **441.5 ms** | **110.4 ms** | **100% (4/4)** |
| **`qwen2.5-coder:14b`** | 14B Code Model | GPU | **43.1 ms** | **86.1 ms** | **436.6 ms** | **109.1 ms** | **100% (4/4)** |

### Benchmark Analysis & Insights:
1. **Single-Pass Latency (~43–48 ms on GPU)**:
   - Evaluates a rubric in a single forward pass without autoregressive decoding. Over **35x faster** than waiting for an LLM to generate 150 words of critique.
2. **2-Pass Symmetric Debiasing (~86–119 ms on GPU)**:
   - Runs both forward ($A=\text{Pass}, B=\text{Fail}$) and reverse ($B=\text{Pass}, A=\text{Fail}$) passes to eliminate pretraining token prior bias. The second pass hits the KV-cache, running in just 40 ms.
3. **Multi-Rubric Batching (~109–171 ms per rubric)**:
   - `evaluate_batch` reuses the candidate prefix across all rubrics, avoiding redundant prompt re-evaluations and preventing KV-cache stampedes on the inference server.
4. **Pure CPU-Only Operation (Zero GPU)**:
   - On a machine with **0 GPUs** (`options={"num_gpu": 0}`), single-pass logit evaluation completes in **~172 ms** on a 4B model—fast enough for local laptop test suites and lightweight CI runners without expensive cloud GPUs.

---

## End-to-End Cascade Time Savings & Pure-CPU Runner Benchmarks

### 1. Multi-Domain Cascade Time Savings (12-Item Test Suite)

Evaluated across **4 non-SDLC and general domains** (RAG Groundedness, Customer Support Policy Compliance, Structured SQL/JSON Safety, Python Code Security):

| Model Configuration | Baseline Generative Judge | Cascade (`trigger="on_ambiguous_or_fail"`) | Cascade (`trigger="on_ambiguous_only"`) | Speedup (Fast-Fail) | Output Tokens Saved | Classification Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 56,002 ms (5,659 tok) | 11,118 ms (1,007 tok) | **1,200 ms (0 tok)** | **46.7x** | **5,659 tokens (100%)** | **12/12 (100%)** |
| **`gemma4:e4b` (Pure CPU, 0 GPU)** | 436,185 ms (5,234 tok) | 105,877 ms (882 tok) | **33,563 ms (0 tok)** | **13.0x** | **5,234 tokens (100%)** | **12/12 (100%)** |
| **`qwen2.5:14b` (GPU)** | 29,144 ms (1,362 tok) | 5,449 ms (197 tok) | **982 ms (0 tok)** | **29.7x** | **1,362 tokens (100%)** | **12/12 (100%)** *(vs. 10/12 baseline)* |
| **`qwen2.5-coder:14b` (GPU)** | 19,642 ms (907 tok) | 4,521 ms (153 tok) | **961 ms (0 tok)** | **20.4x** | **907 tokens (100%)** | **12/12 (100%)** |

---

### 2. Pure-CPU CI/CD Runner Simulation (10-Item CI Test Suite, `options={"num_gpu": 0}`)

Simulating a free 0-GPU GitHub Actions runner using `gemma4:e4b` on pure CPU (`num_gpu=0`):

| Evaluation Strategy | Total CPU Wall Time | Total Tokens Generated | Diagnostic Quality | Accuracy | System RAM Footprint | VRAM Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Generative Judge (`max_tokens=60`)** | 60,257 ms (~60.3s) | 600 tokens | ⚠️ Truncated CoT essays missing final verdict tags | 2/10 (20%) | 8.91 GB | **0 bytes** |
| **`LogitGate(preset="fast_ci") >> JudgeGate`** | **17,476 ms (~17.5s)** | **60 tokens (90% reduction)** | ✅ Crisp 60-token failure reason on semantic failure | **10/10 (100%)** | 8.91 GB | **0 bytes** |

*Key finding*: On CPU CI runners, `LogitGate(preset="fast_ci")` is **3.45x faster** while achieving **100% accuracy** because Layer 0 and Layer 1 resolve clear passes and structural errors with zero tokens, reserving generative diagnostics only for the single semantic failure.

---

### 3. Empirical Layer Resolution Breakdown (16-Item Realistic Stress Test Suite)

Evaluated across clear passes, structural Layer 0 failures, clear semantic failures, and subtle borderline edge cases (subtle omissions, partial compliance, timing side-channels, adversarial overrides):

| Model Configuration | Layer 0 (`ASTGate`) | Layer 1 Decisive (`LogitGate`) | Layer 2 Escalated (`JudgeGate`) | Combined Zero-Token Resolution Rate |
| :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 18.8% (3/16) | 81.2% (13/16) | 0.0% (0/16) | **100.0%** |
| **`gemma4:e4b` (Pure CPU, 0 GPU)** | 18.8% (3/16) | 81.2% (13/16) | 0.0% (0/16) | **100.0%** |
| **`qwen2.5:14b` (GPU)** | 18.8% (3/16) | 81.2% (13/16) | 0.0% (0/16) | **100.0%** |
| **`qwen2.5-coder:14b` (GPU)** | 18.8% (3/16) | 81.2% (13/16) | 0.0% (0/16) | **100.0%** |

---

## When to Use Layer 1 (`LogitGate`) vs. Layer 2 (`JudgeGate`): Understanding $O(1)$ Circuit Depth

A causal transformer produces its first token in a single forward pass over the context window. In computational complexity terms, this corresponds to an **$O(1)$ sequential circuit depth** (bounded strictly by the number of attention layers $L$):

- **Ideal for Layer 1 (`LogitGate`) — $O(1)$ Forward Pass**:
  - **Semantic Rubric Compliance**: Checking if output meets tone, safety, formatting, or negative constraints.
  - **Policy & Safety Gating**: Instant detection of prohibited topics, leaked secrets, or regulatory violations.
  - **Discrete Routing & Categorization**: Selecting between discrete tools, workflow routes, or intent buckets.
  - **Surface Quality & Structure Checks**: Verifying presence of docstrings, required sections, or types.

- **Escalate to Layer 2 (`JudgeGate`) — Autoregressive Generation**:
  - **Multi-Step Deductive Reasoning**: Proof verification or multi-hop logic where conclusions depend on intermediate reasoning steps.
  - **Complex Symbolic Arithmetic**: Calculations requiring scratchpad calculation or token-by-token recurrence.
  - **Diagnostic Reporting**: When a human developer or logging pipeline needs an actionable explanation of *why* an artifact failed.

---

## System Requirements & Hardware Tiers

`LogitGate` adapts dynamically across hardware environments:

| Hardware Tier | Recommended Model | Backend Specifier / Config | Typical Single-Pass Latency | Target Workflows |
| :--- | :--- | :--- | :--- | :--- |
| **0-GPU CI Runners / Air-Gapped** | Offline Deterministic Stub | `stub://default` | **< 0.05 ms** | Unit tests, mock suites, CI smoke tests |
| **CPU-Only Laptops / Edge** | `gemma4:e4b` / `llama3.2:3b` | `ollama://localhost:11434/gemma4:e4b` (`options={"num_gpu": 0}`) | **~170 ms** | Local developer testing without discrete GPU |
| **8 GB VRAM Consumer GPU** | `gemma4:e4b` / `qwen2.5:7b` | `ollama://localhost:11434/gemma4:e4b` | **~40–45 ms** | Desktop development, real-time agent loops |
| **16–24 GB VRAM Workstation** | `qwen2.5-coder:14b` / `qwen2.5:14b` | `ollama://localhost:11434/qwen2.5-coder:14b` | **~40–50 ms** | Production evaluation suites, coding agents |
| **Cloud Inference / Hosted APIs** | `gpt-4o-mini` / `vLLM` | `openai://api.openai.com/v1` or `vllm://localhost:8000/v1` | **~60–120 ms** (network bounded) | High-concurrency enterprise pipelines |

---

## Role-Based Setup & Use-Case Guides

LogitGate is designed to fit directly into diverse software engineering and AI workflows across general and non-SDLC domains:

### 1. CI/CD & DevOps Engineers (Free CPU-Only GitHub Actions / GitLab CI Runners)
Run fast regression suites on free 0-GPU CI runners (standard 2–4 vCPU GitHub Actions runners) without paying for cloud GPU instances:
```python
from logitgate import ASTGate, LogitGate, JudgeGate, Cascade, OllamaBackend

# Pure CPU CI Runner configuration (0 GPU, ~3.2 GB RAM for 3B models / ~8.9 GB RAM for gemma4:e4b)
backend = OllamaBackend(
    model="gemma4:e4b",
    options={"num_gpu": 0, "num_thread": 4},
    keep_alive="5m",
)

# 1-pass fast triage for passes (0 tokens, ~170ms) + concise 60-token diagnostic on failure
ci_pipeline = (
    ASTGate(syntax=True, forbidden_imports=["pickle", "subprocess"])
    >> LogitGate(
        rubric="Candidate code must be clean, secure, and handle edge cases.",
        preset="fast_ci",  # Single-pass 80% pass threshold, debiasing disabled for max speed
        backend=backend,
    )
    >> JudgeGate(
        max_tokens=60,  # 1-2 sentence concise diagnostic for failed CI logs
        trigger="on_ambiguous_or_fail",
        backend=backend,
    )
)

verdict = ci_pipeline.evaluate(candidate_code)
assert verdict.passed, f"CI Gate Failed at {verdict.triggered_layer}: {verdict.reason}"
```
- **Resource Footprint**: Runs in ~2.2–3.2 GB System RAM (`qwen2.5:3b` / `llama3.2:3b`) or ~8.9 GB System RAM (`gemma4:e4b`) with 0 VRAM overhead—fitting comfortably inside a standard 16 GB GitHub Actions (`ubuntu-latest`) VM.
- **Cost Savings**: 100% free runner tier without provisioning \$0.50–\$2.00/hr GPU runners.

---

### 2. RAG & Chatbot Engineers (Hallucination & Groundedness Gating)
Detect hallucinations and verify factual groundedness before streaming an answer to the end user:
```python
from logitgate import ContractGate, LogitGate

# Layer 0 ensures reasonable length bounds in 0ms; Layer 1 verifies grounding in ~45ms
rag_gate = (
    ContractGate(min_length=20, max_length=2000)
    >> LogitGate(
        rubric="Does the candidate answer strictly contain only facts asserted in the context snippet?",
        preset="balanced",
    )
)

context_payload = f"[CONTEXT]\n{retrieved_docs}\n\n[ANSWER]\n{generated_response}"
verdict = rag_gate.evaluate(context_payload)

if not verdict.passed:
    # Trigger fallback or retrieve secondary sources
    generated_response = "I cannot verify that from the documentation. Let me connect you with an agent."
```

---

### 3. AI Agent & LangGraph Builders (Fast Guardrails & Routing)
Enforce structured JSON output contracts in **<0.1ms** and perform sub-50ms conditional routing:
```python
from pydantic import BaseModel
from logitgate import ContractGate, ChoiceGate

class ToolCallPayload(BaseModel):
    tool_name: str
    arguments: dict
    confirm_action: bool

# Layer 0 validates JSON schema instantly; Layer 1 routes intent in ~40ms
agent_router = (
    ContractGate(json_schema=ToolCallPayload)
    >> ChoiceGate(
        question="Select the appropriate downstream handler for this user intent:",
        options={
            "A": "Database Query Execution (read-only safe queries)",
            "B": "Account Management & Authentication changes",
            "C": "General Customer Q&A Response",
        },
    )
)

verdict = agent_router.evaluate(agent_raw_json)
next_node = {"A": "db_node", "B": "auth_node", "C": "qa_node"}[verdict.choice]
```

---

### 4. QA & Evaluation Engineers (Pytest & Multi-Criteria Rubrics)
Auto-calibrate model temperatures and execute multi-criteria rubrics with shared KV-cache:
```bash
# Calibrate your model's temperature on an 8-item micro-dataset
logitgate calibrate --backend "ollama://localhost:11434" --model "qwen2.5:14b"
```

```python
from logitgate import LogitGate

evaluator = LogitGate(
    rubric="Base rubric",
    logit_temperature="auto",  # Automatically uses cached calibrated temperature
)

rubrics = [
    "Is the customer support reply polite and professional?",
    "Does the assistant enforce multi-factor authentication before updating credentials?",
    "Does the response adhere to the company return and refund policy?",
]

# Evaluates all 3 rubrics in a single batch reusing 100% of candidate KV-cache
verdicts = evaluator.evaluate_batch(support_transcript, rubrics)
for r, v in zip(rubrics, verdicts):
    print(f"[{'PASS' if v.passed else 'FAIL'}] P={v.probability:.2f} | {r}")
```

---

## Best Practices: Prompting, Context Length, & Model Sizing

### 1. Optimal Candidate Length & Chunking (500–4,000 Tokens)
- **Attention Density**: First-token logit gates perform optimally on artifacts between **500 and 4,000 tokens**. In causal transformers, very long contexts (>16k tokens) can suffer from attention dilution on subtle constraints.
- **Document Chunking**: For evaluating long documents (e.g. 50-page contracts or huge codebases), break the document into semantic sections or functions and evaluate them individually with `evaluate_batch`.

### 2. Writing Atomic, Observable Rubrics vs. Giant Compound Prompts
- **Single Observable Property**: Write rubrics that test exactly one clear, falsifiable property (e.g. *"Does the reply refuse to reset passwords without 2FA?"* rather than *"Is the reply polite, compliant with 2FA, formatted as markdown, and under 200 words?"*).
- **Prefix-Shared Batching**: Instead of one monolithic prompt, pass a list of atomic rubrics to `evaluate_batch(candidate, rubrics)`. Because all rubrics share the candidate prefix, running 5 atomic rubrics takes only ~110ms per rubric while giving per-criterion visibility.

### 3. Escalation Mechanics: How Symmetric Log-Odds Handle Model Uncertainty
- **Decisive Signal**: When a model is confident that an artifact passes, both forward ($L_1 = s + b$) and reverse ($L_2 = s - b$) log-odds are positive, yielding calibrated probability $P \ge 0.85$ (fast-path pass, 0 tokens).
- **Decisive Failure**: When an artifact clearly violates the rubric, both forward and reverse log-odds are negative, yielding $P \le 0.20$ (decisive failure, 0 tokens).
- **Ambiguity & Escalation**: When small models (e.g. 3B–4B) or complex edge cases yield borderline log-odds (between 0.20 and 0.85), LogitGate flags the item as `AMBIGUOUS` and seamlessly triggers Layer 2 (`JudgeGate`) to perform full chain-of-thought autoregressive critique.

---

## Research Foundations & Academic Citations

LogitGate synthesizes and builds upon foundational breakthroughs in calibration, logit debiasing, and LLM evaluation:

### What is Unique to `LogitGate`
1. **Zero-Extra-VRAM 3-Layer Cascade**: Uses the existing generator model instance for both gating and diagnostics without loading a dedicated secondary judge model into VRAM.
2. **Prefix-Preserving KV Compilation (`compile_batch_binary_prompt`)**: Arranges prompts so the candidate artifact and evaluation rubrics form an identical static prefix across forward and reverse debiasing passes, retaining the entire KV-cache across queries.
3. **Log-Odds Space Symmetric Inversion**: Cancels prior token bias in log-odds space ($\bar{L} = \frac{L_1 + L_2}{2T}$) rather than naive probability averaging, preserving exact ground-truth signal even under extreme pretraining token skew.
4. **Missing-Token Dynamic Floor**: Bounds unobserved token logprobs using residual probability mass $\min(\min_i \text{logprob}_i, \ln(1 - \sum P_i))$, eliminating distortion from arbitrary $-99.0$ constants.

### Foundational Literature Credited

1. **Pretraining Token Prior Bias & Calibration**:
   - Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). *"Calibrate Before Use: Improving Few-Shot Performance of Language Models"*. **ICML 2021**. [arXiv:2102.09690](https://arxiv.org/abs/2102.09690).
   - *Contribution to LogitGate*: Demonstrates that large language models suffer from severe majority-label, recency, and surface-form token priors (such as inherently favoring token `'A'` over `'B'`).

2. **Positional & Label Permutation Bias in LLM Judges**:
   - Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*. **NeurIPS 2023**. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
   - *Contribution to LogitGate*: Establishes the necessity of symmetric permutation testing to cancel order bias and label preference in LLM evaluations.

3. **Probability-Weighted Continuous Expected Scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$)**:
   - Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"*. **EMNLP 2023**. [arXiv:2303.16634](https://arxiv.org/abs/2303.16634).
   - *Contribution to LogitGate*: Provides the mathematical foundation for `ScoreGate`, translating discrete first-token probability distributions over score tokens into smooth, continuous expected value metrics.

4. **Post-Hoc Temperature Scaling & Expected Calibration Error (ECE)**:
   - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *"On Calibration of Modern Neural Networks"*. **ICML 2017**. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
   - *Contribution to LogitGate*: Forms the theoretical basis of `fit_calibration()`, using temperature scaling to align logit confidence with true empirical accuracy and minimize Brier score.

5. **System-1 Single-Token Decision Benchmarking**:
   - Deußer, C., et al. (2026). *"Evaluating and Benchmarking the System One Model Jev"*. [arXiv:2609.37647](https://arxiv.org/abs/2609.37647).
   - *Contribution to LogitGate*: Benchmarks fast, single-token System-1 evaluation pipelines against heavy System-2 generative reasoning judges.

### BibTeX Citation

```bibtex
@software{logitgate2026,
  author = {Jason Bradley},
  title = {LogitGate: Zero-Token Mathematically Debiased LLM Evaluation and Cascaded Routing},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/jasonbdly/logitgate},
  license = {Apache-2.0}
}
```

---

## Mathematical Foundations

### 1. Symmetric Log-Odds Permutation Debiasing

Suppose token `'A'` has an extreme pretraining prior bias $+b$, and the true underlying signal is $s$:

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
  $$P_1 = \sigma(7.0) \approx 0.9991, \quad P_2 = \sigma(-3.0) \approx 0.0474 \implies \bar{P} = 0.5233 \quad (\text{Coin Flip / Failed})$$
- **LogitGate Log-Odds Debiasing**:
  $$\bar{L} = \frac{7.0 + (-3.0)}{2} = 2.0 \implies \sigma(2.0) = \mathbf{0.8808} \quad (\text{Exact True Signal Recovered!})$$

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

Because Pass 1 and Pass 2 share an identical prefix up to `[DECISION MAPPING]`, the entire rubric and candidate context are retained in GPU KV-cache, making Pass 2 near-instantaneous.

---

### 3. Missing-Token Bounded Floor

When target tokens (e.g. `'B'`) do not appear in the top-$K$ logprobs returned by an API, assigning arbitrary $-99.0$ cliffs distorts the symmetric mean. LogitGate dynamically computes the theoretical probability ceiling:

$$\text{floor} = \min\left(\min_{i} \text{logprob}_i, \, \ln\max\left(1.0 - \sum_{i} P_i, \, 10^{-6}\right)\right)$$

This bounds missing tokens strictly to the residual unobserved probability mass.

---

## 3-Layer Cascaded Router Architecture

```
                 Candidate Artifact
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│                   Layer 0:                       │   Syntax / DAG Cycle / Forbidden Import
│                   ASTGate                        ├─────────────────────────────────────────► [FAIL: 0 tokens, ~0.04ms]
└────────────────────────┬─────────────────────────┘
                         │ Passed
                         ▼
┌──────────────────────────────────────────────────┐   P >= pass_threshold (>= 0.85)
│                   Layer 1:                       ├─────────────────────────────────────────► [PASS: 0 tokens, ~40–120ms]
│                  LogitGate                       │
└────────────────────────┬─────────────────────────┘   P <= fail_threshold (<= 0.20)
                         │ Ambiguous                   └─────────────────────────────────────► [FAIL: 0 tokens, ~40–120ms]
                         ▼
┌──────────────────────────────────────────────────┐
│                   Layer 2:                       │   Escalated Generative Critique
│                  JudgeGate                       ├─────────────────────────────────────────► [Actionable Diagnostic Report]
└──────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Core package (offline-ready, zero dependencies except pydantic & httpx)
pip install logitgate

# With all optional integrations (OpenAI, DeepEval, LangGraph, Pytest)
pip install "logitgate[all]"
```

---

## Complete Copy-Pasteable Examples

### 1. Universal Configuration & Environment Variables

Switch backends or thresholds across your entire application or test suite globally without hardcoding parameters in individual gates:

```python
import logitgate

# Configure defaults globally across all gates:
logitgate.configure(
    backend="ollama://localhost:11434/gemma4:e4b",
    pass_threshold=0.85,
    fail_threshold=0.20,
    logit_temperature=1.0,
)

# Any gate instantiated without explicit backend/thresholds inherits these defaults:
from logitgate import LogitGate
gate = LogitGate("Candidate response must follow security best practices.")

# Or control your entire evaluation pipeline via environment variables (zero code changes):
# export LOGITGATE_BACKEND="ollama://localhost:11434/gemma4:e4b"
# export LOGITGATE_PASS_THRESHOLD="0.85"
# export LOGITGATE_FAIL_THRESHOLD="0.20"

# Reset anytime to defaults (offline stub://default):
logitgate.reset_config()
```

---

### 2. 30-Second Quickstart (3-Layer Cascade)

```python
from logitgate import ASTGate, LogitGate, JudgeGate

# Compose the 3-layer pipeline using the >> operator
pipeline = (
    ASTGate(syntax=True, forbidden_imports=["subprocess", "os.system"])
    >> LogitGate(
        rubric="Candidate code must be strictly typed and include clear docstrings.",
        pass_threshold=0.85,
        fail_threshold=0.20,
        backend="ollama://localhost:11434/qwen2.5-coder:14b",
    )
    >> JudgeGate(trigger="on_ambiguous_or_fail")
)

candidate_code = """
def calculate_velocity(distance: float, time_elapsed: float) -> float:
    \"\"\"Calculates average velocity in meters per second.\"\"\"
    if time_elapsed <= 0:
        raise ValueError("time_elapsed must be positive")
    return distance / time_elapsed
"""

verdict = pipeline.evaluate(candidate_code)

if verdict:
    print(f"Passed at {verdict.triggered_layer}! Latency: {verdict.total_latency_ms:.1f}ms (Tokens: {verdict.total_tokens_generated})")
else:
    print(f"Failed at {verdict.triggered_layer}: {verdict.reason}")
```

---

### 3. Multi-Rubric Batch Evaluation with 100% KV-Cache Sharing

Evaluate multiple rubrics concurrently against the same candidate artifact while preserving 100% of the KV-cache prefix:

```python
from logitgate import LogitGate

gate = LogitGate(
    rubric="Shared context evaluator",
    backend="ollama://localhost:11434/qwen2.5-coder:14b",
)

rubrics = [
    "1. Code contains comprehensive type hints.",
    "2. All exceptions are handled explicitly.",
    "3. No security vulnerabilities or unsafe shell invocations.",
    "4. Functions include concise, accurate docstrings.",
]

candidate = """
def fetch_user_record(user_id: str) -> dict:
    \"\"\"Fetches user metadata from the storage engine.\"\"\"
    if not user_id.isalnum():
        raise ValueError("Invalid user_id format")
    return {"id": user_id, "active": True}
"""

# Evaluates all 4 rubrics with forward/reverse debiasing using 1 shared prefix
verdicts = gate.evaluate_batch(candidate, rubrics)

for idx, v in enumerate(verdicts, 1):
    status = "PASS" if v.passed else "FAIL"
    print(f"Rubric #{idx}: [{status}] (P={v.probability:.4f})")
```

---

### 4. Discrete Routing (`ChoiceGate`) & Continuous Scoring (`ScoreGate`)

#### Discrete K-ary Routing (`ChoiceGate`)
```python
from logitgate import ChoiceGate

router = ChoiceGate(
    question="Select the optimal storage architecture for high-throughput write-heavy workloads:",
    options={
        "A": "Append-only LSM-Tree with SSTables (Log-Structured Merge-Tree)",
        "B": "In-memory B-Tree with periodic synchronous disk checkpoints",
        "C": "Relational table with unindexed foreign key cascades",
    },
    backend="ollama://localhost:11434/qwen2.5-coder:14b",
)

verdict = router.evaluate("Workload profile: 100,000 ingest events/sec, append-heavy, rare point reads.")
print(f"Selected Architecture: {verdict.choice} (P={verdict.probability:.4f})")
print(f"All Option Probabilities: {verdict.choice_probabilities}")
```

#### Continuous Rubric Scoring (`ScoreGate`)
Computes continuous expected score $\mathbb{E}[S] = \sum_{s=1}^5 s \cdot P(s)$ with symmetric scale inversion:
```python
from logitgate import ScoreGate

scorer = ScoreGate(
    rubric="Score API architectural documentation quality from 1 (unusable) to 5 (production-grade).",
    min_score=1,
    max_score=5,
    threshold=4.0,
    symmetric_debiasing=True,  # Evaluates both 1..5 and 5..1 to cancel anchoring bias
    backend="ollama://localhost:11434/qwen2.5-coder:14b",
)

verdict = scorer.evaluate(candidate_documentation)
print(f"Continuous Expected Score: {verdict.score:.2f} / 5.0 (Passed: {verdict.passed})")
```

---

### 5. Post-Hoc Temperature Calibration (`fit_calibration`)

Optimize post-hoc temperature $T^*$ on validation samples to minimize Brier score and Expected Calibration Error (ECE):

```python
from logitgate import fit_calibration

# Validation dataset: (forward_log_odds, reverse_log_odds, ground_truth_label)
validation_samples = [
    (+2.1, +1.8, 1),
    (-1.5, -1.9, 0),
    (+0.4, +0.2, 1),
    (-0.8, -0.6, 0),
]

result = fit_calibration(validation_samples, min_temp=0.1, max_temp=5.0)

print(f"Optimal Temperature: T* = {result.optimal_temperature:.3f}")
print(f"Brier Score: {result.brier_score_before:.4f} -> {result.brier_score_after:.4f}")
print(f"Expected Calibration Error (ECE): {result.ece_before:.4f} -> {result.ece_after:.4f}")
```

---

### 6. Command-Line Interface (CLI)

```bash
# Evaluate a candidate text directly against a rubric
logitgate eval \
  --rubric "Response must not contain toxic or harmful instructions" \
  --candidate "To reset your password, visit account settings." \
  --threshold 0.85

# Evaluate candidate from a file with Layer 0 AST syntax check and JSON output
logitgate eval \
  --rubric "Code must be production-ready and fully typed" \
  --file ./src/app.py \
  --ast-syntax \
  --backend "ollama://localhost:11434/qwen2.5-coder:14b" \
  --json

# Pipe candidate directly via standard input
cat schema.json | logitgate eval --rubric "Must be valid JSON schema"
```

---

### 7. Framework Integrations

#### Pytest Integration (`@check`)
```python
import pytest
from logitgate import check

@check(rubric="Function output must be valid JSON with key 'status' equal to 'success'", threshold=0.90)
def test_user_creation_api():
    return '{"status": "success", "user_id": "usr_99823"}'
```

#### DeepEval Integration (`LogitGateSystemOneModel` & `LogitGateMetric`)
```python
from deepeval.metrics import JevEval
from deepeval.test_case import LLMTestCase
from logitgate.integrations.deepeval import LogitGateSystemOneModel, LogitGateMetric

# 1. Use as a fast System-1 judge model for JevEval
system_one = LogitGateSystemOneModel(
    model_name="qwen2.5-coder:14b",
    api_base="http://localhost:11434",
)

metric = JevEval(
    name="Security Baseline Check",
    questions=[{"statement": "Artifact does not invoke raw shell commands"}],
    system_one_model=system_one,
)

# 2. Or use directly as a standalone DeepEval metric
fast_metric = LogitGateMetric(
    rubric="Summary accurately reflects source text without hallucinations",
    threshold=0.85,
)
```

#### LangGraph Integration (`as_conditional_edge`)
```python
from langgraph.graph import StateGraph, END
from logitgate import LogitGate
from logitgate.integrations.langgraph import as_conditional_edge

gate = LogitGate(rubric="Is the generated response safe and helpful?")

# Creates a fast routing function extracting 'content' from graph state
route_verdict = as_conditional_edge(gate, pass_node="publish_response", fail_node="human_escalation")

workflow = StateGraph(...)
workflow.add_conditional_edges("generator", route_verdict)
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for local development setup, coding standards, and testing procedures.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history and version notes.

---

## License

LogitGate is licensed under the [Apache License 2.0](LICENSE).
