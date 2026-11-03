# LogitGate (`logitgate`)

[![Release: v0.1.0](https://img.shields.io/badge/release-v0.1.0-blue.svg)](CHANGELOG.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10–3.15](https://img.shields.io/badge/python-3.10--3.15-blue.svg)](https://www.python.org/downloads/)
[![PEP 561 Typed](https://img.shields.io/badge/typing-PEP%20561-blueviolet.svg)](src/logitgate/py.typed)
[![Typechecked: mypy](https://img.shields.io/badge/typechecked-mypy-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub CI](https://github.com/jasonbdly/LogitGate/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonbdly/LogitGate/actions/workflows/ci.yml)
[![GitLab CI](https://gitlab.com/jasonbdly/logitgate/badges/main/pipeline.svg)](https://gitlab.com/jasonbdly/logitgate/-/pipelines)

**Fast, calibrated first-token logit evaluation and 3-layer cascaded routing for LLMs.**

LogitGate replaces slow, multi-token LLM-as-a-judge critiques with calibrated first-token logit gates and a 3-layer cascaded routing pipeline (`ASTGate >> LogitGate >> JudgeGate`).

---

## How It Works in Plain English

When evaluating whether an LLM output meets a rubric, standard LLM-as-a-judge frameworks prompt another model to write a 200–500 word critique before outputting `PASS` or `FAIL`. This takes **5 to 20 seconds**, generates hundreds of Chain-of-Thought (CoT) tokens, and suffers from positional and label biases.

`LogitGate` uses a **3-layer cascade**:

1. **Layer 0 (`ASTGate` / `ContractGate`)**: Runs deterministic checks in **<0.1 ms** on the CPU (Python AST syntax, JSON/Pydantic schemas, regex patterns, acyclic DAG topology, forbidden imports). If an artifact fails a deterministic contract, it fails immediately with no LLM call.
2. **Layer 1 (`LogitGate`)**: If Layer 0 passes, we pass the rubric and candidate to the model and read its **first-token log-probabilities (`max_tokens=1`)**. In a single forward pass (**~36–41 ms on GPU**, **~173 ms on CPU**), we compute the model's classification confidence without generating multi-token CoT text. To cancel pretraining token priors (such as models favoring `'A'` over `'B'`), we run a second pass with swapped labels and average them in log-odds space (**~90–113 ms on GPU**).
3. **Layer 2 (`JudgeGate`)**: Only when Layer 1 confidence falls into an ambiguous band (e.g., between 20% and 85%) or requires failure diagnostics does the pipeline escalate to generating a full text critique.

In our 24-item multi-domain benchmark suite (with `trigger="on_ambiguous_only"`), Layer 0 and Layer 1 resolved **95.8%–100.0% of items** (8.3% at Layer 0 in <0.1ms; 87.5%–91.7% at Layer 1 in 36–113ms) while reusing the already-loaded model, escalating only genuinely ambiguous cases (`4.2%` on `qwen2.5-coder:14b`) to Layer 2 (`JudgeGate`), as detailed in [Falsification Benchmarks](#falsification-benchmarks-where-does-layer-1-break-down--how-to-mitigate-it).

---

## Who Is LogitGate For? (3 Target Workflows)

1. **CI/CD & LLM Evaluation Suites (`pytest` / DeepEval)**:
   - Run hundreds of automated LLM evaluation checks in seconds during CI runs instead of waiting 15+ minutes for generative judges. Works completely offline with local models or deterministic stubs.
2. **Multi-Agent & LangGraph Routers**:
   - Conditional routing and decision branching in **36–91 ms** per node transition instead of waiting 2–3 seconds for an LLM supervisor agent to generate JSON routing text.
3. **Local & Resource-Constrained LLM Deployments (Laptops, CPU-Only, Single GPU)**:
   - Evaluates using your already-loaded generator model without requiring a second 14–48 GB judge model running in VRAM. Runs on 0-GPU laptops in pure CPU mode or 8 GB consumer GPUs.

---

## Why LogitGate?

Traditional LLM evaluation relies on generative judges (prompting an LLM to generate 100–500 critique tokens). This is **slow (1,500ms–8,000ms)**, **expensive**, and suffers from severe **position and label permutation biases**. Dedicated judge models (like Prometheus or Selene) require gigabytes of extra GPU VRAM and still suffer from high token generation latency.

| Evaluation Approach | Typical Latency | Generated Output Tokens | Extra GPU VRAM | Token & Positional Bias Mitigation | Ambiguity & Failure Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Generative LLM Judge** *(G-Eval, Ragas, MT-Bench)* | 1,500ms – 8,000ms | 100 – 600 CoT tokens | 0 GB (Shared) | ❌ None (or expensive 2x generative re-query) | ✅ Verbose text critique |
| **Fine-Tuned Judge Models** *(Prometheus 2, Selene)* | 800ms – 3,500ms | 80 – 300 CoT tokens | 14 GB – 48 GB (Dedicated weights) | ❌ Susceptible to length & positional bias | ✅ Fine-tuned rubric critique |
| **Naive Raw First-Token Logprobs** | 35ms – 250ms | 1 token (`max_tokens=1`, 0 CoT) | 0 GB (Shared) | ❌ Vulnerable to token prior bias (Zhao et al., 2021) | ❌ No diagnostic rationale |
| **`LogitGate` (3-Layer Cascade)** | **0.04ms (L0) / 36–113ms (L1)** | **0 (L0) / 1 token prefill probe (L1, 0 CoT)** | **0 GB (Reuses loaded model)** | **✅ Symmetric log-odds debiasing** | **✅ Escalates to Layer 2 Judge on failure / ambiguity** |

---

## Real Multi-Model Live Benchmarks

Empirical measurements gathered on local inference across 5 distinct model architectures running under Ollama (`v0.35.0`, `Q4_K_M` 4-bit GGUF quantization, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`). All declarative benchmark datasets ([`benchmarks/data/`](benchmarks/data/)), the benchmark runner ([`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py)), and raw JSON outputs ([`benchmarks/latest_results.json`](benchmarks/latest_results.json)) are included in the repository for independent reproduction.

- **Testbed Hardware**:
  - **CPU & System Memory**: AMD Ryzen 7 5800X3D (8-Core / 16-Thread), 64 GB DDR4 System RAM (used for all `Pure CPU-Only` benchmarks with `options={"num_gpu": 0}`).
  - **GPU 1 (12 GB Sidecar Tier)**: NVIDIA GeForce RTX 4070 (12 GB GDDR6X VRAM) — used for `4B` (`gemma4:e4b`) and `14B` (`qwen2.5:14b`, `qwen2.5-coder:14b`) benchmarks.
  - **GPU 0 (32 GB Primary Tier)**: NVIDIA GeForce RTX 5090 (32 GB GDDR7 VRAM) — used for `26B MoE` (`gemma4:26b`) and `31B Dense` (`gemma4:31b`) benchmarks.
- **Benchmark Methodology & Sample Protocol**:
  - **Multi-Sample Latency Protocol ($N=5$)**: Every latency configuration is preceded by `1` untimed warm-up pass (to load GGUF weights into memory and initialize KV-cache buffers), followed by `N=5` sequential timed trials over a persistent HTTP connection pool (`httpx.Client`), reporting `mean ± std` in milliseconds.
  - **Leak-Free Matched Datasets**: All test items in [`benchmarks/data/`](benchmarks/data/) are stripped of meta-comments and structured as matched pass/fail pairs across 4 domains (24 items in Suite 1; 32 items across 4 reasoning tiers in Suite 2; 12 `ChoiceGate`/`ScoreGate` items in Suite 3; and a 12-section enterprise handbook in Suite 4).
  - **Deterministic Classification**: All Layer 1 probes execute with greedy decoding parameters (`temperature=0.0`, `top_p=1.0`, `max_tokens=1`), yielding deterministic log-probabilities across runs for a fixed prompt and model checkpoint.

| Model Configuration | Model Family | Compute Device | Single-Pass Latency ($N=5$) | 2-Pass Symmetric Latency ($N=5$) | 4-Rubric Batch Total ($N=5$) | Per-Rubric Batch Latency ($N=5$) | 24-Item Multi-Domain Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b`** | 4B General Model | GPU (RTX 4070) | **40.5 ± 6.0 ms** | **112.6 ± 4.6 ms** | **584.9 ± 51.8 ms** | **146.2 ± 12.9 ms** | **23/24 (95.8%)** |
| **`gemma4:e4b`** | 4B General Model | Pure CPU-Only (0 GPU) | **173.0 ± 2.4 ms** | **3,710.5 ± 82.1 ms** | **20,949.5 ± 182.5 ms** | **5,237.4 ± 45.6 ms** | **23/24 (95.8%)** |
| **`qwen2.5:14b`** | 14B General Model | GPU (RTX 4070) | **36.8 ± 6.5 ms** | **90.0 ± 4.1 ms** | **450.1 ± 18.8 ms** | **112.5 ± 4.7 ms** | **23/24 (95.8%)** |
| **`qwen2.5-coder:14b`** | 14B Code Model | GPU (RTX 4070) | **36.5 ± 6.3 ms** | **90.9 ± 3.3 ms** | **447.4 ± 3.9 ms** | **111.8 ± 1.0 ms** | **23/24 (95.8%)** |

### Benchmark Analysis & Insights:
1. **Single-Pass Latency (~36–41 ms on GPU)**:
   - Evaluates a rubric in a single forward pass without autoregressive decoding. Over **35x faster** than waiting for an LLM to generate 150 words of critique.
2. **2-Pass Symmetric Debiasing (~90–113 ms on GPU)**:
   - Runs both forward ($A=\text{Pass}, B=\text{Fail}$) and reverse ($B=\text{Pass}, A=\text{Fail}$) passes to eliminate pretraining token prior bias. The second pass hits the KV-cache, running in ~40–55 ms.
3. **Multi-Rubric Batching (~112–146 ms per rubric)**:
   - `evaluate_batch` reuses the candidate prefix across all rubrics, avoiding redundant prompt re-evaluations and preventing KV-cache stampedes on the inference server.
4. **Pure CPU-Only Operation (Zero GPU)**:
   - On a machine with **0 GPUs** (`options={"num_gpu": 0}`), warm single-pass logit evaluation (`preset="fast_triage"`) completes in **173.0 ± 2.4 ms** (`51.3 s` total across the 24-item suite including cold candidate prefills), while full 2-pass symmetric debiasing (`preset="balanced"`) takes `3.7s–4.3s` per item.

---

## End-to-End Cascade Time Savings & Multi-Gate Benchmarks

### 1. Multi-Domain Cascade Time Savings (24-Item Test Suite)

Evaluated across **24 items spanning 4 domains** ([`benchmarks/data/multi_domain_24.json`](benchmarks/data/multi_domain_24.json), 6 items per domain: RAG Groundedness with subtle scope/regional extrapolations, Customer Support Policy Compliance, SQL/API Query Safety, and Python Security). To ensure a fair comparison, the Baseline Generative Judge also runs with `<think>` suppressed (`think=False`):

| Model Configuration | Baseline Judge Time | Cascade Time (`on_ambiguous_or_fail`) | Cascade Time (`on_ambiguous_only`) | Speedup (`on_ambiguous_only`) | CoT Tokens Saved | Baseline Accuracy | Cascade Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 19,182.5 ms | 11,503.0 ms | **3,279.0 ms** | **5.9x** | 1,756 | 24/24 (100.0%) | **23/24 (95.8%)** |
| **`gemma4:e4b` (Pure CPU, 2-Pass)** | 142,780.6 ms | 143,669.7 ms | **103,426.7 ms** | **1.4x** | 1,086 | 24/24 (100.0%) | **23/24 (95.8%)** |
| **`gemma4:e4b` (Pure CPU, 1-Pass)** | 142,780.6 ms | 91,430.2 ms | **51,287.2 ms** | **2.8x** | 1,086 | 24/24 (100.0%) | **23/24 (95.8%)** |
| **`qwen2.5:14b` (GPU)** | 117,006.7 ms | 53,046.6 ms | **7,939.3 ms** | **14.7x** | 2,183 | 23/24 (95.8%) | **23/24 (95.8%)** |
| **`qwen2.5-coder:14b` (GPU)** | 132,498.2 ms | 65,147.1 ms | **11,743.3 ms** | **11.3x** | 2,578 | 24/24 (100.0%) | **23/24 (95.8%)** |

---

### 2. Fair Pure-CPU CI/CD Runner Benchmark (10-Item CI Test Suite, `options={"num_gpu": 0}`)

Simulating a free 0-GPU GitHub Actions runner using `gemma4:e4b` on pure CPU (`num_gpu=0`) across a 10-item Python CI suite (8 valid Python items, 1 Layer-0 AST syntax failure, 1 semantic PII security leak), with token savings and speedup measured against `Baseline A`:

| Evaluation Strategy | Total CPU Wall Time | Total Output Tokens | Accuracy | Output Token Savings | Speedup vs. Baseline A |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline A: Uncapped CPU Judge (`think=True`, ~436 tok/item)** | 363.0 s | 4,360 | **10/10 (100%)** | 0.0% | 1.00x |
| **Baseline B: Naive Capped CPU Judge (`think=True`, `max_tokens=150`)** | 143.5 s | 1,500 | 2/10 (20%) | 65.6% | 2.53x |
| **Baseline C: Verdict-First CPU Judge (`think=False`, `max_tokens=60`)** | 59.0 s | 600 | **10/10 (100%)** | 86.2% | 6.15x |
| **`LogitGate(preset="fast_triage") >> JudgeGate(max_tokens=60)`** | **22.0 s** | **60** | **10/10 (100%)** | **98.6%** | **16.50x** |

*Key finding*: Why did naive token-capping (`Baseline B`, `max_tokens=150`) fail with 20% accuracy? Because modern reasoning models (`gemma4`, `deepseek-r1`, `qwen3`) emit 300+ hidden `<think>` tokens before writing a single visible character. By suppressing `<think>` in `OllamaBackend.generate` and using Verdict-First prompting (`[VERDICT: PASS/FAIL]` on Line 1 for short caps $\le 80$ tokens), `JudgeGate` produces a complete 60-token diagnostic in ~5.9s per item on CPU (`Baseline C`). Placing `ASTGate >> LogitGate` in front of `JudgeGate` reduces the 10-item suite from **363.0 s (`Baseline A`)** and **59.0 s (`Baseline C`)** down to **22.0 s** (**16.50x faster than `Baseline A`**, **2.68x faster than `Baseline C`**, and **90.0% fewer output tokens than `Baseline C`**) while maintaining **10/10 (100%) accuracy**.

---

### 3. Empirical Layer Resolution Breakdown (24-Item Multi-Domain Suite)

Evaluated across the 24-item multi-domain benchmark suite ([`benchmarks/data/multi_domain_24.json`](benchmarks/data/multi_domain_24.json)):

| Model Configuration | Layer 0 (`ASTGate`) | Layer 1 Decisive (`LogitGate`) | Layer 2 Escalated (`JudgeGate`) | Combined L0 + L1 Resolution Rate |
| :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`gemma4:e4b` (Pure CPU, 0 GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`qwen2.5:14b` (GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`qwen2.5-coder:14b` (GPU)** | 2/24 (8.3%) | 21/24 (87.5%) | 1/24 (4.2%) | **23/24 (95.8%)** |

---

### 4. `ChoiceGate` (Multi-Way Routing) & `ScoreGate` (Continuous 1–5 Scoring) Benchmarks

Evaluated on the 12-item routing and scoring suite ([`benchmarks/data/choice_and_score_12.json`](benchmarks/data/choice_and_score_12.json), comprising 6 three-way operational/architectural routing cases with cross-domain distractors and 6 continuous 1–5 rubric scoring cases across High, Medium, and Low quality tiers):

| Model Architecture | `ChoiceGate` Accuracy (2-Pass) | `ChoiceGate` Latency ($N=5$) | `ScoreGate` Mean (High Tier) | `ScoreGate` Mean (Medium Tier) | `ScoreGate` Mean (Low Tier) | `ScoreGate` Pairwise Rank Accuracy | `ScoreGate` Latency ($N=5$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (4B General)** | **6/6 (100.0%)** | 117.9 ± 2.4 ms | 4.96 | 2.15 | 1.01 | **6/6 (100.0%)** | 167.8 ± 1.2 ms |
| **`qwen2.5:14b` (14B General)** | **6/6 (100.0%)** | 319.1 ± 2.4 ms | 5.00 | 2.96 | 1.00 | **6/6 (100.0%)** | 492.7 ± 7.7 ms |
| **`qwen2.5-coder:14b` (14B Code)** | **6/6 (100.0%)** | 297.2 ± 3.7 ms | 4.28 | 2.61 | 1.43 | **6/6 (100.0%)** | 446.0 ± 16.9 ms |
| **`gemma4:26b` (26B MoE, ~4B act)** | **6/6 (100.0%)** | **101.4 ± 4.3 ms** | 5.00 | 1.98 | 1.00 | **6/6 (100.0%)** | **134.3 ± 1.6 ms** |
| **`gemma4:31b` (31B Dense)** | **6/6 (100.0%)** | 232.2 ± 3.0 ms | 5.00 | 1.64 | 1.00 | **6/6 (100.0%)** | 342.9 ± 3.6 ms |

---

## Falsification Benchmarks: Where Does Layer 1 Break Down & How to Mitigate It?

To establish clear empirical boundaries for first-token logit probing, we evaluated `LogitGate` across **5 model architectures** (`4B`, `14B General`, `14B Code`, `26B MoE`, `31B Dense`), **4 reasoning depth tiers** (32 items total, 8 cases per tier in [`benchmarks/data/reasoning_tiers_32.json`](benchmarks/data/reasoning_tiers_32.json)), **document lengths** (200 to 16,000 tokens across a 12-section enterprise handbook), and **context window allocations** (`num_ctx` from 512 to 131,072).

### 1. Reasoning Depth & Circuit-Depth Boundary (32-Item Suite, 8 Cases per Tier)

A causal transformer evaluates its first token in a single forward pass through $L$ attention layers. This corresponds to a sequential circuit depth of $O(1)$. When a verification task requires sequential recurrence, multi-hop state tracking, scratchpad arithmetic, or resistance against adversarial distractors, first-token probing encounters an architectural boundary:

- **Tier A — Direct Semantic / Format / Policy Compliance (Single-Hop, 8 Cases: 4 Pass, 4 Fail)**:
  RAG groundedness checks (including subtle scope and regional extrapolations), customer support 2FA/CVV policy enforcement, and SQL read-only CTE safety.
- **Tier B — Nuanced Partial Compliance (Compound 4-Part Rubric vs. Atomic Batch, 8 Cases: 4 Pass, 4 Fail)**:
  Testing 4 simultaneous Python engineering rules: (1) valid syntax, (2) Google docstrings with `Args:` and `Returns:`, (3) full parameter and return type annotations, and (4) absence of mutable default arguments (`=[]`, `={}`) and bare `except:` clauses. Failing cases satisfy 3 of the 4 rules and subtly violate only 1 (e.g. missing return annotation `-> float`, mutable list default, or missing `Args:` section).
- **Tier C — Multi-Hop Arithmetic & Algorithmic Loop Invariants (8 Cases: 4 Pass, 4 Fail)**:
  Matched pass/fail counterparts with zero inline comments: multi-item invoice arithmetic (`$92.52` vs `$96.52`), binary search termination (`high = mid - 1` vs `high = mid` infinite loop), sliding window loop bounds (`len(series) - window_size + 1` vs off-by-one `len(series) - window_size`), and half-open interval overlap logic (`max(a_start, b_start) < min(a_end, b_end)` vs `a_start < b_end and a_end < b_start`).
- **Tier D — Adversarial Distractors, Deceptive Docstrings & Self-Correction (8 Cases: 4 Pass, 4 Fail)**:
  Zero inline giveaway comments: a non-atomic cache class whose docstring claims `"Formally verified lock-free concurrent cache"`, an invoice with a \$10 arithmetic error accompanied by `"AUDIT STAMP: All arithmetic sums verified by BillingEngine v4.2"`, an SQL query function claiming `"hardened, SQL-injection-safe lookup"` that interpolates `ORDER BY {order_by}`, a payment function that swallows exceptions and logs `"Settlement pipeline completed"`, plus passing cases with self-correcting support transcripts and runbooks listing forbidden `DROP TABLE` commands inside a policy table.

| Model Architecture | Tier A (Direct) | Tier B (Compound) | Tier B (Atomic Batch) | Tier C (Multi-Hop) | Tier D (Adversarial) | Cascade Accuracy (Compound B) | Cascade Accuracy (Atomic B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (4B General)** | **8/8 (100.0%)** | 6/8 (75.0%) | 7/8 (87.5%) | 7/8 (87.5%) | 6/8 (75.0%) | 27/32 (84.4%) | 28/32 (87.5%) |
| **`qwen2.5:14b` (14B General)** | **8/8 (100.0%)** | 7/8 (87.5%) | 7/8 (87.5%) | 6/8 (75.0%) | 7/8 (87.5%) | 28/32 (87.5%) | 28/32 (87.5%) |
| **`qwen2.5-coder:14b` (14B Code)** | **8/8 (100.0%)** | 7/8 (87.5%) | **8/8 (100.0%)** | 7/8 (87.5%) | 7/8 (87.5%) | 29/32 (90.6%) | **30/32 (93.8%)** |
| **`gemma4:26b` (26B MoE, ~4B act)** | **8/8 (100.0%)** | **8/8 (100.0%)** | **8/8 (100.0%)** | 7/8 (87.5%) | 7/8 (87.5%) | **30/32 (93.8%)** | **30/32 (93.8%)** |
| **`gemma4:31b` (31B Dense)** | **8/8 (100.0%)** | **8/8 (100.0%)** | **8/8 (100.0%)** | 7/8 (87.5%) | 7/8 (87.5%) | **30/32 (93.8%)** | **30/32 (93.8%)** |

#### Crucial Insights & Mitigation Strategies:
1. **Model Capacity & Atomic Decomposition on Compound Rubrics (`Tier B`)**:
   - While the 26B MoE and 31B Dense models achieved **8/8 (100.0%)** even on the monolithic 4-part compound rubric, smaller 4B and 14B models suffered from rubric dilution when 3 of 4 requirements passed and only 1 subtle requirement failed (such as an omitted `-> float` return annotation on a function whose parameters were fully typed).
   - Decomposing the 4-part rubric into atomic rubrics via `evaluate_batch(candidate, rubrics)` lifted `gemma4:e4b` from `6/8 (75.0%)` to `7/8 (87.5%)` and `qwen2.5-coder:14b` from `7/8 (87.5%)` to **8/8 (100.0%)** (where atomic rule #3 produced an ambiguous probability $P=0.81$ that escalated to `JudgeGate`, which caught the missing return type annotation).
2. **Reason-Then-Verdict Escalation in `JudgeGate` (`Tier C` & `Tier D`)**:
   - Even with zero inline comments, Layer 1 (`LogitGate`) accurately detected the binary search infinite loop (`high = mid`), sliding window off-by-one (`range(len(series) - window_size)`), interval overlap bug, deceptive thread-safe cache docstring, and deceptive SQL `ORDER BY` injection across almost all models.
   - However, when presented with the multi-line invoice containing a plausible arithmetic error (`17 * $24.00 + $19.00 = $417.00` instead of `$427.00`) accompanied by `"AUDIT STAMP: All arithmetic sums verified by BillingEngine v4.2"`, **all five models awarded $P \ge 0.94$ false passes at Layer 1**. A single forward pass ($O(1)$ circuit depth) cannot execute multi-digit arithmetic in the presence of authoritative surface distractors.
   - On `tier_c_2_invoice_math_fail` (where no adversarial audit stamp was present), `qwen2.5-coder:14b` produced an ambiguous Layer 1 probability ($P=0.5824$), triggering Layer 2 (`JudgeGate`). Because `JudgeGate` instructs the model to verify calculations step-by-step in bullet points before emitting `[VERDICT: PASS/FAIL]` on the final line, `JudgeGate` computed `$108.85 - $16.33 = $92.52 \ne $96.52` and rejected the faulty invoice.
   - **Recommended Mitigations**: Use `evaluate_batch` for multi-part rubrics, configure `preset="strict"` (`pass_threshold=0.90, fail_threshold=0.25`) to widen the Layer 2 escalation band for subtle logic checks, and route exact arithmetic or schema assertions to `ASTGate` / `ContractGate` validators.

---

### 2. Context Length Scaling & Diverse Multi-Section Handbook Haystack (200 to 16,000 Tokens)

Rather than repeating a single boilerplate paragraph, [`benchmarks/data/haystack_handbook.json`](benchmarks/data/haystack_handbook.json) constructs a realistic 12-section enterprise architecture handbook (covering VPC routing, `config.yaml` schema validation, OIDC federation, HashiCorp Vault credential injection, PostgreSQL HA, PII log redaction, and break-glass procedures—where multiple surrounding sections legitimately discuss `config.yaml`, passwords, and emergency maintenance). We embedded a 1-paragraph policy addendum at the midpoint of the document (either strictly prohibiting hardcoded credentials in `config.yaml` or permitting a fallback password during maintenance) and evaluated `"Does this architecture document strictly prohibit hardcoding plaintext passwords or credentials in config.yaml under all circumstances (including emergency maintenance)?"`:

| Document Length | `gemma4:e4b` Latency | `gemma4:e4b` $\Delta P$ | `qwen2.5:14b` Latency | `qwen2.5:14b` $\Delta P$ | `gemma4:26b` Latency | `gemma4:26b` $\Delta P$ | `gemma4:31b` Latency | `gemma4:31b` $\Delta P$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Short (~200 tok)** | 280.1 ms | 1.0000 | 434.6 ms | 1.0000 | **207.2 ms** | 1.0000 | 738.0 ms | 1.0000 |
| **Medium (~1,500 tok)** | 467.9 ms | 0.9985 | 1,366.2 ms | 1.0000 | **524.6 ms** | 1.0000 | 1,807.9 ms | 1.0000 |
| **Long (~4,500 tok)** | 1,128.1 ms | 0.9506 | 2,611.5 ms | 1.0000 | **827.6 ms** | 1.0000 | 2,858.9 ms | 1.0000 |
| **Very Long (~8,000 tok)** | 1,985.5 ms | 0.9621 | 4,774.7 ms | 1.0000 | **1,229.7 ms** | 1.0000 | 4,406.8 ms | 1.0000 |
| **Massive (~16,000 tok)** | 4,044.0 ms | 0.9448 | 11,122.8 ms | 1.0000 | **2,118.3 ms** | 1.0000 | 7,621.9 ms | 1.0000 |

#### Key Architectural Findings:
- **Realistic Attention Dilution on 4B vs. 14B–31B Models**: In a diverse technical haystack where surrounding sections also mention `config.yaml` and Vault credentials, the 4B model (`gemma4:e4b`) exhibits gradual probability attenuation from $\Delta P = 1.0000$ at 200 tokens down to $\Delta P = 0.9448$ ($P_{\text{compliant}} = 0.9448$, $P_{\text{violating}} = 0.0000$) at 16,000 tokens—still well above the `0.85` pass threshold, while 14B, 26B MoE, and 31B Dense models maintain $\Delta P = 1.0000$ across all lengths.
- **MoE vs. Dense Pre-Fill Advantage**: At 16,000 tokens, **`gemma4:26b` (MoE)** processed the prompt in **2,118.3 ms**—**3.60x faster than `gemma4:31b` (Dense, 7,621.9 ms)** and **5.25x faster than `qwen2.5:14b` (11,122.8 ms)** because MoE routes each token through ~4B active parameters while retaining 26B capacity.

---

### 3. Context Window Squeezing (`num_ctx`) & Hardware Scaling Sweep (512 to 65,536 Tokens)

Configuring the context window size via `LOGITGATE_NUM_CTX` or `logitgate.configure(num_ctx=...)` allows developers to control KV-cache allocation and prevent GPU-to-CPU memory spillover:

#### Allocated Memory Footprint by Context Window (`num_ctx`)

| Model & Hardware Configuration | `num_ctx=512` | `num_ctx=4,096` | `num_ctx=16,384` | `num_ctx=32,768` | `num_ctx=65,536` | Spillover Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (RTX 4070 12 GB)\*** | 9.41 GB VRAM | 9.41 GB VRAM | 9.45 GB VRAM | 9.54 GB VRAM | 9.71 GB VRAM | **100% GPU up to 131,072 tokens** |
| **`gemma4:e4b` (Ryzen 7 5800X3D CPU)\*** | 9.41 GB RAM | 9.41 GB RAM | 9.57 GB RAM | 9.60 GB RAM | 9.60 GB RAM | **Zero VRAM footprint** |
| **`qwen2.5:14b` (RTX 4070 12 GB)** | 8.71 GB VRAM | 9.09 GB VRAM | 10.39 GB VRAM | 10.33 GB VRAM + 2.40 GB RAM | 10.33 GB VRAM + 2.40 GB RAM | **Spills to CPU at $\ge$ 32,768 (>12 GB)** |
| **`gemma4:26b` (MoE, RTX 5090 32 GB)** | 17.42 GB VRAM | 17.69 GB VRAM | 17.72 GB VRAM | 17.84 GB VRAM | 18.18 GB VRAM | **100% GPU up to 65,536 tokens** |
| **`gemma4:31b` (Dense, RTX 5090 32 GB)** | 20.97 GB VRAM | 21.50 GB VRAM | 21.71 GB VRAM | 21.46 GB VRAM | 22.00 GB VRAM | **100% GPU up to 65,536 tokens** |

#### Warm 2-Pass Symmetric Latency by Context Window (`num_ctx`)

| Model & Hardware Configuration | `num_ctx=512` | `num_ctx=4,096` | `num_ctx=16,384` | `num_ctx=32,768` | `num_ctx=65,536` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (RTX 4070 12 GB)\*** | 92.7 ms | 94.5 ms | 101.9 ms | 94.4 ms | 94.5 ms |
| **`gemma4:e4b` (Ryzen 7 5800X3D CPU)\*** | 2,150.0 ms | 2,170.0 ms | 2,200.0 ms | 2,130.0 ms | 2,450.0 ms |
| **`qwen2.5:14b` (RTX 4070 12 GB)** | 94.4 ms | 93.4 ms | 99.7 ms | 314.5 ms | 314.3 ms |
| **`gemma4:26b` (MoE, RTX 5090 32 GB)** | 92.4 ms | 89.9 ms | 86.4 ms | 90.3 ms | 87.9 ms |
| **`gemma4:31b` (Dense, RTX 5090 32 GB)** | 194.8 ms | 184.3 ms | 177.4 ms | 175.1 ms | 182.1 ms |

*\*Note on `gemma4:e4b` memory footprint & 128k scaling: Google's Gemma 4 `e4b` ("Effective 4B") packs `8.0B` total parameters (`8.95 GB` Q4_K_M GGUF weights, including 42-layer Per-Layer Embeddings plus 16-layer Vision and 12-layer Audio encoders, with ~4B active text parameters). At `num_ctx=131,072` (128k tokens), `gemma4:e4b` allocates `10.04 GB` VRAM (`95.9 ms` warm latency), still fitting 100% inside a 12 GB GPU. Text-only 3B–4B Q4 models (`qwen2.5:3b`, `llama3.2:3b`) require only `~2.5–3.2 GB` VRAM/RAM.*

> [!TIP]
> **Why `LogitGate` Survives GPU-to-CPU RAM Spillover (~314 ms vs. 80+ Seconds)**:
> On `qwen2.5:14b` (which uses full global attention across all 48 layers), raising `num_ctx` to `32,768` spills `2.4 GB` of KV-cache into System RAM over PCIe, increasing 2-pass latency from **99.7 ms to 314.5 ms**. In traditional generative LLM-as-a-Judge, that PCIe transfer penalty is paid **sequentially on every generated token** ($200\text{ ms} \times 400\text{ tokens} \approx 80\text{ seconds}$). Because `LogitGate` only evaluates the first-token forward pass (`max_tokens=1`), the PCIe spillover penalty is paid only during prefill—allowing users to evaluate 16k–32k documents in **~0.31 seconds** even when KV-cache spills into System RAM.
> Furthermore, hybrid sliding-window architectures like **`gemma4:e4b` and `gemma4:26b` (MoE)** grow their KV-cache by only **`+0.63 GB` to `+0.76 GB` up to `64k–128k` context**, staying in VRAM at **~88–96 ms**.

---

### 4. Reason-Then-Verdict & Self-Healing Logit Probe in `JudgeGate`

In traditional LLM-as-a-judge frameworks, evaluators often ask for a verdict before reasoning or rely on naive regex/substring matching (`if "FAIL" in text.upper()`). Asking a non-thinking model to emit its verdict on Line 1 forces it to guess before performing step-by-step arithmetic or logic verification, while naive substring matching triggers false failures when a model writes `"The candidate does not fail any requirements"`.

`JudgeGate` handles Layer 2 verdict extraction in three steps:
1. **Reason-Then-Verdict Prompting**: On standard calls, `JudgeGate` instructs the model to verify each requirement or calculation step-by-step in 2–4 concise bullet points first, and emit `[VERDICT: PASS]` or `[VERDICT: FAIL]` on the concluding line (matching the final tag via `re.finditer`). For ultra-short CPU caps (`max_tokens <= 80`), it switches to Verdict-First on Line 1 so the tag is never truncated.
2. **Self-Healing 1-Token Logit Probe**: If an upstream model omits the explicit `[VERDICT: ...]` tag or JSON field, **`JudgeGate` runs a 1-token `LogitGate` probe on its own generated critique text** before falling back to heuristics:
   ```python
   # Inside JudgeGate._parse_verdict:
   probe_gate = LogitGate(
       rubric="Does this review critique conclude that the candidate artifact PASSED the evaluation?",
       backend=self.backend,
       symmetric_debiasing=False,
   )
   is_pass = probe_gate.evaluate(critique_text).probability >= 0.5
   ```
3. **Negation-Safe Line Fallback**: If the backend does not implement `LogitBackend`, it falls back to negation-safe parsing of concluding lines.

---

### 5. Summary: When to Use Layer 1 (`LogitGate`) vs. Escalate to Layer 2 (`JudgeGate`)

- **Use Layer 1 (`LogitGate`) — First-Token Forward Pass ($O(1)$ Circuit Depth)**:
  - **Semantic Rubric & Policy Compliance**: Factual groundedness against context, tone/safety compliance, PII/secret leak checks, and atomic requirement verification via `evaluate_batch`.
  - **Discrete Routing & Classification (`ChoiceGate`)**: Selecting downstream agent tools, intent buckets, or workflow branches in ~40–90 ms.
  - **Continuous Rubric Scoring (`ScoreGate`)**: Probability-weighted expected value scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$) with symmetric scale reversal.
- **Escalate to Layer 2 (`JudgeGate`) — Autoregressive Generation**:
  - **Multi-Step Deductive Reasoning**: Proof verification or multi-hop logic where conclusions depend on intermediate reasoning steps.
  - **Complex Symbolic Arithmetic**: Calculations requiring scratchpad calculation or token-by-token recurrence (or route deterministic math checks to Layer 0 `ASTGate` / `ContractGate`).
  - **Diagnostic Reporting**: When a human developer or logging pipeline needs an actionable explanation of *why* an artifact failed (or replay a CI failure locally via `logitgate explain`).

---

## System Requirements & Hardware Tiers

`LogitGate` adapts dynamically across hardware environments:

| Hardware Tier | Recommended Model | Backend Specifier / Config | Typical Single-Pass Latency | Target Workflows |
| :--- | :--- | :--- | :--- | :--- |
| **0-GPU CI Runners / Air-Gapped** | Offline Deterministic Stub | `stub://default` | **< 0.05 ms** | Unit tests, mock suites, CI smoke tests |
| **CPU-Only Laptops / Edge** | `gemma4:e4b` / `llama3.2:3b` | `ollama://localhost:11434/gemma4:e4b` (`options={"num_gpu": 0}`) | **~170 ms** | Local developer testing without discrete GPU |
| **8–12 GB VRAM Consumer GPU** | `qwen2.5:7b` / `gemma4:e4b` | `ollama://localhost:11434/gemma4:e4b` | **~40–45 ms** | Desktop development, real-time agent loops |
| **16–24 GB VRAM Workstation** | `qwen2.5-coder:14b` / `qwen2.5:14b` | `ollama://localhost:11434/qwen2.5-coder:14b` | **~40–50 ms** | Production evaluation suites, coding agents |
| **Cloud Inference / Hosted APIs** | `gpt-4o-mini` / `vLLM` | `openai://api.openai.com/v1` or `vllm://localhost:8000/v1` | **~60–120 ms** (network bounded) | High-concurrency enterprise pipelines |

### 3 Deployment Topologies: Preventing KV-Cache Eviction & Thrashing

When deploying LLM evaluation pipelines, a common production failure mode is **KV-Cache Thrashing**: when a primary generation model (e.g., an agent synthesizing code) is interrupted by an evaluation judge loading into the same VRAM or context window, evicting the primary generator's KV-cache and forcing expensive full recomputation.

`LogitGate` supports three deployment topologies designed to prevent cache eviction:

1. **Topology 1: Single-Model Shared KV-Cache (Batch Evaluation Suites)**
   - Used when evaluating multiple criteria on the same candidate artifact.
   - Using `evaluate_batch(candidate, rubrics)`, the candidate tokens are evaluated once. Subsequent rubric logit probes reuse the shared `[EVALUATION RUBRICS] + [CANDIDATE ARTIFACT]` prompt prefix in the KV-cache without evicting the context, reducing per-rubric latency to ~110ms.
2. **Topology 2: Heterogeneous Dual-GPU Sidecar (High-Throughput Production)**
   - **GPU 0**: Pinned exclusively to the primary generator model (e.g., `gemma4:26b` MoE or `gemma4:31b` Dense) with 16k–32k context. Its generation KV-cache is **never evicted**.
   - **GPU 1**: Dedicated to a lightweight `LogitGate` sidecar (`qwen2.5:14b` or `gemma4:e4b` in ~8.7–9.4 GB VRAM, or `llama3.2:3b` in ~3.2 GB VRAM).
   - Real-time gating queries complete in ~40–50ms (single-pass) or ~85–120ms (2-pass symmetric) over local IPC (`http://localhost:11435`) without contending for GPU 0's memory bandwidth.
3. **Topology 3: Single-GPU Workstation + Pure-CPU `LogitGate` Sidecar (Memory-Constrained)**
   - 100% of GPU VRAM is preserved for the primary generator model.
   - Layer 1 `LogitGate` runs on host CPU with `options={"num_gpu": 0, "num_ctx": 2048}` in ~170ms (single-pass) using standard System RAM (~3.2 GB for 3B Q4 models like `qwen2.5:3b` / `llama3.2:3b`, or ~9.4 GB for `gemma4:e4b`).
   - Clear passes short-circuit without GPU VRAM interference. Only ambiguous items or failures escalate to the GPU-backed `JudgeGate`.

---

## Role-Based Setup & Use-Case Guides

LogitGate is designed to fit directly into diverse software engineering and AI workflows across general and non-SDLC domains:

### 1. CI/CD Final Merge Gate vs. Local Developer Inner Loop (Speed vs. Correctness)

In production engineering, **local iteration** and **CI/CD merge gates** have different priorities:
- **CI/CD is your final quality gate**: Correctness and minimizing false passes are paramount. However, you do not need to wait 5–20 seconds per test to generate a 400-token LLM essay when Layer 1 already reached a decisive, non-ambiguous conclusion (`P >= 0.85` pass or `P <= 0.20` fail).
- **Local execution is your fast inner loop**: Developers iterating on code or live agent routers want ~40–90ms GPU feedback (or ~170ms 1-pass triage on CPU-only laptops).

| Dimension | Local Inner-Loop / Live Agent Triage | CI/CD Final Merge Gate (Recommended) |
| :--- | :--- | :--- |
| **Primary Goal** | Fast developer feedback / low-latency routing | **High correctness & minimal false passes** without waiting for LLM essays on clear verdicts |
| **Preset & Debiasing** | `preset="balanced"` on GPU (`~86ms`), or `preset="fast_triage"` (`1-pass`, `~170ms`) on CPU-only laptops | **`preset="balanced"` or `"strict"` (`symmetric_debiasing=True`)** — always cancel prior token bias |
| **Multi-Part Rubrics** | Single rubric or `ChoiceGate` | **Atomic rubrics via `evaluate_batch(candidate, rubrics)`** (shared-prefix KV-cache reuse, avoids compound dilution) |
| **Layer 2 (`JudgeGate`) Trigger** | Omit `JudgeGate`, or `trigger="on_ambiguous_only"` | **`trigger="on_ambiguous_only"` + `save_on_fail=True`** (0 CoT tokens on decisive pass *or* decisive fail; only invokes `JudgeGate` to adjudicate ambiguous borderline cases) |

```python
from logitgate import ASTGate, LogitGate, JudgeGate, OllamaBackend

# Pure CPU CI Runner configuration (0 GPU, ~3.2 GB RAM for 3B models / ~9.4 GB RAM for gemma4:e4b)
backend = OllamaBackend(
    model="gemma4:e4b",
    options={"num_gpu": 0, "num_thread": 4},
    keep_alive="5m",
)

# Correctness-first CI Merge Gate:
# - Layer 0 catches syntax/import errors in <0.1ms (no LLM call)
# - Layer 1 runs full 2-pass symmetric debiasing (no CoT tokens on clear pass OR clear fail)
# - Layer 2 ONLY fires when Layer 1 is ambiguous (0.20 < P < 0.85)
ci_pipeline = (
    ASTGate(syntax=True, forbidden_imports=["pickle", "subprocess"])
    >> LogitGate(
        rubric="Candidate code must be clean, secure, and handle edge cases.",
        preset="balanced",  # Full 2-pass symmetric log-odds debiasing for CI correctness
        save_on_fail=True,  # Saves .logitgate/failures/*.json on fail for local `logitgate explain`
        backend=backend,
    )
    >> JudgeGate(
        max_tokens=80,
        trigger="on_ambiguous_only",  # Skip generative essays on non-ambiguous verdicts
        backend=backend,
    )
)

candidate_code = "def add_numbers(a: int, b: int) -> int:\n    return a + b\n"
verdict = ci_pipeline.evaluate(candidate_code)
assert verdict.passed, f"CI Gate Failed at {verdict.triggered_layer}: {verdict.reason}"
```
- **Resource Footprint**: Runs in ~2.2–3.2 GB System RAM (`qwen2.5:3b` / `llama3.2:3b`) or ~9.4 GB System RAM (`gemma4:e4b`) with 0 VRAM overhead—fitting comfortably inside a standard 16 GB GitHub Actions (`ubuntu-latest`) VM.
- **Cost Savings**: Runs on standard CPU CI runners without requiring \$0.50–\$2.00/hr GPU instances.

#### Reproducing Non-Deterministic App Failures Locally (`VerdictSnapshot` & `logitgate explain`)

When testing non-deterministic LLM applications in CI using `trigger="on_ambiguous_only"` or `@check`, clear failures fail immediately at Layer 1 without generating a long critique in CI. However, if a developer re-runs their non-deterministic application locally to debug, the LLM may generate a different output, making reproduction difficult.

LogitGate solves this with deterministic **Verdict Snapshots**:
1. When a test fails in CI (with `LOGITGATE_SAVE_FAILURES=1` or `@check(save_snapshot=True)`), LogitGate freezes the exact candidate output, rubric, and telemetry to `.logitgate/failures/fail_<timestamp>_<test>_<id>.json` (and updates `.logitgate/failures/latest.json`).
2. Developers can replay and diagnose the exact failure locally with a single CLI command without re-running their non-deterministic application:

```bash
# Replay and explain the latest failure with full System-2 reasoning
logitgate explain

# Or explain a specific failure snapshot with custom tokens
logitgate explain .logitgate/failures/fail_20261102_150648_test_rag_a1b2c3d4.json --max-tokens 150
```

Programmatic replay and inspection via Python:
```python
from logitgate import VerdictSnapshot

# Load frozen snapshot from CI
snapshot = VerdictSnapshot.load(".logitgate/failures/latest.json")

# Re-run fast Layer-1 logit evaluation on the frozen candidate
replay_verdict = snapshot.replay()

# Or generate a deep Layer-2 diagnostic critique with reasoning enabled
explanation = snapshot.explain(model="qwen2.5-coder:14b", think=True)
print(explanation.reason)
```

---

### 2. RAG & Chatbot Engineers (Hallucination & Groundedness Gating)
Detect hallucinations and verify factual groundedness before streaming an answer to the end user:
```python
from logitgate import ContractGate, LogitGate

retrieved_docs = "LogitGate v0.1.0 requires Python 3.10 or higher and supports Ollama, vLLM, and OpenAI."
generated_response = "LogitGate v0.1.0 supports Python 3.10+ and runs on Ollama, vLLM, and OpenAI."

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
Enforce structured JSON output contracts in **<0.1ms** and perform ~40–90ms conditional routing:
```python
from pydantic import BaseModel
from logitgate import ContractGate, ChoiceGate

class ToolCallPayload(BaseModel):
    tool_name: str
    arguments: dict
    confirm_action: bool

# Layer 0 validates JSON schema in <0.1ms; Layer 1 routes intent in ~40-90ms
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

agent_raw_json = '{"tool_name": "run_sql", "arguments": {"query": "SELECT count(*) FROM users"}, "confirm_action": false}'
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

support_transcript = (
    "Customer: Please update my account email and process a refund for order #402.\n"
    "Agent: I would be happy to help! Before I can update your email, please complete "
    "the 2FA verification sent to your phone. Your refund for order #402 has been initiated "
    "per our 30-day return policy."
)

# Evaluates all 3 rubrics in a single batch reusing the shared prompt prefix in KV-cache
verdicts = evaluator.evaluate_batch(support_transcript, rubrics)
for r, v in zip(rubrics, verdicts):
    print(f"[{'PASS' if v.passed else 'FAIL'}] P={v.probability:.2f} | {r}")
```

---

## Best Practices: Prompting, Context Length, & Model Sizing

### 1. Document Context Length & When to Chunk
- **Zero-Chunking Sweet Spot (Up to 16,000–32,000 Tokens)**: As shown in our [Needle-in-a-Haystack benchmarks](#2-context-length-scaling--needle-in-a-haystack-boundary-200-to-16000-tokens), modern causal transformers (`gemma4`, `qwen2.5`) maintain strong probability separation ($\Delta P > 0.993$) up to **16,000+ tokens** without manual chunking, and sliding-window/MoE architectures scale cleanly to **32k–64k tokens** in VRAM.
- **When to Chunk (50,000+ Token Codebases or Books)**: If a document exceeds your model's configured context window (`num_ctx`) or exceeds 32k–50k tokens on a dense attention model, split the document into logical sections or files and call `gate.evaluate(chunk)` on each section independently.
- **Multi-Rubric KV Sharing (`evaluate_batch`)**: Note that `gate.evaluate_batch(candidate, rubrics)` is designed for evaluating **multiple atomic rubrics against a single candidate artifact** (sharing the `[EVALUATION RUBRICS] + [CANDIDATE ARTIFACT]` prefix across all $K$ rubrics), rather than evaluating multiple documents against one rubric.

### 2. Writing Atomic, Observable Rubrics vs. Giant Compound Prompts
- **Single Observable Property**: Write rubrics that test one clear, falsifiable property (e.g. *"Does the reply refuse to reset passwords without 2FA?"* rather than *"Is the reply polite, compliant with 2FA, formatted as markdown, and under 200 words?"*).
- **Prefix-Shared Batching**: Instead of one monolithic prompt, pass a list of atomic rubrics to `evaluate_batch(candidate, rubrics)`. Because all rubrics share the candidate prefix, running 4–5 atomic rubrics takes ~110ms per rubric while avoiding compound rubric dilution and giving per-criterion observability.

### 3. Escalation Mechanics: How Symmetric Log-Odds Handle Model Uncertainty
- **Decisive Signal**: When a model is confident that an artifact passes, both forward ($L_1 = s + b$) and reverse ($L_2 = s - b$) log-odds are positive, yielding calibrated probability $P \ge 0.85$ (fast-path pass, no Layer 2 critique).
- **Decisive Failure**: When an artifact clearly violates the rubric, both forward and reverse log-odds are negative, yielding $P \le 0.20$ (decisive failure).
- **Ambiguity & Escalation**: When small models (e.g. 3B–4B) or complex edge cases yield borderline log-odds (between 0.20 and 0.85), LogitGate flags the item as `AMBIGUOUS` and triggers Layer 2 (`JudgeGate`) to generate an autoregressive critique.

---

## Research Foundations & Academic Citations

LogitGate builds upon established methods in calibration, logit debiasing, and LLM evaluation:

### Key Design Decisions in `LogitGate`
1. **Shared-Model 3-Layer Cascade**: Uses the existing generator model instance for both gating and diagnostics without loading a dedicated secondary judge model into VRAM.
2. **Prefix-Preserving KV Compilation (`compile_batch_binary_prompt`)**: Arranges prompts so the candidate artifact and evaluation rubrics form an identical static prefix across forward and reverse debiasing passes, retaining the prefix in the KV-cache across queries.
3. **Log-Odds Space Symmetric Inversion**: Cancels additive prior token bias in log-odds space ($\bar{L} = \frac{L_1 + L_2}{2T}$) rather than probability averaging.
4. **Missing-Token Dynamic Floor**: Bounds unobserved token logprobs using residual probability mass $\min(\min_i \text{logprob}_i, \ln(1 - \sum P_i))$, avoiding distortion from arbitrary $-99.0$ constants.

### Foundational Literature Credited

1. **Pretraining Token Prior Bias & Calibration**:
   - Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). *"Calibrate Before Use: Improving Few-Shot Performance of Language Models"*. **ICML 2021**. [arXiv:2102.09690](https://arxiv.org/abs/2102.09690).
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
   - Deußer, T., Sparrenberg, L., & Sifa, R. (2026). *"Evaluating and Benchmarking the System One Model Jev"*. [arXiv:2609.37647](https://arxiv.org/abs/2609.37647).
   - *Contribution to LogitGate*: Benchmarks fast, single-token System-1 evaluation pipelines against heavy System-2 generative reasoning judges.

### BibTeX Citation

```bibtex
@software{logitgate2026,
  author = {Jason Bradley},
  title = {LogitGate: Calibrated First-Token Logit Evaluation and Cascaded Routing for LLMs},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/jasonbdly/LogitGate},
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
A: The candidate artifact complies fully with the rubric.
B: The candidate artifact violates the rubric or has defects.
Respond with a single character (A or B).
Verdict:
```

Because Pass 1 and Pass 2 share an identical prefix up to `[DECISION MAPPING]`, the rubric and candidate context remain in the GPU KV-cache so Pass 2 only evaluates the short `[DECISION MAPPING]` suffix.

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
│                   ASTGate                        ├─────────────────────────────────────────► [FAIL: no LLM call, ~0.04ms]
└────────────────────────┬─────────────────────────┘
                         │ Passed
                         ▼
┌──────────────────────────────────────────────────┐   P >= pass_threshold (>= 0.85)
│                   Layer 1:                       ├─────────────────────────────────────────► [PASS: 0 CoT tokens, ~40–120ms]
│                  LogitGate                       │
└────────────────────────┬─────────────────────────┘   P <= fail_threshold (<= 0.20)
                         │ Ambiguous                   └─────────────────────────────────────► [FAIL: 0 CoT tokens, ~40–120ms]
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

### 3. Multi-Rubric Batch Evaluation with Shared-Prefix KV Caching

Evaluate multiple rubrics against the same candidate artifact while sharing the `[EVALUATION RUBRICS] + [CANDIDATE ARTIFACT]` prefix in the KV-cache:

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

candidate_documentation = """
# Authentication API (v2)
- **Endpoint**: `POST /v2/auth/token`
- **Request Body**: `{"client_id": "string", "client_secret": "string"}`
- **Responses**: `200 OK` (returns JWT + expiry), `401 Unauthorized`, `429 Too Many Requests`.
"""

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
from typing import TypedDict
from langgraph.graph import StateGraph
from logitgate import LogitGate
from logitgate.integrations.langgraph import as_conditional_edge

class AgentState(TypedDict):
    content: str

gate = LogitGate(rubric="Is the generated response safe and helpful?")

# Creates a fast routing function extracting 'content' from graph state
route_verdict = as_conditional_edge(gate, pass_node="publish_response", fail_node="human_escalation")

workflow = StateGraph(AgentState)
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
