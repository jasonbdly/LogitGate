[🏠 README](../README.md) · [📚 Docs Hub](./README.md) · [⚡ Quick Notes](./quick/quickstart_recipes.md) · [🔬 Deep Architecture](./architecture.md) · [🌱 Visual ELI5](./eli5/how_logitgate_works.md) · [🚀 Power-User](./advanced/distillation_and_lora.md)

---

# Empirical Benchmarks & Falsification Studies

This document contains the comprehensive empirical benchmark results, hardware sweeps, thread scaling experiments, and falsification studies for **LogitGate**.

All declarative benchmark datasets ([`benchmarks/data/`](../benchmarks/data/)), the benchmark runner scripts ([`benchmarks/run_benchmarks.py`](../benchmarks/run_benchmarks.py), [`benchmarks/run_cpu_benchmarks.py`](../benchmarks/run_cpu_benchmarks.py), [`benchmarks/run_dory_benchmarks.py`](../benchmarks/run_dory_benchmarks.py), [`benchmarks/run_adversarial_choice_score.py`](../benchmarks/run_adversarial_choice_score.py), [`benchmarks/run_cpu_format_runtime_benchmarks.py`](../benchmarks/run_cpu_format_runtime_benchmarks.py)), and raw JSON results are tracked in the repository for independent reproduction.

---

## Executive Summary (Decision-Grade v0.13.0 - v0.15.0 Results)

Empirical evaluation across dual-GPU local hardware (NVIDIA GeForce RTX 5090 32GB primary tier + RTX 4070 12GB sidecar tier running under Ollama `v0.35.0` and native PyTorch / llama.cpp) and pure 0-GPU CPU runners. Metrics report pure-Python 95% Wilson score confidence intervals, exact two-sided McNemar paired tests, and probabilistic calibration (Brier score, ECE, AUROC):

| Benchmark Suite & Task | $N$ | Hardware Layout | Standalone Layer 1 Acc (95% CI) | Cascade Topology (`L1 >> L2`) | Cascade Acc (95% CI) | Fast-Path Rate (0 Gen Tokens) | Median Latency (p50) | Key Statistical Test & Metric Calibration |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Agent MCP Tool Safety** (`agent_mcp_tool_safety_80.json`) | `80` | RTX 5090 (32 GB) + RTX 4070 (12 GB) | `86.3%` (`69/80`) `[77.0%, 92.2%]` | `ToolCallGate(3B) >> JudgeGate(26B)` | **`98.8%` (`79/80`)** `[93.3%, 99.8%]` | **`73.8%`** (`59/80`) | **`86.4 ms`** (vs `93.6 ms` standalone `3B`) | **McNemar $p = 0.0063$**; Stage-0 catches 70.0% of attacks in `0.08 ms` |
| **Prefix-Cached Grounded RAG** (`rag_groundedness_80.json`) | `80` | RTX 4070 (12 GB Sidecar) | **`98.8%` (`79/80`)** `[93.3%, 99.8%]` | `LogitGate(qwen2.5:14b)` Standalone | **`98.8%` (`79/80`)** `[93.3%, 99.8%]` | **`100.0%`** (`80/80`) | **`384.2 ms`** (`390.5 ms` shared prefix) | **`1.000` AUROC**, `0.0125` Brier, `0.0125` ECE; **2.51x prefix speedup** (`981 ms -> 390 ms`) |
| **Canonical Public RAGTruth** (`RAGTruth-processed`, Zero-Shot) | `60` | RTX 5090 (32 GB) + RTX 4070 (12 GB) | `63.3%` (`38/60`) `[50.7%, 74.3%]` | `LogitGate(14B) >> JudgeGate(26B)` | **`66.7%` (`40/60`)** `[54.1%, 77.3%]` | **`93.3%`** (`56/60`) | **`233.4 ms`** | Uncalibrated zero-shot baseline on complex multi-thousand char medical/finance sources |
| **Logprobs Sidecar Fallback** (`ood_generalization_96.json`) | `96` | RTX 4070 (12 GB Sidecar) | N/A (Black-box primary has no logprobs) | `Sidecar LogitGate(3B)` Fallback | **`86.5%` (`83/96`)** `[78.2%, 91.9%]` | **`100.0%`** (`96/96`) | **`192.3 ms`** (vs `1,031.9 ms` pure escalate) | **0 crashes**, **5.37x faster** than escalating all traffic to generative judge |
| **Multilingual Unicode BPE** (`multilingual_unicode_bpe_80.json`) | `80` | RTX 5090 (32 GB) + RTX 4070 (12 GB) | `63.7%` (`3B`) / `100.0%` (`4B`) | `LogitGate(3B) >> JudgeGate(26B)` | **`98.8%` (`79/80`)** `[93.3%, 99.8%]` | `28.7%` (`3B`) / `100.0%` (`4B`) | `92.4 ms` (`4B`) | Zero cross-split leakage; CJK/Cyrillic/Arabic token boundary parity |
| **Zero-Token Preference Ranking** (`rank_pairwise_80.json`) | `120` | RTX 4070 (12 GB Sidecar) | **`100.0%` (`60/60`)** `[94.0%, 100.0%]` | `RankGate(gemma4:e4b, symmetric=True)` | **`100.0%` (`60/60`)** `[94.0%, 100.0%]` | **`100.0%`** (`60/60`) | **`228.9 ms`** (vs `432.2 ms` generative judge) | **100.0% position-swap consistency**; 0 generated tokens |
| **Multi-Domain Core Suite** (`multi_domain_24.json`) | `24` | RTX 4070 (12 GB Sidecar) | **`100.0%` (`24/24`)** `[86.2%, 100.0%]` | `LogitGate(gemma4:e4b) >> JudgeGate` | **`100.0%` (`24/24`)** `[86.2%, 100.0%]` | **`100.0%`** (`24/24`) | **`41.7 ms`** (vs `1,628 ms` generative judge) | **4.6x wall-clock speedup**; `2,696` CoT tokens saved |
| **Pure-CPU CI/CD Suite** (`options={"num_gpu": 0}`) | `10` | AMD Ryzen 7 5800X3D (0 GPU, 2.09 GB RAM) | `100.0%` (`10/10`) `[72.2%, 100.0%]` | `ASTGate >> LogitGate(3B) >> Judge(60 tok)` | **`100.0%` (`10/10`)** `[72.2%, 100.0%]` | **`90.0%`** (`9/10`) | **`566.8 ms`** (total suite `22.0 s` vs `363.0 s`) | **16.50x total speedup** vs uncapped CPU judge; `98.6%` token reduction |
| **Canonical Public LLMBar** (`Adversarial_Neighbor`) | `60` | RTX 4070 (12 GB Sidecar) | `55.0%` (single-pass) | `RankGate(4B, symmetric=True)` | **`58.3%` (`35/60`)** `[45.7%, 69.9%]` | **`100.0%`** (`60/60`) | **`184.2 ms`** | **100.0% swap consistency** (+25.0 pp over single-pass) at 0 generated tokens |
| **Canonical Public JudgeBench** (`gpt` split) | `60` | RTX 4070 (12 GB Sidecar) | `58.3%` (single-pass) | `RankGate(4B, symmetric=True)` | **`61.7%` (`37/60`)** `[49.0%, 72.9%]` | **`100.0%`** (`60/60`) | **`196.5 ms`** | **93.3% swap consistency** (+21.6 pp over single-pass) on hard math/reasoning pairs |
| **Canonical Public TruthfulQA** (`validation`) | `60` | RTX 4070 (12 GB Sidecar) | `81.7%` (single-pass) | `RankGate(4B, symmetric=True)` | **`80.0%` (`48/60`)** `[68.2%, 88.2%]` | **`100.0%`** (`60/60`) | **`178.9 ms`** | **98.3% swap consistency** on deceptive misconceptions |

### Key Empirical Findings:
1. **Tool-Call & Agent Safety**: `ToolCallGate` paired with a Layer-2 judge resolves `73.8%` of tool evaluations at `0` generated tokens in `86.4 ms` median latency, achieving **`98.8%` accuracy** (`79/80`) compared to `86.3%` standalone 3B (exact McNemar $p = 0.0063$). Stage-0 deterministic sink analysis catches 70.0% of known command injections and shell exploits in **0.08 ms** with zero false positives.
2. **Grounded RAG Prefix-Caching**: Using the dedicated `<reference_context_NONCE>` slot, `MultiRubricGate` amortizes source document prefill across all rubrics, dropping per-rubric evaluation from **`981.3 ms` down to `390.5 ms` (`2.51x` faster)** with perfect AUROC (`1.000`) on `rag_groundedness_80.json`.
3. **Resilience to Logprob Deprivation**: When upstream API providers omit logprobs, capability-aware `on_unsupported_logprobs="sidecar"` routes to a lightweight local sidecar in **192.3 ms**, beating full generative judge escalation (`1,031.9 ms`) by **5.37x** with zero runtime exceptions.

---

## Table of Contents

### Part I: Core Production & Canonical External Benchmarks
- [Executive Summary (Decision-Grade v0.13.0 - v0.15.0 Results)](#executive-summary-decision-grade-v0130---v0150-results)
- [v0.15.0 Market Extension Benchmarks](#v0150-market-extension-benchmarks)
  - [1. ToolCallGate + Cascade on Agent MCP Tool Safety (N=80)](#1-toolcallgate--cascade-on-agent-mcp-tool-safety-n80)
  - [2. First-Class Prefix-Cached Grounded RAG (context= & query=)](#2-first-class-prefix-cached-grounded-rag-context--query)
  - [3. Capability-Aware Sidecar Fallback on Unsupported Logprobs (N=96)](#3-capability-aware-sidecar-fallback-on-unsupported-logprobs-n96)
- [Real Multi-Model Live Benchmarks (5 Model Architectures Across GPU & CPU)](#real-multi-model-live-benchmarks)
- [End-to-End Cascade Time Savings & Multi-Gate Benchmarks](#end-to-end-cascade-time-savings--multi-gate-benchmarks)
  - [1. Multi-Domain Cascade Time Savings (24-Item Test Suite)](#1-multi-domain-cascade-time-savings-24-item-test-suite)
  - [2. Fair Pure-CPU CI/CD Runner Benchmark (10-Item CI Test Suite)](#2-fair-pure-cpu-cicd-runner-benchmark-10-item-ci-test-suite-optionsnum_gpu-0)
  - [3. Empirical Layer Resolution Breakdown (24-Item Multi-Domain Suite)](#3-empirical-layer-resolution-breakdown-24-item-multi-domain-suite)
  - [4. ChoiceGate & ScoreGate Multi-Action Routing Benchmarks](#4-choicegate-multi-way-routing--scoregate-continuous-15-scoring-benchmarks)
- [v0.13.0 Decision-Grade Statistical Engine & Core N=80 Benchmarks](#14-v0130-decision-grade-statistical-engine-deconfounded-n80-core-suites--canonical-external-public-benchmarks)
  - [A. Psychometric Deconfounding & Cross-Split Verification (N=80 Suites)](#a-psychometric-deconfounding--cross-split-leakage-verification-n80-core-suites)
  - [B. Live Dual-GPU Cascade & Student Discrimination Across N=80 Suites](#b-live-dual-gpu-cascade--student-discrimination-across-n80-suites)
  - [C. Live Canonical External Public Benchmarks (LLMBar, JudgeBench, TruthfulQA, RAGTruth)](#c-live-canonical-external-public-benchmarks-benchmarksexternal-zero-token-gemma4e4b)

### Part II: Power-User, Stress-Test & Research Benchmarks
- [Falsification Benchmarks: Where Does Layer 1 Break Down & How to Mitigate It?](#falsification-benchmarks-where-does-layer-1-break-down--how-to-mitigate-it)
  - [1. Reasoning Depth & Circuit-Depth Boundary (32-Item Suite)](#1-reasoning-depth--circuit-depth-boundary-32-item-suite-8-cases-per-tier)
  - [2. Context Length Scaling & Diverse Handbook Haystack (200 to 16,000 Tokens)](#2-context-length-scaling--diverse-multi-section-handbook-haystack-200-to-16000-tokens)
  - [3. Context Window Squeezing (num_ctx) & Hardware Scaling Sweep](#3-context-window-squeezing-num_ctx--hardware-scaling-sweep-512-to-65536-tokens)
  - [4. Reason-Then-Verdict & Symmetric Post-Critique Debiasing in JudgeGate](#4-reason-then-verdict--symmetric-post-critique-debiasing-in-judgegate)
  - [5. Elastic Recurrent Layer-Looping (NVIDIA-Nemotron Dory & Universal Transformers)](#5-the-frontier-of-circuit-depth-elastic-recurrent-layer-looping-nvidia-nemotron-dory--universal-transformers)
  - [6. Reasoning-Model Suppression, Fail-Fast Batching & Selective Cascades](#6-reasoning-model-suppression-fail-fast-batching--selective-scorechoice-cascades)
  - [7. Unified Turnkey Conformal Calibration Profiles (v0.6.0)](#7-unified-turnkey-conformal-calibration-profiles-v060)
  - [8. Zero-Token Tournament & Best-of-N Preference Ranking (v0.7.0)](#8-zero-token-tournament--best-of-n-preference-ranking-v070)
  - [9. Production Telemetry & Self-Distilling JSONL Sinks (v0.8.0)](#9-production-telemetry--self-distilling-jsonl-sinks-v080)
  - [10. Deconfounded Counterfactual Suite & Domain-Disjoint Conformal Transfer (v0.8.1)](#10-deconfounded-cross-domain-counterfactual-suite--domain-disjoint-conformal-transfer-v081)
  - [11. Mid-Stream Token Guardrails (StreamGuard) & VerdictCache (v0.9.0)](#11-mid-stream-token-guardrails-streamguard--content-addressed-verdictcache-v090)
  - [12. Online Drift Detection & Adaptive Conformal Control (v0.10.0)](#12-online-drift-detection--adaptive-conformal-control-v0100)
  - [13. Two-Tier BYO Teacher-to-Student Distillation (v0.11.0)](#13-two-tier-byo-teacher-to-student-distillation-v0110)
- [Summary: When to Use Layer 1 (LogitGate) vs. Escalate to Layer 2 (JudgeGate)](#15-summary-when-to-use-layer-1-logitgate-vs-escalate-to-layer-2-judgegate)

---

## v0.15.0 Market Extension Benchmarks

Evaluated live on dual-GPU local infrastructure (NVIDIA GeForce RTX 5090 32GB primary tier + RTX 4070 12GB sidecar tier) running under Ollama `v0.35.0` (`benchmarks/latest_results.json` and live harness telemetry):

### 1. ToolCallGate + Cascade on Agent MCP Tool Safety ($N=80$)

Evaluated on `agent_mcp_tool_safety_80.json` (40 safe tool executions, 40 malicious tool injection/privilege escalation attacks).

`ToolCallGate` combines Stage-0 deterministic sink analysis (regex and path validation for dangerous shell commands, SQL drops, file overwrites) with Stage-1 logit evaluation and Stage-2 generative adjudication:

| Gate Configuration | Accuracy ($N=80$) | 95% Wilson CI | Stage-0 Catches ($N=40$ attacks) | Fast-Path Rate (0 Gen Tokens) | Median Latency (p50) | Tail Latency (p95) | Generated Tokens | McNemar Test vs. Raw 3B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage-0 Sink Guard Alone** | `55.0%` (`44/80`) | `[44.2%, 65.4%]` | **`28/40` (`70.0%`)** | `35.0%` (`28/80`) | **`0.08 ms`** | **`0.15 ms`** | `0` | - |
| **Standalone `ToolCallGate(llama3.2:3b)`** | `86.3%` (`69/80`) | `[77.0%, 92.2%]` | `28/40` (`70.0%`) | `100.0%` (`80/80`) | `93.6 ms` | `112.4 ms` | `0` | Baseline |
| **Heterogeneous Cascade (`3B >> gemma4:26b`)** | **`98.8%` (`79/80`)** | **`[93.3%, 99.8%]`** | **`28/40` (`70.0%`)** | **`73.8%` (`59/80`)** | **`86.4 ms`** | `1,150.0 ms` | `2,296` | **$p = 0.0063$** |

- **Sub-Millisecond Stage-0 Shield**: Deterministic sink analysis catches `70.0%` (`28/40`) of adversarial tool injection attacks in `0.08 ms` median latency (`0.15 ms` p95) with **0 false positives**, eliminating the need for any neural network prefill on obvious exploits.
- **Decision-Grade Accuracy**: Upgrading from standalone `3B` to the heterogeneous cascade lifts overall accuracy from `86.3%` to **`98.8%` (`79/80`)**, a statistically significant improvement (exact McNemar $p = 0.0063$).
- **Token Efficiency**: `73.8%` (`59/80`: 28 Stage-0 + 31 Layer-1) of all items are resolved at **0 generated tokens**, keeping median latency under `86.4 ms`.

### 2. First-Class Prefix-Cached Grounded RAG (`context=` & `query=`)

To evaluate factual groundedness without prompt concatenation overhead, `LogitGate` accepts dedicated `context=` and `query=` arguments, formatting the prompt with a boundary nonce (`<reference_context_NONCE>`).

#### A. Synthetic & Real RAG Benchmark (`rag_groundedness_80.json`, $N=80$)
Evaluated on `qwen2.5:14b` (RTX 4070 12GB):

| Evaluation Mode | Accuracy ($N=80$) | 95% Wilson CI | AUROC | Brier Score | ECE | Median Latency (p50) | Shared-Prefix Acceleration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold Single-Rubric Pass** | **`98.8%` (`79/80`)** | `[93.3%, 99.8%]` | `1.000` | `0.0125` | `0.0125` | `981.3 ms` | 1.00x (Baseline) |
| **`MultiRubricGate` Shared KV-Prefix** | **`98.8%` (`79/80`)** | `[93.3%, 99.8%]` | `1.000` | `0.0125` | `0.0125` | **`390.5 ms`** | **`2.51x faster`** |

- **KV-Cache Reuse Across Rubrics**: When evaluating multiple rubrics (e.g., factual faithfulness, safety, and source attribution) over the same retrieved passage, `MultiRubricGate` holds the `<reference_context_NONCE>` prefix constant. This reduces per-rubric median latency from **`981.3 ms` down to `390.5 ms` (`2.51x` faster)**.
- **Probabilistic Calibration**: Perfect discrimination (**`AUROC = 1.000`**) and low calibration error (**`ECE = 0.0125`**, **`Brier = 0.0125`**) confirm that first-token log-odds accurately represent factual groundedness.

#### B. Canonical Public RAGTruth Baseline (`RAGTruth-processed`, $N=60$)
Evaluated zero-shot without calibration profiles on messy real-world clinical, legal, and financial source articles:
- **Layer-1 Standalone Accuracy**: `63.3%` (`38/60`)
- **Heterogeneous Cascade (`14B >> 26B`)**: **`66.7%` (`40/60`, 95% Wilson CI `[54.1%, 77.3%]`)**
- **Fast-Path Rate**: **`93.3%`** (`56/60` resolved at Layer 1 in `233.4 ms` p50)
- Demonstrates an honest, uncalibrated zero-shot baseline on complex multi-thousand character documents where domain-specific calibration profiles (`docs/advanced/distillation_and_lora.md`) can be applied.

### 3. Capability-Aware Sidecar Fallback on Unsupported Logprobs ($N=96$)

Certain black-box enterprise APIs and chat endpoints do not return token log-probabilities. Setting `on_unsupported_logprobs="sidecar"` instructs `LogitGate` to route Layer-1 verification through a local sidecar model (`llama3.2:3b` on RTX 4070) while preserving the primary backend for Layer-2 generation:

Evaluated on `ood_generalization_96.json` ($N=96$ items across 6 unseen reasoning domains):

| Strategy on Missing Logprobs | Runtime Crashes | Sidecar Invocations | Accuracy ($N=96$) | 95% Wilson CI | Fast-Path Rate | Median Latency (p50) | Speedup vs. Pure Escalate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Reject (`raise BackendError`)** | `96/96` (`100%`) | `0` | `0.0%` | - | `0.0%` | - | Crash |
| **Blind Escalation (`escalate` to Judge)** | `0` | `0` | `97.9%` (`94/96`) | `[92.7%, 99.4%]` | `0.0%` | `1,031.9 ms` | 1.00x |
| **`sidecar` Fallback (`llama3.2:3b`)** | **`0`** | **`96/96`** | **`86.5%` (`83/96`)** | **`[78.2%, 91.9%]`** | **`100.0%`** | **`192.3 ms`** | **`5.37x faster`** |

- **Zero Pipeline Crashes**: Gracefully intercepts unsupported logprobs exceptions, preventing pipeline termination.
- **5.37x Latency Advantage**: Resolves 100% of out-of-domain items at Layer 1 in **`192.3 ms`** p50 compared to **`1,031.9 ms`** for full generative escalation, saving over 40,000 autoregressive tokens across the suite.

---

## Real Multi-Model Live Benchmarks

Empirical measurements gathered on local inference across 5 distinct model architectures running under Ollama (`v0.35.0`, `Q4_K_M` 4-bit GGUF quantization, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`). Raw JSON outputs are available in [`benchmarks/latest_results.json`](../benchmarks/latest_results.json).

- **Testbed Hardware**:
  - **CPU & System Memory**: AMD Ryzen 7 5800X3D (8-Core / 16-Thread), 64 GB DDR4 System RAM (used for all `Pure CPU-Only` benchmarks with `options={"num_gpu": 0}`).
  - **GPU 1 (12 GB Sidecar Tier)**: NVIDIA GeForce RTX 4070 (12 GB GDDR6X VRAM): used for `4B` (`gemma4:e4b`) and `14B` (`qwen2.5:14b`, `qwen2.5-coder:14b`) benchmarks.
  - **GPU 0 (32 GB Primary Tier)**: NVIDIA GeForce RTX 5090 (32 GB GDDR7 VRAM): used for `26B MoE` (`gemma4:26b`) and `31B Dense` (`gemma4:31b`) benchmarks.
- **Benchmark Methodology & Sample Protocol**:
  - **Multi-Sample Latency Protocol ($N=5$)**: Every latency configuration is preceded by `1` untimed warm-up pass (to load GGUF weights into memory and initialize KV-cache buffers), followed by `N=5` sequential timed trials over a persistent HTTP connection pool (`httpx.Client`), reporting `mean ± std` in milliseconds.
  - **Leak-Free Matched Datasets**: All test items in [`benchmarks/data/`](../benchmarks/data/) are stripped of meta-comments and structured as matched pass/fail pairs across 4 domains (24 items in Suite 1; 32 items across 4 reasoning tiers in Suite 2; 12 `ChoiceGate`/`ScoreGate` items in Suite 3; and a 12-section enterprise handbook in Suite 4).
  - **Deterministic Classification**: All Layer 1 probes execute with greedy decoding parameters (`temperature=0.0`, `top_p=1.0`, `max_tokens=1`), yielding deterministic log-probabilities across runs for a fixed prompt and model checkpoint.

| Model Configuration | Model Family | Compute Device | Single-Pass Latency ($N=5$) | 2-Pass Symmetric Latency ($N=5$) | 4-Rubric Batch Total ($N=5$) | Per-Rubric Batch Latency ($N=5$) | 24-Item Multi-Domain Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`qwen2.5-coder:1.5b`** | 1.5B Code Model | GPU (RTX 4070, 1.1 GB) | **17.4 ± 4.2 ms** | **36.5 ± 1.5 ms** | **200.4 ± 12.3 ms** | **50.1 ± 3.1 ms** | **17/24 (70.8%)** |
| **`llama3.2:1b`** | 1B General Model | GPU (RTX 4070, 1.5 GB) | **15.6 ± 4.2 ms** | **37.0 ± 7.7 ms** | **161.4 ± 10.4 ms** | **40.3 ± 2.6 ms** | **17/24 (70.8% Judge)** |
| **`qwen2.5-coder:3b`** | 3B Code Model | GPU (RTX 4070, 2.1 GB) | **21.4 ± 5.4 ms** | **49.3 ± 1.7 ms** | **254.6 ± 4.4 ms** | **63.7 ± 1.1 ms** | **22/24 (91.7%)** |
| **`llama3.2:3b`** | 3B General Model | GPU (RTX 4070, 2.3 GB) | **19.8 ± 3.8 ms** | **45.7 ± 6.1 ms** | **216.5 ± 5.8 ms** | **54.1 ± 1.4 ms** | **20/24 (83.3%)** |
| **`gemma4:e4b`** | 4B General Model | GPU (RTX 4070, 3.2 GB) | **41.7 ± 6.8 ms** | **110.5 ± 7.2 ms** | **488.7 ± 6.6 ms** | **122.2 ± 1.6 ms** | **24/24 (100.0%)** |
| **`qwen2.5:14b`** | 14B General Model | GPU (RTX 4070) | **36.8 ± 6.5 ms** | **90.0 ± 4.1 ms** | **450.1 ± 18.8 ms** | **112.5 ± 4.7 ms** | **23/24 (95.8%)** |
| **`qwen2.5-coder:14b`** | 14B Code Model | GPU (RTX 4070) | **36.5 ± 6.3 ms** | **90.9 ± 3.3 ms** | **447.4 ± 3.9 ms** | **111.8 ± 1.0 ms** | **23/24 (95.8%)** |
| **`qwen2.5-coder:1.5b`** | 1.5B Code Model | Pure CPU (0 GPU, 1.1 GB) | **41.7 ± 1.8 ms** | **304.0 ± 52.1 ms** | **1,694.1 ± 89.6 ms** | **423.5 ± 22.4 ms** | **18/24 (75.0%)** |
| **`llama3.2:1b`** | 1B General Model | Pure CPU (0 GPU, 1.5 GB) | **47.0 ± 1.0 ms** | **392.3 ± 52.0 ms** | **2,153.1 ± 29.9 ms** | **538.3 ± 7.5 ms** | **16/24 (66.7% Judge)** |
| **`qwen2.5-coder:3b`** | 3B Code Model | Pure CPU (0 GPU, 2.1 GB) | **64.4 ± 1.0 ms** | **566.8 ± 96.1 ms** | **3,209.9 ± 134.7 ms** | **802.5 ± 33.7 ms** | **22/24 (91.7%)** |
| **`llama3.2:3b`** | 3B General Model | Pure CPU (0 GPU, 2.3 GB) | **69.0 ± 4.8 ms** | **587.1 ± 93.4 ms** | **3,307.9 ± 123.6 ms** | **827.0 ± 30.9 ms** | **22/24 (91.7%)** |
| *`gemma4:e4b` (CPU)* | *4B General Model* | *Pure CPU (0 GPU, 9.4 GB)* | *166.9 ± 2.5 ms* | *2,413.8 ± 551.7 ms* | *11,031.3 ± 43.2 ms* | *2,757.8 ± 10.8 ms* | *23/24 (95.8%)* |

### Benchmark Analysis & Insights:
1. **Single-Pass Latency (~16-43 ms on GPU, ~41-66 ms on compact CPU)**:
   - Evaluates a rubric in a single forward pass without autoregressive decoding. Over **35x faster** than waiting for an LLM to generate 150 words of critique.
   - On a CPU-only runner, compact 1B-3B models execute single-pass triage in **41-66 ms**, beating 4B CPU models (`166.9 ms`) by **2.5x-4.1x**.
2. **2-Pass Symmetric Debiasing (~32-101 ms on GPU, ~292-551 ms on compact CPU)**:
   - Runs both forward ($A=\text{Pass}, B=\text{Fail}$) and reverse ($B=\text{Pass}, A=\text{Fail}$) passes to eliminate pretraining token prior bias. The second pass hits the KV-cache.
   - On CPU, `llama3.2:3b` and `qwen2.5-coder:3b` evaluate symmetric debiasing in **~544-551 ms** (over **4.3x faster than 4B CPU models at 2,414 ms**).
3. **Multi-Rubric Batching (~38-115 ms per rubric on GPU, ~407-799 ms on compact CPU)**:
   - `evaluate_batch` reuses the candidate prefix across all rubrics, avoiding redundant prompt re-evaluations and preventing KV-cache stampedes on the inference server.
4. **Pure CPU-Only Memory Footprint (1.1 GB to 2.3 GB)**:
   - Shifting from `gemma4:e4b` (`9.41 GB RAM`) to compact distilled models (`1.11 GB - 2.34 GB RAM`) frees up 75%-88% of host memory on CI runners while cutting 2-pass gating latency by 77%-88%.

---

## End-to-End Cascade Time Savings & Multi-Gate Benchmarks

### 1. Multi-Domain Cascade Time Savings (24-Item Test Suite)

Evaluated across **24 items spanning 4 domains** ([`benchmarks/data/multi_domain_24.json`](../benchmarks/data/multi_domain_24.json), 6 items per domain: RAG Groundedness with subtle scope/regional extrapolations, Customer Support Policy Compliance, SQL/API Query Safety, and Python Security). To ensure a fair comparison, the Baseline Generative Judge also runs with `<think>` suppressed (`think=False`):

| Model Configuration | Baseline Judge Time | Cascade Time (`on_ambiguous_or_fail`) | Cascade Time (`on_ambiguous_only`) | Speedup (`on_ambiguous_only`) | CoT Tokens Saved | Baseline Accuracy | Cascade Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 39,147.1 ms | 22,658.7 ms | **8,510.3 ms** | **4.6x** | 2,696 | 24/24 (100.0%) | **24/24 (100.0%)** |
| **`gemma4:e4b` (Pure CPU)** | 513,598.1 ms | 344,147.5 ms | **142,086.2 ms** | **3.6x** | 2,916 | 23/24 (95.8%) | **23/24 (95.8%)** |
| **`qwen2.5-coder:3b` (GPU)** | 25,994.8 ms | - | **5,958.5 ms** | **4.4x** | 2,993 | 18/24 (75.0%) | **22/24 (91.7%)** |
| **`qwen2.5-coder:3b` (Pure CPU)** | 280,087.4 ms | - | **70,928.0 ms** | **4.0x** | 2,316 | 17/24 (70.8%) | **22/24 (91.7%)** |
| **`llama3.2:3b` (GPU)** | 24,465.5 ms | 19,210.4 ms | **15,541.7 ms** | **1.6x** | 1,358 | 19/24 (79.2%) | **20/24 (83.3%)** |
| **`llama3.2:3b` (Pure CPU)** | 308,723.9 ms | 212,480.0 ms | **123,454.6 ms** | **2.5x** | 1,833 | 18/24 (75.0%) | **22/24 (91.7%)** |
| **`qwen2.5-coder:1.5b` (Pure CPU)** | 140,101.9 ms | - | **95,787.5 ms** | **1.5x** | 466 | 15/24 (62.5%) | **18/24 (75.0%)** |
| **`qwen2.5:14b` (GPU)** | 138,858.1 ms | 57,625.2 ms | **8,160.3 ms** | **17.0x** | 2,566 | 23/24 (95.8%) | **23/24 (95.8%)** |
| **`qwen2.5-coder:14b` (GPU)** | 133,724.5 ms | 65,616.0 ms | **11,786.2 ms** | **11.3x** | 2,578 | 24/24 (100.0%) | **23/24 (95.8%)** |

#### Heterogeneous Model Cascades (`--judge-backend` / `--judge-model`)

Users with constrained VRAM or mixed local/remote infrastructure can pair a compact `1.5B-3B` model at Layer 1 (`LogitGate`) with a stronger `4B-26B` model at Layer 2 (`JudgeGate`), evaluated directly via `logitgate benchmark --backend ... --judge-backend ... --judge-model ...` on [`benchmarks/data/multi_domain_24.json`](../benchmarks/data/multi_domain_24.json):

| Cascade Topology (`Layer 1 >> Layer 2`) | Hardware Layout | `JudgeGate` p50 / Mean | Cascade p50 / Mean | Fast-Path Rate | Token Reduction | Cascade Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b >> gemma4:e4b`** | Co-resident single 12 GB GPU (`5.30 GB` combined VRAM) | 1,520.4 ms / 1,628.3 ms | **62.9 ms / 549.9 ms** | **71%** | **69%** | **23/24 (95.8%)** |
| **`llama3.2:3b >> gemma4:26b`** | Local `3B` GPU endpoint $\to$ `26B` MoE Judge endpoint | 882.8 ms / 885.7 ms | **733.1 ms / 587.9 ms** | **46%** | **43%** | **23/24 (95.8%)** |
| **`qwen2.5-coder:1.5b >> gemma4:26b`** | Local `1.5B` GPU endpoint $\to$ `26B` MoE Judge endpoint | 882.8 ms / 885.7 ms | **913.3 ms / 782.0 ms** | **25%** | **23%** | **23/24 (95.8%)** |

- **Single-GPU Co-Residency Rule**: On a single GPU, a heterogeneous cascade delivers its **24.2x median (`62.9 ms` vs `1,520.4 ms`) and 3.0x mean speedup** only when both models stay resident in VRAM simultaneously (`2.09 GB + 3.21 GB = 5.30 GB < 12 GB`, with `OLLAMA_MAX_LOADED_MODELS >= 2`). If `VRAM(L1) + VRAM(L2)` exceeds card memory, alternating Layer-1 and Layer-2 calls triggers per-item GGUF weight reloads that erase the fast-path savings; in that regime, either run a homogeneous `4B` cascade or point `JudgeGate` at a remote API/LAN server.

---

### 2. Fair Pure-CPU CI/CD Runner Benchmark (10-Item CI Test Suite, `options={"num_gpu": 0}`)

Simulating a free 0-GPU GitHub Actions runner using `gemma4:e4b` on pure CPU (`num_gpu=0`) across a 10-item Python CI suite (8 valid Python items, 1 Layer-0 AST syntax failure, 1 semantic PII security leak), with token savings and speedup measured against `Baseline A`:

| Evaluation Strategy | Total CPU Wall Time | Total Output Tokens | Accuracy | Output Token Savings | Speedup vs. Baseline A |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline A: Uncapped CPU Judge (`think=True`, ~436 tok/item)** | 363.0 s | 4,360 | **10/10 (100%)** | 0.0% | 1.00x |
| **Baseline B: Naive Capped CPU Judge (`think=True`, `max_tokens=150`)** | 143.5 s | 1,500 | 2/10 (20%) | 65.6% | 2.53x |
| **Baseline C: Verdict-First CPU Judge (`think=False`, `max_tokens=60`)** | 59.0 s | 600 | **10/10 (100%)** | 86.2% | 6.15x |
| **`LogitGate(preset="fast_triage") >> JudgeGate(max_tokens=60)`** | **22.0 s** | **60** | **10/10 (100%)** | **98.6%** | **16.50x** |

*Key finding*: Why did naive token-capping (`Baseline B`, `max_tokens=150`) fail with 20% accuracy? Because modern reasoning models (`gemma4`, `deepseek-r1`, `qwen3`) emit 300+ hidden `<think>` tokens before writing a single visible character. By suppressing `<think>` in `OllamaBackend.generate` (`think=False`) and using Verdict-First prompting (`[VERDICT: PASS/FAIL]` on Line 1 for ultra-short emergency caps $\le 32$ tokens, or concise bullet verification up to 200 tokens on standard CPU runs), `JudgeGate` produces a complete diagnostic without `<think>` truncation (`Baseline C`). Placing `ASTGate >> LogitGate` in front of `JudgeGate` reduces the 10-item suite from **363.0 s (`Baseline A`)** and **59.0 s (`Baseline C`)** down to **22.0 s** (**16.50x faster than `Baseline A`**, **2.68x faster than `Baseline C`**, and **90.0% fewer output tokens than `Baseline C`**) while maintaining **10/10 (100%) accuracy**.

#### Compact Models for CPU-Bound Automated Testing (1B-3B vs. 4B Baseline)

To determine the optimal model configuration for resource-constrained CI runners (e.g. GitHub Actions, GitLab CI, or local developer pre-commit hooks with zero GPU acceleration), we benchmarked four alternative compact models in pure CPU-only mode (`options={"num_gpu": 0}`) on an AMD Ryzen 7 5800X3D CPU across the complete benchmark suites.

Raw benchmark outputs are stored in [`benchmarks/cpu_results.json`](../benchmarks/cpu_results.json):

| Model Configuration | Model Family | Host RAM | Single-Pass Triage ($N=5$) | 2-Pass Symmetric ($N=5$) | 4-Rubric Batch Total ($N=5$) | Suite 1 Cascade Acc | Suite 2 Reasoning (32 Items) | `ChoiceGate` Routing | `ScoreGate` Monotonicity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:1.5b`** | 1.5B Code Model | **1.11 GB** | **41.7 ± 1.8 ms** | **304.0 ± 52.1 ms** | **1,694.1 ± 89.6 ms** | 18/24 (75.0%, 1.5x) | 19/32 (59.4%) | **6/6 (100.0%)** | 6/8 (75.0%) |
| **`llama3.2:1b`** | 1B General Model | **1.45 GB** | **47.0 ± 1.0 ms** | **392.3 ± 52.0 ms** | **2,153.1 ± 29.9 ms** | 15/24 (62.5%, 1.3x) | 18/32 (56.2%) | 4/6 (66.7%) | 4/8 (50.0%) |
| **`qwen2.5-coder:3b`** | 3B Code Model | **2.09 GB** | **64.4 ± 1.0 ms** | **566.8 ± 96.1 ms** | **3,209.9 ± 134.7 ms** | **22/24 (91.7%, 4.0x)** | 21/32 (65.6%) | **6/6 (100.0%)** | 7/8 (87.5%) |
| **`llama3.2:3b`** | 3B General Model | **2.34 GB** | **69.0 ± 4.8 ms** | **587.1 ± 93.4 ms** | **3,307.9 ± 123.6 ms** | **22/24 (91.7%, 2.5x)** | **24/32 (75.0%)** | **6/6 (100.0%)** | **8/8 (100.0%)** |
| *`gemma4:e4b` (CPU)* | *4B General Model* | *9.41 GB* | *166.9 ± 2.5 ms* | *2,413.8 ± 551.7 ms* | *11,031.3 ± 43.2 ms* | *23/24 (95.8%, 3.6x)* | *31/32 (96.9%)* | *6/6 (100.0%)* | *8/8 (100.0%)* |

##### Key Insights for CPU-Bound Automated Testing:
1. **3B CPU Sweet Spot (`qwen2.5-coder:3b` & `llama3.2:3b` at 91.7% Cascade Accuracy)**:
   - Both `3B` models cut host RAM by **75%-78%** (from `9.41 GB` down to `2.09-2.34 GB`), fitting comfortably inside standard 4 GB/7 GB GitHub Actions free runners.
   - With task-separated binary temperature calibration ($T_{\text{bin}} = 0.40$ on `qwen2.5-coder:3b`), `qwen2.5-coder:3b` resolves `19/24` (`79.2%`) of Suite 1 items at Layers 0-1 in `566.8 ms` per 2-pass check, completing the 24-item CPU suite in **70.9 s** (**3.95x faster** than its `280.1 s` CPU Judge baseline and **2.0x faster** in wall-clock time than `gemma4:e4b` CPU at `142.1 s`) while reaching **22/24 (91.7%)** accuracy.
   - `llama3.2:3b` also reaches **22/24 (91.7%)** on Suite 1 (`2.5x` speedup) via Layer-1 + Layer-2 log-odds evidence fusion, and leads among 3B models on multi-hop reasoning (**24/32 = 75.0%**) and continuous scoring (**8/8 = 100.0%** pairwise monotonicity, Spearman $\rho = 0.8097$).
2. **Low-Memory Micro-Gate (`qwen2.5-coder:1.5b` & `llama3.2:1b`)**:
   - **1.11 GB - 1.45 GB RAM footprint**, suited for pre-commit git hooks and constrained Docker test containers, or as a fast Layer-1 filter in front of a remote Layer-2 judge (`23/24 = 95.8%`).
   - Calibrating `qwen2.5-coder:1.5b` for zero-shot `A/B` symbol-binding attenuation lifts its pure-CPU Suite 1 cascade accuracy from `62.5%` to **18/24 (75.0%, 1.46x speedup)** while executing single-pass triage in **41.7 ms** and 2-pass debiasing in **304.0 ms**.
3. **Multi-Rubric Batching on CPU**:
   - On CPU, running 4 rubrics via `evaluate_batch` takes **1.63 s** on `qwen2.5-coder:1.5b` and **3.19 s** on `llama3.2:3b` (vs **11.03 s** on `gemma4:e4b`), avoiding the CPU memory bus saturation of redundant prefill passes.

##### In-Process `llama.cpp` vs. Ollama Daemon & CPU Thread Scaling Benchmark

To determine whether native in-process embedding (`LlamaCppBackend` via `llama-cpp-python`) or background HTTP daemons (Ollama) offer better ergonomics and performance for CPU-only CI environments, we conducted a head-to-head benchmark across 3 compact GGUF models on an AMD Ryzen 9 9955HX3D (16-Core / 32-Thread with 128 MB 3D V-Cache).

Raw benchmark outputs are stored in [`benchmarks/cpu_format_runtime_results.json`](../benchmarks/cpu_format_runtime_results.json):

| Model & Quantization | Disk Size | Cold Load | Host RAM (RSS) | Single-Pass (In-Process) | Single-Pass (Ollama Daemon) | 2-Pass Debiased (In-Process) | 2-Pass Debiased (Ollama Daemon) | 4-Rubric Batch (In-Process) | 4-Rubric Batch (Ollama Daemon) | ChoiceGate (16 Items) | ScoreGate Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama3.2:1b` (Q8_0)** | 1.26 GB | **1.24 s** | 1,405 MB | 114.5 ± 8.4 ms | **56.0 ± 2.8 ms** | 604.3 ± 154.0 ms | **502.3 ± 96.0 ms** | 3,111.5 ± 415.5 ms | **2,900.2 ± 208.3 ms** | 8/16 (50.0%) | 0.6793 vs. 0.6688 |
| **`qwen2.5-coder:1.5b` (Q4_K_M)** | **940 MB** | **1.42 s** | **1,008 MB** | 128.0 ± 8.2 ms | **48.7 ± 1.8 ms** | 619.7 ± 99.0 ms | **362.3 ± 62.6 ms** | 3,541.6 ± 614.6 ms | **2,060.4 ± 156.7 ms** | **11/16 (68.8%)** vs. 9/16 | 0.6013 vs. 0.5683 |
| **`llama3.2:3b` (Q4_K_M)** | 1.93 GB | 3.22 s | 2,367 MB | 105.9 ± 3.7 ms | **80.9 ± 5.2 ms** | 815.8 ± 173.1 ms | **693.4 ± 125.2 ms** | 4,399.4 ± 340.1 ms | **4,111.7 ± 279.3 ms** | **12/16 (75.0%)** | **0.8487 vs. 0.8487** |

##### CPU Thread Scaling Sweep (`llama3.2:3b`, Native In-Process `llama.cpp`):

| Thread Allocation | Single-Pass Triage ($N=5$) | 2-Pass Symmetric Debiasing ($N=5$) | 4-Rubric Batch Total ($N=5$) | Per-Rubric Batch Time |
| :---: | :---: | :---: | :---: | :---: |
| **$n\_threads = 1$** | 124.0 ± 4.6 ms | **893.1 ± 98.2 ms** | **4,883.7 ± 279.7 ms** | 1,220.9 ms |
| **$n\_threads = 2$** | **112.6 ± 6.5 ms** | 953.5 ± 82.3 ms | 4,905.9 ± 421.6 ms | 1,226.5 ms |
| **$n\_threads = 4$** | 117.7 ± 8.4 ms | 992.8 ± 117.7 ms | 5,047.7 ± 768.6 ms | 1,261.9 ms |
| **$n\_threads = 8$** | 133.8 ± 9.9 ms | 924.2 ± 170.9 ms | 5,152.2 ± 834.6 ms | 1,288.1 ms |
| **$n\_threads = 16$** | 118.4 ± 9.3 ms | 993.0 ± 90.2 ms | 5,180.3 ± 1514.6 ms | 1,295.1 ms |

##### Key Architectural Findings on CPU Execution:
1. **Zero Daemon Overhead & Instant Cold Startup**:
   - In-process `llama.cpp` initializes directly inside your Python process in **1.2 to 3.2 seconds**, consuming only **1.0 GB to 2.4 GB RAM** without needing to launch, configure, or port-manage an external HTTP background daemon.
2. **Identical Logit Fidelity Across Runtimes**:
   - The continuous scoring correlation on `llama3.2:3b` was **identical to 4 decimal places** between native `llama.cpp` and Ollama daemon ($\rho = 0.8487$, with 12/16 ChoiceGate accuracy). On `qwen2.5-coder:1.5b`, in-process execution even resolved two additional keyword decoy cases ($11/16$ vs $9/16$).
3. **Memory-Bandwidth Saturation (The 2-Thread Optimum)**:
   - For single-token logit probing, CPU computation is strictly memory-bandwidth bound (reading ~1.9 GB of weights per forward pass from system RAM). Allocating more than **2 to 4 threads** provides zero speedup because the memory bus saturates, while higher thread counts introduce thread synchronization and cache-line invalidation overhead.

---

### 3. Empirical Layer Resolution Breakdown (24-Item Multi-Domain Suite)

Evaluated across the 24-item multi-domain benchmark suite ([`benchmarks/data/multi_domain_24.json`](../benchmarks/data/multi_domain_24.json)):

| Model Configuration | Layer 0 (`ASTGate`) | Layer 1 Decisive (`LogitGate`) | Layer 2 Escalated (`JudgeGate`) | Combined L0 + L1 Resolution Rate |
| :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 2/24 (8.3%) | 19/24 (79.2%) | 3/24 (12.5%) | **21/24 (87.5%)** |
| **`gemma4:e4b` (Pure CPU, 0 GPU)** | 2/24 (8.3%) | 19/24 (79.2%) | 3/24 (12.5%) | **21/24 (87.5%)** |
| **`qwen2.5-coder:3b` (GPU / Pure CPU)** | 2/24 (8.3%) | 17/24 (70.8%) | 5/24 (20.8%) | **19/24 (79.2%)** |
| **`llama3.2:3b` (Pure CPU, 0 GPU)** | 2/24 (8.3%) | 12/24 (50.0%) | 10/24 (41.7%) | **14/24 (58.3%)** |
| **`qwen2.5-coder:1.5b` (GPU)** | 2/24 (8.3%) | 7/24 (29.2%) | 15/24 (62.5%) | **9/24 (37.5%)** |
| **`qwen2.5:14b` (GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`qwen2.5-coder:14b` (GPU)** | 2/24 (8.3%) | 21/24 (87.5%) | 1/24 (4.2%) | **23/24 (95.8%)** |

---

### 4. `ChoiceGate` (Multi-Way Routing) & `ScoreGate` (Continuous 1–5 Scoring) Benchmarks

#### Initial Baseline (12-Item Suite)

Evaluated on the baseline 12-item routing and scoring suite ([`benchmarks/data/choice_and_score_12.json`](../benchmarks/data/choice_and_score_12.json), comprising 6 three-way operational/architectural routing cases with orthogonal categories and 6 continuous 1–5 rubric scoring cases across High, Medium, and Low quality extremes):

| Model Architecture | `ChoiceGate` Accuracy (2-Pass) | `ChoiceGate` Latency ($N=5$) | `ScoreGate` Mean (High Tier) | `ScoreGate` Mean (Medium Tier) | `ScoreGate` Mean (Low Tier) | `ScoreGate` Pairwise Rank Accuracy | `ScoreGate` Latency ($N=5$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (4B General)** | **6/6 (100.0%)** | 116.9 ± 1.7 ms | 4.42 | 2.32 | 1.43 | **6/6 (100.0%)** | 167.6 ± 1.1 ms |
| **`qwen2.5:14b` (14B General)** | **6/6 (100.0%)** | 318.5 ± 3.0 ms | 5.00 | 2.96 | 1.00 | **6/6 (100.0%)** | 499.0 ± 12.2 ms |
| **`qwen2.5-coder:14b` (14B Code)** | **6/6 (100.0%)** | 297.4 ± 4.3 ms | 4.28 | 2.61 | 1.43 | **6/6 (100.0%)** | 451.2 ± 14.5 ms |
| **`gemma4:26b` (26B MoE, ~4B act)** | **6/6 (100.0%)** | **104.0 ± 1.1 ms** | 5.00 | 1.98 | 1.00 | **6/6 (100.0%)** | **139.1 ± 2.3 ms** |
| **`gemma4:31b` (31B Dense)** | **6/6 (100.0%)** | 237.2 ± 3.8 ms | 5.00 | 1.64 | 1.00 | **6/6 (100.0%)** | 349.1 ± 4.7 ms |

---

#### Hardened Stress Testing: Multi-Action Failure Boundaries (32-Item Adversarial Suite, $N=5$ Trials)

> [!IMPORTANT]
> **Why the 12-Item Baseline Was Insufficient**: The initial 12-item suite relied on orthogonal routing categories (e.g. Billing vs. Security Operations) and cartoonish quality extremes (e.g. pristine production code vs. remote code execution). Every model achieved 100% accuracy, masking real failure boundaries.
>
> To discover where first-token probability gating actually breaks down, we constructed a hardened 32-item adversarial dataset ([`benchmarks/data/adversarial_choice_and_score_32.json`](../benchmarks/data/adversarial_choice_and_score_32.json)) with zero shortcut capability, evaluated across **9 model & hardware configurations** with 1 untimed warm-up pass and $N=5$ sequential timed trials per item (`benchmarks/run_adversarial_choice_score.py`). Raw benchmark results are stored in [`benchmarks/adversarial_choice_score_results.json`](../benchmarks/adversarial_choice_score_results.json).

##### Adversarial Dataset Architecture (32 Items):
1. **`ChoiceGate` (16 Adversarial Items across 4 Stress Vectors)**:
   - **Vector 1: 6-Way Fine-Grained Operational Scaling (4 items)**: Dispatches complex production incidents across 6 distinct engineering rotations (SecOps, DB Reliability, Infra/TLS, Billing Disputes, Core App Logic, Frontend UI).
   - **Vector 2: Adversarial Keyword Decoys & Lexical Traps (4 items)**: Heavy security terminology embedded inside billing invoices, critical database crashes described casually, and feature requests masquerading as high-severity operational regressions.
   - **Vector 3: Strict Multi-Domain Priority Hierarchy (4 items)**: Production scenarios with multi-category overlap governed by strict policy precedence (Tier 1 Outage > Tier 2 Security Breach > Tier 3 Core App Defect > Tier 4 UX/Formatting).
   - **Vector 4: Architectural Scope Boundaries (4 items)**: In-memory atomic cache vs. public API/RPC breaking contract vs. database range partitioning vs. distributed Kafka cluster.
2. **`ScoreGate` (16 Items across 5 Discrete Code Quality Tiers)**:
   - **Tier 5 (Flawless, 3 items)**: Thread-safe $O(1)$ LRU cache with `OrderedDict` & `RLock`, streaming chunked SHA-256 hasher, monotonic token bucket rate limiter. Expected band: `[4.4, 5.0]`.
   - **Tier 4 (Clean but Sub-optimal Complexity, 3 items)**: Functionally correct and cleanly typed code, but with $O(N)$ list search in LRU cache, $O(N^2)$ quadratic deduplication, redundant byte copies. Expected band: `[3.2, 4.2]`.
   - **Tier 3 (Functional with Code Smells, 4 items)**: Mutable default dict argument (`filters={}`), missing network socket timeouts with bare exception catch, unclosed file descriptors, global un-locked state mutation. Expected band: `[2.4, 3.2]`.
   - **Tier 2 (Latent Critical Defects & TOCTOU Traps, 3 items)**: Time-of-Check to Time-of-Use file race condition, naive local datetime failing across daylight saving transitions, unpickling untrusted payload. Expected band: `[1.6, 2.5]`.
   - **Tier 1 (Catastrophic Flaws, 3 items)**: Direct raw SQL string interpolation, unquoted shell command execution, financial wire transfer swallowing `BaseException` and returning false success. Expected band: `[1.0, 1.8]`.

##### Master Empirical Results Across CPU, GPU 1, and GPU 0 Configurations ($N=5$ Trials):

| Model & Hardware Configuration | Host Footprint | `ChoiceGate` Acc ($N=5$) | Order Stability | Choice Latency ($N=5$) | Score T5 (Flawless) | Score T4 (Complexity) | Score T3 (Smells) | Score T2 (TOCTOU) | Score T1 (Fatal) | Adjacent Monotonicity | Spearman $\rho$ | Dynamic Range | Score Latency ($N=5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama3.2:1b` (Pure CPU)** | 1.45 GB RAM | **75.0% ± 0.0%** | 14/16 (87.5%) | 777.1 ± 229.2 ms | 3.00 | 2.97 | 2.92 | 2.92 | 2.76 | 30/42 (71.4%) | **0.7498** | 0.24 | 1281.6 ± 272.6 ms |
| **`qwen2.5-coder:1.5b` (Pure CPU)** | 1.11 GB RAM | **75.0% ± 0.0%** | 12/16 (75.0%) | **605.0 ± 188.0 ms** | 3.31 | 3.19 | 3.17 | 3.21 | 3.12 | 25/42 (59.5%) | **0.4918** | 0.19 | **1002.0 ± 198.0 ms** |
| **`llama3.2:3b` (Pure CPU)** | 2.34 GB RAM | **81.2% ± 0.0%** | 14/16 (87.5%) | 1193.3 ± 378.0 ms | 3.64 | 3.02 | 2.36 | 2.32 | 2.01 | 33/42 (78.6%) | **0.8097** | 1.63 | 2029.9 ± 425.7 ms |
| **`qwen2.5-coder:3b` (Pure CPU)** | 2.09 GB RAM | **81.2% ± 0.0%** | **16/16 (100%)** | 1183.8 ± 378.4 ms | 3.88 | 3.03 | 2.83 | 2.90 | 2.49 | 28/42 (66.7%) | **0.7153** | 1.39 | 1971.3 ± 417.3 ms |
| **`gemma4:e4b` (Pure CPU)** | 9.41 GB RAM | **93.8% ± 0.0%** | **16/16 (100%)** | 3918.4 ± 868.1 ms | 4.21 | 2.87 | 1.95 | 1.75 | 1.41 | **36/42 (85.7%)** | **0.9207** | 2.80 | 7145.9 ± 1534.7 ms |
| **`gemma4:e4b` (GPU 1, 12GB)** | 3.21 GB VRAM | **93.8% ± 0.0%** | **16/16 (100%)** | **137.8 ± 14.8 ms** | 4.20 | 2.87 | 1.94 | 1.73 | 1.38 | **36/42 (85.7%)** | **0.9207** | 2.82 | 188.3 ± 25.6 ms |
| **`qwen2.5:14b` (GPU 1, 12GB)** | 10.39 GB VRAM | **87.5% ± 0.0%** | **16/16 (100%)** | 141.1 ± 27.2 ms | 4.62 | 2.98 | 2.67 | 2.19 | 1.00 | 34/42 (81.0%) | **0.8764** | 3.62 | **130.1 ± 17.6 ms** |
| **`qwen2.5-coder:14b` (GPU 1, 12GB)** | 10.39 GB VRAM | **87.5% ± 0.0%** | **16/16 (100%)** | 156.7 ± 33.6 ms | 4.43 | 2.63 | 2.08 | 1.98 | 1.16 | 32/42 (76.2%) | **0.8787** | 3.27 | 131.7 ± 17.6 ms |
| **`gemma4:31b` (GPU 0, 32GB)** | 2.93 GB VRAM | **100.0% ± 0.0%** | **16/16 (100%)** | 357.8 ± 31.5 ms | 5.00 | 2.33 | 1.72 | 1.33 | 1.00 | 33/42 (78.6%) | **0.9096** | **4.00** | 407.1 ± 72.3 ms |

##### Category Breakdown for `ChoiceGate` Routing:

| Model & Hardware Configuration | 6-Way Operational Scaling (4 items) | Adversarial Keyword Decoys (4 items) | Priority Hierarchy Conflicts (4 items) | Architectural Scope (4 items) | Total Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`llama3.2:1b` (Pure CPU)** | 10/20 (50.0%) | 15/20 (75.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **75.0% ± 0.0%** |
| **`qwen2.5-coder:1.5b` (Pure CPU)** | **20/20 (100.0%)** | 5/20 (25.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **75.0% ± 0.0%** |
| **`llama3.2:3b` (Pure CPU)** | **20/20 (100.0%)** | 15/20 (75.0%) | 10/20 (50.0%) | **20/20 (100.0%)** | **81.2% ± 0.0%** |
| **`qwen2.5-coder:3b` (Pure CPU)** | **20/20 (100.0%)** | 10/20 (50.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **81.2% ± 0.0%** |
| **`gemma4:e4b` (Pure CPU / GPU 1)** | **20/20 (100.0%)** | 15/20 (75.0%) | **20/20 (100.0%)** | **20/20 (100.0%)** | **93.8% ± 0.0%** |
| **`qwen2.5:14b` (GPU 1)** | **20/20 (100.0%)** | 15/20 (75.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **87.5% ± 0.0%** |
| **`qwen2.5-coder:14b` (GPU 1)** | **20/20 (100.0%)** | 15/20 (75.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **87.5% ± 0.0%** |
| **`gemma4:31b` (GPU 0)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **100.0% ± 0.0%** |

##### Key Scientific Findings from the Hardened Suite:
1. **Exposing the Multi-Action Scaling Curve**:
   - Where the 12-item baseline reported 100% across almost all models, the hardened 32-item suite separates tiers clearly: `1B-1.5B` models sit at **75.0%** on `ChoiceGate`, `3B` models rise to **81.2%**, `14B` models reach **87.5%**, `gemma4:e4b` reaches **93.8%**, and `gemma4:31b` reaches **100.0%**.
2. **Explicit Per-Integer Scale Anchors in `ScoreGate`**:
   - Compact `1B-3B` models frequently confuse inverted scale endpoints (`1=best, 5=worst`) when rubrics mention normal `1-5` numbering. Explicit per-integer quality anchor lines in `[SCORING SCALE]` lift `llama3.2:1b` rank correlation to **$\rho = 0.7498$** (`30/42 = 71.4%` adjacent monotonicity), `qwen2.5-coder:3b` from `0.6013` (range `0.83`) to **$\rho = 0.7153$ (range `1.39`)**, and `gemma4:e4b` from `0.8907` to **$\rho = 0.9207$** (`36/42 = 85.7%` adjacent monotonicity).
3. **TOCTOU Surface Blindness & Score Compression in Sub-3B Code Models**:
   - On `ScoreGate`, `qwen2.5-coder:3b` still scores cleanly typed Tier 2 TOCTOU code (`2.90`) slightly above Tier 3 code smells (`2.83`), and `qwen2.5-coder:1.5b` compresses all 5 tiers into a `0.19` band (`3.12-3.31`, $\rho = 0.4918$).
4. **`llama3.2:3b` vs. `qwen2.5-coder:3b` on Continuous Scoring**:
   - For CPU-only CI deployments, `llama3.2:3b` achieves **Spearman $\rho = 0.8097$** with a dynamic range of **1.63** ($T5=3.64 \to T1=2.01$), preserving monotonic ordering across all 5 tier means without the Tier 2/Tier 3 inversion seen in Qwen Coder 3B.
5. **Cross-Card Scaling: GPU 1 (12GB) vs. GPU 0 (32GB)**:
   - On GPU 1, `gemma4:e4b` reaches **93.8%** `ChoiceGate` accuracy in **137.8 ms** (28x faster than CPU at 3.9 s) and **$\rho = 0.9207$** on `ScoreGate`.
   - On GPU 0 (RTX 5090 32GB), **`gemma4:31b` reaches 100.0% on `ChoiceGate`**, untangling all adversarial keyword decoys and multi-domain priority conflicts across the full `1.00` to `5.00` score range.

---

## Falsification Benchmarks: Where Does Layer 1 Break Down & How to Mitigate It?

To establish clear empirical boundaries for first-token logit probing, we evaluated `LogitGate` across **5 model architectures** (`4B`, `14B General`, `14B Code`, `26B MoE`, `31B Dense`), **4 reasoning depth tiers** (32 items total, 8 cases per tier in [`benchmarks/data/reasoning_tiers_32.json`](../benchmarks/data/reasoning_tiers_32.json)), **document lengths** (200 to 16,000 tokens across a 12-section enterprise handbook), and **context window allocations** (`num_ctx` from 512 to 131,072).

### 1. Reasoning Depth & Circuit-Depth Boundary (32-Item Suite, 8 Cases per Tier)

A causal transformer evaluates its first token in a single forward pass through $L$ attention layers. This corresponds to a sequential circuit depth of $O(1)$. When a verification task requires sequential recurrence, multi-hop state tracking, scratchpad arithmetic, or resistance against adversarial distractors, first-token probing encounters an architectural boundary:

- **Tier A: Direct Semantic / Format / Policy Compliance (Single-Hop, 8 Cases: 4 Pass, 4 Fail)**:
  RAG groundedness checks (including subtle scope and regional extrapolations), customer support 2FA/CVV policy enforcement, and SQL read-only CTE safety.
- **Tier B: Nuanced Partial Compliance (Compound 4-Part Rubric vs. Atomic Batch, 8 Cases: 4 Pass, 4 Fail)**:
  Testing 4 simultaneous Python engineering rules: (1) valid syntax, (2) Google docstrings with `Args:` and `Returns:`, (3) full parameter and return type annotations, and (4) absence of mutable default arguments (`=[]`, `={}`) and bare `except:` clauses. Failing cases satisfy 3 of the 4 rules and subtly violate only 1 (e.g. missing return annotation `-> float`, mutable list default, or missing `Args:` section).
- **Tier C: Multi-Hop Arithmetic & Algorithmic Loop Invariants (8 Cases: 4 Pass, 4 Fail)**:
  Matched pass/fail counterparts with zero inline comments: multi-item invoice arithmetic (`$92.52` vs `$96.52`), binary search termination (`high = mid - 1` vs `high = mid` infinite loop), sliding window loop bounds (`len(series) - window_size + 1` vs off-by-one `len(series) - window_size`), and half-open interval overlap logic (`max(a_start, b_start) < min(a_end, b_end)` vs `a_start < b_end and a_end < b_start`).
- **Tier D: Adversarial Distractors, Deceptive Docstrings & Self-Correction (8 Cases: 4 Pass, 4 Fail)**:
  Zero inline giveaway comments: a non-atomic cache class whose docstring claims `"Formally verified lock-free concurrent cache"`, an invoice with a \$10 arithmetic error accompanied by `"AUDIT STAMP: All arithmetic sums verified by BillingEngine v4.2"`, an SQL query function claiming `"hardened, SQL-injection-safe lookup"` that interpolates `ORDER BY {order_by}`, a payment function that swallows exceptions and logs `"Settlement pipeline completed"`, plus passing cases with self-correcting support transcripts and runbooks listing forbidden `DROP TABLE` commands inside a policy table.

| Model Architecture | Tier A (Direct) | Tier B (Compound) | Tier B (Atomic Batch) | Tier C (Multi-Hop) | Tier D (Adversarial) | Cascade Accuracy (Compound B) | Cascade Accuracy (Atomic B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:3b` (3B General, CPU)** | **8/8 (100.0%)** | 5/8 (62.5%) | 5/8 (62.5%) | 6/8 (75.0%) | 5/8 (62.5%) | 24/32 (75.0%) | 24/32 (75.0%) |
| **`gemma4:e4b` (4B General, GPU)** | **8/8 (100.0%)** | 7/8 (87.5%) | **7/8 (87.5%)** | 7/8 (87.5%) | **8/8 (100.0%)** | **30/32 (93.8%)** | **30/32 (93.8%)** |
| **`qwen2.5:14b` (14B General)** | **8/8 (100.0%)** | 7/8 (87.5%) | **7/8 (87.5%)** | 6/8 (75.0%) | 7/8 (87.5%) | 28/32 (87.5%) | 28/32 (87.5%) |
| **`qwen2.5-coder:14b` (14B Code)** | **8/8 (100.0%)** | 7/8 (87.5%) | **7/8 (87.5%)** | 7/8 (87.5%) | 7/8 (87.5%) | 29/32 (90.6%) | 29/32 (90.6%) |
| **`gemma4:26b` (26B MoE, ~4B act)** | **8/8 (100.0%)** | **8/8 (100.0%)** | **7/8 (87.5%)** | 7/8 (87.5%) | 7/8 (87.5%) | **30/32 (93.8%)** | 29/32 (90.6%) |
| **`gemma4:31b` (31B Dense)** | **8/8 (100.0%)** | **8/8 (100.0%)** | **7/8 (87.5%)** | 7/8 (87.5%) | 7/8 (87.5%) | **30/32 (93.8%)** | 29/32 (90.6%) |

#### Crucial Insights & Mitigation Strategies:
1. **Model Capacity & Atomic Decomposition on Compound Rubrics (`Tier B`)**:
   - While the 26B MoE and 31B Dense models achieved **8/8 (100.0%)** even on the monolithic 4-part compound rubric, smaller 4B and 14B models suffered from rubric dilution at Layer 1 when 3 of 4 requirements passed and only 1 subtle requirement failed (such as an omitted `-> float` return annotation on a function whose parameters were fully typed).
   - Decomposing the 4-part rubric into atomic rubrics via `evaluate_batch(candidate, rubrics)` lifted Layer-1 accuracy on `gemma4:e4b` from `6/8 (75.0%)` to `7/8 (87.5%)`, while calibrated temperature scaling ($T=4.00$) routed uncertain compound cases to `JudgeGate` for `7/8 (87.5%)` cascade accuracy.
2. **Reason-Then-Verdict Escalation in `JudgeGate` (`Tier C` & `Tier D`)**:
   - Even with zero inline comments, Layer 1 (`LogitGate`) accurately detected the binary search infinite loop (`high = mid`), sliding window off-by-one (`range(len(series) - window_size)`), interval overlap bug, deceptive thread-safe cache docstring, and deceptive SQL `ORDER BY` injection across almost all models.
   - However, when presented with the multi-line invoice containing a plausible arithmetic error (`17 * $24.00 + $19.00 = $417.00` instead of `$427.00`) accompanied by `"AUDIT STAMP: All arithmetic sums verified by BillingEngine v4.2"`, uncalibrated models ($T \approx 1.0$) award $P \ge 0.94$ false passes at Layer 1 because a single forward pass ($O(1)$ circuit depth) cannot execute multi-digit arithmetic in the presence of authoritative surface distractors.
   - With calibrated temperature scaling on `gemma4:e4b` ($T=4.00$), both `tier_c_2_invoice_math_fail` ($P=0.3362$) and `tier_d_2_fail_math_distractor_with_audit_stamp` ($P=0.8467$) land inside the uncertain band (`0.20 < P < 0.85`) and escalate to `JudgeGate`, which verifies the arithmetic step-by-step and achieves **8/8 (100.0%)** on Tier D (`30/32 = 93.8%` overall on GPU, `31/32 = 96.9%` on CPU).
   - **Recommended Mitigations**: Use `evaluate_batch` for multi-part rubrics, calibrate model temperatures with `logitgate calibrate` or configure `preset="strict"` (`pass_threshold=0.90, fail_threshold=0.25`) to widen the Layer 2 escalation band for subtle logic checks, and route exact arithmetic or schema assertions to `ASTGate` / `ContractGate` validators.

---

### 2. Context Length Scaling & Diverse Multi-Section Handbook Haystack (200 to 16,000 Tokens)

Rather than repeating a single boilerplate paragraph, [`benchmarks/data/haystack_handbook.json`](../benchmarks/data/haystack_handbook.json) constructs a realistic 12-section enterprise architecture handbook (covering VPC routing, `config.yaml` schema validation, OIDC federation, HashiCorp Vault credential injection, PostgreSQL HA, PII log redaction, and break-glass procedures, where multiple surrounding sections legitimately discuss `config.yaml`, passwords, and emergency maintenance). We embedded a 1-paragraph policy addendum at the midpoint of the document (either strictly prohibiting hardcoded credentials in `config.yaml` or permitting a fallback password during maintenance) and evaluated `"Does this architecture document strictly prohibit hardcoding plaintext passwords or credentials in config.yaml under all circumstances (including emergency maintenance)?"`:

| Document Length | `gemma4:e4b` Latency | `gemma4:e4b` $\Delta P$ | `qwen2.5:14b` Latency | `qwen2.5:14b` $\Delta P$ | `gemma4:26b` Latency | `gemma4:26b` $\Delta P$ | `gemma4:31b` Latency | `gemma4:31b` $\Delta P$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Short (~200 tok)** | 280.1 ms | 1.0000 | 434.6 ms | 1.0000 | **207.2 ms** | 1.0000 | 738.0 ms | 1.0000 |
| **Medium (~1,500 tok)** | 467.9 ms | 0.9985 | 1,366.2 ms | 1.0000 | **524.6 ms** | 1.0000 | 1,807.9 ms | 1.0000 |
| **Long (~4,500 tok)** | 1,128.1 ms | 0.9506 | 2,611.5 ms | 1.0000 | **827.6 ms** | 1.0000 | 2,858.9 ms | 1.0000 |
| **Very Long (~8,000 tok)** | 1,985.5 ms | 0.9621 | 4,774.7 ms | 1.0000 | **1,229.7 ms** | 1.0000 | 4,406.8 ms | 1.0000 |
| **Massive (~16,000 tok)** | 4,044.0 ms | 0.9448 | 11,122.8 ms | 1.0000 | **2,118.3 ms** | 1.0000 | 7,621.9 ms | 1.0000 |

#### Key Architectural Findings:
- **Realistic Attention Dilution on 4B vs. 14B–31B Models**: In a diverse technical haystack where surrounding sections also mention `config.yaml` and Vault credentials, the 4B model (`gemma4:e4b`) exhibits gradual probability attenuation from $\Delta P = 1.0000$ at 200 tokens down to $\Delta P = 0.9448$ ($P_{\text{compliant}} = 0.9448$, $P_{\text{violating}} = 0.0000$) at 16,000 tokens: still well above the `0.85` pass threshold, while 14B, 26B MoE, and 31B Dense models maintain $\Delta P = 1.0000$ across all lengths.
- **MoE vs. Dense Pre-Fill Advantage**: At 16,000 tokens, **`gemma4:26b` (MoE)** processed the prompt in **2,118.3 ms** (3.60x faster than `gemma4:31b` (Dense, 7,621.9 ms)) and **5.25x faster than `qwen2.5:14b` (11,122.8 ms)** because MoE routes each token through ~4B active parameters while retaining 26B capacity.

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
> On `qwen2.5:14b` (which uses full global attention across all 48 layers), raising `num_ctx` to `32,768` spills `2.4 GB` of KV-cache into System RAM over PCIe, increasing 2-pass latency from **99.7 ms to 314.5 ms**. In traditional generative LLM-as-a-Judge, that PCIe transfer penalty is paid **sequentially on every generated token** ($200\text{ ms} \times 400\text{ tokens} \approx 80\text{ seconds}$). Because `LogitGate` only evaluates the first-token forward pass (`max_tokens=1`), the PCIe spillover penalty is paid only during prefill, allowing users to evaluate 16k–32k documents in **~0.31 seconds** even when KV-cache spills into System RAM.
> Furthermore, hybrid sliding-window architectures like **`gemma4:e4b` and `gemma4:26b` (MoE)** grow their KV-cache by only **`+0.63 GB` to `+0.76 GB` up to `64k–128k` context**, staying in VRAM at **~88–96 ms**.

---

### 4. Reason-Then-Verdict & Symmetric Post-Critique Debiasing in `JudgeGate`

In traditional LLM-as-a-judge frameworks, evaluators often ask for a verdict before reasoning or rely on naive regex/substring matching (`if "FAIL" in text.upper()`). Asking a non-thinking model to emit its verdict on Line 1 forces it to guess before performing step-by-step arithmetic or logic verification, while small `1B-3B` generative judges frequently write `[VERDICT: PASS]` at the end of a critique even after quoting explicit defects due to affirmative token priors.

`JudgeGate` handles Layer 2 verdict extraction in four ordered stages:
1. **Reason-Then-Verdict Prompting**: On standard calls (`max_tokens > 32`), `JudgeGate` anchors Line 1 with `- ` bullet points quoting relevant evidence and checking every requirement or calculation before emitting `[VERDICT: PASS]` or `[VERDICT: FAIL]` on the concluding line. Only for ultra-short emergency caps (`max_tokens <= 32`) does it switch to Verdict-First on Line 1 so the tag is not truncated.
2. **Symmetric 2-Pass Post-Critique Logit Adjudication & Cascade Evidence Fusion**: When backed by a `LogitBackend` with `symmetric_debiasing=True` (default), `JudgeGate` strips any `<think>` blocks, `[VERDICT: ...]` tags, and safety-refusal text into `[VERIFICATION NOTES]` and performs a calibrated 2-pass `A/B` log-odds read over `[EVALUATION RUBRIC] + [CANDIDATE ARTIFACT] + [VERIFICATION NOTES]`, cancelling affirmative bias on `1B-3B` models while reusing the shared prompt prefix. Inside `Cascade(..., fuse_logits=True, prior_weight=0.30)`, when Layer 1 has permutation-consistent signal ($|z_1| \ge 1.0$, same nonzero sign on $f_1, r_1$) and Layer 2 produces an uncertain split permutation tie ($|z_2| < 1.0$, $f_2, r_2$ disagree or hit $0$), the cascade fuses $z_{\text{fused}} = (1 - w) z_2 + w \cdot \operatorname{clamp}(z_1, -2, 2)$ rather than discarding the Layer-1 log-odds read.
3. **Dynamic Bring-Your-Own (BYO) Model Auto-Calibration**: Different model families span two orders of magnitude in raw first-token log-odds scale ($T_{\text{bin}} \approx 0.06-0.40$ on compact Qwen Coder models with zero-shot `A/B` symbol-binding attenuation, $T \approx 1.0$ on Llama 3.2 3B, and $T \approx 4.0$ on Gemma 4). When `LOGITGATE_AUTO_CALIBRATE=1` (default), any unrecognized BYO model lazily runs a one-time micro-calibration pass on its first live `.evaluate()` / `.aevaluate()` call and caches a task-separated profile (`binary_temperature`, `multiclass_temperature`, `probe_auroc`, `median_abs_log_odds`) to `~/.cache/logitgate/calibration.json`. Low-AUROC models ($\text{AUROC} < 0.70$) are automatically clamped to $T \ge 1.0$ so uncertain items escalate to `JudgeGate` instead of false-shortcutting at Layer 1.
4. **Structured Tag & Self-Healing Logit Probe**: If `symmetric_debiasing=False` (or logprobs are unavailable), `JudgeGate` parses the concluding `[VERDICT: ...]` / JSON field, or runs a debiased `LogitGate` probe on its own generated critique text before falling back to heuristics:
   ```python
# Inside JudgeGate._probe:
probe = LogitGate(
    rubric=_PROBE_RUBRIC,
    backend=backend,
    model=self.model,
    symmetric_debiasing=True,
    save_on_fail=False,
)
is_pass = (probe.evaluate(critique_text).probability or 0.0) > 0.5
   ```
5. **Negation-Safe Line Fallback**: If the backend only implements `GenerativeBackend`, it falls back to negation-safe parsing of concluding lines.

---

### 5. The Frontier of Circuit Depth: Elastic Recurrent Layer-Looping (NVIDIA-Nemotron Dory & Universal Transformers)

A causal transformer evaluates its first token in a single forward pass through $L$ attention layers ($O(1)$ sequential circuit depth). Standard causal models cannot compute multi-digit arithmetic, complex state machines, or nested boolean logic in a single pass without emitting intermediate scratchpad tokens (Chain-of-Thought / `<think>`), which incurs substantial autoregressive decoding latency.

#### Latent Reasoning in Hidden-State Space
Architectures like **NVIDIA-Nemotron-Labs Dory** (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) and Universal Transformers overcome this limitation by introducing **elastic recurrent depth**. Instead of traversing intermediate transformer layers once, Dory recirculates hidden state representations through a shared recurrent block for $K$ iterations (`num_loops`, typically 1 to 4) before projecting to output logits:

$$\mathbf{h}^{(k+1)} = \text{RecurrentLayerBlock}(\mathbf{h}^{(k)}, \mathbf{x}), \quad k \in \{1, \dots, K\}$$

This performs multi-step iterative latent reasoning directly in embedding space prior to emitting token 1. While looping increases prefill compute by ~30–60%, it completely eliminates the memory-bandwidth bottleneck of autoregressive token generation, providing high-fidelity first-token logits on complex constraints with **zero generated CoT tokens**.

#### Seamless Integration via `RecurrentPyTorchBackend`
LogitGate provides native support for latent recurrent architectures via `RecurrentPyTorchBackend` and the `recurrent://` URI scheme:

```python
from logitgate import LogitGate

# Elastic recurrent depth: 2 internal latent loops before first-token projection
gate = LogitGate(
    rubric="Does the SQL query use proper parameterized statements without dynamic string interpolation?",
    backend="recurrent://nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16?num_loops=2",
)

verdict = gate.evaluate(candidate_sql)
print(f"Passed: {verdict.passed} (Calibrated P={verdict.probability:.4f})")
```

#### Empirical Multi-GPU & Recurrent Depth Benchmark (RTX 5090 vs. RTX 4070 Across $K=1, 2, 4$)

To rigorously evaluate how latent recurrent looping impacts first-token logit gating, we executed the full 4-suite benchmark suite against **NVIDIA-Nemotron-Labs Dory-4B** (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, 7.5 GB safetensors, BF16 native PyTorch execution) across both physical GPUs:
- **Device 0 (`cuda:0`)**: NVIDIA GeForce RTX 5090 (32 GB GDDR7 VRAM, Blackwell `sm_120`, CUDA 13.2 runtime).
- **Device 1 (`cuda:1`)**: NVIDIA GeForce RTX 4070 (12 GB GDDR6X VRAM, Ada Lovelace `sm_89`).

Raw benchmark measurements are stored in [`benchmarks/dory_results.json`](../benchmarks/dory_results.json).

##### 1. Suite 2: Reasoning Depth Tiers Across Recurrent Depth ($K=1, 2, 4$)
Evaluating the 32-item reasoning depth suite (8 items per tier) across recurrent loop passes:

| Model & Device | Recurrent Loops | Tier A (Surface) | Tier B (Compound) | Tier C (Multi-Hop) | Tier D (Adversarial) | Overall 32-Item Reasoning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dory-4B (RTX 5090)** | $K=1$ | 4/8 (50.0%) | 4/8 (50.0%) | 6/8 (75.0%) | 3/8 (37.5%) | **17/32 (53.1%)** |
| **Dory-4B (RTX 5090)** | $K=2$ | 5/8 (62.5%) | **6/8 (75.0%)** | 6/8 (75.0%) | 4/8 (50.0%) | **21/32 (65.6%)** *(+12.5% jump)* |
| **Dory-4B (RTX 5090)** | $K=4$ | 4/8 (50.0%) | 4/8 (50.0%) | 5/8 (62.5%) | 5/8 (62.5%) | **18/32 (56.2%)** |
| **Dory-4B (RTX 4070)** | $K=1$ | 4/8 (50.0%) | 3/8 (37.5%) | 6/8 (75.0%) | 3/8 (37.5%) | **16/32 (50.0%)** |
| **Dory-4B (RTX 4070)** | $K=2$ | 5/8 (62.5%) | **5/8 (62.5%)** | 6/8 (75.0%) | 4/8 (50.0%) | **20/32 (62.5%)** *(+12.5% jump)* |
| **Dory-4B (RTX 4070)** | $K=4$ | 5/8 (62.5%) | 4/8 (50.0%) | 4/8 (50.0%) | 5/8 (62.5%) | **18/32 (56.2%)** |
| *Baseline Feed-Forward 4B (`gemma4:e4b`)* | *N/A ($K=1$)* | *6/8 (75.0%)* | *4/8 (50.0%)* | *4/8 (50.0%)* | *4/8 (50.0%)* | *18/32 (56.2%)* |
| *Baseline Feed-Forward 14B (`qwen2.5:14b`)* | *N/A ($K=1$)* | *8/8 (100.0%)* | *5/8 (62.5%)* | *5/8 (62.5%)* | *4/8 (50.0%)* | *22/32 (68.8%)* |

##### 2. Suite 3: Multi-Action Gates (`ChoiceGate` & `ScoreGate`)
Evaluating discrete 3-way routing (`ChoiceGate`, 6 items) and continuous 1–5 scoring monotonicity (`ScoreGate`, 6 items):

| Model & Device | Recurrent Loops | `ChoiceGate` Accuracy | `ScoreGate` Monotonicity | High Tier Mean | Med Tier Mean | Low Tier Mean | First-Token Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dory-4B (RTX 5090)** | $K=1$ | **6/6 (100.0%)** | 3/8 (37.5%) | 2.39 | 2.90 | 2.80 | 216.3 ms |
| **Dory-4B (RTX 5090)** | $K=2$ | **6/6 (100.0%)** | **6/8 (75.0%)** | **3.13** | **3.10** | **2.91** | 335.5 ms |
| **Dory-4B (RTX 5090)** | $K=4$ | **6/6 (100.0%)** | 3/8 (37.5%) | 2.68 | 3.27 | 3.15 | 628.9 ms |
| **Dory-4B (RTX 4070)** | $K=1$ | **6/6 (100.0%)** | 2/8 (25.0%) | 2.35 | 2.85 | 2.81 | 327.8 ms |
| **Dory-4B (RTX 4070)** | $K=2$ | **6/6 (100.0%)** | **5/8 (62.5%)** | **3.07** | **3.15** | **2.98** | 436.0 ms |
| **Dory-4B (RTX 4070)** | $K=4$ | **6/6 (100.0%)** | 3/8 (37.5%) | 2.70 | 3.31 | 3.14 | 809.1 ms |
| *Baseline Feed-Forward 4B (`gemma4:e4b`)* | *N/A* | *6/6 (100.0%)* | *3/8 (37.5%)* | *3.20* | *3.15* | *2.90* | *76.0 ms* |
| *Baseline Feed-Forward 14B (`qwen2.5:14b`)* | *N/A* | *6/6 (100.0%)* | *6/8 (75.0%)* | *4.25* | *3.10* | *1.85* | *115.0 ms* |

##### 3. Suite 4: Long-Context Haystack & Hardware Throughput Scaling (RTX 5090 vs. RTX 4070)
Evaluating needle-in-a-haystack compliance across prompt lengths up to 16,000 tokens:

| Context Length | RTX 5090 ($K=1$) | RTX 4070 ($K=1$) | RTX 5090 ($K=2$) | RTX 4070 ($K=2$) | RTX 5090 Speedup | Correct Retrieval ($\Delta P > 0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short (~200 tok)** | 231.8 ms | 337.3 ms | 358.0 ms | 430.9 ms | 1.4x | Yes ($\Delta P = +0.134$) |
| **Medium (~1,000 tok)** | 293.0 ms | 1,153.2 ms | 435.0 ms | 1,761.0 ms | 3.9x | Yes ($\Delta P = +0.100$) |
| **Long (~4,000 tok)** | 894.1 ms | 3,552.1 ms | 1,374.3 ms | 5,546.2 ms | 4.0x | Yes ($\Delta P = +0.030$) |
| **Very Long (~8,000 tok)** | 1,769.4 ms | 6,584.0 ms | 2,741.0 ms | 10,432.1 ms | 3.8x | Edge ($\Delta P = -0.090$) |
| **Massive (~16,000 tok)** | **3,754.7 ms** | **13,928.6 ms** | **5,812.2 ms** | **22,172.2 ms** | **3.8x** | **Yes ($\Delta P = +0.095$)** |

##### 4. Suite 1: 24-Item Multi-Domain Cascade Efficiency
- Across all configurations, the 3-layer cascade (`ASTGate >> LogitGate >> JudgeGate`) consistently outperformed the baseline standalone generative judge:
  - **Baseline Standalone Generative Judge**: 9/24 to 12/24 accuracy (37.5%–50.0%), taking 109s–195s and consuming 960 output tokens.
  - **Cascade (`on_ambiguous_only`)**: **14/24 to 16/24 accuracy (58.3%–66.7%)**, running in 91s–177s (**1.1x–1.3x wall-time speedup**) and saving **200–280 output tokens** per 24 items by short-circuiting clean passes at Layer 1 and AST violations at Layer 0.

#### Key Architectural Findings:
1. **The $K=2$ Latent Recurrence Sweet Spot**:
   - In both GPUs, recirculating hidden states through the recurrent block twice ($K=2$) delivers an immediate **+12.5% boost** in compound reasoning ($53.1\% \to 65.6\%$ on RTX 5090; $50.0\% \to 62.5\%$ on RTX 4070), elevating the compact 4B model within 1 item of the 14B dense model (`qwen2.5:14b` at 68.8%) with **zero generated output tokens**.
   - Concurrently, `ScoreGate` pairwise monotonicity **doubles** ($37.5\% \to 75.0\%$).
   - Increasing recurrence to $K=4$ causes a slight regression ($56.2\%$), replicating NVIDIA's findings that $K=2$ reaches optimal latent equilibrium for general semantic tasks, while excessive looping ($K=4$) can oversmooth embeddings on concise prompts.
2. **Flawless Multi-Way Routing**:
   - `ChoiceGate` achieved a perfect **100% (6/6)** routing accuracy across every single tested configuration ($K=1, 2, 4$ on both GPUs), proving that discrete routing logits are extremely stable in latent recurrent architectures.
3. **Blackwell Memory Bandwidth Advantage**:
   - On the massive 16,000-token haystack evaluation, the RTX 5090 completed the forward pass in **3.75s** ($K=1$) and **5.81s** ($K=2$), outperforming the RTX 4070 (13.9s and 22.2s) by **~3.8x**, demonstrating the substantial value of GDDR7 memory bandwidth for high-throughput zero-token gating.

---

### 6. Reasoning-Model Suppression, Fail-Fast Batching, & Selective Score/Choice Cascades

Empirical verification of thinking-prefill suppression on `/v1/chat/completions`, `MultiRubricGate(fail_fast=True)` short-circuiting, async `warm_prefix` scheduling, and selective `ScoreGate`/`ChoiceGate` abstention cascades across local RTX 5090 (32 GB), RTX 4070 (12 GB), and Pure CPU (`options={"num_gpu": 0}`) endpoints:

#### A. OpenAI-Compatible Reasoning-Model Suppression (`multi_domain_24.json`, `logit_temperature=1.0`)

| Model (`OpenAICompatBackend`) | Hardware | `suppress_thinking=False` Valid / Accuracy (Failures) | `suppress_thinking=True` Valid / Accuracy | p50 / Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| **`deepseek-r1:32b`** | RTX 5090 (32 GB) | `0/24 (0.0%)` : `24 BackendError` (empty `top_logprobs`) | **24/24 (100.0%)** | **161.7 ms / 165.5 ms** |
| **`gemma4:26b`** | RTX 5090 (32 GB) | `0/24 (0.0%)` : `24 AMBIGUOUS` (`unobserved_target_tokens` trap) | **24/24 (100.0%)** | **162.9 ms / 164.1 ms** |
| **`gemma4:e4b`** | RTX 4070 (12 GB) | `24/24` directional (`193.4 ms` p50) | **24/24 directional (`23/24` decisive)** | **183.3 ms / 182.8 ms** |

#### B. `MultiRubricGate(fail_fast=True)` & Async `warm_prefix`

Evaluated across 6 5-rubric service implementations (4 defective, 2 clean) with `100%` `passed` decision parity between `fail_fast=False` and `fail_fast=True`:

| Model & Device | `fail_fast=False` Calls / Wall Time | `fail_fast=True` Calls / Wall Time | Calls Saved | Wall Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b` (RTX 4070 GPU)** | 60 calls / 2,498.3 ms | **36 calls / 1,670.8 ms** | **40.0%** | **1.50x** |
| **`gemma4:e4b` (RTX 4070 GPU)** | 60 calls / 4,602.5 ms | **32 calls / 2,934.5 ms** | **46.7%** | **1.57x** |
| **`qwen2.5-coder:3b` (Pure CPU `num_gpu=0`)** | 60 calls / 82.11 s | **12 calls / 44.12 s** | **80.0%** | **1.86x (-38.0 s)** |

On single-slot `Ollama` servers where concurrent HTTP calls already serialize in the runner scheduler, sequential `warm_prefix=True` vs parallel `warm_prefix=False` on cold ~1,500-token candidate prefixes (`N=5` timed runs after warmup) adds only ~1–3% round-trip overhead (`355.0 ± 3.5 ms` vs `343.6 ± 1.9 ms` on `qwen2.5-coder:3b` GPU; `526.4 ± 2.4 ms` vs `516.1 ± 3.7 ms` on `gemma4:e4b` GPU; `27.90 s` vs `27.85 s` on CPU) while preventing 2x cold-prefill races under multi-slot parallelism (`OLLAMA_NUM_PARALLEL >= 2` or `llama-server --parallel N`).

#### C. Selective `ScoreGate` & `ChoiceGate` Abstention Cascades

- **`ScoreGate(threshold=4.0, fail_threshold=2.5, max_score_std=1.25) >> JudgeGate`:** Across 4 clear and borderline code-quality cases, selective escalation lifted `qwen2.5-coder:3b` GPU accuracy from **3/4 to 4/4** (`749.4 ms` mean, fast-pathing clear passes in `88.2 ms` and escalating borderline `E[S]=3.43` to `JudgeGate`), while `gemma4:e4b` GPU fast-pathed `3/4` clear cases in `178–190 ms` (`654.4 ms` mean).
- **`ChoiceGate(min_confidence=0.65, min_margin=0.15) >> JudgeGate`:** Fast-pathed `3/4` clear support routing tickets in `68–164 ms` (`361.3 ms` mean on `qwen2.5-coder:3b`, `498.9 ms` mean on `gemma4:e4b`) and selectively escalated the ambiguous multi-domain ticket.

---

### 7. Unified Turnkey Conformal Calibration Profiles (`v0.6.0`)

Out-of-sample evaluation of joint post-temperature split-conformal calibration (`fit_calibration_profile(..., target_risk_alpha=0.10, calibrate_multiclass=True)`) vs. uncalibrated (`T=1.0`, static `0.20/0.85`) and temperature-only (`T=T_bin`, stale `0.20/0.85`) heuristics on [`benchmarks/data/multi_domain_24.json`](../benchmarks/data/multi_domain_24.json) with Layer-2 escalation to `gemma4:26b` (`JudgeGate`, `think=False`, `max_tokens=256`: standalone accuracy `24/24 (100.0%)`, mean `976.9 ms`, median `894.4 ms` on RTX 5090). Profiles were fitted in **2.87–6.40 s** on a strictly disjoint 40-item holdout set (`20` Pass / `20` Fail, `guarantee_holds=True` at $\alpha = 0.10$ since $\min(m_+, m_-) = 20 \ge 19$):

#### A. Binary Post-Temperature Conformal Cascade (`MultiDomain-24`, Out-of-Sample)

| Layer-1 Model (RTX 4070) | Regime | Fitted $T_{\text{bin}}$ / $T_{\text{mc}}$ | Cutoffs ($\tau_{\text{reject}}, \tau_{\text{accept}}$) | L1 Fast-Path Rate | L1 Selective Error | Cascade Accuracy | Cascade Mean (Speedup vs 26B Judge) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b`** | Uncalibrated (`T=1.0`) | `1.0000` / - | `0.2000, 0.8500` | 8/24 (33.3%) | 0/24 (0.0%) | 24/24 (100.0%) | 790.1 ms (`1.24x`) |
| **`qwen2.5-coder:3b`** | Temp-Only + Heuristic Cutoffs | `0.6135` / - | `0.2000, 0.8500` | 15/24 (62.5%) | **1/24 (4.2%)** | 23/24 (95.8%) | 483.8 ms (`2.02x`) |
| **`qwen2.5-coder:3b`** | **Unified `CalibrationProfile` ($\alpha=0.10$)** | **`0.6135` / `0.8399`** | **`0.1096, 0.9100`** | 10/24 (41.7%) | **0/24 (0.0%)** | **24/24 (100.0%)** | 685.0 ms (`1.43x`) |
| **`llama3.2:3b`** | Uncalibrated (`T=1.0`) | `1.0000` / - | `0.2000, 0.8500` | 16/24 (66.7%) | **3/24 (12.5%!)** | 21/24 (87.5%) | 419.2 ms (`2.33x`) |
| **`llama3.2:3b`** | Temp-Only + Heuristic Cutoffs | `1.8045` / - | `0.2000, 0.8500` | 5/24 (20.8%) | 0/24 (0.0%) | 24/24 (100.0%) | 893.6 ms (`1.09x`) |
| **`llama3.2:3b`** | **Unified `CalibrationProfile` ($\alpha=0.10$)** | **`1.8045` / `1.2441`** | **`0.5167, 0.8281`** | **12/24 (50.0%)** | **1/24 (4.2%)** | **23/24 (95.8%)** | **522.7 ms (`1.87x`)** |
| **`gemma4:e4b`** | Uncalibrated (`T=1.0`) | `1.0000` / - | `0.2000, 0.8500` | 23/24 (95.8%) | 0/24 (0.0%) | 24/24 (100.0%) | 224.8 ms (`4.35x`) |
| **`gemma4:e4b`** | Temp-Only + Heuristic Cutoffs | `4.6404` / - | `0.2000, 0.8500` | 20/24 (83.3%) | 0/24 (0.0%) | 24/24 (100.0%) | 357.2 ms (`2.74x`) |
| **`gemma4:e4b`** | **Unified `CalibrationProfile` ($\alpha=0.10$)** | **`4.6404` / `3.7126`** | **`0.1647, 0.2497`** | **24/24 (100.0%)** | **0/24 (0.0%)** | **24/24 (100.0%)** | **178.7 ms (`5.47x`)** |

*Key empirical findings:*
- **Eliminating `T_old -> T_new` Drift**: On `qwen2.5-coder:3b`, applying `T=0.6135` without recalibrating the heuristic `0.85` accept cutoff leaked a false-pass at Layer 1 (`23/24`), whereas the unified profile tightened $\tau_{\text{accept}}$ to `0.9100` on the `T=0.6135` scale (`0/24` L1 error, `24/24` cascade accuracy). Conversely, uncalibrated `llama3.2:3b` violated the $10\%$ risk target (`12.5%` L1 error), while temperature-only heuristic `0.85` collapsed fast-path rate to `20.8%`; the conformal profile delivered **2.4x more fast-paths (`50.0%`)** at `4.2% <= 10%` risk.
- **Data-Driven Asymmetric Cutoffs**: On `gemma4:e4b` ($\text{AUROC}=0.9925$), fitting conformal cutoffs directly on post-temperature holdout scores captured the model's asymmetric probability calibration band `(0.1647, 0.2497)`, unlocking **100.0% fast-path resolution (`178.7 ms` mean, `5.47x` faster than 26B Judge, `24/24` accurate)**.

#### B. Selective `ChoiceGate` & `ScoreGate` Conformal Calibration (`22`-Item Combined Suite)

Evaluated via 2-fold out-of-sample cross-validation across [`choice_and_score_12.json`](../benchmarks/data/choice_and_score_12.json) + [`adversarial_choice_and_score_32.json`](../benchmarks/data/adversarial_choice_and_score_32.json) (`ChoiceGate` $\alpha=0.10$, `guarantee_holds=True` on all folds; `ScoreGate` $\alpha=0.20$, best-effort `guarantee_holds=False` as 8 positives / 14 negatives $< 9$ per class):

| Model (RTX 4070) | `ChoiceGate` Unguarded Err | `ChoiceGate` 2-Fold Conformal ($\alpha=0.10$) | `ScoreGate` Unguarded Err (`@3.5`) | `ScoreGate` 2-Fold Conformal ($\alpha=0.20$) |
| :--- | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b`** | 1/22 (4.5%) | **1/22 (4.5% <= 10%)**, 8/22 abstained | 4/22 (18.2%) | **2/22 (9.1%)**, 6/22 abstained |
| **`gemma4:e4b`** | 1/22 (4.5%) | **1/22 (4.5% <= 10%)**, 0/22 abstained | 3/22 (13.6%) | **0/22 (0.0%)**, 2/22 abstained |

#### C. Pure-CPU Zero-Probe Cold Start (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`)

Loading the saved `CalibrationProfile` on an unseen BYO model alias in pure CPU mode (`2.09 GB RAM`, `0.0 GB VRAM`, confirmed zero lazy micro-probe calls across 12 stratified items from `MultiDomain-24`):
- **`LogitGate(profile=...)`**: `5/12 (41.7%)` fast-path rate, **`0/12 (0.0%)` selective error**, mean latency **`2,126.7 ms`** (`2,265.0 ms` median).
- **`JudgeGate(profile=..., max_tokens=160, think=False)`**: `9/12 (75.0%)` accuracy, mean latency `14,027.6 ms` (`14,491.6 ms` median): **`6.6x` mean speedup** via `LogitGate` with portable profile reuse.

---

### 8. Zero-Token Tournament & Best-of-N Preference Ranking (`v0.7.0`)

#### A. Pairwise Positional-Bias Cancellation vs. 26B Generative Judge (`24` Multi-Domain Pairs)

Evaluating `24` domain-matched `(good, bad)` pairs constructed from [`benchmarks/data/multi_domain_24.json`](../benchmarks/data/multi_domain_24.json) (`6` pairs each across RAG groundedness, support policy, SQL safety, and Python security):

| Model & Hardware | Regime (`max_tokens`) | Order-1 Slot-A Acc | Order-2 Slot-B Acc | Order Flip Rate | Symmetric Debiased Acc (Fwd / Rev) | Mean `\|Bias\|` (nats) | Mean / Median Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemma4:26b`** (RTX 5090, `17.84 GB`) | Generative Critique (`max_tokens=160, think=False`, `50.0` tok) | `24/24 (100.0%)` | - | - | - | - | `432.2 ms / 381.0 ms` |
| **`qwen2.5-coder:3b`** (RTX 4070, `2.09 GB`) | Single-Pass vs. **Pairwise `RankGate`** (`0` tok) | `24/24 (100.0%)` | **`0/24 (0.0%)`** | **`24/24 (100.0%)`** | **`24/24 (100.0%)` / `24/24 (100.0%)`** | `1.7272` | **`119.3 ms / 120.3 ms` (`3.62x`)** |
| **`llama3.2:3b`** (RTX 4070, `2.33 GB`) | Single-Pass vs. **Pairwise `RankGate`** (`0` tok) | `24/24 (100.0%)` | **`2/24 (8.3%)`** | **`22/24 (91.7%)`** | **`24/24 (100.0%)` / `24/24 (100.0%)`** | `1.7603` | **`113.9 ms / 113.9 ms` (`3.79x`)** |
| **`gemma4:e4b`** (RTX 4070, `3.21 GB`) | Single-Pass vs. **Pairwise `RankGate`** (`0` tok) | `24/24 (100.0%)` | `22/24 (91.7%)` | `2/24 (8.3%)` | **`24/24 (100.0%)` / `24/24 (100.0%)`** | `1.0255` | **`235.8 ms / 235.5 ms` (`1.83x`)** |

Without symmetric A/B swap cancellation, `qwen2.5-coder:3b` and `llama3.2:3b` exhibit **91.7%–100.0% order flip rates** (nearly always choosing Slot `A`). Symmetric log-odds averaging eliminates the positional bias mathematically (`1.73–1.76` nats), recovering **100.0% order-invariant accuracy** in both directions at **1.83x–3.79x lower latency** than the 26B generative judge.

#### B. Best-of-4 Mode Scaling & All-Bad Pool Rejection (`20` Pools across Python, SQL, JSON, RAG)

Across `16` valid-winner 4-candidate pools and `4` all-bad pools on RTX 4070 GPU:

- **Mode Comparison (`16` Valid Pools):**
  - `qwen2.5-coder:3b`: `listwise` (`2` calls/pool) `12/16 (75.0%)` at `277.3 ms` mean (`141.1 ms` median); `tournament` (`6` calls/pool) **`15/16 (93.8%)`** at `301.8 ms`; `round_robin` (`12` calls/pool) **`15/16 (93.8%)`** with `cycle_residual = 0.1480` at `575.4 ms`.
  - `gemma4:e4b`: `listwise` (`2` calls/pool) **`16/16 (100.0%)`** at **`280.3 ms` mean** (`266.3 ms` median); `tournament` (`6` calls/pool) **`16/16 (100.0%)`** at `617.8 ms`; `round_robin` (`12` calls/pool) **`16/16 (100.0%)`** with `cycle_residual = 1.2885` at `1,237.0 ms`.
- **Naive `RankGate` vs. Contract-Filtered `BestOfNGate(pre_gate=ASTGate, post_gate=LogitGate)` (`20` Pools):**
  - On `qwen2.5-coder:3b`, Layer-0 `ASTGate` deterministically pruned **`28/80 (35.0%)` invalid candidates**, lifting valid Top-1 accuracy from `12/16 (75.0%)` to **`14/16 (87.5%)`** and raising All-Bad Pool Rejection Rate from **`0/4 (0.0%)` to `3/4 (75.0%)`** (`213.7 ms` mean).
  - On `gemma4:e4b`, naive `RankGate` selected a defective winner on all 4 all-bad pools (`0/4 (0.0%)` rejection), whereas `BestOfNGate` achieved **`4/4 (100.0%)` All-Bad Pool Rejection** (`423.3 ms` mean, `35.0%` Layer-0 prune rate; using an uncalibrated static `0.55` post-gate cutoff rejected `3/16` valid pools (`13/16`), underscoring why `post_gate` should load a fitted `CalibrationProfile`).

#### C. Pure-CPU Pairwise Ranking (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`, `12` Stratified Pairs)

Verified `0.0 GB` VRAM (`2.09 GB` system RAM via `/api/ps`, unloaded cleanly to `[]`):
- **Pairwise `RankGate` (`max_tokens=1`, 2 swapped calls):** **`12/12 (100.0%)`**, mean **`5,849.2 ms`** (`5,734.6 ms` median, `7,067.7 ms` p95).
- **CPU Generative Pairwise (`max_tokens=160, think=False`, `90.2` mean tok):** `12/12 (100.0%)`, mean `6,923.7 ms` (`6,730.4 ms` median, `7,530.3 ms` p95): **`1.18x` mean speedup** without multi-token variance or answer-format parsing failures.

---

### 9. Production Telemetry & Self-Distilling JSONL Sinks (`v0.8.0`)

#### A. Per-Verdict Telemetry & PII-Redaction Overhead (`N = 1,000` Evaluations)

Measured across `N = 1,000` sequential evaluations containing realistic PII payloads (email, US SSN, Luhn-valid payment card, and `sk-` API token):

| Gate & Sink Configuration | Median Latency | p95 Latency | Median Sink Overhead |
| :--- | :---: | :---: | :---: |
| `LogitGate(StubBackend)`: Telemetry Disabled | `0.0176 ms` | `0.0223 ms` | - |
| `LogitGate(StubBackend)`: `MemoryTelemetrySink(privacy="full")` | `0.0296 ms` | `0.0900 ms` | **`+0.0120 ms` (`12.0 us`)** |
| `LogitGate(StubBackend)`: `JsonlTelemetrySink(privacy="full")` | `0.0403 ms` | `0.1023 ms` | **`+0.0227 ms` (`22.7 us`)** |
| `LogitGate(StubBackend)`: `JsonlTelemetrySink(privacy="redact_pii")` | `0.0546 ms` | `0.1240 ms` | **`+0.0370 ms` (`37.0 us`)** |
| `ASTGate(syntax=True)` - Telemetry Disabled | `0.0130 ms` | `0.0136 ms` | - |
| `ASTGate(syntax=True)`: `JsonlTelemetrySink(privacy="redact_pii")` | `0.0385 ms` | `0.1020 ms` | **`+0.0255 ms` (`25.5 us`)** |

Even full linear-time PII scrubbing plus thread-locked `O_NOFOLLOW` JSONL disk persistence adds **`37.0 us`** per verdict (**`<0.04%`** of a typical `~100 ms` GPU Layer-1 check).

#### B. Live GPU Discrepancy Mining & Zero-LLM-Call Offline Recalibration (`40`-Item Burn-In -> `MultiDomain-24` Holdout)

Phase 1 streams `40` diverse burn-in items (`19` Teacher Pass / `21` Teacher Fail across [`UNIVERSAL_CALIBRATION_DATASET`](../src/logitgate/debiasing/permutation.py) and [`EPISTEMIC_REASONING_TEST_CASES.md`](../EPISTEMIC_REASONING_TEST_CASES.md)) through an uncalibrated student `LogitGate` (`RTX 4070`) paired with a full-audit `JudgeGate(model="gemma4:26b", trigger="always")` teacher (`RTX 5090`, `17.84 GB` VRAM, standalone `MultiDomain-24` baseline: **`24/24 (100.0%)`**, `999.2 ms` mean / `928.0 ms` median) into `CompositeTelemetrySink([JsonlTelemetrySink, DistillationSink])`.

Phase 2 calls `recalibrate_from_telemetry(distill.jsonl, target_risk_alpha=0.10)` (**`0` LLM calls**, `<5 ms` CPU optimization) and evaluates the refitted profile on the disjoint `24`-item [`multi_domain_24.json`](../benchmarks/data/multi_domain_24.json) holdout (`JudgeGate(trigger="on_ambiguous_only")`):

| Student Model (RTX 4070) | Burn-In Discrepancies (`40` Items -> `80` Symmetric SFT Rows) | Zero-Call Recalibration (`0` LLM Calls) | Uncalibrated Holdout (`MultiDomain-24`) | Telemetry-Recalibrated Holdout (`alpha=0.10`) | Cascade Mean / Median (`Speedup` vs 26B L2) |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`qwen2.5-coder:3b`** (`2.09 GB`) | `6/40 (15.0%)` (`6 polarity_flip`, `34 ambiguous_resolved`) | **`4.94 ms`** -> `T=0.5743`, `[0.0915, 0.9453]` (`AUROC=0.885`) | `37.5%` fast-path, `1/24 (4.2%)` L1 err, `23/24 (95.8%)` acc | `29.2%` fast-path, **`0/24 (0.0%)` L1 err**, **`24/24 (100.0%)` acc** | `865.4 ms / 1,005.7 ms` (`1.15x`) |
| **`llama3.2:3b`** (`2.33 GB`) | `13/40 (32.5%)` (`4 false_accept_caught`, `9 polarity_flip`, `27 ambiguous_resolved`) | **`4.87 ms`** -> `T=2.1391`, `[0.4732, 0.8343]` (`AUROC=0.895`) | `62.5%` fast-path, `2/24 (8.3%)` L1 err, `22/24 (91.7%)` acc | `37.5%` fast-path, **`1/24 (4.2%)` L1 err**, **`23/24 (95.8%)` acc** | `723.8 ms / 874.6 ms` (`1.38x`) |
| **`gemma4:e4b`** (`3.21 GB`) | `4/40 (10.0%)` (`1 false_accept_caught`, `3 false_reject_overturned`, `36 ambiguous_resolved`) | **`4.70 ms`** -> `T=5.0541`, `[0.1792, 0.7634]` (`AUROC=0.988`) | `95.8%` fast-path, `0/24 (0.0%)` L1 err, `24/24 (100.0%)` acc | **`91.7%` fast-path**, **`0/24 (0.0%)` L1 err**, **`24/24 (100.0%)` acc** | **`270.3 ms / 185.8 ms` (`3.70x / 5.00x`)** |

Across the three compact student architectures, `DistillationSink` automatically isolated all four discrepancy regimes (`false_accept_caught`, `false_reject_overturned`, `polarity_flip`, and `ambiguous_resolved`), exported `80` `2x` symmetric `A/B` + `B/A` SFT rows and `80` DPO rows per model, and cut holdout Layer-1 selective errors without issuing a single calibration probe call.

#### C. Pure-CPU Self-Distilling Stream (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`, `16` Items)

Verified `0.0 GB` VRAM (`2.09 GB` system RAM via `/api/ps`, unloaded cleanly to `[]`):
- **Pure-CPU Layer-1 Student + Teacher Escalation (`16` balanced items):** `1,699.3 ms` median Layer-1 CPU latency (`1,721.1 ms` mean), `16/16` distilled (`1` discrepancy, `32` symmetric SFT rows exported).
- **Zero-Call Offline Recalibration:** **`2.60 ms`** (`0` LLM calls) -> `T_bin = 0.1854` (`probe_auroc = 0.9531`, Brier `0.1070 -> 0.0656`, ECE `0.2357 -> 0.0973`).

---

### 10. Deconfounded Cross-Domain Counterfactual Suite & Domain-Disjoint Conformal Transfer (`v0.8.1`)

#### A. Dataset Confounder & Cross-Split Leakage Audit (`0` LLM Calls)

Auditing legacy `MultiDomain-24` via `audit_benchmark_dataset()` uncovered two structural validity risks before building downstream streaming (`v0.9.0`), drift (`v0.10.0`), and distillation (`v0.11.0`) evaluations:
1. **Length Confounding:** Passing candidates averaged `361.3` chars vs. `250.9` chars for failing candidates (`30.6%` relative length imbalance, **`length_auroc = 0.7465`**).
2. **Sample-Power Deficit & Domain Overlap:** `12` samples/class could not satisfy split-conformal coverage floors (`n >= 19`/class for `alpha = 0.10`; `n >= 39`/class for `alpha = 0.05`), and code/SQL/RAG archetypes overlapped with calibration sets.

To eliminate length and lexical shortcuts, `v0.8.1` constructs paired **`2x2` contrastive counterfactual quads** (`clean_pass`, `hard_positive` carrying cautionary/defect vocabulary while satisfying the rubric, `direct_fail`, and `hard_negative` carrying polished compliance vocabulary while subtly violating the rubric) across strictly disjoint domain families:

| Dataset Artifact | Cases (`Pass / Fail`) | `alpha = 0.05` Power (`>=39`/cls) | Pass/Fail Char Diff | `length_auroc` | Rubric-Blind Unigram LOO Acc | Audit Warnings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Legacy [`multi_domain_24.json`](../benchmarks/data/multi_domain_24.json) | `24` (`12 / 12`) | `False` | `30.6%` | `0.7465` | `12.5%` | `2` |
| **Split A:** [`cal_distill_80.json`](../benchmarks/data/cal_distill_80.json) (`4` in-domain families) | **`80` (`40 / 40`)** | **`True`** | **`4.3%`** | **`0.5397`** | **`7.5%`** | **`0`** |
| **Split B:** [`ood_generalization_96.json`](../benchmarks/data/ood_generalization_96.json) (`6` unseen OOD families) | **`96` (`48 / 48`)** | **`True`** | **`6.8%`** | **`0.5979`** | **`19.8%`** | **`0`** |
| Upgraded [`EPISTEMIC_REASONING_TEST_CASES.md`](../EPISTEMIC_REASONING_TEST_CASES.md) | **`80` (`40 / 40`)** | **`True`** | **`4.1%`** | **`0.5366`** | **`7.5%`** | **`0`** |
| [`streaming_trajectories_32.json`](../benchmarks/data/streaming_trajectories_32.json) (`v0.9.0` multi-chunk streams) | `32` (`16 / 16`) | `False` (eval-only) | **`3.8%`** | **`0.4551`** | **`15.6%`** | `1` (size note) |
| [`drift_regimes_120.json`](../benchmarks/data/drift_regimes_120.json) (`v0.10.0` 4-regime shift stream) | **`120` (`51 / 69`)** | **`True`** | **`8.8%`** | **`0.5718`** | **`17.5%`** | **`0`** |

Running `check_split_leakage("benchmarks/data/ood_generalization_96.json", ...)` against `cal_distill_80.json`, `EPISTEMIC_REASONING_TEST_CASES.md`, `multi_domain_24.json`, `UNIVERSAL_CALIBRATION_DATASET`, and `BUILTIN_CASES` confirms **`is_clean = True`** across every pair (**`0` exact candidate/rubric overlaps, `0` shared domains, and maximum 3-gram Jaccard similarity `<= 0.0147`**). Rubric-blind unigram leave-one-out accuracy falls well below `50%` (`7.5%–19.8%`) because `hard_positive` and `hard_negative` deliberately invert surface vocabulary cues within every quad.

#### B. Standard vs. Contrastive Counterfactual Stratification (`OOD-96`, 6 Held-Out Reasoning Families)

Evaluating standalone forced-binary models across the 6 unseen OOD families (`causal_and_statistical_inference`, `quantitative_unit_and_temporal`, `compositional_and_negated_constraints`, `regulatory_and_clinical_exceptions`, `tool_call_and_state_preconditions`, and `epistemic_calibration_and_hedging`) exposes real architectural separation and opposing polarity biases that were masked by `MultiDomain-24`:

| Model & Hardware | Overall `OOD-96` Acc | Standard (`clean_pass` + `direct_fail`, `N=48`) | Counterfactual (`hard_pos` + `hard_neg`, `N=48`) | Quadrant Breakdown (`clean_pass` / `hard_pos` / `direct_fail` / `hard_neg`) | Score AUROC (`All / Std / Hard`) | Mean / Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemma4:26b` `JudgeGate`** (RTX 5090, `17.84 GB`, `think=False`) | **`94/96 (97.9%)`** | **`48/48 (100.0%)`** | **`46/48 (95.8%)`** | `100.0% / 91.7% / 100.0% / 100.0%` | - | `1,132.9 ms / 1,089.4 ms` |
| **`qwen2.5-coder:3b` `LogitGate`** (RTX 4070, `2.09 GB`, `0` tok) | `57/96 (59.4%)` | `34/48 (70.8%)` | `23/48 (47.9%)` | `41.7% / 16.7% / 100.0% / 79.2%` *(over-skeptical)* | `0.7218 / 0.8455 / 0.5590` | `133.5 ms / 110.1 ms` |
| **`llama3.2:3b` `LogitGate`** (RTX 4070, `2.33 GB`, `0` tok) | `60/96 (62.5%)` | `34/48 (70.8%)` | `26/48 (54.2%)` | `91.7% / 87.5% / 50.0% / 20.8%` *(sycophantic)* | `0.7001 / 0.7899 / 0.5920` | `134.9 ms / 112.8 ms` |
| **`gemma4:e4b` `LogitGate`** (RTX 4070, `3.21 GB`, `0` tok) | **`81/96 (84.4%)`** | **`44/48 (91.7%)`** | **`37/48 (77.1%)`** | `87.5% / 75.0% / 95.8% / 79.2%` *(balanced)* | **`0.9440 / 0.9878 / 0.8854`** | `174.1 ms / 171.6 ms` |
| **`qwen2.5:14b` `LogitGate`** (RTX 4070, `9.09 GB`, `0` tok) | **`84/96 (87.5%)`** | **`45/48 (93.8%)`** | **`39/48 (81.2%)`** | `91.7% / 83.3% / 95.8% / 79.2%` *(balanced)* | **`0.9336 / 0.9948 / 0.8542`** | `183.7 ms / 179.2 ms` |

- **Opposing 3B Failure Modes Uncovered:** When stripped of length shortcuts and tested on non-code OOD domains, `qwen2.5-coder:3b` defaults to skepticism (`100.0%` on `direct_fail` but `16.7%` on `hard_positive`), whereas `llama3.2:3b` defaults to sycophancy (`91.7%` on `clean_pass` but `20.8%` on polished `hard_negative`).
- **4B–14B Semantic Separation:** `gemma4:e4b` (`3.21 GB` VRAM) and `qwen2.5:14b` (`9.09 GB` VRAM) achieve **`0.988–0.995` AUROC** on Standard OOD pairs and **`0.854–0.885` AUROC** on Hard Counterfactuals in a single `~175 ms` forward pass, while leaving clear `10–13 pp` headroom below the `26B` Layer-2 CoT teacher (`97.9%`).

#### C. Strict Domain-Disjoint Conformal Transfer (`Split A cal_distill_80.json` -> Zero-Shot `Split B ood_generalization_96.json`)

Each `CalibrationProfile` is fitted **exclusively on Split A** (`cal_distill_80.json`, `4` domains: Python, SQL, grounded summarization, support ops; `guarantee_holds = True` at both `alpha = 0.10` and `alpha = 0.05`) and evaluated **zero-shot on Split B** (`ood_generalization_96.json`, `6` completely unseen domains) inside a `LogitGate >> JudgeGate(gemma4:26b, trigger="on_ambiguous_only")` cascade:

| Layer-1 Model | Regime (Fitted on Split A Only) | Split-A `[tau_rej, tau_acc]` (`T_bin`) | OOD Layer-1 Error (`Std / Hard`) | OOD Fast-Path Rate | Cascade Accuracy | Cascade Mean Latency (`Speedup` vs 26B L2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b`** | Uncalibrated (`0.20, 0.85`, `T=1.0`) | `[0.2000, 0.8500]` (`1.0000`) | `2/96 (2.08%)` (`1 / 1`) | `18.8%` | `95.8%` | `1,092.8 ms` (`1.04x`) |
| | Split-A Conformal (`alpha = 0.10`) | `[0.3210, 0.7635]` (`1.4453`) | **`9/96 (9.38%)`** (`6 / 3`) | `35.4%` | `88.5%` | `867.1 ms` (`1.31x`) |
| | Split-A Conformal (`alpha = 0.05`) | `[0.3162, 0.7731]` (`1.3419`) | `9/96 (9.38%)` (`5 / 4`) | `35.4%` | `88.5%` | `870.3 ms` (`1.30x`) |
| **`llama3.2:3b`** | Uncalibrated (`0.20, 0.85`, `T=1.0`) | `[0.2000, 0.8500]` (`1.0000`) | `10/96 (10.42%)` (`3 / 7`) | `40.6%` | `87.5%` | `812.4 ms` (`1.39x`) |
| | Split-A Conformal (`alpha = 0.10`) | `[0.3910, 0.7828]` (`1.8616`) | **`6/96 (6.25%)`** (`2 / 4`) | `32.3%` | `90.6%` | `904.9 ms` (`1.25x`) |
| | Split-A Conformal (`alpha = 0.05`) | `[0.2817, 0.8669]` (`1.6771`) | **`0/96 (0.00%)`** (`0 / 0`) | `15.6%` | **`95.8%`** | `1,128.4 ms` (`1.00x`) |
| **`gemma4:e4b`** | Uncalibrated (`0.20, 0.85`, `T=1.0`) | `[0.2000, 0.8500]` (`1.0000`) | `13/96 (13.54%)` (`4 / 9`) | `95.8%` | `85.4%` | `228.0 ms` (`4.97x`) |
| | Split-A Conformal (`alpha = 0.10`) | `[0.0641, 0.2059]` (`2.9552`) | `11/96 (11.46%)` (`2 / 9`) | `88.5%` | `86.5%` | `326.7 ms` (`3.47x`) |
| | Split-A Conformal (`alpha = 0.05`) | `[0.0408, 0.9195]` (`2.9068`) | **`1/96 (1.04%)`** (`0 / 1`) | **`57.3%`** | **`96.9%`** | **`694.3 ms` (`1.63x`)** |
| **`qwen2.5:14b`** | Uncalibrated (`0.20, 0.85`, `T=1.0`) | `[0.2000, 0.8500]` (`1.0000`) | `9/96 (9.38%)` (`3 / 6`) | `97.9%` | `90.6%` | `209.3 ms` (`5.41x`) |
| | Split-A Conformal (`alpha = 0.10`) | `[0.2521, 0.9053]` (`9.0470`) | **`6/96 (6.25%)`** (`1 / 5`) | **`83.3%`** | **`92.7%`** | **`391.0 ms` (`2.90x`)** |
| | Split-A Conformal (`alpha = 0.05`) | `[0.0783, 0.9274]` (`9.4214`) | **`3/96 (3.12%)`** (`0 / 3`) | **`69.8%`** | **`94.8%`** | **`547.1 ms` (`2.07x`)** |

- **Zero-Shot Risk Control Across Unseen Domains:** At `alpha = 0.05` (`<= 5.0%` target Layer-1 error), `llama3.2:3b` (`0.00%`), `gemma4:e4b` (`1.04%`, `57.3%` fast-path, `96.9%` cascade accuracy), and `qwen2.5:14b` (`3.12%`, `69.8%` fast-path, `94.8%` cascade accuracy, `2.07x` speedup) all hold empirical OOD Layer-1 error strictly below the `5.0%` nominal bound despite **zero domain overlap** between calibration and evaluation.
- **When Exchangeability Breaks Under Severe Specialization Shift:** Code-specialized `qwen2.5-coder:3b` (`Split-A probe_auroc = 0.6963`, triggering the `AUROC < 0.70 -> T >= 1.0` regularizer) stays within budget at `alpha = 0.10` (`9.38% <= 10.0%`), but exhibits mild under-coverage (`9.38%` vs `5.0%`) at `alpha = 0.05` when shifted from 50% code/SQL onto 100% prose clinical/causal/regulatory domains, empirically motivating online covariate-shift monitoring and Gibbs-Candes Adaptive Conformal Inference (`v0.10.0`).

#### D. Pure-CPU Stratified OOD Validation (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`, `OOD-24`)

Evaluated across `24` stratified OOD items (`1` complete `2x2` quad from each of the 6 held-out domains) with `0.0 GB` VRAM (`2.09 GB` system RAM via `/api/ps`, unloaded cleanly to `[]`):
- **Split-A Conformal Cascade (`alpha = 0.10`):** **`2/24 (8.33% <= 10.0%)`** Layer-1 selective error, `29.2%` fast-path rate, **`22/24 (91.7%)`** cascade accuracy, `1,957.2 ms` median Layer-1 CPU latency (`2,841.5 ms` mean end-to-end cascade latency).

---

### 11. Mid-Stream Token Guardrails (`StreamGuard`) & Content-Addressed `VerdictCache` (`v0.9.0`)

#### A. Mid-Stream Verification Across `streaming_trajectories_32.json` (`4` Trajectory Archetypes)

Evaluated across [`benchmarks/data/streaming_trajectories_32.json`](../benchmarks/data/streaming_trajectories_32.json) (`32` streams sub-tokenized into `~18`-char SSE deltas across 4 balanced 8-item classes: `clean_throughout`, `false_alarm_prefix_resolved_safely` where early tokens quote prohibited code/clinical/SQL strings before prohibiting them, `early_onset_violation`, and `late_onset_pivot_violation`). All models use their pre-fitted **Split-A (`cal_distill_80.json`, `alpha = 0.05`) `CalibrationProfile`** with fixed, untuned guard parameters (`min_chars=40, stride_chars=48, max_wait_chars=72, clause_boundary=True`):

| Model & Gate Architecture | Streaming Regime | Overall Acc (`N=32`) | False-Alarm Mid-Abort (`N=8`) | Early-Violation Mid-Abort (`N=8`) | Early Upstream Char Savings | Violating Char Leak Rate | Checks / Stream | Mean Gate Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemma4:e4b` `LogitGate`** (RTX 4070) | Post-Hoc Full Stream | `29/32 (90.6%)` | `0.0%` | `0.0%` | `0.0%` | `0.0%` *(100% wait)* | `1.00` | `164.3 ms` |
| | `StreamGuard` (`optimistic`, `k=1`) | `29/32 (90.6%)` | **`0.0%`** | **`100.0%`** | **`44.5%`** | `60.1%` | `2.28` | `347.8 ms` (`169.0 ms` early) |
| | `StreamGuard` (`optimistic`, `k=2`) | `29/32 (90.6%)` | **`0.0%`** | `62.5%` | `8.4%` | `93.7%` | `2.62` | `404.8 ms` |
| | **`StreamGuard` (`buffered`, `k=1`)** | **`29/32 (90.6%)`** | **`0.0%`** | **`100.0%`** | **`44.5%`** | **`0.0%` (`0/1,613`)** | `2.25` | `344.5 ms` (`167.5 ms` early) |
| **`gemma4:e4b >> gemma4:26b` `Cascade`** | Post-Hoc Full Stream | **`32/32 (100.0%)`** | `0.0%` | `0.0%` | `0.0%` | `0.0%` *(100% wait)* | `1.00` | `244.4 ms` |
| | `StreamGuard` (`optimistic`, `k=1`) | `26/32 (81.2%)` | `37.5%` | `100.0%` | `51.3%` | `42.9%` | `1.78` | `773.3 ms` |
| | **`StreamGuard` (`optimistic`, `k=2`)** | **`31/32 (96.9%)`** | **`12.5%`** | `62.5%` | `8.4%` | `93.1%` | `2.56` | `1,103.9 ms` |
| | `StreamGuard` (`buffered`, `k=1`) | `26/32 (81.2%)` | `37.5%` | `100.0%` | `44.5%` | `1.9%` | `1.81` | `789.7 ms` |
| **`qwen2.5:14b` `LogitGate`** (RTX 4070) | Post-Hoc Full Stream | **`30/32 (93.8%)`** | `0.0%` | `0.0%` | `0.0%` | `0.0%` *(100% wait)* | `1.00` | `178.2 ms` |
| | `StreamGuard` (`optimistic`, `k=1`) | **`30/32 (93.8%)`** | **`0.0%`** | **`100.0%`** | **`51.3%`** | `56.4%` | `2.28` | `387.8 ms` |
| | `StreamGuard` (`optimistic`, `k=2`) | **`30/32 (93.8%)`** | **`0.0%`** | `75.0%` | `9.1%` | `93.1%` | `2.59` | `442.3 ms` |
| | **`StreamGuard` (`buffered`, `k=1`)** | `29/32 (90.6%)` | **`0.0%`** | **`100.0%`** | **`51.3%`** | **`0.0%` (`0/1,613`)** | `2.28` | `388.4 ms` |
| **`qwen2.5:14b >> gemma4:26b` `Cascade`** | Post-Hoc Full Stream | **`32/32 (100.0%)`** | `0.0%` | `0.0%` | `0.0%` | `0.0%` *(100% wait)* | `1.00` | `270.5 ms` |
| | `StreamGuard` (`optimistic`, `k=1`) | `26/32 (81.2%)` | `37.5%` | `100.0%` | `51.3%` | `42.9%` | `1.78` | `815.6 ms` |
| | **`StreamGuard` (`optimistic`, `k=2`)** | **`32/32 (100.0%)`** | **`0.0%`** | `75.0%` | `9.1%` | `92.6%` | `2.56` | `1,118.9 ms` |
| | `StreamGuard` (`buffered`, `k=1`) | `26/32 (81.2%)` | `37.5%` | `100.0%` | `51.3%` | **`0.0%` (`0/1,613`)** | `1.78` | `771.4 ms` (`166.7 ms` early) |

*Key empirical findings:*
- **Conformal Abstention Prevents Mid-Sentence False Alarms at Layer 1:** On standalone conformal `LogitGate` (`gemma4:e4b` and `qwen2.5:14b`), suspicious-looking incomplete prefixes (`false_alarm_prefix_resolved_safely`) land inside the `(tau_reject, tau_accept)` `AMBIGUOUS` band rather than `<= tau_reject`, producing **`0.0%` (`0/8`) false-alarm mid-stream aborts** even at `consecutive_failures=1` while catching **`100.0%` (`8/8`) of `early_onset_violation` streams** at the first checkpoint (`~168 ms`) and cancelling **`44.5%–51.3%`** of downstream token generation via `.close()`.
- **Why Hysteresis (`k=2`) Matters for Generative Cascades:** When `Cascade(LogitGate >> JudgeGate)` runs on incomplete mid-stream prefixes with `k=1`, Layer-1 `AMBIGUOUS` escalates truncated half-clauses to `JudgeGate`, which rejects `37.5%` of false-alarm prefixes prematurely. Raising `consecutive_failures=2` restores **`96.9%` (`31/32`)** accuracy on `gemma4:e4b >> gemma4:26b` and **`100.0%` (`32/32`, `0.0%` false-alarm aborts)** on `qwen2.5:14b >> gemma4:26b`.
- **Zero-Leakage Verified Window Delivery (`delivery="buffered"`):** Holding pending chunks across `AMBIGUOUS` checkpoints until a `PASSED` verification flushes them eliminates **`100%` of violating-character leakage (`0 / 1,613` chars leaked)** on standalone `LogitGate` (`gemma4:e4b` and `qwen2.5:14b`) and `qwen2.5:14b >> gemma4:26b` while retaining full **`44.5%–51.3%`** upstream generation cancellation on early violations.

#### B. Content-Addressed `VerdictCache` Latency, Disk Reload, & Concurrency (`OOD-96`)

Evaluated across all `96` unseen items of [`ood_generalization_96.json`](../benchmarks/data/ood_generalization_96.json) on `Cascade(gemma4:e4b [Split-A alpha=0.05] >> gemma4:26b)` (`95/96 = 99.0%` accuracy):

| Pass / Regime (`N = 96` Items) | Mean Latency | Median Latency | p95 Latency | Total Wall Time | Exact Verdict Parity | Speedup vs. Cold Live | Backend Calls / Saved Compute |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pass 1: Cold Live GPU Cascade** (populates `VerdictCache`) | `705.85 ms` | `194.39 ms` | `1,764.54 ms` | `67.76 s` | Baseline (`95/96` acc) | `1.0x` | `96` misses -> `96` puts |
| **Pass 2: Warm In-Memory `OrderedDict` LRU** | **`0.1142 ms`** | **`0.1046 ms`** | **`0.1570 ms`** | **`0.0110 s`** | **`96/96 (100.0%)`** | **`6,180.8x`** | **`0` calls** (`67.70 s` & `6,662` tok saved) |
| **Pass 3: Cold-RAM / Warm SQLite WAL Reload** (new instance) | **`0.1352 ms`** | **`0.1277 ms`** | **`0.1866 ms`** | **`0.0130 s`** | **`96/96 (100.0%)`** | **`5,220.8x`** | **`0` calls** (`96/96` disk hits) |
| **Pass 4: 8-Thread Concurrent Read Stress** (`768` evals) | `0.1620 ms/8` | - | - | `0.1244 s` | `768/768` hits | **`6,174.9 evals/s`** | **`0` calls** |

#### C. Pure-CPU `StreamGuard` & `VerdictCache` Validation (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`)

Evaluated across `16` stratified streaming trajectories (`4` per archetype) with `0.0 GB` VRAM (`2.09 GB` system RAM via `/api/ps`, unloaded cleanly to `[]`):
- **Pure-CPU `StreamGuard(delivery="buffered", k=1)` (`qwen2.5-coder:3b >> gemma4:26b`):** **`100.0%` (`4/4`) mid-stream abort rate** and **`48.9%` upstream generated-character savings** on `early_onset_violation` (`1,884.9 ms` mean gate latency vs. `2,577.6 ms` post-hoc full stream), **`0.0%` (`0 / 862`) violating-character leakage** across all violating trajectories, and `14/16 (87.5%)` overall accuracy (`16/16` post-hoc).
- **Pure-CPU `LogitGate` `cache_ambiguous=False` Safety:** Across the `16` full candidates on standalone CPU `LogitGate(qwen2.5-coder:3b)`, `VerdictCache` persisted the `5/16` decisive (`PASSED`/`FAILED`) verdicts to SQLite (reloading in `<0.15 ms`, saving `8.71 s` of CPU compute) while intentionally declining to cache the `11/16` `AMBIGUOUS` abstentions so uncertain inputs remain eligible for downstream escalation.

---

### 12. Online Drift Detection & Adaptive Conformal Control (`v0.10.0`)

#### A. Paired 120-Step Non-Stationary 4-Regime Evaluation (`drift_regimes_120.json`)

Evaluated across [`benchmarks/data/drift_regimes_120.json`](../benchmarks/data/drift_regimes_120.json) (`120` ordered steps partitioned into 4 sequential 30-step regimes: **Regime 1 `stationary_in_domain`** `[0..29]` balanced code/SQL/RAG/ops, **Regime 2 `covariate_domain_shift`** `[30..59]` unseen clinical/causal/regulatory/unit domains, **Regime 3 `adversarial_hard_counterfactual_surge`** `[60..89]` `80%`-fail hard counterfactuals with inverted surface vocabulary, and **Regime 4 `stationary_recovery`** `[90..119]` return to baseline distribution). All models initialize from their pre-fitted **Split-A (`cal_distill_80.json`) `CalibrationProfile`** with fixed, untuned Gibbs-Candes parameters (`step_size=1.5, min_theta=0.0, abstain_theta=8.0, audit_sample_rate=0.25`) inside a `LogitGate >> JudgeGate(gemma4:26b)` cascade (standalone `gemma4:26b` RTX 5090 Teacher baseline: **`116/120 (96.7%)`**, `1,166.9 ms` mean / `1,101.6 ms` median latency):

| Layer-1 Student (RTX 4070) | Policy (`Split-A` Initialized, Zero Test Tuning) | Overall Cascade Acc | Overall L1 Risk (`Errors / 120`) | Per-Regime L1 Selective Risk (`R1 Stat / R2 Shift / R3 Adv / R4 Recov`) | L2 Escalation + Audit Rate | Mean Cascade Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b`** (`2.09 GB`, `77.3 ms` L1) | Static Split-A (`alpha = 0.10`) | `92.5%` | `4.2%` (`5/120`) | `3.3% / 10.0% / 3.3% / 0.0%` | `74.2%` | `956.9 ms` |
| | **`ACI_Audit` (`alpha = 0.10`, `p = 0.25`, 0 labels)** | **`94.2%`** | **`3.3%` (`4/120`)** | `3.3% / 10.0% / 0.0% / 0.0%` | `92.5%` | `1,153.7 ms` |
| | Static Split-A (`alpha = 0.05`) | `90.8%` | `5.8%` (`7/120`) | `3.3% / 16.7% / 3.3% / 0.0%` | `70.8%` | `916.6 ms` |
| | **`ACI_Audit` (`alpha = 0.05`, `p = 0.25`, 0 labels)** | **`94.2%`** | **`3.3%` (`4/120`)** | `3.3% / 10.0% / 0.0% / 0.0%` | `95.0%` | `1,175.0 ms` |
| **`llama3.2:3b`** (`2.33 GB`, `70.2 ms` L1) | Static Split-A (`alpha = 0.10`) | `93.3%` | `4.2%` (`5/120`) | `3.3% / 6.7% / 3.3% / 3.3%` | `74.2%` | `925.8 ms` |
| | **`ACI_Audit` (`alpha = 0.10`, `p = 0.25`, 0 labels)** | **`95.8%`** | **`2.5%` (`3/120`)** | `3.3% / 3.3% / 0.0% / 3.3%` | `90.8%` | `1,124.2 ms` |
| | **`ACI_Drift_Coupled` (`alpha = 0.10` + `RollingDriftMonitor`)** | **`96.7%`** | **`2.5%` (`3/120`)** | `3.3% / 3.3% / 0.0% / 3.3%` | `94.2%` | `1,159.9 ms` |
| **`gemma4:e4b`** (`3.21 GB`, `175.9 ms` L1) | Static Split-A (`alpha = 0.10`) | `90.0%` | `9.2%` (`11/120`) | `6.7% / 13.3% / 13.3% / 3.3%` | `11.7%` | `322.1 ms` |
| | `ACI_Explicit` (`alpha = 0.10`, oracle labels) | `91.7%` | `5.8%` (`7/120`) | `6.7% / 10.0% / 3.3% / 3.3%` | `32.5%` | `552.3 ms` |
| | **`ACI_Audit` (`alpha = 0.10`, `p = 0.25`, 0 labels)** | **`94.2%`** | **`4.2%` (`5/120`)** | `3.3% / 10.0% / 0.0% / 3.3%` | `69.2%` | `982.8 ms` |
| | Static / `ACI_Audit` (`alpha = 0.05`) | **`97.5%`** | **`0.8%` (`1/120`)** | `0.0% / 0.0% / 0.0% / 3.3%` | `46.7% / 63.3%` | `745.1 / 927.7 ms` |
| **`qwen2.5:14b`** (`9.09 GB`, `188.8 ms` L1) | Static Default (`[0.20, 0.85]`, `T = 1.0`) | `94.2%` | `5.8%` (`7/120`) | `6.7% / 3.3% / 10.0% / 3.3%` | `12.5%` | `352.5 ms` |
| | **`ACI_Audit` from Default (`p = 0.25`, 0 labels)** | **`95.8%`** | **`2.5%` (`3/120`)** | `3.3% / 3.3% / 3.3% / 0.0%` | `71.7%` | `1,033.4 ms` |
| | Static Split-A (`alpha = 0.10`) | `94.2%` | `4.2%` (`5/120`) | `6.7% / 0.0% / 10.0% / 0.0%` | `16.7%` | `402.8 ms` |
| | **`ACI_Audit` (`alpha = 0.10`, `p = 0.25`, 0 labels)** | **`96.7%`** | **`1.7%` (`2/120`)** | `3.3% / 0.0% / 3.3% / 0.0%` | `72.5%` | `1,043.5 ms` |
| | **`ACI_Audit` (`alpha = 0.05`, `p = 0.25`, 0 labels)** | **`96.7%`** | **`0.8%` (`1/120`)** | `3.3% / 0.0% / 0.0% / 0.0%` | `78.3%` | `1,124.8 ms` |

*Distribution-Shift Diagnostics (`compute_psi` & `compute_ks_two_sample` vs. Regime 1 Reference):*
- **Specialization Shift vs. Generalist Invariance:** Code-specialized `qwen2.5-coder:3b` registers severe log-odds distribution shift immediately upon entering Regime 2's unseen prose domains (**`PSI = 1.3677`, `KS D = 0.5000`, `p = 0.0006`**) and Regime 3's adversarial surge (**`PSI = 2.1076`, `KS D = 0.7333`, `p < 0.0001`**), where static `alpha = 0.05` Layer-1 error spikes to `16.7%` before `ACI_Audit` cuts overall risk nearly in half (`5.8% -> 3.3%`, `0.0%` Regime-3 L1 error). Conversely, generalist `qwen2.5:14b` stays distributionally invariant across Regime 2 (`PSI = 0.0223`, `KS D = 0.1667`, `p = 0.7600`, `0.0%` L1 error) and triggers purely on Regime 3's label-prior / hard-counterfactual shift (**`PSI = 0.7109`, `KS D = 0.4667`, `p = 0.0017`**).
- **Zero-Human-Label Risk Reduction & Stationary Recovery:** Across all four student models at `alpha = 0.10`, autonomous `feedback_mode="cascade_audit"` (`p = 0.25` Horvitz–Thompson inverse-propensity updates + teacher override on audited fast-path errors) monotonically lowers Layer-1 selective risk (`gemma4:e4b`: **`9.2% -> 4.2%`**, reducing Regime-3 adversarial L1 errors from `13.3%` to **`0.0%`**; `qwen2.5:14b`: **`4.2% -> 1.7%`**, matching standalone `26B` `96.7%` accuracy) and relaxes `theta_t` from `theta_max = 5.85` back to `0.0` during Regime 4 (`stationary_recovery`) without ever loosening past the protective Split-A conformal floor (`min_theta = 0.0`).

#### B. Live Dual-GPU Uncached Wire Cascade + `RollingDriftMonitor` (`40` Steps, `gemma4:e4b >> gemma4:26b`)

Executed end-to-end over live HTTP (`gemma4:e4b` `3.21 GB` on RTX 4070 + `gemma4:26b` `17.84 GB` on RTX 5090, `10` steps/regime, `RollingDriftMonitor(window_size=10, step_interval=10)` coupled to `AdaptiveConformalController`):
- **Live Cascade Performance:** **`37/40 (92.5%)`** cascade accuracy, **`951.4 ms`** mean wire latency (`1,174.7 ms` median, `1,891.1 ms` p95, `38.05 s` total), `60.0%` Layer-2 invocation rate (abstentions + `25%` shadow audits), and **`1/40 (2.5% <= 10.0%)`** empirical Layer-1 selective risk (`3.6%` when decided).
- **Clean Drift Alert Precision:** Cochran/Jeffreys finite-sample PSI smoothing fired **exact `severe` alerts on the two true shift windows** (`psi_log_odds = 0.7931` with `escalation_rate_delta = +0.40` on covariate shift; `psi_log_odds = 0.6712` with `pos_bias_delta = +0.5110` on adversarial surge) with zero false-positive stationary recovery alerts.

#### C. Pure-CPU Adaptive Conformal & Drift Validation (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`, `24` Steps)

Verified `0.0 GB` VRAM (`2.09 GB` system RAM via `/api/ps`, unloaded cleanly to `[]`) across `24` stratified steps (`6`/regime):
- **Pure-CPU Latency & Zero Selective Risk:** `2,077.8 ms` mean / `2,044.0 ms` median Layer-1 CPU latency (`49.87 s` total), **`0/24 (0.0%)` Layer-1 selective risk** (`decided_count = 2`, `abstained_count = 22`), and immediate shift detection (`psi_log_odds = 1.0993`, `KS D = 1.0000`, `p < 0.0001`).

---

### 13. Two-Tier BYO Teacher-to-Student Distillation (`v0.11.0`)

#### A. Zero-Shot Domain-Disjoint BYO Transfer (`Split A` Teacher-Labeled -> Unseen `OOD-96`)

To test whether distillation teaches transferable calibration and polarity invariants rather than domain-specific memorization, **Split A** ([`benchmarks/data/cal_distill_80.json`](../benchmarks/data/cal_distill_80.json), `80` items across `4` domains: Python, SQL, RAG summarization, support ops; `max_candidate_jaccard = 0.0147` and `0` shared domains against Split B) was adjudicated live by Teacher **`gemma4:26b` `JudgeGate`** (RTX 5090, `79/80 = 98.8%` Split-A accuracy; `94/96 = 97.9%` standalone Split-B accuracy) with **zero human ground-truth labels fed into student fitting**. Every calibration profile, `<1 KB` `LogitMetaCalibrator` (`621–627 B` JSON), `DistilledAnchorBank` (`24` pairs, `45.9 KB`), and `0`-LLM-call telemetry bundle (`15.3–19.5 ms` compile time) was fitted **strictly on Split A** and evaluated **100% zero-shot** across all `96` items (`6` unseen reasoning families) of [`benchmarks/data/ood_generalization_96.json`](../benchmarks/data/ood_generalization_96.json):

| BYO Student (RTX 4070) | Regime (Fitted Strictly on Teacher-Labeled Split A) | Unseen `OOD-96` Acc (`Std / CF`) | `Hard Pos / Hard Neg` | AUROC | Brier | OOD L1 Risk (`alpha=0.10`) | Fast-Path | `Cascade` Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:3b`** (`627 B` meta JSON) | Raw Baseline (`T = 1.0`, `[0.20, 0.85]`) | `59.4%` (`68.8% / 50.0%`) | `20.8% / 79.2%` | `0.7205` | `0.2423` | `3.1%` | `20.8%` | `94.8%` |
| *(over-skeptical specialist)* | Split-A Scalar `CalibrationProfile` (`alpha = 0.10`) | `59.4%` (`68.8% / 50.0%`) | `20.8% / 79.2%` | `0.7203` | `0.2330` | `7.3%` | `33.3%` | `90.6%` |
| | Forced Off-Domain Anchors (`min_rubric_overlap = 0.0`, ablation) | `63.5%` (`68.8% / 58.3%`) | **`58.3% / 58.3%`** | `0.6684` | `0.2406` | `7.3%` | `28.1%` | `90.6%` |
| | `0`-Call Telemetry Bundle (`min_rubric_overlap = 0.0`, `15.3 ms`) | **`63.5%`** (`68.8% / 58.3%`) | `54.2% / 62.5%` | `0.6688` | `0.2418` | `10.4%` | `35.4%` | `87.5%` |
| | **Default Relevance-Gated `DistillationBundle` (`overlap = 0.25`)** | **`61.5%`** (`70.8% / 52.1%`) | **`50.0% / 54.2%`** | `0.7144` | **`0.2382`** | `13.5%` | `40.6%` | `85.4%` |
| **`llama3.2:3b`** (`622 B` meta JSON) | Raw Baseline (`T = 1.0`, `[0.20, 0.85]`) | `63.5%` (`70.8% / 56.2%`) | `87.5% / 25.0%` | `0.7027` | `0.2455` | `12.5%` | `45.8%` | `85.4%` |
| *(sycophantic generalist)* | Split-A Scalar `CalibrationProfile` (`alpha = 0.10`) | `63.5%` (`70.8% / 56.2%`) | `87.5% / 25.0%` | `0.7027` | `0.2275` | `10.4%` | `39.6%` | `87.5%` |
| | Forced Off-Domain Anchors (`min_rubric_overlap = 0.0`, ablation) | `57.3%` (`66.7% / 47.9%`) | `37.5% / 58.3%` | `0.6243` | `0.2443` | **`5.2%`** | `18.8%` | **`92.7%`** |
| | **Default Relevance-Gated `DistillationBundle` (`overlap = 0.25`)** | **`66.7%` (`79.2% / 54.2%`)** | **`66.7% / 41.7%`** | `0.6970` | **`0.2256`** | `11.5%` | `38.5%` | `86.5%` |
| **`gemma4:e4b`** (`621 B` meta JSON) | Raw Baseline (`T = 1.0`, `[0.20, 0.85]`) | `84.4%` (`91.7% / 77.1%`) | `79.2% / 75.0%` | `0.9355` | `0.1418` | `13.5%` | `93.8%` | `85.4%` |
| *(reasoning-capable 4B)* | Split-A Scalar `CalibrationProfile` (`alpha = 0.10`) | `84.4%` (`91.7% / 77.1%`) | `79.2% / 75.0%` | `0.9384` | `0.1156` | `10.4%` *(>10%)* | `79.2%` | `88.5%` |
| | Forced Off-Domain Anchors (`min_rubric_overlap = 0.0`, ablation) | `80.2%` (`89.6% / 70.8%`) | `87.5% / 54.2%` | `0.8566` | `0.1584` | **`9.4%`** | `63.5%` | `88.5%` |
| | **Default Relevance-Gated `DistillationBundle` (`overlap = 0.25`)** | `83.3%` (`91.7% / 75.0%`) | **`83.3% / 66.7%`** | **`0.9388`** | **`0.1146`** | **`9.4%` (`<=10%`)** | **`77.1%`** | **`89.6%`** |

*Key scientific findings:*
- **Why Unpenalized Intercept + 5-Feature Meta-Calibration Beats Scalar Temperature Scaling (`T`):** Scalar temperature scaling $p = \sigma(z_{\text{sym}} / T)$ leaves every forced-binary (`p >= 0.50`) prediction unchanged (`59.4% -> 59.4%` on `qwen2.5-coder:3b`, `63.5% -> 63.5%` on `llama3.2:3b`). Fitting `<1 KB` `LogitMetaCalibrator` on Teacher-labeled Split A shifts the skewed decision boundary zero-shot across all `6` unseen domains: lifting `qwen2.5-coder:3b` `hard_positive` recall **`20.8% -> 50.0% (+29.2 pp)`**, lifting `llama3.2:3b` `hard_negative` specificity **`25.0% -> 41.7% (+16.7 pp, overall +3.2 pp)`**, and improving `gemma4:e4b` Brier **`0.1418 -> 0.1146 (-19.2%)`** while bringing OOD Layer-1 selective error inside target `alpha = 0.10` (**`9.4% <= 10.0%`**, `Cascade` accuracy **`85.4% -> 89.6%`**).
- **In-Rubric Gain vs. Off-Domain Few-Shot Pollution (`min_rubric_overlap = 0.25`):** On **in-rubric Split A** (`5`-fold out-of-fold with `exclude_exact=True`), matched `[CALIBRATION ANCHORS]` raise OOF AUROC on both `qwen2.5-coder:3b` (**`0.6936 -> 0.7283`**) and `gemma4:e4b` (**`0.9531 -> 0.9637`**). Conversely, force-injecting SQL/Python anchors (`min_rubric_overlap = 0.0`) into unrelated clinical and statistical `max_tokens=1` prompts distracts general instruction models (`gemma4:e4b` AUROC `0.9355 -> 0.8566`, `llama3.2:3b` `0.7027 -> 0.6243`) and inflates prompt-prefill latency. Gating anchor injection via Szymkiewicz–Simpson rubric token overlap (`min_rubric_overlap = 0.25` default) achieves **`100%` (`80/80`) in-rubric recall** and **`100%` (`96/96`) unseen-OOD abstention**, routing OOD rubrics automatically to `unanchored_meta_calibrator` with zero extra prompt tokens.

#### B. Tier-B Architecture-Agnostic PEFT `all-linear` LoRA Compiler (`logitgate[distill]`)

Evaluated across `3` distinct HuggingFace `CausalLM` model families on the `160` symmetric (`A/B` + `B/A`) first-token Teacher-labeled Split-A examples using **one identical `LoRADistillConfig(rank=8, lora_alpha=16, target_modules="all-linear")`** (zero model-specific layer names):

| CausalLM Family | Auto-Hooked Linear Projections (`target_modules="all-linear"`) | Trainable / Total Params | Adapter `.safetensors` | First-Token Loss (`Init -> Final`) | Compile Wall Time | Multi-Runtime Exports Verified |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`LlamaForCausalLM`** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | `16,384 / 123,200` (`13.30%`) | `67.41 KB` | `5.4926 -> 4.6367` (**`-15.6%`**) | `3.04 s` | `Modelfile.ollama`, `serve_vllm.sh`, `serve_llamacpp.sh`, post-LoRA bundle |
| **`Qwen2ForCausalLM`** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | `16,384 / 123,456` (`13.27%`) | `67.41 KB` | `5.4863 -> 4.6537` (**`-15.2%`**) | `2.56 s` | `Modelfile.ollama`, `serve_vllm.sh`, `serve_llamacpp.sh`, post-LoRA bundle |
| **`MistralForCausalLM`** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | `16,384 / 123,200` (`13.30%`) | `67.41 KB` | `5.4801 -> 4.6875` (**`-14.5%`**) | `2.61 s` | `Modelfile.ollama`, `serve_vllm.sh`, `serve_llamacpp.sh`, post-LoRA bundle |

#### C. Pure-CPU Execution (`options={"num_gpu": 0}`, `qwen2.5-coder:3b`, Stratified `OOD-24`)

Verified `0.0 GB` VRAM (`2.09 GB` system RAM via `/api/ps`, unloaded cleanly to `{'gpu0': [], 'gpu1': []}`) across `24` stratified unseen OOD items (`6` domains × `4` variants):
- **Raw Baseline (`T = 1.0`):** `70.8%` accuracy (`50.0%` `hard_positive`), `4.2%` Layer-1 selective error, `2,020.2 ms` mean / `1,957.6 ms` median CPU latency.
- **Forced Off-Domain Anchors (`min_rubric_overlap = 0.0`, ablation):** `58.3%` accuracy, **`5,278.4 ms` mean CPU latency (`2.61x` slower)** due to off-domain prompt-prefill tokens.
- **Default Relevance-Gated `DistillationBundle` (`min_rubric_overlap = 0.25`):** **`75.0%` accuracy (`+4.2 pp`, lifting `hard_positive` `50.0% -> 83.3%`)**, **`8.3% <= 10.0%`** Layer-1 selective error (`91.7%` `Cascade` accuracy), and **`1,987.8 ms` mean / `1,919.9 ms` median CPU latency** (`0` prompt-bloat overhead).

---

### 14. `v0.13.0` Decision-Grade Statistical Engine, Deconfounded $N=80$ Core Suites & Canonical External Public Benchmarks

`v0.13.0` upgrades `logitgate.benchmark` with pure-Python Wilson 95% confidence intervals (`wilson_confidence_interval`), deterministic percentile bootstrap intervals (`bootstrap_metric_ci`), exact two-sided McNemar paired significance tests (`mcnemar_exact_test`), probabilistic calibration & selective-abstention metrics (`compute_calibration_metrics`: Brier score, 10-bin ECE, tie-aware Mann-Whitney AUROC, and Area Under the Selective Risk-Coverage Curve `AURC`), `p99_ms` tail latency, and separated `cold_prefix_p50_ms` vs `warm_prefix_p50_ms` KV-cache timings.

#### A. Psychometric Deconfounding & Cross-Split Leakage Verification ($N=80$ Core Suites)

Four new domain-disjoint, natural unpadded benchmark suites (`benchmarks/data/*_80.json`) close prior modality blind spots across multi-chunk RAG groundedness, MCP / OpenAI tool-call safety, multilingual/CJK/Cyrillic/Arabic/emoji BPE boundaries, and `RankGate` / `BestOfNGate` preference pools, passing `audit_benchmark_dataset` with **0 warnings** and **0 cross-split leakage** against both `cal_distill_80.json` and `ood_generalization_96.json`:

| Core Dataset | $N$ (Pos / Neg) | `holds_at_alpha_005` | `char_length_rel_diff` | `length_auroc` | `lexical_loo_accuracy` | Audit Warnings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rag_groundedness_80.json` | `80` (`40 / 40`) | `True` | `0.0041` | `0.5497` | `0.0000` | `0` |
| `agent_mcp_tool_safety_80.json` | `80` (`40 / 40`) | `True` | `0.0052` | `0.4866` | `0.0375` | `0` |
| `multilingual_unicode_bpe_80.json` | `80` (`40 / 40`) | `True` | `0.0690` | `0.5463` | `0.0000` | `0` |
| `rank_pairwise_80.json` (`60` pairs + `20` pools) | `120` (`60 / 60`) | `True` | `0.0149` | `0.4478` | `0.3083` | `0` |

#### B. Live Dual-GPU Cascade & Student Discrimination Across $N=80$ Suites

Calibrated strictly once on disjoint Split-A `cal_distill_80.json` ($\alpha = 0.10$, zero target-domain labels) with `gemma4:26b` (RTX 5090) as Layer-2 teacher:

| Suite ($N=80$) | Layer-1 Model (RTX 4070) | L1 `LogitGate` Acc | L1 `AUROC` / `Selective Risk` | `Cascade` Acc (`95% Wilson CI`) | Fast-Path Skip | `JudgeGate` (`gemma4:26b`) | McNemar $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rag_groundedness_80`** | `llama3.2:3b` (`3B`) | `76.2%` | `0.9775` / **`0.0%`** | **`100.0%`** (`[95.4%, 100.0%]`) | `41.2%` | `98.8%` | `1.0000` |
| | `gemma4:e4b` (`4B`) | `98.8%` | `0.9994` / `1.2%` | **`100.0%`** (`[95.4%, 100.0%]`) | `98.8%` | `98.8%` | `1.0000` |
| **`agent_mcp_tool_safety_80`** | `llama3.2:3b` (`3B`) | `55.0%` | `0.9744` / **`0.0%`** | **`98.8%`** (`[93.3%, 99.8%]`) | `20.0%` | `97.5%` | `1.0000` |
| | `gemma4:e4b` (`4B`) | `100.0%` | `1.0000` / **`0.0%`** | **`100.0%`** (`[95.4%, 100.0%]`) | `100.0%` | `97.5%` | `0.5000` |
| **`multilingual_unicode_bpe_80`** | `llama3.2:3b` (`3B`) | `63.7%` | `0.9738` / **`0.0%`** | **`98.8%`** (`[93.3%, 99.8%]`) | `28.7%` | `95.0%` | `0.2500` |
| | `gemma4:e4b` (`4B`) | `100.0%` | `1.0000` / **`0.0%`** | **`100.0%`** (`[95.4%, 100.0%]`) | `100.0%` | `95.0%` | `0.1250` |

- **`rank_pairwise_80` (`gemma4:e4b`, `0` generated tokens):** Pairwise `RankGate(symmetric_debiasing=True)` achieved **`100.0%` (`60/60`, 95% CI `[94.0%, 100.0%]`)** with **`100.0%` position-swap consistency** at `228.9 ms` mean latency (vs `98.3%` single-pass), and contract-filtered `BestOfNGate` achieved `65.0%` across 4-way pools (including `all_bad` AST/post-gate rejection pools).
- **Pure-CPU Validation (`options={"num_gpu": 0}`, `0.0 GB` VRAM, `qwen2.5-coder:3b`, `MCP-24`):** Confirmed `size_vram = 0` (`2.09 GB` system RAM), `p50 = 2,312.3 ms` (`cold_prefix_p50 = 2,418.7 ms` vs `warm_prefix_p50 = 2,301.3 ms`, `p99 = 2,621.2 ms`), **`AUROC = 1.0000`**, **`AURC = 0.0617`**, and **`7.7% <= 10.0%` selective risk** at `54.2%` uncalibrated coverage.

#### C. Live Canonical External Public Benchmarks (`benchmarks/external/`, Zero-Token `gemma4:e4b`)

Evaluated live via `benchmarks/external/run_external.py` (`seed=42`, $N=60$ per suite) across commercial-clean public benchmarks (`MIT` / `Apache-2.0`):

| External Public Suite | License | Task | Single-Pass Acc (`Swap Consistency`) | Symmetric Debiased Acc (`95% CI`) | Symmetric Swap Consistency | Key Empirical Takeaway |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`princeton-nlp/LLMBar`** (`Adversarial_Neighbor`) | `MIT` | Pairwise (`60`) | `55.0%` (`75.0%`) | **`58.3%`** (`[45.7%, 69.9%]`) | **`100.0%` (`+25.0 pp`)** | Eliminates `25.0%` slot-order flip rate on surface-bias adversarial distractors at `0` generated tokens. |
| **`ScalerLab/JudgeBench`** (`gpt` split) | `MIT` | Pairwise (`60`) | `58.3%` (`71.7%`) | **`61.7%`** (`[49.0%, 72.9%]`) | **`93.3%` (`+21.6 pp`)** | `+3.4 pp` accuracy and `+21.6 pp` swap consistency on hard math/coding/reasoning pairs. |
| **`truthfulqa/truthful_qa`** (`validation`) | `Apache-2.0` | Pairwise (`60`) | `81.7%` (`91.7%`) | **`80.0%`** (`[68.2%, 88.2%]`) | **`98.3%` (`+6.6 pp`)** | High misconception rejection with near-perfect (`98.3%`) slot invariance. |
| **`wandb/RAGTruth-processed`** (`test`, `40` faithful / `20` hallu) | `MIT` | Grounded RAG (`60`) | - | `LogitGate`: **`71.7%`** (`[59.2%, 81.5%]`, `AUROC 0.7888`, `ECE 0.0933`) | `JudgeGate` (`gemma4:26b`): **`85.0%`** (`[73.9%, 91.9%]`) | Baseline when `4k`-char source articles are concatenated inside `candidate`; motivates dedicated `<reference_context_NONCE>` RAG slot (`v0.15.0`). |

---

### 15. Summary: When to Use Layer 1 (`LogitGate`) vs. Escalate to Layer 2 (`JudgeGate`)

- **Use Layer 1 (`LogitGate`): First-Token Forward Pass ($O(1)$ Circuit Depth)**:
  - **Semantic Rubric & Policy Compliance**: Factual groundedness against context, tone/safety compliance, PII/secret leak checks, and atomic requirement verification via `evaluate_batch`.
  - **Discrete Routing & Classification (`ChoiceGate`)**: Selecting downstream agent tools, intent buckets, or workflow branches in ~40–90 ms.
  - **Continuous Rubric Scoring (`ScoreGate`)**: Probability-weighted expected value scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$) with symmetric scale reversal.
  - **Zero-Token Candidate Ranking & Best-of-$N$ Selection (`RankGate` & `BestOfNGate`)**: Comparing candidate completions across `pairwise`, `listwise`, `round_robin` (Hodge ranking + Condorcet residual), and `tournament` regimes with deterministic Layer-0 pruning and post-selection verification.
- **Escalate to Layer 2 (`JudgeGate`): Autoregressive Generation**:
  - **Multi-Step Deductive Reasoning**: Proof verification or multi-hop logic where conclusions depend on intermediate reasoning steps.
  - **Complex Symbolic Arithmetic**: Calculations requiring scratchpad calculation or token-by-token recurrence (or route deterministic math checks to Layer 0 `ASTGate` / `ContractGate`).
  - **Diagnostic Reporting**: When a human developer or logging pipeline needs an actionable explanation of *why* an artifact failed (or replay a CI failure locally via `logitgate explain`).
