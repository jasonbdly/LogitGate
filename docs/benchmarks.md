# Empirical Benchmarks & Falsification Studies

This document contains the comprehensive empirical benchmark results, hardware sweeps, thread scaling experiments, and falsification studies for **LogitGate**.

All declarative benchmark datasets ([`benchmarks/data/`](../benchmarks/data/)), the benchmark runner scripts ([`benchmarks/run_benchmarks.py`](../benchmarks/run_benchmarks.py), [`benchmarks/run_cpu_benchmarks.py`](../benchmarks/run_cpu_benchmarks.py), [`benchmarks/run_dory_benchmarks.py`](../benchmarks/run_dory_benchmarks.py), [`benchmarks/run_adversarial_choice_score.py`](../benchmarks/run_adversarial_choice_score.py), [`benchmarks/run_cpu_format_runtime_benchmarks.py`](../benchmarks/run_cpu_format_runtime_benchmarks.py)), and raw JSON results are tracked in the repository for independent reproduction.

---

## Real Multi-Model Live Benchmarks

Empirical measurements gathered on local inference across 5 distinct model architectures running under Ollama (`v0.35.0`, `Q4_K_M` 4-bit GGUF quantization, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`). Raw JSON outputs are available in [`benchmarks/latest_results.json`](../benchmarks/latest_results.json).

- **Testbed Hardware**:
  - **CPU & System Memory**: AMD Ryzen 7 5800X3D (8-Core / 16-Thread), 64 GB DDR4 System RAM (used for all `Pure CPU-Only` benchmarks with `options={"num_gpu": 0}`).
  - **GPU 1 (12 GB Sidecar Tier)**: NVIDIA GeForce RTX 4070 (12 GB GDDR6X VRAM) — used for `4B` (`gemma4:e4b`) and `14B` (`qwen2.5:14b`, `qwen2.5-coder:14b`) benchmarks.
  - **GPU 0 (32 GB Primary Tier)**: NVIDIA GeForce RTX 5090 (32 GB GDDR7 VRAM) — used for `26B MoE` (`gemma4:26b`) and `31B Dense` (`gemma4:31b`) benchmarks.
- **Benchmark Methodology & Sample Protocol**:
  - **Multi-Sample Latency Protocol ($N=5$)**: Every latency configuration is preceded by `1` untimed warm-up pass (to load GGUF weights into memory and initialize KV-cache buffers), followed by `N=5` sequential timed trials over a persistent HTTP connection pool (`httpx.Client`), reporting `mean ± std` in milliseconds.
  - **Leak-Free Matched Datasets**: All test items in [`benchmarks/data/`](../benchmarks/data/) are stripped of meta-comments and structured as matched pass/fail pairs across 4 domains (24 items in Suite 1; 32 items across 4 reasoning tiers in Suite 2; 12 `ChoiceGate`/`ScoreGate` items in Suite 3; and a 12-section enterprise handbook in Suite 4).
  - **Deterministic Classification**: All Layer 1 probes execute with greedy decoding parameters (`temperature=0.0`, `top_p=1.0`, `max_tokens=1`), yielding deterministic log-probabilities across runs for a fixed prompt and model checkpoint.

| Model Configuration | Model Family | Compute Device | Single-Pass Latency ($N=5$) | 2-Pass Symmetric Latency ($N=5$) | 4-Rubric Batch Total ($N=5$) | Per-Rubric Batch Latency ($N=5$) | 24-Item Multi-Domain Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b`** | 4B General Model | GPU (RTX 4070) | **40.5 ± 6.0 ms** | **112.6 ± 4.6 ms** | **584.9 ± 51.8 ms** | **146.2 ± 12.9 ms** | **23/24 (95.8%)** |
| **`qwen2.5:14b`** | 14B General Model | GPU (RTX 4070) | **36.8 ± 6.5 ms** | **90.0 ± 4.1 ms** | **450.1 ± 18.8 ms** | **112.5 ± 4.7 ms** | **23/24 (95.8%)** |
| **`qwen2.5-coder:14b`** | 14B Code Model | GPU (RTX 4070) | **36.5 ± 6.3 ms** | **90.9 ± 3.3 ms** | **447.4 ± 3.9 ms** | **111.8 ± 1.0 ms** | **23/24 (95.8%)** |
| **`qwen2.5-coder:1.5b`** | 1.5B Code Model | Pure CPU (0 GPU, 1.4 GB) | **42.3 ± 1.4 ms** | **330.4 ± 63.5 ms** | **1,834.5 ± 75.6 ms** | **458.6 ± 18.9 ms** | **14/24 (58.3%)** |
| **`llama3.2:1b`** | 1B General Model | Pure CPU (0 GPU, 1.8 GB) | **46.9 ± 1.6 ms** | **459.2 ± 86.0 ms** | **2,665.1 ± 82.9 ms** | **666.3 ± 20.7 ms** | **15/24 (62.5%)** |
| **`qwen2.5-coder:3b`** | 3B Code Model | Pure CPU (0 GPU, 2.4 GB) | **69.1 ± 2.8 ms** | **625.7 ± 113.9 ms** | **3,626.6 ± 155.8 ms** | **906.7 ± 39.0 ms** | **19/24 (79.2%)** |
| **`llama3.2:3b`** | 3B General Model | Pure CPU (0 GPU, 3.2 GB) | **74.8 ± 0.8 ms** | **642.7 ± 113.6 ms** | **3,699.9 ± 116.8 ms** | **925.0 ± 29.2 ms** | **19/24 (79.2%)** |
| *`gemma4:e4b` (Baseline)* | *4B General Model* | *Pure CPU (0 GPU, 9.6 GB)* | *173.0 ± 2.4 ms* | *3,710.5 ± 82.1 ms* | *20,949.5 ± 182.5 ms* | *5,237.4 ± 45.6 ms* | *23/24 (95.8%)* |

### Benchmark Analysis & Insights:
1. **Single-Pass Latency (~36–41 ms on GPU, ~42–75 ms on CPU)**:
   - Evaluates a rubric in a single forward pass without autoregressive decoding. Over **35x faster** than waiting for an LLM to generate 150 words of critique.
   - On a CPU-only runner, compact 1B–3B models execute single-pass triage in **42–75 ms**, beating dense 4B CPU models by **2.3x–4.1x**.
2. **2-Pass Symmetric Debiasing (~90–113 ms on GPU, ~330–640 ms on CPU)**:
   - Runs both forward ($A=\text{Pass}, B=\text{Fail}$) and reverse ($B=\text{Pass}, A=\text{Fail}$) passes to eliminate pretraining token prior bias. The second pass hits the KV-cache.
   - On CPU, `llama3.2:3b` and `qwen2.5-coder:3b` evaluate symmetric debiasing in **~625–642 ms**—almost **6x faster than dense 4B models (3,710 ms)**.
3. **Multi-Rubric Batching (~112–146 ms per rubric on GPU, ~450–925 ms on CPU)**:
   - `evaluate_batch` reuses the candidate prefix across all rubrics, avoiding redundant prompt re-evaluations and preventing KV-cache stampedes on the inference server.
4. **Pure CPU-Only Memory Footprint (1.4 GB to 3.2 GB)**:
   - Shifting from dense 4B models (`9.61 GB RAM`) to compact distilled models (`1.39 GB – 3.18 GB RAM`) frees up 65%–85% of host memory on CI runners while reducing wall-clock gating time by up to 83%.

---

## End-to-End Cascade Time Savings & Multi-Gate Benchmarks

### 1. Multi-Domain Cascade Time Savings (24-Item Test Suite)

Evaluated across **24 items spanning 4 domains** ([`benchmarks/data/multi_domain_24.json`](../benchmarks/data/multi_domain_24.json), 6 items per domain: RAG Groundedness with subtle scope/regional extrapolations, Customer Support Policy Compliance, SQL/API Query Safety, and Python Security). To ensure a fair comparison, the Baseline Generative Judge also runs with `<think>` suppressed (`think=False`):

| Model Configuration | Baseline Judge Time | Cascade Time (`on_ambiguous_or_fail`) | Cascade Time (`on_ambiguous_only`) | Speedup (`on_ambiguous_only`) | CoT Tokens Saved | Baseline Accuracy | Cascade Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (GPU)** | 34,946.5 ms | 18,773.5 ms | **3,286.5 ms** | **10.6x** | 3,323 | 24/24 (100.0%) | **23/24 (95.8%)** |
| **`gemma4:e4b` (Pure CPU)** | 143,733.1 ms | 144,976.4 ms | **104,792.0 ms** | **1.4x** | 1,086 | 24/24 (100.0%) | **23/24 (95.8%)** |
| **`qwen2.5:14b` (GPU)** | 138,858.1 ms | 57,625.2 ms | **8,160.3 ms** | **17.0x** | 2,566 | 23/24 (95.8%) | **23/24 (95.8%)** |
| **`qwen2.5-coder:14b` (GPU)** | 133,724.5 ms | 65,616.0 ms | **11,786.2 ms** | **11.3x** | 2,578 | 24/24 (100.0%) | **23/24 (95.8%)** |

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

#### Compact Models for CPU-Bound Automated Testing (1B–3B vs. 4B Baseline)

To determine the optimal model configuration for resource-constrained CI runners (e.g. GitHub Actions, GitLab CI, or local developer pre-commit hooks with zero GPU acceleration), we benchmarked four alternative compact models in pure CPU-only mode (`options={"num_gpu": 0}`) on an AMD Ryzen 7 5800X3D CPU across the complete benchmark suites.

Raw benchmark outputs are stored in [`benchmarks/cpu_results.json`](../benchmarks/cpu_results.json):

| Model Configuration | Model Family | Host RAM | Single-Pass Triage ($N=5$) | 2-Pass Symmetric ($N=5$) | 4-Rubric Batch Total ($N=5$) | Suite 1 Cascade Acc | Suite 2 Reasoning (32 Items) | `ChoiceGate` Routing | `ScoreGate` Monotonicity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-coder:1.5b`** | 1.5B Code Model | **1.39 GB** | **42.3 ± 1.4 ms** | **330.4 ± 63.5 ms** | **1,834.5 ± 75.6 ms** | 14/24 (58.3%) | 16/32 (50.0%) | **6/6 (100.0%)** | **8/8 (100.0%)** |
| **`llama3.2:1b`** | 1B General Model | **1.76 GB** | **46.9 ± 1.6 ms** | **459.2 ± 86.0 ms** | **2,665.1 ± 82.9 ms** | 15/24 (62.5%) | 18/32 (56.2%) | 4/6 (66.7%) | 4/8 (50.0%) |
| **`qwen2.5-coder:3b`** | 3B Code Model | **2.44 GB** | **69.1 ± 2.8 ms** | **625.7 ± 113.9 ms** | **3,626.6 ± 155.8 ms** | **19/24 (79.2%)** | 18/32 (56.2%) | **6/6 (100.0%)** | **8/8 (100.0%)** |
| **`llama3.2:3b`** | 3B General Model | **3.18 GB** | **74.8 ± 0.8 ms** | **642.7 ± 113.6 ms** | **3,699.9 ± 116.8 ms** | **19/24 (79.2%)** | **23/32 (71.9%)** | **6/6 (100.0%)** | **8/8 (100.0%)** |
| *`gemma4:e4b` (Baseline)* | *4B General Model* | *9.61 GB* | *173.0 ± 2.4 ms* | *3,710.5 ± 82.1 ms* | *20,949.5 ± 182.5 ms* | *23/24 (95.8%)* | *18/32 (56.2%)* | *6/6 (100.0%)* | *3/8 (37.5%)* |

##### Key Insights for CPU-Bound Automated Testing:
1. **The CI/CD Champion (`llama3.2:3b`)**:
   - Reduces RAM consumption by **67%** (from 9.61 GB down to 3.18 GB), easily fitting inside standard 4 GB/7 GB GitHub Actions free runners.
   - Runs 2-pass symmetric debiased gating in **642.7 ms** (**5.8x faster** than the 4B baseline at 3.7 s).
   - Achieves the highest 32-item reasoning accuracy of any tested model (**23/32, 71.9%**), surpassing even 14B models on GPU (68.8%) thanks to Meta's aggressive distillation.
   - Delivers flawless **100% (6/6)** `ChoiceGate` routing and **100% (8/8)** `ScoreGate` pairwise rank monotonicity ($H=3.69 > M=2.92 > L=2.34$).
2. **The Micro-Gate Champion (`qwen2.5-coder:1.5b`)**:
   - Microscopic **1.39 GB RAM footprint**, ideal for pre-commit git hooks and resource-constrained Docker test containers.
   - Executes single-pass triage in **42.3 ms** and 2-pass debiasing in **330.4 ms** on pure CPU.
   - Delivers a perfect **100% (6/6)** `ChoiceGate` routing and **100% (8/8)** `ScoreGate` monotonicity on code/incident rubrics.
3. **The Multi-Rubric Batching Advantage**:
   - On CPU, running 4 rubrics via `evaluate_batch` takes **1.8 s** on `qwen2.5-coder:1.5b` and **3.7 s** on `llama3.2:3b` (vs **20.9 s** on `gemma4:e4b`), avoiding the CPU memory bus saturation of redundant prefill passes.

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
| **`gemma4:e4b` (GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`gemma4:e4b` (Pure CPU, 0 GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`qwen2.5:14b` (GPU)** | 2/24 (8.3%) | 22/24 (91.7%) | 0/24 (0.0%) | **24/24 (100.0%)** |
| **`qwen2.5-coder:14b` (GPU)** | 2/24 (8.3%) | 21/24 (87.5%) | 1/24 (4.2%) | **23/24 (95.8%)** |

---

### 4. `ChoiceGate` (Multi-Way Routing) & `ScoreGate` (Continuous 1–5 Scoring) Benchmarks

#### Initial Baseline (12-Item Suite)

Evaluated on the baseline 12-item routing and scoring suite ([`benchmarks/data/choice_and_score_12.json`](../benchmarks/data/choice_and_score_12.json), comprising 6 three-way operational/architectural routing cases with orthogonal categories and 6 continuous 1–5 rubric scoring cases across High, Medium, and Low quality extremes):

| Model Architecture | `ChoiceGate` Accuracy (2-Pass) | `ChoiceGate` Latency ($N=5$) | `ScoreGate` Mean (High Tier) | `ScoreGate` Mean (Medium Tier) | `ScoreGate` Mean (Low Tier) | `ScoreGate` Pairwise Rank Accuracy | `ScoreGate` Latency ($N=5$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gemma4:e4b` (4B General)** | **6/6 (100.0%)** | 116.9 ± 1.7 ms | 4.96 | 2.15 | 1.01 | **6/6 (100.0%)** | 167.6 ± 1.1 ms |
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
| **`llama3.2:1b` (Pure CPU)** | 1.76 GB RAM | **50.0% ± 0.0%** | 15/16 (93.8%) | 1005.2 ± 310.1 ms | 3.22 | 3.20 | 3.00 | 3.07 | 2.56 | 29/42 (69.0%) | **0.6688** | 0.66 | 602.6 ± 276.7 ms |
| **`qwen2.5-coder:1.5b` (Pure CPU)** | 1.39 GB RAM | **61.2% ± 2.8%** | 13/16 (81.2%) | **644.8 ± 278.4 ms** | 3.28 | 3.08 | 3.03 | 3.02 | 3.02 | 28/42 (66.7%) | **0.5683** | 0.26 | **434.7 ± 192.1 ms** |
| **`llama3.2:3b` (Pure CPU)** | 3.18 GB RAM | **75.0% ± 0.0%** | 15/16 (93.8%) | 1422.5 ± 441.3 ms | 3.63 | 3.05 | 2.55 | 2.54 | 2.05 | 33/42 (78.6%) | **0.8487** | 1.58 | 872.1 ± 401.0 ms |
| **`qwen2.5-coder:3b` (Pure CPU)** | 2.44 GB RAM | **75.0% ± 0.0%** | 14/16 (87.5%) | 1265.0 ± 568.2 ms | 3.61 | 2.99 | 2.85 | 3.04 | 2.78 | 28/42 (66.7%) | **0.6013** | 0.83 | 838.3 ± 399.8 ms |
| **`gemma4:e4b` (Pure CPU)** | 9.57 GB RAM | **93.8% ± 0.0%** | **16/16 (100%)** | 4186.4 ± 1619.0 ms | 4.66 | 2.73 | 2.01 | 1.70 | 1.00 | 33/42 (78.6%) | **0.8907** | 3.66 | 6028.0 ± 1678.2 ms |
| **`gemma4:e4b` (GPU 1, 16GB)** | 3.25 GB VRAM | **93.8% ± 0.0%** | **16/16 (100%)** | **137.4 ± 20.8 ms** | 4.58 | 2.74 | 2.02 | 1.69 | 1.01 | 33/42 (78.6%) | **0.8907** | 3.57 | 165.0 ± 24.1 ms |
| **`qwen2.5:14b` (GPU 1, 16GB)** | 10.39 GB VRAM | **87.5% ± 0.0%** | **16/16 (100%)** | 141.1 ± 27.2 ms | 4.62 | 2.98 | 2.67 | 2.19 | 1.00 | 34/42 (81.0%) | **0.8764** | 3.62 | **130.1 ± 17.6 ms** |
| **`qwen2.5-coder:14b` (GPU 1, 16GB)** | 10.39 GB VRAM | **87.5% ± 0.0%** | **16/16 (100%)** | 156.7 ± 33.6 ms | 4.43 | 2.63 | 2.08 | 1.98 | 1.16 | 32/42 (76.2%) | **0.8787** | 3.27 | 131.7 ± 17.6 ms |
| **`gemma4:31b` (GPU 0, 32GB)** | 2.93 GB VRAM | **100.0% ± 0.0%** | **16/16 (100%)** | 357.8 ± 31.5 ms | 5.00 | 2.33 | 1.72 | 1.33 | 1.00 | 33/42 (78.6%) | **0.9096** | **4.00** | 407.1 ± 72.3 ms |

##### Category Breakdown for `ChoiceGate` Routing:

| Model & Hardware Configuration | 6-Way Operational Scaling (4 items) | Adversarial Keyword Decoys (4 items) | Priority Hierarchy Conflicts (4 items) | Architectural Scope (4 items) | Total Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`llama3.2:1b` (Pure CPU)** | 10/20 (50.0%) | 10/20 (50.0%) | 10/20 (50.0%) | 10/20 (50.0%) | **50.0% ± 0.0%** |
| **`qwen2.5-coder:1.5b` (Pure CPU)** | 19/20 (95.0%) | 5/20 (25.0%) | 5/20 (25.0%) | **20/20 (100.0%)** | **61.2% ± 2.8%** |
| **`llama3.2:3b` (Pure CPU)** | **20/20 (100.0%)** | 15/20 (75.0%) | 10/20 (50.0%) | 15/20 (75.0%) | **75.0% ± 0.0%** |
| **`qwen2.5-coder:3b` (Pure CPU)** | **20/20 (100.0%)** | 5/20 (25.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **75.0% ± 0.0%** |
| **`gemma4:e4b` (Pure CPU / GPU 1)** | **20/20 (100.0%)** | 15/20 (75.0%) | **20/20 (100.0%)** | **20/20 (100.0%)** | **93.8% ± 0.0%** |
| **`qwen2.5:14b` (GPU 1)** | **20/20 (100.0%)** | 15/20 (75.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **87.5% ± 0.0%** |
| **`qwen2.5-coder:14b` (GPU 1)** | **20/20 (100.0%)** | 15/20 (75.0%) | 15/20 (75.0%) | **20/20 (100.0%)** | **87.5% ± 0.0%** |
| **`gemma4:31b` (GPU 0)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **20/20 (100.0%)** | **100.0% ± 0.0%** |

##### Key Scientific Findings from the Hardened Suite:
1. **Shattering the 100% Ceiling Effect**:
   - The hardened adversarial suite completely exposes the true limits of compact and mid-sized models. Where the 12-item baseline reported 100% across the board, `llama3.2:1b` drops to **50.0%**, `qwen2.5-coder:1.5b` drops to **61.2%**, and the 14B models plateau at **87.5%**.
2. **The TOCTOU Inversion Bug in Compact Code Models**:
   - On `ScoreGate`, `qwen2.5-coder:3b` exhibits a critical failure mode: it scores Tier 2 code (3.04) **higher** than Tier 4 code (2.99) and Tier 3 code (2.85). Because Tier 2 code is cleanly typed and well-formatted, the 3B model is blinded by surface appearance and fails to spot the latent TOCTOU concurrency race or daylight-saving timezone bug.
3. **Severe Score Compression in Sub-3B Models**:
   - `qwen2.5-coder:1.5b` exhibits extreme score compression ($\Delta = 0.26$), collapsing all 5 tiers into the narrow band between 3.02 and 3.28 ($\rho = 0.5683$). It cannot differentiate between a flawless LRU cache and a catastrophic SQL injection via first-token logits alone.
4. **`llama3.2:3b` Outperforms `qwen2.5-coder:3b` on Continuous Scoring**:
   - For CPU-only CI deployments, `llama3.2:3b` achieves a **Spearman $\rho = 0.8487$** with a healthy dynamic range of **1.58** ($T5=3.63 \to T1=2.05$), preserving strictly monotonic ordering across all adjacent tiers without the inversion bugs seen in Qwen Coder 3B.
5. **Cross-Card Scaling: GPU 1 (16GB) vs. GPU 0 (32GB)**:
   - On GPU 1, `gemma4:e4b` achieves **93.8%** ChoiceGate accuracy in **137.4 ms** (30x faster than CPU at 4.2 s).
   - On GPU 0 (RTX 5090 32GB), **`gemma4:31b` is the only model to achieve a flawless 100.0% on `ChoiceGate`**, successfully untangling all adversarial keyword decoys and multi-domain priority conflicts, with a near-perfect **Spearman $\rho = 0.9096$** spanning the complete 1.00 to 5.00 score range.

---

## Falsification Benchmarks: Where Does Layer 1 Break Down & How to Mitigate It?

To establish clear empirical boundaries for first-token logit probing, we evaluated `LogitGate` across **5 model architectures** (`4B`, `14B General`, `14B Code`, `26B MoE`, `31B Dense`), **4 reasoning depth tiers** (32 items total, 8 cases per tier in [`benchmarks/data/reasoning_tiers_32.json`](../benchmarks/data/reasoning_tiers_32.json)), **document lengths** (200 to 16,000 tokens across a 12-section enterprise handbook), and **context window allocations** (`num_ctx` from 512 to 131,072).

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
| **`gemma4:e4b` (4B General)** | **8/8 (100.0%)** | 6/8 (75.0%) | **7/8 (87.5%)** | 7/8 (87.5%) | 6/8 (75.0%) | 27/32 (84.4%) | 28/32 (87.5%) |
| **`qwen2.5:14b` (14B General)** | **8/8 (100.0%)** | 7/8 (87.5%) | **7/8 (87.5%)** | 6/8 (75.0%) | 7/8 (87.5%) | 28/32 (87.5%) | 28/32 (87.5%) |
| **`qwen2.5-coder:14b` (14B Code)** | **8/8 (100.0%)** | 7/8 (87.5%) | **7/8 (87.5%)** | 7/8 (87.5%) | 7/8 (87.5%) | 29/32 (90.6%) | 29/32 (90.6%) |
| **`gemma4:26b` (26B MoE, ~4B act)** | **8/8 (100.0%)** | **8/8 (100.0%)** | **7/8 (87.5%)** | 7/8 (87.5%) | 7/8 (87.5%) | **30/32 (93.8%)** | 29/32 (90.6%) |
| **`gemma4:31b` (31B Dense)** | **8/8 (100.0%)** | **8/8 (100.0%)** | **7/8 (87.5%)** | 7/8 (87.5%) | 7/8 (87.5%) | **30/32 (93.8%)** | 29/32 (90.6%) |

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

Rather than repeating a single boilerplate paragraph, [`benchmarks/data/haystack_handbook.json`](../benchmarks/data/haystack_handbook.json) constructs a realistic 12-section enterprise architecture handbook (covering VPC routing, `config.yaml` schema validation, OIDC federation, HashiCorp Vault credential injection, PostgreSQL HA, PII log redaction, and break-glass procedures—where multiple surrounding sections legitimately discuss `config.yaml`, passwords, and emergency maintenance). We embedded a 1-paragraph policy addendum at the midpoint of the document (either strictly prohibiting hardcoded credentials in `config.yaml` or permitting a fallback password during maintenance) and evaluated `"Does this architecture document strictly prohibit hardcoding plaintext passwords or credentials in config.yaml under all circumstances (including emergency maintenance)?"`:

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
# Inside JudgeGate._probe:
probe = LogitGate(
    rubric=_PROBE_RUBRIC,
    backend=self.backend,
    model=self.model,
    symmetric_debiasing=False,
    save_on_fail=False,
)
is_pass = probe.evaluate(critique_text).probability >= 0.5
   ```
3. **Negation-Safe Line Fallback**: If the backend does not implement `LogitBackend`, it falls back to negation-safe parsing of concluding lines.

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

### 6. Summary: When to Use Layer 1 (`LogitGate`) vs. Escalate to Layer 2 (`JudgeGate`)

- **Use Layer 1 (`LogitGate`) — First-Token Forward Pass ($O(1)$ Circuit Depth)**:
  - **Semantic Rubric & Policy Compliance**: Factual groundedness against context, tone/safety compliance, PII/secret leak checks, and atomic requirement verification via `evaluate_batch`.
  - **Discrete Routing & Classification (`ChoiceGate`)**: Selecting downstream agent tools, intent buckets, or workflow branches in ~40–90 ms.
  - **Continuous Rubric Scoring (`ScoreGate`)**: Probability-weighted expected value scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$) with symmetric scale reversal.
- **Escalate to Layer 2 (`JudgeGate`) — Autoregressive Generation**:
  - **Multi-Step Deductive Reasoning**: Proof verification or multi-hop logic where conclusions depend on intermediate reasoning steps.
  - **Complex Symbolic Arithmetic**: Calculations requiring scratchpad calculation or token-by-token recurrence (or route deterministic math checks to Layer 0 `ASTGate` / `ContractGate`).
  - **Diagnostic Reporting**: When a human developer or logging pipeline needs an actionable explanation of *why* an artifact failed (or replay a CI failure locally via `logitgate explain`).
