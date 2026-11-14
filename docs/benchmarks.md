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
| **`deepseek-r1:32b`** | RTX 5090 (32 GB) | `0/24 (0.0%)` — `24 BackendError` (empty `top_logprobs`) | **24/24 (100.0%)** | **161.7 ms / 165.5 ms** |
| **`gemma4:26b`** | RTX 5090 (32 GB) | `0/24 (0.0%)` — `24 AMBIGUOUS` (`unobserved_target_tokens` trap) | **24/24 (100.0%)** | **162.9 ms / 164.1 ms** |
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
| **`qwen2.5-coder:3b`** | Uncalibrated (`T=1.0`) | `1.0000` / — | `0.2000, 0.8500` | 8/24 (33.3%) | 0/24 (0.0%) | 24/24 (100.0%) | 790.1 ms (`1.24x`) |
| **`qwen2.5-coder:3b`** | Temp-Only + Heuristic Cutoffs | `0.6135` / — | `0.2000, 0.8500` | 15/24 (62.5%) | **1/24 (4.2%)** | 23/24 (95.8%) | 483.8 ms (`2.02x`) |
| **`qwen2.5-coder:3b`** | **Unified `CalibrationProfile` ($\alpha=0.10$)** | **`0.6135` / `0.8399`** | **`0.1096, 0.9100`** | 10/24 (41.7%) | **0/24 (0.0%)** | **24/24 (100.0%)** | 685.0 ms (`1.43x`) |
| **`llama3.2:3b`** | Uncalibrated (`T=1.0`) | `1.0000` / — | `0.2000, 0.8500` | 16/24 (66.7%) | **3/24 (12.5%!)** | 21/24 (87.5%) | 419.2 ms (`2.33x`) |
| **`llama3.2:3b`** | Temp-Only + Heuristic Cutoffs | `1.8045` / — | `0.2000, 0.8500` | 5/24 (20.8%) | 0/24 (0.0%) | 24/24 (100.0%) | 893.6 ms (`1.09x`) |
| **`llama3.2:3b`** | **Unified `CalibrationProfile` ($\alpha=0.10$)** | **`1.8045` / `1.2441`** | **`0.5167, 0.8281`** | **12/24 (50.0%)** | **1/24 (4.2%)** | **23/24 (95.8%)** | **522.7 ms (`1.87x`)** |
| **`gemma4:e4b`** | Uncalibrated (`T=1.0`) | `1.0000` / — | `0.2000, 0.8500` | 23/24 (95.8%) | 0/24 (0.0%) | 24/24 (100.0%) | 224.8 ms (`4.35x`) |
| **`gemma4:e4b`** | Temp-Only + Heuristic Cutoffs | `4.6404` / — | `0.2000, 0.8500` | 20/24 (83.3%) | 0/24 (0.0%) | 24/24 (100.0%) | 357.2 ms (`2.74x`) |
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
- **`JudgeGate(profile=..., max_tokens=160, think=False)`**: `9/12 (75.0%)` accuracy, mean latency `14,027.6 ms` (`14,491.6 ms` median) — **`6.6x` mean speedup** via `LogitGate` with portable profile reuse.

---

### 8. Summary: When to Use Layer 1 (`LogitGate`) vs. Escalate to Layer 2 (`JudgeGate`)

- **Use Layer 1 (`LogitGate`) — First-Token Forward Pass ($O(1)$ Circuit Depth)**:
  - **Semantic Rubric & Policy Compliance**: Factual groundedness against context, tone/safety compliance, PII/secret leak checks, and atomic requirement verification via `evaluate_batch`.
  - **Discrete Routing & Classification (`ChoiceGate`)**: Selecting downstream agent tools, intent buckets, or workflow branches in ~40–90 ms.
  - **Continuous Rubric Scoring (`ScoreGate`)**: Probability-weighted expected value scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$) with symmetric scale reversal.
- **Escalate to Layer 2 (`JudgeGate`) — Autoregressive Generation**:
  - **Multi-Step Deductive Reasoning**: Proof verification or multi-hop logic where conclusions depend on intermediate reasoning steps.
  - **Complex Symbolic Arithmetic**: Calculations requiring scratchpad calculation or token-by-token recurrence (or route deterministic math checks to Layer 0 `ASTGate` / `ContractGate`).
  - **Diagnostic Reporting**: When a human developer or logging pipeline needs an actionable explanation of *why* an artifact failed (or replay a CI failure locally via `logitgate explain`).
