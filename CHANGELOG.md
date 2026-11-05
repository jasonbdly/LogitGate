# Changelog

All notable changes to **LogitGate** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.3] - 2026-11-05

### Added

#### Mathematical Debiasing & Surface-Form Token Marginalization
- Added `SurfaceTokenMap` in `logitgate.debiasing.permutation` to map canonical binary poles to single-token surface variants across casing and whitespace tokens (`"A"`, `" A"`, `"a"`, etc.).
- Added `marginalize_pole_logprob` for normalizing pole probabilities strictly over the valid binary decision subspace via logsumexp, eliminating vocabulary probability leakage.
- Added `compute_symmetric_debiased_verdict` to execute symmetric permutation in log-odds space, algebraically cancelling positional and token prior biases and outputting calibrated probability, normalized Shannon entropy in bits, unbiased log-odds, and positional bias magnitude.

#### Shared KV-Cache Prefix Broadcasting
- Added `PrefixBroadcastLogitEvaluator` in `logitgate.backends.pytorch_native` for running Query-First Attention Inverted prefixes once at `batch_size=1` and broadcasting cached key-values across permuted label binding suffixes at `batch_size=2` in ~1.02x single-pass latency.

#### Latent-Recurrent Early Halting
- Added `execute_adaptive_recurrent_loops` and `RecurrentLoopTrace` in `logitgate.backends.dory_recurrent` for early exit in looped transformers when verdict entropy drops below threshold and consecutive loop log-probs stabilize.

#### Split Conformal Cascade Calibration CLI
- Added `calibrate_cascade_thresholds` and `ConformalCascadeThresholds` in `logitgate.calibration.conformal` for calculating distribution-free selective classification bounds ($\tau_{\text{reject}}, \tau_{\text{accept}}$) with finite-sample error rate guarantees bounded by $\le \alpha$.
- Enhanced `logitgate calibrate` CLI to accept `--dataset <path>` and `--alpha <float>`, with native parsing for Markdown tables, JSON, and JSONL datasets.
- Added `EPISTEMIC_REASONING_TEST_CASES.md` reference dataset containing calibrated holdout test cases.

#### PCIe VRAM Spillover Guard
- Added `PCIeSpilloverGuard` and `VRAMWatermarkStatus` in `logitgate.hardware.memory_guard` to monitor physical CUDA VRAM headroom and prevent silent CUDA Unified Memory paging over PCIe into host RAM. Integrated into native PyTorch and recurrent backend passes.

#### ASTGate Failure Codes & Diagnostic Telemetry
- Added standardized `ASTFailureCode` enumeration across syntax, schema, import, pattern, section, DAG cycle, and length violations.
- Recorded `positional_bias_magnitude`, `log_odds_pass`, `shannon_entropy`, `loops_executed`, and `failure_code` on `LayerTrace` and `VerdictSnapshot`.
- Rendered debiasing telemetry, recurrent loop counts, and failure codes directly in `logitgate explain`.

---

## [0.2.2] - 2026-11-04

### Documentation

#### PyPI & Cross-Platform Responsive Unicode Diagrams
- Replaced unsupported raw Mermaid diagram in `README.md` with a compact, mobile-friendly Unicode box-drawing diagram. Ensures clean, instant rendering across PyPI (`pypi.org`), mobile web, terminal CLI documentation viewers, and IDE README previews.

---

## [0.2.1] - 2026-11-04

### Fixed

#### Cascade Pipeline Routing & Intermediate Failure Handling
- Restored multi-gate sequential pipeline execution in `Cascade.evaluate` and `Cascade.aevaluate`. When a `LogitGate` passes, it only short-circuits its immediately paired downstream `JudgeGate` rather than terminating the entire cascade prematurely, allowing subsequent independent validation gates to execute.
- Resolved intermediate non-logit gate failure short-circuiting: failures in intermediate non-logit gates (e.g. `ASTGate`, `ScoreGate`) now correctly abort downstream gate evaluation and return an immediate failure verdict instead of continuing down the chain.
- Corrected `skip_indices` calculation in `Cascade` so downstream judge gates paired with a passed or decisively failed logit gate are skipped cleanly.
- Cascade configuration now raises `ConfigurationError` when initialized with empty gates.

#### JudgeGate Async & Sync Parity
- Aligned async verdict parsing (`_aparse_verdict`) to match sync parsing (`_parse_verdict`) by extracting the final occurrence of `<verdict>...</verdict>` (Reason-Then-Verdict pattern).
- Aligned candidate lines extraction and dogfood probe gate precedence across synchronous and asynchronous judge paths.
- Preserved explicit custom `JudgeGate` rubrics when chained inside a `Cascade`, preventing upstream `LogitGate` rubrics from silently overriding explicit downstream rubrics.
- Enforced `ConfigurationError` validation for invalid trigger values and blank/whitespace rubrics.

#### Deterministic ASTGate Hardening
- Hardened `_detect_dag_cycle` against malformed task graph nodes, requiring non-empty dictionary entries with valid `id` or `task_id` attributes.
- Protected regex compilation in `required_sections` and `forbidden_patterns` with explicit syntax error handling, preventing crashes from unescaped punctuation or regex special characters (e.g., `(v2.0)`).

#### Calibrated Logit & Multi-Choice Debiasing
- Applied temperature scaling across all single-pass triage evaluation paths (`symmetric_debiasing=False`) in `LogitGate.evaluate`, `aevaluate`, `evaluate_batch`, and `aevaluate_batch`.
- Added single-character key validation to `ChoiceGate` initialization (`ConfigurationError`), preventing multi-character label mismatches against subword tokenizers.
- Updated configuration validation across `LogitGate`, `ChoiceGate`, and `ScoreGate` to consistently raise `ConfigurationError` (subclass of `ValueError`).
- Fixed reverse-substring matching in `get_model_temperature` to prevent single-character model aliases (e.g., `"a"`) from unintentionally matching unrelated model identifiers.

#### Backend Kwarg Sanitization
- Filtered out internal LogitGate evaluation kwargs (`save_on_fail`, `snapshot_dir`, `test_name`, `rubric_override`) before dispatching payloads to remote HTTP endpoints in `OpenAICompatBackend` and `OllamaBackend`, preventing HTTP 400 Bad Request responses.

---

## [0.2.0] - 2026-11-04

### Added

#### Hardened 32-Item Adversarial Multi-Action Benchmark Suite
- 32-Item Adversarial Benchmark Suite ([`benchmarks/data/adversarial_choice_and_score_32.json`](benchmarks/data/adversarial_choice_and_score_32.json) & [`benchmarks/run_adversarial_choice_score.py`](benchmarks/run_adversarial_choice_score.py)):
  - 16 `ChoiceGate` adversarial items across 4 stress vectors: 6-way scaling, adversarial keyword decoys, strict priority hierarchy conflicts, and architectural scope boundaries.
  - 16 `ScoreGate` items spanning 5 discrete code tiers (Tier 5 Flawless $\to$ Tier 4 Sub-optimal Complexity $\to$ Tier 3 Code Smells $\to$ Tier 2 Latent Concurrency/TOCTOU Defects $\to$ Tier 1 Catastrophic Flaws).
  - Multi-sample ($N=5$) with untimed warm-up pass evaluated across full 9-configuration matrix (CPU, RTX 4070, RTX 5090), measuring exact pairwise monotonicity, Spearman rank correlation $\rho$, and score dynamic range $\Delta(S_5 - S_1)$.
  - Published empirical dataset and verified results in [`benchmarks/adversarial_choice_score_results.json`](benchmarks/adversarial_choice_score_results.json).

#### In-Process `llama.cpp` Format, Quantization & Thread Scaling Benchmark Suite
- Master CPU Format & Thread Scaling Runner ([`benchmarks/run_cpu_format_runtime_benchmarks.py`](benchmarks/run_cpu_format_runtime_benchmarks.py) & [`benchmarks/cpu_format_runtime_results.json`](benchmarks/cpu_format_runtime_results.json)):
  - Native in-process `llama.cpp` (`LlamaCppBackend`) benchmarked against HTTP daemon architecture on AMD Ryzen 9 9955HX3D (16-core / 32-thread with 128MB 3D V-Cache).
  - Evaluates `Q8_0` (1.26 GB) vs. `Q4_K_M` (940 MB) vs. `Q4_K_M` (1.93 GB) across 1B, 1.5B, and 3B parameter scales.
  - Full CPU thread scaling sweep ($n=1, 2, 4, 8, 16$) demonstrating memory bus saturation and establishing the 2-thread optimum (112ms single forward pass) for single-token logit extraction.

#### Latent Recurrent Network Support & Elastic Layer-Looping
- `RecurrentPyTorchBackend`: Native backend supporting elastic recurrent depth (`num_loops`) via `recurrent://` URI scheme.
- Standalone PyTorch implementation of `DoryForCausalLM` (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) built on normalized-GPT (`nGPT`), `justnorm`, and `slerp_alpha` residual scaling, providing native execution on Blackwell (`sm_120`, RTX 5090) and Ada Lovelace (`sm_89`, RTX 4070) with zero external C++ build dependencies.
- Vocab Head Memory Optimization: projects only final sequence position `h[:, -1:, :]` through the 131,072 vocabulary head, eliminating 8.4 GB allocation overhead and enabling full 16,000-token needle-in-haystack evaluations with zero OOM.
- Master Multi-GPU & Recurrent Depth Benchmark Suite ([`benchmarks/run_dory_benchmarks.py`](benchmarks/run_dory_benchmarks.py) & [`benchmarks/dory_results.json`](benchmarks/dory_results.json)) evaluating 24-item cascade, 32-item reasoning tiers, multi-action gates, and long-context scaling across $K=1, 2, 4$.

#### Compact Models & CPU-Bound CI/CD Evaluation Suite
- Master CPU Benchmark Runner ([`benchmarks/run_cpu_benchmarks.py`](benchmarks/run_cpu_benchmarks.py) & [`benchmarks/cpu_results.json`](benchmarks/cpu_results.json)) evaluating `llama3.2:1b`, `llama3.2:3b`, `qwen2.5-coder:1.5b`, and `qwen2.5-coder:3b` in pure CPU mode (`options={"num_gpu": 0}`).
- Verified 5.8x speedup on 2-pass debiased gating and 67% RAM reduction on `llama3.2:3b` (3.18 GB RAM, 642 ms 2-pass latency, 71.9% reasoning accuracy) compared to dense 4B models on CPU.
- Verified 42.3 ms single-pass gating on `qwen2.5-coder:1.5b` with 1.39 GB RAM footprint and 100% routing accuracy.

---

## [0.1.0] - 2026-11-02

### Added

#### 3-Layer Cascaded Router (`Cascade`)
- Seamless operator chaining (`ASTGate >> LogitGate >> JudgeGate` or pipe `|`).
- Fast-path short-circuiting: high-confidence Layer 1 passes bypass generative Layer 2 entirely.
- Decisive failure short-circuiting when `trigger='on_ambiguous_only'`.
- Monotonic failure enforcement: when Layer 1 decisively fails (`status == FAILED`) and escalates to Layer 2 for diagnostic explanation, `passed=False` is preserved even if the downstream generative critique phrasing is lenient.

#### Layer 0 Deterministic AST & Contract Gate (`ASTGate` / `ContractGate`)
- Sub-millisecond (~0.04ms) deterministic CPU checks without LLM calls.
- Python AST syntax validation (`syntax=True`) and forbidden import detection (`forbidden_imports`) via AST walker with regex fallback.
- General-domain JSON Schema & Pydantic `BaseModel` validation (`json_schema`), character length bounds (`min_length`, `max_length`), forbidden regex patterns (`forbidden_patterns`), and required section verification (`required_sections`).
- Acyclic directed graph (DAG) cycle detection and missing dependency analysis (`acyclic_dag=True`) for agentic task graphs.
- Custom callable validator support (`validators`) with informative diagnostic strings.

#### Layer 1 First-Token Debiased Logit Gate (`LogitGate`)
- 40–120ms GPU latency (~170ms single-pass on pure CPU) using first-token log-probabilities (`max_tokens=1`) without multi-token Chain-of-Thought generation.
- Built-in threshold presets: `preset="strict"` (`0.90 / 0.25`, symmetric 2-pass), `preset="balanced"` (`0.85 / 0.20`, symmetric 2-pass), and `preset="fast_triage"` (`0.80 / 0.20`, single-pass for local CPU iteration & high-throughput triage).
- Symmetric log-odds debiasing ($\bar{L} = \frac{L_1 + L_2}{2T}$) cancelling pretraining token priors and positional bias.
- Prefix-preserving KV-cache prompt compilation (`compile_binary_prompt`), placing rubric and candidate first to share the prompt prefix across forward and reverse passes.
- Multi-rubric batch evaluation (`evaluate_batch`, `aevaluate_batch`) sharing a byte-identical static prefix across all rubrics with KV-cache stampede prevention.
- Dynamic missing-token bounded floor calculation to prevent arbitrary cliff distortion in log-odds space.
- Post-hoc temperature calibration (`fit_calibration`), built-in `KNOWN_MODEL_TEMPERATURES` registry, and automatic calibration caching (`logit_temperature="auto"`, `auto_calibrate_backend`) in `~/.cache/logitgate/calibration.json`.

#### Discrete & Continuous Gating (`ChoiceGate` & `ScoreGate`)
- `ChoiceGate`: K-ary discrete option selection with forward/reverse ordering debiasing and temperature scaling.
- `ScoreGate`: Probability-weighted expected continuous scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$) with optional inverted scale debiasing.

#### Layer 2 Generative Reasoning Judge (`JudgeGate`)
- Escalated diagnostic critique and failure explanation on ambiguous or failing artifacts.
- Configurable trigger policies: `on_ambiguous_or_fail`, `on_ambiguous_only`, and `always`.
- Smart reasoning trace handling (`think=True` by default when uncapped or `max_tokens >= 256`; `think=False` verdict-first mode when tightly capped at `max_tokens < 256`).
- Self-healing 1-token `LogitGate` fallback probe inside `JudgeGate._parse_verdict` when generative output omits explicit `[VERDICT: PASS/FAIL]` tags.

#### Deterministic Failure Replay (`VerdictSnapshot` & `logitgate explain`)
- Captures frozen candidate outputs, rubrics, and Layer 1 telemetry on CI test failures (`VerdictSnapshot.from_verdict`, `LOGITGATE_SAVE_FAILURES=1`, `@check(save_snapshot=True)`).
- Local one-command replay and full System-2 diagnostic explanation (`logitgate explain`) without re-running non-deterministic upstream LLM applications.

#### Inference Engine Backends
- `OllamaBackend`: Native Ollama `/api/chat` client with persistent HTTP connection pooling, hardware `options` (`num_gpu`, `num_ctx`, `num_thread`), and True `<think>` Continuation when `max_tokens` truncates inside a thinking trace.
- `OpenAICompatBackend`: Compatible with vLLM, SGLang, Together AI, Groq, OpenRouter, and OpenAI.
- `LlamaCppBackend`: In-process `llama_cpp.Llama` integration and `llama-server` HTTP API client.
- `StubBackend`: Fully deterministic, zero-network offline stub backend for sub-second testing and CI.

#### Global Configuration, Framework Integrations & CLI
- Global configuration via `logitgate.configure(...)` and `LOGITGATE_*` environment variables (`LOGITGATE_BACKEND`, `LOGITGATE_MODEL`, `LOGITGATE_PASS_THRESHOLD`, `LOGITGATE_FAIL_THRESHOLD`, `LOGITGATE_TEMPERATURE`, `LOGITGATE_NUM_CTX`, `LOGITGATE_SAVE_FAILURES`).
- Pytest: `@check(rubric=..., threshold=..., save_snapshot=...)` decorator and `pytest11` plugin entrypoint.
- DeepEval: `LogitGateSystemOneModel` and `LogitGateMetric`.
- LangGraph: `as_conditional_edge(gate, pass_node, fail_node)` routing helper.
- CLI (`logitgate`): `logitgate eval`, `logitgate calibrate`, and `logitgate explain` subcommands.
- PEP 561 typed (`py.typed`), strict `mypy` and `ruff` compliance, and `CITATION.cff` academic attribution.
