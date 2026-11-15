# Changelog

All notable changes to **LogitGate** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.12.0] - 2026-11-15

### Added
- **Zero-Dependency OpenTelemetry Span Bridge (`logitgate.telemetry.otel`, `logitgate[otel]`):** Added `OtelSpanSink` and `OtelAttributeValue` (re-exported at `logitgate` and `logitgate.telemetry`) to bridge structured `TelemetryRecord` evaluations into OpenTelemetry spans (`<span_prefix>.<gate_name>`) and per-layer `"logitgate.layer"` span events (`record_layer_events=True`). Lazily resolves `opentelemetry.trace.get_tracer(...)` when `opentelemetry-api` is installed, accepts any injected or duck-typed `Tracer` (`start_as_current_span` or `start_span`), degrades to a zero-overhead no-op when `opentelemetry-api` is absent (`require_otel=False`), enforces `privacy="omit_candidate"` by default, omits `None` and non-finite (`NaN`/`+-inf`) values per the OpenTelemetry attribute spec, and supports `error_on_fail=True`.
- **Module Execution Entry Point & Async Sink Lifecycle Parity:** Added `src/logitgate/__main__.py` (`python -m logitgate`) and `aclose()` / `async with` context manager methods on `TelemetrySink` and `CompositeTelemetrySink`.
- **Property-Based Fuzzing & Public API Surface Suites:** Added `hypothesis`-backed property fuzzing (`tests/test_v0120_property_fuzzing.py`) across sigmoid/softmax/logprob boundaries, symmetric debiasing bias cancellation, conformal order statistics, meta-calibrator monotonicity, PSI/KS drift invariants, and linear-time PII/AST scanners, alongside public API export and schema contract tests (`tests/test_v0120_api_contract.py`).

### Changed
- **PyTorch Inference Mode & Post-Training VRAM Hygiene:** Switched `HuggingFaceLogitBackend` and `RecurrentPyTorchBackend` forward passes from `torch.no_grad()` to `torch.inference_mode()`, added `HuggingFaceLogitBackend.close()` / `aclose()` and owned GGUF teardown in `LlamaCppBackend.close()`, and released optimizer buffers after `train_distillation_lora()`.

### Fixed
- **Statistical & Calibration Edge Cases:** Fixed two-sided conformal `min_alpha_guaranteed = ceil(2 / (m + 1))` in `audit_benchmark_dataset()`, enabled boolean `True`/`False` labels in `CalibrationProfile.from_log_odds_samples()`, removed premature `d_stat >= 1.0 -> p = 0.0` clamping in `compute_ks_two_sample()` so small windows preserve exact Stephens survival p-values, guarded `has_finite_logprobs(None)` and `NaN` entries in `logsumexp()`, guarded `NaN`/`inf` search bounds in `_fit_multiclass_temperature()`, and surfaced `[adaptive_log_odds_margin_unmet]` in `LogitGate` verdict reasons when 4-decimal rounded probabilities tie an adaptive conformal boundary.
- **Security, Symlink, & Deserialization Hardening:** Added dynamic import detection (`__import__`, `importlib.import_module`) to `ASTGate(forbidden_imports=...)`, blocked quoted candidate `[VERDICT: PASS]` mentions from spoofing `JudgeGate` verdicts, extended `compile_cot_binary_prompt()` inert-data instructions to `<notes_{nonce}>`, scrubbed generative `JudgeGate` critiques from `TelemetryRecord.reason` under `privacy="omit_candidate"`, hardened `VerdictSnapshot.load()`, `load_calibration_dataset()`, and `CalibrationProfile.load()` with `O_NOFOLLOW` bounded regular-file reads, caught `RecursionError` in `read_json_object_nofollow()`, closed the concurrent `aclose()` window in `_HTTPClientMixin`, synchronized `get_default_telemetry_sink()` / `get_default_verdict_cache()` under `_config_lock`, and sanitized CLI stderr paths.

## [0.11.0] - 2026-11-15

### Added
- **Weight-Free BYO Contrastive Anchor Bank & `<1 KB` Logit Meta-Calibrator (`logitgate.distill`):** Added `AnchorExemplar`, `ContrastiveAnchorPair`, `DistilledAnchorBank`, `resolve_anchor_bank()`, `FEATURE_NAMES`, `LogitFeatureVector`, `extract_logit_features()`, `LogitMetaCalibrator`, `fit_meta_calibrator()`, `afit_meta_calibrator()`, `resolve_meta_calibrator()`, `DistillationBundle`, `compile_distillation_bundle()`, `acompile_distillation_bundle()`, and `resolve_distillation_bundle()` (re-exported at `logitgate` and `logitgate.distill`). `DistilledAnchorBank` discovers natural counterfactual `(PASS, FAIL)` minimal pairs per rubric via lexical overlap, injects slot-neutral `[CALIBRATION ANCHORS]` inside the shared static KV-cache prefix (`compile_binary_prompt` / `compile_batch_binary_prompt`), and gates injection via Szymkiewicz–Simpson rubric token overlap (`min_rubric_overlap=0.25`, `exclude_exact=True`) so out-of-domain rubrics abstain cleanly. `LogitMetaCalibrator` fits L2-regularized Newton–Raphson logistic regression with an unpenalized intercept and monotonicity floor (`w_sym >= 0.05`) over 5 first-token features plus stratified 5-fold out-of-fold split-conformal threshold calibration (including 0-LLM-call fitting from `DistillationRecord` logs and automatic `unanchored_meta_calibrator` OOD fallback).
- **Architecture-Agnostic Open-Weights PEFT LoRA Compiler (`logitgate[distill]`):** Added `LoRADistillConfig`, `SymmetricLoRAExample`, `LoRADistillManifest`, `HuggingFaceLogitBackend`, `prepare_symmetric_lora_examples()`, `emit_deployment_manifests()`, and `train_distillation_lora()` using PEFT `target_modules="all-linear"` with loss masked strictly onto the single first decision token across `2x` symmetric (`A/B` + `B/A`) examples across CausalLM families (`Llama`, `Qwen2`, `Gemma2`, `Mistral`, `Phi`), emitting `Modelfile.ollama`, `serve_vllm.sh`, and `serve_llamacpp.sh`, plus `LlamaCppBackend(lora_path=..., lora_scale=...)` support.
- **Gate, Adaptive Conformal, Cache, Benchmark, & CLI Distillation Wiring:** Wired `anchor_bank`, `meta_calibrator`, and `distillation_bundle` across `LogitGate`, `MultiRubricGate`, `AdaptiveConformalController`, `VerdictCache` fingerprinting (`gate_cache_fingerprint` and `_backend_fingerprint`), `run_benchmark()`, `logitgate distill compile`, `logitgate distill train-lora`, and `--bundle` on `logitgate eval`, `logitgate benchmark`, and `logitgate stream`.

## [0.10.0] - 2026-11-15

### Added
- **Online & Offline Telemetry Drift Detection (`logitgate.telemetry.drift`):** Added `DriftSeverity`, `DriftRecommendedAction`, `DriftReport`, Cochran/Jeffreys finite-sample `compute_psi()`, SciPy-free exact two-pointer Stephens-adjusted `compute_ks_two_sample()`, multi-signal `detect_telemetry_drift()` (combining symmetric log-odds PSI/KS, escalation-rate delta, positional-bias delta, and decisive Layer-1 vs. Layer-2 teacher disagreement rate), and thread-safe sliding-window `RollingDriftMonitor(TelemetrySink)` (re-exported at `logitgate` and `logitgate.telemetry`) with `privacy="omit_candidate"`, automatic initial baseline capture or fixed reference windows, `on_drift` alerting, and cooldown-gated `AdaptiveConformalController.apply_drift_boost()` coupling.
- **Gibbs–Candès Adaptive Conformal Control (`logitgate.calibration.adaptive`):** Added `AdaptiveConformalController`, `AdaptiveStepRecord`, `AdaptiveConformalSummary`, and `AdaptiveFeedbackMode` (`"explicit"` and `"cascade_audit"`, re-exported at `logitgate` and `logitgate.calibration`) for distribution-free online selective risk control under arbitrary covariate and adversarial shifts. Adjusts symmetric log-odds abstention cutoffs (`theta_{t+1} = max(min_theta, theta_t + gamma * (err_t - alpha))`) with a protective Split-A conformal floor (`min_theta=0.0`), linear saturation abstention (`abstain_theta=8.0` -> `force_ambiguous=True`), `.to_profile()`, and dynamic `VerdictCache` threshold fingerprinting.
- **Zero-Human-Label Cascade Shadow Auditing & Drift CLI:** Wired `adaptive_controller` across `LogitGate`, `MultiRubricGate`, and `Cascade` (autonomous `feedback_mode="cascade_audit"` shadow-auditing of fast-path Layer-1 decisions via Tier-2 `JudgeGate` with Horvitz–Thompson inverse-propensity error weighting `1/p` and teacher verdict override), alongside `logitgate telemetry drift <input_path> --reference <ref_path> [--psi-moderate] [--psi-severe] [--ks-alpha] [--fail-on-drift] [--json]`.

### Fixed
- **Privacy Mode Documentation Sync:** Synchronized `PrivacyMode` literal names (`full`, `redact_pii`, `hash_candidate`, `metrics_only`) in `docs/architecture.md`.

## [0.9.0] - 2026-11-15

### Added
- **Mid-Stream Token Guardrails (`logitgate.streaming`):** Added `StreamGuard`, `GuardedStream`, `AsyncGuardedStream`, `StreamGuardSummary`, `StreamCheckRecord`, `StreamAbortedError`, `guard_stream()`, `aguard_stream()`, and `BaseGate.guard_stream()` / `BaseGate.aguard_stream()` (re-exported at `logitgate`) for synchronous and asynchronous verification of live LLM token streams (`str`, OpenAI `ChatCompletionChunk`, Ollama dicts, or custom `chunk_extractor`). Supports clause-aligned character/chunk strides (`min_chars`, `stride_chars`, `stride_chunks`, `clause_boundary`, `max_wait_chars`), consecutive-failure hysteresis (`consecutive_failures`), `abort_on_ambiguous` / `abort_prob_threshold`, `"optimistic"` vs. zero-leakage `"buffered"` chunk delivery (holding pending chunks across `AMBIGUOUS` checkpoints until verified `PASSED`), immediate upstream generator cancellation via `.close()` / `await .aclose()`, exact-prefix `final_check` deduplication, `max_buffer_chars` overflow protection, and single-record stream telemetry emission.
- **Deterministic Content-Addressed Verdict Caching (`logitgate.cache`):** Added `VerdictCache`, `CacheStats`, `CacheWarning`, `gate_cache_fingerprint()`, `compute_verdict_cache_key()`, `resolve_verdict_cache()`, `lookup_cached_verdict()`, and `store_cached_verdict()` (re-exported at `logitgate`). Provides thread-safe bounded `OrderedDict` LRU caching plus optional `0o600` WAL-mode SQLite persistence (`O_NOFOLLOW` symlink rejection, `256 MiB` existing-file cap, `max_disk_entries` pruning, and optional `ttl_seconds`), `cache_ambiguous=False` abstention safety, secret-free canonical SHA-256 fingerprinting across all gates/profiles/backends, and `_normalized_out` artifact preservation across RAM and SQLite reloads.
- **End-to-End Cache, Benchmark, & CLI Integration:** Wired `cache: VerdictCache | str | Path | bool | None = None` across `ASTGate`, `LogitGate`, `MultiRubricGate`, `ChoiceGate`, `ScoreGate`, `RankGate`, `BestOfNGate`, `JudgeGate`, `Cascade` (inheriting child gate caches across `>>` / `|` chaining while preventing duplicate child writes), `logitgate.configure(..., cache=...)`, `LOGITGATE_CACHE_PATH`, and `run_benchmark(..., cache=...)`, plus `logitgate stream` and `--cache` on `logitgate eval` and `logitgate benchmark`.

### Changed
- **Offline Replay & Benchmark Warmup Cache Isolation:** `VerdictSnapshot.replay()` / `areplay()` / `explain()` / `aexplain()` and `run_benchmark()` warmup passes pass `_bypass_cache=True` automatically so live diagnostic replays and benchmark warmups never read or pollute `VerdictCache`.

## [0.8.1] - 2026-11-14

### Added
- **Benchmark Confounder & Cross-Split Leakage Auditor (`logitgate.calibration.audit`):** Added `DatasetAuditReport`, `SplitLeakageReport`, `audit_benchmark_dataset()`, and `check_split_leakage()` (re-exported at `logitgate` and `logitgate.calibration`) to audit JSON/JSONL/Markdown evaluation datasets in `0` LLM calls for Pass-vs-Fail length confounders (exact tie-aware `length_auroc` and per-domain char/word relative differences), rubric-blind Laplace-smoothed unigram leave-one-out shortcuts (`lexical_loo_accuracy`), split-conformal sample-power floors (`holds_at_alpha_010` requiring `>= 19`/class; `holds_at_alpha_005` requiring `>= 39`/class), duplicate candidates, and cross-split leakage via exact candidate/rubric hashing plus an inverted word-3-gram Jaccard shingle index (`O_NOFOLLOW` symlink-hardened, 32 MiB bounded).
- **Deconfounded `2x2` Counterfactual Benchmark & Calibration Datasets:** Added `benchmarks/data/cal_distill_80.json` (`80` items across `20` `2x2` quads in 4 in-domain families), strictly domain-disjoint `benchmarks/data/ood_generalization_96.json` (`96` items across `24` `2x2` quads in 6 held-out reasoning families with `0` shared domains and `max_jaccard <= 0.0147`), `benchmarks/data/streaming_trajectories_32.json` (`32` multi-chunk streaming trajectories), and `benchmarks/data/drift_regimes_120.json` (`120` unique ordered items across 4 non-stationary drift regimes), alongside typed loaders in `benchmarks/dataset.py`.
- **Zero-LLM-Call Dataset Audit CLI Flags (`logitgate benchmark`):** Added `--audit-only`, `--reference-dataset`, and `--strict-audit` flags to `logitgate benchmark` (supporting both human-readable reports and `--json` CI output).

### Changed
- **Upgraded Conformal Sample Power in `EPISTEMIC_REASONING_TEST_CASES.md`:** Expanded `EPISTEMIC_REASONING_TEST_CASES.md` from `20` (`10`/`10`) to `80` (`40`/`40`) length-balanced counterfactual cases (`char_length_rel_diff = 4.1%`, `length_auroc = 0.5366`) so default `--alpha 0.05` conformal calibration satisfies `min_samples_per_class(0.05) = 39` (`guarantee_holds = True`) out of the box.

### Fixed
- **Benchmark Documentation Link:** Fixed `UNIVERSAL_CALIBRATION_DATASET` relative path link in `docs/benchmarks.md` (`../src/logitgate/debiasing/permutation.py`).

## [0.8.0] - 2026-11-14

### Added
- **Production Verdict Telemetry & Privacy Sinks (`logitgate.telemetry`):** Added `TelemetryRecord`, `TelemetrySummary`, `PrivacyMode` (`full`, `redact_pii`, `hash_candidate`, `metrics_only`), linear-time `redact_text()` (emails, `Bearer` headers, `sk-`/`ghp_`/`glpat-`/`AKIA` secrets, US SSNs, and Luhn-validated payment cards) and `hash_candidate()`, `TelemetrySink`, symlink-hardened thread-safe `JsonlTelemetrySink` with size-based file rotation (`max_bytes`, `backup_count`) and configurable `ErrorPolicy` (`warn`, `raise`, `ignore`), `MemoryTelemetrySink`, `CallbackTelemetrySink`, `CompositeTelemetrySink`, `load_telemetry_records()`, `summarize_telemetry()`, and `TelemetryWarning`.
- **Self-Distilling Sinks, Zero-Call Offline Recalibration, & Dataset Export:** Added `DistillationRecord`, `DiscrepancyType` (`false_accept_caught`, `false_reject_overturned`, `polarity_flip`, `ambiguous_resolved`), `DistillationSink` (with bounded SHA-256 `(rubric, candidate)` deduplication), `extract_distillation_record()`, `load_distillation_records()` (accepting both distillation JSONL and raw telemetry JSONL), zero-LLM-call `recalibrate_from_telemetry()`, and `export_distillation_dataset()` (`calibration_jsonl`, `2x` symmetric `A/B`-swapped `sft_jsonl`, and `dpo_jsonl`).
- **End-to-End Gate, Config, Cascade, & CLI Telemetry Integration:** Added `telemetry_sink` across `ASTGate`, `LogitGate`, `MultiRubricGate`, `ChoiceGate`, `ScoreGate`, `RankGate`, `BestOfNGate`, `JudgeGate`, `Cascade`, and `logitgate.configure(..., telemetry_sink=...)` (plus `LOGITGATE_TELEMETRY_PATH` environment configuration and per-call `telemetry_tags`), along with `--telemetry-out` on `logitgate eval` and `logitgate rank`, and `logitgate telemetry summary|export|recalibrate` CLI subcommands.

### Changed
- **Single-Record Composite Deduplication & Offline Replay Isolation:** `Cascade`, `BestOfNGate`, and `JudgeGate` suppress child probe/stage emissions (`_suppress_telemetry=True`) while `Cascade` inherits/multiplexes inner gate sinks across `>>` / `|` chaining and records uncontaminated `teacher_passed` prior to `locked_upstream_fail` or `fuse_logits`. `VerdictSnapshot.replay()`/`areplay()`/`explain()`/`aexplain()` and `BenchmarkRunner` warmups automatically suppress telemetry emission.

## [0.7.0] - 2026-11-14

### Added
- **Zero-Token Preference Ranking (`RankGate` & `BestOfNResult`):** Added `logitgate.gates.rank_gate.RankGate` (re-exported at `logitgate` and `logitgate.gates`) supporting `pairwise`, `listwise` (`derangement`, `cyclic`, `reverse`), `round_robin` (Hodge / Bradley-Terry centered log-worths with `cycle_residual` inconsistency metric), `tournament` (single-elimination bracket with odd-N byes and tree log-worths), and `auto` modes, with `min_confidence`, `min_margin`, `max_positional_bias`, `warm_prefix`, `profile` integration, and sync/async parity (`evaluate`/`aevaluate` and `rank`/`arank`).
- **Contract-Filtered Best-of-N Selection (`BestOfNGate`):** Added `BestOfNGate` (`pre_gate -> RankGate -> post_gate`) for zero-LLM short-circuiting on 0/1 survivors, zero-filled rejected candidate probabilities, all-bad pool rejection via `post_gate`, `_normalized_out` winning-candidate forwarding across `Cascade` pipelines (`BestOfNGate >> LogitGate`), and single-snapshot failure persistence.
- **Prompt Compilers, Stub Debiasing, Snapshot Replay, & CLI:** Added `compile_pairwise_prompt()` and `compile_listwise_rank_prompt()` with randomized nonce boundaries and `query_first` KV-prefix stability in `logitgate.debiasing.inversion`; extended `StubBackend` with `Preferred:` awareness and automatic Pass-2 derangement token shifting; added `RankGate` and `BestOfNGate` reconstruction (with owned child backend cleanup) to `VerdictSnapshot`; and added the `logitgate rank` CLI subcommand (`--rubric`, `--candidate`, `--file`, `--candidates-json`, `--mode`, `--ast-syntax`, `--normalize`, `--verify-winner`, `--profile`, `--json`).

## [0.6.0] - 2026-11-14

### Added
- **Unified Conformal Calibration Profiles (`CalibrationProfile`):** Added versioned (`schema_version=1`) Pydantic calibration artifacts in `logitgate.calibration.profile` (re-exported at `logitgate`) with `from_calibration_result()`, `from_log_odds_samples()`, `with_choice_thresholds()`, `with_score_thresholds()`, symlink-safe atomic `.save()`, and size-bounded `.load()`, alongside strict validation raising `CalibrationError` on boolean/non-finite/out-of-range inputs and legacy field aliases (`optimal_temperature`, `tau_accept`, `tau_reject`).
- **Turnkey Calibration & Selective Conformal Helpers:** Added `fit_calibration_profile()`, `afit_calibration_profile()`, `resolve_calibration_profile()`, `ChoiceConformalThresholds` / `calibrate_choice_thresholds()`, and `ScoreConformalThresholds` / `calibrate_score_thresholds()` to fit temperatures and split-conformal abstention thresholds in a single pass.
- **First-Class `profile` Integration Across Gates, Benchmarks, & CLI:** Added `profile: CalibrationProfile | str | Path | None = None` across `LogitGate`, `MultiRubricGate`, `ChoiceGate`, `ScoreGate`, `JudgeGate`, and `run_benchmark()`, plus `logitgate calibrate --output`, `logitgate eval --profile`, and `logitgate benchmark --profile`, bypassing lazy calibration probes (`_needs_lazy_calibration = False`) when a profile provides calibrated state and emitting `CalibrationWarning` on model mismatch.

### Changed
- **Precedence & `ScoreGate` Default Threshold Resolution:** Enforced explicit precedence (`explicit kwarg > profile > preset > defaults`) across gates, changed `ScoreGate.threshold` default to `None` (resolving to `profile.score_threshold` when provided or `3.5` otherwise), and introduced `CalibrationProfile.effective_multiclass_temperature` with a `0.65` floor for multiclass softmax gates.

### Fixed
- **Post-Temperature Conformal Threshold Consistency in `logitgate calibrate`:** Fixed `T_old -> T_new` conformal drift when saving or applying temperature scaling alongside split-conformal thresholds so threshold fitting operates on the post-temperature calibrated distribution.

## [0.5.1] - 2026-11-13

### Added
- **Selective abstention symmetry on `ScoreGate` & `ChoiceGate`:** Added `fail_threshold: float | None` (creating an `AMBIGUOUS` uncertainty band between `fail_threshold` and `threshold`) and `max_score_std: float | None` (abstaining when score standard deviation exceeds the cap, recording `score_std` in `trace.metadata`) to `ScoreGate`, plus `min_confidence: float | None`, `min_margin: float | None` (`top2_margin`), and `save_on_fail: bool | None` to `ChoiceGate`. Both gates now also abstain with `DecisionStatus.AMBIGUOUS` and `abstained_reason="unobserved_target_tokens"` (setting `score=None` on `ScoreGate` and `choice=None` on `ChoiceGate`) when no valid target token appears in `top_logprobs`, enabling clean escalation to Layer-2 `JudgeGate` inside `Cascade`.
- **Reasoning-model Layer-1 suppression & optional `auto_logit_bias`:** Added `suppress_thinking: bool = True` and `auto_logit_bias: bool = False` to `OpenAICompatBackend` and `LlamaCppBackend`. `OpenAICompatBackend` now instructs reasoning models not to think or explain, defaults `reasoning_effort="none"` with sticky HTTP 400/422 opt-out, automatically falls back to `chat_template_kwargs={"enable_thinking": False}` (with sticky circuit-breaking) when detecting empty/`<think>`/`<|thought|>` prefill traps, and probes `/tokenize` (`vLLM` and `llama.cpp` schemas) when `auto_logit_bias=True`. `LlamaCppBackend` injects `chat_template_kwargs={"enable_thinking": False}` in HTTP Mode B (with sticky HTTP 400/422 opt-out) and computes cached `SurfaceTokenMap` `logit_bias` via `llm.tokenize` in in-process Mode A.
- **`fail_fast` batch short-circuiting & async Pass-1 `warm_prefix` scheduling:** Added `fail_fast: bool = False` to `MultiRubricGate`, `LogitGate.evaluate_batch()`, and `LogitGate.aevaluate_batch()` to halt immediately on the first `DecisionStatus.FAILED` rubric, plus `warm_prefix: bool | None = None` on `LogitGate`, `ChoiceGate`, and `ScoreGate` (and `prefers_sequential_prefix_warmup = True` on `OllamaBackend` and `LlamaCppBackend`) so async multi-pass evaluations warm Pass 1 before dispatching Pass 2+ concurrently.

### Changed
- **Legacy Ollama `"think"` compatibility & custom-key `StubBackend`:** `OllamaBackend` now performs sticky HTTP 400 fallback stripping `"think"` on legacy servers that reject the field (`query_top_logprobs` and `aquery_top_logprobs`), and `StubBackend` derives its default pass/fail mock tokens from custom `allowed_tokens` (so custom `ChoiceGate` keys work out of the box).
- **Snapshot replay parity:** Refactored `VerdictSnapshot.replay()` and `areplay()` onto a shared gate builder supporting `fail_threshold`, `max_score_std`, `min_confidence`, `min_margin`, `warm_prefix`, and `MultiRubricGate(..., fail_fast=...)`.

---

## [0.5.0] - 2026-11-12

### Added
- **Generic BYO dynamic auto-calibration & compact-model temperatures:** Added `KNOWN_BINARY_MODEL_TEMPERATURES` alongside `KNOWN_MODEL_TEMPERATURES` (including calibrated `1B`–`4B` entries for `llama3.2:1b`, `qwen2.5-coder:1.5b`, `llama3.2:3b`, `qwen2.5-coder:3b`, and `gemma4:e4b`), upgraded `fit_calibration()` with AUROC- and log-odds scale-regularized optimization (`O(1/N)` MAP decay and `AUROC < 0.70 -> T >= 1.0` fail-safe clamping), added `auto_calibrate_backend()` / `aauto_calibrate_backend()` and lazy first-use BYO calibration (`LOGITGATE_AUTO_CALIBRATE=1`), and exposed `is_temperature_configured()` + `has_known_or_cached_temperature()`.
- **Symmetric 2-pass post-critique `JudgeGate` adjudication & `Cascade` log-odds fusion:** `JudgeGate(..., symmetric_debiasing=True)` now generates bullet-anchored evidence verification notes (lowering the Verdict-First token cutoff to `<= 32` tokens) and runs calibrated 2-pass `A/B` post-critique logit adjudication (`compile_cot_binary_prompt`, `logit_temperature`) before falling back to structured tags or `_probe()`, while `Cascade(..., fuse_logits=True, prior_weight=0.30)` fuses permutation-consistent Tier-1 priors (`abs(z1) >= 1.0`) with uncertain permutation-split Tier-2 critiques (`abs(z2) < 1.0`) via `fuse_escalated_verdict()`.
- **Heterogeneous cascade benchmarking (`--judge-backend` / `--judge-model`):** `run_benchmark()` and `logitgate benchmark` now support separate Tier-1 and Tier-2 backends/models with combined call metering, pre-warmed lazy BYO calibration, and `--cascade-trigger adjudicate`.
- **Expanded snapshot replay & explain parity:** Added `ChoiceGate`, `JudgeGate(symmetric_debiasing=...)`, and `LogitGate(max_positional_bias=...)` parity in `VerdictSnapshot.replay()` / `areplay()`, and persisted `judge_backend` / `judge_model` in heterogeneous `Cascade` snapshots so `VerdictSnapshot.explain()` / `aexplain()` route cleanly to Tier 2.

### Changed
- **Explicit score-scale anchors & nonce boundary isolation:** `compile_score_prompt()` now emits explicit per-integer quality descriptors across the `[min_score, max_score]` scale, and all prompt compilers accept an optional cryptographic `boundary_nonce` (passed automatically by `LogitGate`, `ChoiceGate`, `ScoreGate`, and `JudgeGate`) so candidates and verification notes are wrapped in unpredictable `<candidate_{nonce}>` / `<notes_{nonce}>` tags.
- **`RecurrentPyTorchBackend` security default:** Changed `trust_remote_code` default from `True` to `False`.

### Fixed
- **ReDoS & adversarial fail-closed hardening:** Replaced potentially backtracking regexes in `JudgeGate` with linear `O(N)` patterns, restricted verdict tag extraction to post-`<think>` text, neutralized spoofed section headers in verification notes, enforced strict `> 0.5` and target-token presence on CoT logit reads, handled safety refusals before `_probe()`, and converted CoT `BackendError` fallbacks cleanly.
- **Systems, concurrency, & resource ownership:** Added per-event-loop `asyncio.Lock` and sync `threading.Lock` singleflight guards for lazy BYO calibration, locked read-modify-write cache persistence, prevented injected `httpx` clients from silent replacement with default clients, preserved caller-owned backends on gate `__init__` validation failures, offloaded `LlamaCppBackend` in-process initialization into `run_in_executor`, forwarded `ASTGate` per-call snapshot kwargs inside `Cascade`, and hardened `sanitize_terminal_text` against ANSI CSI/OSC sequences and Unicode BiDi controls.

---

## [0.4.1] - 2026-11-10

### Added
- **Adjudication escalation policy (`trigger="adjudicate"`):** Added `"adjudicate"` (and `allow_fail_override: bool = False`) to `JudgeGate` and `Cascade` so Tier-2 `JudgeGate` can override Tier-1 (`LogitGate`, `ScoreGate`, `MultiRubricGate`) false-negative failures while keeping deterministic Tier-0 (`ASTGate`) structural failures final.
- **Unobserved target-token and positional-bias abstention:** `LogitGate` now abstains (`DecisionStatus.AMBIGUOUS` with `abstained_reason` recorded in `trace.metadata`) when neither `A` nor `B` surface variants appear in `top_logprobs` on either pass, or when `max_positional_bias: float | None` is set and exceeded under a forward/reverse sign flip.
- **`ChoiceGate` derangement and cyclic permutations:** Added `permutation_mode: Literal["derangement", "cyclic", "reverse"] = "derangement"` to `ChoiceGate` so odd-$K$ option sets (e.g., 3 or 5 choices) shift every option to a distinct letter slot on Pass 2 instead of pinning the middle option to the same letter.
- **Surface token `logit_bias` helper:** Added `SurfaceTokenMap.to_logit_bias(bias=100)` to build OpenAI/vLLM/`llama.cpp`-compatible `logit_bias` dictionaries from discovered surface token IDs.
- **BYO `ASTGate` validators, artifact normalization, and `ScoreGate` snapshots:** Added `custom_validators` to `ASTGate`, `normalize_artifacts: bool = False` to `Cascade` (and `normalize: bool = False` to `@check`), and failure snapshot persistence (`save_on_fail` / `snapshot_dir`) to `ScoreGate`.
- **Expanded framework integrations:** `LogitGateMetric` (DeepEval) now supports `ChoiceGate` and `ScoreGate` in addition to `LogitGate`, and `create_logitgate_node` (LangGraph) supports Pydantic `BaseModel` and dataclass graph states alongside `dict` states.

### Changed
- **Single-character token enforcement on HTTP backends:** `OllamaBackend`, `OpenAICompatBackend`, and HTTP `LlamaCppBackend` now wire `allowed_tokens` into the system prompt and scale `top_k` to at least `2 * len(allowed_tokens)` so conversational prefixes do not push target evaluation tokens out of the `top_logprobs` window.
- **Calibration dataset realism:** Replaced synthetic one-line calibration stubs in `UNIVERSAL_CALIBRATION_DATASET` with multi-constraint code, JSON, SQL, and security policy artifacts.
- **Documentation & benchmarks:** Added citations and architectural boundary comparisons against fine-tuned single-token `logprobs` classifiers (such as Airbnb's `GLOWC` benchmark on sub-2B models).

### Fixed
- **Debiasing & calibration numerical stability:** Hardened logprob extraction, temperature scaling (`PlattCalibrator`), and conformal score quantile handling against degenerate or extreme logprob inputs.
- **Backend URI resolution, thread safety, and reasoning stripping:** Hardened `resolve_logit_backend` URI parsing, reasoning `<think>` tag stripping in `JudgeGate`, CLI input validation/output sanitization, and `StubBackend` Pass-2 response cycling for `ChoiceGate`.

---

## [0.3.1] - 2026-11-07

### Added
- **One-time failure replay notice:** When a fast Layer-1 failure occurs without a Layer-2 diagnostic critique and failure snapshots are not yet enabled, LogitGate prints a single-line hint to `stderr` (and appends a tip to `pytest` `@check` assertion errors) showing how to save and replay snapshots via `logitgate explain`. Suppress with `LOGITGATE_QUIET=1`.
- **Automatic PyPI README sync in CI/CD:** Added `pypi_readme_sync` job in `.gitlab-ci.yml` that publishes a PEP 440 `.postN` release to PyPI whenever `README.md` is updated on `main`.
- **Top-level exports:** Exported `CalibrationWarning` and `get_default_model` from `logitgate`.

### Changed
- **Sensible out-of-the-box defaults:**
  - `OllamaBackend` now defaults to `model="qwen2.5:7b"` (~4.7 GB VRAM) instead of `"qwen2.5-coder:14b"` (~9 GB VRAM).
  - `OllamaBackend` no longer forces `num_ctx=16384` on every request unless `num_ctx` or `LOGITGATE_NUM_CTX` is explicitly set, preventing unnecessary KV-cache reallocations and model reloads on warm Ollama instances.
  - `ScoreGate` now defaults to `symmetric_debiasing=True` (matching `LogitGate` and `ChoiceGate`).
  - `JudgeGate` now defaults to `trigger="on_ambiguous_only"` (matching the `logitgate benchmark` default so decisive Layer-1 failures short-circuit without paying for Layer-2 generation).
  - `ChoiceGate` and `ScoreGate` now resolve `logit_temperature=None` and `"auto"` through `KNOWN_MODEL_TEMPERATURES` and `~/.cache/logitgate/calibration.json`, matching `LogitGate`.
- **Packaging:** Removed unused `openai>=1.0.0` dependency from `logitgate[all]` and `logitgate[openai]` (`OpenAICompatBackend` uses `httpx` natively), and added Python 3.13 to the CI test matrix.

### Fixed
- **URI parsing for `openai://` and `vllm://`:** `resolve_logit_backend` now properly handles shorthand `openai://gpt-4o-mini` and model-embedded `vllm://localhost:8000/v1/Qwen/Qwen2.5-7B-Instruct` URIs, and `infer_backend_spec` preserves models across `OpenAICompatBackend`, `RecurrentPyTorchBackend`, and in-process `LlamaCppBackend`.
- **Concurrency & event-loop safety:**
  - Added thread synchronization locks to in-process `LlamaCppBackend`, `RecurrentPyTorchBackend`, and `StubBackend` so concurrent `asyncio.gather` calls in `aevaluate` cannot race on C++ `llama_context` KV caches or lazy model loading.
  - `_HTTPClientMixin` now tracks event-loop affinity on auto-created `httpx.AsyncClient` instances and recreates the client when called across multiple `asyncio.run()` loops instead of raising `RuntimeError: Event loop is closed`.
  - Bounded `LogitGate.aevaluate_batch` concurrency with an `asyncio.Semaphore(32)`.
- **Numerical & gate hardening:**
  - `LogitGate`, `ChoiceGate`, and `ScoreGate` now raise `BackendError` when a backend returns empty `top_logprobs` instead of silently returning `P=0.50` or passing on a uniform distribution.
  - `extract_token_logprob` filters `NaN`/`inf` values, clamps positive fast-math logprob noise to `<= 0.0`, and deduplicates identical raw token entries.
  - `get_model_temperature` checks exact model matches before longest-prefix substring matches in `calibration.json` and `KNOWN_MODEL_TEMPERATURES`.
  - `JudgeGate`'s internal self-healing `_probe_gate` now runs with `save_on_fail=False` so it never overwrites `.logitgate/failures/latest.json`.
  - `ASTGate` now rejects Python `bool` values for JSON schema `"integer"`/`"number"` fields, catches `from <pkg> import <submodule>` against `forbidden_imports`, and fails when `acyclic_dag=True` is given non-DAG input.

---

## [0.3.0] - 2026-11-05

### Changed (behavior)

These are deliberate fixes. Each one makes runtime behavior match what the docs already promised.

- **`JudgeGate(trigger="always")` now runs after a Layer-1 pass inside a `Cascade`.** It was silently skipped on any upstream PASS, which contradicted the documented "regardless of upstream confidence" semantics.
- **Gates outside the built-in classes now escalate AMBIGUOUS verdicts** under `on_ambiguous_only`, instead of failing without calling the judge. This affects wrappers and custom gates; the built-in `ScoreGate` and `ChoiceGate` never emit AMBIGUOUS.
- **Monotonic failure preservation applies to every tier-1 gate.** Before, it only covered `LogitGate`. A decisive `ScoreGate` FAILED followed by `JudgeGate(trigger="on_ambiguous_or_fail")` now stays FAILED; the judge supplies the diagnostic.
- **`JudgeGate` inherits the rubric of the closest upstream tier-1 gate** (e.g. `ScoreGate`), not only from `LogitGate`.
- **`JudgeGate` no longer treats arbitrary JSON in a critique as a verdict.** Only objects with a `verdict`, `pass`, or `passed` key count, nested objects parse correctly, and `"pass": "false"` now means false.
- **HTTP backends raise `BackendError` subclasses instead of raw `httpx` exceptions.** The original is chained as `__cause__`. `BackendUnavailableError` subclasses `ConnectionError` and `BackendTimeoutError` subclasses `TimeoutError`. Code that caught `httpx.HTTPError` from LogitGate calls should catch `BackendError` instead.
- `JudgeGate` trace latency now includes the self-healing logit probe, when it runs.
- Cascade reasons now name the skipped layer, e.g. `Fast-path pass at LogitGate (JudgeGate skipped): ...`.

### Added
- **`MultiRubricGate`:** Atomic multi-rubric evaluation gate with weakest-link aggregation (fails if any criterion fails), per-rubric trace attribution, shared candidate KV-cache prefixing, and bulleted rubric inheritance in `Cascade >> JudgeGate`. Full snapshot persistence and roundtrip `VerdictSnapshot.replay()` support.
- **Routing traits on `BaseGate`:** `layer_tier`, `can_fast_path`, and `escalation_policy`. `Cascade` routes on these instead of `isinstance`, so custom and wrapped gates get fast-path short-circuiting. Defaults preserve the previous treatment of unrecognized gates. `JudgeGate.trigger` is now an alias of `escalation_policy`.
- **`logitgate benchmark`:** a reproducible judge vs `LogitGate` vs cascade comparison on your own backend. It reports p50/p95/mean latency, generated tokens, LLM calls per case, accuracy, and fast-path rate. `--json` adds environment metadata and a dataset hash.
- **`logitgate explain` rewrite:** a per-layer trace table, decision margins in log-odds against both thresholds, interpretation of entropy and label-order bias, forward/reverse probabilities, `--offline` (no model call), and `--color auto|always|never` (honors `NO_COLOR` / `FORCE_COLOR`). If the replay backend is unreachable, the recorded diagnostics still render.
- **Lazy integration loading (PEP 562):** cut `import logitgate` cold-import latency from ~620ms to ~110ms by deferring `deepeval` and `langgraph` loading until accessed.
- **LangGraph conditional edge enhancements:** added `as_async_conditional_edge`, explicit `ambiguous_node` routing, and actionable error messages when candidate state keys are missing.
- **Conformal finite-sample verification:** added `guarantee_holds` tracking and conservative boundary rounding to `calibrate_cascade_thresholds`, issuing a `CalibrationWarning` when calibration sample size $m < \lceil (1 - \alpha/2) / (\alpha/2) \rceil$.
- **Actionable backend errors** for Ollama, OpenAI-compatible servers, and llama.cpp: how to start the server, `ollama pull <model>` on a missing model, `OPENAI_API_KEY` on 401/403, and cold-load hints on timeouts.
- **`StubBackendWarning`:** emitted once when no backend is configured and the offline stub is used implicitly. Silence it with `LOGITGATE_BACKEND=stub://`.
- **`CalibrationWarning`:** emitted when conformal threshold calibration lacks sufficient samples per class for distribution-free error guarantees.
- CLI exit code `2` for backend errors, printed as one actionable line instead of a traceback.
- `BaseGate.replay_config()` and `snapshot.persist_failure_snapshot()`, so snapshot persistence has one implementation.

### Refactored
- `Cascade.evaluate` and `Cascade.aevaluate` now share one generator-based pipeline. Routing, lookahead, and snapshot persistence live in one place; the two drivers differ only in `gate.evaluate` vs `await gate.aevaluate` (`cascade.py`: 579 to 232 lines).
- The sync/async/batch scoring paths of `LogitGate`, `ChoiceGate`, and `ScoreGate` each collapse into one `_verdict_from_logprobs` (`logit_gate.py`: 1,040 to 692 lines).
- `JudgeGate` parsing is split into pure functions (`_extract_structured_verdict`, `_extract_keyword_verdict`) with module-level compiled regexes. The logit probe gate is now built once and cached; it was rebuilt, with a disk read, on every parse.
- `ASTGate`: Kahn's algorithm uses `collections.deque` (O(V+E) instead of O(V²) on wide fan-outs), cycle and import reports are deterministic, and the import scanner plus user patterns are compiled once.
- Internal numeric records (`SurfaceTokenMap`, `DebiasedLogitResult`, `VRAMWatermarkStatus`, `ConformalCascadeThresholds`, `RecurrentLoopTrace`) are now `@dataclass(frozen=True, slots=True)`.
- Ollama `generate`/`agenerate` share payload and continuation builders.

### Fixed
- README `ScoreGate` example used a nonexistent `scale=` argument. A new test executes every README Python block.
- Removed unused imports in `backends/recurrent.py` (ruff F401).

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
