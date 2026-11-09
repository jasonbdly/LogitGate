---
name: logitgate-audit
description: >-
  Executes a five-track pre-release code quality, security, ML evaluation science,
  and architectural audit across LogitGate. Use before cutting a release, after
  major refactors, or when verifying systems concurrency, numerical stability,
  adversarial robustness, psychometric validity, and CI/packaging parity.
---

# LogitGate Code Quality & Pre-Release Audit Skill

This skill defines a structured, five-track verification workflow for auditing the LogitGate codebase prior to a release or major merge.

When executing a full audit, launch five parallel read-only `research` subagents (one per track below) with zero prior conversation context so they inspect the code strictly as written on disk from distinct engineering, scientific, and adversarial perspectives.

---

## Track 1: Python Systems, Concurrency, & Async Lifecycle

**Scope:** `src/logitgate/backends/`, `src/logitgate/cascade.py`, `src/logitgate/config.py`, `src/logitgate/snapshot.py`, `src/logitgate/hardware/`, `src/logitgate/exceptions.py`

**Checklist:**
1. **Event-Loop Affinity (`_HTTPClientMixin`):**
   - Verify all HTTP backends (`OllamaBackend`, `OpenAICompatBackend`, HTTP `LlamaCppBackend`) inherit `_HTTPClientMixin` and recreate auto-owned `httpx.AsyncClient` instances when called across sequential `asyncio.run()` loops.
   - Verify user-provided `client` / `async_client` instances are never closed by backend `close()` / `aclose()`.
2. **Thread & Coroutine Safety:**
   - Verify in-process backends (`LlamaCppBackend`, `RecurrentPyTorchBackend`, `StubBackend`) guard mutable state, lazy initialization, and non-thread-safe native contexts with locks under concurrent `asyncio.gather` calls.
   - Verify global state in `config.py` and `snapshot.py` is protected by locks.
3. **URI Resolution & Round-Trip Spec Inference:**
   - Verify `resolve_logit_backend` and `infer_backend_spec` preserve host, port, `/v1` prefix, query parameters, and model identifiers across `ollama://`, `vllm://`, `openai://`, `llamacpp://`, `recurrent://`, and `stub://` URIs.
4. **Cascade & Snapshot Lifecycle:**
   - Verify `Cascade` forwards normalized candidates (`_normalized_out` / `normalize_input`) from Layer 0 (`ASTGate`) to downstream `LogitGate` and `JudgeGate` layers, and preserves upstream `probability`, `score`, `choice`, and `choice_probabilities` when non-`None`.
   - Verify atomic file writes in `snapshot.py` clean up temporary files on failure and close auto-created backends in `replay()` / `explain()`.

---

## Track 2: Numerical Stability, Debiasing, & Gate Logic

**Scope:** `src/logitgate/debiasing/`, `src/logitgate/calibration/`, `src/logitgate/gates/`

**Checklist:**
1. **Logprob Extraction & Floating-Point Boundaries:**
   - Verify `extract_token_logprob` filters `NaN` and `±inf`, clamps small positive fast-math logprobs (`> 0.0`) to `0.0`, and deduplicates identical raw token keys before `logsumexp`.
   - Verify `_compute_missing_token_floor` remains finite when cumulative top-$K$ probability approaches or slightly exceeds `1.0`.
   - Verify `clamped_sigmoid`, `softmax_probabilities`, and `marginalize_pole_logprob` cannot overflow (`math.exp(>709)`), divide by zero, or produce `NaN` on saturated `±inf` poles.
2. **Empty Backend Logprobs Guard:**
   - Verify `LogitGate`, `ChoiceGate`, `ScoreGate`, and `auto_calibrate_backend` raise `BackendError` when a backend returns empty `top_logprobs` or all non-finite logprobs.
3. **Threshold Rounding & Input Validation:**
   - Verify `LogitGate` and `ScoreGate` compare `round(prob, 4)` / `round(score, 4)` against thresholds so reported 4-decimal metrics never contradict verdict status.
   - Verify `bool` is rejected for numeric thresholds, lengths, token caps, JSON schema `"integer"` / `"number"` fields, and calibration labels.

---

## Track 3: Packaging, CLI, Integrations, & CI Parity

**Scope:** `pyproject.toml`, `.gitlab-ci.yml`, `src/logitgate/cli.py`, `src/logitgate/benchmark.py`, `src/logitgate/integrations/`, `README.md`, `docs/`, `tests/`

**Checklist:**
1. **Minimal Environment Import Isolation:**
   - Verify `pytest` passes when all optional dependencies (`torch`, `transformers`, `safetensors`, `huggingface_hub`, `llama_cpp`, `deepeval`, `langgraph`) are absent, matching `.gitlab-ci.yml`'s `pip install -e ".[test]"` environment.
2. **CLI Exit Codes & Output Contracts:**
   - Verify `logitgate eval`, `explain`, `calibrate`, and `benchmark` honor `--json` (zero extra `stdout` noise) and exit codes (`0` pass, `1` fail/invalid input, `2` backend error).
3. **Framework Integrations (`pytest`, `DeepEval`, `LangGraph`):**
   - Verify `@check` invalidates cached gates when `logitgate.configure()` changes defaults and forwards `normalize` to `LogitGate`.
   - Verify `LogitGateMetric` respects `preset` thresholds and `LogitGateSystemOneModel` supports both dict and object question items.
4. **Documentation & Snippet Accuracy:**
   - Verify all `README.md` and `docs/` defaults match the implementation (`OllamaBackend` `model="qwen2.5:7b"`, `num_ctx=None`, `ScoreGate` `symmetric_debiasing=True`, `JudgeGate` `trigger="on_ambiguous_only"`), and `pytest tests/test_dx_and_tooling.py` executes all `README.md` and `docs/` code blocks cleanly.

---

## Track 4: Adversarial Security, Prompt Injection, & DoS/ReDoS

**Scope:** `src/logitgate/debiasing/inversion.py`, `src/logitgate/gates/ast_gate.py`, `src/logitgate/gates/judge_gate.py`, `src/logitgate/backends/base.py`, `src/logitgate/snapshot.py`

**Checklist:**
1. **Prompt Injection & Delimiter Spoofing:**
   - Verify untrusted `candidate` artifacts containing spoofed prompt headers (`[EVALUATION RUBRIC]`, `[DECISION PROTOCOL]`, `[VERDICT: PASS]`) or `<think>...</think>` tags cannot hijack Layer 1 or Layer 2 verdict extraction or bypass `ASTGate` `forbidden_imports` / `forbidden_patterns`.
2. **Regex & Parser Denial of Service (ReDoS / $O(N^2)$ Scans):**
   - Verify all regular expressions (`_THINK_BLOCK_RE`, `_FENCE_BLOCK_RE`, `_INLINE_FENCE_RE`, `_IMPORT_STMT_RE`, `_VERDICT_TAG_RE`, `_NEGATED_FAILURE_RE`, `_NEGATED_PASS_RE`) and JSON scanners (`_extract_json_from_text`, `_iter_json_objects`) run in linear time without catastrophic backtracking or quadratic `raw_decode` loops on pathological inputs (e.g., 50,000 unclosed `{` or backticks).
3. **Filesystem & Deserialization Safety:**
   - Verify snapshot filenames sanitize and bound `test_name` slugs, and `ast.parse` catches `RecursionError`, `MemoryError`, `SyntaxError`, and `ValueError`.

---

## Track 5: ML Evaluation Science, Psychometrics, & End-to-End API Symmetry

**Scope:** `src/logitgate/gates/`, `src/logitgate/debiasing/`, `src/logitgate/calibration/`, `src/logitgate/snapshot.py`, `src/logitgate/benchmark.py`

**Checklist:**
1. **Psychometric & Debiasing Validity:**
   - Verify `LogitGate`, `ChoiceGate`, and `ScoreGate` forward/reverse permutation passes properly invert token-to-label/score mappings and combine distributions without uncancelled positional bias or scale-direction errors.
   - Verify conformal calibration quantiles (`calibrate_cascade_thresholds`) and temperature scaling (`fit_calibration`) maintain statistical guarantees and warn via `CalibrationWarning` on degenerate or under-sampled data.
2. **Cross-Gate API & Snapshot Replay Symmetry:**
   - Verify `LogitGate`, `MultiRubricGate`, `ChoiceGate`, `ScoreGate`, `JudgeGate`, `ASTGate`, and `Cascade` maintain consistent `evaluate()` / `aevaluate()` contracts, `replay_config()` serialization, and `VerdictSnapshot.replay()` / `explain()` behavior across all gate types.
