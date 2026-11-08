---
name: logitgate-audit
description: >-
  Executes a multi-track pre-release code quality and architectural audit across
  LogitGate. Use before cutting a release, after major refactors, or when
  verifying systems concurrency, numerical stability, and CI/packaging parity.
---

# LogitGate Code Quality & Pre-Release Audit Skill

This skill defines a structured, three-track verification workflow for auditing the LogitGate codebase prior to a release or major merge.

When executing a full audit, launch three parallel read-only `research` subagents (one per track below) with zero prior conversation context so they inspect the code strictly as written on disk.

---

## Track 1: Python Systems, Concurrency, & Async Lifecycle

**Scope:** `src/logitgate/backends/`, `src/logitgate/cascade.py`, `src/logitgate/config.py`, `src/logitgate/snapshot.py`, `src/logitgate/exceptions.py`

**Checklist:**
1. **Event-Loop Affinity (`_HTTPClientMixin`):**
   - Verify all HTTP backends (`OllamaBackend`, `OpenAICompatBackend`, HTTP `LlamaCppBackend`) inherit `_HTTPClientMixin` and recreate auto-owned `httpx.AsyncClient` instances when called across sequential `asyncio.run()` loops.
   - Verify user-provided `client` / `async_client` instances are never closed by backend `close()` / `aclose()`.
2. **Thread & Coroutine Safety:**
   - Verify in-process backends (`LlamaCppBackend`, `RecurrentPyTorchBackend`, `StubBackend`) guard mutable state, lazy initialization, and non-thread-safe native contexts with locks under concurrent `asyncio.gather` calls.
   - Verify global state in `config.py` and `snapshot.py` is protected by locks.
3. **URI Resolution & Round-Trip Spec Inference:**
   - Verify `resolve_logit_backend` and `infer_backend_spec` preserve host, port, `/v1` prefix, and model identifiers across `ollama://`, `vllm://`, `openai://`, `llamacpp://`, `recurrent://`, and `stub://` URIs.
4. **Exception Hierarchy & Resource Cleanup:**
   - Verify network errors map to `BackendTimeoutError` / `BackendUnavailableError` and malformed responses raise `BackendError`.
   - Verify atomic file writes in `snapshot.py` and `calibration/conformal.py` clean up temporary files on failure.

---

## Track 2: Numerical Stability, Debiasing, & Gate Logic

**Scope:** `src/logitgate/debiasing/`, `src/logitgate/calibration/`, `src/logitgate/gates/`

**Checklist:**
1. **Logprob Extraction & Floating-Point Boundaries:**
   - Verify `extract_token_logprob` filters `NaN` and `±inf`, clamps small positive fast-math logprobs (`> 0.0`) to `0.0`, and deduplicates identical raw token keys before `logsumexp`.
   - Verify `_compute_missing_token_floor` remains finite when cumulative top-$K$ probability approaches or slightly exceeds `1.0`.
   - Verify `clamped_sigmoid` and `softmax_probabilities` cannot overflow (`math.exp(>709)`) or divide by zero.
2. **Empty Backend Logprobs Guard:**
   - Verify `LogitGate`, `ChoiceGate`, and `ScoreGate` raise `BackendError` when a backend returns empty `top_logprobs` or all non-finite logprobs.
3. **Symmetric Permutation Debiasing & Calibration:**
   - Verify `LogitGate`, `ChoiceGate`, and `ScoreGate` resolve `logit_temperature=None` and `"auto"` consistently through `get_model_temperature` (exact match before longest-prefix match).
   - Verify `SurfaceTokenMap` rejects overlapping positive and negative token IDs.
   - Verify `fit_conformal_thresholds` and `fit_calibration` validate input bounds and emit `CalibrationWarning` on degenerate data.
4. **Gate Input & Verdict Validation:**
   - Verify `ASTGate` rejects `bool` for JSON schema `"integer"` / `"number"` fields, checks both `ast.Import` and `ast.ImportFrom` for `forbidden_imports`, and validates `acyclic_dag` inputs.
   - Verify `JudgeGate` passes `save_on_fail=False` to its internal `_probe_gate` and parses `[VERDICT: PASS/FAIL]` and JSON verdicts with word boundaries.

---

## Track 3: Packaging, CLI, Integrations, & CI Parity

**Scope:** `pyproject.toml`, `.gitlab-ci.yml`, `src/logitgate/cli.py`, `src/logitgate/benchmark.py`, `src/logitgate/integrations/`, `README.md`, `docs/`, `tests/`

**Checklist:**
1. **Minimal Environment Import Isolation:**
   - Verify `pytest` passes when all optional dependencies (`torch`, `transformers`, `safetensors`, `huggingface_hub`, `llama_cpp`, `deepeval`, `langgraph`) are absent, matching `.gitlab-ci.yml`'s `pip install -e ".[test]"` environment.
2. **CLI Exit Codes & Output Contracts:**
   - Verify `logitgate eval`, `explain`, `calibrate`, and `benchmark` honor `--json` (zero extra `stdout` noise) and exit codes (`0` pass, `1` fail/invalid input, `2` backend error).
3. **CI/CD Pipeline Integrity:**
   - Verify `.gitlab-ci.yml` `lint` (`ruff` + `mypy src/logitgate tests`), `pytest` matrix (`3.10`–`3.13`), `release`, and `pypi_readme_sync` jobs stay synchronized with `pyproject.toml` and `src/logitgate/__init__.py`.
4. **Documentation & Snippet Accuracy:**
   - Verify all `README.md` and `docs/` defaults match the implementation (`OllamaBackend` `model="qwen2.5:7b"`, `num_ctx=None`, `ScoreGate` `symmetric_debiasing=True`, `JudgeGate` `trigger="on_ambiguous_only"`).
   - Verify `pytest tests/test_readme.py` executes all `README.md` code blocks cleanly.
