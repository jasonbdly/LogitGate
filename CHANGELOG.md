# Changelog

All notable changes to **LogitGate** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-02

### Added
- **3-Layer Cascaded Router (`Cascade`)**:
  - Seamless operator chaining (`ASTGate >> LogitGate >> JudgeGate` or pipe `|`).
  - Fast-path short-circuiting: high-confidence Layer 1 passes bypass generative Layer 2 entirely.
  - Decisive failure short-circuiting when `trigger='on_ambiguous_only'`.
- **Layer 0 Deterministic AST & Contract Gate (`ASTGate` / `ContractGate`)**:
  - Sub-millisecond (~0.04ms) zero-token evaluation on CPU.
  - Python AST syntax validation.
  - Forbidden import detection (`forbidden_imports`) via AST walker with regex fallback.
  - Forbidden pattern matching (`forbidden_patterns`) and required section verification (`required_sections`).
  - Acyclic directed graph (DAG) cycle detection and missing dependency analysis for agentic workflows.
  - Custom callable validator support with informative diagnostic strings.
- **Layer 1 First-Token Debiased Logit Gate (`LogitGate`)**:
  - Sub-50ms latency with 0 generated autoregressive tokens.
  - Symmetric log-odds debiasing ($\bar{L} = \frac{L_1 + L_2}{2T}$) cancelling pretraining token priors and positional bias.
  - Prefix-preserving KV-cache prompt compilation (`compile_binary_prompt`), placing rubric and candidate first to maximize GPU KV-cache hit rate (>99%).
  - Multi-rubric batch evaluation (`evaluate_batch`, `aevaluate_batch`) sharing a byte-identical static prefix across all rubrics with up to 8x speedup.
  - Dynamic missing-token bounded floor calculation to prevent arbitrary cliff distortion in log-odds space.
  - Post-hoc temperature calibration optimization (`fit_calibration`) minimizing Brier score and Expected Calibration Error (ECE).
- **Discrete & Continuous Gating**:
  - `ChoiceGate`: K-ary discrete option selection with forward/reverse ordering debiasing and temperature scaling.
  - `ScoreGate`: Probability-weighted expected continuous scoring ($\mathbb{E}[S] = \sum s \cdot P(s)$) with optional inverted scale debiasing.
- **Layer 2 Generative Reasoning Judge (`JudgeGate`)**:
  - Escalated diagnostic critique and failure explanation on ambiguous or failing artifacts.
  - Configurable trigger policies: `on_ambiguous_or_fail`, `on_ambiguous_only`, and `always`.
  - Reasoning trace suppression stripping `<think>...</think>` tokens for reasoning models.
- **Inference Engine Backends**:
  - `OllamaBackend`: Native Ollama chat endpoint with `think: False` reasoning suppression and connection pooling.
  - `OpenAICompatBackend`: Compatible with vLLM, SGLang, Together AI, Groq, OpenRouter, and OpenAI.
  - `LlamaCppBackend`: In-process `llama_cpp.Llama` integration and `llama-server` HTTP API client.
  - `StubBackend`: Fully deterministic, zero-network offline stub backend for sub-second testing and CI.
- **Framework & Workflow Integrations**:
  - Pytest: `@check(rubric=..., threshold=...)` decorator for test assertions.
  - DeepEval: `LogitGateSystemOneModel` (drop-in System-1 model for `JevEval`) and `LogitGateMetric`.
  - LangGraph: `as_conditional_edge(gate, pass_node, fail_node)` routing function.
- **CLI**:
  - `logitgate eval` command supporting `--rubric`, `--candidate`, `--file`, stdin piping, `--ast-syntax`, and `--json` formatting.
- **Developer Usability & Packaging**:
  - PEP 561 typed (`py.typed`).
  - Informative `__repr__` methods across all gates, backends, traces, and verdicts.
  - Strict input validation raising descriptive `ValueError` exceptions on invalid configurations.
  - Citation metadata conforming to Citation File Format (`CITATION.cff`).
