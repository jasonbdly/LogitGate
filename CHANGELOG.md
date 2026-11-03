# Changelog

All notable changes to **LogitGate** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-02

### Added

#### 3-Layer Cascaded Router (`Cascade`)
- Seamless operator chaining (`ASTGate >> LogitGate >> JudgeGate` or pipe `|`).
- Fast-path short-circuiting: high-confidence Layer 1 passes bypass generative Layer 2 entirely.
- Decisive failure short-circuiting when `trigger='on_ambiguous_only'`.
- Monotonic failure enforcement: when Layer 1 decisively fails (`status == FAILED`) and escalates to Layer 2 for diagnostic explanation, `passed=False` is preserved even if the downstream generative critique phrasing is lenient.

#### Layer 0 Deterministic AST & Contract Gate (`ASTGate` / `ContractGate`)
- Sub-millisecond (~0.04ms) zero-token evaluation on CPU.
- Python AST syntax validation (`syntax=True`) and forbidden import detection (`forbidden_imports`) via AST walker with regex fallback.
- General-domain JSON Schema & Pydantic `BaseModel` validation (`json_schema`), character length bounds (`min_length`, `max_length`), forbidden regex patterns (`forbidden_patterns`), and required section verification (`required_sections`).
- Acyclic directed graph (DAG) cycle detection and missing dependency analysis (`acyclic_dag=True`) for agentic task graphs.
- Custom callable validator support (`validators`) with informative diagnostic strings.

#### Layer 1 First-Token Debiased Logit Gate (`LogitGate`)
- 40–120ms GPU latency (~170ms on pure CPU) with 0 generated autoregressive tokens.
- Built-in threshold presets: `preset="strict"` (`0.90 / 0.25`, symmetric 2-pass), `preset="balanced"` (`0.85 / 0.20`, symmetric 2-pass), and `preset="fast_triage"` (`0.80 / 0.20`, single-pass for local CPU iteration & high-throughput triage).
- Symmetric log-odds debiasing ($\bar{L} = \frac{L_1 + L_2}{2T}$) cancelling pretraining token priors and positional bias.
- Prefix-preserving KV-cache prompt compilation (`compile_binary_prompt`), placing rubric and candidate first to maximize GPU KV-cache hit rate (>99%).
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
