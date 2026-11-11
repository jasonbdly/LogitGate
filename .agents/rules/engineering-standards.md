# LogitGate Engineering & Repository Standards

These standards apply to all code, documentation, tests, and commits in the LogitGate repository.

## 1. Architecture & Concurrency Invariants

- **Async HTTP Client Lifecycle (`_HTTPClientMixin`):**
  All HTTP-backed inference clients (`OllamaBackend`, `OpenAICompatBackend`, HTTP `LlamaCppBackend`) must inherit `_HTTPClientMixin` from `logitgate.backends.base`. Auto-created `httpx.AsyncClient` instances must track the running event loop (`_async_client_loop`) and recreate cleanly when invoked across sequential `asyncio.run()` calls. Never close a user-injected `async_client`.
- **In-Process Backend Thread Safety:**
  `LogitGate.aevaluate`, `ChoiceGate.aevaluate`, and `ScoreGate.aevaluate` issue forward and reverse debiasing passes concurrently via `asyncio.gather`. Any in-process backend with mutable state, lazy initialization, or C++/CUDA context state (`LlamaCppBackend`, `RecurrentPyTorchBackend`, `StubBackend`) must serialize mutations and non-thread-safe native calls with a `threading.Lock` or `threading.RLock`.
- **Bounded Batch Concurrency:**
  Async batch operations (`aevaluate_batch`) must bound in-flight requests with an `asyncio.Semaphore` after warming the shared prefix on item 0 to prevent KV-cache stampedes and socket exhaustion.
- **No Unsolicited `stdout` Writes:**
  Library modules under `src/logitgate/` must never call `print()` to `stdout`, as CLI `--json` consumers and MCP stdio transports rely on clean standard output. Use `logging.getLogger(__name__)` for operational logs, or `sys.stderr` (respecting `LOGITGATE_QUIET=1`) for one-time developer notices.

## 2. Numerical, Psychometric & Validation Invariants

- **Logprob Sanitation, Target-Token Abstention & Single-Char Enforcement:**
  - Always verify that backend `top_logprobs` responses are non-empty via `_ensure_logprobs_present` and raise `BackendError` when empty rather than defaulting to `P = 0.50` or uniform probabilities.
  - `extract_token_logprob` must ignore non-finite values (`NaN`, `±inf`), clamp small positive fast-math logprob noise (`> 0.0`) to `0.0`, and deduplicate identical raw token strings before marginalizing surface variants via `logsumexp`.
  - When neither positive (`A`) nor negative (`B`) surface variants appear in `top_logprobs` on either pass (`has_any_target_token`), `LogitGate` must abstain (`DecisionStatus.AMBIGUOUS` with `abstained_reason` in `trace.metadata`) rather than silently computing `log_odds = 0.0`, and honor `max_positional_bias` abstention on sign-flipping passes.
  - HTTP backends (`OllamaBackend`, `OpenAICompatBackend`, HTTP `LlamaCppBackend`) must wire `allowed_tokens` into the system prompt and scale `top_k >= 2 * len(allowed_tokens)` so conversational prefixes on chat-tuned models do not evict target tokens from `top_logprobs`.
- **Temperature Lookup Precedence & Explicit-Backend Neutrality:**
  - Model temperature resolution (`get_model_temperature`) must check exact model key matches before longest-prefix substring matches so base model names (`qwen2.5`) never shadow more specific variants (`qwen2.5-coder:14b`).
  - When a caller injects an explicit `Backend` instance (e.g., `StubBackend` or custom `Backend`) into a gate or adapter (`LogitGateSystemOneModel`) without specifying `model_name` or `logit_temperature`, default to neutral temperature (`1.0` / `"default"`) rather than applying model-specific temperatures from `KNOWN_MODEL_TEMPERATURES`.
- **Ordinal & Categorical Debiasing (`ScoreGate` & `ChoiceGate`):**
  - Combine forward and reverse debiasing passes in log-probability space across aligned labels/scores (log-linear pooling / geometric mean) before computing expected values or argmax choices. Never average forward and reverse expected scores directly (`0.5 * (E_fwd + E_rev)`), which masks bimodal/J-shaped distributions and leaves token-prior bias uncancelled.
  - `ChoiceGate` must default Pass-2 option permutations to a fixed-point-free `"derangement"` (`shift = max(1, n // 2)`) rather than simple reversal (`"reverse"`), which pins the middle option (`(K - 1) // 2`) to the same letter slot on odd `K`.
- **Cascade Failure Monotonicity vs. Adjudication (`Cascade` & `JudgeGate`):**
  - Keep standard cascade routing fail-closed (`on_ambiguous_only` / `on_ambiguous_or_fail`), while supporting `trigger="adjudicate"` (`allow_fail_override=True`) so Tier-2 `JudgeGate` can rescue Tier-1 (`LogitGate`, `ScoreGate`, `MultiRubricGate`) false negatives while keeping deterministic Tier-0 (`ASTGate`) structural failures strictly final.
- **Conformal Calibration Quantiles:**
  - Finite-sample split conformal thresholds (`calibrate_cascade_thresholds`) must use exact conservative order-statistic indices (`ceil((n + 1) * (1 - alpha))` and symmetric lower rank) and emit `CalibrationWarning` when calibration sample counts are insufficient for non-trivial coverage.
- **AST & Schema Validation (`ASTGate`):**
  - In Python, `isinstance(True, int)` is `True`. JSON schema validation for `"integer"` and `"number"` types must explicitly reject `bool` (`not isinstance(val, bool)`).
  - `forbidden_imports` checks must inspect both `ast.Import` and `ast.ImportFrom` (`node.module` and `f"{node.module}.{alias.name}"`).
- **Internal Probe Isolation (`JudgeGate`):**
  - When `JudgeGate` invokes an internal self-healing `_probe_gate` to classify an untagged critique, it must pass `save_on_fail=False` so internal probes never overwrite user failure snapshots in `.logitgate/failures/latest.json`.

## 3. Adversarial Security & Untrusted Input Isolation

- **Randomized Prompt Boundary Nonces:**
  - Untrusted `candidate` text must never be enclosed solely in static prompt headers. Layer 1 (`build_forward_prompt` / `build_reverse_prompt`) and Layer 2 (`JudgeGate`) prompts must wrap candidate artifacts in per-call randomized boundary nonces so spoofed `[EVALUATION RUBRIC]` or `[VERDICT: PASS]` strings inside candidates cannot hijack verdicts.
- **Linear-Time Parsing & Bounded Scans:**
  - Regular expressions running over untrusted model outputs must avoid nested/ambiguous quantifiers (no catastrophic backtracking), and fallback JSON scanners (`raw_decode` loops) must enforce bounded decode-attempt caps so pathological inputs (e.g., thousands of unclosed `{` or backticks) fail in `O(N)` time.
- **Dual Raw + Normalized Security Scanning (`ASTGate`):**
  - When `normalize_text=True` strips `<think>` blocks or markdown code fences, `forbidden_patterns` must still inspect the raw unstripped candidate in addition to normalized text so disallowed content inside `<think>` tags cannot bypass Layer 0 checks.
- **Terminal Escape Sanitization (`cli.py`):**
  - Sanitize ANSI CSI/OSC escape sequences and C0/C1 control characters before rendering untrusted prompts, candidates, or backend messages to terminal `stdout`/`stderr`, and validate all numeric CLI flags against `NaN`, `±inf`, and domain bounds.

## 4. Packaging, Cross-Surface Parity & Tooling

- Core `logitgate` requires only `httpx` and `pydantic`.
- All optional integrations (`torch`, `transformers`, `safetensors`, `huggingface_hub`, `llama_cpp`, `deepeval`, `langgraph`) must be lazily imported inside the functions or classes that use them so `pytest` passes in minimal `pip install -e ".[test]"` containers without optional extras installed.
- **Cross-Surface Feature Propagation:** Any new gate parameter or normalization capability must be wired consistently through `Cascade`, `VerdictSnapshot` (`replay_config()` and `replay()`), `@check` (`pytest_plugin.py`), and `cli.py`.
- **Integration Polymorphism:** `LangGraphGuardNode` must support `Mapping`, Pydantic `BaseModel` (v1/v2), and `@dataclass` state containers (with both sync and async callable factories), and `LogitGateSystemOneModel` must support `YesNoAnswer`, `ChoiceAnswer`, and `ScoreAnswer` schemas.
- **Local Verification Commands:** Ensure `~/.local/bin` is on `PATH` when invoking `uv` / `uvx` in non-interactive shells (`uvx ruff check src/logitgate tests`, `uv run --with mypy mypy src/logitgate tests`, `.venv/bin/pytest -q`).
- **Centralized GitLab CI & GitHub Storefront Syndication:**
  - `.gitlab-ci.yml` is the single source of truth for testing, linting, PyPI OIDC publishing, GitLab Releases, and syndicating commit statuses and releases to the GitHub mirror (`GITHUB_TOKEN`). Never add duplicate `.github/workflows/` pipelines, and prefer official CLIs (`glab`, `gh`) over custom inline HTTP wrappers in CI scripts.

## 5. Documentation Style

- Technical documentation in `README.md` and `docs/` must remain direct, precise, and grounded in reproducible benchmarks (clearly distinguishing zero-training `3B-14B` cascade performance from fine-tuned single-token `logprobs` classifiers required on sub-2B models).
- Avoid marketing superlatives, unquantified claims, and time-based quickstart claims (use "Quickstart", not "30-Second Quickstart"). Use standard hyphens (`-`) or colons/parentheses rather than em-dashes (`—`) in prose, and use code spans (``odd-`K` ``) rather than inline `$...$` LaTeX math in `CHANGELOG.md` so syndicated GitHub Release notes render cleanly.
- Every Python code block in `README.md` is executed by `tests/test_dx_and_tooling.py` and must run offline using default/stub configurations.

## 6. Git Hygiene & Command Execution

Follow standard industry Git practices (Conventional Commits, 50/72 formatting, and `git bisect`-safe vertical commits):

- **Vertical Slices (Code + Tests + Docs Together):**
  - Each commit must represent **one cohesive logical change** (one feature, one bug fix, or one refactor) along with the unit tests and documentation updates for that change.
  - **Anti-pattern to avoid:** Never split implementation into one commit and its corresponding test updates into a separate `test:` commit, and never group unrelated fixes into a single "grab-bag" commit just because they touch the same directory (`backends/`, `gates/`).
- **Every Commit Must Pass Quality Gates (`git bisect` Safety):**
  - Every individual commit must leave the repository in a green, working state where `ruff check`, `mypy`, and `pytest` pass. Never create intermediate commits that break existing test assertions until a later commit updates the tests.
- **Commit Message Structure (50/72 Rule, Focus on *Why*):**
  - **Subject line:** `<type>(<scope>): <imperative summary>` (`<= 72` characters, ideally `<= 50`, lowercase summary, no trailing period).
  - **Body:** Wrap lines at 72 characters. Explain **why** the change is necessary (the root cause, trade-off, or motivation), not a mechanical bullet-by-bullet inventory repeating the code diff. If a commit requires a long list of unrelated bullets to describe, it is not atomic and must be split into separate commits before committing.
- **Pure Release Commits (`chore(release)`):**
  - `chore(release): bump version to X.Y.Z` commits must only modify the 5 synchronized metadata files (`pyproject.toml`, `src/logitgate/__init__.py`, `CITATION.cff`, `README.md`, `CHANGELOG.md`). All functional changes must be committed in their own vertical commits beforehand.
- **Immutable Published History:**
  - Never amend, rebase, or force-push commits or tags that have already been pushed to `origin/main`.
- **Deterministic Subprocess Timeouts:**
  - Wrap networked CLI commands (`git push`, `git fetch`, remote probes) in `timeout -k 1 <seconds>` so hung network sockets fail deterministically rather than blocking indefinitely.
