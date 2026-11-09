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

## 2. Numerical & Validation Invariants

- **Logprob Sanitation & Presence:**
  - Always verify that backend `top_logprobs` responses are non-empty via `_ensure_logprobs_present` and raise `BackendError` when empty rather than defaulting to `P = 0.50` or uniform probabilities.
  - `extract_token_logprob` must ignore non-finite values (`NaN`, `±inf`), clamp small positive fast-math logprob noise (`> 0.0`) to `0.0`, and deduplicate identical raw token strings before marginalizing surface variants via `logsumexp`.
- **Temperature Lookup Precedence:**
  Model temperature resolution (`get_model_temperature`) must check exact model key matches before longest-prefix substring matches so base model names (`qwen2.5`) never shadow more specific variants (`qwen2.5-coder:14b`).
- **AST & Schema Validation (`ASTGate`):**
  - In Python, `isinstance(True, int)` is `True`. JSON schema validation for `"integer"` and `"number"` types must explicitly reject `bool` (`not isinstance(val, bool)`).
  - `forbidden_imports` checks must inspect both `ast.Import` and `ast.ImportFrom` (`node.module` and `f"{node.module}.{alias.name}"`).
- **Internal Probe Isolation (`JudgeGate`):**
  When `JudgeGate` invokes an internal self-healing `_probe_gate` to classify an untagged critique, it must pass `save_on_fail=False` so internal probes never overwrite user failure snapshots in `.logitgate/failures/latest.json`.

## 3. Packaging & Optional Dependency Isolation

- Core `logitgate` requires only `httpx` and `pydantic`.
- All optional integrations (`torch`, `transformers`, `safetensors`, `huggingface_hub`, `llama_cpp`, `deepeval`, `langgraph`) must be lazily imported inside the functions or classes that use them so `pytest` passes in minimal `pip install -e ".[test]"` containers without optional extras installed.

## 4. Documentation Style

- Technical documentation in `README.md` and `docs/` must remain direct, precise, and grounded in reproducible benchmarks.
- Avoid marketing superlatives, unquantified claims, and time-based quickstart claims (use "Quickstart", not "30-Second Quickstart"). Use standard hyphens (`-`) or colons/parentheses rather than em-dashes (`—`) in prose.
- Every Python code block in `README.md` is executed by `tests/test_dx_and_tooling.py` and must run offline using default/stub configurations.

## 5. Git Hygiene & Command Execution

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
