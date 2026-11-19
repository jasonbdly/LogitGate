[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](../eli5/how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)

---

# Quickstart Recipes (⚡ Quick Notes)

Copy-paste recipes, constructor defaults, and production deployment patterns for core LogitGate primitives.

---

## 1. `LogitGate` (Binary First-Token Classifier)

**Problem & deployment**: Replaces slow generative yes/no judges in CI regression gates, live chatbot guardrails, and output validators by extracting a calibrated `0.0-1.0` pass probability in `~16-90 ms` with `0` generated tokens.

### Copy-Paste Recipe

```python
from logitgate import LogitGate

# Initialize gate with calibrated thresholds and symmetric debiasing
gate = LogitGate(
    rubric="Does the SQL query parameterize all user variables?",
    pass_threshold=0.85,
    fail_threshold=0.20,
    symmetric_debiasing=True,
)

candidate = "SELECT id, email FROM users WHERE id = :user_id"
verdict = gate.evaluate(candidate)

print(f"Passed: {verdict.passed}")              # True
print(f"Status: {verdict.status.value}")        # "passed"
print(f"Confidence: {verdict.probability:.4f}")  # e.g. 0.9982
print(f"Tokens generated: {verdict.total_tokens_generated}")  # 0
```

### Constructor Signature & Defaults

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rubric` | `str` | *(required)* | Evaluation criteria stated as a clear yes/no question. |
| `pass_threshold` | `float` | `0.85` | Probability threshold for decisive `PASSED` status. |
| `fail_threshold` | `float` | `0.20` | Probability threshold below which status is decisive `FAILED`. |
| `backend` | `str \| LogitBackend \| None` | `None` | URI or instance (defaults to `LOGITGATE_BACKEND` or stub). |
| `model` | `str \| None` | `None` | Model override for the backend endpoint. |
| `symmetric_debiasing` | `bool` | `True` | 2-pass permutation canceling token and position priors. |
| `logit_temperature` | `float \| "auto" \| None` | `None` | Temperature scaling divisor ($T$) for sigmoid calibration. |
| `max_positional_bias` | `float \| None` | `None` | Abstains if log-odds difference between passes exceeds this cap (e.g., `3.0`). |
| `on_unsupported_logprobs`| `Literal["raise", "escalate", "sidecar"]` | `"raise"` | Behavior when backend lacks logprobs. |
| `sidecar_backend` | `str \| LogitBackend \| None` | `None` | Alternative backend when using sidecar mode. |

<details>
<summary>💡 <b>Why this works</b></summary>

Standard AI evaluators write 200+ words of explanation before saying yes or no. LogitGate stops right after the model reads the prompt and checks its next-word probabilities directly. Running two quick passes with the labels swapped (`A = Pass, B = Fail`, then `A = Fail, B = Pass`) cancels out any built-in habit of picking `"A"` or `"Yes"`. See [How LogitGate Works](../eli5/how_logitgate_works.md).
</details>

[🔬 Mathematical Proofs & Log-Odds Cancellation](../math_and_debiasing.md)

---

## 2. `Cascade` (3-Layer Routing Pipeline)

**Problem & deployment**: Cuts LLM verification cost and tail latency in code-generation pipelines and production APIs by failing malformed syntax at Layer 0 (`<0.1 ms`), fast-pathing clear passes/fails at Layer 1 (`~16-90 ms`), and escalating only uncertain edge cases (`0.20 < P < 0.85`) to a generative `JudgeGate`.

### Copy-Paste Recipe

```python
from logitgate import ASTGate, Cascade, JudgeGate, LogitGate

# Build cascade using the >> operator
cascade = (
    # Layer 0: Deterministic syntax & import check (<0.1ms, 0 LLM calls)
    ASTGate(syntax=True, forbidden_imports=["os", "subprocess"])
    # Layer 1: Debiased first-token logit triage (~16-90ms, 0 generated tokens)
    >> LogitGate(
        rubric="Does the function validate all input arguments?",
        pass_threshold=0.85,
        fail_threshold=0.20,
    )
    # Layer 2: Chain-of-thought generative judge (only runs if 0.20 < P < 0.85)
    >> JudgeGate(trigger="on_ambiguous_only")
)

# Use as a context manager for clean lifecycle management
with cascade:
    verdict = cascade.evaluate("def add(a: int, b: int) -> int: return a + b")
    print(f"Triggered layer: {verdict.triggered_layer}")
```

### Routing Modes

- `trigger="on_ambiguous_only"` (Default): Layer 2 executes only when Layer 1 confidence is ambiguous (`fail_threshold < P < pass_threshold`). Decisive passes and fails exit immediately.
- `trigger="on_ambiguous_or_fail"`: Layer 2 executes on both ambiguity and failure.
- `allow_fail_override=True`: Allows Layer 2 `JudgeGate` to adjudicate and rescue Layer 1 false negatives while keeping Layer 0 AST structural failures final.

[🔬 Systems Architecture & Cascade Lifecycle](../architecture.md)

---

## 3. `ChoiceGate` (Discrete Categorical Routing)

**Problem & deployment**: Routes support tickets, triage queues, and intent classifiers across discrete categories without token generation, eliminating alphabetical `"A"` slot bias via fixed-point-free derangement permutations (circular slot rotations where no option stays on the same letter).

### Copy-Paste Recipe

```python
from logitgate import ChoiceGate

router = ChoiceGate(
    question="Classify this user support request into the right department:",
    options={
        "A": "Billing and subscription refunds",
        "B": "Authentication and security permissions",
        "C": "Database indexing and SQL performance",
    },
    min_confidence=0.65,
    min_margin=0.15,
    permutation_mode="derangement",
)

verdict = router.evaluate("My credit card was charged twice for invoice #104.")
print(f"Selected route: {verdict.choice}")                  # 'A'
print(f"Confidence: {verdict.probability:.4f}")
print(f"Distribution: {verdict.choice_probabilities}")
```

### Key Parameters & Invariants

- `permutation_mode="derangement"`: Shifts options circularly ($shift = \max(1, K // 2)$) on Pass 2 so no option remains pinned to the same letter slot on odd $K$.
- `min_confidence`: Rejects with `DecisionStatus.AMBIGUOUS` if top choice probability is below this threshold.
- `min_margin`: Rejects with `DecisionStatus.AMBIGUOUS` if difference between top and second choice is below this margin.

<details>
<summary>💡 <b>Why this works</b></summary>

Models prefer option 'A' over 'B' because 'A' appears earlier in the alphabet and earlier in the prompt. By evaluating the question twice with rotated letter assignments, ChoiceGate removes letter favoritism entirely.
</details>

[🔬 Choice Debiasing Math](../math_and_debiasing.md)

---

## 4. `ScoreGate` (Ordinal & Continuous Rubric Scoring)

**Problem & deployment**: Replaces noisy integer-only prompt grading in automated code review, essay/summary grading, and quality dashboards by computing a probability-weighted expected score across discrete anchor logits.

### Copy-Paste Recipe

```python
from logitgate import ScoreGate

scorer = ScoreGate(
    rubric="Rate code clarity and type safety from 1 (broken) to 5 (production-ready):",
    min_score=1,
    max_score=5,
    threshold=4.0,
    fail_threshold=2.5,
    max_score_std=1.25,
    symmetric_debiasing=True,
)

verdict = scorer.evaluate("def add(a: int, b: int) -> int: return a + b")
score_str = f"{verdict.score:.2f} / 5.00" if verdict.score is not None else "N/A"
print(f"Expected Score: {score_str}")
print(f"Passed threshold (>=4.0): {verdict.passed}")
```

### Key Parameters & Invariants

- Combines forward and reverse passes via **log-linear pooling** (geometric mean) in log-probability space before computing expected values. Never averages forward and reverse scores linearly, which distorts bimodal (split between extremes) distributions.
- `max_score_std`: Abstains with `DecisionStatus.AMBIGUOUS` if the posterior distribution standard deviation exceeds this cap (indicating a split or conflicted model).

---

## 5. `MultiRubricGate` (Shared-Prefix Multi-Criterion Checks)

**Problem & deployment**: Evaluates multiple policy, formatting, and correctness checks against a single long artifact (such as a PR diff, legal summary, or RAG response) in one batch, reusing the candidate KV-cache prefix across every criterion.

### Copy-Paste Recipe

```python
from logitgate import MultiRubricGate

rubrics = [
    "Does the code include parameter and return type annotations?",
    "Does the function handle potential None inputs safely?",
    "Does the function avoid catching bare Exception without re-raising?",
]

gate = MultiRubricGate(rubrics, fail_fast=False)
candidate = "def get_user(uid: int | None) -> dict | None: return {'id': uid} if uid else None"

# Aggregate pass/fail verdict
verdict = gate.evaluate(candidate)
print(f"Overall passed: {verdict.passed}")

# Individual breakdown per rubric
for rubric, v in zip(rubrics, gate.evaluate_each(candidate), strict=True):
    status = "PASS" if v.passed else "FAIL"
    print(f"[{status}] P={v.probability:.4f} | {rubric}")
```

### Weakest-Link Aggregation & `fail_fast`

- **Weakest-link verdict**: `MultiRubricGate` passes only when every criterion passes (`PASSED`); any `FAILED` rubric fails the overall verdict, any `AMBIGUOUS` rubric without failures marks it `AMBIGUOUS`, and `verdict.probability` equals the minimum probability across rubrics.
- **`fail_fast=True`**: Stops sequential evaluation immediately on the first `FAILED` rubric instead of scoring remaining criteria.
- `**options`: Forwards any `LogitGate` keyword argument (`pass_threshold`, `fail_threshold`, `backend`, `model`, `symmetric_debiasing`) to the shared evaluator.

[🔬 Production Use Cases for Multi-Rubric Checks](../use_cases.md)

---

## 6. `RankGate` & `BestOfNGate` (Candidate Ranking & Selection)

**Problem & deployment**: Selects the strongest completion from N sampled LLM outputs in code synthesis, RAG reranking, and Best-of-N decoding while filtering out unsafe or malformed drafts before ranking via `pre_gate`.

### Copy-Paste Recipe

```python
from logitgate import ASTGate, BestOfNGate, RankGate

candidates = [
    "def sort_items(items): return sorted(items)",
    "def sort_items(items: list[int]) -> list[int]: return sorted(items)",
    "def sort_items(items): import os; os.system('sort'); return items",
]

# Rank candidates from best to worst
ranker = RankGate("Rate implementation quality and static typing", mode="auto")
ranking = ranker.rank(candidates)
print(f"Top candidate: {ranking.winner}")
print(f"Order: {ranking.ranking}")

# Best-of-N selection with Layer 0 pre-execution safety filter
best_of_n = BestOfNGate(
    rubric="Pick the most production-ready implementation",
    pre_gate=ASTGate(syntax=True, forbidden_imports=["os", "subprocess"]),
)
selection = best_of_n.select(candidates)
print(f"Selected candidate: {selection.winner}")
```

---

## 7. Batch Evaluation (`evaluate_batch` & `aevaluate_batch`)

**Problem & deployment**: Evaluates hundreds or thousands of offline benchmark items or dataset rows in parallel with bounded concurrency, automatic prefix warmup, and per-item fault isolation (`on_error="abstain"`).

### Copy-Paste Recipe

```python
import asyncio
from logitgate import LogitGate

async def run_batch():
    gate = LogitGate("Does the text contain valid JSON?", pass_threshold=0.80)
    candidates = ['{"status": "ok"}', 'not json', '{"user": 1}']

    # Async batch evaluation with concurrency limit and error isolation
    verdicts = await gate.aevaluate_batch(
        candidates=candidates,
        max_concurrency=4,
        on_error="abstain",  # "raise" | "abstain" | "fail"
    )
    for c, v in zip(candidates, verdicts, strict=True):
        print(f"Candidate: {c[:20]}... -> Passed: {v.passed}")

asyncio.run(run_batch())
```

### Concurrency Guarantees

- **Bounded Fan-Out vs. Shared-Prefix Warmup**: Candidate batching (`aevaluate_batch(candidates=...)`) fans out concurrently across items bounded by `max_concurrency` (`asyncio.Semaphore`), whereas multi-rubric batching (`aevaluate_batch(candidate, rubrics)` and `MultiRubricGate`) evaluates rubric 0 first to populate the shared candidate prefix in KV-cache before fanning out the remaining rubrics.
- **Fault Isolation**: With `on_error="abstain"`, unexpected per-item exceptions isolate to `DecisionStatus.AMBIGUOUS` with the exception message recorded in `trace.metadata["abstained_reason"]`. `BaseException` subclasses (`asyncio.CancelledError`, `KeyboardInterrupt`) are never swallowed.

---

## 8. `CalibrationProfile` & `VerdictCache`

**Problem & deployment**: Bounds false-pass error rates via split-conformal calibration in regulated or high-stakes workflows, while `VerdictCache` eliminates redundant GPU calls on repeated CI fixtures and regression inputs with persistent SQLite caching (`~0.11-0.14 ms` cache hits).

### Copy-Paste Recipe

```python
from pathlib import Path
import tempfile
from logitgate import CalibrationProfile, LogitGate, VerdictCache

# 1. Define or load a calibrated profile (e.g. saved by fit_calibration_profile or CalibrationProfile.load(...))
profile = CalibrationProfile(
    model="qwen2.5:7b",
    binary_temperature=0.85,
    pass_threshold=0.88,
    fail_threshold=0.15,
    target_risk_alpha=0.05,
    guarantee_holds=True,
)

# 2. Attach cache and profile to gate
with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "verdicts.sqlite3"
    with VerdictCache(path=cache_path, max_entries=1024) as cache:
        gate = LogitGate(
            rubric="Is the code properly formatted?",
            profile=profile,
            cache=cache,
        )
        v1 = gate.evaluate("def foo(): pass")
        v2 = gate.evaluate("def foo(): pass")  # Cache hit (~0.11-0.14 ms)
        assert v2.traces[0].metadata.get("cache_hit") is True
```

[🔬 Conformal Calibration Proofs & Quantiles](../math_and_debiasing.md)

---

[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](../eli5/how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)
