# Quickstart Recipes (⚡ Quick Notes)

Copy-paste recipes, exact constructor signatures, default parameters, and failure modes for core LogitGate primitives.

---

## 1. `LogitGate` (Binary First-Token Classifier)

Evaluates whether a candidate string satisfies a natural language rubric using debiased first-token log-probabilities.

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
| `backend` | `str \| Backend \| None` | `None` | URI or instance (defaults to `LOGITGATE_BACKEND` or stub). |
| `model` | `str \| None` | `None` | Model override for the backend endpoint. |
| `symmetric_debiasing` | `bool` | `True` | 2-pass permutation canceling token and position priors. |
| `logit_temperature` | `float \| "auto" \| None` | `None` | Temperature scaling divisor ($T$) for sigmoid calibration. |
| `max_positional_bias` | `float` | `3.0` | Abstains if log-odds difference between passes exceeds this delta. |
| `on_unsupported_logprobs`| `Literal["raise", "escalate", "sidecar"]` | `"raise"` | Behavior when backend lacks logprobs. |
| `sidecar_backend` | `str \| Backend \| None` | `None` | Alternative backend when using sidecar mode. |

<details>
<summary>🌱 <b>ELI5 / Why this works</b></summary>

Standard LLM evaluators generate 200+ tokens of critique before saying yes or no. LogitGate stops immediately after the prompt prefill (`max_tokens=1`) and inspects the model's first-token logits. A 2-pass swap (Pass 1: A=Yes, B=No; Pass 2: A=No, B=Yes) analytically cancels the model's affirmative token bias and first-line position preference. See [How LogitGate Works](../eli5/how_logitgate_works.md).
</details>

[🔬 Mathematical Proofs & Log-Odds Cancellation](../math_and_debiasing.md)

---

## 2. `Cascade` (3-Layer Routing Pipeline)

Chains deterministic validation (`Layer 0`), fast logit gating (`Layer 1`), and full step-by-step diagnostic critique (`Layer 2`).

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
- `allow_fail_override=True`: Allows Tier-2 `JudgeGate` to adjudicate and rescue Tier-1 false negatives while keeping Tier-0 AST structural failures final.

[🔬 Systems Architecture & Cascade Lifecycle](../architecture.md)

---

## 3. `ChoiceGate` (Discrete Categorical Routing)

Routes input text across multiple discrete categories without generating tokens, using a fixed-point-free derangement permutation to cancel letter bias.

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
<summary>🌱 <b>ELI5 / Why this works</b></summary>

Models prefer option 'A' over 'B' simply because 'A' appears earlier in the alphabet and earlier in the prompt. By evaluating the question twice with rotated letter assignments, ChoiceGate removes letter favoritism entirely.
</details>

[🔬 Choice Debiasing Math](../math_and_debiasing.md)

---

## 4. `ScoreGate` (Ordinal & Continuous Rubric Scoring)

Computes continuous rubric scores on bounded numeric scales (e.g. 1 to 5) by pooling log-probabilities across both debiasing passes.

### Copy-Paste Recipe

```python
from logitgate import ScoreGate

scorer = ScoreGate(
    rubric="Rate the architectural code quality from 1 (disastrous) to 5 (flawless):",
    min_score=1,
    max_score=5,
    threshold=4.0,
    fail_threshold=2.5,
    max_score_std=1.25,
    symmetric_debiasing=True,
)

verdict = scorer.evaluate("def add(a: int, b: int) -> int: return a + b")
print(f"Expected Score: {verdict.score:.2f} / 5.00")
print(f"Passed threshold (>=4.0): {verdict.passed}")
```

### Key Parameters & Invariants

- Combines forward and reverse passes via **log-linear pooling** (geometric mean) in log-probability space before computing expected values. Never averages forward and reverse scores linearly, which distorts bimodal distributions.
- `max_score_std`: Abstains with `DecisionStatus.AMBIGUOUS` if the posterior distribution variance exceeds this cap (indicating a split or conflicted model).

---

## 5. `MultiRubricGate` (Shared-Prefix Multi-Criterion Checks)

Evaluates multiple distinct rubrics against a single candidate artifact in a single batch, maximizing KV-cache prefix hits across consecutive evaluations.

### Copy-Paste Recipe

```python
from logitgate import MultiRubricGate

rubrics = [
    "Does the code include parameter and return type annotations?",
    "Does the function handle potential None inputs safely?",
    "Does the function avoid catching bare Exception without re-raising?",
]

gate = MultiRubricGate(rubrics, aggregation="all")
candidate = "def get_user(uid: int | None) -> dict | None: return {'id': uid} if uid else None"

# Comprehensive pass/fail verdict
verdict = gate.evaluate(candidate)
print(f"Overall passed: {verdict.passed}")

# Individual breakdown per rubric
for rubric, v in zip(rubrics, gate.evaluate_each(candidate)):
    status = "PASS" if v.passed else "FAIL"
    print(f"[{status}] P={v.probability:.4f} | {rubric}")
```

### Aggregation Modes

- `aggregation="all"`: Passes only if every individual rubric passes.
- `aggregation="any"`: Passes if at least one individual rubric passes.
- `aggregation="mean"`: Passes if the average probability across rubrics exceeds the threshold.

[🔬 Production Use Cases for Multi-Rubric Checks](../use_cases.md)

---

## 6. `RankGate` & `BestOfNGate` (Candidate Ranking & Selection)

Ranks $N$ competing candidate completions or selects the highest-scoring candidate under a unified rubric.

### Copy-Paste Recipe

```python
from logitgate import BestOfNGate, RankGate

candidates = [
    "def sort_items(items): return sorted(items)",
    "def sort_items(items: list[int]) -> list[int]: return sorted(items)",
    "def sort_items(items): import os; os.system('sort'); return items",
]

# Rank candidates from best to worst
ranker = RankGate("Rate implementation quality and static typing", mode="auto")
ranking = ranker.rank(candidates)
print(f"Top candidate: {ranking.ranked_candidates[0]}")

# Best-of-N selection with pre-execution safety filter
best_of_n = BestOfNGate(
    rubric="Pick the most production-ready implementation",
    pre_gate=None,
)
selection = best_of_n.evaluate(candidates)
print(f"Selected candidate: {selection.candidate}")
```

---

## 7. Batch Evaluation & Fault Isolation

Executes batches of evaluations synchronously or asynchronously with concurrency bounds and fault isolation.

### Copy-Paste Recipe

```python
import asyncio
from logitgate import LogitGate

async def run_batch():
    gate = LogitGate("Does the text contain valid JSON?", pass_threshold=0.80)
    candidates = ['{"status": "ok"}', 'not json', '{"user": 1}']

    # Async batch evaluation with concurrency limit and error isolation
    verdicts = await gate.aevaluate_batch(
        candidates,
        max_concurrency=4,
        on_error="abstain",  # "raise" | "abstain" | "fail"
    )
    for c, v in zip(candidates, verdicts):
        print(f"Candidate: {c[:20]}... -> Passed: {v.passed}")

asyncio.run(run_batch())
```

### Concurrency Guarantees

- **KV-Cache Warmup**: Batch evaluation automatically warms the shared prompt prefix on candidate 0 before fanning out concurrent requests.
- **Fault Isolation**: With `on_error="abstain"`, unexpected per-item exceptions isolate to `DecisionStatus.AMBIGUOUS` with the exception message in `trace.metadata["error"]`. `BaseException` subclasses (`asyncio.CancelledError`, `KeyboardInterrupt`) are never swallowed.

---

## 8. `CalibrationProfile` & `VerdictCache`

Applies empirical temperature scaling and split-conformal coverage guarantees, and accelerates repetitive checks with persistent SQLite caching.

### Copy-Paste Recipe

```python
from pathlib import Path
import tempfile
from logitgate import CalibrationProfile, LogitGate, VerdictCache

# 1. Generate split-conformal calibration profile from holdout data
samples = [(2.5, 2.3, 1) for _ in range(25)] + [(-2.5, -2.3, 0) for _ in range(25)]
profile = CalibrationProfile.from_log_odds_samples(
    samples,
    target_risk_alpha=0.05,
    model="qwen2.5:7b",
)

# 2. Attach cache and profile to gate
with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "verdicts.sqlite3"
    with VerdictCache(path=cache_path, max_entries=1024) as cache:
        gate = LogitGate(
            rubric="Is the code properly formatted?",
            cache=cache,
        )
        v1 = gate.evaluate("def foo(): pass")
        v2 = gate.evaluate("def foo(): pass")  # Cache hit (<0.05ms)
        assert v2.traces[0].metadata.get("cache_hit") is True
```

[🔬 Conformal Calibration Proofs & Quantiles](../math_and_debiasing.md)
