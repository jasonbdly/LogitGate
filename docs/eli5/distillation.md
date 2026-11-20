[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [⚡ Quick Notes](../quick/quickstart_recipes.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](./how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)

# How Weight-Free Distillation Works (Tier A)

> **An illustrated walkthrough of teaching a compact 1B-4B model from past Stage-2 reviews in under 20 ms on CPU, without changing model weights or running GPU training.**

---

## Table of Contents

1. [Why Compact Models Struggle Alone (and When You Need This)](#1-why-compact-models-struggle-alone-and-when-you-need-this)
2. [Worked Reference Pairs & Topic Overlap Filtering](#2-worked-reference-pairs--topic-overlap-filtering)
3. [Straightening Harsh or Lenient Scoring Habits](#3-straightening-harsh-or-lenient-scoring-habits)
4. [Compiling a Portable Bundle From Production Logs](#4-compiling-a-portable-bundle-from-production-logs)
5. [Choosing Between Standard LogitGate, Tier A, and Tier B](#5-choosing-between-standard-logitgate-tier-a-and-tier-b)

---

## 1. Why Compact Models Struggle Alone (and When You Need This)

Running a 1B to 3B model on a laptop, CPU runner, or small GPU keeps Stage-1 checks fast and inexpensive. On nuanced domain rules, however, smaller models tend to hit two predictable limits:

- **Uncertain edge cases**: Without seeing what a borderline Pass vs. Fail looks like for your specific rule, the small model hesitates and escalates too often to a slower, larger reviewer.
- **Harsh or lenient scoring habits**: Even after two-pass label swapping cancels `"A"` vs. `"B"` bias, a compact model can still lean systematically too skeptical (rejecting valid borderline outputs) or too lenient (**polarity skew**).

### Open-Book Reference Sheet Plus a Calibrated Dial

You do not always need GPU training to transfer judgment from a larger reviewer (**teacher**) to a compact model (**student**). Think of helping a new reviewer get up to speed on a policy without retraining them from scratch: you hand them **one relevant worked Pass/Fail pair** from past reviews (**distillation**) and adjust their scoring dial to correct any systematic tendency to grade too harshly or too softly.

![Weight-Free Distillation (Tier A)](../assets/tier_a_distillation.svg)

### What Happens Without Touching Model Weights

1. **Collect past decisions automatically**: As your pipeline runs, Stage-2 reviews and calibrated checks write compact pass/fail logs (**telemetry**) to disk.
2. **Compile a `<50 KB` JSON bundle on CPU**: In under `20 ms` with zero extra AI calls, LogitGate extracts a small bank of contrastive Pass/Fail pairs and fits a tiny `<1 KB` score-correction table.
3. **Load the bundle into any runtime**: Because the model's neural weights never change, the same JSON file works on Ollama, vLLM, llama.cpp, or pure CPU.

**Where to use it:** Use Tier-A distillation when running 1B-3B models on CPU or edge hardware, or when you want zero-GPU recalibration directly from production logs. *(Skip it if a standard 3B-4B `Cascade` with calibrated cutoffs already meets your escalation and accuracy targets.)*

---

## 2. Worked Reference Pairs & Topic Overlap Filtering

Showing a compact model a few examples inside the prompt helps it recognize subtle boundary cases, but blind example injection creates two hazards: showing a coding example when grading a refund policy confuses the model, and showing the exact same draft it is currently grading creates circular feedback.

### An Open-Book Index With a Relevance Gate

Instead of stuffing fixed examples into every prompt, LogitGate treats past reviews like a tabbed reference folder. It pulls only **one matched Pass + Fail pair** whose wording closely mirrors the current rule, checks whether the topic genuinely overlaps, and skips the folder altogether when a new topic arrives.

### How Reference Retrieval Stays Safe and Fast

1. **Pair contrasting past reviews**: LogitGate matches historical passing and failing items that look structurally similar (**contrastive pairs**) and trims any `"Option A"` / `"Option B"` wording from teacher notes so the examples stay label-neutral.
2. **Check topic overlap before injecting**: Before adding an example pair to the prompt, LogitGate measures what fraction of the rubric's vocabulary overlaps with the saved pair (**overlap gating**, defaulting to at least `25%`). If the incoming rule is on a new topic (**out-of-domain**), the anchor is skipped cleanly so off-topic examples never pollute the check.
3. **Exclude exact matches & preserve GPU memory**: Any saved item matching the current candidate's text fingerprint (**hash**) is excluded automatically, and selected examples sit ahead of the `A`/`B` decision block so Pass 1 and Pass 2 still reuse 100% of the cached prompt prefix.

**Where to use it:** Multi-domain verification pipelines where small models need crisp examples on known rubrics without breaking when an unfamiliar rubric appears.

---

## 3. Straightening Harsh or Lenient Scoring Habits

Even with good reference examples, a compact model's raw confidence scores can sit off-center: on hard valid examples, a skeptical 3B model might output `P = 0.40` even when the evidence leans toward passing.

### Re-Zeroing a Skewed Sensor From Five Signals

Rather than trusting a single raw score, LogitGate reads **five lightweight signals** already produced during the normal two-pass check:

1. **Combined `A` vs. `B` balance**: The debiased score on a symmetric `+/-` scale (**log-odds**) across Pass 1 and Pass 2.
2. **Direction of label bias**: Whether the model leaned toward `A` or `B` between the two passes.
3. **Size of label bias**: How strongly the model's answer shifted when the labels swapped.
4. **Spread of next-word odds**: How scattered the top candidate probabilities were (**entropy**).
5. **Leftover probability outside `A` and `B`**: How much weight leaked onto filler words (**residual mass**).

### How the `<1 KB` Calibrator Corrects Skew

1. **Fit a tiny correction head from past logs**: Using pure-Python math on CPU (no PyTorch, NumPy, or GPU needed), LogitGate fits a 5-weight logistic curve (`621-627 bytes` in JSON) that maps those five signals to true pass probabilities.
2. **Score live checks in `<0.01 ms`**: At runtime, multiplying five numbers and applying the curve takes microseconds, lifting `qwen2.5-coder:3b` out-of-domain hard-positive recall by **`+29.2 pp`** (`20.8% -> 50.0%`).
3. **Recompute safety cutoffs automatically**: Compilation also updates the data-calibrated (**conformal**) pass/fail thresholds so Stage-1 error stays within your target budget (such as `10%`, `alpha = 0.10`).

**Where to use it:** Whenever a compact model under-scores subtle valid outputs or over-escalates borderline items to Stage 2.

---

## 4. Compiling a Portable Bundle From Production Logs

Compiling a Tier-A bundle requires neither retraining infrastructure nor extra inference spend: you can build it directly from existing telemetry logs or from a labeled holdout dataset.

### Three-Step Workflow

1. **Compile from telemetry (or a labeled dataset)**: Point `compile_distillation_bundle` (or `logitgate distill compile`) at your JSONL telemetry file.
2. **Save the `<50 KB` artifact**: Store the resulting JSON bundle in version control or your deployment artifact store alongside your service config.
3. **Attach to `LogitGate`**: Pass the bundle path once when constructing your gate; anchor retrieval, topic-overlap filtering, and 5-signal calibration run automatically on every check.

<details>
<summary><b>Code Example &amp; CLI Command</b></summary>

**1. Compile and run in Python:**

```python
from pathlib import Path
from logitgate import LogitGate
from logitgate.distill import compile_distillation_bundle

# Compile anchor bank + 5-signal meta-calibrator + conformal thresholds from logs (0 AI calls)
bundle = compile_distillation_bundle(
    telemetry="telemetry.jsonl",
    student_model="llama3.2:3b",
    max_bank_pairs=16,
    target_risk_alpha=0.10,
)

bundle_path = Path("artifacts/llama3b_distill_bundle.json")
bundle.save(bundle_path)

# Attach the portable bundle to an untouched compact model
gate = LogitGate(
    rubric="Does the SQL query use parameterized statements without raw string interpolation?",
    backend="ollama://llama3.2:3b",
    distillation_bundle=bundle_path,
)

verdict = gate.evaluate("cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))")
prob_str = f"{verdict.probability:.4f}" if verdict.probability is not None else "N/A"
print(f"Passed: {verdict.passed} (Calibrated P: {prob_str})")
```

**2. Or compile from the CLI:**

```bash
logitgate distill compile --telemetry .logitgate/telemetry.jsonl --model llama3.2:3b -o bundle.json
```

</details>

---

## 5. Choosing Between Standard LogitGate, Tier A, and Tier B

| Approach | Training Needed | Artifact Size | Prompt Overhead | Best Fit |
|:---|:---|:---:|:---:|:---|
| **Standard `LogitGate` + `Cascade`** | None | `0 KB` | `0` extra tokens | ~90% of workloads using a 3B-14B model |
| **Tier-A Weight-Free Bundle** | `<20 ms` on CPU (0 GPU) | `<50 KB` JSON | `1` matched pair (when in-domain) | Compact 1B-3B edge/CPU models or zero-GPU log recalibration |
| **Tier-B LoRA Fine-Tuning** | Short GPU run | `15-50 MB` adapter | `0` extra tokens | High-volume fixed-domain GPUs needing maximum speed and zero prompt overhead |

---

## Next Steps

- [🧭 Visual Guide: How LoRA Fine-Tuning Works (Tier B)](./fine_tuning.md): When and how to bake your rubric directly into a lightweight adapter.
- [🧭 Visual Guide: How LogitGate Works](./how_logitgate_works.md): Foundational walkthrough of first-token scoring, 3-stage triage, and bias cancellation.
- [🚀 Deep Power-User Guide: Distillation & PEFT LoRA](../advanced/distillation_and_lora.md): Full Newton-Raphson equations, overlap coefficient proofs, and verification scripts.

---

[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [⚡ Quick Notes](../quick/quickstart_recipes.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](./how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)
