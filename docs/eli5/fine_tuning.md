[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [⚡ Quick Notes](../quick/quickstart_recipes.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](./how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)

# How LoRA Fine-Tuning Works (Tier B)

> **An illustrated walkthrough of baking domain rubrics into a compact 15-50 MB plug-in adapter using balanced first-token drills, ready for Ollama, vLLM, or llama.cpp.**

---

## Table of Contents

1. [When Reference Examples Aren't Enough](#1-when-reference-examples-arent-enough)
2. [Plug-In Adapters Instead of Retraining Everything](#2-plug-in-adapters-instead-of-retraining-everything)
3. [Balanced First-Token Drills: Preventing Letter Habits](#3-balanced-first-token-drills-preventing-letter-habits)
4. [Training Once and Deploying Across Runtimes](#4-training-once-and-deploying-across-runtimes)
5. [Choosing Between Standard LogitGate, Tier A, and Tier B](#5-choosing-between-standard-logitgate-tier-a-and-tier-b)

---

## 1. When Reference Examples Aren't Enough

Weight-free distillation ([Tier A](./distillation.md)) gets you far with zero GPU training by attaching a matched Pass/Fail example pair to the prompt. On high-volume, fixed-domain GPU services, however, prepending reference examples on every request still uses prompt tokens and leaves subtle domain nuances outside the model's own weights.

### From Open-Book Cheat Sheet to Trained Instincts

If Tier A is keeping a worked reference sheet on your desk, Tier B is running short, targeted practice drills so the model internalizes your grading boundary directly. At runtime, the model no longer needs extra reference examples in the prompt: it reads your rubric and candidate, and scores Token 1 with zero anchor overhead.

![LoRA Fine-Tuning (Tier B)](../assets/tier_b_lora.svg)

### What Tier B Produces

1. **Frozen base model**: Your 3B open-weights model (`Llama 3.2`, `Qwen 2.5`, `Gemma`, or `Mistral`) keeps its original ~6 GB weights untouched.
2. **Compact plug-in adapter**: Training produces only a `15-50 MB` add-on file (**LoRA adapter**, short for Low-Rank Adaptation) containing the learned rubric adjustments.
3. **Ready-to-run server configs**: LogitGate automatically writes deployment scripts for Ollama, vLLM, and llama.cpp right alongside the adapter.

**Where to use it:** High-throughput, fixed-domain evaluation pipelines running on dedicated GPUs where you want zero extra prompt tokens and tight domain accuracy from a compact 3B model. *(Skip it if standard `LogitGate` or a `<50 KB` Tier-A JSON bundle already meets your latency and escalation targets.)*

---

## 2. Plug-In Adapters Instead of Retraining Everything

Updating all 3 billion parameters of a language model (**full fine-tuning**) demands heavy GPU memory, risks erasing general reading ability, and forces you to store a separate ~6 GB model copy for every task.

### Clip-On Lenses Over a Shared Camera Body

Instead of rebuilding the entire camera body for each assignment, **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** freezes the base model and inserts tiny trainable correction layers across the model's linear projections (`rank=8`). You train a fraction of 1% of the weights, keep the model's core language fluency intact, and swap lightweight `15-50 MB` adapters on top of one shared base model.

### How `all-linear` Transfer Works Across Model Families

1. **Freeze the base weights**: Original model parameters stay read-only during training, cutting GPU memory and preventing catastrophic forgetting.
2. **Attach low-rank updates to every linear layer**: Targeting all linear layers automatically adapts across `Llama 3`, `Qwen 2.5`, `Gemma`, and `Mistral` architectures without hand-picking layer names.
3. **Save only the delta**: The output directory stores only the small adapter weights (`15-50 MB`), never a duplicated 6 GB checkpoint.

**Where to use it:** Teams serving multiple evaluation rubrics or products from one shared base model on vLLM or Ollama.

---

## 3. Balanced First-Token Drills: Preventing Letter Habits

Standard instruction fine-tuning (**Supervised Fine-Tuning**, or **SFT**) on yes/no data has a hidden trap: if `"A"` means Pass on every training row, the fine-tuned model quickly memorizes `"output A"` as a shortcut and wastes training capacity predicting the prompt text instead of the verdict.

### Two Mirrored Drills Per Example, Scored Only on Token 1

LogitGate solves both problems by turning every historical review into a balanced pair of single-token drills (**2x symmetric masked first-token training**):

1. **Mirror every label assignment**: Each labeled example generates two rows: **Drill 1 (Forward)** where `A = Pass, B = Fail`, and **Drill 2 (Swapped)** where `B = Pass, A = Fail`. Because `"A"` and `"B"` appear equally often as the right answer, the adapter cannot cheat by favoring one letter.
2. **Mask the prompt tokens**: Every token in the rubric, context, and candidate receives an ignore label (`-100`) during loss computation (**prompt token masking**).
3. **Optimize exclusively for Token 1**: Only the single verdict position (`A` or `B`) updates the adapter weights, concentrating 100% of the training signal on exact first-token discrimination.

**Where to use it:** Automatically enforced whenever you train via `train_distillation_lora` or `logitgate distill train-lora`.

---

## 4. Training Once and Deploying Across Runtimes

Training an adapter is only useful if it plugs straight into your serving stack without manual glue scripts.

### Three-Step Workflow

1. **Install optional training extras**: Install `pip install "logitgate[distill]"` (PyTorch, Transformers, and PEFT) on your training machine; serving machines do not need these packages.
2. **Run training on telemetry or a labeled dataset**: Pass your historical reviews and base HuggingFace model ID to `train_distillation_lora` (or the CLI).
3. **Deploy with the generated manifest**: LogitGate emits `Modelfile.ollama` (Ollama), `serve_vllm.sh` (vLLM `--enable-lora`), and `serve_llamacpp.sh` (llama.cpp `--lora`) inside your output folder.

<details>
<summary><b>Code Example &amp; CLI Command</b></summary>

**1. Train in Python:**

```python
from logitgate.distill import LoRADistillConfig, train_distillation_lora

config = LoRADistillConfig(
    rank=8,
    lora_alpha=16,
    learning_rate=2e-4,
    epochs=3,
    batch_size=4,
    target_modules="all-linear",
)

manifest = train_distillation_lora(
    dataset="benchmarks/data/cal_distill_80.json",
    base_model="meta-llama/Llama-3.2-3B-Instruct",
    output_dir="distill_output/llama3_lora",
    config=config,
)

print(
    f"Trained {manifest.num_training_examples} symmetric examples. "
    f"Loss: {manifest.initial_loss:.4f} -> {manifest.final_loss:.4f}"
)
# Generated deployment files:
# - distill_output/llama3_lora/Modelfile.ollama
# - distill_output/llama3_lora/serve_vllm.sh
# - distill_output/llama3_lora/serve_llamacpp.sh
```

**2. Or train from the CLI:**

```bash
logitgate distill train-lora --dataset benchmarks/data/cal_distill_80.json --base-model meta-llama/Llama-3.2-3B-Instruct -o distill_output/llama3_lora
```

</details>

---

## 5. Choosing Between Standard LogitGate, Tier A, and Tier B

| Approach | Training Hardware | Artifact Size | Runtime Anchor Tokens | Best Fit |
|:---|:---|:---:|:---:|:---|
| **Standard `LogitGate` + `Cascade`** | None | `0 KB` | `0` | ~90% of production workloads |
| **[Tier-A Weight-Free Bundle](./distillation.md)** | CPU (`<20 ms`) | `<50 KB` JSON | `1` matched pair (in-domain) | Edge/CPU 1B-3B models or zero-GPU recalibration from logs |
| **Tier-B LoRA Adapter** | GPU (short SFT run) | `15-50 MB` | `0` | Dedicated high-volume GPU traffic with fixed domain rubrics |

---

## Next Steps

- [🧭 Visual Guide: How Weight-Free Distillation Works (Tier A)](./distillation.md): Improve compact models in under 20 ms on CPU without GPU training.
- [🧭 Visual Guide: How LogitGate Works](./how_logitgate_works.md): Foundational walkthrough of first-token scoring, 3-stage triage, and bias cancellation.
- [🚀 Deep Power-User Guide: Distillation & PEFT LoRA](../advanced/distillation_and_lora.md): Full training config reference, architecture transfer details, and deployment manifests.

---

[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [⚡ Quick Notes](../quick/quickstart_recipes.md) · [🔬 Deep Architecture](../architecture.md) · [🧭 Visual Guide](./how_logitgate_works.md) · [🚀 Power-User](../advanced/distillation_and_lora.md) · [📊 Benchmarks](../benchmarks.md)
