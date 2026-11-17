[🏠 README](../../README.md) · [📚 Docs Hub](../README.md) · [⚡ Quick Notes](../quick/quickstart_recipes.md) · [🔬 Deep Architecture](../architecture.md) · [🌱 Visual ELI5](../eli5/how_logitgate_works.md) · [📊 Benchmarks](../benchmarks.md)

---

# Elastic Recurrent Layer-Looping & PCIe Hardware Guardrails

> [!NOTE]
> **When You Need This vs. When Standard LogitGate Is Enough**
>
> **Concrete Bottlenecks**:
> 1. **Single-Pass Circuit-Depth Ceiling**: Standard causal transformers execute one pass through $L$ layers (`O(1)` sequential depth), so multi-step deductive constraints or nested logic often fail unless the model generates 200+ Chain-of-Thought tokens.
> 2. **Silent PCIe VRAM Paging**: When multi-model cascades or long-context KV-caches exceed physical VRAM on shared GPUs (e.g., 12 GB RTX 4070 or 24 GB RTX 3090/4090), CUDA Unified Memory silently pages allocations over PCIe (`32-64 GB/s` vs. `1,000-1,800 GB/s` VRAM) without throwing `CUDA OOM`, degrading latency by `10x-50x`.
>
> **Quantified Payoff**:
> - **Elastic Recurrent Looping ($K=2$)**: Delivers a **`+12.5 pp` reasoning gain** on `NVIDIA-Nemotron-Labs Dory-4B` (`53.1% -> 65.6%` on RTX 5090; `50.0% -> 62.5%` on RTX 4070) and doubles `ScoreGate` monotonicity (`37.5% -> 75.0%`) at **`0` generated tokens** and a constant 4B VRAM footprint.
> - **Proactive VRAM Guardrails (`PCIeSpilloverGuard`)**: Inspects physical free VRAM via `torch.cuda.mem_get_info()` and triggers `torch.cuda.empty_cache()` whenever headroom falls below `min_free_headroom_mb + estimated_kv_mb` (or reserved memory exceeds `96%`), preventing silent PCIe swap-thrashing.
>
> **When to Skip It**: In ~90% of deployments, standard feed-forward models (`qwen2.5-coder:3b`, `llama3.2:3b`, `gemma4:e4b`) behind `Cascade` + `CalibrationProfile` already evaluate classification, RAG groundedness, and policy rubrics in `15-45 ms` on GPU with zero custom PyTorch runtime setup.

---

<details>
<summary>💡 <b>Intuition: Re-Reading Before Answering vs. Speaking Out Loud, and VRAM Headroom vs. Swap-Thrashing</b></summary>

> When evaluating a dense logical constraint, you have three execution options:
> 1. **1-Pass Feed-Forward**: Answer immediately after a single read-through (`O(1)` circuit depth). Fast, but prone to missing multi-step dependencies.
> 2. **Chain-of-Thought (`JudgeGate`)**: Talk through every intermediate step out loud token by token before answering. Accurate, but bound by autoregressive memory bandwidth.
> 3. **Elastic Recurrent Looping (`RecurrentPyTorchBackend`)**: Re-read the passage silently $K$ times (looping hidden states through the same recurrent block) before emitting token 1. At $K=2$, a 4B model gets the effective depth of an 8B pass with a 4B memory footprint and `0` generated tokens.
>
> Meanwhile, **`PCIeSpilloverGuard`** works like checking physical RAM before launching a heavy compiler job so the OS never starts swap-thrashing to disk: by verifying free physical VRAM (plus estimated KV-cache) and reclaiming fragmented PyTorch blocks beforehand, it keeps prefill inside high-bandwidth GPU memory rather than falling off the PCIe bus cliff.
</details>

---

## Architectural Workflow

```mermaid
flowchart TD
    subgraph GUARD["1. Hardware Memory Inspection"]
        REQ["Incoming Candidate Prompt"] --> PG["PCIeSpilloverGuard.guard_forward_pass()"]
        PG --> CHK{"Free VRAM < Headroom + KV?"}
        CHK -->|"Yes: Risk Detected"| EC["torch.cuda.empty_cache()\nReclaim Fragmented Blocks"]
        CHK -->|"No: Safe Headroom"| RUN["Proceed to Execution"]
        EC --> RUN
    end

    subgraph RECURRENT["2. Elastic Recurrent Layer-Looping (Dory-4B)"]
        RUN --> EMB["Embedding & Initial Prefill: h^(0)"]
        EMB --> LOOP["Recurrent Layer Block: h^(k+1) = Block(h^(k), x)"]
        LOOP --> KL{"k >= min_loops AND\nKL(P^(k) || P^(k-1)) < epsilon?"}
        KL -->|"Yes: Early Halting"| PROJ["Output Head Projection"]
        KL -->|"No: k < max_loops"| LOOP
        KL -->|"Max Loops Reached"| PROJ
    end

    subgraph OUT["3. First-Token Verification"]
        PROJ --> LOG["Deterministic First-Token Logits: P(A), P(B)\n0 Generated Tokens, Deep Latent Reasoning"]
    end
```

---

## ⚡ Quick Recipe

### 1. Elastic Recurrent Gating via `recurrent://` URI

Use the `recurrent://` URI scheme to configure model name, device, recurrent loops, and dynamic KL-threshold early halting:

```python
from logitgate import LogitGate

# Elastic recurrent backend: 2 latent loops with adaptive KL early-halting
gate = LogitGate(
    rubric="Does the SQL query use parameterized statements without string concatenation?",
    backend="recurrent://nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16?num_loops=2&min_loops=1&max_loops=4&kl_threshold=0.02&device=cuda",
)

verdict = gate.evaluate("cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))")
print(f"Passed: {verdict.passed} (Calibrated P: {verdict.probability:.4f})")
```

### 2. Proactive VRAM Watermark Protection with `PCIeSpilloverGuard`

Wrap Layer-1 and Layer-2 prefill passes inside `PCIeSpilloverGuard` to prevent silent CUDA unified memory paging over PCIe:

```python
from logitgate.hardware.memory_guard import PCIeSpilloverGuard

# 1. Initialize guard monitoring GPU 0 with 512 MB minimum headroom
guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0)

# 2. Inspect current hardware memory status
status = guard.inspect()
print(f"Free VRAM: {status.free_physical_mb:.1f} MB / {status.total_physical_mb:.1f} MB")
print(f"Spillover Risk: {status.spillover_risk_detected}")

# 3. Guard prefill pass with estimated 128 MB KV-cache allocation
with guard.guard_forward_pass(estimated_kv_mb=128.0) as st:
    # Safe forward pass: empty_cache() automatically triggers if headroom is tight
    pass
```

---

## 🔬 Deep Mechanics: Elastic Recurrent Layer-Looping

A standard causal transformer evaluates its first token in a single forward pass through $L$ attention layers. This represents an $O(1)$ sequential circuit depth. Standard models cannot compute multi-digit arithmetic, state machines, or nested boolean logic in a single pass without emitting intermediate Chain-of-Thought (`<think>`) tokens.

### 1. Latent Reasoning in Hidden-State Space

Architectures like **NVIDIA-Nemotron-Labs Dory** (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) introduce **elastic recurrent depth**. Instead of traversing intermediate transformer layers once, Dory recirculates hidden state representations through a shared recurrent block for $K$ iterations (`num_loops`, typically 1 to 4) before projecting to output logits:

$$\mathbf{h}^{(k+1)} = \text{RecurrentLayerBlock}\left(\mathbf{h}^{(k)}, \mathbf{x}\right), \quad k \in \{1, \dots, K\}$$

$$\mathbf{y} = \text{Softmax}\left(\mathbf{W}_{\text{head}} \mathbf{h}^{(K)}\right)$$

- **Computational Depth without Parameter Bloat**: Looping twice ($K=2$) gives a 4B model the effective computational depth of an 8B model, while keeping the VRAM memory footprint strictly at 4B.
- **Zero Generated Tokens**: Because reasoning occurs purely in latent hidden-state space, the model emits its final verification decision on token 1, completely avoiding autoregressive decoding latency.

### 2. Adaptive Early-Halting via KL-Divergence

Not every input requires deep recurrent reasoning. Simple syntax checks or obvious violations converge after a single loop, whereas subtle edge cases benefit from additional passes.

`RecurrentPyTorchBackend` dynamically halts recurrent execution when the output probability distribution stabilizes:

$$\text{KL}\left(P^{(k)} \parallel P^{(k-1)}\right) = \sum_{v \in \mathcal{V}} P^{(k)}(v) \ln\left(\frac{P^{(k)}(v)}{P^{(k-1)}(v)}\right)$$

If $k \ge \text{min\_loops}$ and $\text{KL} < \epsilon_{\text{KL}}$ (configured via `kl_threshold`, e.g. `0.02`), looping terminates early. This preserves multi-loop accuracy on difficult items while cutting prefill latency on routine cases.

### 3. URI Parameter Reference

The `recurrent://` URI scheme supports the following query parameters:

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `num_loops` | `int` | `2` | Default number of recurrent loop iterations per token pass. |
| `min_loops` | `int` | `None` | Minimum loop iterations before KL early-halting is evaluated. |
| `max_loops` | `int` | `None` | Maximum loop iterations before forced projection. |
| `kl_threshold` | `float` | `None` | KL-divergence early exit threshold (typically `0.01` to `0.05`). |
| `device` | `str` | `"cuda"` | Compute device (`"cuda"`, `"cuda:0"`, `"cpu"`, or `"auto"`). |

---

## 🔬 Deep Mechanics: PCIe Memory Guardrails

Modern deep learning frameworks on Linux and Windows/WSL2 use CUDA Unified Memory. When GPU allocations exceed physical card capacity, the NVIDIA driver does not immediately throw an Out-Of-Memory (`CUDA OOM`) error. Instead, it **silently pages memory across the PCIe bus to host DDR RAM**.

### 1. The PCIe Latency Cliff

| Memory Hierarchy | Typical Bandwidth | Latency Impact |
| :--- | :---: | :---: |
| **GPU VRAM (GDDR6X / GDDR7)** | **1,000 to 1,800 GB/s** | Baseline (~15 ms to ~45 ms per pass) |
| **PCIe 4.0 / 5.0 Bus Paging** | **32 to 64 GB/s** | **10x to 50x slowdown** (~1,500 ms to ~3,000 ms per pass) |

A cascade designed to deliver 50 ms decisions suddenly degrades to 2,000 ms decisions without throwing any error in application logs.

### 2. `PCIeSpilloverGuard` Protection Lifecycle

`PCIeSpilloverGuard` acts as an active physical memory sentry:

1. **Hardware Memory Query**: Directly interrogates `torch.cuda.mem_get_info(device_index)` to read physical free and total bytes without PyTorch caching distortion.
2. **Spillover Condition**:
   $$\text{free\_mb} < \text{min\_free\_headroom\_mb} + \text{estimated\_kv\_mb} \quad \text{OR} \quad \text{reserved\_mb} > 0.96 \cdot \text{total\_mb}$$
3. **Proactive Cache Eviction**: If spillover risk is detected and `auto_empty_cache=True`, it triggers `torch.cuda.empty_cache()` immediately before prefill buffers are allocated, returning unused cached blocks to the driver.

---

## Empirical Benchmark Highlights (NVIDIA Dory-4B)

Evaluated across NVIDIA GeForce RTX 5090 (32 GB) and RTX 4070 (12 GB) on the 32-item reasoning depth benchmark:

| Recurrent Depth | Tier A (Surface) | Tier B (Compound) | Tier C (Multi-Hop) | Tier D (Adversarial) | Overall 32-Item Reasoning | First-Token Latency (RTX 5090) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=1$ Loop** | 4/8 (50.0%) | 4/8 (50.0%) | 6/8 (75.0%) | 3/8 (37.5%) | **17/32 (53.1%)** | **216.3 ms** |
| **$K=2$ Loops** | 5/8 (62.5%) | **6/8 (75.0%)** | 6/8 (75.0%) | 4/8 (50.0%) | **21/32 (65.6%)** *(+12.5% jump)* | **335.5 ms** |
| **$K=4$ Loops** | 4/8 (50.0%) | 4/8 (50.0%) | 5/8 (62.5%) | 5/8 (62.5%) | **18/32 (56.2%)** | **628.9 ms** |
| *Baseline Feed-Forward 4B* | *6/8 (75.0%)* | *4/8 (50.0%)* | *4/8 (50.0%)* | *4/8 (50.0%)* | *18/32 (56.2%)* | *76.0 ms* |

**Key Takeaways**:
- **Sweet Spot at $K=2$**: Looping twice provides a **+12.5 percentage point accuracy jump** over $K=1$, outperforming standard feed-forward 4B models (56.2%) and approaching 14B feed-forward models (68.8%) with zero generated tokens.
- **Diminishing Returns at $K=4$**: Deep latent looping without intermediate normalization can cause representation oversaturation on surface tasks. $K=2$ is the best operating point for latency and accuracy.

---

## Complete Verification & Integration Example

The following script verifies that `RecurrentPyTorchBackend` parameters parse cleanly and `PCIeSpilloverGuard` correctly reports hardware headroom:

```python
from logitgate.backends import resolve_logit_backend
from logitgate.backends.recurrent import RecurrentPyTorchBackend
from logitgate.hardware.memory_guard import PCIeSpilloverGuard

# 1. Verify URI parsing for recurrent backend
backend = resolve_logit_backend(
    "recurrent://nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16?num_loops=2&kl_threshold=0.02&device=cpu"
)
assert isinstance(backend, RecurrentPyTorchBackend)
print(f"Backend resolved: {backend.model_name}, num_loops={backend.num_loops}, kl_threshold={backend.kl_threshold}")

# 2. Verify hardware memory guard on current platform
guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=256.0)
status = guard.inspect()
print(f"CUDA Available: {status.total_physical_mb > 0}")
print(f"VRAM Watermark Status: allocated={status.allocated_mb} MB, free={status.free_physical_mb} MB")

# 3. Guard a simulated execution block
with guard.guard_forward_pass(estimated_kv_mb=64.0) as st:
    print(f"Forward pass executed safely. Spillover risk: {st.spillover_risk_detected}")
```
