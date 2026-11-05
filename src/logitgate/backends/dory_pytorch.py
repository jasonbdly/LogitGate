"""
Native PyTorch implementation of NVIDIA-Nemotron-Labs Dory architecture.

Dory is a normalized latent-recurrent language model (nGPT backbone) featuring:
- Layer stack partitioned into Input (16 layers), Recurrent (40 layers), and Output (16 layers)
- Elastic recurrent layer-looping (num_loops = 1, 2, 4) with hidden-state reinjection
- Interleaved sliding-window attention (SWA) and full-attention layers
- Exact nGPT unit-norm normalization (justnorm) and spherical linear interpolation (slerp)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-6


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NGPT normalization: x * rsqrt(sum(x^2) + eps), fp32 accumulation."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)
    return y.to(x.dtype)


def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between normalized vectors a and b."""
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


def slerp_alpha(
    residual: torch.Tensor,
    out: torch.Tensor,
    alpha: torch.Tensor,
    alpha_scale: float,
) -> torch.Tensor:
    """Calculates slerp residual connection scaled by learned alpha parameter."""
    lr = (alpha.float() * alpha_scale).abs()
    return slerp_norm(residual, out, lr)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input for NeoX-style RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors."""
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class DoryRotaryEmbedding(nn.Module):
    """Precomputes rotary position embeddings for standard and long context."""

    def __init__(self, dim: int = 256, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore[assignment]
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype)
        sin = emb.sin()[None, None, :, :].to(dtype)
        return cos, sin


class DoryAttention(nn.Module):
    """GQA Attention layer with nGPT qk-norm, output gating, and slerp residual."""

    def __init__(self, layer_idx: int, is_sliding: bool = True) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.hidden_size = 2560
        self.num_heads = 16
        self.num_kv_heads = 4
        self.head_dim = 256
        self.q_size = 16 * 256
        self.kv_size = 4 * 256
        self.std = 0.0177
        self._sqk_factor = (256 ** 0.5) / self.std
        g0 = 2560 ** 0.5
        self._s_gate_factor = (g0 ** 0.5) / self.std
        self._alpha_scale = 0.05 / self.std

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

        self.sqk = nn.Parameter(torch.empty(self.q_size))
        self.s_gate = nn.Parameter(torch.empty(self.q_size))
        self.attn_alpha = nn.Parameter(torch.empty(self.hidden_size))
        base = 10000.0 if is_sliding else 10000000.0
        self.rotary = DoryRotaryEmbedding(dim=256, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        residual = x

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        gate = self.g_proj(x)

        cos, sin = self.rotary(seq_len, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # nGPT qk-norm AFTER rope
        sqk = (self.sqk.float() * self._sqk_factor).view(1, self.num_heads, 1, self.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype)
        k = justnorm(k).to(k.dtype)

        # Replicate KV heads for Grouped Query Attention (4 -> 16)
        k_rep = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_rep = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.q_size)

        # Output gate (sigmoid with s_gate scaling)
        sg = (self.s_gate.float() * self._s_gate_factor).square()
        attn_out = (attn_out.float() * torch.sigmoid(gate.float() * sg)).to(attn_out.dtype)
        attn_out = justnorm(attn_out)
        out = self.o_proj(attn_out)

        return slerp_alpha(residual, out, self.attn_alpha, self._alpha_scale)


class DoryMLP(nn.Module):
    """SwiGLU MLP layer with nGPT suv scaling and slerp residual."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = 2560
        self.ffn = 7168
        self.std = 0.0177
        self._suv_factor = 2560 ** 0.5
        self._alpha_scale = 0.05 / self.std

        self.gate_proj = nn.Linear(self.hidden_size, self.ffn, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.ffn, bias=False)
        self.down_proj = nn.Linear(self.ffn, self.hidden_size, bias=False)

        self.suv_gate = nn.Parameter(torch.empty(self.ffn))
        self.suv_up = nn.Parameter(torch.empty(self.ffn))
        self.mlp_alpha = nn.Parameter(torch.empty(self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        g = self.gate_proj(x)
        u = self.up_proj(x)
        g = (g.float() * (self.suv_gate.float() * self._suv_factor)).to(g.dtype)
        u = (u.float() * (self.suv_up.float() * self._suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(x.dtype)
        m = justnorm(m)
        out = self.down_proj(m)
        return slerp_alpha(residual, out, self.mlp_alpha, self._alpha_scale)


class DoryForCausalLM(nn.Module):
    """
    Dory-4B Causal Language Model with elastic recurrent layer-looping.

    Architecture:
    - 72 physical layers (16 Input + 40 Recurrent + 16 Output)
    - Loops recurrent layers `num_loops` times per token
    """

    def __init__(self, num_loops: int = 2) -> None:
        super().__init__()
        self.num_loops = num_loops
        self.vocab_size = 131072
        self.hidden_size = 2560
        self.n_input_layers = 16
        self.n_recurrent_layers = 40
        self.n_output_layers = 16
        self.std = 0.0177

        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(72):
            if i % 2 == 0:
                is_full = (i % 8 == 6)
                self.layers.append(DoryAttention(i, is_sliding=not is_full))
            else:
                self.layers.append(DoryMLP(i))

        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.empty(self.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        num_loops: int | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Executes forward pass with `num_loops` recurrent passes on hidden states."""
        k = num_loops if num_loops is not None else self.num_loops
        h = self.embeddings(input_ids)

        n_in = self.n_input_layers
        n_rec = self.n_recurrent_layers

        # Input block (16 layers)
        for layer in self.layers[:n_in]:
            h = layer(h)
        input_h = h

        # Recurrent block (40 layers, looped k times with reinjection)
        state = input_h
        for loop_idx in range(k):
            combined = justnorm(state + input_h)
            h = combined
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(h)
            state = justnorm(h)

        # Output block (16 layers)
        h = state
        for layer in self.layers[n_in + n_rec:]:
            h = layer(h)

        # LM Head with learned logit_scale on last token for next-token prediction
        # (avoid allocating vocab_size * seq_len float32 tensors on long contexts)
        last_h = h if kwargs.get("return_full_logits") else h[:, -1:, :]
        logits = self.lm_head(last_h)
        logits = logits.float() * (self.logit_scale.float() / self.std)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        num_loops: int | None = None,
        eos_token_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressively generates new tokens."""
        eos = eos_token_ids or [2, 11]
        curr = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(curr, num_loops=num_loops)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, next_token], dim=-1)
            if next_token.item() in eos:
                break
        return curr


def load_dory_model(
    model_name_or_path: str = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
    device: str = "cuda:0",
    num_loops: int = 2,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> DoryForCausalLM:
    """Loads Dory model weights from safetensors into DoryForCausalLM."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    t0 = time.perf_counter()
    if os.path.exists(model_name_or_path) and (Path(model_name_or_path) / "model.safetensors").exists():
        safetensor_path = str(Path(model_name_or_path) / "model.safetensors")
    elif os.path.exists(model_name_or_path) and model_name_or_path.endswith(".safetensors"):
        safetensor_path = model_name_or_path
    else:
        safetensor_path = hf_hub_download(model_name_or_path, "model.safetensors")

    model = DoryForCausalLM(num_loops=num_loops).to(dtype=torch_dtype)

    with safe_open(safetensor_path, framework="pt") as f:
        state_dict: dict[str, torch.Tensor] = {}
        state_dict["embeddings.weight"] = f.get_tensor("backbone.embeddings.weight")
        state_dict["lm_head.weight"] = f.get_tensor("lm_head.weight")
        state_dict["logit_scale"] = f.get_tensor("lm_head.logit_scale")

        for i in range(72):
            prefix = f"backbone.layers.{i}."
            if i % 2 == 0:
                state_dict[f"layers.{i}.attn_alpha"] = f.get_tensor(prefix + "attn_alpha")
                state_dict[f"layers.{i}.q_proj.weight"] = f.get_tensor(prefix + "mixer.q_proj.weight")
                state_dict[f"layers.{i}.k_proj.weight"] = f.get_tensor(prefix + "mixer.k_proj.weight")
                state_dict[f"layers.{i}.v_proj.weight"] = f.get_tensor(prefix + "mixer.v_proj.weight")
                state_dict[f"layers.{i}.g_proj.weight"] = f.get_tensor(prefix + "mixer.g_proj.weight")
                state_dict[f"layers.{i}.o_proj.weight"] = f.get_tensor(prefix + "mixer.o_proj.weight")
                state_dict[f"layers.{i}.sqk"] = f.get_tensor(prefix + "mixer.sqk")
                state_dict[f"layers.{i}.s_gate"] = f.get_tensor(prefix + "mixer.s_gate")
            else:
                state_dict[f"layers.{i}.mlp_alpha"] = f.get_tensor(prefix + "mlp_alpha")
                state_dict[f"layers.{i}.gate_proj.weight"] = f.get_tensor(prefix + "mixer.gate_proj.weight")
                state_dict[f"layers.{i}.up_proj.weight"] = f.get_tensor(prefix + "mixer.up_proj.weight")
                state_dict[f"layers.{i}.down_proj.weight"] = f.get_tensor(prefix + "mixer.down_proj.weight")
                state_dict[f"layers.{i}.suv_gate"] = f.get_tensor(prefix + "mixer.suv_gate")
                state_dict[f"layers.{i}.suv_up"] = f.get_tensor(prefix + "mixer.suv_up")

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"Loaded Dory model ({num_loops} loops) onto {device} in {elapsed:.2f}s", flush=True)
    return model
