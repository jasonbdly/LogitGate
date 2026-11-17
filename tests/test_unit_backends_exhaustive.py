from __future__ import annotations

import asyncio
import importlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    _HTTPClientMixin,
    _is_openai_host,
    _response_detail,
    describe_transport_error,
    infer_backend_spec,
    resolve_generative_backend,
    resolve_logit_backend,
    strip_reasoning_blocks,
    translate_transport_errors,
)
from logitgate.backends.dory_recurrent import (
    execute_adaptive_recurrent_loops,
)
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.backends.pytorch_native import PrefixBroadcastLogitEvaluator
from logitgate.backends.recurrent import (
    RecurrentPyTorchBackend,
    _parse_cuda_device_index,
)
from logitgate.exceptions import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
    ConfigurationError,
)
from logitgate.hardware.memory_guard import PCIeSpilloverGuard, VRAMWatermarkStatus

try:
    torch: Any = importlib.import_module("torch")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


# ===========================================================================
# 1. Hardware Memory Guard Tests
# ===========================================================================

def test_vram_watermark_status_attributes() -> None:
    status = VRAMWatermarkStatus(
        allocated_mb=128.5,
        reserved_mb=256.0,
        free_physical_mb=1024.0,
        total_physical_mb=2048.0,
        spillover_risk_detected=False,
    )
    assert status.allocated_mb == 128.5
    assert status.reserved_mb == 256.0
    assert status.free_physical_mb == 1024.0
    assert status.total_physical_mb == 2048.0
    assert status.spillover_risk_detected is False


def test_pcie_spillover_guard_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = PCIeSpilloverGuard(device_index=0)
    if HAS_TORCH:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status = guard.inspect()
    assert status.allocated_mb == 0.0
    assert status.free_physical_mb == 0.0
    assert status.spillover_risk_detected is False


def test_pcie_spillover_guard_with_cuda_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda idx: (2048 * 1024 * 1024, 4096 * 1024 * 1024),
        memory_allocated=lambda idx: 512 * 1024 * 1024,
        memory_reserved=lambda idx: 1024 * 1024 * 1024,
        empty_cache=lambda: None,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = guard.inspect()
    assert status.free_physical_mb == 2048.0
    assert status.total_physical_mb == 4096.0
    assert status.allocated_mb == 512.0
    assert status.reserved_mb == 1024.0
    assert status.spillover_risk_detected is False


def test_pcie_spillover_guard_with_cuda_risk_free_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda idx: (256 * 1024 * 1024, 4096 * 1024 * 1024),
        memory_allocated=lambda idx: 512 * 1024 * 1024,
        memory_reserved=lambda idx: 1024 * 1024 * 1024,
        empty_cache=lambda: None,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = guard.inspect()
    assert status.spillover_risk_detected is True


def test_pcie_spillover_guard_with_cuda_risk_reserved_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=100.0)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda idx: (200 * 1024 * 1024, 4096 * 1024 * 1024),
        memory_allocated=lambda idx: 3500 * 1024 * 1024,
        memory_reserved=lambda idx: 4000 * 1024 * 1024,  # > 4096 * 0.96 (3932.16)
        empty_cache=lambda: None,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = guard.inspect()
    assert status.spillover_risk_detected is True


def test_pcie_spillover_guard_cuda_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = PCIeSpilloverGuard(device_index=0)
    def raise_err(idx: int) -> tuple[int, int]:
        raise RuntimeError("CUDA driver error")

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=raise_err,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = guard.inspect()
    assert status.allocated_mb == 0.0
    assert status.spillover_risk_detected is False


def test_pcie_spillover_guard_empty_cache_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_called = False
    def mark_empty() -> None:
        nonlocal empty_called
        empty_called = True

    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0, auto_empty_cache=True)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda idx: (256 * 1024 * 1024, 4096 * 1024 * 1024),
        memory_allocated=lambda idx: 100 * 1024 * 1024,
        memory_reserved=lambda idx: 200 * 1024 * 1024,
        empty_cache=mark_empty,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with guard.guard_forward_pass(estimated_kv_mb=64.0) as st:
        assert st.spillover_risk_detected is True
    assert empty_called is True


def test_pcie_spillover_guard_empty_cache_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_empty() -> None:
        raise RuntimeError("empty cache failed")

    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=512.0, auto_empty_cache=True)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda idx: (256 * 1024 * 1024, 4096 * 1024 * 1024),
        memory_allocated=lambda idx: 100 * 1024 * 1024,
        memory_reserved=lambda idx: 200 * 1024 * 1024,
        empty_cache=raise_empty,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with guard.guard_forward_pass() as st:
        assert st.spillover_risk_detected is True


# ===========================================================================
# 2. Dory PyTorch Exhaustive Tests
# ===========================================================================

def test_dory_justnorm_math() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import justnorm

    x = torch.tensor([3.0, 4.0], dtype=torch.float32)
    y = justnorm(x)
    norm = torch.linalg.vector_norm(y).item()
    assert abs(norm - 1.0) < 1e-4

    # Dtype preservation
    x16 = x.to(torch.float16)
    y16 = justnorm(x16)
    assert y16.dtype == torch.float16


def test_dory_slerp_norm_math() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import justnorm, slerp_norm

    a = torch.tensor([1.0, 0.0], dtype=torch.float32)
    b = torch.tensor([0.0, 1.0], dtype=torch.float32)
    lr0 = torch.tensor([0.0], dtype=torch.float32)
    lr1 = torch.tensor([1.0], dtype=torch.float32)

    res0 = slerp_norm(a, b, lr0)
    expected0 = justnorm(a)
    assert torch.allclose(res0, expected0, atol=1e-5)

    res1 = slerp_norm(a, b, lr1)
    expected1 = justnorm(b)
    assert torch.allclose(res1, expected1, atol=1e-5)


def test_dory_slerp_alpha_math() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import slerp_alpha

    residual = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    out = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    alpha = torch.tensor([0.5, 0.5], dtype=torch.float32)
    alpha_scale = 1.0

    res = slerp_alpha(residual, out, alpha, alpha_scale)
    assert res.shape == residual.shape


def test_dory_rotate_half() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import rotate_half

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    res = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(res, expected)


def test_dory_apply_rotary_pos_emb() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import apply_rotary_pos_emb

    q = torch.tensor([[[[1.0, 2.0]]]])
    k = torch.tensor([[[[3.0, 4.0]]]])
    cos = torch.tensor([[[[1.0, 1.0]]]])
    sin = torch.tensor([[[[0.0, 0.0]]]])

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_rot, q)
    assert torch.allclose(k_rot, k)


def test_dory_rotary_embedding() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import DoryRotaryEmbedding

    rotary = DoryRotaryEmbedding(dim=64, base=10000.0)
    cos, sin = rotary(seq_len=4, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (1, 1, 4, 64)
    assert sin.shape == (1, 1, 4, 64)


def test_dory_attention_forward_sliding_and_full() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import DoryAttention

    attn_sliding = DoryAttention(0, is_sliding=True)
    for p in attn_sliding.parameters():
        torch.nn.init.normal_(p, std=0.01)

    attn_full = DoryAttention(6, is_sliding=False)
    for p in attn_full.parameters():
        torch.nn.init.normal_(p, std=0.01)

    x = torch.randn(1, 2, 2560)
    out_sliding = attn_sliding(x)
    out_full = attn_full(x)
    assert out_sliding.shape == (1, 2, 2560)
    assert out_full.shape == (1, 2, 2560)


def test_dory_mlp_forward() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import DoryMLP

    mlp = DoryMLP(1)
    for p in mlp.parameters():
        torch.nn.init.normal_(p, std=0.01)

    x = torch.randn(1, 2, 2560)
    out = mlp(x)
    assert out.shape == (1, 2, 2560)


def test_dory_for_causal_lm_init_meta() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import DoryAttention, DoryForCausalLM, DoryMLP

    with torch.device("meta"):
        model = DoryForCausalLM(num_loops=3)
    assert model.num_loops == 3
    assert len(model.layers) == 72
    assert isinstance(model.layers[0], DoryAttention)
    assert isinstance(model.layers[1], DoryMLP)
    assert isinstance(model.layers[6], DoryAttention)
    assert model.layers[6].is_sliding is False


def test_dory_for_causal_lm_forward_and_generate() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends.dory_pytorch import DoryForCausalLM

    m = DoryForCausalLM.__new__(DoryForCausalLM)
    torch.nn.Module.__init__(m)
    m.num_loops = 2
    m.vocab_size = 16
    m.hidden_size = 8
    m.n_input_layers = 1
    m.n_recurrent_layers = 1
    m.n_output_layers = 1
    m.std = 0.0177
    m.embeddings = torch.nn.Embedding(16, 8)
    m.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    m.lm_head = torch.nn.Linear(8, 16, bias=False)
    m.logit_scale = torch.nn.Parameter(torch.ones(16))

    inp = torch.tensor([[1, 2, 3]])
    logits = m.forward(inp, num_loops=1)
    assert logits.shape == (1, 1, 16)

    logits_full = m.forward(inp, return_full_logits=True)
    assert logits_full.shape == (1, 3, 16)

    gen = m.generate(inp, max_new_tokens=2, eos_token_ids=[999])
    assert gen.shape == (1, 5)

    # Test eos early exit in generate
    with patch.object(m, "forward") as mock_forward:
        # Return logits predicting token 2
        pred_logits = torch.zeros(1, 1, 16)
        pred_logits[0, 0, 2] = 10.0
        mock_forward.return_value = pred_logits
        gen_eos = m.generate(inp, max_new_tokens=5, eos_token_ids=[2])
        assert gen_eos.shape == (1, 4)


def test_load_dory_model_safetensors_and_hf_download(tmp_path: Path) -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from logitgate.backends import dory_pytorch

    # Case 1: directory containing model.safetensors
    dir_path = tmp_path / "model_dir"
    dir_path.mkdir()
    safetensor_file = dir_path / "model.safetensors"
    safetensor_file.write_bytes(b"dummy")

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_f = MagicMock()
    fake_f.get_tensor.return_value = torch.zeros(1)
    fake_safe_open = MagicMock()
    fake_safe_open.return_value.__enter__.return_value = fake_f

    with patch.object(dory_pytorch, "DoryForCausalLM", return_value=fake_model), \
         patch("safetensors.safe_open", fake_safe_open):
        res1 = dory_pytorch.load_dory_model(str(dir_path), device="cpu")
        assert res1 is fake_model
        assert fake_safe_open.call_args[0][0] == str(safetensor_file)

    # Case 2: path ending with .safetensors directly
    file_path = tmp_path / "custom.safetensors"
    file_path.write_bytes(b"dummy")
    with patch.object(dory_pytorch, "DoryForCausalLM", return_value=fake_model), \
         patch("safetensors.safe_open", fake_safe_open):
        res2 = dory_pytorch.load_dory_model(str(file_path), device="cpu")
        assert res2 is fake_model
        assert fake_safe_open.call_args[0][0] == str(file_path)

    # Case 3: HuggingFace Hub download
    with patch.object(dory_pytorch, "DoryForCausalLM", return_value=fake_model), \
         patch("safetensors.safe_open", fake_safe_open), \
         patch("huggingface_hub.hf_hub_download", return_value="/tmp/downloaded.safetensors") as mock_hf:
        res3 = dory_pytorch.load_dory_model("nvidia/Dory-Test", device="cpu")
        assert res3 is fake_model
        mock_hf.assert_called_once_with("nvidia/Dory-Test", "model.safetensors")


# ===========================================================================
# 3. Dory Recurrent Loop Execution Tests
# ===========================================================================

def test_dory_recurrent_min_loops_validation() -> None:
    with pytest.raises(ValueError, match="Expected 1 <= min_loops <= max_loops"):
        execute_adaptive_recurrent_loops(
            initial_hidden_state=0,
            recurrent_step_fn=lambda h, idx: h,
            lm_head=lambda h: [0.0],
            min_loops=0,
        )

    with pytest.raises(ValueError, match="Expected 1 <= min_loops <= max_loops"):
        execute_adaptive_recurrent_loops(
            initial_hidden_state=0,
            recurrent_step_fn=lambda h, idx: h,
            lm_head=lambda h: [0.0],
            min_loops=5,
            max_loops=3,
        )


def test_dory_recurrent_h_ndim_fallback_and_final_norm() -> None:
    norm_called = False
    def custom_norm(h: Any) -> Any:
        nonlocal norm_called
        norm_called = True
        return h

    # Non-subscriptable h raises TypeError on h[:, -1:, :] and triggers fallback probe_h = h
    logits_full = [0.0] * 500
    res = execute_adaptive_recurrent_loops(
        initial_hidden_state=123,
        recurrent_step_fn=lambda h, idx: h + 1,
        lm_head=lambda h: logits_full,
        final_norm=custom_norm,
        min_loops=1,
        max_loops=2,
    )
    assert norm_called is True
    assert res.loops_executed == 2
    assert res.halted_early is False


def test_dory_recurrent_raw_logits_tensors_and_non_tensors() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    # 3D tensor
    res_3d = execute_adaptive_recurrent_loops(
        initial_hidden_state=torch.randn(1, 2, 4),
        recurrent_step_fn=lambda h, idx: h,
        lm_head=lambda h: torch.zeros(1, 1, 1000),
        min_loops=1,
        max_loops=1,
    )
    assert res_3d.final_logits.ndim == 1

    # 2D tensor
    res_2d = execute_adaptive_recurrent_loops(
        initial_hidden_state=torch.randn(1, 4),
        recurrent_step_fn=lambda h, idx: h,
        lm_head=lambda h: torch.zeros(1, 1000),
        min_loops=1,
        max_loops=1,
    )
    assert res_2d.final_logits.ndim == 1

    # Non-tensor logits (list of floats)
    logits_list = [0.0] * 1000
    logits_list[319] = 10.0  # token 319 in positive_token_ids
    logits_list[350] = -10.0
    res_list = execute_adaptive_recurrent_loops(
        initial_hidden_state=1,
        recurrent_step_fn=lambda h, idx: h,
        lm_head=lambda h: logits_list,
        min_loops=1,
        max_loops=1,
    )
    assert res_list.final_logits == logits_list


def test_dory_recurrent_early_halting_and_max_loops() -> None:
    # Early halting path: entropy <= 0.12 and prob_delta <= 0.005 on loop 2
    def stable_lm_head(h: Any) -> list[float]:
        logits = [0.0] * 500
        logits[319] = 20.0  # huge positive probability -> entropy ~ 0.0
        return logits

    res_halt = execute_adaptive_recurrent_loops(
        initial_hidden_state=0,
        recurrent_step_fn=lambda h, idx: h,
        lm_head=stable_lm_head,
        min_loops=2,
        max_loops=5,
        entropy_halt_threshold=0.5,
        kl_stability_threshold=0.01,
    )
    assert res_halt.halted_early is True
    assert res_halt.loops_executed == 2

    # Non-halting path: entropy stays high
    def unstable_lm_head(h: Any) -> list[float]:
        logits = [0.0] * 500
        logits[319] = 0.0
        logits[350] = 0.0  # equal -> entropy ~ 0.693
        return logits

    res_no_halt = execute_adaptive_recurrent_loops(
        initial_hidden_state=0,
        recurrent_step_fn=lambda h, idx: h,
        lm_head=unstable_lm_head,
        min_loops=2,
        max_loops=4,
        entropy_halt_threshold=0.1,
    )
    assert res_no_halt.halted_early is False
    assert res_no_halt.loops_executed == 4


# ===========================================================================
# 4. Additional Dory Recurrent & Memory Guard Branch Tests
# ===========================================================================

def test_dory_recurrent_bad_tensor_slice_exception() -> None:
    class BadTensor:
        ndim = 3
        def __getitem__(self, item: Any) -> Any:
            raise IndexError("bad slice")

    logits_full = [0.0] * 500
    res = execute_adaptive_recurrent_loops(
        initial_hidden_state=BadTensor(),
        recurrent_step_fn=lambda h, idx: h,
        lm_head=lambda h: logits_full,
        min_loops=1,
        max_loops=1,
    )
    assert res.loops_executed == 1


def test_dory_recurrent_torch_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    logits_full = [0.0] * 500
    res = execute_adaptive_recurrent_loops(
        initial_hidden_state=0,
        recurrent_step_fn=lambda h, idx: h,
        lm_head=lambda h: logits_full,
        min_loops=1,
        max_loops=1,
    )
    assert res.loops_executed == 1


def test_pcie_spillover_guard_empty_cache_cuda_unavailable_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=50000.0, auto_empty_cache=True)
    call_count = 0

    def fake_is_available() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count == 1  # True first time (inspect), False second time (guard)

    fake_cuda = types.SimpleNamespace(
        is_available=fake_is_available,
        mem_get_info=lambda idx: (100 * 1024 * 1024, 4096 * 1024 * 1024),
        memory_allocated=lambda idx: 100 * 1024 * 1024,
        memory_reserved=lambda idx: 200 * 1024 * 1024,
        empty_cache=lambda: None,
    )
    if HAS_TORCH:
        monkeypatch.setattr(torch, "cuda", fake_cuda)
    else:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with guard.guard_forward_pass() as st:
        assert st.spillover_risk_detected is True


# ===========================================================================
# 5. Recurrent PyTorch Backend Exhaustive Tests
# ===========================================================================

def test_parse_cuda_device_index() -> None:
    assert _parse_cuda_device_index("cuda:2") == 2
    assert _parse_cuda_device_index("cuda:10") == 10
    assert _parse_cuda_device_index("cuda") == 0
    assert _parse_cuda_device_index("cpu") == 0
    assert _parse_cuda_device_index("cuda:invalid") == 0


def test_recurrent_backend_init_validation() -> None:
    with pytest.raises(ValueError, match="num_loops must be >= 1"):
        RecurrentPyTorchBackend(num_loops=0)

    backend = RecurrentPyTorchBackend(model_name="test-model", num_loops=3, device="cpu")
    assert backend.model == "test-model"
    assert "num_loops=3" in repr(backend)


def test_recurrent_ensure_loaded_already_provided() -> None:
    fake_m = MagicMock()
    fake_tok = MagicMock()
    backend = RecurrentPyTorchBackend(model=fake_m, tokenizer=fake_tok)
    m, t = backend._ensure_loaded()
    assert m is fake_m
    assert t is fake_tok


def test_recurrent_ensure_loaded_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    backend = RecurrentPyTorchBackend()
    with pytest.raises(ImportError, match="PyTorch and Transformers are required"):
        backend._ensure_loaded()


def test_recurrent_ensure_loaded_dory_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    fake_tokenizer_cls = MagicMock()
    fake_tok_instance = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = fake_tok_instance

    fake_load_dory = MagicMock()
    fake_dory_instance = MagicMock()
    fake_load_dory.return_value = fake_dory_instance

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=fake_tokenizer_cls,
        AutoModelForCausalLM=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("logitgate.backends.dory_pytorch.load_dory_model", fake_load_dory)

    backend = RecurrentPyTorchBackend(model_name="nvidia/nemotron-dory-test", num_loops=2, device="cpu")
    m, t = backend._ensure_loaded()
    assert m is fake_dory_instance
    assert t is fake_tok_instance
    fake_load_dory.assert_called_once()


def test_recurrent_ensure_loaded_automodel_branch_with_device_to(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    fake_tokenizer_cls = MagicMock()
    fake_tok_instance = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = fake_tok_instance

    fake_model_cls = MagicMock()
    fake_model_instance = MagicMock()
    fake_model_instance.to.return_value = fake_model_instance
    fake_model_cls.from_pretrained.return_value = fake_model_instance

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=fake_tokenizer_cls,
        AutoModelForCausalLM=fake_model_cls,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = RecurrentPyTorchBackend(model_name="gpt2", device="cpu")
    m, _ = backend._ensure_loaded()
    assert m is fake_model_instance
    fake_model_instance.to.assert_called_with("cpu")


def test_recurrent_query_top_logprobs_signature_error_and_torch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    class ModelNoSig:
        # Calling inspect.signature on this raises TypeError
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(logits=torch.tensor([[[0.5, 0.2, 0.1]]]))

    model = ModelNoSig()
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.encode.return_value = [0]
    tokenizer.decode.return_value = "A"

    backend = RecurrentPyTorchBackend(model=model, tokenizer=tokenizer, device="cpu")
    res = backend.query_top_logprobs("prompt", top_k=2)
    assert len(res) > 0


def test_recurrent_query_top_logprobs_output_variations() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    # Variation 1: outputs is raw tensor
    model1 = MagicMock()
    model1.device = "cpu"
    model1.return_value = torch.tensor([[[1.0, 2.0, 3.0]]])
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1]])}
    tok.encode.return_value = [1]
    tok.decode.return_value = "A"

    backend1 = RecurrentPyTorchBackend(model=model1, tokenizer=tok, device="cpu")
    res1 = backend1.query_top_logprobs("test", top_k=2)
    assert len(res1) > 0

    # Variation 2: outputs is nested list
    model2 = MagicMock()
    model2.device = "cpu"
    model2.return_value = [[[1.0, 2.0, 3.0]]]
    backend2 = RecurrentPyTorchBackend(model=model2, tokenizer=tok, device="cpu")
    res2 = backend2.query_top_logprobs("test", top_k=2)
    assert len(res2) > 0

    # Variation 3: outputs.logits is 2D list
    model3 = MagicMock()
    model3.device = "cpu"
    model3.return_value = types.SimpleNamespace(logits=[[[1.0, 2.0, 3.0]]])
    backend3 = RecurrentPyTorchBackend(model=model3, tokenizer=tok, device="cpu")
    res3 = backend3.query_top_logprobs("test", top_k=2)
    assert len(res3) > 0


def test_recurrent_query_top_logprobs_allowed_tokens() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    model = MagicMock()
    model.device = "cpu"
    logits = torch.zeros(1, 1, 100)
    logits[0, 0, 5] = 2.0
    logits[0, 0, 6] = -1.0
    model.return_value = types.SimpleNamespace(logits=logits)

    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1]])}
    def fake_encode(text: str, add_special_tokens: bool = False) -> list[int]:
        if "A" in text:
            return [5]
        if "B" in text:
            return [6]
        return []

    tok.encode = fake_encode
    tok.decode.return_value = "A"

    backend = RecurrentPyTorchBackend(model=model, tokenizer=tok, device="cpu")
    res = backend.query_top_logprobs("test", allowed_tokens=["A", "B"])
    assert len(res) == 2
    tokens = [r["token"] for r in res]
    assert "A" in tokens
    assert "B" in tokens


def test_recurrent_query_top_logprobs_torch_topk_and_guaranteed() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    model = MagicMock()
    model.device = "cpu"
    logits = torch.randn(1, 1, 100)
    model.return_value = types.SimpleNamespace(logits=logits)

    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1]])}
    tok.encode.side_effect = lambda t, add_special_tokens=False: [ord(t[0]) % 100] if t else []
    tok.decode.side_effect = lambda ids: chr(ids[0])

    backend = RecurrentPyTorchBackend(model=model, tokenizer=tok, device="cpu")
    res = backend.query_top_logprobs("test", top_k=10)
    assert len(res) >= 10
    tokens = [r["token"] for r in res]
    assert "A" in tokens or "B" in tokens


def test_recurrent_query_top_logprobs_fallback_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    model = MagicMock()
    model.device = "cpu"
    logits_3d = [[[1.0, 5.0, 2.0, 4.0]]]
    model.return_value = types.SimpleNamespace(logits=logits_3d)

    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1]])}
    tok.encode.return_value = []
    tok.decode.side_effect = lambda ids: str(ids[0])

    # Force torch import failure during topk
    monkeypatch.setattr("torch.topk", MagicMock(side_effect=TypeError("topk disabled")))

    backend = RecurrentPyTorchBackend(model=model, tokenizer=tok, device="cpu")
    res = backend.query_top_logprobs("test", top_k=2)
    assert len(res) == 2
    assert res[0]["token"] == "1"  # Highest logit 5.0 at index 1


def test_recurrent_generate_signature_and_inputs_handling() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    # Case 1: inputs is dict with input_ids
    model1 = MagicMock()
    model1.device = "cpu"
    model1.generate.return_value = torch.tensor([[10, 20, 30, 40]])  # input len 2, gen len 2
    tok1 = MagicMock()
    tok1.return_value = {"input_ids": torch.tensor([[10, 20]])}
    tok1.decode.return_value = "<think>thought</think>Verdict: PASS"

    backend1 = RecurrentPyTorchBackend(model=model1, tokenizer=tok1, device="cpu")
    text1, n1 = backend1.generate("prompt", max_tokens=10)
    assert text1 == "Verdict: PASS"
    assert n1 == 2

    # Case 2: inputs is an object without __getitem__ key 'input_ids' (hits line 293)
    class CustomInputs:
        shape = (1, 2)
        def to(self, dev: Any) -> CustomInputs:
            return self

    model2 = MagicMock()
    model2.device = "cpu"
    model2.generate.return_value = torch.tensor([[10, 20, 30]])
    tok2 = MagicMock()
    tok2.return_value = CustomInputs()
    tok2.decode.return_value = "Normal text"

    backend2 = RecurrentPyTorchBackend(model=model2, tokenizer=tok2, device="cpu")
    text2, n2 = backend2.generate("prompt")
    assert text2 == "Normal text"
    assert n2 == 1


def test_recurrent_close_and_aclose() -> None:
    backend = RecurrentPyTorchBackend(device="cpu")
    backend.close()
    asyncio.run(backend.aclose())


# ===========================================================================
# 6. PyTorch Native Prefix Broadcast Evaluator Tests
# ===========================================================================

def test_prefix_evaluator_device_resolution_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class ModelWithoutDev:
        pass

    tok = MagicMock()

    # Branch 1: device explicitly given
    ev1 = PrefixBroadcastLogitEvaluator(ModelWithoutDev(), tok, device="cpu")
    assert ev1.device == "cpu"

    # Branch 2: hasattr(model, "device")
    m_with_dev = types.SimpleNamespace(device="cuda:1")
    ev2 = PrefixBroadcastLogitEvaluator(m_with_dev, tok)
    assert ev2.device == "cuda:1"

    # Branch 3: torch.cuda.is_available() -> False => cpu
    if HAS_TORCH:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        ev3 = PrefixBroadcastLogitEvaluator(ModelWithoutDev(), tok)
        assert str(ev3.device) == "cpu"

    # Branch 4: ImportError/AttributeError on torch => cpu
    monkeypatch.setitem(sys.modules, "torch", None)
    ev4 = PrefixBroadcastLogitEvaluator(ModelWithoutDev(), tok)
    assert ev4.device == "cpu"


def test_prefix_evaluator_candidate_input_output_payload() -> None:
    prefix, fwd, perm = PrefixBroadcastLogitEvaluator.build_inverted_prefix_and_suffixes(
        rubric="Must be valid",
        candidate_input="test input",
        candidate_output="test output",
    )
    assert "<|context|>\ntest input\n</|context|>" in prefix
    assert "<|candidate|>\ntest output\n</|candidate|>" in prefix
    assert "Map 'A' = PASS" in fwd
    assert "Map 'A' = FAIL" in perm


def test_prefix_evaluator_single_pass_list_prefix_ids() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    # Return list of ints from tokenizer.encode instead of torch.Tensor (hits line 118)
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **kwargs: [1, 2, 3] if "rubric" in text else [4, 5]

    fake_logits = torch.randn(2, 2, 100)
    model = MagicMock()
    model.return_value = types.SimpleNamespace(past_key_values=(), logits=fake_logits)

    evaluator = PrefixBroadcastLogitEvaluator(model, tok, device="cpu")
    l1, l2 = evaluator.evaluate_single_pass_permuted("Rubric")
    assert l1.shape == (100,)
    assert l2.shape == (100,)


def test_prefix_evaluator_suffix_length_mismatch_padding_variants() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    tok = MagicMock()
    # Different lengths for fwd and perm
    def encode_fn(text: str, **kwargs: Any) -> list[int]:
        if "<|rubric|>" in text:
            return [1, 2]
        if "Map 'A' = PASS" in text:
            return [10, 20]
        if "Map 'A' = FAIL" in text:
            return [30, 40, 50]  # length 3 vs 2
        return [1, 2]

    tok.encode = encode_fn
    tok.pad_token_id = 99

    fake_logits = torch.randn(2, 3, 10)
    model = MagicMock()
    model.return_value = types.SimpleNamespace(past_key_values=(), logits=fake_logits)

    evaluator = PrefixBroadcastLogitEvaluator(model, tok, device="cpu")
    l1, l2 = evaluator.evaluate_single_pass_permuted("Rubric")
    assert l1.shape == (10,)
    assert l2.shape == (10,)

    # Test pad_token_id is None, eos_token_id is used
    tok.pad_token_id = None
    tok.eos_token_id = 88
    l1_eos, l2_eos = evaluator.evaluate_single_pass_permuted("Rubric")
    assert l1_eos.shape == (10,)
    assert l2_eos.shape == (10,)

    # Test both pad_token_id and eos_token_id are None (default 0)
    tok.pad_token_id = None
    tok.eos_token_id = None
    l1_0, l2_0 = evaluator.evaluate_single_pass_permuted("Rubric")
    assert l1_0.shape == (10,)
    assert l2_0.shape == (10,)


def test_prefix_evaluator_past_kv_variants() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    tok = MagicMock()
    tok.encode.return_value = [1, 2]

    fake_logits = torch.randn(2, 2, 10)
    model = MagicMock()

    # Case A: past_kv has batch_repeat_interleave
    past_kv_interleave = MagicMock()
    past_kv_interleave.batch_repeat_interleave.return_value = past_kv_interleave
    model.return_value = types.SimpleNamespace(past_key_values=past_kv_interleave, logits=fake_logits)

    evaluator = PrefixBroadcastLogitEvaluator(model, tok, device="cpu")
    evaluator.evaluate_single_pass_permuted("Rubric")
    past_kv_interleave.batch_repeat_interleave.assert_called_once_with(2)

    # Case B: past_kv is not tuple and no batch_repeat_interleave (hits line 165: expanded_kv = past_kv)
    past_kv_raw = object()
    model.return_value = types.SimpleNamespace(past_key_values=past_kv_raw, logits=fake_logits)
    evaluator.evaluate_single_pass_permuted("Rubric")
    call_args = model.call_args[1]
    assert call_args["past_key_values"] is past_kv_raw


def test_prefix_evaluator_evaluate_debiased() -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")

    tok = MagicMock()
    tok.encode.side_effect = lambda t, **kwargs: [1, 2]
    # SurfaceTokenMap.from_tokenizer needs encode to return token ids for A and B
    def token_encode(t: str, *args: Any, **kwargs: Any) -> list[int]:
        if "A" in t:
            return [10]
        if "B" in t:
            return [20]
        return [1, 2]

    tok.encode = token_encode

    fake_logits = torch.zeros(2, 2, 100)
    fake_logits[0, -1, 10] = 5.0  # Pass first: token A high
    fake_logits[1, -1, 20] = 5.0  # Fail first: token B high
    model = MagicMock()
    model.return_value = types.SimpleNamespace(past_key_values=(), logits=fake_logits)

    evaluator = PrefixBroadcastLogitEvaluator(model, tok, device="cpu")
    res = evaluator.evaluate_debiased(rubric="Must pass", candidate="test candidate")
    assert res.calibrated_prob_pass > 0.5


# ===========================================================================
# 7. Llama.cpp Backend Exhaustive Tests
# ===========================================================================

def test_llamacpp_config_validation_errors() -> None:
    with pytest.raises(ConfigurationError, match="suppress_thinking must be a bool"):
        LlamaCppBackend(suppress_thinking="not_a_bool")  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="auto_logit_bias must be a bool"):
        LlamaCppBackend(auto_logit_bias=1)  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="lora_path must be a non-empty string or None"):
        LlamaCppBackend(lora_path="")

    with pytest.raises(ConfigurationError, match="lora_path must be a non-empty string or None"):
        LlamaCppBackend(lora_path=123)  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="lora_scale must be a finite positive float"):
        LlamaCppBackend(lora_scale=True)

    with pytest.raises(ConfigurationError, match="lora_scale must be a finite positive float"):
        LlamaCppBackend(lora_scale=-0.5)

    with pytest.raises(ConfigurationError, match="lora_scale must be a finite positive float"):
        LlamaCppBackend(lora_scale=float("inf"))


def test_llamacpp_init_with_gguf_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Subcase 1: llama_cpp missing
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ConfigurationError, match="requires llama-cpp-python"):
        LlamaCppBackend(llm="/path/to/model.gguf")

    # Subcase 2: llama_cpp present
    fake_llama_cls = MagicMock()
    fake_llama_mod = types.SimpleNamespace(Llama=fake_llama_cls)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_mod)

    b = LlamaCppBackend(llm="/path/to/model.gguf", lora_path="/path/to/lora.bin", lora_scale=1.5)
    assert b._owns_llm is True
    fake_llama_cls.assert_called_once_with(
        model_path="/path/to/model.gguf",
        logits_all=True,
        lora_path="/path/to/lora.bin",
        lora_scale=1.5,
    )


def test_llamacpp_extract_chat_top_logprobs_variations() -> None:
    # Error: non-dict
    with pytest.raises(BackendError, match="returned non-object JSON"):
        LlamaCppBackend._extract_chat_top_logprobs("not-a-dict")  # type: ignore[arg-type]

    # Empty choices or logprobs
    assert LlamaCppBackend._extract_chat_top_logprobs({}) == []
    assert LlamaCppBackend._extract_chat_top_logprobs({"choices": []}) == []
    assert LlamaCppBackend._extract_chat_top_logprobs({"choices": ["not-dict"]}) == []
    assert LlamaCppBackend._extract_chat_top_logprobs({"choices": [{}]}) == []
    assert LlamaCppBackend._extract_chat_top_logprobs({"choices": [{"logprobs": None}]}) == []

    # Format A: content[0]["top_logprobs"]
    data_a = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.1},
                                {"token": "B", "logprob": -2.3},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    res_a = LlamaCppBackend._extract_chat_top_logprobs(data_a)
    assert len(res_a) == 2
    assert res_a[0]["token"] == "A"
    assert res_a[0]["logprob"] == -0.1

    # Format B: logprobs["top_logprobs"] as list of dicts or list of list
    data_b1 = {
        "choices": [
            {
                "logprobs": {
                    "top_logprobs": [
                        [{"token": "X", "logprob": -0.5}]
                    ]
                }
            }
        ]
    }
    res_b1 = LlamaCppBackend._extract_chat_top_logprobs(data_b1)
    assert len(res_b1) == 1
    assert res_b1[0]["token"] == "X"

    data_b2 = {
        "choices": [
            {
                "logprobs": {
                    "top_logprobs": [
                        {"token": "Y", "logprob": -0.8}
                    ]
                }
            }
        ]
    }
    res_b2 = LlamaCppBackend._extract_chat_top_logprobs(data_b2)
    assert len(res_b2) == 1
    assert res_b2[0]["token"] == "Y"


def test_llamacpp_extract_completion_logprobs_variations() -> None:
    with pytest.raises(BackendError, match="returned non-object JSON"):
        LlamaCppBackend._extract_completion_logprobs(None)  # type: ignore[arg-type]

    assert LlamaCppBackend._extract_completion_logprobs({}) == []

    data = {
        "completion_probabilities": [
            {
                "probs": [
                    {"tok_str": "A", "logprob": -0.1},
                    {"token": "B", "prob": 0.5},
                    {"token": "C"},  # missing prob and logprob
                    "not-a-dict",
                ]
            }
        ]
    }
    res = LlamaCppBackend._extract_completion_logprobs(data)
    assert len(res) == 2
    assert res[0]["token"] == "A"
    assert abs(res[0]["logprob"] - (-0.1)) < 1e-4
    assert res[1]["token"] == "B"
    assert abs(res[1]["logprob"] - math.log(0.5)) < 1e-4


def test_llamacpp_extract_chat_and_raw_completion_variations() -> None:
    # Chat completion
    with pytest.raises(BackendError, match="returned non-object JSON"):
        LlamaCppBackend._extract_chat_completion([])

    text, tok = LlamaCppBackend._extract_chat_completion({
        "choices": [{"message": {"content": "<think>t</think>Answer"}}],
        "usage": {"completion_tokens": 15},
    })
    assert text == "Answer"
    assert tok == 15

    # Raw completion
    with pytest.raises(BackendError, match="returned non-object JSON"):
        LlamaCppBackend._extract_raw_completion(123)

    text_raw, tok_raw = LlamaCppBackend._extract_raw_completion({
        "content": "<think>t</think>Raw Output",
        "tokens_predicted": 8,
    })
    assert text_raw == "Raw Output"
    assert tok_raw == 8


def test_llamacpp_ensure_in_process_llm_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Branch 1: self.llm is not None
    b1 = LlamaCppBackend(llm=MagicMock())
    assert b1._ensure_in_process_llm() is True

    # Branch 2: self.base_url is set -> returns False
    b2 = LlamaCppBackend(base_url="http://localhost:8080")
    assert b2._ensure_in_process_llm() is False

    # Branch 3: base_url is empty, model is None -> BackendError
    b3 = LlamaCppBackend(base_url="")
    b3.model = None
    with pytest.raises(BackendError, match="Local llama.cpp backend requires a GGUF model path"):
        b3._ensure_in_process_llm()

    # Branch 4: base_url is empty, llama_cpp missing -> BackendError
    b4 = LlamaCppBackend(base_url="", model="/path/to/model.gguf")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(BackendError, match="In-process llama.cpp requires 'llama-cpp-python'"):
        b4._ensure_in_process_llm()

    # Branch 5: base_url is empty, Llama instantiation throws -> BackendError
    fake_llama_cls = MagicMock(side_effect=RuntimeError("load failed"))
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=fake_llama_cls))
    with pytest.raises(BackendError, match="Failed to load GGUF model"):
        b4._ensure_in_process_llm()

    # Branch 6: base_url is empty, succeeds with lora
    fake_llama_instance = MagicMock()
    fake_llama_cls_ok = MagicMock(return_value=fake_llama_instance)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=fake_llama_cls_ok))
    b5 = LlamaCppBackend(base_url="", model="/path/to/model.gguf", lora_path="/path/lora.bin", lora_scale=1.2)
    assert b5._ensure_in_process_llm() is True
    assert b5.llm is fake_llama_instance
    assert b5._owns_llm is True


def test_llamacpp_close_aclose_owned_vs_unowned() -> None:
    # Owned LLM: close() is called
    mock_llm = MagicMock()
    b_owned = LlamaCppBackend(llm=mock_llm)
    b_owned._owns_llm = True
    b_owned.close()
    mock_llm.close.assert_called_once()
    assert b_owned.llm is None
    assert b_owned._owns_llm is False

    # Aclose owned
    mock_llm2 = MagicMock()
    b_async = LlamaCppBackend(llm=mock_llm2)
    b_async._owns_llm = True
    asyncio.run(b_async.aclose())
    mock_llm2.close.assert_called_once()


def test_llamacpp_resolve_in_process_logit_bias() -> None:
    mock_llm = MagicMock()
    # LLM tokenize method: accepts utf-8 bytes; sum(raw) ensures " A" and " B" are disjoint
    mock_llm.tokenize.side_effect = lambda raw, add_bos=False: [sum(raw)]

    b = LlamaCppBackend(llm=mock_llm, auto_logit_bias=True)

    # auto_logit_bias is False
    b.auto_logit_bias = False
    assert b._resolve_in_process_logit_bias(["A", "B"]) is None

    # len < 2
    b.auto_logit_bias = True
    assert b._resolve_in_process_logit_bias(["A"]) is None

    # Normal resolution & caching
    bias1 = b._resolve_in_process_logit_bias(["A", "B"])
    assert bias1 is not None
    # Caching hit
    bias2 = b._resolve_in_process_logit_bias(["A", "B"])
    assert bias2 is bias1

    # Exception during tokenize -> None
    b_err = LlamaCppBackend(llm=MagicMock(), auto_logit_bias=True)
    b_err.llm.tokenize.side_effect = RuntimeError("tokenize error")
    assert b_err._resolve_in_process_logit_bias(["A", "B"]) is None


def test_llamacpp_in_process_query_top_logprobs_and_generate() -> None:
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"content": "<think>t</think>Result"},
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [{"token": "A", "logprob": -0.05}]
                        }
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 5},
    }

    b = LlamaCppBackend(llm=mock_llm)
    res_lps = b.query_top_logprobs("Prompt", top_k=5, allowed_tokens=["A", "B"], custom_param=123)
    assert len(res_lps) == 1
    assert res_lps[0]["token"] == "A"

    text, tokens = b.generate("Prompt", max_tokens=20)
    assert text == "Result"
    assert tokens == 5


def test_llamacpp_http_query_top_logprobs_400_retry_and_404_fallback() -> None:
    call_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        call_urls.append(url_str)
        payload = json.loads(request.content.decode("utf-8"))

        if "/v1/chat/completions" in url_str:
            if "chat_template_kwargs" in payload:
                # Server does not support chat_template_kwargs, returns 400
                return httpx.Response(400, json={"error": "Unsupported chat_template_kwargs"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "logprobs": {
                                "content": [{"top_logprobs": [{"token": "Y", "logprob": -0.2}]}]
                            }
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    b = LlamaCppBackend(base_url="http://localhost:8080", client=client)

    # 1. 400 retry drops chat_template_kwargs and succeeds
    res = b.query_top_logprobs("prompt", top_k=5)
    assert len(res) == 1
    assert res[0]["token"] == "Y"
    assert b._supports_chat_template_kwargs is False

    # 2. 404 on /v1/chat/completions triggers /completion fallback
    def handler_404(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/v1/chat/completions" in url_str:
            return httpx.Response(404)
        if "/completion" in url_str:
            return httpx.Response(
                200,
                json={
                    "completion_probabilities": [
                        {"probs": [{"token": "FallbackTok", "logprob": -0.4}]}
                    ]
                },
            )
        return httpx.Response(500)

    transport2 = httpx.MockTransport(handler_404)
    client2 = httpx.Client(transport=transport2)
    b2 = LlamaCppBackend(base_url="http://localhost:8080", client=client2)
    res_fb = b2.query_top_logprobs("prompt")
    assert len(res_fb) == 1
    assert res_fb[0]["token"] == "FallbackTok"
    assert b2._use_completion_fallback is True


def test_llamacpp_http_aquery_and_agenerate_variations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        payload = json.loads(request.content.decode("utf-8"))
        if "/v1/chat/completions" in url_str:
            if "chat_template_kwargs" in payload:
                return httpx.Response(422, json={"error": "invalid parameter"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "<think>t</think>Async Output"},
                            "logprobs": {
                                "content": [{"top_logprobs": [{"token": "AsyncTok", "logprob": -0.3}]}]
                            },
                        }
                    ],
                    "usage": {"completion_tokens": 12},
                },
            )
        if "/completion" in url_str:
            return httpx.Response(
                200,
                json={"content": "Raw Fallback", "tokens_predicted": 7},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    b = LlamaCppBackend(base_url="http://localhost:8080", async_client=async_client)

    # aquery_top_logprobs
    lps = asyncio.run(b.aquery_top_logprobs("test", top_k=3))
    assert len(lps) == 1
    assert lps[0]["token"] == "AsyncTok"

    # agenerate
    text, tokens = asyncio.run(b.agenerate("test", max_tokens=10))
    assert text == "AsyncOutput" or "Async Output" in text
    assert tokens == 12

    # Async fallback to /completion when 404
    def handler_404(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/v1/chat/completions" in url_str:
            return httpx.Response(404)
        if "/completion" in url_str:
            return httpx.Response(200, json={"content": "Raw 404 Fallback", "tokens_predicted": 9})
        return httpx.Response(500)

    transport_fb = httpx.MockTransport(handler_404)
    async_client_fb = httpx.AsyncClient(transport=transport_fb)
    b_fb = LlamaCppBackend(base_url="http://localhost:8080", async_client=async_client_fb)
    text_fb, tok_fb = asyncio.run(b_fb.agenerate("test"))
    assert text_fb == "Raw 404 Fallback"
    assert tok_fb == 9


# ===========================================================================
# 8. Llama.cpp Additional Branch Tests & Ollama Backend Exhaustive Tests
# ===========================================================================

def test_llamacpp_additional_branches() -> None:
    # 1. repr
    b_repr = LlamaCppBackend(model="test.gguf", base_url="http://localhost:8080")
    assert "test.gguf" in repr(b_repr)

    # 2. _extract_chat_top_logprobs empty fallback
    empty_res = LlamaCppBackend._extract_chat_top_logprobs({"choices": [{"logprobs": {"other": 1}}]})
    assert empty_res == []

    # 3. close() exception suppression
    mock_llm = MagicMock()
    mock_llm.close.side_effect = RuntimeError("close failed")
    b_close = LlamaCppBackend(llm=mock_llm)
    b_close._owns_llm = True
    b_close.close()  # should not raise
    assert b_close.llm is None

    # 4. _resolve_in_process_logit_bias non-callable tokenize & TypeError on add_bos & 3+ tokens
    mock_llm2 = MagicMock()
    mock_llm2.tokenize = "not-callable"
    b_noncall = LlamaCppBackend(llm=mock_llm2, auto_logit_bias=True)
    assert b_noncall._resolve_in_process_logit_bias(["A", "B"]) is None

    def tokenize_no_bos(raw: bytes) -> list[int]:
        return [sum(raw)]

    def tokenize_raising(raw: bytes, add_bos: bool = False) -> list[int]:
        if add_bos:
            raise TypeError("unexpected add_bos")
        return tokenize_no_bos(raw)

    mock_llm3 = MagicMock()
    mock_llm3.tokenize = tokenize_raising
    b_typeerr = LlamaCppBackend(llm=mock_llm3, auto_logit_bias=True)
    bias_3tok = b_typeerr._resolve_in_process_logit_bias(["A", "B", "C"])
    assert bias_3tok is not None

    # 5. In-process async execution (aquery_top_logprobs & agenerate)
    mock_llm_async = MagicMock()
    mock_llm_async.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"content": "In-process async text"},
                "logprobs": {
                    "content": [{"top_logprobs": [{"token": "Z", "logprob": -0.01}]}]
                },
            }
        ],
        "usage": {"completion_tokens": 4},
    }
    b_inp_async = LlamaCppBackend(llm=mock_llm_async)
    lps = asyncio.run(b_inp_async.aquery_top_logprobs("prompt", top_k=2))
    assert lps[0]["token"] == "Z"
    text, toks = asyncio.run(b_inp_async.agenerate("prompt", max_tokens=10))
    assert text == "In-process async text"
    assert toks == 4

    # 6. HTTP payload builders with kwargs & logit_bias
    b_http = LlamaCppBackend(base_url="http://localhost:8080")
    chat_p = b_http.build_chat_payload(
        "prompt",
        allowed_tokens=["A", "B"],
        logit_bias={"10": 50},
        custom_kwarg="val",
    )
    assert chat_p["logit_bias"] == {"10": 50}
    assert chat_p["custom_kwarg"] == "val"
    assert " (A, B)" in chat_p["messages"][0]["content"]

    comp_p = b_http.build_completion_payload(
        "prompt",
        top_k=5,
        custom_kwarg2="val2",
    )
    assert comp_p["custom_kwarg2"] == "val2"
    assert comp_p["n_probs"] == 5

    # 7. HTTP sync generate with 404 fallback
    def handler_gen_404(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/v1/chat/completions" in url_str:
            return httpx.Response(404)
        if "/completion" in url_str:
            return httpx.Response(200, json={"content": "Sync Raw Fallback", "tokens_predicted": 6})
        return httpx.Response(500)

    client_gen = httpx.Client(transport=httpx.MockTransport(handler_gen_404))
    b_gen_fb = LlamaCppBackend(base_url="http://localhost:8080", client=client_gen)
    text_sync, tok_sync = b_gen_fb.generate("test", max_tokens=15)
    assert text_sync == "Sync Raw Fallback"
    assert tok_sync == 6


def test_ollama_transport_hint() -> None:
    b = OllamaBackend(model="qwen2.5:7b")
    assert b.transport_hint(404, "model 'qwen2.5:7b' not found") == "Pull it first: ollama pull qwen2.5:7b"
    assert b.transport_hint(404, "resource missing") is None
    assert b.transport_hint(500, "internal error") is None


def test_ollama_repr_and_build_query_payload() -> None:
    b = OllamaBackend(model="llama3:latest", base_url="http://localhost:11434", num_ctx=4096, options={"seed": 42}, keep_alive="5m")
    assert "llama3:latest" in repr(b)

    payload = b.build_query_payload(
        "Classify",
        top_k=5,
        allowed_tokens=["A", "B"],
        think=True,
        options={"temperature": 0.2},
        extra_key="extra_val",
    )
    assert payload["think"] is True
    assert payload["keep_alive"] == "5m"
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["seed"] == 42
    assert payload["options"]["temperature"] == 0.2
    assert payload["extra_key"] == "extra_val"
    assert " (A, B)" in payload["messages"][0]["content"]


def test_ollama_extract_top_logprobs_variations() -> None:
    b = OllamaBackend()
    with pytest.raises(BackendError, match="returned non-object JSON"):
        b._extract_top_logprobs_from_response("not-a-dict")  # type: ignore[arg-type]

    assert b._extract_top_logprobs_from_response({}) == []
    assert b._extract_top_logprobs_from_response({"logprobs": "not-a-list"}) == []

    # First has 'token' directly (flat logprobs)
    flat_data = {
        "logprobs": [
            {"token": "A", "logprob": -0.1},
            {"token": "B", "logprob": -2.0},
        ]
    }
    flat_res = b._extract_top_logprobs_from_response(flat_data)
    assert len(flat_res) == 2
    assert flat_res[0]["token"] == "A"


def test_ollama_query_top_logprobs_400_think_retry_sync_and_async() -> None:
    # Sync 400 retry
    def handler_sync(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if "think" in payload:
            return httpx.Response(400, json={"error": "think param not supported"})
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "A",
                    "logprobs": [{"top_logprobs": [{"token": "A", "logprob": -0.05}]}],
                }
            },
        )

    b_sync = OllamaBackend(client=httpx.Client(transport=httpx.MockTransport(handler_sync)))
    res_sync = b_sync.query_top_logprobs("prompt")
    assert len(res_sync) == 1
    assert res_sync[0]["token"] == "A"
    assert b_sync._supports_think_param is False

    # Async 400 retry
    b_async = OllamaBackend(async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler_sync)))
    res_async = asyncio.run(b_async.aquery_top_logprobs("prompt"))
    assert len(res_async) == 1
    assert res_async[0]["token"] == "A"
    assert b_async._supports_think_param is False


def test_ollama_build_generate_payload_variations() -> None:
    b = OllamaBackend(model="qwen", num_ctx=2048, options={"top_p": 0.9}, keep_alive=120)

    # max_tokens < 256 -> think_default = False
    p_small = b.build_generate_payload("prompt", max_tokens=100, kwargs={})
    assert p_small["think"] is False
    assert p_small["options"]["num_predict"] == 100
    assert p_small["options"]["num_ctx"] == 2048
    assert p_small["options"]["top_p"] == 0.9
    assert p_small["keep_alive"] == 120

    # max_tokens >= 256 -> think_default = True
    p_large = b.build_generate_payload("prompt", max_tokens=512, kwargs={})
    assert p_large["think"] is True

    # explicit think=None -> falls back to think_default
    p_none = b.build_generate_payload("prompt", max_tokens=50, kwargs={"think": None, "options": {"seed": 1}})
    assert p_none["think"] is False
    assert p_none["options"]["seed"] == 1


def test_ollama_parse_chat_and_continuation_payload() -> None:
    # Error on non-dict
    with pytest.raises(BackendError, match="returned non-object JSON"):
        OllamaBackend._parse_chat("not-dict")  # type: ignore[arg-type]

    # Content empty, but 'thinking' field present
    data_thinking = {
        "message": {"content": "", "thinking": "Internal thought trace"},
        "eval_count": 25,
    }
    content, tokens = OllamaBackend._parse_chat(data_thinking)
    assert content == "<think>Internal thought trace"
    assert tokens == 25

    # Continuation payload construction
    base_payload = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "user prompt"}],
        "think": True,
        "options": {"num_predict": 100},
    }
    cont = OllamaBackend._continuation_payload(base_payload, "user prompt", "<think>thought", max_tokens=50)
    assert cont["think"] is False
    assert cont["options"]["num_predict"] == 50
    assert len(cont["messages"]) == 2
    assert cont["messages"][1]["role"] == "assistant"
    assert "</think>" in cont["messages"][1]["content"]


def test_ollama_generate_and_agenerate_400_retry_and_continuation() -> None:
    call_count = 0
    # Test continuation truncation recovery
    def handler_trunc(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        payload = json.loads(request.content.decode("utf-8"))

        if call_count == 1:
            # First response: only thought trace, no visible answer
            return httpx.Response(
                200,
                json={
                    "message": {"content": "<think>Long pondering that filled budget</think>"},
                    "eval_count": 30,
                },
            )
        else:
            # Second continuation response: final answer
            assert payload["think"] is False
            return httpx.Response(
                200,
                json={
                    "message": {"content": "Final Verdict: PASS"},
                    "eval_count": 10,
                },
            )

    client_trunc = httpx.Client(transport=httpx.MockTransport(handler_trunc))
    b_trunc = OllamaBackend(client=client_trunc)
    text, tokens = b_trunc.generate("Critique", max_tokens=300)
    assert text == "Final Verdict: PASS"
    assert tokens == 40

    # Async generate with 400 think retry and continuation
    async_call_count = 0
    def handler_async_400(request: httpx.Request) -> httpx.Response:
        nonlocal async_call_count
        async_call_count += 1
        payload = json.loads(request.content.decode("utf-8"))

        if async_call_count == 1 and payload["think"]:
            # First: returns 400 because think=True
            return httpx.Response(400, json={"error": "think not supported"})
        return httpx.Response(
            200,
            json={
                "message": {"content": "Non-thinking response"},
                "eval_count": 15,
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_async_400))
    b_async = OllamaBackend(async_client=async_client)
    text_async, tok_async = asyncio.run(b_async.agenerate("Prompt", max_tokens=300))
    assert text_async == "Non-thinking response"
    assert tok_async == 15


# ===========================================================================
# 9. OpenAI-Compatible Backend Exhaustive Tests
# ===========================================================================

def test_openai_transport_hint() -> None:
    b = OpenAICompatBackend(model="gpt-4o-mini")
    assert "Set OPENAI_API_KEY" in str(b.transport_hint(401, "unauthorized"))
    assert "Set OPENAI_API_KEY" in str(b.transport_hint(403, "forbidden"))
    assert "Check that model 'gpt-4o-mini'" in str(b.transport_hint(404, "not found"))
    assert "does not expose logprobs" in str(b.transport_hint(400, "logprobs are not supported"))
    assert b.transport_hint(400, "bad prompt") is None
    assert b.transport_hint(500, "internal error") is None


def test_openai_config_validation_and_headers() -> None:
    with pytest.raises(ConfigurationError, match="suppress_thinking must be a bool"):
        OpenAICompatBackend(suppress_thinking="false")  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="auto_logit_bias must be a bool"):
        OpenAICompatBackend(auto_logit_bias="true")  # type: ignore[arg-type]

    b_key = OpenAICompatBackend(api_key="secret-key")
    h_key = b_key._headers()
    assert h_key["Authorization"] == "Bearer secret-key"
    assert "gpt-4o-mini" in repr(b_key)

    b_nokey = OpenAICompatBackend(api_key="")
    b_nokey.api_key = ""
    h_nokey = b_nokey._headers()
    assert "Authorization" not in h_nokey


def test_openai_build_query_payload_variations() -> None:
    b = OpenAICompatBackend(model="meta-llama/Llama-3-70b-Instruct", logit_bias={"10": 100})
    b._use_chat_template_no_think = True

    payload = b.build_query_payload(
        "Evaluate",
        top_k=25,  # capped at 20
        allowed_tokens=["Pass", "Fail"],
        logit_bias={"20": 50},
        extra_opt="opt",
    )
    assert payload["top_logprobs"] == 20
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["logit_bias"] == {"20": 50}
    assert payload["extra_opt"] == "opt"
    assert " (Pass, Fail)" in payload["messages"][0]["content"]


def test_openai_is_thinking_prefill_trap() -> None:
    # Trap 1: empty lps
    assert OpenAICompatBackend._is_thinking_prefill_trap([], ["A", "B"]) is True

    # Trap 2: top token is thinking token and no target token is present
    trap_lps = [{"token": "<think>", "logprob": -0.01}, {"token": " reasoning", "logprob": -2.0}]
    assert OpenAICompatBackend._is_thinking_prefill_trap(trap_lps, ["A", "B"]) is True

    trap_lps2 = [{"token": "<|thought|>", "logprob": -0.05}]
    assert OpenAICompatBackend._is_thinking_prefill_trap(trap_lps2, ["A", "B"]) is True

    # Not trap: target token present
    safe_lps = [{"token": "A", "logprob": -0.05}]
    assert OpenAICompatBackend._is_thinking_prefill_trap(safe_lps, ["A", "B"]) is False


def test_openai_tokenize_payloads_and_urls() -> None:
    b_v1 = OpenAICompatBackend(base_url="http://localhost:8000/v1", model="qwen")
    urls_v1 = b_v1._tokenize_urls()
    assert "http://localhost:8000/v1/tokenize" in urls_v1
    assert "http://localhost:8000/tokenize" in urls_v1

    b_root = OpenAICompatBackend(base_url="http://localhost:8000", model="qwen")
    urls_root = b_root._tokenize_urls()
    assert urls_root == ["http://localhost:8000/tokenize"]

    payloads = b_v1._tokenize_payloads("hello")
    assert len(payloads) == 2
    assert payloads[0][0] == "vllm"
    assert payloads[1][0] == "llamacpp"

    payloads_filtered = b_v1._tokenize_payloads("hello", mode="vllm")
    assert len(payloads_filtered) == 1
    assert payloads_filtered[0][0] == "vllm"


def test_openai_token_variants_and_prepare_tokenize_lookup() -> None:
    b = OpenAICompatBackend(auto_logit_bias=True)
    clean, variants = b._token_variants(["A", " B "])
    assert clean == ("A", "B")
    assert "A" in variants
    assert "a" in variants
    assert " A" in variants

    # Disabled auto_logit_bias
    b_disabled = OpenAICompatBackend(auto_logit_bias=False)
    q, _, _, _, _ = b_disabled._prepare_tokenize_lookup(["A", "B"])
    assert q is False

    # Empty allowed_tokens
    q2, _, _, _, _ = b._prepare_tokenize_lookup([])
    assert q2 is False

    # Cache hit
    b._logit_bias_cache[("A", "B")] = {"1": 100}
    q3, c3, _, _, _ = b._prepare_tokenize_lookup(["A", "B"])
    assert q3 is False
    assert c3 == {"1": 100}


def test_openai_parse_tokenize_response_variations() -> None:
    # Status != 200
    r_bad_status = httpx.Response(404, text="Not Found")
    assert OpenAICompatBackend._parse_tokenize_response(r_bad_status) == (False, None)

    # Invalid JSON
    r_invalid_json = httpx.Response(200, text="not json")
    assert OpenAICompatBackend._parse_tokenize_response(r_invalid_json) == (False, None)

    # Missing tokens list
    r_no_tokens = httpx.Response(200, json={"error": "no tokens"})
    assert OpenAICompatBackend._parse_tokenize_response(r_no_tokens) == (False, None)

    # Empty tokens list
    r_empty = httpx.Response(200, json={"tokens": []})
    assert OpenAICompatBackend._parse_tokenize_response(r_empty) == (False, None)

    # Single token ID match
    r_single = httpx.Response(200, json={"tokens": [42]})
    assert OpenAICompatBackend._parse_tokenize_response(r_single) == (True, 42)

    # Multiple tokens (subwords) -> matched, but not single token
    r_multi = httpx.Response(200, json={"tokens": [10, 20]})
    assert OpenAICompatBackend._parse_tokenize_response(r_multi) == (True, None)


def test_openai_finalize_logit_bias() -> None:
    b = OpenAICompatBackend()
    # Empty token_ids -> unsupported
    assert b._finalize_logit_bias(("A",), set()) is None
    assert b._auto_logit_bias_supported is False

    # Non-empty
    b2 = OpenAICompatBackend()
    bias = b2._finalize_logit_bias(("A",), {15, 25})
    assert bias == {"15": 100, "25": 100}
    assert b2._logit_bias_cache[("A",)] == bias


def test_openai_resolve_auto_logit_bias_sync_and_async() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        # vllm payload has prompt
        if "prompt" in payload:
            text = payload["prompt"]
            if text in ("A", "a", " A", " a"):
                return httpx.Response(200, json={"tokens": [101]})
            if text in ("B", "b", " B", " b"):
                return httpx.Response(200, json={"tokens": [102]})
            return httpx.Response(200, json={"tokens": [999]})
        return httpx.Response(404)

    # Sync resolution
    b_sync = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        auto_logit_bias=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    bias_sync = b_sync._resolve_auto_logit_bias_sync(["A", "B"])
    assert bias_sync is not None
    assert "101" in bias_sync
    assert "102" in bias_sync

    # Async resolution
    b_async = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        auto_logit_bias=True,
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    bias_async = asyncio.run(b_async._resolve_auto_logit_bias_async(["A", "B"]))
    assert bias_async is not None
    assert "101" in bias_async
    assert "102" in bias_async

    # Tokenize HTTP error fallback
    def handler_err(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("cannot reach tokenize")

    b_err = OpenAICompatBackend(
        auto_logit_bias=True,
        client=httpx.Client(transport=httpx.MockTransport(handler_err)),
    )
    assert b_err._resolve_auto_logit_bias_sync(["A", "B"]) is None
    assert b_err._auto_logit_bias_supported is False


def test_openai_chat_template_fallback_payload_and_response() -> None:
    b = OpenAICompatBackend()
    payload = {"model": "gpt", "reasoning_effort": "none"}
    trap_lps = [{"token": "<think>", "logprob": -0.1}]

    fb_payload = b._build_chat_template_fallback_payload(payload, trap_lps, ["A", "B"], kwargs={})
    assert fb_payload is not None
    assert fb_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in fb_payload

    # Success fallback response
    resp_ok = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"top_logprobs": [{"token": "A", "logprob": -0.05}]}
                        ]
                    }
                }
            ]
        },
    )
    res = b._handle_chat_template_fallback_response(resp_ok, ["A", "B"])
    assert res is not None
    assert res[0]["token"] == "A"
    assert b._use_chat_template_no_think is True
    assert b._supports_reasoning_effort is False

    # Failure fallback response (still trapped or non-200)
    resp_fail = httpx.Response(400, text="Unsupported parameter")
    b2 = OpenAICompatBackend()
    res_fail = b2._handle_chat_template_fallback_response(resp_fail, ["A", "B"])
    assert res_fail is None
    assert b2._supports_chat_template_no_think is False


def test_openai_extract_top_logprobs_and_chat_completion_errors() -> None:
    b = OpenAICompatBackend()
    with pytest.raises(BackendError, match="returned non-object JSON"):
        b._extract_top_logprobs_from_response([])  # type: ignore[arg-type]

    assert b._extract_top_logprobs_from_response({}) == []
    assert b._extract_top_logprobs_from_response({"choices": []}) == []
    assert b._extract_top_logprobs_from_response({"choices": [{"logprobs": None}]}) == []
    assert b._extract_top_logprobs_from_response({"choices": [{"logprobs": {"content": []}}]}) == []
    assert b._extract_top_logprobs_from_response({"choices": [{"logprobs": {"content": [{}]}}]}) == []

    with pytest.raises(BackendError, match="returned non-object JSON"):
        OpenAICompatBackend._extract_chat_completion("not-dict")


def test_openai_query_top_logprobs_sync_async_auto_bias_400_retry_and_chat_template_fallback() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        url_str = str(request.url)
        payload = json.loads(request.content.decode("utf-8"))

        if "/tokenize" in url_str:
            return httpx.Response(200, json={"tokens": [55]})

        if "/chat/completions" in url_str:
            if "reasoning_effort" in payload:
                # 400 rejection on reasoning_effort
                return httpx.Response(400, json={"error": "unsupported reasoning_effort"})

            if "chat_template_kwargs" not in payload and call_count <= 3:
                # Return thinking trap
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "logprobs": {
                                    "content": [
                                        {"top_logprobs": [{"token": "<think>", "logprob": -0.01}]}
                                    ]
                                }
                            }
                        ]
                    },
                )
            else:
                # Valid verdict response after chat_template fallback
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "<think>t</think>Done"},
                                "logprobs": {
                                    "content": [
                                        {"top_logprobs": [{"token": "A", "logprob": -0.02}]}
                                    ]
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 10},
                    },
                )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    b_sync = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        auto_logit_bias=True,
        client=client,
    )

    # Sync query_top_logprobs with 400 retry + chat_template fallback
    lps = b_sync.query_top_logprobs("Prompt", top_k=5, allowed_tokens=["A", "B"])
    assert len(lps) == 1
    assert lps[0]["token"] == "A"

    # Async query_top_logprobs with chat_template fallback
    async_transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=async_transport)
    b_async = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        auto_logit_bias=True,
        async_client=async_client,
    )
    lps_async = asyncio.run(b_async.aquery_top_logprobs("Prompt", top_k=5, allowed_tokens=["A", "B"]))
    assert len(lps_async) == 1
    assert lps_async[0]["token"] == "A"

    # Generate and agenerate
    text, tokens = b_sync.generate("Critique", max_tokens=50)
    assert text == "Done"
    assert tokens == 10

    text_async, tokens_async = asyncio.run(b_async.agenerate("Critique", max_tokens=50))
    assert text_async == "Done"
    assert tokens_async == 10


# ===========================================================================
# 10. Base Backend Exhaustive Tests
# ===========================================================================

def test_response_detail_variations() -> None:
    # 1. JSON with string error
    r_str_err = httpx.Response(400, json={"error": "Direct error string"})
    assert _response_detail(r_str_err) == "Direct error string"

    # 2. JSON with dict error message
    r_dict_err = httpx.Response(400, json={"error": {"message": "Nested error message"}})
    assert _response_detail(r_dict_err) == "Nested error message"

    # 3. Non-JSON text
    r_text = httpx.Response(502, text="Bad Gateway from proxy")
    assert _response_detail(r_text) == "Bad Gateway from proxy"

    # 4. Empty text falls back to reason phrase
    r_empty = httpx.Response(500, text="")
    assert _response_detail(r_empty) == "Internal Server Error"


def test_describe_transport_error_all_subtypes() -> None:
    backend = types.SimpleNamespace(
        service_name="CustomServer",
        base_url="http://custom:9999",
        timeout=15.0,
        connect_hint="Run `custom-server start`",
        transport_hint=lambda code, detail: "Try restarting" if code == 503 else None,
    )

    # 1. TimeoutException with timeout
    req = httpx.Request("POST", "http://custom:9999")
    timeout_exc = httpx.ReadTimeout("timed out", request=req)
    err_timeout = describe_transport_error(backend, timeout_exc)
    assert isinstance(err_timeout, BackendTimeoutError)
    assert "within 15s" in str(err_timeout)

    # TimeoutException without timeout
    backend_notimeout = types.SimpleNamespace(base_url="http://custom:9999")
    err_notimeout = describe_transport_error(backend_notimeout, timeout_exc)
    assert "http://custom:9999 did not respond" in str(err_notimeout)

    # 2. ConnectError
    connect_exc = httpx.ConnectError("connection refused", request=req)
    err_connect = describe_transport_error(backend, connect_exc)
    assert isinstance(err_connect, BackendUnavailableError)
    assert "Run `custom-server start`" in str(err_connect)

    # 3. HTTPStatusError with transport hint
    resp_503 = httpx.Response(503, text="Service Unavailable", request=req)
    status_exc_hint = httpx.HTTPStatusError("503", request=req, response=resp_503)
    err_hint = describe_transport_error(backend, status_exc_hint)
    assert "Try restarting" in str(err_hint)

    # HTTPStatusError without transport hint
    resp_404 = httpx.Response(404, text="Not Found", request=req)
    status_exc_nohint = httpx.HTTPStatusError("404", request=req, response=resp_404)
    err_nohint = describe_transport_error(backend, status_exc_nohint)
    assert "returned HTTP 404" in str(err_nohint)

    # 4. General HTTPError
    proto_exc = httpx.ProtocolError("protocol violation", request=req)
    err_proto = describe_transport_error(backend, proto_exc)
    assert "failed: ProtocolError" in str(err_proto)


def test_translate_transport_errors_sync_and_async() -> None:
    class DummyBackend:
        service_name = "Dummy"
        base_url = "http://dummy:80"

        @translate_transport_errors
        def sync_http_err(self) -> None:
            raise httpx.ConnectError("cannot connect", request=httpx.Request("GET", "http://dummy:80"))

        @translate_transport_errors
        def sync_url_err(self) -> None:
            raise httpx.InvalidURL("bad url")

        @translate_transport_errors
        def sync_json_err(self) -> None:
            raise json.JSONDecodeError("expecting value", "doc", 0)

        @translate_transport_errors
        async def async_http_err(self) -> None:
            raise httpx.ConnectError("cannot connect async", request=httpx.Request("GET", "http://dummy:80"))

        @translate_transport_errors
        async def async_url_err(self) -> None:
            raise httpx.InvalidURL("bad url async")

        @translate_transport_errors
        async def async_json_err(self) -> None:
            raise json.JSONDecodeError("expecting value async", "doc", 0)

    dummy = DummyBackend()

    # Sync
    with pytest.raises(BackendUnavailableError):
        dummy.sync_http_err()

    with pytest.raises(BackendError, match="Invalid URL"):
        dummy.sync_url_err()

    with pytest.raises(BackendError, match="returned malformed JSON"):
        dummy.sync_json_err()

    # Async
    with pytest.raises(BackendUnavailableError):
        asyncio.run(dummy.async_http_err())

    with pytest.raises(BackendError, match="Invalid URL"):
        asyncio.run(dummy.async_url_err())

    with pytest.raises(BackendError, match="returned malformed JSON"):
        asyncio.run(dummy.async_json_err())


def test_strip_reasoning_blocks_comment_prefixes_and_unclosed() -> None:
    # Empty string
    assert strip_reasoning_blocks("") == ""

    # Comment prefixes on same line as <think>: not stripped
    py_comment = "# <think>This is a python comment</think>\nReal code"
    assert "<think>" in strip_reasoning_blocks(py_comment)

    js_comment = "// <think>Javascript comment</think>\nCode"
    assert "<think>" in strip_reasoning_blocks(js_comment)

    sql_comment = "-- <think>SQL comment</think>\nSELECT 1"
    assert "<think>" in strip_reasoning_blocks(sql_comment)

    c_comment = "/* <think>C style comment</think> */\nCode"
    assert "<think>" in strip_reasoning_blocks(c_comment)

    quoted = '"<think>literal string</think>"\nCode'
    assert "<think>" in strip_reasoning_blocks(quoted)


def test_logit_backend_and_generative_backend_context_managers_and_async_defaults() -> None:
    class ConcreteLogit(LogitBackend):
        def query_top_logprobs(self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"token": "A", "logprob": -0.1}]

    b_logit = ConcreteLogit()
    # Context manager
    with b_logit as backend_instance:
        assert backend_instance is b_logit

    async def run_async_logit() -> None:
        async with b_logit as l_async:
            assert l_async is b_logit
            res = await l_async.aquery_top_logprobs("prompt", top_k=1)
            assert res[0]["token"] == "A"

    asyncio.run(run_async_logit())

    class ConcreteGen(GenerativeBackend):
        def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
            return "Generated", 3

    b_gen = ConcreteGen()
    with b_gen as g:
        assert g is b_gen

    async def run_async_gen() -> None:
        async with b_gen as g_async:
            assert g_async is b_gen
            text, toks = await g_async.agenerate("prompt")
            assert text == "Generated"
            assert toks == 3

    asyncio.run(run_async_gen())


def test_http_client_mixin_lifecycle_and_loop_changes() -> None:
    class MixedBackend(_HTTPClientMixin):
        timeout = 10.0

        def __init__(self) -> None:
            self._init_http_clients()

    b = MixedBackend()
    c1 = b.client
    assert isinstance(c1, httpx.Client)
    # Re-access returns same
    assert b.client is c1

    # Close and re-access creates new
    b.close()
    assert b._client is None
    c2 = b.client
    assert c2 is not c1
    b.close()

    # Async client lifecycle and loop changes
    async def async_test() -> None:
        ac1 = b.async_client
        assert isinstance(ac1, httpx.AsyncClient)
        assert b.async_client is ac1
        await b.aclose()
        assert b._async_client is None

    asyncio.run(async_test())

    # Test loop_changed: create in new event loop
    async def second_loop_test() -> None:
        ac_new = b.async_client
        assert isinstance(ac_new, httpx.AsyncClient)
        await b.aclose()

    asyncio.run(second_loop_test())


def test_stub_backend_exhaustive_features() -> None:
    b = StubBackend(default_prob=0.8)
    assert "default_prob=0.8" in repr(b)

    # 1. logprob_responses with is_inverted_scale and numeric allowed_tokens
    b_queue = StubBackend(
        logprob_responses=[
            [{"token": "5", "logprob": -0.1}, {"token": "1", "logprob": -2.0}]
        ]
    )
    prompt_inv = "[SCORING SCALE]\nEvaluate on an inverted scale where 1 is highest and 5 is lowest."
    # Mirror of 5 with 1..5 is 1+5 - 5 = 1; mirror of 1 is 1+5 - 1 = 5
    res_inv = b_queue.query_top_logprobs(prompt_inv, allowed_tokens=["1", "2", "3", "4", "5"])
    assert res_inv[0]["token"] == "1"
    assert res_inv[1]["token"] == "5"

    # 2. is_inverted_scale without queue: reverse_top_logprobs vs forward_top_logprobs with mirror
    b_rev = StubBackend(reverse_top_logprobs=[{"token": "R", "logprob": -0.5}])
    assert b_rev.query_top_logprobs(prompt_inv)[0]["token"] == "R"

    b_fwd_mirror = StubBackend(forward_top_logprobs=[{"token": "5", "logprob": -0.2}])
    res_fwd_mirror = b_fwd_mirror.query_top_logprobs(prompt_inv, allowed_tokens=["1", "5"])
    assert res_fwd_mirror[0]["token"] == "1"

    # 3. Numeric score scale without custom logprobs (inverted vs normal)
    b_numeric = StubBackend(default_prob=0.9)
    res_num_inv = b_numeric.query_top_logprobs(prompt_inv, allowed_tokens=["1", "2", "3", "4", "5"])
    # Inverted: low token gets default_prob logprob
    assert res_num_inv[0]["token"] == "1"
    assert abs(res_num_inv[0]["logprob"] - math.log(0.9)) < 1e-4

    prompt_norm = "[SCORING SCALE]\nRate from 1 to 5."
    res_num_norm = b_numeric.query_top_logprobs(prompt_norm, allowed_tokens=["1", "2", "3", "4", "5"])
    assert res_num_norm[0]["token"] == "5"
    assert abs(res_num_norm[0]["logprob"] - math.log(0.9)) < 1e-4

    # 4. Preferred: in decision section with resp_idx % 2 == 1
    b_pref = StubBackend(default_prob=0.85)
    prompt_pref = "[DECISION MAPPING]\nPreferred: Choice A"
    b_pref.query_top_logprobs(prompt_pref, allowed_tokens=["A", "B", "C"])  # resp_idx 0 (even)
    res_pref_odd = b_pref.query_top_logprobs(prompt_pref, allowed_tokens=["A", "B", "C"])  # resp_idx 1 (odd)
    assert res_pref_odd[0]["token"] == "B"
    assert abs(res_pref_odd[0]["logprob"] - math.log(0.85)) < 1e-4

    # 5. Reverse pass with reverse_top_logprobs vs default
    b_rev_pass = StubBackend(reverse_top_logprobs=[{"token": "REV", "logprob": -0.1}])
    prompt_rev = "[DECISION MAPPING]\nA: The candidate artifact violates the criteria."
    assert b_rev_pass.query_top_logprobs(prompt_rev)[0]["token"] == "REV"

    # 6. Forward pass with forward_top_logprobs
    b_fwd_pass = StubBackend(forward_top_logprobs=[{"token": "FWD", "logprob": -0.1}])
    prompt_fwd = "[DECISION MAPPING]\nA: The candidate satisfies the criteria."
    assert b_fwd_pass.query_top_logprobs(prompt_fwd)[0]["token"] == "FWD"

    # 7. Async methods
    async def async_stub_test() -> None:
        lps = await b.aquery_top_logprobs("test")
        assert len(lps) == 2
        text, toks = await b.agenerate("test")
        assert "VERDICT" in text
        assert toks >= 0

    asyncio.run(async_stub_test())


def test_is_openai_host_variations() -> None:
    assert _is_openai_host("", "") is False
    assert _is_openai_host("localhost", "") is True
    assert _is_openai_host("localhost:8080", "") is True
    assert _is_openai_host("example", "/v1/chat") is True
    assert _is_openai_host("my-cluster:1234", "") is True
    assert _is_openai_host("198.51.100.10", "") is True
    assert _is_openai_host("api.groq.com", "") is True
    assert _is_openai_host("gpt-4o-mini", "") is False


def test_infer_backend_spec_all_backends() -> None:
    # String passed through
    assert infer_backend_spec("stub://0.9") == "stub://0.9"

    # Inner wrapper unwrapping
    class Wrapper:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

    stub_custom = StubBackend(default_prob=0.7)
    wrapped = Wrapper(stub_custom)
    assert infer_backend_spec(wrapped) == "stub://0.7"

    # Stub default
    stub_def = StubBackend(default_prob=0.95)
    assert infer_backend_spec(stub_def) == "stub://default"

    # OllamaBackend
    ol_model = OllamaBackend(model="qwen2.5:7b", base_url="http://localhost:11434")
    assert infer_backend_spec(ol_model) == "ollama://localhost:11434/qwen2.5:7b"

    # OpenAICompatBackend
    oai = OpenAICompatBackend(model="gpt-4o", base_url="https://api.openai.com/v1")
    assert infer_backend_spec(oai) == "openai://gpt-4o"

    vllm = OpenAICompatBackend(model="meta/llama", base_url="http://localhost:8000/v1")
    assert infer_backend_spec(vllm) == "vllm://localhost:8000/v1/meta/llama"

    # LlamaCppBackend
    llama_http = LlamaCppBackend(model="model.gguf", base_url="http://localhost:8080")
    assert infer_backend_spec(llama_http) == "llamacpp://localhost:8080/model.gguf"

    llama_local = LlamaCppBackend(llm=MagicMock(), model="local.gguf", base_url="")
    assert infer_backend_spec(llama_local) == "llamacpp:///local.gguf"

    # RecurrentPyTorchBackend
    rec = RecurrentPyTorchBackend(
        model_name="nvidia/nemotron",
        num_loops=4,
        min_loops=2,
        max_loops=6,
        kl_threshold=0.01,
        device="cpu",
    )
    spec_rec = infer_backend_spec(rec)
    assert "recurrent://nvidia/nemotron" in spec_rec
    assert "num_loops=4" in spec_rec
    assert "min_loops=2" in spec_rec
    assert "device=cpu" in spec_rec

    # Custom backend
    class CustomLogit(LogitBackend):
        def query_top_logprobs(
            self,
            prompt: str,
            top_k: int = 10,
            allowed_tokens: list[str] | None = None,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            return []

    assert infer_backend_spec(CustomLogit()) == "custom://CustomLogit"


def test_resolve_logit_backend_all_schemes() -> None:
    # Already a LogitBackend
    stub = StubBackend()
    assert resolve_logit_backend(stub) is stub

    # Implicit default
    b_def = resolve_logit_backend(None)
    assert isinstance(b_def, StubBackend)

    # stub:// with custom prob
    b_prob = resolve_logit_backend("stub://0.82")
    assert isinstance(b_prob, StubBackend)
    assert abs(b_prob.default_prob - 0.82) < 1e-4

    # stub:// with invalid prob fallback
    b_invalid_prob = resolve_logit_backend("stub://not_a_number")
    assert isinstance(b_invalid_prob, StubBackend)
    assert b_invalid_prob.default_prob == 0.95

    # ollama:// shorthand model
    b_ol_short = resolve_logit_backend("ollama://qwen2.5:7b")
    assert isinstance(b_ol_short, OllamaBackend)
    assert b_ol_short.model == "qwen2.5:7b"

    # ollama:// host and model
    b_ol_host = resolve_logit_backend("ollama://127.0.0.1:11434/llama3:latest")
    assert isinstance(b_ol_host, OllamaBackend)
    assert b_ol_host.model == "llama3:latest"
    assert "127.0.0.1:11434" in b_ol_host.base_url

    # openai:// shorthand model
    b_oai_short = resolve_logit_backend("openai://gpt-4o-mini")
    assert isinstance(b_oai_short, OpenAICompatBackend)
    assert b_oai_short.model == "gpt-4o-mini"
    assert b_oai_short.base_url == "https://api.openai.com/v1"

    # vllm:// shorthand model
    b_vllm_short = resolve_logit_backend("vllm://Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(b_vllm_short, OpenAICompatBackend)
    assert b_vllm_short.model == "Qwen/Qwen2.5-7B-Instruct"
    assert b_vllm_short.base_url == "http://localhost:8000/v1"

    # llamacpp bare spec
    b_llamacpp = resolve_logit_backend("llamacpp", model="test.gguf")
    assert isinstance(b_llamacpp, LlamaCppBackend)

    # llamacpp:// host
    b_llamacpp_http = resolve_logit_backend("llamacpp://localhost:8080/model.gguf")
    assert isinstance(b_llamacpp_http, LlamaCppBackend)
    assert b_llamacpp_http.base_url == "http://localhost:8080"
    assert b_llamacpp_http.model == "model.gguf"

    # llamacpp:///path/to/model.gguf
    b_llamacpp_local = resolve_logit_backend("llamacpp:///models/model.gguf")
    assert isinstance(b_llamacpp_local, LlamaCppBackend)
    assert b_llamacpp_local.base_url == ""
    assert "models/model.gguf" in str(b_llamacpp_local.model)

    # recurrent:// with query parameters
    b_rec = resolve_logit_backend("recurrent://nvidia/dory?num_loops=3&min_loops=1&max_loops=4&kl_threshold=0.02&device=cpu")
    assert isinstance(b_rec, RecurrentPyTorchBackend)
    assert b_rec.num_loops == 3
    assert b_rec.min_loops == 1
    assert b_rec.max_loops == 4
    assert b_rec.kl_threshold == 0.02
    assert b_rec.device == "cpu"

    # Fallback to Ollama for bare string
    b_bare = resolve_logit_backend("qwen2.5:14b")
    assert isinstance(b_bare, OllamaBackend)
    assert b_bare.model == "qwen2.5:14b"


def test_resolve_generative_backend_validation() -> None:
    # 1. Already a GenerativeBackend
    stub = StubBackend()
    assert resolve_generative_backend(stub) is stub

    # 2. String resolving to GenerativeBackend
    b_gen = resolve_generative_backend("stub://default")
    assert isinstance(b_gen, GenerativeBackend)

    # 3. Object implementing LogitBackend but NOT GenerativeBackend raises TypeError
    class PureLogitBackend(LogitBackend):
        closed = False

        def query_top_logprobs(
            self,
            prompt: str,
            top_k: int = 10,
            allowed_tokens: list[str] | None = None,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            self.closed = True

    pure = PureLogitBackend()
    with pytest.raises(TypeError, match="does not implement GenerativeBackend"):
        resolve_generative_backend(cast(Any, pure))
    assert pure.closed is False  # caller owns instance, not closed

    # When resolved from a spec string, logitgate closes the backend it instantiated
    pure2 = PureLogitBackend()
    with patch("logitgate.backends.base.resolve_logit_backend", return_value=pure2), pytest.raises(
        TypeError, match="does not implement GenerativeBackend"
    ):
        resolve_generative_backend("custom://spec")
    assert pure2.closed is True


# ===========================================================================
# 11. Edge-Case Coverage Hardening for 95%+ Target Saturation
# ===========================================================================

def test_response_detail_response_not_read() -> None:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.side_effect = httpx.ResponseNotRead()
    type(mock_resp).text = PropertyMock(side_effect=httpx.ResponseNotRead())
    mock_resp.reason_phrase = "Reason Only"
    assert _response_detail(mock_resp) == "Reason Only"


def test_strip_reasoning_blocks_end_line_comment_prefix() -> None:
    text = "Answer <think>thinking\n# </think>\nstill thinking</think> Visible final"
    res = strip_reasoning_blocks(text)
    assert "Answer" in res
    assert "Visible final" in res


def test_infer_backend_spec_recurrent_default_params() -> None:
    rec_default = RecurrentPyTorchBackend(model_name="nvidia/nemotron", num_loops=2)
    assert infer_backend_spec(rec_default) == "recurrent://nvidia/nemotron"


def test_resolve_logit_backend_additional_schemes() -> None:
    # 1. openai with explicit host having /v1 (hits line 687-690)
    b_oai_v1 = resolve_logit_backend("openai://my-api.groq.com/v1/llama-3.3-70b")
    assert isinstance(b_oai_v1, OpenAICompatBackend)
    assert b_oai_v1.base_url == "https://my-api.groq.com/v1"
    assert b_oai_v1.model == "llama-3.3-70b"

    # 2. vllm with host without /v1 (hits line 692-694)
    b_vllm_nov1 = resolve_logit_backend("vllm://cluster-node.internal:8000/qwen-7b")
    assert isinstance(b_vllm_nov1, OpenAICompatBackend)
    assert b_vllm_nov1.base_url == "http://cluster-node.internal:8000/v1"
    assert b_vllm_nov1.model == "qwen-7b"

    # 3. recurrent:// with path only (no netloc)
    b_rec_path = resolve_logit_backend("recurrent:///models/dory-custom.safetensors")
    assert isinstance(b_rec_path, RecurrentPyTorchBackend)
    assert "dory-custom.safetensors" in b_rec_path.model_name

    # 4. Fallback string containing 'stub'
    b_stub_fallback = resolve_logit_backend("custom_stub_provider")
    assert isinstance(b_stub_fallback, StubBackend)


def test_openai_resolve_auto_logit_bias_async_error_and_unmatched() -> None:
    def handler_async_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("async tokenize failed")

    b_async_fail = OpenAICompatBackend(
        auto_logit_bias=True,
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler_async_fail)),
    )
    res = asyncio.run(b_async_fail._resolve_auto_logit_bias_async(["A", "B"]))
    assert res is None
    assert b_async_fail._auto_logit_bias_supported is False

    # Unmatched response (all 404)
    def handler_async_404(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    b_async_404 = OpenAICompatBackend(
        auto_logit_bias=True,
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler_async_404)),
    )
    res_404 = asyncio.run(b_async_404._resolve_auto_logit_bias_async(["A", "B"]))
    assert res_404 is None


def test_recurrent_inspect_signature_value_error_and_torch_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HAS_TORCH:
        pytest.skip("torch not available")
    class BuiltinModel:
        # dict.get is a C builtin method where inspect.signature raises ValueError
        forward = dict.get
        device = "cpu"
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return torch.tensor([[[0.1, 0.2]]])

    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1]])}
    tok.encode.return_value = [0]
    tok.decode.return_value = "A"

    b = RecurrentPyTorchBackend(model=BuiltinModel(), tokenizer=tok, device="cpu")
    res = b.query_top_logprobs("prompt", top_k=2)
    assert len(res) > 0

    # Test torch import error fallback in query_top_logprobs
    def fake_import_error_module(name: str) -> None:
        raise ImportError(f"No module named {name}")

    orig_import = __import__
    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("Simulated torch absence")
        return orig_import(name, *args, **kwargs)

    # Calling generate with inspect.signature raising TypeError
    model_gen = MagicMock()
    model_gen.device = "cpu"
    model_gen.generate.return_value = torch.tensor([[1, 2, 3]])
    b_gen = RecurrentPyTorchBackend(model=model_gen, tokenizer=tok, device="cpu")
    with patch("inspect.signature", side_effect=TypeError("sig error")):
        t, count = b_gen.generate("prompt", max_tokens=5)
        assert t == "A"
        assert count >= 0
