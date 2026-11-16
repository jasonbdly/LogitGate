"""Wave 2C Backend, Hardware & Streaming Mutant-Killing Test Suite.

Targets all 8 modules in Wave 2C:
1. logitgate.backends.dory_pytorch
2. logitgate.backends.recurrent
3. logitgate.backends.dory_recurrent
4. logitgate.backends.pytorch_native
5. logitgate.backends.llamacpp
6. logitgate.backends.ollama
7. logitgate.hardware.memory_guard
8. logitgate.streaming
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Explicit top-level imports and symbol mentions for mutation_audit.py AST mapping
from logitgate.backends.dory_recurrent import execute_adaptive_recurrent_loops
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.pytorch_native import PrefixBroadcastLogitEvaluator
from logitgate.backends.recurrent import RecurrentPyTorchBackend
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import BaseGate
from logitgate.hardware.memory_guard import PCIeSpilloverGuard, VRAMWatermarkStatus
from logitgate.models import DecisionStatus, Verdict
from logitgate.streaming import StreamGuard

try:
    import torch

    from logitgate.backends import dory_pytorch
    from logitgate.backends.dory_pytorch import (
        DoryAttention,
        DoryForCausalLM,
        DoryMLP,
        apply_rotary_pos_emb,
        load_dory_model,
        slerp_alpha,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    dory_pytorch = None  # type: ignore[assignment]


# ============================================================================
# Section 1: logitgate.backends.dory_pytorch
# ============================================================================

def test_dory_slerp_alpha_exact_scaling() -> None:
    """Kills line 45 * -> / in slerp_alpha."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    residual = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
    out = torch.tensor([[-0.8, 0.6]], dtype=torch.float32)
    alpha = torch.tensor([[0.25, 0.25]], dtype=torch.float32)
    alpha_scale = 2.0  # (alpha * alpha_scale) = 0.5 vs (alpha / alpha_scale) = 0.125

    res = slerp_alpha(residual, out, alpha, alpha_scale)

    # ya = [0.6, 0.8], yb = [-0.8, 0.6]
    # Under *: lr = 0.5 -> intermediate = [0.6, 0.8] + 0.5 * [-1.4, -0.2] = [-0.1, 0.7]
    # normalized: [-0.1/sqrt(0.50), 0.7/sqrt(0.50)] = [-0.141421, 0.989949]
    # Under /: lr = 0.125 -> intermediate = [0.425, 0.775] -> normalized [0.48083, 0.87681]
    assert res[0, 0].item() < 0.0, "Under *, x-component must be negative"
    assert pytest.approx(res[0, 0].item(), abs=1e-3) == -0.141421
    assert pytest.approx(res[0, 1].item(), abs=1e-3) == 0.989949


def test_dory_apply_rotary_pos_emb_exact_arithmetic() -> None:
    """Kills line 63 (* -> /, + -> -) and line 64 (* -> /, + -> -) in apply_rotary_pos_emb."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    q = torch.tensor([[[[1.2, -0.6]]]], dtype=torch.float32)
    k = torch.tensor([[[[0.8, 1.4]]]], dtype=torch.float32)
    cos = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32)
    sin = torch.tensor([[[[0.8660254, 0.8660254]]]], dtype=torch.float32)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # rotate_half(q) = [0.6, 1.2]
    # q * cos = [0.6, -0.3]
    # rotate_half(q) * sin = [0.519615, 1.03923]
    # q_rot = [0.6 + 0.519615, -0.3 + 1.03923] = [1.119615, 0.73923]
    assert pytest.approx(q_rot[0, 0, 0, 0].item(), abs=1e-4) == 1.119615
    assert pytest.approx(q_rot[0, 0, 0, 1].item(), abs=1e-4) == 0.73923

    # rotate_half(k) = [-1.4, 0.8]
    # k * cos = [0.4, 0.7]
    # rotate_half(k) * sin = [-1.212435, 0.69282]
    # k_rot = [0.4 - 1.212435, 0.7 + 0.69282] = [-0.812435, 1.39282]
    assert pytest.approx(k_rot[0, 0, 0, 0].item(), abs=1e-4) == -0.812435
    assert pytest.approx(k_rot[0, 0, 0, 1].item(), abs=1e-4) == 1.39282


def test_dory_attention_exact_arithmetic_sqk_and_gate() -> None:
    """Kills line 138 (* -> /) and line 150 (* -> /) in DoryAttention."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    torch.manual_seed(12345)
    attn = DoryAttention(layer_idx=0, is_sliding=True)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn_like(p) * 0.5 + 0.5)

    x = torch.randn((1, 2, 2560), dtype=torch.float32)
    out = attn(x)

    # Unmutated forward pass has deterministic sum 0.231731
    # Under mutant line 138 (* -> /): sum is 0.214569
    # Under mutant line 150 (* -> /): sum is NaN
    assert not torch.isnan(out).any(), "Output must not contain NaN"
    assert pytest.approx(out.sum().item(), abs=1e-3) == 0.231731


def test_dory_mlp_exact_arithmetic_suv_factors() -> None:
    """Kills lines 181 and 182 (* -> /) in DoryMLP."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    torch.manual_seed(54321)
    mlp = DoryMLP(layer_idx=1)
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_(torch.randn_like(p) * 0.5 + 0.5)

    x = torch.randn((1, 2, 2560), dtype=torch.float32)
    out = mlp(x)

    # Unmutated forward pass has deterministic sum 70.01345
    # Under mutant line 181 (* -> /): sum is 65.27845
    # Under mutant line 182 (* -> / in u.float() * (...)): sum is 59.06891
    assert pytest.approx(out.sum().item(), abs=1e-2) == 70.01345

    # Secondary check specifically targeting line 182 inner scaling (suv_up.float() * suv_factor)
    # With small inputs where norm is near EPS, inner factor scaling affects eps-stabilized normalization:
    torch.manual_seed(42)
    mlp2 = DoryMLP(layer_idx=1)
    for p in mlp2.parameters():
        torch.nn.init.normal_(p, std=0.01)
    x2 = torch.randn((1, 2, 2560), dtype=torch.float32) * 0.02
    out2 = mlp2(x2)
    # Unmutated sum is -0.574309, mutated is -0.582122
    assert pytest.approx(out2.sum().item(), abs=1e-4) == -0.574309


def test_dory_causal_lm_layer_looping_and_output_block_partition() -> None:
    """Kills line 250 (+ -> -) in DoryForCausalLM.forward."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    model = DoryForCausalLM.__new__(DoryForCausalLM)
    torch.nn.Module.__init__(model)
    model.num_loops = 1
    model.vocab_size = 100
    model.hidden_size = 64
    model.n_input_layers = 16
    model.n_recurrent_layers = 40
    model.n_output_layers = 16
    model.std = 0.0177

    layer_calls = [0] * 72
    layers = []
    for i in range(72):
        class MockLayer(torch.nn.Module):
            def __init__(self, idx: int):
                super().__init__()
                self.idx = idx
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                layer_calls[self.idx] += 1
                return x
        layers.append(MockLayer(i))
    model.layers = torch.nn.ModuleList(layers)
    model.embeddings = torch.nn.Embedding(100, 64)
    model.lm_head = torch.nn.Linear(64, 100, bias=False)
    model.logit_scale = torch.nn.Parameter(torch.ones(100))

    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(x)
    assert out.shape == (1, 1, 100)

    # Output block is layers[16 + 40:] = layers[56:] (16 layers: 56..71)
    # Under mutant + -> -: layers[16 - 40:] = layers[-24:] = layers[48:] (24 layers!)
    # Layer 48 is in recurrent block. For num_loops=1, it must be called exactly 1 time!
    # Under mutant, layer 48 is called 2 times (once in recurrent block, once in output block)!
    assert layer_calls[48] == 1, "Layer 48 must be called exactly once for num_loops=1"
    assert layer_calls[55] == 1, "Layer 55 must be called exactly once for num_loops=1"
    assert layer_calls[56] == 1, "Layer 56 must be called exactly once (output block)"
    assert layer_calls[71] == 1, "Layer 71 must be called exactly once (output block)"


def test_dory_causal_lm_generate_eos_token_override() -> None:
    """Kills line 270 (or -> and) in DoryForCausalLM.generate."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    model = DoryForCausalLM.__new__(DoryForCausalLM)
    torch.nn.Module.__init__(model)
    model.num_loops = 1

    # Mock forward pass returning logits that always predict token 99
    call_step = 0
    def mock_forward(input_ids: torch.Tensor, num_loops: int | None = None, **kwargs: Any) -> torch.Tensor:
        nonlocal call_step
        call_step += 1
        logits = torch.zeros((1, input_ids.shape[1], 200), dtype=torch.float32)
        logits[:, -1, 99] = 100.0  # Predicts token 99
        return logits

    model.forward = mock_forward  # type: ignore[assignment]

    input_ids = torch.tensor([[10]], dtype=torch.long)
    # eos_token_ids=[99]
    # Under 'or': eos = [99] -> stops immediately after generating token 99 (length 2)
    # Under 'and': [99] and [2, 11] = [2, 11] -> does not stop on token 99, continues until max_new_tokens!
    out = model.generate(input_ids, max_new_tokens=5, eos_token_ids=[99])
    assert out.shape[1] == 2, f"Should stop on token 99, got length {out.shape[1]}"


def test_dory_load_model_safetensors_extension_check(tmp_path: Any) -> None:
    """Kills line 294 (and -> or) in load_dory_model."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir()

    fake_file = empty_dir / "downloaded.safetensors"
    fake_file.write_bytes(b"")

    with patch("huggingface_hub.hf_hub_download", return_value=str(fake_file)) as mock_hub, \
         patch("safetensors.safe_open") as mock_safe_open, \
         patch("logitgate.backends.dory_pytorch.DoryForCausalLM") as mock_cls:
        fake_model = MagicMock()
        mock_cls.return_value = fake_model
        mock_safe_open.return_value.__enter__.return_value = MagicMock(get_tensor=lambda k: torch.zeros(1))
        load_dory_model(str(empty_dir), device="cpu")
        mock_hub.assert_called_once()


# ============================================================================
# Section 2: logitgate.backends.dory_recurrent
# ============================================================================

def test_dory_recurrent_h_ndim_boundary_tie() -> None:
    """Kills line 59 (>= -> >) in dory_recurrent."""
    class Tensor3D:
        ndim = 3
        def __init__(self, shape: tuple[int, int, int]):
            self.shape = shape
        def __getitem__(self, item: Any) -> Any:
            return Tensor3D((1, 1, 4))

    last_received_shape = None
    def lm_head_probe(h: Any) -> list[float]:
        nonlocal last_received_shape
        last_received_shape = getattr(h, "shape", None)
        logits = [0.0] * 500
        logits[319] = 10.0
        return logits

    execute_adaptive_recurrent_loops(
        initial_hidden_state=Tensor3D((1, 2, 4)),
        recurrent_step_fn=lambda h, idx: h,
        lm_head=lm_head_probe,
        min_loops=1,
        max_loops=1,
    )
    assert last_received_shape == (1, 1, 4), f"Expected (1, 1, 4) for h.ndim==3, got {last_received_shape}"


def test_dory_recurrent_max_loops_boundary_no_early_halt() -> None:
    """Kills line 98 (< -> <=) in dory_recurrent."""
    step_calls = 0
    def recurrent_step(h: Any, idx: int) -> Any:
        nonlocal step_calls
        step_calls += 1
        return h

    def stable_lm_head(h: Any) -> list[float]:
        logits = [0.0] * 500
        logits[319] = 15.0  # p_pos ~ 1.0, entropy ~ 0.0
        return logits

    trace = execute_adaptive_recurrent_loops(
        initial_hidden_state=[0.0],
        recurrent_step_fn=recurrent_step,
        lm_head=stable_lm_head,
        min_loops=1,
        max_loops=2,
        entropy_halt_threshold=0.5,
        kl_stability_threshold=0.5,
    )
    assert trace.loops_executed == 2
    assert trace.halted_early is False, "When loop_idx == max_loops, halted_early must be False"


def test_dory_recurrent_exact_threshold_ties() -> None:
    """Kills line 100 (<= -> <) on entropy and prob_delta in dory_recurrent."""
    # Test A: exact equality prob_delta == kl_stability_threshold (0.0)
    call_idx = 0
    def controlled_lm_head(h: Any) -> list[float]:
        nonlocal call_idx
        call_idx += 1
        logits = [0.0] * 500
        logits[319] = 2.0
        logits[350] = 0.0
        return logits

    trace_delta = execute_adaptive_recurrent_loops(
        initial_hidden_state=0,
        recurrent_step_fn=lambda h, idx: idx,
        lm_head=controlled_lm_head,
        min_loops=1,
        max_loops=3,
        entropy_halt_threshold=1.0,
        kl_stability_threshold=0.0,  # Exact equality: prob_delta == 0.0
    )
    assert trace_delta.halted_early is True
    assert trace_delta.loops_executed == 2

    # Test B: exact equality entropy == entropy_halt_threshold (1.0)
    def balanced_lm_head(h: Any) -> list[float]:
        logits = [0.0] * 500
        logits[319] = 1.0
        logits[350] = 1.0
        return logits

    trace_entropy = execute_adaptive_recurrent_loops(
        initial_hidden_state=0,
        recurrent_step_fn=lambda h, idx: idx,
        lm_head=balanced_lm_head,
        min_loops=1,
        max_loops=3,
        entropy_halt_threshold=1.0,  # Exact equality: entropy == 1.0
        kl_stability_threshold=0.5,
    )
    assert trace_entropy.halted_early is True
    assert trace_entropy.loops_executed == 2


# ============================================================================
# Section 3: logitgate.backends.llamacpp
# ============================================================================

def test_llamacpp_lora_scale_zero_rejected() -> None:
    """Kills line 57 (<= -> <) in llamacpp."""
    mock_llm = MagicMock()
    with pytest.raises(ConfigurationError, match="lora_scale must be a finite positive float"):
        LlamaCppBackend(llm=mock_llm, lora_scale=0.0)


def test_llamacpp_extract_chat_top_logprobs_null_logprob() -> None:
    """Kills line 123 (and -> or) in llamacpp."""
    data = {
        "choices": [
            {
                "logprobs": {
                    "top_logprobs": [
                        [{"token": "A", "logprob": None}, {"token": "B", "logprob": -0.5}]
                    ]
                }
            }
        ]
    }
    extracted = LlamaCppBackend._extract_chat_top_logprobs(data)
    assert len(extracted) == 1
    assert extracted[0]["token"] == "B"
    assert pytest.approx(extracted[0]["logprob"]) == -0.5


def test_llamacpp_ensure_in_process_llm_passes_lora_path() -> None:
    """Kills line 198 (is not -> is) in llamacpp."""
    pytest.importorskip("llama_cpp")
    mock_llama_cls = MagicMock()
    with patch("llama_cpp.Llama", mock_llama_cls):
        backend = LlamaCppBackend(base_url="", model="test.gguf", lora_path="/tmp/test_lora.bin", lora_scale=0.75)
        backend._ensure_in_process_llm()
        assert mock_llama_cls.call_count == 1
        kwargs = mock_llama_cls.call_args.kwargs
        assert kwargs.get("lora_path") == "/tmp/test_lora.bin"
        assert kwargs.get("lora_scale") == 0.75


def test_llamacpp_aclose_does_not_close_unowned_llm() -> None:
    """Kills line 222 (and -> or) in llamacpp."""
    mock_llm = MagicMock()
    backend = LlamaCppBackend(llm=mock_llm)
    assert backend._owns_llm is False

    with patch.object(backend, "close") as mock_close:
        asyncio.run(backend.aclose())
        mock_close.assert_not_called()


def test_llamacpp_in_process_effective_top_k_multiplier() -> None:
    """Kills line 277 (* -> /) in llamacpp."""
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"logprobs": {"top_logprobs": [[{"token": "A", "logprob": -0.1}]]}}]
    }
    backend = LlamaCppBackend(llm=mock_llm)

    backend._in_process_query_top_logprobs(
        prompt="test",
        top_k=5,
        allowed_tokens=["A", "B", "C", "D"],
    )
    call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
    assert call_kwargs["top_logprobs"] == 8, f"Expected 8, got {call_kwargs['top_logprobs']}"


def test_llamacpp_http_fallback_completion_max_tokens() -> None:
    """Kills line 514 (is not -> is) in llamacpp."""
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    backend._use_completion_fallback = True

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": "ok", "tokens_predicted": 10}
    with patch.object(backend.client, "post", return_value=mock_resp) as mock_post:
        backend.generate("test prompt", max_tokens=42)
        payload = mock_post.call_args.kwargs["json"]
    assert payload.get("n_predict") == 42, f"Expected n_predict=42, got {payload.get('n_predict')}"


# ============================================================================
# Section 4: logitgate.backends.ollama
# ============================================================================

def test_ollama_build_payload_clean_tokens_filter() -> None:
    """Kills line 67 (and -> or) in ollama."""
    backend = OllamaBackend(model="llama3", base_url="http://127.0.0.1:11434")

    payload = backend.build_query_payload(
        prompt="test",
        allowed_tokens=["PASS", "   "],
    )
    sys_content = payload["messages"][0]["content"]
    assert sys_content == "Respond with a single character only (PASS). Do not think or explain."


def test_ollama_build_payload_custom_dict_not_merged_into_options() -> None:
    """Kills line 103 (and -> or) in ollama."""
    backend = OllamaBackend(model="llama3", base_url="http://127.0.0.1:11434")

    payload = backend.build_query_payload(
        prompt="test",
        custom_dict={"foo": "bar"},
    )
    assert "custom_dict" in payload
    assert payload["custom_dict"] == {"foo": "bar"}
    assert "foo" not in payload["options"]


def test_ollama_build_generate_payload_custom_dict_not_merged_into_options() -> None:
    """Kills line 194 (and -> or) in ollama."""
    backend = OllamaBackend(model="llama3", base_url="http://127.0.0.1:11434")

    payload = backend.build_generate_payload(
        prompt="test",
        max_tokens=10,
        kwargs={"extra_meta": {"route": "direct"}},
    )
    assert "extra_meta" in payload
    assert payload["extra_meta"] == {"route": "direct"}
    assert "route" not in payload["options"]


# ============================================================================
# Section 5: logitgate.backends.pytorch_native
# ============================================================================

def test_pytorch_native_prefix_construction_candidate_input_only() -> None:
    """Kills line 55 (or -> and) in pytorch_native."""
    evaluator = PrefixBroadcastLogitEvaluator(model=MagicMock(), tokenizer=MagicMock())

    prefix, _, _ = evaluator.build_inverted_prefix_and_suffixes(
        rubric="Test rubric",
        candidate_input="Exact candidate context input",
        candidate_output="",
    )
    assert "<|context|>\nExact candidate context input\n</|context|>" in prefix


def test_pytorch_native_evaluate_single_pass_pad_token_id_fallback() -> None:
    """Kills line 134 (is -> is not) in pytorch_native."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, **kw: (
        [1, 2] if "Map 'A' = PASS" in text else [1, 2, 3]
    )
    mock_tokenizer.pad_token_id = 999
    mock_tokenizer.eos_token_id = 42

    mock_model = MagicMock()
    fake_prefix_out = MagicMock()
    fake_prefix_out.past_key_values = None
    fake_suffix_out = MagicMock()
    fake_suffix_out.logits = torch.zeros((2, 3, 1000), dtype=torch.float32)
    mock_model.side_effect = [fake_prefix_out, fake_suffix_out]

    evaluator = PrefixBroadcastLogitEvaluator(model=mock_model, tokenizer=mock_tokenizer, device="cpu")
    evaluator.evaluate_single_pass_permuted(rubric="Test rubric", candidate="Test cand")

    suffix_call_args = mock_model.call_args_list[1]
    input_ids = suffix_call_args.kwargs["input_ids"]
    assert input_ids[0, 0].item() == 999, f"Expected pad_id=999, got {input_ids[0, 0].item()}"


def test_pytorch_native_expanded_kv_non_expandable_past_key_values() -> None:
    """Kills lines 153 and 158 (and -> or) in pytorch_native."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    class NonExpandableTensor:
        ndim = 2

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [10, 20]

    mock_model = MagicMock()
    fake_prefix_out = MagicMock()
    fake_prefix_out.past_key_values = ((NonExpandableTensor(), NonExpandableTensor()),)
    fake_suffix_out = MagicMock()
    fake_suffix_out.logits = torch.zeros((2, 2, 100), dtype=torch.float32)
    mock_model.side_effect = [fake_prefix_out, fake_suffix_out]

    evaluator = PrefixBroadcastLogitEvaluator(model=mock_model, tokenizer=mock_tokenizer, device="cpu")
    res_fwd, res_perm = evaluator.evaluate_single_pass_permuted(rubric="rubric", candidate="cand")
    assert res_fwd is not None
    assert res_perm is not None


# ============================================================================
# Section 6: logitgate.backends.recurrent
# ============================================================================

def test_recurrent_ensure_loaded_partial_init() -> None:
    """Kills line 97 (and -> or, is not -> is) in recurrent."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    backend = RecurrentPyTorchBackend(model_name="test-model")
    backend._model = MagicMock()
    backend._tokenizer = None

    fake_tok_cls = MagicMock()
    fake_tok = MagicMock()
    fake_tok_cls.from_pretrained.return_value = fake_tok
    with patch.dict(sys.modules, {"transformers": MagicMock(AutoTokenizer=fake_tok_cls, AutoModelForCausalLM=MagicMock())}):
        _m, tok = backend._ensure_loaded()
        assert tok is not None
        assert tok == fake_tok


def test_recurrent_ensure_loaded_custom_torch_dtype() -> None:
    """Kills line 115 (or -> and) in recurrent."""
    if not HAS_TORCH:
        pytest.skip("torch not available")

    backend = RecurrentPyTorchBackend(model_name="test-model", torch_dtype=torch.float16)
    fake_model_cls = MagicMock()
    fake_tok_cls = MagicMock()
    with patch.dict(sys.modules, {"transformers": MagicMock(AutoTokenizer=fake_tok_cls, AutoModelForCausalLM=fake_model_cls)}):
        backend._ensure_loaded()
        call_kwargs = fake_model_cls.from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.float16


def test_recurrent_ensure_loaded_dory_device_arg() -> None:
    """Kills line 121 (!= -> ==) in recurrent."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    backend = RecurrentPyTorchBackend(model_name="dory-custom", device="cuda:1")
    fake_tok_cls = MagicMock()
    with patch.dict(sys.modules, {"transformers": MagicMock(AutoTokenizer=fake_tok_cls)}), \
         patch("logitgate.backends.dory_pytorch.load_dory_model") as mock_load_dory:
        backend._ensure_loaded()
        call_kwargs = mock_load_dory.call_args.kwargs
        assert call_kwargs["device"] == "cuda:1", f"Expected cuda:1, got {call_kwargs['device']}"


def test_recurrent_ensure_loaded_device_auto_no_to_call() -> None:
    """Kills line 132 (and -> or) in recurrent."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    backend = RecurrentPyTorchBackend(model_name="test-model", device="auto")
    fake_tok_cls = MagicMock()
    fake_model = MagicMock()
    fake_model_cls = MagicMock(from_pretrained=MagicMock(return_value=fake_model))
    with patch.dict(sys.modules, {"transformers": MagicMock(AutoTokenizer=fake_tok_cls, AutoModelForCausalLM=fake_model_cls)}):
        backend._ensure_loaded()
        fake_model.to.assert_not_called()


def test_recurrent_query_top_logprobs_num_loops_kwarg_forward() -> None:
    """Kills line 156 (in -> not in) in recurrent."""
    backend = RecurrentPyTorchBackend(model_name="test-model", num_loops=3)
    received_num_loops = None

    class ModelWithForward:
        device = "cpu"
        def forward(self, input_ids: Any, num_loops: int = 1) -> Any:
            nonlocal received_num_loops
            received_num_loops = num_loops
            res = MagicMock()
            res.logits = [[[-0.5, -1.0]]]
            return res
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    mock_model = ModelWithForward()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [1, 2]}
    mock_tokenizer.batch_decode.return_value = ["A", "B"]

    backend._model = mock_model
    backend._tokenizer = mock_tokenizer

    res = backend.query_top_logprobs("prompt", top_k=2)
    assert len(res) > 0
    assert received_num_loops == 3, f"Expected num_loops=3, got {received_num_loops}"


def test_recurrent_query_top_logprobs_inputs_to_model_device() -> None:
    """Kills line 173 (or -> and) in recurrent."""
    backend = RecurrentPyTorchBackend(model_name="test-model", device="auto")
    mock_model = MagicMock()
    mock_model.device = "cuda:0"
    mock_model.return_value = MagicMock(logits=[[[-0.5, -1.0]]])

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer = MagicMock(return_value=mock_inputs)
    mock_tokenizer.batch_decode.return_value = ["A", "B"]

    backend._model = mock_model
    backend._tokenizer = mock_tokenizer

    backend.query_top_logprobs("test prompt")
    mock_inputs.to.assert_called_with("cuda:0")


def test_recurrent_query_top_logprobs_pure_python_log_softmax_math() -> None:
    """Kills line 199 (- -> +) in recurrent."""
    backend = RecurrentPyTorchBackend(model_name="test-model")
    mock_model = MagicMock()
    mock_model.return_value = [[[2.0, 4.0]]]
    mock_tokenizer = MagicMock(return_value={"input_ids": [1]})
    mock_tokenizer.batch_decode.side_effect = lambda ids: ["token_0", "token_1"]

    backend._model = mock_model
    backend._tokenizer = mock_tokenizer

    with patch.dict(sys.modules, {"torch": None}):
        results = backend.query_top_logprobs("prompt", top_k=2)

    assert results[0]["logprob"] < 0.0, "Log probabilities must be negative"
    assert pytest.approx(results[0]["logprob"], abs=1e-3) == -0.126928


def test_recurrent_generate_max_new_tokens_kwarg() -> None:
    """Kills line 261 (or -> and) in recurrent."""
    backend = RecurrentPyTorchBackend(model_name="test-model")
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_tokenizer = MagicMock()
    mock_input_tensor = MagicMock()
    mock_input_tensor.shape = (1, 1)
    mock_inputs = {"input_ids": mock_input_tensor}
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "generated"

    backend._model = mock_model
    backend._tokenizer = mock_tokenizer

    backend.generate("prompt", max_tokens=50)
    call_kwargs = mock_model.generate.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 50, f"Expected 50, got {call_kwargs['max_new_tokens']}"


def test_recurrent_generate_signature_without_kwargs_no_num_loops() -> None:
    """Kills line 269 (== -> !=) in recurrent."""
    backend = RecurrentPyTorchBackend(model_name="test-model", num_loops=4)
    mock_model = MagicMock()

    def strict_generate(input_ids: Any, max_new_tokens: int = 10, do_sample: bool = False) -> Any:
        return [[10, 20]]
    mock_model.generate = strict_generate

    mock_input_tensor = MagicMock()
    mock_input_tensor.shape = (1, 1)
    mock_inputs = {"input_ids": mock_input_tensor}
    mock_tokenizer = MagicMock(return_value=mock_inputs)
    mock_tokenizer.decode.return_value = "out"

    backend._model = mock_model
    backend._tokenizer = mock_tokenizer

    text, _tokens = backend.generate("prompt")
    assert text == "out"


def test_recurrent_generate_device_not_in_auto_to_device() -> None:
    """Kills line 287 (not in -> in) and line 288 (is not -> is) in recurrent."""
    backend = RecurrentPyTorchBackend(model_name="test-model", device="cpu")
    mock_model = MagicMock(spec=["generate"])  # No .device attribute
    mock_model.generate.return_value = [[1, 2]]

    mock_inputs = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.shape = (1, 1)
    mock_inputs.__getitem__.return_value = mock_tensor
    mock_inputs.__contains__.return_value = True
    mock_tokenizer = MagicMock(return_value=mock_inputs)
    mock_tokenizer.decode.return_value = "done"

    backend._model = mock_model
    backend._tokenizer = mock_tokenizer

    backend.generate("prompt")
    mock_inputs.to.assert_called_with("cpu")


# ============================================================================
# Section 7: logitgate.hardware.memory_guard
# ============================================================================

def test_memory_guard_inspect_free_mb_boundary_tie() -> None:
    """Kills line 52 (< -> <=) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=1024.0)

    free_bytes = int(1024.0 * 1024 * 1024)
    total_bytes = int(10000.0 * 1024 * 1024)
    reserved_bytes = int(1000.0 * 1024 * 1024)

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)), \
         patch("torch.cuda.memory_allocated", return_value=500), \
         patch("torch.cuda.memory_reserved", return_value=reserved_bytes):
        status = guard.inspect()
        assert status.spillover_risk_detected is False


def test_memory_guard_inspect_reserved_mb_boundary_tie() -> None:
    """Kills line 52 (> -> >=) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, min_free_headroom_mb=500.0)

    total_bytes = int(10000.0 * 1024 * 1024)
    reserved_bytes = int(9600.0 * 1024 * 1024)  # exactly 96%
    free_bytes = int(2000.0 * 1024 * 1024)      # safely above headroom

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.mem_get_info", return_value=(free_bytes, total_bytes)), \
         patch("torch.cuda.memory_allocated", return_value=500), \
         patch("torch.cuda.memory_reserved", return_value=reserved_bytes):
        status = guard.inspect()
        assert status.spillover_risk_detected is False


def test_memory_guard_spillover_risk_or_logic() -> None:
    """Kills line 69 (or -> and) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, auto_empty_cache=True)

    status_risk = VRAMWatermarkStatus(
        allocated_mb=0.0,
        reserved_mb=0.0,
        free_physical_mb=0.0,
        total_physical_mb=0.0,
        spillover_risk_detected=True,
    )
    with patch.object(guard, "inspect", return_value=status_risk), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        with guard.guard_forward_pass():
            pass
        mock_empty.assert_called_once()


def test_memory_guard_total_mb_and_free_mb_logic() -> None:
    """Kills line 70 (and -> or) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, auto_empty_cache=True, min_free_headroom_mb=100.0)

    status_safe = VRAMWatermarkStatus(
        allocated_mb=100.0,
        reserved_mb=200.0,
        free_physical_mb=5000.0,
        total_physical_mb=10000.0,
        spillover_risk_detected=False,
    )
    with patch.object(guard, "inspect", return_value=status_safe), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        with guard.guard_forward_pass(estimated_kv_mb=50.0):
            pass
        mock_empty.assert_not_called()


def test_memory_guard_total_physical_mb_zero_boundary() -> None:
    """Kills line 70 (> -> >=) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, auto_empty_cache=True, min_free_headroom_mb=500.0)

    status_zero = VRAMWatermarkStatus(
        allocated_mb=0.0,
        reserved_mb=0.0,
        free_physical_mb=0.0,
        total_physical_mb=0.0,
        spillover_risk_detected=False,
    )
    with patch.object(guard, "inspect", return_value=status_zero), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        with guard.guard_forward_pass(estimated_kv_mb=100.0):
            pass
        mock_empty.assert_not_called()


def test_memory_guard_free_mb_kv_headroom_boundary_tie() -> None:
    """Kills line 71 (< -> <=) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, auto_empty_cache=True, min_free_headroom_mb=1024.0)

    status_exact = VRAMWatermarkStatus(
        allocated_mb=100.0,
        reserved_mb=500.0,
        free_physical_mb=1152.0,
        total_physical_mb=10000.0,
        spillover_risk_detected=False,
    )
    with patch.object(guard, "inspect", return_value=status_exact), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        with guard.guard_forward_pass(estimated_kv_mb=128.0):
            pass
        mock_empty.assert_not_called()


def test_memory_guard_headroom_kv_sum_arithmetic() -> None:
    """Kills line 71 (+ -> -) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, auto_empty_cache=True, min_free_headroom_mb=1000.0)

    status_between = VRAMWatermarkStatus(
        allocated_mb=100.0,
        reserved_mb=500.0,
        free_physical_mb=1200.0,
        total_physical_mb=10000.0,
        spillover_risk_detected=False,
    )
    with patch.object(guard, "inspect", return_value=status_between), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        with guard.guard_forward_pass(estimated_kv_mb=500.0):
            pass
        mock_empty.assert_called_once()


def test_memory_guard_has_risk_and_auto_empty_cache() -> None:
    """Kills line 73 (and -> or) in memory_guard."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    guard = PCIeSpilloverGuard(device_index=0, auto_empty_cache=False)

    status_risk = VRAMWatermarkStatus(
        allocated_mb=0.0,
        reserved_mb=0.0,
        free_physical_mb=0.0,
        total_physical_mb=10000.0,
        spillover_risk_detected=True,
    )
    with patch.object(guard, "inspect", return_value=status_risk), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        with guard.guard_forward_pass():
            pass
        mock_empty.assert_not_called()


# ============================================================================
# Section 8: logitgate.streaming
# ============================================================================

class DummyTestGate(BaseGate):
    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            status=DecisionStatus.PASSED,
            probability=0.95,
            passed=True,
            triggered_layer="dummy_layer",
            total_latency_ms=2.0,
        )


def test_streaming_evaluator_stride_chunks_integer_validation() -> None:
    """Kills line 291 (not x -> x) in streaming."""
    gate = DummyTestGate()
    evaluator = StreamGuard(gate=gate, stride_chunks=5)
    assert evaluator.stride_chunks == 5


def test_streaming_evaluator_max_wait_chars_boundary_tie() -> None:
    """Kills line 409 (>= -> >) in streaming."""
    gate = DummyTestGate()
    evaluator = StreamGuard(
        gate=gate,
        min_chars=10,
        stride_chars=10,
        clause_boundary=True,
        max_wait_chars=30,
    )

    is_ready = evaluator._should_check_now(
        chars_seen=50,
        chars_at_last_check=20,
        chunk_count=5,
        chunks_at_last_check=2,
        accumulated=["incomplete clause without punctuation"],
    )
    assert is_ready is True, "Exact match chars_delta == max_wait_chars must be ready under >="


def test_streaming_evaluator_abort_prob_threshold_boundary_tie() -> None:
    """Kills line 420 (<= -> <) in streaming."""
    gate = DummyTestGate()
    evaluator = StreamGuard(
        gate=gate,
        abort_prob_threshold=0.6,
    )

    verdict = Verdict(
        status=DecisionStatus.FAILED,
        probability=0.6,
        passed=False,
        triggered_layer="dummy_layer",
        total_latency_ms=1.0,
    )
    assert evaluator._is_midstream_strike(verdict) is True


def test_streaming_guarded_stream_final_chunk_index_arithmetic() -> None:
    """Kills line 615 (- -> +) in streaming."""
    gate = DummyTestGate()
    evaluator = StreamGuard(
        gate=gate,
        stride_chars=1000,  # Prevent mid-stream checks
    )

    chunks = ["first chunk ", "second chunk ", "third chunk"]  # 3 chunks
    stream_gen = evaluator.stream(chunks)
    consumed = list(stream_gen)
    assert consumed == chunks

    assert stream_gen.summary is not None
    records = stream_gen.summary.checks
    assert len(records) == 1
    assert records[0].chunk_index == 2, f"Expected chunk_index=2, got {records[0].chunk_index}"
