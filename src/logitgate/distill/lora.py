from __future__ import annotations

import asyncio
import datetime
import math
import os
import shlex
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from logitgate.backends.base import LogitBackend
from logitgate.calibration.profile import CalibrationProfile
from logitgate.debiasing.inversion import compile_binary_prompt
from logitgate.distill._io import atomic_write_json_nofollow, read_json_object_nofollow
from logitgate.distill.anchor_bank import DistilledAnchorBank, _load_generic_cases
from logitgate.distill.bundle import DistillationBundle
from logitgate.distill.meta_calibrator import LogitMetaCalibrator, fit_meta_calibrator
from logitgate.exceptions import BackendError, ConfigurationError

_BINARY_SYS_PROMPT = "Respond with a single character only (A, B). Do not think or explain."


def _require_hf_deps(*, need_peft: bool = True) -> tuple[Any, Any, Any]:
    """Lazily imports ``torch``, ``transformers``, and optionally ``peft``."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ConfigurationError(
            "Open-weights distillation requires PyTorch and HuggingFace Transformers. "
            "Install via: pip install 'logitgate[distill]'"
        ) from exc

    peft_mod: Any = None
    if need_peft:
        try:
            import peft
        except ImportError as exc:
            raise ConfigurationError(
                "LoRA adapter training/loading requires HuggingFace PEFT. "
                "Install via: pip install 'logitgate[distill]'"
            ) from exc
        peft_mod = peft
    return torch, transformers, peft_mod


class LoRADistillConfig(BaseModel):
    """Architecture-agnostic configuration for open-weights BYO LoRA distillation."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: str | list[str] = "all-linear"
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=256)
    grad_accum_steps: int = Field(default=2, ge=1, le=128)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_seq_len: int = Field(default=1024, ge=64, le=32768)
    seed: int = 42
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"

    @model_validator(mode="after")
    def _validate_target_modules(self) -> LoRADistillConfig:
        if isinstance(self.target_modules, str):
            if not self.target_modules.strip():
                raise ConfigurationError("target_modules string cannot be empty.")
        elif isinstance(self.target_modules, list) and (
            not self.target_modules
            or any(not isinstance(m, str) or not m.strip() for m in self.target_modules)
        ):
            raise ConfigurationError(
                "target_modules list must contain one or more non-empty module names."
            )
        return self


class SymmetricLoRAExample(BaseModel):
    """Single prompt + single-token target (`A` or `B`) for first-token symmetric SFT."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    polarity: Literal["forward", "reverse"]
    system_prompt: str = _BINARY_SYS_PROMPT
    user_prompt: str
    target_token: Literal["A", "B"]
    ground_truth_passed: bool


class LoRADistillManifest(BaseModel):
    """Metadata and BYO multi-runtime deployment instructions emitted by `train_distillation_lora`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    base_model: str
    adapter_dir: str
    bundle_path: str | None = None
    profile_path: str | None = None
    ollama_modelfile_path: str
    vllm_launch_script_path: str
    llamacpp_launch_script_path: str
    vllm_cli_args: list[str]
    llamacpp_cli_args: list[str]
    num_training_examples: int
    epochs: int
    rank: int
    lora_alpha: int
    target_modules: str | list[str]
    initial_loss: float
    final_loss: float

    def save(self, path: str | Path) -> Path:
        return atomic_write_json_nofollow(
            path,
            self.model_dump(mode="json"),
            artifact_name="LoRADistillManifest",
        )

    @classmethod
    def load(cls, path: str | Path) -> LoRADistillManifest:
        data = read_json_object_nofollow(path, artifact_name="LoRADistillManifest")
        return cls.model_validate(data)


def prepare_symmetric_lora_examples(
    source: str | Path | Sequence[Any],
    *,
    anchor_bank: DistilledAnchorBank | None = None,
    teacher_labels: Mapping[str, bool] | None = None,
) -> list[SymmetricLoRAExample]:
    """Compiles `2x` symmetric (`A=Pass/B=Fail` and `B=Pass/A=Fail`) first-token training rows."""
    items = _load_generic_cases(source)

    if not items:
        raise ConfigurationError("Cannot prepare LoRA examples from empty dataset.")

    examples: list[SymmetricLoRAExample] = []
    for row in items:
        item_id = row["id"]
        passed = (
            bool(teacher_labels[item_id])
            if teacher_labels is not None and item_id in teacher_labels
            else bool(row["expected"])
        )
        anchor_str = (
            anchor_bank.format_anchor_block(row["rubric"], row["candidate"])
            if anchor_bank is not None
            else None
        )
        fwd_prompt = compile_binary_prompt(
            rubric=row["rubric"],
            candidate=row["candidate"],
            pass_token="A",
            fail_token="B",
            query_first=True,
            anchor_block=anchor_str,
        )
        rev_prompt = compile_binary_prompt(
            rubric=row["rubric"],
            candidate=row["candidate"],
            pass_token="B",
            fail_token="A",
            query_first=True,
            anchor_block=anchor_str,
        )
        examples.append(
            SymmetricLoRAExample(
                example_id=f"{item_id}__fwd",
                polarity="forward",
                user_prompt=fwd_prompt,
                target_token="A" if passed else "B",
                ground_truth_passed=passed,
            )
        )
        examples.append(
            SymmetricLoRAExample(
                example_id=f"{item_id}__rev",
                polarity="reverse",
                user_prompt=rev_prompt,
                target_token="B" if passed else "A",
                ground_truth_passed=passed,
            )
        )
    return examples


def _write_text_nofollow(path: Path, content: str, *, mode: int = 0o644) -> None:
    for anc in (path.parent, *path.parent.parents):
        if anc.exists() and anc.is_symlink():
            raise ConfigurationError(f"Refusing to write under symlinked directory: {anc}")
    if path.is_symlink():
        raise ConfigurationError(f"Refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def emit_deployment_manifests(
    output_dir: str | Path,
    *,
    base_model: str,
    adapter_subdir: str = "adapter",
    ollama_base_model: str | None = None,
    lora_name: str = "logitgate-student",
) -> dict[str, Any]:
    """Writes ready-to-run BYO serving configs for Ollama, vLLM, and llama.cpp."""
    out = Path(output_dir)
    adapter_path = out / adapter_subdir
    ollama_base = ollama_base_model or base_model

    modelfile_path = out / "Modelfile.ollama"
    modelfile_body = (
        f"# LogitGate Distilled Student Adapter (Ollama)\n"
        f"# Build via: ollama create {lora_name} -f {modelfile_path.name}\n"
        f"FROM {ollama_base}\n"
        f"ADAPTER ./{adapter_subdir}\n"
        f"PARAMETER temperature 0\n"
    )
    _write_text_nofollow(modelfile_path, modelfile_body, mode=0o644)

    vllm_args = [
        "vllm",
        "serve",
        base_model,
        "--enable-lora",
        "--lora-modules",
        f"{lora_name}={adapter_path}",
    ]
    vllm_script_path = out / "serve_vllm.sh"
    vllm_body = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"exec {' '.join(shlex.quote(a) for a in vllm_args)} \"$@\"\n"
    )
    _write_text_nofollow(vllm_script_path, vllm_body, mode=0o755)

    llamacpp_args = [
        "llama-server",
        "-m",
        "<path-to-base-model.gguf>",
        "--lora",
        str(out / "adapter.gguf"),
        "--port",
        "8080",
    ]
    llamacpp_script_path = out / "serve_llamacpp.sh"
    llamacpp_body = (
        "#!/usr/bin/env bash\n"
        "# Convert HuggingFace PEFT adapter to GGUF first via llama.cpp:\n"
        f"#   python convert_lora_to_gguf.py {shlex.quote(str(adapter_path))} --outfile {shlex.quote(str(out / 'adapter.gguf'))}\n"
        "set -euo pipefail\n"
        "BASE_GGUF=\"${1:?Usage: ./serve_llamacpp.sh /path/to/base-model.gguf}\"\n"
        "shift\n"
        f"exec llama-server -m \"$BASE_GGUF\" --lora {shlex.quote(str(out / 'adapter.gguf'))} --port 8080 \"$@\"\n"
    )
    _write_text_nofollow(llamacpp_script_path, llamacpp_body, mode=0o755)

    return {
        "ollama_modelfile_path": str(modelfile_path),
        "vllm_launch_script_path": str(vllm_script_path),
        "llamacpp_launch_script_path": str(llamacpp_script_path),
        "vllm_cli_args": vllm_args,
        "llamacpp_cli_args": llamacpp_args,
    }


def _encode_prompt_ids(tokenizer: Any, system_prompt: str, user_prompt: str) -> list[int]:
    """Encodes a binary verification prompt using ``chat_template`` when present."""
    chat_tmpl = getattr(tokenizer, "chat_template", None)
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if chat_tmpl and callable(apply_fn):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            encoded = apply_fn(messages, tokenize=True, add_generation_prompt=True)
            if isinstance(encoded, list) and encoded and isinstance(encoded[0], int):
                return encoded
            input_ids = getattr(encoded, "input_ids", None)
            if isinstance(input_ids, list) and input_ids:
                return list(input_ids[0]) if isinstance(input_ids[0], list) else list(input_ids)
        except (ValueError, TypeError, KeyError, RuntimeError):
            encoded = None
    full_text = f"{system_prompt}\n\n{user_prompt}\n"
    ids = tokenizer.encode(full_text, add_special_tokens=True)
    return [int(i) for i in ids]


def _resolve_single_token_id(tokenizer: Any, token_text: str) -> int:
    for candidate_str in (token_text, f" {token_text}"):
        ids = tokenizer.encode(candidate_str, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    ids_fallback = tokenizer.encode(token_text, add_special_tokens=False)
    if not ids_fallback:
        raise ConfigurationError(f"Tokenizer returned empty IDs for token {token_text!r}.")
    return int(ids_fallback[-1])


class HuggingFaceLogitBackend(LogitBackend):
    """In-process BYO `LogitBackend` for any HuggingFace CausalLM + optional PEFT LoRA adapter."""

    prefers_sequential_prefix_warmup = False

    def __init__(
        self,
        model_obj: Any,
        tokenizer: Any,
        *,
        model_name: str | None = None,
        lora_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_obj = model_obj
        self._tokenizer = tokenizer
        self.model = model_name or getattr(getattr(model_obj, "config", None), "_name_or_path", "hf-local")
        self.lora_path = lora_path
        self._device = device
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        base_model: str | Path,
        *,
        adapter_path: str | Path | None = None,
        device: str = "auto",
        dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto",
        trust_remote_code: bool = False,
    ) -> HuggingFaceLogitBackend:
        torch, transformers, peft_mod = _require_hf_deps(need_peft=adapter_path is not None)
        resolved_dev = _resolve_torch_device(torch, device)
        resolved_dtype = _resolve_torch_dtype(torch, dtype, resolved_dev)

        tok = transformers.AutoTokenizer.from_pretrained(
            str(base_model),
            trust_remote_code=trust_remote_code,
        )
        mdl = transformers.AutoModelForCausalLM.from_pretrained(
            str(base_model),
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
        )
        mdl.to(resolved_dev)
        if adapter_path is not None:
            mdl = peft_mod.PeftModel.from_pretrained(mdl, str(adapter_path))
            mdl.to(resolved_dev)
        mdl.eval()
        return cls(
            mdl,
            tok,
            model_name=str(base_model),
            lora_path=str(adapter_path) if adapter_path is not None else None,
            device=resolved_dev,
        )

    def __repr__(self) -> str:
        return f"HuggingFaceLogitBackend(model={self.model!r}, lora_path={self.lora_path!r})"

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        torch, _, _ = _require_hf_deps(need_peft=False)
        sys_prompt = (
            f"Respond with a single character only ({', '.join(allowed_tokens)}). Do not think or explain."
            if allowed_tokens
            else _BINARY_SYS_PROMPT
        )
        with self._lock:
            input_ids = _encode_prompt_ids(self._tokenizer, sys_prompt, prompt)
            if not input_ids:
                raise BackendError("Tokenizer produced empty input_ids for prompt.")
            dev = self._device
            if dev is None:
                try:
                    dev = str(next(self._model_obj.parameters()).device)
                except StopIteration:
                    dev = "cpu"
            tensor_ids = torch.tensor([input_ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                outputs = self._model_obj(input_ids=tensor_ids)
                next_logits = outputs.logits[0, -1, :].float()
                log_probs = torch.log_softmax(next_logits, dim=-1)

            vocab_len = int(log_probs.shape[-1])
            eff_k = max(1, min(int(top_k), vocab_len))
            top_vals, top_indices = torch.topk(log_probs, k=eff_k)
            selected_ids: dict[int, float] = {
                int(idx_t.item()): float(val_t.item())
                for val_t, idx_t in zip(top_vals, top_indices)
            }
            # Ensure surface variants of allowed_tokens are observable even on uncalibrated tiny models.
            if allowed_tokens:
                for tok_txt in allowed_tokens:
                    for surface in (tok_txt, f" {tok_txt}"):
                        enc = self._tokenizer.encode(surface, add_special_tokens=False)
                        if len(enc) == 1 and 0 <= int(enc[0]) < vocab_len:
                            tid = int(enc[0])
                            selected_ids.setdefault(tid, float(log_probs[tid].item()))

            results: list[dict[str, Any]] = []
            for tid, lp in sorted(selected_ids.items(), key=lambda kv: kv[1], reverse=True):
                tok_str = str(self._tokenizer.decode([tid], skip_special_tokens=False))
                if math.isfinite(lp):
                    results.append({"token": tok_str, "logprob": lp})
            return results

    async def aquery_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.query_top_logprobs,
            prompt,
            top_k=top_k,
            allowed_tokens=allowed_tokens,
            **kwargs,
        )


def _resolve_torch_device(torch: Any, device: str) -> str:
    if device != "auto":
        return device
    if bool(torch.cuda.is_available()):
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    return "cpu"


def _resolve_torch_dtype(torch: Any, dtype: str, device: str) -> Any:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if bool(torch.cuda.is_bf16_supported()) else torch.float16
    return torch.float32


def train_distillation_lora(
    base_model: str | Path | Any,
    dataset: str | Path | Sequence[Any],
    output_dir: str | Path,
    *,
    tokenizer: Any = None,
    config: LoRADistillConfig | None = None,
    anchor_bank: DistilledAnchorBank | None = None,
    teacher_labels: Mapping[str, bool] | None = None,
    calibrate_alpha: float | None = 0.10,
    ollama_base_model: str | None = None,
    trust_remote_code: bool = False,
) -> LoRADistillManifest:
    """Architecture-agnostic BYO LoRA compiler targeting ``all-linear`` first-token log-odds.

    Supports any HuggingFace causal LM checkpoint ID/path (Qwen, Llama, Gemma, Mistral, Phi, etc.)
    or an already-loaded ``(model, tokenizer)`` pair for offline/air-gapped environments.
    """
    torch, transformers, peft_mod = _require_hf_deps(need_peft=True)
    cfg = config or LoRADistillConfig()
    out_dir = Path(output_dir)
    for anc in (out_dir, out_dir.parent, *out_dir.parent.parents):
        if anc.exists() and anc.is_symlink():
            raise ConfigurationError(f"Refusing to write LoRA artifacts under symlink: {anc}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    resolved_dev = _resolve_torch_device(torch, cfg.device)
    resolved_dtype = _resolve_torch_dtype(torch, cfg.dtype, resolved_dev)

    if isinstance(base_model, (str, Path)):
        base_model_name = str(base_model)
        tok = tokenizer or transformers.AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        mdl = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
        )
    else:
        if tokenizer is None:
            raise ConfigurationError(
                "When passing an in-memory model object to train_distillation_lora, tokenizer=... is required."
            )
        mdl = base_model
        tok = tokenizer
        base_model_name = str(
            getattr(getattr(mdl, "config", None), "_name_or_path", None) or "in-memory-causal-lm"
        )

    examples = prepare_symmetric_lora_examples(
        dataset,
        anchor_bank=anchor_bank,
        teacher_labels=teacher_labels,
    )

    lora_cfg = peft_mod.LoraConfig(
        task_type=peft_mod.TaskType.CAUSAL_LM,
        r=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    peft_model = peft_mod.get_peft_model(mdl, lora_cfg)
    peft_model.to(resolved_dev)
    peft_model.train()

    id_a = _resolve_single_token_id(tok, "A")
    id_b = _resolve_single_token_id(tok, "B")
    pad_id = int(getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None) or 0)

    encoded_samples: list[tuple[list[int], list[int]]] = []
    for ex in examples:
        prompt_ids = _encode_prompt_ids(tok, ex.system_prompt, ex.user_prompt)
        if len(prompt_ids) >= cfg.max_seq_len:
            # Keep tail containing the decision mapping and candidate suffix.
            prompt_ids = prompt_ids[-(cfg.max_seq_len - 1) :]
        target_id = id_a if ex.target_token == "A" else id_b
        seq_ids = prompt_ids + [target_id]
        seq_labels = [-100] * len(prompt_ids) + [target_id]
        encoded_samples.append((seq_ids, seq_labels))

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ConfigurationError(
            f"PEFT target_modules={cfg.target_modules!r} matched 0 trainable parameters on {base_model_name}."
        )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    n_samples = len(encoded_samples)
    step_losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n_samples, generator=rng).tolist()
        for start in range(0, n_samples, cfg.batch_size):
            batch_idx = perm[start : start + cfg.batch_size]
            max_len = max(len(encoded_samples[i][0]) for i in batch_idx)
            input_batch: list[list[int]] = []
            mask_batch: list[list[int]] = []
            label_batch: list[list[int]] = []
            for i in batch_idx:
                s_ids, s_lbls = encoded_samples[i]
                pad_n = max_len - len(s_ids)
                # Left-pad so causal position encodings align cleanly across variable prompt lengths.
                input_batch.append([pad_id] * pad_n + s_ids)
                mask_batch.append([0] * pad_n + [1] * len(s_ids))
                label_batch.append([-100] * pad_n + s_lbls)

            t_in = torch.tensor(input_batch, dtype=torch.long, device=resolved_dev)
            t_mask = torch.tensor(mask_batch, dtype=torch.long, device=resolved_dev)
            t_lbl = torch.tensor(label_batch, dtype=torch.long, device=resolved_dev)

            out = peft_model(input_ids=t_in, attention_mask=t_mask, labels=t_lbl)
            loss = out.loss
            loss_val = float(loss.detach().cpu().item())
            if math.isfinite(loss_val):
                step_losses.append(loss_val)

            (loss / float(cfg.grad_accum_steps)).backward()
            micro_step += 1
            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    if micro_step % cfg.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    peft_model.eval()
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))

    # Fit post-LoRA CalibrationProfile + LogitMetaCalibrator if >= 10 items are present.
    raw_items = _load_generic_cases(dataset)
    bundle_path_str: str | None = None
    profile_path_str: str | None = None
    if len(raw_items) >= 10:
        hf_backend = HuggingFaceLogitBackend(
            peft_model,
            tok,
            model_name=base_model_name,
            lora_path=str(adapter_dir),
            device=resolved_dev,
        )
        meta_cal: LogitMetaCalibrator = fit_meta_calibrator(
            hf_backend,
            raw_items,
            model=base_model_name,
            anchor_bank=anchor_bank,
            teacher_labels=teacher_labels,
            target_risk_alpha=calibrate_alpha,
        )
        # Also fit a standard scalar CalibrationProfile for drop-in profile compatibility.
        from logitgate.debiasing.softmax import compute_binary_log_odds

        lo_samples: list[tuple[float, float, int]] = []
        for row in raw_items:
            p_fwd = compile_binary_prompt(
                rubric=row["rubric"],
                candidate=row["candidate"],
                pass_token="A",
                fail_token="B",
                query_first=True,
                anchor_block=(
                    anchor_bank.format_anchor_block(row["rubric"], row["candidate"])
                    if anchor_bank is not None
                    else None
                ),
            )
            p_rev = compile_binary_prompt(
                rubric=row["rubric"],
                candidate=row["candidate"],
                pass_token="B",
                fail_token="A",
                query_first=True,
                anchor_block=(
                    anchor_bank.format_anchor_block(row["rubric"], row["candidate"])
                    if anchor_bank is not None
                    else None
                ),
            )
            lf, _ = compute_binary_log_odds(
                hf_backend.query_top_logprobs(p_fwd, top_k=10, allowed_tokens=["A", "B"]),
                pass_token="A",
                fail_token="B",
            )
            lr, _ = compute_binary_log_odds(
                hf_backend.query_top_logprobs(p_rev, top_k=10, allowed_tokens=["A", "B"]),
                pass_token="B",
                fail_token="A",
            )
            lbl = (
                bool(teacher_labels[row["id"]])
                if teacher_labels is not None and row["id"] in teacher_labels
                else bool(row["expected"])
            )
            lo_samples.append((lf, lr, 1 if lbl else 0))

        prof = CalibrationProfile.from_log_odds_samples(
            lo_samples,
            target_risk_alpha=calibrate_alpha,
            model=base_model_name,
            backend="hf-lora",
        )
        prof_path = out_dir / "calibration_profile.json"
        prof.save(prof_path)
        profile_path_str = str(prof_path)

        bundle = DistillationBundle(
            student_model=base_model_name,
            anchor_bank=anchor_bank,
            meta_calibrator=meta_cal,
            profile=prof,
        )
        b_path = out_dir / "distillation_bundle.json"
        bundle.save(b_path)
        bundle_path_str = str(b_path)

    deploy_info = emit_deployment_manifests(
        out_dir,
        base_model=base_model_name,
        adapter_subdir="adapter",
        ollama_base_model=ollama_base_model,
    )
    manifest = LoRADistillManifest(
        base_model=base_model_name,
        adapter_dir=str(adapter_dir),
        bundle_path=bundle_path_str,
        profile_path=profile_path_str,
        ollama_modelfile_path=deploy_info["ollama_modelfile_path"],
        vllm_launch_script_path=deploy_info["vllm_launch_script_path"],
        llamacpp_launch_script_path=deploy_info["llamacpp_launch_script_path"],
        vllm_cli_args=deploy_info["vllm_cli_args"],
        llamacpp_cli_args=deploy_info["llamacpp_cli_args"],
        num_training_examples=len(encoded_samples),
        epochs=cfg.epochs,
        rank=cfg.rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        initial_loss=round(step_losses[0], 6) if step_losses else 0.0,
        final_loss=round(step_losses[-1], 6) if step_losses else 0.0,
    )
    manifest.save(out_dir / "deployment_manifest.json")
    return manifest
