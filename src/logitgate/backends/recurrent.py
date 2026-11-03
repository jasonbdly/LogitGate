"""
PyTorch & HuggingFace backend with elastic recurrent layer-looping support.

Designed for architectures like NVIDIA-Nemotron-Labs Dory-4B and Universal Transformers
that support dynamic layer looping (`num_loops`) to perform latent reasoning in hidden
state space before producing first-token logits.
"""

from __future__ import annotations

import math
from typing import Any

from logitgate.backends.base import GenerativeBackend, LogitBackend


class RecurrentPyTorchBackend(LogitBackend, GenerativeBackend):
    """
    Inference backend for PyTorch/Transformers models with elastic recurrent layer looping.

    Supports architectures like NVIDIA's Dory (e.g. ``nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16``)
    where hidden states are recirculated through a recurrent layer block for ``num_loops`` iterations
    before computing first-token logits.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local directory (default: ``"nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"``).
    num_loops : int
        Number of internal recurrent loops to execute per token (default: 2; recommended: 1 to 4).
        Higher values increase latent reasoning depth at the cost of ~30-60% prefill compute.
    device : str
        Target compute device (``"cuda"``, ``"cpu"``, or ``"auto"``).
    model : Any, optional
        Pre-instantiated model instance (useful for testing or shared memory pipelines).
    tokenizer : Any, optional
        Pre-instantiated tokenizer instance.
    torch_dtype : Any, optional
        PyTorch dtype (e.g. ``torch.bfloat16``).
    """

    def __init__(
        self,
        model_name: str = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        num_loops: int = 2,
        device: str = "cuda",
        model: Any = None,
        tokenizer: Any = None,
        torch_dtype: Any = None,
    ) -> None:
        if num_loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {num_loops}")

        self.model_name = model_name
        self.num_loops = num_loops
        self.device = device
        self._model = model
        self._tokenizer = tokenizer
        self.torch_dtype = torch_dtype

    def __repr__(self) -> str:
        return f"RecurrentPyTorchBackend(model='{self.model_name}', num_loops={self.num_loops}, device='{self.device}')"

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Lazy-loads model and tokenizer if not already provided."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "PyTorch and Transformers are required for RecurrentPyTorchBackend. "
                "Install them via `pip install logitgate[recurrent]` or `pip install torch transformers`."
            ) from err

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if self._model is None:
            target_dtype = self.torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=target_dtype,
                trust_remote_code=True,
                device_map=self.device if self.device != "auto" else "auto",
            )
            if self.device not in ("auto", "cuda") and hasattr(self._model, "to"):
                self._model = self._model.to(self.device)

        return self._model, self._tokenizer

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Executes a forward pass with ``num_loops`` latent recurrence and extracts first-token logprobs.
        """
        model, tokenizer = self._ensure_loaded()

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to") and self.device not in ("auto",):
            inputs = inputs.to(model.device if hasattr(model, "device") else self.device)

        # Forward pass passing num_loops if supported by model signature
        import inspect

        kwargs: dict[str, Any] = {}
        try:
            fn = model.forward if hasattr(model, "forward") else model
            sig = inspect.signature(fn)
            if "num_loops" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["num_loops"] = self.num_loops
        except (ValueError, TypeError):
            kwargs["num_loops"] = self.num_loops

        outputs = model(**inputs, **kwargs)
        if hasattr(outputs.logits, "ndim"):
            logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)
        else:
            logits = outputs.logits[0][-1]

        # Compute log_softmax
        try:
            import torch

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        except (ImportError, AttributeError, TypeError, ValueError):
            # Fallback numpy log_softmax if torch is mocked or unavailable
            max_l = float(max(logits))
            exp_l = [math.exp(float(x) - max_l) for x in logits]
            sum_exp = sum(exp_l)
            log_probs = [float(x) - max_l - math.log(sum_exp) for x in logits]

        results: list[dict[str, Any]] = []

        if allowed_tokens:
            for tok_str in allowed_tokens:
                # Tokenize variant (both raw and leading-space)
                token_ids = tokenizer.encode(tok_str, add_special_tokens=False)
                if token_ids:
                    tid = token_ids[-1]
                    lp = float(log_probs[tid])
                    results.append({"token": tok_str, "logprob": lp})
            if results:
                return results

        # General top_k
        try:
            import torch

            if isinstance(log_probs, torch.Tensor):
                top_vals, top_indices = torch.topk(log_probs, min(top_k, log_probs.size(-1)))
                for lp, idx in zip(top_vals.tolist(), top_indices.tolist()):
                    token_str = tokenizer.decode([idx])
                    results.append({"token": token_str, "logprob": float(lp)})
                return results
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

        # Fallback list sorting
        indexed = sorted(enumerate(log_probs), key=lambda x: float(x[1]), reverse=True)
        for idx, lp in indexed[:top_k]:
            token_str = tokenizer.decode([idx]) if hasattr(tokenizer, "decode") else str(idx)
            results.append({"token": token_str, "logprob": float(lp)})

        return results

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        """Autoregressive generation with num_loops recurrent reasoning."""
        model, tokenizer = self._ensure_loaded()
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to") and self.device not in ("auto",):
            inputs = inputs.to(model.device if hasattr(model, "device") else self.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or 256,
            "do_sample": False,
        }
        import inspect

        try:
            sig = inspect.signature(model.generate)
            if "num_loops" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                gen_kwargs["num_loops"] = self.num_loops
        except (ValueError, TypeError):
            gen_kwargs["num_loops"] = self.num_loops

        outputs = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, len(generated_ids)
