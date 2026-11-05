from __future__ import annotations

from typing import Any

from logitgate.debiasing.permutation import (
    DebiasedLogitResult,
    SurfaceTokenMap,
    compute_symmetric_debiased_verdict,
)
from logitgate.hardware.memory_guard import PCIeSpilloverGuard


class PrefixBroadcastLogitEvaluator:
    """Executes Query-First Attention Inversion + Symmetric Label Permutation

    in ~1.02x the cost of a single forward pass via KV-cache prefix broadcasting.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = device
        elif hasattr(model, "device"):
            self.device = model.device
        else:
            try:
                import torch

                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            except (ImportError, AttributeError):
                self.device = "cpu"

    @staticmethod
    def build_inverted_prefix_and_suffixes(
        rubric: str,
        candidate: Any = None,
        candidate_input: str = "",
        candidate_output: str = "",
    ) -> tuple[str, str, str]:
        """Constructs the Query-First Attention Inverted prefix and the two

        symmetric label binding suffixes.
        """
        rubric_str = rubric.strip()
        if candidate_input or candidate_output:
            payload = (
                f"<|context|>\n{candidate_input.strip()}\n</|context|>\n"
                f"<|candidate|>\n{candidate_output.strip()}\n</|candidate|>\n"
            )
        else:
            cand_str = str(candidate).strip() if candidate is not None else ""
            payload = f"<|candidate|>\n{cand_str}\n</|candidate|>\n"

        shared_prefix = (
            "<|system|>\n"
            "You are a deterministic evaluation engine. Read the evaluation rubric FIRST, "
            "and evaluate the subsequent candidate payload strictly against that rubric.\n"
            f"<|rubric|>\n{rubric_str}\n</|rubric|>\n"
            f"{payload}"
        )

        suffix_forward = (
            "<|decision_binding|>\n"
            "Map 'A' = PASS (satisfies rubric).\n"
            "Map 'B' = FAIL (violates rubric).\n"
            "Respond with a single token ('A' or 'B'):"
        )

        suffix_permuted = (
            "<|decision_binding|>\n"
            "Map 'A' = FAIL (violates rubric).\n"
            "Map 'B' = PASS (satisfies rubric).\n"
            "Respond with a single token ('A' or 'B'):"
        )
        return shared_prefix, suffix_forward, suffix_permuted

    def evaluate_single_pass_permuted(
        self,
        rubric: str,
        candidate: Any = None,
        candidate_input: str = "",
        candidate_output: str = "",
    ) -> tuple[Any, Any]:
        """1. Runs prefill on shared_prefix with batch_size=1.

        2. Expands past_key_values to batch_size=2.
        3. Runs a single batched forward pass on [suffix_forward, suffix_permuted].
        4. Returns the final-token logit vectors for both permutations.
        """
        import torch

        prefix_str, suffix_fwd_str, suffix_perm_str = self.build_inverted_prefix_and_suffixes(
            rubric=rubric,
            candidate=candidate,
            candidate_input=candidate_input,
            candidate_output=candidate_output,
        )

        # Step 1: Single prefill over shared prefix
        encoded_prefix = self.tokenizer.encode(
            prefix_str, add_special_tokens=True, return_tensors="pt"
        )
        if isinstance(encoded_prefix, torch.Tensor):
            prefix_ids = encoded_prefix.to(self.device)
        else:
            prefix_ids = torch.tensor(encoded_prefix, dtype=torch.long, device=self.device).unsqueeze(0)

        guard = PCIeSpilloverGuard()
        with torch.inference_mode(), guard.guard_forward_pass():
            prefix_out = self.model(
                input_ids=prefix_ids,
                use_cache=True,
                return_dict=True,
            )
            past_kv = prefix_out.past_key_values

            # Step 2: Tokenize suffixes
            suf_fwd_ids = self.tokenizer.encode(suffix_fwd_str, add_special_tokens=False)
            suf_perm_ids = self.tokenizer.encode(suffix_perm_str, add_special_tokens=False)

            if len(suf_fwd_ids) != len(suf_perm_ids):
                max_len = max(len(suf_fwd_ids), len(suf_perm_ids))
                pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(
                    self.tokenizer, "eos_token_id", 0
                )
                suf_fwd_ids = [pad_id] * (max_len - len(suf_fwd_ids)) + suf_fwd_ids
                suf_perm_ids = [pad_id] * (max_len - len(suf_perm_ids)) + suf_perm_ids

            suffix_batch = torch.tensor(
                [suf_fwd_ids, suf_perm_ids], dtype=torch.long, device=self.device
            )

            # Step 3: Broadcast past_key_values across batch dimension (B=1 -> B=2)
            if hasattr(past_kv, "batch_repeat_interleave"):
                expanded_kv = past_kv.batch_repeat_interleave(2)
            elif isinstance(past_kv, tuple):
                expanded_kv = tuple(
                    (
                        (
                            k.expand(2, *([-1] * (k.ndim - 1))).contiguous()
                            if hasattr(k, "expand") and k.ndim > 0
                            else k
                        ),
                        (
                            v.expand(2, *([-1] * (v.ndim - 1))).contiguous()
                            if hasattr(v, "expand") and v.ndim > 0
                            else v
                        ),
                    )
                    for k, v in past_kv
                )
            else:
                expanded_kv = past_kv

            # Step 4: Evaluate both suffixes in parallel
            suffix_out = self.model(
                input_ids=suffix_batch,
                past_key_values=expanded_kv,
                use_cache=False,
                return_dict=True,
            )

            logits = getattr(suffix_out, "logits", suffix_out)
            final_logits = logits[:, -1, :]
            return final_logits[0], final_logits[1]

    def evaluate_debiased(
        self,
        rubric: str,
        candidate: Any = None,
        candidate_input: str = "",
        candidate_output: str = "",
        token_map: SurfaceTokenMap | None = None,
        temperature: float = 1.0,
    ) -> DebiasedLogitResult:
        """Convenience method combining prefix broadcast inference with

        Symmetric Label Permutation log-odds debiasing.
        """
        l1, l2 = self.evaluate_single_pass_permuted(
            rubric=rubric,
            candidate=candidate,
            candidate_input=candidate_input,
            candidate_output=candidate_output,
        )
        effective_map = token_map or SurfaceTokenMap.from_tokenizer(self.tokenizer)
        return compute_symmetric_debiased_verdict(
            logits_pass_first=l1,
            logits_fail_first=l2,
            token_map=effective_map,
            temperature=temperature,
        )
