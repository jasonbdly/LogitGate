from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from logitgate.debiasing.permutation import (
    SurfaceTokenMap,
    compute_shannon_entropy,
    marginalize_pole_logprob,
)


@dataclass(frozen=True, slots=True)
class RecurrentLoopTrace:
    """Result of an adaptive recurrent run: loops used, per-loop verdict entropy, final logits."""

    loops_executed: int
    entropy_trajectory: tuple[float, ...]
    halted_early: bool
    final_logits: Any


def execute_adaptive_recurrent_loops(
    initial_hidden_state: Any,
    recurrent_step_fn: Callable[[Any, int], Any],
    lm_head: Callable[[Any], Any] | Any,
    final_norm: Callable[[Any], Any] | Any | None = None,
    token_map: SurfaceTokenMap | None = None,
    min_loops: int = 2,
    max_loops: int = 8,
    entropy_halt_threshold: float = 0.12,
    kl_stability_threshold: float = 0.005,
) -> RecurrentLoopTrace:
    """Executes latent recurrent loops (e.g., Dory-4B) with early-exit halting when:

      1. Binary verdict entropy drops below `entropy_halt_threshold`, AND
      2. Consecutive loop log-prob shift is below `kl_stability_threshold`.
    """
    h = initial_hidden_state
    entropies: list[float] = []
    prev_p_pos: float | None = None
    last_logits: Any = None

    effective_token_map = token_map or SurfaceTokenMap(
        positive_token_ids=(319, 65),
        negative_token_ids=(350, 66),
    )

    for loop_idx in range(1, max_loops + 1):
        h = recurrent_step_fn(h, loop_idx)

        # Probe only the last token position [-1] to avoid full-sequence vocab projection overhead
        try:
            probe_h = h[:, -1:, :] if hasattr(h, "ndim") and h.ndim >= 3 else h
        except (TypeError, IndexError):
            probe_h = h

        if final_norm is not None:
            normed_last_token = final_norm(probe_h)
        else:
            normed_last_token = probe_h

        raw_logits = lm_head(normed_last_token)

        # Flatten to 1D [vocab_size]
        try:
            import torch

            if isinstance(raw_logits, torch.Tensor):
                logits_last = (
                    raw_logits.squeeze(0).squeeze(0)
                    if raw_logits.ndim == 3
                    else raw_logits.squeeze(0)
                    if raw_logits.ndim == 2
                    else raw_logits
                )
            else:
                logits_last = raw_logits
        except ImportError:
            logits_last = raw_logits

        last_logits = logits_last

        lp_pos, _ = marginalize_pole_logprob(
            logits_last,
            effective_token_map.positive_token_ids,
            effective_token_map.negative_token_ids,
        )
        p_pos = math.exp(lp_pos)
        entropy = compute_shannon_entropy(p_pos)
        entropies.append(entropy)

        if loop_idx >= min_loops and prev_p_pos is not None:
            prob_delta = abs(p_pos - prev_p_pos)
            if entropy <= entropy_halt_threshold and prob_delta <= kl_stability_threshold:
                return RecurrentLoopTrace(
                    loops_executed=loop_idx,
                    entropy_trajectory=tuple(entropies),
                    halted_early=True,
                    final_logits=logits_last,
                )

        prev_p_pos = p_pos

    assert last_logits is not None
    return RecurrentLoopTrace(
        loops_executed=max_loops,
        entropy_trajectory=tuple(entropies),
        halted_early=False,
        final_logits=last_logits,
    )
