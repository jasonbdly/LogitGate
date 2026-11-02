from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compile_binary_prompt(
    rubric: str,
    candidate: Any,
    pass_token: str = "A",
    fail_token: str = "B",
    query_first: bool = True,
    pass_description: str = "The candidate artifact complies fully with the rubric.",
    fail_description: str = "The candidate artifact violates the rubric or has defects.",
) -> str:
    """
    Compiles a prefix-cache-friendly binary evaluation prompt.
    When query_first=True, the rubric and candidate artifact form a 100% byte-identical
    static prefix across both Pass 1 and Pass 2, allowing KV-cache hit rates > 99%.
    """
    cand_str = str(candidate).strip()
    rubric_str = rubric.strip()

    if query_first:
        prefix = f"[EVALUATION RUBRIC]\n{rubric_str}\n\n[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
    else:
        prefix = f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n[EVALUATION RUBRIC]\n{rubric_str}\n\n"

    t_pass = pass_token.strip().upper()
    t_fail = fail_token.strip().upper()

    mapping = {
        t_pass: pass_description,
        t_fail: fail_description,
    }
    sorted_tokens = sorted(mapping.keys())
    option_lines = "\n".join(f"{tok}: {mapping[tok]}" for tok in sorted_tokens)
    tokens_str = f"{sorted_tokens[0]} or {sorted_tokens[1]}" if len(sorted_tokens) == 2 else " or ".join(sorted_tokens)

    suffix = (
        f"[DECISION MAPPING]\n"
        f"{option_lines}\n"
        f"Respond with a single character ({tokens_str}).\n"
        f"Verdict:"
    )
    return prefix + suffix


def compile_batch_binary_prompt(
    rubrics: Sequence[str],
    candidate: Any,
    rubric_index: int,
    pass_token: str = "A",
    fail_token: str = "B",
    query_first: bool = True,
    pass_description: str = "The candidate artifact complies fully with the rubric.",
    fail_description: str = "The candidate artifact violates the rubric or has defects.",
) -> str:
    """
    Compiles a prefix-cache-friendly binary evaluation prompt for a multi-rubric batch.
    Across all 2K passes (forward and reverse for all K rubrics), the prefix containing
    all rubrics and candidate artifact is 100% byte-identical.
    """
    if rubric_index < 0 or rubric_index >= len(rubrics):
        raise IndexError(f"rubric_index {rubric_index} out of bounds for rubrics list of size {len(rubrics)}")

    cand_str = str(candidate).strip()
    rubrics_block = "\n".join(f"Rubric #{i + 1}: {r.strip()}" for i, r in enumerate(rubrics))

    if query_first:
        prefix = f"[EVALUATION RUBRICS]\n{rubrics_block}\n\n[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
    else:
        prefix = f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n[EVALUATION RUBRICS]\n{rubrics_block}\n\n"

    t_pass = pass_token.strip().upper()
    t_fail = fail_token.strip().upper()

    mapping = {
        t_pass: pass_description,
        t_fail: fail_description,
    }
    sorted_tokens = sorted(mapping.keys())
    option_lines = "\n".join(f"{tok}: {mapping[tok]}" for tok in sorted_tokens)
    tokens_str = f"{sorted_tokens[0]} or {sorted_tokens[1]}" if len(sorted_tokens) == 2 else " or ".join(sorted_tokens)

    suffix = (
        f"[DECISION MAPPING — RUBRIC #{rubric_index + 1}]\n"
        f"Target Rubric #{rubric_index + 1}: {rubrics[rubric_index].strip()}\n"
        f"{option_lines}\n"
        f"Respond with a single character ({tokens_str}) for Rubric #{rubric_index + 1}.\n"
        f"Verdict:"
    )
    return prefix + suffix


def compile_choice_prompt(
    question: str,
    candidate: Any,
    options: Mapping[str, str],
    query_first: bool = True,
) -> str:
    """
    Compiles a discrete multi-choice prompt with identical static prefix.
    """
    cand_str = str(candidate).strip()
    q_str = question.strip()

    if query_first:
        prefix = f"[QUESTION]\n{q_str}\n\n[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
    else:
        prefix = f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n[QUESTION]\n{q_str}\n\n"

    option_lines = "\n".join(f"{k}: {v}" for k, v in options.items())
    keys = list(options.keys())
    keys_str = " or ".join(keys)

    suffix = (
        f"[DECISION MAPPING]\n"
        f"{option_lines}\n"
        f"Respond with a single character ({keys_str}).\n"
        f"Choice:"
    )
    return prefix + suffix


def compile_score_prompt(
    rubric: str,
    candidate: Any,
    min_score: int = 1,
    max_score: int = 5,
    query_first: bool = True,
    inverted: bool = False,
) -> str:
    """
    Compiles a bounded integer scoring prompt with identical static prefix.
    When inverted=True, min_score corresponds to highest quality and max_score
    corresponds to lowest quality, enabling symmetric score debiasing with
    byte-identical prefix caching.
    """
    cand_str = str(candidate).strip()
    rubric_str = rubric.strip()

    if query_first:
        prefix = f"[EVALUATION RUBRIC]\n{rubric_str}\n\n[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
    else:
        prefix = f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n[EVALUATION RUBRIC]\n{rubric_str}\n\n"

    if inverted:
        scale_desc = (
            f"Rate the candidate artifact on an inverted scale where {min_score} is highest quality "
            f"and {max_score} is lowest quality based on the rubric.\n"
        )
    else:
        scale_desc = f"Rate the candidate artifact from {min_score} to {max_score} based on the rubric.\n"

    suffix = (
        f"[SCORING SCALE]\n"
        f"{scale_desc}"
        f"Respond with a single integer ({min_score}-{max_score}).\n"
        f"Score:"
    )
    return prefix + suffix
