from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any


def render_candidate(candidate: Any) -> str:
    """Renders an artifact the way the model would see it in production.

    Agents mostly validate *structured* outputs: tool-call arguments, API payloads, Pydantic
    models. ``str(dict)`` gives a Python repr (single quotes, ``True``/``None``) that models read
    as code rather than data, and that never matches what a tool actually emitted. These are
    rendered as indented JSON instead. Strings pass through untouched.
    """
    if isinstance(candidate, str):
        return candidate.strip()
    model_dump_json = getattr(candidate, "model_dump_json", None)  # Pydantic v2 models
    if callable(model_dump_json) and not isinstance(candidate, type):
        return str(model_dump_json(indent=2))
    if dataclasses.is_dataclass(candidate) and not isinstance(candidate, type):
        candidate = dataclasses.asdict(candidate)
    if isinstance(candidate, (Mapping, list, tuple)):
        try:
            return json.dumps(candidate, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            pass  # e.g. non-string dict keys mixed with unorderable types
    return str(candidate).strip()


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
    cand_str = render_candidate(candidate)
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


def compile_cot_binary_prompt(
    rubric: str,
    candidate: Any,
    verification_notes: str,
    pass_token: str = "A",
    fail_token: str = "B",
    query_first: bool = True,
    pass_description: str = "The candidate artifact satisfies all criteria in the evaluation rubric (PASS).",
    fail_description: str = "The candidate artifact violates one or more criteria in the evaluation rubric (FAIL).",
) -> str:
    """Compiles a prefix-cache-friendly binary prompt incorporating step-by-step verification notes.

    Places ``[VERIFICATION NOTES]`` in its own dedicated section outside ``[CANDIDATE ARTIFACT]``
    so small models never mistake reviewer notes for part of the candidate artifact, while keeping
    the entire prefix through ``[VERIFICATION NOTES]`` 100% byte-identical across Pass 1 and Pass 2.
    """
    notes_str = verification_notes.strip()
    if not notes_str or len(notes_str) < 12:
        return compile_binary_prompt(
            rubric=rubric,
            candidate=candidate,
            pass_token=pass_token,
            fail_token=fail_token,
            query_first=query_first,
            pass_description=pass_description,
            fail_description=fail_description,
        )

    cand_str = render_candidate(candidate)
    rubric_str = rubric.strip()
    if query_first:
        prefix = (
            f"[EVALUATION RUBRIC]\n{rubric_str}\n\n"
            f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
            f"[VERIFICATION NOTES]\n{notes_str}\n\n"
        )
    else:
        prefix = (
            f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
            f"[EVALUATION RUBRIC]\n{rubric_str}\n\n"
            f"[VERIFICATION NOTES]\n{notes_str}\n\n"
        )

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
        f"Based strictly on the evaluation rubric, candidate artifact, and verification notes above, "
        f"select the single token representing your verdict:\n"
        f"{option_lines}\n\n"
        f"Respond with the single character {tokens_str} only."
    )
    return prefix + suffix


def compile_batch_binary_prompt(
    rubrics: Sequence[str],
    candidate: Any,
    rubric_index: int,
    pass_token: str = "A",
    fail_token: str = "B",
    query_first: bool = False,
    pass_description: str = "The candidate artifact complies fully with the rubric.",
    fail_description: str = "The candidate artifact violates the rubric or has defects.",
) -> str:
    """
    Compiles an isolated, prefix-cache-friendly binary evaluation prompt for an individual rubric
    within a multi-rubric evaluation set.

    Architecture:
    The candidate artifact forms the shared static prefix:
        [CANDIDATE ARTIFACT]
        {candidate}
    This prefix is 100% byte-identical across ALL K rubrics and across both forward (Pass 1) and
    reverse (Pass 2) passes, allowing the inference server's KV-cache to retain the candidate tokens
    across all queries.

    Crucially, each rubric is evaluated in complete isolation: no other rubrics are loaded into the
    prompt context. This prevents cross-rubric attention leakage and context bloat, scaling
    consistently whether evaluating 2 rubrics or 500 rubrics.
    """
    if rubric_index < 0 or rubric_index >= len(rubrics):
        raise IndexError(f"rubric_index {rubric_index} out of bounds for rubrics list of size {len(rubrics)}")

    cand_str = render_candidate(candidate)
    target_rubric = rubrics[rubric_index].strip()

    t_pass = pass_token.strip().upper()
    t_fail = fail_token.strip().upper()

    mapping = {
        t_pass: pass_description,
        t_fail: fail_description,
    }
    sorted_tokens = sorted(mapping.keys())
    option_lines = "\n".join(f"{tok}: {mapping[tok]}" for tok in sorted_tokens)
    tokens_str = f"{sorted_tokens[0]} or {sorted_tokens[1]}" if len(sorted_tokens) == 2 else " or ".join(sorted_tokens)

    if not query_first:
        prefix = f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
        suffix = (
            f"[EVALUATION RUBRIC]\n"
            f"{target_rubric}\n\n"
            f"[DECISION MAPPING]\n"
            f"{option_lines}\n"
            f"Respond with a single character ({tokens_str}).\n"
            f"Verdict:"
        )
    else:
        prefix = f"[EVALUATION RUBRIC]\n{target_rubric}\n\n[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
        suffix = (
            f"[DECISION MAPPING]\n"
            f"{option_lines}\n"
            f"Respond with a single character ({tokens_str}).\n"
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
    cand_str = render_candidate(candidate)
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
    cand_str = render_candidate(candidate)
    rubric_str = rubric.strip()

    if query_first:
        prefix = f"[EVALUATION RUBRIC]\n{rubric_str}\n\n[CANDIDATE ARTIFACT]\n{cand_str}\n\n"
    else:
        prefix = f"[CANDIDATE ARTIFACT]\n{cand_str}\n\n[EVALUATION RUBRIC]\n{rubric_str}\n\n"

    span = max_score - min_score
    anchors: list[str] = []
    for tok_val in range(min_score, max_score + 1):
        q = (max_score - tok_val) / span if inverted else (tok_val - min_score) / span
        if q >= 0.99:
            desc = "Highest quality (exemplary; satisfies every rubric criterion with no defects)"
        elif q <= 0.01:
            desc = "Lowest quality (severely flawed, unsafe, or violates core rubric criteria)"
        elif q >= 0.65:
            desc = "Good quality (solid compliance with only minor non-critical issues)"
        elif q <= 0.35:
            desc = "Poor quality (notable defects, omissions, or reliability issues)"
        else:
            desc = "Moderate quality (mixed or partial compliance)"
        anchors.append(f"{tok_val}: {desc}")
    anchor_block = "\n".join(anchors)

    if inverted:
        scale_desc = (
            f"Rate the candidate artifact on an inverted scale where {min_score} is highest quality "
            f"and {max_score} is lowest quality based on the rubric "
            f"(override any normal low-to-high numbering mentioned above):\n"
            f"{anchor_block}\n"
        )
    else:
        scale_desc = (
            f"Rate the candidate artifact from {min_score} to {max_score} where {min_score} is lowest quality "
            f"and {max_score} is highest quality based on the rubric:\n"
            f"{anchor_block}\n"
        )

    suffix = (
        f"[SCORING SCALE]\n"
        f"{scale_desc}"
        f"Respond with a single integer ({min_score}-{max_score}).\n"
        f"Score:"
    )
    return prefix + suffix
