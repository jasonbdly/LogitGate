"""Typed loader for LogitGate declarative benchmark datasets in ``benchmarks/data/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(filename: str) -> Any:
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


MULTI_DOMAIN_24_ITEMS: list[dict[str, Any]] = _load_json("multi_domain_24.json")
CAL_DISTILL_80_ITEMS: list[dict[str, Any]] = _load_json("cal_distill_80.json")
OOD_GENERALIZATION_96_ITEMS: list[dict[str, Any]] = _load_json("ood_generalization_96.json")
STREAMING_TRAJECTORIES_32_ITEMS: list[dict[str, Any]] = _load_json("streaming_trajectories_32.json")
DRIFT_REGIMES_120_ITEMS: list[dict[str, Any]] = _load_json("drift_regimes_120.json")
RAG_GROUNDEDNESS_80_ITEMS: list[dict[str, Any]] = _load_json("rag_groundedness_80.json")
AGENT_MCP_TOOL_SAFETY_80_ITEMS: list[dict[str, Any]] = _load_json("agent_mcp_tool_safety_80.json")
MULTILINGUAL_UNICODE_BPE_80_ITEMS: list[dict[str, Any]] = _load_json("multilingual_unicode_bpe_80.json")

_RANK_PAIRWISE_DATA: dict[str, Any] = _load_json("rank_pairwise_80.json")
RANK_PAIRWISE_60_ITEMS: list[dict[str, Any]] = _RANK_PAIRWISE_DATA["pairwise_items"]
RANK_BEST_OF_N_20_POOLS: list[dict[str, Any]] = _RANK_PAIRWISE_DATA["best_of_n_pools"]

_REASONING_DATA: dict[str, Any] = _load_json("reasoning_tiers_32.json")
TIER_B_COMPOUND_RUBRIC: str = _REASONING_DATA["tier_b_compound_rubric"]
TIER_B_ATOMIC_RUBRICS: list[str] = _REASONING_DATA["tier_b_atomic_rubrics"]
TIER_A_8_ITEMS: list[dict[str, Any]] = _REASONING_DATA["tier_a"]
TIER_B_8_ITEMS: list[dict[str, Any]] = _REASONING_DATA["tier_b"]
TIER_C_8_ITEMS: list[dict[str, Any]] = _REASONING_DATA["tier_c"]
TIER_D_8_ITEMS: list[dict[str, Any]] = _REASONING_DATA["tier_d"]

_CHOICE_SCORE_DATA: dict[str, Any] = _load_json("choice_and_score_12.json")
CHOICE_GATE_6_ITEMS: list[dict[str, Any]] = _CHOICE_SCORE_DATA["choice_gate_items"]
SCORE_GATE_6_ITEMS: list[dict[str, Any]] = [
    {**item, "expected_band": tuple(item["expected_band"])}
    for item in _CHOICE_SCORE_DATA["score_gate_items"]
]

_ADV_CHOICE_SCORE_DATA: dict[str, Any] = _load_json("adversarial_choice_and_score_32.json")
ADVERSARIAL_CHOICE_16_ITEMS: list[dict[str, Any]] = _ADV_CHOICE_SCORE_DATA["choice_gate_items"]
ADVERSARIAL_SCORE_16_ITEMS: list[dict[str, Any]] = [
    {**item, "expected_band": tuple(item["expected_band"])}
    for item in _ADV_CHOICE_SCORE_DATA["score_gate_items"]
]

_HAYSTACK_DATA: dict[str, Any] = _load_json("haystack_handbook.json")
HAYSTACK_RUBRIC: str = _HAYSTACK_DATA["rubric"]
NEEDLE_COMPLIANT: str = _HAYSTACK_DATA["needle_compliant"]
NEEDLE_VIOLATING: str = _HAYSTACK_DATA["needle_violating"]
HANDBOOK_SECTIONS: list[str] = _HAYSTACK_DATA["sections"]


def load_multi_domain_24() -> list[dict[str, Any]]:
    return list(MULTI_DOMAIN_24_ITEMS)


def load_cal_distill_80() -> list[dict[str, Any]]:
    return list(CAL_DISTILL_80_ITEMS)


def load_ood_generalization_96() -> list[dict[str, Any]]:
    return list(OOD_GENERALIZATION_96_ITEMS)


def load_streaming_trajectories_32() -> list[dict[str, Any]]:
    return list(STREAMING_TRAJECTORIES_32_ITEMS)


def load_drift_regimes_120() -> list[dict[str, Any]]:
    return list(DRIFT_REGIMES_120_ITEMS)


def load_rag_groundedness_80() -> list[dict[str, Any]]:
    return list(RAG_GROUNDEDNESS_80_ITEMS)


def load_agent_mcp_tool_safety_80() -> list[dict[str, Any]]:
    return list(AGENT_MCP_TOOL_SAFETY_80_ITEMS)


def load_multilingual_unicode_bpe_80() -> list[dict[str, Any]]:
    return list(MULTILINGUAL_UNICODE_BPE_80_ITEMS)


def load_rank_pairwise_80() -> dict[str, Any]:
    return {
        "pairwise_items": list(RANK_PAIRWISE_60_ITEMS),
        "best_of_n_pools": list(RANK_BEST_OF_N_20_POOLS),
    }


def load_reasoning_tiers_32() -> list[dict[str, Any]]:
    return TIER_A_8_ITEMS + TIER_B_8_ITEMS + TIER_C_8_ITEMS + TIER_D_8_ITEMS


def load_choice_and_score_12() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return list(CHOICE_GATE_6_ITEMS), list(SCORE_GATE_6_ITEMS)


def load_adversarial_choice_and_score_32() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return list(ADVERSARIAL_CHOICE_16_ITEMS), list(ADVERSARIAL_SCORE_16_ITEMS)


def load_haystack_handbook() -> dict[str, Any]:
    return dict(_HAYSTACK_DATA)


def build_diverse_haystack_document(target_words: int, needle: str) -> str:
    """Constructs a multi-section technical document with ``needle`` placed at the midpoint."""
    half_target = max(40, target_words // 2)
    left_chunks: list[str] = []
    left_count = 0
    sec_idx = 0
    while left_count < half_target:
        chunk = HANDBOOK_SECTIONS[sec_idx % len(HANDBOOK_SECTIONS)]
        chapter = f"[Chapter {sec_idx // len(HANDBOOK_SECTIONS) + 1}.{sec_idx % len(HANDBOOK_SECTIONS) + 1}]"
        left_chunks.append(f"{chapter}\n{chunk}")
        left_count += len(chunk.split())
        sec_idx += 1

    right_chunks: list[str] = []
    right_count = 0
    while right_count < half_target:
        chunk = HANDBOOK_SECTIONS[sec_idx % len(HANDBOOK_SECTIONS)]
        chapter = f"[Chapter {sec_idx // len(HANDBOOK_SECTIONS) + 1}.{sec_idx % len(HANDBOOK_SECTIONS) + 1}]"
        right_chunks.append(f"{chapter}\n{chunk}")
        right_count += len(chunk.split())
        sec_idx += 1

    return "\n\n".join(left_chunks) + "\n\n" + needle + "\n\n" + "\n\n".join(right_chunks)
