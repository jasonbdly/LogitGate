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
