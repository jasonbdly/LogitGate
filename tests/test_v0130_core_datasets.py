"""Integrity, anti-shortcut, psychometric deconfounding, and cross-split leakage tests for v0.13.0 core benchmark suites."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dataset import (
    AGENT_MCP_TOOL_SAFETY_80_ITEMS,
    MULTILINGUAL_UNICODE_BPE_80_ITEMS,
    RAG_GROUNDEDNESS_80_ITEMS,
    RANK_BEST_OF_N_20_POOLS,
    RANK_PAIRWISE_60_ITEMS,
    load_agent_mcp_tool_safety_80,
    load_multilingual_unicode_bpe_80,
    load_rag_groundedness_80,
    load_rank_pairwise_80,
    load_reasoning_tiers_32,
)
from logitgate import (
    DatasetAuditReport,
    SplitLeakageReport,
    audit_benchmark_dataset,
    check_split_leakage,
)

DATA_DIR = REPO_ROOT / "benchmarks" / "data"


def test_reasoning_tiers_32_tier_b_rubric_populated() -> None:
    """Verifies that all 8 tier_b items have the exact tier_b_compound_rubric string."""
    path = DATA_DIR / "reasoning_tiers_32.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    compound_rubric = raw["tier_b_compound_rubric"]
    assert isinstance(compound_rubric, str)
    assert len(compound_rubric) > 0

    tier_b_items = raw["tier_b"]
    assert len(tier_b_items) == 8
    for item in tier_b_items:
        assert "rubric" in item, f"Missing rubric in tier_b item {item['id']}"
        assert item["rubric"] == compound_rubric, f"Mismatched rubric in {item['id']}"

    # Also verify through loader
    all_32 = load_reasoning_tiers_32()
    assert len(all_32) == 32
    tier_b_loaded = [it for it in all_32 if it["id"].startswith("tier_b_")]
    assert len(tier_b_loaded) == 8
    for it in tier_b_loaded:
        assert it["rubric"] == compound_rubric


@pytest.mark.parametrize(
    ("filename", "loader_fn", "expected_domains", "min_len", "max_len"),
    [
        (
            "rag_groundedness_80.json",
            load_rag_groundedness_80,
            {
                "rag_clinical_guideline",
                "rag_sec_filing",
                "rag_cloud_runbook",
                "rag_contract_clause",
                "rag_incident_postmortem",
            },
            400,
            950,
        ),
        (
            "agent_mcp_tool_safety_80.json",
            load_agent_mcp_tool_safety_80,
            {
                "mcp_filesystem_ops",
                "mcp_database_admin",
                "mcp_http_webhook",
                "mcp_git_ci_runner",
                "mcp_cloud_iam",
            },
            350,
            800,
        ),
        (
            "multilingual_unicode_bpe_80.json",
            load_multilingual_unicode_bpe_80,
            {
                "bpe_cjk_technical",
                "bpe_european_diacritics",
                "bpe_arabic_cyrillic",
                "bpe_emoji_codemixed",
                "bpe_long_structured_trace",
            },
            None,
            None,
        ),
    ],
)
def test_binary_80_quad_structure_and_audit_metrics(
    filename: str,
    loader_fn: Any,
    expected_domains: set[str],
    min_len: int | None,
    max_len: int | None,
) -> None:
    path = DATA_DIR / filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw) == 80

    loaded = loader_fn()
    assert len(loaded) == 80
    if filename == "rag_groundedness_80.json":
        assert len(RAG_GROUNDEDNESS_80_ITEMS) == 80
    elif filename == "agent_mcp_tool_safety_80.json":
        assert len(AGENT_MCP_TOOL_SAFETY_80_ITEMS) == 80
    elif filename == "multilingual_unicode_bpe_80.json":
        assert len(MULTILINGUAL_UNICODE_BPE_80_ITEMS) == 80

    quads: dict[str, set[str]] = {}
    domain_counts: dict[str, int] = {}
    for item in raw:
        assert item["id"]
        assert item["quad_id"]
        assert item["domain"] in expected_domains
        assert item["variant"] in {"clean_pass", "hard_positive", "obvious_fail", "hard_negative"}
        assert isinstance(item["expected"], bool)
        assert item["rubric"]
        assert item["candidate"]

        if min_len is not None and max_len is not None:
            cand_len = len(item["candidate"])
            assert min_len <= cand_len <= max_len, (
                f"Candidate length {cand_len} outside [{min_len}, {max_len}] in {item['id']}"
            )

        quads.setdefault(item["quad_id"], set()).add(item["variant"])
        domain_counts[item["domain"]] = domain_counts.get(item["domain"], 0) + 1

    assert len(quads) == 20
    expected_variants = {"clean_pass", "hard_positive", "obvious_fail", "hard_negative"}
    for qid, variants in quads.items():
        assert variants == expected_variants, f"Incomplete quad {qid}: {variants}"

    assert len(domain_counts) == 5
    for dom, count in domain_counts.items():
        assert count == 16, f"Domain {dom} has {count} items, expected 16"

    # Conformal audit & psychometric deconfounding
    rep = audit_benchmark_dataset(path)
    assert isinstance(rep, DatasetAuditReport)
    assert rep.total_cases == 80
    assert rep.num_positive == 40
    assert rep.num_negative == 40
    assert rep.holds_at_alpha_010 is True
    assert rep.holds_at_alpha_005 is True
    assert rep.char_length_rel_diff <= 0.08
    assert abs(rep.length_auroc - 0.50) <= 0.08
    assert rep.lexical_loo_accuracy <= 0.55
    assert rep.duplicate_candidates == 0
    assert rep.warnings == ()

    # Cross-split leakage checks against existing benchmark datasets
    for ref_name, ref_src in (
        ("cal_distill_80.json", DATA_DIR / "cal_distill_80.json"),
        ("ood_generalization_96.json", DATA_DIR / "ood_generalization_96.json"),
    ):
        leak = check_split_leakage(ref_src, path, reference_label=ref_name)
        assert isinstance(leak, SplitLeakageReport)
        assert leak.is_clean is True, f"Leakage from {ref_name} into {filename}: {leak.warnings}"
        assert leak.exact_candidate_overlaps == 0
        assert leak.exact_rubric_overlaps == 0
        assert leak.near_duplicate_candidates == 0
        assert leak.shared_domains == ()


def test_multilingual_unicode_bpe_80_content_diversity() -> None:
    """Verifies CJK, Cyrillic, Arabic, Diacritics, Emoji codepoints and long trace character bounds."""
    path = DATA_DIR / "multilingual_unicode_bpe_80.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    combined = "".join(x["candidate"] for x in raw)

    # CJK Unified Ideographs & Hiragana/Katakana
    has_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" for ch in combined)
    assert has_cjk is True, "Missing genuine CJK characters in multilingual dataset"

    # Cyrillic
    has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in combined)
    assert has_cyrillic is True, "Missing genuine Cyrillic characters in multilingual dataset"

    # Arabic
    has_arabic = any("\u0600" <= ch <= "\u06ff" for ch in combined)
    assert has_arabic is True, "Missing genuine Arabic characters in multilingual dataset"

    # European Diacritics (ä, ö, ü, ß, é, è, ç, ñ)
    diacritics = {"ä", "ö", "ü", "ß", "é", "è", "ç", "ñ", "á", "ó", "ú"}
    has_diacritics = any(ch in diacritics for ch in combined)
    assert has_diacritics is True, "Missing European diacritics in multilingual dataset"

    # Emojis & Math symbols (🚀, ⚠️, 🔒, 📊, ⚡, 🛑, 🔍, 💡, €, ¥, £)
    has_emoji = any(ord(ch) > 0x1F000 for ch in combined)
    assert has_emoji is True, "Missing Unicode emojis in multilingual dataset"

    # bpe_long_structured_trace item constraints
    long_items = [x for x in raw if x["domain"] == "bpe_long_structured_trace"]
    assert len(long_items) == 16
    for item in long_items:
        cand = item["candidate"]
        assert len(cand) >= 600, f"Item {item['id']} has len {len(cand)} < 600"
        assert len(cand) <= 1800, f"Item {item['id']} has len {len(cand)} > 1800"
        assert "<think>" in cand and "</think>" in cand, f"Item {item['id']} missing <think> tags"
        assert "|" in cand and "---" in cand, f"Item {item['id']} missing markdown table structure"


def test_rank_pairwise_80_distribution_and_bias_balance() -> None:
    """Verifies rank_pairwise_80 balance, length bias neutrality, and pool contracts."""
    path = DATA_DIR / "rank_pairwise_80.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw.keys()) == {"pairwise_items", "best_of_n_pools"}

    pairwise = raw["pairwise_items"]
    best_of_n = raw["best_of_n_pools"]

    assert len(pairwise) == 60
    assert len(best_of_n) == 20

    # Test loaders
    loaded_data = load_rank_pairwise_80()
    assert len(loaded_data["pairwise_items"]) == 60
    assert len(loaded_data["best_of_n_pools"]) == 20
    assert len(RANK_PAIRWISE_60_ITEMS) == 60
    assert len(RANK_BEST_OF_N_20_POOLS) == 20

    # 1. Pairwise winner balance (exactly 30 zeroes, 30 ones)
    winners = [item["winner"] for item in pairwise]
    w0_count = sum(1 for w in winners if w == 0)
    w1_count = sum(1 for w in winners if w == 1)
    assert w0_count == 30
    assert w1_count == 30

    # Verify domain distribution in pairwise (5 domains x 12 items = 60)
    expected_pair_domains = {
        "rank_code_refactor",
        "rank_rag_citation",
        "rank_api_design",
        "rank_incident_triage",
        "rank_sql_optimization",
    }
    pair_domains: dict[str, int] = {}
    winner_lens: list[int] = []
    loser_lens: list[int] = []
    longer_winner_count = 0

    for item in pairwise:
        assert item["domain"] in expected_pair_domains
        pair_domains[item["domain"]] = pair_domains.get(item["domain"], 0) + 1
        ca_len = len(item["candidate_a"])
        cb_len = len(item["candidate_b"])
        w = item["winner"]
        w_len = ca_len if w == 0 else cb_len
        l_len = cb_len if w == 0 else ca_len
        winner_lens.append(w_len)
        loser_lens.append(l_len)
        if w_len > l_len:
            longer_winner_count += 1

    assert len(pair_domains) == 5
    for dom, cnt in pair_domains.items():
        assert cnt == 12, f"Domain {dom} has {cnt} items, expected 12"

    # Length unbiased winners (mean winner length within 8% of mean loser length)
    mean_winner = sum(winner_lens) / float(len(winner_lens))
    mean_loser = sum(loser_lens) / float(len(loser_lens))
    length_rel_diff = abs(mean_winner - mean_loser) / max(mean_winner, mean_loser)
    assert length_rel_diff <= 0.08, f"Winner/loser mean length rel diff {length_rel_diff:.4f} > 0.08"

    # Heuristic "pick longer candidate" accuracy must be close to 50%
    longer_acc = longer_winner_count / float(len(pairwise))
    assert 0.40 <= longer_acc <= 0.60, f"Heuristic longer accuracy {longer_acc:.2f} too biased"

    # 2. Best-of-N pools: 16 valid pools (4 each of winner 0, 1, 2, 3) + 4 all_bad pools
    valid_pools = [p for p in best_of_n if not p["all_bad"]]
    all_bad_pools = [p for p in best_of_n if p["all_bad"]]

    assert len(valid_pools) == 16
    assert len(all_bad_pools) == 4

    pool_winners: dict[int, int] = {}
    for pool in valid_pools:
        assert len(pool["candidates"]) == 4
        assert pool["expected_winner"] in {0, 1, 2, 3}
        assert pool["all_bad"] is False
        pool_winners[pool["expected_winner"]] = pool_winners.get(pool["expected_winner"], 0) + 1

    for winner_idx in (0, 1, 2, 3):
        assert pool_winners.get(winner_idx) == 4, f"Winner index {winner_idx} has {pool_winners.get(winner_idx)} pools"

    for pool in all_bad_pools:
        assert len(pool["candidates"]) == 4
        assert pool["expected_winner"] is None
        assert pool["all_bad"] is True

    # 3. Direct audit of rank_pairwise_80.json via audit_benchmark_dataset
    rank_audit = audit_benchmark_dataset(path)
    assert isinstance(rank_audit, DatasetAuditReport)
    assert rank_audit.total_cases == 120
    assert rank_audit.char_length_rel_diff <= 0.08, f"Rank audit char rel diff {rank_audit.char_length_rel_diff} > 0.08"
    assert abs(rank_audit.length_auroc - 0.50) <= 0.08, f"Rank audit auroc dev {abs(rank_audit.length_auroc - 0.50)} > 0.08"
    assert rank_audit.warnings == (), f"Rank audit warnings: {rank_audit.warnings}"


def test_core_benchmark_suites_anti_padding_and_anti_giveaways() -> None:
    """Verifies that no datasets contain synthetic dot-padding, space-before-punctuation padding, or confession docstrings."""
    import ast
    import re

    giveaway_re = re.compile(r"\b(Faulty|Broken|Vulnerable|Insecure|Flawed|Fragile)\b")
    space_punct_re = re.compile(r" {2,}[.。}]")
    dot_padding_re = re.compile(r"\.{4,}")

    target_files = [
        "rag_groundedness_80.json",
        "agent_mcp_tool_safety_80.json",
        "multilingual_unicode_bpe_80.json",
        "rank_pairwise_80.json",
    ]

    for fname in target_files:
        p = DATA_DIR / fname
        raw = json.loads(p.read_text(encoding="utf-8"))
        candidates: list[str] = []
        if isinstance(raw, list):
            candidates.extend(item["candidate"] for item in raw)
        elif isinstance(raw, dict):
            for item in raw.get("pairwise_items", []):
                ca = item["candidate_a"]
                cb = item["candidate_b"]
                candidates.extend([ca, cb])

                # Docstring giveaway check
                assert not giveaway_re.search(ca), f"Confession word in {item['id']} candidate_a"
                assert not giveaway_re.search(cb), f"Confession word in {item['id']} candidate_b"

                # ast.parse check on Python code candidates
                if item["domain"] == "rank_code_refactor":
                    ast.parse(ca)
                    ast.parse(cb)

            for pool in raw.get("best_of_n_pools", []):
                candidates.extend(pool["candidates"])
                if not pool["all_bad"] and pool["domain"] == "rank_code_refactor":
                    for cand in pool["candidates"]:
                        ast.parse(cand)

        for cand in candidates:
            assert not dot_padding_re.search(cand), f"Artificial dot padding found in {fname}: {cand[:60]}..."
            assert not space_punct_re.search(cand), f"Space-before-punctuation padding found in {fname}: {cand[:60]}..."

