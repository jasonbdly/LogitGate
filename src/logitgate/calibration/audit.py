"""Psychometric, shortcut, and cross-split leakage auditing for evaluation datasets.

Provides zero-LLM-call statistical diagnostics for calibration and benchmark suites:
- Length-confounding detection (`char_length_rel_diff`, `word_length_rel_diff`, exact `length_auroc`)
- Surface-vocabulary shortcut detection ($O(N \\cdot \\bar{L})$ leave-one-out unigram log-odds probe)
- Finite-sample split-conformal power (`min_alpha_guaranteed`, `holds_at_alpha_010`, `holds_at_alpha_005`)
- Cross-split contamination (`check_split_leakage`: exact rubric/candidate overlap, word-shingle Jaccard, and domain overlap)
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logitgate.calibration.conformal import (
    load_calibration_dataset,
    min_samples_per_class,
)
from logitgate.exceptions import CalibrationError

_MAX_AUDIT_DATASET_BYTES = 32 * 1024 * 1024  # 32 MiB cap for local dataset files
_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _NormalizedCase:
    rubric: str
    candidate: str
    label: bool
    domain: str | None = None
    quad_id: str | None = None
    variant: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    """Statistical, shortcut, and conformal-power audit of an evaluation dataset."""

    source: str
    total_cases: int
    num_positive: int
    num_negative: int
    positive_rate: float
    min_alpha_guaranteed: float | None
    holds_at_alpha_010: bool
    holds_at_alpha_005: bool
    pos_mean_chars: float
    neg_mean_chars: float
    char_length_rel_diff: float
    pos_mean_words: float
    neg_mean_words: float
    word_length_rel_diff: float
    length_auroc: float
    lexical_loo_accuracy: float
    unique_rubrics: int
    duplicate_candidates: int
    domains: dict[str, int]
    per_domain_char_rel_diff: dict[str, float]
    quad_variants: dict[str, int]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total_cases": self.total_cases,
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
            "positive_rate": self.positive_rate,
            "min_alpha_guaranteed": self.min_alpha_guaranteed,
            "holds_at_alpha_010": self.holds_at_alpha_010,
            "holds_at_alpha_005": self.holds_at_alpha_005,
            "pos_mean_chars": self.pos_mean_chars,
            "neg_mean_chars": self.neg_mean_chars,
            "char_length_rel_diff": self.char_length_rel_diff,
            "pos_mean_words": self.pos_mean_words,
            "neg_mean_words": self.neg_mean_words,
            "word_length_rel_diff": self.word_length_rel_diff,
            "length_auroc": self.length_auroc,
            "lexical_loo_accuracy": self.lexical_loo_accuracy,
            "unique_rubrics": self.unique_rubrics,
            "duplicate_candidates": self.duplicate_candidates,
            "domains": dict(self.domains),
            "per_domain_char_rel_diff": dict(self.per_domain_char_rel_diff),
            "quad_variants": dict(self.quad_variants),
            "warnings": list(self.warnings),
        }

    def format_summary(self) -> str:
        lines = [
            f"Dataset Integrity Audit: {self.source}",
            "-" * 64,
            (
                f"Cases: {self.total_cases} "
                f"(pass={self.num_positive}, fail={self.num_negative}, "
                f"pass_rate={self.positive_rate:.1%})"
            ),
            (
                f"Conformal Power: min_alpha={self.min_alpha_guaranteed} "
                f"| alpha<=0.10: {self.holds_at_alpha_010} "
                f"| alpha<=0.05: {self.holds_at_alpha_005}"
            ),
            (
                f"Length Parity: pass={self.pos_mean_chars:.1f} chars ({self.pos_mean_words:.1f} words) "
                f"vs fail={self.neg_mean_chars:.1f} chars ({self.neg_mean_words:.1f} words) "
                f"| rel_diff={self.char_length_rel_diff:.1%} | length_auroc={self.length_auroc:.4f}"
            ),
            (
                f"Shortcut & Diversity: lexical_loo_acc={self.lexical_loo_accuracy:.1%} "
                f"| unique_rubrics={self.unique_rubrics} "
                f"| duplicate_candidates={self.duplicate_candidates}"
            ),
        ]
        if self.domains:
            dom_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.domains.items()))
            lines.append(f"Domains ({len(self.domains)}): {dom_str}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("Warnings: none")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SplitLeakageReport:
    """Cross-split contamination report between a reference (train/cal) split and an eval split."""

    reference_source: str
    eval_source: str
    reference_cases: int
    eval_cases: int
    exact_candidate_overlaps: int
    exact_rubric_overlaps: int
    near_duplicate_candidates: int
    max_candidate_jaccard: float
    shared_domains: tuple[str, ...]
    jaccard_threshold: float
    is_clean: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_source": self.reference_source,
            "eval_source": self.eval_source,
            "reference_cases": self.reference_cases,
            "eval_cases": self.eval_cases,
            "exact_candidate_overlaps": self.exact_candidate_overlaps,
            "exact_rubric_overlaps": self.exact_rubric_overlaps,
            "near_duplicate_candidates": self.near_duplicate_candidates,
            "max_candidate_jaccard": self.max_candidate_jaccard,
            "shared_domains": list(self.shared_domains),
            "jaccard_threshold": self.jaccard_threshold,
            "is_clean": self.is_clean,
            "warnings": list(self.warnings),
        }

    def format_summary(self) -> str:
        lines = [
            f"Cross-Split Leakage Check: {self.reference_source} -> {self.eval_source}",
            "-" * 64,
            (
                f"Reference={self.reference_cases} | Eval={self.eval_cases} | Clean={self.is_clean}"
            ),
            (
                f"Overlaps: exact_candidate={self.exact_candidate_overlaps} "
                f"| exact_rubric={self.exact_rubric_overlaps} "
                f"| near_dup(J>={self.jaccard_threshold:.2f})={self.near_duplicate_candidates} "
                f"(max_J={self.max_candidate_jaccard:.4f})"
            ),
        ]
        if self.shared_domains:
            lines.append(f"Shared Domains: {', '.join(self.shared_domains)}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _parse_bool_label(raw_label: Any) -> bool | None:
    if raw_label is None:
        return None
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, (int, float)):
        if not math.isfinite(raw_label):
            raise CalibrationError(f"Non-finite dataset label: {raw_label!r}")
        return bool(raw_label >= 0.5)
    if isinstance(raw_label, str):
        norm = raw_label.strip().lower()
        if norm in ("pass", "passed", "true", "1", "yes"):
            return True
        if norm in ("fail", "failed", "false", "0", "no"):
            return False
    return bool(raw_label)


def _extract_case_from_mapping(item: Mapping[str, Any]) -> _NormalizedCase | None:
    rubric = str(
        item.get("rubric") or item.get("question") or item.get("instruction") or ""
    ).strip()
    raw_cand: Any = None
    for cand_key in ("candidate", "actual_output", "output", "response", "text"):
        if item.get(cand_key) is not None:
            raw_cand = item[cand_key]
            break
    if raw_cand is None and isinstance(item.get("chunks"), list):
        raw_cand = "".join(str(ch) for ch in item["chunks"])
    if raw_cand is None:
        return None
    candidate = (
        json.dumps(raw_cand) if isinstance(raw_cand, (dict, list)) else str(raw_cand)
    ).strip()
    raw_label = item.get("label", item.get("ground_truth", item.get("expected", None)))
    if raw_label is None:
        raw_label = item.get("passed", item.get("pass", None))
    label = _parse_bool_label(raw_label)
    if label is None or not rubric or not candidate:
        return None
    raw_dom = item.get("domain") or item.get("family")
    domain = str(raw_dom).strip() if raw_dom is not None and str(raw_dom).strip() else None
    raw_quad = item.get("quad_id")
    quad_id = str(raw_quad).strip() if raw_quad is not None and str(raw_quad).strip() else None
    raw_var = item.get("variant") or item.get("trajectory_type")
    variant = str(raw_var).strip() if raw_var is not None and str(raw_var).strip() else None
    return _NormalizedCase(
        rubric=rubric,
        candidate=candidate,
        label=label,
        domain=domain,
        quad_id=quad_id,
        variant=variant,
    )


def _load_normalized_cases(source: Any, *, default_label: str = "in_memory") -> tuple[list[_NormalizedCase], str]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        st = os.lstat(p)
        if stat.S_ISLNK(st.st_mode):
            raise CalibrationError(f"Refusing to audit dataset through symlink: {p}")
        if not stat.S_ISREG(st.st_mode):
            raise CalibrationError(f"Dataset path is not a regular file: {p}")
        if st.st_size > _MAX_AUDIT_DATASET_BYTES:
            raise CalibrationError(
                f"Dataset file exceeds {_MAX_AUDIT_DATASET_BYTES} byte limit: {p}"
            )
        suffix = p.suffix.lower()
        if suffix in (".json", ".jsonl"):
            text = p.read_text(encoding="utf-8")
            raw_items: list[Any]
            if suffix == ".jsonl":
                raw_items = [
                    json.loads(line) for line in text.splitlines() if line.strip()
                ]
            else:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_items = parsed
                elif isinstance(parsed, Mapping):
                    raw_items = list(
                        parsed.get("test_cases")
                        or parsed.get("samples")
                        or parsed.get("cases")
                        or parsed.get("data")
                        or []
                    )
                else:
                    raise CalibrationError(f"Unsupported JSON root type in {p}: {type(parsed).__name__}")
            out: list[_NormalizedCase] = []
            for item in raw_items:
                if not isinstance(item, Mapping):
                    raise CalibrationError(f"Expected JSON object in {p}, got {type(item).__name__}")
                nc = _extract_case_from_mapping(item)
                if nc is not None:
                    out.append(nc)
            if not out:
                raise CalibrationError(f"No valid labeled cases found in {p}")
            return out, str(source)

        tuples = load_calibration_dataset(p)
        if not tuples:
            raise CalibrationError(f"No valid labeled cases found in {p}")
        return [
            _NormalizedCase(rubric=r.strip(), candidate=c.strip(), label=bool(y))
            for r, c, y in tuples
        ], str(source)

    if not isinstance(source, Sequence) or isinstance(source, (bytes, bytearray)):
        raise CalibrationError(f"Unsupported dataset source type: {type(source).__name__}")

    cases: list[_NormalizedCase] = []
    for item in source:
        if isinstance(item, Mapping):
            nc = _extract_case_from_mapping(item)
            if nc is not None:
                cases.append(nc)
        elif isinstance(item, tuple) and len(item) >= 3:
            r, c, y = item[0], item[1], item[2]
            lbl = _parse_bool_label(y)
            if lbl is not None and str(r).strip() and str(c).strip():
                cases.append(
                    _NormalizedCase(rubric=str(r).strip(), candidate=str(c).strip(), label=lbl)
                )
        elif hasattr(item, "rubric") and hasattr(item, "candidate") and hasattr(item, "expected"):
            lbl = _parse_bool_label(item.expected)
            if lbl is not None:
                cases.append(
                    _NormalizedCase(
                        rubric=str(item.rubric).strip(),
                        candidate=str(item.candidate).strip(),
                        label=lbl,
                    )
                )
        else:
            raise CalibrationError(f"Unsupported dataset item type: {type(item).__name__}")

    if not cases:
        raise CalibrationError("Dataset contains no valid labeled cases.")
    return cases, default_label


def _compute_exact_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Exact tie-aware Mann-Whitney U AUROC in O(N log N) time."""
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda xy: xy[0])
    n = len(paired)
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        # 1-based average rank for tied block [i, j)
        avg_rank = 0.5 * ((i + 1) + j)
        pos_in_tie = sum(1 for k in range(i, j) if paired[k][1])
        rank_sum_pos += pos_in_tie * avg_rank
        i = j
    u_stat = rank_sum_pos - 0.5 * n_pos * (n_pos + 1)
    return round(u_stat / float(n_pos * n_neg), 4)


def _compute_lexical_loo_accuracy(cases: Sequence[_NormalizedCase]) -> float:
    """O(N * L) leave-one-out Laplace-smoothed multinomial unigram accuracy over candidates."""
    n = len(cases)
    if n < 4:
        return 0.5
    total_pos_docs = sum(1 for c in cases if c.label)
    total_neg_docs = n - total_pos_docs
    if total_pos_docs == 0 or total_neg_docs == 0:
        return 1.0

    doc_tokens: list[Counter[str]] = []
    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    vocab: set[str] = set()
    total_pos_toks = 0
    total_neg_toks = 0

    for c in cases:
        toks = Counter(_TOKEN_RE.findall(c.candidate.lower()))
        doc_tokens.append(toks)
        vocab.update(toks.keys())
        tok_sum = sum(toks.values())
        if c.label:
            pos_counts.update(toks)
            total_pos_toks += tok_sum
        else:
            neg_counts.update(toks)
            total_neg_toks += tok_sum

    v_size = max(len(vocab), 1)
    correct = 0
    for c, toks in zip(cases, doc_tokens):
        doc_len = sum(toks.values())
        if c.label:
            rem_pos_docs = total_pos_docs - 1
            rem_neg_docs = total_neg_docs
            rem_pos_toks = total_pos_toks - doc_len
            rem_neg_toks = total_neg_toks
        else:
            rem_pos_docs = total_pos_docs
            rem_neg_docs = total_neg_docs - 1
            rem_pos_toks = total_pos_toks
            rem_neg_toks = total_neg_toks - doc_len

        if rem_pos_docs <= 0:
            pred = False
        elif rem_neg_docs <= 0:
            pred = True
        else:
            log_score = math.log(rem_pos_docs / float(rem_neg_docs))
            denom_pos = float(rem_pos_toks + v_size)
            denom_neg = float(rem_neg_toks + v_size)
            for tok, cnt in toks.items():
                p_cnt = pos_counts[tok] - (cnt if c.label else 0)
                n_cnt = neg_counts[tok] - (0 if c.label else cnt)
                log_score += cnt * (
                    math.log((p_cnt + 1.0) / denom_pos) - math.log((n_cnt + 1.0) / denom_neg)
                )
            pred = log_score >= 0.0

        if pred == c.label:
            correct += 1

    return round(correct / float(n), 4)


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).lower()


def _word_shingles(text: str, k: int = 3) -> frozenset[str]:
    words = _TOKEN_RE.findall(text.lower())
    if not words:
        return frozenset()
    if len(words) < k:
        return frozenset(words)
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def audit_benchmark_dataset(
    source: Any,
    *,
    label: str | None = None,
    max_char_rel_diff: float = 0.15,
    max_length_auroc_dev: float = 0.12,
    max_lexical_loo_accuracy: float = 0.65,
) -> DatasetAuditReport:
    """Audits a calibration or benchmark dataset for confounders, shortcuts, and sample power."""
    cases, inferred_label = _load_normalized_cases(
        source, default_label=label or "in_memory"
    )
    eff_label = label or inferred_label
    n = len(cases)
    pos = [c for c in cases if c.label]
    neg = [c for c in cases if not c.label]
    n_pos = len(pos)
    n_neg = len(neg)
    pos_rate = round(n_pos / float(n), 4)

    min_cls = min(n_pos, n_neg)
    min_alpha = (
        round(math.ceil((2.0 / float(min_cls + 1)) * 10000.0 - 1e-9) / 10000.0, 4)
        if min_cls >= 1
        else None
    )
    holds_010 = min_cls >= min_samples_per_class(0.10)
    holds_005 = min_cls >= min_samples_per_class(0.05)

    pos_chars = [float(len(c.candidate)) for c in pos]
    neg_chars = [float(len(c.candidate)) for c in neg]
    pos_words = [float(len(c.candidate.split())) for c in pos]
    neg_words = [float(len(c.candidate.split())) for c in neg]

    pos_mean_c = sum(pos_chars) / len(pos_chars) if pos_chars else 0.0
    neg_mean_c = sum(neg_chars) / len(neg_chars) if neg_chars else 0.0
    char_rel = (
        abs(pos_mean_c - neg_mean_c) / max(pos_mean_c, neg_mean_c, 1.0)
        if (pos_chars and neg_chars)
        else 0.0
    )

    pos_mean_w = sum(pos_words) / len(pos_words) if pos_words else 0.0
    neg_mean_w = sum(neg_words) / len(neg_words) if neg_words else 0.0
    word_rel = (
        abs(pos_mean_w - neg_mean_w) / max(pos_mean_w, neg_mean_w, 1.0)
        if (pos_words and neg_words)
        else 0.0
    )

    len_auroc = _compute_exact_auroc(
        [float(len(c.candidate)) for c in cases],
        [c.label for c in cases],
    )
    lex_acc = _compute_lexical_loo_accuracy(cases)

    unique_rubrics = len({_norm_ws(c.rubric) for c in cases})
    norm_cands = [_norm_ws(c.candidate) for c in cases]
    dup_cands = len(norm_cands) - len(set(norm_cands))

    domain_counts: Counter[str] = Counter()
    domain_pos_c: dict[str, list[float]] = {}
    domain_neg_c: dict[str, list[float]] = {}
    variant_counts: Counter[str] = Counter()
    for c in cases:
        if c.domain is not None:
            domain_counts[c.domain] += 1
            if c.label:
                domain_pos_c.setdefault(c.domain, []).append(float(len(c.candidate)))
            else:
                domain_neg_c.setdefault(c.domain, []).append(float(len(c.candidate)))
        if c.variant is not None:
            variant_counts[c.variant] += 1

    per_dom_rel: dict[str, float] = {}
    for dom in sorted(domain_counts):
        dp = domain_pos_c.get(dom, [])
        dn = domain_neg_c.get(dom, [])
        if dp and dn:
            mp = sum(dp) / len(dp)
            mn = sum(dn) / len(dn)
            per_dom_rel[dom] = round(abs(mp - mn) / max(mp, mn, 1.0), 4)

    warnings: list[str] = []
    if n_pos == 0 or n_neg == 0:
        warnings.append("Single-class dataset: requires both positive and negative cases.")
    elif abs(pos_rate - 0.5) > 0.20:
        warnings.append(
            f"Class imbalance detected: positive_rate={pos_rate:.1%} (pass={n_pos}, fail={n_neg})."
        )
    if not holds_010:
        req_010 = min_samples_per_class(0.10)
        warnings.append(
            f"Conformal guarantee at alpha=0.10 requires >= {req_010} cases per class "
            f"(have pass={n_pos}, fail={n_neg})."
        )
    if char_rel > max_char_rel_diff or abs(len_auroc - 0.5) > max_length_auroc_dev:
        warnings.append(
            f"Length confounder detected: pass mean={pos_mean_c:.1f} chars vs fail mean={neg_mean_c:.1f} "
            f"chars (rel_diff={char_rel:.1%}, length_auroc={len_auroc:.4f})."
        )
    if n >= 20 and lex_acc > max_lexical_loo_accuracy:
        warnings.append(
            f"Surface lexical shortcut detected: rubric-blind unigram LOO accuracy is {lex_acc:.1%} "
            f"(> {max_lexical_loo_accuracy:.0%}). Add lexical-trap hard positives and polite hard negatives."
        )
    if dup_cands > 0:
        warnings.append(f"Found {dup_cands} duplicate candidate string(s) within dataset.")

    return DatasetAuditReport(
        source=eff_label,
        total_cases=n,
        num_positive=n_pos,
        num_negative=n_neg,
        positive_rate=pos_rate,
        min_alpha_guaranteed=min_alpha,
        holds_at_alpha_010=holds_010,
        holds_at_alpha_005=holds_005,
        pos_mean_chars=round(pos_mean_c, 2),
        neg_mean_chars=round(neg_mean_c, 2),
        char_length_rel_diff=round(char_rel, 4),
        pos_mean_words=round(pos_mean_w, 2),
        neg_mean_words=round(neg_mean_w, 2),
        word_length_rel_diff=round(word_rel, 4),
        length_auroc=len_auroc,
        lexical_loo_accuracy=lex_acc,
        unique_rubrics=unique_rubrics,
        duplicate_candidates=dup_cands,
        domains=dict(sorted(domain_counts.items())),
        per_domain_char_rel_diff=per_dom_rel,
        quad_variants=dict(sorted(variant_counts.items())),
        warnings=tuple(warnings),
    )


def check_split_leakage(
    reference_source: Any,
    eval_source: Any,
    *,
    reference_label: str | None = None,
    eval_label: str | None = None,
    jaccard_threshold: float = 0.65,
) -> SplitLeakageReport:
    """Checks exact and shingle near-duplicate leakage from a reference split into an eval split."""
    if (
        isinstance(jaccard_threshold, bool)
        or not isinstance(jaccard_threshold, (int, float))
        or not math.isfinite(jaccard_threshold)
        or not (0.0 < float(jaccard_threshold) <= 1.0)
    ):
        raise CalibrationError(
            f"jaccard_threshold must be in (0.0, 1.0], got {jaccard_threshold!r}"
        )
    thresh = float(jaccard_threshold)
    ref_cases, ref_lbl = _load_normalized_cases(
        reference_source, default_label=reference_label or "reference"
    )
    ev_cases, ev_lbl = _load_normalized_cases(
        eval_source, default_label=eval_label or "eval"
    )

    ref_cand_set = {_norm_ws(c.candidate) for c in ref_cases}
    ref_rubric_set = {_norm_ws(c.rubric) for c in ref_cases}
    ref_domains = {c.domain for c in ref_cases if c.domain is not None}
    ev_domains = {c.domain for c in ev_cases if c.domain is not None}
    shared_domains = tuple(sorted(ref_domains & ev_domains))

    exact_cand = sum(1 for c in ev_cases if _norm_ws(c.candidate) in ref_cand_set)
    exact_rub = sum(1 for c in ev_cases if _norm_ws(c.rubric) in ref_rubric_set)

    # Inverted shingle index so candidate similarity stays fast even on large datasets
    ref_shingles = [_word_shingles(c.candidate, k=3) for c in ref_cases]
    posting: dict[str, list[int]] = {}
    for idx, sh in enumerate(ref_shingles):
        for gram in sh:
            posting.setdefault(gram, []).append(idx)

    near_dups = 0
    max_j = 0.0
    for c in ev_cases:
        ev_sh = _word_shingles(c.candidate, k=3)
        if not ev_sh:
            continue
        hit_counts: Counter[int] = Counter()
        for gram in ev_sh:
            for ref_idx in posting.get(gram, ()):
                hit_counts[ref_idx] += 1
        best_item_j = 0.0
        ev_len = len(ev_sh)
        for ref_idx, inter in hit_counts.items():
            union = ev_len + len(ref_shingles[ref_idx]) - inter
            if union > 0:
                j_val = inter / float(union)
                best_item_j = max(best_item_j, j_val)
        max_j = max(max_j, best_item_j)
        if best_item_j >= thresh:
            near_dups += 1

    warnings: list[str] = []
    if exact_cand > 0:
        warnings.append(f"{exact_cand} eval candidate(s) appear verbatim in reference split.")
    if exact_rub > 0:
        warnings.append(f"{exact_rub} eval case(s) reuse exact rubrics from reference split.")
    if near_dups > 0:
        warnings.append(
            f"{near_dups} eval candidate(s) have word-3-gram Jaccard >= {thresh:.2f} with reference split."
        )
    if shared_domains:
        warnings.append(f"Reference and eval splits share domain tag(s): {', '.join(shared_domains)}.")

    is_clean = (
        exact_cand == 0
        and exact_rub == 0
        and near_dups == 0
        and len(shared_domains) == 0
    )

    return SplitLeakageReport(
        reference_source=reference_label or ref_lbl,
        eval_source=eval_label or ev_lbl,
        reference_cases=len(ref_cases),
        eval_cases=len(ev_cases),
        exact_candidate_overlaps=exact_cand,
        exact_rubric_overlaps=exact_rub,
        near_duplicate_candidates=near_dups,
        max_candidate_jaccard=round(max_j, 4),
        shared_domains=shared_domains,
        jaccard_threshold=thresh,
        is_clean=is_clean,
        warnings=tuple(warnings),
    )
