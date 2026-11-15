from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from logitgate.debiasing.inversion import _sanitize_nonce, render_candidate
from logitgate.distill._io import atomic_write_json_nofollow, read_json_object_nofollow
from logitgate.exceptions import ConfigurationError
from logitgate.telemetry.models import DistillationRecord

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS: frozenset[str] = frozenset({
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "candidate",
    "criterion",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "rubric",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "without",
})
_VERDICT_TAG_RE = re.compile(r"\[VERDICT:\s*(?:PASS|FAIL)\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _sha256_norm(text: str) -> str:
    norm = _WHITESPACE_RE.sub(" ", text.strip())
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def _tokenize_bm25(text: str) -> list[str]:
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 2 and tok not in _STOPWORDS
    ]


def _jaccard_tokens(a: str, b: str) -> float:
    sa = set(_tokenize_bm25(a))
    sb = set(_tokenize_bm25(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _condense_rationale(raw_reason: str | None, *, passed: bool, max_chars: int) -> str:
    """Distills a multi-bullet teacher critique into a concise, token-neutral 1-line rationale."""
    default_msg = (
        "Satisfies every required rubric condition despite surface caveats or edge-case phrasing."
        if passed
        else "Violates at least one required constraint, boundary condition, or factual scope."
    )
    if not raw_reason or not raw_reason.strip():
        return default_msg[:max_chars]
    cleaned = _VERDICT_TAG_RE.sub("", raw_reason).strip()
    lines = [
        ln.lstrip("-*•0123456789.) ").strip()
        for ln in cleaned.splitlines()
        if ln.strip()
    ]
    if not lines:
        return default_msg[:max_chars]
    # Prefer the concluding substantive bullet, which typically states the decisive reason.
    chosen = lines[-1] if len(lines[-1]) >= 20 else " ".join(lines)
    chosen = _WHITESPACE_RE.sub(" ", chosen).strip()
    # Scrub standalone "Option A" / "Option B" references so anchors stay 100% slot-neutral.
    chosen = re.sub(r"\boption\s+[ab]\b", "the corresponding option", chosen, flags=re.IGNORECASE)
    if len(chosen) <= max_chars:
        return chosen
    return chosen[: max(16, max_chars - 3)].rstrip() + "..."


def _truncate_anchor_candidate(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(32, max_chars - 3)].rstrip() + "..."


class AnchorExemplar(BaseModel):
    """Single teacher-verified reference exemplar inside a `DistilledAnchorBank`."""

    model_config = ConfigDict(extra="forbid")

    exemplar_id: str
    rubric: str
    candidate: str
    passed: bool
    rationale: str = ""
    domain: str | None = None
    candidate_sha256: str = ""

    @model_validator(mode="after")
    def _validate_exemplar(self) -> AnchorExemplar:
        if not self.rubric or not self.rubric.strip():
            raise ConfigurationError("AnchorExemplar.rubric cannot be empty.")
        if not self.candidate or not self.candidate.strip():
            raise ConfigurationError("AnchorExemplar.candidate cannot be empty.")
        if not self.candidate_sha256:
            self.candidate_sha256 = _sha256_norm(self.candidate)
        return self


class ContrastiveAnchorPair(BaseModel):
    """Paired (PASS, FAIL) contrast exemplars teaching abstract boundary discrimination."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    rubric: str
    domain: str | None = None
    positive: AnchorExemplar
    negative: AnchorExemplar
    priority_score: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_pair(self) -> ContrastiveAnchorPair:
        if not self.positive.passed:
            raise ConfigurationError("ContrastiveAnchorPair.positive must have passed=True.")
        if self.negative.passed:
            raise ConfigurationError("ContrastiveAnchorPair.negative must have passed=False.")
        if not math.isfinite(self.priority_score) or self.priority_score < 0.0:
            raise ConfigurationError("priority_score must be a finite non-negative float.")
        return self


def _load_generic_cases(
    cases_or_path: str | Path | Sequence[Any],
) -> list[dict[str, Any]]:
    """Loads dataset rows from a JSON/JSONL/Markdown file or in-memory sequence."""
    if isinstance(cases_or_path, (str, Path)):
        p = Path(cases_or_path)
        suffix = p.suffix.lower()
        if suffix == ".json":
            raw_text = read_json_or_array_nofollow(p)
            if not isinstance(raw_text, list):
                raise ConfigurationError(f"Dataset JSON '{p}' must contain a list of items.")
            raw_items: Sequence[Any] = raw_text
        elif suffix == ".jsonl":
            from logitgate.telemetry.sinks import (
                _DEFAULT_MAX_READ_BYTES,
                read_regular_text_nofollow,
            )

            txt = read_regular_text_nofollow(
                p, max_bytes=_DEFAULT_MAX_READ_BYTES, label="dataset JSONL"
            )
            raw_items = [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
        elif suffix in (".md", ".markdown"):
            from logitgate.calibration.conformal import load_calibration_dataset

            md_cases = load_calibration_dataset(p)
            return [
                {
                    "id": f"md_{idx}",
                    "rubric": rub,
                    "candidate": cand,
                    "expected": bool(lbl),
                    "domain": None,
                }
                for idx, (rub, cand, lbl) in enumerate(md_cases)
            ]
        else:
            raise ConfigurationError(
                f"Unsupported dataset format '{p.suffix}'; expected .json, .jsonl, or .md"
            )
    else:
        raw_items = cases_or_path

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, Mapping):
            rubric = str(item.get("rubric") or "").strip()
            cand_raw = item.get("candidate")
            cand = render_candidate(cand_raw) if cand_raw is not None else ""
            label_raw = (
                item["expected"]
                if "expected" in item
                else item.get("label", item.get("passed", item.get("teacher_passed")))
            )
            if not isinstance(label_raw, bool) and not (
                isinstance(label_raw, int) and label_raw in (0, 1)
            ):
                continue
            if not rubric or not cand:
                continue
            normalized.append({
                "id": str(item.get("id") or item.get("event_id") or f"item_{idx}"),
                "rubric": rubric,
                "candidate": cand,
                "expected": bool(label_raw),
                "domain": str(item["domain"]) if item.get("domain") else None,
                "rationale": str(
                    item.get("teacher_reason") or item.get("rationale") or item.get("reason") or ""
                ),
            })
        else:
            rubric = str(getattr(item, "rubric", "") or "").strip()
            cand_raw = getattr(item, "candidate", None)
            cand = render_candidate(cand_raw) if cand_raw is not None else ""
            label_raw = getattr(
                item,
                "expected",
                getattr(item, "label", getattr(item, "passed", None)),
            )
            if not isinstance(label_raw, bool) or not rubric or not cand:
                continue
            normalized.append({
                "id": str(getattr(item, "id", getattr(item, "event_id", f"item_{idx}"))),
                "rubric": rubric,
                "candidate": cand,
                "expected": bool(label_raw),
                "domain": getattr(item, "domain", None),
                "rationale": str(getattr(item, "teacher_reason", "") or ""),
            })
    return normalized


def read_json_or_array_nofollow(path: Path) -> Any:
    from logitgate.telemetry.sinks import (
        _DEFAULT_MAX_READ_BYTES,
        read_regular_text_nofollow,
    )

    txt = read_regular_text_nofollow(
        path, max_bytes=_DEFAULT_MAX_READ_BYTES, label="dataset JSON"
    )
    try:
        return json.loads(txt)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid JSON in '{path}': {exc}") from exc


class DistilledAnchorBank(BaseModel):
    """Portable, model-agnostic contrastive exemplar bank for weight-free BYO distillation.

    Compiles teacher-adjudicated ``(PASS, FAIL)`` minimal contrast pairs from Split-A datasets
    or `DistillationSink` telemetry. At evaluation time, retrieves the most rubric-relevant
    contrast pair via zero-dependency Okapi BM25 (excluding exact candidate matches) and formats
    a token-neutral ``[CALIBRATION ANCHORS]`` block inside the shared KV-cache prefix.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    source: str | None = None
    teacher_model: str | None = None
    pairs: list[ContrastiveAnchorPair] = Field(default_factory=list)
    max_pairs: int = Field(default=1, ge=1, le=8)
    min_rubric_overlap: float = Field(default=0.25, ge=0.0, le=1.0)
    max_candidate_chars: int = Field(default=420, ge=64, le=8192)
    max_rationale_chars: int = Field(default=180, ge=32, le=1024)

    @model_validator(mode="after")
    def _validate_bank(self) -> DistilledAnchorBank:
        if self.schema_version != 1:
            raise ConfigurationError(
                f"Unsupported DistilledAnchorBank schema_version={self.schema_version}; expected 1."
            )
        if not self.pairs:
            raise ConfigurationError("DistilledAnchorBank must contain at least one ContrastiveAnchorPair.")
        if not math.isfinite(self.min_rubric_overlap) or not (0.0 <= self.min_rubric_overlap <= 1.0):
            raise ConfigurationError("min_rubric_overlap must be a finite float in [0.0, 1.0].")
        return self

    @classmethod
    def from_dataset(
        cls,
        cases_or_path: str | Path | Sequence[Any],
        *,
        teacher_reasons: Mapping[str, str] | None = None,
        teacher_labels: Mapping[str, bool] | None = None,
        teacher_model: str | None = None,
        max_bank_pairs: int = 24,
        default_retrieve_pairs: int = 1,
        min_rubric_overlap: float = 0.25,
        max_candidate_chars: int = 420,
        max_rationale_chars: int = 180,
    ) -> DistilledAnchorBank:
        """Compiles contrastive minimal-pair anchors without relying on dataset-specific tags.

        Within each rubric, positive and negative items with the highest surface lexical
        similarity (Jaccard overlap) are paired first so the bank naturally captures
        hard-positive / hard-negative counterfactual boundaries on any BYO dataset.
        """
        if isinstance(max_bank_pairs, bool) or not isinstance(max_bank_pairs, int) or max_bank_pairs < 1:
            raise ConfigurationError("max_bank_pairs must be a positive integer.")
        items = _load_generic_cases(cases_or_path)
        if not items:
            raise ConfigurationError("Cannot compile DistilledAnchorBank from empty dataset.")

        by_rubric: dict[str, list[dict[str, Any]]] = {}
        for row in items:
            item_id = row["id"]
            passed = (
                bool(teacher_labels[item_id])
                if teacher_labels is not None and item_id in teacher_labels
                else bool(row["expected"])
            )
            reason = (
                teacher_reasons[item_id]
                if teacher_reasons is not None and item_id in teacher_reasons
                else row.get("rationale", "")
            )
            enriched = {**row, "passed": passed, "rationale": reason}
            by_rubric.setdefault(row["rubric"], []).append(enriched)

        compiled_pairs: list[ContrastiveAnchorPair] = []
        for rubric, group in by_rubric.items():
            positives = [g for g in group if g["passed"]]
            negatives = [g for g in group if not g["passed"]]
            if not positives or not negatives:
                continue

            # Greedily pair (pos, neg) with highest lexical overlap -> natural minimal pairs!
            used_pos: set[int] = set()
            used_neg: set[int] = set()
            candidate_combos: list[tuple[float, int, int]] = []
            for pi, p_item in enumerate(positives):
                for ni, n_item in enumerate(negatives):
                    sim = _jaccard_tokens(p_item["candidate"], n_item["candidate"])
                    has_teacher = 0.25 if (p_item["rationale"] and n_item["rationale"]) else 0.0
                    candidate_combos.append((sim + has_teacher, pi, ni))
            candidate_combos.sort(key=lambda t: t[0], reverse=True)

            for score, pi, ni in candidate_combos:
                if pi in used_pos or ni in used_neg:
                    continue
                used_pos.add(pi)
                used_neg.add(ni)
                p_item = positives[pi]
                n_item = negatives[ni]
                pos_ex = AnchorExemplar(
                    exemplar_id=p_item["id"],
                    rubric=rubric,
                    candidate=_truncate_anchor_candidate(p_item["candidate"], max_candidate_chars * 2),
                    passed=True,
                    rationale=_condense_rationale(
                        p_item["rationale"], passed=True, max_chars=max_rationale_chars
                    ),
                    domain=p_item.get("domain"),
                    candidate_sha256=_sha256_norm(p_item["candidate"]),
                )
                neg_ex = AnchorExemplar(
                    exemplar_id=n_item["id"],
                    rubric=rubric,
                    candidate=_truncate_anchor_candidate(n_item["candidate"], max_candidate_chars * 2),
                    passed=False,
                    rationale=_condense_rationale(
                        n_item["rationale"], passed=False, max_chars=max_rationale_chars
                    ),
                    domain=n_item.get("domain"),
                    candidate_sha256=_sha256_norm(n_item["candidate"]),
                )
                compiled_pairs.append(
                    ContrastiveAnchorPair(
                        pair_id=f"pair_{len(compiled_pairs)}",
                        rubric=rubric,
                        domain=p_item.get("domain") or n_item.get("domain"),
                        positive=pos_ex,
                        negative=neg_ex,
                        priority_score=round(1.0 + max(0.0, score), 4),
                    )
                )

        if not compiled_pairs:
            # Fallback cross-rubric pairing if every rubric had only a single polarity
            all_pos = [
                g for grp in by_rubric.values() for g in grp if g["passed"]
            ]
            all_neg = [
                g for grp in by_rubric.values() for g in grp if not g["passed"]
            ]
            if not all_pos or not all_neg:
                raise ConfigurationError(
                    "DistilledAnchorBank requires at least one passing and one failing item."
                )
            for idx in range(min(len(all_pos), len(all_neg), max_bank_pairs)):
                p_item = all_pos[idx]
                n_item = all_neg[idx]
                compiled_pairs.append(
                    ContrastiveAnchorPair(
                        pair_id=f"pair_{idx}",
                        rubric=p_item["rubric"],
                        domain=p_item.get("domain"),
                        positive=AnchorExemplar(
                            exemplar_id=p_item["id"],
                            rubric=p_item["rubric"],
                            candidate=_truncate_anchor_candidate(p_item["candidate"], max_candidate_chars * 2),
                            passed=True,
                            rationale=_condense_rationale(
                                p_item["rationale"], passed=True, max_chars=max_rationale_chars
                            ),
                            domain=p_item.get("domain"),
                            candidate_sha256=_sha256_norm(p_item["candidate"]),
                        ),
                        negative=AnchorExemplar(
                            exemplar_id=n_item["id"],
                            rubric=n_item["rubric"],
                            candidate=_truncate_anchor_candidate(n_item["candidate"], max_candidate_chars * 2),
                            passed=False,
                            rationale=_condense_rationale(
                                n_item["rationale"], passed=False, max_chars=max_rationale_chars
                            ),
                            domain=n_item.get("domain"),
                            candidate_sha256=_sha256_norm(n_item["candidate"]),
                        ),
                        priority_score=1.0,
                    )
                )

        # Diversify across domains while keeping highest-priority counterfactual pairs first.
        compiled_pairs.sort(key=lambda pr: pr.priority_score, reverse=True)
        selected: list[ContrastiveAnchorPair] = []
        if len(compiled_pairs) <= max_bank_pairs:
            selected = compiled_pairs
        else:
            by_bucket: dict[tuple[str, str], list[ContrastiveAnchorPair]] = {}
            for pr in compiled_pairs:
                by_bucket.setdefault((pr.domain or "_default", pr.rubric), []).append(pr)
            bucket_keys = sorted(by_bucket.keys())
            cursor = 0
            while len(selected) < max_bank_pairs and any(by_bucket.values()):
                bk = bucket_keys[cursor % len(bucket_keys)]
                cursor += 1
                if by_bucket[bk]:
                    selected.append(by_bucket[bk].pop(0))

        source_str = str(cases_or_path) if isinstance(cases_or_path, (str, Path)) else "in_memory_dataset"
        return cls(
            source=source_str,
            teacher_model=teacher_model,
            pairs=selected,
            max_pairs=default_retrieve_pairs,
            min_rubric_overlap=min_rubric_overlap,
            max_candidate_chars=max_candidate_chars,
            max_rationale_chars=max_rationale_chars,
        )

    @classmethod
    def from_distillation_records(
        cls,
        records_or_path: str | Path | Sequence[DistillationRecord],
        *,
        max_bank_pairs: int = 24,
        default_retrieve_pairs: int = 1,
        min_rubric_overlap: float = 0.25,
        max_candidate_chars: int = 420,
        max_rationale_chars: int = 180,
    ) -> DistilledAnchorBank:
        """Compiles a `DistilledAnchorBank` directly from `DistillationSink` telemetry records."""
        if isinstance(records_or_path, (str, Path)):
            from logitgate.telemetry.distill import load_distillation_records

            records = load_distillation_records(records_or_path)
            source_str = str(records_or_path)
        else:
            records = list(records_or_path)
            source_str = "distillation_records"

        if not records:
            raise ConfigurationError("No DistillationRecords available to compile DistilledAnchorBank.")

        rows: list[dict[str, Any]] = []
        teacher_model: str | None = None
        for rec in records:
            if teacher_model is None and rec.teacher_model:
                teacher_model = rec.teacher_model
            disc_bonus = 1.0 if rec.discrepancy_type != "ambiguous_resolved" else 0.0
            rows.append({
                "id": rec.event_id,
                "rubric": rec.rubric,
                "candidate": rec.candidate,
                "expected": rec.label,
                "domain": str(rec.tags.get("domain")) if rec.tags.get("domain") else None,
                "rationale": rec.teacher_reason or "",
                "_disc_bonus": disc_bonus,
            })

        bank = cls.from_dataset(
            rows,
            teacher_model=teacher_model,
            max_bank_pairs=max_bank_pairs,
            default_retrieve_pairs=default_retrieve_pairs,
            min_rubric_overlap=min_rubric_overlap,
            max_candidate_chars=max_candidate_chars,
            max_rationale_chars=max_rationale_chars,
        )
        return bank.model_copy(update={"source": source_str})

    def retrieve(
        self,
        rubric: str,
        candidate: Any = None,
        *,
        k: int | None = None,
        min_rubric_overlap: float | None = None,
        exclude_exact: bool = True,
    ) -> list[ContrastiveAnchorPair]:
        """Retrieves up to ``k`` contrastive pairs ranked by Okapi BM25 rubric relevance.

        Note: Scoring and relevance gating (``min_rubric_overlap``) depend strictly on ``rubric``
        (not ``candidate``) so all candidates evaluated against a gate share a 100% static
        KV-cache prefix, abstaining cleanly (`[]`) on unrelated out-of-domain rubrics while
        ``exclude_exact`` drops any anchor whose candidate matches ``candidate`` verbatim.
        """
        target_k = self.max_pairs if k is None else k
        if target_k <= 0 or not self.pairs:
            return []
        eff_min_overlap = (
            self.min_rubric_overlap if min_rubric_overlap is None else float(min_rubric_overlap)
        )

        cand_hash: str | None = None
        if exclude_exact and candidate is not None:
            cand_hash = _sha256_norm(render_candidate(candidate))

        query_tokens = _tokenize_bm25(rubric)
        query_set = set(query_tokens)

        eligible: list[ContrastiveAnchorPair] = []
        for pr in self.pairs:
            if cand_hash is not None and (
                pr.positive.candidate_sha256 == cand_hash
                or pr.negative.candidate_sha256 == cand_hash
            ):
                continue
            if eff_min_overlap > 0.0:
                rub_set = set(_tokenize_bm25(pr.rubric))
                min_sz = min(len(query_set), len(rub_set))
                overlap = (len(query_set & rub_set) / float(min_sz)) if min_sz > 0 else 0.0
                if overlap < eff_min_overlap:
                    continue
            eligible.append(pr)
        if not eligible:
            return []

        docs = [
            _tokenize_bm25(f"{pr.rubric} {pr.domain or ''}")
            for pr in eligible
        ]
        n_docs = len(docs)
        avg_dl = max(1.0, sum(len(d) for d in docs) / float(n_docs))
        df: Counter[str] = Counter()
        for d in docs:
            for tok in set(d):
                df[tok] += 1

        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, int]] = []
        for idx, (pr, doc_toks) in enumerate(zip(eligible, docs)):
            tf = Counter(doc_toks)
            dl = len(doc_toks)
            bm25 = 0.0
            for q in query_tokens:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                n_q = df.get(q, 0)
                idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                denom = f + k1 * (1.0 - b + b * (dl / avg_dl))
                bm25 += idf * ((f * (k1 + 1.0)) / max(denom, 1e-9))
            tie_break = 0.15 * math.log1p(pr.priority_score)
            scored.append((bm25 + tie_break, idx))

        scored.sort(key=lambda item: (-item[0], item[1]))

        # Diversify across distinct rubrics/domains when target_k > 1.
        chosen: list[ContrastiveAnchorPair] = []
        seen_rubrics: set[str] = set()
        deferred: list[ContrastiveAnchorPair] = []
        for _, idx in scored:
            pr = eligible[idx]
            if pr.rubric not in seen_rubrics:
                chosen.append(pr)
                seen_rubrics.add(pr.rubric)
                if len(chosen) >= target_k:
                    return chosen
            else:
                deferred.append(pr)
        for pr in deferred:
            chosen.append(pr)
            if len(chosen) >= target_k:
                break
        return chosen

    def format_anchor_block(
        self,
        rubric: str,
        candidate: Any = None,
        *,
        boundary_nonce: str | None = None,
        k: int | None = None,
    ) -> str:
        """Formats a token-neutral, nonce-isolated calibration reference block."""
        selected = self.retrieve(rubric, candidate=candidate, k=k, exclude_exact=True)
        if not selected:
            return ""
        safe_nonce = _sanitize_nonce(boundary_nonce)
        tag_sfx = f"_{safe_nonce}" if safe_nonce else ""
        header = (
            "Apply strict requirement verification (ignore surface length or confident tone; "
            "every rubric condition must hold; cautionary caveats alone are not defects):"
        )
        lines: list[str] = [header]
        for idx, pair in enumerate(selected, start=1):
            pos_cand = _truncate_anchor_candidate(pair.positive.candidate, self.max_candidate_chars)
            neg_cand = _truncate_anchor_candidate(pair.negative.candidate, self.max_candidate_chars)
            if pair.rubric.strip() != rubric.strip():
                lines.append(f"Reference Rubric {idx}: {pair.rubric.strip()}")
            lines.append(
                f"<reference_pass{tag_sfx}>\n{pos_cand}\n</reference_pass{tag_sfx}>\n"
                f"Reference Outcome: PASS — {pair.positive.rationale}"
            )
            lines.append(
                f"<reference_fail{tag_sfx}>\n{neg_cand}\n</reference_fail{tag_sfx}>\n"
                f"Reference Outcome: FAIL — {pair.negative.rationale}"
            )
        return "\n".join(lines)

    def cache_fingerprint(self) -> dict[str, Any]:
        pair_digests = [
            f"{p.pair_id}:{p.positive.candidate_sha256[:12]}:{p.negative.candidate_sha256[:12]}"
            for p in self.pairs
        ]
        return {
            "schema_version": self.schema_version,
            "max_pairs": self.max_pairs,
            "min_rubric_overlap": round(self.min_rubric_overlap, 6),
            "max_candidate_chars": self.max_candidate_chars,
            "max_rationale_chars": self.max_rationale_chars,
            "pairs": pair_digests,
        }

    def save(self, path: str | Path) -> Path:
        return atomic_write_json_nofollow(
            path,
            self.model_dump(mode="json"),
            artifact_name="DistilledAnchorBank",
        )

    @classmethod
    def load(cls, path: str | Path) -> DistilledAnchorBank:
        data = read_json_object_nofollow(path, artifact_name="DistilledAnchorBank")
        try:
            return cls.model_validate(data)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid DistilledAnchorBank in '{path}': {exc}"
            ) from exc


def resolve_anchor_bank(
    anchor_bank: DistilledAnchorBank | str | Path | None,
) -> DistilledAnchorBank | None:
    if anchor_bank is None:
        return None
    if isinstance(anchor_bank, DistilledAnchorBank):
        return anchor_bank
    if isinstance(anchor_bank, (str, Path)):
        return DistilledAnchorBank.load(anchor_bank)
    raise ConfigurationError(
        f"anchor_bank must be a DistilledAnchorBank, path, or None, got {type(anchor_bank).__name__}"
    )
