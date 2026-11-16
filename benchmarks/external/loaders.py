"""Zero-dependency HuggingFace Datasets Server loaders, disk caching, and normalizers.

Provides robust fetching via httpx against the public HuggingFace /rows API with
automatic fallback to verified direct CDN URLs, local disk caching under
~/.cache/logitgate/external_benchmarks/, and schema normalizers for all canonical
external public benchmarks.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from benchmarks.external.registry import (
    EXTERNAL_BENCHMARKS,
    ExternalBenchmarkSpec,
    validate_benchmark_license,
)

if TYPE_CHECKING:
    from logitgate.benchmark import BenchmarkCase


@dataclass(frozen=True, slots=True)
class NormalizedBinaryItem:
    """Normalized single-candidate item with binary truth expectation."""

    id: str
    rubric: str
    candidate: str
    expected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedPairwiseItem:
    """Normalized two-candidate comparison item."""

    id: str
    rubric: str
    candidate_a: str
    candidate_b: str
    winner: int  # 0 for candidate_a, 1 for candidate_b
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedPoolItem:
    """Normalized multi-candidate pool item for Best-of-N selection."""

    id: str
    rubric: str
    candidates: list[str]
    expected_winner: int  # 0 <= expected_winner < len(candidates)
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_default_cache_dir() -> Path:
    base = os.environ.get("LOGITGATE_CACHE_DIR")
    if base:
        return Path(base) / "external_benchmarks"
    return Path.home() / ".cache" / "logitgate" / "external_benchmarks"


def _load_local_fixture(path: Path) -> list[dict[str, Any]]:
    """Loads raw rows from a JSON or JSONL fixture file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            raw_rows = data.get("rows", [])
        elif isinstance(data, list):
            raw_rows = data
        else:
            raw_rows = []
    except json.JSONDecodeError:
        raw_rows = []
        for line in text.splitlines():
            line_str = line.strip()
            if line_str:
                raw_rows.append(json.loads(line_str))

    extracted: list[dict[str, Any]] = []
    for item in raw_rows:
        if isinstance(item, dict):
            extracted.append(item.get("row", item))
    return extracted


def _fetch_fallback_content(
    url: str,
    client: httpx.Client | None,
    timeout: float = 30.0,
) -> str:
    """Fetches raw text content from a fallback URL, following redirects."""
    if client is not None:
        try:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.text
        except (TypeError, httpx.RequestError):
            pass

    with httpx.Client(timeout=timeout, follow_redirects=True) as h:
        resp = h.get(url)
        resp.raise_for_status()
        return resp.text


def _fetch_from_fallback_urls(
    spec: ExternalBenchmarkSpec,
    limit: int = 100,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetches and normalizes raw entries directly from direct CDN fallback URLs."""
    if not spec.fallback_urls:
        return []

    norm_key = spec.key.lower()

    if norm_key == "llmbar":
        text = _fetch_fallback_content(spec.fallback_urls[0], client, timeout)
        data = json.loads(text)
        return data[:limit] if isinstance(data, list) else []

    if norm_key == "judgebench":
        text = _fetch_fallback_content(spec.fallback_urls[0], client, timeout)
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line_str = line.strip()
            if line_str:
                rows.append(json.loads(line_str))
                if len(rows) >= limit:
                    break
        return rows

    if norm_key == "truthfulqa":
        text = _fetch_fallback_content(spec.fallback_urls[0], client, timeout)
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for r in reader:
            rows.append(dict(r))
            if len(rows) >= limit:
                break
        return rows

    if norm_key == "halueval":
        text = _fetch_fallback_content(spec.fallback_urls[0], client, timeout)
        rows = []
        for idx, line in enumerate(text.splitlines()):
            line_str = line.strip()
            if not line_str:
                continue
            d = json.loads(line_str)
            if "right_answer" in d and "hallucinated_answer" in d:
                rows.append(
                    {
                        "id": f"halueval_{idx}_right",
                        "knowledge": d.get("knowledge", ""),
                        "question": d.get("question", ""),
                        "answer": d.get("right_answer", ""),
                        "hallucination": "no",
                    }
                )
                rows.append(
                    {
                        "id": f"halueval_{idx}_hallu",
                        "knowledge": d.get("knowledge", ""),
                        "question": d.get("question", ""),
                        "answer": d.get("hallucinated_answer", ""),
                        "hallucination": "yes",
                    }
                )
            else:
                rows.append(d)
            if len(rows) >= limit:
                break
        return rows[:limit]

    if norm_key == "ragtruth":
        resp_url = spec.fallback_urls[0]
        src_url = spec.fallback_urls[1] if len(spec.fallback_urls) > 1 else None
        text_resp = _fetch_fallback_content(resp_url, client, timeout)
        sources: dict[str, dict[str, Any]] = {}
        if src_url:
            text_src = _fetch_fallback_content(src_url, client, timeout)
            for line in text_src.splitlines():
                line_str = line.strip()
                if line_str:
                    d = json.loads(line_str)
                    if "source_id" in d:
                        sources[str(d["source_id"])] = d

        faithful_rows: list[dict[str, Any]] = []
        hallucinated_rows: list[dict[str, Any]] = []
        half_target = max(1, (limit + 1) // 2)
        for line in text_resp.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            d = json.loads(line_str)
            if d.get("split") and d.get("split") != spec.split:
                continue
            if d.get("quality") and d.get("quality") != "good":
                continue
            labels_list = d.get("labels") or []
            is_hallu = isinstance(labels_list, list) and len(labels_list) > 0
            if is_hallu and len(hallucinated_rows) >= half_target:
                continue
            if not is_hallu and len(faithful_rows) >= half_target:
                continue
            src = sources.get(str(d.get("source_id")), {})
            entry = {
                "id": str(d.get("id")),
                "source_id": d.get("source_id"),
                "model": d.get("model"),
                "temperature": d.get("temperature"),
                "labels": labels_list,
                "split": d.get("split", spec.split),
                "quality": d.get("quality"),
                "response": d.get("response", ""),
                "answer": d.get("response", ""),
                "prompt": src.get("prompt", ""),
                "question": src.get("prompt", ""),
                "context": src.get("source_info", ""),
                "source_info": src.get("source_info", ""),
                "task_type": src.get("task_type", ""),
            }
            if is_hallu:
                hallucinated_rows.append(entry)
            else:
                faithful_rows.append(entry)
            if len(faithful_rows) >= half_target and len(hallucinated_rows) >= half_target:
                break

        rows = []
        for idx_pair in range(max(len(faithful_rows), len(hallucinated_rows))):
            if idx_pair < len(faithful_rows) and len(rows) < limit:
                rows.append(faithful_rows[idx_pair])
            if idx_pair < len(hallucinated_rows) and len(rows) < limit:
                rows.append(hallucinated_rows[idx_pair])
        return rows

    # Generic fallback
    text = _fetch_fallback_content(spec.fallback_urls[0], client, timeout)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data[:limit]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        line_str = line.strip()
        if line_str:
            rows.append(json.loads(line_str))
            if len(rows) >= limit:
                break
    return rows


def fetch_raw_hf_rows(
    spec: ExternalBenchmarkSpec,
    limit: int = 100,
    *,
    client: httpx.Client | None = None,
    local_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetches raw rows for an external benchmark specification.

    Tries the public HuggingFace /rows API first, falling back to direct CDN URLs if
    the datasets server returns >= 400 or a network error.
    Honors local fixtures and disk caching.
    """
    if local_path is not None:
        p = Path(local_path)
        if not p.is_file():
            raise FileNotFoundError(f"Local fixture file not found: {p}")
        rows = _load_local_fixture(p)
        return rows[:limit] if limit is not None else rows

    resolved_cache_dir = (
        Path(cache_dir) if cache_dir is not None else _get_default_cache_dir()
    )
    safe_key = spec.key.replace("/", "_")
    safe_cfg = spec.config_name.replace("/", "_")
    safe_split = spec.split.replace("/", "_")
    cache_file = resolved_cache_dir / f"{safe_key}_{safe_cfg}_{safe_split}_{limit}.json"

    if cache_file.is_file() and not force_refresh:
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached_data, dict) and "rows" in cached_data:
                cached_rows = [item.get("row", item) for item in cached_data["rows"]]
                return cached_rows[:limit]
        except (json.JSONDecodeError, OSError):
            pass

    # Remote fetch from HuggingFace Datasets Server with fallback to direct CDN URLs
    headers: dict[str, str] = {"Accept": "application/json"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    all_rows: list[dict[str, Any]] = []
    offset = 0
    max_batch = 100

    try:
        try:
            while len(all_rows) < limit:
                batch_len = min(max_batch, limit - len(all_rows))
                url = (
                    f"https://datasets-server.huggingface.co/rows"
                    f"?dataset={spec.hf_dataset_id}&config={spec.config_name}&split={spec.split}"
                    f"&offset={offset}&length={batch_len}"
                )
                resp = http.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()

                page_rows = payload.get("rows", [])
                if not page_rows:
                    break

                for item in page_rows:
                    if isinstance(item, dict):
                        all_rows.append(item.get("row", item))
                    if len(all_rows) >= limit:
                        break

                offset += len(page_rows)
                if len(page_rows) < batch_len:
                    break
        except (httpx.HTTPStatusError, httpx.RequestError):
            if spec.fallback_urls:
                all_rows = _fetch_from_fallback_urls(
                    spec, limit=limit, client=http, timeout=timeout
                )
            else:
                raise
    finally:
        if owns_client:
            http.close()

    # Save to disk cache
    try:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_payload = {"rows": [{"row": r} for r in all_rows]}
        cache_file.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    except OSError:
        pass

    return all_rows


def normalize_llmbar_row(
    row: dict[str, Any], idx: int = 0, seed: int = 42
) -> NormalizedPairwiseItem:
    """Normalizes an LLMBar row into a NormalizedPairwiseItem."""
    row_id = str(row.get("id") or f"llmbar_{idx}")
    instruction = str(
        row.get("input") or row.get("instruction") or row.get("prompt") or ""
    )
    rubric = (
        f"Does the response adhere to all constraints and accurately follow the instruction?\n\n"
        f"Instruction: {instruction}"
        if instruction
        else "Does the response accurately follow the instruction?"
    )
    output_1 = str(row.get("output_1") or row.get("response_1") or "")
    output_2 = str(row.get("output_2") or row.get("response_2") or "")
    raw_label = row.get("label")

    if raw_label in (1, "1", "output_1") or raw_label == 0:
        orig_winner = 0
    elif raw_label in (2, "2", "output_2"):
        orig_winner = 1
    else:
        orig_winner = 0

    rng = random.Random(f"{seed}:llmbar:{idx}:{row_id}")
    swap = rng.random() < 0.5
    if swap:
        cand_a, cand_b = output_2, output_1
        winner = 1 - orig_winner
    else:
        cand_a, cand_b = output_1, output_2
        winner = orig_winner

    metadata = {
        "dataset": "llmbar",
        "original_input": instruction,
        "original_label": raw_label,
        "swapped": swap,
    }
    return NormalizedPairwiseItem(
        id=row_id,
        rubric=rubric,
        candidate_a=cand_a,
        candidate_b=cand_b,
        winner=winner,
        metadata=metadata,
    )


def normalize_judgebench_row(
    row: dict[str, Any], idx: int = 0, seed: int = 42
) -> NormalizedPairwiseItem:
    """Normalizes a JudgeBench row into a NormalizedPairwiseItem."""
    row_id = str(
        row.get("pair_id")
        or row.get("original_id")
        or row.get("id")
        or f"judgebench_{idx}"
    )
    question = str(row.get("question") or row.get("prompt") or "")
    rubric = (
        f"Which response provides the more accurate, logically sound, and complete answer?\n\n"
        f"Question: {question}"
        if question
        else "Which response is more accurate and logically sound?"
    )
    resp_a = str(row.get("response_A") or row.get("response_a") or "")
    resp_b = str(row.get("response_B") or row.get("response_b") or "")
    raw_label = str(row.get("label", "A>B")).strip()

    if raw_label in ("A>B", "A", "0", 0):
        orig_winner = 0
    else:
        orig_winner = 1

    rng = random.Random(f"{seed}:judgebench:{idx}:{row_id}")
    swap = rng.random() < 0.5
    if swap:
        cand_a, cand_b = resp_b, resp_a
        winner = 1 - orig_winner
    else:
        cand_a, cand_b = resp_a, resp_b
        winner = orig_winner

    metadata = {
        "dataset": "judgebench",
        "pair_id": row_id,
        "source": row.get("source"),
        "response_model": row.get("response_model"),
        "original_label": raw_label,
        "swapped": swap,
    }
    return NormalizedPairwiseItem(
        id=row_id,
        rubric=rubric,
        candidate_a=cand_a,
        candidate_b=cand_b,
        winner=winner,
        metadata=metadata,
    )


def normalize_truthfulqa_row(
    row: dict[str, Any], idx: int = 0, seed: int = 42
) -> NormalizedPairwiseItem:
    """Normalizes a TruthfulQA row into a slot-balanced NormalizedPairwiseItem."""
    row_id = str(row.get("id") or f"truthfulqa_{idx}")
    question = str(row.get("question") or row.get("Question") or "")
    rubric = (
        f"Which answer is factual, truthful, and free of common misconceptions?\n\n"
        f"Question: {question}"
        if question
        else "Which answer is factual, truthful, and free of misconceptions?"
    )
    best_answer = str(
        row.get("best_answer")
        or row.get("Best Answer")
        or (
            row.get("correct_answers", [""])[0]
            if isinstance(row.get("correct_answers"), list)
            else ""
        )
        or row.get("Correct Answers", "")
    )
    incorrect = (
        row.get("incorrect_answers")
        or row.get("Incorrect Answers")
        or row.get("misconceptions")
        or []
    )
    best_incorrect = str(
        row.get("Best Incorrect Answer") or row.get("incorrect_answer") or ""
    )
    if best_incorrect:
        incorrect_answer = best_incorrect
    elif isinstance(incorrect, list) and incorrect:
        incorrect_answer = str(incorrect[0])
    elif isinstance(incorrect, str):
        incorrect_answer = incorrect
    else:
        incorrect_answer = ""

    rng = random.Random(f"{seed}:truthfulqa:{idx}:{row_id}")
    put_best_first = rng.random() < 0.5
    if put_best_first:
        cand_a = best_answer
        cand_b = incorrect_answer
        winner = 0
    else:
        cand_a = incorrect_answer
        cand_b = best_answer
        winner = 1

    metadata = {
        "dataset": "truthfulqa",
        "question": question,
        "best_answer": best_answer,
        "incorrect_answer": incorrect_answer,
        "swapped": not put_best_first,
    }
    return NormalizedPairwiseItem(
        id=row_id,
        rubric=rubric,
        candidate_a=cand_a,
        candidate_b=cand_b,
        winner=winner,
        metadata=metadata,
    )


def normalize_halueval_row(
    row: dict[str, Any], idx: int = 0, seed: int = 42
) -> NormalizedBinaryItem:
    """Normalizes a HaluEval QA row into a NormalizedBinaryItem."""
    row_id = str(row.get("id") or f"halueval_{idx}")
    knowledge = str(row.get("knowledge") or row.get("context") or "")
    question = str(row.get("question") or "")
    if (
        row.get("hallucination") is None
        and row.get("right_answer")
        and row.get("hallucinated_answer")
    ):
        rng = random.Random(f"{seed}:halueval:{idx}:{row_id}")
        use_right = rng.random() < 0.5
        answer = str(row["right_answer"] if use_right else row["hallucinated_answer"])
        has_hallucination = not use_right
        raw_hallu = "no" if use_right else "yes"
    else:
        answer = str(row.get("answer") or row.get("right_answer") or "")
        raw_hallu = str(row.get("hallucination", "no")).strip().lower()
        has_hallucination = raw_hallu in ("yes", "true", "1")
    expected = not has_hallucination

    rubric = "Is the answer fully factual, grounded in the provided knowledge, and free of hallucinations?"
    if knowledge:
        candidate = (
            f"Knowledge:\n{knowledge}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
        )
    else:
        candidate = f"Question:\n{question}\n\nAnswer:\n{answer}"

    metadata = {
        "dataset": "halueval",
        "knowledge": knowledge,
        "question": question,
        "answer": answer,
        "hallucination": raw_hallu,
    }
    return NormalizedBinaryItem(
        id=row_id,
        rubric=rubric,
        candidate=candidate,
        expected=expected,
        metadata=metadata,
    )


def normalize_ragtruth_row(
    row: dict[str, Any], idx: int = 0, seed: int = 42
) -> NormalizedBinaryItem:
    """Normalizes a RAGTruth row into a NormalizedBinaryItem with fine-grained span metadata."""
    del seed
    row_id = str(row.get("id") or f"ragtruth_{idx}")
    question = str(row.get("question") or row.get("query") or row.get("prompt") or "")
    ctx = row.get("context") or row.get("source_info")
    if isinstance(ctx, list):
        context_str = "\n".join(str(c) for c in ctx)
    else:
        context_str = str(ctx or "")

    answer = str(row.get("answer") or row.get("response") or row.get("output") or "")
    raw_labels: Any = (
        row.get("labels")
        or row.get("spans")
        or row.get("hallucination_spans")
        or row.get("hallucination_labels")
    )
    if isinstance(raw_labels, str):
        stripped = raw_labels.strip()
        if stripped.startswith("["):
            try:
                raw_labels = json.loads(stripped)
            except json.JSONDecodeError:
                raw_labels = []
        else:
            raw_labels = []
    processed = row.get("hallucination_labels_processed")
    hallu_label = row.get("hallucination_label") or row.get("hallucination")

    has_hallucination = False
    if isinstance(raw_labels, list) and len(raw_labels) > 0 or isinstance(processed, dict) and sum(
        int(v) for v in processed.values() if isinstance(v, (int, float))
    ) > 0:
        has_hallucination = True
    elif hallu_label is not None:
        has_hallucination = str(hallu_label).strip().lower() in (
            "yes",
            "true",
            "1",
            "hallucination",
        )

    expected = not has_hallucination
    rubric = "Is the answer fully faithful and grounded in the retrieved context with no hallucinated claims?"
    if context_str:
        candidate = (
            f"Context:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
        )
    else:
        candidate = f"Question:\n{question}\n\nAnswer:\n{answer}"

    metadata = {
        "dataset": "ragtruth",
        "question": question,
        "context": context_str,
        "answer": answer,
        "spans": raw_labels,
        "has_hallucination": has_hallucination,
    }
    return NormalizedBinaryItem(
        id=row_id,
        rubric=rubric,
        candidate=candidate,
        expected=expected,
        metadata=metadata,
    )


def normalize_rewardbench2_row(
    row: dict[str, Any], idx: int = 0, seed: int = 42
) -> NormalizedPoolItem:
    """Normalizes a RewardBench-2 row into a 4-way candidate pool with deterministic slot balancing."""
    row_id = str(row.get("id") or f"rewardbench2_{idx}")
    prompt_text = str(row.get("prompt") or "")
    chosen = row.get("chosen")
    if isinstance(chosen, list):
        chosen_text = str(chosen[0] if chosen else "")
    else:
        chosen_text = str(chosen or "")

    rejected = row.get("rejected") or []
    if isinstance(rejected, list):
        rejected_texts = [str(r) for r in rejected]
    else:
        rejected_texts = [str(rejected)]

    pool: list[tuple[str, bool]] = [(chosen_text, True)] + [
        (r, False) for r in rejected_texts
    ]
    rng = random.Random(f"{seed}:rewardbench2:{idx}:{row_id}")
    rng.shuffle(pool)

    candidates = [p[0] for p in pool]
    expected_winner = next(i for i, p in enumerate(pool) if p[1])

    rubric = (
        f"Which response best answers the prompt and satisfies all instructions?\n\n"
        f"Prompt: {prompt_text}"
        if prompt_text
        else "Which response provides the highest quality, most accurate output?"
    )
    metadata = {
        "dataset": "rewardbench2",
        "prompt": prompt_text,
        "orig_chosen": chosen_text,
        "models": row.get("models"),
        "subset": row.get("subset"),
    }
    return NormalizedPoolItem(
        id=row_id,
        rubric=rubric,
        candidates=candidates,
        expected_winner=expected_winner,
        metadata=metadata,
    )


def normalize_benchmark_row(
    key: str,
    row: dict[str, Any],
    idx: int = 0,
    seed: int = 42,
) -> NormalizedBinaryItem | NormalizedPairwiseItem | NormalizedPoolItem:
    """Dispatches a raw benchmark row to the canonical normalizer for key."""
    norm_key = key.lower()
    if norm_key == "llmbar":
        return normalize_llmbar_row(row, idx=idx, seed=seed)
    if norm_key == "judgebench":
        return normalize_judgebench_row(row, idx=idx, seed=seed)
    if norm_key == "truthfulqa":
        return normalize_truthfulqa_row(row, idx=idx, seed=seed)
    if norm_key == "halueval":
        return normalize_halueval_row(row, idx=idx, seed=seed)
    if norm_key == "ragtruth":
        return normalize_ragtruth_row(row, idx=idx, seed=seed)
    if norm_key == "rewardbench2":
        return normalize_rewardbench2_row(row, idx=idx, seed=seed)
    raise ValueError(
        f"Unknown benchmark key '{key}'. Available canonical suites: {sorted(EXTERNAL_BENCHMARKS)}"
    )


def load_external_benchmark(
    spec_or_key: str | ExternalBenchmarkSpec,
    limit: int = 100,
    seed: int = 42,
    *,
    client: httpx.Client | None = None,
    local_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
    timeout: float = 30.0,
) -> list[NormalizedBinaryItem | NormalizedPairwiseItem | NormalizedPoolItem]:
    """Loads and normalizes an external benchmark suite.

    Validates license rules before loading.
    """
    if isinstance(spec_or_key, str):
        norm_key = spec_or_key.lower()
        if norm_key not in EXTERNAL_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark '{spec_or_key}'. Available: {sorted(EXTERNAL_BENCHMARKS)}"
            )
        spec = EXTERNAL_BENCHMARKS[norm_key]
    else:
        spec = spec_or_key

    validate_benchmark_license(spec)
    raw_rows = fetch_raw_hf_rows(
        spec,
        limit=limit,
        client=client,
        local_path=local_path,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        timeout=timeout,
    )

    items: list[NormalizedBinaryItem | NormalizedPairwiseItem | NormalizedPoolItem] = []
    for idx, row in enumerate(raw_rows):
        items.append(normalize_benchmark_row(spec.key, row, idx=idx, seed=seed))
    return items


def pairwise_to_binary_cases(
    items: Sequence[NormalizedPairwiseItem],
) -> list[BenchmarkCase]:
    """Converts normalized pairwise items into standard binary BenchmarkCase instances.

    Each pairwise item is mapped into two balanced cases:
    - (rubric, winner_candidate, True)
    - (rubric, loser_candidate, False)
    """
    from logitgate.benchmark import BenchmarkCase

    cases: list[BenchmarkCase] = []
    for item in items:
        if item.winner == 0:
            winner_text = item.candidate_a
            loser_text = item.candidate_b
        else:
            winner_text = item.candidate_b
            loser_text = item.candidate_a
        cases.append(
            BenchmarkCase(rubric=item.rubric, candidate=winner_text, expected=True)
        )
        cases.append(
            BenchmarkCase(rubric=item.rubric, candidate=loser_text, expected=False)
        )
    return cases
