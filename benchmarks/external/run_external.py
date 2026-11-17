"""Programmatic and CLI runner for Canonical External Public Benchmarks.

Supports:
- Pairwise suites (LLMBar, JudgeBench, TruthfulQA) via RankGate (with symmetric debiasing ablation)
  and Cascade conversion.
- Binary and stream-span suites (RAGTruth, HaluEval) via run_benchmark and StreamGuard mid-stream evaluation.
- Best-of-N suites (RewardBench-2) via BestOfNGate / RankGate 4-way selection.
- Exact binomial accuracy, Wilson 95% confidence intervals, and position-swap consistency rates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.external.loaders import (  # noqa: E402
    NormalizedBinaryItem,
    NormalizedPairwiseItem,
    NormalizedPoolItem,
    load_external_benchmark,
    pairwise_to_binary_cases,
)
from benchmarks.external.registry import EXTERNAL_BENCHMARKS  # noqa: E402
from logitgate import BestOfNGate, LogitGate, RankGate, StreamGuard  # noqa: E402
from logitgate.backends.base import resolve_logit_backend  # noqa: E402
from logitgate.benchmark import BenchmarkCase, run_benchmark  # noqa: E402

DEFAULT_BACKEND = "ollama://localhost:11434"


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Computes Wilson score confidence interval for a binomial proportion.

    Returns (lower_bound, upper_bound) clamped to [0.0, 1.0].
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    z = statistics.NormalDist(0, 1).inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (
        z * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    ) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def compute_latencies(latencies: Sequence[float]) -> dict[str, float]:
    """Computes mean, p50, and p95 latencies in milliseconds."""
    if not latencies:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    sorted_lat = sorted(latencies)
    mean_val = statistics.mean(sorted_lat)
    n = len(sorted_lat)
    p50_idx = max(0, min(n - 1, round(0.50 * (n - 1))))
    p95_idx = max(0, min(n - 1, round(0.95 * (n - 1))))
    return {
        "mean_ms": round(mean_val, 2),
        "p50_ms": round(sorted_lat[p50_idx], 2),
        "p95_ms": round(sorted_lat[p95_idx], 2),
    }


def evaluate_pairwise(
    items: Sequence[NormalizedPairwiseItem],
    backend: Any,
    model: str | None = None,
    *,
    symmetric: bool = True,
    evaluate_cascade: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluates pairwise benchmarks with RankGate and optional Cascade conversion."""
    correct = 0
    consistent = 0
    latencies: list[float] = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        gate = RankGate(
            rubric=item.rubric,
            backend=backend,
            model=model,
            symmetric_debiasing=symmetric,
            mode="pairwise",
        )
        t0 = time.perf_counter()
        res_fwd = gate.rank([item.candidate_a, item.candidate_b])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        pred_winner = res_fwd.winner_index
        is_corr = pred_winner == item.winner
        if is_corr:
            correct += 1

        # Position-swap evaluation for consistency
        res_rev = gate.rank([item.candidate_b, item.candidate_a])
        # In reverse order, slot 0 is candidate_b, slot 1 is candidate_a.
        # Consistency holds if the reverse call chooses the exact same underlying candidate.
        is_cons = (
            pred_winner is not None
            and res_rev.winner_index is not None
            and res_rev.winner_index == (1 - pred_winner)
        )
        if is_cons:
            consistent += 1

        if progress:
            status = "OK" if is_corr else "ERR"
            cons_status = "CONSISTENT" if is_cons else "INCONSISTENT"
            print(
                f"  [{idx:03d}/{total:03d}] {item.id} -> Pred={pred_winner} Exp={item.winner} "
                f"[{status}] ({dt_ms:.1f}ms, {cons_status})",
                flush=True,
            )

    acc = round(correct / max(1, total), 4)
    ci = wilson_score_interval(correct, total)
    cons_rate = round(consistent / max(1, total), 4)
    lat_stats = compute_latencies(latencies)

    out: dict[str, Any] = {
        "task_type": "pairwise",
        "symmetric_debiasing": symmetric,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "accuracy_ci_95": ci,
        "position_swap_consistency_rate": cons_rate,
        "latencies": lat_stats,
    }

    if evaluate_cascade:
        print("  Evaluating Cascade via pairwise_to_binary_cases...", flush=True)
        cases = pairwise_to_binary_cases(items)
        report = run_benchmark(
            cases=cases, backend=backend, model=model, modes=["cascade"]
        )
        out["cascade_report"] = report.to_dict()

    return out


def evaluate_binary(
    items: Sequence[NormalizedBinaryItem],
    backend: Any,
    model: str | None = None,
    *,
    modes: tuple[str, ...] = ("logit", "cascade"),
    use_grounded_context: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluates binary benchmarks via LogitGate's standard run_benchmark runner or grounded evaluation."""
    if not use_grounded_context:
        cases = [
            BenchmarkCase(
                rubric=item.rubric, candidate=item.candidate, expected=item.expected
            )
            for item in items
        ]
        report = run_benchmark(cases=cases, backend=backend, model=model, modes=list(modes))
        return {
            "task_type": "binary",
            "total": len(items),
            "benchmark_report": report.to_dict(),
        }

    from logitgate import BaseGate, Cascade, JudgeGate, LogitGate
    from logitgate.benchmark import BenchmarkReport, ModeStats
    from logitgate.models import DecisionStatus

    modes_stats: dict[str, ModeStats] = {}
    for mode in modes:
        stats = ModeStats(mode=mode)
        for idx, item in enumerate(items, 1):
            has_grounded = bool(
                isinstance(item.metadata, dict)
                and "answer" in item.metadata
                and "context" in item.metadata
            )
            cand = item.metadata["answer"] if has_grounded else item.candidate
            ctx = item.metadata["context"] if has_grounded else None
            q = item.metadata.get("question") if has_grounded else None

            evaluator: Cascade | BaseGate
            if mode == "cascade":
                evaluator = Cascade(
                    rubric=item.rubric, backend=backend, model=model, save_failures=False
                )
            elif mode == "logit":
                evaluator = LogitGate(
                    rubric=item.rubric, backend=backend, model=model, save_on_fail=False
                )
            elif mode == "judge":
                evaluator = JudgeGate(
                    rubric=item.rubric, backend=backend, model=model, save_on_fail=False
                )
            else:
                continue

            t0 = time.perf_counter()
            v = evaluator.evaluate(candidate=cand, context=ctx, query=q)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            stats.latencies_ms.append(dt_ms)
            stats.generated_tokens.append(v.total_tokens_generated)
            stats.llm_calls.append(len(v.traces))
            is_corr = v.passed == item.expected
            stats.correct.append(is_corr)
            stats.ambiguous.append(v.status == DecisionStatus.AMBIGUOUS)
            stats.judge_skipped.append(
                any(t.layer_name.startswith("LogitGate") for t in v.traces)
                and not any(t.layer_name.startswith("JudgeGate") for t in v.traces)
            )
            stats.probabilities.append(v.probability)
            stats.labels.append(item.expected)
            if progress:
                status_str = "OK" if is_corr else "ERR"
                print(
                    f"  [{mode}][{idx:03d}/{len(items):03d}] {item.id} -> Pred={v.passed} Exp={item.expected} [{status_str}] ({dt_ms:.1f}ms)",
                    flush=True,
                )
        modes_stats[mode] = stats

    bench_report = BenchmarkReport(
        metadata={"grounded_context": True, "total_items": len(items)},
        modes=modes_stats,
    )
    return {
        "task_type": "binary",
        "grounded_context": True,
        "total": len(items),
        "benchmark_report": bench_report.to_dict(),
    }


def evaluate_stream_guard_on_ragtruth(
    items: Sequence[NormalizedBinaryItem],
    backend: Any,
    model: str | None = None,
    *,
    profile: Any = None,
    chunk_chars: int = 20,
    stride_chars: int = 20,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluates StreamGuard on RAGTruth responses for hallucination abort catch rate and tokens saved."""
    total_violations = 0
    violations_caught = 0
    total_compliant = 0
    false_alarms = 0
    tokens_saved_list: list[int] = []
    tokens_saved_pct_list: list[float] = []

    for idx, item in enumerate(items, 1):
        is_violation = not item.expected
        if is_violation:
            total_violations += 1
        else:
            total_compliant += 1

        ctx_text = str(item.metadata.get("context") or "").strip()
        q_text = str(item.metadata.get("question") or "").strip()
        prefix_rubric = (
            "Are all factual claims stated so far in this partial response prefix grounded in "
            "the Reference Context without fabricated or contradicted facts? "
            "(Incomplete trailing sentences at the end of a prefix are acceptable.)"
        )
        if item.metadata.get("answer") and (ctx_text or q_text):
            ctx_parts = [prefix_rubric]
            if ctx_text:
                ctx_parts.append(f"Reference Context:\n{ctx_text}")
            if q_text:
                ctx_parts.append(f"Question:\n{q_text}")
            effective_rubric = "\n\n".join(ctx_parts)
        else:
            effective_rubric = item.rubric

        from logitgate.models import DecisionStatus

        gate = LogitGate(
            rubric=effective_rubric,
            backend=backend,
            model=model,
            preset="balanced" if profile is None else None,
            profile=profile,
            save_on_fail=False,
        )
        guard = StreamGuard(
            gate=gate,
            min_chars=min(20, chunk_chars),
            stride_chars=stride_chars,
            raise_on_abort=False,
        )

        stream_text = str(item.metadata.get("answer") or item.candidate)
        total_chars = len(stream_text)
        chunks = [
            stream_text[i : i + chunk_chars]
            for i in range(0, total_chars, chunk_chars)
        ]
        if not chunks:
            chunks = [""]

        guarded = guard.stream(chunks)
        list(guarded)
        summary = guarded.summary
        final_v = guarded.final_verdict

        aborted_or_failed = summary is not None and (
            summary.aborted
            or (final_v is not None and final_v.status == DecisionStatus.FAILED)
        )
        chars_yielded = summary.chars_yielded if summary else total_chars
        saved_chars = max(0, total_chars - chars_yielded)
        saved_tokens = saved_chars // 4

        if is_violation:
            if aborted_or_failed:
                violations_caught += 1
                tokens_saved_list.append(saved_tokens)
                if total_chars > 0:
                    tokens_saved_pct_list.append(saved_chars / total_chars)
        elif aborted_or_failed:
            false_alarms += 1

        if progress:
            print(
                f"  [{idx:03d}/{len(items):03d}] {item.id} -> Violation={is_violation} "
                f"Aborted={aborted_or_failed} SavedTokens={saved_tokens}",
                flush=True,
            )

    catch_rate = round(violations_caught / max(1, total_violations), 4)
    false_alarm_rate = round(false_alarms / max(1, total_compliant), 4)
    avg_tokens_saved = (
        round(statistics.mean(tokens_saved_list), 1) if tokens_saved_list else 0.0
    )
    mean_saved_pct = (
        round(statistics.mean(tokens_saved_pct_list) * 100.0, 1)
        if tokens_saved_pct_list
        else 0.0
    )

    return {
        "total_items": len(items),
        "total_violations": total_violations,
        "violations_caught": violations_caught,
        "violation_catch_rate": catch_rate,
        "total_compliant": total_compliant,
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarm_rate,
        "avg_tokens_saved_on_caught": avg_tokens_saved,
        "mean_tokens_saved_pct": mean_saved_pct,
    }


def evaluate_best_of_n(
    items: Sequence[NormalizedPoolItem],
    backend: Any,
    model: str | None = None,
    *,
    mode: str = "auto",
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluates Best-of-N candidate pool selection accuracy with BestOfNGate / RankGate."""
    correct = 0
    latencies: list[float] = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        gate = BestOfNGate(
            rubric=item.rubric,
            backend=backend,
            model=model,
            mode=mode,  # type: ignore[arg-type]
        )
        t0 = time.perf_counter()
        res = gate.select(item.candidates)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        pred_winner = res.winner_index
        is_corr = pred_winner == item.expected_winner
        if is_corr:
            correct += 1

        if progress:
            status = "OK" if is_corr else "ERR"
            print(
                f"  [{idx:03d}/{total:03d}] {item.id} -> Winner={pred_winner} Exp={item.expected_winner} "
                f"[{status}] ({dt_ms:.1f}ms)",
                flush=True,
            )

    acc = round(correct / max(1, total), 4)
    ci = wilson_score_interval(correct, total)
    lat_stats = compute_latencies(latencies)

    return {
        "task_type": "best_of_n",
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "accuracy_ci_95": ci,
        "latencies": lat_stats,
    }


def run_benchmark_suite(
    suite_key: str,
    *,
    backend_spec: str = DEFAULT_BACKEND,
    model: str | None = None,
    limit: int = 50,
    seed: int = 42,
    symmetric: bool = True,
    eval_stream_guard: bool = False,
    use_grounded_context: bool = False,
    local_path: str | None = None,
    cache_dir: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Executes a single external benchmark suite end-to-end."""
    norm_key = suite_key.lower()
    if norm_key not in EXTERNAL_BENCHMARKS:
        raise ValueError(
            f"Unknown suite '{suite_key}'. Available: {sorted(EXTERNAL_BENCHMARKS)}"
        )

    spec = EXTERNAL_BENCHMARKS[norm_key]
    print(
        f"\n=== [External Benchmark: {spec.key}] {spec.hf_dataset_id} ({spec.task_type}) ===",
        flush=True,
    )
    print(f"  License: {spec.license_spdx} | Citation: {spec.citation_url}", flush=True)

    items = load_external_benchmark(
        spec,
        limit=limit,
        seed=seed,
        local_path=local_path,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    print(f"  Loaded {len(items)} items (limit={limit}, seed={seed})", flush=True)

    resolved_backend = resolve_logit_backend(backend_spec, model=model)
    try:
        if spec.task_type == "pairwise":
            pairwise_items = [
                it for it in items if isinstance(it, NormalizedPairwiseItem)
            ]
            results = evaluate_pairwise(
                pairwise_items, resolved_backend, model=model, symmetric=symmetric
            )
        elif spec.task_type == "best_of_n":
            pool_items = [it for it in items if isinstance(it, NormalizedPoolItem)]
            results = evaluate_best_of_n(pool_items, resolved_backend, model=model)
        elif spec.task_type in ("binary", "stream_span"):
            bin_items = [it for it in items if isinstance(it, NormalizedBinaryItem)]
            if eval_stream_guard and spec.key == "ragtruth":
                results = evaluate_stream_guard_on_ragtruth(
                    bin_items, resolved_backend, model=model
                )
            else:
                results = evaluate_binary(
                    bin_items,
                    resolved_backend,
                    model=model,
                    use_grounded_context=use_grounded_context,
                )
        else:
            raise ValueError(f"Unsupported task_type: {spec.task_type}")

        results["spec"] = {
            "key": spec.key,
            "hf_dataset_id": spec.hf_dataset_id,
            "config_name": spec.config_name,
            "split": spec.split,
            "license_spdx": spec.license_spdx,
            "task_type": spec.task_type,
            "citation_url": spec.citation_url,
        }
        return results
    finally:
        resolved_backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LogitGate Canonical External Public Benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=[*EXTERNAL_BENCHMARKS.keys(), "all"],
        default="llmbar",
        help="Benchmark suite to execute (default: llmbar)",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Backend URI (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--limit", type=int, default=50, help="Maximum number of items to load"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for slot balancing"
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable symmetric swap debiasing (pass --no-symmetric for single-pass ablation)",
    )
    parser.add_argument(
        "--eval-stream-guard",
        action="store_true",
        help="Evaluate StreamGuard violation catch rate and tokens saved (on ragtruth)",
    )
    parser.add_argument(
        "--grounded-context",
        action="store_true",
        help="Enable grounded context mode passing (context, query, answer) from item metadata",
    )
    parser.add_argument(
        "--local-path", default=None, help="Path to local JSON/JSONL fixture file"
    )
    parser.add_argument("--cache-dir", default=None, help="Cache directory override")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass disk cache and refetch from HuggingFace",
    )
    parser.add_argument(
        "--output", default=None, help="Optional output JSON path for results"
    )
    args = parser.parse_args()

    suites_to_run = (
        list(EXTERNAL_BENCHMARKS.keys())
        if args.benchmark == "all"
        else [args.benchmark]
    )
    all_results: dict[str, Any] = {}

    for suite in suites_to_run:
        res = run_benchmark_suite(
            suite,
            backend_spec=args.backend,
            model=args.model,
            limit=args.limit,
            seed=args.seed,
            symmetric=args.symmetric,
            eval_stream_guard=args.eval_stream_guard,
            use_grounded_context=args.grounded_context,
            local_path=args.local_path,
            cache_dir=args.cache_dir,
            force_refresh=args.force_refresh,
        )
        all_results[suite] = res

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nExternal benchmark results written to {out_p}", flush=True)


if __name__ == "__main__":
    main()
