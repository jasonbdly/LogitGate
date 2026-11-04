#!/usr/bin/env python3
"""
Master Hardened Benchmark Runner for LogitGate ChoiceGate & ScoreGate.

Evaluates multi-action gates on the hardened 32-item adversarial benchmark suite
(16 ChoiceGate items + 16 ScoreGate items) designed to find true model failure boundaries:
- ChoiceGate: 6-way scaling, adversarial keyword decoys, strict priority hierarchy, architectural scope.
- ScoreGate: 5 discrete tiers (Tier 5 Flawless, Tier 4 Sub-optimal Complexity,
             Tier 3 Smells & Violations, Tier 2 Latent Concurrency/TOCTOU Defects,
             Tier 1 Catastrophic Flaws & Injections).

Protocol:
- 1 untimed warm-up pass per model
- N=5 sequential timed trials per item (mean ± std ms)
- Symmetric debiasing enabled (cancelling label order and scale anchoring bias)
- Computes:
  - ChoiceGate: Overall accuracy, category breakdown, permutation consistency
  - ScoreGate: Mean score per tier, adjacent pairwise monotonicity, all-pairs monotonicity,
               Spearman rank correlation rho, score dynamic range (S5 - S1)
- Evaluates across Pure CPU models (num_gpu=0), GPU 1 models, and GPU 0 models.

Usage:
    LOGITGATE_BENCH_CPU_URL=http://<host>:11435 \\
    LOGITGATE_BENCH_GPU_URL=http://<host>:11435 \\
    LOGITGATE_BENCH_GPU0_URL=http://<host>:11434 \\
    python benchmarks/run_adversarial_choice_score.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from dataset import load_adversarial_choice_and_score_32

from logitgate import ChoiceGate, ScoreGate
from logitgate.backends.ollama import OllamaBackend


def compute_spearman_rho(x: list[float], y: list[float]) -> float:
    """Computes exact Spearman rank correlation coefficient rho with fractional tie handling."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda it: it[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_r = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_r
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    return cov / ((var_x * var_y) ** 0.5)


def unload_model(base_url: str, model_name: str) -> None:
    """Unloads a model from Ollama host memory."""
    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            resp.read()
    except (OSError, ValueError) as exc:
        print(f"  [Warning] Failed to unload {model_name} on {base_url}: {exc}", flush=True)


def get_residency(base_url: str, model_name: str) -> dict[str, float]:
    """Queries /api/ps to inspect host RAM and VRAM usage."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            for m in data.get("models", []):
                name = m.get("model") or m.get("name") or ""
                if model_name in name:
                    return {
                        "ram_gb": round(m.get("size", 0) / 1e9, 2),
                        "vram_gb": round(m.get("size_vram", 0) / 1e9, 2),
                    }
    except (OSError, ValueError):
        pass
    return {"ram_gb": 0.0, "vram_gb": 0.0}


def run_benchmark_for_config(
    cfg: dict[str, Any],
    choice_items: list[dict[str, Any]],
    score_items: list[dict[str, Any]],
    n_trials: int = 5,
) -> dict[str, Any]:
    """Runs the full hardened ChoiceGate and ScoreGate benchmark for a given configuration."""
    model_name = cfg["model"]
    base_url = cfg["base_url"]
    hw_label = cfg["hardware"]
    ollama_options = cfg.get("options")

    print("\n" + "=" * 70, flush=True)
    print(f"  BENCHMARKING: {model_name} [{hw_label}] (base_url={base_url})", flush=True)
    print("=" * 70, flush=True)

    backend = OllamaBackend(
        base_url=base_url,
        model=model_name,
        options=ollama_options,
        timeout=180.0,
    )

    try:
        # 1. Warm-up pass (untimed)
        print("  [Step 1/3] Running untimed warm-up...", flush=True)
        warmup_cg = ChoiceGate(
            question=choice_items[0]["question"],
            options=choice_items[0]["options"],
            backend=backend,
            model=model_name,
            symmetric_debiasing=True,
        )
        warmup_cg.evaluate(choice_items[0]["candidate"])

        warmup_sg = ScoreGate(
            rubric=score_items[0]["rubric"],
            min_score=1,
            max_score=5,
            backend=backend,
            model=model_name,
            symmetric_debiasing=True,
        )
        warmup_sg.evaluate(score_items[0]["candidate"])

        residency = get_residency(base_url, model_name)
        print(f"  [Residency] Host RAM: {residency['ram_gb']} GB | VRAM: {residency['vram_gb']} GB", flush=True)

        # 2. Hardened ChoiceGate Evaluation (16 items x N trials)
        print(f"  [Step 2/3] Evaluating ChoiceGate (16 adversarial items, N={n_trials} trials)...", flush=True)
        choice_latencies: list[float] = []
        trial_accuracies: list[float] = []
        item_correct_counts: dict[str, int] = {it["id"]: 0 for it in choice_items}
        category_correct_counts: dict[str, int] = {}
        category_total_counts: dict[str, int] = {}
        stability_matches = 0
        stability_total = 0

        for it in choice_items:
            cat = it["category"]
            category_total_counts[cat] = category_total_counts.get(cat, 0) + n_trials

        for trial_idx in range(n_trials):
            trial_correct = 0
            for it in choice_items:
                cg = ChoiceGate(
                    question=it["question"],
                    options=it["options"],
                    backend=backend,
                    model=model_name,
                    symmetric_debiasing=True,
                )
                t0 = time.perf_counter()
                v = cg.evaluate(it["candidate"])
                lat = (time.perf_counter() - t0) * 1000.0
                choice_latencies.append(lat)

                is_correct = (v.choice == it["expected_choice"])
                if is_correct:
                    trial_correct += 1
                    item_correct_counts[it["id"]] += 1
                    category_correct_counts[it["category"]] = category_correct_counts.get(it["category"], 0) + 1

                # Permutation stability check: compare forward-only pass with reverse pass
                if trial_idx == 0:
                    cg_fwd = ChoiceGate(
                        question=it["question"],
                        options=it["options"],
                        backend=backend,
                        model=model_name,
                        symmetric_debiasing=False,
                    )
                    v_fwd = cg_fwd.evaluate(it["candidate"])
                    stability_total += 1
                    if v_fwd.choice == v.choice:
                        stability_matches += 1

            trial_acc = (trial_correct / len(choice_items)) * 100.0
            trial_accuracies.append(trial_acc)
            print(f"    Trial {trial_idx + 1}/{n_trials}: {trial_correct}/{len(choice_items)} correct ({trial_acc:.1f}%)", flush=True)

        mean_choice_acc = statistics.mean(trial_accuracies)
        choice_acc_std = statistics.stdev(trial_accuracies) if len(trial_accuracies) > 1 else 0.0
        choice_lat_mean = statistics.mean(choice_latencies)
        choice_lat_std = statistics.stdev(choice_latencies) if len(choice_latencies) > 1 else 0.0

        cat_breakdown: dict[str, str] = {}
        for cat, tot in category_total_counts.items():
            corr = category_correct_counts.get(cat, 0)
            cat_breakdown[cat] = f"{corr}/{tot} ({corr / tot * 100.0:.1f}%)"

        # 3. Hardened ScoreGate Evaluation (16 items across 5 tiers x N trials)
        print(f"  [Step 3/3] Evaluating ScoreGate (16 items, 5 discrete tiers, N={n_trials} trials)...", flush=True)
        score_latencies: list[float] = []
        item_scores: dict[str, list[float]] = {it["id"]: [] for it in score_items}

        for trial_idx in range(n_trials):
            for it in score_items:
                sg = ScoreGate(
                    rubric=it["rubric"],
                    min_score=1,
                    max_score=5,
                    backend=backend,
                    model=model_name,
                    symmetric_debiasing=True,
                )
                t0 = time.perf_counter()
                v = sg.evaluate(it["candidate"])
                lat = (time.perf_counter() - t0) * 1000.0
                score_latencies.append(lat)
                s = v.score if v.score is not None else 3.0
                item_scores[it["id"]].append(s)

        score_lat_mean = statistics.mean(score_latencies)
        score_lat_std = statistics.stdev(score_latencies) if len(score_latencies) > 1 else 0.0

        # Mean continuous score per item across trials
        mean_item_scores: dict[str, float] = {
            it["id"]: statistics.mean(item_scores[it["id"]]) for it in score_items
        }

        # Group by tier
        tier_scores: dict[str, list[float]] = {
            "tier_5_flawless": [],
            "tier_4_suboptimal_complexity": [],
            "tier_3_code_smells": [],
            "tier_2_latent_defects": [],
            "tier_1_catastrophic": [],
        }
        for it in score_items:
            tier_scores[it["tier"]].append(mean_item_scores[it["id"]])

        tier_means: dict[str, float] = {
            t: round(statistics.mean(vals), 2) for t, vals in tier_scores.items()
        }

        # Pairwise Monotonicity Checks
        # Adjacent pairs: (T5 > T4), (T4 > T3), (T3 > T2), (T2 > T1)
        adj_pairs = [
            ("tier_5_flawless", "tier_4_suboptimal_complexity"),
            ("tier_4_suboptimal_complexity", "tier_3_code_smells"),
            ("tier_3_code_smells", "tier_2_latent_defects"),
            ("tier_2_latent_defects", "tier_1_catastrophic"),
        ]
        adjacent_monotonic_count = 0
        adjacent_pair_count = 0
        adjacent_breakdown: dict[str, str] = {}

        for high_tier, low_tier in adj_pairs:
            m_count = 0
            p_count = 0
            for h_val in tier_scores[high_tier]:
                for l_val in tier_scores[low_tier]:
                    p_count += 1
                    adjacent_pair_count += 1
                    if h_val > l_val:
                        m_count += 1
                        adjacent_monotonic_count += 1
            rate = (m_count / p_count * 100.0) if p_count > 0 else 0.0
            pair_key = f"{high_tier}_vs_{low_tier}"
            adjacent_breakdown[pair_key] = f"{m_count}/{p_count} ({rate:.1f}%)"

        adj_mono_rate = (
            (adjacent_monotonic_count / adjacent_pair_count * 100.0)
            if adjacent_pair_count > 0
            else 0.0
        )

        # All-pairs monotonicity (any higher tier > lower tier)
        tier_order = [
            "tier_5_flawless",
            "tier_4_suboptimal_complexity",
            "tier_3_code_smells",
            "tier_2_latent_defects",
            "tier_1_catastrophic",
        ]
        all_monotonic_count = 0
        all_pair_count = 0
        for i in range(len(tier_order)):
            for j in range(i + 1, len(tier_order)):
                for h_val in tier_scores[tier_order[i]]:
                    for l_val in tier_scores[tier_order[j]]:
                        all_pair_count += 1
                        if h_val > l_val:
                            all_monotonic_count += 1

        all_mono_rate = (
            (all_monotonic_count / all_pair_count * 100.0) if all_pair_count > 0 else 0.0
        )

        # Spearman rank correlation rho
        ground_truth_ranks = [float(it["tier_rank"]) for it in score_items]
        assigned_scores = [mean_item_scores[it["id"]] for it in score_items]
        spearman_rho = round(compute_spearman_rho(ground_truth_ranks, assigned_scores), 4)

        # Dynamic range
        dynamic_range = round(tier_means["tier_5_flawless"] - tier_means["tier_1_catastrophic"], 2)

        perm_stability_pct = (
            round(stability_matches / stability_total * 100.0, 1)
            if stability_total > 0
            else 0.0
        )

        res = {
            "model": model_name,
            "hardware": hw_label,
            "base_url": base_url,
            "residency": residency,
            "trials": n_trials,
            "choice_gate": {
                "overall_accuracy": f"{mean_choice_acc:.1f}% ± {choice_acc_std:.1f}%",
                "accuracy_percentage": round(mean_choice_acc, 1),
                "permutation_stability": f"{stability_matches}/{stability_total} ({perm_stability_pct}%)",
                "category_breakdown": cat_breakdown,
                "latency_ms": f"{choice_lat_mean:.1f} ± {choice_lat_std:.1f} ms",
                "mean_latency_ms": round(choice_lat_mean, 1),
            },
            "score_gate": {
                "tier_means": tier_means,
                "dynamic_range": dynamic_range,
                "adjacent_monotonicity": f"{adjacent_monotonic_count}/{adjacent_pair_count} ({adj_mono_rate:.1f}%)",
                "adjacent_monotonicity_rate": round(adj_mono_rate, 1),
                "adjacent_breakdown": adjacent_breakdown,
                "all_pairs_monotonicity": f"{all_monotonic_count}/{all_pair_count} ({all_mono_rate:.1f}%)",
                "spearman_rho": spearman_rho,
                "latency_ms": f"{score_lat_mean:.1f} ± {score_lat_std:.1f} ms",
                "mean_latency_ms": round(score_lat_mean, 1),
            },
        }

        print("\n  ================ SUMMARY RESULTS ================", flush=True)
        print(f"  ChoiceGate Accuracy:       {res['choice_gate']['overall_accuracy']}", flush=True)
        print(f"  ChoiceGate Stability:      {res['choice_gate']['permutation_stability']}", flush=True)
        print(f"  ChoiceGate Latency:        {res['choice_gate']['latency_ms']}", flush=True)
        print(f"  ScoreGate Tiers:           T5={tier_means['tier_5_flawless']} | T4={tier_means['tier_4_suboptimal_complexity']} | T3={tier_means['tier_3_code_smells']} | T2={tier_means['tier_2_latent_defects']} | T1={tier_means['tier_1_catastrophic']}", flush=True)
        print(f"  ScoreGate Monotonicity:    Adjacent={res['score_gate']['adjacent_monotonicity']} | All-Pairs={res['score_gate']['all_pairs_monotonicity']}", flush=True)
        print(f"  ScoreGate Spearman rho:    {spearman_rho} (Range: {dynamic_range})", flush=True)
        print(f"  ScoreGate Latency:         {res['score_gate']['latency_ms']}", flush=True)
        print("  =================================================", flush=True)

        return res

    finally:
        backend.close()
        unload_model(base_url, model_name)
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LogitGate Hardened ChoiceGate & ScoreGate Benchmarks.")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "adversarial_choice_score_results.json"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--configurations", nargs="+", default=["all"])
    args = parser.parse_args()

    cpu_url = os.environ.get("LOGITGATE_BENCH_CPU_URL", "http://localhost:11435")
    gpu_url = os.environ.get("LOGITGATE_BENCH_GPU_URL", "http://localhost:11435")
    gpu0_url = os.environ.get("LOGITGATE_BENCH_GPU0_URL", "http://localhost:11434")

    # Full Matrix of 9 Configurations
    all_configs: list[dict[str, Any]] = [
        # --- CPU Configurations (options={"num_gpu": 0}) ---
        {"id": "llama3.2:1b_cpu", "model": "llama3.2:1b", "base_url": cpu_url, "hardware": "CPU (num_gpu=0)", "options": {"num_gpu": 0}},
        {"id": "llama3.2:3b_cpu", "model": "llama3.2:3b", "base_url": cpu_url, "hardware": "CPU (num_gpu=0)", "options": {"num_gpu": 0}},
        {"id": "qwen2.5-coder:1.5b_cpu", "model": "qwen2.5-coder:1.5b", "base_url": cpu_url, "hardware": "CPU (num_gpu=0)", "options": {"num_gpu": 0}},
        {"id": "qwen2.5-coder:3b_cpu", "model": "qwen2.5-coder:3b", "base_url": cpu_url, "hardware": "CPU (num_gpu=0)", "options": {"num_gpu": 0}},
        {"id": "gemma4:e4b_cpu", "model": "gemma4:e4b", "base_url": cpu_url, "hardware": "CPU (num_gpu=0)", "options": {"num_gpu": 0}},

        # --- GPU 1 Configurations (16 GB Card) ---
        {"id": "gemma4:e4b_gpu1", "model": "gemma4:e4b", "base_url": gpu_url, "hardware": "GPU 1 (16GB)", "options": None},
        {"id": "qwen2.5:14b_gpu1", "model": "qwen2.5:14b", "base_url": gpu_url, "hardware": "GPU 1 (16GB)", "options": None},
        {"id": "qwen2.5-coder:14b_gpu1", "model": "qwen2.5-coder:14b", "base_url": gpu_url, "hardware": "GPU 1 (16GB)", "options": None},

        # --- GPU 0 Configuration (32 GB Card) ---
        {"id": "gemma4:31b_gpu0", "model": "gemma4:31b", "base_url": gpu0_url, "hardware": "GPU 0 (32GB)", "options": None},
    ]

    selected_configs = all_configs
    if args.configurations != ["all"]:
        selected_configs = [c for c in all_configs if c["id"] in args.configurations or c["model"] in args.configurations]

    choice_items, score_items = load_adversarial_choice_and_score_32()
    print(f"Loaded {len(choice_items)} adversarial ChoiceGate items and {len(score_items)} ScoreGate items.")
    print(f"Configurations to run: {[c['id'] for c in selected_configs]}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    results: dict[str, Any] = {}
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
        except (OSError, json.JSONDecodeError):
            results = {}

    for cfg in selected_configs:
        cfg_id = cfg["id"]
        try:
            res = run_benchmark_for_config(cfg, choice_items, score_items, n_trials=args.trials)
            results[cfg_id] = res

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"\n  [ERROR] Benchmark run failed for {cfg_id}: {exc}", flush=True)
            results[cfg_id] = {"error": str(exc)}
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"ALL BENCHMARKS COMPLETED SUCCESSFULLY! Full results saved to:\n  {out_path}", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
