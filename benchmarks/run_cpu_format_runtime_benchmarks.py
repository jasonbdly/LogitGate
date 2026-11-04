#!/usr/bin/env python3
"""
Benchmark Runner for CPU Model Formats, Quantizations, and Runtime Architectures.

Compares:
1. Runtime Overhead: In-Process llama.cpp (LlamaCppBackend) vs. Ollama HTTP Daemon (OllamaBackend num_gpu=0)
2. Quantization Representations: Q8_0 (1B, 1.32 GB) vs. Q4_K_M (1.5B, 0.94 GB) vs. Q4_K_M (3B, 1.93 GB)
3. Thread Scaling: In-process execution across threads = 1, 2, 4, 8, 16
4. Task Performance: ChoiceGate (16 adversarial items) & ScoreGate (16 items, 5 discrete tiers)

Usage:
    python benchmarks/run_cpu_format_runtime_benchmarks.py --output benchmarks/cpu_format_runtime_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import llama_cpp
from dataset import TIER_B_ATOMIC_RUBRICS, load_adversarial_choice_and_score_32

from logitgate import ChoiceGate, LogitGate, ScoreGate
from logitgate.backends.llamacpp import LlamaCppBackend
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


def get_current_rss_mb() -> float:
    """Returns the peak RSS memory in MB for the current Python process."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in kilobytes; on macOS it's in bytes
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def benchmark_latencies(
    gate_single: LogitGate,
    gate_dp: LogitGate,
    test_cand: str,
    n_trials: int = 5,
) -> dict[str, Any]:
    """Measures single-pass, 2-pass symmetric, and 4-rubric batch latencies (N=5)."""
    # 1. Single-pass latency (triage)
    sp_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        gate_single.evaluate(test_cand)
        sp_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. 2-pass symmetric latency (balanced)
    dp_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        gate_dp.evaluate(test_cand)
        dp_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. 4-rubric batch latency
    b_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        gate_dp.evaluate_batch(test_cand, TIER_B_ATOMIC_RUBRICS)
        b_times.append((time.perf_counter() - t0) * 1000.0)

    sp_mean = statistics.mean(sp_times)
    sp_std = statistics.stdev(sp_times) if len(sp_times) > 1 else 0.0
    dp_mean = statistics.mean(dp_times)
    dp_std = statistics.stdev(dp_times) if len(dp_times) > 1 else 0.0
    b_mean = statistics.mean(b_times)
    b_std = statistics.stdev(b_times) if len(b_times) > 1 else 0.0

    return {
        "single_pass_ms": f"{sp_mean:.1f} ± {sp_std:.1f} ms",
        "single_pass_mean_ms": round(sp_mean, 1),
        "two_pass_ms": f"{dp_mean:.1f} ± {dp_std:.1f} ms",
        "two_pass_mean_ms": round(dp_mean, 1),
        "four_rubric_batch_ms": f"{b_mean:.1f} ± {b_std:.1f} ms",
        "four_rubric_batch_mean_ms": round(b_mean, 1),
        "per_rubric_batch_ms": f"{b_mean / 4.0:.1f} ± {b_std / 4.0:.1f} ms",
    }


def evaluate_adversarial_suite(
    backend: Any,
    model_name: str | None,
    choice_items: list[dict[str, Any]],
    score_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluates the 16 ChoiceGate and 16 ScoreGate items."""
    # ChoiceGate
    cg_correct = 0
    cg_times: list[float] = []
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
        cg_times.append((time.perf_counter() - t0) * 1000.0)
        if v.choice == it["expected_choice"]:
            cg_correct += 1

    cg_acc_pct = (cg_correct / len(choice_items)) * 100.0
    cg_lat_mean = statistics.mean(cg_times)

    # ScoreGate
    tier_scores: dict[str, list[float]] = {
        "tier_5_flawless": [],
        "tier_4_suboptimal_complexity": [],
        "tier_3_code_smells": [],
        "tier_2_latent_defects": [],
        "tier_1_catastrophic": [],
    }
    sg_times: list[float] = []
    scores_by_id: dict[str, float] = {}

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
        sg_times.append((time.perf_counter() - t0) * 1000.0)
        score_val = v.score if v.score is not None else 3.0
        scores_by_id[it["id"]] = score_val
        tier_scores[it["tier"]].append(score_val)

    tier_means = {t: round(statistics.mean(vals), 2) for t, vals in tier_scores.items()}
    sg_lat_mean = statistics.mean(sg_times)

    # Monotonicity
    adj_pairs = [
        ("tier_5_flawless", "tier_4_suboptimal_complexity"),
        ("tier_4_suboptimal_complexity", "tier_3_code_smells"),
        ("tier_3_code_smells", "tier_2_latent_defects"),
        ("tier_2_latent_defects", "tier_1_catastrophic"),
    ]
    adj_m = 0
    adj_tot = 0
    for h, l in adj_pairs:
        for hv in tier_scores[h]:
            for lv in tier_scores[l]:
                adj_tot += 1
                if hv > lv:
                    adj_m += 1

    ground_truth_ranks = [float(it["tier_rank"]) for it in score_items]
    assigned_scores = [scores_by_id[it["id"]] for it in score_items]
    spearman_rho = round(compute_spearman_rho(ground_truth_ranks, assigned_scores), 4)
    dynamic_range = round(tier_means["tier_5_flawless"] - tier_means["tier_1_catastrophic"], 2)

    return {
        "choice_gate_accuracy": f"{cg_correct}/{len(choice_items)} ({cg_acc_pct:.1f}%)",
        "choice_gate_acc_pct": round(cg_acc_pct, 1),
        "choice_gate_latency_ms": round(cg_lat_mean, 1),
        "score_gate_tier_means": tier_means,
        "score_gate_adjacent_monotonicity": f"{adj_m}/{adj_tot} ({adj_m / adj_tot * 100.0:.1f}%)",
        "score_gate_spearman_rho": spearman_rho,
        "score_gate_dynamic_range": dynamic_range,
        "score_gate_latency_ms": round(sg_lat_mean, 1),
    }


def unload_ollama(base_url: str, model_name: str) -> None:
    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            resp.read()
    except (OSError, ValueError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU Format & Runtime Benchmarks.")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "cpu_format_runtime_results.json"))
    args = parser.parse_args()

    choice_items, score_items = load_adversarial_choice_and_score_32()
    test_cand = (
        "def compute_payroll(employees: list[dict], tax_rate: float) -> float:\n"
        "    total = 0.0\n"
        "    for emp in employees:\n"
        "        salary = emp.get('salary', 0.0)\n"
        "        total += salary * (1.0 - tax_rate)\n"
        "    return round(total, 2)\n"
    )

    models_config = [
        {
            "name": "llama3.2:1b",
            "gguf_path": "/tmp/models/llama3.2-1b-q8_0.gguf",
            "quant": "Q8_0",
            "params": "1.2B",
            "filesize_mb": 1260,
        },
        {
            "name": "qwen2.5-coder:1.5b",
            "gguf_path": "/tmp/models/qwen2.5-coder-1.5b-q4_k_m.gguf",
            "quant": "Q4_K_M",
            "params": "1.5B",
            "filesize_mb": 940,
        },
        {
            "name": "llama3.2:3b",
            "gguf_path": "/tmp/models/llama3.2-3b-q4_k_m.gguf",
            "quant": "Q4_K_M",
            "params": "3.2B",
            "filesize_mb": 1930,
        },
    ]

    ollama_url = os.environ.get("LOGITGATE_BENCH_CPU_URL", "http://localhost:11434")
    results: dict[str, Any] = {
        "hardware": "AMD Ryzen 9 9955HX3D (16-Core / 32-Thread, 128MB L3 3D V-Cache)",
        "models": {},
        "thread_scaling": {},
    }

    print("==========================================================================", flush=True)
    print("  CPU RUNTIME & FORMAT BENCHMARK: In-Process llama.cpp vs. Ollama Daemon", flush=True)
    print("==========================================================================", flush=True)

    for m in models_config:
        m_name: str = str(m["name"])
        gguf_p: str = str(m["gguf_path"])
        quant: str = str(m["quant"])
        print(f"\n>>> EVALUATING: {m_name} (Format={quant}, Size={m['filesize_mb']} MB) <<<", flush=True)

        m_res: dict[str, Any] = {"metadata": m}

        # -------------------------------------------------------------
        # 1. Native In-Process llama.cpp Execution (LlamaCppBackend)
        # -------------------------------------------------------------
        print("  [1/2] Benchmarking Native In-Process llama.cpp (threads=8)...", flush=True)
        gc.collect()
        rss_before = get_current_rss_mb()
        t0 = time.perf_counter()
        llm = llama_cpp.Llama(
            model_path=gguf_p,
            n_ctx=2048,
            n_threads=8,
            logits_all=True,
            verbose=False,
        )
        load_time_s = round(time.perf_counter() - t0, 2)
        rss_after = get_current_rss_mb()
        ram_used_mb = round(rss_after - rss_before, 1)
        print(f"    Loaded in {load_time_s}s (RAM footprint: {ram_used_mb} MB)", flush=True)

        llamacpp_backend = LlamaCppBackend(llm=llm)
        gate_sp = LogitGate(rubric="Is this code well-formed?", backend=llamacpp_backend, preset="fast_triage")
        gate_dp = LogitGate(rubric="Is this code well-formed?", backend=llamacpp_backend, preset="balanced")

        # Warm-up
        gate_sp.evaluate(test_cand)

        lat_llamacpp = benchmark_latencies(gate_sp, gate_dp, test_cand, n_trials=5)
        print(f"    Latency: Single={lat_llamacpp['single_pass_ms']} | 2-Pass={lat_llamacpp['two_pass_ms']} | Batch(4)={lat_llamacpp['four_rubric_batch_ms']}", flush=True)

        adv_llamacpp = evaluate_adversarial_suite(llamacpp_backend, None, choice_items, score_items)
        print(f"    ChoiceGate Acc: {adv_llamacpp['choice_gate_accuracy']} ({adv_llamacpp['choice_gate_latency_ms']} ms)", flush=True)
        print(f"    ScoreGate Rho:  {adv_llamacpp['score_gate_spearman_rho']} (Range: {adv_llamacpp['score_gate_dynamic_range']}, {adv_llamacpp['score_gate_latency_ms']} ms)", flush=True)

        m_res["in_process_llamacpp"] = {
            "load_time_s": load_time_s,
            "ram_used_mb": ram_used_mb,
            "latencies": lat_llamacpp,
            "adversarial_suite": adv_llamacpp,
        }

        del llm
        del llamacpp_backend
        gc.collect()

        # -------------------------------------------------------------
        # 2. Ollama HTTP Daemon Execution (num_gpu=0)
        # -------------------------------------------------------------
        print(f"  [2/2] Benchmarking Ollama HTTP Daemon ({m_name}, num_gpu=0)...", flush=True)
        ollama_backend = OllamaBackend(
            base_url=ollama_url,
            model=m_name,
            options={"num_gpu": 0},
            timeout=180.0,
        )
        try:
            gate_sp_ol = LogitGate(rubric="Is this code well-formed?", backend=ollama_backend, model=m_name, preset="fast_triage")
            gate_dp_ol = LogitGate(rubric="Is this code well-formed?", backend=ollama_backend, model=m_name, preset="balanced")

            # Warm-up
            gate_sp_ol.evaluate(test_cand)

            lat_ollama = benchmark_latencies(gate_sp_ol, gate_dp_ol, test_cand, n_trials=5)
            print(f"    Latency: Single={lat_ollama['single_pass_ms']} | 2-Pass={lat_ollama['two_pass_ms']} | Batch(4)={lat_ollama['four_rubric_batch_ms']}", flush=True)

            adv_ollama = evaluate_adversarial_suite(ollama_backend, m_name, choice_items, score_items)
            print(f"    ChoiceGate Acc: {adv_ollama['choice_gate_accuracy']} ({adv_ollama['choice_gate_latency_ms']} ms)", flush=True)
            print(f"    ScoreGate Rho:  {adv_ollama['score_gate_spearman_rho']} (Range: {adv_ollama['score_gate_dynamic_range']}, {adv_ollama['score_gate_latency_ms']} ms)", flush=True)

            # Speedup calculation
            sp_speedup = round(lat_ollama["single_pass_mean_ms"] / max(lat_llamacpp["single_pass_mean_ms"], 1.0), 2)
            dp_speedup = round(lat_ollama["two_pass_mean_ms"] / max(lat_llamacpp["two_pass_mean_ms"], 1.0), 2)
            batch_speedup = round(lat_ollama["four_rubric_batch_mean_ms"] / max(lat_llamacpp["four_rubric_batch_mean_ms"], 1.0), 2)
            print(f"    >>> IN-PROCESS SPEEDUP: Single={sp_speedup}x | 2-Pass={dp_speedup}x | Batch={batch_speedup}x <<<", flush=True)

            m_res["ollama_http_daemon"] = {
                "latencies": lat_ollama,
                "adversarial_suite": adv_ollama,
                "in_process_speedup": {
                    "single_pass": sp_speedup,
                    "two_pass": dp_speedup,
                    "four_rubric_batch": batch_speedup,
                },
            }
        finally:
            ollama_backend.close()
            unload_ollama(ollama_url, m_name)

        results["models"][m_name] = m_res

    # -------------------------------------------------------------
    # 3. CPU Thread Scaling Sweep (llama3.2:3b in-process llama.cpp)
    # -------------------------------------------------------------
    print("\n==========================================================================", flush=True)
    print("  CPU THREAD SCALING SWEEP (llama3.2:3b In-Process llama.cpp)", flush=True)
    print("==========================================================================", flush=True)
    thread_counts = [1, 2, 4, 8, 16]
    model_3b_path = "/tmp/models/llama3.2-3b-q4_k_m.gguf"

    for t_count in thread_counts:
        print(f"  Testing n_threads = {t_count}...", flush=True)
        llm_t = llama_cpp.Llama(
            model_path=model_3b_path,
            n_ctx=2048,
            n_threads=t_count,
            logits_all=True,
            verbose=False,
        )
        b_t = LlamaCppBackend(llm=llm_t)
        g_sp = LogitGate(rubric="Is this code well-formed?", backend=b_t, preset="fast_triage")
        g_dp = LogitGate(rubric="Is this code well-formed?", backend=b_t, preset="balanced")
        g_sp.evaluate(test_cand)

        t_res = benchmark_latencies(g_sp, g_dp, test_cand, n_trials=5)
        print(f"    Threads={t_count:2d} -> Single={t_res['single_pass_ms']} | 2-Pass={t_res['two_pass_ms']} | Batch={t_res['four_rubric_batch_ms']}", flush=True)
        results["thread_scaling"][f"threads_{t_count}"] = t_res

        del llm_t
        del b_t
        gc.collect()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nALL BENCHMARKS COMPLETE! Results written to:\n  {out_file}", flush=True)


if __name__ == "__main__":
    main()
