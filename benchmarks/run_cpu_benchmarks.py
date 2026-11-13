#!/usr/bin/env python3
"""
Automated CPU-Bound Benchmark Runner for LogitGate.

Evaluates alternative compact open-weights models in pure CPU-only mode (options={"num_gpu": 0}):
- llama3.2:1b (1.32 GB)
- llama3.2:3b (2.02 GB)
- qwen2.5-coder:1.5b (0.99 GB)
- qwen2.5-coder:3b (1.93 GB)

Protocol:
- 1 untimed warmup pass per model to initialize CPU memory & KV-cache buffers
- N=5 sequential timed latency trials (mean ± std ms)
- Pure CPU execution (num_gpu=0, host RAM residency recorded via /api/ps)
- Suite 1: 24-Item Multi-Domain Cascade
- Suite 2: 32-Item Reasoning Depth Tiers (Tier A, Tier B compound, Tier B atomic, Tier C, Tier D)
- Suite 3: Multi-Action Gates (ChoiceGate 6 items, ScoreGate 6 items)
- Unload model between runs

Usage:
    LOGITGATE_BENCH_CPU_URL=http://<host>:11435 python benchmarks/run_cpu_benchmarks.py --output benchmarks/cpu_results.json
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

from dataset import (
    CHOICE_GATE_6_ITEMS,
    MULTI_DOMAIN_24_ITEMS,
    SCORE_GATE_6_ITEMS,
    TIER_A_8_ITEMS,
    TIER_B_8_ITEMS,
    TIER_B_ATOMIC_RUBRICS,
    TIER_B_COMPOUND_RUBRIC,
    TIER_C_8_ITEMS,
    TIER_D_8_ITEMS,
)

from logitgate import ASTGate, ChoiceGate, JudgeGate, LogitGate, ScoreGate
from logitgate.backends.ollama import OllamaBackend


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
        print(f"  [Warning] Failed to unload {model_name}: {exc}", flush=True)


def get_ram_residency(base_url: str, model_name: str) -> dict[str, Any]:
    """Queries /api/ps to inspect System RAM and VRAM usage."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            for m in data.get("models", []):
                name = m.get("model") or m.get("name") or ""
                if model_name in name:
                    return {
                        "size_gb": round(m.get("size", 0) / 1e9, 2),
                        "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 2),
                    }
    except (OSError, ValueError):
        pass
    return {"size_gb": 0.0, "size_vram_gb": 0.0}


def measure_cpu_latencies(
    backend: OllamaBackend,
    model_name: str,
    base_url: str,
    n_trials: int = 5,
) -> dict[str, Any]:
    """Measures N=5 sequential latency trials on CPU after 1 untimed warmup pass."""
    test_rubric = "Is this document well-written and free of policy violations?"
    test_cand = (
        "This is a standard technical document describing the deployment and architecture "
        "of a distributed microservice. It specifies health checks, logging formats, and error handling."
    )

    # 1. Warm-up pass (untimed)
    warmup_gate = LogitGate(rubric=test_rubric, backend=backend, model=model_name, preset="fast_triage")
    warmup_gate.evaluate(test_cand)

    # Query /api/ps to verify CPU memory residency
    residency = get_ram_residency(base_url, model_name)
    print(f"  [Residency] RAM: {residency['size_gb']} GB, VRAM: {residency['size_vram_gb']} GB", flush=True)

    # 2. Single-pass latency (preset="fast_triage", N=5)
    sp_gate = LogitGate(rubric=test_rubric, backend=backend, model=model_name, preset="fast_triage")
    sp_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sp_gate.evaluate(test_cand)
        sp_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. 2-pass symmetric latency (preset="balanced", N=5)
    dp_gate = LogitGate(rubric=test_rubric, backend=backend, model=model_name, preset="balanced")
    dp_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        dp_gate.evaluate(test_cand)
        dp_times.append((time.perf_counter() - t0) * 1000.0)

    # 4. 4-rubric batch latency (evaluate_batch, N=5)
    batch_gate = LogitGate(rubric=TIER_B_ATOMIC_RUBRICS[0], backend=backend, model=model_name, preset="balanced")
    batch_times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        batch_gate.evaluate_batch(test_cand, TIER_B_ATOMIC_RUBRICS)
        batch_times.append((time.perf_counter() - t0) * 1000.0)

    sp_mean = statistics.mean(sp_times)
    sp_std = statistics.stdev(sp_times) if len(sp_times) > 1 else 0.0
    dp_mean = statistics.mean(dp_times)
    dp_std = statistics.stdev(dp_times) if len(dp_times) > 1 else 0.0
    b_mean = statistics.mean(batch_times)
    b_std = statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0

    res = {
        "ram_gb": residency["size_gb"],
        "vram_gb": residency["size_vram_gb"],
        "single_pass_ms": f"{sp_mean:.1f} ± {sp_std:.1f} ms",
        "single_pass_mean_ms": round(sp_mean, 1),
        "two_pass_ms": f"{dp_mean:.1f} ± {dp_std:.1f} ms",
        "two_pass_mean_ms": round(dp_mean, 1),
        "four_rubric_batch_total_ms": f"{b_mean:.1f} ± {b_std:.1f} ms",
        "per_rubric_batch_ms": f"{b_mean / 4.0:.1f} ± {b_std / 4.0:.1f} ms",
    }
    print(f"  [Latency N={n_trials}] Single: {res['single_pass_ms']} | 2-Pass: {res['two_pass_ms']} | Batch(4): {res['four_rubric_batch_total_ms']}", flush=True)
    return res


def eval_multi_domain_24_cpu(backend: OllamaBackend, model_name: str) -> dict[str, Any]:
    """Evaluates the 24-item multi-domain suite on CPU (max_tokens=200 for step-by-step diagnostics)."""
    print(f"\n--- [Suite 1: 24-Item Multi-Domain Cascade] {model_name} (CPU) ---", flush=True)
    max_tok = 200

    # 1. Baseline Generative Judge (System-2 CoT on all 24 items)
    t0 = time.perf_counter()
    base_tokens = 0
    base_correct = 0
    for item in MULTI_DOMAIN_24_ITEMS:
        jg = JudgeGate(rubric=item["rubric"], backend=backend, model=model_name, max_tokens=max_tok, think=False)
        v = jg.evaluate(item["candidate"])
        base_tokens += v.total_tokens_generated
        if v.passed == item["expected"]:
            base_correct += 1
    base_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Cascade (on_ambiguous_only)
    t_casc_start = time.perf_counter()
    casc_tokens = 0
    casc_correct = 0
    l0_count = 0
    l1_decisive_count = 0
    l2_escalated_count = 0

    for item in MULTI_DOMAIN_24_ITEMS:
        ast_gate = ASTGate(**item.get("ast_config", {}))
        logit_gate = LogitGate(rubric=item["rubric"], backend=backend, model=model_name, preset="balanced")
        judge_gate = JudgeGate(
            backend=backend,
            model=model_name,
            trigger="on_ambiguous_only",
            max_tokens=max_tok,
            think=False,
        )
        v_c = (ast_gate >> logit_gate >> judge_gate).evaluate(item["candidate"], save_on_fail=False)
        casc_tokens += v_c.total_tokens_generated
        if v_c.triggered_layer == ast_gate.name:
            l0_count += 1
        elif any(t.layer_name == judge_gate.name for t in v_c.traces):
            l2_escalated_count += 1
        else:
            l1_decisive_count += 1
        if v_c.passed == item["expected"]:
            casc_correct += 1

    casc_time_ms = (time.perf_counter() - t_casc_start) * 1000.0
    n = len(MULTI_DOMAIN_24_ITEMS)

    res = {
        "baseline_accuracy": f"{base_correct}/{n} ({round(base_correct / n * 100, 1)}%)",
        "baseline_time_ms": round(base_time_ms, 1),
        "baseline_tokens": base_tokens,
        "cascade_accuracy": f"{casc_correct}/{n} ({round(casc_correct / n * 100, 1)}%)",
        "cascade_time_ms": round(casc_time_ms, 1),
        "cascade_tokens": casc_tokens,
        "speedup_factor": round(base_time_ms / max(casc_time_ms, 1.0), 1),
        "tokens_saved": base_tokens - casc_tokens,
        "layer_resolution": {
            "l0_ast": f"{l0_count}/{n} ({round(l0_count / n * 100, 1)}%)",
            "l1_decisive": f"{l1_decisive_count}/{n} ({round(l1_decisive_count / n * 100, 1)}%)",
            "l2_escalated": f"{l2_escalated_count}/{n} ({round(l2_escalated_count / n * 100, 1)}%)",
        },
    }
    print(f"  --> S1: Base={res['baseline_accuracy']} ({base_time_ms/1000:.1f}s) | Casc={res['cascade_accuracy']} ({casc_time_ms/1000:.1f}s, {res['speedup_factor']}x speedup, saved {res['tokens_saved']} tok)", flush=True)
    return res


def eval_reasoning_tiers_cpu(backend: OllamaBackend, model_name: str) -> dict[str, Any]:
    """Evaluates 32-item reasoning tiers on CPU."""
    print(f"\n--- [Suite 2: 32-Item Reasoning Tiers] {model_name} (CPU) ---", flush=True)

    def _eval_tier(items: list[dict[str, Any]], rubric_override: str | None = None) -> tuple[int, int]:
        l1_ok = 0
        casc_ok = 0
        for it in items:
            rubric = rubric_override or it["rubric"]
            gate = LogitGate(rubric=rubric, backend=backend, model=model_name, preset="balanced")
            jg = JudgeGate(
                backend=backend,
                model=model_name,
                trigger="on_ambiguous_only",
                max_tokens=200,
                think=False,
            )
            v_casc = (gate >> jg).evaluate(it["candidate"], save_on_fail=False)
            p = (
                v_casc.traces[0].calibrated_prob
                if (v_casc.traces and v_casc.traces[0].calibrated_prob is not None)
                else 0.5
            )
            l1_pass = p >= 0.5
            if l1_pass == it["expected"]:
                l1_ok += 1
            if v_casc.passed == it["expected"]:
                casc_ok += 1
        return l1_ok, casc_ok

    t_a_l1, t_a_c = _eval_tier(TIER_A_8_ITEMS)
    t_b_l1, t_b_c = _eval_tier(TIER_B_8_ITEMS, rubric_override=TIER_B_COMPOUND_RUBRIC)
    t_c_l1, t_c_c = _eval_tier(TIER_C_8_ITEMS)
    t_d_l1, t_d_c = _eval_tier(TIER_D_8_ITEMS)

    tot_casc = t_a_c + t_b_c + t_c_c + t_d_c
    tot_l1 = t_a_l1 + t_b_l1 + t_c_l1 + t_d_l1

    res = {
        "overall_cascade_accuracy": f"{tot_casc}/32 ({round(tot_casc / 32 * 100, 1)}%)",
        "overall_l1_accuracy": f"{tot_l1}/32 ({round(tot_l1 / 32 * 100, 1)}%)",
        "tier_a": f"{t_a_c}/8 ({round(t_a_c / 8 * 100, 1)}%)",
        "tier_b_compound": f"{t_b_c}/8 ({round(t_b_c / 8 * 100, 1)}%)",
        "tier_c_multi_hop": f"{t_c_c}/8 ({round(t_c_c / 8 * 100, 1)}%)",
        "tier_d_adversarial": f"{t_d_c}/8 ({round(t_d_c / 8 * 100, 1)}%)",
    }
    print(f"  --> S2: Total={res['overall_cascade_accuracy']} (Tier A: {res['tier_a']}, Tier B: {res['tier_b_compound']}, Tier C: {res['tier_c_multi_hop']}, Tier D: {res['tier_d_adversarial']})", flush=True)
    return res


def eval_choice_and_score_cpu(backend: OllamaBackend, model_name: str) -> dict[str, Any]:
    """Evaluates ChoiceGate (6 items) and ScoreGate (6 items) on CPU."""
    print(f"\n--- [Suite 3: Multi-Action Gates] {model_name} (CPU) ---", flush=True)

    # ChoiceGate (2-pass debiased)
    cg_correct = 0
    for it in CHOICE_GATE_6_ITEMS:
        cg = ChoiceGate(question=it["question"], options=it["options"], backend=backend, model=model_name, symmetric_debiasing=True)
        v = cg.evaluate(it["candidate"])
        if v.choice == it["expected_choice"]:
            cg_correct += 1

    # ScoreGate (monotonicity)
    scores_by_tier: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    for it in SCORE_GATE_6_ITEMS:
        sg = ScoreGate(rubric=it["rubric"], min_score=1, max_score=5, backend=backend, model=model_name, symmetric_debiasing=True)
        v = sg.evaluate(it["candidate"])
        s = v.score if v.score is not None else 3.0
        scores_by_tier[it["tier"]].append(s)

    h_mean = statistics.mean(scores_by_tier["high"])
    m_mean = statistics.mean(scores_by_tier["medium"])
    l_mean = statistics.mean(scores_by_tier["low"])

    monotonic_pairs = 0
    total_pairs = 0
    for h in scores_by_tier["high"]:
        for m in scores_by_tier["medium"]:
            total_pairs += 1
            if h > m:
                monotonic_pairs += 1
    for m in scores_by_tier["medium"]:
        for l in scores_by_tier["low"]:
            total_pairs += 1
            if m > l:
                monotonic_pairs += 1

    res = {
        "choice_gate_accuracy": f"{cg_correct}/6 ({round(cg_correct / 6 * 100, 1)}%)",
        "score_gate_monotonicity": f"{monotonic_pairs}/{total_pairs} ({round(monotonic_pairs / total_pairs * 100, 1)}%)",
        "score_high_mean": round(h_mean, 2),
        "score_med_mean": round(m_mean, 2),
        "score_low_mean": round(l_mean, 2),
    }
    print(f"  --> S3: ChoiceGate={res['choice_gate_accuracy']} | ScoreGate Monotonicity={res['score_gate_monotonicity']} (H={res['score_high_mean']}, M={res['score_med_mean']}, L={res['score_low_mean']})", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LogitGate CPU benchmarks.")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "cpu_results.json"))
    parser.add_argument("--models", nargs="+", default=["llama3.2:1b", "llama3.2:3b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b"])
    args = parser.parse_args()

    base_url = os.environ.get("LOGITGATE_BENCH_CPU_URL", "http://localhost:11435")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    master_results: dict[str, Any] = {}

    for model in args.models:
        print("\n=======================================================", flush=True)
        print(f"  BENCHMARKING CPU CANDIDATE: {model} (num_gpu=0)", flush=True)
        print("=======================================================", flush=True)

        backend = OllamaBackend(base_url=base_url, model=model, options={"num_gpu": 0}, timeout=180.0)
        try:
            model_summary: dict[str, Any] = {}

            # 1. Warmup and N=5 latency trials
            latencies = measure_cpu_latencies(backend, model, base_url, n_trials=5)
            model_summary["latencies"] = latencies

            # 2. Suite 1: 24-Item Multi-Domain Cascade
            s1 = eval_multi_domain_24_cpu(backend, model)
            model_summary["suite_1_multi_domain_24"] = s1

            # 3. Suite 2: 32-Item Reasoning Depth Tiers
            s2 = eval_reasoning_tiers_cpu(backend, model)
            model_summary["suite_2_reasoning_depth_32"] = s2

            # 4. Suite 3: Multi-Action Gates
            s3 = eval_choice_and_score_cpu(backend, model)
            model_summary["suite_3_choice_and_score"] = s3

            master_results[model] = model_summary

            # Incremental save
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(master_results, f, indent=2)

        finally:
            backend.close()
            unload_model(base_url, model)
            time.sleep(1.0)

    print(f"\nALL CPU BENCHMARKS COMPLETE! Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
