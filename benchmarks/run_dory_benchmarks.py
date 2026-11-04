#!/usr/bin/env python3
"""Dory / Recurrent Layer-Looping Benchmark Runner for LogitGate.

Executes the full benchmark suites (Multi-Domain 24-Item, 32-Item Reasoning Depth,
Choice & Score, and Diverse Haystack) against NVIDIA-Nemotron-Labs Dory (4B) across
both available GPU cards:
- Device 0: NVIDIA GeForce RTX 5090 (32 GB)
- Device 1: NVIDIA GeForce RTX 4070 (12 GB)
and across recurrent loop depths (num_loops = 1, 2, 4).

Usage:
    python benchmarks/run_dory_benchmarks.py [--devices cuda:0 cuda:1] [--loops 1 2 4] [--output benchmarks/dory_results.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from dataset import (
    CHOICE_GATE_6_ITEMS,
    HAYSTACK_RUBRIC,
    MULTI_DOMAIN_24_ITEMS,
    NEEDLE_COMPLIANT,
    NEEDLE_VIOLATING,
    SCORE_GATE_6_ITEMS,
    TIER_A_8_ITEMS,
    TIER_B_8_ITEMS,
    TIER_B_ATOMIC_RUBRICS,
    TIER_B_COMPOUND_RUBRIC,
    TIER_C_8_ITEMS,
    TIER_D_8_ITEMS,
    build_diverse_haystack_document,
)

from logitgate import ASTGate, ChoiceGate, JudgeGate, LogitGate, ScoreGate
from logitgate.backends.recurrent import RecurrentPyTorchBackend
from logitgate.models import DecisionStatus


def eval_multi_domain_24(backend: Any, model_label: str) -> dict[str, Any]:
    """Evaluates the 24-item multi-domain suite comparing Baseline Judge vs Cascade."""
    print(f"\n=== [Suite 1: 24-Item Multi-Domain Cascade] {model_label} ===", flush=True)

    max_tok = 40
    # Warmup
    LogitGate(rubric="Warmup check?", backend=backend).evaluate("ok")

    # 1. Baseline Standalone Generative Judge (System-2)
    print("--- Running Baseline Standalone Generative Judge (System-2) ---", flush=True)
    t0 = time.perf_counter()
    base_tokens = 0
    base_correct = 0
    for idx, item in enumerate(MULTI_DOMAIN_24_ITEMS, 1):
        jg = JudgeGate(rubric=item["rubric"], backend=backend, max_tokens=max_tok, think=False)
        t_item = time.perf_counter()
        v = jg.evaluate(item["candidate"])
        dt_ms = (time.perf_counter() - t_item) * 1000.0
        base_tokens += v.total_tokens_generated
        is_corr = (v.passed == item["expected"])
        if is_corr:
            base_correct += 1
        print(f"  [{idx:02d}/24] {item['id']:<38} Exp={item['expected']!s:<5} Judge={v.passed!s:<5} "
              f"[{'OK' if is_corr else 'ERR'}] ({dt_ms:6.1f}ms, {v.total_tokens_generated} tok)", flush=True)
    base_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Cascade Evaluation
    print("--- Running Cascade Evaluation (L0 -> L1 -> L2) ---", flush=True)
    time_ambig_or_fail_ms = 0.0
    tokens_ambig_or_fail = 0
    correct_ambig_or_fail = 0

    time_ambig_only_ms = 0.0
    tokens_ambig_only = 0
    correct_ambig_only = 0

    l0_count = 0
    l1_decisive_count = 0
    l2_escalated_count = 0

    for idx, item in enumerate(MULTI_DOMAIN_24_ITEMS, 1):
        ast_gate = ASTGate(**item.get("ast_config", {}))
        logit_gate = LogitGate(rubric=item["rubric"], backend=backend, preset="balanced")

        t_start = time.perf_counter()
        v_l0 = ast_gate.evaluate(item["candidate"])
        if not v_l0.passed:
            dt_l0 = (time.perf_counter() - t_start) * 1000.0
            l0_count += 1
            time_ambig_or_fail_ms += dt_l0
            time_ambig_only_ms += dt_l0
            if not item["expected"]:
                correct_ambig_or_fail += 1
                correct_ambig_only += 1
            print(f"  [{idx:02d}/24] {item['id']:<38} -> Resolved by L0 ASTGate in {dt_l0:.1f}ms (0 tokens)", flush=True)
            continue

        v_l1 = logit_gate.evaluate(item["candidate"])
        dt_l0_l1 = (time.perf_counter() - t_start) * 1000.0

        if v_l1.status == DecisionStatus.PASSED:
            l1_decisive_count += 1
            time_ambig_or_fail_ms += dt_l0_l1
            time_ambig_only_ms += dt_l0_l1
            if item["expected"]:
                correct_ambig_or_fail += 1
                correct_ambig_only += 1
            print(f"  [{idx:02d}/24] {item['id']:<38} -> Resolved by L1 LogitGate PASS (P={v_l1.probability:.2f}) in {dt_l0_l1:.1f}ms (0 tokens)", flush=True)
        elif v_l1.status == DecisionStatus.FAILED:
            l1_decisive_count += 1
            time_ambig_only_ms += dt_l0_l1
            if not item["expected"]:
                correct_ambig_only += 1

            # on_ambiguous_or_fail generates diagnostic
            t_j = time.perf_counter()
            jg = JudgeGate(rubric=item["rubric"], backend=backend, max_tokens=max_tok, think=False)
            v_j = jg.evaluate(item["candidate"])
            dt_j = (time.perf_counter() - t_j) * 1000.0
            time_ambig_or_fail_ms += dt_l0_l1 + dt_j
            tokens_ambig_or_fail += v_j.total_tokens_generated
            if not item["expected"]:
                correct_ambig_or_fail += 1
            print(f"  [{idx:02d}/24] {item['id']:<38} -> Resolved by L1 LogitGate FAIL (P={v_l1.probability:.2f}) in {dt_l0_l1:.1f}ms (esc diagnostics: {dt_j:.1f}ms)", flush=True)
        else:
            # Ambiguous -> escalate to JudgeGate
            l2_escalated_count += 1
            t_j = time.perf_counter()
            jg = JudgeGate(rubric=item["rubric"], backend=backend, max_tokens=max_tok, think=False)
            v_j = jg.evaluate(item["candidate"])
            dt_j = (time.perf_counter() - t_j) * 1000.0
            time_ambig_or_fail_ms += dt_l0_l1 + dt_j
            time_ambig_only_ms += dt_l0_l1 + dt_j
            tokens_ambig_or_fail += v_j.total_tokens_generated
            tokens_ambig_only += v_j.total_tokens_generated
            if v_j.passed == item["expected"]:
                correct_ambig_or_fail += 1
                correct_ambig_only += 1
            print(f"  [{idx:02d}/24] {item['id']:<38} -> AMBIGUOUS (P={v_l1.probability:.2f}) -> L2 JudgeGate ({dt_j:.1f}ms, {v_j.total_tokens_generated} tok)", flush=True)

    n = len(MULTI_DOMAIN_24_ITEMS)
    speedup_only = round(base_time_ms / max(time_ambig_only_ms, 0.1), 1)
    tokens_saved_only = base_tokens - tokens_ambig_only

    res = {
        "model": model_label,
        "baseline_time_ms": round(base_time_ms, 1),
        "baseline_tokens": base_tokens,
        "baseline_accuracy": f"{base_correct}/{n} ({round(base_correct / n * 100, 1)}%)",
        "cascade_ambig_or_fail_time_ms": round(time_ambig_or_fail_ms, 1),
        "cascade_ambig_or_fail_tokens": tokens_ambig_or_fail,
        "cascade_ambig_or_fail_accuracy": f"{correct_ambig_or_fail}/{n} ({round(correct_ambig_or_fail / n * 100, 1)}%)",
        "cascade_ambig_only_time_ms": round(time_ambig_only_ms, 1),
        "cascade_ambig_only_tokens": tokens_ambig_only,
        "cascade_ambig_only_accuracy": f"{correct_ambig_only}/{n} ({round(correct_ambig_only / n * 100, 1)}%)",
        "speedup_ambig_only": speedup_only,
        "tokens_saved_ambig_only": tokens_saved_only,
        "layer_resolution": {
            "l0_ast": f"{l0_count}/{n} ({round(l0_count / n * 100, 1)}%)",
            "l1_decisive": f"{l1_decisive_count}/{n} ({round(l1_decisive_count / n * 100, 1)}%)",
            "l2_escalated": f"{l2_escalated_count}/{n} ({round(l2_escalated_count / n * 100, 1)}%)",
            "l0_plus_l1_rate": f"{l0_count + l1_decisive_count}/{n} ({round((l0_count + l1_decisive_count) / n * 100, 1)}%)",
        },
    }
    print(f"  --> Baseline: {res['baseline_accuracy']} in {res['baseline_time_ms']}ms", flush=True)
    print(f"  --> Cascade (on_ambiguous_only): {res['cascade_ambig_only_accuracy']} in {res['cascade_ambig_only_time_ms']}ms ({speedup_only}x speedup, saved {tokens_saved_only} tokens)", flush=True)
    return res


def eval_reasoning_tiers(backend: Any, model_label: str) -> dict[str, Any]:
    """Evaluates the 32-item reasoning depth suite (Tiers A, B-Compound, B-Atomic, C, D)."""
    print(f"\n=== [Suite 2: 32-Item Reasoning Depth Tiers] {model_label} ===", flush=True)
    out: dict[str, Any] = {}

    def _run_binary_tier(tier_name: str, items: list[dict[str, Any]], rubric_override: str | None = None) -> dict[str, Any]:
        l1_correct = 0
        l1_fp = 0
        l1_esc = 0
        casc_correct = 0
        rows = []
        n_items = len(items)

        for item in items:
            rubric = rubric_override or item.get("rubric", "")
            gate = LogitGate(rubric=rubric, backend=backend, preset="balanced")
            v_l1 = gate.evaluate(item["candidate"])
            v_casc = v_l1
            if v_l1.status == DecisionStatus.AMBIGUOUS:
                l1_esc += 1
                jg = JudgeGate(rubric=rubric, backend=backend, max_tokens=40, think=False)
                v_casc = jg.evaluate(item["candidate"])

            p = v_l1.probability if v_l1.probability is not None else 0.5
            l1_pass = p >= 0.5

            if v_l1.status == DecisionStatus.PASSED:
                casc_pass = True
            elif v_l1.status == DecisionStatus.FAILED:
                casc_pass = False
            else:
                casc_pass = v_casc.passed

            if not item["expected"] and v_l1.status == DecisionStatus.PASSED:
                l1_fp += 1
            if l1_pass == item["expected"]:
                l1_correct += 1
            if casc_pass == item["expected"]:
                casc_correct += 1

            rows.append({
                "id": item["id"],
                "expected": item["expected"],
                "l1_status": v_l1.status.value,
                "l1_prob": round(p, 4),
                "l1_correct": l1_pass == item["expected"],
                "casc_correct": casc_pass == item["expected"],
            })
            print(f"  [{tier_name}] {item['id']:<35} Exp={item['expected']!s:<5} P={round(p, 2):<4} "
                  f"L1={'OK' if l1_pass == item['expected'] else 'ERR'} Casc={'OK' if casc_pass == item['expected'] else 'ERR'}", flush=True)

        return {
            "l1_correct": l1_correct,
            "total": n_items,
            "l1_accuracy": f"{l1_correct}/{n_items} ({round(l1_correct / n_items * 100, 1)}%)",
            "l1_false_pass": l1_fp,
            "l1_false_pass_rate": f"{l1_fp}/4 ({round(l1_fp / 4 * 100, 1)}%)",
            "l1_escalated": l1_esc,
            "l1_escalation_rate": f"{l1_esc}/{n_items} ({round(l1_esc / n_items * 100, 1)}%)",
            "casc_correct": casc_correct,
            "casc_accuracy": f"{casc_correct}/{n_items} ({round(casc_correct / n_items * 100, 1)}%)",
            "items": rows,
        }

    out["tier_a"] = _run_binary_tier("tier_a", TIER_A_8_ITEMS)
    out["tier_b_compound"] = _run_binary_tier("tier_b_compound", TIER_B_8_ITEMS, rubric_override=TIER_B_COMPOUND_RUBRIC)

    # Tier B Atomic
    print("--- Running Tier B Atomic Isolated Batch ---", flush=True)
    gate_b = LogitGate(rubric="Atomic Checker", backend=backend, preset="balanced")
    l1_b_atom_correct = 0
    l1_b_atom_fp = 0
    l1_b_atom_esc = 0
    casc_b_atom_correct = 0
    atom_rows = []

    for item in TIER_B_8_ITEMS:
        verdicts = gate_b.evaluate_batch(item["candidate"], TIER_B_ATOMIC_RUBRICS)
        probs = [v.probability or 0.5 for v in verdicts]
        has_fail = any(v.status == DecisionStatus.FAILED for v in verdicts)
        has_ambig = any(v.status == DecisionStatus.AMBIGUOUS for v in verdicts)

        if has_fail:
            l1_pass = False
        elif has_ambig:
            l1_b_atom_esc += 1
            l1_pass = all(p >= 0.5 for p in probs)
        else:
            l1_pass = True

        if not item["expected"] and all(v.status == DecisionStatus.PASSED for v in verdicts):
            l1_b_atom_fp += 1
        if l1_pass == item["expected"]:
            l1_b_atom_correct += 1

        rule_passes = []
        for v, r in zip(verdicts, TIER_B_ATOMIC_RUBRICS):
            if v.status == DecisionStatus.PASSED:
                rule_passes.append(True)
            elif v.status == DecisionStatus.FAILED:
                rule_passes.append(False)
            else:
                jg = JudgeGate(rubric=r, backend=backend, max_tokens=40, think=False)
                rule_passes.append(jg.evaluate(item["candidate"]).passed)
        casc_pass = all(rule_passes)
        if casc_pass == item["expected"]:
            casc_b_atom_correct += 1
        atom_rows.append({
            "id": item["id"],
            "expected": item["expected"],
            "probs": [round(p, 4) for p in probs],
            "l1_correct": l1_pass == item["expected"],
            "casc_correct": casc_pass == item["expected"],
        })
        print(f"  [tier_b_atomic] {item['id']:<35} Exp={item['expected']!s:<5} Probs={[round(p, 2) for p in probs]} "
              f"L1={'OK' if l1_pass == item['expected'] else 'ERR'} Casc={'OK' if casc_pass == item['expected'] else 'ERR'}", flush=True)

    n_b = len(TIER_B_8_ITEMS)
    out["tier_b_atomic"] = {
        "l1_correct": l1_b_atom_correct,
        "total": n_b,
        "l1_accuracy": f"{l1_b_atom_correct}/{n_b} ({round(l1_b_atom_correct / n_b * 100, 1)}%)",
        "l1_false_pass": l1_b_atom_fp,
        "l1_escalated": l1_b_atom_esc,
        "casc_correct": casc_b_atom_correct,
        "casc_accuracy": f"{casc_b_atom_correct}/{n_b} ({round(casc_b_atom_correct / n_b * 100, 1)}%)",
        "items": atom_rows,
    }

    out["tier_c"] = _run_binary_tier("tier_c", TIER_C_8_ITEMS)
    out["tier_d"] = _run_binary_tier("tier_d", TIER_D_8_ITEMS)

    tot_comp = out["tier_a"]["casc_correct"] + out["tier_b_compound"]["casc_correct"] + out["tier_c"]["casc_correct"] + out["tier_d"]["casc_correct"]
    tot_atom = out["tier_a"]["casc_correct"] + out["tier_b_atomic"]["casc_correct"] + out["tier_c"]["casc_correct"] + out["tier_d"]["casc_correct"]

    out["summary_32"] = {
        "cascade_compound_b": f"{tot_comp}/32 ({round(tot_comp / 32 * 100, 1)}%)",
        "cascade_atomic_b": f"{tot_atom}/32 ({round(tot_atom / 32 * 100, 1)}%)",
    }
    print(f"  --> Reasoning Summary for {model_label}: Compound={out['summary_32']['cascade_compound_b']}, Atomic={out['summary_32']['cascade_atomic_b']}", flush=True)
    return out


def eval_choice_and_score(backend: Any, model_label: str) -> dict[str, Any]:
    """Evaluates ChoiceGate (6 items) and ScoreGate (6 items)."""
    print(f"\n=== [Suite 3: ChoiceGate & ScoreGate] {model_label} ===", flush=True)
    out: dict[str, Any] = {}

    # ChoiceGate
    choice_correct = 0
    choice_lats = []
    for it in CHOICE_GATE_6_ITEMS:
        gate = ChoiceGate(question=it["question"], options=it["options"], backend=backend, symmetric_debiasing=True)
        t0 = time.perf_counter()
        v = gate.evaluate(it["candidate"])
        lat = (time.perf_counter() - t0) * 1000.0
        choice_lats.append(lat)
        exp = it.get("expected_choice") or it.get("expected")
        if v.choice == exp:
            choice_correct += 1
        print(f"  [ChoiceGate] {it['id']:<32} Exp={exp} Got={v.choice} [{'OK' if v.choice == exp else 'ERR'}] ({lat:5.1f}ms)", flush=True)

    out["choice_gate"] = {
        "correct": choice_correct,
        "total": len(CHOICE_GATE_6_ITEMS),
        "accuracy": f"{choice_correct}/{len(CHOICE_GATE_6_ITEMS)} ({round(choice_correct/len(CHOICE_GATE_6_ITEMS)*100, 1)}%)",
        "mean_latency_ms": round(statistics.mean(choice_lats), 1),
        "std_latency_ms": round(statistics.stdev(choice_lats), 1) if len(choice_lats) > 1 else 0.0,
    }

    # ScoreGate
    score_lats = []
    scores_by_tier: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    in_band_count = 0

    for it in SCORE_GATE_6_ITEMS:
        score_gate = ScoreGate(rubric=it["rubric"], min_score=1, max_score=5, backend=backend, symmetric_debiasing=True)
        t0 = time.perf_counter()
        v = score_gate.evaluate(it["candidate"])
        lat = (time.perf_counter() - t0) * 1000.0
        score_lats.append(lat)
        s = v.score if v.score is not None else 3.0
        scores_by_tier[it["tier"]].append(s)
        lo, hi = it["expected_band"]
        if lo <= s <= hi:
            in_band_count += 1
        print(f"  [ScoreGate] {it['id']:<32} Tier={it['tier']:<6} Score={s:4.2f} (Band: [{lo}, {hi}]) ({lat:5.1f}ms)", flush=True)

    h_mean = statistics.mean(scores_by_tier["high"])
    m_mean = statistics.mean(scores_by_tier["medium"])
    l_mean = statistics.mean(scores_by_tier["low"])

    monotonic = 0
    total_pairs = 0
    for h in scores_by_tier["high"]:
        for m in scores_by_tier["medium"]:
            total_pairs += 1
            if h > m:
                monotonic += 1
    for m in scores_by_tier["medium"]:
        for l in scores_by_tier["low"]:
            total_pairs += 1
            if m > l:
                monotonic += 1

    out["score_gate"] = {
        "score_high_mean": round(h_mean, 2),
        "score_medium_mean": round(m_mean, 2),
        "score_low_mean": round(l_mean, 2),
        "monotonic_pairs": monotonic,
        "total_pairs": total_pairs,
        "monotonicity_rate": f"{monotonic}/{total_pairs} ({round(monotonic/total_pairs*100, 1)}%)",
        "in_band_rate": f"{in_band_count}/6 ({round(in_band_count/6*100, 1)}%)",
        "mean_latency_ms": round(statistics.mean(score_lats), 1),
        "std_latency_ms": round(statistics.stdev(score_lats), 1) if len(score_lats) > 1 else 0.0,
    }
    print(f"  --> ChoiceGate: {out['choice_gate']['accuracy']} in {out['choice_gate']['mean_latency_ms']}ms", flush=True)
    print(f"  --> ScoreGate Monotonicity: {out['score_gate']['monotonicity_rate']} (High={h_mean:.2f}, Med={m_mean:.2f}, Low={l_mean:.2f})", flush=True)
    return out


def eval_diverse_haystack(backend: Any, model_label: str) -> list[dict[str, Any]]:
    """Evaluates the haystack document across context lengths (200 to 16,000 tokens)."""
    print(f"\n=== [Suite 4: Diverse Haystack Context Scaling] {model_label} ===", flush=True)
    tiers = [
        ("Short (~200 tok)", 150),
        ("Medium (~1,500 tok)", 1100),
        ("Long (~4,500 tok)", 3300),
        ("Very Long (~8,000 tok)", 5800),
        ("Massive (~16,000 tok)", 11000),
    ]
    gate = LogitGate(rubric=HAYSTACK_RUBRIC, backend=backend, preset="balanced")
    results = []

    for label, words in tiers:
        cand_pass = build_diverse_haystack_document(words, NEEDLE_COMPLIANT)
        cand_fail = build_diverse_haystack_document(words, NEEDLE_VIOLATING)

        t0 = time.perf_counter()
        v_pass = gate.evaluate(cand_pass)
        t_pass = (time.perf_counter() - t0) * 1000.0

        v_fail = gate.evaluate(cand_fail)
        p_c = v_pass.probability or 0.0
        p_v = v_fail.probability or 0.0
        delta = p_c - p_v
        correct = (p_c >= 0.5 and p_v < 0.5)

        row = {
            "tier": label,
            "warm_latency_ms": round(t_pass, 1),
            "p_compliant": round(p_c, 4),
            "p_violating": round(p_v, 4),
            "delta_p": round(delta, 4),
            "correct_both": correct,
        }
        results.append(row)
        print(f"  [{label}] P_pass={p_c:.4f} P_fail={p_v:.4f} Delta={delta:.4f} [{'OK' if correct else 'ERR'}] ({t_pass:6.1f}ms)", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dory benchmarks on GPUs")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"], help="CUDA devices to benchmark (e.g. cuda:0 cuda:1)")
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4], help="Recurrent loop counts to benchmark (e.g. 1 2 4)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks" / "dory_results.json")
    parser.add_argument("--model-name", default="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
    args = parser.parse_args()

    results: dict[str, Any] = {}

    for dev in args.devices:
        dev_results: dict[str, Any] = {}
        print("\n=======================================================", flush=True)
        print(f"  LOADING MODEL ON DEVICE: {dev}", flush=True)
        print("=======================================================", flush=True)

        from transformers import AutoTokenizer

        from logitgate.backends.dory_pytorch import load_dory_model

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = load_dory_model(args.model_name, device=dev, num_loops=2)

        for k in args.loops:
            label = f"dory-4b_{dev}_k{k}"
            print("\n-------------------------------------------------------", flush=True)
            print(f"  BENCHMARKING: {label} (Device={dev}, num_loops={k})", flush=True)
            print("-------------------------------------------------------", flush=True)

            backend = RecurrentPyTorchBackend(
                model_name=args.model_name,
                num_loops=k,
                device=dev,
                model=model,
                tokenizer=tokenizer,
            )

            k_results = {
                "device": dev,
                "num_loops": k,
                "model_name": args.model_name,
                "suite_1_multi_domain_24": eval_multi_domain_24(backend, label),
                "suite_2_reasoning_depth_32": eval_reasoning_tiers(backend, label),
                "suite_3_choice_and_score": eval_choice_and_score(backend, label),
                "suite_4_diverse_haystack": eval_diverse_haystack(backend, label),
            }
            dev_results[f"k{k}"] = k_results

            args.output.parent.mkdir(parents=True, exist_ok=True)
            results[dev] = dev_results
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        # Clean up model from device
        del model
        del tokenizer
        del backend
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nALL DORY BENCHMARKS COMPLETE! Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
