#!/usr/bin/env python3
"""Master reproducible benchmark runner for LogitGate.

Executes the leak-free multi-domain, reasoning-depth falsification, ChoiceGate,
ScoreGate, and diverse long-context haystack benchmarks sequentially, unloading
models between runs.

Usage:
    LOGITGATE_BENCH_GPU0_URL=http://localhost:11434 \\
    LOGITGATE_BENCH_GPU1_URL=http://localhost:11435 \\
    python benchmarks/run_benchmarks.py --output benchmarks/latest_results.json
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
from logitgate.backends.ollama import OllamaBackend
from logitgate.models import DecisionStatus


def unload_all(urls: list[str]) -> None:
    """Unloads all active models on the target Ollama servers."""
    known = ["gemma4:e4b", "qwen2.5:14b", "qwen2.5-coder:14b", "gemma4:26b", "gemma4:31b"]
    for base_url in set(urls):
        try:
            with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5.0) as resp:
                data = json.loads(resp.read().decode())
                active = [m.get("model") or m.get("name") for m in data.get("models", [])]
        except (OSError, ValueError):
            active = []
        for model in set(active + known):
            if not model:
                continue
            try:
                req = urllib.request.Request(
                    f"{base_url}/api/generate",
                    data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    resp.read()
            except OSError:
                continue


def eval_reasoning_tiers(backend: OllamaBackend, model_name: str) -> dict[str, Any]:
    """Evaluates the 32-item reasoning depth suite (Tiers A, B-Compound, B-Atomic, C, D)."""
    print(f"\n=== [Suite 2: 32-Item Reasoning Depth] Model: {model_name} ===", flush=True)
    out: dict[str, Any] = {}

    def _run_binary_tier(tier_key: str, items: list[dict[str, Any]], rubric_override: str | None = None) -> dict[str, Any]:
        l1_correct = 0
        l1_false_pass = 0
        l1_escalated = 0
        casc_correct = 0
        casc_on_ambig_only_correct = 0
        rows = []

        for item in items:
            rubric = rubric_override or item["rubric"]
            gate = LogitGate(rubric=rubric, backend=backend, model=model_name, preset="balanced")
            v_l1 = gate.evaluate(item["candidate"])
            p = v_l1.probability if v_l1.probability is not None else 0.5
            l1_pass = p >= 0.5
            if l1_pass == item["expected"]:
                l1_correct += 1
            if p >= 0.85 and not item["expected"]:
                l1_false_pass += 1
            if 0.20 < p < 0.85:
                l1_escalated += 1

            # Cascade (on_ambiguous_or_fail)
            if v_l1.status == DecisionStatus.PASSED:
                casc_pass = True
                layer_used = "LogitGate"
            else:
                jg = JudgeGate(
                    rubric=rubric,
                    backend=backend,
                    model=model_name,
                    trigger="on_ambiguous_or_fail",
                    max_tokens=250,
                    think=False,
                )
                v_j = jg.evaluate(item["candidate"])
                # Monotonic fail safety: if L1 was decisively FAILED (p <= 0.20), keep fail
                casc_pass = False if v_l1.status == DecisionStatus.FAILED else v_j.passed
                layer_used = "JudgeGate"

            if casc_pass == item["expected"]:
                casc_correct += 1

            # Cascade (on_ambiguous_only)
            if v_l1.status == DecisionStatus.PASSED:
                ambig_only_pass = True
            elif v_l1.status == DecisionStatus.FAILED:
                ambig_only_pass = False
            else:
                ambig_only_pass = casc_pass
            if ambig_only_pass == item["expected"]:
                casc_on_ambig_only_correct += 1

            rows.append({
                "id": item["id"],
                "expected": item["expected"],
                "prob": round(p, 4),
                "l1_status": v_l1.status.value,
                "l1_correct": l1_pass == item["expected"],
                "casc_passed": casc_pass,
                "casc_correct": casc_pass == item["expected"],
                "layer_used": layer_used,
            })
            print(
                f"  [{tier_key}] {item['id']}: exp={item['expected']} P={p:.4f} ({v_l1.status.value}) "
                f"L1={'OK' if l1_pass == item['expected'] else 'ERR'} "
                f"Casc={'OK' if casc_pass == item['expected'] else 'ERR'}",
                flush=True,
            )

        n = len(items)
        n_fail = sum(1 for it in items if not it["expected"])
        return {
            "l1_correct": l1_correct,
            "total": n,
            "l1_accuracy": f"{l1_correct}/{n} ({round(l1_correct / n * 100, 1)}%)",
            "l1_false_pass": l1_false_pass,
            "l1_false_pass_rate": f"{l1_false_pass}/{n_fail} ({round(l1_false_pass / max(n_fail, 1) * 100, 1)}%)",
            "l1_escalated": l1_escalated,
            "l1_escalation_rate": f"{l1_escalated}/{n} ({round(l1_escalated / n * 100, 1)}%)",
            "casc_correct": casc_correct,
            "casc_accuracy": f"{casc_correct}/{n} ({round(casc_correct / n * 100, 1)}%)",
            "casc_ambig_only_correct": casc_on_ambig_only_correct,
            "items": rows,
        }

    out["tier_a"] = _run_binary_tier("tier_a", TIER_A_8_ITEMS)
    out["tier_b_compound"] = _run_binary_tier("tier_b_compound", TIER_B_8_ITEMS, rubric_override=TIER_B_COMPOUND_RUBRIC)

    # Tier B Atomic via evaluate_batch
    l1_b_atom_correct = 0
    l1_b_atom_fp = 0
    l1_b_atom_esc = 0
    casc_b_atom_correct = 0
    atom_rows = []
    gate_batch = LogitGate(rubric=TIER_B_ATOMIC_RUBRICS[0], backend=backend, model=model_name, preset="balanced")
    for item in TIER_B_8_ITEMS:
        verdicts = gate_batch.evaluate_batch(item["candidate"], TIER_B_ATOMIC_RUBRICS)
        probs = [v.probability if v.probability is not None else 0.5 for v in verdicts]
        l1_pass = all(p >= 0.5 for p in probs)
        if l1_pass == item["expected"]:
            l1_b_atom_correct += 1
        if all(p >= 0.85 for p in probs) and not item["expected"]:
            l1_b_atom_fp += 1
        if any(0.20 < p < 0.85 for p in probs):
            l1_b_atom_esc += 1

        # Cascade per atomic rubric on ambiguous
        rule_passes = []
        for idx, v in enumerate(verdicts):
            if v.status == DecisionStatus.PASSED:
                rule_passes.append(True)
            elif v.status == DecisionStatus.FAILED:
                rule_passes.append(False)
            else:
                jg = JudgeGate(
                    rubric=TIER_B_ATOMIC_RUBRICS[idx],
                    backend=backend,
                    model=model_name,
                    trigger="on_ambiguous_only",
                    max_tokens=250,
                    think=False,
                )
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
        print(
            f"  [tier_b_atomic] {item['id']}: exp={item['expected']} probs={[round(p, 2) for p in probs]} "
            f"L1={'OK' if l1_pass == item['expected'] else 'ERR'} Casc={'OK' if casc_pass == item['expected'] else 'ERR'}",
            flush=True,
        )

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

    total_casc_compound = (
        out["tier_a"]["casc_correct"]
        + out["tier_b_compound"]["casc_correct"]
        + out["tier_c"]["casc_correct"]
        + out["tier_d"]["casc_correct"]
    )
    total_casc_atomic = (
        out["tier_a"]["casc_correct"]
        + out["tier_b_atomic"]["casc_correct"]
        + out["tier_c"]["casc_correct"]
        + out["tier_d"]["casc_correct"]
    )
    out["summary_32"] = {
        "cascade_compound_b": f"{total_casc_compound}/32 ({round(total_casc_compound / 32 * 100, 1)}%)",
        "cascade_atomic_b": f"{total_casc_atomic}/32 ({round(total_casc_atomic / 32 * 100, 1)}%)",
    }
    print(f"  --> 32-Item Summary for {model_name}: {out['summary_32']}", flush=True)
    return out


def eval_multi_domain_24(backend: OllamaBackend, model_name: str, is_cpu: bool = False) -> dict[str, Any]:
    """Evaluates the 24-item multi-domain suite (Baseline Judge vs Cascade modes)."""
    label = f"{model_name} ({'Pure CPU' if is_cpu else 'GPU'})"
    print(f"\n=== [Suite 1: 24-Item Multi-Domain Cascade] {label} ===", flush=True)

    max_tok = 80 if is_cpu else 250
    # Warm-up call
    LogitGate(rubric="Warmup check?", backend=backend, model=model_name).evaluate("ok")

    # 1. Baseline Generative Judge on all 24 items
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

    # 2. Cascade (on_ambiguous_or_fail) and (on_ambiguous_only)
    time_ambig_or_fail_ms = 0.0
    tokens_ambig_or_fail = 0
    correct_ambig_or_fail = 0

    time_ambig_only_ms = 0.0
    tokens_ambig_only = 0
    correct_ambig_only = 0

    l0_count = 0
    l1_decisive_count = 0
    l2_escalated_count = 0

    for item in MULTI_DOMAIN_24_ITEMS:
        ast_gate = ASTGate(**item.get("ast_config", {}))
        logit_gate = LogitGate(rubric=item["rubric"], backend=backend, model=model_name, preset="balanced")

        # Measure L0 + L1
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
        elif v_l1.status == DecisionStatus.FAILED:
            l1_decisive_count += 1
            # on_ambiguous_only stops right here with 0 CoT tokens!
            time_ambig_only_ms += dt_l0_l1
            if not item["expected"]:
                correct_ambig_only += 1

            # on_ambiguous_or_fail invokes JudgeGate for failure diagnostics
            t_j = time.perf_counter()
            jg = JudgeGate(rubric=item["rubric"], backend=backend, model=model_name, max_tokens=max_tok, think=False)
            v_j = jg.evaluate(item["candidate"])
            dt_j = (time.perf_counter() - t_j) * 1000.0
            time_ambig_or_fail_ms += dt_l0_l1 + dt_j
            tokens_ambig_or_fail += v_j.total_tokens_generated
            # Monotonic L1 fail keeps passed=False
            if not item["expected"]:
                correct_ambig_or_fail += 1
        else:
            # AMBIGUOUS: both modes escalate to JudgeGate!
            l2_escalated_count += 1
            t_j = time.perf_counter()
            jg = JudgeGate(rubric=item["rubric"], backend=backend, model=model_name, max_tokens=max_tok, think=False)
            v_j = jg.evaluate(item["candidate"])
            dt_j = (time.perf_counter() - t_j) * 1000.0
            time_ambig_or_fail_ms += dt_l0_l1 + dt_j
            time_ambig_only_ms += dt_l0_l1 + dt_j
            tokens_ambig_or_fail += v_j.total_tokens_generated
            tokens_ambig_only += v_j.total_tokens_generated
            if v_j.passed == item["expected"]:
                correct_ambig_or_fail += 1
                correct_ambig_only += 1

    n = len(MULTI_DOMAIN_24_ITEMS)
    res = {
        "model": label,
        "baseline_time_ms": round(base_time_ms, 1),
        "baseline_tokens": base_tokens,
        "baseline_accuracy": f"{base_correct}/{n} ({round(base_correct / n * 100, 1)}%)",
        "cascade_ambig_or_fail_time_ms": round(time_ambig_or_fail_ms, 1),
        "cascade_ambig_or_fail_tokens": tokens_ambig_or_fail,
        "cascade_ambig_or_fail_accuracy": f"{correct_ambig_or_fail}/{n} ({round(correct_ambig_or_fail / n * 100, 1)}%)",
        "cascade_ambig_only_time_ms": round(time_ambig_only_ms, 1),
        "cascade_ambig_only_tokens": tokens_ambig_only,
        "cascade_ambig_only_accuracy": f"{correct_ambig_only}/{n} ({round(correct_ambig_only / n * 100, 1)}%)",
        "speedup_ambig_only": round(base_time_ms / max(time_ambig_only_ms, 1.0), 1),
        "tokens_saved_ambig_only": base_tokens - tokens_ambig_only,
        "layer_resolution": {
            "l0_ast": f"{l0_count}/{n} ({round(l0_count / n * 100, 1)}%)",
            "l1_decisive": f"{l1_decisive_count}/{n} ({round(l1_decisive_count / n * 100, 1)}%)",
            "l2_escalated": f"{l2_escalated_count}/{n} ({round(l2_escalated_count / n * 100, 1)}%)",
            "l0_plus_l1_rate": f"{l0_count + l1_decisive_count}/{n} ({round((l0_count + l1_decisive_count) / n * 100, 1)}%)",
        },
    }
    print(f"  --> {label}: {json.dumps(res, indent=2)}", flush=True)
    return res


def eval_choice_and_score_gates(backend: OllamaBackend, model_name: str) -> dict[str, Any]:
    """Evaluates ChoiceGate (6 items) and ScoreGate (6 items) with N=5 latency trials."""
    print(f"\n=== [Suite 3: ChoiceGate & ScoreGate] Model: {model_name} ===", flush=True)
    # 1. ChoiceGate accuracy across 6 items
    choice_1p_correct = 0
    choice_2p_correct = 0
    for item in CHOICE_GATE_6_ITEMS:
        cg_1p = ChoiceGate(
            question=item["question"],
            options=item["options"],
            backend=backend,
            model=model_name,
            symmetric_debiasing=False,
        )
        cg_2p = ChoiceGate(
            question=item["question"],
            options=item["options"],
            backend=backend,
            model=model_name,
            symmetric_debiasing=True,
        )
        v1 = cg_1p.evaluate(item["candidate"])
        v2 = cg_2p.evaluate(item["candidate"])
        if v1.choice == item["expected_choice"]:
            choice_1p_correct += 1
        if v2.choice == item["expected_choice"]:
            choice_2p_correct += 1

    # N=5 warm latency trials on ChoiceGate (2-pass symmetric)
    cg_bench = ChoiceGate(
        question=CHOICE_GATE_6_ITEMS[0]["question"],
        options=CHOICE_GATE_6_ITEMS[0]["options"],
        backend=backend,
        model=model_name,
        symmetric_debiasing=True,
    )
    cg_bench.evaluate(CHOICE_GATE_6_ITEMS[0]["candidate"])
    cg_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        cg_bench.evaluate(CHOICE_GATE_6_ITEMS[0]["candidate"])
        cg_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. ScoreGate across 6 items (2 high, 2 medium, 2 low)
    scores_by_tier: dict[str, list[float]] = {"high": [], "medium": [], "low": []}
    in_band_count = 0
    for item in SCORE_GATE_6_ITEMS:
        sg = ScoreGate(
            rubric=item["rubric"],
            min_score=1,
            max_score=5,
            backend=backend,
            model=model_name,
            symmetric_debiasing=True,
        )
        v = sg.evaluate(item["candidate"])
        s = v.score if v.score is not None else 3.0
        scores_by_tier[item["tier"]].append(s)
        lo, hi = item["expected_band"]
        if lo <= s <= hi:
            in_band_count += 1

    # Check pairwise rank order across the 2 domains (code: high > med > low; postmortem: high > med > low)
    rank_pairs_correct = 0
    # Code domain: indices 0 (high), 2 (med), 4 (low)
    # Postmortem domain: indices 1 (high), 3 (med), 5 (low)
    h_mean = statistics.mean(scores_by_tier["high"])
    m_mean = statistics.mean(scores_by_tier["medium"])
    l_mean = statistics.mean(scores_by_tier["low"])
    if scores_by_tier["high"][0] > scores_by_tier["medium"][0] > scores_by_tier["low"][0]:
        rank_pairs_correct += 3
    else:
        if scores_by_tier["high"][0] > scores_by_tier["medium"][0]:
            rank_pairs_correct += 1
        if scores_by_tier["medium"][0] > scores_by_tier["low"][0]:
            rank_pairs_correct += 1
        if scores_by_tier["high"][0] > scores_by_tier["low"][0]:
            rank_pairs_correct += 1

    if scores_by_tier["high"][1] > scores_by_tier["medium"][1] > scores_by_tier["low"][1]:
        rank_pairs_correct += 3
    else:
        if scores_by_tier["high"][1] > scores_by_tier["medium"][1]:
            rank_pairs_correct += 1
        if scores_by_tier["medium"][1] > scores_by_tier["low"][1]:
            rank_pairs_correct += 1
        if scores_by_tier["high"][1] > scores_by_tier["low"][1]:
            rank_pairs_correct += 1

    # N=5 warm latency trials on ScoreGate (2-pass symmetric)
    sg_bench = ScoreGate(
        rubric=SCORE_GATE_6_ITEMS[0]["rubric"],
        min_score=1,
        max_score=5,
        backend=backend,
        model=model_name,
        symmetric_debiasing=True,
    )
    sg_bench.evaluate(SCORE_GATE_6_ITEMS[0]["candidate"])
    sg_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sg_bench.evaluate(SCORE_GATE_6_ITEMS[0]["candidate"])
        sg_times.append((time.perf_counter() - t0) * 1000.0)

    res = {
        "choice_1p_accuracy": f"{choice_1p_correct}/6 ({round(choice_1p_correct / 6 * 100, 1)}%)",
        "choice_2p_accuracy": f"{choice_2p_correct}/6 ({round(choice_2p_correct / 6 * 100, 1)}%)",
        "choice_2p_latency_ms": f"{statistics.mean(cg_times):.1f} ± {statistics.stdev(cg_times):.1f} ms",
        "score_high_mean": round(h_mean, 2),
        "score_medium_mean": round(m_mean, 2),
        "score_low_mean": round(l_mean, 2),
        "score_pairwise_rank_accuracy": f"{rank_pairs_correct}/6 ({round(rank_pairs_correct / 6 * 100, 1)}%)",
        "score_2p_latency_ms": f"{statistics.mean(sg_times):.1f} ± {statistics.stdev(sg_times):.1f} ms",
    }
    print(f"  --> Choice/Score for {model_name}: {res}", flush=True)
    return res


def eval_diverse_haystack(backend: OllamaBackend, model_name: str) -> list[dict[str, Any]]:
    """Evaluates needle-in-a-haystack across diverse 12-section handbook documents (200 to 16,000 tokens)."""
    print(f"\n=== [Suite 4: Diverse Multi-Section Haystack] Model: {model_name} ===", flush=True)
    tiers = [
        ("Short (~200 tok)", 150),
        ("Medium (~1,500 tok)", 1100),
        ("Long (~4,500 tok)", 3300),
        ("Very Long (~8,000 tok)", 5800),
        ("Massive (~16,000 tok)", 11000),
    ]
    gate = LogitGate(rubric=HAYSTACK_RUBRIC, backend=backend, model=model_name, preset="balanced")
    rows = []
    for label, words in tiers:
        doc_clean = build_diverse_haystack_document(words, NEEDLE_COMPLIANT)
        doc_viol = build_diverse_haystack_document(words, NEEDLE_VIOLATING)

        # Warm prefix with compliant doc, then measure violating doc
        v_clean = gate.evaluate(doc_clean)
        t0 = time.perf_counter()
        v_viol = gate.evaluate(doc_viol)
        warm_ms = (time.perf_counter() - t0) * 1000.0

        p_clean = v_clean.probability if v_clean.probability is not None else 0.5
        p_viol = v_viol.probability if v_viol.probability is not None else 0.5
        delta_p = p_clean - p_viol
        row = {
            "tier": label,
            "warm_latency_ms": round(warm_ms, 1),
            "p_compliant": round(p_clean, 4),
            "p_violating": round(p_viol, 4),
            "delta_p": round(delta_p, 4),
            "correct_both": (p_clean >= 0.85 and p_viol <= 0.20),
        }
        rows.append(row)
        print(f"  [{label}] P_clean={p_clean:.4f} P_viol={p_viol:.4f} ΔP={delta_p:.4f} Latency={warm_ms:.1f}ms", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LogitGate reproducible benchmarks.")
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmarks" / "latest_results.json"))
    args = parser.parse_args()

    gpu1_url = os.environ.get("LOGITGATE_BENCH_GPU1_URL", "http://localhost:11434")
    gpu0_url = os.environ.get("LOGITGATE_BENCH_GPU0_URL", "http://localhost:11434")
    all_urls = [gpu1_url, gpu0_url]

    out_path = Path(args.output)
    master: dict[str, Any] = {
        "suite_1_multi_domain_24": {},
        "suite_2_reasoning_depth_32": {},
        "suite_3_choice_and_score": {},
        "suite_4_diverse_haystack": {},
    }

    def _save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(master, f, indent=2)

    unload_all(all_urls)

    # 1. GPU 1 Models (4B, 14B, 14B-Coder)
    for model in ["gemma4:e4b", "qwen2.5:14b", "qwen2.5-coder:14b"]:
        b = OllamaBackend(base_url=gpu1_url, model=model, num_ctx=32768, timeout=180.0)
        try:
            master["suite_2_reasoning_depth_32"][model] = eval_reasoning_tiers(b, model)
            _save()
            master["suite_3_choice_and_score"][model] = eval_choice_and_score_gates(b, model)
            _save()
            master["suite_1_multi_domain_24"][f"{model}_gpu"] = eval_multi_domain_24(b, model, is_cpu=False)
            _save()
            if model in ("gemma4:e4b", "qwen2.5:14b"):
                master["suite_4_diverse_haystack"][model] = eval_diverse_haystack(b, model)
                _save()
        finally:
            b.close()
            unload_all(all_urls)

    # 2. GPU 0 Models (26B MoE, 31B Dense)
    for model in ["gemma4:26b", "gemma4:31b"]:
        b = OllamaBackend(base_url=gpu0_url, model=model, num_ctx=32768, timeout=180.0)
        try:
            master["suite_2_reasoning_depth_32"][model] = eval_reasoning_tiers(b, model)
            _save()
            master["suite_3_choice_and_score"][model] = eval_choice_and_score_gates(b, model)
            _save()
            master["suite_4_diverse_haystack"][model] = eval_diverse_haystack(b, model)
            _save()
        finally:
            b.close()
            unload_all(all_urls)

    print(f"\nALL BENCHMARKS COMPLETE! Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
