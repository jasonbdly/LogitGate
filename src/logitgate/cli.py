from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from logitgate.cascade import Cascade
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import LogitGate


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logitgate",
        description="LogitGate CLI: Calibrated first-token logit evaluation and cascaded routing.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate candidate artifact against a rubric",
    )
    eval_parser.add_argument(
        "--rubric",
        required=True,
        help="Evaluation rubric or question to verify against candidate",
    )
    eval_parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Candidate text/artifact to evaluate",
    )
    eval_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file containing candidate text to evaluate",
    )
    eval_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass probability threshold (defaults to LOGITGATE_PASS_THRESHOLD or 0.85)",
    )
    eval_parser.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Fail probability threshold (defaults to LOGITGATE_FAIL_THRESHOLD or 0.20)",
    )
    eval_parser.add_argument(
        "--ast-syntax",
        action="store_true",
        help="Enable Layer 0 AST syntax validation before logit evaluation",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Output verdict and traces in JSON format",
    )

    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Run micro-calibration dataset to fit optimal temperature or conformal cascade thresholds",
    )
    cal_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    cal_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    cal_parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Optional custom path to calibration cache JSON file",
    )
    cal_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to holdout dataset (JSON, JSONL, or Markdown) for Split Conformal Cascade calibration",
    )
    cal_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target risk bound alpha for selective classification (default: 0.05)",
    )
    cal_parser.add_argument(
        "--json",
        action="store_true",
        help="Output calibration results in JSON format",
    )

    explain_parser = subparsers.add_parser(
        "explain",
        help="Replay frozen failure snapshot and generate full Layer-2 diagnostic explanation",
    )
    explain_parser.add_argument(
        "snapshot_path",
        nargs="?",
        default=".logitgate/failures",
        help="Path to snapshot JSON file or directory containing latest.json (default: .logitgate/failures)",
    )
    explain_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Generative backend specifier (e.g. stub://..., ollama://...)",
    )
    explain_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    explain_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for generative explanation",
    )
    explain_parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Chain-of-Thought reasoning tokens in generative explanation",
    )
    explain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output enriched verdict and traces in JSON format",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        from pathlib import Path

        from logitgate.backends.base import resolve_logit_backend

        backend = resolve_logit_backend(args.backend, model=args.model)
        effective_model = args.model or getattr(backend, "model", "default")

        if args.dataset:
            import json
            from dataclasses import asdict

            from logitgate.calibration.conformal import (
                calibrate_cascade_thresholds,
                load_calibration_dataset,
            )

            samples = load_calibration_dataset(args.dataset)
            if not samples:
                sys.stderr.write(f"Error: No valid samples found in dataset '{args.dataset}'\n")
                return 1

            calibrated_probs: list[float] = []
            ground_truth_labels: list[bool] = []

            for sample_rubric, sample_candidate, sample_y in samples:
                gate = LogitGate(rubric=sample_rubric, backend=backend, model=args.model)
                v = gate.evaluate(sample_candidate)
                prob = v.probability if v.probability is not None else (1.0 if v.passed else 0.0)
                calibrated_probs.append(prob)
                ground_truth_labels.append(bool(sample_y))

            thresholds = calibrate_cascade_thresholds(
                calibrated_probs=calibrated_probs,
                ground_truth_labels=ground_truth_labels,
                target_risk_alpha=args.alpha,
            )

            if args.json:
                print(json.dumps(asdict(thresholds), indent=2))
            else:
                print("=" * 60)
                print("LOGITGATE SPLIT CONFORMAL CASCADE CALIBRATION")
                print("=" * 60)
                print(f"Dataset:                   {args.dataset}")
                print(f"Calibration Samples:       {thresholds.calibration_set_size}")
                print(f"Target Risk (Alpha):       {thresholds.target_risk_alpha:.4f}")
                print("-" * 60)
                print(f"Layer 1 Tau Reject:        {thresholds.tau_reject:.4f}")
                print(f"Layer 1 Tau Accept:        {thresholds.tau_accept:.4f}")
                print(f"Layer 2 Escalation Rate:   {thresholds.empirical_escalation_rate:.2%}")
                print("-" * 60)
                print(f"Guaranteed Layer 1 Error: <= {thresholds.target_risk_alpha:.2%}")
                print("=" * 60)
            return 0

        from logitgate.debiasing.permutation import auto_calibrate_backend

        cache_path = Path(args.cache_path) if args.cache_path else None

        print(f"Calibrating model '{effective_model}' on backend {backend.__class__.__name__}...")
        result = auto_calibrate_backend(
            backend,
            model_key=effective_model,
            save_cache=True,
            cache_path=cache_path,
        )
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(f"Optimal Temperature: {result.optimal_temperature:.4f}")
            print(f"Brier Score: {result.brier_score_before:.6f} -> {result.brier_score_after:.6f}")
            print(f"ECE: {result.ece_before:.6f} -> {result.ece_after:.6f}")
            print("Calibration successfully saved to cache.")
        return 0

    if args.command == "explain":
        from logitgate.snapshot import VerdictSnapshot
        try:
            snapshot = VerdictSnapshot.load(args.snapshot_path)
        except (OSError, ValueError, KeyError) as e:
            sys.stderr.write(f"Error loading snapshot from '{args.snapshot_path}': {e}\n")
            return 1

        think = not args.no_think
        verdict = snapshot.explain(
            backend=args.backend,
            model=args.model,
            max_tokens=args.max_tokens,
            think=think,
        )

        if args.json:
            print(verdict.model_dump_json(indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE FAILURE REPLAY & EXPLANATION")
            print("=" * 60)
            print(f"Snapshot ID:     {snapshot.snapshot_id}")
            if snapshot.test_name:
                print(f"Test Name:       {snapshot.test_name}")
            print(f"Timestamp:       {snapshot.timestamp}")
            print(f"Rubric:          {snapshot.rubric}")
            orig_prob = snapshot.verdict.probability
            prob_str = f"{orig_prob:.4f}" if orig_prob is not None else "N/A"
            print(f"Original Prob:   {prob_str}")
            if snapshot.positional_bias_magnitude is not None:
                print(f"Positional Bias: {snapshot.positional_bias_magnitude:.4f} log-odds")
            if snapshot.log_odds_pass is not None:
                print(f"Unbiased Log-Odds: {snapshot.log_odds_pass:.4f}")
            if snapshot.shannon_entropy is not None:
                print(f"Verdict Entropy: {snapshot.shannon_entropy:.4f} bits")
            if snapshot.loops_executed is not None:
                print(f"Recurrent Loops: {snapshot.loops_executed}")
            if snapshot.failure_code:
                print(f"Failure Code:    {snapshot.failure_code}")
            print(f"Triggered Layer: {snapshot.verdict.triggered_layer}")
            print("-" * 60)
            print(f"Replay Verdict:  {verdict.status.value.upper()}")
            print(f"Replay Latency:  {verdict.total_latency_ms:.2f}ms")
            print(f"Tokens Spent:    {verdict.total_tokens_generated}")
            print("-" * 60)
            print("Diagnostic Critique:")
            print(verdict.reason or "(No reason provided)")
            print("=" * 60)
        return 0

    if args.command != "eval":
        parser.print_help()
        return 1

    candidate: str | None = None
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                candidate = f.read()
        except OSError as e:
            sys.stderr.write(f"Error reading file '{args.file}': {e}\n")
            return 1
    elif args.candidate is not None:
        candidate = args.candidate
    else:
        if not sys.stdin.isatty():
            candidate = sys.stdin.read()
        else:
            sys.stderr.write("Error: Either --candidate or --file must be specified.\n")
            return 1

    logit_gate = LogitGate(
        rubric=args.rubric,
        backend=args.backend,
        model=args.model,
        pass_threshold=args.threshold,
        fail_threshold=args.fail_threshold,
    )

    evaluator: Cascade | LogitGate
    if args.ast_syntax:
        evaluator = Cascade([
            ASTGate(syntax=True),
            logit_gate,
        ])
    else:
        evaluator = logit_gate

    verdict = evaluator.evaluate(candidate)

    if args.json:
        print(verdict.model_dump_json(indent=2))
    else:
        verdict_str = "PASSED" if verdict.passed else "FAILED"
        print(f"Verdict: {verdict_str} (status: {verdict.status.value})")
        if verdict.probability is not None:
            print(f"Confidence: {verdict.probability:.4f}")
        print(f"Triggered Layer: {verdict.triggered_layer}")
        print(f"Total Latency: {verdict.total_latency_ms:.2f}ms")
        if verdict.reason:
            print(f"Reason: {verdict.reason}")

    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
