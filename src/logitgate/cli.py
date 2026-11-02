from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from logitgate.cascade import Cascade
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import LogitGate


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logitgate",
        description="LogitGate CLI: Sub-50ms LLM evaluation and decision engine.",
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
        default="stub://default",
        help="Backend specifier (e.g. stub://default, ollama://..., llamacpp://...)",
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
        default=0.85,
        help="Pass probability threshold (default: 0.85)",
    )
    eval_parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.20,
        help="Fail probability threshold (default: 0.20)",
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
        help="Run micro-calibration dataset to fit and cache optimal logit temperature",
    )
    cal_parser.add_argument(
        "--backend",
        type=str,
        default="stub://default",
        help="Backend specifier (e.g. stub://default, ollama://..., llamacpp://...)",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        from pathlib import Path
        from logitgate.backends.base import resolve_logit_backend
        from logitgate.debiasing.permutation import auto_calibrate_backend

        backend = resolve_logit_backend(args.backend, model=args.model)
        cache_path = Path(args.cache_path) if args.cache_path else None
        effective_model = args.model or getattr(backend, "model", "default")

        print(f"Calibrating model '{effective_model}' on backend {backend.__class__.__name__}...")
        result = auto_calibrate_backend(
            backend,
            model_key=effective_model,
            save_cache=True,
            cache_path=cache_path,
        )
        print(f"Optimal Temperature: {result.optimal_temperature:.4f}")
        print(f"Brier Score: {result.brier_score_before:.6f} -> {result.brier_score_after:.6f}")
        print(f"ECE: {result.ece_before:.6f} -> {result.ece_after:.6f}")
        print("Calibration successfully saved to cache.")
        return 0

    if args.command == "explain":
        from logitgate.snapshot import VerdictSnapshot
        try:
            snapshot = VerdictSnapshot.load(args.snapshot_path)
        except Exception as e:
            sys.stderr.write(f"Error loading snapshot from '{args.snapshot_path}': {e}\n")
            return 1

        think = False if args.no_think else True
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

    candidate: Optional[str] = None
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
