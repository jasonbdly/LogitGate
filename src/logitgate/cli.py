from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TextIO

from logitgate.cascade import Cascade
from logitgate.exceptions import BackendError
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict

if TYPE_CHECKING:
    from logitgate.snapshot import VerdictSnapshot

EXIT_OK = 0
EXIT_FAILED = 1  # verdict failed, or bad input
EXIT_BACKEND_ERROR = 2  # could not reach / use the backend


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
        help="Inspect a failure snapshot and (optionally) replay it through a Layer-2 judge",
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
        "--offline",
        action="store_true",
        help="Only render recorded diagnostics; do not call any model",
    )
    explain_parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize output (default: auto; honors NO_COLOR / FORCE_COLOR)",
    )
    explain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output enriched verdict and traces in JSON format",
    )

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Measure latency, generated tokens, and accuracy: generative judge vs LogitGate vs cascade",
        description=(
            "Runs the same cases through a generative judge, a LogitGate, and a LogitGate >> "
            "JudgeGate cascade on YOUR backend, and reports p50/p95 latency, generated tokens, "
            "LLM calls, and accuracy. Use --json to archive results with full environment metadata."
        ),
    )
    bench_parser.add_argument("--backend", type=str, default=None, help="Backend specifier, e.g. ollama://localhost:11434")
    bench_parser.add_argument("--model", type=str, default=None, help="Model name for the backend")
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSON/JSONL/Markdown with rubric, candidate, label (default: built-in 16-case set)",
    )
    bench_parser.add_argument(
        "--modes",
        type=str,
        default="judge,logit,cascade",
        help="Comma-separated subset of judge,logit,cascade (default: all)",
    )
    bench_parser.add_argument("--repeat", type=int, default=1, help="Timed passes over the dataset per mode (default: 1)")
    bench_parser.add_argument("--warmup", type=int, default=1, help="Untimed calls per mode before timing (default: 1)")
    bench_parser.add_argument("--limit", type=int, default=None, help="Use only the first N cases")
    bench_parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=None,
        help="Cap on judge generation (default: backend default)",
    )
    bench_parser.add_argument(
        "--judge-think",
        action="store_true",
        help="Let the judge emit reasoning tokens (Ollama). Off by default: a fast judge is the fair baseline.",
    )
    bench_parser.add_argument(
        "--cascade-trigger",
        choices=("on_ambiguous_only", "on_ambiguous_or_fail", "always"),
        default="on_ambiguous_only",
        help="JudgeGate escalation policy inside the cascade (default: on_ambiguous_only)",
    )
    bench_parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    bench_parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress on stderr")

    return parser


# ---------------------------------------------------------------------------------------------
# Terminal rendering for `logitgate explain`
# ---------------------------------------------------------------------------------------------


_ANSI_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}


class _Style:
    """ANSI styling that degrades to plain text (pipes, CI logs, NO_COLOR) and ASCII glyphs."""

    def __init__(self, color: bool, stream: TextIO) -> None:
        self.color = color
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            "✓✗◐─═".encode(encoding)
            self.unicode = True
        except (UnicodeEncodeError, LookupError):
            self.unicode = False

    def __call__(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        codes = ";".join(_ANSI_CODES[s] for s in styles)
        return f"\033[{codes}m{text}\033[0m"

    def rule(self, heavy: bool = False, width: int = 72) -> str:
        char = ("═" if heavy else "─") if self.unicode else ("=" if heavy else "-")
        return self(char * width, "dim")

    def status(self, status: DecisionStatus, upper: bool = False) -> str:
        glyphs = {"passed": "✓", "failed": "✗", "ambiguous": "◐"} if self.unicode else {"passed": "+", "failed": "x", "ambiguous": "?"}
        colors = {"passed": "green", "failed": "red", "ambiguous": "yellow"}
        label = status.value.upper() if upper else status.value
        return self(f"{glyphs[status.value]} {label}", colors[status.value])


def _use_color(mode: str, stream: TextIO) -> bool:
    if mode == "always":
        return True
    if mode == "never" or "NO_COLOR" in os.environ:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _entropy_hint(h: float) -> tuple[str, str]:
    # Binary entropy is 0.61 bits at P=0.85 and 0.72 bits at P=0.20, the default thresholds.
    if h >= 0.9:
        return "near coin-flip: the model is guessing. Split compound rubrics into atomic ones, or use a stronger model", "red"
    if h >= 0.6:
        return "uncertain: inside or near the escalation band", "yellow"
    return "confident", "green"


def _bias_hint(bias: float) -> tuple[str, str]:
    # bias = 0.5 * |L_forward - L_reverse|, an estimate of the model's constant preference for
    # one label letter, in log-odds. Symmetric debiasing cancels that constant term exactly.
    if bias < 0.5:
        return "forward and reverse passes agree", "green"
    if bias < 1.5:
        return f"moderate label-order prior (odds x{math.exp(bias):.1f}); cancelled by symmetric debiasing", "yellow"
    message = (
        f"strong label-order prior (odds x{math.exp(bias):.1f}). Debiasing cancels it, but the raw read is "
        "unreliable for this model/rubric pair; try a stronger model or rephrase the rubric"
    )
    return message, "red"


def _decisive_logit_trace(traces: Sequence[LayerTrace]) -> LayerTrace | None:
    for trace in reversed(traces):
        if trace.log_odds_pass is not None or trace.calibrated_prob is not None:
            return trace
    return None


def render_snapshot_report(
    snapshot: VerdictSnapshot,
    style: _Style,
    replay: Verdict | None = None,
    replay_note: str | None = None,
) -> str:
    """Builds the human-readable `logitgate explain` report.

    Answers, in order: what failed, how far from the thresholds it was, whether the model's
    read can be trusted (entropy, label-order bias), and what the Layer-2 judge says.
    """
    lab = 17  # label column width
    out: list[str] = []

    def kv(label: str, value: str, hint: str | None = None, hint_color: str = "dim") -> None:
        line = f"{label:<{lab}}{value}" if len(label) < lab else f"{label} {value}"
        if hint:
            line += "  " + style(f"({hint})", hint_color)
        out.append(line)

    verdict = snapshot.verdict
    out.append(style("LOGITGATE FAILURE REPLAY & EXPLANATION", "bold"))
    out.append(style.rule(heavy=True))
    kv("Snapshot ID:", snapshot.snapshot_id)
    if snapshot.test_name:
        kv("Test Name:", snapshot.test_name)
    kv("Timestamp:", snapshot.timestamp)
    rubric_lines = textwrap.wrap(snapshot.rubric, width=72 - lab) or [""]
    kv("Rubric:", rubric_lines[0])
    out.extend(" " * lab + line for line in rubric_lines[1:])
    candidate = str(snapshot.candidate)
    width = 72 - lab
    wrapped = [w for line in candidate.strip().splitlines() for w in (textwrap.wrap(line, width) or [""])]
    kv("Candidate:", style(wrapped[0] if wrapped else "(empty)", "cyan"), f"{len(candidate)} chars")
    out.extend(" " * lab + style(line, "cyan") for line in wrapped[1:6])
    if len(wrapped) > 6:
        out.append(" " * lab + style(f"... ({len(wrapped) - 6} more lines)", "dim"))

    # Per-layer trace table
    out.append(style.rule())
    out.append(style("Layer trace", "bold"))
    header = f"  {'#':<3}{'layer':<14}{'status':<13}{'latency':>10}{'P(pass)':>9}{'log-odds':>10}{'entropy':>9}{'bias':>7}{'tokens':>8}"
    out.append(style(header, "dim"))
    for idx, t in enumerate(verdict.traces, start=1):
        # Pad the status *before* styling so ANSI codes don't break column alignment.
        status_cell = style.status(t.status) + " " * max(0, 13 - len(t.status.value) - 2)
        out.append(
            f"  {idx:<3}{t.layer_name[:13]:<14}{status_cell}{t.latency_ms:>8.2f}ms"
            f"{_fmt(t.calibrated_prob, '.4f'):>9}{_fmt(t.log_odds_pass, '+.3f'):>10}"
            f"{_fmt(t.shannon_entropy, '.3f'):>9}{_fmt(t.positional_bias_magnitude, '.2f'):>7}{t.tokens_generated:>8}"
        )

    # Diagnostics
    out.append(style.rule())
    out.append(style("Diagnostics", "bold"))
    prob = verdict.probability
    kv("Original Prob:", f"{prob:.4f}" if prob is not None else "N/A", f"{verdict.status.value.upper()} at {verdict.triggered_layer}")

    log_odds = snapshot.log_odds_pass
    pass_th = snapshot.gate_config.get("pass_threshold")
    fail_th = snapshot.gate_config.get("fail_threshold")
    if log_odds is not None and isinstance(pass_th, (int, float)) and isinstance(fail_th, (int, float)):
        lo_pass, lo_fail = _logit(pass_th), _logit(fail_th)
        if log_odds >= lo_pass:
            where = f"{log_odds - lo_pass:.2f} above the pass line"
        elif log_odds <= lo_fail:
            where = f"{lo_fail - log_odds:.2f} below the fail line: decisive"
        else:
            where = f"in the escalation band, {lo_pass - log_odds:.2f} short of passing"
        kv("Decision Margin:", f"{log_odds:+.3f} vs pass {lo_pass:+.3f} (P={pass_th}) / fail {lo_fail:+.3f} (P={fail_th})", where)

    logit_trace = _decisive_logit_trace(verdict.traces)
    if logit_trace is not None and logit_trace.forward_prob is not None:
        rev = logit_trace.reverse_prob
        value = f"P={logit_trace.forward_prob:.4f} (A=pass)" + (f" / P={rev:.4f} (B=pass)" if rev is not None else "")
        kv("Forward/Reverse:", value, None if rev is not None else "single pass: label-order prior not cancelled")

    bias = snapshot.positional_bias_magnitude
    if bias is not None:
        kv("Positional Bias:", f"{bias:.4f} log-odds", *_bias_hint(bias))
    if log_odds is not None:
        out.append(f"Unbiased Log-Odds: {log_odds:.4f}")
    entropy = snapshot.shannon_entropy
    if entropy is not None:
        kv("Verdict Entropy:", f"{entropy:.4f} bits", *_entropy_hint(entropy))
    if snapshot.loops_executed is not None:
        kv("Recurrent Loops:", str(snapshot.loops_executed))
    if snapshot.failure_code:
        kv("Failure Code:", style(snapshot.failure_code, "red"))
    failing = next((t for t in reversed(verdict.traces) if t.status != DecisionStatus.PASSED), None)
    if failing is not None and failing.diagnostic_report and failing.tokens_generated == 0:
        kv("Detail:", failing.diagnostic_report[:200])
    kv("Triggered Layer:", verdict.triggered_layer)

    # Replay
    out.append(style.rule())
    if replay is None:
        recorded = next((t for t in reversed(verdict.traces) if t.tokens_generated > 0 and t.diagnostic_report), None)
        if recorded is not None:
            out.append(style(f"Recorded critique ({recorded.layer_name}, {recorded.tokens_generated} tokens):", "bold"))
            out.extend("  " + line for line in (recorded.diagnostic_report or "").strip().splitlines())
            out.append(style.rule())
        out.append(style(replay_note or "Replay skipped.", "dim"))
        out.append(style.rule(heavy=True))
        return "\n".join(out)

    kv("Replay Verdict:", style.status(replay.status, upper=True))
    kv("Replay Latency:", f"{replay.total_latency_ms:.2f}ms")
    kv("Tokens Spent:", str(replay.total_tokens_generated))
    out.append(style.rule())
    out.append(style("Diagnostic Critique:", "bold"))
    critique = (replay.reason or "(No reason provided)").strip()
    out.extend("  " + line for line in critique.splitlines())
    out.append(style.rule(heavy=True))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------


def _cmd_calibrate(args: argparse.Namespace) -> int:
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
            return EXIT_FAILED

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
            if thresholds.guarantee_holds:
                print(f"Guaranteed Layer 1 Error: <= {thresholds.target_risk_alpha:.2%} "
                      "(marginal; assumes production inputs resemble this dataset)")
            else:
                print(
                    f"Guaranteed Layer 1 Error: NONE. Need >= {thresholds.min_samples_per_class} positives "
                    f"and negatives (have {thresholds.num_positive} / {thresholds.num_negative}); "
                    "thresholds are best-effort."
                )
            print(
                f"Use: LogitGate(..., pass_threshold={thresholds.tau_accept}, "
                f"fail_threshold={thresholds.tau_reject})"
            )
            print("=" * 60)
        return EXIT_OK

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
    return EXIT_OK


def _cmd_explain(args: argparse.Namespace) -> int:
    from logitgate.snapshot import VerdictSnapshot

    try:
        snapshot = VerdictSnapshot.load(args.snapshot_path)
    except (OSError, ValueError, KeyError) as e:
        sys.stderr.write(f"Error loading snapshot from '{args.snapshot_path}': {e}\n")
        return EXIT_FAILED

    style = _Style(_use_color(args.color, sys.stdout), sys.stdout)

    if args.offline:
        if args.json:
            print(snapshot.verdict.model_dump_json(indent=2))
        else:
            print(render_snapshot_report(snapshot, style, replay_note="Replay skipped (--offline). Re-run without --offline for a Layer-2 critique."))
        return EXIT_OK

    try:
        verdict = snapshot.explain(
            backend=args.backend,
            model=args.model,
            max_tokens=args.max_tokens,
            think=not args.no_think,
        )
    except BackendError as exc:
        if not args.json:
            print(render_snapshot_report(snapshot, style, replay_note="Replay failed; see error below."))
        sys.stderr.write(f"error: {exc}\nTip: `logitgate explain --offline {args.snapshot_path}` shows diagnostics without a model.\n")
        return EXIT_BACKEND_ERROR

    if args.json:
        print(verdict.model_dump_json(indent=2))
    else:
        print(render_snapshot_report(snapshot, style, replay=verdict))
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from logitgate.benchmark import format_report, load_cases, run_benchmark, write_json

    try:
        cases, label = load_cases(args.dataset)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error loading dataset '{args.dataset}': {e}\n")
        return EXIT_FAILED
    if args.limit is not None:
        cases = cases[: args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    def progress(line: str) -> None:
        sys.stderr.write(line + "\n")

    try:
        report = run_benchmark(
            cases,
            backend=args.backend,
            model=args.model,
            modes=modes,
            repeat=args.repeat,
            warmup=args.warmup,
            judge_max_tokens=args.judge_max_tokens,
            judge_think=args.judge_think,
            cascade_trigger=args.cascade_trigger,
            dataset_label=label,
            progress=None if (args.quiet or args.json) else progress,
        )
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return EXIT_FAILED

    if args.json:
        write_json(report)
    else:
        print(format_report(report))
    return EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    candidate: str | None = None
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                candidate = f.read()
        except OSError as e:
            sys.stderr.write(f"Error reading file '{args.file}': {e}\n")
            return EXIT_FAILED
    elif args.candidate is not None:
        candidate = args.candidate
    else:
        if not sys.stdin.isatty():
            candidate = sys.stdin.read()
        else:
            sys.stderr.write("Error: Either --candidate or --file must be specified.\n")
            return EXIT_FAILED

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

    return EXIT_OK if verdict.passed else EXIT_FAILED


_COMMANDS: dict[str, Any] = {
    "eval": _cmd_eval,
    "calibrate": _cmd_calibrate,
    "explain": _cmd_explain,
    "benchmark": _cmd_benchmark,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 = passed/ok, 1 = verdict failed or bad input, 2 = backend error."""
    parser = create_parser()
    args = parser.parse_args(argv)

    command = _COMMANDS.get(args.command)
    if command is None:
        parser.print_help()
        return EXIT_FAILED
    try:
        result: int = command(args)
    except BackendError as exc:
        # Backends already phrase these as "what went wrong + what to do"; a traceback adds nothing.
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_BACKEND_ERROR
    return result


if __name__ == "__main__":
    sys.exit(main())
