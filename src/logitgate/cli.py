from __future__ import annotations

import argparse
import math
import os
import re
import sys
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TextIO

from logitgate.cascade import Cascade
from logitgate.exceptions import BackendError, CalibrationError, ConfigurationError
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
        "--profile",
        type=str,
        default=None,
        help="Path to a CalibrationProfile JSON file",
    )
    eval_parser.add_argument(
        "--ast-syntax",
        action="store_true",
        help="Enable Layer 0 AST syntax validation before logit evaluation",
    )
    eval_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Strip <think> blocks and unwrap markdown code fences before Layer 0 validation",
    )
    eval_parser.add_argument(
        "--telemetry-out",
        type=str,
        default=None,
        help="Optional JSONL path to append structured verdict telemetry",
    )
    eval_parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional SQLite VerdictCache path (or 'memory' for in-process LRU)",
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
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save a portable CalibrationProfile JSON file",
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
        "--judge-backend",
        type=str,
        default=None,
        help="Optional separate backend specifier for JudgeGate (heterogeneous cascade)",
    )
    bench_parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional separate model name for JudgeGate (defaults to --model)",
    )
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
        choices=("on_ambiguous_only", "on_ambiguous_or_fail", "adjudicate", "always"),
        default="on_ambiguous_only",
        help="JudgeGate escalation policy inside the cascade (default: on_ambiguous_only)",
    )
    bench_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a CalibrationProfile JSON file for LogitGate / cascade modes",
    )
    bench_parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional SQLite VerdictCache path (or 'memory' for in-process LRU across repeats)",
    )
    bench_parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit dataset for length confounders, lexical shortcuts, conformal power, and split leakage without calling any model",
    )
    bench_parser.add_argument(
        "--reference-dataset",
        type=str,
        default=None,
        help="Optional reference (calibration/training) dataset path to check for cross-split leakage with --audit-only",
    )
    bench_parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="With --audit-only, exit with code 1 if any dataset confounder warnings or cross-split leakage are detected",
    )
    bench_parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    bench_parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress on stderr")

    rank_parser = subparsers.add_parser(
        "rank",
        help="Rank N >= 2 candidate artifacts and select the winner without token generation",
    )
    rank_parser.add_argument(
        "--rubric",
        required=True,
        help="Evaluation rubric or preference criterion for comparing candidates",
    )
    rank_parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate text/artifact to include (repeatable)",
    )
    rank_parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Path to candidate text file to include (repeatable)",
    )
    rank_parser.add_argument(
        "--candidates-json",
        type=str,
        default=None,
        help="Path to a JSON list or key->candidate object of candidates",
    )
    rank_parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional shared task prompt or reference context",
    )
    rank_parser.add_argument(
        "--mode",
        choices=("auto", "pairwise", "listwise", "round_robin", "tournament"),
        default="auto",
        help="Ranking strategy (default: auto)",
    )
    rank_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional selective top-1 probability cutoff below which RankGate abstains",
    )
    rank_parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Optional top-2 probability margin cutoff below which RankGate abstains",
    )
    rank_parser.add_argument(
        "--max-positional-bias",
        type=float,
        default=None,
        help="Optional maximum allowed positional swap bias before abstaining",
    )
    rank_parser.add_argument(
        "--ast-syntax",
        action="store_true",
        help="Pre-filter candidates with ASTGate(syntax=True) before ranking",
    )
    rank_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Strip <think> blocks and unwrap markdown fences during AST pre-filtering",
    )
    rank_parser.add_argument(
        "--verify-winner",
        action="store_true",
        help="Verify the winning candidate against --rubric with a post-ranking LogitGate",
    )
    rank_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a CalibrationProfile JSON file",
    )
    rank_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    rank_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    rank_parser.add_argument(
        "--telemetry-out",
        type=str,
        default=None,
        help="Optional JSONL path to append structured verdict telemetry",
    )
    rank_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BestOfNResult, probabilities, and traces in JSON format",
    )

    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Summarize production telemetry, export SFT/DPO datasets, or zero-call recalibrate profiles",
    )
    tel_sub = telemetry_parser.add_subparsers(dest="telemetry_command", help="Telemetry subcommands")

    tel_summary = tel_sub.add_parser(
        "summary",
        help="Compute pass/escalation/discrepancy/latency summary from a telemetry JSONL file",
    )
    tel_summary.add_argument("input_path", help="Path to telemetry JSONL file")
    tel_summary.add_argument("--json", action="store_true", help="Output summary as JSON")

    tel_export = tel_sub.add_parser(
        "export",
        help="Export teacher-adjudicated escalations to SFT, DPO, or calibration JSONL",
    )
    tel_export.add_argument("input_path", help="Path to distillation or telemetry JSONL file")
    tel_export.add_argument(
        "--output", "-o", required=True, help="Destination JSONL dataset path"
    )
    tel_export.add_argument(
        "--format",
        choices=("sft_jsonl", "calibration_jsonl", "dpo_jsonl"),
        default="sft_jsonl",
        help="Export dataset format (default: sft_jsonl)",
    )
    tel_export.add_argument(
        "--no-symmetric-swaps",
        action="store_true",
        help="Emit only forward A=Pass/B=Fail prompts instead of symmetric A/B + B/A pairs",
    )
    tel_export.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate (rubric, candidate) pairs instead of deduplicating by SHA-256",
    )
    tel_export.add_argument(
        "--discrepancies-only",
        action="store_true",
        help="Export only student/teacher disagreements, omitting concordant ambiguous resolutions",
    )
    tel_export.add_argument("--json", action="store_true", help="Output export metadata as JSON")

    tel_recal = tel_sub.add_parser(
        "recalibrate",
        help="Fit a CalibrationProfile from logged student/teacher escalations with 0 LLM calls",
    )
    tel_recal.add_argument("input_path", help="Path to distillation or telemetry JSONL file")
    tel_recal.add_argument(
        "--output", "-o", type=str, default=None, help="Optional path to save CalibrationProfile JSON"
    )
    tel_recal.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Target selective risk bound alpha (default: 0.10)",
    )
    tel_recal.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional student model name override for the profile",
    )
    tel_recal.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate (rubric, candidate) escalations during fitting",
    )
    tel_recal.add_argument("--json", action="store_true", help="Output CalibrationProfile as JSON")

    tel_drift = tel_sub.add_parser(
        "drift",
        help="Detect covariate, calibration, and teacher-disagreement drift between two telemetry JSONL logs",
    )
    tel_drift.add_argument("input_path", type=str, help="Path to current evaluation telemetry JSONL file")
    tel_drift.add_argument(
        "--reference",
        "-r",
        required=True,
        type=str,
        help="Path to baseline reference telemetry JSONL file",
    )
    tel_drift.add_argument(
        "--psi-moderate",
        type=float,
        default=0.10,
        help="Moderate Population Stability Index (PSI) threshold (default: 0.10)",
    )
    tel_drift.add_argument(
        "--psi-severe",
        type=float,
        default=0.25,
        help="Severe Population Stability Index (PSI) threshold (default: 0.25)",
    )
    tel_drift.add_argument(
        "--ks-alpha",
        type=float,
        default=0.01,
        help="Two-sample Kolmogorov-Smirnov p-value threshold (default: 0.01)",
    )
    tel_drift.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 when drift_detected is True (for CI/monitoring gates)",
    )
    tel_drift.add_argument("--json", action="store_true", help="Output DriftReport as JSON")

    stream_parser = subparsers.add_parser(
        "stream",
        help="Verify a token/text stream mid-generation with StreamGuard early abort or buffered delivery",
    )
    stream_parser.add_argument(
        "--rubric",
        required=True,
        help="Evaluation rubric verified periodically against the growing stream prefix",
    )
    stream_parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Candidate text to simulate streaming in chunks",
    )
    stream_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file to stream in chunks (falls back to stdin when piped)",
    )
    stream_parser.add_argument(
        "--chunk-size",
        type=int,
        default=48,
        help="Simulated chunk size in characters (default: 48)",
    )
    stream_parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Minimum accumulated characters before first mid-stream check (default: 80)",
    )
    stream_parser.add_argument(
        "--stride-chars",
        type=int,
        default=160,
        help="Character stride between consecutive mid-stream checks (default: 160)",
    )
    stream_parser.add_argument(
        "--delivery",
        choices=("optimistic", "buffered"),
        default="optimistic",
        help="Chunk delivery strategy: 'optimistic' (zero-buffer) or 'buffered' (zero-leakage)",
    )
    stream_parser.add_argument(
        "--consecutive-failures",
        type=int,
        default=1,
        help="Consecutive failing checkpoints required before aborting mid-stream (default: 1)",
    )
    stream_parser.add_argument(
        "--no-clause-boundary",
        action="store_true",
        help="Disable clause/sentence boundary alignment for mid-stream checks",
    )
    stream_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass probability threshold",
    )
    stream_parser.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Fail probability threshold",
    )
    stream_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional CalibrationProfile JSON path",
    )
    stream_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    stream_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    stream_parser.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        help="Optional Tier-2 JudgeGate backend specifier for cascaded final verification",
    )
    stream_parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional Tier-2 JudgeGate model name",
    )
    stream_parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional SQLite VerdictCache path (or 'memory' for in-process LRU)",
    )
    stream_parser.add_argument(
        "--telemetry-out",
        type=str,
        default=None,
        help="Optional JSONL path to append structured stream telemetry",
    )
    stream_parser.add_argument(
        "--quiet-stream",
        action="store_true",
        help="Suppress printing yielded text before summary output",
    )
    stream_parser.add_argument(
        "--json",
        action="store_true",
        help="Output StreamGuardSummary and yielded text in JSON format",
    )

    return parser


# ---------------------------------------------------------------------------------------------
# Terminal rendering for `logitgate explain`
# ---------------------------------------------------------------------------------------------


_ANSI_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}
_TERMINAL_ESCAPE_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[@-Z\\-_]"
)
_BIDI_CONTROLS = frozenset(
    "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)


def sanitize_terminal_text(text: str) -> str:
    """Strips ANSI escape sequences, C0/C1 controls, and BiDi overrides from untrusted text."""
    stripped = _TERMINAL_ESCAPE_RE.sub("", text)
    return "".join(
        ch
        for ch in stripped
        if ch not in _BIDI_CONTROLS
        and (ch in ("\n", "\t") or (ord(ch) >= 32 and not (127 <= ord(ch) <= 159)))
    )


def _sanitize_terminal_text(text: str) -> str:
    return sanitize_terminal_text(text)


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
    non_passing = [
        t for t in traces
        if t.status != DecisionStatus.PASSED
        and (t.log_odds_pass is not None or t.calibrated_prob is not None)
    ]
    if non_passing:
        non_passing.sort(
            key=lambda t: (
                0 if t.status == DecisionStatus.FAILED else 1,
                t.calibrated_prob if t.calibrated_prob is not None else 1.0,
            )
        )
        return non_passing[0]
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
    kv("Snapshot ID:", _sanitize_terminal_text(str(snapshot.snapshot_id)))
    if snapshot.test_name:
        kv("Test Name:", _sanitize_terminal_text(snapshot.test_name))
    kv("Timestamp:", _sanitize_terminal_text(snapshot.timestamp))
    safe_rubric = _sanitize_terminal_text(snapshot.rubric)
    rubric_lines = textwrap.wrap(safe_rubric, width=72 - lab) or [""]
    kv("Rubric:", rubric_lines[0])
    out.extend(" " * lab + line for line in rubric_lines[1:])
    raw_candidate = str(snapshot.candidate)
    candidate = _sanitize_terminal_text(raw_candidate)
    width = 72 - lab
    wrapped = [w for line in candidate.strip().splitlines() for w in (textwrap.wrap(line, width) or [""])]
    kv("Candidate:", style(wrapped[0] if wrapped else "(empty)", "cyan"), f"{len(raw_candidate)} chars")
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
        prob_or_score = (
            _fmt(t.calibrated_prob, ".4f")
            if t.calibrated_prob is not None
            else (
                f"s={t.metadata['expected_score']:.2f}"
                if isinstance(t.metadata.get("expected_score"), (int, float))
                else "-"
            )
        )
        out.append(
            f"  {idx:<3}{_sanitize_terminal_text(t.layer_name)[:13]:<14}{status_cell}{t.latency_ms:>8.2f}ms"
            f"{prob_or_score:>9}{_fmt(t.log_odds_pass, '+.3f'):>10}"
            f"{_fmt(t.shannon_entropy, '.3f'):>9}{_fmt(t.positional_bias_magnitude, '.2f'):>7}{t.tokens_generated:>8}"
        )

    # Diagnostics
    out.append(style.rule())
    out.append(style("Diagnostics", "bold"))
    prob = verdict.probability
    kv("Original Prob:", f"{prob:.4f}" if prob is not None else "N/A", f"{verdict.status.value.upper()} at {_sanitize_terminal_text(verdict.triggered_layer or '')}")
    if verdict.score is not None:
        thresh = snapshot.gate_config.get("threshold")
        score_hint = f"threshold={thresh}" if thresh is not None else None
        kv("Original Score:", f"{verdict.score:.4f}", score_hint)

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
        kv("Failure Code:", style(_sanitize_terminal_text(snapshot.failure_code), "red"))
    failing = next((t for t in reversed(verdict.traces) if t.status != DecisionStatus.PASSED), None)
    if failing is not None and failing.diagnostic_report and failing.tokens_generated == 0:
        kv("Detail:", _sanitize_terminal_text(failing.diagnostic_report)[:200])
    kv("Triggered Layer:", _sanitize_terminal_text(verdict.triggered_layer or ""))

    # Replay
    out.append(style.rule())
    if replay is None:
        recorded = next((t for t in reversed(verdict.traces) if t.tokens_generated > 0 and t.diagnostic_report), None)
        if recorded is not None:
            out.append(style(f"Recorded critique ({_sanitize_terminal_text(recorded.layer_name)}, {recorded.tokens_generated} tokens):", "bold"))
            out.extend("  " + line for line in _sanitize_terminal_text(recorded.diagnostic_report or "").strip().splitlines())
            out.append(style.rule())
        out.append(style(replay_note or "Replay skipped.", "dim"))
        out.append(style.rule(heavy=True))
        return "\n".join(out)

    kv("Replay Verdict:", style.status(replay.status, upper=True))
    kv("Replay Latency:", f"{replay.total_latency_ms:.2f}ms")
    kv("Tokens Spent:", str(replay.total_tokens_generated))
    out.append(style.rule())
    out.append(style("Diagnostic Critique:", "bold"))
    critique = _sanitize_terminal_text(replay.reason or "(No reason provided)").strip()
    out.extend("  " + line for line in critique.splitlines())
    out.append(style.rule(heavy=True))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from logitgate.backends.base import infer_backend_spec, resolve_logit_backend
    from logitgate.calibration.profile import (
        CalibrationProfile,
        fit_calibration_profile,
    )

    if args.dataset and (
        isinstance(args.alpha, bool) or not math.isfinite(args.alpha) or not (0.0 < args.alpha < 1.0)
    ):
        sys.stderr.write(f"Error: --alpha must be strictly between 0.0 and 1.0, got {args.alpha}\n")
        return EXIT_FAILED

    backend = resolve_logit_backend(args.backend, model=args.model)
    effective_model = args.model or getattr(backend, "model", None) or getattr(backend, "model_name", None) or "default"
    try:
        if args.dataset:
            import json

            from logitgate.calibration.conformal import load_calibration_dataset

            try:
                samples = load_calibration_dataset(args.dataset)
            except (OSError, ValueError) as e:
                sys.stderr.write(f"Error loading dataset '{args.dataset}': {e}\n")
                return EXIT_FAILED
            if not samples:
                sys.stderr.write(f"Error: No valid samples found in dataset '{args.dataset}'\n")
                return EXIT_FAILED
            if len(samples) < 10:
                sys.stderr.write(
                    f"Error: Calibration requires at least 10 samples, got {len(samples)} in '{args.dataset}'\n"
                )
                return EXIT_FAILED
            if len({bool(y) for _, _, y in samples}) < 2:
                sys.stderr.write(
                    f"Error: Calibration set '{args.dataset}' must contain both positive and negative ground truth labels.\n"
                )
                return EXIT_FAILED

            try:
                profile = fit_calibration_profile(
                    backend=backend,
                    model=args.model,
                    samples=samples,
                    target_risk_alpha=args.alpha,
                    output_path=args.output,
                    cache_path=args.cache_path,
                )
            except (OSError, ValueError, CalibrationError) as e:
                sys.stderr.write(f"Error: {sanitize_terminal_text(str(e))}\n")
                return EXIT_FAILED

            assert profile.fail_threshold is not None
            assert profile.pass_threshold is not None
            assert profile.target_risk_alpha is not None
            assert profile.empirical_escalation_rate is not None

            if args.json:
                payload = {
                    **profile.model_dump(),
                    "optimal_temperature": profile.binary_temperature,
                    "tau_reject": profile.fail_threshold,
                    "tau_accept": profile.pass_threshold,
                }
                print(json.dumps(payload, indent=2))
            else:
                print("=" * 60)
                print("LOGITGATE SPLIT CONFORMAL CASCADE CALIBRATION")
                print("=" * 60)
                print(f"Dataset:                   {sanitize_terminal_text(str(args.dataset))}")
                print(f"Calibration Samples:       {profile.calibration_set_size}")
                print(f"Target Risk (Alpha):       {profile.target_risk_alpha:.4f}")
                print(f"Fitted Temperature:        {profile.binary_temperature:.4f}")
                print("-" * 60)
                print(f"Layer 1 Tau Reject:        {profile.fail_threshold:.4f}")
                print(f"Layer 1 Tau Accept:        {profile.pass_threshold:.4f}")
                print(f"Layer 2 Escalation Rate:   {profile.empirical_escalation_rate:.2%}")
                print("-" * 60)
                if profile.guarantee_holds:
                    print(
                        f"Guaranteed Layer 1 Error: <= {profile.target_risk_alpha:.2%} "
                        "(marginal; assumes production inputs resemble this dataset)"
                    )
                else:
                    print(
                        f"Guaranteed Layer 1 Error: NONE. Need >= {profile.min_samples_per_class} positives "
                        f"and negatives (have {profile.num_positive} / {profile.num_negative}); "
                        "thresholds are best-effort."
                    )
                if args.output:
                    safe_out = sanitize_terminal_text(str(args.output))
                    print(f"Saved Profile:             {safe_out}")
                    print(f'Use: LogitGate(..., profile="{safe_out}")')
                else:
                    print(
                        f"Use: LogitGate(..., pass_threshold={profile.pass_threshold}, "
                        f"fail_threshold={profile.fail_threshold}, "
                        f"logit_temperature={profile.binary_temperature})"
                    )
                print("=" * 60)
            return EXIT_OK

        from logitgate.debiasing.permutation import auto_calibrate_backend

        cache_path = Path(args.cache_path) if args.cache_path else None

        if not args.json:
            print(f"Calibrating model '{sanitize_terminal_text(effective_model)}' on backend {backend.__class__.__name__}...")
        try:
            result = auto_calibrate_backend(
                backend,
                model_key=effective_model,
                save_cache=True,
                cache_path=cache_path,
            )
            if args.output:
                profile = CalibrationProfile.from_calibration_result(
                    result,
                    model=effective_model,
                    backend=infer_backend_spec(backend),
                )
                profile.save(args.output)
        except (OSError, ValueError) as e:
            sys.stderr.write(f"Error saving calibration cache: {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(f"Optimal Temperature: {result.optimal_temperature:.4f}")
            if result.multiclass_temperature is not None:
                print(f"Multiclass Temperature: {result.multiclass_temperature:.4f}")
            if result.probe_auroc is not None:
                print(f"Probe AUROC: {result.probe_auroc:.4f}")
            print(f"Brier Score: {result.brier_score_before:.6f} -> {result.brier_score_after:.6f}")
            print(f"ECE: {result.ece_before:.6f} -> {result.ece_after:.6f}")
            print("Calibration successfully saved to cache.")
            if args.output:
                print(f"Calibration profile saved to {sanitize_terminal_text(str(args.output))}.")
        return EXIT_OK
    finally:
        backend.close()


def _cmd_explain(args: argparse.Namespace) -> int:
    from logitgate.snapshot import VerdictSnapshot

    if args.max_tokens is not None and args.max_tokens < 1:
        sys.stderr.write("Error: --max-tokens must be >= 1\n")
        return EXIT_FAILED

    try:
        snapshot = VerdictSnapshot.load(args.snapshot_path)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
        sys.stderr.write(f"Error loading snapshot from '{args.snapshot_path}': {sanitize_terminal_text(str(e))}\n")
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
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\nTip: `logitgate explain --offline {args.snapshot_path}` shows diagnostics without a model.\n")
        return EXIT_BACKEND_ERROR
    except TypeError as exc:
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\n")
        return EXIT_FAILED

    if args.json:
        print(verdict.model_dump_json(indent=2))
    else:
        print(render_snapshot_report(snapshot, style, replay=verdict))
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from logitgate.benchmark import (
        BUILTIN_CASES,
        format_report,
        load_cases,
        run_benchmark,
        write_json,
    )

    if getattr(args, "audit_only", False):
        import json

        from logitgate.calibration.audit import (
            audit_benchmark_dataset,
            check_split_leakage,
        )

        target_source: Any = args.dataset if args.dataset is not None else BUILTIN_CASES
        target_label = sanitize_terminal_text(str(args.dataset)) if args.dataset is not None else "builtin"
        try:
            audit_rep = audit_benchmark_dataset(target_source, label=target_label)
            leak_rep = None
            if args.reference_dataset is not None:
                leak_rep = check_split_leakage(
                    args.reference_dataset,
                    target_source,
                    reference_label=sanitize_terminal_text(str(args.reference_dataset)),
                    eval_label=target_label,
                )
        except (OSError, ValueError, CalibrationError) as e:
            sys.stderr.write(f"Error auditing dataset: {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED

        if args.json:
            payload: dict[str, Any] = {"audit": audit_rep.to_dict()}
            if leak_rep is not None:
                payload["leakage"] = leak_rep.to_dict()
            print(json.dumps(payload, indent=2))
        else:
            print(sanitize_terminal_text(audit_rep.format_summary()))
            if leak_rep is not None:
                print()
                print(sanitize_terminal_text(leak_rep.format_summary()))

        if getattr(args, "strict_audit", False) and (
            audit_rep.warnings or (leak_rep is not None and not leak_rep.is_clean)
        ):
            return EXIT_FAILED
        return EXIT_OK

    if args.judge_max_tokens is not None and args.judge_max_tokens < 1:
        sys.stderr.write("Error: --judge-max-tokens must be >= 1\n")
        return EXIT_FAILED
    if args.repeat < 1 or args.warmup < 0:
        sys.stderr.write("Error: --repeat must be >= 1 and --warmup must be >= 0\n")
        return EXIT_FAILED

    try:
        cases, label = load_cases(args.dataset)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error loading dataset '{args.dataset}': {sanitize_terminal_text(str(e))}\n")
        return EXIT_FAILED
    if args.limit is not None:
        if args.limit < 1:
            sys.stderr.write("Error: --limit must be >= 1\n")
            return EXIT_FAILED
        cases = cases[: args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    def progress(line: str) -> None:
        sys.stderr.write(line + "\n")

    try:
        cache_arg: Any = (
            True
            if getattr(args, "cache", None) == "memory"
            else (args.cache if getattr(args, "cache", None) is not None else False)
        )
        report = run_benchmark(
            cases,
            backend=args.backend,
            model=args.model,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            modes=modes,
            repeat=args.repeat,
            warmup=args.warmup,
            judge_max_tokens=args.judge_max_tokens,
            judge_think=args.judge_think,
            cascade_trigger=args.cascade_trigger,
            profile=args.profile,
            cache=cache_arg,
            dataset_label=label,
            progress=None if (args.quiet or args.json) else progress,
        )
    except (ValueError, TypeError) as e:
        sys.stderr.write(f"Error: {sanitize_terminal_text(str(e))}\n")
        return EXIT_FAILED

    if args.json:
        write_json(report)
    else:
        print(format_report(report))
    return EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    candidate: str | None = None
    if args.file is not None:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                candidate = f.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading file '{args.file}': {e}\n")
            return EXIT_FAILED
    elif args.candidate is not None:
        candidate = args.candidate
    else:
        if not sys.stdin.isatty():
            try:
                candidate = sys.stdin.read()
            except (OSError, UnicodeDecodeError) as e:
                sys.stderr.write(f"Error reading stdin: {e}\n")
                return EXIT_FAILED
        else:
            sys.stderr.write("Error: Either --candidate or --file must be specified.\n")
            return EXIT_FAILED

    from logitgate.cache import VerdictCache
    from logitgate.telemetry.sinks import JsonlTelemetrySink

    sink = JsonlTelemetrySink(args.telemetry_out) if getattr(args, "telemetry_out", None) else None
    raw_cache_arg = getattr(args, "cache", None)
    cache_obj: VerdictCache | None = None
    try:
        if raw_cache_arg is not None:
            cache_obj = (
                VerdictCache()
                if raw_cache_arg == "memory"
                else VerdictCache(path=raw_cache_arg)
            )
        has_ast = bool(args.ast_syntax or getattr(args, "normalize", False))
        logit_gate = LogitGate(
            rubric=args.rubric,
            backend=args.backend,
            model=args.model,
            pass_threshold=args.threshold,
            fail_threshold=args.fail_threshold,
            profile=args.profile,
            telemetry_sink=sink if not has_ast else None,
            cache=cache_obj if not has_ast else False,
        )
        try:
            evaluator: Cascade | LogitGate
            if has_ast:
                evaluator = Cascade(
                    [
                        ASTGate(
                            syntax=args.ast_syntax,
                            normalize=getattr(args, "normalize", False),
                            cache=False,
                        ),
                        logit_gate,
                    ],
                    telemetry_sink=sink,
                    cache=cache_obj,
                )
            else:
                evaluator = logit_gate

            verdict = evaluator.evaluate(candidate)
        finally:
            logit_gate.backend.close()
    finally:
        if cache_obj is not None:
            cache_obj.close()
        if sink is not None:
            sink.close()

    if args.json:
        print(verdict.model_dump_json(indent=2))
    else:
        verdict_str = verdict.status.value.upper()
        print(f"Verdict: {verdict_str} (status: {verdict.status.value})")
        if verdict.probability is not None:
            print(f"Confidence: {verdict.probability:.4f}")
        print(f"Triggered Layer: {sanitize_terminal_text(verdict.triggered_layer or '')}")
        print(f"Total Latency: {verdict.total_latency_ms:.2f}ms")
        if verdict.reason:
            print(f"Reason: {sanitize_terminal_text(verdict.reason)}")

    return EXIT_OK if verdict.passed else EXIT_FAILED


_MAX_CANDIDATES_JSON_BYTES = 4 * 1024 * 1024


def _cmd_rank(args: argparse.Namespace) -> int:
    import json
    import stat
    from pathlib import Path

    from logitgate.backends.base import resolve_logit_backend
    from logitgate.gates.rank_gate import BestOfNGate

    cand_list: list[Any] = list(args.candidate or [])
    for file_path in args.file or []:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                cand_list.append(f.read())
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading file '{file_path}': {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED

    candidates_input: Any = cand_list
    if args.candidates_json is not None:
        json_path = Path(args.candidates_json)
        try:
            st = json_path.stat()
            if not stat.S_ISREG(st.st_mode):
                sys.stderr.write(f"Error: '{args.candidates_json}' is not a regular file.\n")
                return EXIT_FAILED
            if st.st_size > _MAX_CANDIDATES_JSON_BYTES:
                sys.stderr.write(
                    f"Error: Candidates JSON '{args.candidates_json}' exceeds 4 MiB limit.\n"
                )
                return EXIT_FAILED
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as e:
            sys.stderr.write(
                f"Error loading candidates JSON '{args.candidates_json}': {sanitize_terminal_text(str(e))}\n"
            )
            return EXIT_FAILED

        if isinstance(loaded, dict):
            if not cand_list:
                candidates_input = loaded
            else:
                merged: dict[str, Any] = {str(i): c for i, c in enumerate(cand_list)}
                for k, v in loaded.items():
                    key = str(k)
                    while key in merged:
                        key = f"json_{key}"
                    merged[key] = v
                candidates_input = merged
        elif isinstance(loaded, list):
            cand_list.extend(loaded)
            candidates_input = cand_list
        else:
            sys.stderr.write(
                f"Error: --candidates-json must contain a JSON array or object, got {type(loaded).__name__}.\n"
            )
            return EXIT_FAILED

    if len(candidates_input) < 2:
        sys.stderr.write(
            "Error: rank requires at least 2 candidates via --candidate, --file, or --candidates-json.\n"
        )
        return EXIT_FAILED

    from logitgate.telemetry.sinks import JsonlTelemetrySink

    sink = JsonlTelemetrySink(args.telemetry_out) if getattr(args, "telemetry_out", None) else None
    resolved_backend = resolve_logit_backend(args.backend, model=args.model)
    try:
        pre_gate = (
            ASTGate(syntax=args.ast_syntax, normalize=args.normalize)
            if (args.ast_syntax or args.normalize)
            else None
        )
        post_gate = (
            LogitGate(
                rubric=args.rubric,
                backend=resolved_backend,
                model=args.model,
                profile=args.profile,
                telemetry_sink=False,
            )
            if args.verify_winner
            else None
        )
        gate = BestOfNGate(
            rubric=args.rubric,
            backend=resolved_backend,
            model=args.model,
            pre_gate=pre_gate,
            post_gate=post_gate,
            mode=args.mode,
            context=args.context,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            max_positional_bias=args.max_positional_bias,
            profile=args.profile,
            telemetry_sink=sink,
        )
        result = gate.select(candidates_input)
    finally:
        resolved_backend.close()
        if sink is not None:
            sink.close()

    if args.json:
        payload = {
            **result.verdict.model_dump(),
            "winner": result.winner,
            "winner_key": result.winner_key,
            "winner_index": result.winner_index,
            "ranking": list(result.ranking),
            "surviving_keys": list(result.surviving_keys),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        verdict = result.verdict
        print(f"Verdict: {verdict.status.value.upper()} (status: {verdict.status.value})")
        print(f"Winner Key: {sanitize_terminal_text(str(result.winner_key))}")
        print(f"Ranking: {', '.join(sanitize_terminal_text(k) for k in result.ranking)}")
        probs_str = ", ".join(
            f"{sanitize_terminal_text(k)}={p:.4f}" for k, p in result.choice_probabilities.items()
        )
        print(f"Probabilities: {probs_str}")
        print(f"Triggered Layer: {sanitize_terminal_text(verdict.triggered_layer or '')}")
        print(f"Total Latency: {verdict.total_latency_ms:.2f}ms")
        if verdict.reason:
            print(f"Reason: {sanitize_terminal_text(verdict.reason)}")

    return EXIT_OK if result.verdict.passed else EXIT_FAILED


def _cmd_telemetry(args: argparse.Namespace) -> int:
    import json

    from logitgate.telemetry import (
        detect_telemetry_drift,
        export_distillation_dataset,
        recalibrate_from_telemetry,
        summarize_telemetry,
    )

    sub = getattr(args, "telemetry_command", None)
    if sub == "summary":
        summary = summarize_telemetry(args.input_path)
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE TELEMETRY SUMMARY")
            print("=" * 60)
            print(f"Source:                    {sanitize_terminal_text(str(args.input_path))}")
            print(f"Total Records:             {summary.total_records}")
            print(
                f"Verdicts:                  {summary.passed_count} passed / "
                f"{summary.failed_count} failed / {summary.ambiguous_count} ambiguous"
            )
            print(
                f"Fast-Path Rate:            {summary.fast_path_rate:.2%} "
                f"({summary.fast_path_count}/{summary.total_records})"
            )
            print(
                f"Escalation Rate:           {summary.escalation_rate:.2%} "
                f"({summary.escalated_count}/{summary.total_records})"
            )
            print(
                f"Teacher Discrepancies:     {summary.discrepancy_count} "
                f"({summary.discrepancy_rate:.2%} of escalations)"
            )
            print(
                f"Latency (p50 / p95 / avg): {summary.median_latency_ms:.2f}ms / "
                f"{summary.p95_latency_ms:.2f}ms / {summary.mean_latency_ms:.2f}ms"
            )
            print(f"Total Tokens Generated:    {summary.total_tokens_generated}")
            if summary.by_gate_type:
                gates_str = ", ".join(
                    f"{sanitize_terminal_text(k)}={v}" for k, v in sorted(summary.by_gate_type.items())
                )
                print(f"By Gate Type:              {gates_str}")
            print("=" * 60)
        return EXIT_OK

    if sub == "export":
        rows = export_distillation_dataset(
            args.input_path,
            args.output,
            format=args.format,
            include_symmetric_swaps=not args.no_symmetric_swaps,
            dedupe=not args.no_dedupe,
            discrepancies_only=args.discrepancies_only,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "input_path": str(args.input_path),
                        "output_path": str(args.output),
                        "format": args.format,
                        "rows_written": rows,
                    },
                    indent=2,
                )
            )
        else:
            print(
                f"Exported {rows} rows ({args.format}) to "
                f"{sanitize_terminal_text(str(args.output))}."
            )
        return EXIT_OK

    if sub == "recalibrate":
        profile = recalibrate_from_telemetry(
            args.input_path,
            target_risk_alpha=args.alpha,
            model=args.model,
            output_path=args.output,
            dedupe=not args.no_dedupe,
        )
        if args.json:
            print(profile.model_dump_json(indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE ZERO-CALL TELEMETRY RECALIBRATION")
            print("=" * 60)
            print(f"Telemetry Source:          {sanitize_terminal_text(str(args.input_path))}")
            print(f"Escalation Samples Used:   {profile.calibration_set_size}")
            print(f"Fitted Temperature:        {profile.binary_temperature:.4f}")
            if profile.fail_threshold is not None and profile.pass_threshold is not None:
                print(f"Layer 1 Tau Reject:        {profile.fail_threshold:.4f}")
                print(f"Layer 1 Tau Accept:        {profile.pass_threshold:.4f}")
            if profile.empirical_escalation_rate is not None:
                print(f"Empirical Escalation Rate: {profile.empirical_escalation_rate:.2%}")
            if args.output:
                print(f"Saved Profile:             {sanitize_terminal_text(str(args.output))}")
            print("=" * 60)
        return EXIT_OK

    if sub == "drift":
        report = detect_telemetry_drift(
            args.reference,
            args.input_path,
            psi_moderate=args.psi_moderate,
            psi_severe=args.psi_severe,
            ks_alpha=args.ks_alpha,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE TELEMETRY DRIFT REPORT")
            print("=" * 60)
            print(f"Reference Source:          {sanitize_terminal_text(str(args.reference))} (n={report.reference_count})")
            print(f"Current Source:            {sanitize_terminal_text(str(args.input_path))} (n={report.current_count})")
            print(f"Drift Detected:            {report.drift_detected} (severity={report.severity.upper()})")
            print(f"Recommended Action:        {report.recommended_action}")
            print(f"Log-Odds PSI / KS p-val:   {report.psi_log_odds:.4f} / {report.ks_log_odds_p_value:.6f} (D={report.ks_log_odds_statistic:.4f})")
            print(f"Escalation Rate (Ref/Cur): {report.ref_escalation_rate:.2%} -> {report.cur_escalation_rate:.2%} ({report.escalation_rate_delta:+.2%})")
            print(f"Mean Pos Bias (Ref/Cur):   {report.ref_mean_pos_bias:.4f} -> {report.cur_mean_pos_bias:.4f} ({report.pos_bias_delta:+.4f})")
            if report.teacher_disagreement_rate is not None:
                print(
                    f"Teacher Disagreement Rate: {report.teacher_disagreement_rate:.2%} "
                    f"(n={report.observed_teacher_count})"
                )
            if report.reasons:
                print("Triggers:")
                for reason in report.reasons:
                    print(f"  - {sanitize_terminal_text(reason)}")
            print("=" * 60)
        return EXIT_FAILED if (args.fail_on_drift and report.drift_detected) else EXIT_OK

    sys.stderr.write(
        "Error: Specify a telemetry subcommand: summary, export, recalibrate, or drift.\n"
    )
    return EXIT_FAILED


def _cmd_stream(args: argparse.Namespace) -> int:
    import json

    from logitgate.cache import VerdictCache
    from logitgate.gates.judge_gate import JudgeGate
    from logitgate.streaming import StreamGuard
    from logitgate.telemetry.sinks import JsonlTelemetrySink

    if args.chunk_size < 1:
        sys.stderr.write("Error: --chunk-size must be >= 1\n")
        return EXIT_FAILED

    raw_text: str | None = None
    if args.file is not None:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading file '{args.file}': {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED
    elif args.candidate is not None:
        raw_text = args.candidate
    elif not sys.stdin.isatty():
        try:
            raw_text = sys.stdin.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading stdin: {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED
    else:
        sys.stderr.write("Error: Either --candidate, --file, or piped stdin must be provided.\n")
        return EXIT_FAILED

    chunks = [
        raw_text[i : i + args.chunk_size]
        for i in range(0, len(raw_text), args.chunk_size)
    ] if raw_text else []

    sink = JsonlTelemetrySink(args.telemetry_out) if getattr(args, "telemetry_out", None) else None
    cache_obj: VerdictCache | None = None
    judge_gate: JudgeGate | None = None
    try:
        if args.cache is not None:
            cache_obj = VerdictCache() if args.cache == "memory" else VerdictCache(path=args.cache)
        has_judge = bool(args.judge_backend or args.judge_model)
        logit_gate = LogitGate(
            rubric=args.rubric,
            backend=args.backend,
            model=args.model,
            pass_threshold=args.threshold,
            fail_threshold=args.fail_threshold,
            profile=args.profile,
            save_on_fail=False,
            telemetry_sink=sink if not has_judge else False,
            cache=cache_obj if not has_judge else False,
        )
        try:
            target_gate: Cascade | LogitGate
            if has_judge:
                judge_gate = JudgeGate(
                    rubric=None,
                    backend=args.judge_backend or args.backend,
                    model=args.judge_model or args.model,
                    trigger="on_ambiguous_only",
                    save_on_fail=False,
                    telemetry_sink=False,
                    cache=False,
                )
                target_gate = Cascade(
                    [logit_gate, judge_gate],
                    telemetry_sink=sink,
                    cache=cache_obj,
                )
            else:
                target_gate = logit_gate

            guard = StreamGuard(
                target_gate,
                min_chars=args.min_chars,
                stride_chars=args.stride_chars,
                clause_boundary=not args.no_clause_boundary,
                consecutive_failures=args.consecutive_failures,
                delivery=args.delivery,
                raise_on_abort=False,
                telemetry_sink=sink,
            )
            stream_handle = guard.stream(chunks)
            yielded_parts = list(stream_handle)
            summary = stream_handle.summary
            assert summary is not None
        finally:
            if judge_gate is not None:
                judge_gate.backend.close()
            logit_gate.backend.close()
    finally:
        if cache_obj is not None:
            cache_obj.close()
        if sink is not None:
            sink.close()

    yielded_text = "".join(yielded_parts)
    if args.json:
        payload = {
            **summary.to_dict(),
            "yielded_text": yielded_text,
        }
        print(json.dumps(payload, indent=2))
    else:
        if not args.quiet_stream and yielded_text:
            print(sanitize_terminal_text(yielded_text))
            print("-" * 60)
        status_label = "PASSED" if summary.passed else ("ABORTED_MID_STREAM" if summary.aborted_mid_stream else "FAILED")
        print(f"Stream Status: {status_label}")
        print(f"Delivery Mode: {args.delivery}")
        print(f"Chunks: {summary.chunks_yielded}/{summary.total_chunks_seen} yielded ({summary.chars_yielded}/{summary.chars_seen} chars)")
        print(f"Checks Executed: {summary.checks_executed} ({summary.total_gate_latency_ms:.2f}ms total gate latency)")
        if summary.final_verdict and summary.final_verdict.reason:
            print(f"Reason: {sanitize_terminal_text(summary.final_verdict.reason)}")

    return EXIT_OK if summary.passed else EXIT_FAILED


_COMMANDS: dict[str, Any] = {
    "eval": _cmd_eval,
    "calibrate": _cmd_calibrate,
    "explain": _cmd_explain,
    "benchmark": _cmd_benchmark,
    "rank": _cmd_rank,
    "telemetry": _cmd_telemetry,
    "stream": _cmd_stream,
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
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\n")
        return EXIT_BACKEND_ERROR
    except (ConfigurationError, CalibrationError) as exc:
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\n")
        return EXIT_FAILED
    return result


if __name__ == "__main__":
    sys.exit(main())
