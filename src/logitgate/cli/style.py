from __future__ import annotations

import math
import os
import re
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, TextIO

from logitgate.models import DecisionStatus, LayerTrace, Verdict

if TYPE_CHECKING:
    from logitgate.snapshot import VerdictSnapshot

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
