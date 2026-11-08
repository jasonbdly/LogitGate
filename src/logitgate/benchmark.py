"""Reproducible latency and cost comparison: generative judge vs. `LogitGate` vs. cascade.

Run it against your own model before you trust anyone's numbers, including ours::

    logitgate benchmark --backend ollama://localhost:11434 --model qwen2.5:7b
    logitgate benchmark --backend vllm://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset benchmarks/data/multi_domain_24.json --repeat 3 --json > results.json

Methodology (printed with every report, because hidden methodology is how benchmarks lie):

- Latency is wall-clock around ``gate.evaluate()``: HTTP, prefill, decode, and parsing.
- Each mode runs as a separate pass over the dataset after ``--warmup`` untimed calls, so
  model load time isn't charged to whichever mode happens to run first.
- The judge baseline runs with ``think=False`` by default, its fastest configuration. The
  speedups reported are therefore against a *fast* judge, not a strawman.
- `LogitGate` with symmetric debiasing costs two forward passes per case. "LLM calls / case"
  reports this honestly, alongside generated tokens.
- Servers with prefix caching (vLLM, llama.cpp, Ollama) may reuse KV entries across modes.
  For strict isolation, run one mode per invocation (``--modes logit``).
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    infer_backend_spec,
    resolve_logit_backend,
)
from logitgate.gates.base import BaseGate, EscalationPolicy
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.models import DecisionStatus

MODES = ("judge", "logit", "cascade")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    rubric: str
    candidate: str
    expected: bool


# Clear-cut, balanced (8 pass / 8 fail) cases across the workloads LogitGate targets. Small on
# purpose: the built-in set exists to measure latency and cost, not to rank models. Use
# --dataset for accuracy claims.
BUILTIN_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "Is the ANSWER fully supported by the CONTEXT, with no unsupported claims?",
        "CONTEXT: The Eiffel Tower was completed in 1889 and is 330 metres tall.\n"
        "ANSWER: The Eiffel Tower, finished in 1889, stands about 330 metres tall.",
        True,
    ),
    BenchmarkCase(
        "Is the ANSWER fully supported by the CONTEXT, with no unsupported claims?",
        "CONTEXT: The Eiffel Tower was completed in 1889 and is 330 metres tall.\n"
        "ANSWER: The Eiffel Tower was completed in 1889 and receives 7 million visitors a year.",
        False,
    ),
    BenchmarkCase(
        "Is the ANSWER fully supported by the CONTEXT, with no unsupported claims?",
        "CONTEXT: Our refund window is 30 days from delivery for unused items.\n"
        "ANSWER: You can return unused items within 30 days of delivery for a refund.",
        True,
    ),
    BenchmarkCase(
        "Is the ANSWER fully supported by the CONTEXT, with no unsupported claims?",
        "CONTEXT: Our refund window is 30 days from delivery for unused items.\n"
        "ANSWER: Refunds are available for 90 days, even for used items.",
        False,
    ),
    BenchmarkCase(
        "Does the tool call contain a valid ISO-8601 date in the 'date' field and a positive integer 'guests'?",
        '{"tool": "book_table", "args": {"date": "2026-03-14", "guests": 4}}',
        True,
    ),
    BenchmarkCase(
        "Does the tool call contain a valid ISO-8601 date in the 'date' field and a positive integer 'guests'?",
        '{"tool": "book_table", "args": {"date": "next friday", "guests": -2}}',
        False,
    ),
    BenchmarkCase(
        "Does the support reply refuse to share another customer's personal data?",
        "I'm sorry, I can't share details about other customers' accounts. "
        "I can help with questions about your own order.",
        True,
    ),
    BenchmarkCase(
        "Does the support reply refuse to share another customer's personal data?",
        "Sure! The account under that name lists jane.doe@example.com and 42 Elm Street.",
        False,
    ),
    BenchmarkCase(
        "Is the SQL query read-only (no INSERT, UPDATE, DELETE, DROP, or ALTER)?",
        "SELECT id, total FROM orders WHERE created_at > NOW() - INTERVAL '7 days';",
        True,
    ),
    BenchmarkCase(
        "Is the SQL query read-only (no INSERT, UPDATE, DELETE, DROP, or ALTER)?",
        "SELECT * FROM users; DROP TABLE audit_log;",
        False,
    ),
    BenchmarkCase(
        "Does the Python function return the maximum element of a non-empty list?",
        "def max_elem(xs):\n    best = xs[0]\n    for x in xs[1:]:\n        if x > best:\n            best = x\n    return best",
        True,
    ),
    BenchmarkCase(
        "Does the Python function return the maximum element of a non-empty list?",
        "def max_elem(xs):\n    best = xs[0]\n    for x in xs[1:]:\n        if x < best:\n            best = x\n    return best",
        False,
    ),
    BenchmarkCase(
        "Does the Python code avoid executing user-controlled strings (no eval/exec/os.system on input)?",
        "import json\n\ndef load(payload: str) -> dict:\n    return json.loads(payload)",
        True,
    ),
    BenchmarkCase(
        "Does the Python code avoid executing user-controlled strings (no eval/exec/os.system on input)?",
        "def calc(expr: str):\n    return eval(expr)",
        False,
    ),
    BenchmarkCase(
        "Is the reply written entirely in English?",
        "Your order has shipped and should arrive on Thursday.",
        True,
    ),
    BenchmarkCase(
        "Is the reply written entirely in English?",
        "Votre commande a été expédiée et devrait arriver jeudi.",
        False,
    ),
)


class _MeteredBackend(LogitBackend, GenerativeBackend):
    """Counts forward passes and generation calls without changing backend behavior."""

    def __init__(self, inner: LogitBackend) -> None:
        if not isinstance(inner, GenerativeBackend):
            raise TypeError(f"Benchmarking the judge requires a generative backend, got {inner!r}")
        self.inner = inner
        self.logit_calls = 0
        self.generate_calls = 0

    def __getattr__(self, item: str) -> Any:  # base_url, model, timeout, ... for error messages
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    def __repr__(self) -> str:
        return repr(self.inner)

    def query_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None
    ) -> list[dict[str, Any]]:
        self.logit_calls += 1
        return self.inner.query_top_logprobs(prompt, top_k, allowed_tokens)

    async def aquery_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None
    ) -> list[dict[str, Any]]:
        self.logit_calls += 1
        return await self.inner.aquery_top_logprobs(prompt, top_k, allowed_tokens)

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        self.generate_calls += 1
        assert isinstance(self.inner, GenerativeBackend)
        return self.inner.generate(prompt, max_tokens=max_tokens, **kwargs)

    def close(self) -> None:
        self.inner.close()


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; well-defined for the small samples benchmarks produce."""
    ordered = sorted(values)
    rank = max(1, round(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(slots=True)
class ModeStats:
    mode: str
    latencies_ms: list[float] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    llm_calls: list[int] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)
    ambiguous: list[bool] = field(default_factory=list)
    judge_skipped: list[bool] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.latencies_ms)
        return {
            "mode": self.mode,
            "runs": n,
            "p50_ms": round(_percentile(self.latencies_ms, 50), 2),
            "p95_ms": round(_percentile(self.latencies_ms, 95), 2),
            "mean_ms": round(statistics.fmean(self.latencies_ms), 2),
            "generated_tokens_per_case": round(statistics.fmean(self.generated_tokens), 1),
            "llm_calls_per_case": round(statistics.fmean(self.llm_calls), 2),
            "accuracy": round(sum(self.correct) / n, 4),
            "correct": sum(self.correct),
            "ambiguous_rate": round(sum(self.ambiguous) / n, 4),
            "fast_path_rate": round(sum(self.judge_skipped) / n, 4) if self.mode == "cascade" else None,
        }


@dataclass(slots=True)
class BenchmarkReport:
    metadata: dict[str, Any]
    modes: dict[str, ModeStats]

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "modes": {m: s.summary() for m, s in self.modes.items()}}


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    backend: str | LogitBackend | None = None,
    model: str | None = None,
    modes: Sequence[str] = MODES,
    repeat: int = 1,
    warmup: int = 1,
    judge_max_tokens: int | None = None,
    judge_think: bool = False,
    cascade_trigger: EscalationPolicy = "on_ambiguous_only",
    dataset_label: str = "builtin",
    progress: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    """Runs each mode over every case ``repeat`` times and collects per-call measurements."""
    if not cases:
        raise ValueError("Benchmark needs at least one case")
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown benchmark modes {unknown}; choose from {list(MODES)}")
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be >= 1 and warmup >= 0")

    resolved = resolve_logit_backend(backend, model=model)
    effective_model = model or getattr(resolved, "model", None) or getattr(resolved, "model_name", None)
    metered = _MeteredBackend(resolved)
    # Only Ollama understands `think`; OpenAI-compatible servers may reject unknown fields.
    from logitgate.backends.ollama import OllamaBackend

    effective_think = judge_think if isinstance(resolved, (OllamaBackend, StubBackend)) else None

    def build(mode: str, rubric: str) -> BaseGate:
        logit = LogitGate(rubric=rubric, backend=metered, model=effective_model)
        if mode == "logit":
            return logit
        judge = JudgeGate(
            rubric=rubric if mode == "judge" else None,
            backend=metered,
            model=effective_model,
            trigger="always" if mode == "judge" else cascade_trigger,
            max_tokens=judge_max_tokens,
            think=effective_think,
        )
        return judge if mode == "judge" else logit >> judge

    stats = {mode: ModeStats(mode) for mode in modes}
    try:
        for mode in modes:
            for case in cases[:warmup]:
                build(mode, case.rubric).evaluate(case.candidate)
            for rep in range(repeat):
                for idx, case in enumerate(cases):
                    gate = build(mode, case.rubric)
                    logit_before, gen_before = metered.logit_calls, metered.generate_calls
                    t0 = time.perf_counter()
                    verdict = gate.evaluate(case.candidate)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    generate_calls = metered.generate_calls - gen_before

                    s = stats[mode]
                    s.latencies_ms.append(elapsed_ms)
                    s.generated_tokens.append(verdict.total_tokens_generated)
                    s.llm_calls.append(metered.logit_calls - logit_before + generate_calls)
                    s.correct.append(verdict.passed == case.expected)
                    s.ambiguous.append(verdict.status == DecisionStatus.AMBIGUOUS)
                    s.judge_skipped.append(generate_calls == 0)
                    if progress is not None:
                        progress(f"{mode:<8} rep {rep + 1}/{repeat} case {idx + 1}/{len(cases)} {elapsed_ms:9.1f} ms")
    finally:
        resolved.close()

    from logitgate import __version__

    digest = hashlib.sha256(
        json.dumps([(c.rubric, c.candidate, c.expected) for c in cases]).encode()
    ).hexdigest()[:12]
    metadata = {
        "logitgate_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": infer_backend_spec(resolved),
        "model": model or getattr(resolved, "model", None),
        "is_stub": isinstance(resolved, StubBackend),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"source": dataset_label, "cases": len(cases), "sha256_12": digest},
        "repeat": repeat,
        "warmup": warmup,
        "judge_think": effective_think,
        "judge_max_tokens": judge_max_tokens,
        "cascade_trigger": cascade_trigger,
    }
    return BenchmarkReport(metadata=metadata, modes=stats)


def format_report(report: BenchmarkReport) -> str:
    """Renders a fixed-width table that pastes cleanly into issues and PR descriptions."""
    meta = report.metadata
    rows = [s.summary() for s in report.modes.values()]
    lines = [
        f"LogitGate {meta['logitgate_version']} benchmark | backend={meta['backend']} model={meta['model']}",
        (
            f"dataset={meta['dataset']['source']} ({meta['dataset']['cases']} cases, "
            f"sha256:{meta['dataset']['sha256_12']}) repeat={meta['repeat']} warmup={meta['warmup']} "
            f"judge_think={meta['judge_think']} cascade_trigger={meta['cascade_trigger']}"
        ),
    ]
    if meta["is_stub"]:
        lines.append("NOTE: stub backend. Timings measure harness overhead only, not model latency.")
    header = f"{'mode':<9}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}{'gen tok':>9}{'calls':>7}{'accuracy':>11}{'ambig':>7}{'fast-path':>11}"
    lines += ["", header, "-" * len(header)]
    for r in rows:
        fast = f"{r['fast_path_rate']:.0%}" if r["fast_path_rate"] is not None else "-"
        lines.append(
            f"{r['mode']:<9}{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}{r['mean_ms']:>10.1f}"
            f"{r['generated_tokens_per_case']:>9.1f}{r['llm_calls_per_case']:>7.2f}"
            f"{r['correct']:>6}/{r['runs']:<4}{r['ambiguous_rate']:>7.0%}{fast:>11}"
        )
    judge = report.modes.get("judge")
    if judge is not None and not meta["is_stub"]:
        base = judge.summary()
        lines.append("")
        for r in rows:
            if r["mode"] == "judge" or r["mean_ms"] <= 0:
                continue
            speedup = base["mean_ms"] / r["mean_ms"]
            tokens = base["generated_tokens_per_case"]
            saved = 1.0 - r["generated_tokens_per_case"] / tokens if tokens else 0.0
            lines.append(f"{r['mode']} vs judge: {speedup:.1f}x faster (mean), {saved:.0%} fewer generated tokens")
    lines += [
        "",
        "gen tok = generated tokens per case; calls = forward passes + generate calls per case.",
        "Logit modes count AMBIGUOUS as not-passed. Accuracy on the built-in set is a sanity check, not a leaderboard.",
    ]
    return "\n".join(lines)


def load_cases(path: str | None) -> tuple[list[BenchmarkCase], str]:
    """Loads (rubric, candidate, label) cases from JSON/JSONL/Markdown, or the built-in set."""
    if path is None:
        return list(BUILTIN_CASES), "builtin"
    from logitgate.calibration.conformal import load_calibration_dataset

    return [BenchmarkCase(r, c, bool(y)) for r, c, y in load_calibration_dataset(path)], path


def write_json(report: BenchmarkReport) -> None:
    sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
