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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    infer_backend_spec,
    resolve_generative_backend,
    resolve_logit_backend,
)
from logitgate.gates.base import BaseGate, EscalationPolicy
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.models import DecisionStatus

if TYPE_CHECKING:
    from logitgate.calibration.profile import CalibrationProfile

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

    def __init__(self, inner: LogitBackend, *, require_generative: bool = True) -> None:
        if require_generative and not isinstance(inner, GenerativeBackend):
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
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.logit_calls += 1
        return self.inner.query_top_logprobs(prompt, top_k, allowed_tokens, **kwargs)

    async def aquery_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.logit_calls += 1
        return await self.inner.aquery_top_logprobs(prompt, top_k, allowed_tokens, **kwargs)

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        if not isinstance(self.inner, GenerativeBackend):
            raise TypeError(f"Benchmarking the judge requires a generative backend, got {self.inner!r}")
        self.generate_calls += 1
        return self.inner.generate(prompt, max_tokens=max_tokens, **kwargs)

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        if not isinstance(self.inner, GenerativeBackend):
            raise TypeError(f"Benchmarking the judge requires a generative backend, got {self.inner!r}")
        self.generate_calls += 1
        return await self.inner.agenerate(prompt, max_tokens=max_tokens, **kwargs)

    def close(self) -> None:
        self.inner.close()

    async def aclose(self) -> None:
        await self.inner.aclose()


class _MeteredGenerativeBackend(GenerativeBackend):
    """Meters generation calls on pure GenerativeBackend instances without exposing LogitBackend."""

    def __init__(self, inner: GenerativeBackend) -> None:
        if not isinstance(inner, GenerativeBackend):
            raise TypeError(f"Benchmarking the judge requires a generative backend, got {inner!r}")
        self.inner = inner
        self.logit_calls = 0
        self.generate_calls = 0

    def __getattr__(self, item: str) -> Any:
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    def __repr__(self) -> str:
        return repr(self.inner)

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        self.generate_calls += 1
        return self.inner.generate(prompt, max_tokens=max_tokens, **kwargs)

    async def agenerate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        self.generate_calls += 1
        return await self.inner.agenerate(prompt, max_tokens=max_tokens, **kwargs)

    def close(self) -> None:
        self.inner.close()

    async def aclose(self) -> None:
        await self.inner.aclose()


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


def _extract_backend_clone_kwargs(resolved: Any) -> dict[str, Any]:
    """Extracts transport/runtime options to preserve when cloning a backend for a judge model."""
    from logitgate.backends.ollama import OllamaBackend

    extra: dict[str, Any] = {}
    if getattr(resolved, "timeout", None) is not None:
        extra["timeout"] = resolved.timeout
    if getattr(resolved, "api_key", None):
        extra["api_key"] = resolved.api_key
    if isinstance(resolved, OllamaBackend):
        if getattr(resolved, "num_ctx", None) is not None:
            extra["num_ctx"] = resolved.num_ctx
        if getattr(resolved, "options", None):
            extra["options"] = dict(resolved.options)
    return extra


def _prewarm_calibration_if_needed(target_backend: Any, model_name: str | None) -> None:
    """Runs one-time BYO micro-probe calibration before metered measurement begins."""
    from logitgate.debiasing.permutation import auto_calibrate_backend
    from logitgate.gates.logit_gate import _should_lazy_auto_calibrate

    if model_name and _should_lazy_auto_calibrate(None, model_name, target_backend, binary=True):
        auto_calibrate_backend(target_backend, model_key=model_name, save_cache=True)


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    backend: str | LogitBackend | None = None,
    model: str | None = None,
    judge_backend: str | GenerativeBackend | None = None,
    judge_model: str | None = None,
    modes: Sequence[str] = MODES,
    repeat: int = 1,
    warmup: int = 1,
    judge_max_tokens: int | None = None,
    judge_think: bool = False,
    cascade_trigger: EscalationPolicy = "on_ambiguous_only",
    profile: CalibrationProfile | str | Path | None = None,
    dataset_label: str = "builtin",
    progress: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    """Runs each mode over every case ``repeat`` times and collects per-call measurements.

    Args:
        cases: Sequence of labeled benchmark cases to evaluate.
        backend: Tier-1 backend instance or URI specifier.
        model: Model override for Tier-1 logit gates.
        judge_backend: Optional separate Tier-2 backend instance or URI specifier.
        judge_model: Optional separate Tier-2 model override.
        modes: Subset of ``("judge", "logit", "cascade")`` to run.
        repeat: Number of measured passes per mode (>= 1).
        warmup: Number of untimed warmup evaluations before each mode (>= 0).
        judge_max_tokens: Optional generation token limit for ``JudgeGate`` (>= 1).
        judge_think: Whether to enable reasoning tokens on Ollama judges.
        cascade_trigger: Escalation policy for the ``cascade`` mode.
        profile: Optional `CalibrationProfile` (or path to JSON) applied to ``LogitGate``.
        dataset_label: Human-readable label recorded in report metadata.
        progress: Optional callback invoked after each timed case evaluation.

    Raises:
        ValueError: If parameters are out of valid bounds.
        TypeError: If a requested generative mode lacks a ``GenerativeBackend``.
    """
    from logitgate.calibration.profile import resolve_calibration_profile

    if not cases:
        raise ValueError("Benchmark needs at least one case")
    if not modes:
        raise ValueError(f"Benchmark needs at least one mode; choose from {list(MODES)}")
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown benchmark modes {unknown}; choose from {list(MODES)}")
    if isinstance(repeat, bool) or repeat < 1 or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("repeat must be >= 1 and warmup >= 0")
    if judge_max_tokens is not None and (isinstance(judge_max_tokens, bool) or judge_max_tokens < 1):
        raise ValueError(f"judge_max_tokens must be >= 1, got {judge_max_tokens!r}")

    resolved_profile = resolve_calibration_profile(profile)
    needs_logit = any(m in ("logit", "cascade") for m in modes)
    needs_generative = any(m in ("judge", "cascade") for m in modes)

    owns_backend = False
    resolved: Any = None
    owns_judge_backend = False
    resolved_judge: Any = None
    try:
        if needs_logit:
            owns_backend = not isinstance(backend, LogitBackend)
            resolved = resolve_logit_backend(backend, model=model)
            effective_model = model or getattr(resolved, "model", None) or getattr(resolved, "model_name", None)
            if resolved_profile is None:
                _prewarm_calibration_if_needed(resolved, effective_model)
            metered: _MeteredBackend | None = _MeteredBackend(resolved, require_generative=False)
        else:
            effective_model = model
            metered = None

        if needs_generative:
            heterogeneous_judge = judge_backend is not None or judge_model is not None
            if not needs_logit:
                if judge_backend is not None:
                    owns_judge_backend = not isinstance(judge_backend, GenerativeBackend)
                    resolved_judge = resolve_generative_backend(judge_backend, model=judge_model)
                elif isinstance(backend, GenerativeBackend):
                    owns_judge_backend = False
                    resolved_judge = backend
                elif isinstance(backend, LogitBackend):
                    owns_judge_backend = True
                    resolved_judge = resolve_generative_backend(
                        infer_backend_spec(backend),
                        model=judge_model or model,
                        **_extract_backend_clone_kwargs(backend),
                    )
                else:
                    owns_judge_backend = True
                    resolved_judge = resolve_generative_backend(backend, model=judge_model or model)
            elif heterogeneous_judge:
                if judge_backend is not None:
                    owns_judge_backend = not isinstance(judge_backend, GenerativeBackend)
                    resolved_judge = resolve_generative_backend(judge_backend, model=judge_model)
                elif not isinstance(backend, LogitBackend):
                    owns_judge_backend = True
                    resolved_judge = resolve_generative_backend(backend, model=judge_model)
                elif isinstance(resolved, StubBackend):
                    resolved_judge = resolved
                else:
                    owns_judge_backend = True
                    resolved_judge = resolve_generative_backend(
                        infer_backend_spec(resolved),
                        model=judge_model,
                        **_extract_backend_clone_kwargs(resolved),
                    )
            else:
                resolved_judge = resolved

            if not isinstance(resolved_judge, GenerativeBackend):
                raise TypeError(f"Benchmarking the judge requires a generative backend, got {resolved_judge!r}")
            effective_judge_model = (
                judge_model
                or getattr(resolved_judge, "model", None)
                or getattr(resolved_judge, "model_name", None)
                or effective_model
            )
            if resolved_judge is resolved and metered is not None:
                metered_judge: _MeteredBackend | _MeteredGenerativeBackend | None = metered
            else:
                _prewarm_calibration_if_needed(resolved_judge, effective_judge_model)
                metered_judge = (
                    _MeteredBackend(resolved_judge, require_generative=True)
                    if isinstance(resolved_judge, LogitBackend)
                    else _MeteredGenerativeBackend(resolved_judge)
                )
        else:
            resolved_judge = resolved
            effective_judge_model = effective_model
            metered_judge = metered

        if effective_model is None and resolved_judge is not None and not needs_logit:
            effective_model = effective_judge_model

        # Only Ollama understands `think`; OpenAI-compatible servers may reject unknown fields.
        from logitgate.backends.ollama import OllamaBackend

        effective_think = judge_think if isinstance(resolved_judge, (OllamaBackend, StubBackend)) else None
        judge_profile = resolved_profile if (resolved_judge is resolved and resolved_profile is not None) else None

        def _counts() -> tuple[int, int]:
            l_calls = metered.logit_calls if metered is not None else 0
            g_calls = metered.generate_calls if metered is not None else 0
            if metered_judge is not None and metered_judge is not metered:
                l_calls += metered_judge.logit_calls
                g_calls += metered_judge.generate_calls
            return l_calls, g_calls

        def build(mode: str, rubric: str) -> BaseGate:
            if mode == "judge":
                return JudgeGate(
                    rubric=rubric,
                    backend=metered_judge,
                    model=effective_judge_model,
                    trigger="always",
                    max_tokens=judge_max_tokens,
                    think=effective_think,
                    profile=judge_profile,
                )
            logit = LogitGate(
                rubric=rubric,
                backend=metered,
                model=effective_model,
                save_on_fail=False,
                profile=resolved_profile,
            )
            if mode == "logit":
                return logit
            judge = JudgeGate(
                rubric=None,
                backend=metered_judge,
                model=effective_judge_model,
                trigger=cascade_trigger,
                max_tokens=judge_max_tokens,
                think=effective_think,
                profile=judge_profile,
            )
            return logit >> judge

        stats = {mode: ModeStats(mode) for mode in modes}
        for mode in modes:
            if cases:
                for w_idx in range(warmup):
                    w_case = cases[w_idx % len(cases)]
                    build(mode, w_case.rubric).evaluate(w_case.candidate, save_on_fail=False)
            for rep in range(repeat):
                for idx, case in enumerate(cases):
                    gate = build(mode, case.rubric)
                    logit_before, gen_before = _counts()
                    t0 = time.perf_counter()
                    verdict = gate.evaluate(case.candidate, save_on_fail=False)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    logit_after, gen_after = _counts()
                    generate_calls = gen_after - gen_before

                    s = stats[mode]
                    s.latencies_ms.append(elapsed_ms)
                    s.generated_tokens.append(verdict.total_tokens_generated)
                    s.llm_calls.append((logit_after - logit_before) + generate_calls)
                    s.correct.append(verdict.passed == case.expected)
                    s.ambiguous.append(verdict.status == DecisionStatus.AMBIGUOUS)
                    s.judge_skipped.append(generate_calls == 0)
                    if progress is not None:
                        progress(f"{mode:<8} rep {rep + 1}/{repeat} case {idx + 1}/{len(cases)} {elapsed_ms:9.1f} ms")
    finally:
        if owns_judge_backend and resolved_judge is not None and resolved_judge is not resolved:
            resolved_judge.close()
        if owns_backend and resolved is not None:
            resolved.close()

    from logitgate import __version__

    primary_backend = resolved if resolved is not None else resolved_judge
    digest = hashlib.sha256(
        json.dumps([(c.rubric, c.candidate, c.expected) for c in cases]).encode()
    ).hexdigest()[:12]
    backend_spec = infer_backend_spec(primary_backend)
    judge_backend_spec = infer_backend_spec(resolved_judge if resolved_judge is not None else primary_backend)
    metadata: dict[str, Any] = {
        "logitgate_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend_spec,
        "model": effective_model,
        "judge_backend": judge_backend_spec,
        "judge_model": effective_judge_model,
        "is_stub": (resolved is None or isinstance(resolved, StubBackend))
        and (resolved_judge is None or isinstance(resolved_judge, StubBackend)),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"source": dataset_label, "cases": len(cases), "sha256_12": digest},
        "repeat": repeat,
        "warmup": warmup,
        "judge_think": effective_think,
        "judge_max_tokens": judge_max_tokens,
        "cascade_trigger": cascade_trigger,
    }
    if resolved_profile is not None:
        metadata["profile"] = {
            "binary_temperature": resolved_profile.binary_temperature,
            "pass_threshold": resolved_profile.pass_threshold,
            "fail_threshold": resolved_profile.fail_threshold,
            "target_risk_alpha": resolved_profile.target_risk_alpha,
            "guarantee_holds": resolved_profile.guarantee_holds,
        }
    return BenchmarkReport(metadata=metadata, modes=stats)


def format_report(report: BenchmarkReport) -> str:
    """Renders a fixed-width table that pastes cleanly into issues and PR descriptions."""
    from logitgate.cli import sanitize_terminal_text

    meta = report.metadata
    rows = [s.summary() for s in report.modes.values()]
    safe_backend = sanitize_terminal_text(str(meta.get("backend") or ""))
    safe_model = sanitize_terminal_text(str(meta.get("model") or ""))
    judge_backend_meta = meta.get("judge_backend")
    judge_model_meta = meta.get("judge_model")
    judge_suffix = ""
    if (
        (judge_backend_meta is not None and judge_backend_meta != meta["backend"])
        or (judge_model_meta is not None and judge_model_meta != meta["model"])
    ):
        safe_jb = sanitize_terminal_text(str(judge_backend_meta))
        safe_jm = sanitize_terminal_text(str(judge_model_meta))
        judge_suffix = f" | judge_backend={safe_jb} judge_model={safe_jm}"
    safe_source = sanitize_terminal_text(str(meta["dataset"]["source"]))
    lines = [
        f"LogitGate {meta['logitgate_version']} benchmark | backend={safe_backend} model={safe_model}{judge_suffix}",
        (
            f"dataset={safe_source} ({meta['dataset']['cases']} cases, "
            f"sha256:{meta['dataset']['sha256_12']}) repeat={meta['repeat']} warmup={meta['warmup']} "
            f"judge_think={meta['judge_think']} cascade_trigger={meta['cascade_trigger']}"
        ),
    ]
    prof_meta = meta.get("profile")
    if isinstance(prof_meta, dict):
        lines.append(
            f"profile: T={prof_meta.get('binary_temperature')} "
            f"tau_reject={prof_meta.get('fail_threshold')} "
            f"tau_accept={prof_meta.get('pass_threshold')} "
            f"alpha={prof_meta.get('target_risk_alpha')} "
            f"guarantee_holds={prof_meta.get('guarantee_holds')}"
        )
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
        comp_lines: list[str] = []
        for r in rows:
            if r["mode"] == "judge" or r["mean_ms"] <= 0:
                continue
            speedup = base["mean_ms"] / r["mean_ms"]
            tokens = base["generated_tokens_per_case"]
            saved = 1.0 - r["generated_tokens_per_case"] / tokens if tokens else 0.0
            comp_lines.append(f"{r['mode']} vs judge: {speedup:.1f}x faster (mean), {saved:.0%} fewer generated tokens")
        if comp_lines:
            lines.append("")
            lines.extend(comp_lines)
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
