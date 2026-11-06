from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    resolve_generative_backend,
)
from logitgate.config import get_default_model
from logitgate.debiasing.inversion import render_candidate
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import TIER_GENERATIVE, BaseGate, EscalationPolicy
from logitgate.gates.logit_gate import LogitGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict

TriggerMode = EscalationPolicy  # Public alias kept for backward compatibility.

ParsedVerdict = tuple[bool, DecisionStatus, float | None]
_PASS: ParsedVerdict = (True, DecisionStatus.PASSED, 1.0)
_FAIL: ParsedVerdict = (False, DecisionStatus.FAILED, 0.0)

# Matches [VERDICT: PASS], **[VERDICT: PASS]**, [**VERDICT: PASS**], **Verdict:** FAIL, Verdict: PASS.
_VERDICT_TAG_RE = re.compile(r"\[?\*?\*?VERDICT[\s\*\]]*:[\s\*\[]*(PASS|FAIL)[\*\]]*", re.IGNORECASE)
# "did not fail", "without any failures", "zero rejected": negated failure words must not count as FAIL.
_NEGATED_FAILURE_RE = re.compile(
    r"\b(not|no|without|never|doesn't|does\s+not|did\s+not|didn't|zero)\s+(?:any\s+|to\s+)?"
    r"(?:fail|failed|failure|failures|reject|rejected)\b",
    re.IGNORECASE,
)
_PASS_WORD_RE = re.compile(r"\b(PASS|PASSED|PASSES|APPROVE|APPROVED)\b", re.IGNORECASE)
_FAIL_WORD_RE = re.compile(r"\b(FAIL|FAILED|FAILS|FAILURE|REJECT|REJECTED)\b", re.IGNORECASE)
_TRUTHY_STRINGS = frozenset({"pass", "passed", "true", "yes", "1"})
_PROBE_RUBRIC = (
    "Based on this review critique, did the candidate artifact satisfy all requirements "
    "and PASS the evaluation without any defects or errors?"
)


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yields every top-level JSON object embedded in free text, including nested ones.

    A non-greedy ``\\{.*?\\}`` regex truncates ``{"verdict": "pass", "detail": {...}}`` at the
    first closing brace. `raw_decode` parses balanced objects correctly.
    """
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        idx = text.find("{", end)


def _coerce_pass(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    return bool(value)


def _extract_structured_verdict(text: str) -> ParsedVerdict | None:
    """Reads an explicit verdict the judge was asked to emit: a ``[VERDICT: ...]`` tag or JSON.

    The *last* tag wins because the prompt asks for reasoning first and the verdict last, so
    an early "PASS" inside the reasoning bullets must not override the conclusion.
    """
    tags = _VERDICT_TAG_RE.findall(text)
    if tags:
        return _PASS if tags[-1].upper() == "PASS" else _FAIL

    # Only JSON that actually carries a verdict counts. A critique that quotes the candidate's
    # own JSON (e.g. '{"status": "ok"}') must not be read as a decisive FAIL.
    verdict_objects = [
        obj for obj in _iter_json_objects(text) if any(k in obj for k in ("pass", "passed", "verdict"))
    ]
    if not verdict_objects:
        return None
    data = verdict_objects[-1]
    if "pass" in data:
        is_pass = _coerce_pass(data["pass"])
    elif "passed" in data:
        is_pass = _coerce_pass(data["passed"])
    else:
        is_pass = str(data["verdict"]).strip().lower() == "pass"
    try:
        score = float(data.get("score", 1.0 if is_pass else 0.0))
    except (TypeError, ValueError):
        score = 1.0 if is_pass else 0.0
    return is_pass, DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED, score


def _keyword_signal(text: str, *, tie_break: bool) -> ParsedVerdict | None:
    sanitized = _NEGATED_FAILURE_RE.sub(" ", text)
    pass_match = _PASS_WORD_RE.search(sanitized)
    fail_match = _FAIL_WORD_RE.search(sanitized)
    if pass_match and fail_match:
        # Both on one line ("PASS - no blocking issues; minor failure modes noted"): the
        # leading word is the verdict. Across a whole critique, mixed words are no signal.
        if not tie_break:
            return None
        return _PASS if pass_match.start() < fail_match.start() else _FAIL
    if pass_match:
        return _PASS
    if fail_match:
        return _FAIL
    return None


def _extract_keyword_verdict(text: str) -> ParsedVerdict | None:
    """Last-resort keyword heuristic for critiques with no tag, no JSON, and no logit probe.

    Checks the concluding line first, then the opening line (short ``max_tokens`` prompts ask
    for the verdict up front), then the whole text. Returns ``None`` when the text gives no
    one-sided signal.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    candidate_lines = lines[-1:] + (lines[:1] if len(lines) > 1 else [])
    for line in candidate_lines:
        signal = _keyword_signal(line, tie_break=True)
        if signal is not None:
            return signal
    return _keyword_signal(text, tie_break=False)


class JudgeGate(BaseGate):
    """Generative judge (Layer 2): pays for autoregressive critique only when upstream is unsure.

    Inside a `Cascade` it runs according to its `trigger` (the `escalation_policy` routing
    trait) and inherits the rubric of the closest upstream logit gate when it has none of
    its own. Standalone, it always runs.

    Args:
        rubric: Optional evaluation rubric. If omitted, inherits the rubric from
            upstream LogitGate when executing inside a Cascade.
        backend: Backend instance or specifier URI implementing `GenerativeBackend`.
        model: Optional model identifier string for the backend.
        trigger: Activation policy for this judge layer:
            - 'on_ambiguous_or_fail': Triggers if upstream gate returns AMBIGUOUS or FAILED.
            - 'on_ambiguous_only': Triggers only on AMBIGUOUS; decisive FAILED short-circuits.
            - 'always': Always executes generative critique regardless of upstream confidence.
        max_tokens: Generation cap. At <= 80 the prompt switches to verdict-first so a
            truncated response still carries its tag.
        think: Forwarded to backends that support reasoning traces (e.g. Ollama ``think``).
        name: Identifier for this gate instance.

    Raises:
        ValueError: If trigger is not one of the allowed modes, or rubric is empty whitespace.
    """

    layer_tier = TIER_GENERATIVE
    can_fast_path = False

    def __init__(
        self,
        rubric: str | None = None,
        backend: str | GenerativeBackend | None = None,
        model: str | None = None,
        trigger: TriggerMode = "on_ambiguous_or_fail",
        max_tokens: int | None = None,
        think: bool | None = None,
        name: str = "JudgeGate",
    ) -> None:
        valid_triggers = {"on_ambiguous_or_fail", "on_ambiguous_only", "always"}
        if trigger not in valid_triggers:
            raise ConfigurationError(f"Invalid trigger '{trigger}'. Must be one of {sorted(valid_triggers)}")
        if rubric is not None and not rubric.strip():
            raise ConfigurationError("rubric cannot be empty or whitespace")

        effective_model = model or get_default_model()

        self.rubric = rubric.strip() if rubric is not None else None
        self.backend = resolve_generative_backend(backend, model=model)
        self.model = effective_model
        self.escalation_policy = trigger
        self.max_tokens = max_tokens
        self.think = think
        self.name = name
        self._probe_gate: LogitGate | None = None

    @property
    def trigger(self) -> TriggerMode:
        """Alias for `escalation_policy`, the routing trait `Cascade` reads."""
        return self.escalation_policy

    @trigger.setter
    def trigger(self, value: TriggerMode) -> None:
        self.escalation_policy = value

    def __repr__(self) -> str:
        rubric_repr = (
            repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
            if self.rubric
            else "None (inherited)"
        )
        max_tok_str = f", max_tokens={self.max_tokens}" if self.max_tokens is not None else ""
        return f"JudgeGate(rubric={rubric_repr}, trigger='{self.trigger}'{max_tok_str}, backend={self.backend!r})"

    def _compile_prompt(self, candidate: Any, rubric_override: str | None = None) -> str:
        effective_rubric = (self.rubric or rubric_override or "Evaluate if the artifact complies with standards.").strip()
        if self.max_tokens is not None and self.max_tokens <= 80:
            return (
                f"[EVALUATION RUBRIC]\n{effective_rubric}\n\n"
                f"[CANDIDATE ARTIFACT]\n{render_candidate(candidate)}\n\n"
                f"[INSTRUCTIONS]\n"
                f"Begin your response on the very first line with your verdict tag: [VERDICT: PASS] or [VERDICT: FAIL]\n"
                f"Then provide a concise 1-2 sentence diagnostic critique explaining why the candidate artifact passes or fails the rubric, and end by repeating [VERDICT: PASS] or [VERDICT: FAIL]."
            )
        return (
            f"[EVALUATION RUBRIC]\n{effective_rubric}\n\n"
            f"[CANDIDATE ARTIFACT]\n{render_candidate(candidate)}\n\n"
            f"[INSTRUCTIONS]\n"
            f"First, concisely verify each requirement or calculation in the candidate artifact against the rubric step-by-step (in 2-4 concise bullet points).\n"
            f"After completing your verification, conclude on the very last line with your verdict tag: [VERDICT: PASS] or [VERDICT: FAIL]."
        )

    # -- Verdict parsing -------------------------------------------------------------------
    # Precedence: explicit tag/JSON  ->  1-token logit probe on the critique  ->  keyword
    # heuristics  ->  FAIL. The probe sits above the keyword regexes on purpose: when a critique
    # is cut off by max_tokens before its tag, a calibrated read of the text beats matching
    # words like "passes" that may describe data flow rather than the verdict.

    def _probe(self) -> LogitGate | None:
        backend = self.backend
        if not isinstance(backend, LogitBackend):
            return None
        if isinstance(backend, StubBackend) and backend.logprob_responses is None:
            return None  # The default stub would answer every probe with a fixed P=0.95.
        if self._probe_gate is None:
            self._probe_gate = LogitGate(
                rubric=_PROBE_RUBRIC,
                backend=backend,
                model=self.model,
                symmetric_debiasing=False,
            )
        return self._probe_gate

    @staticmethod
    def _from_probe(probe_verdict: Verdict) -> ParsedVerdict | None:
        if probe_verdict.probability is None:
            return None
        is_pass = probe_verdict.probability >= 0.5
        status = DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED
        return is_pass, status, probe_verdict.probability

    def _parse_verdict(self, text: str) -> ParsedVerdict:
        structured = _extract_structured_verdict(text)
        if structured is not None:
            return structured
        probe = self._probe()
        if probe is not None:
            try:
                probed = self._from_probe(probe.evaluate(text))
            except (ValueError, RuntimeError, OSError, httpx.HTTPError):
                probed = None
            if probed is not None:
                return probed
        return _extract_keyword_verdict(text) or _FAIL

    async def _aparse_verdict(self, text: str) -> ParsedVerdict:
        structured = _extract_structured_verdict(text)
        if structured is not None:
            return structured
        probe = self._probe()
        if probe is not None:
            try:
                probed = self._from_probe(await probe.aevaluate(text))
            except (ValueError, RuntimeError, OSError, httpx.HTTPError):
                probed = None
            if probed is not None:
                return probed
        return _extract_keyword_verdict(text) or _FAIL

    # -- Evaluation ------------------------------------------------------------------------

    def _prepare(self, candidate: Any, rubric_override: str | None, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = self._compile_prompt(candidate, rubric_override=rubric_override)
        gen_kwargs = dict(kwargs)
        if self.think is not None and "think" not in gen_kwargs:
            gen_kwargs["think"] = self.think
        return prompt, gen_kwargs

    def _build_verdict(self, text: str, tokens: int, parsed: ParsedVerdict, t0: float) -> Verdict:
        passed, status, score = parsed
        latency = (time.perf_counter() - t0) * 1000.0
        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            tokens_generated=tokens,
            diagnostic_report=text,
        )
        return Verdict(
            passed=passed,
            status=status,
            score=score,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=tokens,
            reason=text,
            traces=[trace],
        )

    def evaluate(self, candidate: Any, rubric_override: str | None = None, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        prompt, gen_kwargs = self._prepare(candidate, rubric_override, kwargs)
        text, tokens = self.backend.generate(prompt, max_tokens=self.max_tokens, **gen_kwargs)
        return self._build_verdict(text, tokens, self._parse_verdict(text), t0)

    async def aevaluate(self, candidate: Any, rubric_override: str | None = None, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        prompt, gen_kwargs = self._prepare(candidate, rubric_override, kwargs)
        text, tokens = await self.backend.agenerate(prompt, max_tokens=self.max_tokens, **gen_kwargs)
        return self._build_verdict(text, tokens, await self._aparse_verdict(text), t0)
