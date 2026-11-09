from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx

from logitgate.backends.base import (
    EXCLUDED_INTERNAL_KWARGS,
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    resolve_generative_backend,
    strip_reasoning_blocks,
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

# Matches explicit bracketed [VERDICT: PASS], **[VERDICT: PASS]**, [**VERDICT: PASS**].
_BRACKETED_VERDICT_TAG_RE = re.compile(
    r"\[\s*\*{0,2}\s*VERDICT\s*\*{0,2}\s*:\s*\*{0,2}\s*(PASS(?:ED)?|FAIL(?:ED)?)\b[^\]]*\]",
    re.IGNORECASE,
)
# Matches unbracketed **Verdict:** FAIL, Verdict: PASS.
_VERDICT_TAG_RE = re.compile(
    r"\[?\*?\*?VERDICT[\s\*\]]*:[\s\*\[]*(PASS(?:ED)?|FAIL(?:ED)?)\b[\*\]]*",
    re.IGNORECASE,
)
_NEGATED_VERDICT_PREFIX_RE = re.compile(
    r"(?:\b(?:not|no|never|without|cannot|can't|couldn't|shouldn't|wouldn't|won't|isn't|aren't|wasn't|weren't)\s+|\b(?:is|are|was|were|be|been)\s+not\s+)(?:a\s+|an\s+|the\s+)?$",
    re.IGNORECASE,
)
# "did not fail", "without any failures", "zero rejected": negated failure words must not count as FAIL.
_NEGATED_FAILURE_RE = re.compile(
    r"\b(not|no|without|never|doesn't|does\s+not|did\s+not|didn't|cannot|can't|couldn't|could\s+not|"
    r"wouldn't|would\s+not|shouldn't|should\s+not|won't|will\s+not|isn't|is\s+not|"
    r"wasn't|was\s+not|aren't|are\s+not|weren't|were\s+not|hasn't|has\s+not|haven't|have\s+not|zero)\s+"
    r"(?:any\s+|to\s+)?(?:fail|failed|fails|failure|failures|reject|rejected|rejects)\b",
    re.IGNORECASE,
)
# "did not pass", "not approved", "failed to pass": negated pass words must count as FAIL, not PASS.
_NEGATED_PASS_RE = re.compile(
    r"\b(not|no|without|never|doesn't|does\s+not|did\s+not|didn't|cannot|can't|couldn't|could\s+not|"
    r"wouldn't|would\s+not|shouldn't|should\s+not|won't|will\s+not|isn't|is\s+not|wasn't|was\s+not|"
    r"aren't|are\s+not|weren't|were\s+not|hasn't|has\s+not|haven't|have\s+not|mustn't|must\s+not|"
    r"failed\s+to|fails\s+to)\s+(?:fully\s+|strictly\s+)?(?:pass|passed|passes|approve|approved)\b",
    re.IGNORECASE,
)
_PASS_WORD_RE = re.compile(r"\b(PASS|PASSED|PASSES|APPROVE|APPROVED)\b", re.IGNORECASE)
_FAIL_WORD_RE = re.compile(r"\b(FAIL|FAILED|FAILS|FAILURE|REJECT|REJECTED)\b", re.IGNORECASE)
_TRUTHY_STRINGS = frozenset({"pass", "passed", "passes", "approve", "approved", "true", "yes", "1"})
_FALSY_STRINGS = frozenset({"fail", "failed", "fails", "failure", "reject", "rejected", "false", "no", "0"})
_VALID_TRIGGERS = frozenset({"on_ambiguous_or_fail", "on_ambiguous_only", "adjudicate", "always"})
_PROBE_RUBRIC = (
    "Based on this review critique, did the candidate artifact satisfy all requirements "
    "and PASS the evaluation without any defects or errors?"
)
_MAX_JSON_DECODE_ATTEMPTS = 128


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yields every top-level JSON object embedded in free text, including nested ones.

    A non-greedy ``\\{.*?\\}`` regex truncates ``{"verdict": "pass", "detail": {...}}`` at the
    first closing brace. `raw_decode` parses balanced objects correctly.
    """
    decoder = json.JSONDecoder()
    idx = text.find("{")
    attempts = 0
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, RecursionError):
            attempts += 1
            if attempts >= _MAX_JSON_DECODE_ATTEMPTS:
                break
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        idx = text.find("{", end)


def _is_valid_verdict_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and value in (0, 1):
        return True
    if isinstance(value, str):
        return value.strip().lower() in (_TRUTHY_STRINGS | _FALSY_STRINGS)
    return False


def _coerce_pass(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    return bool(value)


def _extract_structured_verdict(text: str) -> ParsedVerdict | None:
    """Reads an explicit verdict the judge was asked to emit: a ``[VERDICT: ...]`` tag or JSON.

    The *last* tag wins because the prompt asks for reasoning first and the verdict last, so
    an early "PASS" inside the reasoning bullets must not override the conclusion. Explicit
    bracketed ``[VERDICT: PASS/FAIL]`` tags take precedence over unbracketed ``Verdict: ...``
    phrases so prose discussing ``Verdict: fail`` in bullet points cannot override a closing
    bracketed tag or vice versa.
    """
    bracketed_tags = _BRACKETED_VERDICT_TAG_RE.findall(text)
    if bracketed_tags:
        return _PASS if bracketed_tags[-1].upper().startswith("PASS") else _FAIL

    unbracketed_tags: list[str] = []
    for match in _VERDICT_TAG_RE.finditer(text):
        prefix = text[max(0, match.start() - 40) : match.start()]
        if _NEGATED_VERDICT_PREFIX_RE.search(prefix):
            continue
        unbracketed_tags.append(match.group(1))
    if unbracketed_tags:
        return _PASS if unbracketed_tags[-1].upper().startswith("PASS") else _FAIL

    # Only JSON that actually carries a recognized verdict value counts. A critique that quotes
    # the candidate's own JSON (e.g. '{"pass": "secret"}') must not be read as a decisive FAIL.
    verdict_objects: list[tuple[dict[str, Any], str]] = []
    for obj in _iter_json_objects(text):
        lowered = {str(k).lower(): v for k, v in obj.items()}
        for key in ("pass", "passed", "verdict"):
            if key in lowered and _is_valid_verdict_value(lowered[key]):
                verdict_objects.append((lowered, key))
                break
    if not verdict_objects:
        return None
    data, matched_key = verdict_objects[-1]
    is_pass = _coerce_pass(data[matched_key])
    raw_score = data.get("score")
    if raw_score is not None and not isinstance(raw_score, bool):
        try:
            score = float(raw_score)
            if not math.isfinite(score):
                score = 1.0 if is_pass else 0.0
        except (TypeError, ValueError):
            score = 1.0 if is_pass else 0.0
    else:
        score = 1.0 if is_pass else 0.0
    return is_pass, DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED, score


def _keyword_signal(text: str, *, tie_break: bool) -> ParsedVerdict | None:
    sanitized = _NEGATED_FAILURE_RE.sub(" ", text)
    # Replace negated pass phrases ("did not pass", "not approved") with "FAIL" so they count as failure
    sanitized = _NEGATED_PASS_RE.sub(" FAIL ", sanitized)
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
            - 'on_ambiguous_or_fail': Triggers if upstream gate returns AMBIGUOUS or FAILED (diagnostic only on FAILED unless allow_fail_override=True).
            - 'on_ambiguous_only': Triggers only on AMBIGUOUS; decisive FAILED short-circuits.
            - 'adjudicate': Triggers on AMBIGUOUS or FAILED, allowing the judge to overturn Tier-1 false negatives.
            - 'always': Always executes generative critique regardless of upstream confidence.
        allow_fail_override: When True (or when trigger='adjudicate'), allows JudgeGate to
            overturn a Tier-1 FAILED verdict with PASSED.
        max_tokens: Generation cap. At <= 80 the prompt switches to verdict-first so a
            truncated response still carries its tag.
        think: Forwarded to backends that support reasoning traces (e.g. Ollama ``think``).
        name: Identifier for this gate instance.

    Raises:
        ConfigurationError: If trigger is not one of the allowed modes, or rubric is empty whitespace.
    """

    layer_tier = TIER_GENERATIVE
    can_fast_path = False

    def __init__(
        self,
        rubric: str | None = None,
        backend: str | GenerativeBackend | None = None,
        model: str | None = None,
        trigger: TriggerMode = "on_ambiguous_only",
        max_tokens: int | None = None,
        think: bool | None = None,
        save_on_fail: bool | None = None,
        allow_fail_override: bool = False,
        name: str = "JudgeGate",
    ) -> None:
        if trigger not in _VALID_TRIGGERS:
            raise ConfigurationError(f"Invalid trigger '{trigger}'. Must be one of {sorted(_VALID_TRIGGERS)}")
        if rubric is not None and not rubric.strip():
            raise ConfigurationError("rubric cannot be empty or whitespace")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            raise ConfigurationError(f"max_tokens must be a positive integer, got {max_tokens!r}")

        self.rubric = rubric.strip() if rubric is not None else None
        self.backend = resolve_generative_backend(backend, model=model)
        effective_model = (
            model
            or getattr(self.backend, "model", None)
            or getattr(self.backend, "model_name", None)
            or get_default_model()
        )
        self.model = effective_model
        self.escalation_policy = trigger
        self.allow_fail_override = bool(allow_fail_override)
        self.can_override_fail = bool(allow_fail_override) or (trigger == "adjudicate")
        self.max_tokens = max_tokens
        self.think = think
        self.save_on_fail = save_on_fail
        self.name = name
        self._probe_gate: LogitGate | None = None

    @property
    def trigger(self) -> TriggerMode:
        """Alias for `escalation_policy`, the routing trait `Cascade` reads."""
        return self.escalation_policy

    @trigger.setter
    def trigger(self, value: TriggerMode) -> None:
        if value not in _VALID_TRIGGERS:
            raise ConfigurationError(f"Invalid trigger '{value}'. Must be one of {sorted(_VALID_TRIGGERS)}")
        self.escalation_policy = value
        self.can_override_fail = self.allow_fail_override or (value == "adjudicate")

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
        inner = getattr(backend, "inner", backend)
        if not isinstance(backend, LogitBackend):
            return None
        if isinstance(inner, StubBackend) and inner.logprob_responses is None:
            return None  # The default stub would answer every probe with a fixed P=0.95.
        if self._probe_gate is None:
            self._probe_gate = LogitGate(
                rubric=_PROBE_RUBRIC,
                backend=backend,
                model=self.model,
                symmetric_debiasing=False,
                save_on_fail=False,
            )
        return self._probe_gate

    @staticmethod
    def _from_probe(probe_verdict: Verdict, text: str = "") -> ParsedVerdict | None:
        if probe_verdict.probability is None:
            return None
        if probe_verdict.status == DecisionStatus.AMBIGUOUS and text:
            kw = _extract_keyword_verdict(text)
            if kw is not None:
                return kw
        is_pass = probe_verdict.probability >= 0.5
        status = DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED
        return is_pass, status, probe_verdict.probability

    def _parse_verdict(self, text: str) -> ParsedVerdict:
        cleaned = strip_reasoning_blocks(text) if text else ""
        if not cleaned or not cleaned.strip():
            return _FAIL
        structured = _extract_structured_verdict(cleaned)
        if structured is not None:
            return structured
        probe = self._probe()
        if probe is not None:
            try:
                probed = self._from_probe(probe.evaluate(cleaned), cleaned)
            except (ValueError, RuntimeError, OSError, httpx.HTTPError):
                probed = None
            if probed is not None:
                return probed
        return _extract_keyword_verdict(cleaned) or _FAIL

    async def _aparse_verdict(self, text: str) -> ParsedVerdict:
        cleaned = strip_reasoning_blocks(text) if text else ""
        if not cleaned or not cleaned.strip():
            return _FAIL
        structured = _extract_structured_verdict(cleaned)
        if structured is not None:
            return structured
        probe = self._probe()
        if probe is not None:
            try:
                probed = self._from_probe(await probe.aevaluate(cleaned), cleaned)
            except (ValueError, RuntimeError, OSError, httpx.HTTPError):
                probed = None
            if probed is not None:
                return probed
        return _extract_keyword_verdict(cleaned) or _FAIL

    # -- Evaluation ------------------------------------------------------------------------

    def _prepare(self, candidate: Any, rubric_override: str | None, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = self._compile_prompt(candidate, rubric_override=rubric_override)
        gen_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDED_INTERNAL_KWARGS}
        if self.think is not None and "think" not in gen_kwargs:
            gen_kwargs["think"] = self.think
        return prompt, gen_kwargs

    def _build_verdict(
        self,
        text: str,
        tokens: int,
        parsed: ParsedVerdict,
        t0: float,
        *,
        candidate: Any = None,
        rubric_override: str | None = None,
    ) -> Verdict:
        passed, status, score = parsed
        latency = (time.perf_counter() - t0) * 1000.0
        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            tokens_generated=tokens,
            diagnostic_report=text,
        )
        verdict = Verdict(
            passed=passed,
            status=status,
            score=score,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=tokens,
            reason=text,
            traces=[trace],
        )
        from logitgate.snapshot import infer_backend_spec, persist_failure_snapshot

        persist_failure_snapshot(
            verdict=verdict,
            candidate=candidate,
            save_on_fail=self.save_on_fail,
            gate_type="JudgeGate",
            gate_config={
                "backend": infer_backend_spec(self.backend),
                "model": self.model,
            },
            rubric=self.rubric or rubric_override,
        )
        return verdict

    def evaluate(self, candidate: Any, rubric_override: str | None = None, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        prompt, gen_kwargs = self._prepare(candidate, rubric_override, kwargs)
        text, tokens = self.backend.generate(prompt, max_tokens=self.max_tokens, **gen_kwargs)
        return self._build_verdict(
            text, tokens, self._parse_verdict(text), t0, candidate=candidate, rubric_override=rubric_override
        )

    async def aevaluate(self, candidate: Any, rubric_override: str | None = None, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        prompt, gen_kwargs = self._prepare(candidate, rubric_override, kwargs)
        text, tokens = await self.backend.agenerate(prompt, max_tokens=self.max_tokens, **gen_kwargs)
        return self._build_verdict(
            text, tokens, await self._aparse_verdict(text), t0, candidate=candidate, rubric_override=rubric_override
        )
