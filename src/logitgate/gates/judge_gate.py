from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, Optional

from logitgate.backends.base import GenerativeBackend, resolve_generative_backend
from logitgate.config import get_default_backend, get_default_model
from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict

TriggerMode = Literal["on_ambiguous_or_fail", "on_ambiguous_only", "always"]


class JudgeGate(BaseGate):
    """Generative System-2 Reasoning Judge (Layer 2).

    Generates actionable diagnostic critiques and explicit failure explanations
    using an autoregressive LLM only when triggered by upstream gate ambiguity or failure.

    Args:
        rubric: Optional evaluation rubric. If omitted, inherits the rubric from
            upstream LogitGate when executing inside a Cascade.
        backend: Backend instance or specifier URI implementing `GenerativeBackend`.
        model: Optional model identifier string for the backend.
        trigger: Activation policy for this judge layer:
            - 'on_ambiguous_or_fail': Triggers if upstream gate returns AMBIGUOUS or FAILED.
            - 'on_ambiguous_only': Triggers only on AMBIGUOUS; decisive FAILED short-circuits.
            - 'always': Always executes generative critique regardless of upstream confidence.
        name: Identifier for this gate instance.

    Raises:
        ValueError: If trigger is not one of the allowed modes, or rubric is empty whitespace.
    """

    def __init__(
        self,
        rubric: Optional[str] = None,
        backend: Optional[str | GenerativeBackend] = None,
        model: Optional[str] = None,
        trigger: TriggerMode = "on_ambiguous_or_fail",
        max_tokens: Optional[int] = None,
        name: str = "JudgeGate",
    ) -> None:
        valid_triggers = {"on_ambiguous_or_fail", "on_ambiguous_only", "always"}
        if trigger not in valid_triggers:
            raise ValueError(f"Invalid trigger '{trigger}'. Must be one of {sorted(valid_triggers)}")
        if rubric is not None and not rubric.strip():
            raise ValueError("rubric cannot be empty or whitespace")

        effective_backend = (
            get_default_backend() if (backend is None or backend == "default") else backend
        )
        effective_model = model or get_default_model()

        self.rubric = rubric.strip() if rubric is not None else None
        self.backend = resolve_generative_backend(effective_backend, model=effective_model)
        self.model = effective_model
        self.trigger = trigger
        self.max_tokens = max_tokens
        self.name = name

    def __repr__(self) -> str:
        rubric_repr = (
            repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
            if self.rubric
            else "None (inherited)"
        )
        max_tok_str = f", max_tokens={self.max_tokens}" if self.max_tokens is not None else ""
        return f"JudgeGate(rubric={rubric_repr}, trigger='{self.trigger}'{max_tok_str}, backend={self.backend!r})"

    def _compile_prompt(self, candidate: Any, rubric_override: Optional[str] = None) -> str:
        effective_rubric = (rubric_override or self.rubric or "Evaluate if the artifact complies with standards.").strip()
        return (
            f"[EVALUATION RUBRIC]\n{effective_rubric}\n\n"
            f"[CANDIDATE ARTIFACT]\n{str(candidate).strip()}\n\n"
            f"[INSTRUCTIONS]\n"
            f"Begin your response on the very first line with your verdict tag: [VERDICT: PASS] or [VERDICT: FAIL]\n"
            f"Then provide a concise diagnostic critique explaining why the candidate artifact passes or fails the rubric, and end by repeating [VERDICT: PASS] or [VERDICT: FAIL]."
        )

    def _parse_verdict(self, text: str) -> tuple[bool, DecisionStatus, Optional[float]]:
        # 1. Standard / Markdown tag check (matches [VERDICT: PASS], **[VERDICT: PASS]**, Verdict: PASS, etc.)
        match = re.search(r"\[?\*?\*?VERDICT[\s\*\]]*:[\s\*\[]*(PASS|FAIL)[\*\]]*", text, re.IGNORECASE)
        if match:
            tag = match.group(1).upper()
            if tag == "PASS":
                return True, DecisionStatus.PASSED, 1.0
            return False, DecisionStatus.FAILED, 0.0

        # 2. JSON check
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                verdict_val = str(data.get("verdict", "")).lower()
                is_pass = data.get("pass", data.get("passed", verdict_val == "pass"))
                score = float(data.get("score", 1.0 if is_pass else 0.0))
                return bool(is_pass), DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED, score
            except Exception:
                pass

        # 3. Explicit keyword heuristic on candidate lines (first line and last non-empty line)
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        candidate_lines = []
        if lines:
            candidate_lines.append(lines[0])
            if len(lines) > 1:
                candidate_lines.append(lines[-1])

        def sanitize_negations(s: str) -> str:
            neg_pattern = r"\b(not|no|without|never|doesn't|does\s+not|did\s+not|didn't|zero)\s+(?:any\s+|to\s+)?(?:fail|failed|failure|failures|reject|rejected)\b"
            return re.sub(neg_pattern, " ", s, flags=re.IGNORECASE)

        pass_re = re.compile(r"\b(PASS|PASSED|PASSES|APPROVE|APPROVED)\b", re.IGNORECASE)
        fail_re = re.compile(r"\b(FAIL|FAILED|FAILS|FAILURE|REJECT|REJECTED)\b", re.IGNORECASE)

        for line in candidate_lines:
            sanitized = sanitize_negations(line)
            has_fail = bool(fail_re.search(sanitized))
            has_pass = bool(pass_re.search(sanitized))
            if has_pass and not has_fail:
                return True, DecisionStatus.PASSED, 1.0
            if has_fail and not has_pass:
                return False, DecisionStatus.FAILED, 0.0
            if has_fail and has_pass:
                p_match = pass_re.search(sanitized)
                f_match = fail_re.search(sanitized)
                if p_match and f_match:
                    if p_match.start() < f_match.start():
                        return True, DecisionStatus.PASSED, 1.0
                    return False, DecisionStatus.FAILED, 0.0

        # 4. Eating Our Own Dogfood: If first/last lines are ambiguous or missing explicit keywords,
        # and backend supports LogitBackend, probe the full critique text with a 1-token LogitGate!
        from logitgate.backends.base import LogitBackend, StubBackend
        if isinstance(self.backend, LogitBackend) and not (isinstance(self.backend, StubBackend) and self.backend.logprob_responses is None):
            try:
                from logitgate.gates.logit_gate import LogitGate
                probe_gate = LogitGate(
                    rubric="Does this review critique conclude that the candidate artifact PASSED the evaluation?",
                    backend=self.backend,
                    model=self.model,
                    symmetric_debiasing=False,
                )
                pv = probe_gate.evaluate(text)
                if pv.probability is not None:
                    is_pass = (pv.probability >= 0.5)
                    return is_pass, DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED, pv.probability
            except Exception:
                pass

        # 5. Full text fallback heuristic
        sanitized_full = sanitize_negations(text)
        has_fail = bool(fail_re.search(sanitized_full))
        has_pass = bool(pass_re.search(sanitized_full))
        if has_pass and not has_fail:
            return True, DecisionStatus.PASSED, 1.0
        if has_fail and not has_pass:
            return False, DecisionStatus.FAILED, 0.0

        return False, DecisionStatus.FAILED, 0.0

    async def _aparse_verdict(self, text: str) -> tuple[bool, DecisionStatus, Optional[float]]:
        # 1. Standard / Markdown tag check
        match = re.search(r"\[?\*?\*?VERDICT[\s\*\]]*:[\s\*\[]*(PASS|FAIL)[\*\]]*", text, re.IGNORECASE)
        if match:
            tag = match.group(1).upper()
            if tag == "PASS":
                return True, DecisionStatus.PASSED, 1.0
            return False, DecisionStatus.FAILED, 0.0

        # 2. JSON check
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                verdict_val = str(data.get("verdict", "")).lower()
                is_pass = data.get("pass", data.get("passed", verdict_val == "pass"))
                score = float(data.get("score", 1.0 if is_pass else 0.0))
                return bool(is_pass), DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED, score
            except Exception:
                pass

        # 3. Candidate lines check
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        candidate_lines = []
        if lines:
            candidate_lines.append(lines[0])
            if len(lines) > 1:
                candidate_lines.append(lines[-1])

        def sanitize_negations(s: str) -> str:
            neg_pattern = r"\b(not|no|without|never|doesn't|does\s+not|did\s+not|didn't|zero)\s+(?:any\s+|to\s+)?(?:fail|failed|failure|failures|reject|rejected)\b"
            return re.sub(neg_pattern, " ", s, flags=re.IGNORECASE)

        pass_re = re.compile(r"\b(PASS|PASSED|PASSES|APPROVE|APPROVED)\b", re.IGNORECASE)
        fail_re = re.compile(r"\b(FAIL|FAILED|FAILS|FAILURE|REJECT|REJECTED)\b", re.IGNORECASE)

        for line in candidate_lines:
            sanitized = sanitize_negations(line)
            has_fail = bool(fail_re.search(sanitized))
            has_pass = bool(pass_re.search(sanitized))
            if has_pass and not has_fail:
                return True, DecisionStatus.PASSED, 1.0
            if has_fail and not has_pass:
                return False, DecisionStatus.FAILED, 0.0
            if has_fail and has_pass:
                p_match = pass_re.search(sanitized)
                f_match = fail_re.search(sanitized)
                if p_match and f_match:
                    if p_match.start() < f_match.start():
                        return True, DecisionStatus.PASSED, 1.0
                    return False, DecisionStatus.FAILED, 0.0

        # 4. Eating Our Own Dogfood: Asynchronous LogitGate probe
        from logitgate.backends.base import LogitBackend, StubBackend
        if isinstance(self.backend, LogitBackend) and not (isinstance(self.backend, StubBackend) and self.backend.logprob_responses is None):
            try:
                from logitgate.gates.logit_gate import LogitGate
                probe_gate = LogitGate(
                    rubric="Does this review critique conclude that the candidate artifact PASSED the evaluation?",
                    backend=self.backend,
                    model=self.model,
                    symmetric_debiasing=False,
                )
                pv = await probe_gate.aevaluate(text)
                if pv.probability is not None:
                    is_pass = (pv.probability >= 0.5)
                    return is_pass, DecisionStatus.PASSED if is_pass else DecisionStatus.FAILED, pv.probability
            except Exception:
                pass

        # 5. Full text fallback heuristic
        sanitized_full = sanitize_negations(text)
        has_fail = bool(fail_re.search(sanitized_full))
        has_pass = bool(pass_re.search(sanitized_full))
        if has_pass and not has_fail:
            return True, DecisionStatus.PASSED, 1.0
        if has_fail and not has_pass:
            return False, DecisionStatus.FAILED, 0.0

        return False, DecisionStatus.FAILED, 0.0

    def evaluate(self, candidate: Any, rubric_override: Optional[str] = None, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        prompt = self._compile_prompt(candidate, rubric_override=rubric_override)
        text, tokens = self.backend.generate(prompt, max_tokens=self.max_tokens)
        latency = (time.perf_counter() - t0) * 1000.0

        passed, status, score = self._parse_verdict(text)
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

    async def aevaluate(self, candidate: Any, rubric_override: Optional[str] = None, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        prompt = self._compile_prompt(candidate, rubric_override=rubric_override)
        text, tokens = await self.backend.agenerate(prompt, max_tokens=self.max_tokens)
        latency = (time.perf_counter() - t0) * 1000.0

        passed, status, score = await self._aparse_verdict(text)
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
