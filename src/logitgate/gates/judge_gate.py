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
            f"Provide a concise diagnostic critique evaluating whether the candidate artifact meets the rubric.\n"
            f"Identify any defects, missing requirements, or compliance issues.\n"
            f"End your review with exactly one verdict tag:\n"
            f"[VERDICT: PASS] or [VERDICT: FAIL]"
        )

    def _parse_verdict(self, text: str) -> tuple[bool, DecisionStatus, Optional[float]]:
        # 1. Standard tag check
        match = re.search(r"\[VERDICT:\s*(PASS|FAIL)\]", text, re.IGNORECASE)
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

        # 3. Simple heuristic fallback
        upper = text.upper()
        if "FAIL" in upper:
            return False, DecisionStatus.FAILED, 0.0
        if "PASS" in upper:
            return True, DecisionStatus.PASSED, 1.0

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
