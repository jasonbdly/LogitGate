from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from logitgate.backends.base import LogitBackend
from logitgate.gates.logit_gate import ChoiceGate, LogitGate, ScoreGate

# Graceful optional imports for DeepEval
try:
    from deepeval.metrics import BaseMetric
    from deepeval.models import DeepEvalBaseSystemOneModel
    from deepeval.models.system_one.schema import (
        ChoiceAnswer,
        ChoiceQuestion,
        NoulAnswer,
        NoulQuestion,
        ScoreAnswer,
        ScoreQuestion,
        SystemOneAnswers,
        SystemOneQuestion,
    )
    from deepeval.test_case import LLMTestCase
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False

    class BaseMetric:  # type: ignore[no-redef]
        def __init__(self) -> None:
            pass

    class DeepEvalBaseSystemOneModel:  # type: ignore[no-redef]
        def __init__(self, model: str = "default") -> None:
            self.model = model

    class NoulAnswer:  # type: ignore[no-redef]
        def __init__(self, probability: float = 0.5) -> None:
            self.probability = probability

    class ChoiceAnswer:  # type: ignore[no-redef]
        def __init__(
            self,
            choice: str = "A",
            probabilities: dict[str, float] | None = None,
            confidence: float = 1.0,
        ) -> None:
            self.choice = choice
            self.probabilities = probabilities or {}
            self.confidence = confidence

    class ScoreAnswer:  # type: ignore[no-redef]
        def __init__(
            self,
            score: float = 3.0,
            probabilities: dict[Any, float] | None = None,
            confidence: float = 1.0,
        ) -> None:
            self.score = score
            self.probabilities = probabilities or {}
            self.confidence = confidence

    class NoulQuestion:  # type: ignore[no-redef]
        def __init__(self, instructions: str = "", statement: str = "") -> None:
            self.instructions = instructions or statement
            self.statement = statement or instructions

    class ChoiceQuestion:  # type: ignore[no-redef]
        def __init__(self, question: str = "", options: dict[str, str] | None = None) -> None:
            self.question = question
            self.options = options or {}

    class ScoreQuestion:  # type: ignore[no-redef]
        def __init__(self, rubric: str = "", min_score: int = 1, max_score: int = 5) -> None:
            self.rubric = rubric
            self.min_score = min_score
            self.max_score = max_score

    class SystemOneAnswers:  # type: ignore[no-redef]
        def __init__(
            self,
            nouls: dict[str, Any] | None = None,
            choices: dict[str, Any] | None = None,
            scores: dict[str, Any] | None = None,
        ) -> None:
            self.nouls = nouls or {}
            if choices is not None:
                self.choices = choices
            if scores is not None:
                self.scores = scores

    class SystemOneQuestion(NoulQuestion):  # type: ignore[no-redef]
        pass

    class LLMTestCase:  # type: ignore[no-redef]
        def __init__(self, input: str = "", actual_output: str = "") -> None:
            self.input = input
            self.actual_output = actual_output


class LogitGateSystemOneModel(DeepEvalBaseSystemOneModel):
    """
    DeepEval System-1 model powered by LogitGate debiased first-token logits.
    Drop-in System-1 judge model for DeepEval JevEval metrics.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        backend: str | LogitBackend | None = None,
        symmetric: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
    ) -> None:
        self.api_base = api_base or "http://localhost:11434"
        self.symmetric = symmetric
        self.logit_temperature = logit_temperature

        if backend is not None:
            from logitgate.backends.base import resolve_logit_backend
            from logitgate.config import get_default_model

            self.backend = resolve_logit_backend(backend, model=model_name)
            self.model_name = (
                model_name
                or getattr(self.backend, "model", None)
                or getattr(self.backend, "model_name", None)
                or get_default_model()
                or "default"
            )
        elif api_base is not None:
            from logitgate.backends.ollama import OllamaBackend
            from logitgate.config import get_default_model

            self.model_name = model_name or get_default_model() or "qwen2.5:7b"
            self.backend = OllamaBackend(base_url=api_base, model=self.model_name)
        else:
            import os

            from logitgate.backends.base import resolve_logit_backend
            from logitgate.backends.ollama import OllamaBackend
            from logitgate.config import LogitGateDefaults, get_default_model

            has_global_backend = (
                LogitGateDefaults.backend is not None
                or bool(os.environ.get("LOGITGATE_BACKEND"))
            )
            if has_global_backend:
                self.backend = resolve_logit_backend(None, model=model_name)
                self.model_name = (
                    model_name
                    or getattr(self.backend, "model", None)
                    or getattr(self.backend, "model_name", None)
                    or get_default_model()
                    or "default"
                )
            else:
                self.model_name = model_name or get_default_model() or "qwen2.5:7b"
                self.backend = OllamaBackend(base_url=self.api_base, model=self.model_name)
        super().__init__(model=self.model_name)

    def load_model(self) -> None:
        return None

    def get_model_name(self) -> str:
        return f"logitgate:{self.model_name}"

    @staticmethod
    def _extract_statement(q: Any) -> str:
        raw: Any = None
        if isinstance(q, Mapping):
            raw = q.get("statement") or q.get("instructions") or q.get("question") or q.get("rubric")
        else:
            raw = (
                getattr(q, "statement", None)
                or getattr(q, "instructions", None)
                or getattr(q, "question", None)
                or getattr(q, "rubric", None)
            )
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        fallback = str(q).strip()
        return fallback if fallback else "Evaluate candidate against rubric."

    @staticmethod
    def _extract_options(q: Any) -> Mapping[str, str] | None:
        opts = q.get("options") if isinstance(q, Mapping) else getattr(q, "options", None)
        if isinstance(opts, Mapping) and len(opts) >= 2:
            return opts
        return None

    @staticmethod
    def _extract_score_bounds(q: Any) -> tuple[int, int] | None:
        if isinstance(q, ScoreQuestion):
            return int(getattr(q, "min_score", 1)), int(getattr(q, "max_score", 5))
        if isinstance(q, Mapping) and ("min_score" in q or "max_score" in q):
            return int(q.get("min_score", 1)), int(q.get("max_score", 5))
        if hasattr(q, "min_score") and hasattr(q, "max_score"):
            return int(q.min_score), int(q.max_score)
        return None

    def decide(
        self,
        state: Any,
        questions: dict[str, Any],
    ) -> tuple[Any, float | None]:
        nouls: dict[str, Any] = {}
        choices: dict[str, Any] = {}
        scores: dict[str, Any] = {}

        binary_qids: list[str] = []
        for q_id, q in questions.items():
            opts = self._extract_options(q)
            score_bounds = self._extract_score_bounds(q)
            if opts is not None:
                c_gate = ChoiceGate(
                    question=self._extract_statement(q),
                    options=opts,
                    backend=self.backend,
                    model=self.model_name,
                    symmetric_debiasing=self.symmetric,
                    logit_temperature=self.logit_temperature,
                )
                v = c_gate.evaluate(state)
                choices[q_id] = ChoiceAnswer(
                    choice=v.choice or next(iter(opts)),
                    probabilities=v.choice_probabilities or {},
                    confidence=v.probability if v.probability is not None else 1.0,
                )
            elif score_bounds is not None:
                min_s, max_s = score_bounds
                s_gate = ScoreGate(
                    rubric=self._extract_statement(q),
                    min_score=min_s,
                    max_score=max_s,
                    threshold=float(min_s),
                    backend=self.backend,
                    model=self.model_name,
                    symmetric_debiasing=self.symmetric,
                    logit_temperature=self.logit_temperature,
                    save_on_fail=False,
                )
                v = s_gate.evaluate(state, save_on_fail=False)
                raw_tok_probs = v.traces[-1].metadata.get("token_probabilities", {}) if v.traces else {}
                int_probs = {int(k): float(val) for k, val in raw_tok_probs.items() if str(k).isdigit()}
                max_tok_prob = max(int_probs.values()) if int_probs else 1.0
                scores[q_id] = ScoreAnswer(
                    score=v.score if v.score is not None else float(min_s),
                    probabilities=int_probs,
                    confidence=round(max_tok_prob, 4),
                )
            else:
                binary_qids.append(q_id)

        if len(binary_qids) > 1:
            statements = [self._extract_statement(questions[qid]) for qid in binary_qids]
            gate = LogitGate(
                rubric=statements[0],
                backend=self.backend,
                model=self.model_name,
                symmetric_debiasing=self.symmetric,
                logit_temperature=self.logit_temperature,
                save_on_fail=False,
            )
            verdicts = gate.evaluate_batch(state, statements)
            for q_id, verdict in zip(binary_qids, verdicts):
                p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
                nouls[q_id] = NoulAnswer(probability=round(p, 4))
        elif len(binary_qids) == 1:
            q_id = binary_qids[0]
            statement = self._extract_statement(questions[q_id])
            gate = LogitGate(
                rubric=statement,
                backend=self.backend,
                model=self.model_name,
                symmetric_debiasing=self.symmetric,
                logit_temperature=self.logit_temperature,
                save_on_fail=False,
            )
            verdict = gate.evaluate(state, save_on_fail=False)
            p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
            nouls[q_id] = NoulAnswer(probability=round(p, 4))

        ans_kwargs: dict[str, Any] = {"nouls": nouls}
        if choices:
            ans_kwargs["choices"] = choices
        if scores:
            ans_kwargs["scores"] = scores
        try:
            return SystemOneAnswers(**ans_kwargs), 0.0001
        except TypeError:
            return SystemOneAnswers(nouls=nouls), 0.0001

    async def a_decide(
        self,
        state: Any,
        questions: dict[str, Any],
    ) -> tuple[Any, float | None]:
        from logitgate.backends.base import StubBackend
        from logitgate.debiasing.permutation import has_known_or_cached_temperature
        from logitgate.gates.logit_gate import (
            _lazy_calibrate_async,
            _should_lazy_auto_calibrate,
        )

        needs_precal = False
        if self.logit_temperature == "auto":
            inner = getattr(self.backend, "inner", self.backend)
            needs_precal = (
                isinstance(self.backend, LogitBackend)
                and not isinstance(inner, StubBackend)
                and bool(self.model_name and self.model_name.strip().lower() not in ("", "default", "stub"))
                and not has_known_or_cached_temperature(self.model_name, binary=True)
            )
        elif self.logit_temperature is None:
            needs_precal = _should_lazy_auto_calibrate(None, self.model_name, self.backend, binary=True)
        if needs_precal:
            await _lazy_calibrate_async(self.model_name, self.backend, binary=True)

        nouls: dict[str, Any] = {}
        choices: dict[str, Any] = {}
        scores: dict[str, Any] = {}

        binary_qids: list[str] = []
        for q_id, q in questions.items():
            opts = self._extract_options(q)
            score_bounds = self._extract_score_bounds(q)
            if opts is not None:
                c_gate = ChoiceGate(
                    question=self._extract_statement(q),
                    options=opts,
                    backend=self.backend,
                    model=self.model_name,
                    symmetric_debiasing=self.symmetric,
                    logit_temperature=self.logit_temperature,
                )
                v = await c_gate.aevaluate(state)
                choices[q_id] = ChoiceAnswer(
                    choice=v.choice or next(iter(opts)),
                    probabilities=v.choice_probabilities or {},
                    confidence=v.probability if v.probability is not None else 1.0,
                )
            elif score_bounds is not None:
                min_s, max_s = score_bounds
                s_gate = ScoreGate(
                    rubric=self._extract_statement(q),
                    min_score=min_s,
                    max_score=max_s,
                    threshold=float(min_s),
                    backend=self.backend,
                    model=self.model_name,
                    symmetric_debiasing=self.symmetric,
                    logit_temperature=self.logit_temperature,
                    save_on_fail=False,
                )
                v = await s_gate.aevaluate(state, save_on_fail=False)
                raw_tok_probs = v.traces[-1].metadata.get("token_probabilities", {}) if v.traces else {}
                int_probs = {int(k): float(val) for k, val in raw_tok_probs.items() if str(k).isdigit()}
                max_tok_prob = max(int_probs.values()) if int_probs else 1.0
                scores[q_id] = ScoreAnswer(
                    score=v.score if v.score is not None else float(min_s),
                    probabilities=int_probs,
                    confidence=round(max_tok_prob, 4),
                )
            else:
                binary_qids.append(q_id)

        if len(binary_qids) > 1:
            statements = [self._extract_statement(questions[qid]) for qid in binary_qids]
            gate = LogitGate(
                rubric=statements[0],
                backend=self.backend,
                model=self.model_name,
                symmetric_debiasing=self.symmetric,
                logit_temperature=self.logit_temperature,
                save_on_fail=False,
            )
            verdicts = await gate.aevaluate_batch(state, statements)
            for q_id, verdict in zip(binary_qids, verdicts):
                p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
                nouls[q_id] = NoulAnswer(probability=round(p, 4))
        elif len(binary_qids) == 1:
            q_id = binary_qids[0]
            statement = self._extract_statement(questions[q_id])
            gate = LogitGate(
                rubric=statement,
                backend=self.backend,
                model=self.model_name,
                symmetric_debiasing=self.symmetric,
                logit_temperature=self.logit_temperature,
                save_on_fail=False,
            )
            verdict = await gate.aevaluate(state, save_on_fail=False)
            p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
            nouls[q_id] = NoulAnswer(probability=round(p, 4))

        ans_kwargs: dict[str, Any] = {"nouls": nouls}
        if choices:
            ans_kwargs["choices"] = choices
        if scores:
            ans_kwargs["scores"] = scores
        try:
            return SystemOneAnswers(**ans_kwargs), 0.0001
        except TypeError:
            return SystemOneAnswers(nouls=nouls), 0.0001


def _extract_test_case_candidate(test_case: Any) -> Any:
    actual = getattr(test_case, "actual_output", None)
    if actual is not None:
        return actual
    inp = getattr(test_case, "input", None)
    if inp is not None:
        return inp
    return str(test_case)


class LogitGateMetric(BaseMetric):
    """
    DeepEval metric evaluating LLMTestCase using a 50ms LogitGate.
    """

    def __init__(
        self,
        rubric: str,
        threshold: float | None = None,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        fail_threshold: float | None = None,
        symmetric_debiasing: bool | None = None,
        logit_temperature: float | Literal["auto"] | None = None,
        preset: Literal["strict", "balanced", "fast_triage"] | None = None,
        name: str = "LogitGate Metric",
        async_mode: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.rubric = rubric
        self.gate = LogitGate(
            rubric=rubric,
            pass_threshold=threshold,
            fail_threshold=fail_threshold,
            backend=backend,
            model=model,
            symmetric_debiasing=symmetric_debiasing,
            logit_temperature=logit_temperature,
            preset=preset,
            save_on_fail=False,
        )
        self.threshold = self.gate.pass_threshold
        self.async_mode = async_mode
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return self.name

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        candidate = _extract_test_case_candidate(test_case)
        verdict = self.gate.evaluate(candidate, save_on_fail=False)
        self.score = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
        self.success = verdict.passed
        self.reason = verdict.reason
        return self.score

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        candidate = _extract_test_case_candidate(test_case)
        verdict = await self.gate.aevaluate(candidate, save_on_fail=False)
        self.score = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
        self.success = verdict.passed
        self.reason = verdict.reason
        return self.score

    def is_successful(self) -> bool:
        if self.success is not None:
            return bool(self.success)
        if self.threshold is None or self.score is None:
            return False
        return self.score >= self.threshold
