from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from logitgate.backends.base import LogitBackend
from logitgate.gates.logit_gate import LogitGate

# Graceful optional imports for DeepEval
try:
    from deepeval.metrics import BaseMetric
    from deepeval.models import DeepEvalBaseSystemOneModel
    from deepeval.models.system_one.schema import (
        NoulAnswer,
        NoulQuestion,
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

    class NoulQuestion:  # type: ignore[no-redef]
        def __init__(self, instructions: str = "", statement: str = "") -> None:
            self.instructions = instructions or statement
            self.statement = statement or instructions

    class SystemOneAnswers:  # type: ignore[no-redef]
        def __init__(self, nouls: dict[str, Any]) -> None:
            self.nouls = nouls

    SystemOneQuestion = NoulQuestion  # type: ignore[misc]

    class LLMTestCase:  # type: ignore[no-redef]
        def __init__(self, input: str = "", actual_output: str = "") -> None:
            self.input = input
            self.actual_output = actual_output


class LogitGateSystemOneModel(DeepEvalBaseSystemOneModel):
    """
    DeepEval System-1 model powered by LogitGate debiased first-token logits.
    Drop-in replacement for DeepEval JevEval System-1 models.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:14b",
        api_base: str = "http://localhost:11434",
        backend: Optional[LogitBackend] = None,
        symmetric: bool = True,
        logit_temperature: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.symmetric = symmetric
        self.logit_temperature = logit_temperature

        if backend is not None:
            self.backend = backend
        else:
            from logitgate.backends.ollama import OllamaBackend
            self.backend = OllamaBackend(base_url=api_base, model=model_name)
        super().__init__(model=model_name)

    def load_model(self) -> None:
        return None

    def get_model_name(self) -> str:
        return f"logitgate:{self.model_name}"

    def decide(
        self,
        state: Any,
        questions: Dict[str, Any],
    ) -> Tuple[Any, Optional[float]]:
        nouls: dict[str, Any] = {}
        if len(questions) > 1:
            q_ids = list(questions.keys())
            statements = [
                getattr(questions[qid], "statement", None)
                or getattr(questions[qid], "instructions", None)
                or str(questions[qid])
                for qid in q_ids
            ]
            gate = LogitGate(
                rubric=statements[0] if (statements and statements[0].strip()) else "Evaluate candidate against rubric.",
                backend=self.backend,
                symmetric_debiasing=self.symmetric,
                logit_temperature=self.logit_temperature,
            )
            verdicts = gate.evaluate_batch(state, statements)
            for q_id, verdict in zip(q_ids, verdicts):
                p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
                nouls[q_id] = NoulAnswer(probability=round(p, 4))
        else:
            for q_id, q in questions.items():
                statement = getattr(q, "statement", None) or getattr(q, "instructions", None) or str(q)
                gate = LogitGate(
                    rubric=statement,
                    backend=self.backend,
                    symmetric_debiasing=self.symmetric,
                    logit_temperature=self.logit_temperature,
                )
                verdict = gate.evaluate(state)
                p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
                nouls[q_id] = NoulAnswer(probability=round(p, 4))
        return SystemOneAnswers(nouls=nouls), 0.0001

    async def a_decide(
        self,
        state: Any,
        questions: Dict[str, Any],
    ) -> Tuple[Any, Optional[float]]:
        nouls: dict[str, Any] = {}
        if len(questions) > 1:
            q_ids = list(questions.keys())
            statements = [
                getattr(questions[qid], "statement", None)
                or getattr(questions[qid], "instructions", None)
                or str(questions[qid])
                for qid in q_ids
            ]
            gate = LogitGate(
                rubric=statements[0] if (statements and statements[0].strip()) else "Evaluate candidate against rubric.",
                backend=self.backend,
                symmetric_debiasing=self.symmetric,
                logit_temperature=self.logit_temperature,
            )
            verdicts = await gate.aevaluate_batch(state, statements)
            for q_id, verdict in zip(q_ids, verdicts):
                p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
                nouls[q_id] = NoulAnswer(probability=round(p, 4))
        else:
            for q_id, q in questions.items():
                statement = getattr(q, "statement", None) or getattr(q, "instructions", None) or str(q)
                gate = LogitGate(
                    rubric=statement,
                    backend=self.backend,
                    symmetric_debiasing=self.symmetric,
                    logit_temperature=self.logit_temperature,
                )
                verdict = await gate.aevaluate(state)
                p = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
                nouls[q_id] = NoulAnswer(probability=round(p, 4))
        return SystemOneAnswers(nouls=nouls), 0.0001


class LogitGateMetric(BaseMetric):
    """
    DeepEval metric evaluating LLMTestCase using a 50ms LogitGate.
    """

    def __init__(
        self,
        rubric: str,
        threshold: float = 0.85,
        backend: str | LogitBackend = "stub://default",
        name: str = "LogitGate Metric",
        async_mode: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.rubric = rubric
        self.threshold = threshold
        self.gate = LogitGate(
            rubric=rubric,
            pass_threshold=threshold,
            backend=backend,
        )
        self.async_mode = async_mode
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None

    @property
    def __name__(self) -> str:
        return self.name

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        candidate = getattr(test_case, "actual_output", None) or getattr(test_case, "input", str(test_case))
        verdict = self.gate.evaluate(candidate)
        self.score = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
        self.success = verdict.passed
        self.reason = verdict.reason
        return self.score

    async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
        candidate = getattr(test_case, "actual_output", None) or getattr(test_case, "input", str(test_case))
        verdict = await self.gate.aevaluate(candidate)
        self.score = verdict.probability if verdict.probability is not None else (1.0 if verdict.passed else 0.0)
        self.success = verdict.passed
        self.reason = verdict.reason
        return self.score

    def is_successful(self) -> bool:
        if self.threshold is None or self.score is None:
            return False
        return self.score >= self.threshold
