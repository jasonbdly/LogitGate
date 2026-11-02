from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING, Any

from logitgate.models import Verdict

if TYPE_CHECKING:
    from logitgate.cascade import Cascade


class BaseGate(abc.ABC):
    """Abstract base class for all evaluation gates and cascaded routing layers.

    All gates support composition via the `>>` and `|` operators to construct
    multi-layer decision pipelines (e.g., `ASTGate >> LogitGate >> JudgeGate`).
    """

    name: str = "BaseGate"

    @abc.abstractmethod
    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates the candidate artifact synchronously.

        Args:
            candidate: Artifact to evaluate (string, dict, code, or structured object).
            **kwargs: Additional layer-specific evaluation parameters.

        Returns:
            Verdict containing decision status, latency, tokens, and telemetry traces.
        """

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates the candidate artifact asynchronously.

        Args:
            candidate: Artifact to evaluate (string, dict, code, or structured object).
            **kwargs: Additional layer-specific evaluation parameters.

        Returns:
            Verdict containing decision status, latency, tokens, and telemetry traces.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(candidate, **kwargs))

    def __rshift__(self, other: BaseGate) -> Cascade:
        """Chains this gate with another gate into a cascaded routing pipeline.

        Example:
            >>> pipeline = ASTGate(syntax=True) >> LogitGate(rubric="Clean code")
        """
        from logitgate.cascade import Cascade
        if isinstance(self, Cascade) and isinstance(other, Cascade):
            return Cascade(gates=list(self.gates) + list(other.gates))
        if isinstance(self, Cascade):
            return Cascade(gates=list(self.gates) + [other])
        if isinstance(other, Cascade):
            return Cascade(gates=[self] + list(other.gates))
        return Cascade(gates=[self, other])

    def __or__(self, other: BaseGate) -> Cascade:
        """Alias for `>>` allowing pipeline chaining with pipe syntax."""
        return self.__rshift__(other)
