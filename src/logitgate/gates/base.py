from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING, Any

from logitgate.models import Verdict

if TYPE_CHECKING:
    from logitgate.cascade import Cascade


class BaseGate(abc.ABC):
    """Abstract base class for all gating layers."""

    @abc.abstractmethod
    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates the candidate synchronously, returning a Verdict."""
        pass

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates the candidate asynchronously, returning a Verdict."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(candidate, **kwargs))

    def __rshift__(self, other: BaseGate) -> Cascade:
        from logitgate.cascade import Cascade
        if isinstance(self, Cascade) and isinstance(other, Cascade):
            return Cascade(gates=list(self.gates) + list(other.gates))
        if isinstance(self, Cascade):
            return Cascade(gates=list(self.gates) + [other])
        if isinstance(other, Cascade):
            return Cascade(gates=[self] + list(other.gates))
        return Cascade(gates=[self, other])

    def __or__(self, other: BaseGate) -> Cascade:
        return self.__rshift__(other)
