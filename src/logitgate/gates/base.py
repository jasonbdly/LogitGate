from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from logitgate.models import Verdict

if TYPE_CHECKING:
    from logitgate.cascade import Cascade

EscalationPolicy = Literal["on_ambiguous_or_fail", "on_ambiguous_only", "always"]
"""When a generative (tier 2) gate runs inside a `Cascade`, relative to the upstream verdict.

- ``on_ambiguous_or_fail``: run on AMBIGUOUS or FAILED upstream. A decisive upstream FAILED
  is preserved; the generative layer only contributes the diagnostic.
- ``on_ambiguous_only``: run only on AMBIGUOUS upstream. A decisive FAILED terminates the
  cascade without paying for generation.
- ``always``: run on every input that reaches it; its verdict is authoritative.
"""

TIER_DETERMINISTIC = 0
TIER_LOGIT = 1
TIER_GENERATIVE = 2


class BaseGate(abc.ABC):
    """Abstract base class for all evaluation gates and cascaded routing layers.

    All gates support composition via the `>>` and `|` operators to construct
    multi-layer decision pipelines (e.g., `ASTGate >> LogitGate >> JudgeGate`).

    Routing traits
    --------------
    `Cascade` never inspects concrete gate classes. It routes on three class-level traits,
    so wrappers, decorators, and third-party gates get the same short-circuiting as the
    built-ins by declaring them:

    - ``layer_tier``: cost tier. ``0`` = deterministic CPU checks (failures are final; asking
      an LLM to second-guess a syntax error wastes money), ``1`` = single forward-pass logit
      reads (failures may be escalated), ``2`` = autoregressive generation (runs only when its
      ``escalation_policy`` admits the upstream verdict).
    - ``can_fast_path``: whether a PASSED verdict from this gate is trusted enough to skip an
      immediately downstream generative gate. Deterministic gates set this to ``False``: a
      passing schema check says nothing about semantics, so the judge must still run.
    - ``escalation_policy``: only consulted on tier-2 gates. See `EscalationPolicy`.

    The defaults (tier 1, fast-path enabled) match how `Cascade` has always treated
    unrecognized gates, so existing custom gates keep their behavior.
    """

    name: str = "BaseGate"
    layer_tier: int = TIER_LOGIT
    can_fast_path: bool = True
    escalation_policy: EscalationPolicy = "on_ambiguous_or_fail"

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

        The default offloads `evaluate` to the loop's thread pool so a blocking custom gate
        never stalls the event loop. Gates with native async I/O should override this.

        Args:
            candidate: Artifact to evaluate (string, dict, code, or structured object).
            **kwargs: Additional layer-specific evaluation parameters.

        Returns:
            Verdict containing decision status, latency, tokens, and telemetry traces.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.evaluate(candidate, **kwargs))

    def replay_config(self) -> dict[str, Any] | None:
        """Returns the `LogitGate` constructor config needed to replay this gate from a snapshot.

        `VerdictSnapshot.replay()` reconstructs a `LogitGate`, so only gates that can be
        faithfully rebuilt that way return a dict. Everything else returns ``None`` and is
        recorded in the snapshot by name only.
        """
        return None

    def __call__(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates the candidate artifact synchronously (callable shorthand for `evaluate`)."""
        return self.evaluate(candidate, **kwargs)

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
