from __future__ import annotations

import abc
import asyncio
import contextlib
import sys
import time
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from logitgate.exceptions import ConfigurationError
from logitgate.models import DecisionStatus, LayerTrace, Verdict

if TYPE_CHECKING:
    from logitgate.cascade import Cascade
    from logitgate.streaming import AsyncGuardedStream, GuardedStream

_T_Stream = TypeVar("_T_Stream")

EscalationPolicy = Literal["on_ambiguous_or_fail", "on_ambiguous_only", "adjudicate", "always"]
BatchErrorPolicy = Literal["raise", "abstain", "fail"]

TIER_DETERMINISTIC = 0
TIER_LOGIT = 1
TIER_GENERATIVE = 2


def _make_batch_error_verdict(
    gate: BaseGate,
    exc: Exception,
    on_error: BatchErrorPolicy,
    t0: float,
) -> Verdict:
    latency = (time.perf_counter() - t0) * 1000.0
    err_msg = f"{type(exc).__name__}: {exc}"
    status = (
        getattr(DecisionStatus, "ABSTAINED", DecisionStatus.AMBIGUOUS)
        if on_error == "abstain"
        else DecisionStatus.FAILED
    )
    trace = LayerTrace(
        layer_name=gate.name,
        status=status,
        latency_ms=latency,
        diagnostic_report=err_msg,
        failure_code="BATCH_EVAL_ERROR" if on_error == "fail" else None,
        metadata={"abstained_reason": err_msg} if on_error == "abstain" else {},
    )
    verdict = Verdict(
        status=status,
        passed=False,
        probability=0.0 if on_error == "fail" else None,
        triggered_layer=gate.name,
        total_latency_ms=latency,
        total_tokens_generated=0,
        reason=err_msg,
        traces=[trace],
    )
    if on_error == "abstain":
        object.__setattr__(verdict, "abstained_reason", err_msg)
    return verdict


def _safe_close(obj: Any) -> None:
    if hasattr(obj, "close"):
        with contextlib.suppress(Exception):
            obj.close()


async def _safe_aclose(obj: Any) -> None:
    if hasattr(obj, "aclose"):
        with contextlib.suppress(Exception):
            await obj.aclose()
    elif hasattr(obj, "close"):
        with contextlib.suppress(Exception):
            obj.close()


class BaseGate(abc.ABC):
    """Abstract base class for all evaluation gates and cascaded routing layers.

    All gates support composition via the `>>` and `|` operators to construct
    multi-layer decision pipelines (e.g., `ASTGate >> LogitGate >> JudgeGate`).
    """

    name: str = "BaseGate"
    layer_tier: int = TIER_LOGIT
    can_fast_path: bool = True
    can_override_fail: bool = False
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

    def evaluate_batch(
        self,
        candidates: Sequence[Any],
        *,
        on_error: BatchErrorPolicy = "raise",
        **kwargs: Any,
    ) -> list[Verdict]:
        """Evaluates a batch of candidates synchronously with fault isolation.

        Args:
            candidates: Sequence of candidate artifacts to evaluate.
            on_error: Error policy when evaluation of an item raises an exception:
                - 'raise': propagate the exception immediately.
                - 'abstain': record an ABSTAINED verdict for that item and continue.
                - 'fail': record a FAILED verdict for that item and continue.
            **kwargs: Extra evaluation parameters passed to `evaluate`.

        Returns:
            List of verdicts in the same order as candidates.
        """
        if on_error not in ("raise", "abstain", "fail"):
            raise ConfigurationError(
                f"on_error must be one of 'raise', 'abstain', or 'fail', got {on_error!r}"
            )
        results: list[Verdict] = []
        for cand in candidates:
            t0 = time.perf_counter()
            if on_error == "raise":
                results.append(self.evaluate(cand, **kwargs))
            else:
                try:
                    results.append(self.evaluate(cand, **kwargs))
                except Exception as exc:
                    results.append(_make_batch_error_verdict(self, exc, on_error, t0))
        return results

    async def aevaluate_batch(
        self,
        candidates: Sequence[Any],
        *,
        max_concurrency: int | None = None,
        on_error: BatchErrorPolicy = "raise",
        **kwargs: Any,
    ) -> list[Verdict]:
        """Evaluates a batch of candidates asynchronously with fault isolation and concurrency control.

        Args:
            candidates: Sequence of candidate artifacts to evaluate.
            max_concurrency: Maximum number of concurrent evaluation tasks (>= 1 or None).
            on_error: Error policy when evaluation of an item raises an exception:
                - 'raise': propagate the exception immediately.
                - 'abstain': record an ABSTAINED verdict for that item and continue.
                - 'fail': record a FAILED verdict for that item and continue.
            **kwargs: Extra evaluation parameters passed to `aevaluate`.

        Returns:
            List of verdicts in the same order as candidates.
        """
        if on_error not in ("raise", "abstain", "fail"):
            raise ConfigurationError(
                f"on_error must be one of 'raise', 'abstain', or 'fail', got {on_error!r}"
            )
        if (
            max_concurrency is not None
            and (
                isinstance(max_concurrency, bool)
                or not isinstance(max_concurrency, int)
                or max_concurrency < 1
            )
        ):
            raise ConfigurationError(
                f"max_concurrency must be an integer >= 1 or None, got {max_concurrency!r}"
            )
        if not candidates:
            return []

        async def _eval_one(cand: Any) -> Verdict:
            t0 = time.perf_counter()
            if on_error == "raise":
                return await self.aevaluate(cand, **kwargs)
            try:
                return await self.aevaluate(cand, **kwargs)
            except Exception as exc:
                return _make_batch_error_verdict(self, exc, on_error, t0)

        if max_concurrency is not None:
            sem = asyncio.Semaphore(max_concurrency)

            async def _sem_eval(cand: Any) -> Verdict:
                async with sem:
                    return await _eval_one(cand)

            return list(await asyncio.gather(*[_sem_eval(c) for c in candidates]))
        return list(await asyncio.gather(*[_eval_one(c) for c in candidates]))

    def close(self) -> None:
        """Safely close any owned resources (backend, cache, child gates). Idempotent."""
        if getattr(self, "_is_closed", False):
            return
        self._is_closed = True

        # Close child gates if present
        closed_children: set[int] = set()
        for child_attr in ("rank_gate", "pre_gate", "post_gate", "_gate"):
            child = getattr(self, child_attr, None)
            if child is not None and id(child) not in closed_children:
                closed_children.add(id(child))
                _safe_close(child)
        for child in getattr(self, "gates", ()) or ():
            if child is not None and id(child) not in closed_children:
                closed_children.add(id(child))
                _safe_close(child)

        # Close backend if this gate owns it
        if getattr(self, "_owns_backend", False):
            backend = getattr(self, "backend", None)
            if backend is not None:
                _safe_close(backend)

        # Close cache if this gate owns it
        if getattr(self, "_owns_cache", False):
            cache = getattr(self, "cache", None)
            if cache is not None:
                _safe_close(cache)

        # Close telemetry sink if this gate owns it
        if getattr(self, "_owns_sink", False):
            sink = getattr(self, "telemetry_sink", None)
            if sink is not None:
                _safe_close(sink)

    async def aclose(self) -> None:
        """Safely and asynchronously close any owned resources. Idempotent."""
        if getattr(self, "_is_closed", False):
            return
        self._is_closed = True

        # Close child gates if present
        closed_children: set[int] = set()
        for child_attr in ("rank_gate", "pre_gate", "post_gate", "_gate"):
            child = getattr(self, child_attr, None)
            if child is not None and id(child) not in closed_children:
                closed_children.add(id(child))
                await _safe_aclose(child)
        for child in getattr(self, "gates", ()) or ():
            if child is not None and id(child) not in closed_children:
                closed_children.add(id(child))
                await _safe_aclose(child)

        # Close backend if this gate owns it
        if getattr(self, "_owns_backend", False):
            backend = getattr(self, "backend", None)
            if backend is not None:
                await _safe_aclose(backend)

        # Close cache if this gate owns it
        if getattr(self, "_owns_cache", False):
            cache = getattr(self, "cache", None)
            if cache is not None:
                await _safe_aclose(cache)

        # Close telemetry sink if this gate owns it
        if getattr(self, "_owns_sink", False):
            sink = getattr(self, "telemetry_sink", None)
            if sink is not None:
                await _safe_aclose(sink)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.aclose()

    def replay_config(self) -> dict[str, Any] | None:
        """Returns the gate constructor config needed to replay this gate from a snapshot."""
        return None

    def as_conditional_edge(
        self,
        pass_node: str | None = None,
        fail_node: str | None = None,
        state_key: str = "content",
        ambiguous_node: str | None = None,
        node_mapping: Mapping[str, str] | None = None,
    ) -> Callable[[Mapping[str, Any] | Any], str]:
        """Wraps this gate as a synchronous LangGraph conditional edge routing callable."""
        from logitgate.integrations.langgraph import as_conditional_edge

        return as_conditional_edge(
            self,
            pass_node=pass_node,
            fail_node=fail_node,
            state_key=state_key,
            ambiguous_node=ambiguous_node,
            node_mapping=node_mapping,
        )

    def as_async_conditional_edge(
        self,
        pass_node: str | None = None,
        fail_node: str | None = None,
        state_key: str = "content",
        ambiguous_node: str | None = None,
        node_mapping: Mapping[str, str] | None = None,
    ) -> Callable[[Mapping[str, Any] | Any], Awaitable[str]]:
        """Wraps this gate as an asynchronous LangGraph conditional edge routing callable."""
        from logitgate.integrations.langgraph import as_async_conditional_edge

        return as_async_conditional_edge(
            self,
            pass_node=pass_node,
            fail_node=fail_node,
            state_key=state_key,
            ambiguous_node=ambiguous_node,
            node_mapping=node_mapping,
        )

    def guard_stream(
        self, source: Iterable[_T_Stream], **guard_kwargs: Any
    ) -> GuardedStream[_T_Stream]:
        """Wraps a synchronous chunk stream with `StreamGuard` using this gate."""
        from logitgate.streaming import guard_stream

        return guard_stream(source, self, **guard_kwargs)

    def aguard_stream(
        self, source: AsyncIterable[_T_Stream], **guard_kwargs: Any
    ) -> AsyncGuardedStream[_T_Stream]:
        """Wraps an asynchronous chunk stream with `StreamGuard` using this gate."""
        from logitgate.streaming import aguard_stream

        return aguard_stream(source, self, **guard_kwargs)

    def __call__(self, candidate: Any, **kwargs: Any) -> Verdict:
        """Evaluates the candidate artifact synchronously (callable shorthand for `evaluate`)."""
        return self.evaluate(candidate, **kwargs)

    def __rshift__(self, other: BaseGate) -> Cascade:
        """Chains this gate with another gate into a cascaded routing pipeline."""
        from logitgate.cascade import Cascade

        if isinstance(self, Cascade) and isinstance(other, Cascade):
            eff_sink = (
                self.telemetry_sink if self.telemetry_sink is not None else other.telemetry_sink
            )
            eff_cache = self.cache if self.cache is not None else other.cache
            return Cascade(
                gates=list(self.gates) + list(other.gates),
                name=self.name,
                fuse_logits=self.fuse_logits,
                prior_weight=self.prior_weight,
                telemetry_sink=eff_sink,
                cache=eff_cache,
            )
        if isinstance(self, Cascade):
            return Cascade(
                gates=[*list(self.gates), other],
                name=self.name,
                fuse_logits=self.fuse_logits,
                prior_weight=self.prior_weight,
                telemetry_sink=self.telemetry_sink,
                cache=self.cache,
            )
        if isinstance(other, Cascade):
            return Cascade(
                gates=[self, *list(other.gates)],
                fuse_logits=other.fuse_logits,
                prior_weight=other.prior_weight,
                telemetry_sink=other.telemetry_sink,
                cache=other.cache,
            )
        return Cascade(gates=[self, other])

    def __or__(self, other: BaseGate) -> Cascade:
        """Alias for `>>` allowing pipeline chaining with pipe syntax."""
        return self.__rshift__(other)
