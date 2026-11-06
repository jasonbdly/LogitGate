from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, Verdict

_MISSING = object()


def _extract(state: Mapping[str, Any] | Any, state_key: str) -> Any:
    """Pulls the candidate out of a LangGraph state (TypedDict/dict or object/Pydantic state).

    A missing key raises instead of falling back to the whole state. Otherwise a typo in
    ``state_key`` would make the gate quietly judge the entire state dict.
    """
    if isinstance(state, Mapping):
        value = state.get(state_key, _MISSING)
        available = sorted(map(str, state.keys()))
    else:
        value = getattr(state, state_key, _MISSING)
        available = sorted(k for k in dir(state) if not k.startswith("_"))
    if value is _MISSING:
        raise KeyError(f"LangGraph state has no {state_key!r} to evaluate; available keys: {available}")
    return value


def _router(pass_node: str, fail_node: str, ambiguous_node: str | None) -> Callable[[Verdict], str]:
    def pick(verdict: Verdict) -> str:
        if verdict.passed:
            return pass_node
        if ambiguous_node is not None and verdict.status == DecisionStatus.AMBIGUOUS:
            return ambiguous_node
        return fail_node

    return pick


def as_conditional_edge(
    gate: BaseGate,
    pass_node: str,
    fail_node: str,
    state_key: str = "content",
    ambiguous_node: str | None = None,
) -> Callable[[Mapping[str, Any] | Any], str]:
    """Returns a LangGraph conditional-edge function that routes on a gate's verdict.

    Args:
        gate: Any gate or cascade.
        pass_node: Node to route to on PASSED.
        fail_node: Node to route to on FAILED (and on AMBIGUOUS when ``ambiguous_node`` is unset).
        state_key: Key (or attribute) of the state holding the artifact to evaluate.
        ambiguous_node: Optional third route for AMBIGUOUS verdicts, e.g. a human-review node.
            Useful when a single `LogitGate` is the router and you'd rather not pay for a
            generative judge inline.

    Example:
        >>> graph.add_conditional_edges("draft", as_conditional_edge(gate, "send", "revise", "reply"))
    """
    pick = _router(pass_node, fail_node, ambiguous_node)

    def route(state: Mapping[str, Any] | Any) -> str:
        return pick(gate.evaluate(_extract(state, state_key)))

    return route


def as_async_conditional_edge(
    gate: BaseGate,
    pass_node: str,
    fail_node: str,
    state_key: str = "content",
    ambiguous_node: str | None = None,
) -> Callable[[Mapping[str, Any] | Any], Awaitable[str]]:
    """Async version of `as_conditional_edge` for graphs run with ``ainvoke`` / ``astream``.

    A sync edge inside an async graph blocks the event loop for the full backend round-trip.
    This one awaits `gate.aevaluate`, which runs symmetric passes concurrently.
    """
    pick = _router(pass_node, fail_node, ambiguous_node)

    async def route(state: Mapping[str, Any] | Any) -> str:
        return pick(await gate.aevaluate(_extract(state, state_key)))

    return route
