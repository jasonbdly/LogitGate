from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from logitgate.exceptions import ConfigurationError
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


def _router(
    pass_node: str | None,
    fail_node: str | None,
    ambiguous_node: str | None,
    node_mapping: Mapping[str, str] | None = None,
) -> Callable[[Verdict], str]:
    if node_mapping is not None:
        mapping = dict(node_mapping)
        if not mapping:
            raise ConfigurationError("node_mapping cannot be empty.")

        def pick_choice(verdict: Verdict) -> str:
            if verdict.choice is not None and verdict.choice in mapping:
                return mapping[verdict.choice]
            if ambiguous_node is not None and verdict.status == DecisionStatus.AMBIGUOUS:
                return ambiguous_node
            if fail_node is not None:
                return fail_node
            raise KeyError(
                f"Gate returned choice {verdict.choice!r}, which is not in node_mapping {sorted(mapping)}"
            )

        return pick_choice

    if pass_node is None or fail_node is None:
        raise ConfigurationError(
            "as_conditional_edge requires either (pass_node and fail_node) or node_mapping"
        )

    def pick(verdict: Verdict) -> str:
        if verdict.passed:
            return pass_node
        if ambiguous_node is not None and verdict.status == DecisionStatus.AMBIGUOUS:
            return ambiguous_node
        return fail_node

    return pick


def as_conditional_edge(
    gate: BaseGate,
    pass_node: str | None = None,
    fail_node: str | None = None,
    state_key: str = "content",
    ambiguous_node: str | None = None,
    node_mapping: Mapping[str, str] | None = None,
) -> Callable[[Mapping[str, Any] | Any], str]:
    """Returns a LangGraph conditional-edge function that routes on a gate's verdict.

    Args:
        gate: Any gate or cascade.
        pass_node: Node to route to on PASSED (for binary gates).
        fail_node: Node to route to on FAILED (and on AMBIGUOUS when ``ambiguous_node`` is unset).
        state_key: Key (or attribute) of the state holding the artifact to evaluate.
        ambiguous_node: Optional third route for AMBIGUOUS verdicts, e.g. a human-review node.
            Useful when a single `LogitGate` is the router and you'd rather not pay for a
            generative judge inline.
        node_mapping: Optional mapping from discrete `ChoiceGate` option keys (e.g. ``"A"``) to
            destination LangGraph node names.

    Example:
        >>> graph.add_conditional_edges("draft", as_conditional_edge(gate, "send", "revise", "reply"))
    """
    pick = _router(pass_node, fail_node, ambiguous_node, node_mapping=node_mapping)

    def route(state: Mapping[str, Any] | Any) -> str:
        return pick(gate.evaluate(_extract(state, state_key)))

    return route


def as_async_conditional_edge(
    gate: BaseGate,
    pass_node: str | None = None,
    fail_node: str | None = None,
    state_key: str = "content",
    ambiguous_node: str | None = None,
    node_mapping: Mapping[str, str] | None = None,
) -> Callable[[Mapping[str, Any] | Any], Awaitable[str]]:
    """Async version of `as_conditional_edge` for graphs run with ``ainvoke`` / ``astream``.

    A sync edge inside an async graph blocks the event loop for the full backend round-trip.
    This one awaits `gate.aevaluate`, which runs symmetric passes concurrently.
    """
    pick = _router(pass_node, fail_node, ambiguous_node, node_mapping=node_mapping)

    async def route(state: Mapping[str, Any] | Any) -> str:
        return pick(await gate.aevaluate(_extract(state, state_key)))

    return route
