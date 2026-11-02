from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from logitgate.gates.base import BaseGate


def as_conditional_edge(
    gate: BaseGate,
    pass_node: str,
    fail_node: str,
    state_key: str = "content",
) -> Callable[[Mapping[str, Any] | Any], str]:
    """
    Returns a conditional edge function for LangGraph workflows.
    Extracts candidate from state using state_key and routes to
    pass_node if the gate passes, else fail_node.
    """
    def route(state: Mapping[str, Any] | Any) -> str:
        candidate: Any
        if isinstance(state, Mapping):
            candidate = state.get(state_key, state)
        else:
            candidate = getattr(state, state_key, state)

        verdict = gate.evaluate(candidate)
        return pass_node if verdict.passed else fail_node

    return route
