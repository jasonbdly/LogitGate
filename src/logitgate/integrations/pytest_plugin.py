from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from logitgate.backends.base import LogitBackend
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import LogitGate


def check(
    rubric: str,
    threshold: float = 0.85,
    backend: str | LogitBackend = "stub://default",
    ast_check: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Pytest decorator to evaluate function output against an LLM rubric
    using a sub-50ms LogitGate.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        gate: Any = LogitGate(rubric=rubric, pass_threshold=threshold, backend=backend)
        if ast_check:
            gate = ASTGate(syntax=True) >> gate

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                res = await fn(*args, **kwargs)
                verdict = await gate.aevaluate(res)
                assert verdict.passed, (
                    f"LogitGate evaluation failed (P={verdict.probability}): {verdict.reason}"
                )
                return res
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                res = fn(*args, **kwargs)
                verdict = gate.evaluate(res)
                assert verdict.passed, (
                    f"LogitGate evaluation failed (P={verdict.probability}): {verdict.reason}"
                )
                return res
            return sync_wrapper

    return decorator
