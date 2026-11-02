from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from logitgate.backends.base import LogitBackend
from logitgate.config import LogitGateDefaults, get_default_backend
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import LogitGate


def check(
    rubric: str,
    threshold: Optional[float] = None,
    backend: Optional[str | LogitBackend] = None,
    ast_check: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Pytest decorator to evaluate function output against an LLM rubric
    using a sub-50ms LogitGate.
    """
    effective_backend = (
        get_default_backend() if (backend is None or backend == "default") else backend
    )
    effective_thresh = (
        LogitGateDefaults.pass_threshold if threshold is None else threshold
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        gate: Any = LogitGate(rubric=rubric, pass_threshold=effective_thresh, backend=effective_backend)
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
