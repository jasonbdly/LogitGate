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
    save_snapshot: Optional[bool] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Pytest decorator to evaluate function output against an LLM rubric
    using a sub-50ms LogitGate.
    """
    from logitgate.config import get_default_save_failures

    effective_backend = (
        get_default_backend() if (backend is None or backend == "default") else backend
    )
    effective_thresh = (
        LogitGateDefaults.pass_threshold if threshold is None else threshold
    )
    effective_save = save_snapshot if save_snapshot is not None else get_default_save_failures()

    def _format_error_msg(verdict: Any) -> str:
        snapshot_path = verdict.traces[-1].metadata.get("snapshot_path") if verdict.traces else None
        if snapshot_path:
            return (
                f"LogitGate evaluation failed (P={verdict.probability}): {verdict.reason}\n"
                f"Snapshot saved to: {snapshot_path}\n"
                f"Reproduce & generate full Layer-2 explanation locally without re-running your app:\n"
                f"  $ logitgate explain {snapshot_path}"
            )
        return f"LogitGate evaluation failed (P={verdict.probability}): {verdict.reason}"

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        gate: Any = LogitGate(rubric=rubric, pass_threshold=effective_thresh, backend=effective_backend)
        if ast_check:
            gate = ASTGate(syntax=True) >> gate

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                res = await fn(*args, **kwargs)
                verdict = await gate.aevaluate(res, save_on_fail=effective_save, test_name=fn.__name__)
                assert verdict.passed, _format_error_msg(verdict)
                return res
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                res = fn(*args, **kwargs)
                verdict = gate.evaluate(res, save_on_fail=effective_save, test_name=fn.__name__)
                assert verdict.passed, _format_error_msg(verdict)
                return res
            return sync_wrapper

    return decorator
