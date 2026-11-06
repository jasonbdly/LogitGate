"""Pytest integration: the ``@check`` decorator plus CLI options and a failure summary.

Registered through the ``pytest11`` entry point, so installing logitgate is enough:

    pytest --logitgate-backend ollama://localhost:11434/qwen2.5:7b --logitgate-save-failures

Without options, tests use whatever `logitgate.configure()` / ``LOGITGATE_BACKEND`` say, so CI
can run the same suite against the offline stub (``LOGITGATE_BACKEND=stub://``) with no GPU.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from logitgate.backends.base import LogitBackend
from logitgate.gates.base import BaseGate
from logitgate.models import Verdict


def check(
    rubric: str,
    threshold: float | None = None,
    backend: str | LogitBackend | None = None,
    ast_check: bool = False,
    save_snapshot: bool | None = None,
    model: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Asserts that a test function's *return value* satisfies ``rubric``.

    The gate is built on the first call, not at import time. That way `logitgate.configure()`
    in ``conftest.py``, fixtures, and ``--logitgate-*`` options all apply, and collecting a
    test module never touches a backend.

    Failures raise a plain `AssertionError` (not an ``assert`` statement), so the check still
    runs under ``python -O``. With snapshots enabled, the message includes the
    ``logitgate explain`` command that reproduces the failure without re-running your app.

    Example:
        >>> @check("The reply apologizes and offers a refund")
        ... def test_refund_reply():
        ...     return support_bot("I was double charged")
    """

    def _gate() -> BaseGate:
        from logitgate.gates.ast_gate import ASTGate
        from logitgate.gates.logit_gate import LogitGate

        gate: BaseGate = LogitGate(rubric=rubric, pass_threshold=threshold, backend=backend, model=model)
        return ASTGate(syntax=True) >> gate if ast_check else gate

    def _save() -> bool:
        from logitgate.config import get_default_save_failures

        return save_snapshot if save_snapshot is not None else get_default_save_failures()

    def _raise_if_failed(verdict: Verdict) -> None:
        if verdict.passed:
            return
        prob = f"{verdict.probability:.4f}" if verdict.probability is not None else "n/a"
        message = f"LogitGate check failed [{verdict.status.value}, P={prob}]: {rubric}\n  {verdict.reason}"
        snapshot_path = verdict.traces[-1].metadata.get("snapshot_path") if verdict.traces else None
        if snapshot_path:
            message += (
                f"\nSnapshot saved to: {snapshot_path}\n"
                "Reproduce & generate full Layer-2 explanation locally without re-running your app:\n"
                f"  $ logitgate explain {snapshot_path}"
            )
        raise AssertionError(message)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        state: dict[str, BaseGate] = {}

        def gate() -> BaseGate:
            if "gate" not in state:
                state["gate"] = _gate()
            return state["gate"]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                res = await fn(*args, **kwargs)
                _raise_if_failed(await gate().aevaluate(res, save_on_fail=_save(), test_name=fn.__name__))
                return res

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            res = fn(*args, **kwargs)
            _raise_if_failed(gate().evaluate(res, save_on_fail=_save(), test_name=fn.__name__))
            return res

        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------------------------
# pytest hooks (no-ops unless the options are used or snapshots are written)
# ---------------------------------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("logitgate", "LogitGate LLM checks")
    group.addoption("--logitgate-backend", default=None, help="Backend for LogitGate checks (overrides LOGITGATE_BACKEND)")
    group.addoption("--logitgate-model", default=None, help="Model for LogitGate checks (overrides LOGITGATE_MODEL)")
    group.addoption(
        "--logitgate-save-failures",
        action="store_true",
        default=False,
        help="Write a replayable snapshot for every failed LogitGate check",
    )
    group.addoption("--logitgate-snapshot-dir", default=None, help="Where failure snapshots go (default: .logitgate/failures)")


def pytest_configure(config: Any) -> None:
    from logitgate.config import configure

    configure(
        backend=config.getoption("logitgate_backend", None),
        model=config.getoption("logitgate_model", None),
        save_failures=True if config.getoption("logitgate_save_failures", False) else None,
        snapshot_dir=config.getoption("logitgate_snapshot_dir", None),
    )


def pytest_terminal_summary(terminalreporter: Any) -> None:
    from logitgate.config import get_default_save_failures
    from logitgate.snapshot import session_snapshot_paths

    paths = session_snapshot_paths()
    if not paths or not get_default_save_failures():
        return
    terminalreporter.section("LogitGate failure snapshots")
    for path in paths:
        terminalreporter.write_line(f"  {path}")
    terminalreporter.write_line(
        f"Inspect offline: logitgate explain --offline {paths[-1]}   |   with a Layer-2 critique: drop --offline"
    )
