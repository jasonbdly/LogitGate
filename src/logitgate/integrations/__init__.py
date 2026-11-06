"""Framework integrations, imported lazily.

DeepEval alone takes ~1 s to import, and this package is reachable from the pytest plugin,
which pytest loads for every session once logitgate is installed. Names resolve on first
attribute access (PEP 562), so `import logitgate` stays cheap for inline, latency-sensitive
callers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logitgate.integrations.deepeval import LogitGateMetric, LogitGateSystemOneModel
    from logitgate.integrations.langgraph import (
        as_async_conditional_edge,
        as_conditional_edge,
    )
    from logitgate.integrations.pytest_plugin import check

_LAZY_EXPORTS = {
    "LogitGateMetric": "logitgate.integrations.deepeval",
    "LogitGateSystemOneModel": "logitgate.integrations.deepeval",
    "as_conditional_edge": "logitgate.integrations.langgraph",
    "as_async_conditional_edge": "logitgate.integrations.langgraph",
    "check": "logitgate.integrations.pytest_plugin",
}

__all__ = [
    "LogitGateMetric",
    "LogitGateSystemOneModel",
    "as_async_conditional_edge",
    "as_conditional_edge",
    "check",
]


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
