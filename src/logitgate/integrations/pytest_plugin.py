"""Pytest integration: the ``@check`` decorator plus CLI options and a failure summary.

Registered through the ``pytest11`` entry point, so installing logitgate is enough:

    pytest --logitgate-backend ollama://localhost:11434/qwen2.5:7b --logitgate-save-failures

Without options, tests use whatever `logitgate.configure()` / ``LOGITGATE_BACKEND`` say, so CI
can run the same suite against the offline stub (``LOGITGATE_BACKEND=stub://``) with no GPU.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from logitgate.backends.base import LogitBackend
from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, Verdict

_CHECK_SNAPSHOT_PATHS: list[str] = []


def check(
    rubric: str = "",
    threshold: float | None = None,
    backend: str | LogitBackend | None = None,
    ast_check: bool | Any = False,
    save_snapshot: bool | None = None,
    model: str | None = None,
    normalize: bool | Callable[[Any], Any] = False,
    *,
    calibration_profile: Any = None,
    distillation_bundle: Any = None,
    choices: Mapping[str, str] | None = None,
    pass_choices: Sequence[str] | None = None,
    min_confidence: float | None = None,
    min_margin: float | None = None,
    score_range: tuple[int, int] | None = None,
    score_threshold: float | None = None,
    rubrics: Sequence[str] | None = None,
    aggregation: Literal["all", "any", "mean"] = "all",
    cache: Any = None,
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
        from logitgate.gates.logit_gate import (
            ChoiceGate,
            LogitGate,
            MultiRubricGate,
            ScoreGate,
        )

        class _PassChoicesChoiceGate(ChoiceGate):
            def __init__(self, *args: Any, pass_choices: Sequence[str] | None = None, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.pass_choices = set(pass_choices) if pass_choices is not None else None

            def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
                v = super().evaluate(candidate, **kwargs)
                if self.pass_choices is not None and v.passed and v.choice not in self.pass_choices:
                    return v.model_copy(update={
                        "passed": False,
                        "status": DecisionStatus.FAILED,
                        "reason": f"ChoiceGate selected '{v.choice}', not in pass_choices {sorted(self.pass_choices)}",
                    })
                return v

            async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
                v = await super().aevaluate(candidate, **kwargs)
                if self.pass_choices is not None and v.passed and v.choice not in self.pass_choices:
                    return v.model_copy(update={
                        "passed": False,
                        "status": DecisionStatus.FAILED,
                        "reason": f"ChoiceGate selected '{v.choice}', not in pass_choices {sorted(self.pass_choices)}",
                    })
                return v

        class _AggregatedMultiRubricGate(MultiRubricGate):
            def __init__(self, rubrics: Sequence[str], aggregation: str = "all", **options: Any) -> None:
                super().__init__(rubrics=rubrics, **options)
                self.aggregation = aggregation

            def _aggregate(self, verdicts: Sequence[Verdict], t0: float) -> Verdict:
                base_v = super()._aggregate(verdicts, t0)
                if self.aggregation == "all":
                    return base_v
                probs = [v.probability for v in verdicts if v.probability is not None]
                passed_count = sum(v.passed for v in verdicts)
                total = len(self.rubrics)
                if self.aggregation == "any":
                    passed = any(v.passed for v in verdicts)
                    status = DecisionStatus.PASSED if passed else (
                        DecisionStatus.AMBIGUOUS if any(v.status == DecisionStatus.AMBIGUOUS for v in verdicts) else DecisionStatus.FAILED
                    )
                    prob = max(probs) if probs else base_v.probability
                    reason = f"{passed_count}/{total} rubrics passed (any passed: {passed})"
                    return base_v.model_copy(update={
                        "passed": passed,
                        "status": status,
                        "probability": prob,
                        "reason": reason,
                    })
                if self.aggregation == "mean":
                    mean_prob = sum(probs) / len(probs) if probs else (base_v.probability or 0.0)
                    pass_thresh = self.pass_threshold
                    passed = mean_prob >= pass_thresh if pass_thresh is not None else (passed_count > total / 2)
                    status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
                    reason = f"mean probability={mean_prob:.4f} across {total} rubrics (threshold={pass_thresh})"
                    return base_v.model_copy(update={
                        "passed": passed,
                        "status": status,
                        "probability": round(mean_prob, 4),
                        "reason": reason,
                    })
                return base_v

        gate: BaseGate
        if choices is not None:
            if pass_choices is not None:
                gate = _PassChoicesChoiceGate(
                    question=rubric,
                    options=choices,
                    pass_choices=pass_choices,
                    backend=backend,
                    model=model,
                    min_confidence=min_confidence,
                    min_margin=min_margin,
                    profile=calibration_profile,
                    cache=cache,
                )
            else:
                gate = ChoiceGate(
                    question=rubric,
                    options=choices,
                    backend=backend,
                    model=model,
                    min_confidence=min_confidence,
                    min_margin=min_margin,
                    profile=calibration_profile,
                    cache=cache,
                )
        elif score_range is not None:
            eff_thresh = score_threshold if score_threshold is not None else threshold
            gate = ScoreGate(
                rubric=rubric,
                min_score=score_range[0],
                max_score=score_range[1],
                threshold=eff_thresh,
                backend=backend,
                model=model,
                profile=calibration_profile,
                cache=cache,
            )
        elif rubrics is not None:
            gate_opts: dict[str, Any] = {
                "backend": backend,
                "model": model,
                "pass_threshold": threshold,
                "profile": calibration_profile,
                "distillation_bundle": distillation_bundle,
                "cache": cache,
            }
            if aggregation != "all":
                gate = _AggregatedMultiRubricGate(
                    rubrics=rubrics,
                    aggregation=aggregation,
                    **gate_opts,
                )
            else:
                gate = MultiRubricGate(
                    rubrics=rubrics,
                    **gate_opts,
                )
        else:
            gate = LogitGate(
                rubric=rubric,
                pass_threshold=threshold,
                backend=backend,
                model=model,
                profile=calibration_profile,
                distillation_bundle=distillation_bundle,
                cache=cache,
            )

        if ast_check is True:
            return ASTGate(syntax=True, normalize=normalize) >> gate
        if isinstance(ast_check, ASTGate):
            if normalize and ast_check.normalize != normalize:
                cloned = ASTGate(
                    syntax=ast_check.syntax,
                    forbidden_imports=ast_check.forbidden_imports,
                    forbidden_patterns=ast_check.forbidden_patterns,
                    required_sections=ast_check.required_sections,
                    acyclic_dag=ast_check.acyclic_dag,
                    json_schema=ast_check.json_schema,
                    min_length=ast_check.min_length,
                    max_length=ast_check.max_length,
                    validators=ast_check.validators,
                    normalize=normalize,
                    save_on_fail=ast_check.save_on_fail,
                    name=ast_check.name,
                )
                return cloned >> gate
            return ast_check >> gate
        if isinstance(ast_check, BaseGate):
            if normalize and not getattr(ast_check, "normalize", False):
                return ASTGate(normalize=normalize) >> ast_check >> gate
            return ast_check >> gate
        if ast_check:
            return ASTGate(ast_check, normalize=normalize) >> gate
        if normalize:
            return ASTGate(normalize=normalize) >> gate
        return gate

    def _save() -> bool:
        from logitgate.config import get_default_save_failures

        return save_snapshot if save_snapshot is not None else get_default_save_failures()

    def _raise_if_failed(verdict: Verdict) -> None:
        if verdict.passed:
            return
        prob = f"{verdict.probability:.4f}" if verdict.probability is not None else "n/a"
        effective_rubric = rubric or (", ".join(rubrics) if rubrics else "Evaluation")
        message = f"LogitGate check failed [{verdict.status.value}, P={prob}]: {effective_rubric}\n  {verdict.reason}"
        snapshot_path = verdict.traces[-1].metadata.get("snapshot_path") if verdict.traces else None
        if snapshot_path:
            if str(snapshot_path) not in _CHECK_SNAPSHOT_PATHS:
                _CHECK_SNAPSHOT_PATHS.append(str(snapshot_path))
            message += (
                f"\nSnapshot saved to: {snapshot_path}\n"
                "Reproduce & generate full Layer-2 explanation locally without re-running your app:\n"
                f"  $ logitgate explain {snapshot_path}"
            )
        else:
            message += (
                "\nTip: re-run pytest with --logitgate-save-failures (or set LOGITGATE_SAVE_FAILURES=1) "
                "to save a snapshot for 'logitgate explain'."
            )
        raise AssertionError(message)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        state: dict[Any, BaseGate] = {}

        def gate() -> BaseGate:
            import os

            from logitgate.config import LogitGateDefaults, is_temperature_configured

            cfg_key = (
                LogitGateDefaults.backend if backend is None else None,
                LogitGateDefaults.model if model is None else None,
                LogitGateDefaults.pass_threshold if threshold is None else None,
                LogitGateDefaults.fail_threshold,
                LogitGateDefaults.logit_temperature,
                LogitGateDefaults.auto_calibrate,
                is_temperature_configured(),
                os.environ.get("LOGITGATE_CACHE_DIR"),
                LogitGateDefaults.num_ctx,
            )
            if cfg_key not in state:
                state.clear()
                state[cfg_key] = _gate()
            return state[cfg_key]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                res = await fn(*args, **kwargs)
                _raise_if_failed(await gate().aevaluate(res, save_on_fail=_save(), test_name=fn.__name__))
                return res

            async_wrapper._logitgate_gate = gate  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            res = fn(*args, **kwargs)
            _raise_if_failed(gate().evaluate(res, save_on_fail=_save(), test_name=fn.__name__))
            return res

        sync_wrapper._logitgate_gate = gate  # type: ignore[attr-defined]
        return sync_wrapper

    return decorator

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

    paths = session_snapshot_paths() if get_default_save_failures() else list(_CHECK_SNAPSHOT_PATHS)
    if not paths:
        return
    terminalreporter.section("LogitGate failure snapshots")
    for path in paths:
        terminalreporter.write_line(f"  {path}")
    terminalreporter.write_line(
        f"Inspect offline: logitgate explain --offline {paths[-1]}   |   with a Layer-2 critique: drop --offline"
    )
