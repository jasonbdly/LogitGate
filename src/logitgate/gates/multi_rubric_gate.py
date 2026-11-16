from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from logitgate.backends.base import LogitBackend
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

if TYPE_CHECKING:
    from logitgate.calibration.adaptive import AdaptiveConformalController
    from logitgate.calibration.profile import CalibrationProfile
    from logitgate.distill.anchor_bank import DistilledAnchorBank
    from logitgate.distill.bundle import DistillationBundle
    from logitgate.distill.meta_calibrator import LogitMetaCalibrator
    from logitgate.gates.logit_gate import LogitGate


class MultiRubricGate(BaseGate):
    """Checks several atomic rubrics against one candidate; passes only if every rubric passes.

    Prefer this to one compound rubric ("is it typed AND documented AND safe?"). A compound
    question blends independent signals into one logit, and a strong "yes" on two criteria can
    mask a "no" on the third. Here each rubric gets its own debiased read, and all reads share
    the candidate's KV-cache prefix (see `LogitGate.evaluate_batch`).

    Aggregation is weakest-link: any FAILED rubric gives FAILED, otherwise any AMBIGUOUS gives
    AMBIGUOUS, and probability is the minimum across rubrics. One trace per rubric is
    recorded (metadata["rubric"]), so `logitgate explain` shows which criterion failed.

    Inside a `Cascade`, a downstream `JudgeGate` inherits all rubrics as a bulleted list.

    Args:
        rubrics: Atomic yes/no criteria, one question each.
        name: Identifier for this gate instance.
        **options: Any `LogitGate` option (backend, model, pass_threshold,
            fail_threshold, symmetric_debiasing, logit_temperature, preset...).
    """

    def __init__(
        self,
        rubrics: Sequence[str],
        name: str = "MultiRubricGate",
        *,
        fail_fast: bool = False,
        **options: Any,
    ) -> None:
        if not isinstance(fail_fast, bool):
            raise ConfigurationError(f"fail_fast must be a bool, got {fail_fast!r}")
        if isinstance(rubrics, (str, bytes)):
            raise ConfigurationError("MultiRubricGate rubrics must be a sequence of strings, not a single string")
        cleaned = [r.strip() for r in rubrics if isinstance(r, str) and r.strip()]
        if len(cleaned) != len(rubrics) or not cleaned:
            raise ConfigurationError("MultiRubricGate needs one or more non-empty rubric strings")
        self.rubrics = cleaned
        self.name = name
        self.fail_fast = fail_fast
        from logitgate.gates.logit_gate import LogitGate

        self._gate = LogitGate(rubric=cleaned[0], name=name, **options)
        self.save_on_fail = self._gate.save_on_fail
        self._owns_backend = False
        self._owns_cache = False
        self._owns_sink = False

    @property
    def gates(self) -> list[LogitGate]:
        return [self._gate]

    @property
    def rubric(self) -> str:
        """All criteria as one bulleted rubric, the form a downstream judge receives."""
        return "\n".join(f"- {r}" for r in self.rubrics)

    @property
    def backend(self) -> LogitBackend:
        return self._gate.backend

    @property
    def model(self) -> str | None:
        return self._gate.model

    @property
    def logit_temperature(self) -> float:
        return self._gate.logit_temperature

    @property
    def pass_threshold(self) -> float:
        return self._gate.pass_threshold

    @property
    def fail_threshold(self) -> float:
        return self._gate.fail_threshold

    @property
    def symmetric_debiasing(self) -> bool:
        return self._gate.symmetric_debiasing

    @property
    def warm_prefix(self) -> bool | None:
        return self._gate.warm_prefix

    @property
    def profile(self) -> CalibrationProfile | None:
        return self._gate.profile

    @property
    def adaptive_controller(self) -> AdaptiveConformalController | None:
        return self._gate.adaptive_controller

    @property
    def anchor_bank(self) -> DistilledAnchorBank | None:
        return self._gate.anchor_bank

    @property
    def meta_calibrator(self) -> LogitMetaCalibrator | None:
        return self._gate.meta_calibrator

    @property
    def distillation_bundle(self) -> DistillationBundle | None:
        return self._gate.distillation_bundle

    @property
    def telemetry_sink(self) -> Any:
        return self._gate.telemetry_sink

    @property
    def cache(self) -> Any:
        return self._gate.cache

    @cache.setter
    def cache(self, val: Any) -> None:
        self._gate.cache = val

    def __repr__(self) -> str:
        return f"MultiRubricGate(rubrics={len(self.rubrics)}, backend={self.backend!r})"

    def evaluate_each(self, candidate: Any) -> list[Verdict]:
        """Per-rubric verdicts, in `rubrics` order."""
        return self._gate.evaluate_batch(candidate, self.rubrics, fail_fast=self.fail_fast)

    async def aevaluate_each(self, candidate: Any) -> list[Verdict]:
        return await self._gate.aevaluate_batch(candidate, self.rubrics, fail_fast=self.fail_fast)

    def _aggregate(self, verdicts: Sequence[Verdict], t0: float) -> Verdict:
        traces: list[LayerTrace] = []
        for i, (rubric, v) in enumerate(zip(self.rubrics, verdicts, strict=False)):
            traces.extend(
                trace.model_copy(
                    update={
                        "layer_name": f"{self.name}[{i}]",
                        "metadata": {**trace.metadata, "rubric": rubric},
                    }
                )
                for trace in v.traces
            )
        statuses = {v.status for v in verdicts}
        if DecisionStatus.FAILED in statuses:
            status = DecisionStatus.FAILED
        elif DecisionStatus.AMBIGUOUS in statuses:
            status = DecisionStatus.AMBIGUOUS
        else:
            status = DecisionStatus.PASSED
        if self._gate.adaptive_controller is not None and traces:
            traces[0] = traces[0].model_copy(
                update={"metadata": {**traces[0].metadata, "raw_l1_status": status.value}}
            )
        probs = [v.probability for v in verdicts if v.probability is not None]
        unresolved = [
            f"{v.status.value.upper()} {rubric!r} (P={v.probability:.4f})"
            for rubric, v in zip(self.rubrics, verdicts, strict=False)
            if not v.passed and v.probability is not None
        ]
        total = len(self.rubrics)
        evaluated = len(verdicts)
        passed_count = sum(v.passed for v in verdicts)
        if evaluated < total:
            reason = f"{passed_count}/{total} rubrics passed (short-circuited at {evaluated}/{total})"
        else:
            reason = f"{passed_count}/{total} rubrics passed"
        if unresolved:
            reason += "; " + "; ".join(unresolved)
        return Verdict(
            passed=status == DecisionStatus.PASSED,
            status=status,
            probability=min(probs) if probs else None,
            triggered_layer=self.name,
            total_latency_ms=(time.perf_counter() - t0) * 1000.0,
            total_tokens_generated=0,
            reason=reason,
            traces=traces,
        )

    def replay_config(self) -> dict[str, Any]:
        config = {k: v for k, v in self._gate.replay_config().items() if k != "rubric"}
        return {
            **config,
            "gate_type": "MultiRubricGate",
            "rubrics": list(self.rubrics),
            "fail_fast": self.fail_fast,
        }

    def _snapshot(
        self,
        candidate: Any,
        verdict: Verdict,
        kwargs: dict[str, Any],
        *,
        _is_cache_hit: bool = False,
    ) -> Verdict:
        from logitgate.cache import store_cached_verdict
        from logitgate.telemetry.sinks import emit_verdict_telemetry

        call_kw = kwargs
        if _is_cache_hit:
            raw_tags = kwargs.get("telemetry_tags")
            merged_tags = dict(raw_tags) if isinstance(raw_tags, dict) else {}
            merged_tags["cache_hit"] = True
            call_kw = {**kwargs, "telemetry_tags": merged_tags}
        else:
            store_cached_verdict(
                self, candidate, verdict, gate_cache=self.cache, call_kwargs=kwargs
            )

        emit_verdict_telemetry(
            verdict,
            candidate=candidate,
            gate_name=self.name,
            gate_type="MultiRubricGate",
            rubric=self.rubric,
            model=self.model,
            backend=self.backend,
            gate_sink=self.telemetry_sink,
            call_kwargs=call_kw,
        )
        if verdict.passed:
            return verdict
        return persist_failure_snapshot(
            verdict,
            candidate=candidate,
            rubric=self.rubric,
            gate_config=self.replay_config(),
            save_on_fail=kwargs.get("save_on_fail", self.save_on_fail),
            test_name=kwargs.get("test_name"),
            snapshot_dir=kwargs.get("snapshot_dir"),
        )

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        self._gate._ensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._snapshot(candidate, hit, kwargs, _is_cache_hit=True)
        t0 = time.perf_counter()
        return self._snapshot(candidate, self._aggregate(self.evaluate_each(candidate), t0), kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        await self._gate._aensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._snapshot(candidate, hit, kwargs, _is_cache_hit=True)
        t0 = time.perf_counter()
        return self._snapshot(candidate, self._aggregate(await self.aevaluate_each(candidate), t0), kwargs)


__all__ = ["MultiRubricGate"]
