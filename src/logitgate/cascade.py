from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from logitgate.gates.ast_gate import ASTGate, ContractGate
from logitgate.gates.base import BaseGate
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict


class Cascade(BaseGate):
    """3-Layer Cascaded Router pipeline.

    Orchestrates sequential execution of gates:
    - Layer 0 (ASTGate): Sub-millisecond deterministic syntax, contract, and DAG checks.
    - Layer 1 (LogitGate): Sub-50ms zero-token debiased confidence gate.
    - Layer 2 (JudgeGate): Diagnostic generative critique triggered only on ambiguous or failing artifacts.

    Fast-path passes at Layer 1 short-circuit Layer 2 entirely, achieving high accuracy
    with zero token generation for the vast majority of requests.

    Args:
        gates: Ordered sequence of gate instances to execute sequentially.
        name: Identifier for this cascade pipeline.

    Raises:
        ValueError: If gates sequence is empty.
    """

    def __init__(self, gates: Sequence[BaseGate], name: str = "Cascade") -> None:
        if not gates:
            raise ValueError("Cascade requires at least one gate")
        self.gates = list(gates)
        self.name = name

    def __repr__(self) -> str:
        pipeline_repr = " >> ".join(repr(g) for g in self.gates)
        return f"Cascade([{pipeline_repr}])"

    def _handle_failure_snapshot(
        self, candidate: Any, verdict: Verdict, current_rubric: str | None = None, **kwargs: Any
    ) -> Verdict:
        if verdict.passed:
            return verdict

        save_on_fail = kwargs.get("save_on_fail")
        from logitgate.config import get_default_save_failures
        should_save = save_on_fail if save_on_fail is not None else get_default_save_failures()
        if not should_save:
            return verdict

        if verdict.traces and "snapshot_path" in verdict.traces[-1].metadata:
            return verdict

        from logitgate.snapshot import VerdictSnapshot
        rubric = current_rubric or getattr(self.gates[0], "rubric", None) or "Cascade evaluation"
        gate_config: dict[str, Any] = {"cascade_gates": [g.name for g in self.gates]}
        for g in self.gates:
            if isinstance(g, LogitGate):
                gate_config.update({
                    "rubric": g.rubric,
                    "pass_threshold": g.pass_threshold,
                    "fail_threshold": g.fail_threshold,
                    "symmetric_debiasing": g.symmetric_debiasing,
                    "logit_temperature": g.logit_temperature,
                })
                if g.model:
                    gate_config["model"] = g.model
                break

        test_name = kwargs.get("test_name")
        snapshot = VerdictSnapshot(
            test_name=test_name,
            rubric=rubric,
            candidate=candidate,
            verdict=verdict,
            gate_config=gate_config,
        )
        saved_path = snapshot.save(directory=kwargs.get("snapshot_dir"))
        if verdict.traces:
            verdict.traces[-1].metadata["snapshot_path"] = str(saved_path)
        return verdict

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        all_traces: list[LayerTrace] = []
        total_tokens = 0
        current_rubric = None
        last_verdict: Verdict | None = None
        gate_kwargs = dict(kwargs)
        gate_kwargs["save_on_fail"] = False

        for i, gate in enumerate(self.gates):
            # 1. Deterministic Layer 0 Gate
            if isinstance(gate, (ASTGate, ContractGate)):
                verdict = gate.evaluate(candidate, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict

                if not verdict.passed:
                    total_latency = (time.perf_counter() - t0) * 1000.0
                    final_v = Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=gate.name,
                        total_latency_ms=total_latency,
                        total_tokens_generated=total_tokens,
                        reason=verdict.reason,
                        traces=all_traces,
                    )
                    return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

            # 2. First-Token Logit Gate Layer 1
            elif isinstance(gate, LogitGate):
                current_rubric = gate.rubric
                verdict = gate.evaluate(candidate, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict

                if verdict.status == DecisionStatus.PASSED:
                    # Fast-path pass: short-circuit Layer 2 entirely!
                    total_latency = (time.perf_counter() - t0) * 1000.0
                    return Verdict(
                        passed=True,
                        status=DecisionStatus.PASSED,
                        probability=verdict.probability,
                        triggered_layer=gate.name,
                        total_latency_ms=total_latency,
                        total_tokens_generated=total_tokens,
                        reason=f"Fast-path pass at {gate.name}: P={verdict.probability:.4f}",
                        traces=all_traces,
                    )

                # Check lookahead to next JudgeGate
                if i + 1 < len(self.gates):
                    next_judge = self.gates[i + 1]
                    if isinstance(next_judge, JudgeGate):
                        if verdict.status == DecisionStatus.FAILED and next_judge.trigger == "on_ambiguous_only":
                            total_latency = (time.perf_counter() - t0) * 1000.0
                            final_v = Verdict(
                                passed=False,
                                status=DecisionStatus.FAILED,
                                probability=verdict.probability,
                                triggered_layer=gate.name,
                                total_latency_ms=total_latency,
                                total_tokens_generated=total_tokens,
                                reason=f"Decisive failure at {gate.name}: P={verdict.probability:.4f}",
                                traces=all_traces,
                            )
                            return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)
                        # Escalate to JudgeGate
                        continue

            # 3. Generative Reasoning Judge Layer 2
            elif isinstance(gate, JudgeGate):
                prev_prob = None
                prev_status = None
                for tr in reversed(all_traces):
                    if tr.calibrated_prob is not None:
                        prev_prob = tr.calibrated_prob
                        prev_status = tr.status
                        break

                verdict = gate.evaluate(candidate, rubric_override=current_rubric, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict

                total_latency = (time.perf_counter() - t0) * 1000.0
                if prev_status == DecisionStatus.FAILED and gate.trigger == "on_ambiguous_or_fail":
                    final_passed = False
                    final_status = DecisionStatus.FAILED
                else:
                    final_passed = verdict.passed
                    final_status = verdict.status

                final_v = Verdict(
                    passed=final_passed,
                    status=final_status,
                    probability=prev_prob,
                    score=verdict.score,
                    triggered_layer=gate.name,
                    total_latency_ms=total_latency,
                    total_tokens_generated=total_tokens,
                    reason=verdict.reason,
                    traces=all_traces,
                )
                return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

            # 4. Generic Gate
            else:
                verdict = gate.evaluate(candidate, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict
                if not verdict.passed and i == len(self.gates) - 1:
                    total_latency = (time.perf_counter() - t0) * 1000.0
                    final_v = Verdict(
                        passed=False,
                        status=verdict.status,
                        probability=verdict.probability,
                        score=verdict.score,
                        choice=verdict.choice,
                        choice_probabilities=verdict.choice_probabilities,
                        triggered_layer=gate.name,
                        total_latency_ms=total_latency,
                        total_tokens_generated=total_tokens,
                        reason=verdict.reason,
                        traces=all_traces,
                    )
                    return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

        total_latency = (time.perf_counter() - t0) * 1000.0
        if last_verdict is not None:
            final_v = Verdict(
                passed=last_verdict.passed,
                status=last_verdict.status,
                probability=last_verdict.probability,
                score=last_verdict.score,
                choice=last_verdict.choice,
                choice_probabilities=last_verdict.choice_probabilities,
                triggered_layer=last_verdict.triggered_layer,
                total_latency_ms=total_latency,
                total_tokens_generated=total_tokens,
                reason=last_verdict.reason,
                traces=all_traces,
            )
            return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer=self.name,
            total_latency_ms=total_latency,
            total_tokens_generated=total_tokens,
            traces=all_traces,
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        all_traces: list[LayerTrace] = []
        total_tokens = 0
        current_rubric = None
        last_verdict: Verdict | None = None
        gate_kwargs = dict(kwargs)
        gate_kwargs["save_on_fail"] = False

        for i, gate in enumerate(self.gates):
            if isinstance(gate, (ASTGate, ContractGate)):
                verdict = await gate.aevaluate(candidate, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict

                if not verdict.passed:
                    total_latency = (time.perf_counter() - t0) * 1000.0
                    final_v = Verdict(
                        passed=False,
                        status=DecisionStatus.FAILED,
                        triggered_layer=gate.name,
                        total_latency_ms=total_latency,
                        total_tokens_generated=total_tokens,
                        reason=verdict.reason,
                        traces=all_traces,
                    )
                    return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

            elif isinstance(gate, LogitGate):
                current_rubric = gate.rubric
                verdict = await gate.aevaluate(candidate, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict

                if verdict.status == DecisionStatus.PASSED:
                    total_latency = (time.perf_counter() - t0) * 1000.0
                    return Verdict(
                        passed=True,
                        status=DecisionStatus.PASSED,
                        probability=verdict.probability,
                        triggered_layer=gate.name,
                        total_latency_ms=total_latency,
                        total_tokens_generated=total_tokens,
                        reason=f"Fast-path pass at {gate.name}: P={verdict.probability:.4f}",
                        traces=all_traces,
                    )

                if i + 1 < len(self.gates):
                    next_judge = self.gates[i + 1]
                    if isinstance(next_judge, JudgeGate):
                        if verdict.status == DecisionStatus.FAILED and next_judge.trigger == "on_ambiguous_only":
                            total_latency = (time.perf_counter() - t0) * 1000.0
                            final_v = Verdict(
                                passed=False,
                                status=DecisionStatus.FAILED,
                                probability=verdict.probability,
                                triggered_layer=gate.name,
                                total_latency_ms=total_latency,
                                total_tokens_generated=total_tokens,
                                reason=f"Decisive failure at {gate.name}: P={verdict.probability:.4f}",
                                traces=all_traces,
                            )
                            return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)
                        continue

            elif isinstance(gate, JudgeGate):
                prev_prob = None
                prev_status = None
                for tr in reversed(all_traces):
                    if tr.calibrated_prob is not None:
                        prev_prob = tr.calibrated_prob
                        prev_status = tr.status
                        break

                verdict = await gate.aevaluate(candidate, rubric_override=current_rubric, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict

                total_latency = (time.perf_counter() - t0) * 1000.0
                if prev_status == DecisionStatus.FAILED and gate.trigger == "on_ambiguous_or_fail":
                    final_passed = False
                    final_status = DecisionStatus.FAILED
                else:
                    final_passed = verdict.passed
                    final_status = verdict.status

                final_v = Verdict(
                    passed=final_passed,
                    status=final_status,
                    probability=prev_prob,
                    score=verdict.score,
                    triggered_layer=gate.name,
                    total_latency_ms=total_latency,
                    total_tokens_generated=total_tokens,
                    reason=verdict.reason,
                    traces=all_traces,
                )
                return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

            else:
                verdict = await gate.aevaluate(candidate, **gate_kwargs)
                all_traces.extend(verdict.traces)
                total_tokens += verdict.total_tokens_generated
                last_verdict = verdict
                if not verdict.passed and i == len(self.gates) - 1:
                    total_latency = (time.perf_counter() - t0) * 1000.0
                    final_v = Verdict(
                        passed=False,
                        status=verdict.status,
                        probability=verdict.probability,
                        score=verdict.score,
                        choice=verdict.choice,
                        choice_probabilities=verdict.choice_probabilities,
                        triggered_layer=gate.name,
                        total_latency_ms=total_latency,
                        total_tokens_generated=total_tokens,
                        reason=verdict.reason,
                        traces=all_traces,
                    )
                    return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

        total_latency = (time.perf_counter() - t0) * 1000.0
        if last_verdict is not None:
            final_v = Verdict(
                passed=last_verdict.passed,
                status=last_verdict.status,
                probability=last_verdict.probability,
                score=last_verdict.score,
                choice=last_verdict.choice,
                choice_probabilities=last_verdict.choice_probabilities,
                triggered_layer=last_verdict.triggered_layer,
                total_latency_ms=total_latency,
                total_tokens_generated=total_tokens,
                reason=last_verdict.reason,
                traces=all_traces,
            )
            return self._handle_failure_snapshot(candidate, final_v, current_rubric, **kwargs)

        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer=self.name,
            total_latency_ms=total_latency,
            total_tokens_generated=total_tokens,
            traces=all_traces,
        )
