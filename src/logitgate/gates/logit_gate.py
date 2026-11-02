from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Literal, Mapping, Optional, Sequence

from logitgate.backends.base import LogitBackend, resolve_logit_backend
from logitgate.config import LogitGateDefaults, get_default_backend, get_default_model
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_binary_prompt,
    compile_choice_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import combine_symmetric_log_odds
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    extract_token_logprob,
)
from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict


class LogitGate(BaseGate):
    """Sub-50ms first-token Logit Gate (Layer 1).

    Computes mathematically debiased binary classification confidence scores directly
    from the language model's first-token logits without generating any output tokens.
    Uses symmetric log-odds debiasing across forward and reverse label mappings to
    cancel prior token biases (e.g. Zhao et al., 2021) and positional biases.

    Args:
        rubric: Clear evaluation criterion or question to test against candidate.
        backend: Backend instance or specifier URI (e.g. 'ollama://', 'openai://', 'stub://').
        model: Optional model identifier string for the backend.
        pass_threshold: Probability threshold required to award a PASSED verdict (default: 0.85).
        fail_threshold: Probability threshold below which a decisive FAILED verdict is awarded (default: 0.20).
        symmetric_debiasing: When True, runs forward (A=Pass, B=Fail) and reverse (B=Pass, A=Fail)
            passes and averages them in log-odds space.
        query_first: When True, compiles prompt with rubric and candidate first, ensuring
            identical KV-cache prefixes across both passes.
        logit_temperature: Post-hoc temperature scaling parameter (default: 1.0).
        name: Identifier for this gate instance.

    Raises:
        ValueError: If rubric is empty, thresholds are outside [0, 1], pass_threshold <= fail_threshold,
            or logit_temperature <= 0.
    """

    def __init__(
        self,
        rubric: str,
        backend: Optional[str | LogitBackend] = None,
        model: Optional[str] = None,
        pass_threshold: Optional[float] = None,
        fail_threshold: Optional[float] = None,
        symmetric_debiasing: Optional[bool] = None,
        query_first: bool = True,
        logit_temperature: Optional[float | Literal["auto"]] = None,
        preset: Optional[Literal["strict", "balanced", "fast_ci"]] = None,
        save_on_fail: Optional[bool] = None,
        name: str = "LogitGate",
    ) -> None:
        if preset is not None:
            if preset == "strict":
                default_pass = 0.90
                default_fail = 0.25
                default_sym = True
            elif preset == "balanced":
                default_pass = 0.85
                default_fail = 0.20
                default_sym = True
            elif preset == "fast_ci":
                default_pass = 0.80
                default_fail = 0.20
                default_sym = False
            else:
                raise ValueError(f"Invalid preset '{preset}'. Must be one of ['strict', 'balanced', 'fast_ci']")
        else:
            default_pass = LogitGateDefaults.pass_threshold
            default_fail = LogitGateDefaults.fail_threshold
            default_sym = True

        effective_pass_thresh = (
            default_pass if pass_threshold is None else pass_threshold
        )
        effective_fail_thresh = (
            default_fail if fail_threshold is None else fail_threshold
        )
        effective_sym = (
            default_sym if symmetric_debiasing is None else symmetric_debiasing
        )

        if not rubric or not rubric.strip():
            raise ValueError("rubric cannot be empty or whitespace")
        if not (0.0 <= effective_pass_thresh <= 1.0):
            raise ValueError(f"pass_threshold must be between 0.0 and 1.0, got {effective_pass_thresh}")
        if not (0.0 <= effective_fail_thresh <= 1.0):
            raise ValueError(f"fail_threshold must be between 0.0 and 1.0, got {effective_fail_thresh}")
        if effective_pass_thresh <= effective_fail_thresh:
            raise ValueError(
                f"pass_threshold ({effective_pass_thresh}) must be strictly greater than fail_threshold ({effective_fail_thresh})"
            )

        effective_backend = (
            get_default_backend() if (backend is None or backend == "default") else backend
        )
        effective_model = model or get_default_model()
        self.backend = resolve_logit_backend(effective_backend, model=effective_model)
        self.model = effective_model

        if logit_temperature == "auto":
            from logitgate.debiasing.permutation import auto_calibrate_backend, get_model_temperature
            temp = get_model_temperature(effective_model)
            if temp == 1.0 and effective_model not in ("default", "stub", None):
                cal_res = auto_calibrate_backend(self.backend, model_key=effective_model)
                temp = cal_res.optimal_temperature
            effective_temp = temp
        elif logit_temperature is not None:
            effective_temp = float(logit_temperature)
        else:
            from logitgate.debiasing.permutation import get_model_temperature
            if LogitGateDefaults.logit_temperature != 1.0:
                effective_temp = LogitGateDefaults.logit_temperature
            else:
                model_temp = get_model_temperature(effective_model)
                effective_temp = model_temp if model_temp != 1.0 else LogitGateDefaults.logit_temperature

        if effective_temp <= 0.0:
            raise ValueError(f"logit_temperature must be positive, got {effective_temp}")

        self.rubric = rubric.strip()
        self.preset = preset
        self.save_on_fail = save_on_fail
        self.pass_threshold = effective_pass_thresh
        self.fail_threshold = effective_fail_thresh
        self.symmetric_debiasing = effective_sym
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.name = name

    def _handle_failure_snapshot(self, candidate: Any, verdict: Verdict, **kwargs: Any) -> Verdict:
        if verdict.passed:
            return verdict

        save_on_fail = kwargs.get("save_on_fail", self.save_on_fail)
        from logitgate.config import get_default_save_failures
        should_save = save_on_fail if save_on_fail is not None else get_default_save_failures()
        if not should_save:
            return verdict

        if verdict.traces and "snapshot_path" in verdict.traces[-1].metadata:
            return verdict

        from logitgate.snapshot import VerdictSnapshot
        gate_config = {
            "rubric": self.rubric,
            "pass_threshold": self.pass_threshold,
            "fail_threshold": self.fail_threshold,
            "symmetric_debiasing": self.symmetric_debiasing,
            "query_first": self.query_first,
            "logit_temperature": self.logit_temperature,
            "preset": self.preset,
            "name": self.name,
        }
        if hasattr(self, "model") and self.model:
            gate_config["model"] = self.model

        test_name = kwargs.get("test_name")
        snapshot = VerdictSnapshot(
            test_name=test_name,
            rubric=self.rubric,
            candidate=candidate,
            verdict=verdict,
            gate_config=gate_config,
        )
        saved_path = snapshot.save(directory=kwargs.get("snapshot_dir"))
        if verdict.traces:
            verdict.traces[-1].metadata["snapshot_path"] = str(saved_path)
        return verdict

    def __repr__(self) -> str:
        rubric_repr = repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
        return (
            f"LogitGate(rubric={rubric_repr}, "
            f"pass_threshold={self.pass_threshold}, "
            f"fail_threshold={self.fail_threshold}, "
            f"backend={self.backend!r})"
        )

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()

        # Pass 1: A=Pass, B=Fail
        prompt1 = compile_binary_prompt(
            rubric=self.rubric,
            candidate=candidate,
            pass_token="A",
            fail_token="B",
            query_first=self.query_first,
        )
        top1 = self.backend.query_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"])
        l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")

        if self.symmetric_debiasing:
            # Pass 2: A=Fail, B=Pass (prefix identical in prompt cache)
            prompt2 = compile_binary_prompt(
                rubric=self.rubric,
                candidate=candidate,
                pass_token="B",
                fail_token="A",
                query_first=self.query_first,
            )
            top2 = self.backend.query_top_logprobs(prompt2, top_k=10, allowed_tokens=["A", "B"])
            l2, p2 = compute_binary_log_odds(top2, pass_token="B", fail_token="A")
            calibrated_prob = combine_symmetric_log_odds(
                forward_log_odds=l1,
                reverse_log_odds=l2,
                logit_temperature=self.logit_temperature,
            )
        else:
            l2, p2 = None, None
            calibrated_prob = p1

        latency = (time.perf_counter() - t0) * 1000.0

        if calibrated_prob >= self.pass_threshold:
            status = DecisionStatus.PASSED
            passed = True
        elif calibrated_prob <= self.fail_threshold:
            status = DecisionStatus.FAILED
            passed = False
        else:
            status = DecisionStatus.AMBIGUOUS
            passed = False

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            calibrated_prob=round(calibrated_prob, 4),
            forward_prob=round(p1, 4),
            reverse_prob=round(p2, 4) if p2 is not None else None,
            forward_log_odds=round(l1, 4),
            reverse_log_odds=round(l2, 4) if l2 is not None else None,
            tokens_generated=0,
        )

        verdict = Verdict(
            passed=passed,
            status=status,
            probability=round(calibrated_prob, 4),
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"LogitGate prob={calibrated_prob:.4f} (pass_thresh={self.pass_threshold}, fail_thresh={self.fail_threshold})",
            traces=[trace],
        )
        return self._handle_failure_snapshot(candidate, verdict, **kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()

        prompt1 = compile_binary_prompt(
            rubric=self.rubric,
            candidate=candidate,
            pass_token="A",
            fail_token="B",
            query_first=self.query_first,
        )

        if self.symmetric_debiasing:
            prompt2 = compile_binary_prompt(
                rubric=self.rubric,
                candidate=candidate,
                pass_token="B",
                fail_token="A",
                query_first=self.query_first,
            )
            top1, top2 = await asyncio.gather(
                self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"]),
                self.backend.aquery_top_logprobs(prompt2, top_k=10, allowed_tokens=["A", "B"]),
            )
            l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")
            l2, p2 = compute_binary_log_odds(top2, pass_token="B", fail_token="A")
            calibrated_prob = combine_symmetric_log_odds(
                forward_log_odds=l1,
                reverse_log_odds=l2,
                logit_temperature=self.logit_temperature,
            )
        else:
            top1 = await self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"])
            l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")
            l2, p2 = None, None
            calibrated_prob = p1

        latency = (time.perf_counter() - t0) * 1000.0

        if calibrated_prob >= self.pass_threshold:
            status = DecisionStatus.PASSED
            passed = True
        elif calibrated_prob <= self.fail_threshold:
            status = DecisionStatus.FAILED
            passed = False
        else:
            status = DecisionStatus.AMBIGUOUS
            passed = False

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            calibrated_prob=round(calibrated_prob, 4),
            forward_prob=round(p1, 4),
            reverse_prob=round(p2, 4) if p2 is not None else None,
            forward_log_odds=round(l1, 4),
            reverse_log_odds=round(l2, 4) if l2 is not None else None,
            tokens_generated=0,
        )

        verdict = Verdict(
            passed=passed,
            status=status,
            probability=round(calibrated_prob, 4),
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"LogitGate prob={calibrated_prob:.4f} (pass_thresh={self.pass_threshold}, fail_thresh={self.fail_threshold})",
            traces=[trace],
        )
        return self._handle_failure_snapshot(candidate, verdict, **kwargs)

    def evaluate_batch(self, candidate: Any, rubrics: Sequence[str]) -> list[Verdict]:
        """
        Evaluates multiple rubrics against candidate artifact using multi-rubric
        batch prompt compilation with 100% byte-identical KV-cache prefix sharing.
        """
        if not rubrics:
            return []

        results = []
        for idx, _ in enumerate(rubrics):
            t0 = time.perf_counter()
            prompt1 = compile_batch_binary_prompt(
                rubrics=rubrics,
                candidate=candidate,
                rubric_index=idx,
                pass_token="A",
                fail_token="B",
                query_first=self.query_first,
            )
            top1 = self.backend.query_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"])
            l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")

            if self.symmetric_debiasing:
                prompt2 = compile_batch_binary_prompt(
                    rubrics=rubrics,
                    candidate=candidate,
                    rubric_index=idx,
                    pass_token="B",
                    fail_token="A",
                    query_first=self.query_first,
                )
                top2 = self.backend.query_top_logprobs(prompt2, top_k=10, allowed_tokens=["A", "B"])
                l2, p2 = compute_binary_log_odds(top2, pass_token="B", fail_token="A")
                calibrated_prob = combine_symmetric_log_odds(
                    forward_log_odds=l1,
                    reverse_log_odds=l2,
                    logit_temperature=self.logit_temperature,
                )
            else:
                l2, p2 = None, None
                calibrated_prob = p1

            latency = (time.perf_counter() - t0) * 1000.0

            if calibrated_prob >= self.pass_threshold:
                status = DecisionStatus.PASSED
                passed = True
            elif calibrated_prob <= self.fail_threshold:
                status = DecisionStatus.FAILED
                passed = False
            else:
                status = DecisionStatus.AMBIGUOUS
                passed = False

            trace = LayerTrace(
                layer_name=self.name,
                status=status,
                latency_ms=latency,
                calibrated_prob=round(calibrated_prob, 4),
                forward_prob=round(p1, 4),
                reverse_prob=round(p2, 4) if p2 is not None else None,
                forward_log_odds=round(l1, 4),
                reverse_log_odds=round(l2, 4) if l2 is not None else None,
                tokens_generated=0,
            )

            results.append(
                Verdict(
                    passed=passed,
                    status=status,
                    probability=round(calibrated_prob, 4),
                    triggered_layer=self.name,
                    total_latency_ms=latency,
                    total_tokens_generated=0,
                    reason=f"LogitGate prob={calibrated_prob:.4f} (pass_thresh={self.pass_threshold}, fail_thresh={self.fail_threshold})",
                    traces=[trace],
                )
            )
        return results

    async def aevaluate_batch(self, candidate: Any, rubrics: Sequence[str]) -> list[Verdict]:
        """
        Asynchronously evaluates multiple rubrics against candidate artifact using
        multi-rubric batch prompt compilation with 100% byte-identical KV-cache prefix sharing.
        """
        if not rubrics:
            return []

        async def _eval_single(idx: int) -> Verdict:
            t0 = time.perf_counter()
            prompt1 = compile_batch_binary_prompt(
                rubrics=rubrics,
                candidate=candidate,
                rubric_index=idx,
                pass_token="A",
                fail_token="B",
                query_first=self.query_first,
            )

            if self.symmetric_debiasing:
                prompt2 = compile_batch_binary_prompt(
                    rubrics=rubrics,
                    candidate=candidate,
                    rubric_index=idx,
                    pass_token="B",
                    fail_token="A",
                    query_first=self.query_first,
                )
                top1, top2 = await asyncio.gather(
                    self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"]),
                    self.backend.aquery_top_logprobs(prompt2, top_k=10, allowed_tokens=["A", "B"]),
                )
                l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")
                l2, p2 = compute_binary_log_odds(top2, pass_token="B", fail_token="A")
                calibrated_prob = combine_symmetric_log_odds(
                    forward_log_odds=l1,
                    reverse_log_odds=l2,
                    logit_temperature=self.logit_temperature,
                )
            else:
                top1 = await self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"])
                l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")
                l2, p2 = None, None
                calibrated_prob = p1

            latency = (time.perf_counter() - t0) * 1000.0

            if calibrated_prob >= self.pass_threshold:
                status = DecisionStatus.PASSED
                passed = True
            elif calibrated_prob <= self.fail_threshold:
                status = DecisionStatus.FAILED
                passed = False
            else:
                status = DecisionStatus.AMBIGUOUS
                passed = False

            trace = LayerTrace(
                layer_name=self.name,
                status=status,
                latency_ms=latency,
                calibrated_prob=round(calibrated_prob, 4),
                forward_prob=round(p1, 4),
                reverse_prob=round(p2, 4) if p2 is not None else None,
                forward_log_odds=round(l1, 4),
                reverse_log_odds=round(l2, 4) if l2 is not None else None,
                tokens_generated=0,
            )

            return Verdict(
                passed=passed,
                status=status,
                probability=round(calibrated_prob, 4),
                triggered_layer=self.name,
                total_latency_ms=latency,
                total_tokens_generated=0,
                reason=f"LogitGate prob={calibrated_prob:.4f} (pass_thresh={self.pass_threshold}, fail_thresh={self.fail_threshold})",
                traces=[trace],
            )

        if len(rubrics) == 1:
            return [await _eval_single(0)]

        # Prevent KV-cache stampede: populate server cache with first item
        first_verdict = await _eval_single(0)
        rest_verdicts = await asyncio.gather(*[_eval_single(idx) for idx in range(1, len(rubrics))])
        return [first_verdict, *rest_verdicts]


class ChoiceGate(BaseGate):
    """Discrete K-ary categorical decision and routing gate.

    Evaluates K discrete options from first-token logits, cancelling label order
    and position bias via symmetric permutation debiasing (Zheng et al., 2023).

    Args:
        question: Decision question or routing prompt.
        options: Mapping of option tokens (e.g., 'A', 'B', 'C') to their descriptions.
        backend: Backend instance or specifier URI.
        model: Optional model identifier string for the backend.
        symmetric_debiasing: When True, queries options in forward and reversed orderings
            and computes the debiased logprob average.
        query_first: When True, compiles prompt with question and candidate first.
        logit_temperature: Temperature scaling factor for softmax normalization (default: 1.0).
        name: Identifier for this gate instance.

    Raises:
        ValueError: If question is empty, options has fewer than 2 items, or logit_temperature <= 0.
    """

    def __init__(
        self,
        question: str,
        options: Mapping[str, str],
        backend: Optional[str | LogitBackend] = None,
        model: Optional[str] = None,
        symmetric_debiasing: bool = True,
        query_first: bool = True,
        logit_temperature: Optional[float] = None,
        name: str = "ChoiceGate",
    ) -> None:
        effective_temp = (
            LogitGateDefaults.logit_temperature if logit_temperature is None else logit_temperature
        )
        if not question or not question.strip():
            raise ValueError("question cannot be empty or whitespace")
        if not options or len(options) < 2:
            raise ValueError(f"ChoiceGate requires at least 2 options, got {len(options) if options else 0}")
        if effective_temp <= 0.0:
            raise ValueError(f"logit_temperature must be positive, got {effective_temp}")

        effective_backend = (
            get_default_backend() if (backend is None or backend == "default") else backend
        )
        effective_model = model or get_default_model()

        self.question = question.strip()
        self.options = dict(options)
        self.backend = resolve_logit_backend(effective_backend, model=effective_model)
        self.model = effective_model
        self.symmetric_debiasing = symmetric_debiasing
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.name = name

    def __repr__(self) -> str:
        q_repr = repr(self.question[:40] + "..." if len(self.question) > 40 else self.question)
        return f"ChoiceGate(question={q_repr}, options={list(self.options.keys())!r}, backend={self.backend!r})"

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        keys = list(self.options.keys())

        # Pass 1: Original option order
        prompt1 = compile_choice_prompt(
            question=self.question,
            candidate=candidate,
            options=self.options,
            query_first=self.query_first,
        )
        top1 = self.backend.query_top_logprobs(prompt1, top_k=10, allowed_tokens=keys)
        logprobs1 = {k: extract_token_logprob(top1, k) for k in keys}

        if self.symmetric_debiasing and len(keys) >= 2:
            # Pass 2: Reversed option mapping
            reversed_values = list(reversed(list(self.options.values())))
            reversed_mapping = dict(zip(keys, reversed_values))
            prompt2 = compile_choice_prompt(
                question=self.question,
                candidate=candidate,
                options=reversed_mapping,
                query_first=self.query_first,
            )
            top2 = self.backend.query_top_logprobs(prompt2, top_k=10, allowed_tokens=keys)
            # Map logprobs back to original option keys
            logprobs2: dict[str, float] = {}
            for i, k in enumerate(keys):
                rev_key = keys[len(keys) - 1 - i]
                logprobs2[k] = extract_token_logprob(top2, rev_key)

            # Average logprobs
            combined_logprobs = {k: (logprobs1[k] + logprobs2[k]) / 2.0 for k in keys}
        else:
            combined_logprobs = logprobs1

        # Temperature scaling before softmax normalization
        temp = max(self.logit_temperature, 1e-6)
        scaled_logprobs = {k: lp / temp for k, lp in combined_logprobs.items()}

        # Normalized softmax probabilities
        max_lp = max(scaled_logprobs.values()) if scaled_logprobs else 0.0
        exp_vals = {k: math.exp(lp - max_lp) for k, lp in scaled_logprobs.items()}
        total_exp = sum(exp_vals.values())
        choice_probs = {k: exp_vals[k] / total_exp for k in keys} if total_exp > 0 else {k: 1.0 / len(keys) for k in keys}

        best_choice = max(choice_probs, key=choice_probs.get)  # type: ignore
        max_prob = choice_probs[best_choice]
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=DecisionStatus.PASSED,
            latency_ms=latency,
            calibrated_prob=round(max_prob, 4),
            metadata={"choice_probabilities": {k: round(v, 4) for k, v in choice_probs.items()}},
        )

        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            probability=round(max_prob, 4),
            choice=best_choice,
            choice_probabilities={k: round(v, 4) for k, v in choice_probs.items()},
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"ChoiceGate selected '{best_choice}' with P={max_prob:.4f}",
            traces=[trace],
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        keys = list(self.options.keys())

        prompt1 = compile_choice_prompt(
            question=self.question,
            candidate=candidate,
            options=self.options,
            query_first=self.query_first,
        )

        if self.symmetric_debiasing and len(keys) >= 2:
            reversed_values = list(reversed(list(self.options.values())))
            reversed_mapping = dict(zip(keys, reversed_values))
            prompt2 = compile_choice_prompt(
                question=self.question,
                candidate=candidate,
                options=reversed_mapping,
                query_first=self.query_first,
            )
            top1, top2 = await asyncio.gather(
                self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=keys),
                self.backend.aquery_top_logprobs(prompt2, top_k=10, allowed_tokens=keys),
            )
            logprobs1 = {k: extract_token_logprob(top1, k) for k in keys}
            logprobs2: dict[str, float] = {}
            for i, k in enumerate(keys):
                rev_key = keys[len(keys) - 1 - i]
                logprobs2[k] = extract_token_logprob(top2, rev_key)

            combined_logprobs = {k: (logprobs1[k] + logprobs2[k]) / 2.0 for k in keys}
        else:
            top1 = await self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=keys)
            logprobs1 = {k: extract_token_logprob(top1, k) for k in keys}
            combined_logprobs = logprobs1

        # Temperature scaling before softmax normalization
        temp = max(self.logit_temperature, 1e-6)
        scaled_logprobs = {k: lp / temp for k, lp in combined_logprobs.items()}

        max_lp = max(scaled_logprobs.values()) if scaled_logprobs else 0.0
        exp_vals = {k: math.exp(lp - max_lp) for k, lp in scaled_logprobs.items()}
        total_exp = sum(exp_vals.values())
        choice_probs = {k: exp_vals[k] / total_exp for k in keys} if total_exp > 0 else {k: 1.0 / len(keys) for k in keys}

        best_choice = max(choice_probs, key=choice_probs.get)  # type: ignore
        max_prob = choice_probs[best_choice]
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=DecisionStatus.PASSED,
            latency_ms=latency,
            calibrated_prob=round(max_prob, 4),
            metadata={"choice_probabilities": {k: round(v, 4) for k, v in choice_probs.items()}},
        )

        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            probability=round(max_prob, 4),
            choice=best_choice,
            choice_probabilities={k: round(v, 4) for k, v in choice_probs.items()},
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"ChoiceGate selected '{best_choice}' with P={max_prob:.4f}",
            traces=[trace],
        )


class ScoreGate(BaseGate):
    """Continuous probability-weighted rubric scoring gate.

    Computes expected continuous quality score:
        E[S] = sum_{s=min_score}^{max_score} s * P(s)
    from first-token logits as formalized by Liu et al. (2023, G-Eval).
    Supports temperature scaling and optional symmetric scale reversal debiasing
    (evaluating both standard 1..5 and inverted 5..1 scale mappings).

    Args:
        rubric: Evaluation rubric describing scoring expectations.
        min_score: Minimum integer score rating (default: 1).
        max_score: Maximum integer score rating (default: 5).
        threshold: Score threshold required for a PASSED verdict (default: 3.5).
        backend: Backend instance or specifier URI.
        model: Optional model identifier string for the backend.
        symmetric_debiasing: When True, queries both normal and inverted scale prompts
            and averages logprobs to eliminate scale-end anchoring bias.
        query_first: When True, compiles prompt with rubric and candidate first.
        logit_temperature: Temperature scaling factor for softmax normalization (default: 1.0).
        name: Identifier for this gate instance.

    Raises:
        ValueError: If rubric is empty, min_score >= max_score, threshold is out of bounds,
            or logit_temperature <= 0.
    """

    def __init__(
        self,
        rubric: str,
        min_score: int = 1,
        max_score: int = 5,
        threshold: float = 3.5,
        backend: Optional[str | LogitBackend] = None,
        model: Optional[str] = None,
        symmetric_debiasing: bool = False,
        query_first: bool = True,
        logit_temperature: Optional[float] = None,
        name: str = "ScoreGate",
    ) -> None:
        effective_temp = (
            LogitGateDefaults.logit_temperature if logit_temperature is None else logit_temperature
        )
        if not rubric or not rubric.strip():
            raise ValueError("rubric cannot be empty or whitespace")
        if min_score >= max_score:
            raise ValueError(f"min_score ({min_score}) must be strictly less than max_score ({max_score})")
        if threshold < min_score or threshold > max_score:
            raise ValueError(
                f"threshold ({threshold}) must be between min_score ({min_score}) and max_score ({max_score})"
            )
        if effective_temp <= 0.0:
            raise ValueError(f"logit_temperature must be positive, got {effective_temp}")

        effective_backend = (
            get_default_backend() if (backend is None or backend == "default") else backend
        )
        effective_model = model or get_default_model()

        self.rubric = rubric.strip()
        self.min_score = min_score
        self.max_score = max_score
        self.threshold = threshold
        self.backend = resolve_logit_backend(effective_backend, model=effective_model)
        self.model = effective_model
        self.symmetric_debiasing = symmetric_debiasing
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.name = name

    def __repr__(self) -> str:
        rubric_repr = repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
        return (
            f"ScoreGate(rubric={rubric_repr}, "
            f"range=[{self.min_score}..{self.max_score}], "
            f"threshold={self.threshold}, "
            f"backend={self.backend!r})"
        )

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        score_tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]

        prompt1 = compile_score_prompt(
            rubric=self.rubric,
            candidate=candidate,
            min_score=self.min_score,
            max_score=self.max_score,
            query_first=self.query_first,
            inverted=False,
        )
        top1 = self.backend.query_top_logprobs(prompt1, top_k=10, allowed_tokens=score_tokens)
        logprobs1 = {s: extract_token_logprob(top1, str(s)) for s in range(self.min_score, self.max_score + 1)}

        if self.symmetric_debiasing:
            prompt2 = compile_score_prompt(
                rubric=self.rubric,
                candidate=candidate,
                min_score=self.min_score,
                max_score=self.max_score,
                query_first=self.query_first,
                inverted=True,
            )
            top2 = self.backend.query_top_logprobs(prompt2, top_k=10, allowed_tokens=score_tokens)
            # In inverted prompt: token s' represents quality score = max_score + min_score - s'
            # So for quality score s, the token queried was s' = max_score + min_score - s
            logprobs2 = {
                s: extract_token_logprob(top2, str(self.max_score + self.min_score - s))
                for s in range(self.min_score, self.max_score + 1)
            }
            combined_logprobs = {
                s: (logprobs1[s] + logprobs2[s]) / 2.0
                for s in range(self.min_score, self.max_score + 1)
            }
        else:
            combined_logprobs = logprobs1

        # Temperature scaling before softmax
        temp = max(self.logit_temperature, 1e-6)
        scaled_logprobs = {s: lp / temp for s, lp in combined_logprobs.items()}

        # Normalized softmax probabilities
        max_lp = max(scaled_logprobs.values()) if scaled_logprobs else 0.0
        exp_vals = {s: math.exp(lp - max_lp) for s, lp in scaled_logprobs.items()}
        total_exp = sum(exp_vals.values())
        score_probs = (
            {s: exp_vals[s] / total_exp for s in scaled_logprobs}
            if total_exp > 0
            else {s: 1.0 / len(scaled_logprobs) for s in scaled_logprobs}
        )

        # Expected score: sum(s * P(s))
        expected_score = sum(s * prob for s, prob in score_probs.items())
        passed = expected_score >= self.threshold
        status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            metadata={
                "expected_score": round(expected_score, 4),
                "token_probabilities": {str(k): round(v, 4) for k, v in score_probs.items()},
            },
        )

        return Verdict(
            passed=passed,
            status=status,
            score=round(expected_score, 4),
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"ScoreGate E[S]={expected_score:.2f} (threshold={self.threshold})",
            traces=[trace],
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        score_tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]

        prompt1 = compile_score_prompt(
            rubric=self.rubric,
            candidate=candidate,
            min_score=self.min_score,
            max_score=self.max_score,
            query_first=self.query_first,
            inverted=False,
        )

        if self.symmetric_debiasing:
            prompt2 = compile_score_prompt(
                rubric=self.rubric,
                candidate=candidate,
                min_score=self.min_score,
                max_score=self.max_score,
                query_first=self.query_first,
                inverted=True,
            )
            top1, top2 = await asyncio.gather(
                self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=score_tokens),
                self.backend.aquery_top_logprobs(prompt2, top_k=10, allowed_tokens=score_tokens),
            )
            logprobs1 = {s: extract_token_logprob(top1, str(s)) for s in range(self.min_score, self.max_score + 1)}
            logprobs2 = {
                s: extract_token_logprob(top2, str(self.max_score + self.min_score - s))
                for s in range(self.min_score, self.max_score + 1)
            }
            combined_logprobs = {
                s: (logprobs1[s] + logprobs2[s]) / 2.0
                for s in range(self.min_score, self.max_score + 1)
            }
        else:
            top1 = await self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=score_tokens)
            logprobs1 = {s: extract_token_logprob(top1, str(s)) for s in range(self.min_score, self.max_score + 1)}
            combined_logprobs = logprobs1

        # Temperature scaling before softmax
        temp = max(self.logit_temperature, 1e-6)
        scaled_logprobs = {s: lp / temp for s, lp in combined_logprobs.items()}

        # Normalized softmax probabilities
        max_lp = max(scaled_logprobs.values()) if scaled_logprobs else 0.0
        exp_vals = {s: math.exp(lp - max_lp) for s, lp in scaled_logprobs.items()}
        total_exp = sum(exp_vals.values())
        score_probs = (
            {s: exp_vals[s] / total_exp for s in scaled_logprobs}
            if total_exp > 0
            else {s: 1.0 / len(scaled_logprobs) for s in scaled_logprobs}
        )

        # Expected score: sum(s * P(s))
        expected_score = sum(s * prob for s, prob in score_probs.items())
        passed = expected_score >= self.threshold
        status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            metadata={
                "expected_score": round(expected_score, 4),
                "token_probabilities": {str(k): round(v, 4) for k, v in score_probs.items()},
            },
        )

        return Verdict(
            passed=passed,
            status=status,
            score=round(expected_score, 4),
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"ScoreGate E[S]={expected_score:.2f} (threshold={self.threshold})",
            traces=[trace],
        )
