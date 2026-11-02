from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Mapping, Optional, Sequence

from logitgate.backends.base import LogitBackend, resolve_logit_backend
from logitgate.debiasing.inversion import (
    compile_binary_prompt,
    compile_choice_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import combine_symmetric_log_odds
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    compute_normalized_probabilities,
    extract_token_logprob,
)
from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict


class LogitGate(BaseGate):
    """
    Sub-50ms first-token Logit Gate (Layer 1).
    Performs symmetric log-odds debiasing, prefix-cache-friendly inversion,
    and temperature calibration without autoregressive token generation.
    """

    def __init__(
        self,
        rubric: str,
        backend: str | LogitBackend = "stub://default",
        model: Optional[str] = None,
        pass_threshold: float = 0.85,
        fail_threshold: float = 0.20,
        symmetric_debiasing: bool = True,
        query_first: bool = True,
        logit_temperature: float = 1.0,
        name: str = "LogitGate",
    ) -> None:
        self.rubric = rubric
        self.backend = resolve_logit_backend(backend, model=model)
        self.model = model
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold
        self.symmetric_debiasing = symmetric_debiasing
        self.query_first = query_first
        self.logit_temperature = logit_temperature
        self.name = name

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

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()

        prompt1 = compile_binary_prompt(
            rubric=self.rubric,
            candidate=candidate,
            pass_token="A",
            fail_token="B",
            query_first=self.query_first,
        )
        top1 = await self.backend.aquery_top_logprobs(prompt1, top_k=10, allowed_tokens=["A", "B"])
        l1, p1 = compute_binary_log_odds(top1, pass_token="A", fail_token="B")

        if self.symmetric_debiasing:
            prompt2 = compile_binary_prompt(
                rubric=self.rubric,
                candidate=candidate,
                pass_token="B",
                fail_token="A",
                query_first=self.query_first,
            )
            top2 = await self.backend.aquery_top_logprobs(prompt2, top_k=10, allowed_tokens=["A", "B"])
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

    def evaluate_batch(self, candidate: Any, rubrics: Sequence[str]) -> list[Verdict]:
        """Evaluates multiple rubrics against candidate artifact."""
        results = []
        for rubric in rubrics:
            gate = LogitGate(
                rubric=rubric,
                backend=self.backend,
                model=self.model,
                pass_threshold=self.pass_threshold,
                fail_threshold=self.fail_threshold,
                symmetric_debiasing=self.symmetric_debiasing,
                query_first=self.query_first,
                logit_temperature=self.logit_temperature,
                name=self.name,
            )
            results.append(gate.evaluate(candidate))
        return results

    async def aevaluate_batch(self, candidate: Any, rubrics: Sequence[str]) -> list[Verdict]:
        """Asynchronously evaluates multiple rubrics against candidate artifact."""
        coros = []
        for rubric in rubrics:
            gate = LogitGate(
                rubric=rubric,
                backend=self.backend,
                model=self.model,
                pass_threshold=self.pass_threshold,
                fail_threshold=self.fail_threshold,
                symmetric_debiasing=self.symmetric_debiasing,
                query_first=self.query_first,
                logit_temperature=self.logit_temperature,
                name=self.name,
            )
            coros.append(gate.aevaluate(candidate))
        return await asyncio.gather(*coros)


class ChoiceGate(BaseGate):
    """
    Evaluates discrete K-ary choice options debiased via forward and reverse option orderings.
    """

    def __init__(
        self,
        question: str,
        options: Mapping[str, str],
        backend: str | LogitBackend = "stub://default",
        model: Optional[str] = None,
        symmetric_debiasing: bool = True,
        query_first: bool = True,
        name: str = "ChoiceGate",
    ) -> None:
        self.question = question
        self.options = dict(options)
        self.backend = resolve_logit_backend(backend, model=model)
        self.model = model
        self.symmetric_debiasing = symmetric_debiasing
        self.query_first = query_first
        self.name = name

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

        # Normalized softmax probabilities
        max_lp = max(combined_logprobs.values()) if combined_logprobs else 0.0
        exp_vals = {k: math.exp(lp - max_lp) for k, lp in combined_logprobs.items()}
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
    """
    Evaluates rubric score on integer scale min_score..max_score by computing
    probability-weighted expected value: E[S] = sum(s * P(s)) from first-token logits.
    """

    def __init__(
        self,
        rubric: str,
        min_score: int = 1,
        max_score: int = 5,
        threshold: float = 3.5,
        backend: str | LogitBackend = "stub://default",
        model: Optional[str] = None,
        query_first: bool = True,
        name: str = "ScoreGate",
    ) -> None:
        self.rubric = rubric
        self.min_score = min_score
        self.max_score = max_score
        self.threshold = threshold
        self.backend = resolve_logit_backend(backend, model=model)
        self.model = model
        self.query_first = query_first
        self.name = name

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        score_tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]

        prompt = compile_score_prompt(
            rubric=self.rubric,
            candidate=candidate,
            min_score=self.min_score,
            max_score=self.max_score,
            query_first=self.query_first,
        )
        top = self.backend.query_top_logprobs(prompt, top_k=10, allowed_tokens=score_tokens)
        token_probs = compute_normalized_probabilities(top, score_tokens)

        # Expected score: sum(s * P(s))
        expected_score = sum(int(tok) * prob for tok, prob in token_probs.items())
        passed = expected_score >= self.threshold
        status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            metadata={
                "expected_score": round(expected_score, 4),
                "token_probabilities": {k: round(v, 4) for k, v in token_probs.items()},
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
