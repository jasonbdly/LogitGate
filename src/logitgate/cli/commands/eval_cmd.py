from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from logitgate.cascade import Cascade
from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text
from logitgate.exceptions import ConfigurationError
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.base import BaseGate
from logitgate.gates.logit_gate import (
    ChoiceGate as _BaseChoiceGate,
)
from logitgate.gates.logit_gate import (
    LogitGate,
    ScoreGate,
)
from logitgate.gates.logit_gate import (
    MultiRubricGate as _BaseMultiRubricGate,
)
from logitgate.models import DecisionStatus, LayerTrace, Verdict


class MultiRubricGate(_BaseMultiRubricGate):
    """CLI MultiRubricGate wrapper supporting configurable aggregation strategies."""

    def __init__(
        self,
        rubrics: Sequence[str],
        name: str = "MultiRubricGate",
        *,
        fail_fast: bool = False,
        aggregation: str = "all",
        **options: Any,
    ) -> None:
        self.aggregation = aggregation
        super().__init__(rubrics, name=name, fail_fast=fail_fast, **options)

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        if self.aggregation == "all":
            return super().evaluate(candidate, **kwargs)

        from logitgate.cache import lookup_cached_verdict

        self._gate._ensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._snapshot(candidate, hit, kwargs, _is_cache_hit=True)

        t0 = time.perf_counter()
        verdicts = self.evaluate_each(candidate)
        traces: list[LayerTrace] = []
        for i, (rubric, v) in enumerate(zip(self.rubrics, verdicts, strict=True)):
            traces.extend(
                trace.model_copy(
                    update={
                        "layer_name": f"{self.name}[{i}]",
                        "metadata": {**trace.metadata, "rubric": rubric},
                    }
                )
                for trace in v.traces
            )

        probs = [v.probability for v in verdicts if v.probability is not None]
        passed_count = sum(v.passed for v in verdicts)
        total = len(self.rubrics)

        if self.aggregation == "any":
            any_passed = any(v.passed for v in verdicts)
            if any_passed:
                status = DecisionStatus.PASSED
            elif any(v.status == DecisionStatus.AMBIGUOUS for v in verdicts):
                status = DecisionStatus.AMBIGUOUS
            else:
                status = DecisionStatus.FAILED
            prob = max(probs) if probs else None
            reason = f"{passed_count}/{total} rubrics passed (aggregation=any)"
        elif self.aggregation == "mean":
            mean_prob = sum(probs) / len(probs) if probs else 0.5
            pass_th = self.pass_threshold
            fail_th = self.fail_threshold
            if mean_prob >= pass_th:
                status = DecisionStatus.PASSED
            elif mean_prob <= fail_th:
                status = DecisionStatus.FAILED
            else:
                status = DecisionStatus.AMBIGUOUS
            prob = mean_prob
            reason = f"{passed_count}/{total} rubrics passed (aggregation=mean, P={mean_prob:.4f})"
        else:
            raise ConfigurationError(f"Unknown aggregation strategy '{self.aggregation}'")

        res_verdict = Verdict(
            passed=status == DecisionStatus.PASSED,
            status=status,
            probability=prob,
            triggered_layer=self.name,
            total_latency_ms=(time.perf_counter() - t0) * 1000.0,
            total_tokens_generated=0,
            reason=reason,
            traces=traces,
        )
        return self._snapshot(candidate, res_verdict, kwargs)


class ChoiceGate(_BaseChoiceGate):
    """CLI ChoiceGate wrapper supporting pass_choices constraint verification."""

    def __init__(
        self,
        question: str,
        options: Mapping[str, str],
        *args: Any,
        pass_choices: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.pass_choices = list(pass_choices) if pass_choices else None
        super().__init__(question, options, *args, **kwargs)

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        verdict = super().evaluate(candidate, **kwargs)
        if self.pass_choices is not None and verdict.passed:
            passed = verdict.choice in self.pass_choices
            status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
            reason = verdict.reason
            if not passed:
                reason = f"Choice '{verdict.choice}' not in expected pass choices {self.pass_choices}"
            verdict = verdict.model_copy(
                update={
                    "passed": passed,
                    "status": status,
                    "reason": reason,
                }
            )
        return verdict


def _cmd_eval(args: argparse.Namespace) -> int:
    candidate: str | None = None
    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as f:
                candidate = f.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(
                f"Error reading file '{sanitize_terminal_text(str(args.file))}': "
                f"{sanitize_terminal_text(str(e))}\n"
            )
            return EXIT_FAILED
    elif args.candidate is not None:
        candidate = args.candidate
    else:
        if not sys.stdin.isatty():
            try:
                candidate = sys.stdin.read()
            except (OSError, UnicodeDecodeError) as e:
                sys.stderr.write(f"Error reading stdin: {sanitize_terminal_text(str(e))}\n")
                return EXIT_FAILED
        else:
            sys.stderr.write("Error: Either --candidate or --file must be specified.\n")
            return EXIT_FAILED

    # Normalize rubric argument: handle single string, list, or tuple
    raw_rubric = getattr(args, "rubric", None)
    if isinstance(raw_rubric, str):
        rubrics_list = [raw_rubric]
    elif isinstance(raw_rubric, (list, tuple)):
        rubrics_list = list(raw_rubric)
    elif raw_rubric is not None:
        rubrics_list = [str(raw_rubric)]
    else:
        rubrics_list = []

    is_tool_call = bool(getattr(args, "tool_call", False))
    if not rubrics_list and not is_tool_call:
        sys.stderr.write("Error: --rubric is required unless --tool-call is specified.\n")
        return EXIT_FAILED

    contexts: list[str] = []
    if getattr(args, "context", None):
        contexts.extend(args.context)
    if getattr(args, "context_file", None):
        from logitgate.telemetry.sinks import read_regular_text_nofollow

        try:
            cf_text = read_regular_text_nofollow(
                Path(args.context_file),
                max_bytes=16 * 1024 * 1024,
                label="Context file",
            )
            contexts.append(cf_text)
        except Exception as e:
            sys.stderr.write(
                f"Error reading context file '{sanitize_terminal_text(str(args.context_file))}': "
                f"{sanitize_terminal_text(str(e))}\n"
            )
            return EXIT_FAILED

    final_context: str | list[str] | None = None
    if len(contexts) == 1:
        final_context = contexts[0]
    elif len(contexts) > 1:
        final_context = contexts

    raw_choice = getattr(args, "choice", None)
    is_choice = bool(raw_choice)
    score_min_raw = getattr(args, "score_min", None)
    score_max_raw = getattr(args, "score_max", None)
    is_score = score_min_raw is not None or score_max_raw is not None
    is_multi_rubric = len(rubrics_list) >= 2

    # Mutual exclusivity checks
    if is_tool_call and (is_choice or is_score):
        raise ConfigurationError("Cannot combine --tool-call with --choice or --score-min/--score-max")
    if is_tool_call and is_multi_rubric:
        raise ConfigurationError("Cannot combine --tool-call with multiple --rubric criteria")
    if is_choice and is_score:
        raise ConfigurationError("Cannot combine --choice with --score-min or --score-max")
    if is_choice and is_multi_rubric:
        raise ConfigurationError("Cannot combine --choice with multiple --rubric criteria")
    if is_score and is_multi_rubric:
        raise ConfigurationError(
            "Cannot combine --score-min/--score-max with multiple --rubric criteria"
        )

    parsed_choices: dict[str, str] = {}
    if is_choice and raw_choice is not None:
        for item in raw_choice:
            if "=" not in item:
                raise ConfigurationError(
                    f"Invalid --choice '{item}'. Expected KEY=TEXT format (e.g. A='Approve')"
                )
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1].strip()
            if not k:
                raise ConfigurationError(f"Choice option key cannot be empty in '{item}'")
            if not v:
                raise ConfigurationError(f"Choice option description cannot be empty in '{item}'")
            parsed_choices[k] = v
        if len(parsed_choices) < 2:
            raise ConfigurationError(
                f"ChoiceGate requires at least 2 options, got {len(parsed_choices)}"
            )
        expected_choice = getattr(args, "expected_choice", None)
        if expected_choice and expected_choice not in parsed_choices:
            raise ConfigurationError(
                f"--expected-choice '{expected_choice}' is not one of the provided options: {list(parsed_choices.keys())}"
            )

    score_min = 1
    score_max = 5
    if is_score:
        score_min = score_min_raw if score_min_raw is not None else 1
        score_max = score_max_raw if score_max_raw is not None else 5
        if score_min >= score_max:
            raise ConfigurationError(
                f"--score-min ({score_min}) must be strictly less than --score-max ({score_max})"
            )

    from logitgate.cache import VerdictCache
    from logitgate.telemetry.sinks import JsonlTelemetrySink

    sink = JsonlTelemetrySink(args.telemetry_out) if getattr(args, "telemetry_out", None) else None
    raw_cache_arg = getattr(args, "cache", None)
    cache_obj: VerdictCache | None = None
    try:
        if raw_cache_arg is not None:
            cache_obj = (
                VerdictCache()
                if raw_cache_arg == "memory"
                else VerdictCache(path=raw_cache_arg)
            )
        has_ast = bool(getattr(args, "ast_syntax", False) or getattr(args, "normalize", False))

        target_gate: BaseGate
        rubric_single = rubrics_list[0] if rubrics_list else ""
        if is_tool_call:
            from logitgate.gates.tool_call_gate import ToolCallGate

            intent_rubric = rubrics_list[0] if rubrics_list else None
            check_intent = bool(rubrics_list)
            target_gate = ToolCallGate(
                allowed_tools=getattr(args, "allowed_tool", None),
                allowed_roots=getattr(args, "allowed_root", None),
                allowed_hosts=getattr(args, "allowed_host", None),
                check_intent=check_intent,
                intent_rubric=intent_rubric,
                user_intent=getattr(args, "query", None),
                backend=args.backend,
                model=args.model,
                pass_threshold=args.threshold,
                fail_threshold=args.fail_threshold,
                on_unsupported_logprobs=getattr(args, "on_unsupported_logprobs", "raise"),
                sidecar_backend=getattr(args, "sidecar_backend", None),
                telemetry_sink=sink if not has_ast else None,
                cache=cache_obj if not has_ast else False,
            )
        elif is_choice:
            target_gate = ChoiceGate(
                question=rubric_single,
                options=parsed_choices,
                pass_choices=[args.expected_choice] if getattr(args, "expected_choice", None) else None,
                backend=args.backend,
                model=args.model,
                min_confidence=getattr(args, "min_confidence", None),
                min_margin=getattr(args, "min_margin", None),
                profile=args.profile,
                context=final_context,
                query=getattr(args, "query", None),
                on_unsupported_logprobs=getattr(args, "on_unsupported_logprobs", "raise"),
                sidecar_backend=getattr(args, "sidecar_backend", None),
                telemetry_sink=sink if not has_ast else None,
                cache=cache_obj if not has_ast else False,
            )
        elif is_score:
            score_th = getattr(args, "score_threshold", None)
            if score_th is None:
                score_th = args.threshold
            target_gate = ScoreGate(
                rubric=rubric_single,
                min_score=score_min,
                max_score=score_max,
                threshold=score_th,
                fail_threshold=args.fail_threshold,
                backend=args.backend,
                model=args.model,
                profile=args.profile,
                context=final_context,
                query=getattr(args, "query", None),
                on_unsupported_logprobs=getattr(args, "on_unsupported_logprobs", "raise"),
                sidecar_backend=getattr(args, "sidecar_backend", None),
                telemetry_sink=sink if not has_ast else None,
                cache=cache_obj if not has_ast else False,
            )
        elif is_multi_rubric:
            target_gate = MultiRubricGate(
                rubrics=rubrics_list,
                aggregation=getattr(args, "aggregation", "all") or "all",
                backend=args.backend,
                model=args.model,
                pass_threshold=args.threshold,
                fail_threshold=args.fail_threshold,
                profile=args.profile,
                distillation_bundle=getattr(args, "bundle", None),
                context=final_context,
                query=getattr(args, "query", None),
                on_unsupported_logprobs=getattr(args, "on_unsupported_logprobs", "raise"),
                sidecar_backend=getattr(args, "sidecar_backend", None),
                telemetry_sink=sink if not has_ast else None,
                cache=cache_obj if not has_ast else False,
            )
        else:
            target_gate = LogitGate(
                rubric=rubric_single,
                backend=args.backend,
                model=args.model,
                pass_threshold=args.threshold,
                fail_threshold=args.fail_threshold,
                profile=args.profile,
                distillation_bundle=getattr(args, "bundle", None),
                context=final_context,
                query=getattr(args, "query", None),
                on_unsupported_logprobs=getattr(args, "on_unsupported_logprobs", "raise"),
                sidecar_backend=getattr(args, "sidecar_backend", None),
                telemetry_sink=sink if not has_ast else None,
                cache=cache_obj if not has_ast else False,
            )

        try:
            evaluator: Cascade | BaseGate
            if has_ast:
                evaluator = Cascade(
                    [
                        ASTGate(
                            syntax=args.ast_syntax,
                            normalize=getattr(args, "normalize", False),
                            cache=False,
                        ),
                        target_gate,
                    ],
                    telemetry_sink=sink,
                    cache=cache_obj,
                )
            else:
                evaluator = target_gate

            verdict = evaluator.evaluate(candidate)
        finally:
            if hasattr(target_gate, "close"):
                target_gate.close()
            elif getattr(target_gate, "backend", None) is not None:
                target_gate.backend.close()
    finally:
        if cache_obj is not None:
            cache_obj.close()
        if sink is not None:
            sink.close()

    if args.json:
        if is_choice or is_score:
            payload = verdict.model_dump(mode="json")
            if is_choice:
                payload["choice"] = verdict.choice
                payload["choice_probabilities"] = verdict.choice_probabilities or {}
            elif is_score:
                payload["score"] = verdict.score
                score_dist: dict[str, float] = {}
                for t in reversed(verdict.traces):
                    if "token_probabilities" in t.metadata:
                        score_dist = t.metadata["token_probabilities"]
                        break
                payload["score_distribution"] = score_dist
            print(json.dumps(payload, indent=2))
        else:
            print(verdict.model_dump_json(indent=2))
    else:
        verdict_str = verdict.status.value.upper()
        print(f"Verdict: {verdict_str} (status: {verdict.status.value})")
        if verdict.probability is not None:
            print(f"Confidence: {verdict.probability:.4f}")
        if is_choice:
            if verdict.choice is not None:
                print(f"Choice: {sanitize_terminal_text(str(verdict.choice))}")
            if verdict.choice_probabilities:
                probs_str = ", ".join(
                    f"{sanitize_terminal_text(k)}={p:.4f}"
                    for k, p in verdict.choice_probabilities.items()
                )
                print(f"Choice Probabilities: {probs_str}")
        elif is_score:
            if verdict.score is not None:
                print(f"Score: {verdict.score:.4f}")
            score_dist_human: dict[str, float] = {}
            for t in reversed(verdict.traces):
                if "token_probabilities" in t.metadata:
                    score_dist_human = t.metadata["token_probabilities"]
                    break
            if score_dist_human:
                dist_str = ", ".join(
                    f"{sanitize_terminal_text(str(k))}={p:.4f}"
                    for k, p in score_dist_human.items()
                )
                print(f"Score Distribution: {dist_str}")
        print(f"Triggered Layer: {sanitize_terminal_text(verdict.triggered_layer or '')}")
        print(f"Total Latency: {verdict.total_latency_ms:.2f}ms")
        if verdict.reason:
            print(f"Reason: {sanitize_terminal_text(verdict.reason)}")

    return EXIT_OK if verdict.passed else EXIT_FAILED
