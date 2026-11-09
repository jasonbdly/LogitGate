from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from logitgate.backends.base import LogitBackend, resolve_logit_backend
from logitgate.config import LogitGateDefaults, get_default_model
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_binary_prompt,
    compile_choice_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import (
    combine_symmetric_log_odds,
    compute_shannon_entropy,
)
from logitgate.debiasing.softmax import (
    clamped_sigmoid as _clamped_sigmoid,
)
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    extract_token_logprob,
    has_any_target_token,
    has_finite_logprobs,
)
from logitgate.debiasing.softmax import (
    softmax_probabilities as _softmax,
)
from logitgate.exceptions import BackendError, ConfigurationError
from logitgate.gates.base import BaseGate
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

TopLogprobs = list[dict[str, Any]]


def _ensure_logprobs_present(top_fwd: TopLogprobs, top_rev: TopLogprobs | None = None) -> None:
    if not has_finite_logprobs(top_fwd) or (top_rev is not None and not has_finite_logprobs(top_rev)):
        raise BackendError(
            "Backend returned no token log-probabilities (top_logprobs is empty). "
            "Ensure the model and inference server support logprobs."
        )


def _resolve_logit_temperature(
    logit_temperature: float | Literal["auto"] | None,
    effective_model: str | None,
    backend: Any,
) -> float:
    """Resolves logit temperature from explicit value, 'auto', global config, or model calibration."""
    if logit_temperature == "auto":
        from logitgate.debiasing.permutation import (
            auto_calibrate_backend,
            get_model_temperature,
        )

        temp = get_model_temperature(effective_model)
        if temp == 1.0 and effective_model not in ("default", "stub", None):
            cal_res = auto_calibrate_backend(backend, model_key=effective_model)
            temp = cal_res.optimal_temperature
        effective_temp = temp
    elif logit_temperature is not None:
        if isinstance(logit_temperature, bool):
            raise ConfigurationError(f"logit_temperature must be a positive float, got {logit_temperature!r}")
        effective_temp = float(logit_temperature)
    else:
        from logitgate.debiasing.permutation import get_model_temperature

        if LogitGateDefaults.logit_temperature != 1.0:
            effective_temp = LogitGateDefaults.logit_temperature
        else:
            model_temp = get_model_temperature(effective_model)
            effective_temp = model_temp if model_temp != 1.0 else LogitGateDefaults.logit_temperature

    if not math.isfinite(effective_temp) or effective_temp <= 0.0:
        raise ConfigurationError(f"logit_temperature must be positive, got {effective_temp}")
    return effective_temp


class LogitGate(BaseGate):
    """First-token Logit Gate (Layer 1).

    Computes calibrated binary classification confidence scores directly from the
    language model's first-token log-probabilities (`max_tokens=1`) without generating
    multi-token Chain-of-Thought critiques. Uses symmetric log-odds debiasing across
    forward and reverse label mappings to cancel prior token biases (Zhao et al., 2021)
    and positional biases.

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
        ConfigurationError: If rubric is empty, thresholds are outside [0, 1],
            pass_threshold <= fail_threshold, or logit_temperature <= 0.
    """

    def __init__(
        self,
        rubric: str,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        pass_threshold: float | None = None,
        fail_threshold: float | None = None,
        symmetric_debiasing: bool | None = None,
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        preset: Literal["strict", "balanced", "fast_triage"] | None = None,
        save_on_fail: bool | None = None,
        max_positional_bias: float | None = None,
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
            elif preset == "fast_triage":
                default_pass = 0.80
                default_fail = 0.20
                default_sym = False
            else:
                raise ConfigurationError(
                    f"Invalid preset '{preset}'. Must be one of ['strict', 'balanced', 'fast_triage']"
                )
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
            raise ConfigurationError("rubric cannot be empty or whitespace")
        if isinstance(effective_pass_thresh, bool) or not (0.0 <= effective_pass_thresh <= 1.0):
            raise ConfigurationError(f"pass_threshold must be between 0.0 and 1.0, got {effective_pass_thresh!r}")
        if isinstance(effective_fail_thresh, bool) or not (0.0 <= effective_fail_thresh <= 1.0):
            raise ConfigurationError(f"fail_threshold must be between 0.0 and 1.0, got {effective_fail_thresh!r}")
        if effective_pass_thresh <= effective_fail_thresh:
            raise ConfigurationError(
                f"pass_threshold ({effective_pass_thresh}) must be strictly greater than fail_threshold ({effective_fail_thresh})"
            )

        if max_positional_bias is not None and (
            isinstance(max_positional_bias, bool)
            or not math.isfinite(float(max_positional_bias))
            or float(max_positional_bias) <= 0.0
        ):
            raise ConfigurationError(
                f"max_positional_bias must be a positive float, got {max_positional_bias!r}"
            )

        self.backend = resolve_logit_backend(backend, model=model)
        effective_model = (
            model
            or getattr(self.backend, "model", None)
            or getattr(self.backend, "model_name", None)
            or get_default_model()
        )
        self.model = effective_model
        effective_temp = _resolve_logit_temperature(logit_temperature, effective_model, self.backend)

        self.rubric = rubric.strip()
        self.preset = preset
        self.save_on_fail = save_on_fail
        self.pass_threshold = effective_pass_thresh
        self.fail_threshold = effective_fail_thresh
        self.symmetric_debiasing = effective_sym
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.max_positional_bias = float(max_positional_bias) if max_positional_bias is not None else None
        self.name = name

    def replay_config(self) -> dict[str, Any]:
        from logitgate.backends.base import infer_backend_spec

        config: dict[str, Any] = {
            "rubric": self.rubric,
            "backend": infer_backend_spec(self.backend),
            "pass_threshold": self.pass_threshold,
            "fail_threshold": self.fail_threshold,
            "symmetric_debiasing": self.symmetric_debiasing,
            "query_first": self.query_first,
            "logit_temperature": self.logit_temperature,
            "preset": self.preset,
            "name": self.name,
        }
        # A backend *instance* carries its own model (e.g. OpenAICompatBackend(model=...)), and
        # `infer_backend_spec` can't encode it for every scheme. Without it, replay would
        # silently run against the scheme's default model.
        model = (
            self.model
            or getattr(self.backend, "model", None)
            or getattr(self.backend, "model_name", None)
        )
        if model:
            config["model"] = model
        return config

    def _handle_failure_snapshot(self, candidate: Any, verdict: Verdict, **kwargs: Any) -> Verdict:
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

    def __repr__(self) -> str:
        rubric_repr = repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
        return (
            f"LogitGate(rubric={rubric_repr}, "
            f"pass_threshold={self.pass_threshold}, "
            f"fail_threshold={self.fail_threshold}, "
            f"backend={self.backend!r})"
        )

    # -- Prompt construction & scoring (shared by sync, async, and batch paths) ------------

    def _binary_prompts(
        self,
        candidate: Any,
        rubrics: Sequence[str] | None = None,
        rubric_index: int = 0,
    ) -> tuple[str, str | None]:
        """Returns the (forward, reverse) prompts; reverse is ``None`` without debiasing.

        Forward binds A=PASS/B=FAIL, reverse binds B=PASS/A=FAIL. Both share a byte-identical
        prefix (rubric + candidate) so the second pass is mostly a KV-cache hit.
        """

        def build(pass_token: str, fail_token: str) -> str:
            if rubrics is None:
                return compile_binary_prompt(
                    rubric=self.rubric,
                    candidate=candidate,
                    pass_token=pass_token,
                    fail_token=fail_token,
                    query_first=self.query_first,
                )
            return compile_batch_binary_prompt(
                rubrics=rubrics,
                candidate=candidate,
                rubric_index=rubric_index,
                pass_token=pass_token,
                fail_token=fail_token,
                query_first=False,
            )

        return build("A", "B"), (build("B", "A") if self.symmetric_debiasing else None)

    def _query(self, prompt: str) -> list[dict[str, Any]]:
        return self.backend.query_top_logprobs(prompt, top_k=10, allowed_tokens=["A", "B"])

    async def _aquery_pair(
        self, forward: str, reverse: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        if reverse is None:
            return await self.backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"]), None
        # Issued concurrently: servers with prefix caching (vLLM, llama.cpp, Ollama) serve the
        # shared prefix once, so the pair costs ~one prefill instead of two.
        top_fwd, top_rev = await asyncio.gather(
            self.backend.aquery_top_logprobs(forward, top_k=10, allowed_tokens=["A", "B"]),
            self.backend.aquery_top_logprobs(reverse, top_k=10, allowed_tokens=["A", "B"]),
        )
        return top_fwd, top_rev

    def _verdict_from_logprobs(
        self,
        top_fwd: list[dict[str, Any]],
        top_rev: list[dict[str, Any]] | None,
        t0: float,
    ) -> Verdict:
        """Turns first-token logprobs into a calibrated, thresholded verdict.

        With a reverse pass, the forward and reverse log-odds are averaged *in log-odds space*.
        A constant preference for token "A" shows up with opposite sign in the two passes and
        cancels exactly (Zhao et al., 2021). Averaging probabilities would only dampen it.
        """
        _ensure_logprobs_present(top_fwd, top_rev)
        fwd_has_target = has_any_target_token(top_fwd, ("A", "B"))
        rev_has_target = True if top_rev is None else has_any_target_token(top_rev, ("A", "B"))
        l1, p1 = compute_binary_log_odds(top_fwd, pass_token="A", fail_token="B")
        temp = max(self.logit_temperature, 1e-6)
        l2: float | None = None
        p2: float | None = None
        pos_bias: float | None = None
        if top_rev is not None:
            l2, p2 = compute_binary_log_odds(top_rev, pass_token="B", fail_token="A")
            calibrated_prob = combine_symmetric_log_odds(
                forward_log_odds=l1,
                reverse_log_odds=l2,
                logit_temperature=self.logit_temperature,
            )
            pos_bias = 0.5 * abs(l1 - l2)
            log_odds_pass = (l1 + l2) / (2.0 * temp)
        else:
            log_odds_pass = l1 / temp
            calibrated_prob = _clamped_sigmoid(log_odds_pass)

        shannon_entropy = compute_shannon_entropy(calibrated_prob)
        latency = (time.perf_counter() - t0) * 1000.0
        prob_rounded = round(calibrated_prob, 4)

        abstained_reason: str | None = None
        if not fwd_has_target or not rev_has_target:
            status = DecisionStatus.AMBIGUOUS
            abstained_reason = "unobserved_target_tokens"
        elif (
            self.max_positional_bias is not None
            and pos_bias is not None
            and l2 is not None
            and (l1 * l2 < 0.0)
            and pos_bias > self.max_positional_bias
        ):
            status = DecisionStatus.AMBIGUOUS
            abstained_reason = f"positional_bias_exceeded ({pos_bias:.4f} > {self.max_positional_bias:.4f})"
        elif prob_rounded >= self.pass_threshold:
            status = DecisionStatus.PASSED
        elif prob_rounded <= self.fail_threshold:
            status = DecisionStatus.FAILED
        else:
            status = DecisionStatus.AMBIGUOUS

        bias_rounded = round(pos_bias, 4) if pos_bias is not None else None
        meta: dict[str, Any] = {
            "positional_bias_magnitude": bias_rounded,
            "log_odds_pass": round(log_odds_pass, 4),
            "shannon_entropy": round(shannon_entropy, 4),
        }
        if abstained_reason is not None:
            meta["abstained_reason"] = abstained_reason
        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            calibrated_prob=prob_rounded,
            forward_prob=round(p1, 4),
            reverse_prob=round(p2, 4) if p2 is not None else None,
            forward_log_odds=round(l1, 4),
            reverse_log_odds=round(l2, 4) if l2 is not None else None,
            positional_bias_magnitude=bias_rounded,
            log_odds_pass=round(log_odds_pass, 4),
            shannon_entropy=round(shannon_entropy, 4),
            tokens_generated=0,
            metadata=meta,
        )
        return Verdict(
            passed=status == DecisionStatus.PASSED,
            status=status,
            probability=prob_rounded,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"LogitGate prob={prob_rounded:.4f} (pass_thresh={self.pass_threshold}, fail_thresh={self.fail_threshold})",
            traces=[trace],
        )

    # -- Public evaluation API --------------------------------------------------------------

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        forward, reverse = self._binary_prompts(candidate)
        top_fwd = self._query(forward)
        top_rev = self._query(reverse) if reverse is not None else None
        return self._handle_failure_snapshot(candidate, self._verdict_from_logprobs(top_fwd, top_rev, t0), **kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        top_fwd, top_rev = await self._aquery_pair(*self._binary_prompts(candidate))
        return self._handle_failure_snapshot(candidate, self._verdict_from_logprobs(top_fwd, top_rev, t0), **kwargs)

    def evaluate_batch(self, candidate: Any, rubrics: Sequence[str]) -> list[Verdict]:
        """Evaluates several atomic rubrics against one candidate.

        The candidate is placed first in every prompt, so all 2 x len(rubrics) queries share
        one cached prefix and only the short rubric suffix is re-prefilled per query.
        """
        if isinstance(rubrics, (str, bytes)):
            raise ConfigurationError("rubrics must be a sequence of rubric strings, not a single string")
        for r in rubrics:
            if not isinstance(r, str) or not r.strip():
                raise ConfigurationError("Each rubric in rubrics must be a non-empty string")
        results = []
        for idx in range(len(rubrics)):
            t0 = time.perf_counter()
            forward, reverse = self._binary_prompts(candidate, rubrics, idx)
            top_fwd = self._query(forward)
            top_rev = self._query(reverse) if reverse is not None else None
            results.append(self._verdict_from_logprobs(top_fwd, top_rev, t0))
        return results

    async def aevaluate_batch(self, candidate: Any, rubrics: Sequence[str]) -> list[Verdict]:
        """Async `evaluate_batch`. The first rubric runs alone to warm the shared prefix.

        Firing every query at a cold server makes each one prefill the full candidate in
        parallel (a cache stampede). One sequential warm-up turns the rest into cache hits.
        """
        if isinstance(rubrics, (str, bytes)):
            raise ConfigurationError("rubrics must be a sequence of rubric strings, not a single string")
        for r in rubrics:
            if not isinstance(r, str) or not r.strip():
                raise ConfigurationError("Each rubric in rubrics must be a non-empty string")
        if not rubrics:
            return []

        sem = asyncio.Semaphore(32)

        async def _eval_single(idx: int) -> Verdict:
            async with sem:
                t0 = time.perf_counter()
                top_fwd, top_rev = await self._aquery_pair(*self._binary_prompts(candidate, rubrics, idx))
                return self._verdict_from_logprobs(top_fwd, top_rev, t0)

        first_verdict = await _eval_single(0)
        rest_verdicts = await asyncio.gather(*[_eval_single(idx) for idx in range(1, len(rubrics))])
        return [first_verdict, *rest_verdicts]


class MultiRubricGate(BaseGate):
    """Checks several *atomic* rubrics against one candidate; passes only if every rubric passes.

    Prefer this to one compound rubric ("is it typed AND documented AND safe?"). A compound
    question blends independent signals into one logit, and a strong "yes" on two criteria can
    mask a "no" on the third. Here each rubric gets its own debiased read, and all reads share
    the candidate's KV-cache prefix (see `LogitGate.evaluate_batch`).

    Aggregation is weakest-link: any FAILED rubric gives FAILED, otherwise any AMBIGUOUS gives
    AMBIGUOUS, and ``probability`` is the minimum across rubrics. One trace per rubric is
    recorded (``metadata["rubric"]``), so `logitgate explain` shows which criterion failed.

    Inside a `Cascade`, a downstream `JudgeGate` inherits all rubrics as a bulleted list.

    Args:
        rubrics: Atomic yes/no criteria, one question each.
        name: Identifier for this gate instance.
        **options: Any `LogitGate` option (``backend``, ``model``, ``pass_threshold``,
            ``fail_threshold``, ``symmetric_debiasing``, ``logit_temperature``, ``preset``...).

    Example:
        >>> gate = MultiRubricGate([
        ...     "Does the function have type annotations?",
        ...     "Does the function handle n <= 0?",
        ... ])
        >>> verdict = gate(candidate_code)
        >>> [t.metadata["rubric"] for t in verdict.traces if t.status != "passed"]
    """

    def __init__(self, rubrics: Sequence[str], name: str = "MultiRubricGate", **options: Any) -> None:
        if isinstance(rubrics, (str, bytes)):
            raise ConfigurationError("MultiRubricGate rubrics must be a sequence of strings, not a single string")
        cleaned = [r.strip() for r in rubrics if isinstance(r, str) and r.strip()]
        if len(cleaned) != len(rubrics) or not cleaned:
            raise ConfigurationError("MultiRubricGate needs one or more non-empty rubric strings")
        self.rubrics = cleaned
        self.name = name
        self._gate = LogitGate(rubric=cleaned[0], name=name, **options)
        self.save_on_fail = self._gate.save_on_fail

    @property
    def rubric(self) -> str:
        """All criteria as one bulleted rubric, the form a downstream judge receives."""
        return "\n".join(f"- {r}" for r in self.rubrics)

    @property
    def backend(self) -> LogitBackend:
        return self._gate.backend

    def __repr__(self) -> str:
        return f"MultiRubricGate(rubrics={len(self.rubrics)}, backend={self.backend!r})"

    def evaluate_each(self, candidate: Any) -> list[Verdict]:
        """Per-rubric verdicts, in `rubrics` order."""
        return self._gate.evaluate_batch(candidate, self.rubrics)

    async def aevaluate_each(self, candidate: Any) -> list[Verdict]:
        return await self._gate.aevaluate_batch(candidate, self.rubrics)

    def _aggregate(self, verdicts: Sequence[Verdict], t0: float) -> Verdict:
        traces: list[LayerTrace] = []
        for i, (rubric, v) in enumerate(zip(self.rubrics, verdicts)):
            for trace in v.traces:
                traces.append(trace.model_copy(update={
                    "layer_name": f"{self.name}[{i}]",
                    "metadata": {**trace.metadata, "rubric": rubric},
                }))
        statuses = {v.status for v in verdicts}
        if DecisionStatus.FAILED in statuses:
            status = DecisionStatus.FAILED
        elif DecisionStatus.AMBIGUOUS in statuses:
            status = DecisionStatus.AMBIGUOUS
        else:
            status = DecisionStatus.PASSED
        probs = [v.probability for v in verdicts if v.probability is not None]
        unresolved = [
            f"{v.status.value.upper()} {rubric!r} (P={v.probability:.4f})"
            for rubric, v in zip(self.rubrics, verdicts)
            if not v.passed and v.probability is not None
        ]
        passed_count = sum(v.passed for v in verdicts)
        reason = f"{passed_count}/{len(verdicts)} rubrics passed"
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
        return {**config, "gate_type": "MultiRubricGate", "rubrics": list(self.rubrics)}

    def _snapshot(self, candidate: Any, verdict: Verdict, kwargs: dict[str, Any]) -> Verdict:
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
        t0 = time.perf_counter()
        return self._snapshot(candidate, self._aggregate(self.evaluate_each(candidate), t0), kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        return self._snapshot(candidate, self._aggregate(await self.aevaluate_each(candidate), t0), kwargs)


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
        ConfigurationError: If question is empty, options has fewer than 2 items, or logit_temperature <= 0.
    """

    def __init__(
        self,
        question: str,
        options: Mapping[str, str],
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        symmetric_debiasing: bool | None = True,
        permutation_mode: Literal["derangement", "cyclic", "reverse"] = "derangement",
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        name: str = "ChoiceGate",
    ) -> None:
        if not question or not question.strip():
            raise ConfigurationError("question cannot be empty or whitespace")
        if permutation_mode not in ("derangement", "cyclic", "reverse"):
            raise ConfigurationError(
                f"Invalid permutation_mode '{permutation_mode}'. Must be one of ['derangement', 'cyclic', 'reverse']"
            )
        if not isinstance(options, Mapping) or len(options) < 2:
            raise ConfigurationError(
                f"ChoiceGate requires at least 2 options, got {len(options) if isinstance(options, Mapping) else type(options).__name__}"
            )
        invalid_keys = [k for k in options if not isinstance(k, str) or len(k.strip()) != 1]
        if invalid_keys:
            raise ConfigurationError(
                f"ChoiceGate option keys must be single characters (e.g. 'A', 'B', 'C'), got: {invalid_keys}"
            )
        normalized_options: dict[str, str] = {}
        seen_upper: set[str] = set()
        for k, v in options.items():
            if not isinstance(v, str) or not v.strip():
                raise ConfigurationError(f"ChoiceGate option description for '{k}' must be a non-empty string")
            k_clean = k.strip()
            k_upper = k_clean.upper()
            if k_upper in seen_upper:
                raise ConfigurationError(
                    f"ChoiceGate option keys must be case-insensitively unique, got duplicate for '{k_clean}'"
                )
            seen_upper.add(k_upper)
            normalized_options[k_clean] = v

        resolved_backend = resolve_logit_backend(backend, model=model)
        effective_model = (
            model
            or getattr(resolved_backend, "model", None)
            or getattr(resolved_backend, "model_name", None)
            or get_default_model()
        )
        effective_temp = _resolve_logit_temperature(logit_temperature, effective_model, resolved_backend)

        self.question = question.strip()
        self.options = normalized_options
        self.backend = resolved_backend
        self.model = effective_model
        self.symmetric_debiasing = True if symmetric_debiasing is None else bool(symmetric_debiasing)
        self.permutation_mode = permutation_mode
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.name = name

    def __repr__(self) -> str:
        q_repr = repr(self.question[:40] + "..." if len(self.question) > 40 else self.question)
        return f"ChoiceGate(question={q_repr}, options={list(self.options.keys())!r}, backend={self.backend!r})"

    def _permuted_key_indices(self) -> list[list[int]]:
        """Returns the list of pass permutations where ``perm[i]`` is the key index assigned to option ``i``.

        - ``derangement`` (2 passes): Pass 1 is identity ``i -> i``; Pass 2 shifts by ``max(1, n // 2)``
          so ``i -> (i + shift) % n``. Unlike list reversal on odd ``n``, this guarantees zero fixed
          points (no middle option stays pinned to the same letter and line in both passes).
        - ``cyclic`` (n passes): Runs all ``n`` cyclic shifts ``(i + s) % n`` for ``s in 0..n-1`` so
          every option visits every letter and slot position exactly once (Zheng et al., 2023).
        - ``reverse`` (2 passes): Legacy list reversal ``i -> n - 1 - i``.
        """
        n = len(self.options)
        identity = list(range(n))
        if not self.symmetric_debiasing:
            return [identity]
        if self.permutation_mode == "cyclic":
            return [[(i + s) % n for i in range(n)] for s in range(n)]
        if self.permutation_mode == "reverse":
            return [identity, [n - 1 - i for i in range(n)]]
        shift = max(1, n // 2)
        return [identity, [(i + shift) % n for i in range(n)]]

    def _choice_prompts_list(self, candidate: Any) -> list[str]:
        keys = list(self.options.keys())
        values = list(self.options.values())
        n = len(keys)
        prompts: list[str] = []
        for perm in self._permuted_key_indices():
            # perm[i] is the key index where option i's description is placed
            slot_to_desc: dict[str, str] = {}
            for key_idx in range(n):
                opt_idx = perm.index(key_idx)
                slot_to_desc[keys[key_idx]] = values[opt_idx]
            prompts.append(
                compile_choice_prompt(
                    question=self.question,
                    candidate=candidate,
                    options=slot_to_desc,
                    query_first=self.query_first,
                )
            )
        return prompts

    def _choice_prompts(self, candidate: Any) -> tuple[str, str | None]:
        prompts = self._choice_prompts_list(candidate)
        return prompts[0], (prompts[1] if len(prompts) > 1 else None)

    def _verdict_from_logprobs_list(self, top_list: list[TopLogprobs], t0: float) -> Verdict:
        if not top_list or any(not has_finite_logprobs(top) for top in top_list):
            raise BackendError(
                "Backend returned no token log-probabilities (top_logprobs is empty). "
                "Ensure the model and inference server support logprobs."
            )
        keys = list(self.options)
        perms = self._permuted_key_indices()[: len(top_list)]
        accum = {k: 0.0 for k in keys}
        for top, perm in zip(top_list, perms):
            # In this pass, option i sits under keys[perm[i]]
            pass_raw = {keys[i]: extract_token_logprob(top, keys[perm[i]]) for i in range(len(keys))}
            m_pass = max(pass_raw.values())
            lse_pass = m_pass + math.log(sum(math.exp(v - m_pass) for v in pass_raw.values()))
            for k in keys:
                accum[k] += (pass_raw[k] - lse_pass) / float(len(perms))

        choice_probs = _softmax(accum, self.logit_temperature)
        best_choice, max_prob = max(choice_probs.items(), key=lambda kv: kv[1])
        rounded = {k: round(v, 4) for k, v in choice_probs.items()}
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=DecisionStatus.PASSED,
            latency_ms=latency,
            calibrated_prob=round(max_prob, 4),
            metadata={"choice_probabilities": rounded, "permutation_mode": self.permutation_mode},
        )
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            probability=round(max_prob, 4),
            choice=best_choice,
            choice_probabilities=rounded,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"ChoiceGate selected '{best_choice}' with P={max_prob:.4f}",
            traces=[trace],
        )

    def _verdict_from_logprobs(self, top_fwd: TopLogprobs, top_rev: TopLogprobs | None, t0: float) -> Verdict:
        top_list = [top_fwd] if top_rev is None else [top_fwd, top_rev]
        return self._verdict_from_logprobs_list(top_list, t0)

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        keys = list(self.options)
        top_k = max(10, 2 * len(keys))
        prompts = self._choice_prompts_list(candidate)
        top_list = [
            self.backend.query_top_logprobs(p, top_k=top_k, allowed_tokens=keys)
            for p in prompts
        ]
        return self._verdict_from_logprobs_list(top_list, t0)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        keys = list(self.options)
        top_k = max(10, 2 * len(keys))
        prompts = self._choice_prompts_list(candidate)
        if len(prompts) == 1:
            top_list = [await self.backend.aquery_top_logprobs(prompts[0], top_k=top_k, allowed_tokens=keys)]
        else:
            top_list = list(
                await asyncio.gather(
                    *[self.backend.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=keys) for p in prompts]
                )
            )
        return self._verdict_from_logprobs_list(top_list, t0)


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
        ConfigurationError: If rubric is empty, min_score >= max_score, threshold is out of bounds,
            or logit_temperature <= 0.
    """

    def __init__(
        self,
        rubric: str,
        min_score: int = 1,
        max_score: int = 5,
        threshold: float = 3.5,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        symmetric_debiasing: bool | None = True,
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        save_on_fail: bool | None = None,
        name: str = "ScoreGate",
    ) -> None:
        if not rubric or not rubric.strip():
            raise ConfigurationError("rubric cannot be empty or whitespace")
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, int)
            or isinstance(max_score, bool)
            or not isinstance(max_score, int)
        ):
            raise ConfigurationError(
                f"min_score and max_score must be integers, got min_score={min_score!r}, max_score={max_score!r}"
            )
        if min_score >= max_score:
            raise ConfigurationError(f"min_score ({min_score}) must be strictly less than max_score ({max_score})")
        if min_score < 0 or max_score > 9:
            raise ConfigurationError(
                f"ScoreGate requires single-digit integer bounds (0 <= min_score < max_score <= 9), "
                f"got min_score={min_score}, max_score={max_score}"
            )
        if (
            isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not (min_score <= threshold <= max_score)
        ):
            raise ConfigurationError(
                f"threshold ({threshold}) must be between min_score ({min_score}) and max_score ({max_score})"
            )

        resolved_backend = resolve_logit_backend(backend, model=model)
        effective_model = (
            model
            or getattr(resolved_backend, "model", None)
            or getattr(resolved_backend, "model_name", None)
            or get_default_model()
        )
        effective_temp = _resolve_logit_temperature(logit_temperature, effective_model, resolved_backend)

        self.rubric = rubric.strip()
        self.min_score = min_score
        self.max_score = max_score
        self.threshold = threshold
        self.backend = resolved_backend
        self.model = effective_model
        self.symmetric_debiasing = True if symmetric_debiasing is None else bool(symmetric_debiasing)
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self.save_on_fail = save_on_fail
        self.name = name

    def replay_config(self) -> dict[str, Any]:
        from logitgate.backends.base import infer_backend_spec

        config: dict[str, Any] = {
            "gate_type": "ScoreGate",
            "rubric": self.rubric,
            "backend": infer_backend_spec(self.backend),
            "min_score": self.min_score,
            "max_score": self.max_score,
            "threshold": self.threshold,
            "symmetric_debiasing": self.symmetric_debiasing,
            "query_first": self.query_first,
            "logit_temperature": self.logit_temperature,
            "name": self.name,
        }
        model = (
            self.model
            or getattr(self.backend, "model", None)
            or getattr(self.backend, "model_name", None)
        )
        if model:
            config["model"] = model
        return config

    def _handle_failure_snapshot(self, candidate: Any, verdict: Verdict, **kwargs: Any) -> Verdict:
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

    def __repr__(self) -> str:
        rubric_repr = repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
        return (
            f"ScoreGate(rubric={rubric_repr}, "
            f"range=[{self.min_score}..{self.max_score}], "
            f"threshold={self.threshold}, "
            f"backend={self.backend!r})"
        )

    def _score_prompts(self, candidate: Any) -> tuple[str, str | None]:
        def build(inverted: bool) -> str:
            return compile_score_prompt(
                rubric=self.rubric,
                candidate=candidate,
                min_score=self.min_score,
                max_score=self.max_score,
                query_first=self.query_first,
                inverted=inverted,
            )

        return build(False), (build(True) if self.symmetric_debiasing else None)

    def _verdict_from_logprobs(self, top_fwd: TopLogprobs, top_rev: TopLogprobs | None, t0: float) -> Verdict:
        _ensure_logprobs_present(top_fwd, top_rev)
        scores = range(self.min_score, self.max_score + 1)
        fwd_raw = {s: extract_token_logprob(top_fwd, str(s)) for s in scores}
        m_fwd = max(fwd_raw.values())
        lse_fwd = m_fwd + math.log(sum(math.exp(v - m_fwd) for v in fwd_raw.values()))
        combined = {s: v - lse_fwd for s, v in fwd_raw.items()}
        if top_rev is not None:
            # On the inverted scale, quality s is written as token (max + min - s).
            mirror = self.max_score + self.min_score
            rev_raw = {s: extract_token_logprob(top_rev, str(mirror - s)) for s in scores}
            m_rev = max(rev_raw.values())
            lse_rev = m_rev + math.log(sum(math.exp(v - m_rev) for v in rev_raw.values()))
            combined = {s: (combined[s] + (rev_raw[s] - lse_rev)) / 2.0 for s in scores}

        score_probs = _softmax(combined, self.logit_temperature)
        expected_score = round(sum(s * prob for s, prob in score_probs.items()), 4)
        passed = expected_score >= self.threshold
        status = DecisionStatus.PASSED if passed else DecisionStatus.FAILED
        latency = (time.perf_counter() - t0) * 1000.0

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            metadata={
                "expected_score": expected_score,
                "token_probabilities": {str(k): round(v, 4) for k, v in score_probs.items()},
            },
        )
        return Verdict(
            passed=passed,
            status=status,
            score=expected_score,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=f"ScoreGate E[S]={expected_score:.2f} (threshold={self.threshold})",
            traces=[trace],
        )

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]
        top_k = max(10, 2 * len(tokens))
        forward, reverse = self._score_prompts(candidate)
        top_fwd = self.backend.query_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
        top_rev = (
            self.backend.query_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens)
            if reverse is not None
            else None
        )
        return self._handle_failure_snapshot(
            candidate, self._verdict_from_logprobs(top_fwd, top_rev, t0), **kwargs
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        t0 = time.perf_counter()
        tokens = [str(s) for s in range(self.min_score, self.max_score + 1)]
        top_k = max(10, 2 * len(tokens))
        forward, reverse = self._score_prompts(candidate)
        top_rev: TopLogprobs | None = None
        if reverse is None:
            top_fwd = await self.backend.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens)
        else:
            top_fwd, top_rev = await asyncio.gather(
                self.backend.aquery_top_logprobs(forward, top_k=top_k, allowed_tokens=tokens),
                self.backend.aquery_top_logprobs(reverse, top_k=top_k, allowed_tokens=tokens),
            )
        return self._handle_failure_snapshot(
            candidate, self._verdict_from_logprobs(top_fwd, top_rev, t0), **kwargs
        )
