from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logitgate.backends.base import LogitBackend
from logitgate.calibration.lifecycle import (
    _validate_optional_bool,
)
from logitgate.exceptions import ConfigurationError
from logitgate.gates.base import BaseGate
from logitgate.gates.rank_gate import (
    BestOfNResult,
    PermutationMode,
    RankGate,
    RankMode,
    _normalize_candidates,
    _tag_traces_with_key,
)
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

if TYPE_CHECKING:
    from logitgate.calibration.profile import CalibrationProfile


def _get_rg_mod() -> Any:
    return sys.modules.get("logitgate.gates.rank_gate")


def _resolve_logit_backend(*args: Any, **kwargs: Any) -> Any:
    mod = _get_rg_mod()
    fn = getattr(mod, "resolve_logit_backend", None) if mod else None
    if fn is not None:
        return fn(*args, **kwargs)
    from logitgate.backends.base import resolve_logit_backend as canonical
    return canonical(*args, **kwargs)


class BestOfNGate(BaseGate):
    """Contract-filtered best-of-N selection pipeline (`pre_gate -> RankGate -> post_gate`).

    Filters candidates with an optional deterministic/cheap ``pre_gate`` (skipping ranking when
    0 or 1 survive), ranks survivors with symmetric swap debiasing, optionally verifies the
    selected winner with ``post_gate`` (catching all-bad candidate pools), and forwards the
    winning artifact downstream in `Cascade` via ``_normalized_out``.
    """

    def __init__(
        self,
        rubric: str,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        *,
        pre_gate: BaseGate | None = None,
        post_gate: BaseGate | None = None,
        mode: RankMode = "auto",
        context: str | None = None,
        symmetric_debiasing: bool | None = True,
        permutation_mode: PermutationMode = "derangement",
        query_first: bool = True,
        logit_temperature: float | Literal["auto"] | None = None,
        min_confidence: float | None = None,
        min_margin: float | None = None,
        max_positional_bias: float | None = None,
        save_on_fail: bool | None = None,
        warm_prefix: bool | None = None,
        profile: CalibrationProfile | str | Path | None = None,
        telemetry_sink: Any = None,
        cache: Any = None,
        name: str = "BestOfNGate",
    ) -> None:
        from logitgate.cache import resolve_verdict_cache
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        if pre_gate is not None and not isinstance(pre_gate, BaseGate):
            raise ConfigurationError(f"pre_gate must be a BaseGate or None, got {type(pre_gate).__name__}")
        if post_gate is not None and not isinstance(post_gate, BaseGate):
            raise ConfigurationError(f"post_gate must be a BaseGate or None, got {type(post_gate).__name__}")
        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        owns_sink = isinstance(telemetry_sink, str) or hasattr(telemetry_sink, "__fspath__")
        try:
            effective_save_on_fail = _validate_optional_bool("save_on_fail", save_on_fail)
            resolved_cache = resolve_verdict_cache(cache)
            self.rank_gate = RankGate(
                rubric=rubric,
                backend=backend,
                model=model,
                mode=mode,
                context=context,
                symmetric_debiasing=symmetric_debiasing,
                permutation_mode=permutation_mode,
                query_first=query_first,
                logit_temperature=logit_temperature,
                min_confidence=min_confidence,
                min_margin=min_margin,
                max_positional_bias=max_positional_bias,
                save_on_fail=False,
                warm_prefix=warm_prefix,
                profile=profile,
                telemetry_sink=False,
                cache=False,
                name=name,
            )
        except BaseException:
            if owns_sink and resolved_sink not in (None, False):
                resolved_sink.close()  # type: ignore[union-attr]
            raise
        self.pre_gate = pre_gate
        self.post_gate = post_gate
        self.backend = self.rank_gate.backend
        self.model = self.rank_gate.model
        self.rubric = self.rank_gate.rubric
        self.profile = self.rank_gate.profile
        self.save_on_fail = effective_save_on_fail
        self.telemetry_sink = resolved_sink
        self.cache = resolved_cache
        self.name = name
        self._owns_backend = False
        self._owns_cache = isinstance(cache, (str, Path))
        self._owns_sink = owns_sink

    def _ensure_calibrated(self) -> None:
        self.rank_gate._ensure_calibrated()
        for child in (self.pre_gate, self.post_gate):
            if child is not None and callable(getattr(child, "_ensure_calibrated", None)):
                child._ensure_calibrated()  # type: ignore[attr-defined]

    async def _aensure_calibrated(self) -> None:
        await self.rank_gate._aensure_calibrated()
        for child in (self.pre_gate, self.post_gate):
            if child is not None and callable(getattr(child, "_aensure_calibrated", None)):
                await child._aensure_calibrated()  # type: ignore[attr-defined]

    def replay_config(self) -> dict[str, Any]:
        cfg = self.rank_gate.replay_config()
        cfg["gate_type"] = "BestOfNGate"
        cfg["name"] = self.name
        if self.pre_gate is not None:
            pre_cfg = self.pre_gate.replay_config()
            if pre_cfg is not None:
                cfg["pre_gate_config"] = pre_cfg
        if self.post_gate is not None:
            post_cfg = self.post_gate.replay_config()
            if post_cfg is not None:
                cfg["post_gate_config"] = post_cfg
        return cfg

    def _handle_failure_snapshot(
        self,
        candidate: Any,
        verdict: Verdict,
        *,
        _entries: Sequence[tuple[str, Any]] | None = None,
        _is_cache_hit: bool = False,
        **kwargs: Any,
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
            has_win = bool(verdict.passed and verdict.choice is not None and _entries is not None)
            win_cand = dict(_entries)[verdict.choice] if (has_win and _entries is not None and verdict.choice is not None) else None
            store_cached_verdict(
                self,
                candidate,
                verdict,
                gate_cache=self.cache,
                call_kwargs=kwargs,
                has_normalized=has_win,
                normalized_artifact=win_cand,
            )

        emit_verdict_telemetry(
            verdict,
            candidate=candidate,
            gate_name=self.name,
            gate_type="BestOfNGate",
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

    def _assemble_pre_stage(
        self,
        entries: list[tuple[str, Any]],
        pre_verdicts: Sequence[Verdict] | None,
        t0: float,
    ) -> tuple[list[tuple[str, Any]], Verdict | None, list[LayerTrace], int, list[str]]:
        """Processes pre-gate filtering results before ranking or sole-survivor promotion."""
        all_keys = [k for k, _ in entries]
        if self.pre_gate is None or pre_verdicts is None:
            return entries, None, [], 0, []

        pre_traces: list[LayerTrace] = []
        pre_tokens = 0
        survivors: list[tuple[str, Any]] = []
        rejected_keys: list[str] = []
        sole_orig_idx = 0
        for orig_idx, ((k, cand), pv) in enumerate(zip(entries, pre_verdicts, strict=False)):
            pre_traces.extend(_tag_traces_with_key(pv.traces, k))
            pre_tokens += pv.total_tokens_generated
            if pv.passed:
                survivors.append((k, cand))
                sole_orig_idx = orig_idx
            else:
                rejected_keys.append(k)

        if not survivors:
            latency = (time.perf_counter() - t0) * 1000.0
            fail_trace = LayerTrace(
                layer_name=self.name,
                status=DecisionStatus.FAILED,
                latency_ms=latency,
                metadata={
                    "surviving_keys": [],
                    "rejected_keys": all_keys,
                    "ranking": list(all_keys),
                    "llm_calls": 0,
                    "winner_index": None,
                },
            )
            verdict = Verdict(
                passed=False,
                status=DecisionStatus.FAILED,
                probability=0.0,
                choice=None,
                choice_probabilities={k: 0.0 for k in all_keys},
                triggered_layer=self.pre_gate.name,
                total_latency_ms=latency,
                total_tokens_generated=pre_tokens,
                reason=f"{self.name}: all {len(entries)} candidates rejected by pre_gate '{self.pre_gate.name}'.",
                traces=[*pre_traces, fail_trace],
            )
            return survivors, verdict, pre_traces, pre_tokens, rejected_keys

        if len(survivors) == 1:
            sole_key = survivors[0][0]
            latency = (time.perf_counter() - t0) * 1000.0
            full_probs = {k: (1.0 if k == sole_key else 0.0) for k in all_keys}
            ranking = [sole_key, *rejected_keys]
            sole_trace = LayerTrace(
                layer_name=self.name,
                status=DecisionStatus.PASSED,
                calibrated_prob=1.0,
                latency_ms=latency,
                metadata={
                    "sole_survivor": True,
                    "surviving_keys": [sole_key],
                    "rejected_keys": rejected_keys,
                    "ranking": ranking,
                    "choice_probabilities": full_probs,
                    "winner_index": sole_orig_idx,
                    "llm_calls": 0,
                },
            )
            verdict = Verdict(
                passed=True,
                status=DecisionStatus.PASSED,
                probability=1.0,
                choice=sole_key,
                choice_probabilities=full_probs,
                triggered_layer=self.name,
                total_latency_ms=latency,
                total_tokens_generated=pre_tokens,
                reason=f"{self.name}: sole surviving candidate '{sole_key}' promoted without ranking.",
                traces=[*pre_traces, sole_trace],
            )
            return survivors, verdict, pre_traces, pre_tokens, rejected_keys

        return survivors, None, pre_traces, pre_tokens, rejected_keys

    def _merge_rank_verdict(
        self,
        entries: list[tuple[str, Any]],
        survivors: Sequence[tuple[str, Any]],
        rejected_keys: Sequence[str],
        rank_v: Verdict,
        pre_traces: Sequence[LayerTrace],
        pre_tokens: int,
        t0: float,
    ) -> Verdict:
        all_keys = [k for k, _ in entries]
        key_to_orig_idx = {k: idx for idx, (k, _) in enumerate(entries)}
        surv_keys = [k for k, _ in survivors]
        full_probs = {
            k: round(float((rank_v.choice_probabilities or {}).get(k, 0.0)), 4)
            for k in all_keys
        }
        orig_meta = rank_v.traces[-1].metadata if rank_v.traces else {}
        surv_ranking = list(orig_meta.get("ranking", surv_keys))
        full_ranking = surv_ranking + [k for k in rejected_keys if k not in surv_ranking]
        remapped_winner_idx = (
            key_to_orig_idx.get(rank_v.choice) if rank_v.choice is not None else None
        )

        merged_traces = list(pre_traces)
        for i, tr in enumerate(rank_v.traces):
            if i == len(rank_v.traces) - 1:
                updated_meta = {
                    **tr.metadata,
                    "surviving_keys": surv_keys,
                    "rejected_keys": list(rejected_keys),
                    "ranking": full_ranking,
                    "choice_probabilities": full_probs,
                    "winner_index": remapped_winner_idx,
                }
                merged_traces.append(tr.model_copy(update={"metadata": updated_meta}))
            else:
                merged_traces.append(tr)

        latency = (time.perf_counter() - t0) * 1000.0
        return Verdict(
            passed=rank_v.passed,
            status=rank_v.status,
            probability=rank_v.probability,
            score=rank_v.score,
            choice=rank_v.choice,
            choice_probabilities=full_probs,
            triggered_layer=self.name if rank_v.triggered_layer == self.rank_gate.name else rank_v.triggered_layer,
            total_latency_ms=latency,
            total_tokens_generated=pre_tokens + rank_v.total_tokens_generated,
            reason=rank_v.reason,
            traces=merged_traces,
        )

    def _merge_post_verdict(
        self,
        entries: list[tuple[str, Any]],
        winner_key: str,
        winner_idx: int,
        base_v: Verdict,
        post_v: Verdict,
        call_kwargs: Mapping[str, Any],
        t0: float,
    ) -> Verdict:
        all_keys = [k for k, _ in entries]
        full_probs = dict(base_v.choice_probabilities or {k: 0.0 for k in all_keys})
        combined_traces = list(base_v.traces) + list(post_v.traces)
        if combined_traces and base_v.traces and post_v.traces:
            base_meta = base_v.traces[-1].metadata
            carry = {
                k: base_meta[k]
                for k in ("surviving_keys", "rejected_keys", "ranking", "winner_index")
                if k in base_meta and k not in combined_traces[-1].metadata
            }
            if carry:
                combined_traces[-1] = combined_traces[-1].model_copy(
                    update={"metadata": {**combined_traces[-1].metadata, **carry}}
                )
        latency = (time.perf_counter() - t0) * 1000.0
        tokens = base_v.total_tokens_generated + post_v.total_tokens_generated
        if post_v.passed:
            norm_out = call_kwargs.get("_normalized_out")
            if isinstance(norm_out, list):
                norm_out.append(entries[winner_idx][1])
        return Verdict(
            passed=post_v.passed,
            status=post_v.status,
            probability=post_v.probability if post_v.probability is not None else base_v.probability,
            score=post_v.score,
            choice=winner_key,
            choice_probabilities=full_probs,
            triggered_layer=post_v.triggered_layer,
            total_latency_ms=latency,
            total_tokens_generated=tokens,
            reason=post_v.reason or base_v.reason,
            traces=combined_traces,
        )

    def _build_rank_kwargs(self, call_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        rk: dict[str, Any] = {
            "save_on_fail": False,
            "_suppress_telemetry": True,
            "_bypass_cache": True,
        }
        if "context" in call_kwargs:
            rk["context"] = call_kwargs["context"]
        return rk

    def _has_active_top_cache(self, call_kwargs: Mapping[str, Any]) -> bool:
        if call_kwargs.get("_bypass_cache"):
            return False
        call_c = call_kwargs.get("cache")
        if call_c is False:
            return False
        if call_c is not None:
            return True
        if self.cache is False:
            return False
        if self.cache is not None:
            return True
        from logitgate.config import get_default_verdict_cache

        return get_default_verdict_cache() is not None

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        entries = _normalize_candidates(candidate)
        self._ensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._handle_failure_snapshot(
                candidate, hit, _is_cache_hit=True, **kwargs
            )
        has_top_cache = self._has_active_top_cache(kwargs)
        t0 = time.perf_counter()
        pre_verdicts = (
            [
                self.pre_gate.evaluate(
                    cand,
                    save_on_fail=False,
                    _suppress_telemetry=True,
                    _bypass_cache=has_top_cache,
                )
                for _, cand in entries
            ]
            if self.pre_gate is not None
            else None
        )
        survivors, short_circuit, pre_traces, pre_tokens, rejected_keys = self._assemble_pre_stage(
            entries, pre_verdicts, t0
        )
        if short_circuit is not None:
            if not short_circuit.passed:
                return self._handle_failure_snapshot(candidate, short_circuit, _entries=entries, **kwargs)
            base_v = short_circuit
        else:
            rank_input: Any = {k: cand for k, cand in survivors}
            rank_v = self.rank_gate.evaluate(rank_input, **self._build_rank_kwargs(kwargs))
            base_v = self._merge_rank_verdict(
                entries, survivors, rejected_keys, rank_v, pre_traces, pre_tokens, t0
            )
            if not base_v.passed:
                return self._handle_failure_snapshot(candidate, base_v, _entries=entries, **kwargs)

        if base_v.choice is None:
            raise RuntimeError("Evaluation passed but no choice was selected in BestOfNGate.")
        winner_idx = next(idx for idx, (k, _) in enumerate(entries) if k == base_v.choice)
        winner_cand = entries[winner_idx][1]

        if self.post_gate is not None:
            post_v = self.post_gate.evaluate(
                winner_cand,
                save_on_fail=False,
                _suppress_telemetry=True,
                _bypass_cache=has_top_cache,
            )
            final_v = self._merge_post_verdict(entries, base_v.choice, winner_idx, base_v, post_v, kwargs, t0)
            return self._handle_failure_snapshot(candidate, final_v, _entries=entries, **kwargs)

        norm_out = kwargs.get("_normalized_out")
        if isinstance(norm_out, list):
            norm_out.append(winner_cand)
        return self._handle_failure_snapshot(candidate, base_v, _entries=entries, **kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        entries = _normalize_candidates(candidate)
        await self._aensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._handle_failure_snapshot(
                candidate, hit, _is_cache_hit=True, **kwargs
            )
        has_top_cache = self._has_active_top_cache(kwargs)
        t0 = time.perf_counter()
        pre_verdicts = (
            list(
                await asyncio.gather(
                    *[
                        self.pre_gate.aevaluate(
                            cand,
                            save_on_fail=False,
                            _suppress_telemetry=True,
                            _bypass_cache=has_top_cache,
                        )
                        for _, cand in entries
                    ]
                )
            )
            if self.pre_gate is not None
            else None
        )
        survivors, short_circuit, pre_traces, pre_tokens, rejected_keys = self._assemble_pre_stage(
            entries, pre_verdicts, t0
        )
        if short_circuit is not None:
            if not short_circuit.passed:
                return self._handle_failure_snapshot(candidate, short_circuit, _entries=entries, **kwargs)
            base_v = short_circuit
        else:
            rank_input: Any = {k: cand for k, cand in survivors}
            rank_v = await self.rank_gate.aevaluate(rank_input, **self._build_rank_kwargs(kwargs))
            base_v = self._merge_rank_verdict(
                entries, survivors, rejected_keys, rank_v, pre_traces, pre_tokens, t0
            )
            if not base_v.passed:
                return self._handle_failure_snapshot(candidate, base_v, _entries=entries, **kwargs)

        if base_v.choice is None:
            raise RuntimeError("Evaluation passed but no choice was selected in BestOfNGate.")
        winner_idx = next(idx for idx, (k, _) in enumerate(entries) if k == base_v.choice)
        winner_cand = entries[winner_idx][1]

        if self.post_gate is not None:
            post_v = await self.post_gate.aevaluate(
                winner_cand,
                save_on_fail=False,
                _suppress_telemetry=True,
                _bypass_cache=has_top_cache,
            )
            final_v = self._merge_post_verdict(entries, base_v.choice, winner_idx, base_v, post_v, kwargs, t0)
            return self._handle_failure_snapshot(candidate, final_v, _entries=entries, **kwargs)

        norm_out = kwargs.get("_normalized_out")
        if isinstance(norm_out, list):
            norm_out.append(winner_cand)
        return self._handle_failure_snapshot(candidate, base_v, _entries=entries, **kwargs)

    def select(self, candidates: Any, **kwargs: Any) -> BestOfNResult:
        """Evaluates candidates through pre-filter, ranking, and post-check, returning BestOfNResult."""
        entries = _normalize_candidates(candidates)
        verdict = self.evaluate(candidates, **kwargs)
        return RankGate._to_result(entries, verdict)

    async def aselect(self, candidates: Any, **kwargs: Any) -> BestOfNResult:
        """Async variant of select."""
        entries = _normalize_candidates(candidates)
        verdict = await self.aevaluate(candidates, **kwargs)
        return RankGate._to_result(entries, verdict)


__all__ = ["BestOfNGate"]
