from __future__ import annotations

import asyncio
import math
import secrets
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logitgate.backends.base import LogitBackend, resolve_logit_backend
from logitgate.config import get_default_model
from logitgate.debiasing.inversion import (
    compile_listwise_rank_prompt,
    compile_pairwise_prompt,
)
from logitgate.debiasing.permutation import compute_shannon_entropy
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    extract_token_logprob,
    has_any_target_token,
)
from logitgate.debiasing.softmax import (
    softmax_probabilities as _softmax,
)
from logitgate.exceptions import BackendError, ConfigurationError
from logitgate.gates.base import BaseGate
from logitgate.gates.logit_gate import (
    TopLogprobs,
    _ensure_logprobs_present,
    _lazy_calibrate_async,
    _lazy_calibrate_sync,
    _resolve_gate_temperature,
    _resolve_warm_prefix,
    _validate_optional_bool,
    _validate_optional_unit_float,
    _warn_if_profile_model_mismatch,
)
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import persist_failure_snapshot

if TYPE_CHECKING:
    from logitgate.calibration.profile import CalibrationProfile

RankMode = Literal["auto", "pairwise", "listwise", "round_robin", "tournament"]
PermutationMode = Literal["derangement", "cyclic", "reverse"]

_VALID_RANK_MODES = frozenset({"auto", "pairwise", "listwise", "round_robin", "tournament"})
_VALID_PERM_MODES = frozenset({"derangement", "cyclic", "reverse"})


@dataclass(frozen=True, slots=True)
class BestOfNResult:
    """Structured result returned by `RankGate.rank` and `BestOfNGate.select`."""

    winner: Any | None
    winner_key: str | None
    winner_index: int | None
    ranking: tuple[str, ...]
    choice_probabilities: dict[str, float]
    verdict: Verdict
    surviving_keys: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.verdict.passed


def _validate_optional_positive_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a positive float, got {value!r}")
    vf = float(value)
    if not math.isfinite(vf) or vf <= 0.0:
        raise ConfigurationError(f"{name} must be a positive float, got {value!r}")
    return vf


def _normalize_context(context: Any) -> str | None:
    if context is None:
        return None
    if not isinstance(context, str):
        raise ConfigurationError(f"context must be a string or None, got {type(context).__name__}")
    stripped = context.strip()
    return stripped or None


def _normalize_candidates(candidates: Any) -> list[tuple[str, Any]]:
    """Normalizes a Sequence or Mapping of >= 2 candidates into ``[(key, value), ...]``."""
    if isinstance(candidates, (str, bytes, bytearray)):
        raise ConfigurationError(
            "candidates must be a sequence or mapping of >= 2 candidate artifacts, not a single string/bytes."
        )
    if isinstance(candidates, Mapping):
        if len(candidates) < 2:
            raise ConfigurationError(f"Ranking requires at least 2 candidates, got {len(candidates)}")
        out: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for raw_key, value in candidates.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ConfigurationError(
                    f"Mapping candidate keys must be non-empty strings, got {raw_key!r}"
                )
            key = raw_key.strip()
            if key in seen:
                raise ConfigurationError(
                    f"Mapping candidate keys must be unique after stripping whitespace, got duplicate {key!r}"
                )
            seen.add(key)
            out.append((key, value))
        return out
    if isinstance(candidates, Sequence):
        if len(candidates) < 2:
            raise ConfigurationError(f"Ranking requires at least 2 candidates, got {len(candidates)}")
        return [(str(i), cand) for i, cand in enumerate(candidates)]
    raise ConfigurationError(
        f"candidates must be a Sequence or Mapping of >= 2 items, got {type(candidates).__name__}"
    )


def _score_pair_from_logprobs(
    top_fwd: TopLogprobs,
    top_rev: TopLogprobs | None,
) -> tuple[float, float, bool, float, float | None, float, float | None]:
    """Computes symmetric debiased log-odds ``log(P(i > j) / P(j > i))`` for one pair."""
    _ensure_logprobs_present(top_fwd, top_rev)
    fwd_has = has_any_target_token(top_fwd, ("A", "B"))
    rev_has = True if top_rev is None else has_any_target_token(top_rev, ("A", "B"))
    delta_1, p_fwd = compute_binary_log_odds(top_fwd, pass_token="A", fail_token="B")
    if top_rev is not None:
        delta_2, p_rev = compute_binary_log_odds(top_rev, pass_token="B", fail_token="A")
        raw_lo = 0.5 * (delta_1 + delta_2)
        debiased_lo = max(-100.0, min(100.0, raw_lo))
        raw_bias = 0.5 * abs(delta_1 - delta_2)
        pos_bias = min(100.0, raw_bias)
        return debiased_lo, pos_bias, (fwd_has and rev_has), delta_1, delta_2, p_fwd, p_rev
    delta_clamped = max(-100.0, min(100.0, delta_1))
    return delta_clamped, 0.0, fwd_has, delta_1, None, p_fwd, None


class RankGate(BaseGate):
    """Zero-token preference ranking gate with symmetric positional swap debiasing.

    Compares ``N >= 2`` candidate artifacts against an evaluation rubric using first-token
    log-probabilities (`max_tokens=1`) across four scalable regimes:

    - ``pairwise`` (N = 2): symmetric A/B swap cancellation.
    - ``listwise`` (2 <= N <= 10): single-token slot softmax with derangement/cyclic/reverse
      permutations.
    - ``round_robin`` (2 <= N <= 16): all N*(N-1)/2 pairwise comparisons with Hodge / Bradley-Terry
      centered log-worths and cycle-inconsistency residual diagnostics.
    - ``tournament`` (2 <= N <= 32): single-elimination bracket with swapped matches, odd-N byes,
      and champion-rooted tree log-worth estimation.
    - ``auto``: selects ``pairwise`` for N=2, ``listwise`` for 3<=N<=6, and ``tournament`` for 7<=N<=32.
    """

    def __init__(
        self,
        rubric: str,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
        *,
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
        name: str = "RankGate",
    ) -> None:
        from logitgate.calibration.profile import resolve_calibration_profile
        from logitgate.telemetry.sinks import resolve_telemetry_sink

        loaded_profile = resolve_calibration_profile(profile)
        resolved_sink = resolve_telemetry_sink(telemetry_sink)
        if not isinstance(rubric, str) or not rubric.strip():
            raise ConfigurationError("rubric cannot be empty or whitespace")
        try:
            mode_ok = mode in _VALID_RANK_MODES
        except TypeError:
            mode_ok = False
        if not mode_ok:
            raise ConfigurationError(
                f"Invalid mode {mode!r}. Must be one of {sorted(_VALID_RANK_MODES)}"
            )
        try:
            perm_ok = permutation_mode in _VALID_PERM_MODES
        except TypeError:
            perm_ok = False
        if not perm_ok:
            raise ConfigurationError(
                f"Invalid permutation_mode {permutation_mode!r}. Must be one of {sorted(_VALID_PERM_MODES)}"
            )
        norm_ctx = _normalize_context(context)
        if symmetric_debiasing is not None and not isinstance(symmetric_debiasing, bool):
            raise ConfigurationError(
                f"symmetric_debiasing must be a bool or None, got {symmetric_debiasing!r}"
            )
        if not isinstance(query_first, bool):
            raise ConfigurationError(f"query_first must be a bool, got {query_first!r}")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("name must be a non-empty string")

        raw_min_conf = (
            min_confidence
            if min_confidence is not None
            else (loaded_profile.choice_min_confidence if loaded_profile is not None else None)
        )
        raw_min_margin = (
            min_margin
            if min_margin is not None
            else (loaded_profile.choice_min_margin if loaded_profile is not None else None)
        )
        effective_min_conf = _validate_optional_unit_float("min_confidence", raw_min_conf)
        effective_min_margin = _validate_optional_unit_float("min_margin", raw_min_margin)
        effective_max_bias = _validate_optional_positive_float("max_positional_bias", max_positional_bias)
        effective_save_on_fail = _validate_optional_bool("save_on_fail", save_on_fail)
        effective_warm_prefix = _validate_optional_bool("warm_prefix", warm_prefix)

        owns_backend = not isinstance(backend, LogitBackend)
        resolved_backend = resolve_logit_backend(backend, model=model)
        try:
            effective_model = (
                model
                or getattr(resolved_backend, "model", None)
                or getattr(resolved_backend, "model_name", None)
                or get_default_model()
            )
            effective_temp, needs_lazy = _resolve_gate_temperature(
                logit_temperature,
                effective_model,
                resolved_backend,
                loaded_profile,
                binary=False,
            )
            from logitgate.cache import resolve_verdict_cache

            resolved_cache = resolve_verdict_cache(cache)
        except BaseException:
            if owns_backend:
                resolved_backend.close()
            raise

        _warn_if_profile_model_mismatch(loaded_profile, effective_model, name)
        self.backend = resolved_backend
        self.model = effective_model
        self.profile = loaded_profile
        self.rubric = rubric.strip()
        self.mode: RankMode = mode
        self.context = norm_ctx
        self.symmetric_debiasing = True if symmetric_debiasing is None else symmetric_debiasing
        self.permutation_mode: PermutationMode = permutation_mode
        self.query_first = query_first
        self.logit_temperature = effective_temp
        self._needs_lazy_calibration = needs_lazy
        self._calibration_lock = threading.Lock()
        self.min_confidence = effective_min_conf
        self.min_margin = effective_min_margin
        self.max_positional_bias = effective_max_bias
        self.save_on_fail = effective_save_on_fail
        self.warm_prefix = effective_warm_prefix
        self.telemetry_sink = resolved_sink
        self.cache = resolved_cache
        self.name = name

    def _ensure_calibrated(self) -> None:
        if not self._needs_lazy_calibration or not self.model:
            return
        with self._calibration_lock:
            if not self._needs_lazy_calibration:
                return
            self.logit_temperature = _lazy_calibrate_sync(self.model, self.backend, binary=False)
            self._needs_lazy_calibration = False

    async def _aensure_calibrated(self) -> None:
        if not self._needs_lazy_calibration or not self.model:
            return
        temp = await _lazy_calibrate_async(self.model, self.backend, binary=False)
        with self._calibration_lock:
            self.logit_temperature = temp
            self._needs_lazy_calibration = False

    def _resolve_effective_mode(self, n: int) -> Literal["pairwise", "listwise", "round_robin", "tournament"]:
        if self.mode == "auto":
            if n == 2:
                return "pairwise"
            if 3 <= n <= 6:
                return "listwise"
            if n <= 32:
                return "tournament"
            raise ConfigurationError(f"RankGate supports at most 32 candidates in 'auto' mode, got {n}")
        if self.mode == "pairwise":
            if n != 2:
                raise ConfigurationError(f"mode='pairwise' requires exactly 2 candidates, got {n}")
            return "pairwise"
        if self.mode == "listwise":
            if not (2 <= n <= 10):
                raise ConfigurationError(f"mode='listwise' supports 2..10 candidates, got {n}")
            return "listwise"
        if self.mode == "round_robin":
            if not (2 <= n <= 16):
                raise ConfigurationError(f"mode='round_robin' supports 2..16 candidates, got {n}")
            return "round_robin"
        if not (2 <= n <= 32):
            raise ConfigurationError(f"mode='tournament' supports 2..32 candidates, got {n}")
        return "tournament"

    def _permuted_key_indices(self, n: int) -> list[list[int]]:
        identity = list(range(n))
        if not self.symmetric_debiasing:
            return [identity]
        if self.permutation_mode == "cyclic":
            return [[(i + s) % n for i in range(n)] for s in range(n)]
        if self.permutation_mode == "reverse":
            return [identity, [n - 1 - i for i in range(n)]]
        shift = max(1, n // 2)
        return [identity, [(i + shift) % n for i in range(n)]]

    def _build_pair_prompts(
        self,
        cand_i: Any,
        cand_j: Any,
        *,
        effective_context: str | None,
        nonce: str,
    ) -> list[str]:
        p1 = compile_pairwise_prompt(
            self.rubric,
            cand_i,
            cand_j,
            context=effective_context,
            token_a="A",
            token_b="B",
            query_first=self.query_first,
            boundary_nonce=nonce,
        )
        if not self.symmetric_debiasing:
            return [p1]
        p2 = compile_pairwise_prompt(
            self.rubric,
            cand_j,
            cand_i,
            context=effective_context,
            token_a="A",
            token_b="B",
            query_first=self.query_first,
            boundary_nonce=nonce,
        )
        return [p1, p2]

    def _build_listwise_prompts(
        self,
        entries: Sequence[tuple[str, Any]],
        *,
        effective_context: str | None,
        nonce: str,
    ) -> tuple[list[str], list[str], list[list[int]]]:
        n = len(entries)
        slot_keys = [chr(ord("A") + i) for i in range(n)]
        perms = self._permuted_key_indices(n)
        prompts: list[str] = []
        for perm in perms:
            slotted = {slot_keys[slot_idx]: entries[perm.index(slot_idx)][1] for slot_idx in range(n)}
            prompts.append(
                compile_listwise_rank_prompt(
                    self.rubric,
                    slotted,
                    context=effective_context,
                    query_first=self.query_first,
                    boundary_nonce=nonce,
                )
            )
        return slot_keys, prompts, perms

    async def _aquery_prompts(
        self,
        prompts: Sequence[str],
        *,
        allowed_tokens: list[str],
        top_k: int,
        warm_first: bool,
    ) -> list[TopLogprobs]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [
                await self.backend.aquery_top_logprobs(
                    prompts[0], top_k=top_k, allowed_tokens=allowed_tokens
                )
            ]
        if warm_first:
            first = await self.backend.aquery_top_logprobs(
                prompts[0], top_k=top_k, allowed_tokens=allowed_tokens
            )
            rest = list(
                await asyncio.gather(
                    *[
                        self.backend.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=allowed_tokens)
                        for p in prompts[1:]
                    ]
                )
            )
            return [first, *rest]
        return list(
            await asyncio.gather(
                *[
                    self.backend.aquery_top_logprobs(p, top_k=top_k, allowed_tokens=allowed_tokens)
                    for p in prompts
                ]
            )
        )

    def _finalize_ranking_verdict(
        self,
        entries: list[tuple[str, Any]],
        worths: dict[str, float],
        *,
        effective_mode: str,
        pos_bias: float,
        all_have_targets: bool,
        llm_calls: int,
        t0: float,
        extra_meta: dict[str, Any] | None = None,
        forward_prob: float | None = None,
        reverse_prob: float | None = None,
        forward_log_odds: float | None = None,
        reverse_log_odds: float | None = None,
        call_kwargs: Mapping[str, Any] | None = None,
    ) -> Verdict:
        keys = [k for k, _ in entries]
        probs = _softmax(worths, self.logit_temperature)
        ordered = sorted(
            enumerate(entries),
            key=lambda item: (-worths[item[1][0]], item[0]),
        )
        ranking = [item[1][0] for item in ordered]
        best_idx, (best_key, best_cand) = ordered[0]
        max_prob = probs[best_key]
        prob_rounded = round(max_prob, 4)
        rounded_probs = {k: round(probs[k], 4) for k in keys}
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = round(sorted_probs[0] - sorted_probs[1], 4) if len(sorted_probs) >= 2 else 1.0
        bias_rounded = round(pos_bias, 4)
        entropy = round(compute_shannon_entropy(max_prob), 4)
        latency = (time.perf_counter() - t0) * 1000.0

        abstained_reason: str | None = None
        if not all_have_targets:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice: str | None = None
            abstained_reason = "unobserved_target_tokens"
            reason = f"{self.name} abstained (AMBIGUOUS): unobserved_target_tokens"
        elif self.max_positional_bias is not None and pos_bias > self.max_positional_bias:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice = best_key
            abstained_reason = f"positional_bias_exceeded ({pos_bias:.4f} > {self.max_positional_bias:.4f})"
            reason = f"{self.name} abstained on '{best_key}': {abstained_reason}"
        elif self.min_confidence is not None and prob_rounded < self.min_confidence:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice = best_key
            abstained_reason = f"below_min_confidence ({prob_rounded:.4f} < {self.min_confidence:.4f})"
            reason = f"{self.name} abstained on '{best_key}': {abstained_reason}"
        elif self.min_margin is not None and margin < self.min_margin:
            status = DecisionStatus.AMBIGUOUS
            passed = False
            selected_choice = best_key
            abstained_reason = f"below_min_margin ({margin:.4f} < {self.min_margin:.4f})"
            reason = f"{self.name} abstained on '{best_key}': {abstained_reason}"
        else:
            status = DecisionStatus.PASSED
            passed = True
            selected_choice = best_key
            reason = f"{self.name} ({effective_mode}) selected '{best_key}' with P={max_prob:.4f}"

        meta: dict[str, Any] = {
            "effective_mode": effective_mode,
            "ranking": ranking,
            "choice_probabilities": rounded_probs,
            "top2_margin": margin,
            "positional_bias_magnitude": bias_rounded,
            "llm_calls": llm_calls,
            "winner_index": None if abstained_reason == "unobserved_target_tokens" else best_idx,
        }
        if extra_meta:
            meta.update(extra_meta)
        if abstained_reason is not None:
            meta["abstained_reason"] = abstained_reason

        if passed and call_kwargs is not None:
            norm_out = call_kwargs.get("_normalized_out")
            if isinstance(norm_out, list):
                norm_out.append(best_cand)

        trace = LayerTrace(
            layer_name=self.name,
            status=status,
            latency_ms=latency,
            calibrated_prob=prob_rounded,
            forward_prob=round(forward_prob, 4) if forward_prob is not None else None,
            reverse_prob=round(reverse_prob, 4) if reverse_prob is not None else None,
            forward_log_odds=round(forward_log_odds, 4) if forward_log_odds is not None else None,
            reverse_log_odds=round(reverse_log_odds, 4) if reverse_log_odds is not None else None,
            positional_bias_magnitude=bias_rounded,
            shannon_entropy=entropy,
            tokens_generated=0,
            metadata=meta,
        )
        return Verdict(
            passed=passed,
            status=status,
            probability=prob_rounded,
            choice=selected_choice,
            choice_probabilities=rounded_probs,
            triggered_layer=self.name,
            total_latency_ms=latency,
            total_tokens_generated=0,
            reason=reason,
            traces=[trace],
        )

    def _compute_pairwise_verdict(
        self,
        entries: list[tuple[str, Any]],
        top_list: Sequence[TopLogprobs],
        t0: float,
        call_kwargs: Mapping[str, Any],
    ) -> Verdict:
        top_fwd = list(top_list[0])
        top_rev = list(top_list[1]) if len(top_list) > 1 else None
        debiased_lo, pos_bias, has_t, d1, d2, p1, p2 = _score_pair_from_logprobs(top_fwd, top_rev)
        worths = {entries[0][0]: 0.5 * debiased_lo, entries[1][0]: -0.5 * debiased_lo}
        return self._finalize_ranking_verdict(
            entries,
            worths,
            effective_mode="pairwise",
            pos_bias=pos_bias,
            all_have_targets=has_t,
            llm_calls=len(top_list),
            t0=t0,
            forward_prob=p1,
            reverse_prob=p2,
            forward_log_odds=d1,
            reverse_log_odds=d2,
            call_kwargs=call_kwargs,
        )

    def _compute_round_robin_verdict(
        self,
        entries: list[tuple[str, Any]],
        pairs: Sequence[tuple[int, int]],
        top_list: Sequence[TopLogprobs],
        t0: float,
        call_kwargs: Mapping[str, Any],
    ) -> Verdict:
        n = len(entries)
        step = 2 if self.symmetric_debiasing else 1
        z_mat: list[list[float]] = [[0.0] * n for _ in range(n)]
        pair_diffs: list[tuple[tuple[int, int], float]] = []
        pair_biases: list[float] = []
        all_have_targets = True

        for idx, (i, j) in enumerate(pairs):
            top_fwd = list(top_list[idx * step])
            top_rev = list(top_list[idx * step + 1]) if step == 2 else None
            z_ij, bias_ij, has_t, _, _, _, _ = _score_pair_from_logprobs(top_fwd, top_rev)
            z_mat[i][j] = z_ij
            z_mat[j][i] = -z_ij
            pair_diffs.append(((i, j), z_ij))
            pair_biases.append(bias_ij)
            all_have_targets = all_have_targets and has_t

        w = [sum(z_mat[i][j] for j in range(n) if j != i) / float(n) for i in range(n)]
        if n >= 3 and pair_diffs:
            sq_errs = [(z_ij - (w[i] - w[j])) ** 2 for (i, j), z_ij in pair_diffs]
            cycle_res = math.sqrt(sum(sq_errs) / float(len(sq_errs)))
        else:
            cycle_res = 0.0

        worths = {entries[i][0]: w[i] for i in range(n)}
        pos_bias = max(pair_biases) if pair_biases else 0.0
        return self._finalize_ranking_verdict(
            entries,
            worths,
            effective_mode="round_robin",
            pos_bias=pos_bias,
            all_have_targets=all_have_targets,
            llm_calls=len(top_list),
            t0=t0,
            extra_meta={"cycle_residual": round(cycle_res, 4), "pairs_evaluated": len(pairs)},
            call_kwargs=call_kwargs,
        )

    def _compute_listwise_verdict(
        self,
        entries: list[tuple[str, Any]],
        slot_keys: list[str],
        perms: Sequence[Sequence[int]],
        top_list: Sequence[TopLogprobs],
        t0: float,
        call_kwargs: Mapping[str, Any],
    ) -> Verdict:
        if not top_list:
            raise BackendError("Backend returned no token log-probabilities (top_logprobs is empty).")
        keys = [k for k, _ in entries]
        n = len(entries)
        accum = {k: 0.0 for k in keys}
        per_pass_raw: list[dict[str, float]] = []
        per_pass_norm: list[dict[str, float]] = []
        all_have_targets = True

        for top, perm in zip(top_list, perms):
            top_l = list(top)
            _ensure_logprobs_present(top_l)
            all_have_targets = all_have_targets and has_any_target_token(top_l, slot_keys)
            pass_raw = {keys[i]: extract_token_logprob(top_l, slot_keys[perm[i]]) for i in range(n)}
            m_pass = max(pass_raw.values())
            lse_pass = m_pass + math.log(sum(math.exp(v - m_pass) for v in pass_raw.values()))
            pass_norm = {k: pass_raw[k] - lse_pass for k in keys}
            per_pass_raw.append(pass_raw)
            per_pass_norm.append(pass_norm)
            for k in keys:
                accum[k] += pass_norm[k] / float(len(perms))

        if len(perms) >= 2:
            if n == 2 and len(perms) == 2:
                diff_0 = per_pass_raw[0][keys[0]] - per_pass_raw[0][keys[1]]
                diff_1 = per_pass_raw[1][keys[0]] - per_pass_raw[1][keys[1]]
                pos_bias = min(100.0, 0.5 * abs(diff_0 - diff_1))
            else:
                pos_bias = min(
                    100.0,
                    0.5 * max((max(p[k] for p in per_pass_norm) - min(p[k] for p in per_pass_norm)) for k in keys),
                )
        else:
            pos_bias = 0.0

        return self._finalize_ranking_verdict(
            entries,
            accum,
            effective_mode="listwise",
            pos_bias=pos_bias,
            all_have_targets=all_have_targets,
            llm_calls=len(top_list),
            t0=t0,
            extra_meta={"permutation_mode": self.permutation_mode},
            call_kwargs=call_kwargs,
        )

    def _compute_tournament_verdict(
        self,
        entries: list[tuple[str, Any]],
        champion_idx: int,
        tree_edges: Sequence[tuple[int, int, float]],
        match_biases: Sequence[float],
        all_have_targets: bool,
        num_rounds: int,
        llm_calls: int,
        t0: float,
        call_kwargs: Mapping[str, Any],
    ) -> Verdict:
        n = len(entries)
        raw_w: dict[int, float] = {champion_idx: 0.0}
        for winner_idx, loser_idx, gap in reversed(tree_edges):
            parent_w = raw_w.get(winner_idx, 0.0)
            raw_w[winner_idx] = parent_w
            raw_w[loser_idx] = parent_w - gap
        mean_w = sum(raw_w.get(i, 0.0) for i in range(n)) / float(n)
        worths = {entries[i][0]: raw_w.get(i, 0.0) - mean_w for i in range(n)}
        pos_bias = max(match_biases) if match_biases else 0.0
        return self._finalize_ranking_verdict(
            entries,
            worths,
            effective_mode="tournament",
            pos_bias=pos_bias,
            all_have_targets=all_have_targets,
            llm_calls=llm_calls,
            t0=t0,
            extra_meta={"rounds": num_rounds, "matches_played": len(tree_edges)},
            call_kwargs=call_kwargs,
        )

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
            gate_type="RankGate",
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

    def replay_config(self) -> dict[str, Any]:
        from logitgate.backends.base import infer_backend_spec

        config: dict[str, Any] = {
            "gate_type": "RankGate",
            "rubric": self.rubric,
            "backend": infer_backend_spec(self.backend),
            "mode": self.mode,
            "symmetric_debiasing": self.symmetric_debiasing,
            "permutation_mode": self.permutation_mode,
            "query_first": self.query_first,
            "logit_temperature": self.logit_temperature,
            "name": self.name,
        }
        if self.context is not None:
            config["context"] = self.context
        if self.min_confidence is not None:
            config["min_confidence"] = self.min_confidence
        if self.min_margin is not None:
            config["min_margin"] = self.min_margin
        if self.max_positional_bias is not None:
            config["max_positional_bias"] = self.max_positional_bias
        if self.warm_prefix is not None:
            config["warm_prefix"] = self.warm_prefix
        model = (
            self.model
            or getattr(self.backend, "model", None)
            or getattr(self.backend, "model_name", None)
        )
        if model:
            config["model"] = model
        return config

    def __repr__(self) -> str:
        rubric_repr = repr(self.rubric[:40] + "..." if len(self.rubric) > 40 else self.rubric)
        return f"RankGate(rubric={rubric_repr}, mode={self.mode!r}, backend={self.backend!r})"

    def _effective_context(self, call_kwargs: Mapping[str, Any]) -> str | None:
        if "context" in call_kwargs:
            return _normalize_context(call_kwargs["context"])
        return self.context

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        entries = _normalize_candidates(candidate)
        eff_mode = self._resolve_effective_mode(len(entries))
        eff_ctx = self._effective_context(kwargs)
        self._ensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._handle_failure_snapshot(
                candidate, hit, _is_cache_hit=True, **kwargs
            )
        t0 = time.perf_counter()
        nonce = secrets.token_hex(8)

        if eff_mode == "pairwise":
            prompts = self._build_pair_prompts(entries[0][1], entries[1][1], effective_context=eff_ctx, nonce=nonce)
            top_list = [self.backend.query_top_logprobs(p, top_k=10, allowed_tokens=["A", "B"]) for p in prompts]
            verdict = self._compute_pairwise_verdict(entries, top_list, t0, kwargs)
            return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

        if eff_mode == "round_robin":
            n = len(entries)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rr_prompts: list[str] = []
            for i, j in pairs:
                rr_prompts.extend(
                    self._build_pair_prompts(entries[i][1], entries[j][1], effective_context=eff_ctx, nonce=nonce)
                )
            top_list = [self.backend.query_top_logprobs(p, top_k=10, allowed_tokens=["A", "B"]) for p in rr_prompts]
            verdict = self._compute_round_robin_verdict(entries, pairs, top_list, t0, kwargs)
            return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

        if eff_mode == "listwise":
            slot_keys, prompts, perms = self._build_listwise_prompts(entries, effective_context=eff_ctx, nonce=nonce)
            top_k = max(10, 2 * len(slot_keys))
            top_list = [self.backend.query_top_logprobs(p, top_k=top_k, allowed_tokens=slot_keys) for p in prompts]
            verdict = self._compute_listwise_verdict(entries, slot_keys, perms, top_list, t0, kwargs)
            return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

        # tournament
        active = list(range(len(entries)))
        tree_edges: list[tuple[int, int, float]] = []
        match_biases: list[float] = []
        all_have_targets = True
        num_rounds = 0
        llm_calls = 0
        while len(active) > 1:
            num_rounds += 1
            round_pairs = [(active[k], active[k + 1]) for k in range(0, len(active) - 1, 2)]
            next_round: list[int] = []
            for i, j in round_pairs:
                prompts = self._build_pair_prompts(entries[i][1], entries[j][1], effective_context=eff_ctx, nonce=nonce)
                tops = [self.backend.query_top_logprobs(p, top_k=10, allowed_tokens=["A", "B"]) for p in prompts]
                llm_calls += len(tops)
                top_fwd = tops[0]
                top_rev = tops[1] if len(tops) > 1 else None
                z_ij, bias_ij, has_t, _, _, _, _ = _score_pair_from_logprobs(top_fwd, top_rev)
                match_biases.append(bias_ij)
                all_have_targets = all_have_targets and has_t
                w_idx, l_idx = (i, j) if z_ij >= 0.0 else (j, i)
                tree_edges.append((w_idx, l_idx, abs(z_ij)))
                next_round.append(w_idx)
            if len(active) % 2 == 1:
                next_round.append(active[-1])
            active = next_round
        verdict = self._compute_tournament_verdict(
            entries,
            active[0],
            tree_edges,
            match_biases,
            all_have_targets,
            num_rounds,
            llm_calls,
            t0,
            kwargs,
        )
        return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        from logitgate.cache import lookup_cached_verdict

        entries = _normalize_candidates(candidate)
        eff_mode = self._resolve_effective_mode(len(entries))
        eff_ctx = self._effective_context(kwargs)
        await self._aensure_calibrated()
        hit = lookup_cached_verdict(
            self, candidate, gate_cache=self.cache, call_kwargs=kwargs
        )
        if hit is not None:
            return self._handle_failure_snapshot(
                candidate, hit, _is_cache_hit=True, **kwargs
            )
        t0 = time.perf_counter()
        nonce = secrets.token_hex(8)
        warm = _resolve_warm_prefix(self.warm_prefix, self.backend)

        if eff_mode == "pairwise":
            prompts = self._build_pair_prompts(entries[0][1], entries[1][1], effective_context=eff_ctx, nonce=nonce)
            top_list = await self._aquery_prompts(
                prompts, allowed_tokens=["A", "B"], top_k=10, warm_first=warm
            )
            verdict = self._compute_pairwise_verdict(entries, top_list, t0, kwargs)
            return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

        if eff_mode == "round_robin":
            n = len(entries)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rr_prompts: list[str] = []
            for i, j in pairs:
                rr_prompts.extend(
                    self._build_pair_prompts(entries[i][1], entries[j][1], effective_context=eff_ctx, nonce=nonce)
                )
            top_list = await self._aquery_prompts(
                rr_prompts, allowed_tokens=["A", "B"], top_k=10, warm_first=warm
            )
            verdict = self._compute_round_robin_verdict(entries, pairs, top_list, t0, kwargs)
            return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

        if eff_mode == "listwise":
            slot_keys, prompts, perms = self._build_listwise_prompts(entries, effective_context=eff_ctx, nonce=nonce)
            top_k = max(10, 2 * len(slot_keys))
            top_list = await self._aquery_prompts(
                prompts, allowed_tokens=slot_keys, top_k=top_k, warm_first=warm
            )
            verdict = self._compute_listwise_verdict(entries, slot_keys, perms, top_list, t0, kwargs)
            return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

        # tournament
        active = list(range(len(entries)))
        tree_edges: list[tuple[int, int, float]] = []
        match_biases: list[float] = []
        all_have_targets = True
        num_rounds = 0
        llm_calls = 0
        step = 2 if self.symmetric_debiasing else 1
        warm_pending = warm
        while len(active) > 1:
            num_rounds += 1
            round_pairs = [(active[k], active[k + 1]) for k in range(0, len(active) - 1, 2)]
            round_prompts: list[str] = []
            for i, j in round_pairs:
                round_prompts.extend(
                    self._build_pair_prompts(entries[i][1], entries[j][1], effective_context=eff_ctx, nonce=nonce)
                )
            round_tops = await self._aquery_prompts(
                round_prompts, allowed_tokens=["A", "B"], top_k=10, warm_first=warm_pending
            )
            warm_pending = False
            llm_calls += len(round_tops)
            next_round: list[int] = []
            for idx, (i, j) in enumerate(round_pairs):
                top_fwd = round_tops[idx * step]
                top_rev = round_tops[idx * step + 1] if step == 2 else None
                z_ij, bias_ij, has_t, _, _, _, _ = _score_pair_from_logprobs(top_fwd, top_rev)
                match_biases.append(bias_ij)
                all_have_targets = all_have_targets and has_t
                w_idx, l_idx = (i, j) if z_ij >= 0.0 else (j, i)
                tree_edges.append((w_idx, l_idx, abs(z_ij)))
                next_round.append(w_idx)
            if len(active) % 2 == 1:
                next_round.append(active[-1])
            active = next_round
        verdict = self._compute_tournament_verdict(
            entries,
            active[0],
            tree_edges,
            match_biases,
            all_have_targets,
            num_rounds,
            llm_calls,
            t0,
            kwargs,
        )
        return self._handle_failure_snapshot(candidate, verdict, _entries=entries, **kwargs)

    @staticmethod
    def _to_result(entries: list[tuple[str, Any]], verdict: Verdict) -> BestOfNResult:
        ranking_meta: dict[str, Any] = verdict.traces[-1].metadata if verdict.traces else {}
        for tr in reversed(verdict.traces):
            if "ranking" in tr.metadata or "surviving_keys" in tr.metadata:
                ranking_meta = tr.metadata
                break
        ranking_raw = ranking_meta.get("ranking")
        ranking = (
            tuple(str(k) for k in ranking_raw)
            if isinstance(ranking_raw, Sequence) and not isinstance(ranking_raw, (str, bytes))
            else tuple(k for k, _ in entries)
        )
        probs = dict(verdict.choice_probabilities or {k: 0.0 for k, _ in entries})
        surv_raw = ranking_meta.get("surviving_keys")
        surviving = (
            tuple(str(k) for k in surv_raw)
            if isinstance(surv_raw, Sequence) and not isinstance(surv_raw, (str, bytes))
            else tuple(k for k, _ in entries)
        )
        if not verdict.passed or verdict.choice is None:
            return BestOfNResult(
                winner=None,
                winner_key=None,
                winner_index=None,
                ranking=ranking,
                choice_probabilities=probs,
                verdict=verdict,
                surviving_keys=surviving,
            )
        key_to_idx_cand = {k: (idx, cand) for idx, (k, cand) in enumerate(entries)}
        w_key = verdict.choice
        w_idx, w_cand = key_to_idx_cand[w_key]
        return BestOfNResult(
            winner=w_cand,
            winner_key=w_key,
            winner_index=w_idx,
            ranking=ranking,
            choice_probabilities=probs,
            verdict=verdict,
            surviving_keys=surviving,
        )

    def rank(self, candidates: Any, **kwargs: Any) -> BestOfNResult:
        """Evaluates ``candidates`` and returns a structured `BestOfNResult`."""
        entries = _normalize_candidates(candidates)
        verdict = self.evaluate(candidates, **kwargs)
        return self._to_result(entries, verdict)

    async def arank(self, candidates: Any, **kwargs: Any) -> BestOfNResult:
        """Async variant of `rank`."""
        entries = _normalize_candidates(candidates)
        verdict = await self.aevaluate(candidates, **kwargs)
        return self._to_result(entries, verdict)


def _tag_traces_with_key(traces: Sequence[LayerTrace], key: str) -> list[LayerTrace]:
    return [
        t.model_copy(update={"metadata": {**t.metadata, "candidate_key": key}})
        for t in traces
    ]


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
        for orig_idx, ((k, cand), pv) in enumerate(zip(entries, pre_verdicts)):
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

        assert base_v.choice is not None
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

        assert base_v.choice is not None
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
        """Evaluates ``candidates`` through pre-filter, ranking, and post-check, returning `BestOfNResult`."""
        entries = _normalize_candidates(candidates)
        verdict = self.evaluate(candidates, **kwargs)
        return RankGate._to_result(entries, verdict)

    async def aselect(self, candidates: Any, **kwargs: Any) -> BestOfNResult:
        """Async variant of `select`."""
        entries = _normalize_candidates(candidates)
        verdict = await self.aevaluate(candidates, **kwargs)
        return RankGate._to_result(entries, verdict)
