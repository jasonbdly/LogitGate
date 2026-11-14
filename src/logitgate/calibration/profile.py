from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import stat
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from logitgate.backends.base import (
    LogitBackend,
    infer_backend_spec,
    resolve_logit_backend,
)
from logitgate.calibration.conformal import (
    ConformalCascadeThresholds,
    _ceil4,
    _floor4,
    _strictly_above4,
    _strictly_below4,
    calibrate_cascade_thresholds,
    load_calibration_dataset,
    min_samples_per_class,
)
from logitgate.debiasing.inversion import compile_binary_prompt
from logitgate.debiasing.permutation import (
    UNIVERSAL_MULTICLASS_CALIBRATION_DATASET,
    CalibrationResult,
    _debiased_multiclass_accum,
    _fit_multiclass_temperature,
    _multiclass_probe_prompts,
    _persist_calibration_profile,
    aauto_calibrate_backend,
    auto_calibrate_backend,
    combine_symmetric_log_odds,
    fit_calibration,
    get_model_temperature,
    has_known_or_cached_temperature,
)
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    has_any_target_token,
    has_finite_logprobs,
)
from logitgate.exceptions import (
    BackendError,
    CalibrationError,
    CalibrationWarning,
    ConfigurationError,
)

_CURRENT_SCHEMA_VERSION = 1
_MAX_PROFILE_BYTES = 1 * 1024 * 1024

_INT_PROFILE_FIELDS = frozenset({
    "schema_version",
    "calibration_set_size",
    "min_samples_per_class",
    "num_positive",
    "num_negative",
})

_FLOAT_PROFILE_FIELDS = frozenset({
    "binary_temperature",
    "multiclass_temperature",
    "pass_threshold",
    "fail_threshold",
    "score_threshold",
    "score_fail_threshold",
    "choice_min_confidence",
    "choice_min_margin",
    "target_risk_alpha",
    "empirical_escalation_rate",
    "probe_auroc",
    "median_abs_log_odds",
    "brier_score_before",
    "brier_score_after",
    "ece_before",
    "ece_after",
})


def _validate_alpha(target_risk_alpha: float) -> float:
    if (
        isinstance(target_risk_alpha, bool)
        or not isinstance(target_risk_alpha, (int, float))
        or not math.isfinite(float(target_risk_alpha))
        or not (0.0 < float(target_risk_alpha) < 1.0)
    ):
        raise CalibrationError(
            f"target_risk_alpha must be strictly between 0.0 and 1.0, got {target_risk_alpha!r}"
        )
    return float(target_risk_alpha)


@dataclass(frozen=True, slots=True)
class ChoiceConformalThresholds:
    """Selective split-conformal abstention thresholds for `ChoiceGate`.

    When ``guarantee_holds`` is ``True``, gating ``ChoiceGate`` acceptance on
    ``P(top_choice) >= min_confidence`` (and ``margin >= min_margin`` if fitted)
    controls the marginal false-routing probability at ``<= target_risk_alpha``
    on exchangeable inputs.
    """

    min_confidence: float
    min_margin: float | None
    target_risk_alpha: float
    empirical_abstention_rate: float
    empirical_accuracy: float
    calibration_set_size: int
    guarantee_holds: bool = True
    min_samples_required: int = 0
    num_errors: int = 0


def calibrate_choice_thresholds(
    choice_probs_list: Sequence[Mapping[str, float]],
    ground_truth_choices: Sequence[str],
    target_risk_alpha: float = 0.05,
    *,
    calibrate_margin: bool = False,
) -> ChoiceConformalThresholds:
    """Fits split-conformal selective confidence (and optional margin) thresholds for `ChoiceGate`.

    Each calibration item receives the false-routing nonconformity score
    ``0.0`` when top-1 matches ``ground_truth_choices[i]`` and ``round(max_prob, 4)``
    otherwise, with a strict one-sided conformal quantile at ``1 - target_risk_alpha``
    and a uniform-tie floor strictly above ``1 / K``.
    """
    if not isinstance(calibrate_margin, bool):
        raise CalibrationError(f"calibrate_margin must be a bool, got {calibrate_margin!r}")
    alpha = _validate_alpha(target_risk_alpha)

    n = len(choice_probs_list)
    if n != len(ground_truth_choices) or n < 10:
        raise CalibrationError("Calibration requires matching sequences of at least 10 samples.")

    extracted: list[tuple[float, float, bool]] = []
    max_k = 2
    for probs, gt in zip(choice_probs_list, ground_truth_choices):
        if not isinstance(probs, Mapping) or len(probs) < 2:
            raise CalibrationError(
                "Each entry in choice_probs_list must be a mapping with at least 2 option probabilities."
            )
        if not isinstance(gt, str) or not gt.strip():
            raise CalibrationError(f"ground_truth_choices must contain non-empty strings, got {gt!r}")

        norm_probs: dict[str, float] = {}
        total_prob = 0.0
        for raw_key, raw_val in probs.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise CalibrationError(f"Option keys must be non-empty strings, got {raw_key!r}")
            norm_key = raw_key.strip().upper()
            if norm_key in norm_probs:
                raise CalibrationError(
                    f"Option keys must be case-insensitively unique, got duplicate for {raw_key!r}"
                )
            if isinstance(raw_val, bool):
                raise CalibrationError(
                    f"Option probabilities must be finite floats in [0.0, 1.0], got {raw_val!r}"
                )
            try:
                pf = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise CalibrationError(
                    f"Option probabilities must be finite floats in [0.0, 1.0], got {raw_val!r}"
                ) from exc
            if not math.isfinite(pf) or not (0.0 <= pf <= 1.0):
                raise CalibrationError(
                    f"Option probabilities must be finite floats in [0.0, 1.0], got {raw_val!r}"
                )
            norm_probs[norm_key] = pf
            total_prob += pf

        if not (0.95 <= total_prob <= 1.05):
            raise CalibrationError(
                f"Option probabilities must sum to ~1.0 (in [0.95, 1.05]), got {total_prob:.6f}"
            )
        gt_norm = gt.strip().upper()
        if gt_norm not in norm_probs:
            raise CalibrationError(
                f"Ground-truth choice {gt!r} is missing from option keys {list(probs.keys())!r}"
            )

        max_k = max(max_k, len(norm_probs))
        best_choice, max_prob = max(norm_probs.items(), key=lambda kv: kv[1])
        prob_rounded = round(max_prob, 4)
        sorted_p = sorted(norm_probs.values(), reverse=True)
        margin = round(sorted_p[0] - sorted_p[1], 4) if len(sorted_p) >= 2 else 1.0
        is_correct = best_choice == gt_norm
        extracted.append((prob_rounded, margin, is_correct))

    required = math.ceil((1.0 - alpha) / alpha - 1e-9)
    rank = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)
    rank_ok = rank <= n

    conf_scores = sorted(0.0 if ok else top1 for top1, _, ok in extracted)
    q_conf = conf_scores[max(min(rank, n), 1) - 1]
    uniform_floor = _strictly_above4(1.0 / float(max_k))
    raw_min_conf = max(_strictly_above4(q_conf), uniform_floor)
    saturated = q_conf >= 1.0 or raw_min_conf > 1.0
    min_conf = min(raw_min_conf, 1.0)

    min_marg: float | None = None
    if calibrate_margin:
        marg_scores = sorted(0.0 if ok else marg for _, marg, ok in extracted)
        q_marg = marg_scores[max(min(rank, n), 1) - 1]
        raw_min_marg = _strictly_above4(q_marg)
        saturated = saturated or q_marg >= 1.0 or raw_min_marg > 1.0
        min_marg = min(raw_min_marg, 1.0)

    guarantee_holds = rank_ok and not saturated
    if not guarantee_holds:
        reasons: list[str] = []
        if not rank_ok:
            reasons.append(f"need >= {required} samples (have {n})")
        if saturated:
            reasons.append("a misclassified sample sits at P=1.0")
        warnings.warn(
            f"ChoiceGate conformal guarantee does not hold at alpha={alpha}: "
            + "; ".join(reasons)
            + ". Thresholds are best-effort only.",
            CalibrationWarning,
            stacklevel=2,
        )

    num_correct = sum(1 for _, _, ok in extracted if ok)
    abstained = sum(
        1
        for top1, marg, _ in extracted
        if top1 < min_conf or (min_marg is not None and marg < min_marg)
    )
    return ChoiceConformalThresholds(
        min_confidence=min_conf,
        min_margin=min_marg,
        target_risk_alpha=alpha,
        empirical_abstention_rate=round(abstained / float(n), 4),
        empirical_accuracy=round(num_correct / float(n), 4),
        calibration_set_size=n,
        guarantee_holds=guarantee_holds,
        min_samples_required=required,
        num_errors=n - num_correct,
    )


@dataclass(frozen=True, slots=True)
class ScoreConformalThresholds:
    """Two-sided split-conformal abstention thresholds for `ScoreGate`.

    Expected scores ``E[S] >= score_threshold`` PASS and ``E[S] <= score_fail_threshold``
    FAIL; scores strictly inside ``(score_fail_threshold, score_threshold)`` return
    ``AMBIGUOUS`` for downstream escalation.
    """

    score_fail_threshold: float
    score_threshold: float
    min_score: int
    max_score: int
    target_risk_alpha: float
    empirical_escalation_rate: float
    calibration_set_size: int
    guarantee_holds: bool = True
    min_samples_per_class: int = 0
    num_positive: int = 0
    num_negative: int = 0


def calibrate_score_thresholds(
    expected_scores: Sequence[float],
    ground_truth_labels: Sequence[bool],
    *,
    min_score: int = 1,
    max_score: int = 5,
    target_risk_alpha: float = 0.05,
) -> ScoreConformalThresholds:
    """Fits two-sided split-conformal pass/fail cutoffs over continuous `ScoreGate` scores."""
    if (
        isinstance(min_score, bool)
        or not isinstance(min_score, int)
        or isinstance(max_score, bool)
        or not isinstance(max_score, int)
        or min_score < 0
        or max_score > 9
        or min_score >= max_score
    ):
        raise CalibrationError(
            f"Score bounds must satisfy 0 <= min_score < max_score <= 9, got min_score={min_score!r}, max_score={max_score!r}"
        )
    alpha = _validate_alpha(target_risk_alpha)

    n = len(expected_scores)
    if n != len(ground_truth_labels) or n < 10:
        raise CalibrationError("Calibration requires matching sequences of at least 10 samples.")

    rounded_scores: list[float] = []
    bool_labels: list[bool] = []
    min_f = float(min_score)
    max_f = float(max_score)
    for raw_s, raw_y in zip(expected_scores, ground_truth_labels):
        if isinstance(raw_s, bool):
            raise CalibrationError(
                f"expected_scores must contain finite floats in [{min_score}, {max_score}], got {raw_s!r}"
            )
        try:
            sf = float(raw_s)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                f"expected_scores must contain finite floats in [{min_score}, {max_score}], got {raw_s!r}"
            ) from exc
        if not math.isfinite(sf) or not (min_f <= sf <= max_f):
            raise CalibrationError(
                f"expected_scores must contain finite floats in [{min_score}, {max_score}], got {raw_s!r}"
            )
        if not isinstance(raw_y, bool) and not (isinstance(raw_y, int) and raw_y in (0, 1)):
            raise CalibrationError(
                f"ground_truth_labels must contain booleans or 0/1 integers, got {raw_y!r}"
            )
        rounded_scores.append(round(sf, 4))
        bool_labels.append(bool(raw_y))

    neg_scores = sorted(s for s, y in zip(rounded_scores, bool_labels) if not y)
    pos_losses = sorted(round(max_f - s, 4) for s, y in zip(rounded_scores, bool_labels) if y)
    if not neg_scores or not pos_losses:
        raise CalibrationError(
            "Calibration set must contain both positive and negative ground truth labels."
        )

    half_alpha = alpha / 2.0
    required = min_samples_per_class(alpha)

    def _conformal_quantile(scores: list[float]) -> tuple[float, bool]:
        m = len(scores)
        rank = math.ceil((m + 1) * (1.0 - half_alpha) - 1e-9)
        return scores[max(min(rank, m), 1) - 1], rank <= m

    q_accept, accept_ok = _conformal_quantile(neg_scores)
    q_reject, reject_ok = _conformal_quantile(pos_losses)
    raw_accept = q_accept
    raw_reject = round(max_f - q_reject, 4)

    tau_accept = _strictly_above4(raw_accept)
    tau_reject = _strictly_below4(raw_reject)
    saturated = (
        raw_accept >= max_f
        or raw_reject <= min_f
        or tau_accept > max_f
        or tau_reject < min_f
    )

    if raw_reject >= raw_accept:
        midpoint = 0.5 * (raw_reject + raw_accept)
        span_pad = 0.05 * float(max_score - min_score)
        tau_reject = min(tau_reject, _floor4(midpoint - span_pad))
        tau_accept = max(tau_accept, _ceil4(midpoint + span_pad))

    tau_accept = min(tau_accept, max_f)
    tau_reject = max(tau_reject, min_f)
    if tau_reject >= tau_accept:
        tau_reject = max(min_f, _strictly_below4(tau_accept))

    guarantee_holds = accept_ok and reject_ok and not saturated
    if not guarantee_holds:
        reasons: list[str] = []
        if not (accept_ok and reject_ok):
            reasons.append(
                f"need >= {required} positives and >= {required} negatives "
                f"(have {len(pos_losses)} / {len(neg_scores)})"
            )
        if saturated:
            reasons.append(f"a calibration score sits at {min_score} or {max_score}")
        warnings.warn(
            f"ScoreGate conformal guarantee does not hold at alpha={alpha}: "
            + "; ".join(reasons)
            + ". Thresholds are best-effort only.",
            CalibrationWarning,
            stacklevel=2,
        )

    escalated = sum(1 for s in rounded_scores if tau_reject < s < tau_accept)
    return ScoreConformalThresholds(
        score_fail_threshold=tau_reject,
        score_threshold=tau_accept,
        min_score=min_score,
        max_score=max_score,
        target_risk_alpha=alpha,
        empirical_escalation_rate=round(escalated / float(n), 4),
        calibration_set_size=n,
        guarantee_holds=guarantee_holds,
        min_samples_per_class=required,
        num_positive=len(pos_losses),
        num_negative=len(neg_scores),
    )


class CalibrationProfile(BaseModel):
    """Portable, self-consistent bundle of temperature scaling and split-conformal cutoffs.

    All thresholds recorded here are calibrated *against the bundled temperature* so gates
    configured with ``profile=...`` always evaluate probabilities on the same scale used to
    derive the conformal bounds.
    """

    model_config = ConfigDict(strict=False, extra="ignore")

    schema_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    model: str | None = None
    backend: str | None = None

    binary_temperature: float = 1.0
    multiclass_temperature: float | None = None

    pass_threshold: float | None = None
    fail_threshold: float | None = None
    score_threshold: float | None = None
    score_fail_threshold: float | None = None
    choice_min_confidence: float | None = None
    choice_min_margin: float | None = None

    target_risk_alpha: float | None = None
    empirical_escalation_rate: float | None = None
    calibration_set_size: int = 0
    guarantee_holds: bool = False
    min_samples_per_class: int = 0
    num_positive: int = 0
    num_negative: int = 0

    probe_auroc: float | None = None
    median_abs_log_odds: float | None = None
    brier_score_before: float | None = None
    brier_score_after: float | None = None
    ece_before: float | None = None
    ece_after: float | None = None

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise CalibrationError(f"Invalid CalibrationProfile: {exc}") from exc

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Any | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> CalibrationProfile:
        try:
            return super().model_validate(
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except ValidationError as exc:
            raise CalibrationError(f"Invalid CalibrationProfile: {exc}") from exc

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases_and_reject_bools(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            return data
        copied: dict[str, Any] = dict(data)
        if copied.get("binary_temperature") is None and "optimal_temperature" in copied:
            copied["binary_temperature"] = copied["optimal_temperature"]
        if copied.get("pass_threshold") is None and "tau_accept" in copied:
            copied["pass_threshold"] = copied["tau_accept"]
        if copied.get("fail_threshold") is None and "tau_reject" in copied:
            copied["fail_threshold"] = copied["tau_reject"]

        for field_name in _INT_PROFILE_FIELDS:
            val = copied.get(field_name)
            if val is not None and isinstance(val, bool):
                raise ValueError(f"{field_name} must be an integer, got bool ({val!r})")
        for field_name in _FLOAT_PROFILE_FIELDS:
            val = copied.get(field_name)
            if val is not None and isinstance(val, bool):
                raise ValueError(f"{field_name} must be a finite float, got bool ({val!r})")
        return copied

    @model_validator(mode="after")
    def _validate_ranges(self) -> CalibrationProfile:
        if self.schema_version < 1:
            raise ValueError(f"schema_version must be >= 1, got {self.schema_version}")
        if self.schema_version > _CURRENT_SCHEMA_VERSION:
            warnings.warn(
                f"CalibrationProfile schema_version={self.schema_version} is newer than "
                f"supported version {_CURRENT_SCHEMA_VERSION}; unknown fields were ignored.",
                CalibrationWarning,
                stacklevel=3,
            )

        if not math.isfinite(self.binary_temperature) or self.binary_temperature <= 0.0:
            raise ValueError(
                f"binary_temperature must be a finite float > 0.0, got {self.binary_temperature!r}"
            )
        if self.multiclass_temperature is not None and (
            not math.isfinite(self.multiclass_temperature) or self.multiclass_temperature <= 0.0
        ):
            raise ValueError(
                f"multiclass_temperature must be a finite float > 0.0, got {self.multiclass_temperature!r}"
            )

        for name in ("pass_threshold", "fail_threshold", "empirical_escalation_rate", "probe_auroc"):
            val = getattr(self, name)
            if val is not None and (not math.isfinite(val) or not (0.0 <= val <= 1.0)):
                raise ValueError(f"{name} must be finite and in [0.0, 1.0], got {val!r}")
        if (
            self.pass_threshold is not None
            and self.fail_threshold is not None
            and self.pass_threshold <= self.fail_threshold
        ):
            raise ValueError(
                f"pass_threshold ({self.pass_threshold}) must be strictly greater than "
                f"fail_threshold ({self.fail_threshold})"
            )

        for name in ("score_threshold", "score_fail_threshold"):
            val = getattr(self, name)
            if val is not None and (not math.isfinite(val) or not (0.0 <= val <= 9.0)):
                raise ValueError(f"{name} must be finite and in [0.0, 9.0], got {val!r}")
        if (
            self.score_threshold is not None
            and self.score_fail_threshold is not None
            and self.score_threshold < self.score_fail_threshold
        ):
            raise ValueError(
                f"score_threshold ({self.score_threshold}) must be >= "
                f"score_fail_threshold ({self.score_fail_threshold})"
            )

        for name in ("choice_min_confidence", "choice_min_margin"):
            val = getattr(self, name)
            if val is not None and (not math.isfinite(val) or not (0.0 < val <= 1.0)):
                raise ValueError(f"{name} must be finite and in (0.0, 1.0], got {val!r}")

        if self.target_risk_alpha is not None and (
            not math.isfinite(self.target_risk_alpha) or not (0.0 < self.target_risk_alpha < 1.0)
        ):
            raise ValueError(
                f"target_risk_alpha must be finite and strictly in (0.0, 1.0), got {self.target_risk_alpha!r}"
            )

        for name in (
            "calibration_set_size",
            "min_samples_per_class",
            "num_positive",
            "num_negative",
        ):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val!r}")

        for name in (
            "median_abs_log_odds",
            "brier_score_before",
            "brier_score_after",
            "ece_before",
            "ece_after",
        ):
            val = getattr(self, name)
            if val is not None and (not math.isfinite(val) or val < 0.0):
                raise ValueError(f"{name} must be finite and >= 0.0, got {val!r}")

        return self

    @property
    def optimal_temperature(self) -> float:
        return self.binary_temperature

    @property
    def tau_accept(self) -> float | None:
        return self.pass_threshold

    @property
    def tau_reject(self) -> float | None:
        return self.fail_threshold

    @property
    def effective_multiclass_temperature(self) -> float:
        """Multiclass softmax temperature, respecting the 0.65 floor when derived from binary temperature."""
        if self.multiclass_temperature is not None:
            return self.multiclass_temperature
        return max(0.65, self.binary_temperature)

    def with_choice_thresholds(self, thresholds: ChoiceConformalThresholds) -> CalibrationProfile:
        """Returns a validated copy populated with conformal `ChoiceGate` cutoffs."""
        data = self.model_dump()
        data["choice_min_confidence"] = thresholds.min_confidence
        data["choice_min_margin"] = thresholds.min_margin
        return self.__class__.model_validate(data)

    def with_score_thresholds(self, thresholds: ScoreConformalThresholds) -> CalibrationProfile:
        """Returns a validated copy populated with conformal `ScoreGate` cutoffs."""
        data = self.model_dump()
        data["score_threshold"] = thresholds.score_threshold
        data["score_fail_threshold"] = thresholds.score_fail_threshold
        return self.__class__.model_validate(data)

    def save(self, path: str | Path) -> Path:
        """Atomically writes this profile as JSON to ``path``, rejecting symlinks."""
        if isinstance(path, str) and not path.strip():
            raise CalibrationError("CalibrationProfile save path cannot be empty.")
        target = Path(path)
        try:
            if target.is_symlink():
                raise CalibrationError(f"Refusing to write CalibrationProfile through symlink: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.model_dump(), indent=2, allow_nan=False) + "\n"
            fd, tmp_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_name)
            fd_owned = True
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    fd_owned = False
                    handle.write(payload)
                    handle.flush()
                os.replace(tmp_path, target)
            except BaseException:
                if fd_owned:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        except CalibrationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise CalibrationError(
                f"Failed to save CalibrationProfile to '{target}': {exc}"
            ) from exc
        return target

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        """Loads and validates a `CalibrationProfile` from a regular JSON file (<= 1 MiB)."""
        if isinstance(path, str) and not path.strip():
            raise CalibrationError("CalibrationProfile load path cannot be empty.")
        target = Path(path)
        try:
            if target.is_symlink():
                raise CalibrationError(f"Refusing to load CalibrationProfile from symlink: {target}")
            st = target.stat()
        except CalibrationError:
            raise
        except OSError as exc:
            raise CalibrationError(f"CalibrationProfile file not found or unreadable: '{target}'") from exc

        if not stat.S_ISREG(st.st_mode):
            raise CalibrationError(f"CalibrationProfile path is not a regular file: '{target}'")
        if st.st_size > _MAX_PROFILE_BYTES:
            raise CalibrationError(
                f"CalibrationProfile file '{target}' exceeds maximum size ({_MAX_PROFILE_BYTES} bytes)."
            )

        try:
            raw_text = target.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
        except (OSError, UnicodeDecodeError, ValueError, TypeError, RecursionError) as exc:
            raise CalibrationError(
                f"Malformed CalibrationProfile JSON in '{target}': {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise CalibrationError(
                f"CalibrationProfile JSON in '{target}' must be an object, got {type(parsed).__name__}"
            )
        return cls.model_validate(parsed)

    @classmethod
    def from_calibration_result(
        cls,
        result: CalibrationResult,
        *,
        model: str | None = None,
        backend: str | None = None,
        thresholds: ConformalCascadeThresholds | None = None,
    ) -> CalibrationProfile:
        """Builds a `CalibrationProfile` from a fitted `CalibrationResult` and optional thresholds."""
        kwargs: dict[str, Any] = {
            "model": model,
            "backend": backend,
            "binary_temperature": result.optimal_temperature,
            "multiclass_temperature": result.multiclass_temperature,
            "probe_auroc": result.probe_auroc,
            "median_abs_log_odds": result.median_abs_log_odds,
            "brier_score_before": result.brier_score_before,
            "brier_score_after": result.brier_score_after,
            "ece_before": result.ece_before,
            "ece_after": result.ece_after,
        }
        if thresholds is not None:
            kwargs.update({
                "pass_threshold": thresholds.tau_accept,
                "fail_threshold": thresholds.tau_reject,
                "target_risk_alpha": thresholds.target_risk_alpha,
                "empirical_escalation_rate": thresholds.empirical_escalation_rate,
                "calibration_set_size": thresholds.calibration_set_size,
                "guarantee_holds": thresholds.guarantee_holds,
                "min_samples_per_class": thresholds.min_samples_per_class,
                "num_positive": thresholds.num_positive,
                "num_negative": thresholds.num_negative,
            })
        return cls(**kwargs)

    @classmethod
    def from_log_odds_samples(
        cls,
        samples: Sequence[tuple[float, float, int]],
        *,
        target_risk_alpha: float | None = 0.05,
        model: str | None = None,
        backend: str | None = None,
        multiclass_temperature: float | None = None,
    ) -> CalibrationProfile:
        """Fits temperature first, then fits split-conformal cascade thresholds on scaled probabilities."""
        if target_risk_alpha is not None:
            alpha = _validate_alpha(target_risk_alpha)
            if len(samples) < 10:
                raise CalibrationError("Calibration requires matching sequences of at least 10 samples.")
            if len({int(y) for _, _, y in samples if not isinstance(y, bool) and y in (0, 1)}) < 2:
                raise CalibrationError(
                    "Calibration set must contain both positive and negative ground truth labels."
                )
        else:
            alpha = None

        cal = fit_calibration(samples)
        if multiclass_temperature is not None:
            if (
                isinstance(multiclass_temperature, bool)
                or not isinstance(multiclass_temperature, (int, float))
                or not math.isfinite(float(multiclass_temperature))
                or float(multiclass_temperature) <= 0.0
            ):
                raise CalibrationError(
                    f"multiclass_temperature must be a finite float > 0.0, got {multiclass_temperature!r}"
                )
            cal = cal.model_copy(update={"multiclass_temperature": float(multiclass_temperature)})

        thresholds: ConformalCascadeThresholds | None = None
        if alpha is not None:
            probs_after = [
                round(combine_symmetric_log_odds(f, r, logit_temperature=cal.optimal_temperature), 4)
                for f, r, _ in samples
            ]
            labels = [bool(y) for _, _, y in samples]
            thresholds = calibrate_cascade_thresholds(probs_after, labels, target_risk_alpha=alpha)

        return cls.from_calibration_result(
            cal,
            model=model,
            backend=backend,
            thresholds=thresholds,
        )


def resolve_calibration_profile(
    profile: CalibrationProfile | str | Path | None,
) -> CalibrationProfile | None:
    """Resolves a `CalibrationProfile` instance or JSON file path; raises `ConfigurationError` on invalid type."""
    if profile is None:
        return None
    if isinstance(profile, CalibrationProfile):
        return profile
    if isinstance(profile, Path):
        return CalibrationProfile.load(profile)
    if isinstance(profile, str):
        if not profile.strip():
            raise ConfigurationError("profile path cannot be empty or whitespace")
        return CalibrationProfile.load(profile)
    raise ConfigurationError(
        f"profile must be a CalibrationProfile instance, file path, or None, got {type(profile).__name__}"
    )


def _prepare_calibration_samples(
    *,
    dataset: str | Path | None,
    samples: Sequence[tuple[str, str, bool | int]] | None,
    target_risk_alpha: float | None,
    calibrate_multiclass: bool,
    save_cache: bool,
) -> tuple[list[tuple[str, str, bool]] | None, float | None]:
    if not isinstance(calibrate_multiclass, bool):
        raise ConfigurationError(
            f"calibrate_multiclass must be a bool, got {calibrate_multiclass!r}"
        )
    if not isinstance(save_cache, bool):
        raise ConfigurationError(f"save_cache must be a bool, got {save_cache!r}")
    if dataset is not None and samples is not None:
        raise ConfigurationError("Specify either 'dataset' or 'samples', not both.")

    alpha = _validate_alpha(target_risk_alpha) if target_risk_alpha is not None else None
    if alpha is not None:
        min_samples_per_class(alpha)

    if dataset is None and samples is None:
        return None, alpha

    raw_samples: Sequence[tuple[Any, Any, Any]]
    if dataset is not None:
        try:
            raw_samples = load_calibration_dataset(dataset)
        except CalibrationError:
            raise
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"Failed to load calibration dataset '{dataset}': {exc}") from exc
    else:
        assert samples is not None
        raw_samples = samples

    if not raw_samples:
        raise CalibrationError("Calibration dataset/samples cannot be empty.")

    normalized: list[tuple[str, str, bool]] = []
    for rubric, candidate, y in raw_samples:
        if not isinstance(rubric, str) or not rubric.strip():
            raise CalibrationError(f"Each calibration sample rubric must be a non-empty string, got {rubric!r}")
        if not isinstance(y, bool) and not (isinstance(y, int) and y in (0, 1)):
            raise CalibrationError(
                f"Each calibration sample label must be a bool or 0/1 integer, got {y!r}"
            )
        normalized.append((rubric.strip(), str(candidate), bool(y)))

    if alpha is not None:
        if len(normalized) < 10:
            raise CalibrationError(
                f"Split-conformal calibration requires at least 10 samples, got {len(normalized)}."
            )
        if len({y for _, _, y in normalized}) < 2:
            raise CalibrationError(
                "Calibration set must contain both positive and negative ground truth labels."
            )

    return normalized, alpha


def _extract_binary_record(
    top_fwd: list[dict[str, Any]],
    top_rev: list[dict[str, Any]],
    y: bool,
    backend_cls_name: str,
) -> tuple[float, float, bool, bool]:
    if not has_finite_logprobs(top_fwd) or not has_finite_logprobs(top_rev):
        raise BackendError(
            f"Backend {backend_cls_name} returned no token log-probabilities during calibration."
        )
    f_lo, _ = compute_binary_log_odds(top_fwd, pass_token="A", fail_token="B")
    r_lo, _ = compute_binary_log_odds(top_rev, pass_token="B", fail_token="A")
    observed = has_any_target_token(top_fwd, ("A", "B")) and has_any_target_token(top_rev, ("A", "B"))
    return f_lo, r_lo, y, observed


def _fit_binary_cal_result(
    records: Sequence[tuple[float, float, bool, bool]],
) -> CalibrationResult:
    obs_lo = [(f, r, 1 if y else 0) for f, r, y, obs in records if obs]
    all_lo = [(f, r, 1 if y else 0) for f, r, y, _ in records]
    fit_samples = obs_lo if (len(obs_lo) >= 4 and len({y for _, _, y in obs_lo}) >= 2) else all_lo
    return fit_calibration(fit_samples)


def _assemble_and_persist_profile(
    records: Sequence[tuple[float, float, bool, bool]],
    cal_res: CalibrationResult,
    *,
    alpha: float | None,
    effective_model: str,
    backend_spec: str,
    mc_temp: float | None,
    output_path: str | Path | None,
    persist_cache: bool,
    cache_path: Path | None,
) -> CalibrationProfile:
    if mc_temp is None and has_known_or_cached_temperature(effective_model, cache_path, binary=False):
        mc_temp = get_model_temperature(effective_model, cache_path, binary=False)
    if mc_temp is not None:
        cal_res = cal_res.model_copy(update={"multiclass_temperature": mc_temp})

    thresholds: ConformalCascadeThresholds | None = None
    if alpha is not None:
        probs_after = [
            round(combine_symmetric_log_odds(f, r, logit_temperature=cal_res.optimal_temperature), 4)
            if obs
            else 0.5
            for f, r, _, obs in records
        ]
        labels = [y for _, _, y, _ in records]
        thresholds = calibrate_cascade_thresholds(probs_after, labels, target_risk_alpha=alpha)

    if persist_cache:
        _persist_calibration_profile(
            effective_model,
            cal_res,
            cache_path=cache_path,
            strict_cache_io=True,
        )

    profile = CalibrationProfile.from_calibration_result(
        cal_res,
        model=effective_model,
        backend=backend_spec,
        thresholds=thresholds,
    )
    if output_path is not None:
        profile.save(output_path)
    return profile


def fit_calibration_profile(
    backend: str | LogitBackend | None = None,
    *,
    model: str | None = None,
    dataset: str | Path | None = None,
    samples: Sequence[tuple[str, str, bool | int]] | None = None,
    target_risk_alpha: float | None = 0.05,
    calibrate_multiclass: bool = False,
    output_path: str | Path | None = None,
    save_cache: bool = False,
    cache_path: str | Path | None = None,
) -> CalibrationProfile:
    """Fits a unified `CalibrationProfile` (temperature + split-conformal cutoffs) on a backend.

    Fits ``binary_temperature`` first on symmetric log-odds and computes split-conformal
    cascade cutoffs strictly on the post-temperature probabilities so temperature scaling
    and Layer-1 thresholds remain mutually consistent.
    """
    norm_samples, alpha = _prepare_calibration_samples(
        dataset=dataset,
        samples=samples,
        target_risk_alpha=target_risk_alpha,
        calibrate_multiclass=calibrate_multiclass,
        save_cache=save_cache,
    )
    persist_cache = save_cache or (cache_path is not None)
    resolved_cache_path = Path(cache_path) if cache_path is not None else None

    owns_backend = not isinstance(backend, LogitBackend)
    resolved = resolve_logit_backend(backend, model=model)
    try:
        effective_model = (
            model
            or getattr(resolved, "model", None)
            or getattr(resolved, "model_name", None)
            or "default"
        )
        backend_spec = infer_backend_spec(resolved)

        if norm_samples is None:
            cal_res = auto_calibrate_backend(
                resolved,
                model_key=effective_model,
                save_cache=persist_cache,
                cache_path=resolved_cache_path,
            )
            profile = CalibrationProfile.from_calibration_result(
                cal_res,
                model=effective_model,
                backend=backend_spec,
            )
            if output_path is not None:
                profile.save(output_path)
            return profile

        records: list[tuple[float, float, bool, bool]] = []
        for rubric, candidate, y in norm_samples:
            nonce = secrets.token_hex(8)
            p_fwd = compile_binary_prompt(
                rubric=rubric,
                candidate=candidate,
                pass_token="A",
                fail_token="B",
                query_first=True,
                boundary_nonce=nonce,
            )
            p_rev = compile_binary_prompt(
                rubric=rubric,
                candidate=candidate,
                pass_token="B",
                fail_token="A",
                query_first=True,
                boundary_nonce=nonce,
            )
            top_fwd = resolved.query_top_logprobs(p_fwd, top_k=10, allowed_tokens=["A", "B"])
            top_rev = resolved.query_top_logprobs(p_rev, top_k=10, allowed_tokens=["A", "B"])
            records.append(_extract_binary_record(top_fwd, top_rev, y, type(resolved).__name__))

        cal_res = _fit_binary_cal_result(records)
        mc_temp: float | None = None
        if calibrate_multiclass:
            try:
                mc_samples: list[tuple[dict[str, float], str]] = []
                for question, cand, options, target_key in UNIVERSAL_MULTICLASS_CALIBRATION_DATASET:
                    p_f, p_p, keys = _multiclass_probe_prompts(question, cand, options)
                    top_f = resolved.query_top_logprobs(p_f, top_k=10, allowed_tokens=keys)
                    top_p = resolved.query_top_logprobs(p_p, top_k=10, allowed_tokens=keys)
                    if not has_finite_logprobs(top_f) or not has_finite_logprobs(top_p):
                        mc_samples = []
                        break
                    mc_samples.append((_debiased_multiclass_accum(top_f, top_p, keys), target_key))
                if mc_samples:
                    mc_temp = _fit_multiclass_temperature(
                        mc_samples,
                        binary_auroc=cal_res.probe_auroc,
                        fallback_temp=cal_res.optimal_temperature,
                    )
            except (BackendError, ValueError, KeyError, RuntimeError):
                mc_temp = None

        return _assemble_and_persist_profile(
            records,
            cal_res,
            alpha=alpha,
            effective_model=effective_model,
            backend_spec=backend_spec,
            mc_temp=mc_temp,
            output_path=output_path,
            persist_cache=persist_cache,
            cache_path=resolved_cache_path,
        )
    finally:
        if owns_backend:
            resolved.close()


async def afit_calibration_profile(
    backend: str | LogitBackend | None = None,
    *,
    model: str | None = None,
    dataset: str | Path | None = None,
    samples: Sequence[tuple[str, str, bool | int]] | None = None,
    target_risk_alpha: float | None = 0.05,
    calibrate_multiclass: bool = False,
    output_path: str | Path | None = None,
    save_cache: bool = False,
    cache_path: str | Path | None = None,
    warm_prefix: bool | None = None,
) -> CalibrationProfile:
    """Async counterpart to `fit_calibration_profile`."""
    if warm_prefix is not None and not isinstance(warm_prefix, bool):
        raise ConfigurationError(f"warm_prefix must be a bool or None, got {warm_prefix!r}")
    norm_samples, alpha = _prepare_calibration_samples(
        dataset=dataset,
        samples=samples,
        target_risk_alpha=target_risk_alpha,
        calibrate_multiclass=calibrate_multiclass,
        save_cache=save_cache,
    )
    persist_cache = save_cache or (cache_path is not None)
    resolved_cache_path = Path(cache_path) if cache_path is not None else None

    owns_backend = not isinstance(backend, LogitBackend)
    resolved = resolve_logit_backend(backend, model=model)
    try:
        effective_model = (
            model
            or getattr(resolved, "model", None)
            or getattr(resolved, "model_name", None)
            or "default"
        )
        backend_spec = infer_backend_spec(resolved)

        if norm_samples is None:
            cal_res = await aauto_calibrate_backend(
                resolved,
                model_key=effective_model,
                save_cache=persist_cache,
                cache_path=resolved_cache_path,
            )
            profile = CalibrationProfile.from_calibration_result(
                cal_res,
                model=effective_model,
                backend=backend_spec,
            )
            if output_path is not None:
                profile.save(output_path)
            return profile

        use_warm = (
            warm_prefix
            if warm_prefix is not None
            else bool(getattr(resolved, "prefers_sequential_prefix_warmup", False))
        )
        records: list[tuple[float, float, bool, bool]] = []
        for rubric, candidate, y in norm_samples:
            nonce = secrets.token_hex(8)
            p_fwd = compile_binary_prompt(
                rubric=rubric,
                candidate=candidate,
                pass_token="A",
                fail_token="B",
                query_first=True,
                boundary_nonce=nonce,
            )
            p_rev = compile_binary_prompt(
                rubric=rubric,
                candidate=candidate,
                pass_token="B",
                fail_token="A",
                query_first=True,
                boundary_nonce=nonce,
            )
            if use_warm:
                top_fwd = await resolved.aquery_top_logprobs(p_fwd, top_k=10, allowed_tokens=["A", "B"])
                top_rev = await resolved.aquery_top_logprobs(p_rev, top_k=10, allowed_tokens=["A", "B"])
            else:
                top_fwd, top_rev = await asyncio.gather(
                    resolved.aquery_top_logprobs(p_fwd, top_k=10, allowed_tokens=["A", "B"]),
                    resolved.aquery_top_logprobs(p_rev, top_k=10, allowed_tokens=["A", "B"]),
                )
            records.append(_extract_binary_record(top_fwd, top_rev, y, type(resolved).__name__))

        cal_res = _fit_binary_cal_result(records)
        mc_temp: float | None = None
        if calibrate_multiclass:
            try:
                mc_samples: list[tuple[dict[str, float], str]] = []
                for question, cand, options, target_key in UNIVERSAL_MULTICLASS_CALIBRATION_DATASET:
                    p_f, p_p, keys = _multiclass_probe_prompts(question, cand, options)
                    if use_warm:
                        top_f = await resolved.aquery_top_logprobs(p_f, top_k=10, allowed_tokens=keys)
                        top_p = await resolved.aquery_top_logprobs(p_p, top_k=10, allowed_tokens=keys)
                    else:
                        top_f, top_p = await asyncio.gather(
                            resolved.aquery_top_logprobs(p_f, top_k=10, allowed_tokens=keys),
                            resolved.aquery_top_logprobs(p_p, top_k=10, allowed_tokens=keys),
                        )
                    if not has_finite_logprobs(top_f) or not has_finite_logprobs(top_p):
                        mc_samples = []
                        break
                    mc_samples.append((_debiased_multiclass_accum(top_f, top_p, keys), target_key))
                if mc_samples:
                    mc_temp = _fit_multiclass_temperature(
                        mc_samples,
                        binary_auroc=cal_res.probe_auroc,
                        fallback_temp=cal_res.optimal_temperature,
                    )
            except (BackendError, ValueError, KeyError, RuntimeError):
                mc_temp = None

        return _assemble_and_persist_profile(
            records,
            cal_res,
            alpha=alpha,
            effective_model=effective_model,
            backend_spec=backend_spec,
            mc_temp=mc_temp,
            output_path=output_path,
            persist_cache=persist_cache,
            cache_path=resolved_cache_path,
        )
    finally:
        if owns_backend:
            await resolved.aclose()
