from __future__ import annotations

import asyncio
import datetime
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from logitgate.backends.base import LogitBackend, resolve_logit_backend
from logitgate.calibration.conformal import (
    ConformalCascadeThresholds,
    calibrate_cascade_thresholds,
)
from logitgate.debiasing.inversion import compile_binary_prompt
from logitgate.debiasing.softmax import clamped_sigmoid, compute_binary_log_odds
from logitgate.distill._io import atomic_write_json_nofollow, read_json_object_nofollow
from logitgate.exceptions import CalibrationError, ConfigurationError
from logitgate.telemetry.models import DistillationRecord

if TYPE_CHECKING:
    from logitgate.distill.anchor_bank import DistilledAnchorBank

FEATURE_NAMES: tuple[str, ...] = (
    "symmetric_log_odds",
    "signed_slot_bias",
    "positional_bias_magnitude",
    "top_k_entropy",
    "unobserved_residual_mass",
)


@dataclass(frozen=True, slots=True)
class LogitFeatureVector:
    """5-dimensional diagnostic feature vector extracted from Layer-1 first-token logprobs."""

    symmetric_log_odds: float
    signed_slot_bias: float
    positional_bias_magnitude: float
    top_k_entropy: float
    unobserved_residual_mass: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.symmetric_log_odds,
            self.signed_slot_bias,
            self.positional_bias_magnitude,
            self.top_k_entropy,
            self.unobserved_residual_mass,
        )


def _pass_distribution_stats(
    top_logprobs: Sequence[Mapping[str, Any]],
    *,
    target_tokens: tuple[str, str] = ("A", "B"),
) -> tuple[float, float]:
    """Computes (top_k_shannon_entropy_nats, residual_non_target_mass) from a single pass."""
    targets = {t.strip().upper() for t in target_tokens}
    by_raw: dict[str, float] = {}
    target_by_raw: dict[str, float] = {}
    for item in top_logprobs:
        if not isinstance(item, Mapping):
            continue
        raw_tok = item.get("token")
        lp = item.get("logprob")
        if raw_tok is None or lp is None or isinstance(lp, bool):
            continue
        try:
            lpf = float(lp)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lpf):
            continue
        lpf = min(lpf, 0.0)
        key = str(raw_tok)
        if key not in by_raw or lpf > by_raw[key]:
            by_raw[key] = lpf
        if key.strip().upper() in targets and (
            key not in target_by_raw or lpf > target_by_raw[key]
        ):
            target_by_raw[key] = lpf

    probs = [math.exp(v) for v in by_raw.values() if v > -50.0]
    total_obs = sum(probs)
    if total_obs > 1.0 and total_obs > 0.0:
        probs = [p / total_obs for p in probs]
        residual_tail = 0.0
    else:
        residual_tail = max(0.0, 1.0 - total_obs)

    entropy = 0.0
    for p in probs:
        if p > 1e-12:
            entropy -= p * math.log(p)
    if residual_tail > 1e-12:
        entropy -= residual_tail * math.log(residual_tail)

    target_mass = min(1.0, sum(math.exp(v) for v in target_by_raw.values() if v > -50.0))
    if total_obs > 1.0 and total_obs > 0.0:
        target_mass = min(1.0, target_mass / total_obs)
    non_target_mass = max(0.0, min(1.0, 1.0 - target_mass))
    return entropy, non_target_mass


def extract_logit_features(
    top_fwd: list[dict[str, Any]],
    top_rev: list[dict[str, Any]] | None = None,
) -> LogitFeatureVector:
    """Extracts the 5-feature Layer-1 vector from forward and optional reverse `top_logprobs`."""
    l_fwd, _ = compute_binary_log_odds(top_fwd, pass_token="A", fail_token="B")
    ent_fwd, res_fwd = _pass_distribution_stats(top_fwd, target_tokens=("A", "B"))
    if top_rev is not None:
        l_rev, _ = compute_binary_log_odds(top_rev, pass_token="B", fail_token="A")
        ent_rev, res_rev = _pass_distribution_stats(top_rev, target_tokens=("A", "B"))
        sym_lo = 0.5 * (l_fwd + l_rev)
        slot_bias = 0.5 * (l_fwd - l_rev)
        pos_mag = abs(slot_bias)
        ent = 0.5 * (ent_fwd + ent_rev)
        res = 0.5 * (res_fwd + res_rev)
    else:
        sym_lo = l_fwd
        slot_bias = 0.0
        pos_mag = 0.0
        ent = ent_fwd
        res = res_fwd
    return LogitFeatureVector(
        symmetric_log_odds=sym_lo,
        signed_slot_bias=slot_bias,
        positional_bias_magnitude=pos_mag,
        top_k_entropy=ent,
        unobserved_residual_mass=res,
    )


def _solve_linear_system(mat: list[list[float]], rhs: list[float]) -> list[float]:
    """Solves ``mat @ x = rhs`` via Gaussian elimination with partial pivoting."""
    d = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for col in range(d):
        pivot = max(range(col, d), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] = 1e-12
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        piv_val = a[col][col]
        for r in range(col + 1, d):
            factor = a[r][col] / piv_val
            if factor == 0.0:
                continue
            for c in range(col, d + 1):
                a[r][c] -= factor * a[col][c]

    sol = [0.0] * d
    for r in range(d - 1, -1, -1):
        acc = a[r][d]
        for c in range(r + 1, d):
            acc -= a[r][c] * sol[c]
        sol[r] = acc / a[r][r]
    return sol


def _fit_ridge_logistic(
    features: Sequence[tuple[float, float, float, float, float]],
    labels: Sequence[bool],
    *,
    l2_penalty: float = 1.0,
    min_sym_weight: float = 0.05,
    max_iter: int = 40,
) -> tuple[float, list[float]]:
    """Fits L2-regularized logistic regression with positive monotonicity on ``w[0]``."""
    n = len(features)
    if n == 0:
        raise CalibrationError("Cannot fit LogitMetaCalibrator on 0 samples.")
    # Parameter vector theta = [intercept, w0, w1, w2, w3, w4]
    theta = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    reg = [1e-4, l2_penalty, l2_penalty, l2_penalty, l2_penalty, l2_penalty]
    inv_n = 1.0 / float(n)

    def _loss(params: list[float]) -> float:
        nll = 0.0
        for x, y in zip(features, labels):
            z = params[0] + sum(params[j + 1] * x[j] for j in range(5))
            # Numerically stable binary cross-entropy: max(z, 0) - z*y + log(1 + exp(-|z|))
            yv = 1.0 if y else 0.0
            az = abs(z)
            nll += max(z, 0.0) - z * yv + (math.log1p(math.exp(-az)) if az < 50.0 else 0.0)
        pen = 0.5 * sum(reg[k] * params[k] * params[k] for k in range(6))
        return nll * inv_n + pen

    for _ in range(max_iter):
        grad = [reg[k] * theta[k] for k in range(6)]
        hess = [[(reg[r] if r == c else 0.0) for c in range(6)] for r in range(6)]

        for x, y in zip(features, labels):
            row = (1.0, x[0], x[1], x[2], x[3], x[4])
            z = sum(theta[k] * row[k] for k in range(6))
            p = clamped_sigmoid(z)
            diff = (p - (1.0 if y else 0.0)) * inv_n
            wt = max(p * (1.0 - p), 1e-8) * inv_n
            for r in range(6):
                grad[r] += diff * row[r]
                hr = hess[r]
                rr = row[r] * wt
                for c in range(r, 6):
                    hr[c] += rr * row[c]
        for r in range(6):
            for c in range(r):
                hess[r][c] = hess[c][r]

        step = _solve_linear_system(hess, grad)
        cur_loss = _loss(theta)
        step_scale = 1.0
        accepted = False
        for _ls in range(12):
            cand = [theta[k] - step_scale * step[k] for k in range(6)]
            cand[1] = max(min_sym_weight, min(20.0, cand[1]))
            cand[0] = max(-15.0, min(15.0, cand[0]))
            for k in range(2, 6):
                cand[k] = max(-15.0, min(15.0, cand[k]))
            if _loss(cand) <= cur_loss + 1e-10:
                delta = max(abs(cand[k] - theta[k]) for k in range(6))
                theta = cand
                accepted = True
                if delta < 1e-7:
                    return theta[0], theta[1:]
                break
            step_scale *= 0.5
        if not accepted:
            break

    theta[1] = max(min_sym_weight, min(20.0, theta[1]))
    return theta[0], theta[1:]


def _auroc_binary(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return round(wins / float(len(pos) * len(neg)), 4)


def _brier_score(probs: Sequence[float], labels: Sequence[bool]) -> float:
    if not probs:
        return 0.25
    return round(
        sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probs, labels)) / float(len(probs)),
        4,
    )


class LogitMetaCalibrator(BaseModel):
    """`<1 KB` L2-regularized multi-feature logit meta-calibrator with OOF conformal cutoffs.

    Corrects both systematic polarity skew (skepticism vs. sycophancy via ``intercept``) and
    slot/entropy/residual uncertainty without modifying base model weights. Enforces
    ``weights[0] >= 0.05`` so calibrated probability remains strictly monotonic in
    ``symmetric_log_odds``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    model: str | None = None
    feature_names: tuple[str, str, str, str, str] = (
        "symmetric_log_odds",
        "signed_slot_bias",
        "positional_bias_magnitude",
        "top_k_entropy",
        "unobserved_residual_mass",
    )
    intercept: float = 0.0
    weights: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0])
    l2_penalty: float = 1.0
    pass_threshold: float | None = None
    fail_threshold: float | None = None
    target_risk_alpha: float | None = None
    calibration_set_size: int = 0
    guarantee_holds: bool | None = None
    oof_auroc: float | None = None
    oof_brier_before: float | None = None
    oof_brier_after: float | None = None

    @model_validator(mode="after")
    def _validate_calibrator(self) -> LogitMetaCalibrator:
        if self.schema_version != 1:
            raise CalibrationError(
                f"Unsupported LogitMetaCalibrator schema_version={self.schema_version}; expected 1."
            )
        if len(self.weights) != 5 or any(
            isinstance(w, bool) or not math.isfinite(float(w)) for w in self.weights
        ):
            raise CalibrationError("LogitMetaCalibrator.weights must contain 5 finite floats.")
        if float(self.weights[0]) <= 0.0:
            raise CalibrationError(
                f"weights[0] (symmetric_log_odds) must be strictly positive for monotonicity, got {self.weights[0]}"
            )
        if isinstance(self.intercept, bool) or not math.isfinite(self.intercept):
            raise CalibrationError("intercept must be a finite float.")
        if isinstance(self.l2_penalty, bool) or not math.isfinite(self.l2_penalty) or self.l2_penalty <= 0.0:
            raise CalibrationError("l2_penalty must be a finite positive float.")
        if (self.pass_threshold is None) != (self.fail_threshold is None):
            raise CalibrationError("pass_threshold and fail_threshold must both be set or both be None.")
        if (
            self.pass_threshold is not None
            and self.fail_threshold is not None
            and not (0.0 <= self.fail_threshold < self.pass_threshold <= 1.0)
        ):
            raise CalibrationError(
                f"Thresholds must satisfy 0.0 <= fail_threshold < pass_threshold <= 1.0, "
                f"got [{self.fail_threshold}, {self.pass_threshold}]"
            )
        return self

    def predict_log_odds(self, features: LogitFeatureVector | Sequence[float]) -> float:
        vals = features.as_tuple() if isinstance(features, LogitFeatureVector) else tuple(features)
        if len(vals) != 5:
            raise CalibrationError(f"Expected 5 features, got {len(vals)}")
        raw = self.intercept + sum(self.weights[j] * float(vals[j]) for j in range(5))
        return max(-50.0, min(50.0, raw))

    def predict_proba(self, features: LogitFeatureVector | Sequence[float]) -> float:
        return clamped_sigmoid(self.predict_log_odds(features))

    @classmethod
    def fit(
        cls,
        features: Sequence[LogitFeatureVector | Sequence[float]],
        labels: Sequence[bool | int],
        *,
        l2_penalty: float = 1.0,
        target_risk_alpha: float | None = 0.10,
        num_oof_folds: int = 5,
        model: str | None = None,
    ) -> LogitMetaCalibrator:
        """Fits weights via Ridge-IRLS and fits out-of-fold split-conformal cascade cutoffs."""
        if isinstance(l2_penalty, bool) or not isinstance(l2_penalty, (int, float)) or float(l2_penalty) <= 0.0:
            raise CalibrationError(f"l2_penalty must be a positive float, got {l2_penalty!r}")
        l2 = float(l2_penalty)
        if len(features) != len(labels) or len(features) < 10:
            raise CalibrationError(
                "LogitMetaCalibrator.fit requires matching sequences of at least 10 samples."
            )

        feat_tuples: list[tuple[float, float, float, float, float]] = []
        bool_labels: list[bool] = []
        for f, y in zip(features, labels):
            tup = f.as_tuple() if isinstance(f, LogitFeatureVector) else tuple(float(v) for v in f)
            if len(tup) != 5 or any(not math.isfinite(v) for v in tup):
                raise CalibrationError("All 5 features in each sample must be finite floats.")
            if not isinstance(y, bool) and not (isinstance(y, int) and y in (0, 1)):
                raise CalibrationError(f"Labels must be booleans or 0/1 ints, got {y!r}")
            feat_tuples.append((tup[0], tup[1], tup[2], tup[3], tup[4]))
            bool_labels.append(bool(y))

        pos_idx = [i for i, y in enumerate(bool_labels) if y]
        neg_idx = [i for i, y in enumerate(bool_labels) if not y]
        if len(pos_idx) < 2 or len(neg_idx) < 2:
            raise CalibrationError(
                "LogitMetaCalibrator.fit requires at least 2 positive and 2 negative samples."
            )

        intercept, weights = _fit_ridge_logistic(feat_tuples, bool_labels, l2_penalty=l2)

        # Stratified K-fold out-of-fold predictions for leak-free conformal thresholds & metrics.
        k_folds = max(2, min(int(num_oof_folds), len(pos_idx), len(neg_idx)))
        folds: list[list[int]] = [[] for _ in range(k_folds)]
        for rank, idx in enumerate(pos_idx):
            folds[rank % k_folds].append(idx)
        for rank, idx in enumerate(neg_idx):
            folds[rank % k_folds].append(idx)

        oof_probs = [0.5] * len(feat_tuples)
        for fold_indices in folds:
            val_set = set(fold_indices)
            tr_feats = [feat_tuples[i] for i in range(len(feat_tuples)) if i not in val_set]
            tr_lbls = [bool_labels[i] for i in range(len(feat_tuples)) if i not in val_set]
            f_int, f_w = _fit_ridge_logistic(tr_feats, tr_lbls, l2_penalty=l2)
            for idx in fold_indices:
                x = feat_tuples[idx]
                z = f_int + sum(f_w[j] * x[j] for j in range(5))
                oof_probs[idx] = round(clamped_sigmoid(z), 4)

        raw_probs = [clamped_sigmoid(x[0]) for x in feat_tuples]
        oof_auroc = _auroc_binary(oof_probs, bool_labels)
        brier_before = _brier_score(raw_probs, bool_labels)
        brier_after = _brier_score(oof_probs, bool_labels)

        thresholds: ConformalCascadeThresholds | None = None
        if target_risk_alpha is not None:
            thresholds = calibrate_cascade_thresholds(
                oof_probs,
                bool_labels,
                target_risk_alpha=target_risk_alpha,
            )

        return cls(
            model=model,
            intercept=round(intercept, 6),
            weights=[round(w, 6) for w in weights],
            l2_penalty=l2,
            pass_threshold=thresholds.tau_accept if thresholds is not None else None,
            fail_threshold=thresholds.tau_reject if thresholds is not None else None,
            target_risk_alpha=thresholds.target_risk_alpha if thresholds is not None else None,
            calibration_set_size=len(feat_tuples),
            guarantee_holds=thresholds.guarantee_holds if thresholds is not None else None,
            oof_auroc=oof_auroc,
            oof_brier_before=brier_before,
            oof_brier_after=brier_after,
        )

    @classmethod
    def from_distillation_records(
        cls,
        records_or_path: str | Path | Sequence[DistillationRecord],
        *,
        l2_penalty: float = 1.0,
        target_risk_alpha: float | None = 0.10,
        model: str | None = None,
    ) -> LogitMetaCalibrator:
        """Fits a `LogitMetaCalibrator` in `0` LLM calls from `DistillationSink` telemetry."""
        if isinstance(records_or_path, (str, Path)):
            from logitgate.telemetry.distill import load_distillation_records

            records = load_distillation_records(records_or_path)
        else:
            records = list(records_or_path)

        feats: list[LogitFeatureVector] = []
        labels: list[bool] = []
        inferred_model = model
        for rec in records:
            if inferred_model is None and rec.student_model:
                inferred_model = rec.student_model
            if rec.student_forward_log_odds is not None and rec.student_reverse_log_odds is not None:
                lf = float(rec.student_forward_log_odds)
                lr = float(rec.student_reverse_log_odds)
                sym = 0.5 * (lf + lr)
                slot = 0.5 * (lf - lr)
                pos_mag = abs(slot)
            elif rec.student_log_odds_pass is not None:
                sym = float(rec.student_log_odds_pass)
                slot = 0.0
                pos_mag = float(rec.student_positional_bias or 0.0)
            else:
                continue
            feats.append(
                LogitFeatureVector(
                    symmetric_log_odds=sym,
                    signed_slot_bias=slot,
                    positional_bias_magnitude=pos_mag,
                    top_k_entropy=0.0,
                    unobserved_residual_mass=0.0,
                )
            )
            labels.append(bool(rec.label))

        return cls.fit(
            feats,
            labels,
            l2_penalty=l2_penalty,
            target_risk_alpha=target_risk_alpha,
            model=inferred_model,
        )

    def cache_fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intercept": round(self.intercept, 6),
            "weights": [round(w, 6) for w in self.weights],
            "pass_threshold": self.pass_threshold,
            "fail_threshold": self.fail_threshold,
        }

    def save(self, path: str | Path) -> Path:
        return atomic_write_json_nofollow(
            path,
            self.model_dump(mode="json"),
            artifact_name="LogitMetaCalibrator",
        )

    @classmethod
    def load(cls, path: str | Path) -> LogitMetaCalibrator:
        data = read_json_object_nofollow(path, artifact_name="LogitMetaCalibrator")
        try:
            return cls.model_validate(data)
        except (CalibrationError, ConfigurationError):
            raise
        except Exception as exc:
            raise CalibrationError(
                f"Invalid LogitMetaCalibrator in '{path}': {exc}"
            ) from exc


def resolve_meta_calibrator(
    calibrator: LogitMetaCalibrator | str | Path | None,
) -> LogitMetaCalibrator | None:
    if calibrator is None:
        return None
    if isinstance(calibrator, LogitMetaCalibrator):
        return calibrator
    if isinstance(calibrator, (str, Path)):
        return LogitMetaCalibrator.load(calibrator)
    raise ConfigurationError(
        f"meta_calibrator must be a LogitMetaCalibrator, path, or None, got {type(calibrator).__name__}"
    )


def fit_meta_calibrator(
    backend: str | LogitBackend,
    cases_or_path: str | Path | Sequence[Any],
    *,
    model: str | None = None,
    anchor_bank: DistilledAnchorBank | None = None,
    teacher_labels: Mapping[str, bool] | None = None,
    l2_penalty: float = 1.0,
    target_risk_alpha: float | None = 0.10,
) -> LogitMetaCalibrator:
    """Probes ``backend`` on ``cases_or_path`` (optionally with ``anchor_bank``) and fits `LogitMetaCalibrator`."""
    from logitgate.distill.anchor_bank import _load_generic_cases

    items = _load_generic_cases(cases_or_path)
    if len(items) < 10:
        raise CalibrationError("fit_meta_calibrator requires at least 10 calibration items.")

    owns = not isinstance(backend, LogitBackend)
    resolved = resolve_logit_backend(backend, model=model)
    eff_model = model or getattr(resolved, "model", None) or getattr(resolved, "model_name", None)
    try:
        feats: list[LogitFeatureVector] = []
        labels: list[bool] = []
        for row in items:
            nonce = secrets.token_hex(8)
            anchor_str = (
                anchor_bank.format_anchor_block(
                    row["rubric"], row["candidate"], boundary_nonce=nonce
                )
                if anchor_bank is not None
                else None
            )
            p_fwd = compile_binary_prompt(
                rubric=row["rubric"],
                candidate=row["candidate"],
                pass_token="A",
                fail_token="B",
                query_first=True,
                boundary_nonce=nonce,
                anchor_block=anchor_str,
            )
            p_rev = compile_binary_prompt(
                rubric=row["rubric"],
                candidate=row["candidate"],
                pass_token="B",
                fail_token="A",
                query_first=True,
                boundary_nonce=nonce,
                anchor_block=anchor_str,
            )
            top_fwd = resolved.query_top_logprobs(p_fwd, top_k=10, allowed_tokens=["A", "B"])
            top_rev = resolved.query_top_logprobs(p_rev, top_k=10, allowed_tokens=["A", "B"])
            feats.append(extract_logit_features(top_fwd, top_rev))
            item_id = row["id"]
            lbl = (
                bool(teacher_labels[item_id])
                if teacher_labels is not None and item_id in teacher_labels
                else bool(row["expected"])
            )
            labels.append(lbl)
    finally:
        if owns:
            resolved.close()

    return LogitMetaCalibrator.fit(
        feats,
        labels,
        l2_penalty=l2_penalty,
        target_risk_alpha=target_risk_alpha,
        model=eff_model,
    )


async def afit_meta_calibrator(
    backend: str | LogitBackend,
    cases_or_path: str | Path | Sequence[Any],
    *,
    model: str | None = None,
    anchor_bank: DistilledAnchorBank | None = None,
    teacher_labels: Mapping[str, bool] | None = None,
    l2_penalty: float = 1.0,
    target_risk_alpha: float | None = 0.10,
    max_concurrency: int = 8,
) -> LogitMetaCalibrator:
    """Async counterpart to `fit_meta_calibrator`."""
    from logitgate.distill.anchor_bank import _load_generic_cases

    items = _load_generic_cases(cases_or_path)
    if len(items) < 10:
        raise CalibrationError("afit_meta_calibrator requires at least 10 calibration items.")

    owns = not isinstance(backend, LogitBackend)
    resolved = resolve_logit_backend(backend, model=model)
    eff_model = model or getattr(resolved, "model", None) or getattr(resolved, "model_name", None)
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    warm = bool(getattr(resolved, "prefers_sequential_prefix_warmup", False))

    async def _probe_one(row: dict[str, Any]) -> tuple[LogitFeatureVector, bool]:
        async with sem:
            nonce = secrets.token_hex(8)
            anchor_str = (
                anchor_bank.format_anchor_block(
                    row["rubric"], row["candidate"], boundary_nonce=nonce
                )
                if anchor_bank is not None
                else None
            )
            p_fwd = compile_binary_prompt(
                rubric=row["rubric"],
                candidate=row["candidate"],
                pass_token="A",
                fail_token="B",
                query_first=True,
                boundary_nonce=nonce,
                anchor_block=anchor_str,
            )
            p_rev = compile_binary_prompt(
                rubric=row["rubric"],
                candidate=row["candidate"],
                pass_token="B",
                fail_token="A",
                query_first=True,
                boundary_nonce=nonce,
                anchor_block=anchor_str,
            )
            if warm:
                top_fwd = await resolved.aquery_top_logprobs(
                    p_fwd, top_k=10, allowed_tokens=["A", "B"]
                )
                top_rev = await resolved.aquery_top_logprobs(
                    p_rev, top_k=10, allowed_tokens=["A", "B"]
                )
            else:
                top_fwd, top_rev = await asyncio.gather(
                    resolved.aquery_top_logprobs(p_fwd, top_k=10, allowed_tokens=["A", "B"]),
                    resolved.aquery_top_logprobs(p_rev, top_k=10, allowed_tokens=["A", "B"]),
                )
            item_id = row["id"]
            lbl = (
                bool(teacher_labels[item_id])
                if teacher_labels is not None and item_id in teacher_labels
                else bool(row["expected"])
            )
            return extract_logit_features(top_fwd, top_rev), lbl

    try:
        pairs = await asyncio.gather(*(_probe_one(r) for r in items))
    finally:
        if owns:
            await resolved.aclose()

    feats = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    return LogitMetaCalibrator.fit(
        feats,
        labels,
        l2_penalty=l2_penalty,
        target_risk_alpha=target_risk_alpha,
        model=eff_model,
    )
