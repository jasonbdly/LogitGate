"""Online and offline covariate, score, and calibration drift detection (`logitgate.telemetry.drift`)."""

from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logitgate.exceptions import ConfigurationError
from logitgate.models import DecisionStatus
from logitgate.telemetry.models import ErrorPolicy, TelemetryRecord
from logitgate.telemetry.sinks import (
    TelemetrySink,
    _find_student_and_teacher_traces,
    _handle_sink_error,
    _validate_privacy_and_bounds,
    load_telemetry_records,
)

if TYPE_CHECKING:
    from logitgate.calibration.adaptive import AdaptiveConformalController

DriftSeverity = Literal["none", "moderate", "severe"]
DriftRecommendedAction = Literal["ok", "monitor", "widen_band_or_recalibrate"]


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Statistical summary comparing a current evaluation window against a reference baseline."""

    drift_detected: bool
    severity: DriftSeverity
    reasons: tuple[str, ...]
    recommended_action: DriftRecommendedAction
    reference_count: int
    current_count: int
    psi_log_odds: float
    ks_log_odds_statistic: float
    ks_log_odds_p_value: float
    psi_prob: float
    ks_prob_statistic: float
    ks_prob_p_value: float
    ref_mean_log_odds: float
    cur_mean_log_odds: float
    ref_mean_pos_bias: float
    cur_mean_pos_bias: float
    pos_bias_delta: float
    ref_escalation_rate: float
    cur_escalation_rate: float
    escalation_rate_delta: float
    observed_teacher_count: int
    teacher_disagreement_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "recommended_action": self.recommended_action,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "psi_log_odds": self.psi_log_odds,
            "ks_log_odds_statistic": self.ks_log_odds_statistic,
            "ks_log_odds_p_value": self.ks_log_odds_p_value,
            "psi_prob": self.psi_prob,
            "ks_prob_statistic": self.ks_prob_statistic,
            "ks_prob_p_value": self.ks_prob_p_value,
            "ref_mean_log_odds": self.ref_mean_log_odds,
            "cur_mean_log_odds": self.cur_mean_log_odds,
            "ref_mean_pos_bias": self.ref_mean_pos_bias,
            "cur_mean_pos_bias": self.cur_mean_pos_bias,
            "pos_bias_delta": self.pos_bias_delta,
            "ref_escalation_rate": self.ref_escalation_rate,
            "cur_escalation_rate": self.cur_escalation_rate,
            "escalation_rate_delta": self.escalation_rate_delta,
            "observed_teacher_count": self.observed_teacher_count,
            "teacher_disagreement_rate": self.teacher_disagreement_rate,
        }


def _validate_float_series(values: Sequence[float], label: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ConfigurationError(f"{label} must be a sequence of floats, not str/bytes")
    out: list[float] = []
    for x in values:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ConfigurationError(f"{label} must contain finite numbers, got {x!r}")
        xf = float(x)
        if not math.isfinite(xf):
            raise ConfigurationError(f"{label} must contain finite numbers, got {x!r}")
        out.append(xf)
    if not out:
        raise ConfigurationError(f"{label} must contain at least 1 sample")
    return out


def compute_psi(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    num_bins: int = 10,
    eps: float = 0.5,
) -> float:
    """Computes Jeffreys-smoothed adaptive quantile-binned Population Stability Index (PSI).

    Adapts bin count via Cochran's minimum-cell rule (``>= 5`` expected samples per bin on
    compact rolling windows) with Jeffreys ``0.5`` pseudo-count smoothing so finite-sample
    windows (e.g., ``N = 30``) do not suffer empty-bin log-ratio blowup.
    ``PSI < 0.10`` indicates negligible shift, ``0.10 <= PSI < 0.25`` moderate shift,
    and ``PSI >= 0.25`` severe distributional shift.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
        raise ConfigurationError(f"num_bins must be an integer >= 2, got {num_bins!r}")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ConfigurationError(f"eps must be a positive finite float, got {eps!r}")

    ref = sorted(_validate_float_series(reference, "reference"))
    cur = _validate_float_series(current, "current")
    eps_f = float(eps)

    n_ref = len(ref)
    n_cur = len(cur)
    eff_bins = min(num_bins, max(2, min(n_ref, n_cur) // 5))
    raw_edges: list[float] = []
    if ref[-1] - ref[0] <= 1e-12:
        center = ref[0]
        raw_edges = [center - 1e-6, center + 1e-6]
    else:
        for k in range(1, eff_bins):
            pos = k * (n_ref - 1) / eff_bins
            lo_idx = math.floor(pos)
            hi_idx = min(n_ref - 1, lo_idx + 1)
            frac = pos - lo_idx
            edge = ref[lo_idx] * (1.0 - frac) + ref[hi_idx] * frac
            if not raw_edges or edge > raw_edges[-1] + 1e-12:
                raw_edges.append(edge)
        if not raw_edges:
            center = ref[0]
            raw_edges = [center - 1e-6, center + 1e-6]

    b_count = len(raw_edges) + 1
    ref_bins = [0] * b_count
    cur_bins = [0] * b_count

    def _bin_index(val: float) -> int:
        lo, hi = 0, len(raw_edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if val <= raw_edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    for v in ref:
        ref_bins[_bin_index(v)] += 1
    for v in cur:
        cur_bins[_bin_index(v)] += 1

    ref_denom = n_ref + b_count * eps_f
    cur_denom = len(cur) + b_count * eps_f
    psi = 0.0
    for r_c, c_c in zip(ref_bins, cur_bins, strict=True):
        p = (r_c + eps_f) / ref_denom
        q = (c_c + eps_f) / cur_denom
        psi += (q - p) * math.log(q / p)
    return round(max(0.0, psi), 6)


def _kolmogorov_survival(d_stat: float, n_ref: int, n_cur: int) -> float:
    if d_stat <= 0.0:
        return 1.0
    d_clamped = min(float(d_stat), 1.0)
    n_eff = (n_ref * n_cur) / (n_ref + n_cur)
    root_n = math.sqrt(n_eff)
    lam = (root_n + 0.12 + 0.11 / root_n) * d_clamped
    if lam <= 1e-6:
        return 1.0
    if lam >= 6.0:
        return 0.0
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * (k * k) * (lam * lam))
        total += term
        if abs(term) < 1e-15:
            break
    return max(0.0, min(1.0, total))


def compute_ks_two_sample(
    reference: Sequence[float],
    current: Sequence[float],
) -> tuple[float, float]:
    """Computes exact two-sample Kolmogorov-Smirnov ``(statistic, p_value)`` without SciPy."""
    ref = sorted(_validate_float_series(reference, "reference"))
    cur = sorted(_validate_float_series(current, "current"))
    n_ref = len(ref)
    n_cur = len(cur)

    i = 0
    j = 0
    d_max = 0.0
    while i < n_ref and j < n_cur:
        v = min(ref[i], cur[j])
        while i < n_ref and ref[i] == v:
            i += 1
        while j < n_cur and cur[j] == v:
            j += 1
        diff = abs((i / n_ref) - (j / n_cur))
        d_max = max(d_max, diff)

    p_val = _kolmogorov_survival(d_max, n_ref, n_cur)
    return round(d_max, 6), round(p_val, 6)


@dataclass(slots=True)
class _ExtractedSignals:
    count: int
    log_odds: list[float]
    probs: list[float]
    pos_biases: list[float]
    escalation_rate: float
    teacher_count: int
    teacher_disagreements: int


def _extract_signals(
    source: Sequence[TelemetryRecord] | Sequence[float] | str | Path,
    label: str,
) -> _ExtractedSignals:
    if isinstance(source, (str, Path)):
        records = load_telemetry_records(source)
        return _extract_from_records(records, label)
    if isinstance(source, Sequence) and not isinstance(source, (bytes, bytearray)):
        if not source:
            raise ConfigurationError(f"{label} cannot be empty")
        first = source[0]
        if isinstance(first, TelemetryRecord):
            for item in source:
                if not isinstance(item, TelemetryRecord):
                    raise ConfigurationError(
                        f"{label} cannot mix TelemetryRecord and {type(item).__name__}"
                    )
            return _extract_from_records(source, label)  # type: ignore[arg-type]
        floats = _validate_float_series(source, label)  # type: ignore[arg-type]
        probs = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) for z in floats]
        return _ExtractedSignals(
            count=len(floats),
            log_odds=floats,
            probs=probs,
            pos_biases=[0.0 for _ in floats],
            escalation_rate=0.0,
            teacher_count=0,
            teacher_disagreements=0,
        )
    raise ConfigurationError(
        f"{label} must be a path, sequence of TelemetryRecord, or sequence of floats"
    )


def _extract_from_records(records: Sequence[TelemetryRecord], label: str) -> _ExtractedSignals:
    if not records:
        raise ConfigurationError(f"{label} contains 0 telemetry records")

    log_odds: list[float] = []
    probs: list[float] = []
    pos_biases: list[float] = []
    escalated_or_ambig = 0
    teacher_count = 0
    teacher_disagreements = 0

    for rec in records:
        student_tr, teacher_tr = _find_student_and_teacher_traces(rec.traces)
        target_tr = student_tr
        if target_tr is None:
            for tr in rec.traces:
                if tr.log_odds_pass is not None or tr.calibrated_prob is not None:
                    target_tr = tr
                    break

        if rec.escalated or rec.status == DecisionStatus.AMBIGUOUS or (
            student_tr is not None and student_tr.status == DecisionStatus.AMBIGUOUS
        ):
            escalated_or_ambig += 1

        if target_tr is not None:
            prob_val = target_tr.calibrated_prob
            lo_val = target_tr.log_odds_pass
            if lo_val is None and prob_val is not None:
                p_clamped = max(1e-6, min(1.0 - 1e-6, float(prob_val)))
                lo_val = math.log(p_clamped / (1.0 - p_clamped))
            if prob_val is None and lo_val is not None:
                prob_val = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(lo_val)))))
            if lo_val is not None and math.isfinite(lo_val):
                log_odds.append(float(lo_val))
            if prob_val is not None and math.isfinite(prob_val):
                probs.append(max(0.0, min(1.0, float(prob_val))))
            if target_tr.positional_bias_magnitude is not None and math.isfinite(
                target_tr.positional_bias_magnitude
            ):
                pos_biases.append(float(target_tr.positional_bias_magnitude))
        elif rec.probability is not None and math.isfinite(rec.probability):
            p_c = max(1e-6, min(1.0 - 1e-6, float(rec.probability)))
            probs.append(float(rec.probability))
            log_odds.append(math.log(p_c / (1.0 - p_c)))

        t_passed: bool | None = rec.teacher_passed
        if t_passed is None and teacher_tr is not None:
            t_passed = teacher_tr.status == DecisionStatus.PASSED
        if (
            t_passed is not None
            and student_tr is not None
            and student_tr.status in (DecisionStatus.PASSED, DecisionStatus.FAILED)
        ):
            teacher_count += 1
            if (student_tr.status == DecisionStatus.PASSED) != bool(t_passed):
                teacher_disagreements += 1

    if not log_odds:
        log_odds = [0.0]
    if not probs:
        probs = [0.5]
    if not pos_biases:
        pos_biases = [0.0]

    return _ExtractedSignals(
        count=len(records),
        log_odds=log_odds,
        probs=probs,
        pos_biases=pos_biases,
        escalation_rate=escalated_or_ambig / len(records),
        teacher_count=teacher_count,
        teacher_disagreements=teacher_disagreements,
    )


def detect_telemetry_drift(
    reference: Sequence[TelemetryRecord] | Sequence[float] | str | Path,
    current: Sequence[TelemetryRecord] | Sequence[float] | str | Path,
    *,
    num_bins: int = 10,
    psi_moderate: float = 0.10,
    psi_severe: float = 0.25,
    ks_alpha: float = 0.01,
    escalation_delta_threshold: float = 0.20,
    pos_bias_delta_threshold: float = 0.50,
    disagreement_threshold: float = 0.15,
    min_teacher_samples: int = 5,
) -> DriftReport:
    """Evaluates covariate, calibration, symmetry, and teacher-disagreement drift."""
    for name, val in (
        ("psi_moderate", psi_moderate),
        ("psi_severe", psi_severe),
        ("ks_alpha", ks_alpha),
        ("escalation_delta_threshold", escalation_delta_threshold),
        ("pos_bias_delta_threshold", pos_bias_delta_threshold),
        ("disagreement_threshold", disagreement_threshold),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) <= 0.0:
            raise ConfigurationError(f"{name} must be a positive finite float, got {val!r}")
    if psi_severe < psi_moderate:
        raise ConfigurationError(
            f"psi_severe ({psi_severe}) must be >= psi_moderate ({psi_moderate})"
        )
    if not (0.0 < float(ks_alpha) < 1.0):
        raise ConfigurationError(f"ks_alpha must be in (0.0, 1.0), got {ks_alpha!r}")
    if isinstance(min_teacher_samples, bool) or not isinstance(min_teacher_samples, int) or min_teacher_samples < 1:
        raise ConfigurationError(
            f"min_teacher_samples must be an integer >= 1, got {min_teacher_samples!r}"
        )

    ref_sig = _extract_signals(reference, "reference")
    cur_sig = _extract_signals(current, "current")

    psi_lo = compute_psi(ref_sig.log_odds, cur_sig.log_odds, num_bins=num_bins)
    ks_lo_d, ks_lo_p = compute_ks_two_sample(ref_sig.log_odds, cur_sig.log_odds)

    psi_p = compute_psi(ref_sig.probs, cur_sig.probs, num_bins=num_bins)
    ks_p_d, ks_p_p = compute_ks_two_sample(ref_sig.probs, cur_sig.probs)

    ref_mean_lo = statistics.mean(ref_sig.log_odds)
    cur_mean_lo = statistics.mean(cur_sig.log_odds)
    ref_mean_bias = statistics.mean(ref_sig.pos_biases)
    cur_mean_bias = statistics.mean(cur_sig.pos_biases)
    bias_delta = cur_mean_bias - ref_mean_bias

    esc_delta = cur_sig.escalation_rate - ref_sig.escalation_rate
    disagree_rate: float | None = (
        round(cur_sig.teacher_disagreements / cur_sig.teacher_count, 4)
        if cur_sig.teacher_count > 0
        else None
    )

    reasons: list[str] = []
    severity: DriftSeverity = "none"

    def _bump(level: DriftSeverity) -> None:
        nonlocal severity
        order = {"none": 0, "moderate": 1, "severe": 2}
        if order[level] > order[severity]:
            severity = level

    if psi_lo >= psi_severe:
        _bump("severe")
        reasons.append(f"psi_log_odds={psi_lo:.4f} >= {psi_severe:.2f}")
    elif psi_lo >= psi_moderate:
        _bump("moderate")
        reasons.append(f"psi_log_odds={psi_lo:.4f} >= {psi_moderate:.2f}")

    if ks_lo_p <= ks_alpha:
        _bump("severe" if ks_lo_p <= ks_alpha * 0.1 else "moderate")
        reasons.append(f"ks_log_odds_p={ks_lo_p:.6f} <= {ks_alpha:.4f} (D={ks_lo_d:.4f})")

    if abs(esc_delta) >= escalation_delta_threshold:
        _bump("severe" if abs(esc_delta) >= escalation_delta_threshold * 1.5 else "moderate")
        reasons.append(
            f"escalation_rate_delta={esc_delta:+.4f} exceeds +/-{escalation_delta_threshold:.2f}"
        )

    if bias_delta >= pos_bias_delta_threshold:
        _bump("moderate")
        reasons.append(
            f"pos_bias_delta={bias_delta:+.4f} >= {pos_bias_delta_threshold:.2f}"
        )

    if (
        disagree_rate is not None
        and cur_sig.teacher_count >= min_teacher_samples
        and disagree_rate >= disagreement_threshold
    ):
        _bump("severe" if disagree_rate >= disagreement_threshold * 1.5 else "moderate")
        reasons.append(
            f"teacher_disagreement_rate={disagree_rate:.4f} >= {disagreement_threshold:.2f} "
            f"(n={cur_sig.teacher_count})"
        )

    drift_detected = severity == "severe" or (severity == "moderate" and len(reasons) >= 2)
    if severity == "severe" or drift_detected:
        action: DriftRecommendedAction = "widen_band_or_recalibrate"
    elif severity == "moderate":
        action = "monitor"
    else:
        action = "ok"

    return DriftReport(
        drift_detected=drift_detected,
        severity=severity,
        reasons=tuple(reasons),
        recommended_action=action,
        reference_count=ref_sig.count,
        current_count=cur_sig.count,
        psi_log_odds=round(psi_lo, 4),
        ks_log_odds_statistic=round(ks_lo_d, 4),
        ks_log_odds_p_value=round(ks_lo_p, 6),
        psi_prob=round(psi_p, 4),
        ks_prob_statistic=round(ks_p_d, 4),
        ks_prob_p_value=round(ks_p_p, 6),
        ref_mean_log_odds=round(ref_mean_lo, 4),
        cur_mean_log_odds=round(cur_mean_lo, 4),
        ref_mean_pos_bias=round(ref_mean_bias, 4),
        cur_mean_pos_bias=round(cur_mean_bias, 4),
        pos_bias_delta=round(bias_delta, 4),
        ref_escalation_rate=round(ref_sig.escalation_rate, 4),
        cur_escalation_rate=round(cur_sig.escalation_rate, 4),
        escalation_rate_delta=round(esc_delta, 4),
        observed_teacher_count=cur_sig.teacher_count,
        teacher_disagreement_rate=disagree_rate,
    )


class RollingDriftMonitor(TelemetrySink):
    """Thread-safe streaming `TelemetrySink` that detects distribution shift over rolling windows.

    Can either compare incoming records against a fixed ``reference`` baseline (records or JSONL path)
    or auto-capture its first ``baseline_window`` records as the reference window, evaluating drift
    every ``check_every`` records over the most recent ``current_window`` evaluations.
    """

    def __init__(
        self,
        reference: Sequence[TelemetryRecord] | str | Path | None = None,
        *,
        baseline_window: int = 30,
        current_window: int = 30,
        check_every: int = 10,
        min_current_samples: int = 10,
        psi_moderate: float = 0.10,
        psi_severe: float = 0.25,
        ks_alpha: float = 0.01,
        on_drift: Callable[[DriftReport], None] | None = None,
        adaptive_controller: AdaptiveConformalController | None = None,
        drift_boost_theta: float = 1.0,
        boost_cooldown_checks: int = 3,
        on_error: ErrorPolicy = "warn",
    ) -> None:
        _, _, validated_error = _validate_privacy_and_bounds("omit_candidate", None, on_error)
        for name, val, floor in (
            ("baseline_window", baseline_window, 5),
            ("current_window", current_window, 5),
            ("check_every", check_every, 1),
            ("min_current_samples", min_current_samples, 5),
            ("boost_cooldown_checks", boost_cooldown_checks, 1),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val < floor:
                raise ConfigurationError(f"{name} must be an integer >= {floor}, got {val!r}")
        if (
            isinstance(drift_boost_theta, bool)
            or not isinstance(drift_boost_theta, (int, float))
            or not math.isfinite(float(drift_boost_theta))
            or float(drift_boost_theta) < 0.0
        ):
            raise ConfigurationError(
                f"drift_boost_theta must be a finite float >= 0.0, got {drift_boost_theta!r}"
            )
        if on_drift is not None and not callable(on_drift):
            raise ConfigurationError("on_drift must be callable or None")

        initial_ref: list[TelemetryRecord] = []
        if reference is not None:
            if isinstance(reference, (str, Path)):
                initial_ref = list(load_telemetry_records(reference))
            elif isinstance(reference, Sequence) and not isinstance(reference, (bytes, bytearray)):
                for item in reference:
                    if not isinstance(item, TelemetryRecord):
                        raise ConfigurationError(
                            f"reference sequence must contain TelemetryRecord, got {type(item).__name__}"
                        )
                    initial_ref.append(item)
            else:
                raise ConfigurationError("reference must be a path, sequence of TelemetryRecord, or None")
            if not initial_ref:
                raise ConfigurationError("Provided reference baseline cannot be empty")

        self._lock = threading.RLock()
        self._baseline_target = baseline_window
        self._baseline_frozen = bool(initial_ref)
        self._reference_records: list[TelemetryRecord] = initial_ref
        self._current_records: deque[TelemetryRecord] = deque(maxlen=current_window)
        self._check_every = check_every
        self._min_current = min(min_current_samples, current_window)
        self._psi_moderate = psi_moderate
        self._psi_severe = psi_severe
        self._ks_alpha = ks_alpha
        self._on_drift = on_drift
        self._adaptive_controller = adaptive_controller
        self._drift_boost_theta = float(drift_boost_theta)
        self._boost_cooldown_checks = boost_cooldown_checks
        self._checks_since_boost = boost_cooldown_checks
        self._on_error = validated_error
        self._ingested_after_baseline = 0
        self._latest_report: DriftReport | None = None
        self._reports: deque[DriftReport] = deque(maxlen=256)

    @property
    def latest_report(self) -> DriftReport | None:
        with self._lock:
            return self._latest_report

    @property
    def reports(self) -> tuple[DriftReport, ...]:
        with self._lock:
            return tuple(self._reports)

    @property
    def baseline_ready(self) -> bool:
        with self._lock:
            return self._baseline_frozen

    def reset_current(self) -> None:
        with self._lock:
            self._current_records.clear()
            self._ingested_after_baseline = 0

    def evaluate_now(self) -> DriftReport | None:
        with self._lock:
            if not self._baseline_frozen or len(self._current_records) < self._min_current:
                return None
            ref_snapshot = list(self._reference_records)
            cur_snapshot = list(self._current_records)
        report = detect_telemetry_drift(
            ref_snapshot,
            cur_snapshot,
            psi_moderate=self._psi_moderate,
            psi_severe=self._psi_severe,
            ks_alpha=self._ks_alpha,
        )
        with self._lock:
            self._latest_report = report
            self._reports.append(report)
        return report

    def emit(self, record: TelemetryRecord) -> None:
        callback_to_fire: Callable[[DriftReport], None] | None = None
        report_to_fire: DriftReport | None = None
        controller_to_boost: AdaptiveConformalController | None = None
        boost_amount = 0.0

        try:
            scrubbed = record.with_privacy(privacy="omit_candidate", max_candidate_chars=None)
            with self._lock:
                if not self._baseline_frozen:
                    self._reference_records.append(scrubbed)
                    if len(self._reference_records) >= self._baseline_target:
                        self._baseline_frozen = True
                    return

                self._current_records.append(scrubbed)
                self._ingested_after_baseline += 1
                if (
                    len(self._current_records) < self._min_current
                    or self._ingested_after_baseline % self._check_every != 0
                ):
                    return
                ref_snapshot = list(self._reference_records)
                cur_snapshot = list(self._current_records)

            report = detect_telemetry_drift(
                ref_snapshot,
                cur_snapshot,
                psi_moderate=self._psi_moderate,
                psi_severe=self._psi_severe,
                ks_alpha=self._ks_alpha,
            )
            with self._lock:
                self._latest_report = report
                self._reports.append(report)
                self._checks_since_boost += 1
                if report.drift_detected:
                    callback_to_fire = self._on_drift
                    report_to_fire = report
                    if (
                        self._adaptive_controller is not None
                        and self._drift_boost_theta > 0.0
                        and self._checks_since_boost >= self._boost_cooldown_checks
                    ):
                        controller_to_boost = self._adaptive_controller
                        boost_amount = self._drift_boost_theta
                        self._checks_since_boost = 0

            if controller_to_boost is not None:
                controller_to_boost.apply_drift_boost(boost_amount)
            if callback_to_fire is not None and report_to_fire is not None:
                callback_to_fire(report_to_fire)
        except Exception as exc:
            _handle_sink_error(exc, self._on_error, "RollingDriftMonitor")
