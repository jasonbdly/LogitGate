"""Online Adaptive Conformal Inference (Gibbs & Candès, 2021) for non-stationary selective cascades."""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from logitgate.calibration.conformal import _ceil4, _floor4
from logitgate.calibration.profile import (
    CalibrationProfile,
    resolve_calibration_profile,
)
from logitgate.config import LogitGateDefaults
from logitgate.debiasing.softmax import clamped_sigmoid
from logitgate.exceptions import ConfigurationError
from logitgate.models import DecisionStatus, Verdict

AdaptiveFeedbackMode = Literal["explicit", "cascade_audit"]

_LOGIT_EPS = 1e-6


def _safe_logit(p: float) -> float:
    clamped = max(_LOGIT_EPS, min(1.0 - _LOGIT_EPS, float(p)))
    return math.log(clamped / (1.0 - clamped))


@dataclass(frozen=True, slots=True)
class AdaptiveStepRecord:
    """Immutable audit record of a single online Adaptive Conformal update step."""

    step: int
    l1_status: DecisionStatus
    ground_truth: bool | None
    decided_at_l1: bool
    raw_error: float
    ips_error: float
    sampling_prob: float
    audited_fast_path: bool
    theta_before: float
    theta_after: float
    fail_threshold_after: float
    pass_threshold_after: float
    force_ambiguous_after: bool


@dataclass(frozen=True, slots=True)
class AdaptiveConformalSummary:
    """Aggregate telemetry of an `AdaptiveConformalController` over a stream of evaluations."""

    total_steps: int
    decided_count: int
    abstained_count: int
    audited_fast_path_count: int
    observed_l1_errors: int
    empirical_selective_risk_overall: float
    empirical_selective_risk_when_decided: float | None
    ips_estimated_selective_risk: float
    target_risk_alpha: float
    step_size: float
    theta_initial: float
    theta_current: float
    theta_max: float
    base_fail_threshold: float
    base_pass_threshold: float
    current_fail_threshold: float
    current_pass_threshold: float
    force_ambiguous: bool
    drift_boosts_applied: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "decided_count": self.decided_count,
            "abstained_count": self.abstained_count,
            "audited_fast_path_count": self.audited_fast_path_count,
            "observed_l1_errors": self.observed_l1_errors,
            "empirical_selective_risk_overall": self.empirical_selective_risk_overall,
            "empirical_selective_risk_when_decided": self.empirical_selective_risk_when_decided,
            "ips_estimated_selective_risk": self.ips_estimated_selective_risk,
            "target_risk_alpha": self.target_risk_alpha,
            "step_size": self.step_size,
            "theta_initial": self.theta_initial,
            "theta_current": self.theta_current,
            "theta_max": self.theta_max,
            "base_fail_threshold": self.base_fail_threshold,
            "base_pass_threshold": self.base_pass_threshold,
            "current_fail_threshold": self.current_fail_threshold,
            "current_pass_threshold": self.current_pass_threshold,
            "force_ambiguous": self.force_ambiguous,
            "drift_boosts_applied": self.drift_boosts_applied,
        }


class AdaptiveConformalController:
    r"""Gibbs-Candès Online Adaptive Conformal Controller over symmetric log-odds margins.

    Split conformal calibration (`fit_calibration_profile`) guarantees selective Layer-1 risk
    $\le \alpha^*$ under exchangeability, which breaks under domain shift or adversarial surges.
    `AdaptiveConformalController` maintains a scalar log-odds band offset $\theta_t \ge \theta_{\min}$
    around the initial thresholds $(\tau_{\text{rej},0}, \tau_{\text{acc},0})$:

    .. math::
        u_t = \operatorname{logit}(\tau_{\text{acc},0}) + \theta_t, \qquad
        \ell_t = \operatorname{logit}(\tau_{\text{rej},0}) - \theta_t

    updating after each evaluation via:

    .. math::
        \theta_{t+1} = \max\!\left(\theta_{\min},\; \theta_t + \eta (\hat{\text{err}}_t - \alpha^*)\right)

    Whenever $\theta_t \ge \theta_{\text{abstain}}$, ``force_ambiguous`` activates so Layer 1
    escalates every input to Layer 2 ($\text{err}_t = 0$), causing $\theta_t$ to decay linearly by
    $\eta \alpha^*$ per step without upper-clip bias. By telescoping over any arbitrary
    non-exchangeable adversarial sequence of length $T$, time-averaged selective Layer-1 risk satisfies
    $\frac{1}{T}\sum_{t=1}^T \mathbb{E}[\text{err}_t] \le \alpha^* + \frac{\theta_{\text{abstain}} + \eta/\rho_{\min} - \theta_1}{\eta T}$.

    Two feedback modes are supported without double-counting:
    - ``"explicit"`` (default when ``audit_sample_rate == 0.0``): caller supplies delayed ground truth
      via :meth:`observe`.
    - ``"cascade_audit"`` (auto-selected when ``audit_sample_rate > 0.0``): `Cascade` shadow-audits
      fast-path Layer-1 decisions via Tier-2 `JudgeGate` with probability ``audit_sample_rate``,
      updating $\theta_t$ via unbiased Horvitz-Thompson inverse-propensity error $\hat{\text{err}}_t = \text{err}_t / p$
      with zero human labels.
    """

    def __init__(
        self,
        profile: CalibrationProfile | str | Path | None = None,
        *,
        pass_threshold: float | None = None,
        fail_threshold: float | None = None,
        target_risk_alpha: float | None = None,
        step_size: float = 1.5,
        min_theta: float = 0.0,
        abstain_theta: float = 8.0,
        min_band_gap: float = 0.01,
        feedback_mode: AdaptiveFeedbackMode | None = None,
        audit_sample_rate: float = 0.0,
        override_on_audit: bool = True,
        max_history: int = 512,
        seed: int | None = 42,
    ) -> None:
        loaded_profile = resolve_calibration_profile(profile)

        raw_pass = (
            pass_threshold
            if pass_threshold is not None
            else (
                loaded_profile.pass_threshold
                if loaded_profile is not None and loaded_profile.pass_threshold is not None
                else LogitGateDefaults.pass_threshold
            )
        )
        raw_fail = (
            fail_threshold
            if fail_threshold is not None
            else (
                loaded_profile.fail_threshold
                if loaded_profile is not None and loaded_profile.fail_threshold is not None
                else LogitGateDefaults.fail_threshold
            )
        )
        raw_alpha = (
            target_risk_alpha
            if target_risk_alpha is not None
            else (
                loaded_profile.target_risk_alpha
                if loaded_profile is not None and loaded_profile.target_risk_alpha is not None
                else 0.05
            )
        )

        for name, val in (("pass_threshold", raw_pass), ("fail_threshold", raw_fail)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise ConfigurationError(f"{name} must be a finite float in [0.0, 1.0], got {val!r}")
            if not (0.0 <= float(val) <= 1.0):
                raise ConfigurationError(f"{name} must be in [0.0, 1.0], got {val!r}")
        base_pass = round(float(raw_pass), 4)
        base_fail = round(float(raw_fail), 4)
        if base_pass <= base_fail:
            raise ConfigurationError(
                f"pass_threshold ({base_pass}) must be strictly greater than fail_threshold ({base_fail})"
            )

        if (
            isinstance(raw_alpha, bool)
            or not isinstance(raw_alpha, (int, float))
            or not math.isfinite(float(raw_alpha))
            or not (0.0 < float(raw_alpha) < 1.0)
        ):
            raise ConfigurationError(
                f"target_risk_alpha must be strictly in (0.0, 1.0), got {raw_alpha!r}"
            )
        if (
            isinstance(step_size, bool)
            or not isinstance(step_size, (int, float))
            or not math.isfinite(float(step_size))
            or float(step_size) <= 0.0
        ):
            raise ConfigurationError(f"step_size must be a positive finite float, got {step_size!r}")
        if (
            isinstance(min_theta, bool)
            or not isinstance(min_theta, (int, float))
            or not math.isfinite(float(min_theta))
            or float(min_theta) > 0.0
        ):
            raise ConfigurationError(f"min_theta must be a finite float <= 0.0, got {min_theta!r}")
        if (
            isinstance(abstain_theta, bool)
            or not isinstance(abstain_theta, (int, float))
            or not math.isfinite(float(abstain_theta))
            or float(abstain_theta) <= 0.0
        ):
            raise ConfigurationError(
                f"abstain_theta must be a positive finite float, got {abstain_theta!r}"
            )
        if (
            isinstance(min_band_gap, bool)
            or not isinstance(min_band_gap, (int, float))
            or not math.isfinite(float(min_band_gap))
            or not (0.0 < float(min_band_gap) < 0.5)
        ):
            raise ConfigurationError(f"min_band_gap must be in (0.0, 0.5), got {min_band_gap!r}")
        if (
            isinstance(audit_sample_rate, bool)
            or not isinstance(audit_sample_rate, (int, float))
            or not math.isfinite(float(audit_sample_rate))
            or not (0.0 <= float(audit_sample_rate) <= 1.0)
        ):
            raise ConfigurationError(
                f"audit_sample_rate must be in [0.0, 1.0], got {audit_sample_rate!r}"
            )
        if not isinstance(override_on_audit, bool):
            raise ConfigurationError(f"override_on_audit must be a bool, got {override_on_audit!r}")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
            raise ConfigurationError(f"max_history must be an integer >= 1, got {max_history!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ConfigurationError(f"seed must be an int or None, got {seed!r}")

        rate_f = float(audit_sample_rate)
        if feedback_mode is None:
            resolved_mode: AdaptiveFeedbackMode = "cascade_audit" if rate_f > 0.0 else "explicit"
        elif feedback_mode in ("explicit", "cascade_audit"):
            resolved_mode = feedback_mode
        else:
            raise ConfigurationError(
                f"feedback_mode must be 'explicit', 'cascade_audit', or None, got {feedback_mode!r}"
            )

        if resolved_mode == "cascade_audit" and rate_f <= 0.0:
            raise ConfigurationError(
                "audit_sample_rate must be > 0.0 when feedback_mode='cascade_audit'"
            )
        if resolved_mode == "explicit" and rate_f > 0.0:
            raise ConfigurationError(
                "audit_sample_rate must be 0.0 when feedback_mode='explicit' "
                "(omit feedback_mode or set feedback_mode='cascade_audit')"
            )

        self._lock = threading.RLock()
        self._profile = loaded_profile
        self._base_pass = base_pass
        self._base_fail = base_fail
        self._base_pass_lo = _safe_logit(base_pass)
        self._base_fail_lo = _safe_logit(base_fail)
        self._alpha = float(raw_alpha)
        self._step_size = float(step_size)
        self._min_theta = float(min_theta)
        self._abstain_theta = float(abstain_theta)
        self._min_band_gap = float(min_band_gap)
        self._feedback_mode: AdaptiveFeedbackMode = resolved_mode
        self._audit_sample_rate = rate_f
        self._override_on_audit = override_on_audit
        self._seed = seed
        self._rng = random.Random(seed)

        self._theta = 0.0
        self._theta_max = 0.0
        self._total_steps = 0
        self._decided_count = 0
        self._abstained_count = 0
        self._audited_fast_path_count = 0
        self._observed_l1_errors = 0
        self._ips_error_sum = 0.0
        self._drift_boosts_applied = 0
        self._history: deque[AdaptiveStepRecord] = deque(maxlen=max_history)

    @property
    def profile(self) -> CalibrationProfile | None:
        return self._profile

    @property
    def base_pass_threshold(self) -> float:
        return self._base_pass

    @property
    def base_fail_threshold(self) -> float:
        return self._base_fail

    @property
    def target_risk_alpha(self) -> float:
        return self._alpha

    @property
    def step_size(self) -> float:
        return self._step_size

    @property
    def feedback_mode(self) -> AdaptiveFeedbackMode:
        return self._feedback_mode

    @property
    def audit_sample_rate(self) -> float:
        return self._audit_sample_rate

    @property
    def override_on_audit(self) -> bool:
        return self._override_on_audit

    @property
    def theta(self) -> float:
        with self._lock:
            return self._theta

    @property
    def history(self) -> tuple[AdaptiveStepRecord, ...]:
        with self._lock:
            return tuple(self._history)

    def _compute_state_unlocked(self) -> tuple[float, float, float, float, float, bool]:
        """Returns ``(theta, fail_t, pass_t, fail_lo, pass_lo, force_ambiguous)``."""
        theta = self._theta
        force_ambig = theta >= self._abstain_theta
        pass_lo = self._base_pass_lo + theta
        fail_lo = self._base_fail_lo - theta

        if abs(theta) <= 1e-12:
            return 0.0, self._base_fail, self._base_pass, self._base_fail_lo, self._base_pass_lo, False

        raw_pass = clamped_sigmoid(pass_lo)
        raw_fail = clamped_sigmoid(fail_lo)
        if theta > 0.0:
            pass_t = min(1.0, max(self._base_pass, _ceil4(raw_pass)))
            fail_t = max(0.0, min(self._base_fail, _floor4(raw_fail)))
        else:
            pass_t = round(raw_pass, 4)
            fail_t = round(raw_fail, 4)

        if pass_t - fail_t < self._min_band_gap:
            mid = 0.5 * (pass_t + fail_t)
            half = 0.5 * self._min_band_gap
            fail_t = round(max(0.0, mid - half), 4)
            pass_t = round(min(1.0, mid + half), 4)
            if pass_t <= fail_t:
                fail_t = max(0.0, round(pass_t - 0.0001, 4))
            pass_lo = _safe_logit(pass_t)
            fail_lo = _safe_logit(fail_t)

        return theta, fail_t, pass_t, fail_lo, pass_lo, force_ambig

    def current_thresholds(self) -> tuple[float, float, bool]:
        """Returns ``(fail_threshold, pass_threshold, force_ambiguous)`` for the current step."""
        with self._lock:
            _, fail_t, pass_t, _, _, force_ambig = self._compute_state_unlocked()
            return fail_t, pass_t, force_ambig

    def current_decision_state(self) -> tuple[float, float, float, float, float, bool]:
        """Returns ``(theta, fail_t, pass_t, fail_lo, pass_lo, force_ambiguous)`` atomically."""
        with self._lock:
            return self._compute_state_unlocked()

    def cache_fingerprint(self) -> dict[str, Any]:
        """Deterministic snapshot of active adaptive cutoffs for `VerdictCache` key isolation."""
        with self._lock:
            theta, fail_t, pass_t, _, _, force_ambig = self._compute_state_unlocked()
            return {
                "theta": round(theta, 6),
                "fail_threshold": fail_t,
                "pass_threshold": pass_t,
                "force_ambiguous": force_ambig,
                "feedback_mode": self._feedback_mode,
            }

    def should_audit_fast_path(self) -> bool:
        """Draws a thread-safe Bernoulli trial with probability ``audit_sample_rate``."""
        with self._lock:
            if self._feedback_mode != "cascade_audit" or self._audit_sample_rate <= 0.0:
                return False
            if self._audit_sample_rate >= 1.0:
                return True
            return self._rng.random() < self._audit_sample_rate

    @staticmethod
    def _extract_l1_status(verdict_or_status: Verdict | DecisionStatus | str) -> DecisionStatus:
        if isinstance(verdict_or_status, DecisionStatus):
            return verdict_or_status
        if isinstance(verdict_or_status, str):
            try:
                return DecisionStatus(verdict_or_status.strip().lower())
            except ValueError as exc:
                raise ConfigurationError(f"Invalid DecisionStatus {verdict_or_status!r}") from exc
        if isinstance(verdict_or_status, Verdict):
            for tr in verdict_or_status.traces:
                raw_l1 = (tr.metadata or {}).get("raw_l1_status")
                if isinstance(raw_l1, str):
                    try:
                        return DecisionStatus(raw_l1)
                    except ValueError:
                        pass
            if verdict_or_status.traces:
                return verdict_or_status.traces[0].status
            return verdict_or_status.status
        raise ConfigurationError(
            f"Expected Verdict or DecisionStatus, got {type(verdict_or_status).__name__}"
        )

    def observe(
        self,
        verdict_or_status: Verdict | DecisionStatus | str,
        ground_truth: bool,
        *,
        sampling_prob: float = 1.0,
    ) -> AdaptiveStepRecord:
        """Updates $\\theta_{t+1}$ from an observed ground-truth label (``True``=Pass, ``False``=Fail)."""
        if not isinstance(ground_truth, bool):
            raise ConfigurationError(f"ground_truth must be a bool, got {ground_truth!r}")
        if (
            isinstance(sampling_prob, bool)
            or not isinstance(sampling_prob, (int, float))
            or not math.isfinite(float(sampling_prob))
            or not (0.0 < float(sampling_prob) <= 1.0)
        ):
            raise ConfigurationError(
                f"sampling_prob must be in (0.0, 1.0], got {sampling_prob!r}"
            )
        l1_status = self._extract_l1_status(verdict_or_status)
        return self._step(
            l1_status=l1_status,
            ground_truth=ground_truth,
            sampling_prob=float(sampling_prob),
            audited_fast_path=False,
        )

    def record_cascade_outcome(
        self,
        *,
        l1_status: DecisionStatus,
        teacher_passed: bool | None,
        sampling_prob: float,
        audited_fast_path: bool,
    ) -> AdaptiveStepRecord:
        """Internal hook invoked by `Cascade` when ``feedback_mode='cascade_audit'``."""
        return self._step(
            l1_status=l1_status,
            ground_truth=teacher_passed,
            sampling_prob=sampling_prob,
            audited_fast_path=audited_fast_path,
        )

    def _step(
        self,
        *,
        l1_status: DecisionStatus,
        ground_truth: bool | None,
        sampling_prob: float,
        audited_fast_path: bool,
    ) -> AdaptiveStepRecord:
        with self._lock:
            theta_before = self._theta
            decided = l1_status in (DecisionStatus.PASSED, DecisionStatus.FAILED)

            raw_err = 0.0
            ips_err = 0.0
            if decided and ground_truth is not None:
                l1_passed = l1_status == DecisionStatus.PASSED
                if l1_passed != ground_truth:
                    raw_err = 1.0
                    ips_err = 1.0 / max(sampling_prob, 1e-6)

            theta_next = max(
                self._min_theta,
                theta_before + self._step_size * (ips_err - self._alpha),
            )
            self._theta = theta_next
            self._theta_max = max(self._theta_max, theta_next)

            self._total_steps += 1
            if decided:
                self._decided_count += 1
            else:
                self._abstained_count += 1
            if audited_fast_path:
                self._audited_fast_path_count += 1
            if raw_err > 0.0:
                self._observed_l1_errors += 1
            self._ips_error_sum += ips_err

            _, fail_after, pass_after, _, _, force_after = self._compute_state_unlocked()
            rec = AdaptiveStepRecord(
                step=self._total_steps,
                l1_status=l1_status,
                ground_truth=ground_truth,
                decided_at_l1=decided,
                raw_error=raw_err,
                ips_error=round(ips_err, 6),
                sampling_prob=round(sampling_prob, 6),
                audited_fast_path=audited_fast_path,
                theta_before=round(theta_before, 6),
                theta_after=round(theta_next, 6),
                fail_threshold_after=fail_after,
                pass_threshold_after=pass_after,
                force_ambiguous_after=force_after,
            )
            self._history.append(rec)
            return rec

    def apply_drift_boost(self, delta_theta: float = 1.0) -> tuple[float, float, bool]:
        """Proactively widens the conformal escalation band when covariate drift is detected."""
        if (
            isinstance(delta_theta, bool)
            or not isinstance(delta_theta, (int, float))
            or not math.isfinite(float(delta_theta))
            or float(delta_theta) < 0.0
        ):
            raise ConfigurationError(
                f"delta_theta must be a finite float >= 0.0, got {delta_theta!r}"
            )
        with self._lock:
            self._theta = min(self._abstain_theta, self._theta + float(delta_theta))
            self._theta_max = max(self._theta_max, self._theta)
            if float(delta_theta) > 0.0:
                self._drift_boosts_applied += 1
            _, fail_t, pass_t, _, _, force_ambig = self._compute_state_unlocked()
            return fail_t, pass_t, force_ambig

    def reset(self) -> None:
        """Resets $\\theta_t$ and step counters back to initial Split-A calibration state."""
        with self._lock:
            self._theta = 0.0
            self._theta_max = 0.0
            self._total_steps = 0
            self._decided_count = 0
            self._abstained_count = 0
            self._audited_fast_path_count = 0
            self._observed_l1_errors = 0
            self._ips_error_sum = 0.0
            self._drift_boosts_applied = 0
            self._history.clear()
            self._rng = random.Random(self._seed)

    def to_profile(self) -> CalibrationProfile:
        """Exports the controller's active thresholds as a portable `CalibrationProfile`."""
        with self._lock:
            _, fail_t, pass_t, _, _, _ = self._compute_state_unlocked()
            if self._profile is not None:
                return self._profile.model_copy(
                    update={
                        "fail_threshold": fail_t,
                        "pass_threshold": pass_t,
                        "target_risk_alpha": self._alpha,
                    }
                )
            return CalibrationProfile(
                model="default",
                binary_temperature=1.0,
                fail_threshold=fail_t,
                pass_threshold=pass_t,
                target_risk_alpha=self._alpha,
            )

    def summary(self) -> AdaptiveConformalSummary:
        """Returns an immutable summary of selective risk and threshold adaptation."""
        with self._lock:
            theta, fail_t, pass_t, _, _, force_ambig = self._compute_state_unlocked()
            overall_risk = (
                round(self._observed_l1_errors / self._total_steps, 4)
                if self._total_steps > 0
                else 0.0
            )
            cond_risk = (
                round(self._observed_l1_errors / self._decided_count, 4)
                if self._decided_count > 0
                else None
            )
            ips_risk = (
                round(self._ips_error_sum / self._total_steps, 4)
                if self._total_steps > 0
                else 0.0
            )
            return AdaptiveConformalSummary(
                total_steps=self._total_steps,
                decided_count=self._decided_count,
                abstained_count=self._abstained_count,
                audited_fast_path_count=self._audited_fast_path_count,
                observed_l1_errors=self._observed_l1_errors,
                empirical_selective_risk_overall=overall_risk,
                empirical_selective_risk_when_decided=cond_risk,
                ips_estimated_selective_risk=ips_risk,
                target_risk_alpha=self._alpha,
                step_size=self._step_size,
                theta_initial=0.0,
                theta_current=round(theta, 6),
                theta_max=round(self._theta_max, 6),
                base_fail_threshold=self._base_fail,
                base_pass_threshold=self._base_pass,
                current_fail_threshold=fail_t,
                current_pass_threshold=pass_t,
                force_ambiguous=force_ambig,
                drift_boosts_applied=self._drift_boosts_applied,
            )
