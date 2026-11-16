from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from logitgate.backends.base import LogitBackend
from logitgate.calibration.profile import CalibrationProfile
from logitgate.distill._io import atomic_write_json_nofollow, read_json_object_nofollow
from logitgate.distill.anchor_bank import DistilledAnchorBank
from logitgate.distill.meta_calibrator import (
    LogitMetaCalibrator,
    afit_meta_calibrator,
    fit_meta_calibrator,
)
from logitgate.exceptions import CalibrationError, ConfigurationError
from logitgate.telemetry.models import DistillationRecord


class DistillationBundle(BaseModel):
    """Unified portable JSON artifact packaging Tier-A BYO distillation components.

    Can carry:
    - ``anchor_bank`` alone (100% model-agnostic — transfers zero-shot across any student/API),
    - ``meta_calibrator`` alone (student-specific 5-feature polarity & uncertainty head), or
    - ``anchor_bank + meta_calibrator (+ optional profile)`` fitted jointly on Split A.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    teacher_model: str | None = None
    student_model: str | None = None
    anchor_bank: DistilledAnchorBank | None = None
    meta_calibrator: LogitMetaCalibrator | None = None
    unanchored_meta_calibrator: LogitMetaCalibrator | None = None
    profile: CalibrationProfile | None = None

    @model_validator(mode="after")
    def _validate_bundle(self) -> DistillationBundle:
        if self.schema_version != 1:
            raise ConfigurationError(
                f"Unsupported DistillationBundle schema_version={self.schema_version}; expected 1."
            )
        if (
            self.anchor_bank is None
            and self.meta_calibrator is None
            and self.unanchored_meta_calibrator is None
            and self.profile is None
        ):
            raise ConfigurationError(
                "DistillationBundle must contain at least one of anchor_bank, meta_calibrator, or profile."
            )
        return self

    def cache_fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "anchor_bank": (
                self.anchor_bank.cache_fingerprint() if self.anchor_bank is not None else None
            ),
            "meta_calibrator": (
                self.meta_calibrator.cache_fingerprint()
                if self.meta_calibrator is not None
                else None
            ),
            "unanchored_meta_calibrator": (
                self.unanchored_meta_calibrator.cache_fingerprint()
                if self.unanchored_meta_calibrator is not None
                else None
            ),
            "profile": (
                self.profile.model_dump(mode="json", exclude={"created_at"})
                if self.profile is not None
                else None
            ),
        }

    def save(self, path: str | Path) -> Path:
        return atomic_write_json_nofollow(
            path,
            self.model_dump(mode="json"),
            artifact_name="DistillationBundle",
        )

    @classmethod
    def load(cls, path: str | Path) -> DistillationBundle:
        data = read_json_object_nofollow(path, artifact_name="DistillationBundle")
        try:
            return cls.model_validate(data)
        except (ConfigurationError, CalibrationError):
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid DistillationBundle in '{path}': {exc}"
            ) from exc


def resolve_distillation_bundle(
    bundle: DistillationBundle | str | Path | None,
) -> DistillationBundle | None:
    if bundle is None:
        return None
    if isinstance(bundle, DistillationBundle):
        return bundle
    if isinstance(bundle, (str, Path)):
        return DistillationBundle.load(bundle)
    raise ConfigurationError(
        f"distillation_bundle must be a DistillationBundle, path, or None, got {type(bundle).__name__}"
    )


def compile_distillation_bundle(
    *,
    dataset: str | Path | Sequence[Any] | None = None,
    telemetry: str | Path | Sequence[DistillationRecord] | None = None,
    backend: str | LogitBackend | None = None,
    student_model: str | None = None,
    teacher_model: str | None = None,
    teacher_reasons: Mapping[str, str] | None = None,
    teacher_labels: Mapping[str, bool] | None = None,
    include_anchors: bool = True,
    fit_meta: bool = True,
    max_bank_pairs: int = 24,
    default_retrieve_pairs: int = 1,
    min_rubric_overlap: float = 0.25,
    l2_penalty: float = 1.0,
    target_risk_alpha: float | None = 0.10,
) -> DistillationBundle:
    """Turnkey compiler building a portable `DistillationBundle` from Split-A data or telemetry.

    - When ``backend`` is ``None`` and ``dataset`` is given, builds a **100% model-agnostic**
      `DistilledAnchorBank` (`0` LLM calls) that transfers across any BYO student endpoint.
    - When ``telemetry`` is given without ``backend``, builds `DistilledAnchorBank` AND fits
      `LogitMetaCalibrator` + `CalibrationProfile` in `0` LLM calls directly from recorded
      student-teacher log-odds!
    - When ``backend`` and ``dataset`` are given, builds `DistilledAnchorBank` first and then
      fits `LogitMetaCalibrator` on the anchored student prompt distribution.
    """
    if (dataset is None) == (telemetry is None):
        raise ConfigurationError(
            "Provide exactly one of dataset=... or telemetry=... to compile_distillation_bundle."
        )

    anchor_bank: DistilledAnchorBank | None = None
    meta_cal: LogitMetaCalibrator | None = None
    unanchored_meta: LogitMetaCalibrator | None = None
    prof: CalibrationProfile | None = None

    if telemetry is not None:
        from logitgate.telemetry.distill import (
            load_distillation_records,
            recalibrate_from_telemetry,
        )

        recs = (
            load_distillation_records(telemetry)
            if isinstance(telemetry, (str, Path))
            else list(telemetry)
        )
        if include_anchors:
            anchor_bank = DistilledAnchorBank.from_distillation_records(
                recs,
                max_bank_pairs=max_bank_pairs,
                default_retrieve_pairs=default_retrieve_pairs,
                min_rubric_overlap=min_rubric_overlap,
            )
        if fit_meta and len(recs) >= 10:
            if backend is not None:
                rows = [
                    {
                        "id": r.event_id,
                        "rubric": r.rubric,
                        "candidate": r.candidate,
                        "expected": r.label,
                    }
                    for r in recs
                ]
                meta_cal = fit_meta_calibrator(
                    backend,
                    rows,
                    model=student_model,
                    anchor_bank=anchor_bank,
                    l2_penalty=l2_penalty,
                    target_risk_alpha=target_risk_alpha,
                )
                if anchor_bank is not None and anchor_bank.min_rubric_overlap > 0.0:
                    unanchored_meta = fit_meta_calibrator(
                        backend,
                        rows,
                        model=student_model,
                        anchor_bank=None,
                        l2_penalty=l2_penalty,
                        target_risk_alpha=target_risk_alpha,
                    )
            else:
                meta_cal = LogitMetaCalibrator.from_distillation_records(
                    recs,
                    l2_penalty=l2_penalty,
                    target_risk_alpha=target_risk_alpha,
                    model=student_model,
                )
                prof = recalibrate_from_telemetry(
                    recs,
                    target_risk_alpha=target_risk_alpha,
                    model=student_model,
                )
        eff_teacher = teacher_model or (anchor_bank.teacher_model if anchor_bank else None)
        eff_student = student_model or (meta_cal.model if meta_cal else None)
        return DistillationBundle(
            teacher_model=eff_teacher,
            student_model=eff_student,
            anchor_bank=anchor_bank,
            meta_calibrator=meta_cal,
            unanchored_meta_calibrator=unanchored_meta,
            profile=prof,
        )

    if dataset is None:
        raise ConfigurationError("dataset must not be None")
    if include_anchors:
        anchor_bank = DistilledAnchorBank.from_dataset(
            dataset,
            teacher_reasons=teacher_reasons,
            teacher_labels=teacher_labels,
            teacher_model=teacher_model,
            max_bank_pairs=max_bank_pairs,
            default_retrieve_pairs=default_retrieve_pairs,
            min_rubric_overlap=min_rubric_overlap,
        )
    if fit_meta and backend is not None:
        meta_cal = fit_meta_calibrator(
            backend,
            dataset,
            model=student_model,
            anchor_bank=anchor_bank,
            teacher_labels=teacher_labels,
            l2_penalty=l2_penalty,
            target_risk_alpha=target_risk_alpha,
        )
        if anchor_bank is not None and anchor_bank.min_rubric_overlap > 0.0:
            unanchored_meta = fit_meta_calibrator(
                backend,
                dataset,
                model=student_model,
                anchor_bank=None,
                teacher_labels=teacher_labels,
                l2_penalty=l2_penalty,
                target_risk_alpha=target_risk_alpha,
            )
    eff_student = student_model or (meta_cal.model if meta_cal else None)
    return DistillationBundle(
        teacher_model=teacher_model,
        student_model=eff_student,
        anchor_bank=anchor_bank,
        meta_calibrator=meta_cal,
        unanchored_meta_calibrator=unanchored_meta,
        profile=prof,
    )


async def acompile_distillation_bundle(
    *,
    dataset: str | Path | Sequence[Any] | None = None,
    telemetry: str | Path | Sequence[DistillationRecord] | None = None,
    backend: str | LogitBackend | None = None,
    student_model: str | None = None,
    teacher_model: str | None = None,
    teacher_reasons: Mapping[str, str] | None = None,
    teacher_labels: Mapping[str, bool] | None = None,
    include_anchors: bool = True,
    fit_meta: bool = True,
    max_bank_pairs: int = 24,
    default_retrieve_pairs: int = 1,
    min_rubric_overlap: float = 0.25,
    l2_penalty: float = 1.0,
    target_risk_alpha: float | None = 0.10,
    max_concurrency: int = 8,
) -> DistillationBundle:
    """Async counterpart to `compile_distillation_bundle`."""
    if backend is None or not fit_meta:
        return compile_distillation_bundle(
            dataset=dataset,
            telemetry=telemetry,
            backend=None,
            student_model=student_model,
            teacher_model=teacher_model,
            teacher_reasons=teacher_reasons,
            teacher_labels=teacher_labels,
            include_anchors=include_anchors,
            fit_meta=fit_meta,
            max_bank_pairs=max_bank_pairs,
            default_retrieve_pairs=default_retrieve_pairs,
            min_rubric_overlap=min_rubric_overlap,
            l2_penalty=l2_penalty,
            target_risk_alpha=target_risk_alpha,
        )
    if (dataset is None) == (telemetry is None):
        raise ConfigurationError(
            "Provide exactly one of dataset=... or telemetry=... to acompile_distillation_bundle."
        )

    anchor_bank: DistilledAnchorBank | None = None
    if telemetry is not None:
        from logitgate.telemetry.distill import load_distillation_records

        recs = (
            load_distillation_records(telemetry)
            if isinstance(telemetry, (str, Path))
            else list(telemetry)
        )
        if include_anchors:
            anchor_bank = DistilledAnchorBank.from_distillation_records(
                recs,
                max_bank_pairs=max_bank_pairs,
                default_retrieve_pairs=default_retrieve_pairs,
                min_rubric_overlap=min_rubric_overlap,
            )
        rows: str | Path | Sequence[Any] = [
            {
                "id": r.event_id,
                "rubric": r.rubric,
                "candidate": r.candidate,
                "expected": r.label,
            }
            for r in recs
        ]
    else:
        if dataset is None:
            raise ConfigurationError("dataset must not be None")
        if include_anchors:
            anchor_bank = DistilledAnchorBank.from_dataset(
                dataset,
                teacher_reasons=teacher_reasons,
                teacher_labels=teacher_labels,
                teacher_model=teacher_model,
                max_bank_pairs=max_bank_pairs,
                default_retrieve_pairs=default_retrieve_pairs,
                min_rubric_overlap=min_rubric_overlap,
            )
        rows = dataset

    meta_cal = await afit_meta_calibrator(
        backend,
        rows,
        model=student_model,
        anchor_bank=anchor_bank,
        teacher_labels=teacher_labels,
        l2_penalty=l2_penalty,
        target_risk_alpha=target_risk_alpha,
        max_concurrency=max_concurrency,
    )
    unanchored_meta: LogitMetaCalibrator | None = None
    if anchor_bank is not None and anchor_bank.min_rubric_overlap > 0.0:
        unanchored_meta = await afit_meta_calibrator(
            backend,
            rows,
            model=student_model,
            anchor_bank=None,
            teacher_labels=teacher_labels,
            l2_penalty=l2_penalty,
            target_risk_alpha=target_risk_alpha,
            max_concurrency=max_concurrency,
        )
    eff_student = student_model or meta_cal.model
    return DistillationBundle(
        teacher_model=teacher_model,
        student_model=eff_student,
        anchor_bank=anchor_bank,
        meta_calibrator=meta_cal,
        unanchored_meta_calibrator=unanchored_meta,
    )
