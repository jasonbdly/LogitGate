from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any, Literal

from logitgate.calibration.profile import CalibrationProfile
from logitgate.debiasing.inversion import compile_binary_prompt
from logitgate.exceptions import CalibrationError, ConfigurationError
from logitgate.models import DecisionStatus
from logitgate.telemetry.models import (
    DiscrepancyType,
    DistillationRecord,
    ErrorPolicy,
    TelemetryRecord,
    hash_candidate,
    truncate_text,
)
from logitgate.telemetry.sinks import (
    _DEFAULT_MAX_READ_BYTES,
    TelemetrySink,
    _find_student_and_teacher_traces,
    _handle_sink_error,
    _validate_privacy_and_bounds,
    _validate_target_file_path,
    open_append_nofollow,
    read_regular_text_nofollow,
    rotate_jsonl_files,
)

DistillationMode = Literal["all_escalations", "discrepancies_only"]
ExportFormat = Literal["sft_jsonl", "calibration_jsonl", "dpo_jsonl"]

_VALID_DISTILL_MODES: frozenset[str] = frozenset({"all_escalations", "discrepancies_only"})
_VALID_EXPORT_FORMATS: frozenset[str] = frozenset(
    {"sft_jsonl", "calibration_jsonl", "dpo_jsonl"}
)
_BINARY_SYSTEM_PROMPT = "Respond with a single character only (A, B). Do not think or explain."


def extract_distillation_record(
    record: TelemetryRecord,
    *,
    mode: DistillationMode = "all_escalations",
    max_candidate_chars: int | None = 16384,
) -> DistillationRecord | None:
    """Extracts a `DistillationRecord` from a multi-tier `TelemetryRecord`, or ``None`` if ineligible.

    Eligibility requires:
    1. A non-empty ``rubric`` and non-omitted ``candidate`` string.
    2. Both a Tier-1 binary logit student trace and a Tier-2 generative teacher trace.
    3. When ``mode='discrepancies_only'``, a disagreement between student and teacher
       (``false_accept_caught``, ``false_reject_overturned``, or ``polarity_flip``).
    """
    if mode not in _VALID_DISTILL_MODES:
        raise ConfigurationError(
            f"Invalid distillation mode={mode!r}: must be one of {sorted(_VALID_DISTILL_MODES)}"
        )
    if not record.rubric or not record.rubric.strip() or record.candidate is None:
        return None

    student_tr, teacher_tr = _find_student_and_teacher_traces(record.traces)
    if student_tr is None or teacher_tr is None:
        return None
    # Skip MultiRubricGate sub-traces whose log-odds belong to only one sub-criterion.
    if "rubric" in student_tr.metadata:
        return None

    if record.teacher_passed is not None:
        teacher_passed = bool(record.teacher_passed)
    elif "judge_log_odds" in teacher_tr.metadata and isinstance(
        teacher_tr.metadata["judge_log_odds"], (int, float)
    ):
        teacher_passed = float(teacher_tr.metadata["judge_log_odds"]) >= 0.0
    else:
        teacher_passed = teacher_tr.status == DecisionStatus.PASSED

    if "judge_log_odds" in teacher_tr.metadata and isinstance(
        teacher_tr.metadata["judge_log_odds"], (int, float)
    ):
        raw_j_lo = float(teacher_tr.metadata["judge_log_odds"])
        teacher_lo: float | None = raw_j_lo
        teacher_prob: float | None = round(
            1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw_j_lo)))), 4
        )
    else:
        teacher_lo = teacher_tr.log_odds_pass
        teacher_prob = teacher_tr.calibrated_prob

    s_status = student_tr.status
    s_lo = student_tr.log_odds_pass

    discrepancy: DiscrepancyType
    if s_status == DecisionStatus.PASSED and not teacher_passed:
        discrepancy = "false_accept_caught"
    elif s_status == DecisionStatus.FAILED and teacher_passed:
        discrepancy = "false_reject_overturned"
    elif (
        s_status == DecisionStatus.AMBIGUOUS
        and s_lo is not None
        and ((s_lo > 0.0) != teacher_passed)
    ):
        discrepancy = "polarity_flip"
    else:
        discrepancy = "ambiguous_resolved"

    if mode == "discrepancies_only" and discrepancy == "ambiguous_resolved":
        return None

    cand_str = truncate_text(record.candidate, max_candidate_chars)
    dedupe_digest = hash_candidate(f"{record.rubric.strip()}\0{cand_str}")
    return DistillationRecord(
        event_id=record.event_id,
        timestamp=record.timestamp,
        rubric=record.rubric.strip(),
        candidate=cand_str,
        label=teacher_passed,
        discrepancy_type=discrepancy,
        student_layer=student_tr.layer_name,
        student_model=record.model,
        student_status=s_status,
        student_prob=student_tr.calibrated_prob,
        student_forward_log_odds=student_tr.forward_log_odds,
        student_reverse_log_odds=student_tr.reverse_log_odds,
        student_log_odds_pass=s_lo,
        student_positional_bias=student_tr.positional_bias_magnitude,
        teacher_layer=teacher_tr.layer_name,
        teacher_model=record.teacher_model or record.model,
        teacher_passed=teacher_passed,
        teacher_prob=teacher_prob,
        teacher_log_odds_pass=teacher_lo,
        teacher_reason=(
            truncate_text(teacher_tr.diagnostic_report, max_candidate_chars)
            if teacher_tr.diagnostic_report is not None
            else None
        ),
        cascade_passed=record.passed,
        candidate_sha256=dedupe_digest,
        tags=dict(record.tags),
    )


class DistillationSink(TelemetrySink):
    """Self-distilling JSONL sink collecting teacher-adjudicated Layer-1 escalations.

    Every JSONL line written by `DistillationSink` contains ``{"rubric", "candidate", "label"}``
    at the top level (natively compatible with ``load_calibration_dataset()`` and
    ``logitgate calibrate --dataset``) PLUS ``student_forward_log_odds`` and
    ``student_reverse_log_odds`` for zero-LLM-call online profile recalibration via
    ``recalibrate_from_telemetry()``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: DistillationMode = "all_escalations",
        dedupe_by_hash: bool = True,
        max_dedupe_keys: int = 50000,
        max_candidate_chars: int | None = 16384,
        max_bytes: int | None = None,
        backup_count: int = 0,
        fsync: bool = False,
        on_error: ErrorPolicy = "warn",
    ) -> None:
        target = _validate_target_file_path(path, "DistillationSink")
        if mode not in _VALID_DISTILL_MODES:
            raise ConfigurationError(
                f"Invalid mode={mode!r}: must be one of {sorted(_VALID_DISTILL_MODES)}"
            )
        _, val_chars, val_err = _validate_privacy_and_bounds("full", max_candidate_chars, on_error)
        if not isinstance(dedupe_by_hash, bool):
            raise ConfigurationError(f"dedupe_by_hash must be a bool, got {dedupe_by_hash!r}")
        if (
            isinstance(max_dedupe_keys, bool)
            or not isinstance(max_dedupe_keys, int)
            or max_dedupe_keys < 1
        ):
            raise ConfigurationError(
                f"max_dedupe_keys must be an integer >= 1, got {max_dedupe_keys!r}"
            )
        if max_bytes is not None and (
            isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 256
        ):
            raise ConfigurationError(
                f"max_bytes must be an integer >= 256 or None, got {max_bytes!r}"
            )
        if isinstance(backup_count, bool) or not isinstance(backup_count, int) or backup_count < 0:
            raise ConfigurationError(
                f"backup_count must be an integer >= 0, got {backup_count!r}"
            )
        if not isinstance(fsync, bool):
            raise ConfigurationError(f"fsync must be a bool, got {fsync!r}")

        self.path: Path = target
        self.mode: DistillationMode = mode
        self.dedupe_by_hash: bool = dedupe_by_hash
        self.max_dedupe_keys: int = max_dedupe_keys
        self.max_candidate_chars: int | None = val_chars
        self.max_bytes: int | None = max_bytes
        self.backup_count: int = backup_count
        self.fsync: bool = fsync
        self.on_error: ErrorPolicy = val_err
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._handle: IO[str] | None = None
        self._closed = False
        self._records_written = 0

    @property
    def records_written(self) -> int:
        return self._records_written

    def _ensure_open_locked(self) -> IO[str]:
        if self._handle is None:
            self._handle = open_append_nofollow(self.path)
        return self._handle

    def _rotate_if_needed_locked(self, incoming_bytes: int) -> None:
        if self.max_bytes is None:
            return
        current_size = 0
        if self._handle is not None:
            self._handle.flush()
            try:
                current_size = os.fstat(self._handle.fileno()).st_size
            except OSError:
                current_size = 0
        elif self.path.exists() and not self.path.is_symlink():
            try:
                current_size = self.path.stat().st_size
            except OSError:
                current_size = 0
        if current_size > 0 and current_size + incoming_bytes > self.max_bytes:
            if self._handle is not None:
                try:
                    self._handle.close()
                finally:
                    self._handle = None
            rotate_jsonl_files(self.path, self.backup_count)

    def emit(self, record: TelemetryRecord) -> None:
        try:
            distilled = extract_distillation_record(
                record,
                mode=self.mode,
                max_candidate_chars=self.max_candidate_chars,
            )
            if distilled is None:
                return
            line = json.dumps(distilled.model_dump(mode="json"), allow_nan=False) + "\n"
            encoded_len = len(line.encode("utf-8"))
            with self._lock:
                if self._closed:
                    return
                if self.dedupe_by_hash:
                    key = distilled.candidate_sha256
                    if key in self._seen:
                        self._seen.move_to_end(key)
                        return
                    self._seen[key] = None
                    if len(self._seen) > self.max_dedupe_keys:
                        self._seen.popitem(last=False)
                self._rotate_if_needed_locked(encoded_len)
                handle = self._ensure_open_locked()
                handle.write(line)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
                self._records_written += 1
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            _handle_sink_error(exc, self.on_error, "DistillationSink")

    def flush(self) -> None:
        with self._lock:
            if self._handle is not None and not self._closed:
                try:
                    self._handle.flush()
                    if self.fsync:
                        os.fsync(self._handle.fileno())
                except OSError as exc:
                    _handle_sink_error(exc, self.on_error, "DistillationSink")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except OSError:
                    pass
                finally:
                    self._handle = None


def _load_distillation_or_telemetry_jsonl(
    path: str | Path,
    *,
    max_bytes: int = _DEFAULT_MAX_READ_BYTES,
) -> list[DistillationRecord]:
    target = _validate_target_file_path(path, "Distillation/Telemetry JSONL")
    raw_text = read_regular_text_nofollow(
        target, max_bytes=max_bytes, label="Distillation/Telemetry JSONL"
    )
    records: list[DistillationRecord] = []
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise TypeError("JSONL line must decode to an object")
            if "discrepancy_type" in parsed and "teacher_passed" in parsed and "label" in parsed:
                records.append(DistillationRecord.model_validate(parsed))
            else:
                tel = TelemetryRecord.model_validate(parsed)
                dist = extract_distillation_record(tel, mode="all_escalations")
                if dist is not None:
                    records.append(dist)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ConfigurationError(
                f"Malformed JSONL record at '{target}' line {lineno}: {exc}"
            ) from exc
    return records


def load_distillation_records(
    path: str | Path,
    *,
    max_bytes: int = _DEFAULT_MAX_READ_BYTES,
) -> list[DistillationRecord]:
    """Loads `DistillationRecord` items from a distillation or raw telemetry JSONL file."""
    return _load_distillation_or_telemetry_jsonl(path, max_bytes=max_bytes)


def _normalize_distillation_source(
    source: str | Path | Sequence[DistillationRecord | TelemetryRecord],
) -> list[DistillationRecord]:
    if isinstance(source, (str, Path)):
        return _load_distillation_or_telemetry_jsonl(source)
    out: list[DistillationRecord] = []
    for item in source:
        if isinstance(item, DistillationRecord):
            out.append(item)
        elif isinstance(item, TelemetryRecord):
            dist = extract_distillation_record(item, mode="all_escalations")
            if dist is not None:
                out.append(dist)
        else:
            raise ConfigurationError(
                f"Expected DistillationRecord or TelemetryRecord, got {type(item).__name__}"
            )
    return out


def recalibrate_from_telemetry(
    source: str | Path | Sequence[DistillationRecord | TelemetryRecord],
    *,
    target_risk_alpha: float | None = 0.10,
    model: str | None = None,
    backend: str | None = None,
    multiclass_temperature: float | None = None,
    output_path: str | Path | None = None,
    dedupe: bool = True,
) -> CalibrationProfile:
    """Fits a unified `CalibrationProfile` directly from logged student/teacher escalations with 0 LLM calls.

    Uses the Tier-1 student's recorded ``(student_forward_log_odds, student_reverse_log_odds)``
    paired against the Tier-2 teacher's authoritative verdict ``label`` to fit post-hoc
    temperature scaling AND split-conformal cascade thresholds in milliseconds.
    """
    if not isinstance(dedupe, bool):
        raise ConfigurationError(f"dedupe must be a bool, got {dedupe!r}")
    records = _normalize_distillation_source(source)
    if not records:
        raise CalibrationError(
            "No teacher-adjudicated binary escalation records found in telemetry source."
        )

    samples: list[tuple[float, float, int]] = []
    seen_hashes: set[str] = set()
    inferred_model: str | None = model

    for rec in records:
        if dedupe:
            if rec.candidate_sha256 in seen_hashes:
                continue
            seen_hashes.add(rec.candidate_sha256)
        f_lo = rec.student_forward_log_odds
        if f_lo is None or not math.isfinite(f_lo):
            continue
        r_lo = (
            rec.student_reverse_log_odds
            if (rec.student_reverse_log_odds is not None and math.isfinite(rec.student_reverse_log_odds))
            else f_lo
        )
        if inferred_model is None and rec.student_model:
            inferred_model = rec.student_model
        samples.append((float(f_lo), float(r_lo), 1 if rec.teacher_passed else 0))

    if len(samples) < 4:
        raise CalibrationError(
            f"Telemetry recalibration requires at least 4 finite student log-odds records "
            f"(>= 10 when target_risk_alpha is set), got {len(samples)}."
        )
    if len({y for _, _, y in samples}) < 2:
        raise CalibrationError(
            "Telemetry recalibration requires both positive (PASS) and negative (FAIL) teacher verdicts."
        )

    profile = CalibrationProfile.from_log_odds_samples(
        samples,
        target_risk_alpha=target_risk_alpha,
        model=inferred_model,
        backend=backend,
        multiclass_temperature=multiclass_temperature,
    )
    if output_path is not None:
        profile.save(output_path)
    return profile


def export_distillation_dataset(
    source: str | Path | Sequence[DistillationRecord | TelemetryRecord],
    output_path: str | Path,
    *,
    format: ExportFormat = "sft_jsonl",
    include_symmetric_swaps: bool = True,
    dedupe: bool = True,
    discrepancies_only: bool = False,
) -> int:
    """Exports teacher-adjudicated telemetry into a fine-tuning or calibration JSONL dataset.

    Formats:
    - ``"sft_jsonl"``: ChatML ``{"messages": [...]}`` pairs using `compile_binary_prompt`
      (2x symmetric ``A/B`` and ``B/A`` examples per record when ``include_symmetric_swaps=True``
      so fine-tuned students inherit zero positional bias).
    - ``"calibration_jsonl"``: ``{"rubric", "candidate", "label"}`` compatible with
      `load_calibration_dataset` and `logitgate calibrate --dataset`.
    - ``"dpo_jsonl"``: Direct Preference Optimization ``{"prompt", "chosen", "rejected", "rubric", "label"}``
      rows across forward (and optional reverse) binary label mappings.

    Returns the total number of JSONL rows written.
    """
    if format not in _VALID_EXPORT_FORMATS:
        raise ConfigurationError(
            f"Invalid export format={format!r}: must be one of {sorted(_VALID_EXPORT_FORMATS)}"
        )
    if not isinstance(include_symmetric_swaps, bool):
        raise ConfigurationError(
            f"include_symmetric_swaps must be a bool, got {include_symmetric_swaps!r}"
        )
    if not isinstance(dedupe, bool):
        raise ConfigurationError(f"dedupe must be a bool, got {dedupe!r}")
    if not isinstance(discrepancies_only, bool):
        raise ConfigurationError(
            f"discrepancies_only must be a bool, got {discrepancies_only!r}"
        )
    target = _validate_target_file_path(output_path, "Export output")

    records = _normalize_distillation_source(source)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for rec in records:
        if discrepancies_only and rec.discrepancy_type == "ambiguous_resolved":
            continue
        if dedupe:
            if rec.candidate_sha256 in seen:
                continue
            seen.add(rec.candidate_sha256)

        if format == "calibration_jsonl":
            rows.append(
                {
                    "rubric": rec.rubric,
                    "candidate": rec.candidate,
                    "label": rec.teacher_passed,
                    "discrepancy_type": rec.discrepancy_type,
                }
            )
            continue

        fwd_prompt = compile_binary_prompt(
            rubric=rec.rubric,
            candidate=rec.candidate,
            pass_token="A",
            fail_token="B",
            query_first=True,
        )
        fwd_chosen = "A" if rec.teacher_passed else "B"
        fwd_rejected = "B" if rec.teacher_passed else "A"

        if format == "sft_jsonl":
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": _BINARY_SYSTEM_PROMPT},
                        {"role": "user", "content": fwd_prompt},
                        {"role": "assistant", "content": fwd_chosen},
                    ],
                    "label": rec.teacher_passed,
                    "orientation": "forward",
                    "discrepancy_type": rec.discrepancy_type,
                }
            )
            if include_symmetric_swaps:
                rev_prompt = compile_binary_prompt(
                    rubric=rec.rubric,
                    candidate=rec.candidate,
                    pass_token="B",
                    fail_token="A",
                    query_first=True,
                )
                rev_chosen = "B" if rec.teacher_passed else "A"
                rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": _BINARY_SYSTEM_PROMPT},
                            {"role": "user", "content": rev_prompt},
                            {"role": "assistant", "content": rev_chosen},
                        ],
                        "label": rec.teacher_passed,
                        "orientation": "reverse",
                        "discrepancy_type": rec.discrepancy_type,
                    }
                )
        else:  # dpo_jsonl
            rows.append(
                {
                    "prompt": fwd_prompt,
                    "chosen": fwd_chosen,
                    "rejected": fwd_rejected,
                    "rubric": rec.rubric,
                    "label": rec.teacher_passed,
                    "orientation": "forward",
                    "discrepancy_type": rec.discrepancy_type,
                }
            )
            if include_symmetric_swaps:
                rev_prompt = compile_binary_prompt(
                    rubric=rec.rubric,
                    candidate=rec.candidate,
                    pass_token="B",
                    fail_token="A",
                    query_first=True,
                )
                rev_chosen = "B" if rec.teacher_passed else "A"
                rev_rejected = "A" if rec.teacher_passed else "B"
                rows.append(
                    {
                        "prompt": rev_prompt,
                        "chosen": rev_chosen,
                        "rejected": rev_rejected,
                        "rubric": rec.rubric,
                        "label": rec.teacher_passed,
                        "orientation": "reverse",
                        "discrepancy_type": rec.discrepancy_type,
                    }
                )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or target.is_symlink():
        raise ConfigurationError(f"Refusing to write export dataset through symlink: '{target}'")

    payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
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
        if target.is_symlink():
            raise ConfigurationError(f"Refusing to overwrite symlink export target: '{target}'")
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

    return len(rows)
