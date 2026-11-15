from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from logitgate.models import DecisionStatus, LayerTrace

PrivacyMode = Literal["full", "hash_only", "omit_candidate", "redact_pii"]
DiscrepancyType = Literal[
    "false_accept_caught",
    "false_reject_overturned",
    "polarity_flip",
    "ambiguous_resolved",
]
ErrorPolicy = Literal["warn", "raise", "ignore"]

VALID_PRIVACY_MODES: frozenset[str] = frozenset(
    {"full", "hash_only", "omit_candidate", "redact_pii"}
)
VALID_ERROR_POLICIES: frozenset[str] = frozenset({"warn", "raise", "ignore"})

# Linear-time, zero-backtracking PII/secret redaction patterns.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}\b"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,512}")
_API_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,256}|gh[pousr]_[A-Za-z0-9]{20,255}"
    r"|glpat-[A-Za-z0-9_-]{16,128}|AKIA[0-9A-Z]{16})\b"
)
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_PAN_CANDIDATE_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


def _luhn_valid(digits: str) -> bool:
    if len(digits) < 13 or len(digits) > 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for idx, ch in enumerate(digits):
        d = ord(ch) - 48
        if idx % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _replace_pan(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = "".join(ch for ch in raw if "0" <= ch <= "9")
    return "[REDACTED_PAN]" if _luhn_valid(digits) else raw


def truncate_text(text: str, max_chars: int | None) -> str:
    """Truncates ``text`` to ``max_chars`` characters in O(1) slice time."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def redact_text(text: str, *, max_chars: int | None = 16384) -> str:
    """Linear-time redaction of emails, Bearer/API tokens, US SSNs, and Luhn-valid PANs."""
    bounded = truncate_text(text, max_chars)
    scrubbed = _BEARER_RE.sub("Bearer [REDACTED_TOKEN]", bounded)
    scrubbed = _API_TOKEN_RE.sub("[REDACTED_SECRET]", scrubbed)
    scrubbed = _EMAIL_RE.sub("[REDACTED_EMAIL]", scrubbed)
    scrubbed = _SSN_RE.sub("[REDACTED_SSN]", scrubbed)
    scrubbed = _PAN_CANDIDATE_RE.sub(_replace_pan, scrubbed)
    return scrubbed


def hash_candidate(rendered: str) -> str:
    """Returns the lowercase hex SHA-256 digest of a rendered candidate string."""
    return hashlib.sha256(rendered.encode("utf-8", errors="replace")).hexdigest()


def apply_privacy(
    rendered_candidate: str,
    *,
    privacy: PrivacyMode = "full",
    max_candidate_chars: int | None = 16384,
) -> tuple[str | None, str, int]:
    """Applies ``privacy`` and ``max_candidate_chars`` to a rendered candidate string.

    Returns ``(stored_candidate, candidate_sha256, candidate_length)``.
    """
    cand_len = len(rendered_candidate)
    digest = hash_candidate(rendered_candidate)
    if privacy in ("hash_only", "omit_candidate"):
        return None, digest, cand_len
    if privacy == "redact_pii":
        return redact_text(rendered_candidate, max_chars=max_candidate_chars), digest, cand_len
    return truncate_text(rendered_candidate, max_candidate_chars), digest, cand_len


class TelemetryRecord(BaseModel):
    """Structured production observability record for a single gate or cascade verdict."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    event_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    gate_name: str
    gate_type: str
    model: str | None = None
    backend: str | None = None
    teacher_model: str | None = None
    teacher_passed: bool | None = None
    rubric: str | None = None
    candidate: str | None = None
    candidate_sha256: str
    candidate_length: int
    passed: bool
    status: DecisionStatus
    probability: float | None = None
    score: float | None = None
    choice: str | None = None
    choice_probabilities: dict[str, float] | None = None
    triggered_layer: str | None = None
    escalated: bool = False
    fast_pathed: bool = False
    total_latency_ms: float
    total_tokens_generated: int = 0
    reason: str | None = None
    traces: list[LayerTrace] = Field(default_factory=list)
    tags: dict[str, Any] = Field(default_factory=dict)

    def with_privacy(
        self,
        *,
        privacy: PrivacyMode,
        max_candidate_chars: int | None,
    ) -> TelemetryRecord:
        """Returns a privacy-filtered copy suitable for sink persistence."""
        if privacy == "full" and (
            self.candidate is None
            or max_candidate_chars is None
            or len(self.candidate) <= max_candidate_chars
        ):
            return self
        if privacy in ("hash_only", "omit_candidate"):
            scrubbed_traces = [
                tr.model_copy(update={"diagnostic_report": None})
                if tr.diagnostic_report is not None and privacy == "omit_candidate"
                else tr
                for tr in self.traces
            ]
            has_generative_critique = self.gate_type == "JudgeGate" or any(
                tr.diagnostic_report is not None or tr.tokens_generated > 0 for tr in self.traces
            )
            scrubbed_reason = (
                None if (privacy == "omit_candidate" and has_generative_critique) else self.reason
            )
            return self.model_copy(
                update={
                    "candidate": None,
                    "reason": scrubbed_reason,
                    "traces": scrubbed_traces,
                }
            )
        if privacy == "redact_pii":
            red_cand = (
                redact_text(self.candidate, max_chars=max_candidate_chars)
                if self.candidate is not None
                else None
            )
            red_reason = (
                redact_text(self.reason, max_chars=max_candidate_chars)
                if self.reason is not None
                else None
            )
            red_traces = [
                tr.model_copy(
                    update={
                        "diagnostic_report": redact_text(
                            tr.diagnostic_report, max_chars=max_candidate_chars
                        )
                    }
                )
                if tr.diagnostic_report is not None
                else tr
                for tr in self.traces
            ]
            return self.model_copy(
                update={
                    "candidate": red_cand,
                    "reason": red_reason,
                    "traces": red_traces,
                }
            )
        return self.model_copy(
            update={
                "candidate": (
                    truncate_text(self.candidate, max_candidate_chars)
                    if self.candidate is not None
                    else None
                )
            }
        )


class DistillationRecord(BaseModel):
    """Self-distilling student/teacher escalation record.

    Top-level ``rubric``, ``candidate``, and ``label`` match ``load_calibration_dataset()``
    natively, while ``student_forward_log_odds`` and ``student_reverse_log_odds`` allow
    zero-LLM-call profile recalibration via ``recalibrate_from_telemetry()``.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    event_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    rubric: str
    candidate: str
    label: bool
    discrepancy_type: DiscrepancyType

    student_layer: str
    student_model: str | None = None
    student_status: DecisionStatus
    student_prob: float | None = None
    student_forward_log_odds: float | None = None
    student_reverse_log_odds: float | None = None
    student_log_odds_pass: float | None = None
    student_positional_bias: float | None = None

    teacher_layer: str
    teacher_model: str | None = None
    teacher_passed: bool
    teacher_prob: float | None = None
    teacher_log_odds_pass: float | None = None
    teacher_reason: str | None = None

    cascade_passed: bool
    candidate_sha256: str
    tags: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TelemetrySummary:
    """Aggregated operational statistics across a stream of `TelemetryRecord` events."""

    total_records: int
    passed_count: int
    failed_count: int
    ambiguous_count: int
    fast_path_count: int
    fast_path_rate: float
    escalated_count: int
    escalation_rate: float
    discrepancy_count: int
    discrepancy_rate: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    total_tokens_generated: int
    by_gate_type: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "ambiguous_count": self.ambiguous_count,
            "fast_path_count": self.fast_path_count,
            "fast_path_rate": self.fast_path_rate,
            "escalated_count": self.escalated_count,
            "escalation_rate": self.escalation_rate,
            "discrepancy_count": self.discrepancy_count,
            "discrepancy_rate": self.discrepancy_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "total_tokens_generated": self.total_tokens_generated,
            "by_gate_type": dict(self.by_gate_type),
        }
