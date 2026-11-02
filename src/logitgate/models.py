from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class DecisionStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


class LayerTrace(BaseModel):
    layer_name: str
    status: DecisionStatus
    latency_ms: float
    calibrated_prob: Optional[float] = None
    forward_prob: Optional[float] = None
    reverse_prob: Optional[float] = None
    forward_log_odds: Optional[float] = None
    reverse_log_odds: Optional[float] = None
    tokens_generated: int = 0
    diagnostic_report: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    passed: bool
    status: DecisionStatus
    probability: Optional[float] = None
    score: Optional[float] = None
    choice: Optional[str] = None
    choice_probabilities: Optional[dict[str, float]] = None
    triggered_layer: str
    total_latency_ms: float
    total_tokens_generated: int = 0
    reason: Optional[str] = None
    traces: list[LayerTrace] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed
