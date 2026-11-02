from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from logitgate.backends.base import GenerativeBackend, LogitBackend
from logitgate.config import get_default_snapshot_dir
from logitgate.models import Verdict


def _compute_snapshot_id(timestamp: str, rubric: str, candidate: Any) -> str:
    raw = f"{timestamp}:{rubric}:{candidate!s}".encode()
    return hashlib.sha256(raw).hexdigest()[:8]


class VerdictSnapshot(BaseModel):
    """
    Deterministic failure snapshot capturing candidate artifact, rubric, verdict,
    and gate telemetry for offline replay and Layer-2 explanation without re-running the system under test.
    """
    model_config = {"arbitrary_types_allowed": True}

    snapshot_id: str = ""
    timestamp: str = ""
    test_name: str | None = None
    rubric: str
    candidate: Any
    verdict: Verdict
    gate_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _populate_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ts = data.get("timestamp")
            if not ts:
                ts = datetime.now(timezone.utc).isoformat()
                data["timestamp"] = ts
            sid = data.get("snapshot_id")
            if not sid:
                rub = str(data.get("rubric", ""))
                cand = data.get("candidate", "")
                data["snapshot_id"] = _compute_snapshot_id(ts, rub, cand)
        return data

    def save(self, directory: str | Path | None = None) -> Path:
        """
        Saves this snapshot as JSON to directory (defaulting to LOGITGATE_SNAPSHOT_DIR or .logitgate/failures),
        named fail_{timestamp_compact}_{test_slug}_{snapshot_id}.json, and updates latest.json in that directory.
        """
        target_dir = Path(directory) if directory is not None else Path(
            os.environ.get("LOGITGATE_SNAPSHOT_DIR", get_default_snapshot_dir())
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            ts_compact = dt.strftime("%Y%m%d_%H%M%S")
        except (ValueError, TypeError):
            ts_compact = re.sub(r"[^0-9]", "", self.timestamp)[:15] or "snapshot"

        if self.test_name:
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", self.test_name).strip("_")
            test_slug = slug if slug else "test"
        else:
            test_slug = "test"

        filename = f"fail_{ts_compact}_{test_slug}_{self.snapshot_id}.json"
        file_path = target_dir / filename
        content = self.model_dump_json(indent=2)
        file_path.write_text(content, encoding="utf-8")

        latest_path = target_dir / "latest.json"
        latest_path.write_text(content, encoding="utf-8")

        return file_path

    @classmethod
    def load(cls, path: str | Path) -> VerdictSnapshot:
        """
        Loads a VerdictSnapshot from a file or directory.
        If path is a directory, loads latest.json (or the newest *.json file).
        If path is a file, loads and validates the snapshot JSON.
        """
        p = Path(path)
        if p.is_dir():
            latest = p / "latest.json"
            if latest.exists():
                target = latest
            else:
                candidates = [f for f in p.glob("fail_*.json")]
                if not candidates:
                    candidates = [f for f in p.glob("*.json")]
                if not candidates:
                    raise FileNotFoundError(f"No snapshot JSON files found in directory '{p}'")
                target = max(candidates, key=lambda f: f.stat().st_mtime)
        elif p.is_file():
            target = p
        else:
            raise FileNotFoundError(f"Snapshot path '{path}' does not exist")

        content = target.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.model_validate(data)

    def replay(
        self,
        backend: str | LogitBackend | None = None,
        model: str | None = None,
    ) -> Verdict:
        """
        Re-runs Layer 1 LogitGate using self.rubric, self.candidate, and self.gate_config
        against the specified (or default) backend.
        """
        from logitgate.gates.logit_gate import LogitGate

        gate_kwargs = dict(self.gate_config)
        allowed_keys = {
            "pass_threshold",
            "fail_threshold",
            "symmetric_debiasing",
            "query_first",
            "logit_temperature",
            "preset",
            "name",
        }
        init_kwargs = {k: v for k, v in gate_kwargs.items() if k in allowed_keys}
        effective_backend = backend if backend is not None else gate_kwargs.get("backend")
        effective_model = model if model is not None else gate_kwargs.get("model")

        gate = LogitGate(
            rubric=self.rubric,
            backend=effective_backend,
            model=effective_model,
            **init_kwargs,
        )
        return gate.evaluate(self.candidate)

    def explain(
        self,
        backend: str | GenerativeBackend | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        think: bool | None = True,
    ) -> Verdict:
        """
        Runs JudgeGate on the exact frozen self.candidate from the snapshot with trigger='always',
        attaches the original snapshot traces + the new JudgeGate trace, and returns the enriched
        Verdict with the full diagnostic reason.
        """
        from logitgate.gates.judge_gate import JudgeGate

        judge = JudgeGate(
            rubric=self.rubric,
            backend=backend,
            model=model,
            trigger="always",
            max_tokens=max_tokens,
            think=think,
        )
        judge_verdict = judge.evaluate(self.candidate)
        combined_traces = list(self.verdict.traces) + list(judge_verdict.traces)
        total_latency = self.verdict.total_latency_ms + judge_verdict.total_latency_ms
        total_tokens = self.verdict.total_tokens_generated + judge_verdict.total_tokens_generated

        return Verdict(
            passed=judge_verdict.passed,
            status=judge_verdict.status,
            probability=self.verdict.probability if self.verdict.probability is not None else judge_verdict.probability,
            score=judge_verdict.score if judge_verdict.score is not None else self.verdict.score,
            choice=self.verdict.choice,
            choice_probabilities=self.verdict.choice_probabilities,
            triggered_layer=judge_verdict.triggered_layer,
            total_latency_ms=total_latency,
            total_tokens_generated=total_tokens,
            reason=judge_verdict.reason,
            traces=combined_traces,
        )
