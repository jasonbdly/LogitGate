from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from logitgate.backends.base import GenerativeBackend, LogitBackend, StubBackend
from logitgate.config import get_default_snapshot_dir
from logitgate.models import DecisionStatus, Verdict


def _compute_snapshot_id(timestamp: str, rubric: str, candidate: Any) -> str:
    raw = f"{timestamp}:{rubric}:{candidate!s}".encode()
    return hashlib.sha256(raw).hexdigest()[:8]


_SESSION_SNAPSHOTS: list[Path] = []
_NOTIFIED_SNAPSHOT_SAVED: bool = False
_NOTIFIED_SNAPSHOT_HINT: bool = False


def reset_snapshot_notices() -> None:
    """Resets one-time console notices and session snapshot list (used by config reset and tests)."""
    global _NOTIFIED_SNAPSHOT_SAVED, _NOTIFIED_SNAPSHOT_HINT
    _NOTIFIED_SNAPSHOT_SAVED = False
    _NOTIFIED_SNAPSHOT_HINT = False
    _SESSION_SNAPSHOTS.clear()


def _should_log_console_notice() -> bool:
    if os.environ.get("LOGITGATE_QUIET", "0").lower() in ("1", "true", "yes"):
        return False
    return "PYTEST_CURRENT_TEST" not in os.environ


def session_snapshot_paths() -> list[Path]:
    """Snapshots written by gates in this process, oldest first (feeds the pytest summary)."""
    return list(_SESSION_SNAPSHOTS)


def persist_failure_snapshot(
    verdict: Verdict,
    *,
    candidate: Any,
    rubric: str,
    gate_config: dict[str, Any],
    save_on_fail: bool | None,
    test_name: str | None = None,
    snapshot_dir: str | Path | None = None,
) -> Verdict:
    """Writes a replayable snapshot for a failed verdict and records its path on the last trace.

    Shared by every gate that persists failures, so the save policy (explicit flag, then
    `configure()` / ``LOGITGATE_SAVE_FAILURES``) and the "save at most once" guard live in one
    place. The guard matters for cascades: inner gates run with ``save_on_fail=False`` and the
    cascade writes a single snapshot containing the full multi-layer trace.
    """
    global _NOTIFIED_SNAPSHOT_SAVED, _NOTIFIED_SNAPSHOT_HINT
    if verdict.passed:
        return verdict

    from logitgate.config import get_default_save_failures

    should_save = save_on_fail if save_on_fail is not None else get_default_save_failures()
    if not should_save:
        if (
            save_on_fail is None
            and not _NOTIFIED_SNAPSHOT_HINT
            and verdict.triggered_layer != "JudgeGate"
            and _should_log_console_notice()
        ):
            _NOTIFIED_SNAPSHOT_HINT = True
            sys.stderr.write(
                f"[logitgate] Fast failure at {verdict.triggered_layer} without Layer-2 critique. "
                "Set LOGITGATE_SAVE_FAILURES=1 (or pass save_on_fail=True) to save a snapshot "
                "for 'logitgate explain'.\n"
            )
        return verdict
    if verdict.traces and "snapshot_path" in verdict.traces[-1].metadata:
        return verdict

    snapshot = VerdictSnapshot.from_verdict(
        verdict=verdict,
        rubric=rubric,
        candidate=candidate,
        test_name=test_name,
        gate_config=gate_config,
    )
    saved_path = snapshot.save(directory=snapshot_dir)
    if len(_SESSION_SNAPSHOTS) >= 1000:
        _SESSION_SNAPSHOTS.pop(0)
    _SESSION_SNAPSHOTS.append(saved_path)
    if verdict.traces:
        verdict.traces[-1].metadata["snapshot_path"] = str(saved_path)
    if not _NOTIFIED_SNAPSHOT_SAVED and _should_log_console_notice():
        _NOTIFIED_SNAPSHOT_SAVED = True
        sys.stderr.write(
            f"[logitgate] Saved failure snapshot to {saved_path}. "
            f"Replay with Layer-2 critique: logitgate explain {saved_path}\n"
        )
    return verdict


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

    @classmethod
    def from_verdict(
        cls,
        verdict: Verdict,
        rubric: str,
        candidate: Any,
        test_name: str | None = None,
        gate_config: dict[str, Any] | None = None,
    ) -> VerdictSnapshot:
        """Constructs a VerdictSnapshot directly from a Verdict and evaluation context."""
        return cls(
            test_name=test_name,
            rubric=rubric,
            candidate=candidate,
            verdict=verdict,
            gate_config=gate_config or {},
        )

    @property
    def positional_bias_magnitude(self) -> float | None:
        for t in self.verdict.traces:
            if t.positional_bias_magnitude is not None:
                return t.positional_bias_magnitude
            if "positional_bias_magnitude" in t.metadata and t.metadata["positional_bias_magnitude"] is not None:
                return float(t.metadata["positional_bias_magnitude"])
        return None

    @property
    def log_odds_pass(self) -> float | None:
        for t in self.verdict.traces:
            if t.log_odds_pass is not None:
                return t.log_odds_pass
            if "log_odds_pass" in t.metadata and t.metadata["log_odds_pass"] is not None:
                return float(t.metadata["log_odds_pass"])
        return None

    @property
    def shannon_entropy(self) -> float | None:
        for t in self.verdict.traces:
            if t.shannon_entropy is not None:
                return t.shannon_entropy
            if "shannon_entropy" in t.metadata and t.metadata["shannon_entropy"] is not None:
                return float(t.metadata["shannon_entropy"])
        return None

    @property
    def loops_executed(self) -> int | None:
        for t in self.verdict.traces:
            if t.loops_executed is not None:
                return t.loops_executed
            if "loops_executed" in t.metadata and t.metadata["loops_executed"] is not None:
                return int(t.metadata["loops_executed"])
        return None

    @property
    def failure_code(self) -> str | None:
        for t in self.verdict.traces:
            if t.failure_code is not None:
                return t.failure_code
            if "failure_code" in t.metadata and t.metadata["failure_code"] is not None:
                return str(t.metadata["failure_code"])
        return None

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

        rubrics = gate_kwargs.get("rubrics")
        if isinstance(rubrics, list) and rubrics:
            from logitgate.gates.logit_gate import MultiRubricGate

            multi_gate = MultiRubricGate(
                rubrics=rubrics,
                backend=effective_backend,
                model=effective_model,
                **init_kwargs,
            )
            return multi_gate.evaluate(self.candidate)

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

        effective_backend = backend if backend is not None else self.gate_config.get("backend")
        effective_model = model if model is not None else self.gate_config.get("model")

        judge = JudgeGate(
            rubric=self.rubric,
            backend=effective_backend,
            model=effective_model,
            trigger="always",
            max_tokens=max_tokens,
            think=think,
        )
        judge_verdict = judge.evaluate(self.candidate)
        combined_traces = list(self.verdict.traces) + list(judge_verdict.traces)
        total_latency = self.verdict.total_latency_ms + judge_verdict.total_latency_ms
        total_tokens = self.verdict.total_tokens_generated + judge_verdict.total_tokens_generated

        final_passed = judge_verdict.passed
        final_status = judge_verdict.status
        final_reason = judge_verdict.reason

        if (
            self.verdict.status == DecisionStatus.FAILED
            and isinstance(judge.backend, StubBackend)
            and judge.backend.diagnostic_text == "[VERDICT: PASS] All criteria met."
        ):
            final_passed = False
            final_status = DecisionStatus.FAILED
            final_reason = (
                f"[VERDICT: FAIL] Candidate failed rubric '{self.rubric}' "
                f"(Layer 1 P={self.verdict.probability:.4f})."
                if self.verdict.probability is not None
                else (self.verdict.reason or f"[VERDICT: FAIL] Candidate failed rubric '{self.rubric}'.")
            )

        return Verdict(
            passed=final_passed,
            status=final_status,
            probability=self.verdict.probability if self.verdict.probability is not None else judge_verdict.probability,
            score=judge_verdict.score if judge_verdict.score is not None else self.verdict.score,
            choice=self.verdict.choice,
            choice_probabilities=self.verdict.choice_probabilities,
            triggered_layer=judge_verdict.triggered_layer,
            total_latency_ms=total_latency,
            total_tokens_generated=total_tokens,
            reason=final_reason,
            traces=combined_traces,
        )
