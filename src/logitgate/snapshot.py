from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    infer_backend_spec,
)
from logitgate.config import get_default_snapshot_dir
from logitgate.models import DecisionStatus, LayerTrace, Verdict

__all__ = [
    "VerdictSnapshot",
    "infer_backend_spec",
    "persist_failure_snapshot",
    "reset_snapshot_notices",
    "session_snapshot_paths",
]


def _compute_snapshot_id(timestamp: str, rubric: str, candidate: Any) -> str:
    raw = f"{timestamp}:{rubric}:{candidate!s}".encode()
    return hashlib.sha256(raw).hexdigest()[:8]


def _atomic_write_text(path: Path, content: str) -> None:
    """Writes text to a temporary file in the same directory and atomically replaces `path`."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    fd_owned = True
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd_owned = False
            handle.write(content)
            handle.flush()
        os.replace(tmp_path, path)
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


_snapshot_lock = threading.RLock()
_SESSION_SNAPSHOTS: list[Path] = []
_NOTIFIED_SNAPSHOT_SAVED: bool = False
_NOTIFIED_SNAPSHOT_HINT: bool = False


def reset_snapshot_notices() -> None:
    """Resets one-time console notices and session snapshot list (used by config reset and tests)."""
    global _NOTIFIED_SNAPSHOT_SAVED, _NOTIFIED_SNAPSHOT_HINT
    with _snapshot_lock:
        _NOTIFIED_SNAPSHOT_SAVED = False
        _NOTIFIED_SNAPSHOT_HINT = False
        _SESSION_SNAPSHOTS.clear()
    plugin_mod = sys.modules.get("logitgate.integrations.pytest_plugin")
    if plugin_mod is not None:
        check_paths = getattr(plugin_mod, "_CHECK_SNAPSHOT_PATHS", None)
        if isinstance(check_paths, list):
            check_paths.clear()


def _should_log_console_notice() -> bool:
    if os.environ.get("LOGITGATE_QUIET", "0").lower() in ("1", "true", "yes"):
        return False
    return "PYTEST_CURRENT_TEST" not in os.environ


def session_snapshot_paths() -> list[Path]:
    """Snapshots written by gates in this process, oldest first (feeds the pytest summary)."""
    with _snapshot_lock:
        return list(_SESSION_SNAPSHOTS)


def persist_failure_snapshot(
    verdict: Verdict,
    *,
    candidate: Any,
    rubric: str | None = None,
    gate_type: str | None = None,
    gate_config: dict[str, Any] | None = None,
    save_on_fail: bool | None = None,
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
        emit_hint = False
        with _snapshot_lock:
            if (
                save_on_fail is None
                and not _NOTIFIED_SNAPSHOT_HINT
                and verdict.triggered_layer != "JudgeGate"
                and _should_log_console_notice()
            ):
                _NOTIFIED_SNAPSHOT_HINT = True
                emit_hint = True
        if emit_hint:
            sys.stderr.write(
                f"[logitgate] Fast failure at {verdict.triggered_layer} without Layer-2 critique. "
                "Set LOGITGATE_SAVE_FAILURES=1 (or pass save_on_fail=True) to save a snapshot "
                "for 'logitgate explain'.\n"
            )
        return verdict
    if verdict.traces and "snapshot_path" in verdict.traces[-1].metadata:
        return verdict

    effective_cfg = dict(gate_config) if gate_config else {}
    if gate_type and "gate_type" not in effective_cfg:
        effective_cfg["gate_type"] = gate_type
    effective_rubric = rubric if rubric else f"{gate_type or verdict.triggered_layer or 'Gate'} evaluation"

    snapshot = VerdictSnapshot.from_verdict(
        verdict=verdict,
        rubric=effective_rubric,
        candidate=candidate,
        test_name=test_name,
        gate_config=effective_cfg,
    )
    saved_path = snapshot.save(directory=snapshot_dir)
    emit_saved = False
    with _snapshot_lock:
        if len(_SESSION_SNAPSHOTS) >= 1000:
            _SESSION_SNAPSHOTS.pop(0)
        _SESSION_SNAPSHOTS.append(saved_path)
        if verdict.traces:
            verdict.traces[-1].metadata["snapshot_path"] = str(saved_path)
        if not _NOTIFIED_SNAPSHOT_SAVED and _should_log_console_notice():
            _NOTIFIED_SNAPSHOT_SAVED = True
            emit_saved = True
    if emit_saved:
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
            data = dict(data)
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

    def _diagnostic_traces(self) -> list[LayerTrace]:
        """Orders traces so failing/ambiguous traces with lowest probability are inspected first."""
        non_passing = [t for t in self.verdict.traces if t.status != DecisionStatus.PASSED]
        non_passing.sort(
            key=lambda t: (
                0 if t.status == DecisionStatus.FAILED else 1,
                t.calibrated_prob if t.calibrated_prob is not None else 1.0,
            )
        )
        passing = [t for t in reversed(self.verdict.traces) if t.status == DecisionStatus.PASSED]
        return non_passing + passing

    @property
    def positional_bias_magnitude(self) -> float | None:
        for t in self._diagnostic_traces():
            if t.positional_bias_magnitude is not None:
                return t.positional_bias_magnitude
            if "positional_bias_magnitude" in t.metadata and t.metadata["positional_bias_magnitude"] is not None:
                return float(t.metadata["positional_bias_magnitude"])
        return None

    @property
    def log_odds_pass(self) -> float | None:
        for t in self._diagnostic_traces():
            if t.log_odds_pass is not None:
                return t.log_odds_pass
            if "log_odds_pass" in t.metadata and t.metadata["log_odds_pass"] is not None:
                return float(t.metadata["log_odds_pass"])
        return None

    @property
    def shannon_entropy(self) -> float | None:
        for t in self._diagnostic_traces():
            if t.shannon_entropy is not None:
                return t.shannon_entropy
            if "shannon_entropy" in t.metadata and t.metadata["shannon_entropy"] is not None:
                return float(t.metadata["shannon_entropy"])
        return None

    @property
    def loops_executed(self) -> int | None:
        for t in self._diagnostic_traces():
            if t.loops_executed is not None:
                return t.loops_executed
            if "loops_executed" in t.metadata and t.metadata["loops_executed"] is not None:
                return int(t.metadata["loops_executed"])
        return None

    @property
    def failure_code(self) -> str | None:
        for t in self._diagnostic_traces():
            if t.failure_code is not None:
                return t.failure_code
            if "failure_code" in t.metadata and t.metadata["failure_code"] is not None:
                return str(t.metadata["failure_code"])
        return None

    def save(self, directory: str | Path | None = None) -> Path:
        """
        Saves this snapshot as JSON to directory (defaulting to configure(snapshot_dir=...)
        or LOGITGATE_SNAPSHOT_DIR or .logitgate/failures), named
        fail_{timestamp_compact}_{test_slug}_{snapshot_id}.json, and updates latest.json in that directory.
        """
        target_dir = Path(directory) if directory is not None else Path(get_default_snapshot_dir())
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            ts_compact = dt.strftime("%Y%m%d_%H%M%S")
        except (ValueError, TypeError):
            ts_compact = re.sub(r"[^0-9]", "", self.timestamp)[:15] or "snapshot"

        if self.test_name:
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", self.test_name).strip("_")[:80].strip("_")
            test_slug = slug if slug else "test"
        else:
            test_slug = "test"

        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.snapshot_id)).strip("_")[:32] or "snap"
        filename = f"fail_{ts_compact}_{test_slug}_{safe_id}.json"
        file_path = target_dir / filename
        content = self.model_dump_json(indent=2, fallback=str)
        _atomic_write_text(file_path, content)

        latest_path = target_dir / "latest.json"
        _atomic_write_text(latest_path, content)

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
        try:
            data = json.loads(content)
        except RecursionError as e:
            raise ValueError(f"Malformed snapshot JSON (maximum nesting depth exceeded): {e}") from e
        return cls.model_validate(data)

    @staticmethod
    def _resolve_replay_backend_and_model(
        backend: str | LogitBackend | GenerativeBackend | None,
        model: str | None,
        gate_config: dict[str, Any],
        *,
        prefer_judge: bool = False,
    ) -> tuple[Any, str | None]:
        has_judge_cfg = prefer_judge and (
            gate_config.get("judge_backend") is not None or gate_config.get("judge_model") is not None
        )
        cfg_backend = (
            gate_config.get("judge_backend")
            if (prefer_judge and gate_config.get("judge_backend") is not None)
            else gate_config.get("backend")
        )
        cfg_model = gate_config.get("judge_model") if has_judge_cfg else gate_config.get("model")
        if backend is not None:
            effective_backend: Any = backend
        elif isinstance(cfg_backend, str) and (
            cfg_backend.startswith("custom://")
            or (model is not None and cfg_backend.startswith("stub://"))
        ):
            effective_backend = None
        else:
            effective_backend = cfg_backend
        effective_model = (
            model
            if model is not None
            else (cfg_model if backend is None else None)
        )
        return effective_backend, effective_model

    def replay(
        self,
        backend: str | LogitBackend | GenerativeBackend | None = None,
        model: str | None = None,
    ) -> Verdict:
        """
        Re-runs Layer 1 LogitGate, MultiRubricGate, ScoreGate, or Layer 2 JudgeGate using
        self.rubric, self.candidate, and self.gate_config against the specified (or default) backend.
        """
        from logitgate.gates.logit_gate import LogitGate, ScoreGate

        gate_kwargs = dict(self.gate_config)
        effective_backend, effective_model = self._resolve_replay_backend_and_model(
            backend, model, gate_kwargs
        )
        owns_backend = not isinstance(effective_backend, LogitBackend)

        if gate_kwargs.get("gate_type") == "JudgeGate":
            from logitgate.gates.judge_gate import JudgeGate

            judge_allowed = {
                "max_tokens",
                "think",
                "symmetric_debiasing",
                "logit_temperature",
                "name",
            }
            judge_init = {k: v for k, v in gate_kwargs.items() if k in judge_allowed}
            judge_owns = not isinstance(effective_backend, GenerativeBackend)
            judge_gate = JudgeGate(
                rubric=self.rubric,
                backend=effective_backend,
                model=effective_model,
                trigger="always",
                **judge_init,
            )
            try:
                return judge_gate.evaluate(self.candidate, save_on_fail=False)
            finally:
                if judge_owns:
                    judge_gate.backend.close()

        if gate_kwargs.get("gate_type") == "ScoreGate":
            score_allowed = {
                "min_score",
                "max_score",
                "threshold",
                "symmetric_debiasing",
                "query_first",
                "logit_temperature",
                "name",
            }
            score_init = {k: v for k, v in gate_kwargs.items() if k in score_allowed}
            score_gate = ScoreGate(
                rubric=self.rubric,
                backend=effective_backend,
                model=effective_model,
                **score_init,
            )
            try:
                return score_gate.evaluate(self.candidate, save_on_fail=False)
            finally:
                if owns_backend:
                    score_gate.backend.close()

        if gate_kwargs.get("gate_type") == "ChoiceGate":
            from logitgate.gates.logit_gate import ChoiceGate

            options = gate_kwargs.get("options")
            if not isinstance(options, dict) or len(options) < 2:
                raise ValueError("ChoiceGate snapshot is missing valid 'options' metadata for replay")
            choice_allowed = {
                "symmetric_debiasing",
                "permutation_mode",
                "query_first",
                "logit_temperature",
                "name",
            }
            choice_init = {k: v for k, v in gate_kwargs.items() if k in choice_allowed}
            choice_gate = ChoiceGate(
                question=gate_kwargs.get("question") or self.rubric,
                options=options,
                backend=effective_backend,
                model=effective_model,
                **choice_init,
            )
            try:
                return choice_gate.evaluate(self.candidate, save_on_fail=False)
            finally:
                if owns_backend:
                    choice_gate.backend.close()

        allowed_keys = {
            "pass_threshold",
            "fail_threshold",
            "symmetric_debiasing",
            "query_first",
            "logit_temperature",
            "max_positional_bias",
            "preset",
            "name",
        }
        init_kwargs = {k: v for k, v in gate_kwargs.items() if k in allowed_keys}

        rubrics = gate_kwargs.get("rubrics")
        if isinstance(rubrics, list) and rubrics:
            from logitgate.gates.logit_gate import MultiRubricGate

            multi_gate = MultiRubricGate(
                rubrics=rubrics,
                backend=effective_backend,
                model=effective_model,
                **init_kwargs,
            )
            try:
                return multi_gate.evaluate(self.candidate, save_on_fail=False)
            finally:
                if owns_backend:
                    multi_gate.backend.close()

        gate = LogitGate(
            rubric=self.rubric,
            backend=effective_backend,
            model=effective_model,
            **init_kwargs,
        )
        try:
            return gate.evaluate(self.candidate, save_on_fail=False)
        finally:
            if owns_backend:
                gate.backend.close()

    async def areplay(
        self,
        backend: str | LogitBackend | GenerativeBackend | None = None,
        model: str | None = None,
    ) -> Verdict:
        """Async version of `replay`."""
        from logitgate.gates.logit_gate import LogitGate, ScoreGate

        gate_kwargs = dict(self.gate_config)
        effective_backend, effective_model = self._resolve_replay_backend_and_model(
            backend, model, gate_kwargs
        )
        owns_backend = not isinstance(effective_backend, LogitBackend)

        if gate_kwargs.get("gate_type") == "JudgeGate":
            from logitgate.gates.judge_gate import JudgeGate

            judge_allowed = {
                "max_tokens",
                "think",
                "symmetric_debiasing",
                "logit_temperature",
                "name",
            }
            judge_init = {k: v for k, v in gate_kwargs.items() if k in judge_allowed}
            judge_owns = not isinstance(effective_backend, GenerativeBackend)
            judge_gate = JudgeGate(
                rubric=self.rubric,
                backend=effective_backend,
                model=effective_model,
                trigger="always",
                **judge_init,
            )
            try:
                return await judge_gate.aevaluate(self.candidate, save_on_fail=False)
            finally:
                if judge_owns:
                    await judge_gate.backend.aclose()

        if gate_kwargs.get("gate_type") == "ScoreGate":
            score_allowed = {
                "min_score",
                "max_score",
                "threshold",
                "symmetric_debiasing",
                "query_first",
                "logit_temperature",
                "name",
            }
            score_init = {k: v for k, v in gate_kwargs.items() if k in score_allowed}
            score_gate = ScoreGate(
                rubric=self.rubric,
                backend=effective_backend,
                model=effective_model,
                **score_init,
            )
            try:
                return await score_gate.aevaluate(self.candidate, save_on_fail=False)
            finally:
                if owns_backend:
                    await score_gate.backend.aclose()

        if gate_kwargs.get("gate_type") == "ChoiceGate":
            from logitgate.gates.logit_gate import ChoiceGate

            options = gate_kwargs.get("options")
            if not isinstance(options, dict) or len(options) < 2:
                raise ValueError("ChoiceGate snapshot is missing valid 'options' metadata for replay")
            choice_allowed = {
                "symmetric_debiasing",
                "permutation_mode",
                "query_first",
                "logit_temperature",
                "name",
            }
            choice_init = {k: v for k, v in gate_kwargs.items() if k in choice_allowed}
            choice_gate = ChoiceGate(
                question=gate_kwargs.get("question") or self.rubric,
                options=options,
                backend=effective_backend,
                model=effective_model,
                **choice_init,
            )
            try:
                return await choice_gate.aevaluate(self.candidate, save_on_fail=False)
            finally:
                if owns_backend:
                    await choice_gate.backend.aclose()

        allowed_keys = {
            "pass_threshold",
            "fail_threshold",
            "symmetric_debiasing",
            "query_first",
            "logit_temperature",
            "max_positional_bias",
            "preset",
            "name",
        }
        init_kwargs = {k: v for k, v in gate_kwargs.items() if k in allowed_keys}

        rubrics = gate_kwargs.get("rubrics")
        if isinstance(rubrics, list) and rubrics:
            from logitgate.gates.logit_gate import MultiRubricGate

            multi_gate = MultiRubricGate(
                rubrics=rubrics,
                backend=effective_backend,
                model=effective_model,
                **init_kwargs,
            )
            try:
                return await multi_gate.aevaluate(self.candidate, save_on_fail=False)
            finally:
                if owns_backend:
                    await multi_gate.backend.aclose()

        gate = LogitGate(
            rubric=self.rubric,
            backend=effective_backend,
            model=effective_model,
            **init_kwargs,
        )
        try:
            return await gate.aevaluate(self.candidate, save_on_fail=False)
        finally:
            if owns_backend:
                await gate.backend.aclose()

    def _merge_explained_verdict(self, judge: Any, judge_verdict: Verdict) -> Verdict:
        combined_traces = list(self.verdict.traces) + list(judge_verdict.traces)
        total_latency = self.verdict.total_latency_ms + judge_verdict.total_latency_ms
        total_tokens = self.verdict.total_tokens_generated + judge_verdict.total_tokens_generated

        final_passed = judge_verdict.passed
        final_status = judge_verdict.status
        final_reason = judge_verdict.reason
        final_score = self.verdict.score if self.verdict.score is not None else judge_verdict.score

        if (
            self.verdict.status == DecisionStatus.FAILED
            and isinstance(judge.backend, StubBackend)
            and judge.backend.diagnostic_text == "[VERDICT: PASS] All criteria met."
        ):
            final_passed = False
            final_status = DecisionStatus.FAILED
            final_score = self.verdict.score if self.verdict.score is not None else 0.0
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
            score=final_score,
            choice=self.verdict.choice,
            choice_probabilities=self.verdict.choice_probabilities,
            triggered_layer=judge_verdict.triggered_layer,
            total_latency_ms=total_latency,
            total_tokens_generated=total_tokens,
            reason=final_reason,
            traces=combined_traces,
        )

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

        effective_backend, effective_model = self._resolve_replay_backend_and_model(
            backend, model, self.gate_config, prefer_judge=True
        )
        owns_backend = not isinstance(effective_backend, GenerativeBackend)
        judge_temp = (
            self.gate_config.get("logit_temperature")
            if self.gate_config.get("gate_type") == "JudgeGate"
            else None
        )

        judge = JudgeGate(
            rubric=self.rubric,
            backend=effective_backend,
            model=effective_model,
            trigger="always",
            max_tokens=max_tokens,
            think=think,
            symmetric_debiasing=self.gate_config.get("symmetric_debiasing", True),
            logit_temperature=judge_temp,
        )
        try:
            judge_verdict = judge.evaluate(self.candidate, save_on_fail=False)
        finally:
            if owns_backend:
                judge.backend.close()
        return self._merge_explained_verdict(judge, judge_verdict)

    async def aexplain(
        self,
        backend: str | GenerativeBackend | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        think: bool | None = True,
    ) -> Verdict:
        """Async version of `explain`."""
        from logitgate.gates.judge_gate import JudgeGate

        effective_backend, effective_model = self._resolve_replay_backend_and_model(
            backend, model, self.gate_config, prefer_judge=True
        )
        owns_backend = not isinstance(effective_backend, GenerativeBackend)
        judge_temp = (
            self.gate_config.get("logit_temperature")
            if self.gate_config.get("gate_type") == "JudgeGate"
            else None
        )

        judge = JudgeGate(
            rubric=self.rubric,
            backend=effective_backend,
            model=effective_model,
            trigger="always",
            max_tokens=max_tokens,
            think=think,
            symmetric_debiasing=self.gate_config.get("symmetric_debiasing", True),
            logit_temperature=judge_temp,
        )
        try:
            judge_verdict = await judge.aevaluate(self.candidate, save_on_fail=False)
        finally:
            if owns_backend:
                await judge.backend.aclose()
        return self._merge_explained_verdict(judge, judge_verdict)
