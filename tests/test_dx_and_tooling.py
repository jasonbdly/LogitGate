"""Developer-experience guarantees: actionable errors, honest defaults, tooling, and data layout."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from logitgate import (
    ASTGate,
    BackendError,
    DecisionStatus,
    JudgeGate,
    LayerTrace,
    LogitGate,
    OllamaBackend,
    OpenAICompatBackend,
    StubBackend,
    Verdict,
    VerdictSnapshot,
)
from logitgate.cli import main
from logitgate.exceptions import (
    BackendTimeoutError,
    BackendUnavailableError,
    StubBackendWarning,
)
from logitgate.gates.ast_gate import _detect_dag_cycle
from logitgate.gates.judge_gate import (
    _extract_keyword_verdict,
    _extract_structured_verdict,
)

# ----------------------------------------------------------------------------------------------
# ASTGate: Kahn's algorithm on a deque
# ----------------------------------------------------------------------------------------------


def test_dag_accepts_diamond_and_disconnected_components() -> None:
    tasks: list[dict[str, Any]] = [
        {"id": "fetch"},
        {"id": "parse", "deps": ["fetch"]},
        {"id": "lint", "deps": ["fetch"]},
        {"id": "report", "dependencies": ["parse", "lint"]},
        {"task_id": "standalone"},
    ]
    assert _detect_dag_cycle(tasks) is None


def test_dag_cycle_report_is_deterministic_and_input_ordered() -> None:
    tasks: list[dict[str, Any]] = [
        {"id": "z_root"},
        {"id": "b", "deps": ["c", "z_root"]},
        {"id": "c", "deps": ["b"]},
        {"id": "a", "deps": ["c"]},
    ]
    assert _detect_dag_cycle(tasks) == "Cycle detected in task dependencies among: ['b', 'c', 'a']"


def test_dag_self_loop_and_missing_dependency() -> None:
    assert "Cycle detected" in (_detect_dag_cycle([{"id": "x", "deps": ["x"]}]) or "")
    assert "non-existent task 'ghost'" in (_detect_dag_cycle([{"id": "x", "deps": ["ghost"]}]) or "")


def test_dag_wide_fan_out_is_linear_time() -> None:
    # A 100k-leaf fan-out puts every node in the ready queue at once: the case where
    # list.pop(0) degrades Kahn's algorithm to O(V^2) (minutes) and a deque stays O(V + E).
    n = 100_000
    tasks = [{"id": "root"}] + [{"id": f"t{i}", "deps": ["root"]} for i in range(n)]
    t0 = time.perf_counter()
    assert _detect_dag_cycle(tasks) is None
    assert time.perf_counter() - t0 < 5.0


def test_ast_gate_patterns_compile_once_and_keep_literal_fallback() -> None:
    gate = ASTGate(forbidden_patterns=["eval(", r"rm\s+-rf"], required_sections=["## Usage (v2)"])
    assert gate._forbidden_res[0][1] is None  # invalid regex -> literal substring
    assert gate._forbidden_res[1][1] is not None
    assert gate.evaluate("x = eval(1)\n## Usage (v2)").status == DecisionStatus.FAILED
    assert gate.evaluate("rm   -rf /\n## Usage (v2)").traces[0].failure_code == "FORBIDDEN_PATTERN"
    assert gate.evaluate("print('ok')\n## Usage (v2)").passed is True


# ----------------------------------------------------------------------------------------------
# JudgeGate verdict extraction
# ----------------------------------------------------------------------------------------------


def test_structured_verdict_handles_nested_json_and_string_booleans() -> None:
    assert _extract_structured_verdict('Result: {"verdict": "pass", "details": {"checks": 3}}') == (True, DecisionStatus.PASSED, 1.0)
    assert _extract_structured_verdict('{"pass": "false", "score": 0.2}') == (False, DecisionStatus.FAILED, 0.2)
    assert _extract_structured_verdict("reasoning... [VERDICT: PASS] ... final [VERDICT: FAIL]")[0] is False  # type: ignore[index]


def test_quoted_candidate_json_is_not_mistaken_for_a_verdict() -> None:
    # Previously any JSON object in the critique produced a decisive FAIL.
    assert _extract_structured_verdict('The tool returned {"status": "ok", "code": 200} as required.') is None


def test_keyword_heuristic_returns_none_without_signal() -> None:
    assert _extract_keyword_verdict("The implementation handles every edge case.") is None
    assert _extract_keyword_verdict("Did not fail any check.\nApproved.") == (True, DecisionStatus.PASSED, 1.0)


def test_logit_probe_outranks_keyword_heuristics_and_is_cached() -> None:
    # "passes" here describes data flow, not the verdict; the calibrated probe says FAIL.
    probe_says_fail = [[{"token": "A", "logprob": -4.0}, {"token": "B", "logprob": -0.02}]]
    judge = JudgeGate(backend=StubBackend(logprob_responses=probe_says_fail))
    critique = "The function passes the raw user string straight into a SQL query"
    assert judge._parse_verdict(critique)[0] is False
    probe = judge._probe_gate
    judge._parse_verdict(critique)
    assert judge._probe_gate is probe


async def test_judge_sync_async_parse_parity_on_corpus() -> None:
    judge = JudgeGate(backend=StubBackend())
    corpus = [
        "**Verdict:** PASS",
        "- ok\n- ok\n[VERDICT: FAIL]",
        '{"passed": true}',
        "No failures were found. Approved.",
        "The code failed on empty input.",
        "Nothing conclusive here.",
    ]
    for text in corpus:
        assert judge._parse_verdict(text) == await judge._aparse_verdict(text), text


# ----------------------------------------------------------------------------------------------
# Actionable backend errors
# ----------------------------------------------------------------------------------------------


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_connection_refused_is_actionable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    gate = LogitGate("Is this factual?", backend=OllamaBackend(client=_client(refuse)))
    with pytest.raises(BackendUnavailableError) as exc_info:
        gate.evaluate("Paris is in France")

    message = str(exc_info.value)
    assert message.startswith("Cannot connect to Ollama at http://localhost:11434. Is ollama running?")
    assert "backend='stub://'" in message
    assert isinstance(exc_info.value, ConnectionError)  # generic retry logic still works
    assert isinstance(exc_info.value, BackendError)
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


def test_ollama_missing_model_suggests_pull() -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "qwen2.5:7b" not found, try pulling it first'})

    backend = OllamaBackend(model="qwen2.5:7b", client=_client(not_found))
    with pytest.raises(BackendError, match=r"ollama pull qwen2\.5:7b"):
        backend.query_top_logprobs("prompt")


def test_openai_auth_failure_points_at_api_key() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    backend = OpenAICompatBackend(client=_client(unauthorized))
    with pytest.raises(BackendError, match="Incorrect API key provided.*OPENAI_API_KEY"):
        backend.generate("prompt")


async def test_async_timeout_is_translated() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    backend = OllamaBackend(timeout=5.0, async_client=httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    with pytest.raises(BackendTimeoutError, match="did not respond within 5s"):
        await backend.agenerate("prompt")


def test_cli_reports_unreachable_backend_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--rubric", "r", "--candidate", "c", "--backend", "ollama://127.0.0.1:9/m"])
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: Cannot connect to Ollama at http://127.0.0.1:9")
    assert "Traceback" not in err


# ----------------------------------------------------------------------------------------------
# Implicit stub warning
# ----------------------------------------------------------------------------------------------


@pytest.fixture
def clean_config(monkeypatch: pytest.MonkeyPatch) -> Any:
    from logitgate.config import reset_config

    monkeypatch.delenv("LOGITGATE_BACKEND", raising=False)
    reset_config()
    yield
    reset_config()


def test_implicit_stub_warns_once(clean_config: Any) -> None:
    with pytest.warns(StubBackendWarning, match="fixed P\\(pass\\)=0.95"):
        LogitGate("Is this factual?")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", StubBackendWarning)
        LogitGate("Is this factual?")  # second construction stays quiet


@pytest.mark.parametrize("how", ["argument", "env", "configure"])
def test_explicit_stub_does_not_warn(clean_config: Any, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    import warnings

    from logitgate import configure

    with warnings.catch_warnings():
        warnings.simplefilter("error", StubBackendWarning)
        if how == "argument":
            LogitGate("r", backend="stub://default")
        elif how == "env":
            monkeypatch.setenv("LOGITGATE_BACKEND", "stub://default")
            LogitGate("r")
        else:
            configure(backend="stub://default")
            LogitGate("r")


# ----------------------------------------------------------------------------------------------
# `logitgate benchmark`
# ----------------------------------------------------------------------------------------------


class _SlowJudgeBackend(StubBackend):
    """Logit reads are instant; generation costs 4 ms. Enough to prove the harness measures."""

    def generate(self, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> tuple[str, int]:
        time.sleep(0.004)
        return super().generate(prompt, max_tokens=max_tokens, **kwargs)


def test_benchmark_counts_calls_tokens_and_fast_path() -> None:
    from logitgate.benchmark import BUILTIN_CASES, format_report, run_benchmark

    backend = _SlowJudgeBackend(default_prob=0.97, diagnostic_text="[VERDICT: PASS]", tokens_generated=120)
    report = run_benchmark(BUILTIN_CASES[:6], backend=backend, warmup=0)
    rows = {mode: stats.summary() for mode, stats in report.modes.items()}

    assert rows["judge"]["llm_calls_per_case"] == 1.0
    assert rows["judge"]["generated_tokens_per_case"] == 120.0
    assert rows["logit"]["llm_calls_per_case"] == 2.0  # symmetric debiasing = 2 forward passes
    assert rows["cascade"]["fast_path_rate"] == 1.0
    assert rows["cascade"]["generated_tokens_per_case"] == 0.0
    assert rows["cascade"]["mean_ms"] < rows["judge"]["mean_ms"]
    assert "cascade" in format_report(report)


def test_benchmark_cascade_escalates_ambiguous_cases() -> None:
    from logitgate.benchmark import BUILTIN_CASES, run_benchmark

    backend = StubBackend(default_prob=0.5, diagnostic_text="[VERDICT: FAIL]", tokens_generated=80)
    report = run_benchmark(BUILTIN_CASES[:4], backend=backend, modes=["cascade"], warmup=0)
    row = report.modes["cascade"].summary()
    assert row["fast_path_rate"] == 0.0
    assert row["llm_calls_per_case"] == 3.0
    assert row["generated_tokens_per_case"] == 80.0


def test_cli_benchmark_json_is_reproducible_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--backend", "stub://", "--limit", "4", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert set(data["modes"]) == {"judge", "logit", "cascade"}
    meta = data["metadata"]
    assert meta["is_stub"] is True
    assert meta["dataset"]["cases"] == 4
    assert re.fullmatch(r"[0-9a-f]{12}", meta["dataset"]["sha256_12"])


def test_cli_benchmark_rejects_unknown_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--backend", "stub://", "--modes", "judge,vibes"]) == 1
    assert "vibes" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------
# `logitgate explain`
# ----------------------------------------------------------------------------------------------


def _ambiguous_snapshot(tmp_path: Path, backend: str = "ollama://127.0.0.1:9/m") -> Path:
    trace = LayerTrace(
        layer_name="LogitGate",
        status=DecisionStatus.AMBIGUOUS,
        latency_ms=48.0,
        calibrated_prob=0.6373,
        forward_prob=0.81,
        reverse_prob=0.42,
        positional_bias_magnitude=0.8864,
        log_odds_pass=0.5636,
        shannon_entropy=0.9449,
    )
    verdict = Verdict(
        passed=False,
        status=DecisionStatus.AMBIGUOUS,
        probability=0.6373,
        triggered_layer="LogitGate",
        total_latency_ms=48.0,
        traces=[trace],
    )
    snap = VerdictSnapshot(
        test_name="test_privacy",
        rubric="Refuses to disclose personal data",
        candidate="I can't share her email, but she is a customer.",
        verdict=verdict,
        gate_config={"backend": backend, "pass_threshold": 0.85, "fail_threshold": 0.2},
    )
    return snap.save(directory=tmp_path)


def test_explain_offline_renders_margins_without_calling_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _ambiguous_snapshot(tmp_path)  # backend points at a dead port: must not be contacted
    assert main(["explain", str(path), "--offline", "--color", "never"]) == 0
    out = capsys.readouterr().out
    assert "Decision Margin: +0.564 vs pass +1.735 (P=0.85) / fail -1.386 (P=0.2)" in out
    assert "1.17 short of passing" in out
    assert "near coin-flip" in out
    assert "moderate label-order prior" in out
    assert "\x1b[" not in out


def test_explain_replay_failure_still_shows_diagnostics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _ambiguous_snapshot(tmp_path)
    assert main(["explain", str(path)]) == 2
    captured = capsys.readouterr()
    assert "Decision Margin" in captured.out
    assert "Cannot connect to Ollama" in captured.err
    assert "--offline" in captured.err


def test_explain_color_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    path = _ambiguous_snapshot(tmp_path)
    assert main(["explain", str(path), "--offline", "--color", "always"]) == 0
    assert "\x1b[" in capsys.readouterr().out
    monkeypatch.setenv("NO_COLOR", "1")
    assert main(["explain", str(path), "--offline"]) == 0
    assert "\x1b[" not in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------
# Data layout: hot-path records are frozen + slotted; public results stay Pydantic
# ----------------------------------------------------------------------------------------------


def test_internal_records_are_frozen_slotted_dataclasses() -> None:
    from logitgate import (
        ConformalCascadeThresholds,
        DebiasedLogitResult,
        RecurrentLoopTrace,
        SurfaceTokenMap,
        VRAMWatermarkStatus,
    )

    t_map = SurfaceTokenMap(positive_token_ids=[3, 1], negative_token_ids=(2,))  # type: ignore[arg-type]
    c_thresh = ConformalCascadeThresholds(0.2, 0.8, 0.05, 0.1, 50)
    instances: list[Any] = [
        t_map,
        DebiasedLogitResult(0.9, 0.4, 2.2, 0.1, 0.9, 0.88),
        VRAMWatermarkStatus(1.0, 2.0, 3.0, 4.0, False),
        c_thresh,
        RecurrentLoopTrace(3, (0.9, 0.5, 0.1), True, None),
    ]
    for obj in instances:
        assert dataclasses.is_dataclass(obj)
        assert not hasattr(obj, "__dict__"), type(obj).__name__
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, dataclasses.fields(obj)[0].name, 0)
    assert t_map.positive_token_ids == (1, 3)  # __post_init__ normalization survives slots
    assert dataclasses.asdict(c_thresh)["tau_accept"] == 0.8


# ----------------------------------------------------------------------------------------------
# Docs that execute
# ----------------------------------------------------------------------------------------------


def test_readme_python_examples_run(clean_config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ```python block in the README runs, in order, against the offline stub."""
    monkeypatch.setenv("LOGITGATE_BACKEND", "stub://default")
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert blocks, "README has no python examples"
    namespace: dict[str, Any] = {}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"README.md[python block {i}]", "exec"), namespace)  # noqa: S102 - trusted repo docs
        except Exception as exc:  # noqa: BLE001 - surface which block broke
            pytest.fail(f"README python block {i} failed: {type(exc).__name__}: {exc}\n---\n{block}")


def test_sensible_out_of_box_defaults_and_failure_replay_notices(
    clean_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from logitgate import ChoiceGate, JudgeGate, LogitGate, ScoreGate, check
    from logitgate.backends.ollama import OllamaBackend

    # 1. OllamaBackend defaults to qwen2.5:7b and omits num_ctx unless configured
    ollama = OllamaBackend()
    assert ollama.model == "qwen2.5:7b"
    assert ollama.num_ctx is None
    assert "num_ctx" not in ollama.build_query_payload("hi")["options"]
    assert "num_ctx" not in ollama.build_generate_payload("hi", None, {})["options"]
    assert OllamaBackend(num_ctx=8192).build_query_payload("hi")["options"]["num_ctx"] == 8192

    # 2. ScoreGate defaults to symmetric_debiasing=True, and ChoiceGate/ScoreGate use KNOWN_MODEL_TEMPERATURES
    score_gate = ScoreGate("Rate quality", model="qwen2.5:7b", backend="stub://default")
    choice_gate = ChoiceGate("Pick one", {"A": "x", "B": "y"}, model="qwen2.5:7b", backend="stub://default")
    assert score_gate.symmetric_debiasing is True
    assert score_gate.logit_temperature == pytest.approx(1.05)
    assert choice_gate.logit_temperature == pytest.approx(1.05)

    # 3. JudgeGate defaults to trigger='on_ambiguous_only'
    judge = JudgeGate(backend="stub://default")
    assert judge.trigger == "on_ambiguous_only"

    # 4. @check failure without --logitgate-save-failures includes tip on how to enable snapshots
    @check("Must be polite", backend="stub://0.05")
    def bad_reply() -> str:
        return "Go away"

    with pytest.raises(AssertionError, match="--logitgate-save-failures"):
        bad_reply()

    # 5. Outside pytest, a fast Layer-1 failure emits a one-time console notice to stderr
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    gate = LogitGate("Must pass", backend="stub://0.05")
    gate.evaluate("bad candidate")
    err1 = capsys.readouterr().err
    assert "logitgate explain" in err1
    assert "LOGITGATE_SAVE_FAILURES=1" in err1

    # Second failure does not spam stderr again
    gate.evaluate("bad candidate 2")
    assert capsys.readouterr().err == ""

    # And when save_on_fail=True, it logs the saved snapshot path once
    gate.evaluate("bad candidate 3", save_on_fail=True, snapshot_dir=str(tmp_path))
    err_saved = capsys.readouterr().err
    assert "Saved failure snapshot to" in err_saved
    assert "logitgate explain" in err_saved


def test_staff_audit_fixes_uri_resolution_and_roundtrip(clean_config: Any) -> None:
    from logitgate import ChoiceGate, ConfigurationError, LogitGate, ScoreGate
    from logitgate.backends.base import infer_backend_spec, resolve_logit_backend
    from logitgate.backends.llamacpp import LlamaCppBackend
    from logitgate.backends.openai_compat import OpenAICompatBackend
    from logitgate.backends.recurrent import RecurrentPyTorchBackend

    # 1. openai://gpt-4o-mini resolves to https://api.openai.com/v1 with model='gpt-4o-mini'
    b_oai = resolve_logit_backend("openai://gpt-4o-mini")
    assert isinstance(b_oai, OpenAICompatBackend)
    assert b_oai.base_url == "https://api.openai.com/v1"
    assert b_oai.model == "gpt-4o-mini"
    assert infer_backend_spec(b_oai) == "openai://gpt-4o-mini"

    # 2. vllm://localhost:8000/v1/Qwen/Qwen2.5-7B-Instruct separates base_url and model
    b_vllm = resolve_logit_backend("vllm://localhost:8000/v1/Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(b_vllm, OpenAICompatBackend)
    assert b_vllm.base_url == "http://localhost:8000/v1"
    assert b_vllm.model == "Qwen/Qwen2.5-7B-Instruct"
    assert infer_backend_spec(b_vllm) == "vllm://localhost:8000/v1/Qwen/Qwen2.5-7B-Instruct"

    # 3. vllm://localhost:8000 (without /v1) normalizes base_url to /v1
    b_vllm_bare = resolve_logit_backend("vllm://localhost:8000")
    assert isinstance(b_vllm_bare, OpenAICompatBackend)
    assert b_vllm_bare.base_url == "http://localhost:8000/v1"

    # 4. RecurrentPyTorchBackend infer_backend_spec and .model property
    b_rec = RecurrentPyTorchBackend(model_name="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
    assert b_rec.model == "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
    assert infer_backend_spec(b_rec) == "recurrent://nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

    # 5. In-process LlamaCppBackend infer_backend_spec
    b_llama_inproc = LlamaCppBackend(llm=object(), model="qwen.gguf")
    assert infer_backend_spec(b_llama_inproc) == "llamacpp:///qwen.gguf"

    # 6. URI-embedded model is picked up by LogitGate/ChoiceGate/ScoreGate for temperature lookup
    g_uri = LogitGate("Check", backend="ollama://localhost:11434/gemma4:e4b")
    assert g_uri.model == "gemma4:e4b"
    assert g_uri.logit_temperature == pytest.approx(0.95)

    # 7. ScoreGate works out-of-the-box on default StubBackend and rejects multi-token bounds
    s_default = ScoreGate("Rate quality", backend="stub://default")
    v_score = s_default.evaluate("Great code")
    assert v_score.passed is True
    assert v_score.score is not None and v_score.score > 4.5

    with pytest.raises(ConfigurationError, match="single-digit integer bounds"):
        ScoreGate("Rate quality", min_score=1, max_score=10)

    # 8. ChoiceGate rejects case-insensitive duplicate keys and strips key whitespace
    with pytest.raises(ConfigurationError, match="case-insensitively unique"):
        ChoiceGate("Pick", {"a": "opt 1", "A": "opt 2"})
    cg = ChoiceGate("Pick", {" A ": "opt 1", "B": "opt 2"}, backend="stub://default")
    assert list(cg.options.keys()) == ["A", "B"]


def test_staff_audit_fixes_numerical_ast_and_async_loop(tmp_path: Path) -> None:
    import asyncio

    from logitgate import (
        ASTGate,
        BackendError,
        ChoiceGate,
        JudgeGate,
        LogitGate,
        ScoreGate,
    )
    from logitgate.backends.ollama import OllamaBackend
    from logitgate.debiasing.permutation import get_model_temperature
    from logitgate.debiasing.softmax import extract_token_logprob

    # 1. Empty top_logprobs raises BackendError in LogitGate, ChoiceGate, ScoreGate
    empty_stub = StubBackend(forward_top_logprobs=[])
    with pytest.raises(BackendError, match="no token log-probabilities"):
        LogitGate("Check", backend=empty_stub).evaluate("x")
    with pytest.raises(BackendError, match="no token log-probabilities"):
        ChoiceGate("Pick", {"A": "1", "B": "2"}, backend=empty_stub).evaluate("x")
    with pytest.raises(BackendError, match="no token log-probabilities"):
        ScoreGate("Score", backend=empty_stub).evaluate("x")

    # 2. extract_token_logprob ignores NaN/inf, clamps >0 logprobs, and deduplicates identical raw tokens
    noisy_lps = [
        {"token": "A", "logprob": float("nan")},
        {"token": "A", "logprob": 0.05},  # positive fast-math noise -> clamped to 0.0
        {"token": "A", "logprob": 0.05},  # duplicate raw token "A" -> deduplicated
        {"token": "B", "logprob": float("inf")},
    ]
    assert extract_token_logprob(noisy_lps, "A") == 0.0

    # 3. get_model_temperature prefers exact match over shorter substring in cache
    cache_file = tmp_path / "calibration.json"
    cache_file.write_text(json.dumps({"qwen2.5": 1.30, "qwen2.5-coder:14b": 1.08}), encoding="utf-8")
    assert get_model_temperature("qwen2.5-coder:14b", cache_path=cache_file) == pytest.approx(1.08)

    # 4. ASTGate: bool is rejected for int/float schema, from X import Y is caught, acyclic_dag rejects non-DAG
    schema_gate = ASTGate(json_schema={"type": "object", "properties": {"count": {"type": "integer"}}})
    assert schema_gate.evaluate({"count": True}).passed is False
    assert ASTGate(json_schema={"count": int}).evaluate({"count": False}).passed is False

    imp_gate = ASTGate(forbidden_imports=["urllib.request", "subprocess"])
    assert imp_gate.evaluate("from urllib import request").passed is False
    assert imp_gate.evaluate("import os, subprocess\ndef broken(:").passed is False

    dag_gate = ASTGate(acyclic_dag=True)
    assert dag_gate.evaluate("not valid json").passed is False
    assert dag_gate.evaluate([{"id": 0}, {"id": 1, "deps": [0]}]).passed is True
    assert dag_gate.evaluate([{"id": 1}, {"id": 1}]).passed is False
    assert dag_gate.evaluate([{"id": 1, "dependencies": "not_a_list"}]).passed is False

    # 5. JudgeGate: {"verdict": "PASSED"} and [VERDICT: PASSED] parse as PASS, and "Verdict: FAILURES: 0" does not match tag
    jg = JudgeGate(
        rubric="Check",
        backend=StubBackend(diagnostic_text='{"verdict": "PASSED", "score": 1.0}'),
    )
    assert jg.evaluate("x").passed is True

    # 6. httpx.AsyncClient recreates cleanly across multiple asyncio.run() loops
    backend = OllamaBackend()

    async def _get_client() -> Any:
        return backend.async_client

    loop1_client = asyncio.run(_get_client())
    loop2_client = asyncio.run(_get_client())
    assert loop1_client is not loop2_client
    asyncio.run(backend.aclose())
