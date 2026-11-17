from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

import logitgate
import logitgate.gates
from benchmarks.external.loaders import NormalizedBinaryItem
from benchmarks.external.run_external import (
    evaluate_binary,
    run_benchmark_suite,
)
from logitgate import (
    GeminiBackend,
    NormalizedToolCall,
    ToolCallFailureCode,
    ToolCallGate,
    UnsupportedLogprobsError,
    UnsupportedLogprobsPolicy,
    VerdictSnapshot,
    check,
    normalize_tool_calls,
    render_reference_context,
)
from logitgate.backends.base import StubBackend
from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.commands.eval_cmd import _cmd_eval
from logitgate.cli.parser import create_parser
from logitgate.exceptions import ConfigurationError
from logitgate.models import DecisionStatus

# ---------------------------------------------------------------------------
# 1. Public API Exports & Contract Parity
# ---------------------------------------------------------------------------


def test_public_api_exports_and_clean_all() -> None:
    """Verifies all Milestone 3 additions are exposed and cleanly sorted in __all__."""
    assert GeminiBackend is not None
    assert NormalizedToolCall is not None
    assert ToolCallFailureCode is not None
    assert ToolCallGate is not None
    assert UnsupportedLogprobsError is not None
    assert UnsupportedLogprobsPolicy is not None
    assert normalize_tool_calls is not None
    assert render_reference_context is not None

    top_level_all = logitgate.__all__
    assert len(top_level_all) == len(set(top_level_all)), "Duplicate symbols in logitgate.__all__"

    expected_top_symbols = [
        "GeminiBackend",
        "NormalizedToolCall",
        "ToolCallFailureCode",
        "ToolCallGate",
        "UnsupportedLogprobsError",
        "UnsupportedLogprobsPolicy",
        "normalize_tool_calls",
        "render_reference_context",
    ]
    for sym in expected_top_symbols:
        assert sym in top_level_all, f"Expected {sym} in logitgate.__all__"
        assert hasattr(logitgate, sym), f"Expected logitgate to export {sym}"

    gates_all = logitgate.gates.__all__
    assert len(gates_all) == len(set(gates_all)), "Duplicate symbols in logitgate.gates.__all__"
    assert gates_all == sorted(gates_all), "logitgate.gates.__all__ must be sorted alphabetically"

    expected_gate_symbols = [
        "NormalizedToolCall",
        "ToolCallFailureCode",
        "ToolCallGate",
        "normalize_tool_calls",
    ]
    for sym in expected_gate_symbols:
        assert sym in gates_all, f"Expected {sym} in logitgate.gates.__all__"
        assert hasattr(logitgate.gates, sym), f"Expected logitgate.gates to export {sym}"


# ---------------------------------------------------------------------------
# 2. ToolCallGate VerdictSnapshot.replay() and areplay() Parity
# ---------------------------------------------------------------------------


def test_tool_call_gate_pure_layer0_snapshot_replay_roundtrip(tmp_path: Path) -> None:
    """Pure Layer-0 deterministic ToolCallGate failure snapshot round-trips through replay/areplay."""
    gate = ToolCallGate(
        check_intent=False,
        allowed_tools=["read_file"],
        allowed_roots=["/safe/workspace"],
        name="ContractToolGate",
    )
    assert gate.backend is None

    malicious_call = {
        "name": "read_file",
        "arguments": {"path": "/safe/workspace/../../etc/shadow"},
    }
    verdict = gate.evaluate(
        malicious_call,
        save_on_fail=True,
        snapshot_dir=tmp_path,
        test_name="test_path_traversal",
    )
    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert verdict.traces[0].failure_code == ToolCallFailureCode.PATH_TRAVERSAL.value

    snapshot_path_str = verdict.traces[-1].metadata.get("snapshot_path")
    assert snapshot_path_str is not None
    snapshot_path = Path(snapshot_path_str)
    assert snapshot_path.is_file()

    loaded = VerdictSnapshot.load(snapshot_path)
    assert loaded.gate_config.get("gate_type") == "ToolCallGate"
    assert loaded.gate_config.get("check_intent") is False

    replayed = loaded.replay()
    assert replayed.passed is False
    assert replayed.status == DecisionStatus.FAILED
    assert replayed.traces[0].failure_code == ToolCallFailureCode.PATH_TRAVERSAL.value

    replayed_async = asyncio.run(loaded.areplay())
    assert replayed_async.passed is False
    assert replayed_async.status == DecisionStatus.FAILED
    assert replayed_async.traces[0].failure_code == ToolCallFailureCode.PATH_TRAVERSAL.value

    # Mutated candidate that complies with the contract
    safe_call = {
        "name": "read_file",
        "arguments": {"path": "/safe/workspace/config.json"},
    }
    loaded_safe = loaded.model_copy(update={"candidate": safe_call})
    replayed_safe = loaded_safe.replay()
    assert replayed_safe.passed is True
    assert replayed_safe.status == DecisionStatus.PASSED


def test_tool_call_gate_layer1_intent_snapshot_replay_roundtrip(tmp_path: Path) -> None:
    """Layer-1 intent ToolCallGate failure snapshot round-trips through replay/areplay with backends."""
    backend_fail = StubBackend(supports_logprobs=True, default_prob=0.15)
    gate = ToolCallGate(
        check_intent=True,
        backend=backend_fail,
        intent_rubric="Confirm the tool call accurately reads configuration",
        pass_threshold=0.85,
        allowed_tools=["read_file"],
        name="IntentToolGate",
    )
    assert gate.backend is backend_fail

    valid_contract_call = {
        "name": "read_file",
        "arguments": {"path": "config.json"},
    }
    verdict = gate.evaluate(
        valid_contract_call,
        save_on_fail=True,
        snapshot_dir=tmp_path,
        test_name="test_intent_fail",
    )
    assert verdict.passed is False
    assert verdict.probability == pytest.approx(0.15, rel=1e-4)
    assert len(verdict.traces) == 2
    assert verdict.traces[0].status == DecisionStatus.PASSED
    assert verdict.traces[1].status == DecisionStatus.FAILED

    snapshot_path_str = verdict.traces[-1].metadata.get("snapshot_path")
    assert snapshot_path_str is not None
    loaded = VerdictSnapshot.load(Path(snapshot_path_str))
    assert loaded.gate_config.get("gate_type") == "ToolCallGate"
    assert loaded.gate_config.get("check_intent") is True

    # Replay with same failing backend
    replayed_fail = loaded.replay(backend=StubBackend(supports_logprobs=True, default_prob=0.15))
    assert replayed_fail.passed is False
    assert replayed_fail.probability == pytest.approx(0.15, rel=1e-4)

    # Replay with passing backend
    replayed_pass = loaded.replay(backend=StubBackend(supports_logprobs=True, default_prob=0.93))
    assert replayed_pass.passed is True
    assert replayed_pass.probability == pytest.approx(0.93, rel=1e-4)

    # Replay async with passing backend
    async def _run_async_replay() -> None:
        replayed_async = await loaded.areplay(backend=StubBackend(supports_logprobs=True, default_prob=0.91))
        assert replayed_async.passed is True
        assert replayed_async.probability == pytest.approx(0.91, rel=1e-4)

    asyncio.run(_run_async_replay())


# ---------------------------------------------------------------------------
# 3. @check Pytest Plugin Parity
# ---------------------------------------------------------------------------


def test_check_decorator_rag_context_and_query_sync() -> None:
    """@check decorator accepts context and query and passes sync execution."""
    @check(
        rubric="The response accurately answers based on the reference context",
        context="The server port is 8080 and bind address is 0.0.0.0.",
        query="What is the server port?",
        backend="stub://0.94",
    )
    def handle_request() -> str:
        return "The server runs on port 8080."

    res = handle_request()
    assert res == "The server runs on port 8080."


def test_check_decorator_rag_context_and_query_async() -> None:
    """@check decorator accepts sequence context and query and passes async execution."""
    @check(
        rubric="The response accurately summarizes findings",
        context=["Report Section 1: Revenue up 14%.", "Report Section 2: Margin 22%."],
        query="What is the revenue growth?",
        backend="stub://0.93",
    )
    async def async_fetch_summary() -> str:
        return "Revenue grew by 14%."

    res = asyncio.run(async_fetch_summary())
    assert res == "Revenue grew by 14%."


def test_check_decorator_sidecar_fallback_and_raise() -> None:
    """@check decorator handles unsupported logprobs with sidecar fallback or raise."""
    unsupported_backend = StubBackend(supports_logprobs=False)

    @check(
        rubric="Response is helpful",
        backend=unsupported_backend,
        on_unsupported_logprobs="sidecar",
        sidecar_backend="stub://0.92",
    )
    def test_sidecar_pass() -> str:
        return "Helpful response text."

    assert test_sidecar_pass() == "Helpful response text."

    @check(
        rubric="Response is helpful",
        backend=unsupported_backend,
        on_unsupported_logprobs="raise",
    )
    def test_raise_fail() -> str:
        return "Any text."

    with pytest.raises(UnsupportedLogprobsError, match="does not support logprobs"):
        test_raise_fail()


def test_check_decorator_choice_and_score_and_multi_rubric_with_context() -> None:
    """@check decorator routes context/query across ChoiceGate, ScoreGate, and MultiRubricGate."""
    @check(
        choices={"A": "Affirmative", "B": "Negative"},
        pass_choices=["A"],
        context="Contract specifies full indemnification.",
        query="Is indemnification covered?",
        backend="stub://0.92",
    )
    def check_choice() -> str:
        return "Indemnification is covered."

    assert check_choice() == "Indemnification is covered."

    # ChoiceGate without pass_choices
    @check(
        choices={"A": "Affirmative", "B": "Negative"},
        context="Contract specifies full indemnification.",
        query="Is indemnification covered?",
        backend="stub://0.92",
    )
    def check_choice_no_pass_choices() -> str:
        return "Indemnification is covered."

    assert check_choice_no_pass_choices() == "Indemnification is covered."

    # Async choice gate with pass choices
    @check(
        choices={"A": "Affirmative", "B": "Negative"},
        pass_choices=["A"],
        context="Contract specifies full indemnification.",
        query="Is indemnification covered?",
        backend="stub://0.92",
    )
    async def async_check_choice() -> str:
        return "Indemnification is covered."

    assert asyncio.run(async_check_choice()) == "Indemnification is covered."

    # Async choice gate with failing pass_choices
    @check(
        choices={"A": "Affirmative", "B": "Negative"},
        pass_choices=["B"],  # Will fail because stub chooses A
        context="Contract specifies full indemnification.",
        query="Is indemnification covered?",
        backend="stub://0.92",
    )
    async def async_check_choice_fail() -> str:
        return "Indemnification is covered."

    with pytest.raises(AssertionError, match="not in pass_choices"):
        asyncio.run(async_check_choice_fail())

    # MultiRubricGate aggregation any and mean
    @check(
        rubrics=["Accurate", "Concise"],
        aggregation="any",
        backend="stub://0.10",
    )
    def check_multi_any_fail() -> str:
        return "Poor response."

    with pytest.raises(AssertionError, match="rubrics passed"):
        check_multi_any_fail()

    @check(
        rubrics=["Accurate", "Concise"],
        aggregation="mean",
        threshold=0.85,
        backend="stub://0.90",
    )
    def check_multi_mean_pass() -> str:
        return "Good response."

    assert check_multi_mean_pass() == "Good response."

    @check(
        score_range=(1, 5),
        score_threshold=3.5,
        context="Service level is exceptional.",
        query="Rate service quality.",
        backend="stub://0.92",
    )
    def check_score() -> str:
        return "Great service."

    assert check_score() == "Great service."

    @check(
        rubrics=["Accurate", "Concise"],
        context="System status is healthy.",
        query="How is the system?",
        backend="stub://0.92",
    )
    def check_multi() -> str:
        return "System is healthy."

    assert check_multi() == "System is healthy."


# ---------------------------------------------------------------------------
# 4. logitgate eval CLI Parity
# ---------------------------------------------------------------------------


def test_cli_eval_pure_layer0_tool_call(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI eval supports pure Layer-0 ToolCallGate evaluation without --rubric."""
    parser = create_parser()

    # 1. Failing destructive command
    args_fail = parser.parse_args([
        "eval",
        "--tool-call",
        "--candidate",
        json.dumps({"name": "execute_command", "arguments": {"command": "rm -rf /"}}),
    ])
    exit_fail = _cmd_eval(args_fail)
    assert exit_fail == EXIT_FAILED
    out_fail = capsys.readouterr().out
    assert "Verdict: FAILED" in out_fail
    assert "destructive" in out_fail.lower() or "blocked" in out_fail.lower()

    # 2. Passing tool call with allowed-tool and allowed-root
    args_pass = parser.parse_args([
        "eval",
        "--tool-call",
        "--allowed-tool",
        "read_file",
        "--allowed-root",
        "/var/data",
        "--candidate",
        json.dumps({"name": "read_file", "arguments": {"path": "/var/data/log.txt"}}),
    ])
    exit_pass = _cmd_eval(args_pass)
    assert exit_pass == EXIT_OK
    out_pass = capsys.readouterr().out
    assert "Verdict: PASSED" in out_pass

    # 3. Passing tool call with JSON output
    args_json = parser.parse_args([
        "eval",
        "--tool-call",
        "--allowed-tool",
        "read_file",
        "--allowed-root",
        "/var/data",
        "--candidate",
        json.dumps({"name": "read_file", "arguments": {"path": "/var/data/log.txt"}}),
        "--json",
    ])
    assert _cmd_eval(args_json) == EXIT_OK
    parsed_json = json.loads(capsys.readouterr().out)
    assert parsed_json["passed"] is True
    assert parsed_json["status"] == "passed"


def test_cli_eval_layer1_intent_tool_call(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI eval supports Layer-1 intent ToolCallGate with rubric and backend."""
    parser = create_parser()

    # Pass intent with high prob backend
    args_pass = parser.parse_args([
        "eval",
        "--tool-call",
        "--rubric",
        "Ensure proposed tool call correctly looks up user details",
        "--backend",
        "stub://0.92",
        "--query",
        "Find profile for user 42",
        "--candidate",
        json.dumps({"name": "get_user", "arguments": {"user_id": 42}}),
    ])
    assert _cmd_eval(args_pass) == EXIT_OK
    out_pass = capsys.readouterr().out
    assert "Verdict: PASSED" in out_pass
    assert "Confidence: 0.9200" in out_pass

    # Fail intent with low prob backend
    args_fail = parser.parse_args([
        "eval",
        "--tool-call",
        "--rubric",
        "Ensure proposed tool call correctly looks up user details",
        "--backend",
        "stub://0.18",
        "--threshold",
        "0.85",
        "--candidate",
        json.dumps({"name": "get_user", "arguments": {"user_id": 42}}),
    ])
    assert _cmd_eval(args_fail) == EXIT_FAILED
    out_fail = capsys.readouterr().out
    assert "Verdict: FAILED" in out_fail
    assert "Confidence: 0.1800" in out_fail


def test_cli_eval_context_and_context_file_and_query(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI eval handles repeatable --context, --context-file, and --query."""
    parser = create_parser()

    # 1. Multiple --context flags
    args_contexts = parser.parse_args([
        "eval",
        "--rubric",
        "Answers according to facts",
        "--context",
        "Context Chunk Alpha.",
        "--context",
        "Context Chunk Beta.",
        "--query",
        "What are the chunks?",
        "--backend",
        "stub://0.94",
        "--candidate",
        "Both Alpha and Beta are present.",
        "--json",
    ])
    assert _cmd_eval(args_contexts) == EXIT_OK
    out_data = json.loads(capsys.readouterr().out)
    assert out_data["passed"] is True

    # 2. Context file
    ctx_file = tmp_path / "reference.txt"
    ctx_file.write_text("Reference content from disk.", encoding="utf-8")

    args_file = parser.parse_args([
        "eval",
        "--rubric",
        "Accurate report",
        "--context-file",
        str(ctx_file),
        "--backend",
        "stub://0.93",
        "--candidate",
        "Verified report.",
    ])
    assert _cmd_eval(args_file) == EXIT_OK
    out_file = capsys.readouterr().out
    assert "Verdict: PASSED" in out_file

    # 3. Missing context file produces clean error
    args_bad_file = parser.parse_args([
        "eval",
        "--rubric",
        "Accurate report",
        "--context-file",
        str(tmp_path / "nonexistent.txt"),
        "--backend",
        "stub://0.93",
        "--candidate",
        "Text",
    ])
    assert _cmd_eval(args_bad_file) == EXIT_FAILED
    err_text = capsys.readouterr().err
    assert "Error reading context file" in err_text


def test_cli_eval_sidecar_backend_flag() -> None:
    """CLI eval supports --on-unsupported-logprobs sidecar --sidecar-backend."""
    parser = create_parser()

    def _mock_resolve_backend(spec: str, **kw: Any) -> StubBackend:
        if "unsupported" in str(spec):
            return StubBackend(supports_logprobs=False)
        return StubBackend(supports_logprobs=True, default_prob=0.91)

    with patch("logitgate.gates.logit_gate.resolve_logit_backend", side_effect=_mock_resolve_backend):
        args = parser.parse_args([
            "eval",
            "--rubric",
            "Verify compliance",
            "--backend",
            "unsupported://model",
            "--on-unsupported-logprobs",
            "sidecar",
            "--sidecar-backend",
            "stub://0.91",
            "--candidate",
            "Compliant content",
        ])
        assert _cmd_eval(args) == EXIT_OK


def test_cli_eval_validation_and_mutual_exclusion(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI eval enforces validation rules on --rubric and --tool-call combinations."""
    parser = create_parser()

    # 1. Missing --rubric when --tool-call is NOT specified
    args_no_rubric = parser.parse_args([
        "eval",
        "--candidate",
        "Sample candidate",
    ])
    assert _cmd_eval(args_no_rubric) == EXIT_FAILED
    err = capsys.readouterr().err
    assert "--rubric is required unless --tool-call is specified" in err

    # 2. Cannot combine --tool-call with --choice
    args_tool_choice = parser.parse_args([
        "eval",
        "--tool-call",
        "--choice",
        "A=Allow",
        "--choice",
        "B=Deny",
        "--candidate",
        "{}",
    ])
    with pytest.raises(ConfigurationError, match="Cannot combine --tool-call with --choice"):
        _cmd_eval(args_tool_choice)

    # 3. Cannot combine --tool-call with multiple --rubric criteria
    args_tool_multi = parser.parse_args([
        "eval",
        "--tool-call",
        "--rubric",
        "Rubric 1",
        "--rubric",
        "Rubric 2",
        "--candidate",
        "{}",
    ])
    with pytest.raises(ConfigurationError, match="Cannot combine --tool-call with multiple --rubric"):
        _cmd_eval(args_tool_multi)


def test_cli_eval_choice_and_score_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI eval human output formats ChoiceGate and ScoreGate verdicts correctly."""
    parser = create_parser()

    # ChoiceGate human output
    args_choice = parser.parse_args([
        "eval",
        "--rubric",
        "Select classification",
        "--choice",
        "A=Approve",
        "--choice",
        "B=Reject",
        "--backend",
        "stub://0.92",
        "--context",
        "Risk profile low",
        "--query",
        "Approve or reject?",
        "--candidate",
        "Low risk transaction",
    ])
    assert _cmd_eval(args_choice) == EXIT_OK
    out_choice = capsys.readouterr().out
    assert "Choice:" in out_choice
    assert "Choice Probabilities:" in out_choice

    # ScoreGate human output
    args_score = parser.parse_args([
        "eval",
        "--rubric",
        "Rate satisfaction",
        "--score-min",
        "1",
        "--score-max",
        "5",
        "--score-threshold",
        "3.5",
        "--backend",
        "stub://0.92",
        "--context",
        "User is satisfied",
        "--query",
        "Satisfaction rating?",
        "--candidate",
        "Great experience",
    ])
    assert _cmd_eval(args_score) == EXIT_OK
    out_score = capsys.readouterr().out
    assert "Score:" in out_score


# ---------------------------------------------------------------------------
# 5. Grounded RAG Mode in run_external.py
# ---------------------------------------------------------------------------


def test_run_external_grounded_rag_mode_evaluation() -> None:
    """evaluate_binary with use_grounded_context=True routes (context, query, answer) to gates."""
    grounded_item_1 = NormalizedBinaryItem(
        id="rag_1",
        rubric="Are factual statements in the answer grounded in context?",
        candidate="Ignored candidate string in grounded mode",
        expected=True,
        metadata={
            "answer": "The battery life is 18 hours.",
            "context": "Specifications: The device battery lasts up to 18 hours.",
            "question": "What is the battery life?",
        },
    )
    grounded_item_2 = NormalizedBinaryItem(
        id="rag_2",
        rubric="Are factual statements in the answer grounded in context?",
        candidate="Ignored candidate string in grounded mode",
        expected=False,
        metadata={
            "answer": "The battery life is 90 hours.",
            "context": "Specifications: The device battery lasts up to 18 hours.",
            "question": "What is the battery life?",
        },
    )

    stub_backend = StubBackend(supports_logprobs=True, default_prob=0.92)

    # Run grounded context evaluation
    res = evaluate_binary(
        [grounded_item_1, grounded_item_2],
        backend=stub_backend,
        modes=("logit", "cascade"),
        use_grounded_context=True,
        progress=False,
    )

    assert res["task_type"] == "binary"
    assert res["grounded_context"] is True
    assert res["total"] == 2

    report = res["benchmark_report"]
    assert "logit" in report["modes"]
    assert "cascade" in report["modes"]

    logit_stats = report["modes"]["logit"]
    assert logit_stats["runs"] == 2
    assert logit_stats["accuracy"] == 0.5  # Item 1 matches expected True, Item 2 fails match

    cascade_stats = report["modes"]["cascade"]
    assert cascade_stats["runs"] == 2


def test_run_benchmark_suite_grounded_context_wiring() -> None:
    """run_benchmark_suite accepts use_grounded_context and passes to evaluate_binary."""
    mock_items = [
        NormalizedBinaryItem(
            id="rag_mock",
            rubric="Groundedness check",
            candidate="Raw",
            expected=True,
            metadata={"answer": "42", "context": "The answer is 42", "question": "What is answer?"},
        )
    ]

    with patch("benchmarks.external.run_external.load_external_benchmark", return_value=mock_items), \
         patch("benchmarks.external.run_external.resolve_logit_backend") as mock_resolve:
        mock_resolve.return_value = StubBackend(supports_logprobs=True, default_prob=0.93)

        res = run_benchmark_suite(
            "ragtruth",
            backend_spec="stub://0.93",
            limit=1,
            use_grounded_context=True,
        )
        assert res["task_type"] == "binary"
        assert res.get("grounded_context") is True
        assert res["total"] == 1


# ---------------------------------------------------------------------------
# 6. Mutation Boundary & Edge Cases
# ---------------------------------------------------------------------------


def test_eval_cmd_mean_aggregation_exact_boundary_ties() -> None:
    """Verifies exact equality tie on fail_threshold (0.20) in eval_cmd mean aggregation."""
    from logitgate.cli.commands.eval_cmd import MultiRubricGate as CliMultiRubricGate
    from logitgate.models import LayerTrace, Verdict

    gate = CliMultiRubricGate(
        rubrics=["r1", "r2"],
        aggregation="mean",
        pass_threshold=0.85,
        fail_threshold=0.20,
        backend=StubBackend(default_prob=0.20),
    )
    with patch.object(
        gate,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.20, triggered_layer="l1", total_latency_ms=1.0, traces=[LayerTrace(layer_name="l1", status=DecisionStatus.FAILED, latency_ms=1.0)]),
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.20, triggered_layer="l2", total_latency_ms=1.0, traces=[LayerTrace(layer_name="l2", status=DecisionStatus.FAILED, latency_ms=1.0)]),
        ],
    ):
        v_tie = gate.evaluate("candidate text")
        # Exactly on fail_threshold must be FAILED, not AMBIGUOUS
        assert v_tie.status == DecisionStatus.FAILED
        assert v_tie.passed is False


def test_pytest_plugin_mean_aggregation_unweighted_half_tie() -> None:
    """Verifies exact half-pass tie (1/2 passed) in pytest plugin mean aggregation when threshold=None."""
    @check(
        rubrics=["r1", "r2"],
        aggregation="mean",
        threshold=None,
        backend="stub://0.50",
    )
    def test_func() -> str:
        return "response"

    gate_inst = test_func._logitgate_gate()  # type: ignore[attr-defined]
    from logitgate.models import LayerTrace, Verdict

    with patch.object(
        gate_inst,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.90, triggered_layer="l1", total_latency_ms=1.0, traces=[LayerTrace(layer_name="l1", status=DecisionStatus.PASSED, latency_ms=1.0)]),
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.10, triggered_layer="l2", total_latency_ms=1.0, traces=[LayerTrace(layer_name="l2", status=DecisionStatus.FAILED, latency_ms=1.0)]),
        ],
    ):
        v = gate_inst.evaluate("response")
        # 1 passed out of 2 is NOT > 2 / 2 (1 > 1 is False). Must be FAILED!
        assert v.passed is False
        assert v.status == DecisionStatus.FAILED


def test_eval_cmd_ast_syntax_with_telemetry(tmp_path: Path) -> None:
    """Verifies that --ast-syntax cascades with target gate and telemetry output."""
    parser = create_parser()
    tel_file = tmp_path / "eval_telemetry.jsonl"
    args = parser.parse_args([
        "eval",
        "--rubric",
        "Valid Python function",
        "--ast-syntax",
        "--backend",
        "stub://0.92",
        "--telemetry-out",
        str(tel_file),
        "--candidate",
        "def compute_total(a, b):\n    return a + b\n",
    ])
    assert _cmd_eval(args) == EXIT_OK
    assert tel_file.is_file()
    lines = [json.loads(line) for line in tel_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) >= 1
    assert lines[0]["passed"] is True

