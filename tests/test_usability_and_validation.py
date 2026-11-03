from __future__ import annotations

import pytest

from logitgate import (
    ASTGate,
    Cascade,
    ChoiceGate,
    DecisionStatus,
    JudgeGate,
    LayerTrace,
    LlamaCppBackend,
    LogitGate,
    OllamaBackend,
    OpenAICompatBackend,
    ScoreGate,
    StubBackend,
    Verdict,
)

# ==============================================================================
# 1. Usability & REPL __repr__ Tests
# ==============================================================================

def test_verdict_repr():
    """Verdict.__repr__ displays concise, informative attributes for debugging."""
    v_pass = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        probability=0.9234,
        triggered_layer="LogitGate",
        total_latency_ms=48.25,
        total_tokens_generated=0,
    )
    r = repr(v_pass)
    assert "status='passed'" in r
    assert "prob=0.9234" in r
    assert "layer='LogitGate'" in r
    assert "latency=48.25ms" in r
    assert "tokens=0" in r

    v_choice = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        choice="B",
        probability=0.8800,
        triggered_layer="ChoiceGate",
        total_latency_ms=12.50,
    )
    r_c = repr(v_choice)
    assert "choice='B'" in r_c
    assert "layer='ChoiceGate'" in r_c

    v_score = Verdict(
        passed=True,
        status=DecisionStatus.PASSED,
        score=4.50,
        triggered_layer="ScoreGate",
        total_latency_ms=15.10,
    )
    r_s = repr(v_score)
    assert "score=4.50" in r_s
    assert "layer='ScoreGate'" in r_s


def test_layer_trace_repr():
    """LayerTrace.__repr__ displays layer name, status, and latency."""
    trace = LayerTrace(
        layer_name="ASTGate",
        status=DecisionStatus.PASSED,
        latency_ms=0.04,
        tokens_generated=0,
    )
    r = repr(trace)
    assert "layer='ASTGate'" in r
    assert "status=passed" in r
    assert "latency=0.04ms" in r


def test_ast_gate_repr():
    """ASTGate.__repr__ lists configured checks."""
    gate = ASTGate(syntax=True, forbidden_imports=["os", "sys"], acyclic_dag=True)
    r = repr(gate)
    assert "ASTGate" in r
    assert "syntax=True" in r
    assert "forbidden_imports=['os', 'sys']" in r
    assert "acyclic_dag=True" in r


def test_logit_gate_repr():
    """LogitGate.__repr__ displays truncated rubric, thresholds, and backend."""
    backend = StubBackend(default_prob=0.95)
    gate = LogitGate(
        rubric="Candidate code must be strictly typed and contain docstrings.",
        backend=backend,
        pass_threshold=0.88,
        fail_threshold=0.15,
    )
    r = repr(gate)
    assert "LogitGate" in r
    assert "pass_threshold=0.88" in r
    assert "fail_threshold=0.15" in r
    assert "StubBackend" in r


def test_choice_gate_repr():
    """ChoiceGate.__repr__ displays question, option keys, and backend."""
    gate = ChoiceGate(
        question="Select the optimal cache strategy:",
        options={"A": "LRU cache", "B": "LFU cache", "C": "No cache"},
    )
    r = repr(gate)
    assert "ChoiceGate" in r
    assert "options=['A', 'B', 'C']" in r


def test_score_gate_repr():
    """ScoreGate.__repr__ displays scale range, threshold, and backend."""
    gate = ScoreGate(
        rubric="Score code clarity from 1 to 5.",
        min_score=1,
        max_score=5,
        threshold=4.0,
    )
    r = repr(gate)
    assert "ScoreGate" in r
    assert "range=[1..5]" in r
    assert "threshold=4.0" in r


def test_judge_gate_repr():
    """JudgeGate.__repr__ displays trigger policy and backend."""
    gate = JudgeGate(rubric="Critique edge cases", trigger="on_ambiguous_only")
    r = repr(gate)
    assert "JudgeGate" in r
    assert "trigger='on_ambiguous_only'" in r


def test_cascade_repr():
    """Cascade.__repr__ shows operator-chained gate pipeline representation."""
    pipeline = (
        ASTGate(syntax=True)
        >> LogitGate(rubric="Must be safe")
        >> JudgeGate(trigger="on_ambiguous_or_fail")
    )
    r = repr(pipeline)
    assert "Cascade([" in r
    assert "ASTGate" in r
    assert ">> LogitGate" in r
    assert ">> JudgeGate" in r


def test_backends_repr():
    """Backend classes display clean representations."""
    stub = StubBackend(default_prob=0.90)
    assert repr(stub) == "StubBackend(default_prob=0.9)"

    ollama = OllamaBackend(model="qwen2.5-coder:14b", base_url="http://localhost:11434")
    assert "OllamaBackend(model='qwen2.5-coder:14b'" in repr(ollama)

    openai = OpenAICompatBackend(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
    assert "OpenAICompatBackend(model='gpt-4o-mini'" in repr(openai)

    llamacpp = LlamaCppBackend(model="meta-llama-3.1-8b", base_url="http://localhost:8080")
    assert "LlamaCppBackend(model='meta-llama-3.1-8b'" in repr(llamacpp)


# ==============================================================================
# 2. Input Validation Tests
# ==============================================================================

def test_logit_gate_validation():
    """LogitGate validates rubric, thresholds, and temperature."""
    # Empty rubric
    with pytest.raises(ValueError, match="rubric cannot be empty"):
        LogitGate(rubric="")
    with pytest.raises(ValueError, match="rubric cannot be empty"):
        LogitGate(rubric="   ")

    # Invalid thresholds range
    with pytest.raises(ValueError, match="pass_threshold must be between 0.0 and 1.0"):
        LogitGate(rubric="Valid", pass_threshold=1.5)
    with pytest.raises(ValueError, match="pass_threshold must be between 0.0 and 1.0"):
        LogitGate(rubric="Valid", pass_threshold=-0.1)
    with pytest.raises(ValueError, match="fail_threshold must be between 0.0 and 1.0"):
        LogitGate(rubric="Valid", fail_threshold=-0.05)
    with pytest.raises(ValueError, match="fail_threshold must be between 0.0 and 1.0"):
        LogitGate(rubric="Valid", fail_threshold=1.05)

    # pass_threshold <= fail_threshold
    with pytest.raises(ValueError, match="must be strictly greater than fail_threshold"):
        LogitGate(rubric="Valid", pass_threshold=0.50, fail_threshold=0.50)
    with pytest.raises(ValueError, match="must be strictly greater than fail_threshold"):
        LogitGate(rubric="Valid", pass_threshold=0.40, fail_threshold=0.60)

    # Invalid temperature
    with pytest.raises(ValueError, match="logit_temperature must be positive"):
        LogitGate(rubric="Valid", logit_temperature=0.0)
    with pytest.raises(ValueError, match="logit_temperature must be positive"):
        LogitGate(rubric="Valid", logit_temperature=-1.5)


def test_choice_gate_validation():
    """ChoiceGate validates question, options count, and temperature."""
    # Empty question
    with pytest.raises(ValueError, match="question cannot be empty"):
        ChoiceGate(question="", options={"A": "opt A", "B": "opt B"})
    with pytest.raises(ValueError, match="question cannot be empty"):
        ChoiceGate(question="   \t", options={"A": "opt A", "B": "opt B"})

    # Fewer than 2 options
    with pytest.raises(ValueError, match="ChoiceGate requires at least 2 options"):
        ChoiceGate(question="Valid?", options={})
    with pytest.raises(ValueError, match="ChoiceGate requires at least 2 options"):
        ChoiceGate(question="Valid?", options={"A": "Solo option"})

    # Invalid temperature
    with pytest.raises(ValueError, match="logit_temperature must be positive"):
        ChoiceGate(
            question="Valid?",
            options={"A": "opt A", "B": "opt B"},
            logit_temperature=0.0,
        )


def test_score_gate_validation():
    """ScoreGate validates rubric, min/max score order, and threshold range."""
    # Empty rubric
    with pytest.raises(ValueError, match="rubric cannot be empty"):
        ScoreGate(rubric="")

    # min_score >= max_score
    with pytest.raises(ValueError, match="min_score .* must be strictly less than max_score"):
        ScoreGate(rubric="Valid", min_score=5, max_score=5)
    with pytest.raises(ValueError, match="min_score .* must be strictly less than max_score"):
        ScoreGate(rubric="Valid", min_score=5, max_score=1)

    # Threshold out of range
    with pytest.raises(ValueError, match="threshold .* must be between min_score"):
        ScoreGate(rubric="Valid", min_score=1, max_score=5, threshold=0.5)
    with pytest.raises(ValueError, match="threshold .* must be between min_score"):
        ScoreGate(rubric="Valid", min_score=1, max_score=5, threshold=5.5)

    # Invalid temperature
    with pytest.raises(ValueError, match="logit_temperature must be positive"):
        ScoreGate(rubric="Valid", logit_temperature=-0.5)


def test_judge_gate_validation():
    """JudgeGate validates trigger modes and rubric."""
    # Invalid trigger mode
    with pytest.raises(ValueError, match="Invalid trigger 'unsupported_mode'"):
        JudgeGate(trigger="unsupported_mode")  # type: ignore

    # Whitespace-only rubric
    with pytest.raises(ValueError, match="rubric cannot be empty or whitespace"):
        JudgeGate(rubric="   ")

    # None rubric is allowed (inherited from cascade)
    gate = JudgeGate(rubric=None)
    assert gate.rubric is None


def test_cascade_validation():
    """Cascade validates that at least one gate is provided."""
    with pytest.raises(ValueError, match="Cascade requires at least one gate"):
        Cascade(gates=[])


# ==============================================================================
# 3. Configurability & Hardware Options Tests
# ==============================================================================

def test_global_configure_and_reset():
    """logitgate.configure updates global defaults and reset_config restores them."""
    from logitgate import LogitGateDefaults, configure, reset_config

    reset_config()
    assert LogitGateDefaults.backend == "stub://default"
    assert LogitGateDefaults.pass_threshold == 0.85
    assert LogitGateDefaults.fail_threshold == 0.20
    assert LogitGateDefaults.logit_temperature == 1.0

    configure(
        backend="stub://0.77",
        pass_threshold=0.92,
        fail_threshold=0.15,
        logit_temperature=0.5,
    )
    assert LogitGateDefaults.backend == "stub://0.77"
    assert LogitGateDefaults.pass_threshold == 0.92
    assert LogitGateDefaults.fail_threshold == 0.15
    assert LogitGateDefaults.logit_temperature == 0.5

    # Gate created without explicit params inherits configured defaults
    gate = LogitGate(rubric="Test rubric")
    assert gate.pass_threshold == 0.92
    assert gate.fail_threshold == 0.15
    assert gate.logit_temperature == 0.5
    assert isinstance(gate.backend, StubBackend)
    assert gate.backend.default_prob == 0.77

    reset_config()
    assert LogitGateDefaults.backend == "stub://default"
    assert LogitGateDefaults.pass_threshold == 0.85


def test_env_var_resolution(monkeypatch):
    """LOGITGATE_* environment variables resolve dynamically."""
    from logitgate import LogitGateDefaults, reset_config

    reset_config()
    monkeypatch.setenv("LOGITGATE_BACKEND", "stub://0.88")
    monkeypatch.setenv("LOGITGATE_PASS_THRESHOLD", "0.90")
    monkeypatch.setenv("LOGITGATE_FAIL_THRESHOLD", "0.10")
    monkeypatch.setenv("LOGITGATE_TEMPERATURE", "1.5")

    assert LogitGateDefaults.backend == "stub://0.88"
    assert LogitGateDefaults.pass_threshold == 0.90
    assert LogitGateDefaults.fail_threshold == 0.10
    assert LogitGateDefaults.logit_temperature == 1.5

    gate = LogitGate(rubric="Env test")
    assert gate.pass_threshold == 0.90
    assert gate.fail_threshold == 0.10
    assert gate.logit_temperature == 1.5
    assert gate.backend.default_prob == 0.88

    judge = JudgeGate(rubric="Judge env test")
    assert isinstance(judge.backend, StubBackend)
    assert judge.backend.default_prob == 0.88

    reset_config()


def test_ollama_backend_options():
    """OllamaBackend accepts and merges custom options and keep_alive."""
    backend = OllamaBackend(
        base_url="http://localhost:11434",
        model="gemma4:e4b",
        options={"num_gpu": 0, "num_thread": 8},
        keep_alive="5m",
    )
    assert backend.options == {"num_gpu": 0, "num_thread": 8}
    assert backend.keep_alive == "5m"

    payload = backend.build_query_payload("Evaluate code", top_k=5)
    assert payload["keep_alive"] == "5m"
    assert payload["options"]["num_gpu"] == 0
    assert payload["options"]["num_thread"] == 8
    assert payload["options"]["num_predict"] == 1
    assert payload["options"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_aevaluate_batch_stampede_prevention():
    """aevaluate_batch warms KV-cache with first prompt before gathering rest."""
    backend = StubBackend(default_prob=0.95)
    gate = LogitGate(rubric="Base rubric", backend=backend)
    rubrics = ["Rubric 1", "Rubric 2", "Rubric 3"]
    verdicts = await gate.aevaluate_batch("Candidate artifact", rubrics)
    assert len(verdicts) == 3
    assert all(v.passed for v in verdicts)


# ==============================================================================
# 4. Calibration, Presets, ContractGate & max_tokens Tests
# ==============================================================================

def test_auto_calibrate_backend(tmp_path):
    """auto_calibrate_backend fits temperature on micro-calibration dataset and caches it."""
    from logitgate import (
        KNOWN_MODEL_TEMPERATURES,
        auto_calibrate_backend,
        get_model_temperature,
    )

    backend = StubBackend(default_prob=0.92)
    cache_file = tmp_path / "calibration.json"
    res = auto_calibrate_backend(backend, model_key="test-model", save_cache=True, cache_path=cache_file)

    assert res.optimal_temperature > 0.0
    assert cache_file.exists()
    assert get_model_temperature("test-model", cache_path=cache_file) == res.optimal_temperature
    assert get_model_temperature("qwen2.5-coder:14b") == KNOWN_MODEL_TEMPERATURES["qwen2.5-coder:14b"]
    assert get_model_temperature("non-existent-model-xyz") == 1.0


def test_logit_gate_presets():
    """LogitGate initializes with threshold & debiasing presets."""
    gate_triage = LogitGate(rubric="Test rubric", preset="fast_triage")
    assert gate_triage.pass_threshold == 0.80
    assert gate_triage.fail_threshold == 0.20
    assert gate_triage.symmetric_debiasing is False

    gate_strict = LogitGate(rubric="Test rubric", preset="strict")
    assert gate_strict.pass_threshold == 0.90
    assert gate_strict.fail_threshold == 0.25
    assert gate_strict.symmetric_debiasing is True

    gate_bal = LogitGate(rubric="Test rubric", preset="balanced")
    assert gate_bal.pass_threshold == 0.85
    assert gate_bal.fail_threshold == 0.20
    assert gate_bal.symmetric_debiasing is True

    with pytest.raises(ValueError, match="Invalid preset"):
        LogitGate(rubric="Test rubric", preset="invalid_preset")  # type: ignore


def test_contract_gate_json_schema_and_length():
    """ContractGate validates JSON schema, Pydantic models, and min/max length in 0ms."""
    from pydantic import BaseModel

    from logitgate import ContractGate

    class UserPayload(BaseModel):
        id: int
        name: str

    gate_pydantic = ContractGate(json_schema=UserPayload)
    assert gate_pydantic.evaluate('{"id": 42, "name": "LogitGate"}').passed
    assert not gate_pydantic.evaluate('{"id": "not_int", "name": "LogitGate"}').passed
    assert not gate_pydantic.evaluate('invalid json string').passed

    gate_dict = ContractGate(json_schema={"required": ["action", "status"]})
    assert gate_dict.evaluate('{"action": "deploy", "status": "ok"}').passed
    assert not gate_dict.evaluate('{"action": "deploy"}').passed

    gate_len = ContractGate(min_length=5, max_length=15)
    assert not gate_len.evaluate("abc").passed
    assert not gate_len.evaluate("this string exceeds fifteen characters").passed
    assert gate_len.evaluate("valid length").passed


def test_judge_gate_max_tokens():
    """JudgeGate supports max_tokens for concise CPU diagnostics."""
    stub = StubBackend()
    judge = JudgeGate(rubric="Critique", max_tokens=60, backend=stub)
    assert judge.max_tokens == 60
    assert "max_tokens=60" in repr(judge)

    v = judge.evaluate("Candidate code")
    assert v.passed is True


def test_judge_gate_parse_verdict_robustness():
    """JudgeGate._parse_verdict handles markdown bolding and avoids false failures on negations."""
    judge = JudgeGate(rubric="Critique", backend=StubBackend())

    # Case 1: Markdown bolding & variations
    assert judge._parse_verdict("**[VERDICT: PASS]**")[0] is True
    assert judge._parse_verdict("[**VERDICT: PASS**]")[0] is True
    assert judge._parse_verdict("**Verdict:** PASS")[0] is True
    assert judge._parse_verdict("Verdict: PASS")[0] is True
    assert judge._parse_verdict("**Verdict**: FAIL")[0] is False
    assert judge._parse_verdict("[VERDICT: FAIL]")[0] is False

    # Case 2: Negation safety ("does not fail" in body shouldn't trigger FAIL)
    qwen_sample = (
        "**[VERDICT: PASS]**\n"
        "The candidate artifact is well-formed.\n"
        "The candidate does not fail any requirements, and all assertions succeed."
    )
    passed, _status, score = judge._parse_verdict(qwen_sample)
    assert passed is True
    assert score == 1.0

    # Case 3: Fallback without explicit tag but with negation
    fallback_sample = (
        "Evaluation summary: The implementation passes all tests without fail.\n"
        "No failures were detected."
    )
    assert judge._parse_verdict(fallback_sample)[0] is True

    # Case 4: Explicit failure
    fail_sample = "Evaluation summary: The code failed on edge cases."
    assert judge._parse_verdict(fail_sample)[0] is False

    # Case 5: Reason-then-Verdict default prompt vs Verdict-first short cap prompt
    prompt = judge._compile_prompt("def add(a, b): return a + b")
    assert "conclude on the very last line with your verdict tag: [VERDICT: PASS] or [VERDICT: FAIL]" in prompt
    short_judge = JudgeGate(rubric="Critique", max_tokens=60, backend=StubBackend())
    short_prompt = short_judge._compile_prompt("def add(a, b): return a + b")
    assert "Begin your response on the very first line with your verdict tag: [VERDICT: PASS] or [VERDICT: FAIL]" in short_prompt

    # Case 6: Dogfood LogitGate probe when text has zero explicit keywords
    # Review text without "PASS" or "FAIL":
    ambiguous_review = "Detailed evaluation notes: All boundary conditions behave as specified and unit test execution was verified."
    # StubBackend with custom logprobs simulating model answering YES (PASS)
    stub_pass = StubBackend(logprob_responses=[[{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -3.5}]])
    judge_dogfood = JudgeGate(rubric="Critique", backend=stub_pass)
    assert judge_dogfood._parse_verdict(ambiguous_review)[0] is True



def test_cli_calibrate(tmp_path, capsys):
    """CLI calibrate subcommand runs auto-calibration and writes cache."""
    from logitgate.cli import main

    cache_file = str(tmp_path / "cli_cal.json")
    ret = main([
        "calibrate",
        "--backend", "stub://0.95",
        "--model", "custom-ci-model",
        "--cache-path", cache_file,
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Optimal Temperature" in captured.out
    assert (tmp_path / "cli_cal.json").exists()


# ==============================================================================
# 9. VerdictSnapshot & Failure Reproduction Tests
# ==============================================================================

def test_verdict_snapshot_save_load(tmp_path):
    from logitgate.snapshot import VerdictSnapshot

    v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.15,
        triggered_layer="LogitGate",
        total_latency_ms=25.0,
        total_tokens_generated=0,
        reason="Failed threshold",
        traces=[
            LayerTrace(
                layer_name="LogitGate",
                status=DecisionStatus.FAILED,
                latency_ms=25.0,
                calibrated_prob=0.15,
            )
        ],
    )
    snap = VerdictSnapshot(
        test_name="test_login_auth",
        rubric="Must not reveal passwords",
        candidate="pwd = 123",
        verdict=v,
        gate_config={"pass_threshold": 0.85, "fail_threshold": 0.20},
    )

    assert len(snap.snapshot_id) == 8
    assert snap.timestamp
    assert snap.test_name == "test_login_auth"

    save_dir = tmp_path / "failures"
    saved_file = snap.save(directory=save_dir)
    assert saved_file.exists()
    assert (save_dir / "latest.json").exists()

    # Load from directory
    loaded_dir = VerdictSnapshot.load(save_dir)
    assert loaded_dir.snapshot_id == snap.snapshot_id
    assert loaded_dir.test_name == "test_login_auth"
    assert loaded_dir.candidate == "pwd = 123"
    assert loaded_dir.verdict.probability == 0.15

    # Load from file
    loaded_file = VerdictSnapshot.load(saved_file)
    assert loaded_file.snapshot_id == snap.snapshot_id
    assert loaded_file.rubric == "Must not reveal passwords"


def test_verdict_snapshot_replay_and_explain():
    from logitgate.snapshot import VerdictSnapshot

    v = Verdict(
        passed=False,
        status=DecisionStatus.FAILED,
        probability=0.15,
        triggered_layer="LogitGate",
        total_latency_ms=25.0,
        total_tokens_generated=0,
        reason="Failed threshold",
        traces=[
            LayerTrace(
                layer_name="LogitGate",
                status=DecisionStatus.FAILED,
                latency_ms=25.0,
                calibrated_prob=0.15,
            )
        ],
    )
    snap = VerdictSnapshot(
        test_name="test_repro",
        rubric="Candidate code must be clean and safe.",
        candidate="eval('1 + 1')",
        verdict=v,
        gate_config={"pass_threshold": 0.85, "fail_threshold": 0.20},
    )

    # Replay with StubBackend
    replayed = snap.replay(backend=StubBackend(default_prob=0.95))
    assert replayed.passed is True
    assert replayed.status == DecisionStatus.PASSED

    # Explain with StubBackend
    stub_judge = StubBackend(
        diagnostic_text="[VERDICT: FAIL] The candidate uses dangerous eval().",
        tokens_generated=22,
    )
    explained = snap.explain(backend=stub_judge)
    assert explained.passed is False
    assert explained.status == DecisionStatus.FAILED
    assert explained.triggered_layer == "JudgeGate"
    assert "[VERDICT: FAIL] The candidate uses dangerous eval()." in (explained.reason or "")
    assert len(explained.traces) == 2
    assert explained.traces[0].layer_name == "LogitGate"
    assert explained.traces[1].layer_name == "JudgeGate"
    assert explained.total_tokens_generated == 22


def test_logit_gate_save_on_fail(tmp_path):
    from pathlib import Path

    from logitgate.snapshot import VerdictSnapshot

    gate = LogitGate(rubric="Must be safe", backend=StubBackend(default_prob=0.10))
    verdict = gate.evaluate(
        "unsafe payload",
        save_on_fail=True,
        snapshot_dir=tmp_path / "logit_fails",
        test_name="test_safety",
    )
    assert verdict.passed is False
    assert "snapshot_path" in verdict.traces[-1].metadata
    snap_path = Path(verdict.traces[-1].metadata["snapshot_path"])
    assert snap_path.exists()

    snap = VerdictSnapshot.load(snap_path)
    assert snap.test_name == "test_safety"
    assert snap.candidate == "unsafe payload"


def test_logit_gate_configure_save_failures(tmp_path):
    from logitgate.config import configure, reset_config

    configure(save_failures=True, snapshot_dir=str(tmp_path / "global_fails"))
    try:
        gate = LogitGate(rubric="Must be valid", backend=StubBackend(default_prob=0.10))
        verdict = gate.evaluate("invalid data")
        assert verdict.passed is False
        assert "snapshot_path" in verdict.traces[-1].metadata
        assert (tmp_path / "global_fails" / "latest.json").exists()
    finally:
        reset_config()


def test_cascade_save_on_fail(tmp_path):
    from logitgate.snapshot import VerdictSnapshot

    cascade = ASTGate(syntax=True) >> LogitGate(
        rubric="Clean code",
        backend=StubBackend(default_prob=0.10),
    )
    verdict = cascade.evaluate(
        "x = 1",
        save_on_fail=True,
        snapshot_dir=tmp_path / "cascade_fails",
        test_name="test_cascade_pipeline",
    )
    assert verdict.passed is False
    assert "snapshot_path" in verdict.traces[-1].metadata
    snap = VerdictSnapshot.load(tmp_path / "cascade_fails")
    assert snap.test_name == "test_cascade_pipeline"


def test_pytest_plugin_check_save_snapshot_failure(tmp_path, monkeypatch):
    from logitgate.integrations.pytest_plugin import check
    from logitgate.snapshot import VerdictSnapshot

    monkeypatch.setenv("LOGITGATE_SNAPSHOT_DIR", str(tmp_path / "check_fails"))

    @check("Must pass criteria", backend=StubBackend(default_prob=0.05), save_snapshot=True)
    def my_flaky_test():
        return "failing candidate"

    with pytest.raises(AssertionError) as exc_info:
        my_flaky_test()

    err_msg = str(exc_info.value)
    assert "Snapshot saved to:" in err_msg
    assert "logitgate explain" in err_msg
    assert (tmp_path / "check_fails" / "latest.json").exists()

    snap = VerdictSnapshot.load(tmp_path / "check_fails")
    assert snap.test_name == "my_flaky_test"
    assert snap.candidate == "failing candidate"


def test_base_gate_callable_and_from_verdict(tmp_path, monkeypatch):
    """BaseGate.__call__ delegates to evaluate() and VerdictSnapshot.from_verdict works cleanly."""
    from logitgate.cli import main
    from logitgate.snapshot import VerdictSnapshot

    gate = LogitGate(rubric="Must be valid", backend=StubBackend(default_prob=0.95))
    verdict = gate("valid candidate")
    assert verdict.passed is True
    assert verdict.status == DecisionStatus.PASSED

    # VerdictSnapshot.from_verdict factory
    fail_gate = LogitGate(rubric="Must not fail", backend=StubBackend(default_prob=0.10))
    fail_verdict = fail_gate("bad candidate")
    snap = VerdictSnapshot.from_verdict(
        verdict=fail_verdict,
        rubric=fail_gate.rubric,
        candidate="bad candidate",
        test_name="test_factory",
        gate_config={"backend": "stub://0.10", "pass_threshold": 0.85, "fail_threshold": 0.20},
    )
    assert snap.test_name == "test_factory"
    assert snap.verdict.passed is False

    # Explain on failed snapshot with default StubBackend preserves FAILED status
    explained = snap.explain()
    assert explained.passed is False
    assert explained.status == DecisionStatus.FAILED

    # CLI eval respects LOGITGATE_BACKEND and LOGITGATE_PASS_THRESHOLD when flags omitted
    monkeypatch.setenv("LOGITGATE_BACKEND", "stub://0.95")
    monkeypatch.setenv("LOGITGATE_PASS_THRESHOLD", "0.80")
    ret = main(["eval", "--rubric", "Check CLI env", "--candidate", "hello world", "--json"])
    assert ret == 0

