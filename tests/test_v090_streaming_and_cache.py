from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
import time
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from logitgate import (
    ASTGate,
    BestOfNGate,
    CacheWarning,
    Cascade,
    ChoiceGate,
    ConfigurationError,
    DecisionStatus,
    JudgeGate,
    LogitBackend,
    LogitGate,
    MemoryTelemetrySink,
    MultiRubricGate,
    RankGate,
    ScoreGate,
    StreamAbortedError,
    StreamGuard,
    StubBackend,
    Verdict,
    VerdictCache,
    VerdictSnapshot,
    aguard_stream,
    compute_verdict_cache_key,
    configure,
    guard_stream,
    reset_config,
)
from logitgate.backends.base import GenerativeBackend
from logitgate.benchmark import BUILTIN_CASES, run_benchmark
from logitgate.cli import main as cli_main


class CountingRuleBackend(LogitBackend, GenerativeBackend):
    """Deterministic counting backend driven by substring rules in the candidate block."""

    def __init__(self, *, fail_substrings: tuple[str, ...] = ("SECRET_LEAK", "DROP TABLE")) -> None:
        self.fail_substrings = fail_substrings
        self.logit_calls = 0
        self.generate_calls = 0

    def _candidate_slice(self, prompt: str) -> str:
        marker = "[CANDIDATE ARTIFACT]"
        if marker in prompt:
            start = prompt.find(marker) + len(marker)
            for end_marker in ("[DECISION MAPPING]", "[SCORING SCALE]", "[VERIFICATION NOTES]", "[EVALUATION RUBRIC]"):
                end = prompt.find(end_marker, start)
                if end != -1:
                    return prompt[start:end]
            return prompt[start:]
        return prompt

    def _logprobs_for(self, prompt: str, allowed_tokens: list[str] | None) -> list[dict[str, Any]]:
        self.logit_calls += 1
        if allowed_tokens and set(allowed_tokens) >= {"1", "2", "3", "4", "5"}:
            cand = self._candidate_slice(prompt)
            is_bad = any(tok in cand for tok in self.fail_substrings)
            inverted = "inverted scale" in prompt
            best_tok = "1" if inverted else "5"
            worst_tok = "5" if inverted else "1"
            win = worst_tok if is_bad else best_tok
            return [
                {"token": s, "logprob": -0.05 if s == win else -5.0}
                for s in ("1", "2", "3", "4", "5")
            ]

        if allowed_tokens and "C" in allowed_tokens:
            # ChoiceGate: find whichever slot maps to "Billing"
            win_slot = "A"
            for slot in ("A", "B", "C"):
                if f"{slot}: Billing" in prompt:
                    win_slot = slot
                    break
            return [
                {"token": k, "logprob": -0.05 if k == win_slot else -4.00}
                for k in ("A", "B", "C")
            ]

        if "[CANDIDATE A]" in prompt and "[CANDIDATE B]" in prompt:
            # Pairwise RankGate
            a_start = prompt.find("[CANDIDATE A]")
            b_start = prompt.find("[CANDIDATE B]")
            dec_start = prompt.find("[DECISION MAPPING]")
            cand_a = prompt[a_start:b_start]
            cand_b = prompt[b_start : (dec_start if dec_start != -1 else len(prompt))]
            a_bad = any(tok in cand_a for tok in self.fail_substrings)
            b_bad = any(tok in cand_b for tok in self.fail_substrings)
            win_tok = "B" if (a_bad and not b_bad) else "A"
            lose_tok = "A" if win_tok == "B" else "B"
            return [{"token": win_tok, "logprob": -0.05}, {"token": lose_tok, "logprob": -4.00}]

        cand = self._candidate_slice(prompt)
        is_bad = any(tok in cand for tok in self.fail_substrings)
        is_ambig = "AMBIGUOUS_TOKEN" in cand
        if is_ambig and "[VERIFICATION NOTES]" not in prompt:
            return [{"token": "A", "logprob": -0.6931}, {"token": "B", "logprob": -0.6931}]

        rev_binding = (
            "B: The candidate artifact complies" in prompt
            or "B: The candidate artifact satisfies" in prompt
        )
        pass_tok = "B" if rev_binding else "A"
        fail_tok = "A" if rev_binding else "B"
        chosen = fail_tok if is_bad else pass_tok
        other = pass_tok if is_bad else fail_tok
        return [{"token": chosen, "logprob": -0.02}, {"token": other, "logprob": -4.50}]

    def query_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._logprobs_for(prompt, allowed_tokens)

    async def aquery_top_logprobs(
        self, prompt: str, top_k: int = 10, allowed_tokens: list[str] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        return self._logprobs_for(prompt, allowed_tokens)

    def generate(
        self, prompt: str, max_tokens: int | None = None, **kwargs: Any
    ) -> tuple[str, int]:
        self.generate_calls += 1
        cand = self._candidate_slice(prompt)
        is_bad = any(tok in cand for tok in self.fail_substrings)
        tag = "[VERDICT: FAIL]" if is_bad else "[VERDICT: PASS]"
        return f"- Verified candidate content.\n{tag}", 12

    async def agenerate(
        self, prompt: str, max_tokens: int | None = None, **kwargs: Any
    ) -> tuple[str, int]:
        await asyncio.sleep(0)
        return self.generate(prompt, max_tokens=max_tokens, **kwargs)


@pytest.fixture(autouse=True)
def _clean_global_config() -> Iterator[None]:
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Part 1: StreamGuard (Sync & Async Mid-Stream Guardrails)
# ---------------------------------------------------------------------------


def test_stream_guard_clean_sync_and_exact_final_dedup() -> None:
    backend = CountingRuleBackend()
    sink = MemoryTelemetrySink()
    gate = LogitGate(
        "No secret leaks allowed.",
        backend=backend,
        logit_temperature=1.0,
        telemetry_sink=sink,
    )
    chunks = [
        "The quarterly report shows steady revenue growth across regions. ",
        "Operating margins improved by two percentage points year over year.",
    ]
    # Second chunk ends with '.' and crosses min_chars + stride -> mid-stream check runs at exact end.
    # Final check must deduplicate and NOT re-run the backend!
    handle = gate.guard_stream(chunks, min_chars=40, stride_chars=40)
    output = "".join(handle)
    assert output == "".join(chunks)
    assert handle.summary is not None
    assert handle.summary.passed is True
    assert handle.summary.aborted is False
    assert handle.summary.checks_executed == 2
    # 2 checks * 2 symmetric passes = 4 logit calls (NOT 6; final check deduplicated!)
    assert backend.logit_calls == 4
    assert len(sink.records) == 1
    assert sink.records[0].tags["stream_guard"] is True
    assert sink.records[0].tags["aborted_mid_stream"] is False


def test_stream_guard_early_abort_cancels_sync_generator_and_buffered_zero_leak() -> None:
    backend = CountingRuleBackend()
    gate = LogitGate("No secret leaks.", backend=backend, logit_temperature=1.0)

    closed_flag = False
    produced_chunks = 0

    def upstream() -> Iterator[str]:
        nonlocal closed_flag, produced_chunks
        pieces = [
            "Here is your safe summary introduction sentence. ",
            "Wait, exposing SECRET_LEAK=sk-live-9999 right now. ",
            "More sensitive data that must never even be generated 1. ",
            "More sensitive data that must never even be generated 2. ",
        ]
        try:
            for p in pieces:
                produced_chunks += 1
                yield p
        finally:
            closed_flag = True

    # Buffered mode: chunk 0 passes -> yielded; chunk 1 fails -> aborted BEFORE yielding chunk 1!
    guard = StreamGuard(gate, min_chars=20, stride_chars=20, delivery="buffered")
    collected: list[str] = []
    with pytest.raises(StreamAbortedError) as exc_info:
        for ch in guard.stream(upstream()):
            collected.append(ch)  # noqa: PERF402 - preserve chunks yielded prior to raise

    err = exc_info.value
    assert err.is_final is False
    assert err.chunk_index == 1
    assert "SECRET_LEAK" in err.partial_text
    assert "SECRET_LEAK" not in err.yielded_text
    assert "".join(collected) == "Here is your safe summary introduction sentence. "
    assert closed_flag is True
    assert produced_chunks == 2  # Chunks 2 and 3 were never generated!
    assert guard.last_summary is not None
    assert guard.last_summary.aborted_mid_stream is True
    assert guard.last_summary.chars_yielded < guard.last_summary.chars_seen


def test_stream_guard_async_early_abort_and_aclose() -> None:
    backend = CountingRuleBackend()
    gate = LogitGate("No SQL destructive statements.", backend=backend, logit_temperature=1.0)

    aclosed = False
    produced = 0

    async def aupstream() -> AsyncIterator[dict[str, str]]:
        nonlocal aclosed, produced
        items = [
            {"response": "SELECT id, name FROM accounts WHERE active = 1; "},
            {"response": "DROP TABLE accounts; "},
            {"response": "SELECT * FROM audit_logs; "},
        ]
        try:
            for item in items:
                produced += 1
                yield item
        finally:
            aclosed = True

    async def _run() -> None:
        yielded: list[dict[str, str]] = []
        astream = aguard_stream(
            aupstream(),
            gate,
            min_chars=20,
            stride_chars=15,
            delivery="buffered",
        )
        with pytest.raises(StreamAbortedError) as exc_info:
            async for chunk in astream:
                yielded.append(chunk)  # noqa: PERF401 - loop preserves yielded items before exception
        assert exc_info.value.chunk_index == 1
        assert len(yielded) == 1
        assert "DROP TABLE" not in yielded[0]["response"]
        assert astream.summary is not None
        assert astream.summary.aborted_mid_stream is True

    asyncio.run(_run())
    assert aclosed is True
    assert produced == 2


def test_stream_guard_hysteresis_prevents_transient_false_alarm() -> None:
    backend = CountingRuleBackend(fail_substrings=("TRANSIENT_BAD",))

    class PrefixRecoveringGate(LogitGate):
        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            text = str(candidate)
            # Fails if 'DANGER' appears without 'RESOLVED_SAFELY'
            if "DANGER" in text and "RESOLVED_SAFELY" not in text:
                return super().evaluate("TRANSIENT_BAD", **kwargs)
            return super().evaluate("All good.", **kwargs)

    gate = PrefixRecoveringGate("Must resolve danger safely.", backend=backend, logit_temperature=1.0)
    chunks = [
        "Warning: DANGER detected in user query. ",
        "Request blocked and RESOLVED_SAFELY without executing action.",
    ]

    # With consecutive_failures=1, aborts prematurely on chunk 0:
    with pytest.raises(StreamAbortedError):
        list(guard_stream(chunks, gate, min_chars=10, stride_chars=10, consecutive_failures=1))

    # With consecutive_failures=2, survives transient chunk 0 and passes at chunk 1!
    handle = guard_stream(
        chunks,
        gate,
        min_chars=10,
        stride_chars=10,
        consecutive_failures=2,
        delivery="buffered",
    )
    out = "".join(handle)
    assert out == "".join(chunks)
    assert handle.summary is not None
    assert handle.summary.passed is True


def test_stream_guard_clause_boundary_openai_chunks_and_overflow() -> None:
    backend = CountingRuleBackend()
    gate = LogitGate("Safe content.", backend=backend, logit_temperature=1.0)

    # OpenAI ChatCompletionChunk-style objects
    def _mk_oai(txt: str | None) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=txt))])

    oai_chunks = [
        _mk_oai("word " * 10),  # 50 chars, NO clause terminator -> waits
        _mk_oai("still no punctuation " * 2),  # 92 chars, still < max_wait_chars -> waits
        _mk_oai("now ends sentence."),  # hits period -> checks once!
        _mk_oai(None),  # empty keepalive delta
    ]
    handle = guard_stream(
        oai_chunks,
        gate,
        min_chars=30,
        stride_chars=30,
        clause_boundary=True,
        max_wait_chars=200,
    )
    items = list(handle)
    assert len(items) == 4
    assert handle.summary is not None
    assert handle.summary.checks_executed == 1  # Checked at sentence end, deduplicated at EOF!

    # Buffer overflow protection with raise_on_abort=False
    ov_handle = guard_stream(
        ["x" * 60, "y" * 60],
        gate,
        max_buffer_chars=100,
        raise_on_abort=False,
    )
    yielded = list(ov_handle)
    assert yielded == ["x" * 60]
    assert ov_handle.summary is not None
    assert ov_handle.summary.aborted is True
    assert ov_handle.final_verdict is not None
    assert ov_handle.final_verdict.traces[0].failure_code == "STREAM_BUFFER_OVERFLOW"

    with pytest.raises(ConfigurationError):
        guard_stream("raw_string_not_chunk_iterable", gate)


# ---------------------------------------------------------------------------
# Part 2: VerdictCache (Memory LRU, SQLite WAL, Key Isolation, Normalization)
# ---------------------------------------------------------------------------


def test_verdict_cache_memory_lru_eviction_ttl_and_ambiguous_policy() -> None:
    backend = CountingRuleBackend()
    cache = VerdictCache(max_entries=2, ttl_seconds=0.08)
    gate = LogitGate(
        "No secret leaks.",
        backend=backend,
        logit_temperature=1.0,
        cache=cache,
    )

    v1 = gate.evaluate("Clean artifact A")
    assert v1.passed is True
    assert backend.logit_calls == 2

    # Warm hit: 0 additional backend calls!
    v1_hit = gate.evaluate("Clean artifact A")
    assert v1_hit.passed is True
    assert backend.logit_calls == 2
    assert v1_hit.traces[0].metadata["cache_hit"] is True
    assert cache.stats().hits == 1
    assert cache.stats().misses == 1

    # Deep copy check: mutating returned verdict does not mutate cached copy
    v1_hit.reason = "MUTATED"
    assert gate.evaluate("Clean artifact A").reason != "MUTATED"

    # AMBIGUOUS not cached by default
    v_amb = gate.evaluate("Contains AMBIGUOUS_TOKEN here")
    assert v_amb.status == DecisionStatus.AMBIGUOUS
    calls_after_amb1 = backend.logit_calls
    gate.evaluate("Contains AMBIGUOUS_TOKEN here")
    assert backend.logit_calls == calls_after_amb1 + 2

    # Eviction when exceeding max_entries=2
    gate.evaluate("Clean artifact B")
    gate.evaluate("Clean artifact C")  # Evicts oldest ("Clean artifact A")
    assert cache.stats().evictions >= 1
    calls_before = backend.logit_calls
    gate.evaluate("Clean artifact A")
    assert backend.logit_calls == calls_before + 2  # Recomputed after LRU eviction

    # TTL expiration
    time.sleep(0.10)
    calls_before_ttl = backend.logit_calls
    gate.evaluate("Clean artifact C")
    assert backend.logit_calls == calls_before_ttl + 2


def test_verdict_cache_sqlite_persistence_and_security_hardening(tmp_path: Path) -> None:
    db_path = tmp_path / "cache" / "verdicts.sqlite3"
    backend = CountingRuleBackend()

    with VerdictCache(path=db_path, max_disk_entries=10) as cache1:
        gate1 = LogitGate("No secret leaks.", backend=backend, logit_temperature=1.0, cache=cache1)
        jgate1 = JudgeGate(backend=backend, logit_temperature=1.0)
        cascade1 = Cascade([gate1, jgate1], cache=cache1)

        v_pass = cascade1.evaluate("Completely safe response.")
        v_amb_escalated = cascade1.evaluate("AMBIGUOUS_TOKEN but safe overall.")
        assert v_pass.passed is True
        assert v_amb_escalated.passed is True
        assert cache1.stats().disk_entries == 2

    # Permissions must be 0o600
    mode = stat.S_IMODE(db_path.stat().st_mode)
    assert mode == 0o600

    # New process/instance with cold memory hits directly from SQLite with 0 backend calls!
    backend2 = CountingRuleBackend()
    with VerdictCache(path=db_path) as cache2:
        gate2 = LogitGate("No secret leaks.", backend=backend2, logit_temperature=1.0)
        jgate2 = JudgeGate(backend=backend2, logit_temperature=1.0)
        cascade2 = Cascade([gate2, jgate2], cache=cache2)

        hit1 = cascade2.evaluate("Completely safe response.")
        hit2 = cascade2.evaluate("AMBIGUOUS_TOKEN but safe overall.")
        assert hit1.passed is True
        assert hit2.passed is True
        assert backend2.logit_calls == 0
        assert backend2.generate_calls == 0
        assert cache2.stats().hits == 2
        assert cache2.stats().saved_tokens == 12

        # Corrupt row eviction emits CacheWarning and falls back cleanly
        assert cache2._conn is not None
        some_key = next(iter(cache2._mem.keys()))
        cache2._mem.clear()
        cache2._conn.execute(
            "UPDATE verdict_cache SET verdict_json = ? WHERE cache_key = ?",
            ("{not valid json", some_key),
        )
        cache2._conn.commit()
        with pytest.warns(CacheWarning):
            recomputed = cascade2.evaluate("Completely safe response.")
        assert recomputed.passed is True
        assert backend2.logit_calls == 2

    # Security checks: reject ':memory:' and symlinks
    with pytest.raises(ConfigurationError):
        VerdictCache(path=":memory:")

    sym_file = tmp_path / "symlink_cache.db"
    sym_file.symlink_to(db_path)
    with pytest.raises(ConfigurationError, match="symlink"):
        VerdictCache(path=sym_file)

    sym_dir = tmp_path / "sym_dir"
    sym_dir.symlink_to(tmp_path / "cache")
    with pytest.raises(ConfigurationError, match="symlink"):
        VerdictCache(path=sym_dir / "nested.db")


def test_verdict_cache_key_isolation_and_all_gates_plus_normalized_artifacts(tmp_path: Path) -> None:
    backend = CountingRuleBackend()
    db_path = tmp_path / "all_gates.db"

    # Key isolation across AST validator bytecode, rubric, threshold, JudgeGate rubric_override, RankGate context
    ast_short = ASTGate(validators=[lambda s: len(s) > 3])
    ast_long = ASTGate(validators=[lambda s: len(s) > 20])
    assert compute_verdict_cache_key(ast_short, "hello") != compute_verdict_cache_key(ast_long, "hello")

    lg_a = LogitGate("Rubric 1", backend=backend, pass_threshold=0.80, logit_temperature=1.0)
    lg_b = LogitGate("Rubric 2", backend=backend, pass_threshold=0.80, logit_temperature=1.0)
    lg_c = LogitGate("Rubric 1", backend=backend, pass_threshold=0.90, logit_temperature=1.0)
    assert compute_verdict_cache_key(lg_a, "x") != compute_verdict_cache_key(lg_b, "x")
    assert compute_verdict_cache_key(lg_a, "x") != compute_verdict_cache_key(lg_c, "x")

    # JudgeGate rubric_override isolation
    with VerdictCache(path=db_path) as cache:
        jg = JudgeGate(rubric=None, backend=backend, logit_temperature=1.0, cache=cache)
        jg.evaluate("Candidate text", rubric_override="Check policy alpha")
        jg.evaluate("Candidate text", rubric_override="Check policy beta")
        assert backend.generate_calls == 2  # Distinct rubric_override -> must NOT collide!
        jg.evaluate("Candidate text", rubric_override="Check policy alpha")
        assert backend.generate_calls == 2  # Exact match -> cache hit!

        # ChoiceGate, ScoreGate, MultiRubricGate
        cg = ChoiceGate(
            "Classify intent",
            {"A": "Billing", "B": "Support", "C": "Sales"},
            backend=backend,
            logit_temperature=1.0,
            cache=cache,
        )
        sg = ScoreGate("Rate clarity", backend=backend, logit_temperature=1.0, cache=cache)
        mrg = MultiRubricGate(["No leaks", "Clear prose"], backend=backend, logit_temperature=1.0, cache=cache)

        assert cg.evaluate("Invoice question").choice == "A"
        calls_snap = backend.logit_calls
        assert cg.evaluate("Invoice question").choice == "A"
        assert backend.logit_calls == calls_snap

        assert sg.evaluate("Clear explanation").passed is True
        calls_snap = backend.logit_calls
        assert sg.evaluate("Clear explanation").passed is True
        assert backend.logit_calls == calls_snap

        assert mrg.evaluate("Safe clear prose").passed is True
        calls_snap = backend.logit_calls
        assert asyncio.run(mrg.aevaluate("Safe clear prose")).passed is True
        assert backend.logit_calls == calls_snap

        # RankGate context override isolation & BestOfNGate >> LogitGate _normalized_out preservation
        rg = RankGate("Pick safer SQL", backend=backend, logit_temperature=1.0, cache=cache)
        cands = ["SELECT 1;", "DROP TABLE users;"]
        rg.evaluate(cands, context="Read-only analytics")
        calls_snap = backend.logit_calls
        rg.evaluate(cands, context="Admin migration")  # Different context -> cache miss!
        assert backend.logit_calls > calls_snap

        # ASTGate(normalize=True) >> LogitGate preserves normalized code into _normalized_out across SQLite reload!
        norm_pipe = Cascade(
            [
                ASTGate(syntax=True, normalize=True),
                LogitGate("Valid Python", backend=backend, logit_temperature=1.0),
            ],
            cache=cache,
        )
        fenced = "<think>drafting</think>\n```python\ndef add(a, b):\n    return a + b\n```"
        box1: list[Any] = []
        v_norm1 = norm_pipe.evaluate(fenced, _normalized_out=box1)
        assert v_norm1.passed is True
        assert box1 == ["def add(a, b):\n    return a + b"]

        best_pipe = Cascade(
            [
                BestOfNGate("Pick best code", backend=backend, pre_gate=ASTGate(syntax=True), logit_temperature=1.0),
                LogitGate("No secret leaks", backend=backend, logit_temperature=1.0),
            ],
            cache=cache,
        )
        best_box1: list[Any] = []
        v_best1 = best_pipe.evaluate(["def ok():\n    return 1\n", "def bad(:\n"], _normalized_out=best_box1)
        assert v_best1.passed is True
        assert best_box1 == ["def ok():\n    return 1\n"]

    # Reopen from SQLite disk with cold RAM and verify _normalized_out is still populated!
    backend_cold = CountingRuleBackend()
    with VerdictCache(path=db_path) as cache_reloaded:
        norm_pipe2 = Cascade(
            [
                ASTGate(syntax=True, normalize=True),
                LogitGate("Valid Python", backend=backend_cold, logit_temperature=1.0),
            ],
            cache=cache_reloaded,
        )
        box2: list[Any] = []
        v_norm2 = norm_pipe2.evaluate(fenced, _normalized_out=box2)
        assert v_norm2.passed is True
        assert box2 == ["def add(a, b):\n    return a + b"]
        assert backend_cold.logit_calls == 0

        best_pipe2 = Cascade(
            [
                BestOfNGate("Pick best code", backend=backend_cold, pre_gate=ASTGate(syntax=True), logit_temperature=1.0),
                LogitGate("No secret leaks", backend=backend_cold, logit_temperature=1.0),
            ],
            cache=cache_reloaded,
        )
        best_box2: list[Any] = []
        v_best2 = best_pipe2.evaluate(["def ok():\n    return 1\n", "def bad(:\n"], _normalized_out=best_box2)
        assert v_best2.passed is True
        assert best_box2 == ["def ok():\n    return 1\n"]
        assert backend_cold.logit_calls == 0


def test_global_configure_cache_snapshot_bypass_and_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "global_cache.sqlite3"
    monkeypatch.setenv("LOGITGATE_CACHE_PATH", str(db_path))
    reset_config()

    backend = CountingRuleBackend()
    gate = LogitGate(
        "No secret leaks.",
        backend=backend,
        logit_temperature=1.0,
        save_on_fail=True,
    )
    snap_dir = tmp_path / "snaps"

    # Failing evaluation caches FAILED verdict AND saves snapshot
    v_fail = gate.evaluate("Leaking SECRET_LEAK now", snapshot_dir=snap_dir)
    assert v_fail.passed is False
    assert backend.logit_calls == 2

    # Second call hits global env-configured SQLite cache
    v_fail_hit = gate.evaluate("Leaking SECRET_LEAK now", save_on_fail=False)
    assert v_fail_hit.passed is False
    assert backend.logit_calls == 2

    # Snapshot replay & explain MUST bypass global cache and execute live!
    snap = VerdictSnapshot.load(snap_dir)
    snap.replay(backend=backend)
    assert backend.logit_calls == 4
    snap.explain(backend=backend)
    assert backend.generate_calls == 1
    assert backend.logit_calls == 6  # JudgeGate symmetric CoT debiasing ran 2 live logit passes

    # Explicit cache=False bypasses global cache
    gate.evaluate("Leaking SECRET_LEAK now", cache=False, save_on_fail=False)
    assert backend.logit_calls == 8

    # Multi-thread + asyncio concurrency stress on SQLite VerdictCache
    shared_cache = VerdictCache(path=tmp_path / "concurrent.db", max_entries=64)
    configure(cache=shared_cache)
    c_gate = LogitGate("No secret leaks.", backend=backend, logit_temperature=1.0)

    def _worker(i: int) -> bool:
        v = c_gate.evaluate(f"Concurrent payload {i % 12}")
        return v.passed

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(60)))
    assert all(results)

    async def _aworkers() -> list[Verdict]:
        return list(
            await asyncio.gather(
                *[c_gate.aevaluate(f"Concurrent payload {i % 12}") for i in range(40)]
            )
        )

    aresults = asyncio.run(_aworkers())
    assert all(v.passed for v in aresults)
    assert shared_cache.stats().hits >= 88
    shared_cache.close()


def test_cli_stream_eval_and_benchmark_with_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_file = tmp_path / "cli_cache.db"
    tel_file = tmp_path / "stream_tel.jsonl"

    # 1. logitgate stream --json
    rc = cli_main(
        [
            "stream",
            "--rubric",
            "Is the text compliant?",
            "--candidate",
            "First clean clause here. Second clean clause concludes.",
            "--chunk-size",
            "20",
            "--min-chars",
            "15",
            "--stride-chars",
            "20",
            "--delivery",
            "buffered",
            "--backend",
            "stub://default",
            "--cache",
            str(cache_file),
            "--telemetry-out",
            str(tel_file),
            "--json",
        ]
    )
    assert rc == 0
    stream_out = json.loads(capsys.readouterr().out)
    assert stream_out["passed"] is True
    assert stream_out["aborted"] is False
    assert "First clean clause" in stream_out["yielded_text"]
    assert tel_file.exists()

    # 2. logitgate eval --cache (populated on call 1, hit on call 2)
    eval_out: dict[str, Any] = {}
    for _ in range(2):
        rc_eval = cli_main(
            [
                "eval",
                "--rubric",
                "Is the text compliant?",
                "--candidate",
                "Cached CLI candidate",
                "--backend",
                "stub://default",
                "--cache",
                str(cache_file),
                "--json",
            ]
        )
        assert rc_eval == 0
        eval_out = json.loads(capsys.readouterr().out)
    assert eval_out["traces"][0]["metadata"].get("cache_hit") is True

    # 3. run_benchmark with cache=True across repeat=2
    rep = run_benchmark(
        BUILTIN_CASES[:4],
        backend=StubBackend(default_prob=0.95),
        modes=("logit",),
        repeat=2,
        warmup=1,
        cache=True,
    )
    assert rep.metadata["cache"]["hits"] == 4
    assert rep.metadata["cache"]["misses"] == 4
    assert sqlite3.sqlite_version  # sanity check stdlib sqlite3 active
    assert os.path.exists(cache_file)


def test_stream_guard_buffered_holds_ambiguous_until_verified() -> None:
    """Buffered delivery must NOT flush pending chunks on an AMBIGUOUS mid-stream checkpoint."""

    class AmbiguousThenFailGate(LogitGate):
        def __init__(self) -> None:
            super().__init__(
                "Check safety",
                backend=StubBackend(default_prob=0.50),
                pass_threshold=0.85,
                fail_threshold=0.20,
                logit_temperature=1.0,
            )
            self.calls = 0

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            self.calls += 1
            if self.calls == 1:
                return super().evaluate(candidate, **kwargs)  # prob=0.50 -> AMBIGUOUS
            fail_gate = LogitGate(
                "Check safety",
                backend=StubBackend(default_prob=0.02),
                pass_threshold=0.85,
                fail_threshold=0.20,
                logit_temperature=1.0,
            )
            return fail_gate.evaluate(candidate, **kwargs)  # prob=0.02 -> FAILED

    gate = AmbiguousThenFailGate()
    guard = StreamGuard(
        gate,
        min_chars=10,
        stride_chars=15,
        clause_boundary=False,
        delivery="buffered",
        raise_on_abort=False,
    )
    chunks = ["Borderline start. ", "Explicit violation payload."]
    yielded = list(guard.stream(chunks))
    assert yielded == []
    assert guard.last_summary is not None
    assert guard.last_summary.chars_yielded == 0
    assert guard.last_summary.aborted is True

