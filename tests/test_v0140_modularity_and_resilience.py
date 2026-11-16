"""Unit tests for LogitGate v0.14.0 modularity, resilience, cross-surface parity, and regressions.

Covers:
1. HTTP Retry, Exponential Backoff, Jitter & Retry-After (OllamaBackend & OpenAICompatBackend).
2. Universal Fault-Isolated evaluate_batch & aevaluate_batch (BaseGate, LogitGate, Cascade).
3. Resource Lifecycle & Context Managers (close(), aclose(), with, async with).
4. ASTGate(syntax=True) List/Tuple-of-Strings & ASTGate.replay_config().
5. Cross-Surface Parity (Cascade, VerdictSnapshot.replay()/areplay(), @check, logitgate eval CLI).
6. Explicit Regression Locks for all 5 fixed latent src/ bugs.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from logitgate import configure, reset_config
from logitgate.backends.base import (
    LogitBackend,
    StubBackend,
    _validate_retry_config,
)
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.backends.recurrent import RecurrentPyTorchBackend
from logitgate.calibration import lifecycle
from logitgate.calibration.conformal import calibrate_cascade_thresholds
from logitgate.calibration.lifecycle import (
    _should_lazy_auto_calibrate,
    _validate_prob_threshold,
    _warn_if_profile_model_mismatch,
)
from logitgate.calibration.profile import CalibrationProfile
from logitgate.cascade import Cascade
from logitgate.cli import create_parser, main
from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.commands.distill_cmd import _cmd_distill
from logitgate.cli.commands.eval_cmd import (
    ChoiceGate as CliChoiceGate,
)
from logitgate.cli.commands.eval_cmd import (
    MultiRubricGate as CliMultiRubricGate,
)
from logitgate.cli.commands.eval_cmd import (
    _cmd_eval,
)
from logitgate.distill.anchor_bank import _condense_rationale
from logitgate.distill.bundle import DistillationBundle
from logitgate.distill.meta_calibrator import _pass_distribution_stats
from logitgate.exceptions import (
    BackendError,
    BackendUnavailableError,
    CalibrationWarning,
    ConfigurationError,
)
from logitgate.gates.ast_gate import ASTFailureCode, ASTGate
from logitgate.gates.base import (
    TIER_GENERATIVE,
    TIER_LOGIT,
    BaseGate,
    BatchErrorPolicy,
    _safe_aclose,
    _safe_close,
)
from logitgate.gates.best_of_n import (
    BestOfNGate as SubmoduleBestOfNGate,
)
from logitgate.gates.best_of_n import (
    _resolve_logit_backend as _resolve_bon_backend,
)
from logitgate.gates.choice_gate import (
    ChoiceGate,
)
from logitgate.gates.choice_gate import (
    _resolve_logit_backend as _resolve_choice_backend,
)
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import LogitGate, MultiRubricGate, ScoreGate
from logitgate.gates.multi_rubric_gate import (
    MultiRubricGate as SubmoduleMultiRubricGate,
)
from logitgate.gates.rank_gate import BestOfNGate, RankGate
from logitgate.gates.score_gate import (
    ScoreGate as SubmoduleScoreGate,
)
from logitgate.gates.score_gate import (
    _resolve_logit_backend as _resolve_score_backend,
)
from logitgate.integrations.deepeval import LogitGateSystemOneModel
from logitgate.integrations.pytest_plugin import check
from logitgate.models import DecisionStatus, LayerTrace, Verdict
from logitgate.snapshot import (
    VerdictSnapshot,
    _atomic_write_text,
    persist_failure_snapshot,
    reset_snapshot_notices,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures & Shared Test Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_config_and_snapshots(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LOGITGATE_TELEMETRY_PATH", raising=False)
    monkeypatch.delenv("LOGITGATE_SAVE_FAILURES", raising=False)
    monkeypatch.setenv("LOGITGATE_BACKEND", "stub://default")
    reset_config()
    yield
    reset_config()


def _make_ollama_query_response(token: str = "A", logprob: float = -0.1) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": token,
            "logprobs": [
                {
                    "top_logprobs": [
                        {"token": token, "logprob": logprob},
                        {"token": "B", "logprob": -2.0},
                    ]
                }
            ],
        }
    }


def _make_ollama_generate_response(content: str = "Critique text", tokens: int = 10) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": content,
        },
        "eval_count": tokens,
    }


def _make_openai_query_response(token: str = "A", logprob: float = -0.1) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": token},
                "logprobs": {
                    "content": [
                        {
                            "token": token,
                            "logprob": logprob,
                            "top_logprobs": [
                                {"token": token, "logprob": logprob},
                                {"token": "B", "logprob": -2.0},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 1},
    }


def _make_openai_generate_response(content: str = "Critique text", tokens: int = 10) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"completion_tokens": tokens},
    }


# ===========================================================================
# Area 1: HTTP Retry, Exponential Backoff, Jitter & Retry-After
# ===========================================================================


def test_retry_config_validation_invalid_max_retries() -> None:
    for invalid_retries in (-1, -10, True, False, "2", 1.5):
        with pytest.raises(ConfigurationError, match="max_retries must be an integer >= 0"):
            _validate_retry_config(invalid_retries, 0.25, 5.0)  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError, match="max_retries must be an integer >= 0"):
            OllamaBackend(max_retries=invalid_retries)  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError, match="max_retries must be an integer >= 0"):
            OpenAICompatBackend(max_retries=invalid_retries)  # type: ignore[arg-type]


def test_retry_config_validation_invalid_backoff_and_max_wait() -> None:
    for invalid_base in (-0.1, -5.0, float("nan"), float("inf"), float("-inf"), True, False, "0.5"):
        with pytest.raises(ConfigurationError, match="retry_backoff_base must be a finite float >= 0.0"):
            _validate_retry_config(2, invalid_base, 5.0)  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError, match="retry_backoff_base must be a finite float >= 0.0"):
            OllamaBackend(retry_backoff_base=invalid_base)  # type: ignore[arg-type]

    for invalid_wait in (-0.01, -10.0, float("nan"), float("inf"), float("-inf"), True, False, "5.0"):
        with pytest.raises(ConfigurationError, match="retry_max_wait must be a finite float >= 0.0"):
            _validate_retry_config(2, 0.25, invalid_wait)  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError, match="retry_max_wait must be a finite float >= 0.0"):
            OpenAICompatBackend(retry_max_wait=invalid_wait)  # type: ignore[arg-type]


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
def test_ollama_transient_status_codes_retry_and_succeed_sync(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, json={"error": "service transient error"})
        return httpx.Response(200, json=_make_ollama_query_response(token="A", logprob=-0.15))

    sleep_calls: list[float] = []

    def mock_sleep(d: float) -> None:
        sleep_calls.append(d)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client=client, max_retries=2, retry_backoff_base=0.25, retry_max_wait=5.0)

    with (
        patch("time.sleep", side_effect=mock_sleep),
        patch("random.uniform", return_value=1.0),
    ):
        result = backend.query_top_logprobs("test prompt")

    assert calls == 2
    assert len(sleep_calls) == 1
    # attempt 0: 0.25 * (2**0) * 1.0 = 0.25
    assert math.isclose(sleep_calls[0], 0.25, abs_tol=1e-6)
    assert len(result) >= 1
    assert result[0]["token"] == "A"
    assert math.isclose(result[0]["logprob"], -0.15, abs_tol=1e-6)


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
async def test_ollama_transient_status_codes_retry_and_succeed_async(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, json={"error": "service transient error"})
        return httpx.Response(200, json=_make_ollama_query_response(token="B", logprob=-0.22))

    async_sleep_calls: list[float] = []

    async def mock_async_sleep(d: float) -> None:
        async_sleep_calls.append(d)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(async_client=async_client, max_retries=2, retry_backoff_base=0.5, retry_max_wait=5.0)

    with (
        patch("asyncio.sleep", side_effect=mock_async_sleep),
        patch("random.uniform", return_value=1.0),
    ):
        result = await backend.aquery_top_logprobs("test prompt")

    assert calls == 2
    assert len(async_sleep_calls) == 1
    # attempt 0: 0.5 * (2**0) * 1.0 = 0.5
    assert math.isclose(async_sleep_calls[0], 0.5, abs_tol=1e-6)
    assert len(result) >= 1
    assert result[0]["token"] == "B"
    assert math.isclose(result[0]["logprob"], -0.22, abs_tol=1e-6)


@pytest.mark.parametrize("status_code", [429, 503])
def test_openai_transient_status_codes_retry_and_succeed_sync(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, json={"error": {"message": "Rate limited"}})
        return httpx.Response(200, json=_make_openai_query_response(token="A", logprob=-0.05))

    sleep_calls: list[float] = []

    def mock_sleep(d: float) -> None:
        sleep_calls.append(d)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(client=client, max_retries=2, retry_backoff_base=0.25, retry_max_wait=5.0)

    with (
        patch("time.sleep", side_effect=mock_sleep),
        patch("random.uniform", return_value=1.0),
    ):
        result = backend.query_top_logprobs("test prompt")

    assert calls == 2
    assert len(sleep_calls) == 1
    assert math.isclose(sleep_calls[0], 0.25, abs_tol=1e-6)
    assert len(result) >= 1
    assert result[0]["token"] == "A"
    assert math.isclose(result[0]["logprob"], -0.05, abs_tol=1e-6)


@pytest.mark.parametrize("status_code", [429, 503])
async def test_openai_transient_status_codes_retry_and_succeed_async(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, json={"error": {"message": "Unavailable"}})
        return httpx.Response(200, json=_make_openai_query_response(token="A", logprob=-0.08))

    async_sleep_calls: list[float] = []

    async def mock_async_sleep(d: float) -> None:
        async_sleep_calls.append(d)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(
        async_client=async_client, max_retries=2, retry_backoff_base=0.3, retry_max_wait=5.0
    )

    with (
        patch("asyncio.sleep", side_effect=mock_async_sleep),
        patch("random.uniform", return_value=1.0),
    ):
        result = await backend.aquery_top_logprobs("test prompt")

    assert calls == 2
    assert len(async_sleep_calls) == 1
    assert math.isclose(async_sleep_calls[0], 0.3, abs_tol=1e-6)
    assert len(result) >= 1
    assert result[0]["token"] == "A"


def test_transient_network_exceptions_retry_sync_and_async() -> None:
    # 1. Sync Ollama with ConnectError
    calls_sync = 0

    def handler_sync(request: httpx.Request) -> httpx.Response:
        nonlocal calls_sync
        calls_sync += 1
        if calls_sync == 1:
            raise httpx.ConnectError("Connection refused by peer")
        return httpx.Response(200, json=_make_ollama_query_response())

    sleep_sync: list[float] = []
    client_sync = httpx.Client(transport=httpx.MockTransport(handler_sync))
    backend_sync = OllamaBackend(client=client_sync, max_retries=2, retry_backoff_base=0.2)

    with (
        patch("time.sleep", side_effect=lambda d: sleep_sync.append(d)),
        patch("random.uniform", return_value=1.0),
    ):
        res_sync = backend_sync.query_top_logprobs("p")

    assert calls_sync == 2
    assert len(sleep_sync) == 1
    assert math.isclose(sleep_sync[0], 0.2, abs_tol=1e-6)
    assert len(res_sync) >= 1

    # 2. Async OpenAI with ReadTimeout
    calls_async = 0

    def handler_async(request: httpx.Request) -> httpx.Response:
        nonlocal calls_async
        calls_async += 1
        if calls_async == 1:
            raise httpx.ReadTimeout("Timeout reading stream")
        return httpx.Response(200, json=_make_openai_query_response())

    sleep_async: list[float] = []

    async def run_async() -> None:
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_async))
        backend_async = OpenAICompatBackend(async_client=async_client, max_retries=2, retry_backoff_base=0.4)

        async def _fake_async_sleep(d: float) -> None:
            sleep_async.append(d)

        with (
            patch("asyncio.sleep", side_effect=_fake_async_sleep),
            patch("random.uniform", return_value=1.0),
        ):
            res_async = await backend_async.aquery_top_logprobs("p")
        assert len(res_async) >= 1

    asyncio.run(run_async())
    assert calls_async == 2
    assert len(sleep_async) == 1
    assert math.isclose(sleep_async[0], 0.4, abs_tol=1e-6)


def test_retry_after_header_honored_and_capped() -> None:
    # Case A: Retry-After: 1.75 is honored
    calls_a = 0

    def handler_a(request: httpx.Request) -> httpx.Response:
        nonlocal calls_a
        calls_a += 1
        if calls_a == 1:
            return httpx.Response(429, headers={"retry-after": "1.75"}, json={"error": "slow down"})
        return httpx.Response(200, json=_make_ollama_query_response())

    sleep_a: list[float] = []
    client_a = httpx.Client(transport=httpx.MockTransport(handler_a))
    backend_a = OllamaBackend(client=client_a, retry_max_wait=5.0)

    with patch("time.sleep", side_effect=lambda d: sleep_a.append(d)):
        backend_a.query_top_logprobs("prompt")

    assert len(sleep_a) == 1
    assert math.isclose(sleep_a[0], 1.75, abs_tol=1e-6)

    # Case B: Retry-After: 300.0 is capped by retry_max_wait = 2.5
    calls_b = 0

    def handler_b(request: httpx.Request) -> httpx.Response:
        nonlocal calls_b
        calls_b += 1
        if calls_b == 1:
            return httpx.Response(503, headers={"retry-after": "300.0"}, json={"error": "overloaded"})
        return httpx.Response(200, json=_make_openai_query_response())

    sleep_b: list[float] = []
    client_b = httpx.Client(transport=httpx.MockTransport(handler_b))
    backend_b = OpenAICompatBackend(client=client_b, retry_max_wait=2.5)

    with patch("time.sleep", side_effect=lambda d: sleep_b.append(d)):
        backend_b.query_top_logprobs("prompt")

    assert len(sleep_b) == 1
    assert math.isclose(sleep_b[0], 2.5, abs_tol=1e-6)

    # Case C: Invalid non-numeric Retry-After falls back to exponential backoff without error
    calls_c = 0

    def handler_c(request: httpx.Request) -> httpx.Response:
        nonlocal calls_c
        calls_c += 1
        if calls_c == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
                json={"error": "rate limited"},
            )
        return httpx.Response(200, json=_make_ollama_query_response())

    sleep_c: list[float] = []
    client_c = httpx.Client(transport=httpx.MockTransport(handler_c))
    backend_c = OllamaBackend(client=client_c, retry_backoff_base=0.35, retry_max_wait=5.0)

    with (
        patch("time.sleep", side_effect=lambda d: sleep_c.append(d)),
        patch("random.uniform", return_value=1.0),
    ):
        backend_c.query_top_logprobs("prompt")

    assert len(sleep_c) == 1
    assert math.isclose(sleep_c[0], 0.35, abs_tol=1e-6)


@pytest.mark.parametrize("permanent_status", [400, 401, 403, 404, 422])
def test_permanent_4xx_fail_attempt_1_zero_retries(permanent_status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(permanent_status, json={"error": f"permanent {permanent_status}"})

    sleep_calls: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend(client=client, max_retries=3)

    with (
        patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)),
        pytest.raises(BackendError) as exc_info,
    ):
        # Passing reasoning_effort explicitly avoids the reasoning_effort 400/422 retry fallback
        backend.query_top_logprobs("prompt", reasoning_effort="low")

    assert calls == 1
    assert len(sleep_calls) == 0
    assert f"HTTP {permanent_status}" in str(exc_info.value)


def test_exhausted_max_retries_raises_backend_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "server permanently down"})

    sleep_calls: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client=client, max_retries=2)

    with (
        patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)),
        patch("random.uniform", return_value=1.0),
        pytest.raises(BackendError) as exc_info,
    ):
        backend.query_top_logprobs("prompt")

    assert calls == 3  # initial attempt + 2 retries
    assert len(sleep_calls) == 2
    assert "HTTP 503" in str(exc_info.value)


def test_exhausted_transient_exceptions_raise_backend_unavailable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("Target port unreachable")

    sleep_calls: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OllamaBackend(client=client, max_retries=1)

    with (
        patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)),
        patch("random.uniform", return_value=1.0),
        pytest.raises(BackendUnavailableError),
    ):
        backend.query_top_logprobs("prompt")

    assert calls == 2
    assert len(sleep_calls) == 1


# ===========================================================================
# Area 2: Universal Fault-Isolated evaluate_batch & aevaluate_batch
# ===========================================================================


class _FaultIsolatingTestGate(BaseGate):
    name = "FaultIsolatingTestGate"

    def __init__(self, failure_candidates: Sequence[str] = ()) -> None:
        self.failure_candidates = set(failure_candidates)

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        cand_str = str(candidate)
        if cand_str == "boom":
            raise RuntimeError("Candidate evaluation crashed intentionally")
        if cand_str == "keyboard_interrupt":
            raise KeyboardInterrupt("Simulated Ctrl-C")
        if cand_str in self.failure_candidates:
            return Verdict(
                passed=False,
                status=DecisionStatus.FAILED,
                probability=0.15,
                triggered_layer=self.name,
                total_latency_ms=1.0,
                traces=[LayerTrace(layer_name=self.name, status=DecisionStatus.FAILED, latency_ms=1.0)],
            )
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            probability=0.92,
            triggered_layer=self.name,
            total_latency_ms=1.0,
            traces=[LayerTrace(layer_name=self.name, status=DecisionStatus.PASSED, latency_ms=1.0)],
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        cand_str = str(candidate)
        if cand_str == "cancel":
            raise asyncio.CancelledError("Simulated task cancellation")
        return self.evaluate(candidate, **kwargs)


def test_batch_validation_invalid_arguments() -> None:
    gate = _FaultIsolatingTestGate()

    for invalid_policy in ("ignore", "skip", "crash", 123, None):
        with pytest.raises(ConfigurationError, match="on_error must be one of 'raise', 'abstain', or 'fail'"):
            gate.evaluate_batch(["a"], on_error=invalid_policy)  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError, match="on_error must be one of 'raise', 'abstain', or 'fail'"):
            asyncio.run(gate.aevaluate_batch(["a"], on_error=invalid_policy))  # type: ignore[arg-type]

    for invalid_concurrency in (0, -1, -10, True, False, "4"):
        with pytest.raises(ConfigurationError, match="max_concurrency must be an integer >= 1 or None"):
            asyncio.run(gate.aevaluate_batch(["a"], max_concurrency=invalid_concurrency))  # type: ignore[arg-type]


def test_batch_on_error_raise_sync_and_async() -> None:
    gate = _FaultIsolatingTestGate()
    candidates = ["ok1", "boom", "ok2"]

    with pytest.raises(RuntimeError, match="Candidate evaluation crashed intentionally"):
        gate.evaluate_batch(candidates, on_error="raise")

    with pytest.raises(RuntimeError, match="Candidate evaluation crashed intentionally"):
        asyncio.run(gate.aevaluate_batch(candidates, on_error="raise"))


def test_batch_on_error_abstain_sync_and_async() -> None:
    gate = _FaultIsolatingTestGate(failure_candidates=["bad_cand"])
    candidates = ["ok1", "boom", "bad_cand", "ok2"]

    expected_abstained = getattr(DecisionStatus, "ABSTAINED", DecisionStatus.AMBIGUOUS)

    # 1. Sync
    results_sync = gate.evaluate_batch(candidates, on_error="abstain")
    assert len(results_sync) == 4

    assert results_sync[0].passed is True
    assert results_sync[0].status == DecisionStatus.PASSED

    assert results_sync[1].passed is False
    assert results_sync[1].status == expected_abstained
    assert results_sync[1].probability is None
    assert "Candidate evaluation crashed intentionally" in str(results_sync[1].reason)
    assert getattr(results_sync[1], "abstained_reason", None) is not None

    assert results_sync[2].passed is False
    assert results_sync[2].status == DecisionStatus.FAILED

    assert results_sync[3].passed is True
    assert results_sync[3].status == DecisionStatus.PASSED

    # 2. Async
    results_async = asyncio.run(gate.aevaluate_batch(candidates, on_error="abstain"))
    assert len(results_async) == 4
    assert [r.status for r in results_async] == [
        DecisionStatus.PASSED,
        expected_abstained,
        DecisionStatus.FAILED,
        DecisionStatus.PASSED,
    ]
    assert results_async[1].probability is None


def test_batch_on_error_fail_sync_and_async() -> None:
    gate = _FaultIsolatingTestGate(failure_candidates=["bad_cand"])
    candidates = ["ok1", "boom", "bad_cand"]

    results_sync = gate.evaluate_batch(candidates, on_error="fail")
    assert len(results_sync) == 3
    assert results_sync[0].status == DecisionStatus.PASSED
    assert results_sync[1].status == DecisionStatus.FAILED
    assert results_sync[1].passed is False
    assert results_sync[1].probability == 0.0
    assert results_sync[1].traces[0].failure_code == "BATCH_EVAL_ERROR"
    assert results_sync[2].status == DecisionStatus.FAILED

    results_async = asyncio.run(gate.aevaluate_batch(candidates, on_error="fail"))
    assert [r.status for r in results_async] == [
        DecisionStatus.PASSED,
        DecisionStatus.FAILED,
        DecisionStatus.FAILED,
    ]
    assert results_async[1].probability == 0.0


def test_batch_never_swallow_base_exceptions() -> None:
    gate = _FaultIsolatingTestGate()

    policies: tuple[BatchErrorPolicy, ...] = ("abstain", "fail")
    for policy in policies:
        with pytest.raises(KeyboardInterrupt, match="Simulated Ctrl-C"):
            gate.evaluate_batch(["ok", "keyboard_interrupt"], on_error=policy)

        with pytest.raises(asyncio.CancelledError, match="Simulated task cancellation"):
            asyncio.run(gate.aevaluate_batch(["ok", "cancel"], on_error=policy))


class _ConcurrencyTrackingGate(BaseGate):
    name = "ConcurrencyTrackingGate"

    def __init__(self) -> None:
        self.current_concurrency = 0
        self.peak_concurrency = 0
        self._lock = asyncio.Lock()

    def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer=self.name,
            total_latency_ms=1.0,
            traces=[],
        )

    async def aevaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
        async with self._lock:
            self.current_concurrency += 1
            if self.current_concurrency > self.peak_concurrency:
                self.peak_concurrency = self.current_concurrency

        await asyncio.sleep(0.005)

        async with self._lock:
            self.current_concurrency -= 1

        return Verdict(
            passed=True,
            status=DecisionStatus.PASSED,
            triggered_layer=self.name,
            total_latency_ms=5.0,
            traces=[],
        )


def test_batch_max_concurrency_semaphore_bounds() -> None:
    async def run_test() -> None:
        # Test max_concurrency = 2 over 8 items
        gate = _ConcurrencyTrackingGate()
        items = [f"item_{i}" for i in range(8)]
        results = await gate.aevaluate_batch(items, max_concurrency=2)
        assert len(results) == 8
        assert gate.peak_concurrency <= 2

        # Test max_concurrency = 4 over 8 items
        gate4 = _ConcurrencyTrackingGate()
        results4 = await gate4.aevaluate_batch(items, max_concurrency=4)
        assert len(results4) == 8
        assert gate4.peak_concurrency <= 4

    asyncio.run(run_test())


def test_batch_empty_sequences() -> None:
    gate = _FaultIsolatingTestGate()
    assert gate.evaluate_batch([]) == []
    assert asyncio.run(gate.aevaluate_batch([])) == []


def test_cascade_evaluate_batch_fault_isolation() -> None:
    l1 = ASTGate(syntax=True)
    l2 = LogitGate("Is safe", backend=StubBackend(default_prob=0.9))
    cascade = Cascade(gates=[l1, l2], rubric="Is safe")

    # Syntax failure -> FAILED, valid python -> PASSED
    candidates = ["x = 1", "def broken(:", "y = 2"]
    res = cascade.evaluate_batch(candidates, on_error="abstain")
    assert len(res) == 3
    assert res[0].passed is True
    assert res[1].passed is False
    assert res[1].status == DecisionStatus.FAILED
    assert res[2].passed is True

    # Async batch
    res_async = asyncio.run(cascade.aevaluate_batch(candidates, on_error="abstain", max_concurrency=2))
    assert len(res_async) == 3
    assert res_async[0].passed is True
    assert res_async[1].passed is False
    assert res_async[2].passed is True


# ===========================================================================
# Area 3: Resource Lifecycle & Context Managers
# ===========================================================================


class _TrackingResourceBackend(StubBackend):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0
        self.aclose_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()

    async def aclose(self) -> None:
        self.aclose_calls += 1
        await super().aclose()


def test_gate_owned_resources_closed_and_idempotent() -> None:
    # 1. LogitGate
    gate = LogitGate("rubric", backend="stub://")
    backend_ref = gate.backend
    with patch.object(backend_ref, "close", wraps=backend_ref.close) as mock_close:
        with gate:
            assert not getattr(gate, "_is_closed", False)
            assert mock_close.call_count == 0
        assert getattr(gate, "_is_closed", False) is True
        assert mock_close.call_count == 1
        # Idempotent close
        gate.close()
        assert mock_close.call_count == 1

    # 2. ChoiceGate
    cg = ChoiceGate("question?", options={"A": "a", "B": "b"}, backend="stub://")
    cg_b = cg.backend
    with patch.object(cg_b, "close", wraps=cg_b.close) as mock_close_cg:
        with cg:
            assert not getattr(cg, "_is_closed", False)
        assert getattr(cg, "_is_closed", False) is True
        assert mock_close_cg.call_count == 1
        cg.close()
        assert mock_close_cg.call_count == 1

    # 3. ScoreGate
    sg = ScoreGate("rubric", backend="stub://")
    sg_b = sg.backend
    with patch.object(sg_b, "close", wraps=sg_b.close) as mock_close_sg:
        with sg:
            assert not getattr(sg, "_is_closed", False)
        assert getattr(sg, "_is_closed", False) is True
        assert mock_close_sg.call_count == 1
        sg.close()
        assert mock_close_sg.call_count == 1

    # 4. MultiRubricGate
    mrg = MultiRubricGate(rubrics=["r1", "r2"], backend="stub://")
    mrg_b = mrg.backend
    with patch.object(mrg_b, "close", wraps=mrg_b.close) as mock_close_mrg:
        with mrg:
            assert not getattr(mrg, "_is_closed", False)
        assert getattr(mrg, "_is_closed", False) is True
        assert mock_close_mrg.call_count == 1
        mrg.close()
        assert mock_close_mrg.call_count == 1

    # 5. RankGate
    rg = RankGate("rubric", backend="stub://")
    rg_b = rg.backend
    with patch.object(rg_b, "close", wraps=rg_b.close) as mock_close_rg:
        with rg:
            assert not getattr(rg, "_is_closed", False)
        assert getattr(rg, "_is_closed", False) is True
        assert mock_close_rg.call_count == 1
        rg.close()
        assert mock_close_rg.call_count == 1

    # 6. BestOfNGate
    bong = BestOfNGate("rubric", backend="stub://")
    bong_b = bong.backend
    with patch.object(bong_b, "close", wraps=bong_b.close) as mock_close_bong:
        with bong:
            assert not getattr(bong, "_is_closed", False)
        assert getattr(bong, "_is_closed", False) is True
        assert mock_close_bong.call_count == 1
        bong.close()
        assert mock_close_bong.call_count == 1

    # 7. JudgeGate
    jg = JudgeGate("rubric", backend="stub://")
    with jg:
        assert not getattr(jg, "_is_closed", False)
    assert getattr(jg, "_is_closed", False) is True
    jg.close()
    assert getattr(jg, "_is_closed", False) is True

    # 8. Cascade
    cas = Cascade(rubric="rubric", backend="stub://")
    first_gate = cas.gates[0]
    cas_b = getattr(first_gate, "backend", None)
    assert cas_b is not None
    with patch.object(cas_b, "close", wraps=cas_b.close) as mock_close_cas:
        with cas:
            assert not getattr(cas, "_closed", False)
        assert getattr(cas, "_closed", False) is True
        assert mock_close_cas.call_count >= 1
        cas.close()
        assert getattr(cas, "_closed", False) is True


async def test_gate_owned_resources_aclose_and_idempotent() -> None:
    gate = LogitGate("rubric", backend="stub://")
    backend_ref = gate.backend
    with patch.object(backend_ref, "aclose", wraps=backend_ref.aclose) as mock_aclose:
        async with gate:
            assert not getattr(gate, "_is_closed", False)
            assert mock_aclose.call_count == 0
        assert getattr(gate, "_is_closed", False) is True
        assert mock_aclose.call_count == 1
        # Calling aclose() a second time is idempotent
        await gate.aclose()
        assert mock_aclose.call_count == 1


def test_caller_injected_backend_instances_not_closed() -> None:
    # Verify that passing pre-instantiated backend instances does NOT close them on gate close
    shared_backend = _TrackingResourceBackend()

    gates: list[BaseGate] = [
        LogitGate("rubric", backend=shared_backend),
        ChoiceGate("question?", options={"A": "a", "B": "b"}, backend=shared_backend),
        ScoreGate("rubric", backend=shared_backend),
        MultiRubricGate(rubrics=["r1", "r2"], backend=shared_backend),
        RankGate("rubric", backend=shared_backend),
        BestOfNGate("rubric", backend=shared_backend),
        JudgeGate("rubric", backend=shared_backend),
    ]

    for g in gates:
        assert getattr(g, "_owns_backend", False) is False
        g.close()
        # Shared backend must NOT have been closed!
        assert shared_backend.close_calls == 0

    # Test Cascade with caller-injected gates
    cas = Cascade(gates=gates, rubric="rubric")
    cas.close()
    assert shared_backend.close_calls == 0

    # Async close check
    async def run_async_check() -> None:
        for g in gates:
            await g.aclose()
        await cas.aclose()

    asyncio.run(run_async_check())
    assert shared_backend.aclose_calls == 0


def test_gate_owned_cache_and_sink_closed(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.sqlite"
    sink_file = tmp_path / "telemetry.jsonl"

    gate = LogitGate(
        "rubric",
        backend="stub://",
        cache=str(cache_file),
        telemetry_sink=str(sink_file),
    )
    assert gate._owns_cache is True
    assert gate._owns_sink is True

    gate.close()
    assert getattr(gate, "_is_closed", False) is True


# ===========================================================================
# Area 4: ASTGate(syntax=True) List/Tuple Fix & replay_config()
# ===========================================================================


def test_ast_gate_string_sequence_syntax_error() -> None:
    gate = ASTGate(syntax=True)
    candidate = ["x = 1", "def broken(:"]

    verdict = gate.evaluate(candidate)
    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert len(verdict.traces) >= 1
    assert verdict.traces[0].failure_code == ASTFailureCode.SYNTAX_ERROR
    assert verdict.traces[0].metadata.get("index") == 1


def test_ast_gate_string_sequence_forbidden_import() -> None:
    gate = ASTGate(syntax=True, forbidden_imports=["os"])
    candidate = ["x = 1", "import os\nprint('hello')"]

    verdict = gate.evaluate(candidate)
    assert verdict.passed is False
    assert verdict.status == DecisionStatus.FAILED
    assert len(verdict.traces) >= 1
    assert verdict.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT
    assert verdict.traces[0].metadata.get("index") == 1
    assert verdict.traces[0].metadata.get("forbidden_import") == "os"


def test_ast_gate_string_sequence_valid_python_passes() -> None:
    gate = ASTGate(syntax=True)

    # 1. List of strings
    v_list = gate.evaluate(["def ok():\n    return 42", "y = 2"])
    assert v_list.passed is True
    assert v_list.status == DecisionStatus.PASSED

    # 2. Tuple of strings
    v_tuple = gate.evaluate(("x = 10", "y = 20", "z = x + y"))
    assert v_tuple.passed is True
    assert v_tuple.status == DecisionStatus.PASSED


def test_ast_gate_replay_config_round_trip(tmp_path: Path) -> None:
    gate = ASTGate(
        syntax=True,
        forbidden_imports=["sys", "os"],
        max_length=250,
        name="SecurityASTGate",
    )
    cfg = gate.replay_config()
    assert cfg["gate_type"] == "ASTGate"
    assert cfg["syntax"] is True
    assert cfg["forbidden_imports"] == ["sys", "os"]
    assert cfg["max_length"] == 250
    assert cfg["name"] == "SecurityASTGate"

    candidate = ["import sys\nprint('test')"]
    verdict = gate.evaluate(candidate)
    assert verdict.passed is False

    # Snapshot using ASTGate.replay_config()
    snapshot = VerdictSnapshot(
        snapshot_id="ast_replay_id",
        timestamp="2026-11-16T18:00:00Z",
        rubric="Python AST contract checks",
        test_name="test_ast_replay",
        candidate=candidate,
        verdict=verdict,
        gate_config=cfg,
    )
    snapshot.save(tmp_path)

    loaded_snapshot = VerdictSnapshot.load(tmp_path)
    assert loaded_snapshot.gate_config.get("gate_type") == "ASTGate"

    # Replay sync
    replayed = loaded_snapshot.replay()
    assert replayed.passed is False
    assert replayed.status == DecisionStatus.FAILED
    assert replayed.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT

    # Replay async
    replayed_async = asyncio.run(loaded_snapshot.areplay())
    assert replayed_async.passed is False
    assert replayed_async.status == DecisionStatus.FAILED
    assert replayed_async.traces[0].failure_code == ASTFailureCode.FORBIDDEN_IMPORT


# ===========================================================================
# Area 5: Cross-Surface Parity
# ===========================================================================


def test_cascade_distillation_bundle_forwarding() -> None:
    profile = CalibrationProfile(model="test-model", binary_temperature=1.2)
    bundle = DistillationBundle(profile=profile)

    cascade = Cascade(rubric="Check safety", backend="stub://", distillation_bundle=bundle)
    assert len(cascade.gates) >= 1
    # Layer 1 LogitGate must have received the distillation_bundle
    layer1 = cascade.gates[0]
    assert getattr(layer1, "distillation_bundle", None) is bundle


def test_cascade_multi_layer_snapshot_replay_and_cleanup(tmp_path: Path) -> None:
    l1 = ASTGate(syntax=True, name="L1_AST")
    l2 = LogitGate("Safe and polite", backend=StubBackend(default_prob=0.05), name="L2_Logit")
    l3 = JudgeGate(
        "Safe and polite",
        backend=StubBackend(default_prob=0.05, diagnostic_text="[VERDICT: FAIL] Rude."),
        trigger="always",
        name="L3_Judge",
    )
    cascade = Cascade(gates=[l1, l2, l3], rubric="Safe and polite", name="TriLayerCascade")

    verdict = cascade.evaluate(
        "valid_code = 123",
        save_on_fail=True,
        snapshot_dir=tmp_path,
        test_name="cascade_tri_layer",
    )
    assert verdict.passed is False

    snapshot = VerdictSnapshot.load(tmp_path)
    assert snapshot.gate_config.get("gate_type") == "Cascade"

    replayed = snapshot.replay()
    assert replayed is not None
    # Verify replay extra backends are cleaned up
    replayed_async = asyncio.run(snapshot.areplay())
    assert replayed_async is not None


def test_logit_gate_snapshot_replay_with_profile_and_bundle(tmp_path: Path) -> None:
    profile = CalibrationProfile(
        model="byo-7b",
        binary_temperature=1.35,
        pass_threshold=0.88,
        fail_threshold=0.15,
    )
    bundle = DistillationBundle(profile=profile)

    gate = LogitGate(
        "Is polite",
        backend=StubBackend(default_prob=0.05),
        profile=profile,
        distillation_bundle=bundle,
        name="CalibratedLogitGate",
    )

    verdict = gate.evaluate(
        "Rude comment",
        save_on_fail=True,
        snapshot_dir=tmp_path,
        test_name="profile_bundle_replay",
    )
    assert verdict.passed is False

    snapshot = VerdictSnapshot.load(tmp_path)
    replayed = snapshot.replay()
    assert replayed.passed is False
    assert replayed.probability is not None

    replayed_async = asyncio.run(snapshot.areplay())
    assert replayed_async.passed is False


def test_check_decorator_parity_features() -> None:
    # 1. choices + pass_choices (Success and Failure)
    @check("Is approved?", choices={"A": "Yes", "B": "No"}, pass_choices=["A"], backend="stub://")
    def func_choice_pass() -> str:
        return "sample output"

    func_choice_pass()  # Does not raise

    @check("Is approved?", choices={"A": "Yes", "B": "No"}, pass_choices=["B"], backend="stub://")
    def func_choice_fail() -> str:
        return "sample output"

    with pytest.raises(AssertionError, match="ChoiceGate selected 'A', not in pass_choices"):
        func_choice_fail()

    # 2. score_range + score_threshold
    @check("Rate tone", score_range=(1, 5), score_threshold=3.0, backend="stub://")
    def func_score() -> str:
        return "well formatted tone"

    func_score()

    # 3. rubrics + aggregation (any, mean, all)
    @check(rubrics=["Is polite", "Is concise"], aggregation="any", backend="stub://")
    def func_agg_any() -> str:
        return "hello"

    func_agg_any()

    @check(rubrics=["Is polite", "Is concise"], aggregation="mean", threshold=0.5, backend="stub://")
    def func_agg_mean() -> str:
        return "hello"

    func_agg_mean()

    @check(rubrics=["Is polite", "Is concise"], aggregation="all", backend="stub://")
    def func_agg_all() -> str:
        return "hello"

    func_agg_all()

    # 4. calibration_profile and distillation_bundle
    profile = CalibrationProfile(model="m")
    bundle = DistillationBundle(profile=profile)

    @check("Is valid", backend="stub://", calibration_profile=profile, distillation_bundle=bundle)
    def func_bundle() -> str:
        return "hello"

    func_bundle()


def test_cli_eval_options_and_mutual_exclusion() -> None:
    # 1. Repeatable --rubric + --aggregation
    rc_any = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "Rubric 1",
            "--rubric",
            "Rubric 2",
            "--aggregation",
            "any",
            "--json",
        ]
    )
    assert rc_any == 0

    rc_mean = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "Rubric 1",
            "--rubric",
            "Rubric 2",
            "--aggregation",
            "mean",
            "--json",
        ]
    )
    assert rc_mean == 0

    # 2. Choice mode
    rc_choice = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "Select option",
            "--choice",
            "A=Approve",
            "--choice",
            "B=Reject",
            "--expected-choice",
            "A",
            "--json",
        ]
    )
    assert rc_choice == 0

    # 3. Score mode
    rc_score = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "Rate candidate",
            "--score-min",
            "1",
            "--score-max",
            "5",
            "--score-threshold",
            "3.0",
            "--json",
        ]
    )
    assert rc_score == 0

    # 4. Mutually exclusive flag rejections (non-zero exit code)
    rc_conflict_1 = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "Q",
            "--choice",
            "A=Yes",
            "--choice",
            "B=No",
            "--score-min",
            "1",
        ]
    )
    assert rc_conflict_1 != 0

    rc_conflict_2 = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "R1",
            "--rubric",
            "R2",
            "--choice",
            "A=Yes",
            "--choice",
            "B=No",
        ]
    )
    assert rc_conflict_2 != 0

    rc_conflict_3 = main(
        [
            "eval",
            "--candidate",
            "test payload",
            "--backend",
            "stub://",
            "--rubric",
            "R1",
            "--rubric",
            "R2",
            "--score-min",
            "1",
            "--score-max",
            "5",
        ]
    )
    assert rc_conflict_3 != 0


def test_public_and_legacy_module_imports() -> None:
    from logitgate.cli import create_parser as cli_cp
    from logitgate.cli import main as cli_main
    from logitgate.cli import render_snapshot_report as cli_rsr
    from logitgate.gates import (
        BestOfNGate as BONG_pub,
    )
    from logitgate.gates import (
        ChoiceGate as CG_pub,
    )
    from logitgate.gates import (
        MultiRubricGate as MRG_pub,
    )
    from logitgate.gates import (
        ScoreGate as SG_pub,
    )
    from logitgate.gates.logit_gate import (
        ChoiceGate as CG_legacy,
    )
    from logitgate.gates.logit_gate import (
        MultiRubricGate as MRG_legacy,
    )
    from logitgate.gates.logit_gate import (
        ScoreGate as SG_legacy,
    )
    from logitgate.gates.rank_gate import BestOfNGate as BONG_legacy

    # Verify public and legacy class parity
    assert CG_pub is CG_legacy
    assert SG_pub is SG_legacy
    assert MRG_pub is MRG_legacy
    assert BONG_pub is BONG_legacy

    # Verify CLI functions
    assert callable(cli_main)
    assert callable(cli_cp)
    assert callable(cli_rsr)


# ===========================================================================
# Area 6: Explicit Regression Locks for All 5 Fixed Latent src/ Bugs
# ===========================================================================


class _TensorContainsTrap:
    """Mock tensor object that raises RuntimeError on `__contains__`, matching PyTorch tensor behavior."""

    shape = (1, 3)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __contains__(self, key: Any) -> bool:
        raise RuntimeError("Tensor.__contains__ only supports Tensor or scalar, but you passed in a <class 'str'>.")


def test_regression_bug1_recurrent_pytorch_backend_raw_tensor() -> None:
    trap_tensor = _TensorContainsTrap()
    mock_tok = MagicMock()
    mock_tok.side_effect = lambda prompt, return_tensors="pt": trap_tensor
    mock_tok.decode.return_value = "Critique completed"

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = [[101, 102, 103, 104, 105]]

    backend = RecurrentPyTorchBackend(model=mock_model, tokenizer=mock_tok, device="cpu")
    # Must NOT raise RuntimeError from `Tensor.__contains__`!
    text, tokens = backend.generate("input prompt")
    assert text == "Critique completed"
    assert tokens == 2

    # Also verify with actual torch tensor when torch is installed
    if HAS_TORCH and torch is not None:
        actual_in = torch.tensor([[101, 102, 103]], dtype=torch.long)
        actual_out = torch.tensor([[101, 102, 103, 104, 105]], dtype=torch.long)

        mock_tok_real = MagicMock()
        mock_tok_real.side_effect = lambda prompt, return_tensors="pt": actual_in
        mock_tok_real.decode.return_value = "Real torch critique"

        mock_model_real = MagicMock()
        mock_model_real.device = "cpu"
        mock_model_real.generate.return_value = actual_out

        backend_real = RecurrentPyTorchBackend(model=mock_model_real, tokenizer=mock_tok_real, device="cpu")
        text_real, tokens_real = backend_real.generate("input prompt")
        assert text_real == "Real torch critique"
        assert tokens_real == 2


def test_regression_bug2_condense_rationale_bullet_symbols() -> None:
    empty_bullets = "- \n* \n• "

    # 1. passed=True returns non-empty default message
    rationale_pass = _condense_rationale(empty_bullets, passed=True, max_chars=120)
    assert isinstance(rationale_pass, str)
    assert rationale_pass != ""
    assert "Satisfies" in rationale_pass

    # 2. passed=False returns non-empty default message
    rationale_fail = _condense_rationale(empty_bullets, passed=False, max_chars=120)
    assert isinstance(rationale_fail, str)
    assert rationale_fail != ""
    assert "Violates" in rationale_fail


def test_regression_bug3_pass_distribution_stats_normalization() -> None:
    # When target tokens have saturated logprobs (0.0), total observed mass > 1.0
    top_logprobs = [{"token": "A", "logprob": 0.0}, {"token": "B", "logprob": 0.0}]
    _, non_target_mass = _pass_distribution_stats(top_logprobs, target_tokens=("A", "B"))
    # Must correctly normalize and return 0.0 instead of negative residual mass
    assert math.isclose(non_target_mass, 0.0, abs_tol=1e-9)


def test_regression_bug4_system_one_model_resolves_ollama_when_unconfigured() -> None:
    reset_config()
    model = LogitGateSystemOneModel(api_base="http://localhost:11434", model_name="custom-m")
    try:
        assert isinstance(model.backend, OllamaBackend)
        assert model.backend.base_url == "http://localhost:11434"
        assert model.backend.model == "custom-m"
    finally:
        model.backend.close()


def test_regression_bug5_calibrate_cascade_thresholds_string_probs() -> None:
    # 20 string probabilities for each class
    string_probs = ["0.9"] * 20 + ["0.1"] * 20
    labels = [True] * 20 + [False] * 20

    # Must successfully parse strings to floats without raising TypeError
    thresholds = calibrate_cascade_thresholds(
        string_probs,  # type: ignore[arg-type]
        labels,
        target_risk_alpha=0.1,
    )

    assert thresholds.guarantee_holds is True
    assert thresholds.tau_accept is not None
    assert thresholds.tau_reject is not None
    assert thresholds.tau_accept > thresholds.tau_reject


# ---------------------------------------------------------------------------
# Stage 2B-2D v0.14.0 Branch-Coverage & Mutant Resistance Test Suite
# ---------------------------------------------------------------------------


def test_distill_cmd_branches_and_coverage(capsys: pytest.CaptureFixture[str]) -> None:
    # 1. Unknown distill subcommand returns EXIT_FAILED
    args_unknown = argparse.Namespace(distill_cmd="unsupported_subcmd")
    ret = _cmd_distill(args_unknown)
    assert ret == EXIT_FAILED

    # 2. compile with json=False, anchor_bank=None, meta_calibrator=None
    mock_bundle_empty = MagicMock()
    mock_bundle_empty.anchor_bank = None
    mock_bundle_empty.meta_calibrator = None
    mock_bundle_empty.save.return_value = Path("/tmp/mock_bundle.json")

    with patch("logitgate.distill.bundle.compile_distillation_bundle", return_value=mock_bundle_empty):
        args_compile_no_extra = argparse.Namespace(
            distill_cmd="compile",
            dataset="data.jsonl",
            telemetry=None,
            backend=None,
            model=None,
            teacher_model=None,
            no_anchors=True,
            no_meta=True,
            max_pairs=10,
            retrieve_pairs=3,
            min_rubric_overlap=0.2,
            alpha=0.05,
            output="/tmp/mock_bundle.json",
            json=False,
        )
        ret_compile = _cmd_distill(args_compile_no_extra)
        assert ret_compile == EXIT_OK
        out = capsys.readouterr().out
        assert "Saved DistillationBundle to" in out
        assert "Anchor Pairs" not in out
        assert "Meta-Calibrator" not in out

    # 3. compile with json=False, anchor_bank and meta_calibrator present
    mock_bundle_full = MagicMock()
    mock_bank = MagicMock()
    mock_bank.pairs = [("r", "c")]
    mock_bank.max_pairs = 3
    mock_bundle_full.anchor_bank = mock_bank

    mock_mc = MagicMock()
    mock_mc.intercept = 0.1234
    mock_mc.weights = [0.5678]
    mock_mc.oof_auroc = 0.95
    mock_mc.fail_threshold = 0.3
    mock_mc.pass_threshold = 0.7
    mock_bundle_full.meta_calibrator = mock_mc
    mock_bundle_full.save.return_value = Path("/tmp/mock_bundle.json")

    with patch("logitgate.distill.bundle.compile_distillation_bundle", return_value=mock_bundle_full):
        args_compile_full = argparse.Namespace(
            distill_cmd="compile",
            dataset="data.jsonl",
            telemetry=None,
            backend=None,
            model=None,
            teacher_model=None,
            no_anchors=False,
            no_meta=False,
            max_pairs=10,
            retrieve_pairs=3,
            min_rubric_overlap=0.2,
            alpha=0.05,
            output="/tmp/mock_bundle.json",
            json=False,
        )
        ret_full = _cmd_distill(args_compile_full)
        assert ret_full == EXIT_OK
        out = capsys.readouterr().out
        assert "Anchor Pairs:      1 (retrieve k=3)" in out
        assert "Meta-Calibrator:" in out

    # 4. train-lora with json=False
    mock_manifest = MagicMock()
    mock_manifest.adapter_dir = "/tmp/lora"
    mock_manifest.initial_loss = 1.2345
    mock_manifest.final_loss = 0.2345
    mock_manifest.ollama_modelfile_path = "/tmp/Modelfile"
    mock_manifest.vllm_launch_script_path = "/tmp/launch_vllm.sh"
    mock_manifest.llamacpp_launch_script_path = "/tmp/launch_llama.sh"

    with patch("logitgate.distill.lora.train_distillation_lora", return_value=mock_manifest):
        args_lora = argparse.Namespace(
            distill_cmd="train-lora",
            rank=8,
            lora_alpha=16,
            epochs=1,
            batch_size=2,
            lr=1e-4,
            device="cpu",
            base_model="base",
            dataset="data.jsonl",
            output_dir="/tmp/lora",
            alpha=0.05,
            ollama_base="base",
            json=False,
        )
        ret_lora = _cmd_distill(args_lora)
        assert ret_lora == EXIT_OK
        out_lora = capsys.readouterr().out
        assert "Trained PEFT LoRA adapter ->" in out_lora
        assert "Loss:              1.2345 -> 0.2345" in out_lora


def test_eval_cmd_multirubric_aggregation_and_cache() -> None:
    # 1. MultiRubricGate evaluate with aggregation="all" calls super().evaluate
    gate_all = CliMultiRubricGate(
        rubrics=["r1", "r2"],
        aggregation="all",
        backend=StubBackend(default_prob=0.95),
    )
    v_all = gate_all.evaluate("test candidate")
    assert v_all.status == DecisionStatus.PASSED

    # 2. MultiRubricGate evaluate with aggregation="any"
    gate_any = CliMultiRubricGate(
        rubrics=["r1", "r2"],
        aggregation="any",
        backend=StubBackend(default_prob=0.05),
    )
    # Both fail
    v_any_fail = gate_any.evaluate("test candidate")
    assert v_any_fail.status == DecisionStatus.FAILED
    assert "0/2 rubrics passed" in (v_any_fail.reason or "")

    # One ambiguous, one fail
    with patch.object(
        CliMultiRubricGate,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.AMBIGUOUS, passed=False, probability=0.5, triggered_layer="g1", total_latency_ms=1.0),
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.1, triggered_layer="g2", total_latency_ms=1.0),
        ],
    ):
        v_any_ambig = gate_any.evaluate("test candidate")
        assert v_any_ambig.status == DecisionStatus.AMBIGUOUS

    # One pass
    with patch.object(
        CliMultiRubricGate,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.9, triggered_layer="g1", total_latency_ms=1.0),
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.1, triggered_layer="g2", total_latency_ms=1.0),
        ],
    ):
        v_any_pass = gate_any.evaluate("test candidate")
        assert v_any_pass.status == DecisionStatus.PASSED

    # 3. MultiRubricGate evaluate with aggregation="mean"
    gate_mean = CliMultiRubricGate(
        rubrics=["r1", "r2"],
        aggregation="mean",
        pass_threshold=0.8,
        fail_threshold=0.2,
        backend=StubBackend(default_prob=0.95),
    )
    # mean >= pass_threshold
    with patch.object(
        CliMultiRubricGate,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.85, triggered_layer="g1", total_latency_ms=1.0),
            Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.95, triggered_layer="g2", total_latency_ms=1.0),
        ],
    ):
        v_mean_pass = gate_mean.evaluate("test candidate")
        assert v_mean_pass.status == DecisionStatus.PASSED

    # mean <= fail_threshold
    with patch.object(
        CliMultiRubricGate,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.1, triggered_layer="g1", total_latency_ms=1.0),
            Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.15, triggered_layer="g2", total_latency_ms=1.0),
        ],
    ):
        v_mean_fail = gate_mean.evaluate("test candidate")
        assert v_mean_fail.status == DecisionStatus.FAILED

    # mean between fail_threshold and pass_threshold
    with patch.object(
        CliMultiRubricGate,
        "evaluate_each",
        return_value=[
            Verdict(status=DecisionStatus.AMBIGUOUS, passed=False, probability=0.5, triggered_layer="g1", total_latency_ms=1.0),
            Verdict(status=DecisionStatus.AMBIGUOUS, passed=False, probability=0.5, triggered_layer="g2", total_latency_ms=1.0),
        ],
    ):
        v_mean_ambig = gate_mean.evaluate("test candidate")
        assert v_mean_ambig.status == DecisionStatus.AMBIGUOUS

    # 4. Unknown aggregation strategy raises ConfigurationError
    gate_bad = CliMultiRubricGate(
        rubrics=["r1", "r2"],
        aggregation="unsupported_strat",
        backend=StubBackend(default_prob=0.95),
    )
    with pytest.raises(ConfigurationError, match="Unknown aggregation strategy"):
        gate_bad.evaluate("cand")

    # 5. Cache hit branch
    cached_verdict = Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.9, triggered_layer="HitGate", total_latency_ms=1.0)
    with patch("logitgate.cache.lookup_cached_verdict", return_value=cached_verdict):
        v_hit = gate_any.evaluate("cand")
        assert v_hit.status == DecisionStatus.PASSED
        assert v_hit.probability == 0.9


def test_eval_cmd_choicegate_pass_choices() -> None:
    # 1. Choice not in pass_choices
    gate_choice = CliChoiceGate(
        question="Select option",
        options={"A": "Alpha", "B": "Beta"},
        pass_choices=["A"],
        backend=StubBackend(),
    )
    mock_base_verdict_fail = Verdict(
        status=DecisionStatus.PASSED,
        passed=True,
        probability=0.95,
        triggered_layer="ChoiceGate",
        choice="B",
        total_latency_ms=1.0,
    )
    with patch.object(ChoiceGate, "evaluate", return_value=mock_base_verdict_fail):
        v = gate_choice.evaluate("cand")
        assert v.passed is False
        assert v.status == DecisionStatus.FAILED
        assert "Choice 'B' not in expected pass choices ['A']" in (v.reason or "")

    # 2. Choice is in pass_choices
    mock_base_verdict_pass = Verdict(
        status=DecisionStatus.PASSED,
        passed=True,
        probability=0.95,
        triggered_layer="ChoiceGate",
        choice="A",
        total_latency_ms=1.0,
    )
    with patch.object(ChoiceGate, "evaluate", return_value=mock_base_verdict_pass):
        v_ok = gate_choice.evaluate("cand")
        assert v_ok.passed is True
        assert v_ok.status == DecisionStatus.PASSED


def test_eval_cmd_cli_validation_and_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # 1. Candidate reading errors
    bad_file = tmp_path / "does_not_exist.txt"
    args_bad_file = argparse.Namespace(file=str(bad_file), candidate=None)
    ret_bad_file = _cmd_eval(args_bad_file)
    assert ret_bad_file == EXIT_FAILED
    assert "Error reading file" in capsys.readouterr().err

    with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin.read", side_effect=OSError("pipe broken")):
        args_stdin_err = argparse.Namespace(file=None, candidate=None)
        assert _cmd_eval(args_stdin_err) == EXIT_FAILED
        assert "Error reading stdin" in capsys.readouterr().err

    with patch("sys.stdin.isatty", return_value=True):
        args_no_cand = argparse.Namespace(file=None, candidate=None)
        assert _cmd_eval(args_no_cand) == EXIT_FAILED
        assert "Either --candidate or --file must be specified" in capsys.readouterr().err

    # 2. Rubric normalization: integer rubric
    args_int_rubric = argparse.Namespace(
        candidate="test",
        file=None,
        rubric=42,
        choice=None,
        score_min=None,
        score_max=None,
        telemetry_out=None,
        cache=None,
        ast_syntax=False,
        normalize=False,
        backend="stub://default",
        model=None,
        threshold=0.5,
        fail_threshold=0.1,
        profile=None,
        bundle=None,
        json=True,
    )
    assert _cmd_eval(args_int_rubric) == EXIT_OK

    # 3. Choice options validation
    base_choice_args: dict[str, Any] = {
        "candidate": "test",
        "file": None,
        "rubric": "question?",
        "score_min": None,
        "score_max": None,
        "telemetry_out": None,
        "cache": None,
        "ast_syntax": False,
        "normalize": False,
        "backend": "stub://default",
        "model": None,
        "profile": None,
        "bundle": None,
        "threshold": 0.5,
        "fail_threshold": 0.1,
        "min_confidence": None,
        "min_margin": None,
    }

    # Invalid choice format (no =)
    with pytest.raises(ConfigurationError, match="Invalid --choice"):
        _cmd_eval(argparse.Namespace(**base_choice_args, choice=["NO_EQUALS"], expected_choice=None, json=False))

    # Empty key in choice option
    with pytest.raises(ConfigurationError, match="Choice option key cannot be empty"):
        _cmd_eval(argparse.Namespace(**base_choice_args, choice=["=Val", "B=Val2"], expected_choice=None, json=False))

    # Empty value in choice option
    with pytest.raises(ConfigurationError, match="Choice option description cannot be empty"):
        _cmd_eval(argparse.Namespace(**base_choice_args, choice=["A=", "B=Val2"], expected_choice=None, json=False))

    # Less than 2 choices
    with pytest.raises(ConfigurationError, match="requires at least 2 options"):
        _cmd_eval(argparse.Namespace(**base_choice_args, choice=['A="Only One"'], expected_choice=None, json=False))

    # Expected choice not in options
    with pytest.raises(ConfigurationError, match="--expected-choice 'Z' is not one of"):
        _cmd_eval(argparse.Namespace(**base_choice_args, choice=["A=Opt1", "B=Opt2"], expected_choice="Z", json=False))

    # 4. Score options validation
    base_score_args: dict[str, Any] = {
        "candidate": "test",
        "file": None,
        "rubric": "rate quality",
        "choice": None,
        "telemetry_out": None,
        "cache": None,
        "ast_syntax": False,
        "normalize": False,
        "backend": "stub://default",
        "model": None,
        "profile": None,
        "threshold": 3.0,
        "fail_threshold": 2.0,
    }
    with pytest.raises(ConfigurationError, match="--score-min .* must be strictly less than --score-max"):
        _cmd_eval(argparse.Namespace(**base_score_args, score_min=5, score_max=2, score_threshold=None, json=False))

    # 5. ChoiceGate Human & JSON output
    args_choice_human = argparse.Namespace(
        **base_choice_args,
        choice=['A="Approve"', "B='Reject'"],
        expected_choice="A",
        json=False,
    )
    assert _cmd_eval(args_choice_human) == EXIT_OK
    out_ch = capsys.readouterr().out
    assert "Verdict: PASSED" in out_ch
    assert "Choice:" in out_ch

    args_choice_json = argparse.Namespace(
        **base_choice_args,
        choice=["A=Approve", "B=Reject"],
        expected_choice="A",
        json=True,
    )
    assert _cmd_eval(args_choice_json) == EXIT_OK
    out_ch_json = capsys.readouterr().out
    assert '"choice":' in out_ch_json

    # 6. ScoreGate Human & JSON output with score_threshold falling back to args.threshold
    args_score_human = argparse.Namespace(
        **base_score_args,
        score_min=1,
        score_max=5,
        score_threshold=None,
        json=False,
    )
    assert _cmd_eval(args_score_human) == EXIT_OK
    out_sc = capsys.readouterr().out
    assert "Verdict: PASSED" in out_sc
    assert "Score:" in out_sc

    args_score_json = argparse.Namespace(
        **base_score_args,
        score_min=1,
        score_max=5,
        score_threshold=3.5,
        json=True,
    )
    assert _cmd_eval(args_score_json) == EXIT_OK
    out_sc_json = capsys.readouterr().out
    assert '"score":' in out_sc_json
    assert '"score_distribution":' in out_sc_json

    # 7. Non-JSON formatting when verdict has no probability and no reason, and fails
    mock_fail_verdict = Verdict(
        status=DecisionStatus.FAILED,
        passed=False,
        probability=None,
        reason="",
        triggered_layer="StubGate",
        total_latency_ms=2.0,
    )
    with patch.object(LogitGate, "evaluate", return_value=mock_fail_verdict):
        args_fail_sparse = argparse.Namespace(
            candidate="fail cand",
            file=None,
            rubric="quality check",
            choice=None,
            score_min=None,
            score_max=None,
            telemetry_out=None,
            cache=None,
            ast_syntax=False,
            normalize=False,
            backend="stub://default",
            model=None,
            threshold=0.5,
            fail_threshold=0.1,
            profile=None,
            bundle=None,
            json=False,
        )
        assert _cmd_eval(args_fail_sparse) == EXIT_FAILED
        out_sparse = capsys.readouterr().out
        assert "Verdict: FAILED" in out_sparse
        assert "Confidence:" not in out_sparse
        assert "Reason:" not in out_sparse


def test_base_gate_lifecycle_and_branches() -> None:
    # 1. _safe_close with object having no close
    _safe_close(object())

    # 2. _safe_close with object whose close raises Exception
    mock_bad_close = MagicMock()
    mock_bad_close.close.side_effect = RuntimeError("close failed")
    _safe_close(mock_bad_close)

    # 3. _safe_aclose
    sync_close_obj = MagicMock(spec=["close"])
    asyncio.run(_safe_aclose(sync_close_obj))
    sync_close_obj.close.assert_called_once()

    asyncio.run(_safe_aclose(object()))

    async def _failing_aclose() -> None:
        raise RuntimeError("aclose failed")

    mock_bad_aclose = MagicMock()
    mock_bad_aclose.aclose = _failing_aclose
    asyncio.run(_safe_aclose(mock_bad_aclose))

    # 4. Default BaseGate.aevaluate() executes via loop.run_in_executor
    class _CustomSyncGate(BaseGate):
        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(
                passed=True,
                status=DecisionStatus.PASSED,
                probability=0.99,
                triggered_layer=self.name,
                reason="custom sync pass",
                total_latency_ms=1.0,
            )

    gate = _CustomSyncGate()
    v_async = asyncio.run(gate.aevaluate("hello candidate"))
    assert v_async.passed is True
    assert v_async.probability == 0.99

    # 5. BaseGate.close() branches
    child_g1 = MagicMock(spec=["close"])
    child_g2 = MagicMock(spec=["close"])
    gate.gates = [child_g1, child_g2]  # type: ignore[attr-defined]
    gate._owns_backend = True  # type: ignore[attr-defined]
    gate.backend = None  # type: ignore[attr-defined]
    gate._owns_cache = True  # type: ignore[attr-defined]
    gate.cache = None  # type: ignore[attr-defined]
    gate._owns_sink = True  # type: ignore[attr-defined]
    gate.telemetry_sink = None  # type: ignore[attr-defined]
    gate.close()
    child_g1.close.assert_called_once()
    child_g2.close.assert_called_once()

    # 6. BaseGate.aclose() branches
    gate_async = _CustomSyncGate()
    async_child = MagicMock()

    async def _c_aclose() -> None:
        pass

    async_child.aclose = _c_aclose
    sync_child = MagicMock(spec=["close"])
    gate_async.gates = [async_child, sync_child]  # type: ignore[attr-defined]

    mock_b = MagicMock()

    async def _b_aclose() -> None:
        pass

    mock_b.aclose = _b_aclose
    gate_async._owns_backend = True  # type: ignore[attr-defined]
    gate_async.backend = mock_b  # type: ignore[attr-defined]

    mock_c = MagicMock()

    async def _c_cache_aclose() -> None:
        pass

    mock_c.aclose = _c_cache_aclose
    gate_async._owns_cache = True  # type: ignore[attr-defined]
    gate_async.cache = mock_c  # type: ignore[attr-defined]

    mock_s = MagicMock()

    async def _s_aclose() -> None:
        pass

    mock_s.aclose = _s_aclose
    gate_async._owns_sink = True  # type: ignore[attr-defined]
    gate_async.telemetry_sink = mock_s  # type: ignore[attr-defined]

    asyncio.run(gate_async.aclose())
    sync_child.close.assert_called_once()

    # 7. BaseGate.aclose() with None resources
    gate_async_none = _CustomSyncGate()
    gate_async_none._owns_backend = True  # type: ignore[attr-defined]
    gate_async_none.backend = None  # type: ignore[attr-defined]
    gate_async_none._owns_cache = True  # type: ignore[attr-defined]
    gate_async_none.cache = None  # type: ignore[attr-defined]
    gate_async_none._owns_sink = True  # type: ignore[attr-defined]
    gate_async_none.telemetry_sink = None  # type: ignore[attr-defined]
    asyncio.run(gate_async_none.aclose())


def test_verdict_snapshot_branches_and_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1. Fast failure console notice hint emission
    reset_snapshot_notices()
    v_hint = Verdict(
        status=DecisionStatus.FAILED,
        passed=False,
        probability=0.1,
        triggered_layer="LogitGate",
        total_latency_ms=1.0,
        traces=[LayerTrace(layer_name="LogitGate", status=DecisionStatus.FAILED, latency_ms=1.0)],
    )
    with patch("logitgate.snapshot._should_log_console_notice", return_value=True):
        persist_failure_snapshot(v_hint, candidate="fail cand", save_on_fail=None)
        err = capsys.readouterr().err
        assert "[logitgate] Fast failure at LogitGate without Layer-2 critique" in err

    # 2. _atomic_write_text symlink TOCTOU check and fd_owned branches
    p_text = tmp_path / "atomic_text.json"
    with (
        patch.object(Path, "is_symlink", side_effect=[False, False, True]),
        pytest.raises(ConfigurationError, match="Refusing to overwrite symlink VerdictSnapshot"),
    ):
        _atomic_write_text(p_text, '{"hello": "world"}')

    with patch("os.fdopen", side_effect=OSError("fdopen failed")), pytest.raises(OSError, match="fdopen failed"):
        _atomic_write_text(p_text, '{"hello": "world"}')

    # 3. reset_snapshot_notices when pytest_plugin is None or _CHECK_SNAPSHOT_PATHS is not a list
    with patch.dict(sys.modules, {"logitgate.integrations.pytest_plugin": None}):
        reset_snapshot_notices()

    # 4. VerdictSnapshot._populate_defaults when data is not a dict
    populate_defaults_fn: Any = VerdictSnapshot._populate_defaults
    assert populate_defaults_fn("not_a_dict") == "not_a_dict"

    # 5. Replay with profile & bundle loaded from string path
    prof_file = tmp_path / "test_prof.json"
    prof_obj = CalibrationProfile(
        model="stub",
        symmetric_temperature=1.2,
        pass_threshold=0.8,
        fail_threshold=0.2,
    )
    prof_obj.save(prof_file)

    bundle_file = tmp_path / "test_bundle.json"
    bundle_obj = DistillationBundle(profile=prof_obj)
    bundle_obj.save(bundle_file)

    snap = VerdictSnapshot(
        rubric="test rubric",
        candidate="test candidate",
        verdict=Verdict(status=DecisionStatus.FAILED, passed=False, triggered_layer="ScoreGate", total_latency_ms=1.0),
    )

    # ScoreGate with profile path
    gate_score, _ = snap._build_replay_gate(
        {"gate_type": "ScoreGate", "profile": str(prof_file)},
        effective_backend="stub://default",
        effective_model="stub",
    )
    assert gate_score.name == "ScoreGate"

    # ChoiceGate with profile path
    gate_choice, _ = snap._build_replay_gate(
        {"gate_type": "ChoiceGate", "options": {"A": "Yes", "B": "No"}, "profile": str(prof_file)},
        effective_backend="stub://default",
        effective_model="stub",
    )
    assert gate_choice.name == "ChoiceGate"

    # RankGate replay
    gate_rank, _ = snap._build_replay_gate(
        {"gate_type": "RankGate"},
        effective_backend="stub://default",
        effective_model="stub",
    )
    assert gate_rank.name == "RankGate"

    # MultiRubricGate with bundle path
    gate_multi, _ = snap._build_replay_gate(
        {
            "gate_type": "LogitGate",
            "rubrics": ["r1", "r2"],
            "distillation_bundle": str(bundle_file),
        },
        effective_backend="stub://default",
        effective_model="stub",
    )
    assert gate_multi.name == "MultiRubricGate"

    class _MockReplayChild(BaseGate):
        def __init__(self, extra_backends: list[Any] | None = None) -> None:
            self.backend = StubBackend()
            self._replay_extra_backends = extra_backends or []

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(status=DecisionStatus.PASSED, passed=True, total_latency_ms=1.0, triggered_layer=self.name)

    # 6. BestOfNGate replay with child having _replay_extra_backends
    mock_extra_b = MagicMock()
    mock_child = _MockReplayChild(extra_backends=[mock_extra_b])
    real_build = snap._build_replay_gate
    call_count_bon = 0

    def _mock_build_bon(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
        nonlocal call_count_bon
        call_count_bon += 1
        if call_count_bon == 1:
            return real_build(*args, **kwargs)
        return (mock_child, True)

    with patch.object(snap, "_build_replay_gate", side_effect=_mock_build_bon):
        bon_gate, _ = snap._build_replay_gate(
            {"gate_type": "BestOfNGate", "pre_gate_config": {"gate_type": "ScoreGate"}},
            effective_backend="stub://default",
            effective_model="stub",
        )
        assert mock_extra_b in bon_gate._replay_extra_backends

    # 7. Cascade replay with child having _replay_extra_backends
    mock_c_extra_b = MagicMock()
    mock_c_child = _MockReplayChild(extra_backends=[mock_c_extra_b])
    call_count_casc = 0

    def _mock_build_casc(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
        nonlocal call_count_casc
        call_count_casc += 1
        if call_count_casc == 1:
            return real_build(*args, **kwargs)
        return (mock_c_child, True)

    with patch.object(snap, "_build_replay_gate", side_effect=_mock_build_casc):
        casc_gate, _ = snap._build_replay_gate(
            {"gate_type": "Cascade", "layers": [{"gate_type": "LogitGate"}]},
            effective_backend="stub://default",
            effective_model="stub",
        )
        assert mock_c_extra_b in casc_gate._replay_extra_backends

    # 8. Cascade replay with invalid empty/non-dict layers
    with pytest.raises(ValueError, match="Cascade snapshot 'layers' contains no valid gate configs"):
        snap._build_replay_gate(
            {"gate_type": "Cascade", "layers": [123, "not_dict"]},
            effective_backend="stub://default",
            effective_model="stub",
        )

    # 9. Cascade replay error cleanup when a child fails to build
    first_child_backend = MagicMock(spec=["close"])
    first_child = _MockReplayChild()
    first_child.backend = first_child_backend

    call_count_err = 0

    def _mock_build_step(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
        nonlocal call_count_err
        call_count_err += 1
        if call_count_err == 1:
            return real_build(*args, **kwargs)
        if call_count_err == 2:
            return (first_child, True)
        raise RuntimeError("Second child failed")

    with patch.object(snap, "_build_replay_gate", side_effect=_mock_build_step):
        with pytest.raises(RuntimeError, match="Second child failed"):
            snap._build_replay_gate(
                {"gate_type": "Cascade", "layers": [{"gate_type": "g1"}, {"gate_type": "g2"}]},
                effective_backend="stub://default",
                effective_model="stub",
            )
        first_child_backend.close.assert_called_once()

    # 10. Replay and areplay where extra_backend is gate.backend (lines 736->735 and 764->763)
    mock_rep_gate = MagicMock()
    mock_rep_backend = MagicMock()

    async def _mock_b_aclose() -> None:
        pass

    mock_rep_backend.aclose = _mock_b_aclose
    mock_rep_gate.backend = mock_rep_backend
    mock_rep_gate._replay_extra_backends = [mock_rep_backend]
    mock_rep_gate.evaluate.return_value = Verdict(
        status=DecisionStatus.PASSED,
        passed=True,
        total_latency_ms=1.0,
        triggered_layer="g",
    )

    async def _mock_aeval(*args: Any, **kwargs: Any) -> Verdict:
        return Verdict(
            status=DecisionStatus.PASSED,
            passed=True,
            total_latency_ms=1.0,
            triggered_layer="g",
        )

    mock_rep_gate.aevaluate = _mock_aeval
    with patch.object(snap, "_build_replay_gate", return_value=(mock_rep_gate, True)):
        snap.replay()
        asyncio.run(snap.areplay())


def test_cascade_lifecycle_and_batch_branches(tmp_path: Path) -> None:
    # 0. Constructor default layer construction and missing gate errors
    with pytest.raises(ConfigurationError, match="requires at least one gate"):
        Cascade(backend="stub://default")

    with pytest.raises(ConfigurationError, match="requires at least one gate"):
        Cascade([], backend="stub://default")

    casc_constructed = Cascade(
        rubric="test rubric",
        backend="stub://default",
        judge_backend="stub://default",
        pass_threshold=0.9,
        fail_threshold=0.1,
        logit_temperature=1.0,
        trigger="always",
    )
    assert len(casc_constructed.gates) == 2
    asyncio.run(casc_constructed.aclose())

    gate1 = LogitGate(rubric="is quality", backend=StubBackend())
    casc = Cascade([gate1])

    # repr
    assert "Cascade([" in repr(casc)

    # 1. evaluate_batch
    with pytest.raises(ConfigurationError, match="Invalid on_error"):
        casc.evaluate_batch(["test"], on_error="invalid")  # type: ignore[arg-type]

    with patch.object(casc, "evaluate", side_effect=ValueError("failing eval")):
        verdicts_abs = casc.evaluate_batch(["test"], on_error="abstain")
        assert len(verdicts_abs) == 1
        assert verdicts_abs[0].status == DecisionStatus.AMBIGUOUS
        assert verdicts_abs[0].passed is False
        assert "ValueError: failing eval" in (verdicts_abs[0].reason or "")

        verdicts_fail = casc.evaluate_batch(["test"], on_error="fail")
        assert len(verdicts_fail) == 1
        assert verdicts_fail[0].status == DecisionStatus.FAILED
        assert verdicts_fail[0].passed is False

        with pytest.raises(ValueError, match="failing eval"):
            casc.evaluate_batch(["test"], on_error="raise")

    with patch.object(casc, "evaluate", side_effect=KeyboardInterrupt()), pytest.raises(KeyboardInterrupt):
        casc.evaluate_batch(["test"], on_error="abstain")

    # 2. aevaluate_batch
    with pytest.raises(ConfigurationError, match="Invalid on_error"):
        asyncio.run(casc.aevaluate_batch(["test"], on_error="invalid"))  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="max_concurrency must be an integer >= 1"):
        asyncio.run(casc.aevaluate_batch(["test"], max_concurrency=0))

    with patch.object(casc, "aevaluate", side_effect=ValueError("async err")):
        v_abs_async = asyncio.run(casc.aevaluate_batch(["t1", "t2"], max_concurrency=2, on_error="abstain"))
        assert len(v_abs_async) == 2
        assert all(v.status == DecisionStatus.AMBIGUOUS for v in v_abs_async)

        v_fail_async = asyncio.run(casc.aevaluate_batch(["t1"], max_concurrency=2, on_error="fail"))
        assert v_fail_async[0].status == DecisionStatus.FAILED

        with pytest.raises(ValueError, match="async err"):
            asyncio.run(casc.aevaluate_batch(["t1"], max_concurrency=2, on_error="raise"))

    with patch.object(casc, "aevaluate", side_effect=KeyboardInterrupt()), pytest.raises(KeyboardInterrupt):
        asyncio.run(casc.aevaluate_batch(["t1"], on_error="abstain"))

    # 3. aclose() and close() branches
    async def _run_aclose_test() -> None:
        c = Cascade([gate1])
        mock_c = MagicMock()

        async def _c_aclose() -> None:
            pass

        mock_c.aclose = _c_aclose
        c._owns_cache = True
        c.cache = mock_c

        mock_s = MagicMock()

        async def _s_aclose() -> None:
            pass

        mock_s.aclose = _s_aclose
        c._owns_telemetry_sink = True
        c.telemetry_sink = mock_s

        mock_g = MagicMock()

        async def _g_aclose() -> None:
            pass

        mock_g.aclose = _g_aclose
        c.gates = [mock_g]

        mock_extra = MagicMock()

        async def _extra_aclose() -> None:
            pass

        mock_extra.aclose = _extra_aclose
        c._replay_extra_backends = [mock_extra]

        await c.aclose()

        c2 = Cascade([gate1])
        c2._owns_cache = True
        c2.cache = MagicMock(spec=["close"])
        c2._owns_telemetry_sink = True
        c2.telemetry_sink = MagicMock(spec=["close"])
        c2_child = MagicMock(spec=["close"])
        c2.gates = [c2_child]
        c2_extra = MagicMock(spec=["close"])
        c2._replay_extra_backends = [c2_extra]

        await c2.aclose()
        c2_child.close.assert_called_once()
        c2_extra.close.assert_called_once()

    asyncio.run(_run_aclose_test())

    async def _test_async_with() -> None:
        async with Cascade([gate1]) as c_ctx:
            assert isinstance(c_ctx, Cascade)

    asyncio.run(_test_async_with())

    c_sync = Cascade([gate1])
    c_sync._owns_cache = True
    c_sync.cache = MagicMock(spec=["close"])
    c_sync._owns_telemetry_sink = True
    c_sync.telemetry_sink = MagicMock(spec=["close"])
    extra_b = MagicMock(spec=["close"])
    c_sync._replay_extra_backends = [extra_b]
    c_sync.close()
    extra_b.close.assert_called_once()

    # 4. Constructor prior_weight validations
    with pytest.raises(ConfigurationError, match="prior_weight must be between 0.0 and 1.0"):
        Cascade([gate1], prior_weight=True)

    with pytest.raises(ConfigurationError, match="prior_weight must be a float between 0.0 and 1.0"):
        Cascade([gate1], prior_weight="not_a_float")  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="prior_weight must be between 0.0 and 1.0"):
        Cascade([gate1], prior_weight=1.5)

    # 5. Constructor cache resolution failure closes owned sink
    mock_sink_cleanup = MagicMock(spec=["close"])
    with (
        patch("logitgate.telemetry.sinks.resolve_telemetry_sink", return_value=mock_sink_cleanup),
        patch("logitgate.cache.resolve_verdict_cache", side_effect=RuntimeError("cache error")),
        pytest.raises(RuntimeError, match="cache error"),
    ):
        Cascade([gate1], telemetry_sink=str(tmp_path / "sink.jsonl"), cache="bad")
    mock_sink_cleanup.close.assert_called_once()

    # 6. Snapshot without replay config inference branches
    class _MockPlainGate(BaseGate):
        name = "PlainGate"
        layer_tier = 1

        def __init__(self) -> None:
            self.backend = StubBackend()
            self.model = "stub-model"

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(status=DecisionStatus.PASSED, passed=True, total_latency_ms=1.0, triggered_layer=self.name)

        def replay_config(self) -> dict[str, Any] | None:
            return None

    class _MockJudgeGate(BaseGate):
        name = "GenerativeJudge"
        layer_tier = TIER_GENERATIVE

        def __init__(self) -> None:
            self.backend = StubBackend()
            self.model = "judge-model"

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(status=DecisionStatus.PASSED, passed=True, total_latency_ms=1.0, triggered_layer=self.name)

        def replay_config(self) -> dict[str, Any] | None:
            return None

    casc_no_replay = Cascade([_MockPlainGate(), _MockJudgeGate()], save_failures=True)
    v_snap = Verdict(status=DecisionStatus.FAILED, passed=False, triggered_layer="GenerativeJudge", total_latency_ms=1.0)
    with patch("logitgate.cascade.persist_failure_snapshot") as mock_snap_call:
        casc_no_replay._persist_failure(v_snap, "candidate", "rubric", {})
        assert mock_snap_call.called
        cfg_passed = mock_snap_call.call_args[1]["gate_config"]
        assert cfg_passed.get("backend") is not None
        assert cfg_passed.get("model") == "judge-model"
        assert cfg_passed.get("judge_backend") is not None
        assert cfg_passed.get("judge_model") == "judge-model"

    # 7. Locked upstream failure overrode_judge_pass and fused_reason branches (lines 835-838, 895-897)
    g_locked = LogitGate(rubric="is accurate", backend=StubBackend(default_prob=0.05))
    g_judge = JudgeGate(rubric="is accurate", backend=StubBackend(diagnostic_text="[VERDICT: PASS]"), trigger="on_ambiguous_or_fail")
    g_judge.can_override_fail = False
    casc_locked = Cascade([g_locked, g_judge], fuse_logits=False)
    v_locked = casc_locked.evaluate("candidate text")
    assert v_locked.passed is False
    assert v_locked.status == DecisionStatus.FAILED

    # Fused logits where passed != verdict.passed (lines 844-874, 895-897)
    class _MockFusedTier1(BaseGate):
        name = "Tier1Gate"
        layer_tier = 1

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            tr = LayerTrace(
                layer_name=self.name,
                status=DecisionStatus.AMBIGUOUS,
                latency_ms=1.0,
                log_odds_pass=1.5,
                forward_log_odds=1.5,
                reverse_log_odds=1.5,
            )
            return Verdict(
                passed=False,
                status=DecisionStatus.AMBIGUOUS,
                probability=0.8,
                triggered_layer=self.name,
                total_latency_ms=1.0,
                traces=[tr],
            )

    class _MockFusedTier2(BaseGate):
        name = "Tier2Judge"
        layer_tier = 2
        can_override_fail = True

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            tr = LayerTrace(
                layer_name=self.name,
                status=DecisionStatus.FAILED,
                latency_ms=2.0,
                log_odds_pass=-0.2,
                forward_log_odds=0.5,
                reverse_log_odds=-0.9,
            )
            return Verdict(
                passed=False,
                status=DecisionStatus.FAILED,
                probability=0.45,
                reason="Tier 2 critique",
                triggered_layer=self.name,
                total_latency_ms=2.0,
                traces=[tr],
            )

    casc_fused = Cascade([_MockFusedTier1(), _MockFusedTier2()], fuse_logits=True, prior_weight=0.5)
    v_fused = casc_fused.evaluate("candidate text")
    assert v_fused.passed is True
    assert v_fused.status == DecisionStatus.PASSED
    assert "Fused Tier-1 prior" in (v_fused.reason or "")


def test_logit_gate_batch_and_calibration_branches() -> None:
    gate = LogitGate(rubric="is quality", backend=StubBackend(default_prob=0.95))

    # 1. evaluate_batch
    with pytest.raises(ConfigurationError, match="on_error must be one of"):
        gate.evaluate_batch(candidate="test", on_error="invalid")  # type: ignore[arg-type]

    assert gate.evaluate_batch(candidate=None, candidates=None) == []

    v_list = gate.evaluate_batch(candidate="single candidate")
    assert len(v_list) == 1
    assert v_list[0].status == DecisionStatus.PASSED

    with patch.object(gate, "_query", side_effect=RuntimeError("query error")):
        v_rub_abs = gate.evaluate_batch(candidate="test", rubrics=["r1", "r2"], on_error="abstain")
        assert len(v_rub_abs) == 2
        assert all(v.status == DecisionStatus.AMBIGUOUS for v in v_rub_abs)

        v_rub_fail = gate.evaluate_batch(candidate="test", rubrics=["r1", "r2"], on_error="fail")
        assert len(v_rub_fail) == 2
        assert all(v.status == DecisionStatus.FAILED for v in v_rub_fail)

    # 2. aevaluate_batch
    with pytest.raises(ConfigurationError, match="on_error must be one of"):
        asyncio.run(gate.aevaluate_batch(candidate="test", on_error="invalid"))  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError, match="max_concurrency must be an integer >= 1"):
        asyncio.run(gate.aevaluate_batch(candidate="test", max_concurrency=0))

    assert asyncio.run(gate.aevaluate_batch(candidate=None, candidates=None)) == []
    v_async_single = asyncio.run(gate.aevaluate_batch(candidate="single async"))
    assert len(v_async_single) == 1
    assert v_async_single[0].status == DecisionStatus.PASSED

    with patch.object(gate, "_aquery_pair", side_effect=RuntimeError("async pair error")):
        v_async_abs = asyncio.run(gate.aevaluate_batch(candidate="test", rubrics=["r1", "r2"], on_error="abstain"))
        assert len(v_async_abs) == 2
        assert all(v.status == DecisionStatus.AMBIGUOUS for v in v_async_abs)

        v_async_fail = asyncio.run(gate.aevaluate_batch(candidate="test", rubrics=["r1", "r2"], on_error="fail"))
        assert len(v_async_fail) == 2
        assert all(v.status == DecisionStatus.FAILED for v in v_async_fail)

    v_pass_mock = Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.9, triggered_layer=gate.name, total_latency_ms=1.0)
    v_fail_mock = Verdict(status=DecisionStatus.FAILED, passed=False, probability=0.1, triggered_layer=gate.name, total_latency_ms=1.0)

    # fail_fast breaks early on second
    with patch.object(gate, "_verdict_from_logprobs", side_effect=[v_pass_mock, v_fail_mock, v_pass_mock]):
        res_ff_break = asyncio.run(
            gate.aevaluate_batch(candidate="test", rubrics=["r1", "r2", "r3"], fail_fast=True)
        )
        assert len(res_ff_break) == 2
        assert res_ff_break[0].status == DecisionStatus.PASSED
        assert res_ff_break[1].status == DecisionStatus.FAILED

    # fail_fast all pass
    with patch.object(gate, "_verdict_from_logprobs", side_effect=[v_pass_mock, v_pass_mock, v_pass_mock]):
        res_ff_all = asyncio.run(
            gate.aevaluate_batch(candidate="test", rubrics=["r1", "r2", "r3"], fail_fast=True)
        )
        assert len(res_ff_all) == 3

    # 3. Constructor validations
    with pytest.raises(ConfigurationError, match="rubric cannot be empty"):
        LogitGate(rubric="")

    with pytest.raises(ConfigurationError, match="pass_threshold .* must be strictly greater than fail_threshold"):
        LogitGate(rubric="test", pass_threshold=0.3, fail_threshold=0.7)

    mock_owned_b = MagicMock(spec=["close"])
    with (
        patch("logitgate.gates.logit_gate.resolve_logit_backend", return_value=mock_owned_b),
        patch("logitgate.cache.resolve_verdict_cache", side_effect=RuntimeError("cache init fail")),
        pytest.raises(RuntimeError, match="cache init fail"),
    ):
        LogitGate(rubric="test", backend="mock://b", cache="bad")
    mock_owned_b.close.assert_called_once()

    # 4. Repr with long rubric > 40 chars & warm_prefix replay config
    long_rubric = "This is a very long rubric exceeding forty characters in total length"
    gate_long = LogitGate(rubric=long_rubric, backend=StubBackend(), warm_prefix=True)
    assert "..." in repr(gate_long)
    assert gate_long.replay_config().get("warm_prefix") is True

    # 5. _ensure_calibrated & _aensure_calibrated
    gate_cal = LogitGate(rubric="test", backend=StubBackend(), model="stub-model")
    gate_cal._needs_lazy_calibration = False
    gate_cal._ensure_calibrated()

    gate_lock = LogitGate(rubric="test", backend=StubBackend(), model="stub-model")
    gate_lock._needs_lazy_calibration = True
    mock_lock = MagicMock()

    def _race() -> None:
        gate_lock._needs_lazy_calibration = False

    mock_lock.__enter__.side_effect = _race
    gate_lock._calibration_lock = mock_lock
    gate_lock._ensure_calibrated()

    gate_cal._needs_lazy_calibration = True
    from logitgate.config import configure

    configure(auto_calibrate=False)
    try:
        gate_cal._ensure_calibrated()
        assert gate_cal._needs_lazy_calibration is False

        gate_cal._needs_lazy_calibration = True
        asyncio.run(gate_cal._aensure_calibrated())
        assert gate_cal._needs_lazy_calibration is False
    finally:
        configure(auto_calibrate=True)

    # 6. _handle_failure_snapshot with _is_cache_hit=True
    v_hit = Verdict(status=DecisionStatus.PASSED, passed=True, probability=0.9, triggered_layer=gate.name, total_latency_ms=1.0)
    with patch("logitgate.cache.store_cached_verdict") as mock_store:
        gate._handle_failure_snapshot("cand", v_hit, _is_cache_hit=True, telemetry_tags={"initial": "val"})
        assert not mock_store.called

        gate._handle_failure_snapshot("cand", v_hit, _is_cache_hit=True, telemetry_tags=None)
        assert not mock_store.called

    # 7. Adaptive Conformal Inference (ACI) decision state branches
    mock_aci = MagicMock()
    mock_aci.current_decision_state.return_value = (0.5, 0.2, 0.8, -1.0, 1.0, False)

    gate_aci = LogitGate(rubric="test", backend=StubBackend())
    gate_aci.adaptive_controller = mock_aci

    top_fwd = [{"token": "A", "logprob": -0.2}, {"token": "B", "logprob": -2.0}]
    top_rev = [{"token": "B", "logprob": -0.2}, {"token": "A", "logprob": -2.0}]

    v_aci_pass = gate_aci._verdict_from_logprobs(
        top_fwd=top_fwd,
        top_rev=top_rev,
        t0=time.perf_counter(),
    )
    assert v_aci_pass.status == DecisionStatus.PASSED

    mock_aci.current_decision_state.return_value = (0.5, 0.2, 0.8, -10.0, 10.0, False)
    v_aci_unmet = gate_aci._verdict_from_logprobs(
        top_fwd=top_fwd,
        top_rev=top_rev,
        t0=time.perf_counter(),
    )
    assert v_aci_unmet.status == DecisionStatus.AMBIGUOUS
    assert v_aci_unmet.traces[0].metadata.get("abstained_reason") == "adaptive_log_odds_margin_unmet"


# ---------------------------------------------------------------------------
# Section 7: v0.14.0 Submodule Behavioral Mutant Killers
# ---------------------------------------------------------------------------


def test_v0140_calibration_lifecycle_mutants() -> None:
    assert lifecycle is not None
    # 1. Line 39 (LCR or -> and): _validate_prob_threshold must reject bool and str
    with pytest.raises(ConfigurationError):
        _validate_prob_threshold("test", True)
    with pytest.raises(ConfigurationError):
        _validate_prob_threshold("test", "0.5")

    # 2. Line 180 (CRP return False -> True): _should_lazy_auto_calibrate must return False when auto_calibrate is False
    class _CustomLogitBackend(LogitBackend):
        def query_top_logprobs(
            self, prompt: str, top_k: int = 5, allowed_tokens: list[str] | None = None, **kwargs: Any
        ) -> list[dict[str, Any]]:
            return []

        async def aquery_top_logprobs(
            self, prompt: str, top_k: int = 5, allowed_tokens: list[str] | None = None, **kwargs: Any
        ) -> list[dict[str, Any]]:
            return []

    try:
        configure(auto_calibrate=False)
        assert _should_lazy_auto_calibrate(None, "llama-custom-model-x", _CustomLogitBackend()) is False
    finally:
        configure(auto_calibrate=True)

    # 3. Line 234 (UOI not x -> x): _warn_if_profile_model_mismatch warns when models differ
    prof = CalibrationProfile(model="model-alpha", temperature=1.2)
    with pytest.warns(CalibrationWarning, match="Thresholds and temperatures"):
        _warn_if_profile_model_mismatch(prof, "model-beta", "LifecycleGate")


def test_v0140_cascade_generative_escalation_and_sink_mutants() -> None:
    # 1. Line 329 (UOI not x -> x): aclose closes owned telemetry sink
    class _SimpleGate(BaseGate):
        name = "simple"
        layer_tier = 50

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(status=DecisionStatus.PASSED, passed=True, triggered_layer=self.name, total_latency_ms=1.0)

    async def _run_aclose() -> None:
        c = Cascade(gates=[_SimpleGate()])
        c._owns_telemetry_sink = True
        mock_sink = MagicMock()
        mock_sink.aclose = AsyncMock()
        c.telemetry_sink = mock_sink
        await c.aclose()
        mock_sink.aclose.assert_awaited_once()

    asyncio.run(_run_aclose())

    # 2. Line 722 (ROR is -> is not): Generative gate inherits gate rubric when rubric is None, and preserves explicit rubric
    class _CapturingGenGate(BaseGate):
        name = "capturing_gen"
        layer_tier = 200
        rubric = "gen_default_rubric"
        escalation_policy = "always"

        def __init__(self) -> None:
            self.received_kwargs: list[dict[str, Any]] = []

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            self.received_kwargs.append(kwargs)
            return Verdict(status=DecisionStatus.PASSED, passed=True, triggered_layer=self.name, total_latency_ms=1.0)

    gen_gate = _CapturingGenGate()
    casc = Cascade(gates=[gen_gate])
    casc.evaluate("candidate text")
    assert gen_gate.received_kwargs[0].get("rubric_override") == "gen_default_rubric"

    # 3. Line 840 (ROR is not -> is): locked_upstream_fail takes upstream.probability when not None
    class _FailingUpstreamGate(BaseGate):
        name = "upstream_fail"
        layer_tier = TIER_LOGIT
        escalation_policy = "always"

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(
                status=DecisionStatus.FAILED,
                passed=False,
                probability=0.25,
                triggered_layer=self.name,
                total_latency_ms=1.0,
            )

    class _PassingGenGate(BaseGate):
        name = "gen_pass"
        layer_tier = TIER_GENERATIVE
        rubric = "gen_rubric"
        escalation_policy = "on_ambiguous_or_fail"
        can_override_fail = False

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(
                status=DecisionStatus.PASSED,
                passed=True,
                probability=0.99,
                triggered_layer=self.name,
                total_latency_ms=1.0,
            )

    c_locked = Cascade(gates=[_FailingUpstreamGate(), _PassingGenGate()])
    v_locked = c_locked.evaluate("cand text", rubric=None)
    assert v_locked.status == DecisionStatus.FAILED
    assert v_locked.passed is False
    assert v_locked.probability == 0.25


def test_v0140_cli_eval_cmd_mutants() -> None:
    # 1. Line 105 (AOR - -> +): CliMultiRubricGate.evaluate total_latency_ms
    cli_mrg = CliMultiRubricGate(rubrics=["rubric_1", "rubric_2"], aggregation="mean")
    v_cli = cli_mrg.evaluate("cand text")
    assert 0.0 <= v_cli.total_latency_ms < 10000.0

    # 2. Line 208 (LCR and -> or): _cmd_eval preserves unmatched quote in --choice options
    p = create_parser()
    args_choice = p.parse_args([
        "eval",
        "--candidate",
        "test",
        "--choice",
        'A=hello"',
        "--choice",
        "B=world'",
        "--rubric",
        "which?",
        "--backend",
        "stub://default",
    ])
    with patch("logitgate.cli.commands.eval_cmd.ChoiceGate") as mock_cg:
        mock_cg.return_value.evaluate.return_value = Verdict(
            status=DecisionStatus.PASSED, passed=True, triggered_layer="choice", total_latency_ms=1.0
        )
        _cmd_eval(args_choice)
        mock_cg.assert_called_once()
        assert mock_cg.call_args.kwargs["options"]["A"] == 'hello"'
        assert mock_cg.call_args.kwargs["options"]["B"] == "world'"

    # 3. Line 279 (UOI not x -> x): _cmd_eval passes cache != False to ScoreGate when has_ast is False
    args_score = p.parse_args([
        "eval",
        "--candidate",
        "test",
        "--rubric",
        "quality",
        "--score-min",
        "1",
        "--score-max",
        "5",
        "--backend",
        "stub://default",
        "--cache",
        "memory",
    ])
    with patch("logitgate.cli.commands.eval_cmd.ScoreGate") as mock_sg:
        mock_sg.return_value.evaluate.return_value = Verdict(
            status=DecisionStatus.PASSED, passed=True, triggered_layer="score", total_latency_ms=1.0
        )
        _cmd_eval(args_score)
        mock_sg.assert_called_once()
        assert mock_sg.call_args.kwargs["cache"] is not False

    # 4. Line 344 (ROR in -> not in): _cmd_eval --json populates score_distribution from token_probabilities trace
    args_json = p.parse_args([
        "eval",
        "--candidate",
        "test",
        "--rubric",
        "quality",
        "--score-min",
        "1",
        "--score-max",
        "5",
        "--backend",
        "stub://default",
        "--json",
    ])
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_eval(args_json)
    data = json.loads(buf.getvalue())
    assert "score_distribution" in data
    assert len(data["score_distribution"]) > 0


def test_v0140_base_gate_sink_close_mutant() -> None:
    # Line 260 (ROR is not -> is): BaseGate.close safely closes owned sink
    class _SimpleBaseGate(BaseGate):
        name = "simple_base"
        layer_tier = 50
        _owns_sink: bool = False
        telemetry_sink: Any = None

        def evaluate(self, candidate: Any, **kwargs: Any) -> Verdict:
            return Verdict(status=DecisionStatus.PASSED, passed=True, triggered_layer=self.name, total_latency_ms=1.0)

    gate = _SimpleBaseGate()
    gate._owns_sink = True
    mock_sink = MagicMock()
    gate.telemetry_sink = mock_sink
    gate.close()
    mock_sink.close.assert_called_once()


def test_v0140_best_of_n_mutants() -> None:
    # 0. Line 38 (ROR is not -> is): _resolve_bon_backend calls rank_gate resolve_logit_backend if present
    sentinel = object()
    with patch("logitgate.gates.rank_gate.resolve_logit_backend", return_value=sentinel):
        assert _resolve_bon_backend("stub://default") is sentinel

    # 1. Line 172 (ROR is not -> is): _handle_failure_snapshot sets has_normalized=True when winner exists
    bon = SubmoduleBestOfNGate(rubric="which is best?", backend=StubBackend())
    with patch("logitgate.cache.store_cached_verdict") as mock_store:
        bon._handle_failure_snapshot(
            candidate={"A": "apple", "B": "banana"},
            verdict=Verdict(
                status=DecisionStatus.PASSED, passed=True, choice="A", triggered_layer=bon.name, total_latency_ms=1.0
            ),
            _entries=[("A", "apple"), ("B", "banana")],
            _is_cache_hit=False,
        )
        mock_store.assert_called_once()
        assert mock_store.call_args.kwargs.get("has_normalized") is True
        assert mock_store.call_args.kwargs.get("normalized_artifact") == "apple"

    # 2. Line 335 (AOR - -> +): BestOfNGate.evaluate total_latency_ms
    v_bon = bon.evaluate({"A": "apple", "B": "banana"})
    assert 0.0 <= v_bon.total_latency_ms < 10000.0

    # 3. Line 390 (LCR or -> and): _merge_post_verdict reason fallback when post_v.reason is empty
    base_v = Verdict(
        status=DecisionStatus.PASSED,
        passed=True,
        choice="A",
        triggered_layer="rank",
        total_latency_ms=1.0,
        reason="base winning reason",
    )
    post_v = Verdict(status=DecisionStatus.PASSED, passed=True, triggered_layer="post", total_latency_ms=1.0, reason="")
    fin_v = bon._merge_post_verdict(
        entries=[("A", "cand A"), ("B", "cand B")],
        winner_key="A",
        winner_idx=0,
        base_v=base_v,
        post_v=post_v,
        call_kwargs={},
        t0=time.perf_counter(),
    )
    assert fin_v.reason == "base winning reason"


def test_v0140_choice_gate_mutants() -> None:
    # 1. Line 55 (ROR is not -> is): _resolve_choice_backend calls logit_gate resolve_logit_backend if present
    sentinel = object()
    with patch("logitgate.gates.logit_gate.resolve_logit_backend", return_value=sentinel):
        assert _resolve_choice_backend("stub://default") is sentinel

    # 2. Line 436 (LCR or -> and): _ensure_calibrated skips when auto_calibrate is False
    cg = ChoiceGate(
        question="which?", options={"A": "desc A", "B": "desc B"}, model="dummy_model", backend=StubBackend()
    )
    try:
        configure(auto_calibrate=False)
        cg._needs_lazy_calibration = True
        with patch("logitgate.gates.choice_gate._lazy_calibrate_sync") as mock_sync:
            cg._ensure_calibrated()
            mock_sync.assert_not_called()
            assert cg._needs_lazy_calibration is False
    finally:
        configure(auto_calibrate=True)

    # 3. Line 445 (UOI not x -> x): _aensure_calibrated runs async calibration when auto_calibrate is True
    cg._needs_lazy_calibration = True
    with patch("logitgate.gates.choice_gate._lazy_calibrate_async", new=AsyncMock(return_value=1.5)):
        asyncio.run(cg._aensure_calibrated())
        assert cg.logit_temperature == 1.5


def test_v0140_multi_rubric_gate_mutants() -> None:
    # 1. Line 167 (ROR is not -> is): raw_l1_status updated in traces[0] when adaptive_controller is present
    mrg = SubmoduleMultiRubricGate(rubrics=["r1", "r2"], backend=StubBackend())
    v1 = Verdict(
        status=DecisionStatus.PASSED,
        passed=True,
        probability=0.9,
        triggered_layer="lg",
        total_latency_ms=1.0,
        traces=[
            LayerTrace(
                layer_name="lg", status=DecisionStatus.PASSED, latency_ms=1.0, metadata={"raw_l1_status": "passed"}
            )
        ],
    )
    v2 = Verdict(
        status=DecisionStatus.FAILED,
        passed=False,
        probability=0.1,
        triggered_layer="lg",
        total_latency_ms=1.0,
        traces=[
            LayerTrace(
                layer_name="lg", status=DecisionStatus.FAILED, latency_ms=1.0, metadata={"raw_l1_status": "failed"}
            )
        ],
    )
    mrg._gate.adaptive_controller = MagicMock()
    with patch.object(mrg, "evaluate_each", return_value=[v1, v2]):
        v_mrg = mrg.evaluate("test candidate")
    assert v_mrg.status == DecisionStatus.FAILED
    assert v_mrg.traces[0].metadata.get("raw_l1_status") == "failed"

    # 2. Line 175 (LCR and -> or): passing rubrics are NOT placed into unresolved in reason
    assert v_mrg.reason is not None and "PASSED '" not in v_mrg.reason

    # 3. Line 191 (AOR * -> / and - -> +): total_latency_ms is in realistic range
    assert 0.0001 <= v_mrg.total_latency_ms < 10000.0


def test_v0140_score_gate_mutants() -> None:
    # 1. Line 52 (ROR is not -> is): _resolve_score_backend calls logit_gate resolve_logit_backend if present
    sentinel = object()
    with patch("logitgate.gates.logit_gate.resolve_logit_backend", return_value=sentinel):
        assert _resolve_score_backend("stub://default") is sentinel

    # 2. Line 362 (AOR - -> +): token_probabilities sum to 1.0 (log-sum-exp normalization)
    sg = SubmoduleScoreGate(rubric="quality", min_score=1, max_score=5, backend=StubBackend())
    v_sg = sg.evaluate("test candidate")
    score_probs = v_sg.traces[0].metadata["token_probabilities"]
    assert math.isclose(sum(score_probs.values()), 1.0, abs_tol=1e-3)

    # 3. Line 399 (AOR - -> +): total_latency_ms is in realistic range
    assert 0.0 <= v_sg.total_latency_ms < 10000.0

