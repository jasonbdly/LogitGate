"""Unit tests for the LogitGate v0.13.0 Canonical External Public Benchmark Harness.

Covers:
- Commercial-clean license validation and blocked prefix rejection.
- Schema normalizers for all 6 registered suites (LLMBar, JudgeBench, RAGTruth, HaluEval, TruthfulQA, RewardBench-2).
- Deterministic 50/50 slot balancing and candidate shuffling.
- Disk caching and MockTransport single-fetch verification.
- End-to-end evaluation with StubBackend across pairwise, binary, stream-span, and best-of-n suites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.external.loaders import (
    NormalizedBinaryItem,
    NormalizedPairwiseItem,
    NormalizedPoolItem,
    fetch_raw_hf_rows,
    load_external_benchmark,
    normalize_benchmark_row,
    pairwise_to_binary_cases,
)
from benchmarks.external.registry import (
    ALLOWED_LICENSES,
    BLOCKED_LICENSES_PREFIX,
    EXTERNAL_BENCHMARKS,
    ExternalBenchmarkSpec,
    validate_benchmark_license,
)
from benchmarks.external.run_external import (
    evaluate_best_of_n,
    evaluate_binary,
    evaluate_pairwise,
    evaluate_stream_guard_on_ragtruth,
    wilson_score_interval,
)
from logitgate.backends.base import StubBackend


class TestLicenseValidation:
    """Verifies commercial-clean license contracts for external benchmarks."""

    def test_canonical_benchmarks_all_allowed(self) -> None:
        """Every canonical benchmark registered in EXTERNAL_BENCHMARKS must satisfy license policy."""
        assert len(EXTERNAL_BENCHMARKS) == 6
        for key, spec in EXTERNAL_BENCHMARKS.items():
            assert spec.key == key
            assert spec.license_spdx in ALLOWED_LICENSES
            # Must not raise
            validate_benchmark_license(spec)

    def test_rejects_non_commercial_prefixes(self) -> None:
        """Benchmarks with CC-BY-NC or other non-commercial licenses must be rejected."""
        toxic_chat_spec = ExternalBenchmarkSpec(
            key="toxic-chat",
            hf_dataset_id="lmsys/toxic-chat",
            config_name="default",
            split="test",
            license_spdx="CC-BY-NC-4.0",
            task_type="binary",
            description="Non-commercial toxic chat dataset.",
            citation_url="https://huggingface.co/datasets/lmsys/toxic-chat",
        )
        with pytest.raises(ValueError, match="blocked prefix 'CC-BY-NC'"):
            validate_benchmark_license(toxic_chat_spec)

    @pytest.mark.parametrize("blocked_prefix", BLOCKED_LICENSES_PREFIX)
    def test_rejects_all_blocked_prefixes(self, blocked_prefix: str) -> None:
        spec = ExternalBenchmarkSpec(
            key="blocked_test",
            hf_dataset_id="test/blocked",
            config_name="default",
            split="test",
            license_spdx=f"{blocked_prefix}-1.0",
            task_type="binary",
            description="Test blocked prefix.",
            citation_url="https://example.com",
        )
        with pytest.raises(ValueError, match="blocked prefix"):
            validate_benchmark_license(spec)

    def test_rejects_unregistered_clean_licenses(self) -> None:
        """Even non-copyleft licenses not explicitly in ALLOWED_LICENSES must be rejected."""
        bsd_spec = ExternalBenchmarkSpec(
            key="bsd_test",
            hf_dataset_id="test/bsd",
            config_name="default",
            split="test",
            license_spdx="BSD-3-Clause",
            task_type="binary",
            description="BSD licensed dataset.",
            citation_url="https://example.com",
        )
        with pytest.raises(ValueError, match="not in allowed commercial licenses"):
            validate_benchmark_license(bsd_spec)


class TestRowNormalizers:
    """Verifies raw HuggingFace row conversion for all 6 suites."""

    def test_normalize_llmbar_row(self) -> None:
        row = {
            "input": "Summarize the text in 10 words.",
            "output_1": "Concise summary adhering strictly to the ten word length limit.",
            "output_2": "This is a very long response that completely violates the ten word limit constraint.",
            "label": 1,
        }
        item = normalize_benchmark_row("llmbar", row, idx=0, seed=123)
        assert isinstance(item, NormalizedPairwiseItem)
        assert "Summarize the text in 10 words." in item.rubric
        assert item.metadata["original_label"] == 1
        # The winner candidate must correspond to output_1
        winner_text = item.candidate_a if item.winner == 0 else item.candidate_b
        assert winner_text == row["output_1"]

    def test_normalize_judgebench_row(self) -> None:
        row = {
            "pair_id": "jb_uuid_101",
            "question": "What is the time complexity of quicksort in the worst case?",
            "response_A": "O(n^2) when poor pivots are chosen repeatedly.",
            "response_B": "O(log n) always regardless of pivot selection.",
            "label": "A>B",
            "source": "mmlu-pro-cs",
        }
        item = normalize_benchmark_row("judgebench", row, idx=0, seed=42)
        assert isinstance(item, NormalizedPairwiseItem)
        assert "quicksort" in item.rubric
        winner_text = item.candidate_a if item.winner == 0 else item.candidate_b
        assert winner_text == row["response_A"]

    def test_normalize_truthfulqa_row(self) -> None:
        row = {
            "question": "What happens if you swallow watermelon seeds?",
            "best_answer": "They pass through the digestive system harmlessly.",
            "incorrect_answers": ["A watermelon tree will grow inside your stomach."],
        }
        item = normalize_benchmark_row("truthfulqa", row, idx=0, seed=42)
        assert isinstance(item, NormalizedPairwiseItem)
        assert "watermelon" in item.rubric
        winner_text = item.candidate_a if item.winner == 0 else item.candidate_b
        assert winner_text == row["best_answer"]

    def test_normalize_halueval_row(self) -> None:
        row_grounded = {
            "knowledge": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answer": "Paris is the capital.",
            "hallucination": "no",
        }
        item_g = normalize_benchmark_row("halueval", row_grounded, idx=0)
        assert isinstance(item_g, NormalizedBinaryItem)
        assert item_g.expected is True
        assert "Paris" in item_g.candidate

        row_hallu = {
            "knowledge": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answer": "Berlin is the capital of France.",
            "hallucination": "yes",
        }
        item_h = normalize_benchmark_row("halueval", row_hallu, idx=1)
        assert isinstance(item_h, NormalizedBinaryItem)
        assert item_h.expected is False

    def test_normalize_ragtruth_row(self) -> None:
        row_clean = {
            "id": "rag_1",
            "question": "What is the warranty period?",
            "context": "The standard warranty covers parts for 2 years.",
            "answer": "Parts are covered under warranty for 2 years.",
            "labels": [],
        }
        item_c = normalize_benchmark_row("ragtruth", row_clean, idx=0)
        assert isinstance(item_c, NormalizedBinaryItem)
        assert item_c.expected is True
        assert item_c.metadata["has_hallucination"] is False

        row_hallu = {
            "id": "rag_2",
            "question": "What is the warranty period?",
            "context": "The standard warranty covers parts for 2 years.",
            "answer": "Parts and labor are covered for 10 years with free lifetime replacements.",
            "labels": [{"start": 16, "end": 40}],
        }
        item_h = normalize_benchmark_row("ragtruth", row_hallu, idx=1)
        assert isinstance(item_h, NormalizedBinaryItem)
        assert item_h.expected is False
        assert item_h.metadata["has_hallucination"] is True

    def test_normalize_rewardbench2_row(self) -> None:
        row = {
            "id": "rb2_001",
            "prompt": "Explain gradient descent in one sentence.",
            "chosen": [
                "Gradient descent iteratively updates parameters in the direction of steepest loss decrease."
            ],
            "rejected": [
                "It ascends the hill randomly.",
                "It calculates the second derivative always.",
                "It multiplies everything by zero.",
            ],
        }
        item = normalize_benchmark_row("rewardbench2", row, idx=0, seed=42)
        assert isinstance(item, NormalizedPoolItem)
        assert len(item.candidates) == 4
        # Winner must point to the chosen response
        assert item.candidates[item.expected_winner] == row["chosen"][0]

    def test_unknown_suite_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark key 'unknown_suite'"):
            normalize_benchmark_row("unknown_suite", {})


class TestSlotBalancing:
    """Verifies deterministic seeded slot-balancing so gold answers aren't fixed to slot 0."""

    def test_truthfulqa_50_50_balance(self) -> None:
        n_samples = 60
        winners: list[int] = []
        for i in range(n_samples):
            row = {
                "id": f"tqa_{i}",
                "question": f"Question {i}?",
                "best_answer": f"Truth {i}",
                "incorrect_answers": [f"Myth {i}"],
            }
            item = normalize_benchmark_row("truthfulqa", row, idx=i, seed=42)
            assert isinstance(item, NormalizedPairwiseItem)
            winners.append(item.winner)

        count_0 = winners.count(0)
        count_1 = winners.count(1)
        # Verify both slots are populated and roughly balanced
        assert count_0 > 0.30 * n_samples
        assert count_1 > 0.30 * n_samples
        assert count_0 + count_1 == n_samples

        # Verify determinism: same seed yields identical sequence
        repeat_winners: list[int] = []
        for i in range(n_samples):
            row = {
                "id": f"tqa_{i}",
                "question": f"Question {i}?",
                "best_answer": f"Truth {i}",
                "incorrect_answers": [f"Myth {i}"],
            }
            item = normalize_benchmark_row("truthfulqa", row, idx=i, seed=42)
            assert isinstance(item, NormalizedPairwiseItem)
            repeat_winners.append(item.winner)
        assert winners == repeat_winners

    def test_rewardbench2_pool_slot_shuffling(self) -> None:
        n_samples = 60
        chosen_slots: list[int] = []
        for i in range(n_samples):
            row = {
                "id": f"rb2_{i}",
                "prompt": f"Prompt {i}",
                "chosen": ["Gold Answer"],
                "rejected": ["Bad 1", "Bad 2", "Bad 3"],
            }
            item = normalize_benchmark_row("rewardbench2", row, idx=i, seed=42)
            assert isinstance(item, NormalizedPoolItem)
            chosen_slots.append(item.expected_winner)

        # The chosen item must be distributed across slots (not stuck in slot 0)
        unique_slots = set(chosen_slots)
        assert len(unique_slots) == 4  # Covers 0, 1, 2, 3
        assert (
            0 in unique_slots
            and 1 in unique_slots
            and 2 in unique_slots
            and 3 in unique_slots
        )


class TestPairwiseToBinaryCases:
    """Verifies conversion of pairwise items into balanced binary BenchmarkCase pairs."""

    def test_pairwise_to_binary_conversion(self) -> None:
        items = [
            NormalizedPairwiseItem(
                id="p1",
                rubric="Rubric 1",
                candidate_a="Answer A (Best)",
                candidate_b="Answer B (Worst)",
                winner=0,
            ),
            NormalizedPairwiseItem(
                id="p2",
                rubric="Rubric 2",
                candidate_a="Answer A (Worst)",
                candidate_b="Answer B (Best)",
                winner=1,
            ),
        ]
        cases = pairwise_to_binary_cases(items)
        assert len(cases) == 4

        # Case 1 & 2 from item p1 (winner=0 -> cand_a is true, cand_b is false)
        assert cases[0].candidate == "Answer A (Best)"
        assert cases[0].expected is True
        assert cases[1].candidate == "Answer B (Worst)"
        assert cases[1].expected is False

        # Case 3 & 4 from item p2 (winner=1 -> cand_b is true, cand_a is false)
        assert cases[2].candidate == "Answer B (Best)"
        assert cases[2].expected is True
        assert cases[3].candidate == "Answer A (Worst)"
        assert cases[3].expected is False


class TestDiskCacheAndMockTransport:
    """Verifies HuggingFace Datasets Server querying and offline disk caching."""

    def test_disk_cache_hit_on_second_load(self, tmp_path: Path) -> None:
        hf_calls = 0

        def mock_hf_handler(request: httpx.Request) -> httpx.Response:
            nonlocal hf_calls
            hf_calls += 1
            payload = {
                "rows": [
                    {
                        "row_idx": 0,
                        "row": {
                            "input": "Instruction 1",
                            "output_1": "Output 1",
                            "output_2": "Output 2",
                            "label": 1,
                        },
                    },
                    {
                        "row_idx": 1,
                        "row": {
                            "input": "Instruction 2",
                            "output_1": "Output 1",
                            "output_2": "Output 2",
                            "label": 2,
                        },
                    },
                ],
                "num_rows_per_page": 2,
            }
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(mock_hf_handler)
        client = httpx.Client(transport=transport)

        spec = EXTERNAL_BENCHMARKS["llmbar"]

        # First load: cache miss -> queries mock server
        rows_1 = fetch_raw_hf_rows(spec, limit=2, client=client, cache_dir=tmp_path)
        assert len(rows_1) == 2
        assert hf_calls == 1

        # Second load: cache hit -> must not call mock server again
        rows_2 = fetch_raw_hf_rows(spec, limit=2, client=client, cache_dir=tmp_path)
        assert len(rows_2) == 2
        assert hf_calls == 1  # Remained 1!
        assert rows_1 == rows_2

    def test_local_path_fixture_loading(self, tmp_path: Path) -> None:
        fixture_file = tmp_path / "mock_judgebench.json"
        fixture_data = [
            {
                "pair_id": "loc_1",
                "question": "Question 1",
                "response_A": "Ans A",
                "response_B": "Ans B",
                "label": "A>B",
            },
            {
                "pair_id": "loc_2",
                "question": "Question 2",
                "response_A": "Ans A",
                "response_B": "Ans B",
                "label": "B>A",
            },
        ]
        fixture_file.write_text(json.dumps(fixture_data), encoding="utf-8")

        items = load_external_benchmark("judgebench", local_path=fixture_file)
        assert len(items) == 2
        assert isinstance(items[0], NormalizedPairwiseItem)
        assert isinstance(items[1], NormalizedPairwiseItem)

    def test_fallback_urls_on_hf_rows_503(self, tmp_path: Path) -> None:
        """When /rows returns 503, loader must automatically fetch from fallback_urls."""

        def mock_handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "datasets-server.huggingface.co" in url_str:
                return httpx.Response(503, text="Service Temporarily Unavailable")
            if "githubusercontent.com" in url_str or "resolve/main" in url_str:
                sample_data = [
                    {
                        "input": "Fallback Instruction",
                        "output_1": "Fallback 1",
                        "output_2": "Fallback 2",
                        "label": 1,
                    }
                ]
                return httpx.Response(200, json=sample_data)
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_handler)
        client = httpx.Client(transport=transport)
        spec = EXTERNAL_BENCHMARKS["llmbar"]

        rows = fetch_raw_hf_rows(spec, limit=1, client=client, cache_dir=tmp_path)
        assert len(rows) == 1
        assert rows[0]["input"] == "Fallback Instruction"


class TestWilsonScoreInterval:
    """Verifies Wilson 95% confidence interval calculations."""

    def test_wilson_ci_bounds(self) -> None:
        assert wilson_score_interval(0, 0) == (0.0, 0.0)
        low, high = wilson_score_interval(90, 100)
        assert 0.82 < low < 0.83
        assert 0.94 < high < 0.95
        assert 0.0 <= low <= high <= 1.0

        low_0, high_0 = wilson_score_interval(0, 10)
        assert low_0 == 0.0
        assert high_0 > 0.20

        low_full, high_full = wilson_score_interval(10, 10)
        assert low_full > 0.70
        assert high_full == 1.0


class TestRunExternalExecution:
    """Verifies end-to-end evaluation with StubBackend across all benchmark modalities."""

    def test_pairwise_evaluation_with_stub_backend(self) -> None:
        items = [
            NormalizedPairwiseItem(
                id="p1",
                rubric="Pick better",
                candidate_a="Best",
                candidate_b="Bad",
                winner=0,
            ),
            NormalizedPairwiseItem(
                id="p2",
                rubric="Pick better",
                candidate_a="Bad",
                candidate_b="Best",
                winner=1,
            ),
        ]
        # StubBackend default_prob=0.95 favors candidate_a (A > B)
        stub = StubBackend(default_prob=0.95)
        res = evaluate_pairwise(items, stub, symmetric=True, progress=False)

        assert res["task_type"] == "pairwise"
        assert res["total"] == 2
        assert "accuracy" in res
        assert "position_swap_consistency_rate" in res
        assert "latencies" in res
        assert res["latencies"]["mean_ms"] >= 0.0

    def test_binary_evaluation_with_stub_backend(self) -> None:
        items = [
            NormalizedBinaryItem(
                id="b1", rubric="Correct?", candidate="True fact", expected=True
            ),
            NormalizedBinaryItem(
                id="b2", rubric="Correct?", candidate="False myth", expected=False
            ),
        ]
        stub = StubBackend(default_prob=0.95)
        res = evaluate_binary(items, stub, modes=("logit",))
        assert res["task_type"] == "binary"
        assert res["total"] == 2
        assert "benchmark_report" in res

    def test_stream_guard_on_ragtruth_with_stub_backend(self) -> None:
        items = [
            NormalizedBinaryItem(
                id="rag_clean",
                rubric="Check grounding",
                candidate="Fully grounded facts without any hallucination.",
                expected=True,
                metadata={"answer": "Fully grounded facts without any hallucination."},
            ),
            NormalizedBinaryItem(
                id="rag_hallu",
                rubric="Check grounding",
                candidate="This text contains hallucinated claims about nonexistent things.",
                expected=False,
                metadata={
                    "answer": "This text contains hallucinated claims about nonexistent things."
                },
            ),
        ]
        # Low prob on hallucination causes StreamGuard abort
        stub_fail = StubBackend(default_prob=0.01)
        res = evaluate_stream_guard_on_ragtruth(
            items, stub_fail, chunk_chars=15, stride_chars=15, progress=False
        )

        assert res["total_items"] == 2
        assert res["total_violations"] == 1
        assert res["total_compliant"] == 1
        assert res["violations_caught"] == 1
        assert res["violation_catch_rate"] == 1.0
        assert res["avg_tokens_saved_on_caught"] >= 0

    def test_best_of_n_evaluation_with_stub_backend(self) -> None:
        items = [
            NormalizedPoolItem(
                id="rb1",
                rubric="Pick best",
                candidates=["Gold", "Junk 1", "Junk 2", "Junk 3"],
                expected_winner=0,
            ),
            NormalizedPoolItem(
                id="rb2",
                rubric="Pick best",
                candidates=["Junk 1", "Gold", "Junk 2", "Junk 3"],
                expected_winner=1,
            ),
        ]
        stub = StubBackend(default_prob=0.95)
        res = evaluate_best_of_n(items, stub, progress=False)

        assert res["task_type"] == "best_of_n"
        assert res["total"] == 2
        assert "accuracy" in res
        assert "accuracy_ci_95" in res
        assert "latencies" in res


def test_ragtruth_processed_json_string_labels_and_halueval_raw_pairs() -> None:
    from benchmarks.external.loaders import (
        normalize_halueval_row,
        normalize_ragtruth_row,
    )

    # wandb/RAGTruth-processed stores hallucination_labels as a JSON string + query/output keys
    rt_clean = normalize_ragtruth_row(
        {
            "id": "24",
            "query": "Summarize news:",
            "context": "Article text.",
            "output": "Faithful summary.",
            "hallucination_labels": "[]",
            "hallucination_labels_processed": {"evident_conflict": 0, "baseless_info": 0},
        }
    )
    assert rt_clean.expected is True

    rt_hallu = normalize_ragtruth_row(
        {
            "id": "28",
            "query": "Summarize news:",
            "context": "Article text.",
            "output": "Contradicted summary.",
            "hallucination_labels": '[{"start": 0, "end": 12, "label_type": "Evident Conflict"}]',
            "hallucination_labels_processed": {"evident_conflict": 1, "baseless_info": 0},
        }
    )
    assert rt_hallu.expected is False
    assert rt_hallu.metadata["has_hallucination"] is True

    # RUCAIBox/HaluEval raw lines provide right_answer + hallucinated_answer without hallucination key
    he_labels = [
        normalize_halueval_row(
            {
                "knowledge": "Fact.",
                "question": f"Q{i}?",
                "right_answer": "True answer.",
                "hallucinated_answer": "False answer.",
            },
            idx=i,
            seed=42,
        ).expected
        for i in range(40)
    ]
    assert 12 <= sum(he_labels) <= 28

