"""Canonical External Public Benchmark Harness for LogitGate.

Provides zero-dependency loading from HuggingFace Datasets Server, commercial-clean
licensing guardrails, schema normalizers, and unified evaluation runners for
LLMBar, JudgeBench, RAGTruth, HaluEval, TruthfulQA, and RewardBench-2.
"""

from __future__ import annotations

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

__all__ = [
    "ALLOWED_LICENSES",
    "BLOCKED_LICENSES_PREFIX",
    "EXTERNAL_BENCHMARKS",
    "ExternalBenchmarkSpec",
    "NormalizedBinaryItem",
    "NormalizedPairwiseItem",
    "NormalizedPoolItem",
    "fetch_raw_hf_rows",
    "load_external_benchmark",
    "normalize_benchmark_row",
    "pairwise_to_binary_cases",
    "validate_benchmark_license",
]
