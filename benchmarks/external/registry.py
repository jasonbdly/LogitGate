"""Canonical External Public Benchmark Registry for LogitGate.

Defines commercial-clean licensing guardrails and metadata specifications for
evaluating LogitGate on canonical external datasets without heavy dataset dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ALLOWED_LICENSES: frozenset[str] = frozenset(
    {"MIT", "Apache-2.0", "ODC-By-1.0", "CC-BY-4.0"}
)
BLOCKED_LICENSES_PREFIX: tuple[str, ...] = ("CC-BY-NC", "GPL", "AGPL", "Proprietary")


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkSpec:
    """Specification and licensing contract for an external benchmark suite."""

    key: str
    hf_dataset_id: str
    config_name: str
    split: str
    license_spdx: str
    task_type: Literal["binary", "pairwise", "best_of_n", "stream_span"]
    description: str
    citation_url: str
    fallback_urls: tuple[str, ...] = ()


def validate_benchmark_license(spec: ExternalBenchmarkSpec) -> None:
    """Validates that a benchmark specification complies with commercial licensing rules.

    Raises:
        ValueError: If the license starts with a blocked prefix (e.g. non-commercial or copyleft)
            or is not explicitly registered in ALLOWED_LICENSES.
    """
    for blocked_prefix in BLOCKED_LICENSES_PREFIX:
        if spec.license_spdx.startswith(blocked_prefix):
            raise ValueError(
                f"Benchmark '{spec.key}' uses blocked license '{spec.license_spdx}' "
                f"(starts with blocked prefix '{blocked_prefix}'). "
                f"Commercial deployment requires non-copyleft, unrestricted licenses."
            )
    if spec.license_spdx not in ALLOWED_LICENSES:
        raise ValueError(
            f"Benchmark '{spec.key}' uses license '{spec.license_spdx}', which is not in "
            f"allowed commercial licenses: {sorted(ALLOWED_LICENSES)}."
        )


EXTERNAL_BENCHMARKS: dict[str, ExternalBenchmarkSpec] = {
    "llmbar": ExternalBenchmarkSpec(
        key="llmbar",
        hf_dataset_id="princeton-nlp/LLMBar",
        config_name="LLMBar",
        split="Adversarial_Neighbor",
        license_spdx="MIT",
        task_type="pairwise",
        description="Objective instruction-following response pairs with adversarial surface-bias distractors from Princeton NLP.",
        citation_url="https://github.com/princeton-nlp/LLMBar",
        fallback_urls=(
            "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar/Adversarial/Neighbor/dataset.json",
        ),
    ),
    "judgebench": ExternalBenchmarkSpec(
        key="judgebench",
        hf_dataset_id="ScalerLab/JudgeBench",
        config_name="default",
        split="gpt",
        license_spdx="MIT",
        task_type="pairwise",
        description="Objective reasoning, coding, and math response pairs for LLM-as-a-judge evaluation from ScalerLab.",
        citation_url="https://huggingface.co/datasets/ScalerLab/JudgeBench",
        fallback_urls=(
            "https://huggingface.co/datasets/ScalerLab/JudgeBench/resolve/main/data/gpt-00000-of-00001.jsonl",
        ),
    ),
    "ragtruth": ExternalBenchmarkSpec(
        key="ragtruth",
        hf_dataset_id="wandb/RAGTruth-processed",
        config_name="default",
        split="test",
        license_spdx="MIT",
        task_type="stream_span",
        description="Real LLM RAG responses with grounded context and fine-grained hallucination labels and spans.",
        citation_url="https://huggingface.co/datasets/wandb/RAGTruth-processed",
        fallback_urls=(
            "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
            "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
        ),
    ),
    "halueval": ExternalBenchmarkSpec(
        key="halueval",
        hf_dataset_id="pminervini/HaluEval",
        config_name="qa_samples",
        split="data",
        license_spdx="MIT",
        task_type="binary",
        description="Question answering groundedness and hallucination pairs for evaluation from pminervini.",
        citation_url="https://huggingface.co/datasets/pminervini/HaluEval",
        fallback_urls=(
            "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
        ),
    ),
    "truthfulqa": ExternalBenchmarkSpec(
        key="truthfulqa",
        hf_dataset_id="truthfulqa/truthful_qa",
        config_name="generation",
        split="validation",
        license_spdx="Apache-2.0",
        task_type="pairwise",
        description="Truthful vs misconception answer pairs constructed from best_answer vs incorrect_answers.",
        citation_url="https://huggingface.co/datasets/truthfulqa/truthful_qa",
        fallback_urls=(
            "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        ),
    ),
    "rewardbench2": ExternalBenchmarkSpec(
        key="rewardbench2",
        hf_dataset_id="allenai/reward-bench-2",
        config_name="default",
        split="test",
        license_spdx="ODC-By-1.0",
        task_type="best_of_n",
        description="4-way human-preference candidate pools for reward and rank evaluation from AllenAI.",
        citation_url="https://huggingface.co/datasets/allenai/reward-bench-2",
        fallback_urls=(),
    ),
}

# Validate all registered canonical suites at module import time
for _spec in EXTERNAL_BENCHMARKS.values():
    validate_benchmark_license(_spec)
