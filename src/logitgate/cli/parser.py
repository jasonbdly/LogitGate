from __future__ import annotations

import argparse


def _register_eval_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate candidate artifact against a rubric",
    )
    eval_parser.add_argument(
        "--rubric",
        action="append",
        default=None,
        help="Evaluation rubric or question to verify against candidate (repeatable for MultiRubricGate)",
    )
    eval_parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Candidate text/artifact to evaluate",
    )
    eval_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file containing candidate text to evaluate",
    )
    eval_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass probability threshold (defaults to LOGITGATE_PASS_THRESHOLD or 0.85)",
    )
    eval_parser.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Fail probability threshold (defaults to LOGITGATE_FAIL_THRESHOLD or 0.20)",
    )
    eval_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a CalibrationProfile JSON file",
    )
    eval_parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to a DistillationBundle JSON file (anchors + meta-calibrator)",
    )
    eval_parser.add_argument(
        "--ast-syntax",
        action="store_true",
        help="Enable Layer 0 AST syntax validation before logit evaluation",
    )
    eval_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Strip <think> blocks and unwrap markdown code fences before Layer 0 validation",
    )
    eval_parser.add_argument(
        "--telemetry-out",
        type=str,
        default=None,
        help="Optional JSONL path to append structured verdict telemetry",
    )
    eval_parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional SQLite VerdictCache path (or 'memory' for in-process LRU)",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Output verdict and traces in JSON format",
    )
    eval_parser.add_argument(
        "--aggregation",
        choices=("all", "any", "mean"),
        default="all",
        help="Multi-rubric verdict aggregation strategy: 'all', 'any', or 'mean' (default: all)",
    )
    eval_parser.add_argument(
        "--choice",
        action="append",
        default=[],
        help="Discrete choice option in KEY=TEXT format (repeatable, minimum 2 options)",
    )
    eval_parser.add_argument(
        "--expected-choice",
        type=str,
        default=None,
        help="Optional expected choice key required for a PASSED verdict",
    )
    eval_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional minimum top-choice probability cutoff below which ChoiceGate abstains",
    )
    eval_parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Optional top-2 choice probability margin cutoff below which ChoiceGate abstains",
    )
    eval_parser.add_argument(
        "--score-min",
        type=int,
        default=None,
        help="Minimum integer score rating for ScoreGate (0 <= min < max <= 9)",
    )
    eval_parser.add_argument(
        "--score-max",
        type=int,
        default=None,
        help="Maximum integer score rating for ScoreGate (0 <= min < max <= 9)",
    )
    eval_parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Score threshold required for a PASSED verdict in ScoreGate",
    )
    eval_parser.add_argument(
        "--context",
        action="append",
        default=None,
        help="Retrieved context chunk to condition the evaluation (repeatable)",
    )
    eval_parser.add_argument(
        "--context-file",
        type=str,
        default=None,
        help="Path to a text file containing reference context",
    )
    eval_parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="User query or question corresponding to the candidate response",
    )
    eval_parser.add_argument(
        "--on-unsupported-logprobs",
        choices=("raise", "escalate", "sidecar"),
        default="raise",
        help="Policy when backend does not support logprobs: 'raise', 'escalate', or 'sidecar' (default: raise)",
    )
    eval_parser.add_argument(
        "--sidecar-backend",
        type=str,
        default=None,
        help="Sidecar backend specification for logprob estimation when primary backend does not support logprobs",
    )
    eval_parser.add_argument(
        "--tool-call",
        action="store_true",
        help="Evaluate candidate tool calls via ToolCallGate (deterministic Layer-0 contract and optional Layer-1 intent)",
    )
    eval_parser.add_argument(
        "--allowed-tool",
        action="append",
        default=None,
        help="Permitted tool name for ToolCallGate (repeatable)",
    )
    eval_parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="Permitted filesystem root directory for ToolCallGate (repeatable)",
    )
    eval_parser.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        help="Permitted network host for ToolCallGate (repeatable)",
    )
    return eval_parser


def _register_calibrate_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Run micro-calibration dataset to fit optimal temperature or conformal cascade thresholds",
    )
    cal_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    cal_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    cal_parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Optional custom path to calibration cache JSON file",
    )
    cal_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to holdout dataset (JSON, JSONL, or Markdown) for Split Conformal Cascade calibration",
    )
    cal_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target risk bound alpha for selective classification (default: 0.05)",
    )
    cal_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save a portable CalibrationProfile JSON file",
    )
    cal_parser.add_argument(
        "--json",
        action="store_true",
        help="Output calibration results in JSON format",
    )
    return cal_parser


def _register_explain_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    explain_parser = subparsers.add_parser(
        "explain",
        help="Inspect a failure snapshot and (optionally) replay it through a Layer-2 judge",
    )
    explain_parser.add_argument(
        "snapshot_path",
        nargs="?",
        default=".logitgate/failures",
        help="Path to snapshot JSON file or directory containing latest.json (default: .logitgate/failures)",
    )
    explain_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Generative backend specifier (e.g. stub://..., ollama://...)",
    )
    explain_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    explain_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for generative explanation",
    )
    explain_parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Chain-of-Thought reasoning tokens in generative explanation",
    )
    explain_parser.add_argument(
        "--offline",
        action="store_true",
        help="Only render recorded diagnostics; do not call any model",
    )
    explain_parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize output (default: auto; honors NO_COLOR / FORCE_COLOR)",
    )
    explain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output enriched verdict and traces in JSON format",
    )
    return explain_parser


def _register_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Measure latency, generated tokens, and accuracy: generative judge vs LogitGate vs cascade",
        description=(
            "Runs the same cases through a generative judge, a LogitGate, and a LogitGate >> "
            "JudgeGate cascade on YOUR backend, and reports p50/p95 latency, generated tokens, "
            "LLM calls, and accuracy. Use --json to archive results with full environment metadata."
        ),
    )
    bench_parser.add_argument(
        "--backend", type=str, default=None, help="Backend specifier, e.g. ollama://localhost:11434"
    )
    bench_parser.add_argument("--model", type=str, default=None, help="Model name for the backend")
    bench_parser.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        help="Optional separate backend specifier for JudgeGate (heterogeneous cascade)",
    )
    bench_parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional separate model name for JudgeGate (defaults to --model)",
    )
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSON/JSONL/Markdown with rubric, candidate, label (default: built-in 16-case set)",
    )
    bench_parser.add_argument(
        "--modes",
        type=str,
        default="judge,logit,cascade",
        help="Comma-separated subset of judge,logit,cascade (default: all)",
    )
    bench_parser.add_argument(
        "--repeat", type=int, default=1, help="Timed passes over the dataset per mode (default: 1)"
    )
    bench_parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed calls per mode before timing (default: 1)"
    )
    bench_parser.add_argument("--limit", type=int, default=None, help="Use only the first N cases")
    bench_parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=None,
        help="Cap on judge generation (default: backend default)",
    )
    bench_parser.add_argument(
        "--judge-think",
        action="store_true",
        help="Let the judge emit reasoning tokens (Ollama). Off by default: a fast judge is the fair baseline.",
    )
    bench_parser.add_argument(
        "--cascade-trigger",
        choices=("on_ambiguous_only", "on_ambiguous_or_fail", "adjudicate", "always"),
        default="on_ambiguous_only",
        help="JudgeGate escalation policy inside the cascade (default: on_ambiguous_only)",
    )
    bench_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a CalibrationProfile JSON file for LogitGate / cascade modes",
    )
    bench_parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to a DistillationBundle JSON file for LogitGate / cascade modes",
    )
    bench_parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional SQLite VerdictCache path (or 'memory' for in-process LRU across repeats)",
    )
    bench_parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit dataset for length confounders, lexical shortcuts, conformal power, and split leakage without calling any model",
    )
    bench_parser.add_argument(
        "--reference-dataset",
        type=str,
        default=None,
        help="Optional reference (calibration/training) dataset path to check for cross-split leakage with --audit-only",
    )
    bench_parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="With --audit-only, exit with code 1 if any dataset confounder warnings or cross-split leakage are detected",
    )
    bench_parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    bench_parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress on stderr")
    return bench_parser


def _register_rank_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    rank_parser = subparsers.add_parser(
        "rank",
        help="Rank N >= 2 candidate artifacts and select the winner without token generation",
    )
    rank_parser.add_argument(
        "--rubric",
        required=True,
        help="Evaluation rubric or preference criterion for comparing candidates",
    )
    rank_parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate text/artifact to include (repeatable)",
    )
    rank_parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Path to candidate text file to include (repeatable)",
    )
    rank_parser.add_argument(
        "--candidates-json",
        type=str,
        default=None,
        help="Path to a JSON list or key->candidate object of candidates",
    )
    rank_parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional shared task prompt or reference context",
    )
    rank_parser.add_argument(
        "--mode",
        choices=("auto", "pairwise", "listwise", "round_robin", "tournament"),
        default="auto",
        help="Ranking strategy (default: auto)",
    )
    rank_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional selective top-1 probability cutoff below which RankGate abstains",
    )
    rank_parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Optional top-2 probability margin cutoff below which RankGate abstains",
    )
    rank_parser.add_argument(
        "--max-positional-bias",
        type=float,
        default=None,
        help="Optional maximum allowed positional swap bias before abstaining",
    )
    rank_parser.add_argument(
        "--ast-syntax",
        action="store_true",
        help="Pre-filter candidates with ASTGate(syntax=True) before ranking",
    )
    rank_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Strip <think> blocks and unwrap markdown fences during AST pre-filtering",
    )
    rank_parser.add_argument(
        "--verify-winner",
        action="store_true",
        help="Verify the winning candidate against --rubric with a post-ranking LogitGate",
    )
    rank_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a CalibrationProfile JSON file",
    )
    rank_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    rank_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    rank_parser.add_argument(
        "--telemetry-out",
        type=str,
        default=None,
        help="Optional JSONL path to append structured verdict telemetry",
    )
    rank_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BestOfNResult, probabilities, and traces in JSON format",
    )
    return rank_parser


def _register_telemetry_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Summarize production telemetry, export SFT/DPO datasets, or zero-call recalibrate profiles",
    )
    tel_sub = telemetry_parser.add_subparsers(dest="telemetry_command", help="Telemetry subcommands")

    tel_summary = tel_sub.add_parser(
        "summary",
        help="Compute pass/escalation/discrepancy/latency summary from a telemetry JSONL file",
    )
    tel_summary.add_argument("input_path", help="Path to telemetry JSONL file")
    tel_summary.add_argument("--json", action="store_true", help="Output summary as JSON")

    tel_export = tel_sub.add_parser(
        "export",
        help="Export teacher-adjudicated escalations to SFT, DPO, or calibration JSONL",
    )
    tel_export.add_argument("input_path", help="Path to distillation or telemetry JSONL file")
    tel_export.add_argument(
        "--output", "-o", required=True, help="Destination JSONL dataset path"
    )
    tel_export.add_argument(
        "--format",
        choices=("sft_jsonl", "calibration_jsonl", "dpo_jsonl"),
        default="sft_jsonl",
        help="Export dataset format (default: sft_jsonl)",
    )
    tel_export.add_argument(
        "--no-symmetric-swaps",
        action="store_true",
        help="Emit only forward A=Pass/B=Fail prompts instead of symmetric A/B + B/A pairs",
    )
    tel_export.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate (rubric, candidate) pairs instead of deduplicating by SHA-256",
    )
    tel_export.add_argument(
        "--discrepancies-only",
        action="store_true",
        help="Export only student/teacher disagreements, omitting concordant ambiguous resolutions",
    )
    tel_export.add_argument("--json", action="store_true", help="Output export metadata as JSON")

    tel_recal = tel_sub.add_parser(
        "recalibrate",
        help="Fit a CalibrationProfile from logged student/teacher escalations with 0 LLM calls",
    )
    tel_recal.add_argument("input_path", help="Path to distillation or telemetry JSONL file")
    tel_recal.add_argument(
        "--output", "-o", type=str, default=None, help="Optional path to save CalibrationProfile JSON"
    )
    tel_recal.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Target selective risk bound alpha (default: 0.10)",
    )
    tel_recal.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional student model name override for the profile",
    )
    tel_recal.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate (rubric, candidate) escalations during fitting",
    )
    tel_recal.add_argument("--json", action="store_true", help="Output CalibrationProfile as JSON")

    tel_drift = tel_sub.add_parser(
        "drift",
        help="Detect covariate, calibration, and teacher-disagreement drift between two telemetry JSONL logs",
    )
    tel_drift.add_argument("input_path", type=str, help="Path to current evaluation telemetry JSONL file")
    tel_drift.add_argument(
        "--reference",
        "-r",
        required=True,
        type=str,
        help="Path to baseline reference telemetry JSONL file",
    )
    tel_drift.add_argument(
        "--psi-moderate",
        type=float,
        default=0.10,
        help="Moderate Population Stability Index (PSI) threshold (default: 0.10)",
    )
    tel_drift.add_argument(
        "--psi-severe",
        type=float,
        default=0.25,
        help="Severe Population Stability Index (PSI) threshold (default: 0.25)",
    )
    tel_drift.add_argument(
        "--ks-alpha",
        type=float,
        default=0.01,
        help="Two-sample Kolmogorov-Smirnov p-value threshold (default: 0.01)",
    )
    tel_drift.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 when drift_detected is True (for CI/monitoring gates)",
    )
    tel_drift.add_argument("--json", action="store_true", help="Output DriftReport as JSON")
    return telemetry_parser


def _register_stream_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    stream_parser = subparsers.add_parser(
        "stream",
        help="Verify a token/text stream mid-generation with StreamGuard early abort or buffered delivery",
    )
    stream_parser.add_argument(
        "--rubric",
        required=True,
        help="Evaluation rubric verified periodically against the growing stream prefix",
    )
    stream_parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Candidate text to simulate streaming in chunks",
    )
    stream_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file to stream in chunks (falls back to stdin when piped)",
    )
    stream_parser.add_argument(
        "--chunk-size",
        type=int,
        default=48,
        help="Simulated chunk size in characters (default: 48)",
    )
    stream_parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Minimum accumulated characters before first mid-stream check (default: 80)",
    )
    stream_parser.add_argument(
        "--stride-chars",
        type=int,
        default=160,
        help="Character stride between consecutive mid-stream checks (default: 160)",
    )
    stream_parser.add_argument(
        "--delivery",
        choices=("optimistic", "buffered"),
        default="optimistic",
        help="Chunk delivery strategy: 'optimistic' (zero-buffer) or 'buffered' (zero-leakage)",
    )
    stream_parser.add_argument(
        "--consecutive-failures",
        type=int,
        default=1,
        help="Consecutive failing checkpoints required before aborting mid-stream (default: 1)",
    )
    stream_parser.add_argument(
        "--no-clause-boundary",
        action="store_true",
        help="Disable clause/sentence boundary alignment for mid-stream checks",
    )
    stream_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Pass probability threshold",
    )
    stream_parser.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Fail probability threshold",
    )
    stream_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional CalibrationProfile JSON path",
    )
    stream_parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Optional DistillationBundle JSON path",
    )
    stream_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend specifier (defaults to LOGITGATE_BACKEND or stub://default)",
    )
    stream_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name for backend",
    )
    stream_parser.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        help="Optional Tier-2 JudgeGate backend specifier for cascaded final verification",
    )
    stream_parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional Tier-2 JudgeGate model name",
    )
    stream_parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional SQLite VerdictCache path (or 'memory' for in-process LRU)",
    )
    stream_parser.add_argument(
        "--telemetry-out",
        type=str,
        default=None,
        help="Optional JSONL path to append structured stream telemetry",
    )
    stream_parser.add_argument(
        "--quiet-stream",
        action="store_true",
        help="Suppress printing yielded text before summary output",
    )
    stream_parser.add_argument(
        "--json",
        action="store_true",
        help="Output StreamGuardSummary and yielded text in JSON format",
    )
    return stream_parser


def _register_distill_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    distill_parser = subparsers.add_parser(
        "distill",
        help="Compile BYO teacher-to-student distillation bundles (Tier A) or PEFT LoRA adapters (Tier B)",
    )
    distill_sub = distill_parser.add_subparsers(dest="distill_cmd", required=True)

    distill_compile = distill_sub.add_parser(
        "compile",
        help="Compile a portable DistillationBundle JSON (DistilledAnchorBank + optional LogitMetaCalibrator)",
    )
    distill_compile.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Split-A dataset path (.json, .jsonl, or .md) to mine contrastive minimal-pair anchors",
    )
    distill_compile.add_argument(
        "--telemetry",
        type=str,
        default=None,
        help="DistillationSink / TelemetrySink JSONL path to compile anchors and fit 0-call meta-calibrator",
    )
    distill_compile.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Output path for the compiled DistillationBundle JSON file",
    )
    distill_compile.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional student backend specifier to fit a 5-feature LogitMetaCalibrator on --dataset",
    )
    distill_compile.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional student model identifier",
    )
    distill_compile.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Optional teacher model identifier recorded in bundle metadata",
    )
    distill_compile.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Target out-of-fold split-conformal selective risk alpha (default: 0.10)",
    )
    distill_compile.add_argument(
        "--max-pairs",
        type=int,
        default=24,
        help="Maximum contrastive pairs stored in DistilledAnchorBank (default: 24)",
    )
    distill_compile.add_argument(
        "--retrieve-pairs",
        type=int,
        default=1,
        help="Default contrastive pairs retrieved per evaluation query (default: 1)",
    )
    distill_compile.add_argument(
        "--min-rubric-overlap",
        type=float,
        default=0.25,
        help="Minimum Szymkiewicz-Simpson rubric token overlap [0.0..1.0] required before injecting anchors; abstains cleanly on OOD rubrics (default: 0.25)",
    )
    distill_compile.add_argument(
        "--no-anchors",
        action="store_true",
        help="Omit DistilledAnchorBank (fit student LogitMetaCalibrator only)",
    )
    distill_compile.add_argument(
        "--no-meta",
        action="store_true",
        help="Omit LogitMetaCalibrator (compile 100% model-agnostic DistilledAnchorBank only)",
    )
    distill_compile.add_argument("--json", action="store_true", help="Output bundle metadata as JSON")

    distill_lora = distill_sub.add_parser(
        "train-lora",
        help="Train an architecture-agnostic PEFT LoRA adapter (target_modules='all-linear') for any open-weights CausalLM",
    )
    distill_lora.add_argument(
        "--base-model",
        required=True,
        type=str,
        help="HuggingFace model ID or local checkpoint directory (Qwen, Llama, Gemma, Mistral, Phi, etc.)",
    )
    distill_lora.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset or distillation JSONL file (.json, .jsonl, .md)",
    )
    distill_lora.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=str,
        help="Output directory for adapter/, calibration_profile.json, Modelfile.ollama, and launch scripts",
    )
    distill_lora.add_argument("--rank", type=int, default=8, help="LoRA rank r (default: 8)")
    distill_lora.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    distill_lora.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    distill_lora.add_argument("--batch-size", type=int, default=4, help="Micro-batch size (default: 4)")
    distill_lora.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    distill_lora.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Post-LoRA conformal calibration target risk alpha (default: 0.10)",
    )
    distill_lora.add_argument(
        "--ollama-base",
        type=str,
        default=None,
        help="Optional Ollama base tag written into Modelfile.ollama FROM line",
    )
    distill_lora.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, mps)")
    distill_lora.add_argument("--json", action="store_true", help="Output LoRADistillManifest as JSON")
    return distill_parser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logitgate",
        description="LogitGate CLI: Calibrated first-token logit evaluation and cascaded routing.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _register_eval_parser(subparsers)
    _register_calibrate_parser(subparsers)
    _register_explain_parser(subparsers)
    _register_benchmark_parser(subparsers)
    _register_rank_parser(subparsers)
    _register_telemetry_parser(subparsers)
    _register_stream_parser(subparsers)
    _register_distill_parser(subparsers)

    return parser
