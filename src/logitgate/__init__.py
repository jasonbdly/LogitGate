from typing import TYPE_CHECKING, Any

from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    infer_backend_spec,
    resolve_generative_backend,
    resolve_logit_backend,
)
from logitgate.backends.dory_recurrent import (
    RecurrentLoopTrace,
    execute_adaptive_recurrent_loops,
)
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.backends.pytorch_native import PrefixBroadcastLogitEvaluator
from logitgate.backends.recurrent import (
    RecurrentPyTorchBackend,
)
from logitgate.cache import (
    CacheStats,
    VerdictCache,
    compute_verdict_cache_key,
    gate_cache_fingerprint,
    resolve_verdict_cache,
)
from logitgate.calibration.adaptive import (
    AdaptiveConformalController,
    AdaptiveConformalSummary,
    AdaptiveFeedbackMode,
    AdaptiveStepRecord,
)
from logitgate.calibration.audit import (
    DatasetAuditReport,
    SplitLeakageReport,
    audit_benchmark_dataset,
    check_split_leakage,
)
from logitgate.calibration.conformal import (
    ConformalCascadeThresholds,
    calibrate_cascade_thresholds,
    load_calibration_dataset,
    min_samples_per_class,
)
from logitgate.calibration.profile import (
    CalibrationProfile,
    ChoiceConformalThresholds,
    ScoreConformalThresholds,
    afit_calibration_profile,
    calibrate_choice_thresholds,
    calibrate_score_thresholds,
    fit_calibration_profile,
    resolve_calibration_profile,
)
from logitgate.cascade import Cascade
from logitgate.config import (
    LogitGateDefaults,
    configure,
    get_default_auto_calibrate,
    get_default_backend,
    get_default_model,
    get_default_save_failures,
    get_default_snapshot_dir,
    get_default_telemetry_sink,
    get_default_verdict_cache,
    reset_config,
)
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_binary_prompt,
    compile_choice_prompt,
    compile_cot_binary_prompt,
    compile_listwise_rank_prompt,
    compile_pairwise_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import (
    KNOWN_BINARY_MODEL_TEMPERATURES,
    KNOWN_MODEL_TEMPERATURES,
    CalibrationResult,
    DebiasedLogitResult,
    SurfaceTokenMap,
    aauto_calibrate_backend,
    auto_calibrate_backend,
    combine_symmetric_log_odds,
    compute_shannon_entropy,
    compute_symmetric_debiased_verdict,
    fit_calibration,
    get_model_temperature,
    has_known_or_cached_temperature,
    marginalize_pole_logprob,
)
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    compute_normalized_probabilities,
    extract_token_logprob,
)
from logitgate.exceptions import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
    CacheWarning,
    CalibrationError,
    CalibrationWarning,
    ConfigurationError,
    ContractViolationError,
    LogitGateError,
    StreamAbortedError,
    StubBackendWarning,
    TelemetryWarning,
)
from logitgate.gates.ast_gate import (
    ASTFailureCode,
    ASTGate,
    ContractGate,
    normalize_candidate,
)
from logitgate.gates.base import BaseGate, EscalationPolicy
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import ChoiceGate, LogitGate, MultiRubricGate, ScoreGate
from logitgate.gates.rank_gate import BestOfNGate, BestOfNResult, RankGate
from logitgate.hardware.memory_guard import (
    PCIeSpilloverGuard,
    VRAMWatermarkStatus,
)
from logitgate.models import (
    DecisionStatus,
    LayerTrace,
    Verdict,
)
from logitgate.snapshot import VerdictSnapshot
from logitgate.streaming import (
    AsyncGuardedStream,
    GuardedStream,
    StreamCheckRecord,
    StreamGuard,
    StreamGuardSummary,
    aguard_stream,
    guard_stream,
)
from logitgate.telemetry import (
    CallbackTelemetrySink,
    CompositeTelemetrySink,
    DiscrepancyType,
    DistillationMode,
    DistillationRecord,
    DistillationSink,
    DriftRecommendedAction,
    DriftReport,
    DriftSeverity,
    ErrorPolicy,
    ExportFormat,
    JsonlTelemetrySink,
    MemoryTelemetrySink,
    PrivacyMode,
    RollingDriftMonitor,
    TelemetryRecord,
    TelemetrySink,
    TelemetrySummary,
    compute_ks_two_sample,
    compute_psi,
    detect_telemetry_drift,
    export_distillation_dataset,
    extract_distillation_record,
    load_distillation_records,
    load_telemetry_records,
    recalibrate_from_telemetry,
    redact_text,
    resolve_telemetry_sink,
    summarize_telemetry,
)

if TYPE_CHECKING:
    from logitgate.integrations.deepeval import LogitGateMetric, LogitGateSystemOneModel
    from logitgate.integrations.langgraph import (
        as_async_conditional_edge,
        as_conditional_edge,
    )
    from logitgate.integrations.pytest_plugin import check

# Integrations resolve on first access; see logitgate/integrations/__init__.py for why.
_LAZY_INTEGRATIONS = {
    "LogitGateMetric",
    "LogitGateSystemOneModel",
    "as_async_conditional_edge",
    "as_conditional_edge",
    "check",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_INTEGRATIONS:
        from logitgate import integrations

        value = getattr(integrations, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_INTEGRATIONS})

__version__ = "0.10.0"

__all__ = [
    "KNOWN_BINARY_MODEL_TEMPERATURES",
    "KNOWN_MODEL_TEMPERATURES",
    "ASTFailureCode",
    "ASTGate",
    "AdaptiveConformalController",
    "AdaptiveConformalSummary",
    "AdaptiveFeedbackMode",
    "AdaptiveStepRecord",
    "AsyncGuardedStream",
    "BackendError",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "BaseGate",
    "BestOfNGate",
    "BestOfNResult",
    "CacheStats",
    "CacheWarning",
    "CalibrationError",
    "CalibrationProfile",
    "CalibrationResult",
    "CalibrationWarning",
    "CallbackTelemetrySink",
    "Cascade",
    "ChoiceConformalThresholds",
    "ChoiceGate",
    "CompositeTelemetrySink",
    "ConfigurationError",
    "ConformalCascadeThresholds",
    "ContractGate",
    "ContractViolationError",
    "DatasetAuditReport",
    "DebiasedLogitResult",
    "DecisionStatus",
    "DiscrepancyType",
    "DistillationMode",
    "DistillationRecord",
    "DistillationSink",
    "DriftRecommendedAction",
    "DriftReport",
    "DriftSeverity",
    "ErrorPolicy",
    "EscalationPolicy",
    "ExportFormat",
    "GenerativeBackend",
    "GuardedStream",
    "JsonlTelemetrySink",
    "JudgeGate",
    "LayerTrace",
    "LlamaCppBackend",
    "LogitBackend",
    "LogitGate",
    "LogitGateDefaults",
    "LogitGateError",
    "LogitGateMetric",
    "LogitGateSystemOneModel",
    "MemoryTelemetrySink",
    "MultiRubricGate",
    "OllamaBackend",
    "OpenAICompatBackend",
    "PCIeSpilloverGuard",
    "PrefixBroadcastLogitEvaluator",
    "PrivacyMode",
    "RankGate",
    "RecurrentLoopTrace",
    "RecurrentPyTorchBackend",
    "RollingDriftMonitor",
    "ScoreConformalThresholds",
    "ScoreGate",
    "SplitLeakageReport",
    "StreamAbortedError",
    "StreamCheckRecord",
    "StreamGuard",
    "StreamGuardSummary",
    "StubBackend",
    "StubBackendWarning",
    "SurfaceTokenMap",
    "TelemetryRecord",
    "TelemetrySink",
    "TelemetrySummary",
    "TelemetryWarning",
    "VRAMWatermarkStatus",
    "Verdict",
    "VerdictCache",
    "VerdictSnapshot",
    "aauto_calibrate_backend",
    "afit_calibration_profile",
    "aguard_stream",
    "as_async_conditional_edge",
    "as_conditional_edge",
    "audit_benchmark_dataset",
    "auto_calibrate_backend",
    "calibrate_cascade_thresholds",
    "calibrate_choice_thresholds",
    "calibrate_score_thresholds",
    "check",
    "check_split_leakage",
    "combine_symmetric_log_odds",
    "compile_batch_binary_prompt",
    "compile_binary_prompt",
    "compile_choice_prompt",
    "compile_cot_binary_prompt",
    "compile_listwise_rank_prompt",
    "compile_pairwise_prompt",
    "compile_score_prompt",
    "compute_binary_log_odds",
    "compute_ks_two_sample",
    "compute_normalized_probabilities",
    "compute_psi",
    "compute_shannon_entropy",
    "compute_symmetric_debiased_verdict",
    "compute_verdict_cache_key",
    "configure",
    "detect_telemetry_drift",
    "execute_adaptive_recurrent_loops",
    "export_distillation_dataset",
    "extract_distillation_record",
    "extract_token_logprob",
    "fit_calibration",
    "fit_calibration_profile",
    "gate_cache_fingerprint",
    "get_default_auto_calibrate",
    "get_default_backend",
    "get_default_model",
    "get_default_save_failures",
    "get_default_snapshot_dir",
    "get_default_telemetry_sink",
    "get_default_verdict_cache",
    "get_model_temperature",
    "guard_stream",
    "has_known_or_cached_temperature",
    "infer_backend_spec",
    "load_calibration_dataset",
    "load_distillation_records",
    "load_telemetry_records",
    "marginalize_pole_logprob",
    "min_samples_per_class",
    "normalize_candidate",
    "recalibrate_from_telemetry",
    "redact_text",
    "reset_config",
    "resolve_calibration_profile",
    "resolve_generative_backend",
    "resolve_logit_backend",
    "resolve_telemetry_sink",
    "resolve_verdict_cache",
    "summarize_telemetry",
]
