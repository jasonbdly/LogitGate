from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    infer_backend_spec,
    resolve_generative_backend,
    resolve_logit_backend,
)
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.backends.recurrent import RecurrentPyTorchBackend
from logitgate.cascade import Cascade
from logitgate.config import (
    LogitGateDefaults,
    configure,
    get_default_backend,
    get_default_save_failures,
    get_default_snapshot_dir,
    reset_config,
)
from logitgate.debiasing.inversion import (
    compile_batch_binary_prompt,
    compile_binary_prompt,
    compile_choice_prompt,
    compile_score_prompt,
)
from logitgate.debiasing.permutation import (
    KNOWN_MODEL_TEMPERATURES,
    CalibrationResult,
    auto_calibrate_backend,
    combine_symmetric_log_odds,
    fit_calibration,
    get_model_temperature,
)
from logitgate.debiasing.softmax import (
    compute_binary_log_odds,
    compute_normalized_probabilities,
    extract_token_logprob,
)
from logitgate.exceptions import (
    BackendError,
    CalibrationError,
    ConfigurationError,
    ContractViolationError,
    LogitGateError,
)
from logitgate.gates.ast_gate import ASTGate, ContractGate
from logitgate.gates.base import BaseGate
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import ChoiceGate, LogitGate, ScoreGate
from logitgate.integrations.deepeval import LogitGateMetric, LogitGateSystemOneModel
from logitgate.integrations.langgraph import as_conditional_edge
from logitgate.integrations.pytest_plugin import check
from logitgate.models import (
    DecisionStatus,
    LayerTrace,
    Verdict,
)
from logitgate.snapshot import VerdictSnapshot

__version__ = "0.2.1"

__all__ = [
    "KNOWN_MODEL_TEMPERATURES",
    "ASTGate",
    "BackendError",
    "BaseGate",
    "CalibrationError",
    "CalibrationResult",
    "Cascade",
    "ChoiceGate",
    "ConfigurationError",
    "ContractGate",
    "ContractViolationError",
    "DecisionStatus",
    "GenerativeBackend",
    "JudgeGate",
    "LayerTrace",
    "LlamaCppBackend",
    "LogitBackend",
    "LogitGate",
    "LogitGateDefaults",
    "LogitGateError",
    "LogitGateMetric",
    "LogitGateSystemOneModel",
    "OllamaBackend",
    "OpenAICompatBackend",
    "RecurrentPyTorchBackend",
    "ScoreGate",
    "StubBackend",
    "Verdict",
    "VerdictSnapshot",
    "as_conditional_edge",
    "auto_calibrate_backend",
    "check",
    "combine_symmetric_log_odds",
    "compile_batch_binary_prompt",
    "compile_binary_prompt",
    "compile_choice_prompt",
    "compile_score_prompt",
    "compute_binary_log_odds",
    "compute_normalized_probabilities",
    "configure",
    "extract_token_logprob",
    "fit_calibration",
    "get_default_backend",
    "get_default_save_failures",
    "get_default_snapshot_dir",
    "get_model_temperature",
    "infer_backend_spec",
    "reset_config",
    "resolve_generative_backend",
    "resolve_logit_backend",
]
