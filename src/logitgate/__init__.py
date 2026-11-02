from logitgate.backends.base import (
    GenerativeBackend,
    LogitBackend,
    StubBackend,
    resolve_generative_backend,
    resolve_logit_backend,
)
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
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
    extract_token_logprob,
)
from logitgate.gates.ast_gate import ASTGate, ContractGate
from logitgate.gates.base import BaseGate
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import ChoiceGate, LogitGate, ScoreGate
from logitgate.integrations.pytest_plugin import check
from logitgate.models import (
    DecisionStatus,
    LayerTrace,
    Verdict,
)
from logitgate.snapshot import VerdictSnapshot

__version__ = "0.1.0"

__all__ = [
    "KNOWN_MODEL_TEMPERATURES",
    "ASTGate",
    "BaseGate",
    "CalibrationResult",
    "Cascade",
    "ChoiceGate",
    "ContractGate",
    "DecisionStatus",
    "GenerativeBackend",
    "JudgeGate",
    "LayerTrace",
    "LlamaCppBackend",
    "LogitBackend",
    "LogitGate",
    "LogitGateDefaults",
    "OllamaBackend",
    "OpenAICompatBackend",
    "ScoreGate",
    "StubBackend",
    "Verdict",
    "VerdictSnapshot",
    "auto_calibrate_backend",
    "check",
    "combine_symmetric_log_odds",
    "compile_batch_binary_prompt",
    "compile_binary_prompt",
    "compile_choice_prompt",
    "compile_score_prompt",
    "compute_binary_log_odds",
    "configure",
    "extract_token_logprob",
    "fit_calibration",
    "get_default_backend",
    "get_default_save_failures",
    "get_default_snapshot_dir",
    "get_model_temperature",
    "reset_config",
    "resolve_generative_backend",
    "resolve_logit_backend",
]
