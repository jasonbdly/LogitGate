from logitgate.config import (
    LogitGateDefaults,
    configure,
    reset_config,
    get_default_backend,
    get_default_save_failures,
    get_default_snapshot_dir,
)
from logitgate.models import (
    Verdict,
    LayerTrace,
    DecisionStatus,
)
from logitgate.snapshot import VerdictSnapshot
from logitgate.debiasing.permutation import (
    fit_calibration,
    CalibrationResult,
    combine_symmetric_log_odds,
    auto_calibrate_backend,
    get_model_temperature,
    KNOWN_MODEL_TEMPERATURES,
)
from logitgate.debiasing.softmax import (
    extract_token_logprob,
    compute_binary_log_odds,
)
from logitgate.debiasing.inversion import (
    compile_binary_prompt,
    compile_batch_binary_prompt,
    compile_choice_prompt,
    compile_score_prompt,
)
from logitgate.backends.base import (
    LogitBackend,
    GenerativeBackend,
    StubBackend,
    resolve_logit_backend,
    resolve_generative_backend,
)
from logitgate.backends.ollama import OllamaBackend
from logitgate.backends.openai_compat import OpenAICompatBackend
from logitgate.backends.llamacpp import LlamaCppBackend
from logitgate.gates.base import BaseGate
from logitgate.gates.ast_gate import ASTGate, ContractGate
from logitgate.gates.logit_gate import LogitGate, ChoiceGate, ScoreGate
from logitgate.gates.judge_gate import JudgeGate
from logitgate.cascade import Cascade
from logitgate.integrations.pytest_plugin import check

__version__ = "0.1.0"

__all__ = [
    "ASTGate",
    "ContractGate",
    "LogitGate",
    "ChoiceGate",
    "ScoreGate",
    "JudgeGate",
    "Cascade",
    "Verdict",
    "VerdictSnapshot",
    "LayerTrace",
    "DecisionStatus",
    "StubBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "LlamaCppBackend",
    "fit_calibration",
    "CalibrationResult",
    "auto_calibrate_backend",
    "get_model_temperature",
    "KNOWN_MODEL_TEMPERATURES",
    "check",
    "combine_symmetric_log_odds",
    "extract_token_logprob",
    "compute_binary_log_odds",
    "compile_binary_prompt",
    "compile_batch_binary_prompt",
    "compile_choice_prompt",
    "compile_score_prompt",
    "LogitBackend",
    "GenerativeBackend",
    "resolve_logit_backend",
    "resolve_generative_backend",
    "BaseGate",
    "LogitGateDefaults",
    "configure",
    "reset_config",
    "get_default_backend",
    "get_default_save_failures",
    "get_default_snapshot_dir",
]
