from __future__ import annotations

from logitgate.cli.commands.benchmark_cmd import _cmd_benchmark
from logitgate.cli.commands.calibrate_cmd import _cmd_calibrate
from logitgate.cli.commands.common import (
    EXIT_BACKEND_ERROR,
    EXIT_FAILED,
    EXIT_OK,
)
from logitgate.cli.commands.distill_cmd import _cmd_distill
from logitgate.cli.commands.eval_cmd import _cmd_eval
from logitgate.cli.commands.explain_cmd import _cmd_explain
from logitgate.cli.commands.rank_cmd import _cmd_rank
from logitgate.cli.commands.stream_cmd import _cmd_stream
from logitgate.cli.commands.telemetry_cmd import _cmd_telemetry

__all__ = [
    "EXIT_BACKEND_ERROR",
    "EXIT_FAILED",
    "EXIT_OK",
    "_cmd_benchmark",
    "_cmd_calibrate",
    "_cmd_distill",
    "_cmd_eval",
    "_cmd_explain",
    "_cmd_rank",
    "_cmd_stream",
    "_cmd_telemetry",
]
