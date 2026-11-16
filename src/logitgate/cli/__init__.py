from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

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
from logitgate.cli.parser import create_parser
from logitgate.cli.style import (
    _bias_hint,
    _decisive_logit_trace,
    _entropy_hint,
    _fmt,
    _logit,
    _sanitize_terminal_text,
    _Style,
    _use_color,
    render_snapshot_report,
    sanitize_terminal_text,
)
from logitgate.exceptions import (
    BackendError,
    CalibrationError,
    ConfigurationError,
    LogitGateError,
)

_COMMANDS: dict[str, Any] = {
    "eval": _cmd_eval,
    "calibrate": _cmd_calibrate,
    "explain": _cmd_explain,
    "benchmark": _cmd_benchmark,
    "rank": _cmd_rank,
    "telemetry": _cmd_telemetry,
    "stream": _cmd_stream,
    "distill": _cmd_distill,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 = passed/ok, 1 = verdict failed or bad input, 2 = backend error."""
    parser = create_parser()
    args = parser.parse_args(argv)

    command = _COMMANDS.get(args.command)
    if command is None:
        parser.print_help()
        return EXIT_FAILED
    try:
        result: int = command(args)
    except BackendError as exc:
        # Backends already phrase these as "what went wrong + what to do"; a traceback adds nothing.
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\n")
        return EXIT_BACKEND_ERROR
    except (ConfigurationError, CalibrationError, LogitGateError) as exc:
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\n")
        return EXIT_FAILED
    return result


__all__ = [
    "EXIT_BACKEND_ERROR",
    "EXIT_FAILED",
    "EXIT_OK",
    "_COMMANDS",
    "_Style",
    "_bias_hint",
    "_cmd_benchmark",
    "_cmd_calibrate",
    "_cmd_distill",
    "_cmd_eval",
    "_cmd_explain",
    "_cmd_rank",
    "_cmd_stream",
    "_cmd_telemetry",
    "_decisive_logit_trace",
    "_entropy_hint",
    "_fmt",
    "_logit",
    "_sanitize_terminal_text",
    "_use_color",
    "create_parser",
    "main",
    "render_snapshot_report",
    "sanitize_terminal_text",
]


if __name__ == "__main__":
    sys.exit(main())
