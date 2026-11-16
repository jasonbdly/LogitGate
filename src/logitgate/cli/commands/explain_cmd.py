from __future__ import annotations

import argparse
import sys

from logitgate.cli.commands.common import EXIT_BACKEND_ERROR, EXIT_FAILED, EXIT_OK
from logitgate.cli.style import (
    _Style,
    _use_color,
    render_snapshot_report,
    sanitize_terminal_text,
)
from logitgate.exceptions import BackendError
from logitgate.snapshot import VerdictSnapshot


def _cmd_explain(args: argparse.Namespace) -> int:
    if args.max_tokens is not None and args.max_tokens < 1:
        sys.stderr.write("Error: --max-tokens must be >= 1\n")
        return EXIT_FAILED

    try:
        snapshot = VerdictSnapshot.load(args.snapshot_path)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
        sys.stderr.write(f"Error loading snapshot from '{args.snapshot_path}': {sanitize_terminal_text(str(e))}\n")
        return EXIT_FAILED

    style = _Style(_use_color(args.color, sys.stdout), sys.stdout)

    if args.offline:
        if args.json:
            print(snapshot.verdict.model_dump_json(indent=2))
        else:
            print(
                render_snapshot_report(
                    snapshot,
                    style,
                    replay_note="Replay skipped (--offline). Re-run without --offline for a Layer-2 critique.",
                )
            )
        return EXIT_OK

    try:
        verdict = snapshot.explain(
            backend=args.backend,
            model=args.model,
            max_tokens=args.max_tokens,
            think=not args.no_think,
        )
    except BackendError as exc:
        if not args.json:
            print(render_snapshot_report(snapshot, style, replay_note="Replay failed; see error below."))
        sys.stderr.write(
            f"error: {sanitize_terminal_text(str(exc))}\nTip: `logitgate explain --offline {args.snapshot_path}` shows diagnostics without a model.\n"
        )
        return EXIT_BACKEND_ERROR
    except TypeError as exc:
        sys.stderr.write(f"error: {sanitize_terminal_text(str(exc))}\n")
        return EXIT_FAILED

    if args.json:
        print(verdict.model_dump_json(indent=2))
    else:
        print(render_snapshot_report(snapshot, style, replay=verdict))
    return EXIT_OK
