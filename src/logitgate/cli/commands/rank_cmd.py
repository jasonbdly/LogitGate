from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

from logitgate.backends.base import resolve_logit_backend
from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text
from logitgate.gates.ast_gate import ASTGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.gates.rank_gate import BestOfNGate
from logitgate.telemetry.sinks import JsonlTelemetrySink

_MAX_CANDIDATES_JSON_BYTES = 4 * 1024 * 1024


def _cmd_rank(args: argparse.Namespace) -> int:
    cand_list: list[Any] = list(args.candidate or [])
    for file_path in args.file or []:
        try:
            with open(file_path, encoding="utf-8") as f:
                cand_list.append(f.read())
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading file '{file_path}': {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED

    candidates_input: Any = cand_list
    if args.candidates_json is not None:
        json_path = Path(args.candidates_json)
        try:
            st = json_path.stat()
            if not stat.S_ISREG(st.st_mode):
                sys.stderr.write(f"Error: '{args.candidates_json}' is not a regular file.\n")
                return EXIT_FAILED
            if st.st_size > _MAX_CANDIDATES_JSON_BYTES:
                sys.stderr.write(
                    f"Error: Candidates JSON '{args.candidates_json}' exceeds 4 MiB limit.\n"
                )
                return EXIT_FAILED
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as e:
            sys.stderr.write(
                f"Error loading candidates JSON '{args.candidates_json}': {sanitize_terminal_text(str(e))}\n"
            )
            return EXIT_FAILED

        if isinstance(loaded, dict):
            if not cand_list:
                candidates_input = loaded
            else:
                merged: dict[str, Any] = {str(i): c for i, c in enumerate(cand_list)}
                for k, v in loaded.items():
                    key = str(k)
                    while key in merged:
                        key = f"json_{key}"
                    merged[key] = v
                candidates_input = merged
        elif isinstance(loaded, list):
            cand_list.extend(loaded)
            candidates_input = cand_list
        else:
            sys.stderr.write(
                f"Error: --candidates-json must contain a JSON array or object, got {type(loaded).__name__}.\n"
            )
            return EXIT_FAILED

    if len(candidates_input) < 2:
        sys.stderr.write(
            "Error: rank requires at least 2 candidates via --candidate, --file, or --candidates-json.\n"
        )
        return EXIT_FAILED

    sink = JsonlTelemetrySink(args.telemetry_out) if getattr(args, "telemetry_out", None) else None
    resolved_backend = resolve_logit_backend(args.backend, model=args.model)
    try:
        pre_gate = (
            ASTGate(syntax=args.ast_syntax, normalize=args.normalize)
            if (args.ast_syntax or args.normalize)
            else None
        )
        post_gate = (
            LogitGate(
                rubric=args.rubric,
                backend=resolved_backend,
                model=args.model,
                profile=args.profile,
                telemetry_sink=False,
            )
            if args.verify_winner
            else None
        )
        gate = BestOfNGate(
            rubric=args.rubric,
            backend=resolved_backend,
            model=args.model,
            pre_gate=pre_gate,
            post_gate=post_gate,
            mode=args.mode,
            context=args.context,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            max_positional_bias=args.max_positional_bias,
            profile=args.profile,
            telemetry_sink=sink,
        )
        result = gate.select(candidates_input)
    finally:
        resolved_backend.close()
        if sink is not None:
            sink.close()

    if args.json:
        payload = {
            **result.verdict.model_dump(),
            "winner": result.winner,
            "winner_key": result.winner_key,
            "winner_index": result.winner_index,
            "ranking": list(result.ranking),
            "surviving_keys": list(result.surviving_keys),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        verdict = result.verdict
        print(f"Verdict: {verdict.status.value.upper()} (status: {verdict.status.value})")
        print(f"Winner Key: {sanitize_terminal_text(str(result.winner_key))}")
        print(f"Ranking: {', '.join(sanitize_terminal_text(k) for k in result.ranking)}")
        probs_str = ", ".join(
            f"{sanitize_terminal_text(k)}={p:.4f}" for k, p in result.choice_probabilities.items()
        )
        print(f"Probabilities: {probs_str}")
        print(f"Triggered Layer: {sanitize_terminal_text(verdict.triggered_layer or '')}")
        print(f"Total Latency: {verdict.total_latency_ms:.2f}ms")
        if verdict.reason:
            print(f"Reason: {sanitize_terminal_text(verdict.reason)}")

    return EXIT_OK if result.verdict.passed else EXIT_FAILED
