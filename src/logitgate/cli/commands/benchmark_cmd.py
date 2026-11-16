from __future__ import annotations

import argparse
import sys
from typing import Any

from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from logitgate.benchmark import (
        BUILTIN_CASES,
        format_report,
        load_cases,
        run_benchmark,
        write_json,
    )

    if getattr(args, "audit_only", False):
        import json

        from logitgate.calibration.audit import (
            audit_benchmark_dataset,
            check_split_leakage,
        )
        from logitgate.exceptions import CalibrationError

        target_source: Any = args.dataset if args.dataset is not None else BUILTIN_CASES
        target_label = sanitize_terminal_text(str(args.dataset)) if args.dataset is not None else "builtin"
        try:
            audit_rep = audit_benchmark_dataset(target_source, label=target_label)
            leak_rep = None
            if args.reference_dataset is not None:
                leak_rep = check_split_leakage(
                    args.reference_dataset,
                    target_source,
                    reference_label=sanitize_terminal_text(str(args.reference_dataset)),
                    eval_label=target_label,
                )
        except (OSError, ValueError, CalibrationError) as e:
            sys.stderr.write(f"Error auditing dataset: {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED

        if args.json:
            payload: dict[str, Any] = {"audit": audit_rep.to_dict()}
            if leak_rep is not None:
                payload["leakage"] = leak_rep.to_dict()
            print(json.dumps(payload, indent=2))
        else:
            print(sanitize_terminal_text(audit_rep.format_summary()))
            if leak_rep is not None:
                print()
                print(sanitize_terminal_text(leak_rep.format_summary()))

        if getattr(args, "strict_audit", False) and (
            audit_rep.warnings or (leak_rep is not None and not leak_rep.is_clean)
        ):
            return EXIT_FAILED
        return EXIT_OK

    if args.judge_max_tokens is not None and args.judge_max_tokens < 1:
        sys.stderr.write("Error: --judge-max-tokens must be >= 1\n")
        return EXIT_FAILED
    if args.repeat < 1 or args.warmup < 0:
        sys.stderr.write("Error: --repeat must be >= 1 and --warmup must be >= 0\n")
        return EXIT_FAILED

    try:
        cases, label = load_cases(args.dataset)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"Error loading dataset '{args.dataset}': {sanitize_terminal_text(str(e))}\n")
        return EXIT_FAILED
    if args.limit is not None:
        if args.limit < 1:
            sys.stderr.write("Error: --limit must be >= 1\n")
            return EXIT_FAILED
        cases = cases[: args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    def progress(line: str) -> None:
        sys.stderr.write(line + "\n")

    try:
        cache_arg: Any = (
            True
            if getattr(args, "cache", None) == "memory"
            else (args.cache if getattr(args, "cache", None) is not None else False)
        )
        report = run_benchmark(
            cases,
            backend=args.backend,
            model=args.model,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            modes=modes,
            repeat=args.repeat,
            warmup=args.warmup,
            judge_max_tokens=args.judge_max_tokens,
            judge_think=args.judge_think,
            cascade_trigger=args.cascade_trigger,
            profile=args.profile,
            distillation_bundle=getattr(args, "bundle", None),
            cache=cache_arg,
            dataset_label=label,
            progress=None if (args.quiet or args.json) else progress,
        )
    except (ValueError, TypeError) as e:
        sys.stderr.write(f"Error: {sanitize_terminal_text(str(e))}\n")
        return EXIT_FAILED

    if args.json:
        write_json(report)
    else:
        print(format_report(report))
    return EXIT_OK
