from __future__ import annotations

import argparse
import json
import sys

from logitgate.cache import VerdictCache
from logitgate.cascade import Cascade
from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text
from logitgate.gates.judge_gate import JudgeGate
from logitgate.gates.logit_gate import LogitGate
from logitgate.streaming import StreamGuard
from logitgate.telemetry.sinks import JsonlTelemetrySink


def _cmd_stream(args: argparse.Namespace) -> int:
    if args.chunk_size < 1:
        sys.stderr.write("Error: --chunk-size must be >= 1\n")
        return EXIT_FAILED

    raw_text: str | None = None
    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as f:
                raw_text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading file '{args.file}': {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED
    elif args.candidate is not None:
        raw_text = args.candidate
    elif not sys.stdin.isatty():
        try:
            raw_text = sys.stdin.read()
        except (OSError, UnicodeDecodeError) as e:
            sys.stderr.write(f"Error reading stdin: {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED
    else:
        sys.stderr.write("Error: Either --candidate, --file, or piped stdin must be provided.\n")
        return EXIT_FAILED

    chunks = [
        raw_text[i : i + args.chunk_size]
        for i in range(0, len(raw_text), args.chunk_size)
    ] if raw_text else []

    sink = JsonlTelemetrySink(args.telemetry_out) if getattr(args, "telemetry_out", None) else None
    cache_obj: VerdictCache | None = None
    judge_gate: JudgeGate | None = None
    try:
        if args.cache is not None:
            cache_obj = VerdictCache() if args.cache == "memory" else VerdictCache(path=args.cache)
        has_judge = bool(args.judge_backend or args.judge_model)
        logit_gate = LogitGate(
            rubric=args.rubric,
            backend=args.backend,
            model=args.model,
            pass_threshold=args.threshold,
            fail_threshold=args.fail_threshold,
            profile=args.profile,
            distillation_bundle=getattr(args, "bundle", None),
            save_on_fail=False,
            telemetry_sink=sink if not has_judge else False,
            cache=cache_obj if not has_judge else False,
        )
        try:
            target_gate: Cascade | LogitGate
            if has_judge:
                judge_gate = JudgeGate(
                    rubric=None,
                    backend=args.judge_backend or args.backend,
                    model=args.judge_model or args.model,
                    trigger="on_ambiguous_only",
                    save_on_fail=False,
                    telemetry_sink=False,
                    cache=False,
                )
                target_gate = Cascade(
                    [logit_gate, judge_gate],
                    telemetry_sink=sink,
                    cache=cache_obj,
                )
            else:
                target_gate = logit_gate

            guard = StreamGuard(
                target_gate,
                min_chars=args.min_chars,
                stride_chars=args.stride_chars,
                clause_boundary=not args.no_clause_boundary,
                consecutive_failures=args.consecutive_failures,
                delivery=args.delivery,
                raise_on_abort=False,
                telemetry_sink=sink,
            )
            stream_handle = guard.stream(chunks)
            yielded_parts = list(stream_handle)
            summary = stream_handle.summary
            if summary is None:
                raise RuntimeError("Stream execution finished without producing a summary.")
        finally:
            if judge_gate is not None:
                judge_gate.backend.close()
            logit_gate.backend.close()
    finally:
        if cache_obj is not None:
            cache_obj.close()
        if sink is not None:
            sink.close()

    yielded_text = "".join(yielded_parts)
    if args.json:
        payload = {
            **summary.to_dict(),
            "yielded_text": yielded_text,
        }
        print(json.dumps(payload, indent=2))
    else:
        if not args.quiet_stream and yielded_text:
            print(sanitize_terminal_text(yielded_text))
            print("-" * 60)
        status_label = "PASSED" if summary.passed else ("ABORTED_MID_STREAM" if summary.aborted_mid_stream else "FAILED")
        print(f"Stream Status: {status_label}")
        print(f"Delivery Mode: {args.delivery}")
        print(f"Chunks: {summary.chunks_yielded}/{summary.total_chunks_seen} yielded ({summary.chars_yielded}/{summary.chars_seen} chars)")
        print(f"Checks Executed: {summary.checks_executed} ({summary.total_gate_latency_ms:.2f}ms total gate latency)")
        if summary.final_verdict and summary.final_verdict.reason:
            print(f"Reason: {sanitize_terminal_text(summary.final_verdict.reason)}")

    return EXIT_OK if summary.passed else EXIT_FAILED
