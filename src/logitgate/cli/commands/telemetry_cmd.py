from __future__ import annotations

import argparse
import sys

from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text


def _cmd_telemetry(args: argparse.Namespace) -> int:
    import json

    from logitgate.telemetry import (
        detect_telemetry_drift,
        export_distillation_dataset,
        recalibrate_from_telemetry,
        summarize_telemetry,
    )

    sub = getattr(args, "telemetry_command", None)
    if sub == "summary":
        summary = summarize_telemetry(args.input_path)
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE TELEMETRY SUMMARY")
            print("=" * 60)
            print(f"Source:                    {sanitize_terminal_text(str(args.input_path))}")
            print(f"Total Records:             {summary.total_records}")
            print(
                f"Verdicts:                  {summary.passed_count} passed / "
                f"{summary.failed_count} failed / {summary.ambiguous_count} ambiguous"
            )
            print(
                f"Fast-Path Rate:            {summary.fast_path_rate:.2%} "
                f"({summary.fast_path_count}/{summary.total_records})"
            )
            print(
                f"Escalation Rate:           {summary.escalation_rate:.2%} "
                f"({summary.escalated_count}/{summary.total_records})"
            )
            print(
                f"Teacher Discrepancies:     {summary.discrepancy_count} "
                f"({summary.discrepancy_rate:.2%} of escalations)"
            )
            print(
                f"Latency (p50 / p95 / avg): {summary.median_latency_ms:.2f}ms / "
                f"{summary.p95_latency_ms:.2f}ms / {summary.mean_latency_ms:.2f}ms"
            )
            print(f"Total Tokens Generated:    {summary.total_tokens_generated}")
            if summary.by_gate_type:
                gates_str = ", ".join(
                    f"{sanitize_terminal_text(k)}={v}" for k, v in sorted(summary.by_gate_type.items())
                )
                print(f"By Gate Type:              {gates_str}")
            print("=" * 60)
        return EXIT_OK

    if sub == "export":
        rows = export_distillation_dataset(
            args.input_path,
            args.output,
            format=args.format,
            include_symmetric_swaps=not args.no_symmetric_swaps,
            dedupe=not args.no_dedupe,
            discrepancies_only=args.discrepancies_only,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "input_path": str(args.input_path),
                        "output_path": str(args.output),
                        "format": args.format,
                        "rows_written": rows,
                    },
                    indent=2,
                )
            )
        else:
            print(
                f"Exported {rows} rows ({args.format}) to "
                f"{sanitize_terminal_text(str(args.output))}."
            )
        return EXIT_OK

    if sub == "recalibrate":
        profile = recalibrate_from_telemetry(
            args.input_path,
            target_risk_alpha=args.alpha,
            model=args.model,
            output_path=args.output,
            dedupe=not args.no_dedupe,
        )
        if args.json:
            print(profile.model_dump_json(indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE ZERO-CALL TELEMETRY RECALIBRATION")
            print("=" * 60)
            print(f"Telemetry Source:          {sanitize_terminal_text(str(args.input_path))}")
            print(f"Escalation Samples Used:   {profile.calibration_set_size}")
            print(f"Fitted Temperature:        {profile.binary_temperature:.4f}")
            if profile.fail_threshold is not None and profile.pass_threshold is not None:
                print(f"Layer 1 Tau Reject:        {profile.fail_threshold:.4f}")
                print(f"Layer 1 Tau Accept:        {profile.pass_threshold:.4f}")
            if profile.empirical_escalation_rate is not None:
                print(f"Empirical Escalation Rate: {profile.empirical_escalation_rate:.2%}")
            if args.output:
                print(f"Saved Profile:             {sanitize_terminal_text(str(args.output))}")
            print("=" * 60)
        return EXIT_OK

    if sub == "drift":
        report = detect_telemetry_drift(
            args.reference,
            args.input_path,
            psi_moderate=args.psi_moderate,
            psi_severe=args.psi_severe,
            ks_alpha=args.ks_alpha,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("=" * 60)
            print("LOGITGATE TELEMETRY DRIFT REPORT")
            print("=" * 60)
            print(f"Reference Source:          {sanitize_terminal_text(str(args.reference))} (n={report.reference_count})")
            print(f"Current Source:            {sanitize_terminal_text(str(args.input_path))} (n={report.current_count})")
            print(f"Drift Detected:            {report.drift_detected} (severity={report.severity.upper()})")
            print(f"Recommended Action:        {report.recommended_action}")
            print(f"Log-Odds PSI / KS p-val:   {report.psi_log_odds:.4f} / {report.ks_log_odds_p_value:.6f} (D={report.ks_log_odds_statistic:.4f})")
            print(f"Escalation Rate (Ref/Cur): {report.ref_escalation_rate:.2%} -> {report.cur_escalation_rate:.2%} ({report.escalation_rate_delta:+.2%})")
            print(f"Mean Pos Bias (Ref/Cur):   {report.ref_mean_pos_bias:.4f} -> {report.cur_mean_pos_bias:.4f} ({report.pos_bias_delta:+.4f})")
            if report.teacher_disagreement_rate is not None:
                print(
                    f"Teacher Disagreement Rate: {report.teacher_disagreement_rate:.2%} "
                    f"(n={report.observed_teacher_count})"
                )
            if report.reasons:
                print("Triggers:")
                for reason in report.reasons:
                    print(f"  - {sanitize_terminal_text(reason)}")
            print("=" * 60)
        return EXIT_FAILED if (args.fail_on_drift and report.drift_detected) else EXIT_OK

    sys.stderr.write(
        "Error: Specify a telemetry subcommand: summary, export, recalibrate, or drift.\n"
    )
    return EXIT_FAILED
