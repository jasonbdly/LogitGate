from __future__ import annotations

import argparse
import math
import sys

from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from logitgate.backends.base import infer_backend_spec, resolve_logit_backend
    from logitgate.calibration.profile import (
        CalibrationProfile,
        fit_calibration_profile,
    )
    from logitgate.exceptions import CalibrationError

    if args.dataset and (
        isinstance(args.alpha, bool) or not math.isfinite(args.alpha) or not (0.0 < args.alpha < 1.0)
    ):
        sys.stderr.write(f"Error: --alpha must be strictly between 0.0 and 1.0, got {args.alpha}\n")
        return EXIT_FAILED

    backend = resolve_logit_backend(args.backend, model=args.model)
    effective_model = args.model or getattr(backend, "model", None) or getattr(backend, "model_name", None) or "default"
    try:
        if args.dataset:
            import json

            from logitgate.calibration.conformal import load_calibration_dataset

            try:
                samples = load_calibration_dataset(args.dataset)
            except (OSError, ValueError) as e:
                sys.stderr.write(f"Error loading dataset '{args.dataset}': {e}\n")
                return EXIT_FAILED
            if not samples:
                sys.stderr.write(f"Error: No valid samples found in dataset '{args.dataset}'\n")
                return EXIT_FAILED
            if len(samples) < 10:
                sys.stderr.write(
                    f"Error: Calibration requires at least 10 samples, got {len(samples)} in '{args.dataset}'\n"
                )
                return EXIT_FAILED
            if len({bool(y) for _, _, y in samples}) < 2:
                sys.stderr.write(
                    f"Error: Calibration set '{args.dataset}' must contain both positive and negative ground truth labels.\n"
                )
                return EXIT_FAILED

            try:
                profile = fit_calibration_profile(
                    backend=backend,
                    model=args.model,
                    samples=samples,
                    target_risk_alpha=args.alpha,
                    output_path=args.output,
                    cache_path=args.cache_path,
                )
            except (OSError, ValueError, CalibrationError) as e:
                sys.stderr.write(f"Error: {sanitize_terminal_text(str(e))}\n")
                return EXIT_FAILED

            if (
                profile.fail_threshold is None
                or profile.pass_threshold is None
                or profile.target_risk_alpha is None
                or profile.empirical_escalation_rate is None
            ):
                raise RuntimeError("Calibration profile is missing required conformal cascade thresholds.")

            if args.json:
                payload = {
                    **profile.model_dump(),
                    "optimal_temperature": profile.binary_temperature,
                    "tau_reject": profile.fail_threshold,
                    "tau_accept": profile.pass_threshold,
                }
                print(json.dumps(payload, indent=2))
            else:
                print("=" * 60)
                print("LOGITGATE SPLIT CONFORMAL CASCADE CALIBRATION")
                print("=" * 60)
                print(f"Dataset:                   {sanitize_terminal_text(str(args.dataset))}")
                print(f"Calibration Samples:       {profile.calibration_set_size}")
                print(f"Target Risk (Alpha):       {profile.target_risk_alpha:.4f}")
                print(f"Fitted Temperature:        {profile.binary_temperature:.4f}")
                print("-" * 60)
                print(f"Layer 1 Tau Reject:        {profile.fail_threshold:.4f}")
                print(f"Layer 1 Tau Accept:        {profile.pass_threshold:.4f}")
                print(f"Layer 2 Escalation Rate:   {profile.empirical_escalation_rate:.2%}")
                print("-" * 60)
                if profile.guarantee_holds:
                    print(
                        f"Guaranteed Layer 1 Error: <= {profile.target_risk_alpha:.2%} "
                        "(marginal; assumes production inputs resemble this dataset)"
                    )
                else:
                    print(
                        f"Guaranteed Layer 1 Error: NONE. Need >= {profile.min_samples_per_class} positives "
                        f"and negatives (have {profile.num_positive} / {profile.num_negative}); "
                        "thresholds are best-effort."
                    )
                if args.output:
                    safe_out = sanitize_terminal_text(str(args.output))
                    print(f"Saved Profile:             {safe_out}")
                    print(f'Use: LogitGate(..., profile="{safe_out}")')
                else:
                    print(
                        f"Use: LogitGate(..., pass_threshold={profile.pass_threshold}, "
                        f"fail_threshold={profile.fail_threshold}, "
                        f"logit_temperature={profile.binary_temperature})"
                    )
                print("=" * 60)
            return EXIT_OK

        from logitgate.debiasing.permutation import auto_calibrate_backend

        cache_path = Path(args.cache_path) if args.cache_path else None

        if not args.json:
            print(f"Calibrating model '{sanitize_terminal_text(effective_model)}' on backend {backend.__class__.__name__}...")
        try:
            result = auto_calibrate_backend(
                backend,
                model_key=effective_model,
                save_cache=True,
                cache_path=cache_path,
            )
            if args.output:
                profile = CalibrationProfile.from_calibration_result(
                    result,
                    model=effective_model,
                    backend=infer_backend_spec(backend),
                )
                profile.save(args.output)
        except (OSError, ValueError) as e:
            sys.stderr.write(f"Error saving calibration cache: {sanitize_terminal_text(str(e))}\n")
            return EXIT_FAILED
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(f"Optimal Temperature: {result.optimal_temperature:.4f}")
            if result.multiclass_temperature is not None:
                print(f"Multiclass Temperature: {result.multiclass_temperature:.4f}")
            if result.probe_auroc is not None:
                print(f"Probe AUROC: {result.probe_auroc:.4f}")
            print(f"Brier Score: {result.brier_score_before:.6f} -> {result.brier_score_after:.6f}")
            print(f"ECE: {result.ece_before:.6f} -> {result.ece_after:.6f}")
            print("Calibration successfully saved to cache.")
            if args.output:
                print(f"Calibration profile saved to {sanitize_terminal_text(str(args.output))}.")
        return EXIT_OK
    finally:
        backend.close()
