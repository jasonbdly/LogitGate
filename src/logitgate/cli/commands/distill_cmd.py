from __future__ import annotations

import argparse
import json

from logitgate.cli.commands.common import EXIT_FAILED, EXIT_OK
from logitgate.cli.style import sanitize_terminal_text


def _cmd_distill(args: argparse.Namespace) -> int:
    if args.distill_cmd == "compile":
        from logitgate.distill.bundle import compile_distillation_bundle

        bundle = compile_distillation_bundle(
            dataset=args.dataset,
            telemetry=args.telemetry,
            backend=args.backend,
            student_model=args.model,
            teacher_model=args.teacher_model,
            include_anchors=not args.no_anchors,
            fit_meta=not args.no_meta,
            max_bank_pairs=args.max_pairs,
            default_retrieve_pairs=args.retrieve_pairs,
            min_rubric_overlap=args.min_rubric_overlap,
            target_risk_alpha=args.alpha,
        )
        saved = bundle.save(args.output)
        if args.json:
            payload = {
                "output_path": str(saved),
                "has_anchor_bank": bundle.anchor_bank is not None,
                "num_anchor_pairs": len(bundle.anchor_bank.pairs) if bundle.anchor_bank else 0,
                "has_meta_calibrator": bundle.meta_calibrator is not None,
                "meta_oof_auroc": (
                    bundle.meta_calibrator.oof_auroc if bundle.meta_calibrator else None
                ),
                "pass_threshold": (
                    bundle.meta_calibrator.pass_threshold if bundle.meta_calibrator else None
                ),
                "fail_threshold": (
                    bundle.meta_calibrator.fail_threshold if bundle.meta_calibrator else None
                ),
                "has_profile": bundle.profile is not None,
            }
            print(json.dumps(payload, indent=2))
        else:
            print(f"Saved DistillationBundle to {sanitize_terminal_text(str(saved))}")
            if bundle.anchor_bank is not None:
                print(f"  Anchor Pairs:      {len(bundle.anchor_bank.pairs)} (retrieve k={bundle.anchor_bank.max_pairs})")
            if bundle.meta_calibrator is not None:
                mc = bundle.meta_calibrator
                print(
                    f"  Meta-Calibrator:   intercept={mc.intercept:+.4f}, w_sym={mc.weights[0]:.4f}, "
                    f"OOF AUROC={mc.oof_auroc}, thresholds=[{mc.fail_threshold}, {mc.pass_threshold}]"
                )
        return EXIT_OK

    if args.distill_cmd == "train-lora":
        from logitgate.distill.lora import LoRADistillConfig, train_distillation_lora

        cfg = LoRADistillConfig(
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )
        manifest = train_distillation_lora(
            args.base_model,
            args.dataset,
            args.output_dir,
            config=cfg,
            calibrate_alpha=args.alpha,
            ollama_base_model=args.ollama_base,
        )
        if args.json:
            print(manifest.model_dump_json(indent=2))
        else:
            print(f"Trained PEFT LoRA adapter -> {sanitize_terminal_text(manifest.adapter_dir)}")
            print(f"  Loss:              {manifest.initial_loss:.4f} -> {manifest.final_loss:.4f}")
            print(f"  Ollama Modelfile:  {sanitize_terminal_text(manifest.ollama_modelfile_path)}")
            print(f"  vLLM Launch:       {sanitize_terminal_text(manifest.vllm_launch_script_path)}")
            print(f"  llama.cpp Launch:  {sanitize_terminal_text(manifest.llamacpp_launch_script_path)}")
        return EXIT_OK

    return EXIT_FAILED
