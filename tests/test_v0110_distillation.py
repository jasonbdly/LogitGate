from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
from typing import Any

import pytest

from logitgate import (
    AdaptiveConformalController,
    ConfigurationError,
    DecisionStatus,
    DistillationBundle,
    DistillationRecord,
    DistilledAnchorBank,
    HuggingFaceLogitBackend,
    LlamaCppBackend,
    LogitBackend,
    LogitFeatureVector,
    LogitGate,
    LogitMetaCalibrator,
    LoRADistillConfig,
    MultiRubricGate,
    VerdictCache,
    acompile_distillation_bundle,
    compile_binary_prompt,
    emit_deployment_manifests,
    extract_logit_features,
    gate_cache_fingerprint,
    prepare_symmetric_lora_examples,
    train_distillation_lora,
)
from logitgate.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_A_PATH = REPO_ROOT / "benchmarks" / "data" / "cal_distill_80.json"


class _SkewedStudentBackend(LogitBackend):
    """Simulates a systematically skeptical or sycophantic 3B model that responds to anchors."""

    def __init__(self, *, skew_log_odds: float = -1.6) -> None:
        self.skew_log_odds = skew_log_odds
        self.prompts_seen: list[str] = []

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del top_k, allowed_tokens, kwargs
        self.prompts_seen.append(prompt)
        # Extract candidate section
        cand_part = (
            prompt.split("[CANDIDATE ARTIFACT]", 1)[-1]
            .split("[EVALUATION RUBRIC]", 1)[0]
            .split("[CALIBRATION ANCHORS]", 1)[0]
            .split("[DECISION MAPPING]", 1)[0]
        )
        map_part = prompt.split("[DECISION MAPPING]", 1)[-1]
        a_is_pass = "A: The candidate artifact complies" in map_part

        is_true_pass = "VALID_SIGNAL" in cand_part
        true_lo = 2.0 if is_true_pass else -2.0
        # Anchors partially mitigate raw skew in-context
        eff_skew = self.skew_log_odds * (0.35 if "[CALIBRATION ANCHORS]" in prompt else 1.0)
        sym_pass_lo = true_lo + eff_skew
        a_lo = sym_pass_lo if a_is_pass else -sym_pass_lo
        p_a = 1.0 / (1.0 + math.exp(-a_lo))
        p_b = max(1e-9, 1.0 - p_a)
        return [
            {"token": "A", "logprob": math.log(max(p_a * 0.92, 1e-9))},
            {"token": "B", "logprob": math.log(max(p_b * 0.92, 1e-9))},
            {"token": "The", "logprob": math.log(0.08)},
        ]


def test_anchor_bank_compilation_kv_prefix_parity_and_self_exclusion(tmp_path: Path) -> None:
    bank = DistilledAnchorBank.from_dataset(SPLIT_A_PATH, max_bank_pairs=16, default_retrieve_pairs=1)
    assert len(bank.pairs) == 16
    for pair in bank.pairs:
        assert pair.positive.passed is True
        assert pair.negative.passed is False
        # Token-neutral rationale: must not hardcode Option A or Option B
        assert "option a" not in pair.positive.rationale.lower()
        assert "option b" not in pair.negative.rationale.lower()

    # Exact self-exclusion: passing an anchor's own candidate never retrieves that anchor!
    sample_pair = bank.pairs[0]
    retrieved_self = bank.retrieve(
        sample_pair.rubric,
        candidate=sample_pair.positive.candidate,
        k=1,
        exclude_exact=True,
    )
    assert all(p.pair_id != sample_pair.pair_id for p in retrieved_self)

    # 100% KV-cache prefix parity across Pass 1 (A=Pass, B=Fail) and Pass 2 (B=Pass, A=Fail)
    nonce = "deadbeef1234"
    anchor_str = bank.format_anchor_block(
        "Does the SQL query parameterize user inputs safely?",
        "SELECT * FROM accounts WHERE id = :id",
        boundary_nonce=nonce,
    )
    p_fwd = compile_binary_prompt(
        rubric="Does the SQL query parameterize user inputs safely?",
        candidate="SELECT * FROM accounts WHERE id = :id",
        pass_token="A",
        fail_token="B",
        query_first=True,
        boundary_nonce=nonce,
        anchor_block=anchor_str,
    )
    p_rev = compile_binary_prompt(
        rubric="Does the SQL query parameterize user inputs safely?",
        candidate="SELECT * FROM accounts WHERE id = :id",
        pass_token="B",
        fail_token="A",
        query_first=True,
        boundary_nonce=nonce,
        anchor_block=anchor_str,
    )
    fwd_prefix = p_fwd.split("[DECISION MAPPING]", 1)[0]
    rev_prefix = p_rev.split("[DECISION MAPPING]", 1)[0]
    assert "[CALIBRATION ANCHORS]" in fwd_prefix
    assert fwd_prefix == rev_prefix

    # Save/Load + symlink rejection
    saved_path = bank.save(tmp_path / "anchors.json")
    reloaded = DistilledAnchorBank.load(saved_path)
    assert reloaded.cache_fingerprint() == bank.cache_fingerprint()

    sym = tmp_path / "symlink_anchors.json"
    sym.symlink_to(saved_path)
    with pytest.raises(ConfigurationError, match="symlink"):
        DistilledAnchorBank.load(sym)
    with pytest.raises(ConfigurationError, match="symlink"):
        bank.save(sym)


def test_logit_meta_calibrator_polarity_skew_and_monotonicity(tmp_path: Path) -> None:
    # Build 40 synthetic samples where student has a severe -1.8 skeptical offset
    feats: list[LogitFeatureVector] = []
    labels: list[bool] = []
    for i in range(40):
        y = (i % 2) == 0
        raw_sym = (1.8 if y else -1.8) - 1.8 + (0.15 if (i % 4 == 0) else -0.10)
        feats.append(
            LogitFeatureVector(
                symmetric_log_odds=raw_sym,
                signed_slot_bias=0.2,
                positional_bias_magnitude=0.2,
                top_k_entropy=0.25,
                unobserved_residual_mass=0.05,
            )
        )
        labels.append(y)

    mc = LogitMetaCalibrator.fit(feats, labels, l2_penalty=0.5, target_risk_alpha=0.10, model="skewed-3b")
    assert mc.weights[0] >= 0.05
    assert mc.intercept > 0.5  # Positive intercept counteracts negative skepticism!
    assert mc.oof_brier_before is not None and mc.oof_brier_after is not None
    assert mc.oof_brier_after < mc.oof_brier_before
    assert mc.pass_threshold is not None and mc.fail_threshold is not None
    assert 0.0 <= mc.fail_threshold < mc.pass_threshold <= 1.0

    # Strict monotonicity in symmetric_log_odds
    p_low = mc.predict_proba(LogitFeatureVector(-2.0, 0.0, 0.0, 0.1, 0.01))
    p_mid = mc.predict_proba(LogitFeatureVector(0.0, 0.0, 0.0, 0.1, 0.01))
    p_high = mc.predict_proba(LogitFeatureVector(2.0, 0.0, 0.0, 0.1, 0.01))
    assert p_low < p_mid < p_high

    p = mc.save(tmp_path / "meta.json")
    mc2 = LogitMetaCalibrator.load(p)
    assert mc2.cache_fingerprint() == mc.cache_fingerprint()


@pytest.mark.asyncio
async def test_distillation_bundle_end_to_end_gate_cache_and_concurrency(tmp_path: Path) -> None:
    dataset = [
        {
            "id": f"item_{i}",
            "rubric": "Does the artifact carry VALID_SIGNAL without defects?",
            "candidate": (
                f"Artifact {i}: VALID_SIGNAL present with clean bounds."
                if i % 2 == 0
                else f"Artifact {i}: MISSING_SIGNAL broken bounds."
            ),
            "expected": (i % 2 == 0),
            "domain": "synthetic",
        }
        for i in range(40)
    ]
    backend = _SkewedStudentBackend(skew_log_odds=-2.1)

    # Without distillation bundle, strong -2.1 skepticism causes valid candidates to miss PASS!
    raw_gate = LogitGate(
        rubric="Does the artifact carry VALID_SIGNAL without defects?",
        backend=backend,
        pass_threshold=0.80,
        fail_threshold=0.20,
        save_on_fail=False,
    )
    v_raw = raw_gate.evaluate("Unseen test case: VALID_SIGNAL verified.")
    assert v_raw.status != DecisionStatus.PASSED

    bundle = await acompile_distillation_bundle(
        dataset=dataset,
        backend=backend,
        student_model="skewed-student",
        teacher_model="teacher-26b",
        target_risk_alpha=0.10,
    )
    assert bundle.meta_calibrator is not None and bundle.meta_calibrator.guarantee_holds is True
    bundle_path = bundle.save(tmp_path / "bundle.json")

    cache = VerdictCache()
    aci = AdaptiveConformalController(distillation_bundle=bundle_path, target_risk_alpha=0.10)
    distilled_gate = LogitGate(
        rubric="Does the artifact carry VALID_SIGNAL without defects?",
        backend=backend,
        distillation_bundle=bundle_path,
        adaptive_controller=aci,
        cache=cache,
        save_on_fail=False,
    )
    v_distilled = distilled_gate.evaluate("Unseen test case: VALID_SIGNAL verified.")
    assert v_distilled.status == DecisionStatus.PASSED
    assert v_distilled.traces[0].metadata.get("meta_calibrated") is True
    assert v_distilled.traces[0].metadata.get("anchored") is True
    assert bundle.unanchored_meta_calibrator is not None

    # Disjoint OOD rubric (overlap < min_rubric_overlap=0.20) abstains from off-domain anchors
    # and routes cleanly to `unanchored_meta_calibrator` (`anchored=False`, `meta_calibrated=True`)!
    ood_gate = LogitGate(
        rubric="Is renal dosing contraindicated under acute glomerular filtration decline?",
        backend=backend,
        distillation_bundle=bundle_path,
        save_on_fail=False,
    )
    v_ood = ood_gate.evaluate("Unseen clinical candidate: VALID_SIGNAL verified.")
    assert v_ood.status == DecisionStatus.PASSED
    assert v_ood.traces[0].metadata.get("meta_calibrated") is True
    assert v_ood.traces[0].metadata.get("anchored") is False

    # Cache fingerprint isolation between raw_gate and distilled_gate + LlamaCppBackend LoRA scale
    assert gate_cache_fingerprint(raw_gate) != gate_cache_fingerprint(distilled_gate)
    lcpp_a = LlamaCppBackend(model="m.gguf", lora_path="/tmp/a.gguf", lora_scale=0.5)
    lcpp_b = LlamaCppBackend(model="m.gguf", lora_path="/tmp/a.gguf", lora_scale=1.0)
    try:
        g_a = LogitGate("Rubric?", backend=lcpp_a, save_on_fail=False)
        g_b = LogitGate("Rubric?", backend=lcpp_b, save_on_fail=False)
        assert gate_cache_fingerprint(g_a) != gate_cache_fingerprint(g_b)
    finally:
        lcpp_a.close()
        lcpp_b.close()

    # MultiRubricGate forwarding + 8-thread concurrency
    mrg = MultiRubricGate(
        [
            "Does the artifact carry VALID_SIGNAL?",
            "Are bounds clean?",
        ],
        backend=backend,
        distillation_bundle=bundle,
        save_on_fail=False,
    )
    assert mrg.anchor_bank is not None
    assert mrg.meta_calibrator is not None

    def _worker(idx: int) -> bool:
        cand = f"Thread {idx}: VALID_SIGNAL clean." if idx % 2 == 0 else f"Thread {idx}: broken."
        res = mrg.evaluate(cand)
        return res.passed == (idx % 2 == 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_worker, range(24)))
    assert all(outcomes)


def test_zero_call_telemetry_bundle_and_cli_compile(tmp_path: Path) -> None:
    recs: list[DistillationRecord] = []
    for i in range(20):
        passed = (i % 2) == 0
        lf = (1.5 if passed else -1.5) + 0.4
        lr = (1.5 if passed else -1.5) - 0.4
        recs.append(
            DistillationRecord(
                event_id=f"ev_{i}",
                timestamp="2026-11-15T00:00:00+00:00",
                rubric="Is the SQL statement free of string concatenation?",
                candidate=(
                    f"SELECT name FROM users WHERE id = $1 -- case {i}"
                    if passed
                    else f"SELECT name FROM users WHERE id = '{i}'"
                ),
                label=passed,
                discrepancy_type="polarity_flip" if i < 6 else "ambiguous_resolved",
                student_layer="LogitGate",
                student_model="student-3b",
                student_status=DecisionStatus.AMBIGUOUS,
                student_prob=0.5,
                student_forward_log_odds=lf,
                student_reverse_log_odds=lr,
                student_log_odds_pass=0.5 * (lf + lr),
                student_positional_bias=0.4,
                teacher_layer="JudgeGate",
                teacher_model="teacher-26b",
                teacher_passed=passed,
                teacher_prob=0.98 if passed else 0.02,
                teacher_log_odds_pass=3.8 if passed else -3.8,
                teacher_reason="- Parameterized placeholder verified.\n[VERDICT: PASS]"
                if passed
                else "- Unescaped string interpolation.\n[VERDICT: FAIL]",
                cascade_passed=passed,
                candidate_sha256=f"sha_{i}",
            )
        )
    jsonl_path = tmp_path / "distill.jsonl"
    jsonl_path.write_text(
        "\n".join(r.model_dump_json() for r in recs) + "\n",
        encoding="utf-8",
    )
    out_bundle = tmp_path / "compiled_bundle.json"
    rc = cli_main([
        "distill",
        "compile",
        "--telemetry",
        str(jsonl_path),
        "--output",
        str(out_bundle),
        "--json",
    ])
    assert rc == 0
    loaded = DistillationBundle.load(out_bundle)
    assert loaded.anchor_bank is not None
    assert loaded.meta_calibrator is not None
    assert loaded.profile is not None
    bundle_gate = LogitGate(
        "Is the SQL statement free of string concatenation?",
        backend="stub://pass",
        distillation_bundle=out_bundle,
        save_on_fail=False,
    )
    bundle_aci = AdaptiveConformalController(distillation_bundle=out_bundle)
    if loaded.meta_calibrator.pass_threshold is not None:
        assert bundle_gate.pass_threshold == loaded.meta_calibrator.pass_threshold
        assert bundle_aci.base_pass_threshold == loaded.meta_calibrator.pass_threshold

    # CLI eval with --bundle against stub backend
    rc_eval = cli_main([
        "eval",
        "--rubric",
        "Is the SQL statement free of string concatenation?",
        "--candidate",
        "SELECT 1",
        "--bundle",
        str(out_bundle),
        "--backend",
        "stub://pass",
        "--json",
    ])
    assert rc_eval == 0


def test_lora_examples_and_deployment_manifests_pure_python(tmp_path: Path) -> None:
    mini_cases = [
        {"id": "c1", "rubric": "Is x > 0?", "candidate": "x = 5", "expected": True},
        {"id": "c2", "rubric": "Is x > 0?", "candidate": "x = -3", "expected": False},
    ]
    examples = prepare_symmetric_lora_examples(mini_cases)
    assert len(examples) == 4
    assert examples[0].polarity == "forward" and examples[0].target_token == "A"
    assert examples[1].polarity == "reverse" and examples[1].target_token == "B"
    assert examples[2].polarity == "forward" and examples[2].target_token == "B"
    assert examples[3].polarity == "reverse" and examples[3].target_token == "A"

    info = emit_deployment_manifests(
        tmp_path / "lora_out",
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        ollama_base_model="qwen2.5-coder:3b",
    )
    modelfile_txt = Path(info["ollama_modelfile_path"]).read_text(encoding="utf-8")
    assert "FROM qwen2.5-coder:3b" in modelfile_txt
    assert "ADAPTER ./adapter" in modelfile_txt
    vllm_txt = Path(info["vllm_launch_script_path"]).read_text(encoding="utf-8")
    assert "--enable-lora" in vllm_txt

    feat = extract_logit_features(
        [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.5}],
        [{"token": "B", "logprob": -0.2}, {"token": "A", "logprob": -2.1}],
    )
    assert feat.symmetric_log_odds > 1.5
    assert 0.0 <= feat.unobserved_residual_mass <= 1.0


def test_architecture_agnostic_lora_training_across_model_families(tmp_path: Path) -> None:
    """Verifies `target_modules='all-linear'` LoRA training across both Llama and Qwen2 architectures."""
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers
    except ImportError:
        with pytest.raises(ConfigurationError, match="logitgate\\[distill\\]"):
            train_distillation_lora(
                "dummy/model",
                [{"id": "1", "rubric": "R?", "candidate": "C", "expected": True}],
                tmp_path / "should_fail",
            )
        return

    class _ASCIIWordTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            stripped = text.strip()
            if stripped == "A":
                return [10]
            if stripped == "B":
                return [11]
            return [((ord(ch) % 90) + 20) for ch in text[-96:]] or [2]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            if ids == [10]:
                return "A"
            if ids == [11]:
                return "B"
            return f"tok_{ids[0]}"

    dataset = [
        {
            "id": f"d_{i}",
            "rubric": "Does code return 42?",
            "candidate": "return 42" if i % 2 == 0 else "return 0",
            "expected": (i % 2 == 0),
        }
        for i in range(12)
    ]
    tok = _ASCIIWordTokenizer()

    # Family 1: Llama architecture (gate_proj/up_proj/down_proj + q/k/v/o_proj)
    llama_cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    llama_model = transformers.LlamaForCausalLM(llama_cfg)
    manifest_llama = train_distillation_lora(
        llama_model,
        dataset,
        tmp_path / "llama_lora",
        tokenizer=tok,
        config=LoRADistillConfig(rank=4, lora_alpha=8, epochs=4, batch_size=4, grad_accum_steps=1, learning_rate=5e-3, device="cpu"),
        calibrate_alpha=0.10,
    )
    assert Path(manifest_llama.adapter_dir, "adapter_config.json").is_file()
    assert manifest_llama.bundle_path is not None and Path(manifest_llama.bundle_path).is_file()
    assert manifest_llama.final_loss < manifest_llama.initial_loss

    # Family 2: Qwen2 architecture — exact same pipeline & config, zero code changes!
    qwen_cfg = transformers.Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    qwen_model = transformers.Qwen2ForCausalLM(qwen_cfg)
    manifest_qwen = train_distillation_lora(
        qwen_model,
        dataset,
        tmp_path / "qwen_lora",
        tokenizer=tok,
        config=LoRADistillConfig(rank=4, lora_alpha=8, epochs=4, batch_size=4, grad_accum_steps=1, learning_rate=5e-3, device="cpu"),
        calibrate_alpha=0.10,
    )
    assert Path(manifest_qwen.adapter_dir, "adapter_config.json").is_file()
    adapter_cfg = json.loads(Path(manifest_qwen.adapter_dir, "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_cfg["r"] == 4

    # Plug HuggingFaceLogitBackend directly into LogitGate!
    hf_backend = HuggingFaceLogitBackend(qwen_model, tok, model_name="tiny-qwen2", device="cpu")
    gate = LogitGate("Does code return 42?", backend=hf_backend, distillation_bundle=manifest_qwen.bundle_path, save_on_fail=False)
    verdict = gate.evaluate("return 42")
    assert verdict.probability is not None and 0.0 <= verdict.probability <= 1.0
