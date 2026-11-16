from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

try:
    import torch
    import transformers
except ImportError:
    torch = None  # type: ignore[assignment]
    transformers = None  # type: ignore[assignment]

_TorchModule: Any = torch.nn.Module if torch is not None else object

from logitgate import (
    BackendError,
    CalibrationError,
    CalibrationProfile,
    ConfigurationError,
    DecisionStatus,
    DistillationBundle,
    DistillationRecord,
    DistilledAnchorBank,
    HuggingFaceLogitBackend,
    LogitBackend,
    LogitFeatureVector,
    LogitMetaCalibrator,
    LoRADistillConfig,
    LoRADistillManifest,
    acompile_distillation_bundle,
    afit_meta_calibrator,
    compile_distillation_bundle,
    emit_deployment_manifests,
    extract_logit_features,
    fit_meta_calibrator,
    prepare_symmetric_lora_examples,
    resolve_anchor_bank,
    resolve_distillation_bundle,
    resolve_meta_calibrator,
    train_distillation_lora,
)
from logitgate.distill._io import (
    atomic_write_json_nofollow,
    read_json_object_nofollow,
)
from logitgate.distill.anchor_bank import (
    AnchorExemplar,
    ContrastiveAnchorPair,
    _condense_rationale,
    _jaccard_tokens,
    _load_generic_cases,
    _truncate_anchor_candidate,
)
from logitgate.distill.lora import (
    _encode_prompt_ids,
    _require_hf_deps,
    _resolve_single_token_id,
    _resolve_torch_device,
    _resolve_torch_dtype,
    _write_text_nofollow,
)
from logitgate.distill.meta_calibrator import (
    _auroc_binary,
    _brier_score,
    _fit_ridge_logistic,
    _pass_distribution_stats,
    _solve_linear_system,
)

# ===========================================================================
# Section 1: Tests for logitgate.distill._io
# ===========================================================================


def test_io_atomic_write_json_nofollow_empty_path() -> None:
    with pytest.raises(ConfigurationError, match="save path cannot be empty"):
        atomic_write_json_nofollow("   ", {"key": "val"}, artifact_name="TestArtifact")


def test_io_atomic_write_json_nofollow_symlink_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    sym_dir = tmp_path / "sym_dir"
    sym_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="under symlinked directory"):
        atomic_write_json_nofollow(
            sym_dir / "artifact.json",
            {"key": "val"},
            artifact_name="TestArtifact",
        )


def test_io_atomic_write_json_nofollow_symlink_target(tmp_path: Path) -> None:
    target_real = tmp_path / "target_real.json"
    target_real.write_text("{}", encoding="utf-8")
    target_sym = tmp_path / "target_sym.json"
    target_sym.symlink_to(target_real)

    with pytest.raises(ConfigurationError, match="through symlink"):
        atomic_write_json_nofollow(
            target_sym,
            {"key": "val"},
            artifact_name="TestArtifact",
        )


def test_io_atomic_write_json_nofollow_fchmod_failure_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "chmod_fail.json"

    def _fchmod_raise(fd: int, mode: int) -> None:
        raise OSError("fchmod unsupported on this filesystem")

    monkeypatch.setattr(os, "fchmod", _fchmod_raise)
    out = atomic_write_json_nofollow(
        target,
        {"status": "ok"},
        artifact_name="TestArtifact",
    )
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8")) == {"status": "ok"}


def test_io_atomic_write_json_nofollow_cleanup_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "fail_replace.json"

    def _replace_raise(src: Path, dst: Path) -> None:
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(os, "replace", _replace_raise)
    with pytest.raises(ConfigurationError, match="Failed to save"):
        atomic_write_json_nofollow(target, {"hello": "world"}, artifact_name="Artifact")

    leftover_tmps = list(tmp_path.glob("*.tmp"))
    assert not leftover_tmps


def test_io_atomic_write_json_nofollow_symlink_created_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "mid_write.json"
    real = tmp_path / "other.json"
    real.write_text("{}", encoding="utf-8")

    orig_dumps = json.dumps

    def _dumps_hook(*args: Any, **kwargs: Any) -> str:
        target.symlink_to(real)
        return orig_dumps(*args, **kwargs)

    monkeypatch.setattr(json, "dumps", _dumps_hook)
    with pytest.raises(ConfigurationError, match="Refusing to overwrite symlinked"):
        atomic_write_json_nofollow(target, {"x": 1}, artifact_name="MidWriteArtifact")


def test_io_atomic_write_json_nofollow_fd_owned_cleanup_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "fd_owned_err.json"

    def _fchmod_raise(fd: int, mode: int) -> None:
        raise RuntimeError("boom before fdopen")

    def _close_raise(fd: int) -> None:
        raise OSError("close failed")

    monkeypatch.setattr(os, "fchmod", _fchmod_raise)
    monkeypatch.setattr(os, "close", _close_raise)
    with pytest.raises(RuntimeError, match="boom before fdopen"):
        atomic_write_json_nofollow(target, {"x": 1}, artifact_name="FdOwnedArtifact")


def test_io_atomic_write_json_nofollow_unserializable_payload(tmp_path: Path) -> None:
    target = tmp_path / "unserializable.json"
    with pytest.raises(ConfigurationError, match="Failed to save"):
        atomic_write_json_nofollow(
            target,
            {"bad": object()},
            artifact_name="BadArtifact",
        )


def test_io_read_json_object_nofollow_empty_path() -> None:
    with pytest.raises(ConfigurationError, match="load path cannot be empty"):
        read_json_object_nofollow("", artifact_name="EmptyPathArtifact")


def test_io_read_json_object_nofollow_symlink(tmp_path: Path) -> None:
    target_real = tmp_path / "real.json"
    target_real.write_text('{"a": 1}', encoding="utf-8")
    target_sym = tmp_path / "sym.json"
    target_sym.symlink_to(target_real)

    with pytest.raises(ConfigurationError, match="from symlink"):
        read_json_object_nofollow(target_sym, artifact_name="SymlinkArtifact")


def test_io_read_json_object_nofollow_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "non_existent.json"
    with pytest.raises(ConfigurationError, match="Failed to open"):
        read_json_object_nofollow(target, artifact_name="MissingArtifact")


def test_io_read_json_object_nofollow_directory_not_regular_file(tmp_path: Path) -> None:
    sub_dir = tmp_path / "sub_directory"
    sub_dir.mkdir()
    with pytest.raises(ConfigurationError, match="not a regular file"):
        read_json_object_nofollow(sub_dir, artifact_name="DirArtifact")


def test_io_read_json_object_nofollow_oversize_st_size(tmp_path: Path) -> None:
    target = tmp_path / "big_file.json"
    target.write_text('{"data": "' + "x" * 200 + '"}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="exceeds maximum size"):
        read_json_object_nofollow(target, artifact_name="BigArtifact", max_bytes=50)


def test_io_read_json_object_nofollow_oversize_utf8_encoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "multibyte.json"
    content = '{"emoji": "' + "🚀" * 30 + '"}'
    target.write_text(content, encoding="utf-8")

    orig_fstat = os.fstat

    def _fstat_fake(fd: int) -> os.stat_result:
        res = orig_fstat(fd)
        return os.stat_result((
            res.st_mode,
            res.st_ino,
            res.st_dev,
            res.st_nlink,
            res.st_uid,
            res.st_gid,
            10,
            res.st_atime,
            res.st_mtime,
            res.st_ctime,
        ))

    monkeypatch.setattr(os, "fstat", _fstat_fake)
    with pytest.raises(ConfigurationError, match="exceeds maximum size"):
        read_json_object_nofollow(target, artifact_name="MultibyteArtifact", max_bytes=20)


def test_io_read_json_object_nofollow_invalid_utf8_decode(tmp_path: Path) -> None:
    target = tmp_path / "invalid_utf8.json"
    target.write_bytes(b"\x80\x81\xff\xfe")
    with pytest.raises(ConfigurationError, match="Failed to read"):
        read_json_object_nofollow(target, artifact_name="InvalidUtf8Artifact")


def test_io_read_json_object_nofollow_invalid_json_syntax(tmp_path: Path) -> None:
    target = tmp_path / "broken_syntax.json"
    target.write_text("{this is not valid json:", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        read_json_object_nofollow(target, artifact_name="BrokenSyntaxArtifact")


def test_io_read_json_object_nofollow_array_instead_of_dict(tmp_path: Path) -> None:
    target = tmp_path / "array.json"
    target.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must contain a top-level JSON object, got list"):
        read_json_object_nofollow(target, artifact_name="ArrayArtifact")


def test_io_read_json_object_nofollow_success(tmp_path: Path) -> None:
    target = tmp_path / "valid.json"
    target.write_text('{"success": true, "code": 200}', encoding="utf-8")
    result = read_json_object_nofollow(target, artifact_name="ValidArtifact")
    assert result == {"success": True, "code": 200}


# ===========================================================================
# Section 2: Tests for logitgate.distill.bundle
# ===========================================================================


def _build_minimal_anchor_bank() -> DistilledAnchorBank:
    pair = ContrastiveAnchorPair(
        pair_id="p1",
        rubric="Is query safe?",
        positive=AnchorExemplar(
            exemplar_id="ex1",
            rubric="Is query safe?",
            candidate="SELECT 1",
            passed=True,
            rationale="Harmless query.",
        ),
        negative=AnchorExemplar(
            exemplar_id="ex2",
            rubric="Is query safe?",
            candidate="DROP TABLE users;",
            passed=False,
            rationale="Destructive SQL query.",
        ),
        priority_score=1.5,
    )
    return DistilledAnchorBank(pairs=[pair], min_rubric_overlap=0.25)


def test_bundle_validation_schema_version_rejection() -> None:
    bank = _build_minimal_anchor_bank()
    with pytest.raises((ConfigurationError, ValidationError), match="Unsupported DistillationBundle schema_version=2"):
        DistillationBundle(schema_version=2, anchor_bank=bank)


def test_bundle_validation_empty_components_rejection() -> None:
    with pytest.raises((ConfigurationError, ValidationError), match="must contain at least one of"):
        DistillationBundle(
            anchor_bank=None,
            meta_calibrator=None,
            unanchored_meta_calibrator=None,
            profile=None,
        )


def test_bundle_cache_fingerprint_variations() -> None:
    bank = _build_minimal_anchor_bank()
    bundle_anchors_only = DistillationBundle(anchor_bank=bank)
    fp1 = bundle_anchors_only.cache_fingerprint()
    assert fp1["anchor_bank"] is not None
    assert fp1["meta_calibrator"] is None
    assert fp1["unanchored_meta_calibrator"] is None
    assert fp1["profile"] is None

    mc = LogitMetaCalibrator(
        intercept=0.2,
        weights=[1.1, 0.1, 0.05, 0.01, 0.02],
        pass_threshold=0.85,
        fail_threshold=0.15,
    )
    unanchored_mc = LogitMetaCalibrator(
        intercept=0.1,
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        pass_threshold=0.80,
        fail_threshold=0.20,
    )
    prof = CalibrationProfile(
        model="dummy",
        backend="hf",
        tau_accept=0.85,
        tau_reject=0.15,
        target_risk_alpha=0.10,
    )
    bundle_full = DistillationBundle(
        anchor_bank=bank,
        meta_calibrator=mc,
        unanchored_meta_calibrator=unanchored_mc,
        profile=prof,
    )
    fp2 = bundle_full.cache_fingerprint()
    assert fp2["anchor_bank"] is not None
    assert fp2["meta_calibrator"] is not None
    assert fp2["unanchored_meta_calibrator"] is not None
    assert fp2["profile"] is not None


def test_bundle_save_and_load_roundtrip(tmp_path: Path) -> None:
    bank = _build_minimal_anchor_bank()
    bundle = DistillationBundle(
        teacher_model="teacher-26b",
        student_model="student-3b",
        anchor_bank=bank,
    )
    out_path = bundle.save(tmp_path / "bundle.json")
    assert out_path.is_file()

    loaded = DistillationBundle.load(out_path)
    assert loaded.teacher_model == "teacher-26b"
    assert loaded.student_model == "student-3b"
    assert loaded.anchor_bank is not None
    assert len(loaded.anchor_bank.pairs) == 1


def test_bundle_load_corrupted_json_or_invalid_schema(tmp_path: Path) -> None:
    p_corrupt = tmp_path / "corrupt.json"
    p_corrupt.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        DistillationBundle.load(p_corrupt)

    p_bad_schema = tmp_path / "bad_schema.json"
    p_bad_schema.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported DistillationBundle"):
        DistillationBundle.load(p_bad_schema)


def test_resolve_distillation_bundle_variations(tmp_path: Path) -> None:
    assert resolve_distillation_bundle(None) is None

    bank = _build_minimal_anchor_bank()
    bundle = DistillationBundle(anchor_bank=bank)
    assert resolve_distillation_bundle(bundle) is bundle

    saved = bundle.save(tmp_path / "resolved_bundle.json")
    loaded = resolve_distillation_bundle(saved)
    assert isinstance(loaded, DistillationBundle)
    assert loaded.anchor_bank is not None

    with pytest.raises(ConfigurationError, match="distillation_bundle must be a DistillationBundle"):
        resolve_distillation_bundle(12345)  # type: ignore[arg-type]


def test_compile_distillation_bundle_arg_validation() -> None:
    with pytest.raises(ConfigurationError, match="Provide exactly one of dataset=... or telemetry=..."):
        compile_distillation_bundle()

    with pytest.raises(ConfigurationError, match="Provide exactly one of dataset=... or telemetry=..."):
        compile_distillation_bundle(dataset=[{"id": "1"}], telemetry=[])


def test_compile_distillation_bundle_telemetry_with_backend() -> None:
    recs: list[DistillationRecord] = [
        DistillationRecord(
            event_id=f"ev_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is code safe?",
            candidate=f"print({i})",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_model="student-3b",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_model="teacher-26b",
            teacher_passed=(i % 2 == 0),
            teacher_reason="Safe execution." if i % 2 == 0 else "Unsafe code.",
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"hash_{i}",
        )
        for i in range(12)
    ]
    bundle = compile_distillation_bundle(
        telemetry=recs,
        backend="stub://pass",
        student_model="student-3b",
        teacher_model="teacher-26b",
        include_anchors=True,
        fit_meta=True,
        min_rubric_overlap=0.25,
    )
    assert bundle.anchor_bank is not None
    assert bundle.meta_calibrator is not None
    assert bundle.unanchored_meta_calibrator is not None


def test_compile_distillation_bundle_telemetry_anchors_and_meta_flags() -> None:
    recs: list[DistillationRecord] = [
        DistillationRecord(
            event_id=f"ev_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is code safe?",
            candidate=f"print({i})",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_model="student-3b",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_model="teacher-26b",
            teacher_passed=(i % 2 == 0),
            teacher_reason="Safe execution." if i % 2 == 0 else "Unsafe code.",
            student_forward_log_odds=(1.0 if i % 2 == 0 else -1.0),
            student_reverse_log_odds=(1.0 if i % 2 == 0 else -1.0),
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"hash_{i}",
        )
        for i in range(12)
    ]
    bundle = compile_distillation_bundle(
        telemetry=recs,
        include_anchors=False,
        fit_meta=True,
        student_model="student-3b",
    )
    assert bundle.anchor_bank is None
    assert bundle.meta_calibrator is not None
    assert bundle.profile is not None


def test_compile_distillation_bundle_dataset_with_and_without_backend() -> None:
    dataset = [
        {
            "id": f"item_{i}",
            "rubric": "Is code safe?",
            "candidate": f"print({i})",
            "expected": (i % 2 == 0),
        }
        for i in range(12)
    ]
    bundle_anchor_only = compile_distillation_bundle(
        dataset=dataset,
        backend=None,
        include_anchors=True,
        fit_meta=True,
    )
    assert bundle_anchor_only.anchor_bank is not None
    assert bundle_anchor_only.meta_calibrator is None

    bundle_with_backend = compile_distillation_bundle(
        dataset=dataset,
        backend="stub://pass",
        include_anchors=True,
        fit_meta=True,
        min_rubric_overlap=0.25,
    )
    assert bundle_with_backend.anchor_bank is not None
    assert bundle_with_backend.meta_calibrator is not None
    assert bundle_with_backend.unanchored_meta_calibrator is not None


@pytest.mark.asyncio
async def test_acompile_distillation_bundle_delegates_and_validates() -> None:
    dataset = [
        {
            "id": f"item_{i}",
            "rubric": "Is code safe?",
            "candidate": f"print({i})",
            "expected": (i % 2 == 0),
        }
        for i in range(12)
    ]
    bundle = await acompile_distillation_bundle(
        dataset=dataset,
        backend=None,
    )
    assert bundle.anchor_bank is not None
    assert bundle.meta_calibrator is None

    bundle_no_meta = await acompile_distillation_bundle(
        dataset=dataset,
        backend="stub://pass",
        fit_meta=False,
    )
    assert bundle_no_meta.anchor_bank is not None
    assert bundle_no_meta.meta_calibrator is None

    with pytest.raises(ConfigurationError, match="Provide exactly one of dataset=... or telemetry=..."):
        await acompile_distillation_bundle(backend="stub://pass")


@pytest.mark.asyncio
async def test_acompile_distillation_bundle_telemetry_with_backend() -> None:
    recs = [
        DistillationRecord(
            event_id=f"ev_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is code safe?",
            candidate=f"print({i})",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_model="student-3b",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_model="teacher-26b",
            teacher_passed=(i % 2 == 0),
            teacher_reason="Safe execution." if i % 2 == 0 else "Unsafe code.",
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"hash_{i}",
        )
        for i in range(12)
    ]
    bundle = await acompile_distillation_bundle(
        telemetry=recs,
        backend="stub://pass",
        student_model="student-3b",
        include_anchors=True,
        fit_meta=True,
        min_rubric_overlap=0.25,
    )
    assert bundle.anchor_bank is not None
    assert bundle.meta_calibrator is not None
    assert bundle.unanchored_meta_calibrator is not None


@pytest.mark.asyncio
async def test_acompile_distillation_bundle_dataset_overlap_zero_and_no_anchors() -> None:
    dataset = [
        {
            "id": f"item_{i}",
            "rubric": "Is code safe?",
            "candidate": f"print({i})",
            "expected": (i % 2 == 0),
        }
        for i in range(12)
    ]
    bundle_zero_overlap = await acompile_distillation_bundle(
        dataset=dataset,
        backend="stub://pass",
        include_anchors=True,
        fit_meta=True,
        min_rubric_overlap=0.0,
    )
    assert bundle_zero_overlap.anchor_bank is not None
    assert bundle_zero_overlap.meta_calibrator is not None
    assert bundle_zero_overlap.unanchored_meta_calibrator is None

    bundle_no_anchors = await acompile_distillation_bundle(
        dataset=dataset,
        backend="stub://pass",
        include_anchors=False,
        fit_meta=True,
    )
    assert bundle_no_anchors.anchor_bank is None
    assert bundle_no_anchors.meta_calibrator is not None


# ===========================================================================
# Section 3: Tests for logitgate.distill.anchor_bank
# ===========================================================================


def test_anchor_bank_jaccard_empty_tokens() -> None:
    assert _jaccard_tokens("", "") == 0.0
    assert _jaccard_tokens("hello", "") == 0.0
    assert _jaccard_tokens("", "world") == 0.0
    assert _jaccard_tokens("identical query tokens", "identical query tokens") == 1.0


def test_anchor_bank_condense_rationale_empty_or_whitespace() -> None:
    msg_pass = _condense_rationale("", passed=True, max_chars=120)
    assert "Satisfies every required" in msg_pass

    msg_fail = _condense_rationale("   ", passed=False, max_chars=120)
    assert "Violates at least one" in msg_fail

    # Stripping verdict tag leaves empty critique -> default message
    msg_verdict = _condense_rationale("[VERDICT: PASS]", passed=True, max_chars=120)
    assert "Satisfies every required" in msg_verdict


def test_anchor_bank_condense_rationale_bullet_stripping_and_slot_neutrality() -> None:
    critique = (
        "- Candidate includes proper validation.\n"
        "- Conforms with Option A constraints.\n"
        "[VERDICT: PASS]"
    )
    condensed = _condense_rationale(critique, passed=True, max_chars=150)
    assert "[VERDICT: PASS]" not in condensed
    assert "option a" not in condensed.lower()
    assert "the corresponding option" in condensed.lower()

    short_critique = "- OK.\n- Fine."
    condensed_short = _condense_rationale(short_critique, passed=True, max_chars=150)
    assert "OK. Fine." in condensed_short


def test_anchor_bank_condense_rationale_truncation_boundary() -> None:
    long_critique = "Verification completed: " + ("x" * 200)
    condensed = _condense_rationale(long_critique, passed=True, max_chars=50)
    assert len(condensed) <= 50
    assert condensed.endswith("...")


def test_anchor_bank_truncate_candidate_boundary() -> None:
    assert _truncate_anchor_candidate("short text", 100) == "short text"
    long_text = "word " * 50
    truncated = _truncate_anchor_candidate(long_text, 40)
    assert len(truncated) <= 40
    assert truncated.endswith("...")


def test_anchor_exemplar_validation_empty_rubric_or_candidate() -> None:
    with pytest.raises((ConfigurationError, ValidationError), match="AnchorExemplar.rubric cannot be empty"):
        AnchorExemplar(
            exemplar_id="1",
            rubric="   ",
            candidate="print('hi')",
            passed=True,
        )

    with pytest.raises((ConfigurationError, ValidationError), match="AnchorExemplar.candidate cannot be empty"):
        AnchorExemplar(
            exemplar_id="1",
            rubric="Is safe?",
            candidate="",
            passed=True,
        )


def test_contrastive_anchor_pair_validation_polarity_and_priority() -> None:
    ex_pass = AnchorExemplar(exemplar_id="1", rubric="R", candidate="A", passed=True)
    ex_fail = AnchorExemplar(exemplar_id="2", rubric="R", candidate="B", passed=False)

    with pytest.raises((ConfigurationError, ValidationError), match="positive must have passed=True"):
        ContrastiveAnchorPair(
            pair_id="p1",
            rubric="R",
            positive=ex_fail,
            negative=ex_fail,
        )

    with pytest.raises((ConfigurationError, ValidationError), match="negative must have passed=False"):
        ContrastiveAnchorPair(
            pair_id="p1",
            rubric="R",
            positive=ex_pass,
            negative=ex_pass,
        )

    with pytest.raises(ConfigurationError, match="priority_score must be a finite non-negative float"):
        pair_bad = ContrastiveAnchorPair.model_construct(
            pair_id="p1",
            rubric="R",
            positive=ex_pass,
            negative=ex_fail,
            priority_score=-1.0,
        )
        cast(Any, pair_bad)._validate_pair()


def test_generic_cases_json_not_list_rejection(tmp_path: Path) -> None:
    p = tmp_path / "dict_dataset.json"
    p.write_text('{"item": 1}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must contain a list of items"):
        _load_generic_cases(p)


def test_generic_cases_invalid_json_rejection(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{broken", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        _load_generic_cases(p)


def test_generic_cases_jsonl_and_markdown_support(tmp_path: Path) -> None:
    p_jsonl = tmp_path / "data.jsonl"
    rows = [
        {"id": "j1", "rubric": "Is SQL safe?", "candidate": "SELECT 1", "expected": True},
        {"id": "j2", "rubric": "Is SQL safe?", "candidate": "DROP TABLE", "expected": False},
    ]
    p_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded_jsonl = _load_generic_cases(p_jsonl)
    assert len(loaded_jsonl) == 2

    p_md = tmp_path / "cases.md"
    p_md.write_text(
        "# Benchmark\n\n"
        "| Rubric | Candidate | Expected |\n"
        "| --- | --- | --- |\n"
        "| Is SQL safe? | SELECT 1 | true |\n"
        "| Is SQL safe? | DROP TABLE | false |\n",
        encoding="utf-8",
    )
    loaded_md = _load_generic_cases(p_md)
    assert len(loaded_md) >= 2


def test_generic_cases_unsupported_format_rejection(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported dataset format '.csv'"):
        _load_generic_cases(p)


def test_generic_cases_row_filtering_and_object_duck_typing() -> None:
    class _DuckItem:
        def __init__(self, item_id: str, rubric: str, candidate: str, expected: Any) -> None:
            self.id = item_id
            self.rubric = rubric
            self.candidate = candidate
            self.expected = expected
            self.domain = "test_domain"
            self.teacher_reason = "Reason."

    mixed_raw = [
        {"id": "m1", "rubric": "", "candidate": "Cand", "expected": True},
        {"id": "m2", "rubric": "R", "candidate": "", "expected": True},
        {"id": "m3", "rubric": "R", "candidate": "Cand", "expected": "maybe"},
        {"id": "m4", "rubric": "Valid Rubric", "candidate": "Valid Cand", "expected": True},
        _DuckItem("d1", "Duck Rubric", "Duck Cand", True),
        _DuckItem("d2", "Duck Rubric", "Duck Cand", "invalid_expected"),
    ]
    loaded = _load_generic_cases(mixed_raw)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "m4"
    assert loaded[1]["id"] == "d1"


def test_anchor_bank_validation_schema_empty_pairs_and_overlap_bounds() -> None:
    pair = ContrastiveAnchorPair(
        pair_id="p1",
        rubric="R",
        positive=AnchorExemplar(exemplar_id="1", rubric="R", candidate="A", passed=True),
        negative=AnchorExemplar(exemplar_id="2", rubric="R", candidate="B", passed=False),
    )
    with pytest.raises((ConfigurationError, ValidationError), match="Unsupported DistilledAnchorBank schema_version=2"):
        DistilledAnchorBank(schema_version=2, pairs=[pair])

    with pytest.raises((ConfigurationError, ValidationError), match="must contain at least one ContrastiveAnchorPair"):
        DistilledAnchorBank(pairs=[])

    with pytest.raises(ConfigurationError, match="min_rubric_overlap must be a finite float in"):
        bank_bad = DistilledAnchorBank.model_construct(pairs=[pair], min_rubric_overlap=-0.5)
        cast(Any, bank_bad)._validate_bank()


def test_anchor_bank_from_dataset_invalid_max_bank_pairs() -> None:
    with pytest.raises(ConfigurationError, match="max_bank_pairs must be a positive integer"):
        DistilledAnchorBank.from_dataset([{"id": "1"}], max_bank_pairs=True)

    with pytest.raises(ConfigurationError, match="max_bank_pairs must be a positive integer"):
        DistilledAnchorBank.from_dataset([{"id": "1"}], max_bank_pairs=0)


def test_anchor_bank_from_dataset_empty_cases_rejection() -> None:
    with pytest.raises(ConfigurationError, match="Cannot compile DistilledAnchorBank from empty dataset"):
        DistilledAnchorBank.from_dataset([])


def test_anchor_bank_fallback_cross_rubric_pairing() -> None:
    cases = [
        {"id": "1", "rubric": "Rubric A", "candidate": "SELECT 1", "expected": True},
        {"id": "2", "rubric": "Rubric A", "candidate": "SELECT 2", "expected": True},
        {"id": "3", "rubric": "Rubric B", "candidate": "DROP 1", "expected": False},
        {"id": "4", "rubric": "Rubric B", "candidate": "DROP 2", "expected": False},
    ]
    bank = DistilledAnchorBank.from_dataset(cases, max_bank_pairs=2)
    assert len(bank.pairs) == 2
    assert bank.pairs[0].positive.passed is True
    assert bank.pairs[0].negative.passed is False


def test_anchor_bank_fallback_cross_rubric_all_pos_or_all_neg_rejection() -> None:
    all_pos = [
        {"id": "1", "rubric": "Rubric A", "candidate": "Pass 1", "expected": True},
        {"id": "2", "rubric": "Rubric B", "candidate": "Pass 2", "expected": True},
    ]
    with pytest.raises(ConfigurationError, match="requires at least one passing and one failing item"):
        DistilledAnchorBank.from_dataset(all_pos)


def test_anchor_bank_domain_diversification_bucket_slicing() -> None:
    cases: list[dict[str, Any]] = []
    for d_idx in range(6):
        domain_name = f"domain_{d_idx}"
        cases.append({
            "id": f"p_{d_idx}",
            "rubric": f"Rubric in {domain_name}",
            "candidate": f"Valid candidate in {domain_name}",
            "expected": True,
            "domain": domain_name,
        })
        cases.append({
            "id": f"n_{d_idx}",
            "rubric": f"Rubric in {domain_name}",
            "candidate": f"Invalid candidate in {domain_name}",
            "expected": False,
            "domain": domain_name,
        })
    bank = DistilledAnchorBank.from_dataset(cases, max_bank_pairs=4)
    assert len(bank.pairs) == 4
    domains_selected = {p.domain for p in bank.pairs}
    assert len(domains_selected) == 4


def test_anchor_bank_from_distillation_records_empty_and_bonus() -> None:
    with pytest.raises(ConfigurationError, match="No DistillationRecords available"):
        DistilledAnchorBank.from_distillation_records([])

    recs = [
        DistillationRecord(
            event_id="e1",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate="SELECT 1",
            label=True,
            discrepancy_type="polarity_flip",
            student_layer="LogitGate",
            student_model="student-3b",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=True,
            cascade_passed=True,
            candidate_sha256="sha1",
        ),
        DistillationRecord(
            event_id="e2",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate="DROP TABLE",
            label=False,
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_model="student-3b",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=False,
            cascade_passed=False,
            candidate_sha256="sha2",
        ),
    ]
    bank = DistilledAnchorBank.from_distillation_records(recs)
    assert len(bank.pairs) == 1
    assert bank.pairs[0].priority_score >= 1.0


def test_anchor_bank_retrieve_zero_k_or_empty_bank() -> None:
    bank = _build_minimal_anchor_bank()
    assert bank.retrieve("Is query safe?", k=0) == []
    empty_bank = bank.model_copy(update={"pairs": []})
    assert empty_bank.retrieve("Is query safe?", k=1) == []


def test_anchor_bank_retrieve_exclude_exact_candidate_sha256() -> None:
    bank = _build_minimal_anchor_bank()
    sample_pair = bank.pairs[0]
    retrieved = bank.retrieve(
        sample_pair.rubric,
        candidate=sample_pair.positive.candidate,
        exclude_exact=True,
    )
    assert sample_pair not in retrieved

    retrieved_neg = bank.retrieve(
        sample_pair.rubric,
        candidate=sample_pair.negative.candidate,
        exclude_exact=True,
    )
    assert sample_pair not in retrieved_neg


def test_anchor_bank_retrieve_min_rubric_overlap_filtering() -> None:
    bank = _build_minimal_anchor_bank()
    retrieved = bank.retrieve(
        "Acute glomerular nephritis with pulmonary embolism",
        k=1,
        min_rubric_overlap=0.25,
    )
    assert retrieved == []


def test_anchor_bank_retrieve_bm25_scoring_and_duplicate_rubric_deferral() -> None:
    pair1 = ContrastiveAnchorPair(
        pair_id="p1",
        rubric="Is SQL query parameterized?",
        positive=AnchorExemplar(exemplar_id="1", rubric="Is SQL query parameterized?", candidate="P1", passed=True),
        negative=AnchorExemplar(exemplar_id="2", rubric="Is SQL query parameterized?", candidate="N1", passed=False),
        priority_score=1.2,
    )
    pair2 = ContrastiveAnchorPair(
        pair_id="p2",
        rubric="Is SQL query parameterized?",
        positive=AnchorExemplar(exemplar_id="3", rubric="Is SQL query parameterized?", candidate="P2", passed=True),
        negative=AnchorExemplar(exemplar_id="4", rubric="Is SQL query parameterized?", candidate="N2", passed=False),
        priority_score=1.1,
    )
    bank = DistilledAnchorBank(pairs=[pair1, pair2], max_pairs=2, min_rubric_overlap=0.1)
    retrieved = bank.retrieve("Is SQL query parameterized?", k=2)
    assert len(retrieved) == 2
    assert retrieved[0].pair_id == "p1"
    assert retrieved[1].pair_id == "p2"


def test_anchor_bank_format_anchor_block_empty_retrieve() -> None:
    bank = _build_minimal_anchor_bank()
    formatted = bank.format_anchor_block("Acute glomerular nephritis with pulmonary embolism", k=1)
    assert formatted == ""


def test_anchor_bank_format_anchor_block_different_rubric_header() -> None:
    pair = ContrastiveAnchorPair(
        pair_id="p1",
        rubric="Does code parameterize SQL?",
        positive=AnchorExemplar(exemplar_id="1", rubric="Does code parameterize SQL?", candidate="SELECT :id", passed=True),
        negative=AnchorExemplar(exemplar_id="2", rubric="Does code parameterize SQL?", candidate="SELECT '1'", passed=False),
    )
    bank = DistilledAnchorBank(pairs=[pair], min_rubric_overlap=0.1)
    block = bank.format_anchor_block("Does SQL query use parameterization?", boundary_nonce="test_nonce")
    assert "Reference Rubric 1: Does code parameterize SQL?" in block
    assert "<reference_pass_test_nonce>" in block
    assert "<reference_fail_test_nonce>" in block


def test_resolve_anchor_bank_variations(tmp_path: Path) -> None:
    assert resolve_anchor_bank(None) is None
    bank = _build_minimal_anchor_bank()
    assert resolve_anchor_bank(bank) is bank

    saved = bank.save(tmp_path / "anchors.json")
    loaded = resolve_anchor_bank(saved)
    assert isinstance(loaded, DistilledAnchorBank)

    with pytest.raises(ConfigurationError, match="anchor_bank must be a DistilledAnchorBank"):
        resolve_anchor_bank(999)  # type: ignore[arg-type]


def test_anchor_bank_load_validation_error_wrapping(tmp_path: Path) -> None:
    p = tmp_path / "bad_bank.json"
    p.write_text(json.dumps({"schema_version": 1, "pairs": []}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid DistilledAnchorBank"):
        DistilledAnchorBank.load(p)


# ===========================================================================
# Section 4: Tests for logitgate.distill.meta_calibrator
# ===========================================================================


def test_meta_calibrator_pass_distribution_stats_dirty_data_handling() -> None:
    dirty_top_logprobs: list[Any] = [
        "not a mapping",
        {"token": None, "logprob": -0.5},
        {"token": "A", "logprob": True},
        {"token": "A", "logprob": "not-a-number"},
        {"token": "A", "logprob": float("nan")},
        {"token": "A", "logprob": float("inf")},
        {"token": "A", "logprob": -0.2},
        {"token": "A", "logprob": -0.5},
        {"token": "B", "logprob": -1.5},
        {"token": "C", "logprob": -2.0},
    ]
    entropy, non_target_mass = _pass_distribution_stats(dirty_top_logprobs, target_tokens=("A", "B"))
    assert entropy >= 0.0
    assert 0.0 <= non_target_mass <= 1.0


def test_meta_calibrator_pass_distribution_stats_total_obs_over_one() -> None:
    top_lps = [
        {"token": "A", "logprob": 0.0},
        {"token": "B", "logprob": 0.0},
    ]
    entropy, non_target_mass = _pass_distribution_stats(top_lps, target_tokens=("A", "B"))
    assert entropy > 0.0
    # Observed total_obs = 2.0 on target tokens A and B; normalized target_mass = 1.0 -> non_target_mass = 0.0
    assert non_target_mass == 0.0


def test_meta_calibrator_extract_logit_features_forward_only() -> None:
    top_fwd = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.4}]
    feat = extract_logit_features(top_fwd, top_rev=None)
    assert feat.signed_slot_bias == 0.0
    assert feat.positional_bias_magnitude == 0.0
    assert feat.symmetric_log_odds > 1.0


def test_solve_linear_system_near_zero_pivot() -> None:
    mat = [
        [1e-15, 1.0],
        [1.0, 2.0],
    ]
    rhs = [1.0, 3.0]
    sol = _solve_linear_system(mat, rhs)
    assert len(sol) == 2
    assert all(math.isfinite(x) for x in sol)


def test_fit_ridge_logistic_empty_features_rejection() -> None:
    with pytest.raises(CalibrationError, match="Cannot fit LogitMetaCalibrator on 0 samples"):
        _fit_ridge_logistic([], [])


def test_auroc_binary_no_pos_neg_and_ties() -> None:
    assert _auroc_binary([0.5, 0.6], [True, True]) == 0.5
    assert _auroc_binary([0.5, 0.6], [False, False]) == 0.5
    assert _auroc_binary([0.5, 0.5], [True, False]) == 0.5


def test_brier_score_empty_sequence() -> None:
    assert _brier_score([], []) == 0.25


def test_meta_calibrator_validation_schema_weights_intercept_and_penalties() -> None:
    with pytest.raises((CalibrationError, ValidationError), match="Unsupported LogitMetaCalibrator schema_version=2"):
        LogitMetaCalibrator(schema_version=2)

    with pytest.raises((CalibrationError, ValidationError), match="weights must contain 5 finite floats"):
        LogitMetaCalibrator(weights=[1.0, 0.0])

    with pytest.raises((CalibrationError, ValidationError), match="weights must contain 5 finite floats"):
        LogitMetaCalibrator(weights=[float("nan"), 0.0, 0.0, 0.0, 0.0])

    with pytest.raises((CalibrationError, ValidationError), match="must be strictly positive for monotonicity"):
        LogitMetaCalibrator(weights=[-0.1, 0.0, 0.0, 0.0, 0.0])

    with pytest.raises((CalibrationError, ValidationError), match="intercept must be a finite float"):
        LogitMetaCalibrator(intercept=float("nan"))

    with pytest.raises((CalibrationError, ValidationError), match="l2_penalty must be a finite positive float"):
        LogitMetaCalibrator(l2_penalty=-1.0)


def test_meta_calibrator_validation_thresholds_consistency() -> None:
    with pytest.raises((CalibrationError, ValidationError), match="pass_threshold and fail_threshold must both be set"):
        LogitMetaCalibrator(pass_threshold=0.8, fail_threshold=None)

    with pytest.raises((CalibrationError, ValidationError), match="Thresholds must satisfy 0.0 <= fail_threshold < pass_threshold <= 1.0"):
        LogitMetaCalibrator(pass_threshold=0.4, fail_threshold=0.6)


def test_meta_calibrator_predict_feature_count_mismatch_and_clamping() -> None:
    cal = LogitMetaCalibrator()
    with pytest.raises(CalibrationError, match="Expected 5 features, got 3"):
        cal.predict_log_odds([1.0, 2.0, 3.0])

    extreme_cal = LogitMetaCalibrator(intercept=100.0, weights=[20.0, 0.0, 0.0, 0.0, 0.0])
    lo = extreme_cal.predict_log_odds(LogitFeatureVector(10.0, 0.0, 0.0, 0.0, 0.0))
    assert lo == 50.0


def test_meta_calibrator_fit_invalid_l2_or_small_sample_rejection() -> None:
    with pytest.raises(CalibrationError, match="l2_penalty must be a positive float"):
        LogitMetaCalibrator.fit([], [], l2_penalty=True)

    with pytest.raises(CalibrationError, match="requires matching sequences of at least 10 samples"):
        LogitMetaCalibrator.fit([LogitFeatureVector(1.0, 0.0, 0.0, 0.0, 0.0)] * 5, [True] * 5)


def test_meta_calibrator_fit_dirty_features_or_labels_rejection() -> None:
    dirty_feats = [LogitFeatureVector(float("nan"), 0.0, 0.0, 0.0, 0.0)] + [
        LogitFeatureVector(1.0, 0.0, 0.0, 0.0, 0.0)
    ] * 9
    with pytest.raises(CalibrationError, match="All 5 features in each sample must be finite floats"):
        LogitMetaCalibrator.fit(dirty_feats, [True] * 10)

    clean_feats = [LogitFeatureVector(1.0, 0.0, 0.0, 0.0, 0.0)] * 10
    with pytest.raises(CalibrationError, match="Labels must be booleans or 0/1 ints"):
        LogitMetaCalibrator.fit(clean_feats, cast(Any, ["maybe"] * 10))


def test_meta_calibrator_fit_class_imbalance_rejection() -> None:
    clean_feats = [LogitFeatureVector(1.0, 0.0, 0.0, 0.0, 0.0)] * 10
    with pytest.raises(CalibrationError, match="requires at least 2 positive and 2 negative samples"):
        LogitMetaCalibrator.fit(clean_feats, [True] + [False] * 9)


def test_meta_calibrator_fit_target_risk_alpha_none() -> None:
    feats = [
        LogitFeatureVector(1.5 if i % 2 == 0 else -1.5, 0.0, 0.0, 0.1, 0.05)
        for i in range(12)
    ]
    labels = [(i % 2 == 0) for i in range(12)]
    mc = LogitMetaCalibrator.fit(feats, labels, target_risk_alpha=None)
    assert mc.pass_threshold is None
    assert mc.fail_threshold is None


def test_meta_calibrator_from_distillation_records_sparse_log_odds() -> None:
    recs: list[DistillationRecord] = []
    for i in range(12):
        passed = (i % 2 == 0)
        recs.append(
            DistillationRecord(
                event_id=f"ev_{i}",
                timestamp="2026-11-15T00:00:00+00:00",
                rubric="Is clean?",
                candidate=f"Cand {i}",
                label=passed,
                discrepancy_type="ambiguous_resolved",
                student_layer="LogitGate",
                student_model="student-3b",
                student_status=DecisionStatus.AMBIGUOUS,
                student_log_odds_pass=(1.5 if passed else -1.5),
                student_positional_bias=0.2,
                teacher_layer="JudgeGate",
                teacher_passed=passed,
                cascade_passed=passed,
                candidate_sha256=f"hash_{i}",
            )
        )
    mc = LogitMetaCalibrator.from_distillation_records(recs)
    assert mc.weights[0] >= 0.05
    assert mc.calibration_set_size == 12


def test_meta_calibrator_load_invalid_wrapping(tmp_path: Path) -> None:
    p = tmp_path / "bad_mc.json"
    p.write_text(json.dumps({"schema_version": 1, "weights": [-5.0]}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="Invalid LogitMetaCalibrator"):
        LogitMetaCalibrator.load(p)


def test_resolve_meta_calibrator_variations(tmp_path: Path) -> None:
    assert resolve_meta_calibrator(None) is None
    mc = LogitMetaCalibrator()
    assert resolve_meta_calibrator(mc) is mc

    saved = mc.save(tmp_path / "mc.json")
    loaded = resolve_meta_calibrator(saved)
    assert isinstance(loaded, LogitMetaCalibrator)

    with pytest.raises(ConfigurationError, match="meta_calibrator must be a LogitMetaCalibrator"):
        resolve_meta_calibrator(42)  # type: ignore[arg-type]


def test_fit_meta_calibrator_item_count_validation() -> None:
    with pytest.raises(CalibrationError, match="requires at least 10 calibration items"):
        fit_meta_calibrator("stub://pass", [{"id": "1", "rubric": "R", "candidate": "C", "expected": True}])


def test_fit_meta_calibrator_backend_ownership_and_closing() -> None:
    dataset = [
        {"id": f"item_{i}", "rubric": "Is safe?", "candidate": f"code_{i}", "expected": (i % 2 == 0)}
        for i in range(12)
    ]
    mc = fit_meta_calibrator("stub://pass", dataset)
    assert mc.weights[0] >= 0.05


@pytest.mark.asyncio
async def test_afit_meta_calibrator_item_count_validation() -> None:
    with pytest.raises(CalibrationError, match="requires at least 10 calibration items"):
        await afit_meta_calibrator("stub://pass", [{"id": "1", "rubric": "R", "candidate": "C", "expected": True}])


class _SequentialWarmupBackend(LogitBackend):
    prefers_sequential_prefix_warmup = True

    def query_top_logprobs(
        self,
        prompt: str,
        top_k: int = 10,
        allowed_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del prompt, top_k, allowed_tokens, kwargs
        return [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.3}]


@pytest.mark.asyncio
async def test_afit_meta_calibrator_sequential_warmup_vs_parallel() -> None:
    dataset = [
        {"id": f"item_{i}", "rubric": "Is safe?", "candidate": f"code_{i}", "expected": (i % 2 == 0)}
        for i in range(12)
    ]
    backend = _SequentialWarmupBackend()
    mc = await afit_meta_calibrator(backend, dataset)
    assert mc.weights[0] >= 0.05


# ===========================================================================
# Section 5: Tests for logitgate.distill.lora
# ===========================================================================


def test_lora_require_hf_deps_missing_torch_or_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BlockModule:
        def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
            if fullname in ("torch", "transformers"):
                raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)

    monkeypatch.setattr(sys, "meta_path", [_BlockModule(), *sys.meta_path])
    for mod in ["torch", "transformers"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ConfigurationError, match="requires PyTorch and HuggingFace Transformers"):
        _require_hf_deps(need_peft=False)


def test_lora_require_hf_deps_missing_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class _BlockPeft:
        def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
            if fullname == "peft" or fullname.startswith("peft."):
                raise ModuleNotFoundError("No module named 'peft'", name="peft")

    monkeypatch.setattr(sys, "meta_path", [_BlockPeft(), *sys.meta_path])
    for mod in [m for m in list(sys.modules) if m == "peft" or m.startswith("peft.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(ConfigurationError, match="requires HuggingFace PEFT"):
        _require_hf_deps(need_peft=True)


def test_lora_distill_config_target_modules_string_validation() -> None:
    with pytest.raises((ConfigurationError, ValidationError), match="target_modules string cannot be empty"):
        LoRADistillConfig(target_modules="   ")

    cfg = LoRADistillConfig(target_modules="all-linear")
    assert cfg.target_modules == "all-linear"


def test_lora_distill_config_target_modules_list_validation() -> None:
    with pytest.raises((ConfigurationError, ValidationError), match="target_modules list must contain one or more non-empty"):
        LoRADistillConfig(target_modules=[])

    with pytest.raises((ConfigurationError, ValidationError), match="target_modules list must contain one or more non-empty"):
        LoRADistillConfig(target_modules=["q_proj", "   "])

    with pytest.raises((ConfigurationError, ValidationError), match="target_modules list must contain one or more non-empty"):
        LoRADistillConfig(target_modules=["q_proj", ""])

    cfg = LoRADistillConfig(target_modules=["q_proj", "v_proj"])
    assert cfg.target_modules == ["q_proj", "v_proj"]


def test_lora_distill_config_field_bounds_and_bool_rejection() -> None:
    with pytest.raises(ValidationError):
        LoRADistillConfig(rank=0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(rank=257)
    with pytest.raises(ValidationError):
        LoRADistillConfig(lora_alpha=0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(lora_dropout=-0.1)
    with pytest.raises(ValidationError):
        LoRADistillConfig(lora_dropout=1.0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(learning_rate=0.0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(epochs=0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(batch_size=0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(grad_accum_steps=0)
    with pytest.raises(ValidationError):
        LoRADistillConfig(max_seq_len=63)


def test_lora_distill_manifest_save_and_load(tmp_path: Path) -> None:
    manifest = LoRADistillManifest(
        base_model="Qwen/Qwen2.5-Coder-3B",
        adapter_dir=str(tmp_path / "adapter"),
        ollama_modelfile_path=str(tmp_path / "Modelfile.ollama"),
        vllm_launch_script_path=str(tmp_path / "serve_vllm.sh"),
        llamacpp_launch_script_path=str(tmp_path / "serve_llamacpp.sh"),
        vllm_cli_args=["vllm", "serve"],
        llamacpp_cli_args=["llama-server"],
        num_training_examples=24,
        epochs=3,
        rank=8,
        lora_alpha=16,
        target_modules="all-linear",
        initial_loss=1.5,
        final_loss=0.3,
    )
    p = manifest.save(tmp_path / "manifest.json")
    loaded = LoRADistillManifest.load(p)
    assert loaded.base_model == "Qwen/Qwen2.5-Coder-3B"
    assert loaded.num_training_examples == 24
    assert loaded.final_loss < loaded.initial_loss


def test_prepare_symmetric_lora_examples_empty_dataset_rejection() -> None:
    with pytest.raises(ConfigurationError, match="Cannot prepare LoRA examples from empty dataset"):
        prepare_symmetric_lora_examples([])


def test_prepare_symmetric_lora_examples_with_anchor_bank_and_teacher_labels() -> None:
    bank = _build_minimal_anchor_bank()
    dataset = [
        {"id": "c1", "rubric": "Is SQL safe?", "candidate": "SELECT 1", "expected": True},
        {"id": "c2", "rubric": "Is SQL safe?", "candidate": "DROP TABLE", "expected": False},
    ]
    examples = prepare_symmetric_lora_examples(
        dataset,
        anchor_bank=bank,
        teacher_labels={"c1": True, "c2": False},
    )
    assert len(examples) == 4
    assert any("[CALIBRATION ANCHORS]" in ex.user_prompt for ex in examples)


def test_write_text_nofollow_symlink_rejections(tmp_path: Path) -> None:
    real_f = tmp_path / "real.txt"
    real_f.write_text("ok", encoding="utf-8")
    sym_f = tmp_path / "sym.txt"
    sym_f.symlink_to(real_f)
    with pytest.raises(ConfigurationError, match="Refusing to write through symlink"):
        _write_text_nofollow(sym_f, "fail")

    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    sym_dir = tmp_path / "sym_dir"
    sym_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="Refusing to write under symlinked directory"):
        _write_text_nofollow(sym_dir / "target.txt", "fail")


def test_emit_deployment_manifests_shell_quoting_and_options(tmp_path: Path) -> None:
    info = emit_deployment_manifests(
        tmp_path / "deploy_out",
        base_model="models/my custom coder v1",
        ollama_base_model="my-coder:latest",
        lora_name="custom-student",
    )
    modelfile_txt = Path(info["ollama_modelfile_path"]).read_text(encoding="utf-8")
    assert "FROM my-coder:latest" in modelfile_txt
    assert "ollama create custom-student" in modelfile_txt

    vllm_txt = Path(info["vllm_launch_script_path"]).read_text(encoding="utf-8")
    assert "'models/my custom coder v1'" in vllm_txt

    llamacpp_txt = Path(info["llamacpp_launch_script_path"]).read_text(encoding="utf-8")
    assert "convert_lora_to_gguf.py" in llamacpp_txt


def test_encode_prompt_ids_chat_template_list_and_input_ids() -> None:
    class _TokTemplateList:
        chat_template = "{{ message }}"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
            del messages, kwargs
            return [101, 102, 103]

    assert _encode_prompt_ids(_TokTemplateList(), "Sys", "User") == [101, 102, 103]

    class _TokTemplate2D:
        chat_template = "{{ message }}"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            del messages, kwargs
            return type("ChatOutput", (), {"input_ids": [[201, 202]]})()

    assert _encode_prompt_ids(_TokTemplate2D(), "Sys", "User") == [201, 202]

    class _TokTemplate1D:
        chat_template = "{{ message }}"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            del messages, kwargs
            return type("ChatOutput", (), {"input_ids": [301, 302]})()

    assert _encode_prompt_ids(_TokTemplate1D(), "Sys", "User") == [301, 302]

    class _TokTemplateFail:
        chat_template = "{{ message }}"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            del messages, kwargs
            raise RuntimeError("Chat template syntax error")

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del text, add_special_tokens
            return [401, 402]

    assert _encode_prompt_ids(_TokTemplateFail(), "Sys", "User") == [401, 402]


def test_resolve_single_token_id_multi_token_and_empty_error() -> None:
    class _TokDirect:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [10] if text == "A" else [11]

    assert _resolve_single_token_id(_TokDirect(), "A") == 10

    class _TokPrefixed:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if text == " A":
                return [10]
            return [1, 2]

    assert _resolve_single_token_id(_TokPrefixed(), "A") == 10

    class _TokFallback:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [1, 2, 99]

    assert _resolve_single_token_id(_TokFallback(), "A") == 99

    class _TokEmpty:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return []

    with pytest.raises(ConfigurationError, match="Tokenizer returned empty IDs for token"):
        _resolve_single_token_id(_TokEmpty(), "A")


def test_resolve_torch_device_and_dtype_variations() -> None:
    class _MockTorch:
        float32 = "f32"
        float16 = "f16"
        bfloat16 = "bf16"

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def is_bf16_supported() -> bool:
                return True

        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return False

    mock_torch = _MockTorch()
    assert _resolve_torch_device(mock_torch, "cpu") == "cpu"
    assert _resolve_torch_device(mock_torch, "auto") == "cuda"

    mock_torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    mock_torch.backends.mps.is_available = lambda: True  # type: ignore[method-assign]
    assert _resolve_torch_device(mock_torch, "auto") == "mps"

    mock_torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
    assert _resolve_torch_device(mock_torch, "auto") == "cpu"

    assert _resolve_torch_dtype(mock_torch, "float32", "cpu") == "f32"
    assert _resolve_torch_dtype(mock_torch, "float16", "cpu") == "f16"
    assert _resolve_torch_dtype(mock_torch, "bfloat16", "cpu") == "bf16"
    assert _resolve_torch_dtype(mock_torch, "auto", "cpu") == "f32"
    assert _resolve_torch_dtype(mock_torch, "auto", "cuda:0") == "bf16"

    mock_torch.cuda.is_bf16_supported = lambda: False  # type: ignore[method-assign]
    assert _resolve_torch_dtype(mock_torch, "auto", "cuda:0") == "f16"


def test_hf_logit_backend_query_top_logprobs_empty_tokens_error() -> None:
    pytest.importorskip("torch")

    class _EmptyTok:
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del text, add_special_tokens
            return []

    backend = HuggingFaceLogitBackend(object(), _EmptyTok())
    with pytest.raises(BackendError, match="Tokenizer produced empty input_ids"):
        backend.query_top_logprobs("test")


def test_hf_logit_backend_query_top_logprobs_allowed_tokens_and_device_fallback() -> None:
    torch = pytest.importorskip("torch")

    class _TinyModel(_TorchModule):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 32)

        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            del kwargs
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            logits[0, -1, 10] = 3.0
            logits[0, -1, 11] = -1.0
            return type("Out", (), {"logits": logits})()

    class _TinyTok:
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            s = text.strip()
            if s == "A":
                return [10]
            if s == "B":
                return [11]
            return [10, 11]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            m = {10: "A", 11: "B"}
            return "".join(m.get(i, f"tok_{i}") for i in ids)

    model = _TinyModel()
    tok = _TinyTok()
    backend = HuggingFaceLogitBackend(model, tok, device=None)
    assert "HuggingFaceLogitBackend" in repr(backend)

    res = backend.query_top_logprobs("Is code safe?", top_k=5, allowed_tokens=["A", "B"])
    assert len(res) >= 2
    toks = [r["token"] for r in res]
    assert "A" in toks
    assert "B" in toks

    class _NoParamModel(_TorchModule):
        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            del kwargs
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            logits[0, -1, 10] = 2.0
            return type("Out", (), {"logits": logits})()

    backend_no_param = HuggingFaceLogitBackend(_NoParamModel(), tok, device=None)
    res_no_param = backend_no_param.query_top_logprobs("Test", top_k=2)
    assert len(res_no_param) > 0


@pytest.mark.asyncio
async def test_hf_logit_backend_close_and_aclose() -> None:
    backend = HuggingFaceLogitBackend(object(), object())
    await backend.aclose()
    assert backend._model_obj is None
    assert backend._tokenizer is None


def test_train_distillation_lora_symlink_output_dir_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")

    class _MockPeftMod:
        pass

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    real_d = tmp_path / "real_lora"
    real_d.mkdir()
    sym_d = tmp_path / "sym_lora"
    sym_d.symlink_to(real_d, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="Refusing to write LoRA artifacts under symlink"):
        train_distillation_lora("dummy/model", [{"id": "1", "rubric": "R", "candidate": "C", "expected": True}], sym_d)


def test_train_distillation_lora_missing_tokenizer_for_in_memory_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")

    class _MockPeftMod:
        pass

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    with pytest.raises(ConfigurationError, match="tokenizer=... is required"):
        train_distillation_lora(object(), [{"id": "1", "rubric": "R", "candidate": "C", "expected": True}], tmp_path / "out", tokenizer=None)


def test_train_distillation_lora_end_to_end_mocked_peft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class _MockPeftModel(_TorchModule):
        def __init__(self, base_model: Any) -> None:
            super().__init__()
            self.base_model = base_model
            self.trainable = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None) -> Any:
            del attention_mask, labels
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            logits[0, -1, 10] = 2.0
            logits[0, -1, 11] = -1.0
            loss = torch.tensor(0.5, requires_grad=True) * self.trainable
            return type("Out", (), {"logits": logits, "loss": loss})()

        def save_pretrained(self, save_dir: str) -> None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_config.json").write_text(json.dumps({"r": 4}), encoding="utf-8")

    class _MockPeftMod:
        class TaskType:
            CAUSAL_LM = "CAUSAL_LM"

        @staticmethod
        def LoraConfig(**kwargs: Any) -> Any:
            return type("LoraConfigObj", (), kwargs)()

        @staticmethod
        def get_peft_model(model: Any, config: Any) -> Any:
            del config
            return _MockPeftModel(model)

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    class _BaseModel(_TorchModule):
        config = type("Config", (), {"_name_or_path": "tiny-model"})()

        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            del kwargs
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            logits[0, -1, 10] = 2.0
            logits[0, -1, 11] = -1.0
            return type("Out", (), {"logits": logits})()

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            s = text.strip()
            if s == "A":
                return [10]
            if s == "B":
                return [11]
            return [10, 11, 12]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            m = {10: "A", 11: "B"}
            return "".join(m.get(i, f"tok_{i}") for i in ids)

    dataset = [
        {"id": f"item_{i}", "rubric": "Is code clean?", "candidate": f"print({i})", "expected": (i % 2 == 0)}
        for i in range(12)
    ]
    out_dir = tmp_path / "lora_run"
    manifest = train_distillation_lora(
        _BaseModel(),
        dataset,
        out_dir,
        tokenizer=_Tok(),
        config=LoRADistillConfig(
            rank=4,
            lora_alpha=8,
            epochs=2,
            batch_size=4,
            grad_accum_steps=2,
            learning_rate=1e-3,
            device="cpu",
            max_seq_len=64,
        ),
    )
    assert manifest.num_training_examples == 24
    assert Path(manifest.adapter_dir, "adapter_config.json").is_file()
    assert manifest.bundle_path is not None and Path(manifest.bundle_path).is_file()
    assert manifest.profile_path is not None and Path(manifest.profile_path).is_file()
    assert Path(manifest.ollama_modelfile_path).is_file()
    assert Path(manifest.vllm_launch_script_path).is_file()
    assert Path(manifest.llamacpp_launch_script_path).is_file()


def test_train_distillation_lora_under_ten_items_no_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class _MockPeftModel(_TorchModule):
        def __init__(self, base_model: Any) -> None:
            super().__init__()
            self.base_model = base_model
            self.trainable = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None) -> Any:
            del attention_mask, labels
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            loss = torch.tensor(0.5, requires_grad=True) * self.trainable
            return type("Out", (), {"logits": logits, "loss": loss})()

        def save_pretrained(self, save_dir: str) -> None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_config.json").write_text("{}", encoding="utf-8")

    class _MockPeftMod:
        class TaskType:
            CAUSAL_LM = "CAUSAL_LM"

        @staticmethod
        def LoraConfig(**kwargs: Any) -> Any:
            return type("LoraConfigObj", (), kwargs)()

        @staticmethod
        def get_peft_model(model: Any, config: Any) -> Any:
            del config
            return _MockPeftModel(model)

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    class _BaseModel(_TorchModule):
        pass

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            s = text.strip()
            if s == "A":
                return [10]
            if s == "B":
                return [11]
            return [10, 11]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del ids, skip_special_tokens
            return "tok"

    dataset = [
        {"id": f"item_{i}", "rubric": "Is clean?", "candidate": f"print({i})", "expected": (i % 2 == 0)}
        for i in range(4)
    ]
    out_dir = tmp_path / "lora_small"
    manifest = train_distillation_lora(
        _BaseModel(),
        dataset,
        out_dir,
        tokenizer=_Tok(),
        config=LoRADistillConfig(
            rank=4,
            lora_alpha=8,
            epochs=1,
            batch_size=4,
            grad_accum_steps=2,
            learning_rate=1e-3,
            device="cpu",
        ),
    )
    assert manifest.bundle_path is None
    assert manifest.profile_path is None


def test_train_distillation_lora_zero_trainable_params_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class _UntrainableModel(_TorchModule):
        def __init__(self) -> None:
            super().__init__()
            self.fixed = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    class _MockPeftMod:
        class TaskType:
            CAUSAL_LM = "CAUSAL_LM"

        @staticmethod
        def LoraConfig(**kwargs: Any) -> Any:
            return type("LoraConfigObj", (), kwargs)()

        @staticmethod
        def get_peft_model(model: Any, config: Any) -> Any:
            del config
            return model

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            return [10] if text.strip() == "A" else [11]

    with pytest.raises(ConfigurationError, match="matched 0 trainable parameters"):
        train_distillation_lora(
            _UntrainableModel(),
            [{"id": "1", "rubric": "R", "candidate": "C", "expected": True}],
            tmp_path / "out",
            tokenizer=_Tok(),
        )


def test_anchor_bank_from_distillation_records_path_loading(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    recs = [
        DistillationRecord(
            event_id="e1",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate="SELECT 1",
            label=True,
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=True,
            cascade_passed=True,
            candidate_sha256="sha1",
        ),
        DistillationRecord(
            event_id="e2",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate="DROP TABLE",
            label=False,
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=False,
            cascade_passed=False,
            candidate_sha256="sha2",
        ),
    ]
    records_path.write_text("\n".join(r.model_dump_json() for r in recs) + "\n", encoding="utf-8")
    bank = DistilledAnchorBank.from_distillation_records(records_path)
    assert len(bank.pairs) == 1
    assert bank.source == str(records_path)


def test_anchor_bank_load_reraises_configuration_error(tmp_path: Path) -> None:
    p = tmp_path / "bad_schema_bank.json"
    p.write_text(json.dumps({"schema_version": 2, "pairs": []}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported DistilledAnchorBank schema_version=2"):
        DistilledAnchorBank.load(p)


def test_bundle_load_reraises_configuration_error(tmp_path: Path) -> None:
    p = tmp_path / "bad_schema_bundle.json"
    p.write_text(json.dumps({"schema_version": 2, "anchor_bank": None}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported DistillationBundle schema_version=2"):
        DistillationBundle.load(p)


def test_bundle_compile_edge_branches() -> None:
    recs = [
        DistillationRecord(
            event_id=f"e_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate=f"SELECT {i}",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=(i % 2 == 0),
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"sha_{i}",
        )
        for i in range(4)
    ]
    # Telemetry with recs < 10 and backend=None -> fit_meta skipped (branch 174->214)
    bundle_small_telemetry = compile_distillation_bundle(telemetry=recs, backend=None)
    assert bundle_small_telemetry.anchor_bank is not None
    assert bundle_small_telemetry.meta_calibrator is None

    cases = [
        {"id": f"c_{i}", "rubric": "Is query safe?", "candidate": f"SELECT {i}", "expected": (i % 2 == 0)}
        for i in range(12)
    ]
    # Dataset with include_anchors=False and backend="stub://pass" (branch 226->236, 246->256)
    bundle_no_anchors = compile_distillation_bundle(
        dataset=cases,
        backend="stub://pass",
        include_anchors=False,
    )
    assert bundle_no_anchors.anchor_bank is None
    assert bundle_no_anchors.meta_calibrator is not None


@pytest.mark.asyncio
async def test_bundle_acompile_edge_branches() -> None:
    recs = [
        DistillationRecord(
            event_id=f"e_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate=f"SELECT {i}",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=(i % 2 == 0),
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"sha_{i}",
        )
        for i in range(12)
    ]
    # Telemetry with include_anchors=False and backend="stub://pass" (branch 317->324)
    bundle = await acompile_distillation_bundle(
        telemetry=recs,
        backend="stub://pass",
        include_anchors=False,
    )
    assert bundle.anchor_bank is None
    assert bundle.meta_calibrator is not None


def test_meta_calibrator_from_distillation_records_path_and_skips(tmp_path: Path) -> None:
    p = tmp_path / "mc_records.jsonl"
    recs = [
        DistillationRecord(
            event_id=f"e_{i}",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate=f"SELECT {i}",
            label=(i % 2 == 0),
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_status=DecisionStatus.AMBIGUOUS,
            student_log_odds_pass=(1.5 if i % 2 == 0 else -1.5),
            student_positional_bias=0.2,
            teacher_layer="JudgeGate",
            teacher_passed=(i % 2 == 0),
            cascade_passed=(i % 2 == 0),
            candidate_sha256=f"sha_{i}",
        )
        for i in range(12)
    ]
    # Append record with no student log odds at all -> triggers continue (line 461)
    recs.append(
        DistillationRecord(
            event_id="e_skipped",
            timestamp="2026-11-15T00:00:00+00:00",
            rubric="Is query safe?",
            candidate="SELECT 999",
            label=True,
            discrepancy_type="ambiguous_resolved",
            student_layer="LogitGate",
            student_status=DecisionStatus.AMBIGUOUS,
            teacher_layer="JudgeGate",
            teacher_passed=True,
            cascade_passed=True,
            candidate_sha256="sha_skipped",
        )
    )
    p.write_text("\n".join(r.model_dump_json() for r in recs) + "\n", encoding="utf-8")
    mc = LogitMetaCalibrator.from_distillation_records(p)
    assert mc.calibration_set_size == 12


def test_meta_calibrator_load_reraises_calibration_error(tmp_path: Path) -> None:
    p = tmp_path / "bad_schema_mc.json"
    p.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="Unsupported LogitMetaCalibrator schema_version=2"):
        LogitMetaCalibrator.load(p)


@pytest.mark.asyncio
async def test_hf_logit_backend_aquery_top_logprobs() -> None:
    torch = pytest.importorskip("torch")

    class _TinyModel(_TorchModule):
        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            del kwargs
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            logits[0, -1, 10] = 2.0
            logits[0, -1, 11] = -1.0
            return type("Out", (), {"logits": logits})()

    class _TinyTok:
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            s = text.strip()
            if s == "A":
                return [10]
            if s == "B":
                return [11]
            return [10, 11]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            m = {10: "A", 11: "B"}
            return "".join(m.get(i, f"tok_{i}") for i in ids)

    backend = HuggingFaceLogitBackend(_TinyModel(), _TinyTok(), device="cpu")
    res = await backend.aquery_top_logprobs("Is code safe?", top_k=2)
    assert len(res) == 2


def test_hf_logit_backend_from_pretrained_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class _MockPeftModel(_TorchModule):
        @staticmethod
        def from_pretrained(model: Any, path: str) -> Any:
            del path
            return model

    class _MockPeftMod:
        PeftModel = _MockPeftModel

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    class _FakeModel(_TorchModule):
        config = type("Config", (), {"_name_or_path": "fake-model"})()

        def to(self, device: Any) -> Any:
            del device
            return self

        def eval(self) -> Any:
            return self

    class _FakeTok:
        pass

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: _FakeTok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: _FakeModel())

    backend = HuggingFaceLogitBackend.from_pretrained(
        "dummy-org/dummy-model",
        adapter_path=tmp_path / "adapter",
        device="cpu",
    )
    assert backend.model == "dummy-org/dummy-model"
    assert backend.lora_path == str(tmp_path / "adapter")


def test_train_distillation_lora_string_base_model_and_remainder_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class _MockPeftModel(_TorchModule):
        def __init__(self, base_model: Any) -> None:
            super().__init__()
            self.base_model = base_model
            self.trainable = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None) -> Any:
            del attention_mask, labels
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 32), dtype=torch.float32)
            logits[0, -1, 10] = 2.0
            logits[0, -1, 11] = -1.0
            loss = torch.tensor(0.5, requires_grad=True) * self.trainable
            return type("Out", (), {"logits": logits, "loss": loss})()

        def save_pretrained(self, save_dir: str) -> None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_config.json").write_text("{}", encoding="utf-8")

    class _MockPeftMod:
        class TaskType:
            CAUSAL_LM = "CAUSAL_LM"

        @staticmethod
        def LoraConfig(**kwargs: Any) -> Any:
            return type("LoraConfigObj", (), kwargs)()

        @staticmethod
        def get_peft_model(model: Any, config: Any) -> Any:
            del config
            return _MockPeftModel(model)

    monkeypatch.setitem(sys.modules, "peft", _MockPeftMod)

    class _BaseModel(_TorchModule):
        config = type("Config", (), {"_name_or_path": "fake-base-model"})()

        def to(self, device: Any) -> Any:
            del device
            return self

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            del add_special_tokens
            s = text.strip()
            if s == "A":
                return [10]
            if s == "B":
                return [11]
            # Return 80 tokens to trigger prompt_ids >= max_seq_len (line 570)
            return [12] * 80

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            del ids, skip_special_tokens
            return "tok"

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: _Tok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: _BaseModel())

    # 12 items -> 24 examples. batch_size=5 -> 5 batches. 5 % 2 == 1 != 0 (triggers remainder step line 628-630)
    dataset = [
        {"id": f"item_{i}", "rubric": "Is safe?", "candidate": f"print({i})", "expected": (i % 2 == 0)}
        for i in range(12)
    ]
    out_dir = tmp_path / "lora_str"
    manifest = train_distillation_lora(
        "fake-org/fake-model",
        dataset,
        out_dir,
        config=LoRADistillConfig(
            rank=4,
            lora_alpha=8,
            epochs=1,
            batch_size=5,
            grad_accum_steps=2,
            learning_rate=1e-3,
            device="cpu",
            max_seq_len=64,
        ),
    )
    assert manifest.base_model == "fake-org/fake-model"
    assert manifest.bundle_path is not None

