import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import image_quality
from app.image_quality import (
    CodexSemanticVerifier,
    FrameMasks,
    GateResult,
    MaskArtifact,
    POC_PROFILE_V1,
    QualityProfile,
    SemanticVerdict,
    evaluate_image_quality,
    evaluate_reference_packs,
    load_frame_masks,
    mask_manifest_receipt,
    quality_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image)
    return path


def _mask(path: Path, region: tuple[slice, slice], shape=(32, 32)) -> Path:
    data = np.zeros(shape, dtype=np.uint8)
    data[region] = 255
    return _png(path, data)


def _artifact(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return {
        "path": path.name,
        "sha256": _sha(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def _person_mask_artifact(path: Path, source: dict, *, purpose: str) -> dict:
    artifact = _artifact(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    alpha = image if image.ndim == 2 else image[:, :, 3]
    nonzero = int(np.count_nonzero(alpha))
    producer = {"provider": "fake", "action": "SegmentPerson", "model": "fake-v1"}
    params = {"ImageURL": "https://private.invalid/signed"}
    request = {
        **producer,
        "purpose": purpose,
        "source_sha256": source["sha256"],
        "width": source["width"],
        "height": source["height"],
        "frame_pts": "1.25",
        "params": params,
        "cache_version": "mask-cache-v1",
    }
    artifact["producer_receipt"] = {
        "schema": "duet.image-mask-producer", "version": 1,
        "producer": producer, "purpose": purpose,
        "source": {**source, "frame_pts": "1.25"},
        "request_sha256": _canonical_sha(request),
        "params": params, "cache_version": "mask-cache-v1",
        "mask": {
            **artifact, "size": path.stat().st_size, "mime_type": "image/png",
            "alpha_nonzero_pixels": nonzero,
            "alpha_transparent_pixels": int(alpha.size) - nonzero,
        },
    }
    return artifact


def _scene_mask_artifact(path: Path, source: dict, plan_sha256: str) -> dict:
    artifact = _artifact(path)
    receipt = {
        "schema": "duet.scene-mask.producer", "version": 1,
        "backend": "remote_gpu", "model": "sam2.1-base-plus",
        "model_version": "checkpoint-sha256:abc", "endpoint_identity": "worker:v1",
        "plan_sha256": plan_sha256, "scene_id": "SCENE_01",
        "component_id": "COMPONENT_01", "shot_id": "SHOT_01", "frame_id": "FRAME_01",
        "frame_sha256": source["sha256"], "reference_frame_id": "FRAME_01",
        "request_sha256": "1" * 64, "propagation_job_sha256": "2" * 64,
        "propagation_scope": "hard_cut_shot_only", "membership_engine": "sam2",
        "edge_refiner": "birefnet", "edge_refinement_scope": "sam2_uncertain_edges_only",
        "fallback": "none",
    }
    return {
        "purpose": "scene_component", "channel": "grayscale_alpha",
        "component_id": "COMPONENT_01", "shot_id": "SHOT_01", "frame_id": "FRAME_01",
        **artifact, "byte_size": path.stat().st_size, "producer_receipt": receipt,
    }


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _plan() -> dict:
    return {
        "version": 2,
        "phase": "plan",
        "eligible": True,
        "segment_indices": [1],
        "person_plans": [{"id": "PERSON_01"}],
        "scene_plans": [{"id": "SCENE_01", "segments": [1]}],
        "segments": [{
            "segment_index": 1,
            "persons": [{
                "id": "PERSON_01",
                "state": "replace",
                "observable_frames": [1],
            }],
            "scene": {"scene_id": "SCENE_01"},
        }],
    }


def _profile(**changes) -> QualityProfile:
    values = {
        **POC_PROFILE_V1.thresholds_dict(),
        "max_global_exposure_delta_l": 100.0,
        "max_protected_exposure_delta_l": 100.0,
        "max_global_white_point_delta_e": 100.0,
        "max_protected_white_point_delta_e": 100.0,
        "max_global_contrast_relative_delta": 10.0,
        "max_protected_contrast_relative_delta": 10.0,
        "max_global_cct_proxy_delta": 10.0,
        "max_protected_cct_proxy_delta": 10.0,
        "min_person_local_delta_e": 0.1,
        "min_scene_local_delta_e": 0.1,
        "min_scene_edge_change_ratio": 0.0,
        "max_scene_edge_change_ratio": 1.0,
        "min_protected_edge_iou": 0.0,
        "max_composition_centroid_shift": 1.0,
        "min_mask_pixels": 4,
        **changes,
    }
    return QualityProfile(
        name="test",
        version="test-v1",
        calibration="test_only",
        thresholds=values,
    )


class _Semantic:
    def __init__(self, status="pass", code=None):
        self.status = status
        self.code = code

    def verify(self, plan, source_frames, output_frames, *, deterministic_metrics):
        assert "gates" in deterministic_metrics
        return SemanticVerdict(
            status=self.status,
            code=self.code,
            checks={"project": self.status},
            verdict_sha256="a" * 64,
        )


def _frame_files(tmp_path: Path):
    source = np.full((32, 32, 3), 90, dtype=np.uint8)
    output = source.copy()
    source[4:12, 4:12] = (30, 70, 150)
    output[4:12, 4:12] = (130, 50, 30)
    source[14:28, 2:18] = (70, 90, 110)
    output[14:28, 2:18] = (30, 130, 60)
    source[16:26:3, 3:17] = (180, 180, 180)
    output[15:27:4, 3:17] = (20, 20, 20)
    source_path = _png(tmp_path / "source.png", source)
    output_path = _png(tmp_path / "output.png", output)
    person = _mask(tmp_path / "person.png", (slice(4, 12), slice(4, 12)))
    scene = _mask(tmp_path / "scene.png", (slice(14, 28), slice(2, 18)))
    protected = np.full((32, 32), 255, dtype=np.uint8)
    protected[4:12, 4:12] = 0
    protected[14:28, 2:18] = 0
    protected_path = _png(tmp_path / "protected.png", protected)
    masks = FrameMasks(
        version=1,
        segment_index=1,
        frame_index=1,
        persons={"PERSON_01": MaskArtifact.from_path(person, person.name)},
        scene=MaskArtifact.from_path(scene, scene.name),
        protected_non_target=MaskArtifact.from_path(protected_path, protected_path.name),
        producer_receipt_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )
    return source_path, output_path, masks


def test_all_gates_and_semantics_must_pass(tmp_path):
    source, output, masks = _frame_files(tmp_path)
    receipt = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=[masks],
        profile=_profile(), semantic_verifier=_Semantic(),
    )
    assert receipt.publishable is True
    assert receipt.status == "pass"
    assert receipt.provider_retry_allowed is False
    assert receipt.control_mode == "pixel_masks"
    assert all(item.status == "pass" for item in receipt.gates)
    payload = receipt.to_dict()
    assert payload["sha256"] == _canonical_sha({k: v for k, v in payload.items() if k != "sha256"})


def test_missing_masks_is_unknown_and_never_publishable(tmp_path):
    source, output, _ = _frame_files(tmp_path)
    receipt = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=None,
        profile=_profile(), semantic_verifier=_Semantic(),
    )
    assert receipt.status == "unknown"
    assert receipt.publishable is False
    assert receipt.control_mode == "soft_control"
    assert "input_masks_missing" in {item.code for item in receipt.gates}


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        ("fail", "semantic_source_identity_residual", "fail"),
        ("unknown", "semantic_person_continuity_unknown", "unknown"),
    ],
)
def test_semantic_fail_or_unknown_cannot_publish(tmp_path, status, code, expected):
    source, output, masks = _frame_files(tmp_path)
    receipt = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=[masks],
        profile=_profile(), semantic_verifier=_Semantic(status, code),
    )
    assert receipt.status == expected
    assert receipt.publishable is False
    assert receipt.provider_retry_allowed is False
    assert receipt.semantic.code == code


def test_dimension_change_is_a_stable_failure(tmp_path):
    source, _, masks = _frame_files(tmp_path)
    output = _png(tmp_path / "wrong-size.png", np.zeros((16, 16, 3), dtype=np.uint8))
    receipt = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=[masks],
        profile=_profile(), semantic_verifier=_Semantic(),
    )
    assert receipt.publishable is False
    assert receipt.status == "fail"
    assert "dimension_mismatch" in receipt.failure_codes


def test_global_exposure_threshold_is_profile_driven(tmp_path):
    source = _png(tmp_path / "dark.png", np.full((32, 32, 3), 10, np.uint8))
    output = _png(tmp_path / "bright.png", np.full((32, 32, 3), 240, np.uint8))
    receipt = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=None,
        profile=_profile(max_global_exposure_delta_l=1.0),
        semantic_verifier=_Semantic(),
    )
    assert "global_exposure_drift" in receipt.failure_codes


def test_local_color_gate_rejects_noop_edit(tmp_path):
    source, _, masks = _frame_files(tmp_path)
    receipt = evaluate_image_quality(
        _plan(), [source], [source], frame_masks=[masks],
        profile=_profile(min_person_local_delta_e=2.0, min_scene_local_delta_e=2.0),
        semantic_verifier=_Semantic(),
    )
    assert "scene_local_color_change_too_small" in receipt.failure_codes


def test_scene_edge_and_protected_structure_are_independent_gates(tmp_path):
    source, _, masks = _frame_files(tmp_path)
    original = cv2.imread(str(source))
    scene_mask = cv2.imread(str(masks.scene.path), cv2.IMREAD_GRAYSCALE) > 0
    person_mask = cv2.imread(
        str(masks.persons["PERSON_01"].path), cv2.IMREAD_GRAYSCALE
    ) > 0
    recolored = original.copy()
    recolored[scene_mask | person_mask] = 255 - recolored[scene_mask | person_mask]
    recolored_path = _png(tmp_path / "recolored.png", recolored)
    scene_receipt = evaluate_image_quality(
        _plan(), [source], [recolored_path], frame_masks=[masks],
        profile=_profile(min_scene_edge_change_ratio=0.99),
        semantic_verifier=_Semantic(),
    )
    assert "scene_edge_change_too_small" in scene_receipt.failure_codes

    damaged = original.copy()
    protected = cv2.imread(
        str(masks.protected_non_target.path), cv2.IMREAD_GRAYSCALE
    ) > 0
    yy, xx = np.indices(protected.shape)
    damaged[protected] = np.where(((xx[protected] + yy[protected]) % 2)[:, None], 255, 0)
    damaged[scene_mask | person_mask] = 255 - damaged[scene_mask | person_mask]
    damaged_path = _png(tmp_path / "damaged.png", damaged)
    protected_receipt = evaluate_image_quality(
        _plan(), [source], [damaged_path], frame_masks=[masks],
        profile=_profile(min_protected_edge_iou=0.9),
        semantic_verifier=_Semantic(),
    )
    assert "protected_structure_drift" in protected_receipt.failure_codes


def test_custom_gate_is_pluggable_and_part_of_publish_barrier(tmp_path):
    source, output, masks = _frame_files(tmp_path)

    class Reject:
        name = "remote_depth"
        version = "remote-depth-v1"

        def evaluate(self, context, profile):
            return GateResult(self.name, self.version, "fail", "scene_depth_change_missing", {})

    receipt = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=[masks],
        profile=_profile(), semantic_verifier=_Semantic(), gates=[Reject()],
    )
    assert receipt.publishable is False
    assert "scene_depth_change_missing" in receipt.failure_codes


def test_manifest_getter_and_loader_verify_binding_and_files(tmp_path):
    source, _, _ = _frame_files(tmp_path)
    for name in ("person.png", "protected.png"):
        alpha = cv2.imread(str(tmp_path / name), cv2.IMREAD_GRAYSCALE)
        bgra = np.zeros((32, 32, 4), dtype=np.uint8)
        bgra[:, :, :3] = 40
        bgra[:, :, 3] = alpha
        _png(tmp_path / name, bgra)
    inventory = [{
        "segment_index": 1,
        "frame_index": 1,
        "source": {
            "path": source.name,
            "sha256": _sha(source),
            "width": 32,
            "height": 32,
        },
        "person_ids": ["PERSON_01"],
    }]
    plan_sha = "d" * 64
    frame = {
        "segment_index": 1,
        "frame_index": 1,
        "source": inventory[0]["source"],
        "persons": [{
            "person_id": "PERSON_01",
            **_person_mask_artifact(tmp_path / "person.png", inventory[0]["source"], purpose="person"),
        }],
        "scene": _scene_mask_artifact(tmp_path / "scene.png", inventory[0]["source"], plan_sha),
        "protected_non_target": _person_mask_artifact(
            tmp_path / "protected.png", inventory[0]["source"],
            purpose="protected_non_target_people",
        ),
    }
    manifest = {
        "schema": "duet.image_edit_masks",
        "version": 1,
        "plan_sha256": plan_sha,
        "frames": [frame],
    }
    manifest["sha256"] = _canonical_sha(manifest)
    meta = {"_image_edit_masks": manifest}

    canonical = mask_manifest_receipt(
        meta, plan_sha256=plan_sha, frame_inventory=inventory
    )
    assert canonical == manifest
    loaded = load_frame_masks(tmp_path, canonical, inventory)
    assert loaded[0].persons["PERSON_01"].sha256 == _sha(tmp_path / "person.png")
    assert loaded[0].manifest_sha256 == manifest["sha256"]

    tampered = json.loads(json.dumps(manifest))
    tampered["frames"][0]["persons"][0]["producer_receipt"]["purpose"] = "protected_non_target_people"
    tampered["sha256"] = _canonical_sha({key: value for key, value in tampered.items() if key != "sha256"})
    assert mask_manifest_receipt(
        {"_image_edit_masks": tampered}, plan_sha256=plan_sha,
        frame_inventory=inventory,
    ) is None

    tampered_scene = json.loads(json.dumps(manifest))
    tampered_scene["frames"][0]["scene"]["producer_receipt"]["membership_engine"] = "bbox"
    tampered_scene["sha256"] = _canonical_sha({
        key: value for key, value in tampered_scene.items() if key != "sha256"
    })
    assert mask_manifest_receipt(
        {"_image_edit_masks": tampered_scene}, plan_sha256=plan_sha,
        frame_inventory=inventory,
    ) is None


def test_manifest_rejects_escape_tamper_and_whole_frame_mask(tmp_path):
    source, _, _ = _frame_files(tmp_path)
    inventory = [{
        "segment_index": 1,
        "frame_index": 1,
        "source": {"path": source.name, "sha256": _sha(source), "width": 32, "height": 32},
        "person_ids": ["PERSON_01"],
    }]
    whole = _png(tmp_path / "whole.png", np.full((32, 32), 255, np.uint8))
    plan_sha = "e" * 64

    def manifest(person_path: str, person_sha: str):
        person = _person_mask_artifact(
            whole if person_path == whole.name else tmp_path / "person.png",
            inventory[0]["source"], purpose="person",
        )
        person["path"] = person_path
        person["sha256"] = person_sha
        person["producer_receipt"]["mask"]["path"] = person_path
        person["producer_receipt"]["mask"]["sha256"] = person_sha
        raw = {
            "schema": "duet.image_edit_masks", "version": 1, "plan_sha256": plan_sha,
            "frames": [{
                "segment_index": 1, "frame_index": 1, "source": inventory[0]["source"],
                "persons": [{
                    "person_id": "PERSON_01",
                    **person,
                }],
                "scene": _scene_mask_artifact(
                    tmp_path / "scene.png", inventory[0]["source"], plan_sha
                ),
                "protected_non_target": _person_mask_artifact(
                    tmp_path / "protected.png", inventory[0]["source"],
                    purpose="protected_non_target_people",
                ),
            }],
        }
        raw["sha256"] = _canonical_sha(raw)
        return raw

    with pytest.raises(ValueError, match="project-relative"):
        load_frame_masks(tmp_path, manifest("../outside.png", "0" * 64), inventory)
    with pytest.raises(ValueError, match="invalid mask manifest|whole-frame"):
        load_frame_masks(tmp_path, manifest(whole.name, _sha(whole)), inventory)


def test_codex_semantic_verifier_stages_same_skill_and_parses_exact_v2(tmp_path):
    source, output, _ = _frame_files(tmp_path)
    skill = tmp_path / "SKILL.md"
    skill.write_text("name: image-postprocess\n", encoding="utf-8")
    plan = _plan()
    plan_sha = image_quality.canonical_sha256(plan)

    class Runner:
        def run_isolated(self, workdir, prompt, *, session_dir):
            request = json.loads((workdir / "work" / "request.json").read_text())
            assert request["phase"] == "verify"
            assert (workdir / "SKILL.md").read_text() == skill.read_text()
            assert json.loads((workdir / "work" / "frozen_plan.json").read_text())["sha256"] == plan_sha
            assert (workdir / "work" / "segments" / "1" / "source" / "01.png").is_file()
            assert (workdir / "work" / "segments" / "1" / "output" / "01.png").is_file()
            verdict = {
                "version": 2, "phase": "verify", "plan_sha256": plan_sha,
                "segment_indices": [1], "passed": True, "reason": None,
                "segments": [{
                    "segment_index": 1, "passed": True,
                    "person_checks": [{
                        "person_id": "PERSON_01",
                        "identity_changed": {"status": "pass", "evidence": "different"},
                        "source_identity_absent": {"status": "pass", "evidence": "absent"},
                        "local_color_change": {"status": "pass", "evidence": "different"},
                    }],
                    "scene_checks": {
                        "semantic_change": {"status": "pass", "evidence": "semantic changed"},
                        "geometry_change": {"status": "pass", "evidence": "shape changed"},
                        "depth_change": {"status": "pass", "evidence": "depth changed"},
                        "layout_change": {"status": "pass", "evidence": "layout changed"},
                        "local_color_change": {"status": "pass", "evidence": "different"},
                    },
                    "invariants": {
                        "lighting_preservation": {"status": "pass", "evidence": "same global light"},
                        "interaction_preservation": {"status": "pass", "evidence": "same relation"},
                        "cross_frame_continuity": {"status": "pass", "evidence": "continuous"},
                    },
                }],
                "project_checks": {
                    key: {"status": "pass", "evidence": "ok"}
                    for key in (
                        "narrative_person_completeness", "no_identity_swap",
                        "no_unplanned_person", "person_identity_continuity", "scene_continuity",
                    )
                },
            }
            (workdir / "work" / "image_verification.json").write_text(json.dumps(verdict))

    result = CodexSemanticVerifier(
        Runner(), skill_path=skill, session_dir=tmp_path
    ).verify(plan, [source], [output], deterministic_metrics={"gates": []})
    assert result.status == "pass"
    assert result.code is None
    assert "project.no_identity_swap" in result.checks


def test_codex_semantic_verifier_maps_unknown_to_stable_code(tmp_path):
    source, output, _ = _frame_files(tmp_path)
    skill = tmp_path / "SKILL.md"
    skill.write_text("name: image-postprocess\n", encoding="utf-8")

    class Runner:
        def run_isolated(self, workdir, prompt, *, session_dir):
            plan_sha = json.loads((workdir / "work" / "frozen_plan.json").read_text())["sha256"]
            # Structurally invalid/incomplete model output is unknown, never success.
            (workdir / "work" / "image_verification.json").write_text(json.dumps({
                "version": 2, "phase": "verify", "plan_sha256": plan_sha,
                "segment_indices": [1], "passed": True,
            }))

    result = CodexSemanticVerifier(
        Runner(), skill_path=skill, session_dir=tmp_path
    ).verify(_plan(), [source], [output], deterministic_metrics={"gates": []})
    assert result.status == "unknown"
    assert result.code == "semantic_verdict_invalid"


def test_profile_is_mandatory(tmp_path):
    source, output, masks = _frame_files(tmp_path)
    with pytest.raises(TypeError):
        evaluate_image_quality(  # type: ignore[call-arg]
            _plan(), [source], [output], frame_masks=[masks], semantic_verifier=_Semantic()
        )


def test_durable_quality_receipt_is_revalidated_against_frames(tmp_path):
    source, output, masks = _frame_files(tmp_path)
    path = tmp_path / "quality.json"
    created = evaluate_image_quality(
        _plan(), [source], [output], frame_masks=[masks],
        profile=_profile(), semantic_verifier=_Semantic(), receipt_path=path,
    )
    loaded = quality_receipt(
        path,
        plan_sha256=created.plan_sha256,
        mask_manifest_sha256=created.mask_manifest_sha256,
        source_frames=[source],
        output_frames=[output],
    )
    assert loaded is not None and loaded["publishable"] is True
    _png(output, np.zeros((32, 32, 3), dtype=np.uint8))
    assert quality_receipt(
        path,
        plan_sha256=created.plan_sha256,
        mask_manifest_sha256=created.mask_manifest_sha256,
        source_frames=[source],
        output_frames=[output],
    ) is None


class _PackSemantic:
    def verify_reference_packs(
        self, plan, source_slots, generated_packs, *, deterministic_metrics
    ):
        checks = {}
        for target in ("identity_changed", "source_identity_absent", "multi_view_consistency", "local_color_change"):
            checks[f"person.PERSON_01.{target}"] = "pass"
        for target in ("semantic_change", "geometry_change", "depth_change", "layout_change", "local_color_change"):
            checks[f"scene.SCENE_01.{target}"] = "pass"
        for target in (
            "global_light_direction_preservation", "global_exposure_preservation",
            "global_wb_cct_preservation", "global_tone_curve_preservation",
        ):
            checks[f"project.{target}"] = "pass"
        return SemanticVerdict("pass", None, checks, "f" * 64)


def _mask_manifest_for_plan(plan):
    manifest = {
        "schema": "duet.image_edit_masks",
        "version": 1,
        "plan_sha256": image_quality.canonical_sha256(plan),
        "frames": [{"bound": True}],
    }
    manifest["sha256"] = _canonical_sha(manifest)
    return manifest


def test_reference_packs_gate_people_and_scene_before_frame_posts(tmp_path):
    source, output, _ = _frame_files(tmp_path)
    output_2 = tmp_path / "output-2.png"
    changed = cv2.imread(str(output))
    changed[5, 5] = (1, 2, 3)
    _png(output_2, changed)
    plan = _plan()
    receipt = evaluate_reference_packs(
        plan,
        {"PERSON_01": [source], "SCENE_01": [source]},
        {"PERSON_01": [output, output_2], "SCENE_01": [output]},
        mask_manifest=_mask_manifest_for_plan(plan),
        profile=_profile(min_person_local_delta_e=0.0, min_scene_local_delta_e=0.0),
        semantic_verifier=_PackSemantic(),
    )
    assert receipt.publishable is True
    assert receipt.control_mode == "reference_packs"


def test_reference_pack_missing_multiview_or_semantics_cannot_publish(tmp_path):
    source, output, _ = _frame_files(tmp_path)
    plan = _plan()
    receipt = evaluate_reference_packs(
        plan,
        {"PERSON_01": [source], "SCENE_01": [source]},
        {"PERSON_01": [output], "SCENE_01": [output]},
        mask_manifest=_mask_manifest_for_plan(plan),
        profile=_profile(),
        semantic_verifier=None,
    )
    assert receipt.publishable is False
    assert "reference_person_multiview_missing" in receipt.failure_codes
    assert "semantic_reference_pack_verifier_missing" in receipt.failure_codes
