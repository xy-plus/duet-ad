import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app import image_optimization
from app import video_visual_reconcile as reconcile


def _sha(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> str:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


def _plan() -> dict:
    return {
        "version": 3,
        "phase": "plan",
        "segment_indices": [0],
        "eligible": True,
        "reason": None,
        "person_plans": [{
            "id": "PERSON_01",
            "source_identity": "旧人物静态外观",
            "replacement_identity": "新人物静态外观",
            "wardrobe_change": "新服装静态设计",
            "local_color_change": "人物局部固有色变化",
            "reference": {"segment_index": 0, "frame_index": 1},
            "observable_segments": [0],
        }],
        "scene_plans": [{
            "id": "SCENE_01",
            "source_scene": "旧场景静态外观",
            "replacement_scene": "新场景静态外观",
            "semantic_change": "新环境语义",
            "geometry_changes": ["新环境几何"],
            "depth_changes": ["新环境纵深"],
            "layout_changes": ["新环境布局"],
            "local_color_change": "场景局部固有色变化",
            "reference": {"segment_index": 0, "frame_index": 1},
            "segments": [0],
        }],
        "segments": [{
            "segment_index": 0,
            "persons": [{
                "id": "PERSON_01",
                "state": "replace",
                "observable_frames": [1, 2],
                "target_region": "人物完整目标域",
                "boundary": "人物可见边界",
            }],
            "scene": {
                "scene_id": "SCENE_01",
                "target_region": "场景完整目标域",
                "boundary": "场景停止边界",
                "layout_reference_frame_index": 1,
            },
            "protected_non_target_people": [],
            "protected_relations": ["可见关系冻结"],
            "frame_constraints": [
                {
                    "frame_index": 1,
                    "visible_body_parts": "帧一可见部位",
                    "pose_skeleton": "帧一姿态骨架",
                    "contact_points": "帧一接触点",
                    "occlusion_order": "帧一遮挡顺序",
                    "out_of_frame_crop": "帧一裁切",
                    "non_person_entity_ledger": {
                        "entities": [{
                            "entity_id": "ENTITY_01",
                            "description": "帧一静态实体",
                            "visibility": "full",
                        }],
                        "relations": [{
                            "subject_id": "ENTITY_01",
                            "predicate": "contacts",
                            "object_id": "PERSON_01",
                        }],
                    },
                },
                {
                    "frame_index": 2,
                    "visible_body_parts": "帧二可见部位",
                    "pose_skeleton": "帧二姿态骨架",
                    "contact_points": "帧二接触点",
                    "occlusion_order": "帧二遮挡顺序",
                    "out_of_frame_crop": "帧二裁切",
                    "non_person_entity_ledger": {
                        "entities": [{
                            "entity_id": "ENTITY_01",
                            "description": "帧二静态实体",
                            "visibility": "edge_fragment",
                        }],
                        "relations": [{
                            "subject_id": "ENTITY_01",
                            "predicate": "contacts",
                            "object_id": "PERSON_01",
                        }],
                    },
                },
            ],
            "photometric_contract": {
                "light_direction": "冻结光向",
                "light_quality": "冻结光质",
                "exposure_or_intensity": "冻结曝光",
                "wb_cct": "冻结白平衡",
                "global_contrast": "冻结对比",
                "tone_curve": "冻结曲线",
            },
        }],
    }


def _action(index: int) -> dict:
    return {
        "initial_state": f"state-{index}-before",
        "motion": f"motion-{index}",
        "result_state": f"state-{index}-after",
    }


def _camera(index: int) -> dict:
    return {
        "shot_scale": "medium",
        "angle": "eye-level",
        "movement": "locked" if index == 1 else "pan-right",
        "composition": f"composition-{index}",
        "focus": "subject-plane",
    }


def _time_base() -> dict:
    return {"numerator": 1, "denominator": 90_000}


def _source(frame_hashes: list[str]) -> dict:
    frames = [
        {
            "segment_index": 0,
            "frame_index": index,
            "source_file": f"work/source/{index:02d}.png",
            "source_frame_sha256": digest,
            "source_pts": pts,
            "source_time_base": _time_base(),
        }
        for index, (digest, pts) in enumerate(
            zip(frame_hashes, (0, 45_000), strict=True), 1
        )
    ]
    return {
        "schema": "duet.source-visual-ir",
        "version": 1,
        "phase": "source_visual",
        "frame_manifest_sha256": _sha("frame-manifest"),
        "old_visual_prompt_sha256": _sha("old-visual-prompt"),
        "frames": frames,
        "events": [
            {
                "event_index": index,
                "segment_index": 0,
                "frame_refs": [index],
                "actor_refs": ["PERSON_01"],
                "scene_ref": "SCENE_01",
                "entity_refs": [{
                    "frame_index": index, "entity_id": "ENTITY_01",
                }],
                "action": _action(index),
                "camera": _camera(index),
                "timing": {
                    "start_source_pts": frame["source_pts"],
                    "end_source_pts": frame["source_pts"],
                    "source_time_base": _time_base(),
                    "pace": "normal",
                    "transition": "start" if index == 1 else "continue",
                },
            }
            for index, frame in enumerate(frames, 1)
        ],
    }


def _output_receipt(
    *, plan_sha256: str, index: int, source_sha256: str,
    optimized_file: str, optimized_sha256: str,
) -> dict:
    return {
        "schema": "duet.image-optimization-output",
        "version": 1,
        "plan_sha256": plan_sha256,
        "segment_index": 0,
        "frame_index": index,
        "source_frame_sha256": source_sha256,
        "output": {"path": optimized_file, "sha256": optimized_sha256},
    }


def _unified(source: dict, plan: dict, verified: dict) -> dict:
    return {
        "version": 1,
        "phase": "reconcile_after_image_optimization",
        "eligible": True,
        "reason": None,
        "source_evidence_binding": {
            "frame_manifest_sha256": source["frame_manifest_sha256"],
            "old_visual_prompt_sha256": source["old_visual_prompt_sha256"],
        },
        "target_static_plan_binding": {
            "image_plan_sha256": image_optimization.plan_sha256(plan),
            "image_verification_sha256": verified["verification_sha256"],
        },
        "frame_bindings": [
            {
                "segment_index": frame["segment_index"],
                "frame_index": frame["frame_index"],
                "source_frame_sha256": frame["source_frame_sha256"],
                "source_pts": frame["source_pts"],
                "source_time_base": deepcopy(frame["source_time_base"]),
                "optimized_image_sha256": output["optimized_image_sha256"],
                "output_receipt_sha256": output["output_receipt_sha256"],
            }
            for frame, output in zip(source["frames"], verified["frames"], strict=True)
        ],
        "preserved_beats": [
            {
                "beat_index": event["event_index"],
                **{
                    key: deepcopy(event[key])
                    for key in (
                        "segment_index", "frame_refs", "actor_refs", "scene_ref",
                        "entity_refs", "action", "camera", "timing",
                    )
                },
            }
            for event in source["events"]
        ],
        "conflicts": [],
    }


def _artifacts(root: Path) -> dict:
    source_blobs = [b"source-one", b"source-two"]
    optimized_blobs = [b"optimized-one", b"optimized-two"]
    for index, data in enumerate(source_blobs, 1):
        path = root / f"work/source/{index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for index, data in enumerate(optimized_blobs, 1):
        path = root / f"work/optimized/{index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    plan = image_optimization.canonical_plan_v3(_plan(), [0], {0: 2})
    plan_digest = image_optimization.plan_sha256(plan)
    source = _source([_sha(data) for data in source_blobs])
    output_receipts = []
    verified_frames = []
    for index, (source_blob, optimized_blob) in enumerate(
        zip(source_blobs, optimized_blobs, strict=True), 1
    ):
        optimized_file = f"work/optimized/{index:02d}.png"
        receipt = _output_receipt(
            plan_sha256=plan_digest,
            index=index,
            source_sha256=_sha(source_blob),
            optimized_file=optimized_file,
            optimized_sha256=_sha(optimized_blob),
        )
        receipt_file = root / f"work/output-receipts/{index:02d}.json"
        receipt_sha256 = _write_json(receipt_file, receipt)
        output_receipts.append(receipt)
        verified_frames.append({
            "segment_index": 0,
            "frame_index": index,
            "source_frame_sha256": _sha(source_blob),
            "optimized_file": optimized_file,
            "optimized_image_sha256": _sha(optimized_blob),
            "output_receipt_file": receipt_file.relative_to(root).as_posix(),
            "output_receipt_sha256": receipt_sha256,
        })
    verified = {
        "schema": "duet.image-optimization-verified-outputs",
        "version": 1,
        "plan_sha256": plan_digest,
        "verification_sha256": _sha("image-verification"),
        "passed": True,
        "frames": verified_frames,
    }
    unified = _unified(source, plan, verified)
    paths = {
        "source_visual_ir_path": root / "work/source_visual_ir.json",
        "image_plan_path": root / "work/frozen_image_plan.json",
        "verified_output_receipt_path": root / "work/verified_outputs.json",
        "unified_visual_ir_path": root / "work/unified_visual_ir.json",
    }
    _write_json(paths["source_visual_ir_path"], source)
    _write_json(paths["image_plan_path"], plan)
    _write_json(paths["verified_output_receipt_path"], verified)
    _write_json(paths["unified_visual_ir_path"], unified)
    return {
        "plan": plan,
        "source": source,
        "verified": verified,
        "unified": unified,
        "output_receipts": output_receipts,
        "paths": paths,
    }


def _canonical(bundle: dict, unified: dict | None = None) -> dict:
    return reconcile.canonical_unified_visual_ir(
        bundle["unified"] if unified is None else unified,
        source_visual_ir=bundle["source"],
        image_plan=bundle["plan"],
        verified_output_receipt=bundle["verified"],
    )


def test_canonicalizer_accepts_only_exact_closed_world_event_projection(tmp_path):
    bundle = _artifacts(tmp_path)
    canonical = _canonical(bundle)

    assert canonical == bundle["unified"]
    assert canonical is not bundle["unified"]
    assert canonical["preserved_beats"][1]["action"] == bundle["source"][
        "events"
    ][1]["action"]
    assert canonical["frame_bindings"][1]["source_time_base"] == {
        "numerator": 1, "denominator": 90_000,
    }


@pytest.mark.parametrize("damage", ["added", "removed", "reordered"])
def test_event_add_remove_or_reorder_becomes_ineligible(tmp_path, damage):
    bundle = _artifacts(tmp_path)
    unified = deepcopy(bundle["unified"])
    beats = unified["preserved_beats"]
    if damage == "added":
        extra = deepcopy(beats[-1])
        extra["beat_index"] = 3
        beats.append(extra)
    elif damage == "removed":
        beats.pop()
    else:
        beats.reverse()

    canonical = _canonical(bundle, unified)

    assert canonical["eligible"] is False
    assert canonical["reason"] == "reconciliation_unknown"
    assert canonical["preserved_beats"] == []
    assert canonical["conflicts"][0]["code"] == "reconciliation_unknown"


def test_event_content_change_and_incomplete_id_mapping_become_ineligible(tmp_path):
    bundle = _artifacts(tmp_path)
    changed = deepcopy(bundle["unified"])
    changed["preserved_beats"][0]["action"]["motion"] = "different-motion"
    assert _canonical(bundle, changed)["reason"] == "reconciliation_unknown"

    source = deepcopy(bundle["source"])
    source["events"][0]["actor_refs"] = ["PERSON_99"]
    unmapped = _unified(source, bundle["plan"], bundle["verified"])
    canonical = reconcile.canonical_unified_visual_ir(
        unmapped,
        source_visual_ir=source,
        image_plan=bundle["plan"],
        verified_output_receipt=bundle["verified"],
    )
    assert canonical["eligible"] is False
    assert canonical["reason"] == "reconciliation_unknown"


def test_static_plan_text_in_dynamic_source_becomes_ineligible(tmp_path):
    bundle = _artifacts(tmp_path)
    source = deepcopy(bundle["source"])
    source["events"][0]["action"]["motion"] = bundle["plan"][
        "person_plans"
    ][0]["source_identity"]
    unified = _unified(source, bundle["plan"], bundle["verified"])

    canonical = reconcile.canonical_unified_visual_ir(
        unified,
        source_visual_ir=source,
        image_plan=bundle["plan"],
        verified_output_receipt=bundle["verified"],
    )

    assert canonical["eligible"] is False
    assert canonical["reason"] == "source_static_semantics_leaked"


def test_ineligible_conflicts_cannot_smuggle_static_free_text(tmp_path):
    bundle = _artifacts(tmp_path)
    failure = {
        "version": 1,
        "phase": "reconcile_after_image_optimization",
        "eligible": False,
        "reason": "source_static_semantics_leaked",
        "source_evidence_binding": None,
        "target_static_plan_binding": None,
        "frame_bindings": [],
        "preserved_beats": [],
        "conflicts": [{
            "code": "source_static_semantics_leaked",
            "segment_index": 0,
            "frame_index": 1,
            "evidence_refs": ["旧人物静态外观"],
        }],
    }

    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.canonical_unified_visual_ir(
            failure,
            source_visual_ir=bundle["source"],
            image_plan=bundle["plan"],
            verified_output_receipt=bundle["verified"],
        )
    assert caught.value.code == "unified_visual_ir_invalid"


@pytest.mark.parametrize("damage", ["plan_hash", "missing_frame", "frame_order"])
def test_hash_or_verified_frame_mapping_drift_becomes_ineligible(tmp_path, damage):
    bundle = _artifacts(tmp_path)
    unified = deepcopy(bundle["unified"])
    verified = deepcopy(bundle["verified"])
    if damage == "plan_hash":
        unified["target_static_plan_binding"]["image_plan_sha256"] = "0" * 64
    elif damage == "missing_frame":
        verified["frames"].pop()
    else:
        verified["frames"].reverse()

    canonical = reconcile.canonical_unified_visual_ir(
        unified,
        source_visual_ir=bundle["source"],
        image_plan=bundle["plan"],
        verified_output_receipt=verified,
    )

    assert canonical["eligible"] is False
    assert canonical["reason"] in {
        "receipt_binding_mismatch", "frame_mapping_missing",
    }


@pytest.mark.parametrize("damage", ["float_pts", "bad_time_base", "extra_key"])
def test_source_and_unified_schemas_reject_noncanonical_values(tmp_path, damage):
    bundle = _artifacts(tmp_path)
    source = deepcopy(bundle["source"])
    unified = deepcopy(bundle["unified"])
    if damage == "float_pts":
        source["frames"][0]["source_pts"] = 0.0
    elif damage == "bad_time_base":
        source["frames"][0]["source_time_base"] = {
            "numerator": 2, "denominator": 180_000,
        }
    else:
        unified["provider_prompt"] = "must never enter this contract"

    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.canonical_unified_visual_ir(
            unified,
            source_visual_ir=source,
            image_plan=bundle["plan"],
            verified_output_receipt=bundle["verified"],
        )
    assert caught.value.code in {
        "source_visual_ir_invalid", "unified_visual_ir_invalid",
    }


def test_freeze_binding_and_loader_reject_any_bound_byte_drift(tmp_path):
    bundle = _artifacts(tmp_path)
    frozen = reconcile.freeze(tmp_path, **bundle["paths"])
    binding = reconcile.receipt_binding(tmp_path, frozen)

    loaded = reconcile.load_bound(tmp_path, binding)
    assert loaded.unified_visual_ir_sha256 == frozen.unified_visual_ir_sha256
    assert reconcile.receipt_binding(tmp_path, loaded) == binding

    bundle["paths"]["unified_visual_ir_path"].write_bytes(
        bundle["paths"]["unified_visual_ir_path"].read_bytes() + b" "
    )
    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.load_bound(tmp_path, binding)
    assert caught.value.code == "receipt_mismatch"


def test_freeze_rejects_optimized_bytes_or_output_receipt_drift(tmp_path):
    bundle = _artifacts(tmp_path)
    (tmp_path / "work/optimized/01.png").write_bytes(b"changed-output")
    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.freeze(tmp_path, **bundle["paths"])
    assert caught.value.code == "hash_drift"


def test_freeze_rejects_rehashed_output_receipt_that_does_not_bind_plan(tmp_path):
    bundle = _artifacts(tmp_path)
    receipt_path = tmp_path / "work/output-receipts/01.json"
    forged = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged["plan_sha256"] = "0" * 64
    forged_sha = _write_json(receipt_path, forged)
    bundle["verified"]["frames"][0]["output_receipt_sha256"] = forged_sha
    bundle["unified"]["frame_bindings"][0]["output_receipt_sha256"] = forged_sha
    _write_json(
        bundle["paths"]["verified_output_receipt_path"], bundle["verified"]
    )
    _write_json(bundle["paths"]["unified_visual_ir_path"], bundle["unified"])

    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.freeze(tmp_path, **bundle["paths"])
    assert caught.value.code == "output_receipt_invalid"


def test_loader_rejects_receipt_path_escape_before_read(tmp_path):
    bundle = _artifacts(tmp_path)
    frozen = reconcile.freeze(tmp_path, **bundle["paths"])
    binding = reconcile.receipt_binding(tmp_path, frozen)
    binding["source_visual_ir"]["path"] = "../outside.json"

    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.load_bound(tmp_path, binding)
    assert caught.value.code == "receipt_invalid"

    bundle = _artifacts(tmp_path)
    (tmp_path / "work/output-receipts/01.json").write_bytes(b"{}\n")
    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.freeze(tmp_path, **bundle["paths"])
    assert caught.value.code == "hash_drift"


def test_projector_is_deterministic_and_emits_no_static_or_provider_prompt(tmp_path):
    bundle = _artifacts(tmp_path)
    frozen = reconcile.freeze(tmp_path, **bundle["paths"])

    first = reconcile.project(frozen)
    second = reconcile.project(frozen)

    assert first == second
    assert first["events"] == bundle["source"]["events"]
    assert [item["optimized_file"] for item in first["frames"]] == [
        "work/optimized/01.png", "work/optimized/02.png",
    ]
    rendered = json.dumps(first, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "旧人物静态外观", "新人物静态外观",
        "旧场景静态外观", "新场景静态外观",
        "provider_prompt", "visual_prompt", "dialogue",
    ):
        assert forbidden not in rendered


def test_projector_revalidates_frozen_paths_before_exposing_projection(tmp_path):
    bundle = _artifacts(tmp_path)
    frozen = reconcile.freeze(tmp_path, **bundle["paths"])
    (tmp_path / "work/optimized/02.png").write_bytes(b"changed-after-freeze")

    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.project(frozen)
    assert caught.value.code == "hash_drift"


def test_projector_rejects_a_canonical_ineligible_ir(tmp_path):
    bundle = _artifacts(tmp_path)
    damaged = deepcopy(bundle["unified"])
    damaged["preserved_beats"].pop()
    ineligible = _canonical(bundle, damaged)
    _write_json(bundle["paths"]["unified_visual_ir_path"], ineligible)
    frozen = reconcile.freeze(tmp_path, **bundle["paths"])

    with pytest.raises(reconcile.VideoVisualReconcileError) as caught:
        reconcile.project(frozen)
    assert caught.value.code == "reconciliation_unknown"
