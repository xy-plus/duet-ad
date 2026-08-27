import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import scene_masks


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _plan(**changes) -> scene_masks.SceneMaskPlan:
    plan = scene_masks.SceneMaskPlan(
        plan_sha256=_sha("frozen-plan"),
        scene_id="scene-7",
        components=("wall",),
        frames=(
            scene_masks.Frame("f1", "work/frames/f1.png", _sha("f1"), 4, 3, 0),
            scene_masks.Frame("f2", "work/frames/f2.png", _sha("f2"), 4, 3, 1),
            scene_masks.Frame("f3", "work/frames/f3.png", _sha("f3"), 4, 3, 2),
            scene_masks.Frame("f4", "work/frames/f4.png", _sha("f4"), 4, 3, 3),
        ),
        hard_cut_chain=(
            scene_masks.HardCutShot("shot-a", ("f1", "f2")),
            scene_masks.HardCutShot("shot-b", ("f3", "f4")),
        ),
        references=(
            scene_masks.ReferenceFrame("wall", "shot-a", "f1"),
            scene_masks.ReferenceFrame("wall", "shot-b", "f3"),
        ),
        box_prompts=(
            scene_masks.BoxPrompt("wall", "f1", 0, 0, 2, 2),
            scene_masks.BoxPrompt("wall", "f3", 1, 0, 3, 2),
        ),
        point_prompts=(
            scene_masks.PointPrompt("wall", "f1", 1, 1, True),
            scene_masks.PointPrompt("wall", "f3", 2, 1, True),
        ),
        people_count=0,
        people_protection_known=True,
        protection_masks=(),
        model="sam2.1-base-plus",
        model_version="checkpoint-sha256:abc123",
        endpoint_identity="gpu-mask-worker-a:v1",
    )
    return replace(plan, **changes)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


def _write_mask(path: Path, width: int = 4, height: int = 3, *, fill: str = "partial"):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((height, width), dtype=np.uint8)
    if fill == "partial":
        pixels[0, 0] = 255
    elif fill == "whole":
        pixels[:] = 255
    assert cv2.imwrite(str(path), pixels)


def _producer_receipt(plan, payload, *, component_id, shot_id, frame_id, output):
    job = next(
        item
        for item in payload["propagation_jobs"]
        if item["component_id"] == component_id and item["shot_id"] == shot_id
    )
    frame = next(item for item in payload["frames"] if item["frame_id"] == frame_id)
    return {
        "schema": "duet.scene-mask.producer",
        "version": 1,
        "backend": plan.backend,
        "model": plan.model,
        "model_version": plan.model_version,
        "endpoint_identity": plan.endpoint_identity,
        "plan_sha256": plan.plan_sha256,
        "scene_id": plan.scene_id,
        "component_id": component_id,
        "shot_id": shot_id,
        "frame_id": frame_id,
        "frame_sha256": frame["sha256"],
        "reference_frame_id": job["reference_frame_id"],
        "request_sha256": scene_masks.canonical_json_sha256(payload),
        "propagation_job_sha256": scene_masks.canonical_json_sha256(job),
        "propagation_scope": "hard_cut_shot_only",
        "membership_engine": "sam2",
        "edge_refiner": "birefnet",
        "edge_refinement_scope": "sam2_uncertain_edges_only",
        "fallback": "none",
        "output": dict(output),
    }


def _mask_item(root, plan, payload, *, component_id, shot_id, frame_id, fill="partial"):
    relative = f"work/scene-masks/{component_id}/{frame_id}.png"
    path = root / relative
    _write_mask(path, fill=fill)
    raw = path.read_bytes()
    output = {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
        "width": 4,
        "height": 3,
    }
    return {
        "purpose": "scene_component",
        "channel": "grayscale_alpha",
        "component_id": component_id,
        "shot_id": shot_id,
        "frame_id": frame_id,
        **output,
        "producer_receipt": _producer_receipt(
            plan,
            payload,
            component_id=component_id,
            shot_id=shot_id,
            frame_id=frame_id,
            output=output,
        ),
    }


def _all_mask_items(root, plan, payload):
    shot_for_frame = {
        frame_id: shot.shot_id
        for shot in plan.hard_cut_chain
        for frame_id in shot.frame_ids
    }
    return [
        _mask_item(
            root,
            plan,
            payload,
            component_id=component,
            shot_id=shot_for_frame[frame.frame_id],
            frame_id=frame.frame_id,
        )
        for component in plan.components
        for frame in plan.frames
    ]


def _advance(root, receipt, plan, handler):
    with _client(handler) as client:
        return scene_masks.advance(
            plan,
            endpoint="https://worker.example/api",
            output_root=root,
            receipt_path=receipt,
            client=client,
            timeout_s=3,
        )


def test_request_freezes_complete_plan_and_persists_receipt_before_post(tmp_path):
    plan = _plan()
    receipt = tmp_path / "work/scene-masks/receipt.json"
    calls = []

    def handler(request: httpx.Request):
        calls.append(request.method)
        if request.method == "POST":
            landed = json.loads(receipt.read_text(encoding="utf-8"))
            assert landed["status"] == "submitting"
            payload = json.loads(request.content)
            assert payload["schema"] == "duet.scene-mask.request"
            assert payload["backend"] == scene_masks.DEFAULT_BACKEND == "sam2_birefnet"
            assert payload["plan_sha256"] == plan.plan_sha256
            assert payload["scene_id"] == "scene-7"
            assert payload["model"] == {
                "name": "sam2.1-base-plus",
                "version": "checkpoint-sha256:abc123",
            }
            assert payload["endpoint_identity"] == "gpu-mask-worker-a:v1"
            assert payload["frames"] == [
                {
                    "frame_id": frame.frame_id,
                    "path": frame.path,
                    "sha256": frame.sha256,
                    "width": frame.width,
                    "height": frame.height,
                    "pts": frame.pts,
                }
                for frame in plan.frames
            ]
            assert [job["frame_ids"] for job in payload["propagation_jobs"]] == [
                ["f1", "f2"],
                ["f3", "f4"],
            ]
            assert payload["contracts"] == {
                "propagation_scope": "hard_cut_shot_only",
                "membership_engine": "sam2",
                "edge_refinement": "birefnet_uncertain_edges_only",
                "fallback": "none",
            }
            return httpx.Response(202, json={"task_id": "task-123"})
        assert request.url.path == "/api/tasks/task-123"
        return httpx.Response(200, json={"status": "running"})

    result = _advance(tmp_path, receipt, plan, handler)

    assert result == scene_masks.SceneMaskResult(status="running", task_id="task-123")
    assert calls == ["POST", "GET"]
    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["status"] == "running"
    assert state["task_id"] == "task-123"
    assert state["request_sha256"] == scene_masks.canonical_json_sha256(state["request"])


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"box_prompts": (), "point_prompts": ()}, "prompt_missing"),
        ({"people_protection_known": False}, "people_protection_unknown"),
        ({"people_count": 2, "protection_masks": ()}, "people_protection_unknown"),
    ],
)
def test_unsafe_or_incomplete_input_fails_closed_before_receipt_or_network(
    tmp_path, change, code
):
    plan = _plan(**change)
    receipt = tmp_path / "work/scene-masks/receipt.json"

    def handler(_request):
        raise AssertionError("invalid plan must not reach worker")

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, receipt, plan, handler)

    assert caught.value.code == code
    assert not receipt.exists()


def test_post_timeout_becomes_submission_unknown_and_is_never_resent(tmp_path):
    plan = _plan()
    receipt = tmp_path / "work/scene-masks/receipt.json"
    calls = []

    def timeout_handler(request):
        calls.append(request.method)
        raise httpx.ReadTimeout("contains-sensitive-upstream-detail", request=request)

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, receipt, plan, timeout_handler)
    assert caught.value.code == "submission_unknown"
    assert calls == ["POST"]
    state = json.loads(receipt.read_text(encoding="utf-8"))
    assert state["status"] == "submission_unknown"
    assert "contains-sensitive-upstream-detail" not in receipt.read_text(encoding="utf-8")

    def must_not_send(_request):
        raise AssertionError("submission_unknown must never send again")

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, receipt, plan, must_not_send)
    assert caught.value.code == "submission_unknown"


def test_known_task_recovers_with_get_only_after_query_failure(tmp_path):
    plan = _plan()
    receipt = tmp_path / "work/scene-masks/receipt.json"
    first_calls = []

    def first_handler(request):
        first_calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "known-task"})
        raise httpx.ReadTimeout("query failed", request=request)

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, receipt, plan, first_handler)
    assert caught.value.code == "worker_query_failed"
    assert first_calls == ["POST", "GET"]
    assert json.loads(receipt.read_text())["task_id"] == "known-task"

    resumed_calls = []

    def resumed_handler(request):
        resumed_calls.append(request.method)
        assert request.url.path == "/api/tasks/known-task"
        return httpx.Response(200, json={"status": "running"})

    assert _advance(tmp_path, receipt, plan, resumed_handler).status == "running"
    assert resumed_calls == ["GET"]


def test_concurrent_callers_share_one_receipt_and_cross_post_only_once(tmp_path):
    plan = _plan()
    receipt = tmp_path / "work/scene-masks/receipt.json"
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "single-task"})
        return httpx.Response(200, json={"status": "running"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_advance, tmp_path, receipt, plan, handler)
            for _ in range(2)
        ]
        results = [future.result() for future in futures]

    assert results == [
        scene_masks.SceneMaskResult(status="running", task_id="single-task"),
        scene_masks.SceneMaskResult(status="running", task_id="single-task"),
    ]
    assert calls.count("POST") == 1
    assert calls.count("GET") == 2


def test_success_validates_complete_per_frame_masks_and_producer_receipts(tmp_path):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    masks = _all_mask_items(tmp_path, plan, payload)
    receipt = tmp_path / "work/scene-masks/receipt.json"

    def handler(request):
        if request.method == "POST":
            assert json.loads(request.content) == payload
            return httpx.Response(202, json={"task_id": "done-task"})
        return httpx.Response(
            200,
            json={"status": "succeeded", "result": {"masks": masks}},
        )

    result = _advance(tmp_path, receipt, plan, handler)

    assert result.status == "succeeded"
    assert result.task_id == "done-task"
    assert {(item.purpose, item.channel) for item in result.masks} == {
        ("scene_component", "grayscale_alpha")
    }
    for item in result.masks:
        assert item.producer_receipt["output"] == {
            "path": item.path,
            "sha256": item.sha256,
            "byte_size": item.byte_size,
            "width": item.width,
            "height": item.height,
        }
    assert [item.path for item in result.masks] == [item["path"] for item in masks]
    state = json.loads(receipt.read_text())
    assert state["status"] == "succeeded"
    assert state["masks"] == masks

    def no_network(_request):
        raise AssertionError("validated local result should be reusable")

    assert _advance(tmp_path, receipt, plan, no_network) == result


def test_public_consumer_loader_returns_canonical_item_and_immutable_packed_mask(
    tmp_path,
):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    raw = _all_mask_items(tmp_path, plan, payload)[0]

    loaded = scene_masks.load_validated_scene_mask(
        tmp_path,
        raw,
        expected_plan_sha256=plan.plan_sha256,
        expected_source_sha256=plan.frames[0].sha256,
    )

    assert isinstance(loaded, scene_masks.ValidatedSceneMask)
    assert loaded.item == scene_masks.SceneMaskItem(
        purpose="scene_component",
        channel="grayscale_alpha",
        component_id="wall",
        shot_id="shot-a",
        frame_id="f1",
        path=raw["path"],
        sha256=raw["sha256"],
        byte_size=raw["byte_size"],
        width=4,
        height=3,
        producer_receipt=raw["producer_receipt"],
    )
    assert loaded.pixel_count == 12
    assert loaded.active_pixels == 1
    unpacked = np.unpackbits(
        np.frombuffer(loaded.packed_mask, dtype=np.uint8),
        bitorder="little",
        count=loaded.pixel_count,
    )
    assert unpacked.tolist() == [1] + [0] * 11
    with pytest.raises(FrozenInstanceError):
        loaded.active_pixels = 2
    with pytest.raises(TypeError):
        loaded.item.producer_receipt["output"]["sha256"] = "0" * 64
    assert scene_masks.load_validated_scene_mask(
        tmp_path,
        loaded.item,
        expected_plan_sha256=plan.plan_sha256,
        expected_source_sha256=plan.frames[0].sha256,
    ) == loaded


@pytest.mark.parametrize("expected_field", ["plan", "source"])
def test_public_consumer_loader_rejects_wrong_expected_provenance(
    tmp_path, expected_field
):
    plan = _plan()
    raw = _all_mask_items(tmp_path, plan, scene_masks.request_payload(plan))[0]
    expected_plan = plan.plan_sha256
    expected_source = plan.frames[0].sha256
    if expected_field == "plan":
        expected_plan = "0" * 64
    else:
        expected_source = "0" * 64

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        scene_masks.load_validated_scene_mask(
            tmp_path,
            raw,
            expected_plan_sha256=expected_plan,
            expected_source_sha256=expected_source,
        )
    assert caught.value.code == "worker_output_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [("component_id", "floor"), ("shot_id", "shot-b"), ("frame_id", "f2")],
)
def test_public_consumer_loader_binds_outer_component_shot_and_frame_to_receipt(
    tmp_path, field, value
):
    plan = _plan()
    raw = _all_mask_items(tmp_path, plan, scene_masks.request_payload(plan))[0]
    raw[field] = value

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        scene_masks.load_validated_scene_mask(
            tmp_path,
            raw,
            expected_plan_sha256=plan.plan_sha256,
            expected_source_sha256=plan.frames[0].sha256,
        )
    assert caught.value.code == "worker_output_invalid"


def test_public_consumer_loader_reuses_nofollow_png_and_output_receipt_guards(tmp_path):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    attacks = []

    tampered = _all_mask_items(tmp_path, plan, payload)[0]
    tampered["producer_receipt"]["output"]["byte_size"] += 1
    attacks.append(tampered)

    whole = _mask_item(
        tmp_path,
        plan,
        payload,
        component_id="wall",
        shot_id="shot-a",
        frame_id="f1",
        fill="whole",
    )
    whole_path = tmp_path / "work/scene-masks/wall/f1-whole.png"
    _write_mask(whole_path, fill="whole")
    whole_bytes = whole_path.read_bytes()
    whole.update(
        path="work/scene-masks/wall/f1-whole.png",
        sha256=hashlib.sha256(whole_bytes).hexdigest(),
        byte_size=len(whole_bytes),
    )
    whole["producer_receipt"]["output"] = {
        key: whole[key]
        for key in ("path", "sha256", "byte_size", "width", "height")
    }
    attacks.append(whole)

    outside = tmp_path.parent / f"consumer-outside-{tmp_path.name}.png"
    _write_mask(outside)
    symlinked = _all_mask_items(tmp_path, plan, payload)[0]
    link = tmp_path / "work/scene-masks/wall/consumer-link.png"
    link.symlink_to(outside)
    external = outside.read_bytes()
    symlinked["path"] = "work/scene-masks/wall/consumer-link.png"
    symlinked["sha256"] = hashlib.sha256(external).hexdigest()
    symlinked["byte_size"] = len(external)
    symlinked["producer_receipt"]["output"] = {
        key: symlinked[key]
        for key in ("path", "sha256", "byte_size", "width", "height")
    }
    attacks.append(symlinked)

    try:
        for raw in attacks:
            with pytest.raises(scene_masks.SceneMaskError) as caught:
                scene_masks.load_validated_scene_mask(
                    tmp_path,
                    raw,
                    expected_plan_sha256=plan.plan_sha256,
                    expected_source_sha256=plan.frames[0].sha256,
                )
            assert caught.value.code == "worker_output_invalid"
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize("fill", ["empty", "whole"])
def test_empty_or_whole_frame_mask_is_rejected(tmp_path, fill):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    masks = _all_mask_items(tmp_path, plan, payload)
    masks[0] = _mask_item(
        tmp_path,
        plan,
        payload,
        component_id="wall",
        shot_id="shot-a",
        frame_id="f1",
        fill=fill,
    )

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "bad-mask"})
        return httpx.Response(200, json={"status": "succeeded", "result": {"masks": masks}})

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, tmp_path / "receipt.json", plan, handler)
    assert caught.value.code == "worker_output_invalid"


def test_missing_frame_or_cross_hard_cut_producer_receipt_is_rejected(tmp_path):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    complete = _all_mask_items(tmp_path, plan, payload)
    cases = [complete[:-1], [dict(item) for item in complete]]
    cases[1][2]["producer_receipt"] = dict(cases[1][2]["producer_receipt"])
    cases[1][2]["producer_receipt"]["reference_frame_id"] = "f1"

    for index, masks in enumerate(cases):
        receipt = tmp_path / f"receipt-{index}.json"

        def handler(request, masks=masks, index=index):
            if request.method == "POST":
                return httpx.Response(202, json={"task_id": f"bad-{index}"})
            return httpx.Response(
                200, json={"status": "succeeded", "result": {"masks": masks}}
            )

        with pytest.raises(scene_masks.SceneMaskError) as caught:
            _advance(tmp_path, receipt, plan, handler)
        assert caught.value.code == "worker_output_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("membership_engine", "birefnet"),
        ("edge_refinement_scope", "whole_mask"),
        ("fallback", "bbox"),
    ],
)
def test_birefnet_cannot_define_membership_and_fallbacks_are_rejected(
    tmp_path, field, value
):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    masks = _all_mask_items(tmp_path, plan, payload)
    masks[0]["producer_receipt"][field] = value

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "unsafe-producer"})
        return httpx.Response(200, json={"status": "succeeded", "result": {"masks": masks}})

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, tmp_path / "receipt.json", plan, handler)
    assert caught.value.code == "worker_output_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "work/scene-masks/wall/other.png"),
        ("sha256", "0" * 64),
        ("byte_size", 1),
        ("width", 3),
        ("height", 2),
    ],
)
def test_producer_receipt_must_independently_bind_exact_output_item(
    tmp_path, field, value
):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    masks = _all_mask_items(tmp_path, plan, payload)
    masks[0]["producer_receipt"]["output"][field] = value

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "unbound-output"})
        return httpx.Response(
            200, json={"status": "succeeded", "result": {"masks": masks}}
        )

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, tmp_path / "receipt.json", plan, handler)
    assert caught.value.code == "worker_output_invalid"


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink"])
def test_output_path_must_be_relative_contained_regular_png(tmp_path, attack):
    plan = _plan()
    payload = scene_masks.request_payload(plan)
    masks = _all_mask_items(tmp_path, plan, payload)
    outside = tmp_path.parent / f"outside-{tmp_path.name}.png"
    _write_mask(outside)
    raw = outside.read_bytes()
    if attack == "traversal":
        masks[0]["path"] = f"../{outside.name}"
    elif attack == "absolute":
        masks[0]["path"] = str(outside)
    else:
        link = tmp_path / "work/scene-masks/wall/link.png"
        link.symlink_to(outside)
        masks[0]["path"] = "work/scene-masks/wall/link.png"
    masks[0]["sha256"] = hashlib.sha256(raw).hexdigest()
    masks[0]["byte_size"] = len(raw)

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "path-attack"})
        return httpx.Response(200, json={"status": "succeeded", "result": {"masks": masks}})

    try:
        with pytest.raises(scene_masks.SceneMaskError) as caught:
            _advance(tmp_path, tmp_path / "receipt.json", plan, handler)
        assert caught.value.code == "worker_output_invalid"
    finally:
        outside.unlink(missing_ok=True)


def test_changed_plan_cannot_reuse_or_advance_existing_receipt(tmp_path):
    plan = _plan()
    receipt = tmp_path / "receipt.json"

    def initial_handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "bound-task"})
        return httpx.Response(200, json={"status": "running"})

    _advance(tmp_path, receipt, plan, initial_handler)

    def no_network(_request):
        raise AssertionError("receipt mismatch must fail before network")

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(
            tmp_path,
            receipt,
            replace(plan, model_version="different-checkpoint"),
            no_network,
        )
    assert caught.value.code == "receipt_mismatch"


def test_provider_failure_maps_to_safe_error_without_persisting_raw_detail(tmp_path):
    plan = _plan()
    receipt = tmp_path / "receipt.json"
    secret = "Bearer super-secret-token / internal/tenant/path"

    def handler(_request):
        return httpx.Response(500, json={"error": {"message": secret}})

    with pytest.raises(scene_masks.SceneMaskError) as caught:
        _advance(tmp_path, receipt, plan, handler)

    assert caught.value.code == "worker_submit_rejected"
    assert secret not in receipt.read_text(encoding="utf-8")
    assert secret not in str(caught.value)
