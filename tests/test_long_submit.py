import hashlib
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import (
    h3,
    long_generation,
    long_video,
    pipeline,
    postprocess,
    prepared_input,
    stitch,
    storage,
)
from app import main as main_module
from app.main import (
    _SubmitError,
    _long_fit_required,
    _long_validation_paths,
    _resume_long_generation,
    _validate_long_submit_payload,
    create_app,
)
from conftest import AUTH, make_settings


_REAL_STITCHED_OUTPUT_IS_REUSABLE = long_generation.stitched_output_is_reusable


@pytest.fixture(autouse=True)
def _mock_provider_bound_segment_outputs(monkeypatch):
    """Coordinator tests mock H3; receipt/media validation is covered in H3 tests."""
    monkeypatch.setattr(
        h3,
        "output_is_reusable",
        lambda request, *_args, **_kwargs: (
            request.workdir.joinpath("generated.mp4").is_file()
            and request.workdir.joinpath("generated.mp4").stat().st_size > 0
        ),
    )
    monkeypatch.setattr(
        long_generation,
        "stitched_output_is_reusable",
        lambda plan, _dialogue_mode, **_kwargs: (
            plan.root.joinpath("generated.mp4").is_file()
            and plan.root.joinpath("generated.mp4").stat().st_size > 0
        ),
    )


def _png(path: Path, value: int, *, width: int = 90, height: int = 160) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _make_long(settings, *, joins=("hard_cut",), dialogue_text="源台词",
               segment_duration=14.000000000000004, duration=None,
               fit_required=False, landscape_first_indices=(),
               landscape_end_indices=(), legacy=False,
               dialogue_classification=None):
    planned = None
    if duration is None:
        duration = segment_duration * len(joins)
    else:
        planned = long_video.plan_segments(duration, [(0.0, duration)], [])
        joins = tuple(item["join_mode"] for item in planned)
    meta = storage.new_conversation(settings.data_dir, "long", "source.mp4")
    cid = meta["id"]
    root = settings.data_dir / cid
    source = root / "source.mp4"
    source.write_bytes(b"source-video")
    receipt_input = []
    public_segments = []
    chain_no = 0
    for index, join_mode in enumerate(joins, 1):
        if planned is None:
            if join_mode == "hard_cut":
                chain_no += 1
            start_s = segment_duration * (index - 1)
            end_s = segment_duration * index
            chain_id = f"chain-{chain_no:03d}"
            line = {"text": dialogue_text, "start_s": 1.0, "end_s": 2.0}
            if dialogue_classification is not None:
                line["classification"] = dialogue_classification
            local_dialogue = (line,)
        else:
            start_s = planned[index - 1]["start_s"]
            end_s = planned[index - 1]["end_s"]
            chain_id = planned[index - 1]["chain_id"]
            local_dialogue = ()
        segdir = root / "work" / "segments" / str(index)
        work = segdir / "work"
        (segdir / "source.mp4").parent.mkdir(parents=True, exist_ok=True)
        (segdir / "source.mp4").write_bytes(f"segment-{index}".encode())
        key = work / "keyframes" / "01.png"
        first = work / "anchors" / "first.png"
        last = work / "anchors" / "last.png"
        _png(key, 20 + index)
        if index in landscape_first_indices:
            _png(first, 40 + index, width=160, height=90)
        else:
            _png(first, 40 + index)
        if index in landscape_end_indices:
            _png(last, 60 + index, width=160, height=90)
        else:
            _png(last, 60 + index)
        visual_text = f"第{index}段局部动作"
        visual = work / "visual_prompt.txt"
        visual.write_text(visual_text, encoding="utf-8")
        prompt_text = "不要生成背景音乐\n" + prepared_input.compose_final_prompt(
            long_video.compose_segment_visual_prompt(visual_text), local_dialogue
        )
        final = work / "prompt.txt"
        final.write_text(prompt_text, encoding="utf-8")
        segment = {
            "index": index,
            "start_s": start_s,
            "end_s": end_s,
            "chain_id": chain_id,
            "join_mode": join_mode,
            "source": f"segments/{index}/source.mp4",
            "keyframes": ["01.png"],
            "keyframe_paths": [f"segments/{index}/work/keyframes/01.png"],
            "first_frame_path": f"segments/{index}/work/anchors/first.png",
            "last_frame_path": f"segments/{index}/work/anchors/last.png",
            "visual_prompt": visual_text,
            "prompt": prompt_text,
            "dialogue": list(local_dialogue),
            "lines": [dialogue_text],
        }
        public_segments.append(segment)
        receipt_input.append({
            **segment,
            "source_path": segdir / "source.mp4",
            "keyframe_paths": [key],
            "first_frame_path": first,
            "last_frame_path": last,
            "visual_prompt_path": visual,
            "final_prompt_path": final,
        })
    if legacy or duration <= long_video.SHORT_VIDEO_MAX_S:
        def artifact(path):
            path = Path(path)
            return {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        bound_segments = []
        for item in receipt_input:
            dialogue = item["dialogue"]
            bound_segments.append({
                "index": item["index"],
                "start_s": item["start_s"],
                "end_s": item["end_s"],
                "chain_id": item["chain_id"],
                "join_mode": item["join_mode"],
                "source": artifact(item["source_path"]),
                "keyframes": [artifact(path) for path in item["keyframe_paths"]],
                "anchors": [
                    {"role": "first", **artifact(item["first_frame_path"])},
                    {"role": "end", **artifact(item["last_frame_path"])},
                ],
                "visual_prompt": artifact(item["visual_prompt_path"]),
                "final_prompt": artifact(item["final_prompt_path"]),
                "dialogue": {
                    "count": len(dialogue),
                    "sha256": hashlib.sha256(
                        long_video._canonical_bytes(dialogue)
                    ).hexdigest(),
                },
            })
        receipt_path = root / long_video.PLAN_RECEIPT_FILENAME
        receipt_path.write_bytes(long_video._canonical_bytes({
            "schema": "duet.long-video-plan",
            "version": (
                long_video.LEGACY_PLAN_RECEIPT_VERSION
                if legacy else long_video.PLAN_RECEIPT_VERSION
            ),
            "source": artifact(source),
            "video": {"duration_s": duration},
            "workflow": h3.H3_BOUNDARY_WORKFLOW if legacy else h3.H3_WORKFLOW,
            "segments": bound_segments,
        }))
    else:
        receipt_path = long_video.write_plan_receipt(
            root, source=source, duration_s=duration, segments=receipt_input,
            workflow=h3.H3_WORKFLOW,
        )
    receipt = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    storage.update_meta(
        settings.data_dir, cid, status="done", duration_s=duration,
        voice_mode="keep", fit_required=fit_required, segments=public_segments,
        long_video_plan_receipt=receipt_path.name,
    )
    return cid, receipt


def _payload(
    receipt, request_id="parent-request-123", mode="auto", fit="none",
    aspect_ratio="9:16", resolution="768p", fast_mode=None,
    dialogue_delivery=None,
):
    payload = {
        "confirm": True,
        "client_request_id": request_id,
        "dialogue_mode": mode,
        "fit_mode": fit,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "expected_plan_receipt": receipt,
    }
    if fast_mode is not None:
        payload["fast_mode"] = fast_mode
    if dialogue_delivery is not None:
        payload["dialogue_delivery"] = dialogue_delivery
    return payload


def test_long_v4_optimized_dialogue_requires_explicit_delivery(tmp_path):
    settings = make_settings(tmp_path)
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    storage.update_meta(
        settings.data_dir,
        cid,
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": True,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    meta = storage.load_meta(settings.data_dir, cid)

    with pytest.raises(_SubmitError) as caught:
        _validate_long_submit_payload(meta, _payload(receipt))

    assert caught.value.status == 409
    assert caught.value.detail == "dialogue_delivery_required"
    parsed = _validate_long_submit_payload(
        meta, _payload(receipt, dialogue_delivery="off_screen")
    )
    assert parsed[-2] == "off_screen"


def test_long_submit_fast_mode_defaults_off_and_rejects_non_boolean(tmp_path):
    settings = make_settings(tmp_path)
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)

    assert _validate_long_submit_payload(meta, _payload(receipt))[-1] is False
    assert _validate_long_submit_payload(meta, _payload(receipt, fast_mode=True))[-1] is True
    for invalid in (0, 1, "true", None, [], {}):
        payload = _payload(receipt)
        payload["fast_mode"] = invalid
        with pytest.raises(_SubmitError) as caught:
            _validate_long_submit_payload(meta, payload)
        assert caught.value.status == 422
        assert caught.value.detail == "invalid_fast_mode"


def test_fast_mode_prepares_all_segments_then_submits_before_any_poll(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(
        settings, joins=("hard_cut", "continue", "continue")
    )
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    prepared = []
    submitted = []
    polled = []

    def prepare(request):
        prepared.append(request)
        return h3.H3Result("not_started", "attempt")

    def submit(request):
        assert len(prepared) == len(plan.segments)
        submitted.append(request)
        return h3.H3Result("h3_running", "attempt")

    def resume(request):
        assert len(submitted) == len(plan.segments)
        polled.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "attempt")

    monkeypatch.setattr(h3, "prepare", prepare)
    monkeypatch.setattr(h3, "submit", submit)
    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(
        long_generation, "_extract_last_frame",
        lambda *_a: pytest.fail("fast mode must not read a generated tail"),
    )
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan)

    assert len(prepared) == len(submitted) == len(polled) == 3
    assert all(request.mode == "reference" for request in submitted)
    assert [request.keyframes for request in submitted] == [
        segment.keyframes for segment in plan.segments
    ]
    assert all(request.first_frame is None and request.last_frame is None for request in submitted)
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "succeeded"
    assert stored["fast_mode"] is True


def test_receiptless_legacy_postprocess_keeps_operation_and_never_starts_h3(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    root = settings.data_dir / cid
    frame_refs = []
    for index in (1, 2):
        output = (
            root / "work" / "segments" / str(index) / "work"
            / "postprocessed" / "01.png"
        )
        _png(output, 220 + index)
        frame_refs.append(f"segments/{index}/work/postprocessed/01.png")
    meta = storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={
            "status": "done",
            "options": {"remove_subtitle": True, "remove_brand": False},
            "frames": frame_refs,
            "error": None,
        },
    )

    before = storage.load_meta(settings.data_dir, cid)
    assert before == meta
    assert all(key not in before for key in (
        "_postprocess_receipt",
        "_image_optimization",
        "_prompt_fusion",
    ))
    assert before.get("generation") is None
    provider_calls = []
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: provider_calls.append("coordinator"),
    )
    for method in ("prepare", "submit", "resume", "retry"):
        monkeypatch.setattr(
            h3,
            method,
            lambda *_args, _method=method, **_kwargs: provider_calls.append(
                _method
            ),
        )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="postprocess_artifacts_invalid",
    ):
        long_generation.freeze_plan(root, meta, receipt, "none", "auto")

    after = storage.load_meta(settings.data_dir, cid)
    assert after == before
    assert after["id"] == cid
    assert after["postprocess"]["status"] == "done"
    assert after.get("generation") is None
    assert provider_calls == []


def test_existing_unmarked_boundary_attempt_keeps_boundary_recovery(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    payload = json.loads((root / long_video.PLAN_RECEIPT_FILENAME).read_text())
    payload["workflow"] = h3.H3_BOUNDARY_WORKFLOW
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(
        long_video._canonical_bytes(payload)
    )
    receipt = hashlib.sha256(
        (root / long_video.PLAN_RECEIPT_FILENAME).read_bytes()
    ).hexdigest()
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={"status": "resume_required", "segments": []},
    )
    meta = storage.load_meta(settings.data_dir, cid)

    plan = long_generation.freeze_plan(root, meta, receipt, "none", "auto")

    assert plan.workflow == h3.H3_BOUNDARY_WORKFLOW


def test_fast_mode_submit_workers_enter_concurrently(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    submit_barrier = threading.Barrier(2)
    submit_threads = []
    threads_lock = threading.Lock()
    monkeypatch.setattr(
        h3, "prepare", lambda _request: h3.H3Result("not_started", "attempt")
    )

    def submit(_request):
        with threads_lock:
            submit_threads.append(threading.get_ident())
        submit_barrier.wait(timeout=2)
        return h3.H3Result("h3_running", "attempt")

    def resume(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "attempt")

    monkeypatch.setattr(h3, "submit", submit)
    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan)

    assert len(submit_threads) == 2
    assert len(set(submit_threads)) == 2
    assert (
        storage.load_meta(settings.data_dir, cid)["generation"]["status"]
        == "succeeded"
    )


def test_fast_mode_preflight_failure_makes_zero_provider_posts(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    posts = []

    def prepare(request):
        if request.workdir.name == "2":
            raise h3.H3Error("attempt_claim_failed")
        return h3.H3Result("not_started", "attempt")

    monkeypatch.setattr(h3, "prepare", prepare)
    monkeypatch.setattr(h3, "submit", lambda request: posts.append(request))

    long_generation.run(settings, cid, plan)

    assert posts == []
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "failed"


def test_fast_mode_ambiguous_prepared_child_locks_without_posting_siblings(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(
        h3, "prepare", lambda request: h3.H3Result(
            "submission_unknown" if request.workdir.name == "1" else "not_started",
            "attempt",
        ),
    )
    monkeypatch.setattr(
        h3, "submit", lambda _request: pytest.fail("ambiguous preflight must make zero POSTs")
    )

    long_generation.run(settings, cid, plan)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert stored["segments"][0]["status"] == "submission_unknown"


def test_fast_mode_unknown_keeps_polling_submitted_siblings(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(
        settings, joins=("hard_cut", "continue", "continue")
    )
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    polled = []
    monkeypatch.setattr(
        h3, "prepare", lambda _request: h3.H3Result("not_started", "attempt")
    )
    monkeypatch.setattr(
        h3, "submit", lambda request: h3.H3Result(
            "submission_unknown" if request.workdir.name == "2" else "h3_running",
            "attempt",
        ),
    )

    def resume(request):
        polled.append(int(request.workdir.name))
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "attempt")

    monkeypatch.setattr(h3, "resume", resume)

    long_generation.run(settings, cid, plan)

    assert sorted(polled) == [1, 3]
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert [item["status"] for item in stored["segments"]] == [
        "succeeded", "submission_unknown", "succeeded",
    ]


def test_fast_mode_retry_reuses_successful_downstream_segment(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(
        settings, joins=("hard_cut", "continue", "continue")
    )
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(
        h3, "prepare", lambda _request: h3.H3Result("not_started", "attempt")
    )
    monkeypatch.setattr(
        h3, "submit", lambda _request: h3.H3Result("h3_running", "attempt")
    )

    def resume(request):
        if request.workdir.name == "2":
            return h3.H3Result(
                "failed", "attempt", error_code="h3_provider_failed"
            )
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "attempt")

    monkeypatch.setattr(h3, "resume", resume)
    long_generation.run(settings, cid, plan)
    failed = storage.load_meta(settings.data_dir, cid)["generation"]

    retry = long_generation.initial_generation(
        settings, cid, plan, "parent-request-999", 2, failed, fast_mode=True
    )

    assert [item["status"] for item in retry["segments"]] == [
        "succeeded", "not_started", "succeeded",
    ]


def test_fast_mode_startup_is_get_only_and_leaves_prepared_child_unpaid(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    for item in generation["segments"]:
        item.update(
            status="queued", attempt=1,
            child_request_id=long_generation.child_request_id(
                "parent-request-123", receipt, item["index"]
            ),
        )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(
        h3, "submit", lambda _request: pytest.fail("startup must never POST")
    )
    monkeypatch.setattr(
        h3, "resume", lambda _request: h3.H3Result("not_started", "attempt")
    )

    long_generation.run(settings, cid, plan, startup=True)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "resume_required"
    assert [item["status"] for item in stored["segments"]] == ["queued", "queued"]


def test_fast_startup_auto_retries_only_provider_failed_child(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "hard_cut"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    generation.update(status="failed", error="long_video_segment_failed")
    generation["segments"][0].update(
        status="failed", error="h3_provider_failed", attempt=1,
        child_request_id=long_generation.child_request_id(
            "parent-request-123", receipt, 1
        ),
    )
    generation["segments"][1].update(
        status="succeeded", error=None, attempt=1,
        child_request_id=long_generation.child_request_id(
            "parent-request-123", receipt, 2
        ),
    )
    plan.segments[1].workdir.joinpath("generated.mp4").write_bytes(b"sibling")
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    resumed = []

    def resume(request):
        resumed.append(int(request.workdir.name))
        request.workdir.joinpath("generated.mp4").write_bytes(b"retried")
        return h3.H3Result("succeeded", "000002")

    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(h3, "output_is_reusable", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        long_generation, "bound_reusable_segment_indices", lambda *_a, **_kw: frozenset({2})
    )
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan, startup=True)

    assert resumed == [1]
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert [item["status"] for item in stored["segments"]] == ["succeeded", "succeeded"]


def test_serial_provider_auto_retry_success_precedes_downstream_submit(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=False
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    events = []

    def start(request):
        index = int(request.workdir.name)
        events.append("provider-auto-retried-1" if index == 1 else "submitted-2")
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "000002" if index == 1 else "000001")

    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(h3, "output_is_reusable", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        long_generation,
        "_extract_last_frame",
        lambda _v, output: (_png(output, 200) or output),
    )
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan)

    assert events == ["provider-auto-retried-1", "submitted-2"]


def test_serial_startup_provider_retry_success_automatically_starts_downstream(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=False
    )
    generation.update(status="failed", error="long_video_segment_failed")
    generation["segments"][0].update(
        status="failed", error="h3_provider_failed", attempt=1,
        child_request_id=long_generation.child_request_id(
            "parent-request-123", receipt, 1
        ),
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    resumed = []
    started = []

    def resume(request):
        resumed.append(int(request.workdir.name))
        request.workdir.joinpath("generated.mp4").write_bytes(b"recovered")
        return h3.H3Result("succeeded", "000002")

    def start(request):
        started.append(int(request.workdir.name))
        request.workdir.joinpath("generated.mp4").write_bytes(b"downstream")
        return h3.H3Result("succeeded", "000001")

    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(
        long_generation,
        "_extract_last_frame",
        lambda _v, output: (_png(output, 200) or output),
    )
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan, startup=True)

    assert resumed == [1]
    assert started == [2]
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("root_error", "segment_error", "expected"),
    [
        ("long_video_segment_failed", "h3_provider_failed", 1),
        ("submission_unknown", "submission_unknown", 0),
        ("long_video_segment_failed", "download_invalid_video", 0),
    ],
)
def test_long_startup_scanner_only_claims_provider_failed_root(
    tmp_path, monkeypatch, root_error, segment_error, expected,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    generation.update(status="failed", error=root_error)
    generation["segments"][0].update(
        status="failed", error=segment_error, attempt=1,
        child_request_id=long_generation.child_request_id(
            "parent-request-123", receipt, 1
        ),
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    calls = []
    monkeypatch.setattr(
        long_generation, "run", lambda *_args, **kwargs: calls.append(kwargs.get("startup"))
    )

    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/conversations/{cid}", headers=AUTH).status_code == 200
        for thread in client.app.state.h3_resume_threads:
            thread.join(timeout=2)

    assert calls == [True] * expected


def test_fast_mode_startup_revalidates_completed_child_output(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    generation["segments"][0].update(
        status="running", attempt=1,
        child_request_id=long_generation.child_request_id(
            "parent-request-123", receipt, 1
        ),
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(
        h3, "resume", lambda _request: h3.H3Result("succeeded", "attempt")
    )

    long_generation.run(settings, cid, plan, startup=True)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "failed"
    assert stored["segments"][0]["error"] == "long_video_segment_output_invalid"


def test_fast_resume_claim_schedules_exactly_one_coordinator(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    generation["status"] = "resume_required"
    generation["segments"][0].update(
        status="resume_required", attempt=1,
        child_request_id=long_generation.child_request_id(
            "parent-request-123", receipt, 1
        ),
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        aspect_ratio="9:16", resolution="768p",
        frozen_plan_receipt=receipt, generation=generation,
    )
    launch = threading.Barrier(3)
    first_entered = threading.Event()
    release = threading.Event()
    first_finished = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def fake_run(_settings, _cid, _plan):
        with calls_lock:
            calls.append(len(calls) + 1)
            call_number = calls[-1]
        first_entered.set()
        release.wait(timeout=2)
        current = storage.load_meta(settings.data_dir, cid)["generation"]
        if call_number == 1:
            succeeded = {
                **current, "status": "succeeded", "error": None, "stage": "stitch"
            }
            succeeded["segments"][0] = {
                **succeeded["segments"][0], "status": "succeeded", "error": None
            }
            storage.update_meta(settings.data_dir, cid, generation=succeeded)
            first_finished.set()
        else:
            first_finished.wait(timeout=2)
            storage.update_meta(
                settings.data_dir, cid,
                generation={
                    **storage.load_meta(settings.data_dir, cid)["generation"],
                    "status": "submission_unknown", "error": "submission_unknown",
                },
            )

    monkeypatch.setattr(long_generation, "run", fake_run)

    def submit_once():
        launch.wait(timeout=2)
        return client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH,
            json=_payload(receipt, fast_mode=True),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit_once) for _ in range(2)]
        launch.wait(timeout=2)
        assert first_entered.wait(timeout=2)
        deadline = time.monotonic() + 2
        while not any(future.done() for future in futures) and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        responses = [future.result(timeout=2) for future in futures]

    assert [response.status_code for response in responses] == [202, 202]
    assert calls == [1]
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "succeeded"
    assert stored["client_request_id"] == "parent-request-123"
    assert stored["segments"][0]["child_request_id"] == generation["segments"][0]["child_request_id"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_fast_startup_missing_or_corrupt_attempt_locks_without_post(
    tmp_path, monkeypatch, corrupt,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    child_id = long_generation.child_request_id("parent-request-123", receipt, 1)
    generation["status"] = "running"
    generation["segments"][0].update(
        status="queued", attempt=1, child_request_id=child_id
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        aspect_ratio="9:16", resolution="768p",
        frozen_plan_receipt=receipt, generation=generation,
    )
    if corrupt:
        attempt = (
            plan.segments[0].workdir / ".h3" / "attempts" / "000001" / "attempt.json"
        )
        attempt.parent.mkdir(parents=True)
        attempt.write_text(
            json.dumps({"client_request_id": child_id}), encoding="utf-8"
        )
    monkeypatch.setattr(
        h3, "_submit_h3", lambda *_a, **_kw: pytest.fail("recovery must make zero POSTs")
    )

    long_generation.run(settings, cid, plan, startup=True)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert stored["client_request_id"] == "parent-request-123"
    assert stored["segments"][0]["attempt"] == 1
    assert stored["segments"][0]["child_request_id"] == child_id


@pytest.mark.parametrize("corrupt", [False, True])
def test_fast_explicit_resume_missing_or_corrupt_attempt_becomes_unknown(
    enabled, monkeypatch, corrupt,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    child_id = long_generation.child_request_id("parent-request-123", receipt, 1)
    generation["status"] = "resume_required"
    generation["segments"][0].update(
        status="queued", attempt=1, child_request_id=child_id
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        aspect_ratio="9:16", resolution="768p",
        frozen_plan_receipt=receipt, generation=generation,
    )
    if corrupt:
        attempt = (
            plan.segments[0].workdir / ".h3" / "attempts" / "000001" / "attempt.json"
        )
        attempt.parent.mkdir(parents=True)
        attempt.write_text(
            json.dumps({"client_request_id": child_id}), encoding="utf-8"
        )
    monkeypatch.setattr(
        h3, "_submit_h3", lambda *_a, **_kw: pytest.fail("missing attempt must make zero POSTs")
    )

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, fast_mode=True),
    )
    retry = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(
            receipt, request_id="parent-request-999", fast_mode=True
        ),
    )

    assert response.status_code == 202
    assert retry.status_code == 409
    assert retry.json() == {"detail": "submission_outcome_unknown"}
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert stored["client_request_id"] == "parent-request-123"
    assert stored["segments"][0]["attempt"] == 1
    assert stored["segments"][0]["child_request_id"] == child_id


def test_fast_explicit_resume_submits_provably_prepared_child(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    child_id = long_generation.child_request_id("parent-request-123", receipt, 1)
    generation["status"] = "resume_required"
    generation["segments"][0].update(
        status="queued", attempt=1, child_request_id=child_id
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        aspect_ratio="9:16", resolution="768p",
        frozen_plan_receipt=receipt, generation=generation,
    )
    request = long_generation._request(
        settings, cid, plan, plan.segments[0], "parent-request-123", "none",
        frozen_child_id=child_id, prepare_inputs=False, fast_mode=True,
    )
    assert h3.prepare(request).attempt_id == "000001"
    submitted = []

    def submit(prepared_request):
        inspected = h3.inspect(prepared_request)
        assert inspected.status == "ready_to_submit"
        assert inspected.attempt_id == "000001"
        submitted.append(prepared_request.client_request_id)
        return h3.H3Result("h3_running", "000001")

    def resume(resumed_request):
        resumed_request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "000001")

    monkeypatch.setattr(h3, "submit", submit)
    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, fast_mode=True),
    )

    assert response.status_code == 202
    assert submitted == [child_id]
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"


def test_fast_mode_is_frozen_across_failed_parent_retry(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1, fast_mode=True
    )
    generation.update(status="failed", error="long_video_segment_failed")
    generation["segments"][0].update(status="failed", error="h3_provider_failed")
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        aspect_ratio="9:16", resolution="768p",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(
        h3, "submit", lambda _request: pytest.fail("CAS rejection must make zero POSTs")
    )

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(
            receipt, request_id="parent-request-999", fast_mode=False
        ),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "resume_parameters_changed"}
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["fast_mode"] is True


def test_fast_mode_submit_api_persists_choice_and_uses_parallel_anchor(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    requests = []
    monkeypatch.setattr(
        h3, "prepare", lambda _request: h3.H3Result("not_started", "attempt")
    )
    monkeypatch.setattr(
        h3, "submit", lambda request: (
            requests.append(request) or h3.H3Result("h3_running", "attempt")
        ),
    )

    def resume(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "attempt")

    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, fast_mode=True),
    )

    assert response.status_code == 202
    assert len(requests) == 2
    assert requests[1].first_frame == requests[0].last_frame
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["fast_mode"] is True
    assert "child_request_id" not in json.dumps(detail["generation"])


def _fake_stitch(calls):
    def invoke(**kwargs):
        calls.append(kwargs)
        kwargs["output"].write_bytes(b"joined")
    return invoke


def test_long_validation_fingerprint_tracks_selected_aspect_fit_outputs(tmp_path):
    settings = make_settings(tmp_path, enable_pipeline=False)
    cid, _receipt = _make_long(settings, landscape_first_indices=(1,))
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    meta.update(
        aspect_ratio="16:9",
        resolution="480p",
        fit_mode="crop",
        postprocess={
            "status": "done",
            "frames": ["segments/1/work/postprocessed/01.png"],
        },
        generation={"fit_layout": long_generation.FIT_LAYOUT_ASPECT},
    )
    selected = (
        root / "work" / "segments" / "1" / "work"
        / "postprocessed" / "01.png"
    )
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_bytes(
        (root / "work" / "segments" / "1" / "work" / "keyframes" / "01.png")
        .read_bytes()
    )

    paths = _long_validation_paths(root, meta)

    expected_root = root / "work" / "segments" / "1" / "work" / "h3_frames"
    assert expected_root / "16x9" / "crop" / "first" / "first.png" in paths
    assert expected_root / "crop" / "first" / "first.png" not in paths
    assert selected in paths
    assert expected_root / "16x9" / "crop" / "keyframes" / "01.png" in paths


def test_pre_marker_recovery_rejects_ambiguous_complete_fit_layouts(tmp_path):
    settings = make_settings(tmp_path, enable_pipeline=False)
    cid, receipt = _make_long(
        settings, fit_required=True, landscape_first_indices=(1,)
    )
    root = settings.data_dir / cid
    legacy_meta = storage.load_meta(settings.data_dir, cid)
    long_generation.freeze_plan(root, legacy_meta, receipt, "crop", "auto")
    semantic_meta = {
        **legacy_meta,
        "aspect_ratio": "9:16",
        "resolution": "768p",
    }
    long_generation.freeze_plan(
        root,
        semantic_meta,
        receipt,
        "crop",
        "auto",
        aspect_ratio="9:16",
        resolution="768p",
    )

    with pytest.raises(long_generation.LongGenerationError) as raised:
        long_generation.freeze_plan(
            root,
            semantic_meta,
            receipt,
            "crop",
            "auto",
            prepare_fit=False,
        )

    assert raised.value.code == "frame_fit_failed"


def _small_video(path: Path, color: str = "black") -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s=32x32:r=5:d=1", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(path),
        ],
        check=True,
        capture_output=True,
    )


def _with_boundary_bounds(plan, *, receipt_version):
    segment = replace(plan.segments[0], start_s=27.52, end_s=37.52)
    return replace(
        plan,
        segments=(segment,),
        receipt_version=receipt_version,
        workflow=h3.H3_BOUNDARY_WORKFLOW,
    )


def test_startup_reconciles_half_frozen_long_submit_without_provider(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=False, enable_h3_submit=False
    )
    cid, receipt = _make_long(settings)
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    claimed = storage.claim_submission_input(
        settings.data_dir, cid, "request-old-long"
    )
    assert claimed
    long_generation.freeze_plan(
        settings.data_dir / cid, claimed, receipt, "crop", "auto"
    )
    cdir = settings.data_dir / cid
    before = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    provider_calls = []
    monkeypatch.setattr(h3, "start", lambda *_args, **_kwargs: provider_calls.append(1))

    with TestClient(create_app(settings)):
        pass

    recovered = storage.load_meta(settings.data_dir, cid)
    after = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    assert recovered["status"] == "done"
    assert recovered["error"] == "submission_recovery_required"
    assert recovered["generation"] is None
    assert recovered["_input_owner"] is None
    assert provider_calls == []
    assert after == before
    assert storage.claim_submission_input(
        settings.data_dir, cid, "request-new-long"
    )


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    with TestClient(create_app(settings)) as client:
        yield settings, client


def _make_prompt_fusion_claim(
    settings, *, status: str, error: str | None,
):
    cid, _receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={"status": "done", "segments": []},
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    claimed = storage.claim_submission_input(
        settings.data_dir, cid, "durable-request-123"
    )
    assert claimed is not None
    owner = {**claimed["_input_owner"], "fast_mode": False}
    storage.update_meta(
        settings.data_dir,
        cid,
        _input_owner=owner,
        _prompt_fusion={
            "version": long_generation.PROMPT_FUSION_VERSION,
            "status": status,
            "error": error,
            "input_sha256": "a" * 64,
            "image_acceptance_sha256": "b" * 64,
            "manifest_sha256": "d" * 64 if status == "done" else None,
        },
        dialogue_mode="auto",
        dialogue_delivery="auto",
        resolved_dialogue_delivery="off_screen",
        fit_mode="none",
        aspect_ratio="9:16",
        resolution="768p",
    )
    return cid, owner


def test_running_prompt_fusion_is_observed_without_requeue_or_second_producer(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings, status="running", error=None
    )
    before = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("running must not produce"),
    )

    retry = main_module._prompt_fusion_continuation_step(
        settings, cid, object(), owner
    )

    assert retry is True
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"] == before


def test_running_prompt_fusion_has_only_one_owner_timer(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings, status="running", error=None
    )
    timers = []

    class Timer:
        def __init__(self, interval, function):
            self.interval = interval
            self.function = function
            self.daemon = False
            self.started = False
            timers.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(main_module.threading, "Timer", Timer)
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("running must not produce"),
    )

    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )
    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )

    assert len(timers) == 1
    assert timers[0].started is True and timers[0].daemon is True
    key = main_module._prompt_fusion_timer_key(settings, cid, owner)
    with main_module._PROMPT_FUSION_TIMER_LOCK:
        main_module._PROMPT_FUSION_TIMERS.pop(key, None)


def test_retryable_technical_fusion_failure_auto_continues_same_owner(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings,
        status="failed",
        error="codex timed out after 30.0s",
    )
    producer_calls = []
    continuation_calls = []
    scheduled = []

    def produce(_settings, received_cid, _runner):
        producer_calls.append(received_cid)
        current = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={**current, "status": "done", "error": None},
        )
        return "done"

    def schedule(*args):
        scheduled.append(args[1])
        main_module._produce_prompt_fusion_and_continue(*args)

    monkeypatch.setattr(pipeline, "produce_prompt_fusion", produce)
    monkeypatch.setattr(
        main_module,
        "_continue_after_prompt_fusion",
        lambda _settings, received_cid, received_owner: (
            continuation_calls.append((received_cid, received_owner)) or True
        ),
    )
    monkeypatch.setattr(
        main_module, "_schedule_prompt_fusion_continuation", schedule
    )

    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )

    assert scheduled == [cid]
    assert producer_calls == [cid]
    assert continuation_calls == [(cid, owner)]
    assert storage.load_meta(settings.data_dir, cid)["_input_owner"] == owner


@pytest.mark.parametrize(
    "error",
    [
        "prompt_fusion_output_invalid",
        "prompt fusion raw output is missing",
        "prompt fusion raw output is invalid",
    ],
)
def test_invalid_or_missing_fusion_output_auto_continues_same_operation(
    tmp_path, monkeypatch, error,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings, status="failed", error=error
    )
    producer_calls = []
    continuation_calls = []
    scheduled = []

    def produce(_settings, received_cid, _runner):
        producer_calls.append(received_cid)
        current = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={**current, "status": "done", "error": None},
        )
        return "done"

    def schedule(*args):
        scheduled.append(args[1])
        main_module._produce_prompt_fusion_and_continue(*args)

    monkeypatch.setattr(pipeline, "produce_prompt_fusion", produce)
    monkeypatch.setattr(
        main_module,
        "_continue_after_prompt_fusion",
        lambda _settings, received_cid, received_owner: (
            continuation_calls.append((received_cid, received_owner)) or True
        ),
    )
    monkeypatch.setattr(
        main_module, "_schedule_prompt_fusion_continuation", schedule
    )

    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )

    assert scheduled == [cid]
    assert producer_calls == [cid]
    assert continuation_calls == [(cid, owner)]
    assert storage.load_meta(settings.data_dir, cid)["_input_owner"] == owner


@pytest.mark.parametrize(
    "error",
    [
        "prompt fusion input drifted",
        "prompt fusion Skill drifted",
        "prompt fusion manifest drifted",
    ],
)
def test_fusion_evidence_drift_never_reposts(
    tmp_path, monkeypatch, error,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings, status="failed", error=error
    )
    before = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("drift must not produce"),
    )
    monkeypatch.setattr(
        main_module,
        "_schedule_prompt_fusion_continuation",
        lambda *_args, **_kwargs: pytest.fail("drift must not schedule"),
    )

    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )

    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"] == before


def test_quality_score_does_not_trigger_fusion_retry(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings, status="done", error=None
    )
    current = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={**current, "quality_score": 0.0},
    )
    continuation_calls = []
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("score must not produce"),
    )
    monkeypatch.setattr(
        main_module,
        "_continue_after_prompt_fusion",
        lambda _settings, received_cid, received_owner: (
            continuation_calls.append((received_cid, received_owner)) or True
        ),
    )

    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )

    assert continuation_calls == [(cid, owner)]


def test_finalize_exception_auto_continues_done_fusion_without_new_producer(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, owner = _make_prompt_fusion_claim(
        settings, status="done", error=None
    )
    continuation_calls = []
    scheduled = []

    def continue_once(_settings, received_cid, received_owner):
        continuation_calls.append((received_cid, received_owner))
        if len(continuation_calls) == 1:
            raise OSError("transient local finalize error")
        return True

    def schedule(*args):
        scheduled.append(args[1])
        main_module._produce_prompt_fusion_and_continue(*args)

    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("done Fusion must be reused"),
    )
    monkeypatch.setattr(
        main_module, "_continue_after_prompt_fusion", continue_once
    )
    monkeypatch.setattr(
        main_module, "_schedule_prompt_fusion_continuation", schedule
    )

    main_module._produce_prompt_fusion_and_continue(
        settings, cid, object(), owner
    )

    assert scheduled == [cid]
    assert continuation_calls == [(cid, owner), (cid, owner)]
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "done"


def test_prompt_fusion_refresh_is_one_202_operation_without_second_submit_job(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={"status": "done", "segments": []},
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    fusion_calls = []
    coordinator_calls = []
    finalize_calls = []
    frozen_plan = object()

    def finalize(*_args, **_kwargs):
        finalize_calls.append(True)
        if len(finalize_calls) > 1:
            return "c" * 64
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={
                "version": long_generation.PROMPT_FUSION_VERSION,
                "status": "queued",
                "error": None,
                "input_sha256": "a" * 64,
                "image_acceptance_sha256": "b" * 64,
                "manifest_sha256": None,
            },
        )
        raise long_generation.LongGenerationError(
            "prompt_fusion_refresh_required"
        )

    def produce(_settings, received_cid, _runner):
        fusion_calls.append(received_cid)
        current = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={**current, "status": "done"},
        )
        return "done"

    monkeypatch.setattr(
        long_generation, "finalize_multimodal_plan", finalize
    )
    monkeypatch.setattr(pipeline, "produce_prompt_fusion", produce)
    monkeypatch.setattr(
        long_generation,
        "freeze_plan",
        lambda *_args, **_kwargs: frozen_plan,
    )
    monkeypatch.setattr(
        long_generation,
        "initial_generation",
        lambda _settings, _cid, plan, request_id, attempt, _old,
        *, fast_mode=False: {
            "status": "queued",
            "error": None,
            "attempt": attempt,
            "client_request_id": request_id,
            "stage": "h3",
            "fast_mode": fast_mode,
            "segments": [],
        } if plan is frozen_plan else pytest.fail("wrong frozen plan"),
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: coordinator_calls.append(True),
    )
    payload = _payload(receipt, dialogue_delivery="auto")

    first = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
    )
    stored_after_first = storage.load_meta(settings.data_dir, cid)
    replay = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={**payload, "client_request_id": "different-request-456"},
    )

    accepted = {
        "operation_id": cid,
        "status": "running",
        "stage": "prompt_fusion",
    }
    assert first.status_code == 202, first.json()
    assert replay.status_code == 202, replay.json()
    assert first.json() == accepted
    assert replay.json() == {
        "operation_id": cid,
        "status": "running",
        "stage": "h3",
        "attempt": 1,
    }
    assert fusion_calls == [cid]
    assert finalize_calls == [True, True]
    assert coordinator_calls == [True]
    assert stored_after_first["_input_owner"] is None
    assert stored_after_first["frozen_plan_receipt"] == "c" * 64
    assert stored_after_first["generation"]["client_request_id"] == (
        payload["client_request_id"]
    )


def test_startup_continues_done_fusion_claim_into_same_generation(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, _receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={"status": "done", "segments": []},
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    claimed = storage.claim_submission_input(
        settings.data_dir, cid, "durable-request-123"
    )
    assert claimed is not None
    owner = {**claimed["_input_owner"], "fast_mode": False}
    storage.update_meta(
        settings.data_dir,
        cid,
        _input_owner=owner,
        _prompt_fusion={
            "version": long_generation.PROMPT_FUSION_VERSION,
            "status": "done",
            "error": None,
            "input_sha256": "a" * 64,
            "image_acceptance_sha256": "b" * 64,
            "manifest_sha256": "d" * 64,
        },
        dialogue_mode="auto",
        dialogue_delivery="auto",
        resolved_dialogue_delivery="off_screen",
        fit_mode="none",
        aspect_ratio="9:16",
        resolution="768p",
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")

    frozen_plan = object()
    coordinator_calls = []
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("done Fusion must be reused"),
    )
    monkeypatch.setattr(
        long_generation,
        "finalize_multimodal_plan",
        lambda *_args, **_kwargs: "c" * 64,
    )
    monkeypatch.setattr(
        long_generation,
        "freeze_plan",
        lambda *_args, **_kwargs: frozen_plan,
    )
    monkeypatch.setattr(
        long_generation,
        "initial_generation",
        lambda _settings, _cid, plan, request_id, attempt, _old,
        *, fast_mode=False: {
            "status": "queued",
            "error": None,
            "attempt": attempt,
            "client_request_id": request_id,
            "stage": "h3",
            "fast_mode": fast_mode,
            "segments": [],
        } if plan is frozen_plan else pytest.fail("wrong frozen plan"),
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: coordinator_calls.append(True),
    )

    with TestClient(create_app(settings)) as client:
        for thread in client.app.state.prompt_fusion_recovery_threads:
            thread.join(timeout=5)

    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["_input_owner"] is None
    assert recovered["frozen_plan_receipt"] == "c" * 64
    assert recovered["generation"]["client_request_id"] == (
        "durable-request-123"
    )
    assert coordinator_calls == [True]


def test_startup_starts_only_pristine_durable_v4_generation(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "durable-request-123", 1
    )
    assert all(
        item["status"] == "not_started"
        and item["child_request_id"] is None
        for item in generation["segments"]
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
        dialogue_mode="auto",
        dialogue_delivery="auto",
        fit_mode="none",
        aspect_ratio=plan.aspect_ratio,
        resolution=plan.resolution,
        frozen_plan_receipt=receipt,
        generation=generation,
    )
    calls = []
    monkeypatch.setattr(
        long_generation, "freeze_plan", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **kwargs: calls.append(kwargs.get("startup")),
    )

    _resume_long_generation(settings, cid)

    assert calls == [False]


def test_technical_acceptance_auto_claim_reaches_fusion_and_h3_once(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, _receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={"status": "done", "segments": []},
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    monkeypatch.setattr(
        postprocess,
        "image_acceptance_status",
        lambda *_args, **_kwargs: {
            "required": True,
            "accepted": True,
            "expected_meta_sha256": "a" * 64,
        },
    )
    finalize_calls = []

    def finalize(*_args, **_kwargs):
        finalize_calls.append(True)
        if len(finalize_calls) > 1:
            return "c" * 64
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={
                "version": long_generation.PROMPT_FUSION_VERSION,
                "status": "queued",
                "error": None,
                "input_sha256": "a" * 64,
                "image_acceptance_sha256": "b" * 64,
                "manifest_sha256": None,
            },
        )
        raise long_generation.LongGenerationError(
            "prompt_fusion_refresh_required"
        )

    def produce(_settings, _cid, _runner):
        current = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={**current, "status": "done"},
        )
        return "done"

    frozen_plan = object()
    coordinator_calls = []
    monkeypatch.setattr(
        long_generation, "finalize_multimodal_plan", finalize
    )
    monkeypatch.setattr(pipeline, "produce_prompt_fusion", produce)
    monkeypatch.setattr(
        long_generation,
        "freeze_plan",
        lambda *_args, **_kwargs: frozen_plan,
    )
    monkeypatch.setattr(
        long_generation,
        "initial_generation",
        lambda _settings, _cid, plan, request_id, attempt, _old,
        *, fast_mode=False: {
            "status": "queued",
            "error": None,
            "attempt": attempt,
            "client_request_id": request_id,
            "stage": "h3",
            "fast_mode": fast_mode,
            "segments": [],
        } if plan is frozen_plan else pytest.fail("wrong frozen plan"),
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: coordinator_calls.append(True),
    )

    main_module._start_automatic_v4_generation(
        settings, cid, object()
    )

    stored = storage.load_meta(settings.data_dir, cid)
    assert stored["_input_owner"] is None
    assert stored["generation"]["client_request_id"] == f"auto-{cid}"
    assert finalize_calls == [True, True]
    assert coordinator_calls == [True]


def _make_automatic_pre_fusion_claim(
    settings, monkeypatch, *, request_id: str | None = None,
):
    cid, _receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={"status": "done", "segments": []},
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    monkeypatch.setattr(
        postprocess,
        "image_acceptance_status",
        lambda *_args, **_kwargs: {
            "required": True,
            "accepted": True,
            "expected_meta_sha256": "a" * 64,
        },
    )
    claimed = storage.claim_submission_input(
        settings.data_dir,
        cid,
        request_id or f"auto-{cid}",
    )
    assert claimed is not None
    owner = {**claimed["_input_owner"], "fast_mode": False}
    storage.update_meta(
        settings.data_dir,
        cid,
        _input_owner=owner,
        dialogue_mode="auto",
        dialogue_delivery="auto",
        resolved_dialogue_delivery="on_screen",
        fit_mode="none",
        aspect_ratio="9:16",
        resolution="768p",
    )
    return cid, owner


def _install_automatic_pre_fusion_harness(
    settings, cid, monkeypatch,
):
    finalize_process_generations = []
    fusion_calls = []
    coordinator_calls = []
    frozen_plan = object()

    def finalize(*_args, **_kwargs):
        current = storage.load_meta(settings.data_dir, cid)
        finalize_process_generations.append(
            current["_input_owner"]["process_generation"]
        )
        if current.get("_prompt_fusion") is None:
            storage.update_meta(
                settings.data_dir,
                cid,
                _prompt_fusion={
                    "version": long_generation.PROMPT_FUSION_VERSION,
                    "status": "queued",
                    "error": None,
                    "input_sha256": "a" * 64,
                    "image_acceptance_sha256": "b" * 64,
                    "manifest_sha256": None,
                },
            )
            raise long_generation.LongGenerationError(
                "prompt_fusion_refresh_required"
            )
        return "c" * 64

    def produce(_settings, received_cid, _runner):
        fusion_calls.append(received_cid)
        current = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={**current, "status": "done"},
        )
        return "done"

    monkeypatch.setattr(
        long_generation, "finalize_multimodal_plan", finalize
    )
    monkeypatch.setattr(pipeline, "produce_prompt_fusion", produce)
    monkeypatch.setattr(
        long_generation,
        "freeze_plan",
        lambda *_args, **_kwargs: frozen_plan,
    )
    monkeypatch.setattr(
        long_generation,
        "initial_generation",
        lambda _settings, _cid, plan, request_id, attempt, _old,
        *, fast_mode=False: {
            "status": "queued",
            "error": None,
            "attempt": attempt,
            "client_request_id": request_id,
            "stage": "h3",
            "fast_mode": fast_mode,
            "segments": [],
        } if plan is frozen_plan else pytest.fail("wrong frozen plan"),
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: coordinator_calls.append(True),
    )
    monkeypatch.setattr(
        h3,
        "start",
        lambda *_args, **_kwargs: pytest.fail("provider must not start"),
    )
    return finalize_process_generations, fusion_calls, coordinator_calls


def test_automatic_pre_fusion_claim_reuses_owner_and_one_producer(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, owner = _make_automatic_pre_fusion_claim(
        settings, monkeypatch
    )
    finalize_generations, fusion_calls, coordinator_calls = (
        _install_automatic_pre_fusion_harness(
            settings, cid, monkeypatch
        )
    )
    monkeypatch.setattr(
        storage,
        "claim_submission_input",
        lambda *_args, **_kwargs: pytest.fail("must reuse durable owner"),
    )

    main_module._start_automatic_v4_generation(settings, cid, object())
    main_module._start_automatic_v4_generation(settings, cid, object())

    recovered = storage.load_meta(settings.data_dir, cid)
    assert owner["request_id"] == f"auto-{cid}"
    assert recovered["_input_owner"] is None
    assert recovered["generation"]["client_request_id"] == f"auto-{cid}"
    assert finalize_generations == [
        storage.PROCESS_GENERATION,
        storage.PROCESS_GENERATION,
    ]
    assert fusion_calls == [cid]
    assert coordinator_calls == [True]


def test_startup_atomically_resumes_automatic_pre_fusion_claim(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    cid, old_owner = _make_automatic_pre_fusion_claim(
        settings, monkeypatch
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    finalize_generations, fusion_calls, coordinator_calls = (
        _install_automatic_pre_fusion_harness(
            settings, cid, monkeypatch
        )
    )
    monkeypatch.setattr(
        storage,
        "claim_submission_input",
        lambda *_args, **_kwargs: pytest.fail("restart must reuse owner"),
    )

    with TestClient(create_app(settings)) as client:
        deadline = time.monotonic() + 5
        while (
            storage.load_meta(settings.data_dir, cid).get("generation") is None
            and time.monotonic() < deadline
        ):
            client.get("/api/health", headers=AUTH)
            time.sleep(0.01)
        assert len(client.app.state.operation_recovery_tasks) == 1

    recovered = storage.load_meta(settings.data_dir, cid)
    assert old_owner["process_generation"] == "boot-old"
    assert recovered["_input_owner"] is None
    assert recovered["generation"]["client_request_id"] == f"auto-{cid}"
    assert finalize_generations == ["boot-new", "boot-new"]
    assert fusion_calls == [cid]
    assert coordinator_calls == [True]


@pytest.mark.parametrize("drift", ["snapshot", "nonauto"])
def test_automatic_pre_fusion_recovery_rejects_drift_and_nonauto_owner(
    tmp_path, monkeypatch, drift,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, owner = _make_automatic_pre_fusion_claim(
        settings,
        monkeypatch,
        request_id="manual-request-123" if drift == "nonauto" else None,
    )
    if drift == "snapshot":
        plan_path = settings.data_dir / cid / long_video.PLAN_RECEIPT_FILENAME
        plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    monkeypatch.setattr(
        storage,
        "claim_submission_input",
        lambda *_args, **_kwargs: pytest.fail("must not create a new owner"),
    )
    monkeypatch.setattr(
        long_generation,
        "finalize_multimodal_plan",
        lambda *_args, **_kwargs: pytest.fail("invalid claim must not resume"),
    )
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("Fusion must not start"),
    )
    monkeypatch.setattr(
        h3,
        "start",
        lambda *_args, **_kwargs: pytest.fail("provider must not start"),
    )

    main_module._start_automatic_v4_generation(settings, cid, object())
    assert not main_module._prompt_fusion_continuation_step(
        settings, cid, object(), owner
    )

    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["_input_owner"] == owner
    assert recovered.get("_prompt_fusion") is None
    assert recovered.get("generation") is None


def test_automatic_pre_fusion_local_error_retries_same_owner_only(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, owner = _make_automatic_pre_fusion_claim(
        settings, monkeypatch
    )
    monkeypatch.setattr(
        storage,
        "claim_submission_input",
        lambda *_args, **_kwargs: pytest.fail("must not create a new owner"),
    )

    def fail_locally(*_args, **_kwargs):
        raise OSError("local interruption")

    monkeypatch.setattr(
        long_generation,
        "finalize_multimodal_plan",
        fail_locally,
    )
    monkeypatch.setattr(
        pipeline,
        "produce_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail("Fusion producer not queued"),
    )
    monkeypatch.setattr(
        h3,
        "start",
        lambda *_args, **_kwargs: pytest.fail("provider must not start"),
    )

    assert main_module._prompt_fusion_continuation_step(
        settings, cid, object(), owner
    )

    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["_input_owner"] == owner
    assert recovered.get("_prompt_fusion") is None
    assert recovered.get("generation") is None


def test_current_operation_replay_ignores_new_id_and_never_reposts_unknown(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
        generation={
            "status": "submission_unknown",
            "error": "submission_unknown",
            "attempt": 1,
            "client_request_id": "parent-request-123",
            "stage": "h3",
            "segments": [],
        },
    )
    monkeypatch.setattr(
        h3, "start", lambda *_args, **_kwargs: pytest.fail("must not POST")
    )
    monkeypatch.setattr(
        h3, "retry", lambda *_args, **_kwargs: pytest.fail("must not retry")
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: pytest.fail("unknown is GET-only"),
    )

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(
            receipt,
            request_id="another-request-456",
            resolution="480p",
        ),
    )

    assert response.status_code == 202
    assert response.json() == {
        "operation_id": cid,
        "status": "running",
        "stage": "h3",
        "attempt": 1,
    }
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert stored["client_request_id"] == "parent-request-123"


def test_long_v4_combined_first_promotion_binds_settings_and_is_unpaid(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={
            "status": "done",
            "options": {
                "remove_subtitle": True,
                "remove_brand": False,
                "optimize_image": True,
            },
            "frames": ["segments/1/work/postprocessed/01.png"],
        },
        _image_optimization={"version": 4},
        _v4_canvas_execution={"version": 1},
    )
    promotion_calls = []
    provider_calls = []

    def finalize(root, meta, expected, fit_mode, dialogue_mode, **kwargs):
        assert kwargs["settings"] is settings
        assert meta["postprocess"]["options"] == {
            "remove_subtitle": True,
            "remove_brand": False,
            "optimize_image": True,
        }
        assert meta["_image_optimization"]["version"] == 4
        assert meta["_v4_canvas_execution"]["version"] == 1
        promotion_calls.append(
            (root, expected, fit_mode, dialogue_mode)
        )
        return "f" * 64

    monkeypatch.setattr(
        long_generation, "finalize_multimodal_plan", finalize
    )
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **_kwargs: provider_calls.append("run"),
    )

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(receipt),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": {
        "code": "long_video_plan_changed",
        "message": "音画计划已冻结，请刷新并确认后重试。",
    }}
    assert promotion_calls == [(
        settings.data_dir / cid, receipt, "none", "auto",
    )]
    assert provider_calls == []
    assert storage.load_meta(settings.data_dir, cid).get("generation") is None


def test_long_v4_optimized_nine_frames_final_freeze_binds_settings(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    base_plan = long_generation.freeze_plan(
        root,
        storage.load_meta(settings.data_dir, cid),
        receipt,
        "none",
        "auto",
        settings=settings,
    )
    frame_refs = []
    for index in range(1, 10):
        relative = f"segments/1/work/postprocessed/{index:02d}.png"
        _png(root / "work" / relative, 100 + index)
        frame_refs.append(relative)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={
            "status": "done",
            "options": {
                "remove_subtitle": True,
                "remove_brand": False,
                "optimize_image": True,
            },
            "frames": frame_refs,
        },
        _image_optimization={"version": 4},
        _v4_canvas_execution={"version": 1},
    )
    freeze_calls = []
    coordinator_calls = []
    expected_settings = settings

    def finalize(root_, meta, expected, fit_mode, dialogue_mode, **kwargs):
        assert kwargs["settings"] is settings
        assert root_ == root
        assert expected == receipt
        assert fit_mode == "none"
        assert dialogue_mode == "auto"
        assert meta["postprocess"]["frames"] == frame_refs
        assert all((root / "work" / relative).is_file() for relative in frame_refs)
        return None

    def freeze(
        root_, meta, expected, fit_mode, dialogue_mode, *,
        aspect_ratio, resolution, dialogue_delivery, prepare_fit, settings,
    ):
        assert settings is expected_settings
        assert dialogue_delivery == "auto"
        assert meta["postprocess"]["frames"] == frame_refs
        freeze_calls.append((
            root_, expected, fit_mode, dialogue_mode,
            aspect_ratio, resolution, prepare_fit,
        ))
        return base_plan

    monkeypatch.setattr(
        long_generation, "finalize_multimodal_plan", finalize
    )
    monkeypatch.setattr(long_generation, "freeze_plan", freeze)
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *args: coordinator_calls.append(args),
    )

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(receipt),
    )

    assert response.status_code == 202
    assert response.json() == {"status": "queued", "attempt": 1}
    assert freeze_calls == [(
        root, receipt, "none", "auto", "9:16", "768p", True,
    )]
    assert coordinator_calls == [(settings, cid, base_plan)]


def test_long_paid_boundary_revalidation_failure_makes_zero_h3_post(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "long-producer-toctou", 1,
    )
    storage.update_meta(
        settings.data_dir, cid,
        dialogue_mode="auto", fit_mode="none", frozen_plan_receipt=receipt,
        generation=generation,
    )
    checks = []

    def reject(_plan, segment, _request):
        checks.append(segment.index)
        raise long_generation.LongGenerationError(
            "speaker_visibility_output_hash_mismatch"
        )

    monkeypatch.setattr(
        long_generation, "_revalidate_speaker_authority", reject
    )
    monkeypatch.setattr(
        h3, "start", lambda _request: pytest.fail("drifted producer must make zero H3 POST")
    )

    long_generation.run(settings, cid, plan)

    assert checks == [1]
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert stored["segments"][0]["error"] == "submission_unknown"


def test_long_plan_cas_and_detail_contract_do_not_expose_task_id(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["plan_receipt"] == receipt
    assert detail["segment_count"] == 1
    wrong = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload("0" * 64),
    )
    assert wrong.status_code == 409
    assert wrong.json() == {"detail": "long_video_plan_changed"}
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    def start(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "provider-task-secret")
    monkeypatch.setattr(h3, "start", start)
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    generation = client.get(f"/api/conversations/{cid}", headers=AUTH).json()["generation"]
    assert generation["segments"] == [{
        "index": 1, "chain_id": "chain-001", "join_mode": "hard_cut",
        "status": "succeeded", "attempt": 1, "error": None,
    }]
    assert "provider-task-secret" not in str(generation)


def test_legacy_long_detail_and_submit_derive_fit_from_frozen_h3_anchors(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(
        settings, fit_required=None, landscape_first_indices=(1,)
    )
    calls = []
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    def start(request):
        calls.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")

    monkeypatch.setattr(h3, "start", start)

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    rejected = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, fit="none"),
    )
    accepted = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, request_id="legacy-fit-request-2", fit="crop"),
    )

    assert detail.status_code == 200
    assert detail.json()["fit_required"] is True
    assert rejected.status_code == 422
    assert rejected.json() == {"detail": "fit_mode_required"}
    assert accepted.status_code == 202
    assert len(calls) == 1
    assert storage.load_meta(settings.data_dir, cid)["fit_required"] is None


def test_legacy_long_detail_derives_false_from_all_portrait_h3_anchors(enabled):
    settings, client = enabled
    cid, _receipt = _make_long(settings, fit_required=None)

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert detail.status_code == 200
    assert detail.json()["fit_required"] is False
    assert storage.load_meta(settings.data_dir, cid)["fit_required"] is None


def test_legacy_long_invalid_anchor_path_fails_closed_before_paid_submit(
    enabled, monkeypatch, tmp_path,
):
    settings, client = enabled
    cid, _receipt = _make_long(settings, fit_required=None)
    root = settings.data_dir / cid
    outside = tmp_path / "outside.png"
    _png(outside, 99, width=160, height=90)
    plan_path = root / long_video.PLAN_RECEIPT_FILENAME
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["segments"][0]["anchors"][0]["path"] = "../outside.png"
    payload["segments"][0]["anchors"][0]["sha256"] = hashlib.sha256(
        outside.read_bytes()
    ).hexdigest()
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(h3, "start", lambda request: calls.append(request))

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    submitted = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, fit="none"),
    )

    assert detail.status_code == 200
    assert detail.json()["fit_required"] is None
    assert submitted.status_code == 409
    assert submitted.json() == {"detail": "fit_requirement_unknown"}
    assert calls == []


@pytest.mark.parametrize(
    ("status", "fit_mode", "stored_required", "expected"),
    [
        ("running", "none", True, False),
        ("failed", "crop", False, True),
        ("resume_required", "pad", None, True),
    ],
)
def test_frozen_long_fit_uses_frozen_mode_without_reinterpreting_anchors(
    enabled, status, fit_mode, stored_required, expected,
):
    settings, client = enabled
    cid, receipt = _make_long(
        settings, fit_required=stored_required, landscape_first_indices=(1,)
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode=fit_mode,
        frozen_plan_receipt=receipt,
        generation={
            "status": status,
            "client_request_id": "frozen-request-1",
            "attempt": 1,
            "segments": [],
        },
    )
    meta = storage.load_meta(settings.data_dir, cid)

    effective = _long_fit_required(settings.data_dir / cid, meta)
    parsed = _validate_long_submit_payload(
        {**meta, "fit_required": effective},
        _payload(receipt, request_id="frozen-request-1", fit=fit_mode),
    )
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert effective is expected
    assert parsed[1] == fit_mode
    assert detail.status_code == 200
    assert detail.json()["fit_required"] is expected
    assert storage.load_meta(settings.data_dir, cid)["fit_required"] is stored_required


def test_legacy_long_ignores_continue_source_first_when_deriving_fit(enabled):
    settings, client = enabled
    cid, _receipt = _make_long(
        settings,
        joins=("hard_cut", "continue"),
        fit_required=None,
        landscape_first_indices=(2,),
    )

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert detail.status_code == 200
    assert detail.json()["fit_required"] is False


def test_legacy_long_uses_continue_end_when_deriving_fit(enabled):
    settings, client = enabled
    cid, _receipt = _make_long(
        settings,
        joins=("hard_cut", "continue"),
        fit_required=None,
        landscape_end_indices=(2,),
    )

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert detail.status_code == 200
    assert detail.json()["fit_required"] is True


def test_frozen_receipt_without_generation_still_derives_from_h3_anchors(enabled):
    settings, client = enabled
    cid, receipt = _make_long(
        settings, fit_required=None, landscape_first_indices=(1,)
    )
    storage.update_meta(
        settings.data_dir, cid, frozen_plan_receipt=receipt, fit_mode="none"
    )

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert detail.status_code == 200
    assert detail.json()["fit_required"] is True


def test_active_frozen_null_fit_change_reaches_locked_parameter_cas(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(
        settings, fit_required=None, landscape_first_indices=(1,)
    )
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "frozen-request-1", 1
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation={**generation, "status": "running"},
    )
    calls = []
    monkeypatch.setattr(h3, "start", lambda request: calls.append(request))

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(receipt, request_id="frozen-request-1", fit="crop"),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "resume_parameters_changed"}
    assert calls == []


@pytest.mark.parametrize("mode", ["auto", "none"])
def test_stale_long_page_requires_refresh_before_credentials_or_provider(
    tmp_path, monkeypatch, mode
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token=""
    )
    cid, _receipt = _make_long(settings)
    root = settings.data_dir / cid
    meta_path = root / "meta.json"
    receipt_path = root / long_video.PLAN_RECEIPT_FILENAME
    before_meta = meta_path.read_bytes()
    before_receipt = receipt_path.read_bytes()
    provider_calls = []
    monkeypatch.setattr(h3, "start", lambda request: provider_calls.append(request))

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/submit",
            headers=AUTH,
            json={
                "confirm": True,
                "client_request_id": "stale-page-request",
                "dialogue_mode": mode,
                "fit_mode": "none",
            },
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "code": "client_refresh_required",
            "message": "页面版本已更新，请刷新页面后重试。",
        }
    }
    assert provider_calls == []
    assert meta_path.read_bytes() == before_meta
    assert receipt_path.read_bytes() == before_receipt
    assert not (root / ".h3").exists()


def test_structured_submit_error_rejects_non_public_fields():
    with pytest.raises(TypeError, match="safe code and message"):
        _SubmitError(
            409,
            {
                "code": "client_refresh_required",
                "message": "safe",
                "provider_error": "must-not-escape",
            },
        )


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        (
            {
                "confirm": True,
                "client_request_id": "stale-page-request",
                "dialogue_mode": "auto",
                "fit_mode": "none",
                "unexpected": True,
            },
            "invalid_submit_request",
        ),
        (
            {
                "confirm": True,
                "client_request_id": "stale-page-request",
                "dialogue_mode": "edit",
                "fit_mode": "none",
            },
            "long_video_audio_mode_unsupported",
        ),
        (
            {
                "confirm": True,
                "client_request_id": "bad",
                "dialogue_mode": "none",
                "fit_mode": "none",
            },
            "invalid_submit_request",
        ),
        (
            {
                "confirm": True,
                "client_request_id": "stale-page-request",
                "dialogue_mode": "none",
                "fit_mode": "bad",
            },
            "invalid_submit_request",
        ),
    ],
)
def test_malformed_long_payloads_are_not_misclassified_as_stale_page(
    enabled, payload, expected_detail
):
    settings, client = enabled
    cid, _receipt = _make_long(settings)

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
    )

    assert response.status_code == 422
    assert response.json() == {"detail": expected_detail}


def test_15_seconds_one_post_and_none_rebuilds_prompt_without_source_dialogue(
    enabled, monkeypatch
):
    settings, client = enabled
    seen = []
    stitch_calls = []
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch(stitch_calls))
    def start(request):
        seen.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    monkeypatch.setattr(h3, "start", start)
    auto_cid, auto_receipt = _make_long(settings)
    none_cid, none_receipt = _make_long(settings)
    assert client.post(
        f"/api/conversations/{auto_cid}/submit", headers=AUTH,
        json=_payload(auto_receipt, mode="auto"),
    ).status_code == 202
    assert client.post(
        f"/api/conversations/{none_cid}/submit", headers=AUTH,
        json=_payload(none_receipt, request_id="parent-request-456", mode="none"),
    ).status_code == 202
    assert len(seen) == 2
    assert "源台词" in seen[0].prompt
    assert "源台词" not in seen[1].prompt
    assert "无台词" in seen[1].prompt
    assert seen[1].prompt.startswith("不要生成背景音乐\n")
    assert [call["audio_mode"] for call in stitch_calls] == ["mute", "mute"]


def test_30_second_continue_uses_each_segments_reference_frames(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    seen = []
    tail_bytes = settings.data_dir / cid / "work" / "tail-fixture.png"
    stale_tail = settings.data_dir / cid / "work" / "stale-tail.png"
    _png(tail_bytes, 222)
    _png(stale_tail, 111)
    generated_tail = (
        settings.data_dir / cid / "work" / "segments" / "1"
        / "work" / "generated_last.png"
    )
    generated_tail.parent.mkdir(parents=True, exist_ok=True)
    generated_tail.write_bytes(stale_tail.read_bytes())
    extract_calls = []
    def extract_tail(_video, output):
        extract_calls.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(tail_bytes.read_bytes())
        return output
    monkeypatch.setattr(long_generation, "_extract_last_frame", extract_tail)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    def start(request):
        seen.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", str(len(seen)))
    monkeypatch.setattr(h3, "start", start)
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    assert extract_calls == []
    assert [request.workdir.name for request in seen] == ["1", "2"]
    assert all(request.mode == "reference" for request in seen)
    assert seen[1].first_frame is None and seen[1].last_frame is None
    assert seen[1].keyframes


def test_new_parent_retry_reuses_segment_references_without_tail_extraction(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    root = settings.data_dir / cid
    fresh = []
    extract_calls = []

    def extract_tail(_video, output):
        extract_calls.append(output)
        fixture = root / "work" / f"fresh-tail-{len(extract_calls)}.png"
        _png(fixture, 170 + len(extract_calls))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(fixture.read_bytes())
        fresh.append(fixture.read_bytes())
        return output

    starts = []
    segment_two_attempts = 0

    def start(request):
        nonlocal segment_two_attempts
        starts.append(request)
        index = int(request.workdir.name)
        if index == 2:
            segment_two_attempts += 1
            if segment_two_attempts == 1:
                return h3.H3Result("failed", "task", error_code="h3_provider_failed")
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")

    monkeypatch.setattr(long_generation, "_extract_last_frame", extract_tail)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    monkeypatch.setattr(h3, "start", start)

    first = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt),
    )
    stale = root / "work" / "stale-retry-tail.png"
    _png(stale, 33)
    generated_tail = root / "work" / "segments" / "1" / "work" / "generated_last.png"
    generated_tail.write_bytes(stale.read_bytes())
    second = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, request_id="retry-parent-request-2"),
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert extract_calls == []
    second_segment_requests = [request for request in starts if request.workdir.name == "2"]
    assert len(second_segment_requests) == 2
    assert all(request.mode == "reference" for request in second_segment_requests)
    assert second_segment_requests[-1].keyframes


def test_reference_resume_does_not_require_frozen_continue_tail(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    plan.segments[0].workdir.joinpath("generated.mp4").write_bytes(b"segment")
    generation = long_generation.initial_generation(
        settings, cid, plan, "resume-parent-request", 1
    )
    generation["status"] = "resume_required"
    generation["segments"][0].update(
        status="succeeded", attempt=1, child_request_id="child-one"
    )
    generation["segments"][1].update(
        status="resume_required", attempt=1, child_request_id="child-two"
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    extracts = []
    starts = []
    resumes = []
    monkeypatch.setattr(
        long_generation, "_extract_last_frame",
        lambda *_args: extracts.append(1),
    )
    monkeypatch.setattr(h3, "start", lambda request: starts.append(request))
    monkeypatch.setattr(
        h3, "resume",
        lambda request: resumes.append(request)
        or h3.H3Result("retryable_failure", "child-two", retryable=True,
                       error_code="h3_query_failed"),
    )

    long_generation.run(settings, cid, plan)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert extracts == []
    assert starts == []
    assert len(resumes) == 1
    assert resumes[0].mode == "reference"
    assert stored["status"] == "resume_required"
    assert stored["client_request_id"] == "resume-parent-request"
    assert stored["segments"][1]["status"] == "resume_required"
    assert stored["segments"][1]["error"] == "h3_query_failed"
    assert stored["segments"][1]["child_request_id"] == long_generation.child_request_id(
        "resume-parent-request", receipt, 2
    )
    assert stored["segments"][1]["attempt"] == 1

    retry = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(receipt, request_id="retry-after-missing-tail"),
    )

    assert retry.status_code == 409
    assert retry.json() == {"detail": "resume_request_id_mismatch"}
    assert starts == []


def test_all_long_segments_and_continue_share_frozen_generation_parameters(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    seen = []
    tail = settings.data_dir / cid / "tail.png"
    _png(tail, 222)

    def extract_tail(_video, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(tail.read_bytes())
        return output

    def start(request):
        seen.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", str(len(seen)))

    monkeypatch.setattr(long_generation, "_extract_last_frame", extract_tail)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    monkeypatch.setattr(h3, "start", start)

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(
            receipt, fit="crop", aspect_ratio="16:9", resolution="480p"
        ),
    )

    assert response.status_code == 202
    assert len(seen) == 2
    assert {
        (request.aspect_ratio, request.resolution)
        for request in seen
    } == {("16:9", "480p")}
    assert {
        h3._input_manifest(request)["request"]["provider_resolution"]
        for request in seen
    } == {"480p横"}
    meta = storage.load_meta(settings.data_dir, cid)
    assert (meta["aspect_ratio"], meta["resolution"]) == ("16:9", "480p")


def test_plan_freeze_failure_makes_zero_posts(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    prompt = settings.data_dir / cid / "work" / "segments" / "1" / "work" / "prompt.txt"
    prompt.write_text("tampered", encoding="utf-8")
    calls = []
    monkeypatch.setattr(h3, "start", lambda request: calls.append(request))
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "long_video_plan_invalid"}
    assert calls == []
    assert meta["generation"] is None


def test_anchor_swap_before_bound_read_fails_sha_and_makes_zero_posts(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    anchor = (
        settings.data_dir / cid / "work" / "segments" / "1"
        / "work" / "anchors" / "first.png"
    )
    original = anchor.read_bytes()
    replacement_path = anchor.with_name("replacement.png")
    _png(replacement_path, 211)
    replacement = replacement_path.read_bytes()
    anchor.write_bytes(replacement)
    real_read_bytes = Path.read_bytes
    restored = False

    def swap_then_restore(path):
        nonlocal restored
        data = real_read_bytes(path)
        if path == anchor and not restored:
            anchor.write_bytes(original)
            restored = True
        return data

    monkeypatch.setattr(Path, "read_bytes", swap_then_restore)
    calls = []
    monkeypatch.setattr(h3, "start", lambda request: calls.append(request))

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "long_video_plan_invalid"}
    assert restored is True
    assert calls == []


def test_anchor_swap_after_bound_read_cannot_replace_reference_request_bytes(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    anchor = (
        settings.data_dir / cid / "work" / "segments" / "1"
        / "work" / "anchors" / "first.png"
    )
    original = anchor.read_bytes()
    keyframe = (
        settings.data_dir / cid / "work" / "segments" / "1"
        / "work" / "keyframes" / "01.png"
    )
    keyframe_bytes = keyframe.read_bytes()
    replacement_path = anchor.with_name("replacement.png")
    _png(replacement_path, 233)
    replacement = replacement_path.read_bytes()
    real_read_bytes = Path.read_bytes
    swapped = False

    def snapshot_then_swap(path):
        nonlocal swapped
        data = real_read_bytes(path)
        if path == anchor and not swapped:
            anchor.write_bytes(replacement)
            swapped = True
        return data

    monkeypatch.setattr(Path, "read_bytes", snapshot_then_swap)
    seen = []

    def start(request):
        anchor.write_bytes(original)
        seen.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")

    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt),
    )

    assert response.status_code == 202
    assert swapped is True
    assert len(seen) == 1
    assert seen[0].mode == "reference"
    assert seen[0].first_frame is None and seen[0].last_frame is None
    assert seen[0].keyframes[0] == (keyframe, keyframe_bytes)


def test_unknown_locks_batch_and_stops_continue_segment(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    calls = []
    monkeypatch.setattr(
        h3, "start",
        lambda request: calls.append(request) or h3.H3Result("submission_unknown", "secret"),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert response.status_code == 202
    assert len(calls) == 1
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["status"] == "submission_unknown"
    assert detail["generation"]["segments"][1]["status"] == "not_started"


def test_hard_cut_chains_run_with_global_concurrency_cap_two(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(
        settings, joins=("hard_cut", "hard_cut", "hard_cut")
    )
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls = []
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    def start(request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(int(request.workdir.name))
        time.sleep(0.03)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        with lock:
            active -= 1
        return h3.H3Result("succeeded", "task")

    monkeypatch.setattr(h3, "start", start)
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    assert sorted(calls) == [1, 2, 3]
    assert maximum == 2


def test_failed_new_parent_only_retries_failed_and_downstream(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue", "continue"))
    calls = []
    monkeypatch.setattr(long_generation, "_extract_last_frame", lambda _v, output: (
        output.parent.mkdir(parents=True, exist_ok=True) or _png(output, 200) or output
    ))
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    def start(request):
        index = int(request.workdir.name)
        calls.append(index)
        if calls == [1, 2]:
            return h3.H3Result("failed", "task", error_code="h3_provider_failed")
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    # First segment succeeds; second fails; third cannot start.
    first = True
    def sequenced(request):
        nonlocal first
        index = int(request.workdir.name)
        calls.append(index)
        if index == 2 and first:
            first = False
            return h3.H3Result("failed", "task", error_code="h3_provider_failed")
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    monkeypatch.setattr(h3, "start", sequenced)
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    assert calls == [1, 2]
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, request_id="parent-request-999"),
    ).status_code == 202
    assert calls == [1, 2, 2, 3]


def test_stitch_failure_retry_is_local_only(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    posts = []
    def start(request):
        posts.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", lambda **_kw: (_ for _ in ()).throw(RuntimeError()))
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    assert len(posts) == 1


def test_stitch_receipt_publish_failure_can_rebuild_over_existing_output_without_h3(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    posts = []

    def start(request):
        posts.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")

    stitch_attempts = []

    def publish_then_fail_once(**kwargs):
        stitch_attempts.append(kwargs)
        kwargs["output"].write_bytes(b"complete-local-output")
        if len(stitch_attempts) == 1:
            raise OSError("injected receipt publish failure")

    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", publish_then_fail_once)

    first = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert first.status_code == 202
    assert (settings.data_dir / cid / "generated.mp4").read_bytes() == b"complete-local-output"
    assert storage.load_meta(settings.data_dir, cid)["generation"]["stage"] == "stitch"

    retried = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert retried.status_code == 202
    assert len(stitch_attempts) == 2
    assert len(posts) == 1
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"


def test_detail_retry_paid_count_uses_segment_files_and_stitch_is_free(
    enabled,
):
    settings, client = enabled
    cid, _receipt = _make_long(settings, joins=("hard_cut", "hard_cut"))
    root = settings.data_dir / cid
    generation = {
        "status": "failed",
        "error": "long_video_segment_failed",
        "attempt": 1,
        "client_request_id": "parent-request-123",
        "stage": "h3",
        "segments": [
            {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut",
             "status": "succeeded", "attempt": 1, "error": None},
            {"index": 2, "chain_id": "chain-002", "join_mode": "hard_cut",
             "status": "succeeded", "attempt": 1, "error": None},
        ],
    }
    (root / "work" / "segments" / "2" / "generated.mp4").write_bytes(b"segment")
    storage.update_meta(settings.data_dir, cid, generation=generation)

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["retry_paid_segment_count"] == 2
    assert all("path" not in item and "task_id" not in item
               for item in detail["generation"]["segments"])

    generation.update(stage="stitch", error="long_video_stitch_failed")
    (root / "generated.mp4").write_bytes(b"previous-published-video")
    storage.update_meta(settings.data_dir, cid, generation=generation)
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["has_video"] is False
    assert detail["generation"]["retry_paid_segment_count"] == 2


def test_detail_retry_paid_count_uses_complete_frozen_segment_set(
    enabled,
):
    settings, client = enabled
    cid, _receipt = _make_long(
        settings, joins=("hard_cut", "hard_cut", "hard_cut")
    )
    root = settings.data_dir / cid
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=_receipt,
    )
    for index in (1, 2, 3):
        (root / "work" / "segments" / str(index) / "generated.mp4").write_bytes(
            b"segment"
        )

    def segment(index):
        return {
            "index": index,
            "chain_id": f"chain-{index:03d}",
            "join_mode": "hard_cut",
            "status": "succeeded",
            "attempt": 1,
            "error": None,
            "child_request_id": f"child-{index}",
        }

    cases = (
        ([segment(1), segment(3)], 3),
        ([segment(1), segment(1), segment(3)], 3),
        ([segment(2), segment(1), segment(3)], 3),
        ([segment(1), segment(2), segment(3)], 0),
    )
    for segments, expected in cases:
        storage.update_meta(settings.data_dir, cid, generation={
            "status": "failed",
            "error": "long_video_segment_failed",
            "attempt": 1,
            "client_request_id": "parent-request-123",
            "stage": "h3",
            "segments": segments,
        })
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
        assert detail["generation"]["retry_paid_segment_count"] == expected


def test_retry_initialization_uses_same_fail_closed_reuse_contract(enabled):
    settings, _client = enabled
    cid, receipt = _make_long(
        settings, joins=("hard_cut", "hard_cut", "hard_cut")
    )
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(root, meta, receipt, "none", "auto")
    for segment in plan.segments:
        (segment.workdir / "generated.mp4").write_bytes(b"segment")
    old = {
        "segments": [
            {"index": 2, "status": "succeeded", "attempt": 1},
            {"index": 1, "status": "succeeded", "attempt": 1},
            {"index": 3, "status": "succeeded", "attempt": 1},
        ]
    }
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-new", 2, old
    )
    assert [item["status"] for item in generation["segments"]] == [
        "not_started", "not_started", "not_started"
    ]


def test_startup_recovery_only_resumes_attempted_segments(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    plan = long_generation.freeze_plan(settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
                                       receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["segments"][0].update(status="running", attempt=1)
    storage.update_meta(settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
                        frozen_plan_receipt=receipt, generation=generation)
    resumed = []
    monkeypatch.setattr(h3, "resume", lambda request: resumed.append(request) or h3.H3Result("h3_running", "task"))
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("startup must not POST"))
    monkeypatch.setattr(long_generation, "_extract_last_frame", lambda _v, output: output)
    _resume_long_generation(settings, cid)
    assert [request.workdir.name for request in resumed] == ["1"]
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["segments"][1]["status"] == "not_started"


@pytest.mark.parametrize("fit_mode", ["crop", "pad"])
def test_current_first_failure_restart_uses_persisted_fit_layout_get_only(
    tmp_path, monkeypatch, fit_mode,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    client = TestClient(create_app(settings))
    cid, receipt = _make_long(
        settings, fit_required=True, landscape_first_indices=(1,)
    )
    submitted = []

    def first_start(request):
        submitted.append(request)
        return h3.H3Result(
            "retryable_failure",
            "000001",
            retryable=True,
            error_code="h3_query_failed",
        )

    monkeypatch.setattr(h3, "start", first_start)
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(receipt, fit=fit_mode),
    )
    assert response.status_code == 202
    assert len(submitted) == 1
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["generation"]["status"] == "resume_required"
    assert (meta["aspect_ratio"], meta["resolution"], meta["fit_mode"]) == (
        "9:16", "768p", fit_mode
    )
    assert meta["generation"]["fit_layout"] == long_generation.FIT_LAYOUT_ASPECT

    fit_base = (
        settings.data_dir / cid / "work" / "segments" / "1" / "work"
        / "h3_frames"
    )
    legacy_first = fit_base / fit_mode / "keyframes" / "01.png"
    semantic_first = fit_base / "9x16" / fit_mode / "keyframes" / "01.png"
    frozen_first = semantic_first
    other_first = legacy_first
    assert frozen_first.is_file()
    assert not other_first.exists()
    frozen_bytes = frozen_first.read_bytes()

    monkeypatch.setattr(
        long_generation.frame_fit,
        "fit_frames",
        lambda *_a, **_kw: pytest.fail("restart must not re-fit frozen inputs"),
    )
    monkeypatch.setattr(
        h3, "start", lambda _request: pytest.fail("restart must not POST")
    )
    resumed = []

    def resume(request):
        resumed.append(request)
        return h3.H3Result(
            "retryable_failure",
            "000001",
            retryable=True,
            error_code="h3_query_failed",
        )

    monkeypatch.setattr(h3, "resume", resume)
    _resume_long_generation(settings, cid)

    assert len(resumed) == 1
    assert resumed[0].keyframes[0] == (frozen_first, frozen_bytes)
    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["generation"]["status"] == "resume_required"
    assert (recovered["aspect_ratio"], recovered["resolution"]) == (
        "9:16", "768p"
    )


@pytest.mark.parametrize("fit_mode", ["crop", "pad"])
@pytest.mark.parametrize("layout", ["legacy", "semantic"])
def test_pre_marker_restart_discovers_existing_fit_layout_get_only(
    tmp_path, monkeypatch, fit_mode, layout,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art"
    )
    cid, receipt = _make_long(
        settings, fit_required=True, landscape_first_indices=(1,)
    )
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    if layout == "semantic":
        meta = storage.update_meta(
            settings.data_dir, cid, aspect_ratio="9:16", resolution="768p"
        )
        plan = long_generation.freeze_plan(
            root,
            meta,
            receipt,
            fit_mode,
            "auto",
            aspect_ratio="9:16",
            resolution="768p",
        )
    else:
        plan = long_generation.freeze_plan(
            root, meta, receipt, fit_mode, "auto"
        )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation.pop("fit_layout")
    generation["status"] = "resume_required"
    generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(
        settings.data_dir,
        cid,
        aspect_ratio="9:16",
        resolution="768p",
        fit_mode=fit_mode,
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation=generation,
    )

    fit_base = root / "work" / "segments" / "1" / "work" / "h3_frames"
    legacy_first = fit_base / fit_mode / "keyframes" / "01.png"
    semantic_first = fit_base / "9x16" / fit_mode / "keyframes" / "01.png"
    frozen_first = legacy_first if layout == "legacy" else semantic_first
    other_first = semantic_first if layout == "legacy" else legacy_first
    assert frozen_first.is_file()
    assert not other_first.exists()
    frozen_bytes = frozen_first.read_bytes()

    monkeypatch.setattr(
        long_generation.frame_fit,
        "fit_frames",
        lambda *_a, **_kw: pytest.fail("restart must not re-fit frozen inputs"),
    )
    monkeypatch.setattr(
        h3, "start", lambda _request: pytest.fail("restart must not POST")
    )
    resumed = []

    def resume(request):
        resumed.append(request)
        return h3.H3Result(
            "retryable_failure",
            "000001",
            retryable=True,
            error_code="h3_query_failed",
        )

    monkeypatch.setattr(h3, "resume", resume)
    _resume_long_generation(settings, cid)

    assert len(resumed) == 1
    assert resumed[0].keyframes[0] == (frozen_first, frozen_bytes)
    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["generation"]["status"] == "resume_required"
    assert (recovered["aspect_ratio"], recovered["resolution"]) == (
        "9:16", "768p"
    )


def test_active_same_parent_is_idempotent_without_refreeze_or_second_coordinator(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
        receipt, "none", "auto",
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["status"] = "running"
    generation["segments"][0]["status"] = "running"
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(
        long_generation, "freeze_plan",
        lambda *_a, **_kw: pytest.fail("active idempotent replay must not refreeze"),
    )
    monkeypatch.setattr(
        long_generation, "run",
        lambda *_a, **_kw: pytest.fail("active idempotent replay must not reschedule"),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert response.status_code == 202
    assert response.json() == {"status": "running", "attempt": 1}


def test_startup_uses_frozen_receipt_cas_and_fails_closed_on_plan_drift(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
        receipt, "none", "auto",
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["segments"][0].update(status="running", attempt=1)
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    receipt_path = settings.data_dir / cid / long_video.PLAN_RECEIPT_FILENAME
    receipt_path.write_text(receipt_path.read_text() + " ", encoding="utf-8")
    monkeypatch.setattr(h3, "resume", lambda _request: pytest.fail("drifted plan must not GET"))
    _resume_long_generation(settings, cid)
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "submission_unknown"


def test_startup_aggregates_resume_required_and_never_queues_unstarted(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
        receipt, "none", "auto",
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["segments"][0].update(status="running", attempt=1)
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    monkeypatch.setattr(h3, "resume", lambda _request: h3.H3Result("h3_running", "task"))
    _resume_long_generation(settings, cid)
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "resume_required"
    assert stored["segments"][1]["status"] == "not_started"


def test_resume_same_child_does_not_increment_segment_attempt(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(settings.data_dir / cid, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["status"] = "resume_required"
    generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
                        frozen_plan_receipt=receipt, generation=generation)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    def resume(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("resume must not POST"))
    monkeypatch.setattr(h3, "resume", resume)
    long_generation.run(settings, cid, plan)
    assert storage.load_meta(settings.data_dir, cid)["generation"]["segments"][0]["attempt"] == 1


@pytest.mark.parametrize(
    ("persisted_delivery", "expected_delivery"),
    [("off_screen", "off_screen"), (None, "auto")],
)
def test_long_startup_freeze_uses_persisted_dialogue_delivery(
    tmp_path, monkeypatch, persisted_delivery, expected_delivery,
):
    settings = make_settings(tmp_path)
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(root, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["status"] = "resume_required"
    generation["segments"][0]["status"] = "resume_required"
    changes = {
        "fit_mode": "none",
        "dialogue_mode": "auto",
        "frozen_plan_receipt": receipt,
        "generation": generation,
    }
    if persisted_delivery is not None:
        changes["dialogue_delivery"] = persisted_delivery
    storage.update_meta(settings.data_dir, cid, **changes)
    observed = []

    def freeze(*_args, **kwargs):
        observed.append(kwargs.get("dialogue_delivery"))
        return plan

    monkeypatch.setattr(long_generation, "freeze_plan", freeze)
    monkeypatch.setattr(
        long_generation,
        "run",
        lambda *_args, **kwargs: observed.append(kwargs.get("startup")),
    )

    _resume_long_generation(settings, cid)

    assert observed == [expected_delivery, True]


def test_boundary_reuse_validation_receives_original_segment_target(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, segment_duration=10.84, legacy=True)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(settings.data_dir / cid, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["segments"][0].update(
        status="succeeded", attempt=1, child_request_id="child-1"
    )
    plan.segments[0].workdir.joinpath("generated.mp4").write_bytes(b"segment")
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        generation=generation,
    )
    targets = []

    def reusable(_request, *_args, expected_duration_s=None, **_kwargs):
        targets.append(expected_duration_s)
        return True

    monkeypatch.setattr(h3, "output_is_reusable", reusable)

    assert long_generation.bound_reusable_segment_indices(
        settings, cid, plan, generation
    ) == frozenset({1})
    assert targets == [10.84]


@pytest.mark.parametrize(
    ("receipt_version", "expected_duration_s"),
    [
        (long_video.PLAN_RECEIPT_VERSION, 10.0),
        (long_video.LEGACY_PLAN_RECEIPT_VERSION, 37.52 - 27.52),
    ],
)
def test_boundary_reuse_uses_receipt_version_segment_duration(
    tmp_path, monkeypatch, receipt_version, expected_duration_s,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, meta, receipt, "none", "auto"
    )
    plan = _with_boundary_bounds(plan, receipt_version=receipt_version)
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["segments"][0].update(
        status="succeeded", attempt=1, child_request_id="child-1"
    )
    plan.segments[0].workdir.joinpath("generated.mp4").write_bytes(b"segment")
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        generation=generation,
    )
    targets = []
    monkeypatch.setattr(
        h3,
        "output_is_reusable",
        lambda _request, *_args, expected_duration_s=None, **_kwargs: (
            targets.append(expected_duration_s) or True
        ),
    )

    assert long_generation.bound_reusable_segment_indices(
        settings, cid, plan, generation
    ) == frozenset({1})
    assert targets == [expected_duration_s]


@pytest.mark.parametrize(
    ("receipt_version", "expected_duration_s"),
    [
        (long_video.PLAN_RECEIPT_VERSION, 10.0),
        (long_video.LEGACY_PLAN_RECEIPT_VERSION, 37.52 - 27.52),
    ],
)
def test_stitch_input_uses_receipt_version_segment_duration(
    tmp_path, monkeypatch, receipt_version, expected_duration_s,
):
    settings = make_settings(tmp_path)
    cid, receipt = _make_long(settings)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid,
        storage.load_meta(settings.data_dir, cid),
        receipt,
        "none",
        "auto",
    )
    plan = _with_boundary_bounds(plan, receipt_version=receipt_version)
    calls = []
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch(calls))
    monkeypatch.setattr(long_generation, "stitched_output_is_reusable", lambda *_a: True)

    long_generation._stitch(settings, cid, plan, "auto")

    assert len(calls) == 1
    assert calls[0]["segments"][0].target_duration_s == expected_duration_s


@pytest.mark.parametrize(
    ("receipt_version", "expected_duration_s", "resume"),
    [
        (long_video.PLAN_RECEIPT_VERSION, 10.0, False),
        (long_video.LEGACY_PLAN_RECEIPT_VERSION, 37.52 - 27.52, True),
    ],
)
def test_success_validation_uses_receipt_version_segment_duration(
    tmp_path, monkeypatch, receipt_version, expected_duration_s, resume,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid,
        storage.load_meta(settings.data_dir, cid),
        receipt,
        "none",
        "auto",
    )
    plan = _with_boundary_bounds(plan, receipt_version=receipt_version)
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    if resume:
        generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        generation=generation,
    )
    targets = []

    def succeed(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")

    monkeypatch.setattr(
        h3,
        "start",
        (lambda _request: pytest.fail("legacy recovery must not POST"))
        if resume
        else succeed,
    )
    monkeypatch.setattr(
        h3,
        "resume",
        succeed if resume else lambda _request: pytest.fail("new task must POST"),
    )
    monkeypatch.setattr(
        h3,
        "output_is_reusable",
        lambda _request, *_args, expected_duration_s=None, **_kwargs: (
            targets.append(expected_duration_s) or True
        ),
    )
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan)

    assert targets == [expected_duration_s]
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"


def test_startup_revalidates_previous_output_invalid_without_new_provider_call(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(
        settings, segment_duration=10.84, legacy=True
    )
    root = settings.data_dir / cid
    plan = long_generation.freeze_plan(
        root, storage.load_meta(settings.data_dir, cid), receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["status"] = "failed"
    generation["error"] = "long_video_segment_failed"
    generation["segments"][0].update(
        status="failed",
        attempt=1,
        error="long_video_segment_output_invalid",
        child_request_id="child-1",
    )
    plan.segments[0].workdir.joinpath("generated.mp4").write_bytes(b"paid-segment")
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        generation=generation,
    )
    targets = []
    monkeypatch.setattr(
        h3,
        "output_is_reusable",
        lambda _request, *_args, expected_duration_s=None, **_kwargs: (
            targets.append(expected_duration_s) or True
        ),
    )
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not POST"))
    monkeypatch.setattr(h3, "resume", lambda _request: pytest.fail("must not GET"))
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan, startup=True)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert targets == [10.84]
    assert stored["segments"][0]["status"] == "succeeded"
    assert stored["status"] == "succeeded"


def test_legacy_over_ten_segment_never_starts_a_new_paid_attempt(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, segment_duration=11.0, legacy=True)
    root = settings.data_dir / cid
    plan = long_generation.freeze_plan(
        root, storage.load_meta(settings.data_dir, cid), receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        generation=generation,
    )
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("legacy plan must not POST"))

    long_generation.run(settings, cid, plan)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "failed"
    assert stored["segments"][0]["error"] == "long_video_legacy_plan_read_only"


@pytest.mark.parametrize("duration", [11.0, 13.0, 15.0])
def test_legacy_boundary_attempts_up_to_fifteen_seconds_remain_get_only_recoverable(
    tmp_path, monkeypatch, duration,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, segment_duration=duration, legacy=True)
    root = settings.data_dir / cid
    plan = long_generation.freeze_plan(
        root, storage.load_meta(settings.data_dir, cid), receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        generation=generation,
    )
    resumed = []
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("recovery must not POST"))
    monkeypatch.setattr(
        h3, "resume",
        lambda request: resumed.append(request) or h3.H3Result("h3_running", "task"),
    )

    long_generation.run(settings, cid, plan)

    assert [request.duration for request in resumed] == [int(duration)]


def test_long_resume_required_missing_attempt_locks_batch_without_post(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(settings.data_dir / cid, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["status"] = "resume_required"
    generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    calls = []
    monkeypatch.setattr(h3, "start", lambda _request: calls.append("start"))
    monkeypatch.setattr(
        h3, "resume",
        lambda _request: calls.append("resume") or h3.H3Result("not_started", None),
    )
    long_generation.run(settings, cid, plan)
    assert calls == ["resume"]
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "submission_unknown"


def test_new_receipt_rejects_segment_whose_provider_duration_exceeds_fourteen(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    with pytest.raises(long_video.LongVideoError, match="long_video_plan_invalid_segment"):
        _make_long(
            settings,
            joins=("hard_cut", "continue"),
            segment_duration=14.000001,
        )


def test_new_receipt_rejects_true_six_decimal_subsecond_segment(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    joins = ("hard_cut",) + ("continue",) * 15

    with pytest.raises(
        long_video.LongVideoError,
        match="long_video_plan_invalid_segment",
    ):
        _make_long(settings, joins=joins, segment_duration=0.999999)


def test_freeze_rejects_new_multisegment_receipt_with_over_fourteen_provider_duration(
    tmp_path,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, _receipt = _make_long(
        settings,
        joins=("hard_cut", "hard_cut"),
        segment_duration=15.0,
        legacy=True,
    )
    root = settings.data_dir / cid
    path = root / long_video.PLAN_RECEIPT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = long_video.PLAN_RECEIPT_VERSION
    path.write_bytes(long_video._canonical_bytes(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        long_generation.LongGenerationError, match="long_video_plan_invalid"
    ):
        long_generation.freeze_plan(
            root,
            storage.load_meta(settings.data_dir, cid),
            digest,
            "none",
            "auto",
        )


def test_freeze_accepts_new_single_segment_at_fifteen_seconds(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, segment_duration=15.0)
    root = settings.data_dir / cid

    frozen = long_generation.freeze_plan(
        root,
        storage.load_meta(settings.data_dir, cid),
        receipt,
        "none",
        "auto",
    )

    assert len(frozen.segments) == 1
    assert long_video.provider_duration_s(
        frozen.segments[0].start_s,
        frozen.segments[0].end_s,
        receipt_version=frozen.receipt_version,
    ) == 15


@pytest.mark.parametrize(
    "duration", [math.nextafter(15.0, math.inf)]
)
def test_positive_float_boundary_overflow_plans_and_freezes_provider_safe_segments(
    tmp_path, duration
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, duration=duration)

    frozen = long_generation.freeze_plan(
        settings.data_dir / cid,
        storage.load_meta(settings.data_dir, cid),
        receipt,
        "none",
        "auto",
    )

    assert frozen.segments
    assert all(
        long_video.provider_duration_s(item.start_s, item.end_s) <= 14
        for item in frozen.segments
    )


def test_every_new_long_generation_request_is_at_most_fourteen_seconds(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, duration=31.0)
    root = settings.data_dir / cid
    plan = long_generation.freeze_plan(
        root, storage.load_meta(settings.data_dir, cid), receipt, "none", "auto"
    )
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    requests = []

    def start(request):
        requests.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", f"task-{request.duration}")

    def extract_tail(_video, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"tail-frame")
        return output

    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(long_generation, "_extract_last_frame", extract_tail)
    monkeypatch.setattr(long_generation, "_fit_anchor", lambda path, _output, _fit: path)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))

    long_generation.run(settings, cid, plan)

    assert requests
    assert all(request.duration <= 14 for request in requests)


def test_legacy_binary_float_duration_rebuilds_original_thirteen_second_request(
    tmp_path,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    root = tmp_path / "legacy"
    root.mkdir()
    anchor = root / "anchor.png"
    anchor.write_bytes(b"anchor")
    segment = long_generation.FrozenSegment(
        index=1,
        start_s=25.52,
        end_s=37.52,
        chain_id="chain-001",
        join_mode="hard_cut",
        workdir=root,
        first_frame=anchor,
        first_frame_data=b"anchor",
        last_frame=anchor,
        last_frame_data=b"anchor",
        prompt="prompt",
    )
    plan = long_generation.FrozenPlan(
        root=root,
        source=root / "source.mp4",
        receipt="legacy",
        segments=(segment,),
        receipt_version=long_video.LEGACY_PLAN_RECEIPT_VERSION,
        workflow=h3.H3_BOUNDARY_WORKFLOW,
    )

    request = long_generation._request(
        settings, "cid", plan, segment, "parent-request-123", "none"
    )

    assert request.duration == 13


def test_reference_continue_does_not_depend_on_generated_tail(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    def start(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    monkeypatch.setattr(h3, "start", start)
    monkeypatch.setattr(
        long_generation, "_extract_last_frame",
        lambda *_a: (_ for _ in ()).throw(long_generation.LongGenerationError("tail_failed")),
    )
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    ).status_code == 202
    generation = client.get(f"/api/conversations/{cid}", headers=AUTH).json()["generation"]
    assert generation["status"] == "succeeded"
    assert all(item["status"] == "succeeded" for item in generation["segments"])


def test_resume_required_segment_runs_at_most_once_per_coordinator_invocation(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(settings.data_dir / cid, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    storage.update_meta(
        settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
        frozen_plan_receipt=receipt, generation=generation,
    )
    calls = 0
    stop = threading.Event()

    def query_failure(_request):
        nonlocal calls
        calls += 1
        if stop.is_set():
            return h3.H3Result("failed", "task", error_code="h3_provider_failed")
        return h3.H3Result(
            "retryable_failure", "task", retryable=True,
            error_code="h3_query_failed",
        )

    monkeypatch.setattr(h3, "start", query_failure)
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(long_generation.run, settings, cid, plan)
    timed_out = False
    try:
        future.result(timeout=0.2)
    except TimeoutError:
        timed_out = True
        stop.set()
        future.result(timeout=1)
    finally:
        pool.shutdown()
    assert not timed_out
    assert calls == 1
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "resume_required"


@pytest.mark.parametrize(
    "changed", ["plan", "dialogue", "fit", "aspect_ratio", "resolution"]
)
def test_failed_retry_rejects_all_frozen_parameter_changes_with_zero_new_posts(
    enabled, monkeypatch, changed
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    if changed == "fit":
        storage.update_meta(settings.data_dir, cid, fit_required=True)
    calls = []
    monkeypatch.setattr(long_generation, "_extract_last_frame", lambda _v, output: (
        output.parent.mkdir(parents=True, exist_ok=True) or _png(output, 200) or output
    ))

    def start(request):
        index = int(request.workdir.name)
        calls.append(index)
        if index == 2:
            return h3.H3Result("failed", "task", error_code="h3_provider_failed")
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")

    monkeypatch.setattr(h3, "start", start)
    initial_fit = "crop" if changed == "fit" else "none"
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(receipt, fit=initial_fit),
    ).status_code == 202
    assert calls == [1, 2]
    calls.clear()

    retry_receipt = receipt
    retry_mode = "auto"
    retry_fit = initial_fit
    retry_aspect = "9:16"
    retry_resolution = "768p"
    if changed == "plan":
        path = settings.data_dir / cid / long_video.PLAN_RECEIPT_FILENAME
        path.write_text(path.read_text() + " ", encoding="utf-8")
        retry_receipt = hashlib.sha256(path.read_bytes()).hexdigest()
    elif changed == "dialogue":
        retry_mode = "none"
    elif changed == "fit":
        retry_fit = "pad"
    elif changed == "aspect_ratio":
        retry_aspect = "16:9"
        retry_fit = "crop"
    else:
        retry_resolution = "480p"
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(
            retry_receipt, request_id="parent-request-999",
            mode=retry_mode, fit=retry_fit,
            aspect_ratio=retry_aspect, resolution=retry_resolution,
        ),
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "resume_parameters_changed"}
    assert calls == []


def test_plan_receipt_read_error_degrades_to_none(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cid, _receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    target = settings.data_dir / cid / long_video.PLAN_RECEIPT_FILENAME
    original = Path.read_bytes

    def fail_target(path):
        if path == target:
            raise OSError("unreadable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_target)
    assert long_generation.plan_receipt(settings.data_dir / cid, meta) is None


def test_long_succeeded_missing_top_output_only_restitches_valid_bound_segments(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "hard_cut"))
    root = settings.data_dir / cid
    segments = []
    for index in (1, 2):
        (root / "work" / "segments" / str(index) / "generated.mp4").write_bytes(
            b"segment"
        )
        segments.append({
            "index": index,
            "chain_id": f"chain-{index:03d}",
            "join_mode": "hard_cut",
            "status": "succeeded",
            "attempt": 1,
            "error": None,
            "child_request_id": f"child-{index}",
        })
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation={
            "status": "succeeded",
            "error": None,
            "attempt": 1,
            "client_request_id": "parent-request-123",
            "stage": "stitch",
            "segments": segments,
        },
    )
    stitch_calls = []
    monkeypatch.setattr(
        long_generation.stitch, "stitch_video", _fake_stitch(stitch_calls)
    )
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not POST"))
    monkeypatch.setattr(h3, "resume", lambda _request: pytest.fail("must not GET"))
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert response.status_code == 202
    assert len(stitch_calls) == 1
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"
    replay = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert replay.status_code == 202
    assert replay.json()["status"] == "succeeded"
    assert len(stitch_calls) == 1


def test_stitch_retry_with_invalid_segments_requires_new_paid_confirmation(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    (root / "work" / "segments" / "1" / "generated.mp4").write_bytes(b"broken")
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation={
            "status": "failed",
            "error": "long_video_stitch_failed",
            "attempt": 1,
            "client_request_id": "parent-request-123",
            "stage": "stitch",
            "segments": [{
                "index": 1,
                "chain_id": "chain-001",
                "join_mode": "hard_cut",
                "status": "succeeded",
                "attempt": 1,
                "error": None,
                "child_request_id": "child-1",
            }],
        },
    )
    monkeypatch.setattr(h3, "output_is_reusable", lambda *_a, **_kw: False)
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not POST"))
    monkeypatch.setattr(h3, "resume", lambda _request: pytest.fail("must not GET"))
    monkeypatch.setattr(
        long_generation.stitch, "stitch_video",
        lambda **_kwargs: pytest.fail("invalid segment must not stitch"),
    )
    first = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert first.status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "failed"
    assert generation["stage"] == "h3"
    assert generation["segments"][0]["error"] == "long_video_segment_output_invalid"
    second = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert second.status_code == 409
    assert second.json() == {"detail": "new client_request_id required"}


def test_long_resume_corrupt_plan_converges_to_unknown(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(settings.data_dir / cid, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation["status"] = "resume_required"
    generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation=generation,
    )
    path = settings.data_dir / cid / long_video.PLAN_RECEIPT_FILENAME
    path.write_text(path.read_text() + " ", encoding="utf-8")
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not POST"))
    monkeypatch.setattr(h3, "resume", lambda _request: pytest.fail("must not GET"))
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(receipt)
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "submission_outcome_unknown"}
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "submission_unknown"


@pytest.mark.parametrize("entrypoint", ["startup", "submit"])
@pytest.mark.parametrize("damage", ["missing", "duplicate", "reordered"])
def test_long_succeeded_recovery_rejects_malformed_segment_state_atomically(
    enabled, monkeypatch, entrypoint, damage
):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "hard_cut"))
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(root, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation.update(status="succeeded", stage="stitch")
    for item in generation["segments"]:
        item.update(
            status="succeeded",
            attempt=1,
            child_request_id=f"child-{item['index']}",
        )
        (root / "work" / "segments" / str(item["index"]) / "generated.mp4").write_bytes(
            f"segment-{item['index']}".encode()
        )
    if damage == "missing":
        generation["segments"] = generation["segments"][:1]
    elif damage == "duplicate":
        generation["segments"][1] = dict(generation["segments"][0])
    else:
        generation["segments"].reverse()
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation=generation,
    )
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != "meta.json"
    }
    provider_calls = []
    monkeypatch.setattr(h3, "start", lambda request: provider_calls.append("POST"))
    monkeypatch.setattr(h3, "resume", lambda request: provider_calls.append("GET"))

    if entrypoint == "startup":
        _resume_long_generation(settings, cid)
    else:
        response = client.post(
            f"/api/conversations/{cid}/submit",
            headers=AUTH,
            json=_payload(receipt),
        )
        assert response.status_code == 409
        assert response.json() == {"detail": "submission_outcome_unknown"}

    recovered = storage.load_meta(settings.data_dir, cid)["generation"]
    assert recovered["status"] == "submission_unknown"
    assert recovered["error"] == "submission_unknown"
    assert recovered["attempt"] == 1
    assert provider_calls == []
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != "meta.json"
    }
    assert after == before


@pytest.mark.parametrize(
    "damage", ["missing_receipt", "corrupt_receipt", "zero", "tampered", "wrong_duration"]
)
def test_stitched_output_reuse_validates_receipt_bytes_and_target_duration(
    tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(
        long_generation,
        "stitched_output_is_reusable",
        _REAL_STITCHED_OUTPUT_IS_REUSABLE,
    )
    root = tmp_path / "conversation"
    segment_root = root / "work" / "segments" / "1"
    segment_root.mkdir(parents=True)
    source = root / "source.mp4"
    segment_video = segment_root / "generated.mp4"
    _small_video(source)
    _small_video(segment_video, "blue")
    anchor = segment_root / "anchor.png"
    anchor.write_bytes(b"unused")
    plan = long_generation.FrozenPlan(
        root=root,
        source=source,
        receipt="frozen-plan",
        segments=(long_generation.FrozenSegment(
            index=1,
            start_s=0.0,
            end_s=11.0,
            chain_id="chain-001",
            join_mode="hard_cut",
            workdir=segment_root,
            first_frame=anchor,
            first_frame_data=b"unused",
            last_frame=anchor,
            last_frame_data=b"unused",
            prompt="prompt",
        ),),
    )
    long_generation._stitch(None, "unused", plan, "none")
    assert _REAL_STITCHED_OUTPUT_IS_REUSABLE(plan, "none") is True
    output = root / "generated.mp4"
    receipt_path = root / long_generation.stitch.RECEIPT_FILENAME
    if damage == "missing_receipt":
        receipt_path.unlink()
    elif damage == "corrupt_receipt":
        receipt_path.write_bytes(b"\xff")
    elif damage == "zero":
        output.write_bytes(b"")
    elif damage == "tampered":
        data = output.read_bytes()
        output.write_bytes(data[:-32] + b"x" * 32)
    else:
        _small_video(output, "red")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["output"].update(
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            size=output.stat().st_size,
            duration_s=1.0,
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert _REAL_STITCHED_OUTPUT_IS_REUSABLE(plan, "none") is False


@pytest.mark.parametrize(
    ("receipt_version", "expected_duration_s"),
    [
        (long_video.PLAN_RECEIPT_VERSION, 10.0),
        (long_video.LEGACY_PLAN_RECEIPT_VERSION, 37.52 - 27.52),
    ],
)
def test_stitched_receipt_and_success_validation_use_receipt_version_duration(
    tmp_path, monkeypatch, receipt_version, expected_duration_s,
):
    monkeypatch.setattr(
        long_generation,
        "stitched_output_is_reusable",
        _REAL_STITCHED_OUTPUT_IS_REUSABLE,
    )
    settings = make_settings(tmp_path)
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    plan = long_generation.freeze_plan(
        root,
        storage.load_meta(settings.data_dir, cid),
        receipt,
        "none",
        "auto",
    )
    plan = _with_boundary_bounds(plan, receipt_version=receipt_version)
    segment_output = plan.segments[0].workdir / "generated.mp4"
    segment_output.write_bytes(b"segment-output")
    output = root / "generated.mp4"
    output.write_bytes(b"stitched-output")
    receipt_path = root / stitch.RECEIPT_FILENAME
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "duet.stitch",
                "version": 1,
                "segments": [
                    {
                        "index": 1,
                        "path": str(segment_output.resolve()),
                        "sha256": hashlib.sha256(
                            segment_output.read_bytes()
                        ).hexdigest(),
                        "target_duration_s": expected_duration_s,
                        "output_frames": 240,
                        "join_mode": "hard_cut",
                    }
                ],
                "audio": {
                    "mode": "mute",
                    "source": str(plan.source.resolve()),
                    "source_sha256": hashlib.sha256(
                        plan.source.read_bytes()
                    ).hexdigest(),
                    "source_has_audio": False,
                },
                "output": {
                    "name": "generated.mp4",
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "size": output.stat().st_size,
                    "duration_s": expected_duration_s,
                    "fps": stitch.FPS,
                },
            }
        ),
        encoding="utf-8",
    )
    observed = {"budgets": [], "validation": []}

    def frame_budgets(segments):
        observed["budgets"].append(
            [segment.target_duration_s for segment in segments]
        )
        return [240]

    def validate_output(_output, duration_s, audio_mode, source_has_audio):
        observed["validation"].append(
            (duration_s, audio_mode, source_has_audio)
        )
        return SimpleNamespace(duration_s=expected_duration_s)

    monkeypatch.setattr(stitch, "_frame_budgets", frame_budgets)
    monkeypatch.setattr(
        stitch,
        "_probe",
        lambda _path: SimpleNamespace(has_audio=False),
    )
    monkeypatch.setattr(stitch, "_validate_output", validate_output)

    assert _REAL_STITCHED_OUTPUT_IS_REUSABLE(plan, "none") is True
    assert observed == {
        "budgets": [[expected_duration_s]],
        "validation": [(expected_duration_s, "mute", False)],
    }


@pytest.mark.parametrize("entrypoint", ["startup", "submit"])
@pytest.mark.parametrize("damage", ["missing", "zero", "tampered", "receipt"])
def test_long_invalid_top_output_is_hidden_and_restitched_without_provider(
    enabled, monkeypatch, entrypoint, damage
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    root = settings.data_dir / cid
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(root, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(
        settings, cid, plan, "parent-request-123", 1
    )
    generation.update(status="succeeded", stage="stitch")
    segment = generation["segments"][0]
    segment.update(status="succeeded", attempt=1, child_request_id="child-1")
    plan.segments[0].workdir.joinpath("generated.mp4").write_bytes(b"segment")
    output = root / "generated.mp4"
    top_receipt = root / long_generation.stitch.RECEIPT_FILENAME
    output.write_bytes(b"valid-top")
    top_receipt.write_bytes(b"valid-receipt")
    if damage == "missing":
        output.unlink()
    elif damage == "zero":
        output.write_bytes(b"")
    elif damage == "tampered":
        output.write_bytes(b"tampered")
    else:
        top_receipt.write_bytes(b"broken")
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        frozen_plan_receipt=receipt,
        generation=generation,
    )

    def top_is_valid(current_plan, _dialogue_mode):
        try:
            return (
                current_plan.root.joinpath("generated.mp4").read_bytes() == b"valid-top"
                and current_plan.root.joinpath(
                    long_generation.stitch.RECEIPT_FILENAME
                ).read_bytes() == b"valid-receipt"
            )
        except OSError:
            return False

    stitch_calls = []

    def restitch(**kwargs):
        stitch_calls.append(kwargs)
        kwargs["output"].write_bytes(b"valid-top")
        kwargs["output"].with_name(
            long_generation.stitch.RECEIPT_FILENAME
        ).write_bytes(b"valid-receipt")

    monkeypatch.setattr(
        long_generation, "stitched_output_is_reusable", top_is_valid
    )
    monkeypatch.setattr(long_generation.stitch, "stitch_video", restitch)
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not POST"))
    monkeypatch.setattr(h3, "resume", lambda _request: pytest.fail("must not GET"))
    assert client.get(f"/api/conversations/{cid}", headers=AUTH).json()["has_video"] is False
    listed = client.get("/api/conversations", headers=AUTH).json()
    assert next(item for item in listed if item["id"] == cid)["has_video"] is False

    if entrypoint == "startup":
        _resume_long_generation(settings, cid)
    else:
        response = client.post(
            f"/api/conversations/{cid}/submit",
            headers=AUTH,
            json=_payload(receipt),
        )
        assert response.status_code == 202

    assert len(stitch_calls) == 1
    recovered = storage.load_meta(settings.data_dir, cid)["generation"]
    assert recovered["status"] == "succeeded"
    assert recovered["attempt"] == 1
    assert client.get(f"/api/conversations/{cid}", headers=AUTH).json()["has_video"] is True
    listed = client.get("/api/conversations", headers=AUTH).json()
    assert next(item for item in listed if item["id"] == cid)["has_video"] is True


def _native_audio_plan(settings, *, segment_count: int = 2):
    meta = storage.new_conversation(settings.data_dir, "native audio", "source.mp4")
    cid = meta["id"]
    root = settings.data_dir / cid
    source = root / "source.mp4"
    source.write_bytes(b"source")
    segments = []
    states = []
    for index in range(1, segment_count + 1):
        workdir = root / "work" / "segments" / str(index)
        workdir.mkdir(parents=True, exist_ok=True)
        anchor = workdir / "anchor.png"
        segments.append(long_generation.FrozenSegment(
            index=index,
            start_s=float(index - 1),
            end_s=float(index),
            chain_id="chain-001",
            join_mode="hard_cut" if index == 1 else "continue",
            workdir=workdir,
            first_frame=anchor,
            first_frame_data=b"frame",
            last_frame=anchor,
            last_frame_data=b"frame",
            prompt=f"segment {index}",
        ))
        states.append({
            "index": index,
            "chain_id": "chain-001",
            "join_mode": "hard_cut" if index == 1 else "continue",
            "status": "succeeded",
            "attempt": 1,
            "error": None,
            "child_request_id": f"child-{index}",
            "h3_attempt_id": f"{index:06d}",
            "context_ir": {"status": "succeeded"},
        })
    plan = long_generation.FrozenPlan(
        root=root,
        source=source,
        receipt="frozen-plan",
        segments=tuple(segments),
        workflow="test-h3-multimodal",
    )
    generation = {
        "status": "succeeded",
        "error": None,
        "attempt": 1,
        "client_request_id": "parent-request",
        "stage": "h3",
        "fit_layout": long_generation.FIT_LAYOUT_ASPECT,
        "fast_mode": False,
        "workflow": plan.workflow,
        "audio_route": dict(long_generation.H3_NATIVE_AUDIO_ROUTE),
        "segments": states,
    }
    storage.update_meta(
        settings.data_dir,
        cid,
        fit_mode="none",
        dialogue_mode="auto",
        aspect_ratio="9:16",
        resolution="768p",
        generation=generation,
    )
    return cid, plan, generation


def _make_tone_video(path: Path, *, color: str, frequency: int, duration: float):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"color=c={color}:s=160x120:r=24:d={duration}",
            "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )


def _tone_amplitude(samples: np.ndarray, frequency: int) -> float:
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)
    return float(spectrum[np.argmin(np.abs(frequencies - frequency))])


def test_run_h3_native_audio_never_restores_source_track(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, plan, generation = _native_audio_plan(settings)
    _make_tone_video(plan.source, color="black", frequency=220, duration=2)
    _make_tone_video(
        plan.segments[0].workdir / "generated.mp4",
        color="red",
        frequency=440,
        duration=1,
    )
    _make_tone_video(
        plan.segments[1].workdir / "generated.mp4",
        color="blue",
        frequency=880,
        duration=1,
    )
    monkeypatch.setattr(
        h3, "H3_MULTIMODAL_WORKFLOWS", frozenset({plan.workflow}), raising=False
    )
    monkeypatch.setattr(h3, "is_multimodal_request", lambda _request: True, raising=False)
    monkeypatch.setattr(
        long_generation,
        "bound_reusable_segment_indices",
        lambda *_args, **_kwargs: frozenset({1, 2}),
    )
    monkeypatch.setattr(
        long_generation,
        "_request",
        lambda _settings, _cid, _plan, segment, *_args, **_kwargs: SimpleNamespace(
            segment_index=segment.index
        ),
    )
    timeline_calls = []

    def load_timeline(request, attempt_id):
        timeline_calls.append((request.segment_index, attempt_id))
        return {
            "schema": "duet.h3.media_timeline",
            "version": 1,
            "decode_complete": True,
            "video": {"index": 0},
            "audio": {"index": 1, "decoded_sha256": f"{request.segment_index:064x}"},
        }

    monkeypatch.setattr(h3, "load_media_timeline_receipt", load_timeline)
    monkeypatch.setattr(
        long_generation, "stitched_output_is_reusable",
        _REAL_STITCHED_OUTPUT_IS_REUSABLE,
    )

    long_generation.run(settings, cid, plan)

    assert timeline_calls == [(1, "000001"), (2, "000002")]
    output = plan.root / "generated.mp4"
    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
            "-f", "f32le", "-ac", "1", "-ar", "48000", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = np.frombuffer(decoded, dtype=np.float32)
    first = samples[int(0.15 * 48000):int(0.85 * 48000)]
    second = samples[int(1.15 * 48000):int(1.85 * 48000)]
    assert _tone_amplitude(first, 440) > 20 * _tone_amplitude(first, 220)
    assert _tone_amplitude(second, 880) > 20 * _tone_amplitude(second, 220)
    receipt = json.loads((plan.root / stitch.RECEIPT_FILENAME).read_text())
    assert receipt["version"] == 2
    assert receipt["audio"]["mode"] == "provider_generated"
    assert [item["attempt_id"] for item in receipt["audio"]["provider_segments"]] == [
        "000001", "000002",
    ]
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"
    exact_provider_media = {
        index: (
            f"{index:06d}",
            {
                "schema": "duet.h3.media_timeline",
                "version": 1,
                "decode_complete": True,
                "video": {"index": 0},
                "audio": {"index": 1, "decoded_sha256": f"{index:064x}"},
            },
        )
        for index in (1, 2)
    }
    assert _REAL_STITCHED_OUTPUT_IS_REUSABLE(
        plan,
        "auto",
        generation=generation,
        provider_media=exact_provider_media,
    )
    assert not _REAL_STITCHED_OUTPUT_IS_REUSABLE(plan, "auto")
    mismatched_generation = {
        **generation,
        "segments": [dict(item) for item in generation["segments"]],
    }
    mismatched_generation["segments"][0]["h3_attempt_id"] = "999999"
    assert not _REAL_STITCHED_OUTPUT_IS_REUSABLE(
        plan,
        "auto",
        generation=mismatched_generation,
        provider_media=exact_provider_media,
    )
    mismatched_provider_media = dict(exact_provider_media)
    mismatched_provider_media[1] = (
        "000001",
        {
            **exact_provider_media[1][1],
            "audio": {"index": 1, "decoded_sha256": "f" * 64},
        },
    )
    assert not _REAL_STITCHED_OUTPUT_IS_REUSABLE(
        plan,
        "auto",
        generation=generation,
        provider_media=mismatched_provider_media,
    )


def test_h3_native_submission_unknown_never_calls_stitch(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, plan, generation = _native_audio_plan(settings, segment_count=1)
    generation["status"] = "running"
    generation["segments"][0]["status"] = "running"
    storage.update_meta(settings.data_dir, cid, generation=generation)
    monkeypatch.setattr(
        h3, "H3_MULTIMODAL_WORKFLOWS", frozenset({plan.workflow}), raising=False
    )
    monkeypatch.setattr(
        long_generation, "bound_reusable_segment_indices", lambda *_args: frozenset()
    )
    monkeypatch.setattr(
        long_generation,
        "_request",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        h3,
        "resume",
        lambda _request: h3.H3Result("submission_unknown", "000001"),
    )
    monkeypatch.setattr(
        stitch,
        "stitch_video",
        lambda **_kwargs: pytest.fail("unknown H3 attempt must not stitch"),
    )

    long_generation.run(settings, cid, plan, startup=True)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "submission_unknown"
    assert stored["segments"][0]["h3_attempt_id"] == "000001"


@pytest.mark.parametrize("damage", ["missing_timeline", "missing_audio"])
def test_h3_native_missing_audio_evidence_never_falls_back_to_source(
    tmp_path, monkeypatch, damage,
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, plan, _generation = _native_audio_plan(settings, segment_count=1)
    monkeypatch.setattr(
        h3, "H3_MULTIMODAL_WORKFLOWS", frozenset({plan.workflow}), raising=False
    )
    monkeypatch.setattr(h3, "is_multimodal_request", lambda _request: True, raising=False)
    monkeypatch.setattr(
        long_generation, "bound_reusable_segment_indices", lambda *_args: frozenset({1})
    )
    monkeypatch.setattr(
        long_generation, "_request", lambda *_args, **_kwargs: SimpleNamespace()
    )

    def damaged_timeline(_request, _attempt_id):
        if damage == "missing_timeline":
            raise h3.ReceiptError("media_timeline_missing")
        return {
            "schema": "duet.h3.media_timeline",
            "version": 1,
            "decode_complete": True,
            "video": {"index": 0},
            "audio": None,
        }

    monkeypatch.setattr(h3, "load_media_timeline_receipt", damaged_timeline)
    stitch_calls = []
    monkeypatch.setattr(stitch, "stitch_video", _fake_stitch(stitch_calls))

    long_generation.run(settings, cid, plan)

    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    if damage == "missing_timeline":
        assert stitch_calls == []
        assert stored["status"] == "failed"
        assert stored["error"] == "long_video_h3_native_audio_invalid"
    else:
        assert stored["status"] == "succeeded"
        assert len(stitch_calls) == 1
        assert stitch_calls[0]["audio_mode"] == "provider_generated"
        timeline = stitch_calls[0]["segments"][0].provider_media_timeline
        assert timeline is not None
        assert timeline["audio"] is None
