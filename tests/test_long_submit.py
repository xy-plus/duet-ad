import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import h3, long_generation, long_video, prepared_input, storage
from app.main import _resume_long_generation, create_app
from conftest import AUTH, make_settings


def _png(path: Path, value: int) -> None:
    image = np.full((160, 90, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _make_long(settings, *, joins=("hard_cut",), dialogue_text="源台词",
               segment_duration=15.0):
    duration = segment_duration * len(joins)
    meta = storage.new_conversation(settings.data_dir, "long", "source.mp4")
    cid = meta["id"]
    root = settings.data_dir / cid
    source = root / "source.mp4"
    source.write_bytes(b"source-video")
    receipt_input = []
    public_segments = []
    chain_no = 0
    for index, join_mode in enumerate(joins, 1):
        if join_mode == "hard_cut":
            chain_no += 1
        chain_id = f"chain-{chain_no:03d}"
        segdir = root / "work" / "segments" / str(index)
        work = segdir / "work"
        (segdir / "source.mp4").parent.mkdir(parents=True, exist_ok=True)
        (segdir / "source.mp4").write_bytes(f"segment-{index}".encode())
        key = work / "keyframes" / "01.png"
        first = work / "anchors" / "first.png"
        last = work / "anchors" / "last.png"
        _png(key, 20 + index)
        _png(first, 40 + index)
        _png(last, 60 + index)
        visual_text = f"第{index}段局部动作"
        visual = work / "visual_prompt.txt"
        visual.write_text(visual_text, encoding="utf-8")
        local_dialogue = ({"text": dialogue_text, "start_s": 1.0, "end_s": 2.0},)
        prompt_text = "不要生成背景音乐\n" + prepared_input.compose_final_prompt(
            long_video.compose_segment_visual_prompt(visual_text), local_dialogue
        )
        final = work / "prompt.txt"
        final.write_text(prompt_text, encoding="utf-8")
        segment = {
            "index": index,
            "start_s": segment_duration * (index - 1),
            "end_s": segment_duration * index,
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
    receipt_path = long_video.write_plan_receipt(
        root, source=source, duration_s=duration, segments=receipt_input,
        workflow=h3.H3_BOUNDARY_WORKFLOW,
    )
    receipt = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    storage.update_meta(
        settings.data_dir, cid, status="done", duration_s=duration,
        voice_mode="keep", fit_required=False, segments=public_segments,
        long_video_plan_receipt=receipt_path.name,
    )
    return cid, receipt


def _payload(receipt, request_id="parent-request-123", mode="auto", fit="none"):
    return {
        "confirm": True,
        "client_request_id": request_id,
        "dialogue_mode": mode,
        "fit_mode": fit,
        "expected_plan_receipt": receipt,
    }


def _fake_stitch(calls):
    def invoke(**kwargs):
        calls.append(kwargs)
        kwargs["output"].write_bytes(b"joined")
    return invoke


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    with TestClient(create_app(settings)) as client:
        yield settings, client


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
    assert [call["audio_mode"] for call in stitch_calls] == ["keep", "mute"]


def test_30_second_continue_uses_generated_tail_and_two_posts(enabled, monkeypatch):
    settings, client = enabled
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    seen = []
    tail_bytes = (settings.data_dir / cid / "work" / "tail-fixture.png")
    _png(tail_bytes, 222)
    def extract_tail(_video, output):
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
    assert [request.workdir.name for request in seen] == ["1", "2"]
    assert seen[1].first_frame[1] == tail_bytes.read_bytes()


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


def test_startup_recovery_only_resumes_attempted_segments(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, joins=("hard_cut", "continue"))
    plan = long_generation.freeze_plan(settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
                                       receipt, "none", "auto")
    generation = long_generation.initial_generation(plan, "parent-request-123", 1)
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


def test_active_same_parent_is_idempotent_without_refreeze_or_second_coordinator(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, receipt = _make_long(settings)
    plan = long_generation.freeze_plan(
        settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
        receipt, "none", "auto",
    )
    generation = long_generation.initial_generation(plan, "parent-request-123", 1)
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
    generation = long_generation.initial_generation(plan, "parent-request-123", 1)
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
    generation = long_generation.initial_generation(plan, "parent-request-123", 1)
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
    generation = long_generation.initial_generation(plan, "parent-request-123", 1)
    generation["status"] = "resume_required"
    generation["segments"][0].update(status="resume_required", attempt=1)
    storage.update_meta(settings.data_dir, cid, fit_mode="none", dialogue_mode="auto",
                        frozen_plan_receipt=receipt, generation=generation)
    monkeypatch.setattr(long_generation.stitch, "stitch_video", _fake_stitch([]))
    def start(request):
        request.workdir.joinpath("generated.mp4").write_bytes(b"segment")
        return h3.H3Result("succeeded", "task")
    monkeypatch.setattr(h3, "start", start)
    long_generation.run(settings, cid, plan)
    assert storage.load_meta(settings.data_dir, cid)["generation"]["segments"][0]["attempt"] == 1


def test_freeze_rejects_segment_whose_ceil_duration_exceeds_15(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings, segment_duration=15.0000005)
    with pytest.raises(long_generation.LongGenerationError, match="long_video_plan_invalid"):
        long_generation.freeze_plan(
            settings.data_dir / cid, storage.load_meta(settings.data_dir, cid),
            receipt, "none", "auto",
        )


def test_local_continue_request_failure_is_structured_failed_not_coordinator_crash(
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
    assert generation["status"] == "failed"
    assert generation["segments"][1]["error"] == "tail_failed"


def test_resume_required_segment_runs_at_most_once_per_coordinator_invocation(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, receipt = _make_long(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    plan = long_generation.freeze_plan(settings.data_dir / cid, meta, receipt, "none", "auto")
    generation = long_generation.initial_generation(plan, "parent-request-123", 1)
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


@pytest.mark.parametrize("changed", ["plan", "dialogue", "fit"])
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
    if changed == "plan":
        path = settings.data_dir / cid / long_video.PLAN_RECEIPT_FILENAME
        path.write_text(path.read_text() + " ", encoding="utf-8")
        retry_receipt = hashlib.sha256(path.read_bytes()).hexdigest()
    elif changed == "dialogue":
        retry_mode = "none"
    else:
        retry_fit = "pad"
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(
            retry_receipt, request_id="parent-request-999",
            mode=retry_mode, fit=retry_fit,
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
