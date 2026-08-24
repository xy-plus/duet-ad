import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app import storage


def test_new_conversation_layout(tmp_path):
    meta = storage.new_conversation(tmp_path, note="我的笔记", orig_name="x.mp4")
    cdir = tmp_path / meta["id"]
    assert cdir.is_dir()
    assert (cdir / "work").is_dir()
    saved = json.loads((cdir / "meta.json").read_text())
    for key in (
        "id", "title", "note", "status", "error", "created_at", "updated_at",
        "schema_version", "duration_s", "fit_required", "dialogue_mode", "generation",
    ):
        assert key in saved
    assert saved["status"] == "queued"
    assert saved["error"] is None
    assert saved["title"] == "我的笔记"
    assert saved["note"] == "我的笔记"
    assert saved["schema_version"] == 2
    assert saved["dialogue_mode"] == "auto"
    assert saved["generation"] is None
    assert saved["voice_mode"] == "keep"
    assert len(saved["id"]) == 32


def test_title_falls_back_to_sanitized_filename(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="../../etc/\x00pass\\wd.mp4")
    assert "/" not in meta["title"] and "\x00" not in meta["title"]
    assert meta["title"] == "wd"
    meta = storage.new_conversation(tmp_path, note="", orig_name="a" * 200 + ".mp4")
    assert len(meta["title"]) <= 80


def test_load_meta_rejects_bad_id(tmp_path):
    assert storage.load_meta(tmp_path, "..") is None
    assert storage.load_meta(tmp_path, "a" * 31) is None
    assert storage.load_meta(tmp_path, "g" * 32) is None
    assert storage.load_meta(tmp_path, "0" * 32) is None  # 合法格式但不存在


def test_list_conversations(tmp_path):
    a = storage.new_conversation(tmp_path, note="first", orig_name="a.mp4")
    b = storage.new_conversation(tmp_path, note="second", orig_name="b.mp4")
    items = storage.list_conversations(tmp_path)
    assert {m["id"] for m in items} == {a["id"], b["id"]}
    assert storage.list_conversations(tmp_path / "empty") == []


def test_concurrent_meta_updates_are_atomic_and_do_not_lose_fields(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    barrier = threading.Barrier(2)

    def update(**changes):
        barrier.wait()
        storage.update_meta(tmp_path, meta["id"], **changes)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(update, generation={"status": "running"}),
            pool.submit(update, segments=[{"index": 1, "status": "succeeded"}]),
        ]
        for future in futures:
            future.result()
    stored = storage.load_meta(tmp_path, meta["id"])
    assert stored["generation"] == {"status": "running"}
    assert stored["segments"] == [{"index": 1, "status": "succeeded"}]


@pytest.mark.parametrize("first_owner", ["pipeline", "submit"])
def test_pipeline_and_submission_claim_share_one_atomic_owner(
    tmp_path, monkeypatch, first_owner,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    storage.update_meta(tmp_path, meta["id"], status="done", duration_s=9.2)
    entered = threading.Event()
    release = threading.Event()
    writes = []
    original_write = storage._write_meta

    def blocked_write(cdir, payload):
        writes.append(payload.get("_input_owner"))
        entered.set()
        assert release.wait(timeout=5)
        original_write(cdir, payload)

    monkeypatch.setattr(storage, "_write_meta", blocked_write)

    def claim(owner):
        if owner == "pipeline":
            return storage.claim_pipeline_input(tmp_path, meta["id"])
        return storage.claim_submission_input(tmp_path, meta["id"], "request-123456")

    second_owner = "submit" if first_owner == "pipeline" else "pipeline"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(claim, first_owner)
        assert entered.wait(timeout=5)
        second = pool.submit(claim, second_owner)
        release.set()
        results = {first_owner: first.result(), second_owner: second.result()}

    assert results[first_owner] is not None
    assert results[second_owner] is None
    assert len(writes) == 1
    provider_calls = 1 if results["submit"] is not None else 0
    assert provider_calls == (1 if first_owner == "submit" else 0)


def test_new_process_generation_reclaims_unfrozen_pipeline_owner(tmp_path, monkeypatch):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    old = storage.claim_pipeline_input(tmp_path, meta["id"])
    assert old["_input_owner"]["process_generation"] == "boot-old"

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    recovered = storage.claim_pipeline_input(tmp_path, meta["id"])

    assert recovered is not None
    assert recovered["status"] == "processing"
    assert recovered["_input_owner"] == {
        "kind": "pipeline", "process_generation": "boot-new",
        "frozen_input_snapshot": {},
    }


def test_new_process_generation_reruns_pipeline_after_only_partial_prompt(
    tmp_path, monkeypatch,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(tmp_path, meta["id"])
    prompt = tmp_path / meta["id"] / "work" / "prompt.txt"
    prompt.write_text("partial, not frozen", encoding="utf-8")

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    assert storage.claim_pipeline_input(tmp_path, meta["id"]) is not None


def test_new_process_generation_does_not_rerun_pipeline_after_receipt_landed(
    tmp_path, monkeypatch,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(tmp_path, meta["id"])
    receipt = tmp_path / meta["id"] / "prepared_input.json"
    receipt.write_bytes(b"frozen")

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    before = (tmp_path / meta["id"] / "meta.json").read_bytes()
    assert storage.claim_pipeline_input(tmp_path, meta["id"]) is None
    assert (tmp_path / meta["id"] / "meta.json").read_bytes() == before


def test_frozen_reconciliation_claim_preserves_original_snapshot_across_crashes(
    tmp_path, monkeypatch,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cdir = tmp_path / meta["id"]
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    old = storage.claim_pipeline_input(tmp_path, meta["id"])
    assert old["_input_owner"]["frozen_input_snapshot"] == {}
    (cdir / "prepared_input.json").write_bytes(b"half-frozen")

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-reconcile-1")
    first = storage.claim_stale_input_reconciliations(tmp_path)
    assert first[0][1]["frozen_input_snapshot"] == {}

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-reconcile-2")
    second = storage.claim_stale_input_reconciliations(tmp_path)
    assert second[0][1]["frozen_input_snapshot"] == {}


def test_new_process_generation_reclaims_unfrozen_submit_owner_with_new_request(
    tmp_path, monkeypatch,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    storage.update_meta(tmp_path, meta["id"], status="done", duration_s=9.2)
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_submission_input(tmp_path, meta["id"], "request-old")

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    recovered = storage.claim_submission_input(tmp_path, meta["id"], "request-new")

    assert recovered is not None
    assert recovered["_input_owner"] == {
        "kind": "submit", "process_generation": "boot-new",
        "request_id": "request-new", "frozen_input_snapshot": {},
    }


def test_new_process_reclaims_submit_owner_when_preexisting_receipt_is_unchanged(
    tmp_path, monkeypatch,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cdir = tmp_path / meta["id"]
    (cdir / "prepared_input.json").write_bytes(b"pipeline-receipt")
    storage.update_meta(
        tmp_path, meta["id"], status="done", duration_s=9.2,
        prepared_input_receipt="prepared_input.json",
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_submission_input(tmp_path, meta["id"], "request-old")

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    recovered = storage.claim_submission_input(tmp_path, meta["id"], "request-new")

    assert recovered is not None
    assert recovered["_input_owner"]["request_id"] == "request-new"
    assert recovered["_input_owner"]["frozen_input_snapshot"] == {
        "prepared_input.json": hashlib.sha256(b"pipeline-receipt").hexdigest(),
    }


@pytest.mark.parametrize("frozen", ["generation", "prepared", "plan", "fit"])
def test_new_process_cannot_take_over_submit_owner_after_frozen_artifact(
    tmp_path, monkeypatch, frozen,
):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cdir = tmp_path / meta["id"]
    storage.update_meta(tmp_path, meta["id"], status="done", duration_s=9.2)
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_submission_input(tmp_path, meta["id"], "request-old")
    if frozen == "generation":
        storage.update_meta(tmp_path, meta["id"], generation={"status": "queued"})
    elif frozen == "prepared":
        (cdir / "prepared_input.json").write_bytes(b"frozen-prepared")
    elif frozen == "plan":
        (cdir / "long_video_plan.json").write_bytes(b"frozen-plan")
    else:
        fit = cdir / "work" / "h3_frames" / "crop"
        fit.mkdir(parents=True)
        (fit / "01.png").write_bytes(b"frozen-fit")
    before = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file()
    }

    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    assert storage.claim_submission_input(tmp_path, meta["id"], "request-new") is None

    after = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file()
    }
    assert after == before


def test_finish_input_claim_is_bound_to_exact_process_generation(tmp_path, monkeypatch):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    claimed = storage.claim_pipeline_input(tmp_path, meta["id"])
    owner = claimed["_input_owner"]
    before = (tmp_path / meta["id"] / "meta.json").read_bytes()

    wrong = {**owner, "process_generation": "boot-new"}
    assert storage.finish_input_claim(
        tmp_path, meta["id"], wrong, status="done"
    ) is None
    assert (tmp_path / meta["id"] / "meta.json").read_bytes() == before
    assert storage.finish_input_claim(
        tmp_path, meta["id"], owner, status="done"
    )["status"] == "done"


def test_legacy_meta_without_owner_remains_claimable(tmp_path, monkeypatch):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    claimed = storage.claim_pipeline_input(tmp_path, meta["id"])
    assert claimed is not None


def test_resolve_file_whitelist(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cdir = tmp_path / meta["id"]
    (cdir / "source.mov").write_bytes(b"s")
    (cdir / "preview.mp4").write_bytes(b"p")
    (cdir / "generated.mp4").write_bytes(b"g")
    (cdir / "work" / "contact_sheet.jpg").write_bytes(b"c")
    (cdir / "work" / "keyframes").mkdir()
    (cdir / "work" / "keyframes" / "k01.jpg").write_bytes(b"k")

    cid = meta["id"]
    assert storage.resolve_file(tmp_path, cid, "source.mp4") == (cdir / "source.mov").resolve()
    assert storage.resolve_file(tmp_path, cid, "preview.mp4") == (cdir / "preview.mp4").resolve()
    assert storage.resolve_file(tmp_path, cid, "generated.mp4") == (cdir / "generated.mp4").resolve()
    assert storage.resolve_file(tmp_path, cid, "contact_sheet.jpg") == (cdir / "work" / "contact_sheet.jpg").resolve()
    assert storage.resolve_file(tmp_path, cid, "keyframes/k01.jpg") == (cdir / "work" / "keyframes" / "k01.jpg").resolve()


def test_resolve_file_rejects_traversal_and_unknown(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cid = meta["id"]
    assert storage.resolve_file(tmp_path, cid, "../meta.json") is None
    assert storage.resolve_file(tmp_path, cid, "keyframes/../meta.json") is None
    assert storage.resolve_file(tmp_path, cid, "keyframes/sub/dir.jpg") is None
    assert storage.resolve_file(tmp_path, cid, "keyframes/") is None
    assert storage.resolve_file(tmp_path, cid, "meta.json") is None
    assert storage.resolve_file(tmp_path, cid, "preview.exe") is None
    assert storage.resolve_file(tmp_path, "..", "preview.mp4") is None


def test_resolve_file_missing_on_disk(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    assert storage.resolve_file(tmp_path, meta["id"], "preview.mp4") is None
    assert storage.resolve_file(tmp_path, meta["id"], "source.mp4") is None
    assert storage.resolve_file(tmp_path, meta["id"], "keyframes/nope.jpg") is None


def test_resolve_file_postprocessed_and_segments(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cid = meta["id"]
    cdir = tmp_path / cid
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    (cdir / "work" / "postprocessed" / "01.png").write_bytes(b"p")
    (cdir / "work" / "segments" / "2" / "work" / "keyframes").mkdir(parents=True)
    (cdir / "work" / "segments" / "2" / "work" / "keyframes" / "01.png").write_bytes(b"k")
    (cdir / "work" / "segments" / "2" / "work" / "postprocessed").mkdir(parents=True)
    (cdir / "work" / "segments" / "2" / "work" / "postprocessed" / "01.png").write_bytes(b"s")

    assert storage.resolve_file(tmp_path, cid, "postprocessed/01.png") == \
        (cdir / "work" / "postprocessed" / "01.png").resolve()
    assert storage.resolve_file(tmp_path, cid, "segments/2/work/keyframes/01.png") == \
        (cdir / "work" / "segments" / "2" / "work" / "keyframes" / "01.png").resolve()
    assert storage.resolve_file(tmp_path, cid, "segments/2/work/postprocessed/01.png") == \
        (cdir / "work" / "segments" / "2" / "work" / "postprocessed" / "01.png").resolve()


def test_resolve_file_rejects_bad_segments_and_postprocessed(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cid = meta["id"]
    # N 必须为正整数、fn 必须是纯文件名、目录必须是白名单两类；穿越/越界一律 None
    for name in ("segments/0/work/keyframes/a.png", "segments/x/work/keyframes/a.png",
                 "segments/1/keyframes/a.png",  # 缺 work/ 层级
                 "segments/1/meta.json", "segments/1/work/keyframes/",
                 "segments/1/work/keyframes/../prompt.txt",
                 "segments/1/work/postprocessed/../keyframes/a.png",
                 "segments/1/work/keyframes/sub/a.png",
                 "segments/1/work/../meta.json",
                 "postprocessed/", "postprocessed/../meta.json",
                 "postprocessed/sub/a.png"):
        assert storage.resolve_file(tmp_path, cid, name) is None, name
    # 合法格式但磁盘上不存在 → None
    assert storage.resolve_file(tmp_path, cid, "postprocessed/nope.png") is None
    assert storage.resolve_file(tmp_path, cid, "segments/1/work/keyframes/nope.png") is None


def test_probe_video_returns_duration_and_dimensions_in_one_probe(tmp_path, monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = json.dumps({
            "format": {"duration": "14.27"},
            "streams": [{"width": 1080, "height": 1920, "duration": "14.25"}],
        })
        stderr = ""

    monkeypatch.setattr(
        storage.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Completed(),
    )
    info = storage.probe_video(tmp_path / "source.mp4")
    assert info.duration_s == 14.25
    assert (info.width, info.height) == (1080, 1920)
    assert len(calls) == 1


def test_probe_video_uses_video_stream_not_longer_audio_or_container(tmp_path, monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "format": {"duration": "16.787007"},
            "streams": [{
                "width": 1080, "height": 1920, "duration": "16.766667",
                "duration_ts": "503", "time_base": "1/30",
            }],
        }),
        stderr="",
    )
    monkeypatch.setattr(storage.subprocess, "run", lambda *_a, **_kw: completed)
    assert storage.probe_video(tmp_path / "source.mp4").duration_s == 16.766667


def test_probe_video_falls_back_to_duration_ts_time_base(tmp_path, monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "format": {"duration": "99"},
            "streams": [{
                "width": 320, "height": 240, "duration": "N/A",
                "duration_ts": "503", "time_base": "1/30",
            }],
        }),
        stderr="",
    )
    monkeypatch.setattr(storage.subprocess, "run", lambda *_a, **_kw: completed)
    assert storage.probe_video(tmp_path / "source.mp4").duration_s == pytest.approx(503 / 30)


def test_probe_video_rejects_bool_duration_and_uses_duration_ts(tmp_path, monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"streams": [{
            "width": 320, "height": 240, "duration": True,
            "duration_ts": "503", "time_base": "1/30",
        }]}),
        stderr="",
    )
    monkeypatch.setattr(storage.subprocess, "run", lambda *_a, **_kw: completed)
    assert storage.probe_video(tmp_path / "source.mp4").duration_s == pytest.approx(503 / 30)


def test_probe_video_rejects_bool_ticks_and_uses_decoded_fallback(tmp_path, monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"streams": [{
            "width": 320, "height": 240, "duration": False,
            "duration_ts": True, "time_base": "1/30",
        }]}),
        stderr="",
    )
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda prop: {storage.cv2.CAP_PROP_FRAME_COUNT: 600,
                          storage.cv2.CAP_PROP_FPS: 30}[prop],
        release=lambda: None,
    )
    monkeypatch.setattr(storage.subprocess, "run", lambda *_a, **_kw: completed)
    monkeypatch.setattr(storage.cv2, "VideoCapture", lambda _path: capture)
    assert storage.probe_video(tmp_path / "source.mp4").duration_s == 20.0


def test_probe_video_falls_back_to_decoded_frame_timeline(tmp_path, monkeypatch):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "format": {"duration": "99"},
            "streams": [{"width": 320, "height": 240}],
        }),
        stderr="",
    )
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda prop: {storage.cv2.CAP_PROP_FRAME_COUNT: 503,
                          storage.cv2.CAP_PROP_FPS: 30}[prop],
        release=lambda: None,
    )
    monkeypatch.setattr(storage.subprocess, "run", lambda *_a, **_kw: completed)
    monkeypatch.setattr(storage.cv2, "VideoCapture", lambda _path: capture)
    assert storage.probe_video(tmp_path / "source.mp4").duration_s == pytest.approx(503 / 30)


@pytest.mark.parametrize("stream_duration", ["0", "nan", "inf", "-1"])
def test_probe_video_rejects_invalid_stream_duration_without_decoded_fallback(
    tmp_path, monkeypatch, stream_duration,
):
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({
            "format": {"duration": "99"},
            "streams": [{
                "width": 320, "height": 240, "duration": stream_duration,
            }],
        }),
        stderr="",
    )
    capture = SimpleNamespace(isOpened=lambda: False, release=lambda: None)
    monkeypatch.setattr(storage.subprocess, "run", lambda *_a, **_kw: completed)
    monkeypatch.setattr(storage.cv2, "VideoCapture", lambda _path: capture)
    with pytest.raises(storage.UploadError, match="duration"):
        storage.probe_video(tmp_path / "source.mp4")


def test_probe_video_rejects_missing_or_invalid_dimensions(tmp_path, monkeypatch):
    class Completed:
        returncode = 0
        stdout = json.dumps({"format": {"duration": "1"}, "streams": []})
        stderr = ""

    monkeypatch.setattr(storage.subprocess, "run", lambda *args, **kwargs: Completed())
    try:
        storage.probe_video(tmp_path / "source.mp4")
    except storage.UploadError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("missing video dimensions must be rejected")
