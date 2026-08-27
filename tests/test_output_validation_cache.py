import json
import threading
import time

from fastapi.testclient import TestClient

from app import main as main_module, storage
from app.main import create_app
from conftest import AUTH, make_settings


def _succeeded_conversation(settings, cid="same-cid"):
    cdir = settings.data_dir / cid
    (cdir / "work").mkdir(parents=True)
    (cdir / "generated.mp4").write_bytes(b"published")
    artifacts = {
        "source": "source.mp4",
        "visual_prompt": "work/visual_prompt.txt",
        "final_prompt": "work/prompt.txt",
    }
    for name, relative in artifacts.items():
        path = cdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    receipt = {
        "bindings": {
            "source": {"path": artifacts["source"], "sha256": "unused"},
            "normalized_audio": None,
            "keyframes": [],
            "visual_prompt": {
                "path": artifacts["visual_prompt"], "sha256": "unused"
            },
            "final_prompt": {
                "path": artifacts["final_prompt"], "sha256": "unused"
            },
        }
    }
    (cdir / "prepared_input.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return {
        "id": cid,
        "prepared_input_receipt": "prepared_input.json",
        "generation": {
            "status": "succeeded",
            "client_request_id": "request-cache-test",
        },
    }


def test_generated_video_validation_cache_hits_and_meta_change_invalidates(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    calls = []

    def validate(_settings, _meta):
        calls.append((_settings, _meta))
        return True

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)

    assert main_module._has_valid_generated_video(settings, meta) is True
    assert main_module._has_valid_generated_video(settings, meta) is True
    assert len(calls) == 1

    changed = {
        **meta,
        "generation": {
            **meta["generation"],
            "client_request_id": "request-cache-changed",
        },
    }
    assert main_module._has_valid_generated_video(settings, changed) is True
    assert len(calls) == 2


def test_generated_video_validation_cache_invalidates_every_artifact_class(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)
    assert main_module._has_valid_generated_video(settings, meta) is False

    paths = [
        settings.data_dir / meta["id"] / "generated.mp4",
        settings.data_dir / meta["id"] / "source.mp4",
        settings.data_dir / meta["id"] / "prepared_input.json",
    ]
    for index, path in enumerate(paths, 1):
        path.write_bytes(f"mutation-{index}".encode())
        assert main_module._has_valid_generated_video(settings, meta) is False

    assert calls == 1 + len(paths)


def test_generated_video_validation_cache_ignores_unrelated_files_and_meta(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)
    assert main_module._has_valid_generated_video(settings, meta) is True

    cdir = settings.data_dir / meta["id"]
    unrelated = cdir / "work" / "unrelated-keyframe.png"
    unrelated.write_bytes(b"not bound by either receipt")
    changed = {**meta, "updated_at": "later", "postprocess": {"status": "done"}}
    assert main_module._has_valid_generated_video(settings, changed) is True
    assert calls == 1


def test_validation_fingerprint_hashes_bound_speaker_timing_raw_bytes(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    cdir = settings.data_dir / meta["id"]
    timing = cdir / "work" / "speaker_timing.json"
    timing.write_bytes(b'{"timing":"aaaa"}')
    prepared = cdir / "prepared_input.json"
    receipt = json.loads(prepared.read_text(encoding="utf-8"))
    receipt["multimodal"] = {
        "schema": "duet.h3-project-multimodal",
        "version": 3,
        "speaker_timing": {
            "path": "work/speaker_timing.json",
            "sha256": "expected-bound-sha",
            "canonical_sha256": "expected-canonical-sha",
        },
    }
    prepared.write_text(json.dumps(receipt), encoding="utf-8")

    before_stat = timing.stat()
    before = main_module._generated_video_validation_fingerprint(cdir, meta)
    timing.write_bytes(b'{"timing":"bbbb"}')
    real_stat = main_module.Path.stat

    def stable_stat(path, *args, **kwargs):
        if str(path) == str(timing):
            return before_stat
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(main_module.Path, "stat", stable_stat)
    after = main_module._generated_video_validation_fingerprint(cdir, meta)

    assert before is not None
    assert after is not None
    assert after != before


def test_long_video_cache_invalidates_plan_segment_state_and_stitch_artifacts(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    cid = "long-cid"
    cdir = settings.data_dir / cid
    segment = cdir / "work" / "segments" / "1"
    attempt = segment / ".h3" / "attempts" / "000001" / "attempt.json"
    attempt.parent.mkdir(parents=True)
    bindings = {
        "source": "source.mp4",
        "segment_source": "work/segments/1/source.mp4",
        "anchor_first": "work/segments/1/first.png",
        "anchor_end": "work/segments/1/end.png",
        "visual": "work/segments/1/visual.txt",
        "final": "work/segments/1/final.txt",
    }
    for name, relative in bindings.items():
        path = cdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for path in (
        segment / "generated.mp4",
        segment / ".h3" / "session.json",
        attempt,
        cdir / "generated.mp4",
        cdir / "stitch-receipt.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    def artifact(name):
        return {"path": bindings[name], "sha256": "unused"}

    plan = {
        "source": artifact("source"),
        "segments": [{
            "index": 1,
            "source": artifact("segment_source"),
            "keyframes": [],
            "anchors": [
                {"role": "first", **artifact("anchor_first")},
                {"role": "end", **artifact("anchor_end")},
            ],
            "visual_prompt": artifact("visual"),
            "final_prompt": artifact("final"),
        }],
    }
    (cdir / "long_video_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    meta = {
        "id": cid,
        "duration_s": 20,
        "fit_mode": "none",
        "dialogue_mode": "auto",
        "segments": [{"index": 1}],
        "frozen_plan_receipt": "receipt",
        "long_video_plan_receipt": "long_video_plan.json",
        "generation": {
            "status": "succeeded",
            "client_request_id": "request-long-cache",
            "segments": [{"index": 1}],
        },
    }
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)
    assert main_module._has_valid_generated_video(settings, meta) is True

    relevant = [
        cdir / bindings["anchor_first"],
        segment / "generated.mp4",
        attempt,
        cdir / "stitch-receipt.json",
        cdir / "generated.mp4",
    ]
    for index, path in enumerate(relevant, 1):
        path.write_bytes(f"changed-{index}".encode())
        assert main_module._has_valid_generated_video(settings, meta) is True

    assert calls == 1 + len(relevant)


def test_generated_video_validation_cache_separates_data_roots(tmp_path, monkeypatch):
    first = make_settings(tmp_path / "first")
    second = make_settings(tmp_path / "second")
    first_meta = _succeeded_conversation(first)
    second_meta = _succeeded_conversation(second)
    calls = []

    def validate(settings, _meta):
        calls.append(settings.data_dir)
        return True

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)

    assert main_module._has_valid_generated_video(first, first_meta) is True
    assert main_module._has_valid_generated_video(second, second_meta) is True
    assert main_module._has_valid_generated_video(first, first_meta) is True
    assert calls == [first.data_dir, second.data_dir]


def test_generated_video_validation_cache_coalesces_concurrent_misses(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    start = threading.Barrier(5)
    calls = 0
    calls_lock = threading.Lock()

    def validate(_settings, _meta):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return True

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)

    results = []

    def worker():
        start.wait()
        results.append(main_module._has_valid_generated_video(settings, meta))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert results == [True] * 5
    assert calls == 1


def test_generated_video_validation_cache_does_not_persist_false(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    results = iter((False, True))
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return next(results)

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)

    assert main_module._has_valid_generated_video(settings, meta) is False
    assert main_module._has_valid_generated_video(settings, meta) is True
    assert calls == 2


def test_concurrent_false_validation_waiters_do_not_deadlock(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _succeeded_conversation(settings)
    start = threading.Barrier(5)
    calls = 0
    calls_lock = threading.Lock()

    def validate(_settings, _meta):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.01)
        return False

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)
    results = []

    def worker():
        start.wait()
        results.append(main_module._has_valid_generated_video(settings, meta))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert results == [False] * 5
    assert calls == 5


def test_repeated_list_and_detail_share_one_expensive_validation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(main_module, "_validate_generated_video_uncached", validate)
    with TestClient(create_app(settings)) as client:
        meta = storage.new_conversation(settings.data_dir, "cache", "source.mp4")
        storage.update_meta(
            settings.data_dir,
            meta["id"],
            generation={
                "status": "succeeded",
                "client_request_id": "request-cache-test",
            },
            prepared_input_receipt="prepared_input.json",
        )
        (settings.data_dir / meta["id"] / "generated.mp4").write_bytes(b"video")

        assert client.get("/api/conversations", headers=AUTH).status_code == 200
        assert client.get(
            f"/api/conversations/{meta['id']}", headers=AUTH
        ).status_code == 200
        assert client.get("/api/conversations", headers=AUTH).status_code == 200

    assert calls == 1


def test_generated_video_validation_cache_is_bounded_lru():
    cache = main_module._GeneratedVideoValidationCache(2)
    calls = []

    def validate(name):
        calls.append(name)
        return True

    for name in ("a", "b", "c", "a"):
        assert cache.get_or_validate(
            (name,), "same", lambda: "same", lambda name=name: validate(name)
        ) is True

    assert calls == ["a", "b", "c", "a"]
