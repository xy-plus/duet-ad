import threading

from fastapi.testclient import TestClient

from app import main as main_module, pipeline, storage
from app.main import create_app
from conftest import AUTH, make_settings


def _ready_queued(settings, *, request_id="request-orphan-123"):
    meta = storage.new_conversation(
        settings.data_dir,
        note="orphan",
        orig_name="clip.mp4",
        client_request_id=request_id,
    )
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"durable-upload")
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        duration_s=1.0,
        source_width=320,
        source_height=240,
        dialogue_mode="auto",
        voice_lines=[],
    )
    return meta


def test_ready_queued_pipeline_input_is_claimed_once(tmp_path):
    settings = make_settings(tmp_path, enable_pipeline=True)
    meta = _ready_queued(settings)

    claimed = storage.claim_ready_queued_pipeline_input(
        settings.data_dir, meta["id"]
    )

    assert claimed is not None
    assert claimed["status"] == "processing"
    assert claimed["_input_owner"]["kind"] == "pipeline"
    assert storage.claim_ready_queued_pipeline_input(
        settings.data_dir, meta["id"]
    ) is None


def test_startup_recovers_ready_queued_input(tmp_path, monkeypatch):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = _ready_queued(settings)
    called = threading.Event()

    def fake_run(_settings, cid, _runner, **kwargs):
        owner = kwargs["claimed_owner"]
        assert cid == meta["id"]
        assert storage.finish_input_claim(
            settings.data_dir,
            cid,
            owner,
            status="failed",
            error="recovered-test",
        )
        called.set()

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(create_app(settings)):
        assert called.wait(timeout=1)

    assert storage.load_meta(settings.data_dir, meta["id"])["error"] == (
        "recovered-test"
    )


def test_idempotent_create_requeues_ready_orphan(
    tmp_path, video_1s, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    called = threading.Event()
    expected = {}

    def fake_run(_settings, cid, _runner, **kwargs):
        assert cid == expected["cid"]
        assert kwargs.get("claimed_owner")
        called.set()

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(create_app(settings)) as client:
        meta = _ready_queued(settings)
        expected["cid"] = meta["id"]
        with video_1s.open("rb") as source:
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", source, "video/mp4")},
                data={"client_request_id": "request-orphan-123"},
            )
        assert response.status_code == 200
        assert called.wait(timeout=1)

    assert response.json()["id"] == meta["id"]


def test_idempotent_create_finishes_incomplete_orphan_upload(
    tmp_path, video_1s, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    called = threading.Event()

    def fake_run(_settings, _cid, _runner, **kwargs):
        assert kwargs.get("claimed_owner") is None
        called.set()

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(create_app(settings)) as client:
        meta = storage.new_conversation(
            settings.data_dir,
            note="partial",
            orig_name="clip.mp4",
            client_request_id="request-partial-123",
        )
        cdir = settings.data_dir / meta["id"]
        (cdir / "source.mp4").write_bytes(b"partial")
        with video_1s.open("rb") as source:
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", source, "video/mp4")},
                data={"client_request_id": "request-partial-123"},
            )
        assert response.status_code == 200
        assert called.wait(timeout=1)

    assert response.json()["id"] == meta["id"]
    recovered = storage.load_meta(settings.data_dir, meta["id"])
    assert recovered["duration_s"] > 0


def test_pipeline_gate_releases_and_closes_after_runner_exception(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_pipeline=True)
    meta = _ready_queued(settings)
    gate = threading.Semaphore(1)

    def fail_run(_settings, cid, _runner, **_kwargs):
        claimed = storage.claim_pipeline_input(settings.data_dir, cid)
        assert claimed is not None
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(pipeline, "run", fail_run)

    assert main_module._run_pipeline_under_gate(
        settings, meta["id"], object(), gate
    ) is False
    assert gate.acquire(blocking=False) is True
    gate.release()
    failed = storage.load_meta(settings.data_dir, meta["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "pipeline_internal_error"


def test_pipeline_gate_timeout_closes_waiter(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_pipeline=True)
    meta = _ready_queued(settings)
    gate = threading.Semaphore(0)
    monkeypatch.setattr(main_module, "_PIPELINE_GATE_WAIT_S", 0.01)

    assert main_module._run_pipeline_under_gate(
        settings, meta["id"], object(), gate
    ) is False
    failed = storage.load_meta(settings.data_dir, meta["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "pipeline_gate_timeout"
