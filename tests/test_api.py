import hashlib
import json

import pytest

from conftest import AUTH, make_settings
from fastapi.testclient import TestClient

from app import h3, storage
from app.main import create_app


def _make_conv(client, video_1s, note=""):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")},
                        data={"note": note})
    assert r.status_code == 201
    return r.json()["id"]


def _write_context_ir_state(settings, cid, *, status="succeeded", prompt="优化后的精确提示词"):
    root = settings.data_dir / cid / ".h3"
    attempt_dir = root / "attempts" / "000001"
    attempt_dir.mkdir(parents=True)
    (root / "session.json").write_text(
        json.dumps({"schema_version": h3.SCHEMA_VERSION, "cid": cid}),
        encoding="utf-8",
    )
    input_manifest = {"fixture": "context-ir"}
    ir = {
        "status": status,
        "task_id": "provider-ir-task-secret",
        "credential": "provider-credential-secret",
    }
    if prompt is not None:
        ir.update(
            optimized_prompt=prompt,
            optimized_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
    state = {
        "schema_version": h3.SCHEMA_VERSION,
        "cid": cid,
        "attempt_id": "000001",
        "client_request_id": "request-context-ir",
        "input": input_manifest,
        "input_receipt": h3.canonical_json_sha256(input_manifest),
        "status": "ready_for_h3" if status == "succeeded" else "ir_running",
        "retryable": False,
        "ir": ir,
        "h3": {
            "status": "ready",
            "task_id": "provider-h3-task-secret",
            "provider_result_url": "https://provider.invalid/private-result",
        },
    }
    path = attempt_dir / "attempt.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def test_list_conversations_shape(client, video_1s):
    cid = _make_conv(client, video_1s, note="n1")
    r = client.get("/api/conversations", headers=AUTH)
    assert r.status_code == 200
    (item,) = r.json()
    assert set(item) == {"id", "title", "note", "status", "created_at", "has_video"}
    assert item["id"] == cid
    assert item["status"] == "queued"
    assert item["has_video"] is False


def test_detail_shape(client, video_1s):
    cid = _make_conv(client, video_1s, note="n1")
    r = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "id": cid,
        "title": "n1",
        "note": "n1",
        "status": "queued",
        "error": None,
        "created_at": r.json()["created_at"],
        "updated_at": r.json()["updated_at"],
        "keyframes": [],
        "prompt": None,
        "segments": [],
        "voice_lines": [],
        "read_only": False,
        "duration_s": body["duration_s"],
        "fit_required": None,
        "fit_mode": None,
        "dialogue": {"mode": "auto", "lines": [], "auto_lines": []},
        "receipt_version": None,
        "generation": None,
        "context_ir": {"status": "not_started", "available": False, "sha256": None},
        "has_source": True,
        "has_video": False,
        "submit_enabled": False,
        "postprocess": None,
        "postprocess_enabled": False,
    }
    assert 0.9 <= body["duration_s"] <= 1.1


def test_context_ir_details_are_small_and_prompt_is_loaded_only_from_authenticated_endpoint(tmp_path):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    prompt = "精确保留空格  和换行\n<d>原样台词</d>"
    _write_context_ir_state(settings, meta["id"], prompt=prompt)
    (settings.data_dir / meta["id"] / "generated.mp4").write_bytes(b"video")

    with TestClient(create_app(settings)) as client:
        detail_response = client.get(f"/api/conversations/{meta['id']}", headers=AUTH)
        unauthorized = client.get(f"/api/conversations/{meta['id']}/context-ir")
        prompt_response = client.get(
            f"/api/conversations/{meta['id']}/context-ir", headers=AUTH
        )

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert detail_response.status_code == 200
    assert detail_response.json()["context_ir"] == {
        "status": "succeeded",
        "available": True,
        "sha256": digest,
    }
    detail_text = detail_response.text
    for secret in (
        prompt,
        "provider-ir-task-secret",
        "provider-h3-task-secret",
        "provider-credential-secret",
        "https://provider.invalid/private-result",
    ):
        assert secret not in detail_text
    assert unauthorized.status_code == 401
    assert prompt_response.status_code == 200
    assert prompt_response.json() == {
        "status": "succeeded",
        "prompt": prompt,
        "sha256": digest,
        "dialogue_valid": True,
    }


def test_context_ir_endpoint_distinguishes_absent_running_and_unknown_conversation(tmp_path):
    settings = make_settings(tmp_path)
    absent = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    running = storage.new_conversation(settings.data_dir, note="", orig_name="b.mp4")
    _write_context_ir_state(settings, running["id"], status="running", prompt=None)

    with TestClient(create_app(settings)) as client:
        absent_detail = client.get(
            f"/api/conversations/{absent['id']}", headers=AUTH
        ).json()
        running_detail = client.get(
            f"/api/conversations/{running['id']}", headers=AUTH
        ).json()
        absent_response = client.get(
            f"/api/conversations/{absent['id']}/context-ir", headers=AUTH
        )
        running_response = client.get(
            f"/api/conversations/{running['id']}/context-ir", headers=AUTH
        )
        missing_response = client.get(
            f"/api/conversations/{'0' * 32}/context-ir", headers=AUTH
        )

    assert absent_detail["context_ir"] == {
        "status": "not_started",
        "available": False,
        "sha256": None,
    }
    assert running_detail["context_ir"] == {
        "status": "running",
        "available": False,
        "sha256": None,
    }
    assert absent_response.status_code == 404
    assert absent_response.json() == {"detail": "context_ir_not_found"}
    assert running_response.status_code == 409
    assert running_response.json() == {"detail": "context_ir_not_ready"}
    assert missing_response.status_code == 404
    assert missing_response.json() == {"detail": "not found"}


@pytest.mark.parametrize("tamper", ["hash", "status"])
def test_context_ir_hash_or_status_tamper_fails_closed(tmp_path, tamper):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    state_path = _write_context_ir_state(settings, meta["id"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if tamper == "hash":
        state["ir"]["optimized_prompt_sha256"] = "0" * 64
    else:
        state["ir"]["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        detail = client.get(f"/api/conversations/{meta['id']}", headers=AUTH)
        prompt_response = client.get(
            f"/api/conversations/{meta['id']}/context-ir", headers=AUTH
        )

    assert detail.json()["context_ir"] == {
        "status": "invalid",
        "available": False,
        "sha256": None,
    }
    assert "优化后的精确提示词" not in detail.text
    assert prompt_response.status_code == 409
    assert prompt_response.json() == {"detail": "context_ir_invalid"}


def test_detail_submit_enabled_follows_config(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True)
    with TestClient(create_app(settings)) as c:
        meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
        r = c.get(f"/api/conversations/{meta['id']}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["submit_enabled"] is True


def test_old_meta_remains_readable_but_is_derived_read_only(tmp_path):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    stored = storage.load_meta(settings.data_dir, meta["id"])
    stored.pop("schema_version")
    stored.pop("dialogue_mode")
    (settings.data_dir / meta["id"] / "meta.json").write_text(
        __import__("json").dumps(stored), encoding="utf-8"
    )
    with TestClient(create_app(settings)) as client:
        detail = client.get(f"/api/conversations/{meta['id']}", headers=AUTH).json()
    assert detail["read_only"] is True
    assert detail["dialogue"] == {"mode": "auto", "lines": [], "auto_lines": []}


def test_creation_preserves_voice_modes_and_translate_requires_language(tmp_path, video_1s):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        with open(video_1s, "rb") as source:
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", source, "video/mp4")},
                data={"voice_mode": "translate"},
            )
        assert response.status_code == 422
        with open(video_1s, "rb") as source:
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", source, "video/mp4")},
                data={"voice_mode": "rewrite"},
            )
        assert response.status_code == 201
        meta = storage.load_meta(settings.data_dir, response.json()["id"])
    assert meta["voice_mode"] == "rewrite"
    assert meta["dialogue_mode"] == "auto"


def test_postprocess_rejects_legacy_read_only_session(tmp_path):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    stored = storage.load_meta(settings.data_dir, meta["id"])
    stored.pop("schema_version")
    (settings.data_dir / meta["id"] / "meta.json").write_text(
        __import__("json").dumps(stored), encoding="utf-8"
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{meta['id']}/postprocess",
            headers=AUTH,
            json={"confirm": True, "options": {"remove_subtitle": True}},
        )
    assert response.status_code == 409
    assert response.json() == {"detail": "read_only"}


def test_detail_404(client):
    assert client.get(f"/api/conversations/{'0' * 32}", headers=AUTH).status_code == 404
    assert client.get("/api/conversations/..", headers=AUTH).status_code == 404


def test_files_endpoint(client, video_1s, settings):
    cid = _make_conv(client, video_1s)
    cdir = settings.data_dir / cid
    (cdir / "generated.mp4").write_bytes(b"fake-video")
    (cdir / "work" / "contact_sheet.jpg").write_bytes(b"sheet")
    (cdir / "work" / "keyframes").mkdir()
    (cdir / "work" / "keyframes" / "k01.jpg").write_bytes(b"k")

    r = client.get(f"/api/conversations/{cid}/files/generated.mp4", headers=AUTH)
    assert r.status_code == 200 and r.content == b"fake-video"
    r = client.get(f"/api/conversations/{cid}/files/source.mp4", headers=AUTH)
    assert r.status_code == 200  # 上传落盘的源视频（_make_conv 已产出 source.<ext>）
    r = client.get(f"/api/conversations/{cid}/files/contact_sheet.jpg", headers=AUTH)
    assert r.status_code == 200 and r.content == b"sheet"
    r = client.get(f"/api/conversations/{cid}/files/keyframes/k01.jpg", headers=AUTH)
    assert r.status_code == 200 and r.content == b"k"

    # 生成物落盘后 has_* 翻真
    r = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert r.json()["has_source"] is True
    assert r.json()["has_video"] is True


def test_files_traversal_404(client, video_1s):
    cid = _make_conv(client, video_1s)
    for name in ["../meta.json",
                 "keyframes/..%2Fmeta.json",
                 "keyframes/..%2F..%2Fmeta.json",
                 "keyframes/sub/dir.jpg"]:
        r = client.get(f"/api/conversations/{cid}/files/{name}", headers=AUTH)
        assert r.status_code == 404, name


def test_files_not_whitelisted_404(client, video_1s):
    cid = _make_conv(client, video_1s)
    for name in ["meta.json", "preview.exe", "work/meta.json", "keyframes/nope.jpg"]:
        r = client.get(f"/api/conversations/{cid}/files/{name}", headers=AUTH)
        assert r.status_code == 404, name


def test_files_requires_auth(client, video_1s):
    cid = _make_conv(client, video_1s)
    assert client.get(f"/api/conversations/{cid}/files/preview.mp4").status_code == 401
