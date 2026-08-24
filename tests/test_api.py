import json
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings
from app import main as main_module, storage
from app.main import create_app


def _make_conv(client, video_1s, note=""):
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data={"note": note},
        )
    assert response.status_code == 201
    return response.json()["id"]


def test_list_conversations_shape(client, video_1s):
    cid = _make_conv(client, video_1s, note="n1")
    response = client.get("/api/conversations", headers=AUTH)
    assert response.status_code == 200
    (item,) = response.json()
    assert set(item) == {
        "id", "title", "note", "status", "navigation_status", "created_at",
        "has_video",
    }
    assert item["id"] == cid
    assert item["status"] == "queued"
    assert item["has_video"] is False


def test_html_entrypoints_and_app_contract_are_never_cached(client):
    for path in ("/", "/index.html", "/app.js", "/styles.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        conditional = client.get(
            path, headers={"If-None-Match": response.headers["etag"]}
        )
        assert conditional.status_code == 304
        assert conditional.headers["cache-control"] == "no-store"

    assert "no-store" not in client.get("/api/health").headers.get(
        "cache-control", ""
    )


def test_detail_shape_has_no_context_ir_contract(client, video_1s):
    cid = _make_conv(client, video_1s, note="n1")
    response = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert "context_ir" not in body
    assert set(body) == {
        "id", "title", "note", "status", "error", "created_at", "updated_at",
        "keyframes", "prompt", "source_prompt", "source_prompt_sha256", "segments",
        "voice_lines", "read_only", "duration_s", "fit_required", "fit_mode",
        "aspect_ratio", "resolution", "fit_profiles",
        "dialogue", "receipt_version", "generation", "has_source", "has_video",
        "navigation_status", "submit_enabled", "postprocess", "postprocess_enabled",
    }
    assert body["generation"] is None
    assert body["has_source"] is True
    assert 0.9 <= body["duration_s"] <= 1.1


@pytest.mark.parametrize(
    ("analysis", "generation_status", "has_video", "postprocess_status", "expected"),
    (
        ("queued", "succeeded", True, "done", "analysis_queued"),
        ("processing", "succeeded", True, "done", "analysis_processing"),
        ("failed", "succeeded", True, "done", "analysis_failed"),
        ("unexpected", None, False, None, "analysis_unknown"),
        ("done", None, True, "done", "analysis_complete"),
        ("done", "queued", False, None, "generation_queued"),
        ("done", "running", False, None, "generation_running"),
        ("done", "failed", False, None, "generation_failed"),
        ("done", "submission_unknown", False, None, "generation_submission_unknown"),
        ("done", "resume_required", False, None, "generation_resume_required"),
        ("done", "unexpected", True, "done", "generation_unknown"),
        ("done", "succeeded", False, None, "output_missing"),
        ("done", "succeeded", True, None, "completed"),
        ("done", "succeeded", True, "running", "postprocessing"),
        ("done", "succeeded", True, "failed", "postprocess_failed"),
        ("done", "succeeded", True, "done", "completed"),
    ),
)
def test_navigation_status_matrix_is_authoritative_and_consistent(
    tmp_path, monkeypatch, analysis, generation_status, has_video,
    postprocess_status, expected,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, note="nav", orig_name="a.mp4")
    cid = meta["id"]
    generation = None if generation_status is None else {
        "status": generation_status,
        "error": None,
        "attempt": 1,
        "client_request_id": "request-nav-test",
        "stage": "h3",
        "task_id": "secret-provider-task",
    }
    postprocess = (
        None if postprocess_status is None else {
            "status": postprocess_status,
        }
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        status=analysis,
        generation=generation,
        postprocess=postprocess,
    )
    output = settings.data_dir / cid / "generated.mp4"
    if has_video:
        output.write_bytes(b"accepted-by-test-double")
    monkeypatch.setattr(
        main_module,
        "_has_valid_generated_video",
        lambda _settings, candidate: (
            _settings.data_dir / candidate["id"] / "generated.mp4"
        ).is_file(),
    )

    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/conversations", headers=AUTH).json()[0]
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()

    assert listed["navigation_status"] == expected
    assert detail["navigation_status"] == expected
    assert listed["has_video"] is has_video
    assert detail["has_video"] is has_video
    assert "secret-provider-task" not in json.dumps(detail)


def test_removed_context_ir_endpoints_are_not_registered(tmp_path):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    path = f"/api/conversations/{meta['id']}/context-ir"
    with TestClient(create_app(settings)) as client:
        assert client.get(path, headers=AUTH).status_code == 404
        assert client.post(path, headers=AUTH, json={}).status_code in {404, 405}
        assert client.patch(path, headers=AUTH, json={}).status_code in {404, 405}
        assert client.post(path + "/translation", headers=AUTH, json={}).status_code in {404, 405}


def test_detail_submit_enabled_follows_config(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True)
    with TestClient(create_app(settings)) as client:
        meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
        response = client.get(f"/api/conversations/{meta['id']}", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["submit_enabled"] is True


def test_old_meta_remains_readable_but_is_derived_read_only(tmp_path):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
    stored = storage.load_meta(settings.data_dir, meta["id"])
    stored.pop("schema_version")
    stored.pop("dialogue_mode")
    (settings.data_dir / meta["id"] / "meta.json").write_text(
        json.dumps(stored), encoding="utf-8"
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
        json.dumps(stored), encoding="utf-8"
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
    assert client.get(f"/api/conversations/{cid}/files/generated.mp4", headers=AUTH).content == b"fake-video"
    assert client.get(f"/api/conversations/{cid}/files/source.mp4", headers=AUTH).status_code == 200
    assert client.get(f"/api/conversations/{cid}/files/contact_sheet.jpg", headers=AUTH).content == b"sheet"
    assert client.get(f"/api/conversations/{cid}/files/keyframes/k01.jpg", headers=AUTH).content == b"k"
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["has_source"] is True
    assert detail["has_video"] is True


def test_files_traversal_and_unknown_names_are_404(client, video_1s):
    cid = _make_conv(client, video_1s)
    names = ["../meta.json", "keyframes/..%2Fmeta.json", "meta.json", "preview.exe"]
    for name in names:
        assert client.get(f"/api/conversations/{cid}/files/{name}", headers=AUTH).status_code == 404


def test_files_require_auth(client, video_1s):
    cid = _make_conv(client, video_1s)
    assert client.get(f"/api/conversations/{cid}/files/preview.mp4").status_code == 401
