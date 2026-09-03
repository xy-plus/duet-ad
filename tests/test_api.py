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
        "id", "title", "note", "status", "analysis_status",
        "navigation_status", "created_at", "updated_at", "duration_s",
        "segment_count", "thumbnail_path", "has_video", "project_progress",
    }
    assert item["id"] == cid
    assert item["status"] == "queued"
    assert item["has_video"] is False
    assert item["segment_count"] == 0
    assert item["thumbnail_path"] is None


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
        "id", "title", "note", "status", "analysis_status", "error",
        "created_at", "updated_at",
        "keyframes", "prompt", "source_prompt", "source_prompt_sha256", "segments",
        "voice_lines", "read_only", "duration_s", "fit_required", "fit_mode",
        "aspect_ratio", "resolution", "fit_profiles",
            "dialogue", "dialogue_review", "receipt_version", "generation", "has_source", "has_video",
        "navigation_status", "submit_enabled", "postprocess", "postprocess_enabled",
        "postprocess_capabilities", "image_optimization_prompt",
            "image_acceptance", "element_index",
            "generation_config", "generation_config_sha256",
            "project_progress", "effective_request", "input_receipt",
            "creation_input",
        }
    assert body["generation"] is None
    assert body["has_source"] is True
    assert 0.9 <= body["duration_s"] <= 1.1
    assert body["element_index"] is None
    assert body["creation_input"] is None
    assert "skill_milestone" not in body


def test_detail_returns_backend_published_element_index(tmp_path):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir, note="frozen", orig_name="a.mp4",
    )
    root = settings.data_dir / meta["id"]
    element_index = {
        "people": {"person-01": {"description": "人物", "occurrences": []}},
        "entities": {"entity-01": {"description": "杯子", "occurrences": []}},
        "scenes": {"scene-01": {"description": "室内", "occurrences": []}},
        "relations": {
            "relation-01": {
                "subject_key": "person-01", "predicate": "拿着",
                "object_key": "entity-01",
            }
        },
    }
    (root / "work").mkdir(parents=True, exist_ok=True)
    (root / "work" / "element_index.json").write_text(
        json.dumps(element_index, ensure_ascii=False), encoding="utf-8",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get(
            f"/api/conversations/{meta['id']}", headers=AUTH,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["element_index"] == element_index
    assert "skill_milestone" not in body


@pytest.mark.parametrize(
    (
        "analysis", "generation_status", "file_exists", "postprocess_status",
        "expected",
    ),
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
        # A running postprocess without its private frozen receipt is failed closed on startup.
        ("done", "succeeded", True, "running", "postprocess_failed"),
        ("done", "succeeded", True, "failed", "postprocess_failed"),
        ("done", "succeeded", True, "done", "completed"),
    ),
)
def test_navigation_status_matrix_is_authoritative_and_consistent(
    tmp_path, analysis, generation_status, file_exists,
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
    if file_exists:
        output.write_bytes(b"accepted-by-test-double")

    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/conversations", headers=AUTH).json()[0]
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()

    expected_has_video = bool(
        analysis == "done"
        and generation_status in {None, "succeeded"}
        and file_exists
    )
    assert listed["navigation_status"] == expected
    assert detail["navigation_status"] == expected
    assert listed["has_video"] is expected_has_video
    assert detail["has_video"] is expected_has_video
    assert "secret-provider-task" not in json.dumps(detail)


def test_list_and_detail_publish_persisted_completed_output_without_decoding(
    tmp_path,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir, note="fast list", orig_name="a.mp4"
    )
    cid = meta["id"]
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        generation={"status": "succeeded"},
        _postprocess_receipt={"version": 4, "options": {}},
    )
    (settings.data_dir / cid / "generated.mp4").write_bytes(
        b"persisted-but-not-decodable"
    )

    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/conversations", headers=AUTH)
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert listed.status_code == 200
    assert listed.json()[0]["has_video"] is True
    assert listed.json()[0]["project_progress"] == {
        "percent": 100,
        "status": "succeeded",
    }
    assert detail.status_code == 200
    assert detail.json()["has_video"] is True
    assert detail.json()["project_progress"] == {
        "percent": 100,
        "status": "succeeded",
    }


@pytest.mark.parametrize("generation_status", [None, "succeeded"])
def test_terminal_postprocess_failure_overrides_public_operation_not_analysis(
    tmp_path, monkeypatch, generation_status,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir, note="failed delivery", orig_name="a.mp4"
    )
    generation = None if generation_status is None else {
        "status": generation_status,
        "error": None,
        "attempt": 1,
        "client_request_id": "request-postprocess-failed",
        "stage": "h3",
    }
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        status="done",
        error=None,
        generation=generation,
        postprocess={
            "status": "failed",
            "error": "provider_rejected",
            "options": {"optimize_image": True},
            "frames": [],
            "segments": [],
        },
    )
    monkeypatch.setattr(
        main_module.postprocess, "recover_running", lambda _settings: []
    )

    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/conversations", headers=AUTH).json()[0]
        detail = client.get(
            f"/api/conversations/{meta['id']}", headers=AUTH
        ).json()

    assert listed["status"] == "failed"
    assert listed["analysis_status"] == "done"
    assert listed["navigation_status"] == "postprocess_failed"
    assert listed["has_video"] is False
    assert detail["status"] == "failed"
    assert detail["analysis_status"] == "done"
    assert detail["error"] == "provider_rejected"
    assert detail["navigation_status"] == "postprocess_failed"
    assert detail["has_video"] is False
    if generation_status is None:
        assert detail["generation"] is None
    else:
        assert detail["generation"]["status"] == "succeeded"


def test_detail_does_not_republish_legacy_internal_analysis_error(tmp_path):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir, note="legacy failure", orig_name="a.mp4"
    )
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        status="failed",
        error=(
            "LEGACY_MODEL_TAIL tokens used 42,452 "
            'provider_body={"authorization":"Bearer secret"}'
        ),
    )

    with TestClient(create_app(settings)) as client:
        detail = client.get(
            f"/api/conversations/{meta['id']}", headers=AUTH
        ).json()

    assert detail["status"] == "failed"
    assert detail["analysis_status"] == "failed"
    assert detail["error"] == "pipeline_failed"
    assert "LEGACY_MODEL_TAIL" not in json.dumps(detail)
    assert "tokens used" not in json.dumps(detail)
    assert "Bearer secret" not in json.dumps(detail)


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


def test_detail_auto_dialogue_only_publishes_spoken_with_scoped_legacy_compat(
    tmp_path,
):
    settings = make_settings(tmp_path)
    lines = [
        {
            "text": "spoken line",
            "start_s": 0.0,
            "end_s": 1.0,
            "classification": "spoken",
            "kept": True,
        },
        {
            "text": "sung lyrics",
            "start_s": 1.0,
            "end_s": 2.0,
            "classification": "sung",
            "kept": True,
        },
        {
            "text": "unclassified current line",
            "start_s": 2.0,
            "end_s": 3.0,
            "kept": True,
        },
        {
            "text": "discarded spoken line",
            "start_s": 3.0,
            "end_s": 4.0,
            "classification": "spoken",
            "kept": False,
        },
    ]
    current = storage.new_conversation(
        settings.data_dir, note="", orig_name="current.mp4"
    )
    storage.update_meta(
        settings.data_dir,
        current["id"],
        dialogue_mode="auto",
        voice_line_provenance=lines,
    )

    legacy = storage.new_conversation(
        settings.data_dir, note="", orig_name="legacy.mp4"
    )
    legacy_stored = storage.load_meta(settings.data_dir, legacy["id"])
    legacy_stored.pop("schema_version")
    legacy_stored["dialogue_mode"] = "auto"
    legacy_stored["voice_line_provenance"] = lines
    (settings.data_dir / legacy["id"] / "meta.json").write_text(
        json.dumps(legacy_stored), encoding="utf-8"
    )

    with TestClient(create_app(settings)) as client:
        current_detail = client.get(
            f"/api/conversations/{current['id']}", headers=AUTH
        ).json()
        legacy_detail = client.get(
            f"/api/conversations/{legacy['id']}", headers=AUTH
        ).json()

    spoken = [{"text": "spoken line", "start_s": 0.0, "end_s": 1.0}]
    assert current_detail["read_only"] is False
    assert current_detail["dialogue"] == {
        "mode": "auto",
        "lines": spoken,
        "auto_lines": spoken,
    }
    legacy_lines = [
        *spoken,
        {
            "text": "unclassified current line",
            "start_s": 2.0,
            "end_s": 3.0,
        },
    ]
    assert legacy_detail["read_only"] is True
    assert legacy_detail["dialogue"] == {
        "mode": "auto",
        "lines": legacy_lines,
        "auto_lines": legacy_lines,
    }


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
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
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
    generated = video_1s.read_bytes()
    (cdir / "generated.mp4").write_bytes(generated)
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        generation={"status": "succeeded"},
    )
    (cdir / "work" / "contact_sheet.jpg").write_bytes(b"sheet")
    (cdir / "work" / "keyframes").mkdir()
    (cdir / "work" / "keyframes" / "k01.jpg").write_bytes(b"k")
    assert client.get(
        f"/api/conversations/{cid}/files/generated.mp4", headers=AUTH
    ).content == generated
    assert client.get(f"/api/conversations/{cid}/files/source.mp4", headers=AUTH).status_code == 200
    assert client.get(f"/api/conversations/{cid}/files/contact_sheet.jpg", headers=AUTH).content == b"sheet"
    assert client.get(f"/api/conversations/{cid}/files/keyframes/k01.jpg", headers=AUTH).content == b"k"
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["has_source"] is True
    assert detail["has_video"] is True


def test_generated_file_uses_persisted_completion_and_presence_contract(
    client, video_1s, settings,
):
    cid = _make_conv(client, video_1s)
    generated = settings.data_dir / cid / "generated.mp4"
    generated.write_bytes(b"not-a-video")
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        generation={"status": "succeeded"},
    )
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert detail.status_code == 200
    assert detail.json()["has_video"] is True
    assert detail.json()["project_progress"] == {
        "percent": 100,
        "status": "succeeded",
    }
    artifact = client.get(
        f"/api/conversations/{cid}/files/generated.mp4", headers=AUTH
    )
    assert artifact.status_code == 200
    assert artifact.content == b"not-a-video"

    missing_cid = _make_conv(client, video_1s)
    storage.update_meta(
        settings.data_dir,
        missing_cid,
        status="done",
        generation={"status": "succeeded"},
    )
    missing = client.get(
        f"/api/conversations/{missing_cid}/files/generated.mp4", headers=AUTH
    )
    assert missing.status_code == 404


def test_files_traversal_and_unknown_names_are_404(client, video_1s):
    cid = _make_conv(client, video_1s)
    names = ["../meta.json", "keyframes/..%2Fmeta.json", "meta.json", "preview.exe"]
    for name in names:
        assert client.get(f"/api/conversations/{cid}/files/{name}", headers=AUTH).status_code == 404


def test_files_require_auth(client, video_1s):
    cid = _make_conv(client, video_1s)
    assert client.get(f"/api/conversations/{cid}/files/preview.mp4").status_code == 401
