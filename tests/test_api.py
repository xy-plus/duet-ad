from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import UploadTooLarge, _write_upload, create_app
from app.store import ProjectStore


AUTH = {"Authorization": "Bearer test-secret"}
GENERATION_REQUEST = {
    "version": 1,
    "output": {"aspect_ratio": "16:9", "resolution": "480p", "fit_mode": "auto"},
    "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
    "dialogue": {"mode": "auto_rewrite", "target_language": "日语"},
    "fusion": {"agent": "codex"},
    "replacement_guidance": {"instruction": "替换产品", "image_field": "replacement_image"},
}


class SuccessfulPipeline:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def run(self, project_id: str) -> None:
        project = self.store.path(project_id)
        output = project / "output" / "generated.mp4"
        output.write_bytes(b"generated-video")
        os.chmod(output, 0o600)
        self.store.update(
            project_id,
            status="done",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={"status": "succeeded", "percent": 100, "stage": "completed"},
            generation={"status": "succeeded", "stage": "completed", "error": None},
        )


class ExplodingPipeline:
    def run(self, _project_id: str) -> None:
        raise RuntimeError("background exploded")


class ResumablePipeline:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.validated: list[str] = []
        self.resumed: list[str] = []
        self.pre_validated: list[str] = []
        self.pre_resumed: list[str] = []

    def run(self, _project_id: str) -> None:
        raise AssertionError("full pipeline must not run during Picsart resume")

    def validate_r2v_resume(self, project_id: str) -> None:
        self.validated.append(project_id)

    def validate_pre_picsart_resume(self, project_id: str) -> None:
        self.pre_validated.append(project_id)

    def resume_r2v(self, project_id: str) -> None:
        self.resumed.append(project_id)
        self._succeed(project_id)

    def resume_pre_picsart(self, project_id: str) -> None:
        self.pre_resumed.append(project_id)
        self._succeed(project_id)

    def _succeed(self, project_id: str) -> None:
        project = self.store.path(project_id)
        output = project / "output" / "generated.mp4"
        output.write_bytes(b"resumed-video")
        self.store.update(
            project_id,
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={
                "status": "succeeded", "percent": 100, "stage": "completed",
            },
            generation={"status": "succeeded", "stage": "completed", "error": None},
        )


def settings(tmp_path: Path, *, max_upload_mb: int = 1) -> Settings:
    return Settings(
        access_token="test-secret",
        data_dir=tmp_path / "data",
        public_base_url="https://studio.example",
        max_upload_mb=max_upload_mb,
        store_lock_enabled=False,
    )


def successful_factory(_settings: Settings, store: ProjectStore, _register_asset):
    return SuccessfulPipeline(store)


def create_payload(**overrides):
    data = {
        "generation_request": json.dumps(GENERATION_REQUEST, ensure_ascii=False),
        "client_request_id": "request-1",
    }
    data.update(overrides)
    return data


def wait_for_status(client: TestClient, project_id: str, status: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/api/conversations/{project_id}", headers=AUTH)
        assert response.status_code == 200
        if response.json()["status"] == status:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"project did not reach {status}")


def test_default_and_blank_access_tokens_are_rejected(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="ACCESS_TOKEN"):
        Settings(access_token="")
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ACCESS_TOKEN"):
        get_settings()


@pytest.mark.parametrize(
    "base",
    ["", "http://public.example", "https://127.0.0.1:8000", "https://user@public.example"],
)
def test_public_base_url_must_be_a_public_https_origin(base: str):
    with pytest.raises(ValueError, match="PUBLIC_BASE_URL"):
        Settings(access_token="test-secret", public_base_url=base)


def test_fusion_agent_configuration_is_closed() -> None:
    assert Settings(
        access_token="test-secret",
        public_base_url="https://studio.example",
    ).fusion_agent == "codex"
    with pytest.raises(ValueError, match="FUSION_AGENT"):
        Settings(
            access_token="test-secret",
            public_base_url="https://studio.example",
            fusion_agent="other",
        )


def test_fusion_capability_and_request_are_frozen_at_ingress(tmp_path: Path) -> None:
    configured = Settings(
        access_token="test-secret",
        data_dir=tmp_path / "data",
        public_base_url="https://studio.example",
        fusion_agent="deepseek",
        store_lock_enabled=False,
    )
    app = create_app(configured, pipeline_factory=successful_factory)
    with TestClient(app) as client:
        capability = client.get("/api/capabilities", headers=AUTH).json()
        assert capability["minimal_creation"]["fusion"] == {
            "agents": ["codex", "deepseek"],
            "default_agent": "deepseek",
        }
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data={"generation_request": "{}", "client_request_id": "fusion-default"},
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        assert response.status_code == 201
        assert response.json()["effective_request"]["fusion"] == {
            "agent": "deepseek",
        }

        for index, invalid_fusion in enumerate((None, {}, {"agent": "other"})):
            invalid = client.post(
                "/api/conversations",
                headers=AUTH,
                data={
                    "generation_request": json.dumps({"fusion": invalid_fusion}),
                    "client_request_id": f"fusion-invalid-{index}",
                },
                files={"file": ("source.mp4", b"source", "video/mp4")},
            )
            assert invalid.status_code == 400
            assert invalid.json()["detail"]["code"] == "invalid_fusion_agent"


def test_creation_request_rejects_oversized_declared_body_before_parsing(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            headers={
                **AUTH,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(16 * 1024 * 1024),
            },
            content=b"",
        )
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "request_too_large"


def test_login_and_api_auth_fail_closed_with_structured_json_errors(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        unauthorized = client.get("/api/capabilities")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["detail"]["code"] == "unauthorized"

        malformed = client.post("/api/login", content=b"{", headers={"Content-Type": "application/json"})
        assert malformed.status_code == 400
        assert malformed.json()["detail"] == {"code": "invalid_json", "message": "请求体必须是 JSON 对象"}

        non_object = client.post("/api/login", json=["test-secret"])
        assert non_object.status_code == 400
        assert non_object.json()["detail"]["code"] == "invalid_json"

        assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
        assert client.post("/api/login", json={"token": "test-secret"}).json() == {"ok": True}
        assert client.get("/api/capabilities", headers=AUTH).status_code == 200


def test_failed_project_picsart_resume_requires_confirmation_and_skips_full_run(
    tmp_path: Path,
):
    app = create_app(
        settings(tmp_path),
        pipeline_factory=lambda _settings, store, _register_asset: ResumablePipeline(store),
    )
    project_id, _project, _state = app.state.store.create({}, "resume-request")
    app.state.store.update(
        project_id,
        status="failed",
        project_progress={"status": "failed", "percent": 57, "stage": "pipeline_failed"},
        generation={"status": "failed", "stage": "pipeline_failed", "error": "failed"},
    )

    with TestClient(app) as client:
        unconfirmed = client.post(
            f"/api/conversations/{project_id}/resume-picsart",
            headers=AUTH,
            json={"confirm": False},
        )
        assert unconfirmed.status_code == 400
        assert unconfirmed.json()["detail"]["code"] == "confirmation_required"

        accepted = client.post(
            f"/api/conversations/{project_id}/resume-picsart",
            headers=AUTH,
            json={"confirm": True},
        )
        assert accepted.status_code == 202
        assert accepted.json()["project_progress"]["stage"] == "picsart_resume_queued"
        terminal = wait_for_status(client, project_id, "succeeded")
        assert terminal["has_video"] is True
        assert app.state.pipeline.validated == [project_id]
        assert app.state.pipeline.resumed == [project_id]

        repeated = client.post(
            f"/api/conversations/{project_id}/resume-picsart",
            headers=AUTH,
            json={"confirm": True},
        )
        assert repeated.status_code == 409
        assert repeated.json()["detail"]["code"] == "project_not_failed"


def test_failed_project_pre_picsart_resume_is_explicit_and_uses_its_own_operation(
    tmp_path: Path,
):
    app = create_app(
        settings(tmp_path),
        pipeline_factory=lambda _settings, store, _register_asset: ResumablePipeline(store),
    )
    project_id, _project, _state = app.state.store.create({}, "pre-resume-request")
    app.state.store.update(
        project_id,
        status="failed",
        project_progress={"status": "failed", "percent": 24, "stage": "pipeline_failed"},
        generation={"status": "failed", "stage": "pipeline_failed", "error": "failed"},
    )

    with TestClient(app) as client:
        unconfirmed = client.post(
            f"/api/conversations/{project_id}/resume-pre-picsart",
            headers=AUTH,
            json={"confirm": False},
        )
        assert unconfirmed.status_code == 400
        assert unconfirmed.json()["detail"]["code"] == "confirmation_required"

        accepted = client.post(
            f"/api/conversations/{project_id}/resume-pre-picsart",
            headers=AUTH,
            json={"confirm": True},
        )
        assert accepted.status_code == 202
        assert accepted.json()["project_progress"]["stage"] == "pre_picsart_resume_queued"
        terminal = wait_for_status(client, project_id, "succeeded")
        assert terminal["has_video"] is True
        assert app.state.pipeline.pre_validated == [project_id]
        assert app.state.pipeline.pre_resumed == [project_id]
        assert app.state.pipeline.validated == []
        assert app.state.pipeline.resumed == []


def test_create_uses_exact_recovery_shape_fixed_paths_and_four_public_statuses(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={
                "file": ("../../nested\\clip.mp4", b"source-video", "video/mp4"),
                "replacement_image": ("../images\\product.png", b"replacement", "image/png"),
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        project_id = body["id"]
        assert body["status"] == "queued"
        assert set(body["creation_input"]) == {"version", "source", "replacement_image"}
        assert body["creation_input"] == {
            "version": 1,
            "source": {"mode": "upload", "filename": "clip.mp4", "bytes": len(b"source-video")},
            "replacement_image": {
                "filename": "product.png",
                "bytes": len(b"replacement"),
                "media_type": "image/png",
                "preview_url": f"/api/conversations/{project_id}/creation-input/replacement-image",
            },
        }

        project = app.state.store.path(project_id)
        assert {path.name for path in (project / "inputs").iterdir()} == {"source.mp4", "replacement-image"}
        assert (project / "inputs" / "source.mp4").read_bytes() == b"source-video"
        assert (project / "inputs" / "replacement-image").read_bytes() == b"replacement"
        assert not (tmp_path / "clip.mp4").exists()
        assert os.stat(project).st_mode & 0o777 == 0o700
        assert os.stat(project / "state.json").st_mode & 0o777 == 0o600
        assert os.stat(project / "inputs" / "source.mp4").st_mode & 0o777 == 0o600

        terminal = wait_for_status(client, project_id, "succeeded")
        assert terminal["project_progress"] == {"status": "succeeded", "percent": 100, "stage": "completed"}
        assert terminal["status"] in {"queued", "running", "succeeded", "failed"}
        assert client.get(f"/api/conversations/{project_id}/files/source.mp4", headers=AUTH).content == b"source-video"
        assert client.get(f"/api/conversations/{project_id}/files/generated.mp4", headers=AUTH).content == b"generated-video"
        replacement = client.get(
            f"/api/conversations/{project_id}/creation-input/replacement-image",
            headers=AUTH,
        )
        assert replacement.status_code == 200
        assert replacement.headers["content-type"].startswith("image/png")
        assert replacement.content == b"replacement"


def test_generation_request_and_missing_source_errors_are_stable_and_create_nothing(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        for raw in ("{", "[]", "null", '"text"'):
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                data={"generation_request": raw},
                files={"file": ("source.mp4", b"x", "video/mp4")},
            )
            assert response.status_code == 400
            assert response.json()["detail"]["code"] == "invalid_generation_request"
        missing = client.post(
            "/api/conversations",
            headers=AUTH,
            data={"generation_request": "{}"},
        )
        assert missing.status_code == 400
        assert missing.json()["detail"] == {"code": "source_missing", "message": "没有提供视频"}
        assert client.get("/api/conversations", headers=AUTH).json() == []


def test_source_and_replacement_limits_reject_before_project_creation(tmp_path: Path):
    app = create_app(settings(tmp_path, max_upload_mb=1), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        source = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={"file": ("large.mp4", b"x" * (1024 * 1024 + 1), "video/mp4")},
        )
        assert source.status_code == 413
        assert source.json()["detail"]["code"] == "upload_too_large"

        replacement = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={
                "file": ("source.mp4", b"x", "video/mp4"),
                "replacement_image": ("large.png", b"y" * (10 * 1024 * 1024 + 1), "image/png"),
            },
        )
        assert replacement.status_code == 413
        assert replacement.json()["detail"]["code"] == "upload_too_large"
        assert client.get("/api/conversations", headers=AUTH).json() == []


def test_stream_writer_never_publishes_or_leaves_partial_oversized_upload(tmp_path: Path):
    upload = UploadFile(filename="source.mp4", file=io.BytesIO(b"123456"), size=None)
    destination = tmp_path / "source.mp4"
    with pytest.raises(UploadTooLarge):
        asyncio.run(_write_upload(upload, destination, 5))
    assert not destination.exists()
    assert list(tmp_path.glob(".source.mp4.*.tmp")) == []


def test_reference_url_create_uses_injected_pinned_transport_without_network(tmp_path: Path):
    requests: list[httpx.Request] = []

    async def resolver(hostname: str, port: int):
        assert (hostname, port) == ("media.example", 443)
        return ["93.184.216.34"]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"remote-video")

    app = create_app(
        settings(tmp_path),
        pipeline_factory=successful_factory,
        reference_transport=httpx.MockTransport(handler),
        reference_resolver=resolver,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(reference_url="https://media.example/video.mp4"),
        )
        assert response.status_code == 201, response.text
        assert response.json()["creation_input"] == {
            "version": 1,
            "source": {"mode": "link", "reference_url": "https://media.example/video.mp4"},
            "replacement_image": None,
        }
        project = app.state.store.path(response.json()["id"])
        assert (project / "inputs" / "source.mp4").read_bytes() == b"remote-video"
    assert len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "media.example"
    assert requests[0].extensions["sni_hostname"] == "media.example"


def test_files_endpoint_exposes_only_two_fixed_aliases_even_if_state_is_poisoned(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        project_id = response.json()["id"]
        project = app.state.store.path(project_id)
        app.state.store.update(project_id, source_path="logs/trace.jsonl")

        assert client.get(f"/api/conversations/{project_id}/files/source.mp4").status_code == 401
        assert client.get(f"/api/conversations/{project_id}/files/source.mp4", headers=AUTH).content == b"source"
        for relative in (
            "inputs/source.mp4",
            "logs/trace.jsonl",
            "state.json",
            "%2e%2e%2flogs%2ftrace.jsonl",
            "source.mp4/extra",
        ):
            denied = client.get(
                f"/api/conversations/{project_id}/files/{relative}",
                headers=AUTH,
                follow_redirects=False,
            )
            assert denied.status_code == 404, (relative, denied.status_code, denied.text)
        assert (project / "logs" / "trace.jsonl").is_file()


def test_provider_asset_signature_binds_project_asset_id_token_and_real_path(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        first = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={"file": ("source.mp4", b"first", "video/mp4")},
        ).json()
        second = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={"file": ("source.mp4", b"second", "video/mp4")},
        ).json()
        source = app.state.store.path(first["id"]) / "inputs" / "source.mp4"
        signed = urlsplit(app.state.register_asset(first["id"], source))
        token = parse_qs(signed.query)["token"][0]
        asset_id = signed.path.rsplit("/", 1)[-1]
        assert asset_id.endswith(".mp4")

        delivered = client.get(f"{signed.path}?{signed.query}")
        assert delivered.status_code == 200
        assert delivered.content == b"first"
        metadata = client.head(f"{signed.path}?{signed.query}")
        assert metadata.status_code == 200
        assert metadata.content == b""
        assert metadata.headers["content-length"] == str(len(b"first"))
        assert metadata.headers["content-type"].startswith("video/mp4")
        assert client.get(f"{signed.path}?token=wrong").status_code == 404
        assert client.head(f"{signed.path}?token=wrong").status_code == 404
        assert client.get(
            f"/internal/provider-assets/{first['id']}/different?token={token}"
        ).status_code == 404
        assert client.get(
            f"/internal/provider-assets/{second['id']}/{asset_id}?token={token}"
        ).status_code == 404

        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"outside")
        with pytest.raises(ValueError, match="inside its project"):
            app.state.register_asset(first["id"], outside)
        symlink = app.state.store.path(first["id"]) / "inputs" / "escape.mp4"
        symlink.symlink_to(outside)
        with pytest.raises(ValueError, match="inside its project"):
            app.state.register_asset(first["id"], symlink)


def test_provider_asset_preserves_replacement_image_media_type(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        created = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={
                "file": ("source.mp4", b"source", "video/mp4"),
                "replacement_image": ("product.png", b"replacement", "image/png"),
            },
        ).json()
        replacement = (
            app.state.store.path(created["id"]) / "inputs" / "replacement-image"
        )
        signed = urlsplit(app.state.register_asset(created["id"], replacement))
        assert signed.path.endswith(".png")

        delivered = client.get(f"{signed.path}?{signed.query}")
        assert delivered.status_code == 200
        assert delivered.headers["content-type"].startswith("image/png")
        assert delivered.content == b"replacement"


def test_unhandled_background_failure_is_persisted_as_failed(tmp_path: Path):
    app = create_app(
        settings(tmp_path),
        pipeline_factory=lambda _settings, _store, _register_asset: ExplodingPipeline(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(),
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        assert response.status_code == 201
        failed = wait_for_status(client, response.json()["id"], "failed")
        assert failed["project_progress"]["status"] == "failed"
        assert failed["error"]["code"] == "pipeline_failed"
        assert failed["error"]["message"] == "后台任务执行失败"
        assert failed["generation"]["error"] == "pipeline_failed"
        assert "background exploded" not in json.dumps(failed)


def test_store_lock_permissions_and_restart_scan_fail_safe(tmp_path: Path):
    locked_root = tmp_path / "locked-projects"
    first = ProjectStore(locked_root)
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            ProjectStore(locked_root)
        assert os.stat(locked_root).st_mode & 0o777 == 0o700
        assert os.stat(locked_root / ".process.lock").st_mode & 0o777 == 0o600
    finally:
        first.close()
    reopened = ProjectStore(locked_root)
    reopened.close()

    store = ProjectStore(tmp_path / "recovery-projects", acquire_process_lock=False)
    malformed_id, malformed_dir, _ = store.create({}, "")
    wrong_id, wrong_dir, wrong_state = store.create({}, "")
    percent_id, percent_dir, percent_state = store.create({}, "")
    active_id, _active_dir, _ = store.create({}, "")
    (malformed_dir / "state.json").write_text("{", encoding="utf-8")
    wrong_state["id"] = "0" * 32
    (wrong_dir / "state.json").write_text(json.dumps(wrong_state), encoding="utf-8")
    percent_state["project_progress"]["percent"] = 999
    (percent_dir / "state.json").write_text(json.dumps(percent_state), encoding="utf-8")

    store.fail_incomplete_after_restart()

    for project_id in (malformed_id, wrong_id, percent_id):
        repaired = store.read(project_id)
        assert repaired["id"] == project_id
        assert repaired["status"] == "failed"
        assert repaired["error"]["code"] == "invalid_state"
        assert os.stat(store.path(project_id) / "state.json").st_mode & 0o777 == 0o600
    restarted = store.read(active_id)
    assert restarted["status"] == "failed"
    assert restarted["error"]["code"] == "process_restarted"
