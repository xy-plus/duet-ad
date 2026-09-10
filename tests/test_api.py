from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import UploadTooLarge, _write_upload, create_app
from app.store import (
    ProjectStateError,
    ProjectStore,
    canonical_json_sha256,
    creation_request_digest,
)


AUTH = {"Authorization": "Bearer test-secret"}
GENERATION_REQUEST = {
    "version": 1,
    "output": {"aspect_ratio": "16:9", "resolution": "480p", "fit_mode": "auto"},
    "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
    "dialogue": {"mode": "auto_rewrite", "target_language": "日语"},
    "replacement_guidance": None,
}
REPLACEMENT_GUIDANCE = {
    "instruction": "替换产品",
    "image_field": "replacement_image",
}


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_subtitles_json_rejects_non_boolean(value):
    from fastapi import HTTPException
    from app.main import _parse_generation_request
    with pytest.raises(HTTPException) as error:
        _parse_generation_request(json.dumps({**GENERATION_REQUEST, "add_subtitles": value}), "internal")
    assert error.value.status_code == 422


@pytest.mark.parametrize("value", [True, False])
def test_subtitles_json_normalization_and_legacy_default(value):
    from app.main import _parse_generation_request
    assert _parse_generation_request(json.dumps({**GENERATION_REQUEST, "add_subtitles": value}), "internal")["add_subtitles"] is value
    assert _parse_generation_request(json.dumps(GENERATION_REQUEST), "internal")["add_subtitles"] is False


def test_subtitle_fingerprint_preserves_legacy_and_binds_enabled():
    from app.store import generation_request_sha256
    legacy = canonical_json_sha256(GENERATION_REQUEST)
    assert generation_request_sha256(GENERATION_REQUEST) == legacy
    assert generation_request_sha256({**GENERATION_REQUEST, "add_subtitles": False}) == legacy
    assert generation_request_sha256({**GENERATION_REQUEST, "add_subtitles": True}) != legacy


def test_internal_legacy_project_replays_with_subtitles_off(tmp_path, monkeypatch):
    import importlib
    main_module = importlib.import_module("app.main")
    parser = main_module._parse_generation_request
    def legacy_parser(*args):
        value = parser(*args)
        value.pop("add_subtitles", None)
        return value
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        with monkeypatch.context() as legacy:
            legacy.setattr(main_module, "_parse_generation_request", legacy_parser)
            created = client.post("/api/conversations", headers=AUTH,
                                  data=create_payload(client_request_id="legacy-subtitles"),
                                  files={"file": ("source.mp4", b"x", "video/mp4")})
        assert created.status_code == 201
        assert "add_subtitles" not in created.json()["effective_request"]
        for enabled, status in [(False, 200), (True, 409)]:
            replay = client.post("/api/conversations", headers=AUTH,
                                 data=create_payload(client_request_id="legacy-subtitles", generation_request=json.dumps({**GENERATION_REQUEST, "add_subtitles": enabled})),
                                 files={"file": ("source.mp4", b"x", "video/mp4")})
            assert replay.status_code == status
            if not enabled:
                assert replay.json()["id"] == created.json()["id"]


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
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={"status": "succeeded", "percent": 100, "stage": "completed"},
            generation={"status": "succeeded", "stage": "completed", "error": None},
        )


class ExplodingPipeline:
    def run(self, _project_id: str) -> None:
        raise RuntimeError("background exploded")


class BlockingCountingPipeline:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.run_count = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, project_id: str) -> None:
        self.run_count += 1
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release pipeline")
        SuccessfulPipeline(self.store).run(project_id)


class NeverRunPipeline:
    def __init__(self) -> None:
        self.run_count = 0

    def run(self, _project_id: str) -> None:
        self.run_count += 1
        raise AssertionError("idempotency replay must not start the pipeline")


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


def create_payload(*, replacement_guidance=None, **overrides):
    generation_request = {
        **GENERATION_REQUEST,
        "replacement_guidance": replacement_guidance,
    }
    data = {
        "generation_request": json.dumps(generation_request, ensure_ascii=False),
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


def test_get_settings_requires_the_canonical_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ACCESS_TOKEN", "test-access")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://studio.example")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "old-key-must-not-be-used")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        get_settings()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "  canonical-deepseek-key  ")
    assert get_settings().deepseek_api_key == "canonical-deepseek-key"


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
    ).fusion_agent == "deepseek"
    assert Settings(
        access_token="test-secret",
        public_base_url="https://studio.example",
        fusion_agent="codex",
    ).fusion_agent == "codex"
    for unsupported_agent in ("gpt", "other"):
        with pytest.raises(ValueError, match="FUSION_AGENT"):
            Settings(
                access_token="test-secret",
                public_base_url="https://studio.example",
                fusion_agent=unsupported_agent,
            )


def test_fusion_is_internal_and_client_cannot_select_it(tmp_path: Path) -> None:
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
        assert "fusion" not in capability["minimal_creation"]
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="fusion-default"),
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        assert response.status_code == 201
        assert response.json()["effective_request"]["fusion"] == {
            "agent": "deepseek",
        }

        for index, client_fusion in enumerate(
            (None, {}, {"agent": "codex"}, {"agent": "other"})
        ):
            rejected = client.post(
                "/api/conversations",
                headers=AUTH,
                data={
                    "generation_request": json.dumps({
                        **GENERATION_REQUEST,
                        "fusion": client_fusion,
                    }),
                    "client_request_id": f"fusion-client-value-{index}",
                },
                files={"file": ("source.mp4", b"source", "video/mp4")},
            )
            assert rejected.status_code == 422
            assert rejected.json()["detail"]["code"] == "invalid_generation_request"
        assert len(client.get("/api/conversations", headers=AUTH).json()) == 1


def test_creation_request_rejects_oversized_declared_body_before_parsing(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/conversations",
            headers={
                **AUTH,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(64 * 1024 * 1024),
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
            data=create_payload(replacement_guidance=REPLACEMENT_GUIDANCE),
            files={
                "file": ("../../nested\\clip.mp4", b"source-video", "video/mp4"),
                "replacement_image": ("../images\\product.png", b"replacement", "image/png"),
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        project_id = body["id"]
        assert body["status"] == "queued"
        assert set(body["creation_input"]) == {"version", "source", "replacement_image", "replacement_images"}
        assert body["creation_input"] == {
            "version": 1,
            "source": {"mode": "upload", "filename": "clip.mp4", "bytes": len(b"source-video")},
            "replacement_image": {
                "filename": "product.png",
                "bytes": len(b"replacement"),
                "media_type": "image/png",
                "preview_url": f"/api/conversations/{project_id}/creation-input/replacement-image",
            },
            "replacement_images": [{
                "filename": "product.png",
                "bytes": len(b"replacement"),
                "media_type": "image/png",
                "preview_url": f"/api/conversations/{project_id}/creation-input/replacement-images/1",
            }],
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


def test_create_client_request_id_replays_same_project_without_rewriting_or_rerunning(
    tmp_path: Path,
):
    app = create_app(
        settings(tmp_path),
        pipeline_factory=lambda _settings, store, _register_asset: BlockingCountingPipeline(
            store
        ),
    )
    pipeline = app.state.pipeline
    request_with_replacement = {
        **GENERATION_REQUEST,
        "replacement_guidance": REPLACEMENT_GUIDANCE,
    }
    with TestClient(app) as client:
        try:
            first = client.post(
                "/api/conversations",
                headers=AUTH,
                data=create_payload(
                    client_request_id="durable-request-1",
                    replacement_guidance=REPLACEMENT_GUIDANCE,
                ),
                files={
                    "file": ("source.mp4", b"source-video", "video/mp4"),
                    "replacement_image": (
                        "product.png",
                        b"replacement-image",
                        "image/png",
                    ),
                },
            )
            assert first.status_code == 201, first.text
            assert pipeline.started.wait(timeout=2)
            project_id = first.json()["id"]
            source = app.state.store.path(project_id) / "inputs" / "source.mp4"
            replacement = (
                app.state.store.path(project_id) / "inputs" / "replacement-image"
            )
            source_stat = source.stat()
            replacement_stat = replacement.stat()

            replay = client.post(
                "/api/conversations",
                headers=AUTH,
                data=create_payload(
                    client_request_id="durable-request-1",
                    generation_request=json.dumps(
                        request_with_replacement,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ),
                ),
                files={
                    "file": ("source.mp4", b"source-video", "video/mp4"),
                    "replacement_image": (
                        "product.png",
                        b"replacement-image",
                        "image/png",
                    ),
                },
            )
            assert replay.status_code == 200, replay.text
            assert replay.json() == first.json()
            assert pipeline.run_count == 1
            assert source.stat().st_ino == source_stat.st_ino
            assert source.stat().st_mtime_ns == source_stat.st_mtime_ns
            assert replacement.stat().st_ino == replacement_stat.st_ino
            assert replacement.stat().st_mtime_ns == replacement_stat.st_mtime_ns
            assert len(client.get("/api/conversations", headers=AUTH).json()) == 1
            assert list(app.state.store.root.glob(".create-staging-*")) == []

            receipt = first.json()["input_receipt"]
            assert receipt["source"] == {
                "sha256": hashlib.sha256(b"source-video").hexdigest(),
                "bytes": len(b"source-video"),
            }
            assert receipt["replacement_image"] == {
                "sha256": hashlib.sha256(b"replacement-image").hexdigest(),
                "bytes": len(b"replacement-image"),
            }
            index_path = app.state.store.idempotency_index_path
            index_text = index_path.read_text(encoding="utf-8")
            assert os.stat(index_path).st_mode & 0o777 == 0o600
            assert "durable-request-1" not in index_text
            assert "source-video" not in index_text
            assert "replacement-image" not in index_text
            assert len(json.loads(index_text)["entries"]) == 1
        finally:
            pipeline.release.set()
        wait_for_status(client, first.json()["id"], "succeeded")


def test_create_client_request_id_conflicts_on_any_bound_input_change(tmp_path: Path):
    app = create_app(
        settings(tmp_path),
        pipeline_factory=lambda _settings, store, _register_asset: BlockingCountingPipeline(
            store
        ),
    )
    pipeline = app.state.pipeline
    request_with_replacement = {
        **GENERATION_REQUEST,
        "replacement_guidance": REPLACEMENT_GUIDANCE,
    }
    with TestClient(app) as client:
        try:
            first = client.post(
                "/api/conversations",
                headers=AUTH,
                data=create_payload(
                    client_request_id="durable-conflict-1",
                    replacement_guidance=REPLACEMENT_GUIDANCE,
                ),
                files={
                    "file": ("source.mp4", b"source-a", "video/mp4"),
                    "replacement_image": ("product.png", b"image-a", "image/png"),
                },
            )
            assert first.status_code == 201
            assert pipeline.started.wait(timeout=2)

            changed_request = {
                **request_with_replacement,
                "output": {
                    **GENERATION_REQUEST["output"],
                    "resolution": "768p",
                },
            }
            conflicts = [
                (
                    create_payload(
                        client_request_id="durable-conflict-1",
                        replacement_guidance=REPLACEMENT_GUIDANCE,
                    ),
                    {
                        "file": ("source.mp4", b"source-b", "video/mp4"),
                        "replacement_image": (
                            "product.png",
                            b"image-a",
                            "image/png",
                        ),
                    },
                ),
                (
                    create_payload(
                        client_request_id="durable-conflict-1",
                        generation_request=json.dumps(changed_request),
                    ),
                    {
                        "file": ("source.mp4", b"source-a", "video/mp4"),
                        "replacement_image": (
                            "product.png",
                            b"image-a",
                            "image/png",
                        ),
                    },
                ),
                (
                    create_payload(
                        client_request_id="durable-conflict-1",
                        replacement_guidance=REPLACEMENT_GUIDANCE,
                    ),
                    {
                        "file": ("source.mp4", b"source-a", "video/mp4"),
                        "replacement_image": (
                            "product.png",
                            b"image-b",
                            "image/png",
                        ),
                    },
                ),
            ]
            for data, files in conflicts:
                conflict = client.post(
                    "/api/conversations",
                    headers=AUTH,
                    data=data,
                    files=files,
                )
                assert conflict.status_code == 409, conflict.text
                assert conflict.json()["detail"]["code"] == (
                    "client_request_id_conflict"
                )
            assert pipeline.run_count == 1
            assert len(client.get("/api/conversations", headers=AUTH).json()) == 1
        finally:
            pipeline.release.set()
        wait_for_status(client, first.json()["id"], "succeeded")


@pytest.mark.parametrize("client_request_id", [None, "", "   ", "x" * 65])
def test_client_request_id_is_required_nonempty_and_bounded(
    tmp_path: Path,
    client_request_id: str | None,
):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        data = create_payload()
        if client_request_id is None:
            data.pop("client_request_id")
        else:
            data["client_request_id"] = client_request_id
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=data,
            files={"file": ("source.mp4", b"same-source", "video/mp4")},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "invalid_client_request_id"
        assert client.get("/api/conversations", headers=AUTH).json() == []


def test_client_request_id_is_trimmed_before_idempotency_lookup(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        first = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="  trimmed-request  "),
            files={"file": ("source.mp4", b"same-source", "video/mp4")},
        )
        assert first.status_code == 201
        project_id = first.json()["id"]
        wait_for_status(client, project_id, "succeeded")

        replay = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="trimmed-request"),
            files={"file": ("source.mp4", b"same-source", "video/mp4")},
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == project_id
        assert replay.json()["client_request_id"] == "trimmed-request"
        assert len(client.get("/api/conversations", headers=AUTH).json()) == 1


@pytest.mark.parametrize("client_request_id", ["", "   "])
def test_store_create_from_staging_rejects_blank_client_request_id(
    tmp_path: Path,
    client_request_id: str,
):
    store = ProjectStore(tmp_path / "projects", acquire_process_lock=False)
    staging = store.root / ".create-staging-test"
    staging.mkdir()
    staged_source = staging / "source.mp4"
    staged_source.write_bytes(b"source")
    effective_request = {
        **GENERATION_REQUEST,
        "fusion": {"agent": "deepseek"},
    }
    generation_digest = canonical_json_sha256(effective_request)
    source_digest = hashlib.sha256(b"source").hexdigest()
    input_receipt = {
        "version": 1,
        "client_request_id": client_request_id,
        "generation_request_sha256": generation_digest,
        "source": {"sha256": source_digest, "bytes": len(b"source")},
        "replacement_image": None,
    }

    with pytest.raises(ValueError, match="client_request_id must be nonempty"):
        store.create_from_staging(
            effective_request,
            client_request_id,
            request_digest=creation_request_digest(
                generation_digest,
                source_digest,
                None,
            ),
            input_receipt=input_receipt,
            source_snapshot={
                "mode": "upload",
                "filename": "source.mp4",
                "bytes": len(b"source"),
            },
            staged_source=staged_source,
            replacement_snapshot=None,
            staged_replacement=None,
        )

    assert store.list() == []
    assert not store.idempotency_index_path.exists()


def test_client_request_id_replay_survives_app_restart(tmp_path: Path):
    configured = settings(tmp_path)
    first_app = create_app(configured, pipeline_factory=successful_factory)
    with TestClient(first_app) as client:
        created = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="restart-request-1"),
            files={"file": ("source.mp4", b"restart-source", "video/mp4")},
        )
        assert created.status_code == 201
        project_id = created.json()["id"]
        wait_for_status(client, project_id, "succeeded")

    second_app = create_app(
        configured,
        pipeline_factory=lambda _settings, _store, _register_asset: NeverRunPipeline(),
    )
    with TestClient(second_app) as client:
        replay = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="restart-request-1"),
            files={"file": ("source.mp4", b"restart-source", "video/mp4")},
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["id"] == project_id
        assert replay.json()["status"] == "succeeded"
        assert second_app.state.pipeline.run_count == 0
        assert len(client.get("/api/conversations", headers=AUTH).json()) == 1


@pytest.mark.parametrize("damage", ["missing", "malformed", "incomplete"])
def test_client_request_id_index_damage_fails_closed_without_new_project(
    tmp_path: Path,
    damage: str,
):
    configured = settings(tmp_path)
    first_app = create_app(configured, pipeline_factory=successful_factory)
    with TestClient(first_app) as client:
        created = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="damaged-index-request"),
            files={"file": ("source.mp4", b"damaged-index-source", "video/mp4")},
        )
        assert created.status_code == 201
        project_id = created.json()["id"]
        wait_for_status(client, project_id, "succeeded")
        index_path = first_app.state.store.idempotency_index_path

    if damage == "missing":
        index_path.unlink()
    elif damage == "malformed":
        index_path.write_text("{", encoding="utf-8")
    else:
        index_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": {
                        "0" * 64: {
                            "project_id": project_id,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    second_app = create_app(
        configured,
        pipeline_factory=lambda _settings, _store, _register_asset: NeverRunPipeline(),
    )
    with TestClient(second_app) as client:
        replay = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="damaged-index-request"),
            files={"file": ("source.mp4", b"damaged-index-source", "video/mp4")},
        )
        assert replay.status_code == 503, replay.text
        assert replay.json()["detail"]["code"] == "idempotency_index_unavailable"
        assert second_app.state.pipeline.run_count == 0
        projects = client.get("/api/conversations", headers=AUTH).json()
        assert [item["id"] for item in projects] == [project_id]
        assert list(second_app.state.store.root.glob(".create-staging-*")) == []


def test_generation_request_accepts_only_the_closed_v1_shape(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        invalid_requests = [
            None,
            "{",
            "[]",
            "null",
            '"text"',
            json.dumps({}),
            json.dumps({**GENERATION_REQUEST, "extra": True}),
            json.dumps({**GENERATION_REQUEST, "version": True}),
            json.dumps({**GENERATION_REQUEST, "version": 2}),
            json.dumps({
                **GENERATION_REQUEST,
                "output": {"aspect_ratio": "16:9", "resolution": "480p"},
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "output": {
                    **GENERATION_REQUEST["output"],
                    "extra": True,
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "processing": {
                    "optimize_image": 1,
                    "remove_subtitle": True,
                    "remove_logo": True,
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "processing": {
                    "optimize_image": False,
                    "remove_subtitle": True,
                    "remove_logo": True,
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "processing": {
                    **GENERATION_REQUEST["processing"],
                    "extra": False,
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {"mode": "manual", "target_language": "日语"},
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {"mode": "generate", "target_language": "日语"},
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {
                    "mode": "generate",
                    "target_language": "日语",
                    "brief": "   ",
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {
                    "mode": "generate",
                    "target_language": "日语",
                    "brief": "长" * 1001,
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {"mode": "auto_rewrite", "target_language": "日语", "brief": "介绍产品"},
            }),
            json.dumps({**GENERATION_REQUEST, "dialogue": None}),
            json.dumps({**GENERATION_REQUEST, "dialogue": {}}),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {"target_language": "日语", "brief": "介绍产品"},
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "dialogue": {"mode": "auto_rewrite"},
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "replacement_guidance": {
                    "instruction": "替换产品",
                    "image_field": "image",
                },
            }),
            json.dumps({
                **GENERATION_REQUEST,
                "replacement_guidance": {
                    "instruction": "   ",
                    "image_field": "replacement_image",
                },
            }),
            '{"version":1,"version":1,"output":{},"processing":{},"dialogue":{},"replacement_guidance":null}',
        ]
        for index, raw in enumerate(invalid_requests):
            data = create_payload(client_request_id=f"invalid-generation-{index}")
            if raw is None:
                data.pop("generation_request")
            else:
                data["generation_request"] = raw
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                data=data,
                files={"file": ("source.mp4", b"x", "video/mp4")},
            )
            assert response.status_code == 422
            assert response.json()["detail"]["code"] == "invalid_generation_request"

        blank_language = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(
                client_request_id="blank-target-language",
                generation_request=json.dumps({
                    **GENERATION_REQUEST,
                    "dialogue": {
                        "mode": "auto_rewrite",
                        "target_language": "   ",
                    },
                }),
            ),
            files={"file": ("source.mp4", b"x", "video/mp4")},
        )
        assert blank_language.status_code == 422
        assert blank_language.json()["detail"]["code"] == "target_language_required"
        assert client.get("/api/conversations", headers=AUTH).json() == []


def test_generation_request_accepts_generate_mode_with_brief(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        capability = client.get("/api/capabilities", headers=AUTH).json()
        assert capability["minimal_creation"]["dialogue"] == {
            "modes": ["auto_rewrite", "generate"],
            "mode": "auto_rewrite",
            "translation": True,
            "max_brief_chars": 1000,
            "optional": True,
            "default": {
                "mode": "auto_rewrite",
                "target_language": "与原视频相同",
            },
        }
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(
                client_request_id="generate-dialogue",
                generation_request=json.dumps({
                    **GENERATION_REQUEST,
                    "dialogue": {
                        "mode": "generate",
                        "target_language": " 日语 ",
                        "brief": " 用轻松口吻介绍新产品 ",
                    },
                }),
            ),
            files={"file": ("source.mp4", b"x", "video/mp4")},
        )
        assert response.status_code == 201
        assert response.json()["effective_request"]["dialogue"] == {
            "mode": "generate",
            "target_language": "日语",
            "brief": "用轻松口吻介绍新产品",
        }


def test_generation_request_fills_dialogue_defaults_for_legacy_clients(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        legacy_mode = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(
                client_request_id="legacy-no-mode",
                generation_request=json.dumps({
                    **GENERATION_REQUEST,
                    "dialogue": {"target_language": " 中文 "},
                }),
            ),
            files={"file": ("source.mp4", b"x", "video/mp4")},
        )
        assert legacy_mode.status_code == 201
        assert legacy_mode.json()["effective_request"]["dialogue"] == {
            "mode": "auto_rewrite",
            "target_language": "中文",
        }

        legacy_dialogue = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(
                client_request_id="legacy-no-dialogue",
                generation_request=json.dumps({
                    key: value
                    for key, value in GENERATION_REQUEST.items()
                    if key != "dialogue"
                }),
            ),
            files={"file": ("source.mp4", b"x", "video/mp4")},
        )
        assert legacy_dialogue.status_code == 201
        assert legacy_dialogue.json()["effective_request"]["dialogue"] == {
            "mode": "auto_rewrite",
            "target_language": "与原视频相同",
        }


def test_creation_multipart_surface_rejects_extra_and_duplicate_fields(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        extra = client.post(
            "/api/conversations",
            headers=AUTH,
            data={
                **create_payload(client_request_id="extra-multipart-field"),
                "unexpected": "value",
            },
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        assert extra.status_code == 422
        assert extra.json()["detail"]["code"] == "invalid_creation_form"

        duplicate = client.post(
            "/api/conversations",
            headers=AUTH,
            files=[
                ("file", ("source.mp4", b"source", "video/mp4")),
                ("client_request_id", (None, "duplicate-one")),
                ("client_request_id", (None, "duplicate-two")),
                (
                    "generation_request",
                    (None, json.dumps(GENERATION_REQUEST, ensure_ascii=False)),
                ),
            ],
        )
        assert duplicate.status_code == 422
        assert duplicate.json()["detail"]["code"] == "invalid_creation_form"
        assert client.get("/api/conversations", headers=AUTH).json() == []
        assert list(app.state.store.root.glob(".create-staging-*")) == []


def test_source_and_replacement_inputs_must_match_the_v1_contract(tmp_path: Path):
    app = create_app(settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        missing_source = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="missing-source"),
        )
        assert missing_source.status_code == 400
        assert missing_source.json()["detail"]["code"] == "source_exactly_one_required"

        duplicate_source = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(
                client_request_id="duplicate-source",
                reference_url="https://media.example/source.mp4",
            ),
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        assert duplicate_source.status_code == 400
        assert duplicate_source.json()["detail"]["code"] == "source_exactly_one_required"

        missing_replacement_image = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(
                client_request_id="missing-replacement-image",
                replacement_guidance=REPLACEMENT_GUIDANCE,
            ),
            files={"file": ("source.mp4", b"source", "video/mp4")},
        )
        assert missing_replacement_image.status_code == 422
        assert missing_replacement_image.json()["detail"]["code"] == (
            "replacement_pair_required"
        )

        missing_replacement_guidance = client.post(
            "/api/conversations",
            headers=AUTH,
            data=create_payload(client_request_id="missing-replacement-guidance"),
            files={
                "file": ("source.mp4", b"source", "video/mp4"),
                "replacement_image": ("product.png", b"image", "image/png"),
            },
        )
        assert missing_replacement_guidance.status_code == 422
        assert missing_replacement_guidance.json()["detail"]["code"] == (
            "replacement_pair_required"
        )
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
            data=create_payload(replacement_guidance=REPLACEMENT_GUIDANCE),
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
            "replacement_images": [],
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
            data=create_payload(client_request_id="request-2"),
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
        voice = app.state.store.path(first["id"]) / "work" / "voice-reference.wav"
        voice.write_bytes(b"RIFFvoice-reference")
        voice_signed = urlsplit(app.state.register_asset(first["id"], voice))
        assert voice_signed.path.endswith(".wav")
        voice_response = client.head(f"{voice_signed.path}?{voice_signed.query}")
        assert voice_response.status_code == 200
        assert voice_response.headers["content-type"].startswith("audio/wav")
        assert client.get(f"{signed.path}?token=wrong").status_code == 404
        assert client.head(f"{signed.path}?token=wrong").status_code == 404
        assert client.get(
            f"/internal/provider-assets/{first['id']}/different?token={token}"
        ).status_code == 404
        assert client.get(
            f"/internal/provider-assets/{second['id']}/{asset_id}?token={token}"
        ).status_code == 404
        source.write_bytes(b"other")
        assert len(b"other") == len(b"first")
        assert client.get(f"{signed.path}?{signed.query}").status_code == 404

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
            data=create_payload(replacement_guidance=REPLACEMENT_GUIDANCE),
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


@pytest.mark.parametrize(
    ("unsupported_status", "project_progress"),
    [
        (
            "processing",
            {"status": "running", "percent": 1, "stage": "running"},
        ),
        (
            "done",
            {"status": "succeeded", "percent": 100, "stage": "completed"},
        ),
    ],
)
def test_store_rejects_unsupported_status_values_without_normalizing(
    tmp_path: Path,
    unsupported_status: str,
    project_progress: dict,
):
    store = ProjectStore(tmp_path / "projects", acquire_process_lock=False)
    project_id, _project, _state = store.create({}, "unsupported-status-test")

    with pytest.raises(ProjectStateError, match="invalid top-level project status"):
        store.update(
            project_id,
            status=unsupported_status,
            project_progress=project_progress,
        )

    assert store.read(project_id)["status"] == "queued"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.pop("schema"),
        lambda state: state.pop("version"),
        lambda state: state.update({"version": 0}),
        lambda state: state.update({"version": True}),
        lambda state: state.update({"version": 1.0}),
    ],
)
def test_store_rejects_unsupported_project_state_contract(
    tmp_path: Path,
    mutate,
):
    store = ProjectStore(tmp_path / "projects", acquire_process_lock=False)
    project_id, project, state = store.create({}, "unsupported-state-contract")
    mutate(state)
    (project / "state.json").write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ProjectStateError, match="unsupported project state schema"):
        store.read(project_id)


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
    malformed_id, malformed_dir, _ = store.create({}, "malformed-state")
    wrong_id, wrong_dir, wrong_state = store.create({}, "wrong-id-state")
    percent_id, percent_dir, percent_state = store.create({}, "bad-percent-state")
    unsupported_id, unsupported_dir, unsupported_state = store.create(
        {}, "unsupported-state"
    )
    active_id, _active_dir, _ = store.create({}, "active-state")
    (malformed_dir / "state.json").write_text("{", encoding="utf-8")
    wrong_state["id"] = "0" * 32
    (wrong_dir / "state.json").write_text(json.dumps(wrong_state), encoding="utf-8")
    percent_state["project_progress"]["percent"] = 999
    (percent_dir / "state.json").write_text(json.dumps(percent_state), encoding="utf-8")
    unsupported_state["version"] = 0
    (unsupported_dir / "state.json").write_text(
        json.dumps(unsupported_state), encoding="utf-8"
    )
    frozen_invalid_states = {
        project_id: (store.path(project_id) / "state.json").read_bytes()
        for project_id in (malformed_id, wrong_id, percent_id, unsupported_id)
    }

    store.fail_incomplete_after_restart()

    for project_id, frozen_payload in frozen_invalid_states.items():
        assert (store.path(project_id) / "state.json").read_bytes() == frozen_payload
        with pytest.raises(ProjectStateError):
            store.read(project_id)
    restarted = store.read(active_id)
    assert restarted["status"] == "failed"
    assert restarted["error"]["code"] == "process_restarted"


def _running_project(store: ProjectStore, client_request_id: str) -> tuple[str, Path]:
    project_id, project, _state = store.create({}, client_request_id)
    store.update(
        project_id,
        status="running",
        project_progress={"status": "running", "percent": 50, "stage": "stitch"},
        generation={"status": "running", "stage": "stitch", "error": None},
    )
    return project_id, project


def test_restart_downgrades_unproven_submission_started_receipts(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects", acquire_process_lock=False)
    proven_id, proven_dir = _running_project(store, "proven-submit")
    unproven_id, _unproven_dir = _running_project(store, "unproven-submit")
    for project_id in (proven_id, unproven_id):
        store.update(
            project_id,
            picsart_submission_receipts={
                "version": 1,
                "segments": {
                    "1": [{
                        "status": "submission_started",
                        "run_identity": "picsart",
                        "segment_index": 1,
                    }],
                },
            },
        )
    # Only the proven project has submit-log evidence that the provider
    # accepted the task.
    evidence_dir = (
        proven_dir / "logs" / "picsart-submit" / "segment-0001" / "attempt-1"
    )
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "output.json").write_text(
        json.dumps({
            "provider_record": {"outcome": "accepted", "task_id": "task-1"},
        }),
        encoding="utf-8",
    )

    restarted = store.fail_incomplete_after_restart()

    unproven = store.read(unproven_id)
    assert unproven["status"] == "failed"
    assert (
        unproven["picsart_submission_receipts"]["segments"]["1"][0]["status"]
        == "rejected"
    )
    proven = store.read(proven_id)
    assert proven["status"] == "failed"
    assert (
        proven["picsart_submission_receipts"]["segments"]["1"][0]["status"]
        == "submission_started"
    )
    by_project = {item["project_id"]: item for item in restarted}
    assert by_project[unproven_id]["downgraded_receipts"] == [{
        "segment_index": 1,
        "run_identity": "picsart",
    }]
    assert by_project[proven_id]["downgraded_receipts"] == []


def test_restart_auto_resumes_projects_that_pass_a_resume_gate(tmp_path: Path):
    configured = settings(tmp_path)
    store = ProjectStore(
        configured.data_dir / "projects", acquire_process_lock=False,
    )
    project_id, project = _running_project(store, "auto-resume-after-restart")
    pipelines: list[ResumablePipeline] = []
    app = create_app(
        configured,
        pipeline_factory=lambda _settings, app_store, _register_asset: (
            pipelines.append(ResumablePipeline(app_store)) or pipelines[-1]
        ),
    )

    with TestClient(app) as client:
        state = wait_for_status(client, project_id, "succeeded")

    assert state["error"] is None
    assert pipelines[0].validated == [project_id]
    assert pipelines[0].resumed == [project_id]
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "auto_resumed_after_restart" in trace
    assert '"operation": "resume_r2v"' in trace


def test_restart_auto_resume_stays_failed_with_guidance_when_gates_reject(
    tmp_path: Path,
):
    class BlockedPipeline:
        def validate_r2v_resume(self, _project_id: str) -> None:
            raise ValueError("submission_unknown receipts need reconciliation")

        def validate_pre_picsart_resume(self, _project_id: str) -> None:
            raise ValueError("no reusable pre-Picsart artifacts")

    configured = settings(tmp_path)
    store = ProjectStore(
        configured.data_dir / "projects", acquire_process_lock=False,
    )
    project_id, project = _running_project(store, "auto-resume-blocked")
    app = create_app(
        configured,
        pipeline_factory=lambda _settings, _store, _register_asset: BlockedPipeline(),
    )

    with TestClient(app) as client:
        response = client.get(f"/api/conversations/{project_id}", headers=AUTH)
        assert response.status_code == 200
        state = response.json()
        assert state["status"] == "failed"
        assert state["error"]["code"] == "process_restarted"

    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "auto_resume_after_restart_blocked" in trace
    assert "submission_unknown" in trace
