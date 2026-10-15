"""Independent HTTP contract checks for ordered replacement references."""

import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import public_credits
from app.main import create_app
from test_public_quotes import (
    BASE, RecordingPipeline, assert_idle, confirm, quote_env, quote_video, record_path,
)


@pytest.fixture
def reference_images():
    return [subprocess.run(
        ["/usr/bin/ffmpeg", "-threads", "1", "-f", "lavfi", "-i", f"color=c={color}:s=16x16",
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        check=True, capture_output=True,
    ).stdout for color in ("red", "blue", "green", "yellow")]


def image_fields(images):
    return [("replacement_image", (f"view-{i}.png", data, "image/png"))
            for i, data in enumerate(images, 1)]


def create_or_quote(client, auth, video, images, *, path=BASE, key="multiref01"):
    data = {"replacement_instruction": "Use all views of the same product"} if images else {}
    return client.post(path, headers={**auth, "Idempotency-Key": key}, data=data,
                       files=[("source_video", ("source.mp4", video.read_bytes(), "video/mp4")),
                              *image_fields(images)])


def assert_frozen_references(app, job, images):
    project_id = job["id"].removeprefix("vg_")
    state = app.state.store.read(project_id)
    receipt = state["input_receipt"]
    expected = [{"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for data in images]
    actual = receipt.get("replacement_images",
                         [] if receipt["replacement_image"] is None else [receipt["replacement_image"]])
    assert actual == expected
    assert receipt["replacement_image"] == (expected[0] if images else None)
    paths = state.get("replacement_image_paths", [] if not images else [state["replacement_image_path"]])
    assert len(paths) == len(images)
    assert [(app.state.store.path(project_id) / path).read_bytes() for path in paths] == images
    if images:
        assert state["replacement_image_path"] == paths[0] == "inputs/replacement-image"
    assert receipt["version"] == (2 if len(images) > 1 else 1)
    return state


@pytest.mark.parametrize("count", [0, 1, 2, 4])
@pytest.mark.parametrize("path", [BASE, "/api/v1/video-generations"])
def test_public_creation_and_confirmation_preserve_ordered_originals(
    quote_env, quote_video, reference_images, count, path,
):
    app, client, auth, settings = quote_env
    images = reference_images[:count]
    response = create_or_quote(client, auth, quote_video, images, path=path)
    assert response.status_code == 201, response.text
    if path == BASE:
        assert_idle(app, settings)
        response = confirm(client, auth, response.json()["quote_id"])
        assert response.status_code == 201, response.text
    assert_frozen_references(app, response.json(), images)
    assert app.state.pipeline.started.wait(2)
    assert len(app.state.pipeline.calls) == 1


@pytest.mark.parametrize("path", [BASE, "/api/v1/video-generations"])
def test_idempotency_binds_every_reference_and_its_order(
    quote_env, quote_video, reference_images, path,
):
    app, client, auth, settings = quote_env
    response = create_or_quote(client, auth, quote_video, reference_images, path=path)
    assert response.status_code == 201, response.text
    replay = create_or_quote(client, auth, quote_video, reference_images, path=path)
    assert replay.status_code == 200, replay.text
    assert replay.json().get("quote_id", replay.json().get("id")) == response.json().get("quote_id", response.json().get("id"))
    for changed in (reference_images[::-1], reference_images[:-1],
                    [*reference_images[:3], reference_images[0]]):
        conflict = create_or_quote(client, auth, quote_video, changed, path=path)
        assert conflict.status_code == 409, conflict.text


def test_four_reference_url_quote_replay_survives_url_expiry(
    quote_env, quote_video, reference_images, monkeypatch,
):
    app, client, auth, settings = quote_env

    async def download(url, destination, **kwargs):
        shutil.copyfile(quote_video, destination)
        return destination.stat().st_size

    monkeypatch.setattr("app.main.download_reference_url", download)
    headers = {**auth, "Idempotency-Key": "multirefurl01"}
    data = {"source_video_url": "https://example.com/temporary.mp4",
            "replacement_instruction": "Use every product view"}
    original = client.post(BASE, headers=headers, data=data, files=image_fields(reference_images))
    assert original.status_code == 201, original.text

    async def expired(*args, **kwargs):
        raise OSError("signed URL expired")

    monkeypatch.setattr("app.main.download_reference_url", expired)
    replay = client.post(BASE, headers=headers, data=data, files=image_fields(reference_images))
    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()
    reordered = client.post(BASE, headers=headers, data=data, files=image_fields(reference_images[::-1]))
    assert reordered.status_code == 409, reordered.text
    changed_last = client.post(BASE, headers=headers, data=data,
                               files=image_fields([*reference_images[:3], reference_images[0]]))
    assert changed_last.status_code == 409, changed_last.text
    assert_idle(app, settings)
    created = confirm(client, auth, original.json()["quote_id"])
    assert created.status_code == 201, created.text
    assert_frozen_references(app, created.json(), reference_images)


def test_concurrent_multiref_confirmation_does_not_duplicate(
    quote_env, quote_video, reference_images,
):
    app, client, auth, settings = quote_env
    response = create_or_quote(client, auth, quote_video, reference_images)
    assert response.status_code == 201, response.text
    quote_id = response.json()["quote_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: confirm(client, auth, quote_id), range(2)))
    assert sorted(result.status_code for result in results) == [200, 201]
    assert len({result.json()["id"] for result in results}) == 1
    assert_frozen_references(app, results[0].json(), reference_images)
    assert app.state.pipeline.started.wait(2)
    assert len(app.state.pipeline.calls) == 1
    assert public_credits.balance(settings.data_dir, "quote_owner")["reserved"] == 60


def test_confirmation_checks_integrity_of_fourth_frozen_reference(
    quote_env, quote_video, reference_images,
):
    app, client, auth, settings = quote_env
    response = create_or_quote(client, auth, quote_video, reference_images)
    assert response.status_code == 201, response.text
    quote_id = response.json()["quote_id"]
    directory = record_path(settings, quote_id).parent
    matching = [path for path in directory.rglob("*")
                if path.is_file() and path.read_bytes() == reference_images[3]]
    assert len(matching) == 1
    matching[0].write_bytes(reference_images[0])
    response = confirm(client, auth, quote_id)
    assert response.status_code >= 400, response.text
    assert_idle(app, settings)


def test_public_openapi_describes_repeated_image_uploads(quote_env):
    _app, client, auth, _settings = quote_env
    schema = client.get("/api/v1/openapi.json").json()
    for path in (BASE, "/api/v1/video-generations"):
        body = schema["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        if "$ref" in body:
            body = schema["components"]["schemas"][body["$ref"].split("/")[-1]]
        field = body["properties"]["replacement_image"]
        variants = field.get("anyOf", [field])
        arrays = [value for value in variants if value.get("type") == "array"]
        assert len(arrays) == 1, field
        item = arrays[0]["items"]
        assert item["type"] == "string"
        assert item.get("format") == "binary" or item.get("contentMediaType") == "application/octet-stream"


def test_all_frozen_views_have_individual_previews(quote_env, quote_video, reference_images):
    app, client, auth, _settings = quote_env
    response = create_or_quote(client, auth, quote_video, reference_images,
                               path="/api/v1/video-generations")
    assert response.status_code == 201, response.text
    state = assert_frozen_references(app, response.json(), reference_images)
    snapshots = state["creation_input"]["replacement_images"]
    assert len(snapshots) == 4
    for snapshot, original in zip(snapshots, reference_images):
        preview = client.get(snapshot["preview_url"],
                             headers={"Authorization": "Bearer private-test-secret"})
        assert preview.status_code == 200, preview.text
        assert preview.content == original
        assert preview.headers["content-type"] == "image/png"
    project_id = response.json()["id"].removeprefix("vg_")
    legacy = client.get(f"/api/conversations/{project_id}/creation-input/replacement-image",
                        headers={"Authorization": "Bearer private-test-secret"})
    assert legacy.status_code == 200
    assert legacy.content == reference_images[0]
    assert legacy.headers["content-type"] == "image/png"


def test_internal_store_replay_rejects_fourth_persisted_image_tampering(
    quote_env, quote_video, reference_images,
):
    app, client, _auth, _settings = quote_env
    request = {"version": 1,
               "output": {"aspect_ratio": "9:16", "resolution": "768p", "fit_mode": "auto"},
               "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
               "dialogue": {"mode": "auto_rewrite", "target_language": "English"},
               "replacement_guidance": {"instruction": "Use all views", "image_field": "replacement_image"}}

    def create():
        return client.post("/api/conversations",
                           headers={"Authorization": "Bearer private-test-secret"},
                           data={"client_request_id": "integrity001", "generation_request": json.dumps(request)},
                           files=[("file", ("source.mp4", quote_video.read_bytes(), "video/mp4")), *image_fields(reference_images)])

    created = create()
    assert created.status_code == 201, created.text
    state = assert_frozen_references(app, created.json(), reference_images)
    directory = app.state.store.path(created.json()["id"].removeprefix("vg_"))
    (directory / state["replacement_image_paths"][3]).write_bytes(reference_images[0])
    replay = create()
    assert replay.status_code >= 400, replay.text
    assert len(app.state.store.list()) <= 1


@pytest.mark.parametrize("path", [BASE, "/api/v1/video-generations"])
def test_five_references_are_rejected_without_reservation(
    quote_env, quote_video, reference_images, path,
):
    app, client, auth, settings = quote_env
    response = create_or_quote(client, auth, quote_video, [*reference_images, reference_images[0]], path=path)
    assert response.status_code == 422, response.text
    assert_idle(app, settings)


@pytest.mark.parametrize("path", [BASE, "/api/v1/video-generations", "/api/conversations"])
def test_four_images_are_not_limited_by_single_image_total_body_budget(
    quote_env, quote_video, reference_images, path,
):
    _app, _client, auth, settings = quote_env
    configured = replace(settings, max_upload_mb=1)
    app = create_app(configured, pipeline_factory=lambda *_: RecordingPipeline())
    # Valid PNGs with trailing data are accepted by the media decoder. Each is
    # below the existing per-file limit; together they exceed the old 10 MiB cap.
    images = [data + b"\0" * (3 * 1024 * 1024 - len(data)) for data in reference_images]
    with TestClient(app) as client:
        try:
            if path == "/api/conversations":
                request = {"version": 1,
                           "output": {"aspect_ratio": "9:16", "resolution": "768p", "fit_mode": "auto"},
                           "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
                           "dialogue": {"mode": "auto_rewrite", "target_language": "English"},
                           "replacement_guidance": {"instruction": "Use all views", "image_field": "replacement_image"}}
                response = client.post(path, headers={"Authorization": "Bearer private-test-secret"},
                                       data={"client_request_id": "bodybudget01", "generation_request": json.dumps(request)},
                                       files=[("file", ("source.mp4", quote_video.read_bytes(), "video/mp4")), *image_fields(images)])
            else:
                response = create_or_quote(client, auth, quote_video, images, path=path)
            assert response.status_code == 201, response.text
        finally:
            app.state.pipeline.release.set()
