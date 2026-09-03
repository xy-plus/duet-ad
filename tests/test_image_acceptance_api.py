"""Image acceptance compatibility calls project onto one operation."""

from fastapi.testclient import TestClient
import pytest

from conftest import AUTH, make_settings

from app import frame_fit, long_generation, postprocess, storage
from app.main import create_app


def _conversation(settings) -> tuple[str, dict]:
    created = storage.new_conversation(
        settings.data_dir, "manual image acceptance", "source.mp4"
    )
    storage.update_meta(
        settings.data_dir,
        created["id"],
        status="done",
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False,
                "remove_brand": False,
                "optimize_image": True,
            },
        },
    )
    return created["id"], storage.load_meta(settings.data_dir, created["id"])


def test_detail_projects_persisted_image_acceptance_without_strict_validation(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    calls: list[str] = []

    def strict_acceptance(*_args, **_kwargs):
        calls.append("image_acceptance_status")
        return {
            "required": False,
            "accepted": False,
            "expected_meta_sha256": None,
        }

    def freeze_plan(*_args, **_kwargs):
        calls.append("freeze_plan")
        return type("FrozenPlan", (), {"segments": ()})()

    def frame_fit_call(*_args, **_kwargs):
        calls.append("frame_fit")
        return False

    monkeypatch.setattr(postprocess, "image_acceptance_status", strict_acceptance)
    monkeypatch.setattr(long_generation, "plan_receipt", lambda *_args: "frozen-plan")
    monkeypatch.setattr(long_generation, "freeze_plan", freeze_plan)
    monkeypatch.setattr(
        frame_fit, "frame_bytes_require_fit", frame_fit_call
    )
    monkeypatch.setattr(frame_fit, "frames_require_fit", frame_fit_call)
    monkeypatch.setattr(frame_fit, "fit_frames", frame_fit_call)

    with TestClient(create_app(settings)) as client:
        cid, _meta = _conversation(settings)
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={"status": "done"},
            # The detail view is deliberately a persisted-state projection.
            # Even an opaque historical dict must not trigger artifact reads.
            _image_user_acceptance={},
        )
        expected_meta = storage.load_meta(settings.data_dir, cid)
        response = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["image_acceptance"] == {
        "required": True,
        "accepted": True,
        "expected_meta_sha256": postprocess._image_acceptance_meta_sha256(
            expected_meta
        ),
    }
    assert calls == []


def test_list_projects_persisted_summary_without_detail_validation(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("list reads must not enter detail validation")

    monkeypatch.setattr(postprocess, "image_acceptance_status", forbidden)
    monkeypatch.setattr(long_generation, "plan_receipt", forbidden)
    monkeypatch.setattr(long_generation, "freeze_plan", forbidden)
    monkeypatch.setattr(frame_fit, "frame_bytes_require_fit", forbidden)
    monkeypatch.setattr(frame_fit, "frames_require_fit", forbidden)
    monkeypatch.setattr(frame_fit, "fit_frames", forbidden)

    with TestClient(create_app(settings)) as client:
        cid, _meta = _conversation(settings)
        thumbnail_path = "segments/1/work/anchors/persisted-first.png"
        persisted = storage.update_meta(
            settings.data_dir,
            cid,
            duration_s=37.25,
            segments=[
                {"index": 1, "first_frame_path": thumbnail_path},
                {
                    "index": 2,
                    "first_frame_path": (
                        "segments/2/work/anchors/persisted-first.png"
                    ),
                },
            ],
        )
        response = client.get("/api/conversations", headers=AUTH)

    assert response.status_code == 200
    summaries = {item["id"]: item for item in response.json()}
    assert {
        key: summaries[cid][key]
        for key in (
            "updated_at", "duration_s", "segment_count", "thumbnail_path"
        )
    } == {
        "updated_at": persisted["updated_at"],
        "duration_s": 37.25,
        "segment_count": 2,
        "thumbnail_path": thumbnail_path,
    }


def test_accept_images_forwards_exact_payload_and_returns_same_operation(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, _meta = _conversation(settings)
    payload = {"confirm": True, "expected_meta_sha256": "b" * 64}
    calls = []

    def accept(received_settings, received_cid, received_payload):
        calls.append((received_settings, received_cid, received_payload))
        return {
            "required": True,
            "accepted": True,
            "expected_meta_sha256": "c" * 64,
            "receipt_sha256": "must-not-leak",
        }

    monkeypatch.setattr(postprocess, "accept_images", accept, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/image-acceptance",
            headers=AUTH,
            json=payload,
        )

    assert response.status_code == 202
    assert response.json() == {
        "operation_id": cid,
        "status": "running",
        "stage": "postprocess",
    }
    assert calls == [(settings, cid, payload)]


def test_accept_images_requires_auth_before_calling_core(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cid, _meta = _conversation(settings)
    calls = []
    monkeypatch.setattr(
        postprocess,
        "accept_images",
        lambda *_args: calls.append(True),
        raising=False,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/image-acceptance",
            json={"confirm": True, "expected_meta_sha256": "a" * 64},
        )

    assert response.status_code == 401
    assert calls == []


def test_accept_images_generation_started_is_idempotent_operation_read(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, _meta = _conversation(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={"status": "queued", "client_request_id": "frozen"},
    )

    calls = []

    def reject(*_args):
        calls.append(True)
        raise AssertionError("existing operation must bypass acceptance")

    monkeypatch.setattr(postprocess, "accept_images", reject, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/image-acceptance",
            headers=AUTH,
            json={"confirm": True, "expected_meta_sha256": "a" * 64},
        )

    assert response.status_code == 202
    assert response.json() == {
        "operation_id": cid,
        "status": "running",
        "stage": "generation",
    }
    assert calls == []


@pytest.mark.parametrize(
    ("payload", "status", "detail"),
    [
        ({"confirm": True}, 422, "invalid_image_acceptance_request"),
        ({"confirm": False, "expected_meta_sha256": "a" * 64}, 409,
         "confirmation required"),
        ({"confirm": True, "expected_meta_sha256": "0" * 64}, 409,
         "image_acceptance_changed"),
    ],
)
def test_accept_images_preserves_core_validation_and_cas_errors(
    tmp_path, monkeypatch, payload, status, detail,
):
    settings = make_settings(tmp_path)
    cid, _meta = _conversation(settings)

    def reject(*_args):
        raise postprocess.PostprocessError(status, detail)

    monkeypatch.setattr(postprocess, "accept_images", reject, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/image-acceptance",
            headers=AUTH,
            json=payload,
        )

    if status == 422:
        assert response.status_code == status
        assert response.json() == {"detail": detail}
    else:
        assert response.status_code == 202
        assert response.json() == {
            "operation_id": cid,
            "status": "running",
            "stage": "postprocess",
        }
