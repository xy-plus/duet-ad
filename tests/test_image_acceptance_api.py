"""Manual image acceptance HTTP projection and command seam."""

from fastapi.testclient import TestClient
import pytest

from conftest import AUTH, make_settings

from app import postprocess, storage
from app.main import create_app


def _conversation(settings) -> tuple[str, dict]:
    created = storage.new_conversation(
        settings.data_dir, "manual image acceptance", "source.mp4"
    )
    storage.update_meta(settings.data_dir, created["id"], status="done")
    return created["id"], storage.load_meta(settings.data_dir, created["id"])


def test_detail_projects_only_fixed_image_acceptance_shape(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cid, expected_meta = _conversation(settings)
    calls = []

    def status(received_settings, received_cid, received_meta):
        calls.append((received_settings, received_cid, received_meta))
        return {
            "required": True,
            "accepted": False,
            "expected_meta_sha256": "a" * 64,
            "private_receipt_path": "must-not-leak.json",
        }

    monkeypatch.setattr(
        postprocess, "image_acceptance_status", status, raising=False
    )

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/api/conversations/{cid}", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["image_acceptance"] == {
        "required": True,
        "accepted": False,
        "expected_meta_sha256": "a" * 64,
    }
    assert calls == [(settings, cid, expected_meta)]


def test_accept_images_forwards_exact_payload_and_returns_public_shape(
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

    assert response.status_code == 200
    assert response.json() == {
        "required": True,
        "accepted": True,
        "expected_meta_sha256": "c" * 64,
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


def test_accept_images_preserves_generation_started_rejection(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid, _meta = _conversation(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={"status": "queued", "client_request_id": "frozen"},
    )

    def reject(received_settings, received_cid, _payload):
        assert received_settings is settings
        assert received_cid == cid
        assert isinstance(
            storage.load_meta(settings.data_dir, cid).get("generation"), dict
        )
        raise postprocess.PostprocessError(409, "image_acceptance_frozen")

    monkeypatch.setattr(postprocess, "accept_images", reject, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/image-acceptance",
            headers=AUTH,
            json={"confirm": True, "expected_meta_sha256": "a" * 64},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "image_acceptance_frozen"}


@pytest.mark.parametrize(
    ("payload", "status", "detail"),
    [
        ({"confirm": True}, 422, "invalid_image_acceptance_request"),
        (
            {"confirm": False, "expected_meta_sha256": "a" * 64},
            409,
            "confirmation required",
        ),
        (
            {"confirm": True, "expected_meta_sha256": "0" * 64},
            409,
            "image_acceptance_changed",
        ),
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

    assert response.status_code == status
    assert response.json() == {"detail": detail}
