"""Receipt, recovery, security, and adapter contract for image masks."""

from __future__ import annotations

import json
import stat
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import image_masks


def _png(width: int = 6, height: int = 4, *, alpha: str = "partial") -> bytes:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, :3] = (30, 20, 10)
    if alpha == "partial":
        pixels[1:-1, 1:-1, 3] = 255
    elif alpha == "empty":
        pixels[:, :, 3] = 0
    elif alpha == "full":
        pixels[:, :, 3] = 255
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(alpha)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return encoded.tobytes()


def _source_png(width: int = 6, height: int = 4) -> bytes:
    pixels = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return encoded.tobytes()


class FakeAdapter:
    descriptor = image_masks.ProviderDescriptor(
        provider="fake", action="SegmentPerson", model="fake-v1"
    )
    capabilities = image_masks.ProviderCapabilities(
        mask_scope="all_people_union",
        identity_binding="sole_visible_person",
        supported_purposes=("person",),
    )

    def __init__(
        self,
        receipt: Path,
        result: bytes | None = None,
        *,
        expected_person_id: str = "person-1",
    ):
        self.receipt = receipt
        self.result = result or _png()
        self.expected_person_id = expected_person_id
        self.submit_calls = 0
        self.get_calls = 0
        self.download_calls = 0
        self.submit_results: list[object] = [
            image_masks.ProviderResponse(
                request_id="req-1", result_url="https://private.invalid/result.png"
            )
        ]
        self.get_results: list[object] = []

    def submit(self, request: image_masks.ProviderMaskRequest):
        self.submit_calls += 1
        assert request.action == self.descriptor.action
        assert request.person_instance.person_id == self.expected_person_id
        assert json.loads(self.receipt.read_text())["status"] == "submitting"
        result = self.submit_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def get(self, task_id: str):
        self.get_calls += 1
        assert task_id == "task-1"
        assert json.loads(self.receipt.read_text())["status"] == "accepted"
        result = self.get_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def download(self, url: str) -> bytes:
        self.download_calls += 1
        current = json.loads(self.receipt.read_text())
        assert current["status"] == "response_received"
        assert url == "https://private.invalid/result.png"
        return self.result


class FakeInstanceAdapter(FakeAdapter):
    capabilities = image_masks.ProviderCapabilities(
        mask_scope="person_instance",
        identity_binding="provider_person_id",
        person_id_param="PersonId",
        supported_purposes=("person", "protected_non_target_people"),
    )


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "frames" / "0001.png"
    source.parent.mkdir()
    source.write_bytes(_source_png())
    return root, source


def _person_instance(
    root: Path,
    *,
    person_id: str = "person-1",
    visible_person_ids: tuple[str, ...] = ("person-1",),
    frame_pts: str = "1.25",
    roster_path: str = "work/person-rosters/0001.json",
) -> image_masks.PersonInstanceRequest:
    source = root / "frames/0001.png"
    payload = {
        "schema": "duet.person-roster",
        "version": 1,
        "source": {
            "path": "frames/0001.png",
            "sha256": image_masks.sha256_bytes(source.read_bytes()),
            "width": 6,
            "height": 4,
            "frame_pts": frame_pts,
        },
        "person_ids": sorted(visible_person_ids),
    }
    encoded = json.dumps(payload, sort_keys=True).encode() + b"\n"
    absolute = root / roster_path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(encoded)
    return image_masks.PersonInstanceRequest(
        person_id=person_id,
        visible_person_ids=visible_person_ids,
        person_roster_receipt_path=roster_path,
        person_roster_receipt_sha256=image_masks.sha256_bytes(encoded),
    )


def _generate(root: Path, adapter: FakeAdapter, **overrides):
    person_instance = overrides.pop("person_instance", None) or _person_instance(root)
    values = dict(
        project_root=root,
        source_path="frames/0001.png",
        output_path="work/masks/0001.png",
        receipt_path="work/masks/0001.attempt.json",
        provider=adapter,
        purpose="person",
        person_instance=person_instance,
        frame_pts="1.250000",
        params={"ImageURL": "https://private.invalid/source.png?signature=secret"},
        cache_version="mask-cache-v1",
    )
    values.update(overrides)
    return image_masks.generate_mask(**values)


def _direct_provider_request(adapter, *, action="SegmentHDBody"):
    capability = adapter.capabilities
    instance = image_masks.PersonInstanceRequest(
        person_id="person-1",
        visible_person_ids=("person-1",),
        person_roster_receipt_path="work/person-rosters/0001.json",
        person_roster_receipt_sha256="c" * 64,
    )
    params = {"ImageURL": "https://private.invalid/source.png"}
    request_hash = image_masks.request_sha256(
        provider="aliyun_viapi",
        action=action,
        model="imageseg-20191230",
        purpose="person",
        provider_capability=capability,
        person_instance=instance,
        source_sha256="b" * 64,
        width=6,
        height=4,
        frame_pts="1.25",
        params=params,
        cache_version="mask-cache-v1",
    )
    return image_masks.ProviderMaskRequest(
        provider="aliyun_viapi",
        action=action,
        model="imageseg-20191230",
        request_sha256=request_hash,
        purpose="person",
        provider_capability=capability,
        person_instance=instance,
        source_sha256="b" * 64,
        width=6,
        height=4,
        frame_pts="1.25",
        params=params,
        cache_version="mask-cache-v1",
    )


def test_success_freezes_request_and_publishes_consumer_dto(tmp_path):
    root, source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    person_instance = _person_instance(root)

    result = _generate(root, adapter, person_instance=person_instance)

    assert result.path == root / "work/masks/0001.png"
    assert result.path.read_bytes() == adapter.result
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "succeeded"
    assert payload["version"] == 2
    assert payload["history"] == [
        "prepared", "submitting", "response_received", "downloaded",
        "validated", "succeeded",
    ]
    assert payload["request"] == {
        "schema": "duet.image-mask-request",
        "version": 2,
        "provider": "fake",
        "action": "SegmentPerson",
        "model": "fake-v1",
        "purpose": "person",
        "provider_capability": {
            "mask_scope": "all_people_union",
            "identity_binding": "sole_visible_person",
            "person_id_param": None,
            "supported_purposes": ["person"],
        },
        "person_instance": {
            "person_id": "person-1",
            "visible_person_ids": ["person-1"],
            "person_roster_receipt_path": "work/person-rosters/0001.json",
            "person_roster_receipt_sha256": person_instance.person_roster_receipt_sha256,
            "provider_person_id": None,
        },
        "source": {
            "path": "frames/0001.png",
            "sha256": image_masks.sha256_bytes(source.read_bytes()),
            "width": 6,
            "height": 4,
        },
        "frame_pts": "1.25",
        "params": {
            "ImageURL": "https://private.invalid/source.png?signature=secret"
        },
        "cache_version": "mask-cache-v1",
        "request_sha256": payload["request"]["request_sha256"],
    }
    assert payload["request"]["request_sha256"] == image_masks.request_sha256(
        provider="fake",
        action="SegmentPerson",
        model="fake-v1",
        purpose="person",
        provider_capability=adapter.capabilities,
        person_instance=person_instance,
        source_sha256=image_masks.sha256_bytes(source.read_bytes()),
        width=6,
        height=4,
        frame_pts="1.25",
        params={"ImageURL": "https://private.invalid/source.png?signature=secret"},
        cache_version="mask-cache-v1",
    )
    producer = result.producer_receipt
    assert producer == payload["producer_receipt"]
    assert producer["schema"] == "duet.image-mask-producer"
    assert producer["version"] == 2
    assert producer["producer"] == {
        "provider": "fake", "action": "SegmentPerson", "model": "fake-v1"
    }
    assert producer["purpose"] == "person"
    assert producer["provider_capability"] == payload["request"]["provider_capability"]
    assert producer["person_instance"] == payload["request"]["person_instance"]
    assert producer["source"]["frame_pts"] == "1.25"
    assert producer["mask"]["path"] == "work/masks/0001.png"
    assert producer["mask"]["width"] == 6
    assert producer["mask"]["height"] == 4
    assert producer["mask"]["alpha_nonzero_pixels"] == 8
    assert producer["mask"]["alpha_transparent_pixels"] == 16
    landing = root / payload["landing_path"]
    assert landing.is_file()
    assert stat.S_IMODE(landing.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.path.stat().st_mode) == 0o600
    assert adapter.submit_calls == adapter.download_calls == 1
    assert adapter.get_calls == 0


@pytest.mark.parametrize("purpose", ["scene", "background", "", None])
def test_rejects_scene_and_unknown_purposes_without_provider_call(tmp_path, purpose):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, purpose=purpose)

    assert raised.value.code == "invalid_mask_purpose"
    assert adapter.submit_calls == adapter.get_calls == adapter.download_calls == 0


def test_union_provider_rejects_protected_non_target_people_before_post(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, purpose="protected_non_target_people")

    assert raised.value.code == "person_instance_unavailable"
    assert adapter.submit_calls == 0
    assert not adapter.receipt.exists()


def test_union_provider_rejects_multi_person_frame_before_post(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    instance = _person_instance(
        root,
        person_id="person-1",
        visible_person_ids=("person-1", "person-2"),
    )

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, person_instance=instance)

    assert raised.value.code == "person_instance_unavailable"
    assert adapter.submit_calls == 0
    assert not adapter.receipt.exists()


@pytest.mark.parametrize("damage", ["bytes", "source", "people"])
def test_person_roster_must_be_authoritative_before_post(tmp_path, damage):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    instance = _person_instance(root)
    roster_path = root / instance.person_roster_receipt_path
    payload = json.loads(roster_path.read_text())
    if damage == "bytes":
        roster_path.write_bytes(roster_path.read_bytes() + b" ")
    elif damage == "source":
        payload["source"]["sha256"] = "f" * 64
        roster_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        instance = image_masks.PersonInstanceRequest(
            person_id=instance.person_id,
            visible_person_ids=instance.visible_person_ids,
            person_roster_receipt_path=instance.person_roster_receipt_path,
            person_roster_receipt_sha256=image_masks.sha256_bytes(
                roster_path.read_bytes()
            ),
        )
    else:
        payload["person_ids"] = ["person-1", "person-2"]
        roster_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        instance = image_masks.PersonInstanceRequest(
            person_id=instance.person_id,
            visible_person_ids=instance.visible_person_ids,
            person_roster_receipt_path=instance.person_roster_receipt_path,
            person_roster_receipt_sha256=image_masks.sha256_bytes(
                roster_path.read_bytes()
            ),
        )

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, person_instance=instance)

    assert raised.value.code == "person_roster_receipt_mismatch"
    assert adapter.submit_calls == 0
    assert not adapter.receipt.exists()


def test_instance_provider_binds_person_id_and_allows_protected_person(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeInstanceAdapter(root / "work/masks/0001.attempt.json")

    result = _generate(
        root,
        adapter,
        purpose="protected_non_target_people",
        person_instance=_person_instance(
            root,
            person_id="person-1",
            visible_person_ids=("person-1", "target-person"),
        ),
        params={
            "ImageURL": "https://private.invalid/source.png?signature=secret",
            "PersonId": "person-1",
        },
    )

    instance = result.producer_receipt["person_instance"]
    assert instance["person_id"] == instance["provider_person_id"] == "person-1"
    assert instance["visible_person_ids"] == ["person-1", "target-person"]


def test_instance_provider_produces_distinct_receipts_for_each_person(tmp_path):
    root, _source = _paths(tmp_path)
    first = np.zeros((4, 6, 4), dtype=np.uint8)
    first[:, 1, 3] = 255
    second = np.zeros((4, 6, 4), dtype=np.uint8)
    second[:, 4, 3] = 255
    ok_first, first_png = cv2.imencode(".png", first)
    ok_second, second_png = cv2.imencode(".png", second)
    assert ok_first and ok_second
    first_instance = _person_instance(
        root, person_id="person-1", visible_person_ids=("person-1", "person-2")
    )
    second_instance = image_masks.PersonInstanceRequest(
        person_id="person-2",
        visible_person_ids=("person-1", "person-2"),
        person_roster_receipt_path=first_instance.person_roster_receipt_path,
        person_roster_receipt_sha256=first_instance.person_roster_receipt_sha256,
    )
    first_adapter = FakeInstanceAdapter(
        root / "work/masks/person-1.attempt.json", result=first_png.tobytes()
    )
    second_adapter = FakeInstanceAdapter(
        root / "work/masks/person-2.attempt.json",
        result=second_png.tobytes(),
        expected_person_id="person-2",
    )

    first_result = _generate(
        root,
        first_adapter,
        output_path="work/masks/person-1.png",
        receipt_path="work/masks/person-1.attempt.json",
        person_instance=first_instance,
        params={"ImageURL": "https://private.invalid/source.png", "PersonId": "person-1"},
    )
    second_result = _generate(
        root,
        second_adapter,
        output_path="work/masks/person-2.png",
        receipt_path="work/masks/person-2.attempt.json",
        person_instance=second_instance,
        params={"ImageURL": "https://private.invalid/source.png", "PersonId": "person-2"},
    )

    first_receipt = first_result.producer_receipt
    second_receipt = second_result.producer_receipt
    assert first_receipt["person_instance"]["provider_person_id"] == "person-1"
    assert second_receipt["person_instance"]["provider_person_id"] == "person-2"
    assert first_receipt["request_sha256"] != second_receipt["request_sha256"]
    assert first_receipt["mask"]["sha256"] != second_receipt["mask"]["sha256"]


@pytest.mark.parametrize("provider_person_id", [None, "person-2"])
def test_instance_provider_rejects_missing_or_wrong_identity_binding_before_post(
    tmp_path, provider_person_id
):
    root, _source = _paths(tmp_path)
    adapter = FakeInstanceAdapter(root / "work/masks/0001.attempt.json")
    params = {"ImageURL": "https://private.invalid/source.png"}
    if provider_person_id is not None:
        params["PersonId"] = provider_person_id

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, params=params)

    assert raised.value.code == "provider_identity_binding_invalid"
    assert adapter.submit_calls == 0
    assert not adapter.receipt.exists()


def test_post_timeout_without_identifier_is_terminal_and_never_resubmits(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    adapter.submit_results = [TimeoutError("secret URL https://private.invalid/leak")]

    with pytest.raises(image_masks.MaskError) as first:
        _generate(root, adapter)
    with pytest.raises(image_masks.MaskError) as second:
        _generate(root, adapter)

    assert first.value.code == second.value.code == "submission_unknown"
    assert str(first.value) == "submission_unknown"
    assert "private.invalid" not in str(first.value)
    assert json.loads(receipt.read_text())["status"] == "submission_unknown"
    assert adapter.submit_calls == 1
    assert adapter.get_calls == adapter.download_calls == 0


def test_accepted_task_recovers_with_get_only(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    adapter.submit_results = [
        image_masks.ProviderResponse(request_id="req-1", task_id="task-1")
    ]
    adapter.get_results = [
        image_masks.ProviderResponse(request_id="req-1", task_id="task-1"),
        image_masks.ProviderResponse(
            request_id="req-1",
            task_id="task-1",
            result_url="https://private.invalid/result.png",
        ),
    ]

    with pytest.raises(image_masks.MaskError) as pending:
        _generate(root, adapter)
    result = _generate(root, adapter)

    assert pending.value.code == "provider_pending"
    assert result.path.is_file()
    assert adapter.submit_calls == 1
    assert adapter.get_calls == 2
    assert adapter.download_calls == 1
    assert json.loads(receipt.read_text())["history"].count("accepted") == 1


def test_accepted_get_timeout_remains_get_only_and_retryable(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    adapter.submit_results = [
        image_masks.ProviderResponse(request_id="req-1", task_id="task-1")
    ]
    adapter.get_results = [
        TimeoutError("GET https://private.invalid/task/task-1 failed"),
        image_masks.ProviderResponse(
            request_id="req-1",
            task_id="task-1",
            result_url="https://private.invalid/result.png",
        ),
    ]

    with pytest.raises(image_masks.MaskError) as first:
        _generate(root, adapter)
    result = _generate(root, adapter)

    assert first.value.code == "provider_query_failed"
    assert first.value.retryable is True
    assert "private.invalid" not in str(first.value)
    assert result.path.is_file()
    assert adapter.submit_calls == 1
    assert adapter.get_calls == 2


def test_uncertain_submission_with_task_id_switches_to_get_not_post(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    adapter.submit_results = [
        image_masks.SubmissionUncertain(request_id="req-1", task_id="task-1")
    ]
    adapter.get_results = [
        image_masks.ProviderResponse(
            request_id="req-1", task_id="task-1",
            result_url="https://private.invalid/result.png",
        )
    ]

    result = _generate(root, adapter)

    assert result.path.is_file()
    assert adapter.submit_calls == adapter.get_calls == adapter.download_calls == 1


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (b"not-a-png", "mask_not_png"),
        (_png() + b"trailing-junk", "mask_not_png"),
        (_png(7, 4), "mask_dimensions_mismatch"),
        (_png(alpha="empty"), "mask_alpha_empty"),
        (_png(alpha="full"), "mask_alpha_full_frame"),
    ],
)
def test_invalid_provider_output_is_never_published(tmp_path, result, code):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt, result=result)

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter)

    assert raised.value.code == code
    assert not (root / "work/masks/0001.png").exists()
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "failed"
    assert payload["error_code"] == code
    assert "result_url" not in payload.get("public_error", {})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_path", "../outside.png"),
        ("source_path", "/tmp/outside.png"),
        ("output_path", "../outside.png"),
        ("receipt_path", "/tmp/receipt.json"),
    ],
)
def test_project_paths_must_be_relative_and_cannot_traverse(tmp_path, field, value):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, **{field: value})

    assert raised.value.code == "unsafe_project_path"
    assert adapter.submit_calls == 0


def test_rejects_source_parent_and_destination_symlinks(tmp_path):
    root, source = _paths(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.png").write_bytes(source.read_bytes())
    (root / "linked").symlink_to(outside, target_is_directory=True)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")

    with pytest.raises(image_masks.MaskError) as parent_error:
        _generate(root, adapter, source_path="linked/source.png")
    assert parent_error.value.code == "unsafe_project_path"

    output = root / "work/masks/0001.png"
    output.parent.mkdir(parents=True)
    output.symlink_to(outside / "stolen.png")
    with pytest.raises(image_masks.MaskError) as output_error:
        _generate(root, adapter)
    assert output_error.value.code == "unsafe_project_path"
    assert not (outside / "stolen.png").exists()
    assert adapter.submit_calls == 0


@pytest.mark.parametrize("leaf", ["0001.attempt.json", ".0001.attempt.json.provider-result"])
def test_rejects_receipt_and_private_landing_symlinks(tmp_path, leaf):
    root, _source = _paths(tmp_path)
    outside = tmp_path / "outside-receipt"
    outside.write_bytes(b"do-not-overwrite")
    directory = root / "work/masks"
    directory.mkdir(parents=True)
    (directory / leaf).symlink_to(outside)
    adapter = FakeAdapter(directory / "0001.attempt.json")

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter)

    assert raised.value.code == "unsafe_project_path"
    assert outside.read_bytes() == b"do-not-overwrite"
    assert adapter.submit_calls == 0


def test_response_received_receipt_recovers_download_without_post(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    original_download = adapter.download

    def interrupted_download(url: str) -> bytes:
        adapter.download_calls += 1
        assert json.loads(receipt.read_text())["status"] == "response_received"
        raise TimeoutError(f"download failed: {url}")

    adapter.download = interrupted_download
    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter)
    assert raised.value.code == "mask_download_failed"

    adapter.download = original_download
    result = _generate(root, adapter)
    assert result.path.is_file()
    assert adapter.submit_calls == 1
    assert adapter.download_calls == 2


def test_succeeded_receipt_reuses_verified_local_output_without_network(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    first = _generate(root, adapter)

    second = _generate(root, adapter)

    assert second == first
    assert adapter.submit_calls == adapter.download_calls == 1
    assert adapter.get_calls == 0


def test_validated_landing_recovers_local_publish_without_network(tmp_path, monkeypatch):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    real_write = image_masks._atomic_project_bytes
    failed_once = False

    def fail_first_publish(project_root, relative, payload):
        nonlocal failed_once
        if relative == "work/masks/0001.png" and not failed_once:
            failed_once = True
            raise image_masks.MaskError("mask_output_write_failed")
        real_write(project_root, relative, payload)

    monkeypatch.setattr(image_masks, "_atomic_project_bytes", fail_first_publish)
    with pytest.raises(image_masks.MaskError) as first:
        _generate(root, adapter)
    assert first.value.code == "mask_output_write_failed"
    assert json.loads(receipt.read_text())["status"] == "validated"

    monkeypatch.setattr(image_masks, "_atomic_project_bytes", real_write)
    result = _generate(root, adapter)

    assert result.path.is_file()
    assert adapter.submit_calls == adapter.download_calls == 1
    assert adapter.get_calls == 0


def test_request_drift_fails_closed_without_network(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    _generate(root, adapter)

    changed_instance = _person_instance(root, frame_pts="1.251")
    with pytest.raises(image_masks.MaskError) as raised:
        _generate(
            root,
            adapter,
            frame_pts="1.251",
            person_instance=changed_instance,
        )

    assert raised.value.code == "mask_receipt_mismatch"
    assert adapter.submit_calls == adapter.download_calls == 1


def test_person_roster_receipt_drift_fails_closed_without_network(tmp_path):
    root, _source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)
    _generate(root, adapter)
    roster_path = root / "work/person-rosters/0001.json"
    roster_path.write_bytes(roster_path.read_bytes() + b" ")
    changed_instance = image_masks.PersonInstanceRequest(
        person_id="person-1",
        visible_person_ids=("person-1",),
        person_roster_receipt_path="work/person-rosters/0001.json",
        person_roster_receipt_sha256=image_masks.sha256_bytes(roster_path.read_bytes()),
    )

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, person_instance=changed_instance)

    assert raised.value.code == "mask_receipt_mismatch"
    assert adapter.submit_calls == adapter.download_calls == 1


def test_aliyun_segment_hd_body_adapter_maps_injected_request_and_dto():
    calls = []

    def request(action, params):
        calls.append((action, params))
        return {
            "RequestId": "ali-req-1",
            "Data": {
                "ImageURL": (
                    "https://result.oss-cn-shanghai.aliyuncs.com/"
                    "ali-mask.png?token=secret"
                )
            },
        }

    adapter = image_masks.AliyunVIAPISegmentHDBody(request=request, download=lambda _url: b"png")
    provider_request = _direct_provider_request(adapter)

    response = adapter.submit(provider_request)

    assert adapter.descriptor == image_masks.ProviderDescriptor(
        provider="aliyun_viapi",
        action="SegmentHDBody",
        model="imageseg-20191230",
    )
    assert calls == [
        ("SegmentHDBody", {"ImageURL": "https://private.invalid/source.png"})
    ]
    assert response == image_masks.ProviderResponse(
        request_id="ali-req-1",
        result_url="https://result.oss-cn-shanghai.aliyuncs.com/ali-mask.png?token=secret",
    )


def test_provider_request_rejects_hash_that_does_not_bind_instance():
    adapter = image_masks.AliyunVIAPISegmentHDBody(
        request=lambda _action, _params: {}, download=lambda _url: b"png"
    )
    good = _direct_provider_request(adapter)

    with pytest.raises(image_masks.MaskError) as raised:
        image_masks.ProviderMaskRequest(
            **{**good.__dict__, "request_sha256": "f" * 64}
        )

    assert raised.value.code == "provider_request_invalid"


def test_aliyun_protocol_errors_are_safe_and_do_not_echo_response():
    secret = "https://result.oss-cn-shanghai.aliyuncs.com/result.png?token=do-not-leak"
    adapter = image_masks.AliyunVIAPISegmentHDBody(
        request=lambda _action, _params: {"RequestId": "req-1", "Data": {"ImageURL": secret}},
        download=lambda _url: b"png",
    )
    request = _direct_provider_request(adapter, action="WrongAction")

    with pytest.raises(image_masks.MaskError) as raised:
        adapter.submit(request)

    assert raised.value.code == "provider_request_invalid"
    assert str(raised.value) == "provider_request_invalid"
    assert secret not in str(raised.value)


def test_aliyun_result_url_is_provider_scoped_and_safe():
    secret = "https://127.0.0.1/internal?token=do-not-leak"
    adapter = image_masks.AliyunVIAPISegmentHDBody(
        request=lambda _action, _params: {
            "RequestId": "req-1", "Data": {"ImageURL": secret}
        },
        download=lambda _url: b"png",
    )
    request = _direct_provider_request(adapter)

    with pytest.raises(image_masks.MaskError) as raised:
        adapter.submit(request)

    assert raised.value.code == "provider_protocol_error"
    assert secret not in str(raised.value)


def _consumer_expectations(root: Path, producer: dict):
    source = producer["source"]
    instance = producer["person_instance"]
    return {
        "expected_source": image_masks.MaskSourceExpectation(
            path=source["path"],
            sha256=source["sha256"],
            width=source["width"],
            height=source["height"],
            frame_pts=source["frame_pts"],
        ),
        "expected_person_id": instance["person_id"],
        "expected_visible_person_ids": tuple(instance["visible_person_ids"]),
        "expected_roster": image_masks.MaskRosterExpectation(
            path=instance["person_roster_receipt_path"],
            sha256=instance["person_roster_receipt_sha256"],
        ),
        "expected_purpose": producer["purpose"],
    }


def test_consumer_loader_returns_immutable_canonical_receipt_and_packed_mask(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    generated = _generate(root, adapter)

    loaded = image_masks.load_validated_mask(
        root,
        generated.producer_receipt,
        **_consumer_expectations(root, generated.producer_receipt),
    )

    assert json.loads(loaded.canonical_receipt) == generated.producer_receipt
    assert loaded.canonical_receipt_sha256 == image_masks.sha256_bytes(
        loaded.canonical_receipt
    )
    assert loaded.project_relative_path == "work/masks/0001.png"
    assert loaded.mask_sha256 == generated.producer_receipt["mask"]["sha256"]
    assert loaded.width == 6
    assert loaded.height == 4
    assert loaded.foreground_pixels == 8
    assert loaded.packed_encoding == "row-major-alpha-gt-zero-packbits-little-v1"
    unpacked = np.unpackbits(
        np.frombuffer(loaded.packed_mask, dtype=np.uint8),
        count=24,
        bitorder="little",
    ).reshape(4, 6)
    assert unpacked.tolist() == [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    with pytest.raises(FrozenInstanceError):
        loaded.width = 7


def test_consumer_loader_accepts_project_relative_succeeded_attempt_path(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    generated = _generate(root, adapter)

    loaded = image_masks.load_validated_mask(
        root,
        str(generated.receipt_path.relative_to(root)),
        **_consumer_expectations(root, generated.producer_receipt),
    )

    assert json.loads(loaded.canonical_receipt) == generated.producer_receipt
    assert loaded.mask_sha256 == generated.producer_receipt["mask"]["sha256"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifact: artifact.update(version=1),
        lambda artifact: artifact["person_instance"].update(person_id="person-2"),
        lambda artifact: artifact["person_instance"].update(
            visible_person_ids=["person-1", "person-2"]
        ),
        lambda artifact: artifact["provider_capability"].update(
            mask_scope="person_instance"
        ),
        lambda artifact: artifact.update(request_sha256="f" * 64),
        lambda artifact: artifact["source"].update(frame_pts="1.251"),
        lambda artifact: artifact["mask"].update(alpha_nonzero_pixels=9),
        lambda artifact: artifact.update(unexpected="private-response"),
    ],
)
def test_consumer_loader_rejects_noncanonical_or_mismatched_artifact(
    tmp_path, mutate
):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    generated = _generate(root, adapter)
    artifact = deepcopy(generated.producer_receipt)
    mutate(artifact)

    with pytest.raises(image_masks.MaskError) as raised:
        image_masks.load_validated_mask(
            root,
            artifact,
            **_consumer_expectations(root, generated.producer_receipt),
        )

    assert raised.value.code == "mask_artifact_mismatch"
    assert "private-response" not in str(raised.value)


@pytest.mark.parametrize("target", ["source", "mask"])
def test_consumer_loader_rejects_symlinked_bound_files(tmp_path, target):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    generated = _generate(root, adapter)
    producer = generated.producer_receipt
    relative = producer["source"]["path"] if target == "source" else producer["mask"]["path"]
    path = root / relative
    backup = root / f"outside-{target}.bin"
    backup.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(backup)

    with pytest.raises(image_masks.MaskError) as raised:
        image_masks.load_validated_mask(
            root, producer, **_consumer_expectations(root, producer)
        )

    assert raised.value.code == "unsafe_project_path"


def test_consumer_loader_rejects_roster_bytes_changed_after_generation(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    generated = _generate(root, adapter)
    producer = generated.producer_receipt
    roster = root / producer["person_instance"]["person_roster_receipt_path"]
    roster.write_bytes(roster.read_bytes() + b" ")

    with pytest.raises(image_masks.MaskError) as raised:
        image_masks.load_validated_mask(
            root, producer, **_consumer_expectations(root, producer)
        )

    assert raised.value.code == "person_roster_receipt_mismatch"


@pytest.mark.parametrize(
    ("replacement", "code"),
    [(b"not-a-png", "mask_not_png"), (_png(alpha="full"), "mask_alpha_full_frame")],
)
def test_consumer_loader_revalidates_persisted_png_bytes(tmp_path, replacement, code):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")
    generated = _generate(root, adapter)
    producer = generated.producer_receipt
    (root / producer["mask"]["path"]).write_bytes(replacement)

    with pytest.raises(image_masks.MaskError) as raised:
        image_masks.load_validated_mask(
            root, producer, **_consumer_expectations(root, producer)
        )

    assert raised.value.code == code
