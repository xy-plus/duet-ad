"""Receipt, recovery, security, and adapter contract for image masks."""

from __future__ import annotations

import json
import stat
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

    def __init__(self, receipt: Path, result: bytes | None = None):
        self.receipt = receipt
        self.result = result or _png()
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


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "frames" / "0001.png"
    source.parent.mkdir()
    source.write_bytes(_source_png())
    return root, source


def _generate(root: Path, adapter: FakeAdapter, **overrides):
    values = dict(
        project_root=root,
        source_path="frames/0001.png",
        output_path="work/masks/0001.png",
        receipt_path="work/masks/0001.attempt.json",
        provider=adapter,
        purpose="person",
        frame_pts="1.250000",
        params={"ImageURL": "https://private.invalid/source.png?signature=secret"},
        cache_version="mask-cache-v1",
    )
    values.update(overrides)
    return image_masks.generate_mask(**values)


def test_success_freezes_request_and_publishes_consumer_dto(tmp_path):
    root, source = _paths(tmp_path)
    receipt = root / "work/masks/0001.attempt.json"
    adapter = FakeAdapter(receipt)

    result = _generate(root, adapter)

    assert result.path == root / "work/masks/0001.png"
    assert result.path.read_bytes() == adapter.result
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "succeeded"
    assert payload["history"] == [
        "prepared", "submitting", "response_received", "downloaded",
        "validated", "succeeded",
    ]
    assert payload["request"] == {
        "provider": "fake",
        "action": "SegmentPerson",
        "model": "fake-v1",
        "purpose": "person",
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
    assert producer["producer"] == {
        "provider": "fake", "action": "SegmentPerson", "model": "fake-v1"
    }
    assert producer["purpose"] == "person"
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


def test_allows_protected_non_target_people_purpose(tmp_path):
    root, _source = _paths(tmp_path)
    adapter = FakeAdapter(root / "work/masks/0001.attempt.json")

    result = _generate(root, adapter, purpose="protected_non_target_people")

    assert result.producer_receipt["purpose"] == "protected_non_target_people"


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

    with pytest.raises(image_masks.MaskError) as raised:
        _generate(root, adapter, frame_pts="1.251")

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
    provider_request = image_masks.ProviderMaskRequest(
        provider="aliyun_viapi",
        action="SegmentHDBody",
        model="imageseg-20191230",
        request_sha256="a" * 64,
        purpose="person",
        source_sha256="b" * 64,
        width=6,
        height=4,
        frame_pts="1.25",
        params={"ImageURL": "https://private.invalid/source.png"},
        cache_version="mask-cache-v1",
    )

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


def test_aliyun_protocol_errors_are_safe_and_do_not_echo_response():
    secret = "https://result.oss-cn-shanghai.aliyuncs.com/result.png?token=do-not-leak"
    adapter = image_masks.AliyunVIAPISegmentHDBody(
        request=lambda _action, _params: {"RequestId": "req-1", "Data": {"ImageURL": secret}},
        download=lambda _url: b"png",
    )
    request = image_masks.ProviderMaskRequest(
        provider="aliyun_viapi",
        action="WrongAction",
        model="imageseg-20191230",
        request_sha256="a" * 64,
        purpose="person",
        source_sha256="b" * 64,
        width=6,
        height=4,
        frame_pts="1.25",
        params={"ImageURL": "https://private.invalid/source.png"},
        cache_version="mask-cache-v1",
    )

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
    request = image_masks.ProviderMaskRequest(
        provider="aliyun_viapi",
        action="SegmentHDBody",
        model="imageseg-20191230",
        purpose="person",
        source_sha256="b" * 64,
        width=6,
        height=4,
        frame_pts="1.25",
        request_sha256="a" * 64,
        params={"ImageURL": "https://private.invalid/source.png"},
        cache_version="mask-cache-v1",
    )

    with pytest.raises(image_masks.MaskError) as raised:
        adapter.submit(request)

    assert raised.value.code == "provider_protocol_error"
    assert secret not in str(raised.value)
