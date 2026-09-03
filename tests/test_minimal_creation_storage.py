import asyncio
import hashlib
import json
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import storage


class ChunkedUpload:
    def __init__(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str | None = "application/octet-stream",
        chunk_size: int = 7,
    ):
        self.data = data
        self.filename = filename
        self.content_type = content_type
        self.chunk_size = chunk_size
        self.offset = 0

    async def read(self, _size: int) -> bytes:
        chunk = self.data[self.offset:self.offset + self.chunk_size]
        self.offset += len(chunk)
        return chunk


def _effective_request(replacement_guidance=None):
    return {
        "version": 1,
        "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
        },
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "rewrite",
            "script": "冻结的新台词",
            "language": {"mode": "source"},
        },
        "replacement_guidance": replacement_guidance,
    }


def _meta(cid):
    return {
        "schema_version": 2,
        "id": cid,
        "title": "v1",
        "note": "",
        "status": "queued",
        "error": None,
        "created_at": "2026-09-03T00:00:00+00:00",
        "updated_at": "2026-09-03T00:00:00+00:00",
        "keyframes": [],
        "prompt": None,
        "voice_mode": "rewrite",
        "duration_s": 1.0,
        "fit_required": None,
        "dialogue_mode": "rewrite",
        "generation": None,
    }


def _encoded_image(ext: str) -> bytes:
    image = np.full((8, 9, 3), (10, 80, 170), dtype=np.uint8)
    ok, encoded = cv2.imencode(ext, image)
    assert ok
    return encoded.tobytes()


def test_staged_source_is_hidden_then_published_with_frozen_receipt(
    tmp_path, monkeypatch,
):
    source_bytes = b"source-video-bytes"
    renames = []
    original_rename = storage.os.rename

    def record_rename(source, destination):
        renames.append((Path(source), Path(destination)))
        original_rename(source, destination)

    monkeypatch.setattr(storage.os, "rename", record_rename)
    with storage.staged_creation(tmp_path) as (cid, staging):
        (staging / "meta.json").write_text(
            json.dumps(_meta(cid)), encoding="utf-8"
        )
        assert staging.name.startswith(".")
        assert stat.S_IMODE(staging.stat().st_mode) == 0o700
        assert storage.list_conversations(tmp_path) == []
        (staging / "meta.json").unlink()

        source = asyncio.run(storage.save_creation_source(
            staging,
            ChunkedUpload(source_bytes, filename="clip.mp4", chunk_size=3),
            len(source_bytes),
        ))
        assert stat.S_IMODE(source.path.stat().st_mode) == 0o600
        effective_request = _effective_request()
        published = storage.publish_staged_creation(
            tmp_path,
            staging,
            cid,
            meta=_meta(cid),
            effective_request=effective_request,
            client_request_id="minimal-create-000001",
            source=source,
            source_filename="folder\\clip.mp4",
            source_reference_url=None,
        )
        assert not staging.exists()

    final = tmp_path / cid
    assert final.is_dir()
    assert renames == [(staging, final)]
    saved = json.loads((final / "meta.json").read_text(encoding="utf-8"))
    canonical = json.dumps(
        effective_request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert saved == published
    assert saved["effective_request"] == effective_request
    assert saved["input_receipt"] == {
        "version": 1,
        "client_request_id": "minimal-create-000001",
        "generation_request_sha256": hashlib.sha256(canonical).hexdigest(),
        "source": {
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "bytes": len(source_bytes),
        },
        "replacement_image": None,
    }
    assert saved["creation_input"] == {
        "version": 1,
        "source": {
            "mode": "upload",
            "filename": "clip.mp4",
            "bytes": len(source_bytes),
        },
        "replacement_image": None,
    }
    assert "_minimal_replacement_image_path" not in saved
    assert storage.list_conversations(tmp_path)[0]["id"] == cid
    updated = storage.update_meta(tmp_path, cid, status="processing")
    assert updated is not None
    assert updated["creation_input"] == saved["creation_input"]
    assert updated["error"] is None
    before = (final / "meta.json").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        storage.update_meta(tmp_path, cid, effective_request={"version": 1})
    assert (final / "meta.json").read_bytes() == before

    def remove_receipt(meta):
        meta.pop("input_receipt")

    with pytest.raises(ValueError, match="immutable"):
        storage.mutate_meta(tmp_path, cid, remove_receipt)
    assert (final / "meta.json").read_bytes() == before

    with pytest.raises(ValueError, match="immutable"):
        storage.update_meta(tmp_path, cid, creation_input=None)
    assert (final / "meta.json").read_bytes() == before


@pytest.mark.parametrize(
    ("media_type", "encoder_ext", "stored_ext", "declared_type"),
    [
        ("image/jpeg", ".jpg", ".jpg", "image/jpeg"),
        ("image/jpeg", ".jpg", ".jpg", "image/png"),
        ("image/png", ".png", ".png", None),
        ("image/webp", ".webp", ".webp", "image/gif"),
    ],
)
def test_replacement_image_uses_decoded_type_and_private_receipt(
    tmp_path, media_type, encoder_ext, stored_ext, declared_type,
):
    image_bytes = _encoded_image(encoder_ext)
    source_bytes = b"video"
    guidance = {
        "instruction": "把杯子替换成参考产品",
        "image_field": "replacement_image",
    }
    with storage.staged_creation(tmp_path) as (cid, staging):
        source = asyncio.run(storage.save_creation_source(
            staging,
            ChunkedUpload(source_bytes, filename="source.webm"),
            len(source_bytes),
        ))
        replacement = asyncio.run(storage.save_creation_replacement_image(
            staging,
            ChunkedUpload(
                image_bytes,
                filename="untrusted.bin",
                content_type=declared_type,
                chunk_size=5,
            ),
            len(image_bytes),
        ))
        assert replacement.path == (
            staging / "inputs" / f"replacement_image{stored_ext}"
        )
        assert stat.S_IMODE(replacement.path.stat().st_mode) == 0o600
        assert replacement.media_type == media_type
        assert cv2.imread(str(replacement.path), cv2.IMREAD_UNCHANGED) is not None
        published = storage.publish_staged_creation(
            tmp_path,
            staging,
            cid,
            meta=_meta(cid),
            effective_request=_effective_request(guidance),
            client_request_id="minimal-create-000002",
            source=source,
            source_filename="source.webm",
            source_reference_url=None,
            replacement_image=replacement,
            replacement_image_filename="catalog/product-original.png",
        )

    assert published["_minimal_replacement_image_path"] == (
        f"inputs/replacement_image{stored_ext}"
    )
    assert published["effective_request"]["replacement_guidance"] == guidance
    assert published["input_receipt"]["replacement_image"] == {
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "bytes": len(image_bytes),
    }
    assert set(published["input_receipt"]["replacement_image"]) == {
        "sha256", "bytes",
    }
    assert published["creation_input"] == {
        "version": 1,
        "source": {
            "mode": "upload",
            "filename": "source.webm",
            "bytes": len(source_bytes),
        },
        "replacement_image": {
            "filename": "product-original.png",
            "bytes": len(image_bytes),
            "media_type": media_type,
            "preview_url": (
                f"/api/conversations/{cid}/creation-input/replacement-image"
            ),
        },
    }


@pytest.mark.parametrize(
    ("data", "declared_type", "code", "status_code"),
    [
        (b"\x89PNG\r\n\x1a\nnot-decodable", "image/png", "invalid_replacement_image", 422),
        (b"GIF89a-not-supported", "image/png", "unsupported_replacement_media_type", 415),
    ],
)
def test_replacement_image_rejects_false_or_undecodable_media(
    tmp_path, data, declared_type, code, status_code,
):
    with pytest.raises(storage.CreationStorageError) as caught:
        with storage.staged_creation(tmp_path) as (_cid, staging):
            failed_staging = staging
            asyncio.run(storage.save_creation_replacement_image(
                staging,
                ChunkedUpload(
                    data,
                    filename="claimed.png",
                    content_type=declared_type,
                ),
                len(data),
            ))
    assert caught.value.code == code
    assert caught.value.status_code == status_code
    assert not failed_staging.exists()
    assert storage.list_conversations(tmp_path) == []


def test_oversize_and_final_conflict_leave_no_staging_or_overwrite(tmp_path):
    with pytest.raises(storage.CreationStorageError) as caught:
        with storage.staged_creation(tmp_path) as (_cid, staging):
            oversize_staging = staging
            asyncio.run(storage.save_creation_source(
                staging,
                ChunkedUpload(b"123456", filename="source.mov", chunk_size=2),
                5,
            ))
    assert caught.value.code == "source_too_large"
    assert caught.value.status_code == 413
    assert not oversize_staging.exists()

    with storage.staged_creation(tmp_path) as (cid, staging):
        source = asyncio.run(storage.save_creation_source(
            staging,
            ChunkedUpload(b"video", filename="source.mp4"),
            5,
        ))
        final = tmp_path / cid
        final.mkdir()
        (final / "marker").write_bytes(b"existing")
        with pytest.raises(storage.CreationStorageError) as conflict:
            storage.publish_staged_creation(
                tmp_path,
                staging,
                cid,
                meta=_meta(cid),
                effective_request=_effective_request(),
                client_request_id="minimal-create-conflict",
                source=source,
                source_filename="source.mp4",
                source_reference_url=None,
            )
        assert conflict.value.code == "creation_id_conflict"
        assert conflict.value.status_code == 409

    assert (final / "marker").read_bytes() == b"existing"
    assert not staging.exists()


def test_replacement_image_enforces_exact_byte_limit(tmp_path):
    image_bytes = _encoded_image(".png")
    with storage.staged_creation(tmp_path) as (_cid, staging):
        frozen = asyncio.run(storage.save_creation_replacement_image(
            staging,
            ChunkedUpload(
                image_bytes,
                filename="image.png",
                content_type="image/png",
            ),
            len(image_bytes),
        ))
        assert frozen.bytes == len(image_bytes)

    with pytest.raises(storage.CreationStorageError) as caught:
        with storage.staged_creation(tmp_path) as (_cid, staging):
            asyncio.run(storage.save_creation_replacement_image(
                staging,
                ChunkedUpload(
                    image_bytes,
                    filename="image.png",
                    content_type="image/png",
                ),
                len(image_bytes) - 1,
            ))
    assert caught.value.code == "replacement_image_too_large"
    assert caught.value.status_code == 413


def test_freeze_downloaded_source_hashes_in_chunks_and_enforces_limit(tmp_path):
    with storage.staged_creation(tmp_path) as (_cid, staging):
        downloaded = staging / "source.mp4"
        downloaded.write_bytes(b"downloaded-source")
        frozen = storage.freeze_creation_source_file(downloaded, 100)
        assert frozen.sha256 == hashlib.sha256(b"downloaded-source").hexdigest()
        assert frozen.bytes == len(b"downloaded-source")

    with pytest.raises(storage.CreationStorageError) as caught:
        with storage.staged_creation(tmp_path) as (_cid, staging):
            downloaded = staging / "source.mp4"
            downloaded.write_bytes(b"too-large")
            storage.freeze_creation_source_file(downloaded, 3)
    assert caught.value.code == "source_too_large"


def test_publish_rehashes_frozen_input_before_exposing_conversation(tmp_path):
    with storage.staged_creation(tmp_path) as (cid, staging):
        source = asyncio.run(storage.save_creation_source(
            staging,
            ChunkedUpload(b"original", filename="source.mp4"),
            8,
        ))
        source.path.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="bytes changed"):
            storage.publish_staged_creation(
                tmp_path,
                staging,
                cid,
                meta=_meta(cid),
                effective_request=_effective_request(),
                client_request_id="minimal-create-tamper",
                source=source,
                source_filename="source.mp4",
                source_reference_url=None,
            )
    assert not (tmp_path / cid).exists()


def test_publish_failure_before_atomic_rename_leaves_no_visible_project(
    tmp_path, monkeypatch,
):
    source_bytes = b"validated-source"

    def fail_before_publish(_source, _destination):
        raise OSError("injected pre-publish failure")

    monkeypatch.setattr(storage.os, "rename", fail_before_publish)
    with pytest.raises(OSError, match="pre-publish failure"):
        with storage.staged_creation(tmp_path) as (cid, staging):
            source = asyncio.run(storage.save_creation_source(
                staging,
                ChunkedUpload(source_bytes, filename="source.mp4"),
                len(source_bytes),
            ))
            storage.publish_staged_creation(
                tmp_path,
                staging,
                cid,
                meta=_meta(cid),
                effective_request=_effective_request(),
                client_request_id="minimal-pre-publish-failure",
                source=source,
                source_filename="source.mp4",
                source_reference_url=None,
            )

    assert storage.list_conversations(tmp_path) == []
    assert not (tmp_path / cid).exists()
    assert not staging.exists()


def test_creation_lock_is_private_hidden_and_rejects_symlink(tmp_path):
    with storage.creation_lock(tmp_path):
        lock_path = tmp_path / ".minimal-creation.lock"
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        assert storage.list_conversations(tmp_path) == []

    lock_path.unlink()
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    lock_path.symlink_to(target)
    with pytest.raises(OSError):
        with storage.creation_lock(tmp_path):
            pass
    assert target.read_bytes() == b"unchanged"
