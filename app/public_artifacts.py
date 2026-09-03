"""Immutable, receipt-bound publication of public video artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app import storage


_DIRECTORY = "public-v1-artifact"
_CONTENT = "content.mp4"
_MANIFEST = "manifest.json"
_CHUNK = 1024 * 1024


class PublicArtifactError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _publish_lock(cdir: Path) -> Iterator[None]:
    lock_path = cdir / ".public-v1-artifact.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PublicArtifactError("artifact_lock_invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load(cdir: Path) -> tuple[Path, dict[str, Any]] | None:
    root = cdir / _DIRECTORY
    manifest_path = root / _MANIFEST
    content = root / _CONTENT
    if not root.exists() and not manifest_path.exists() and not content.exists():
        return None
    try:
        expected_root = cdir.resolve(strict=True) / _DIRECTORY
        if (
            root.is_symlink()
            or root.resolve(strict=True) != expected_root
            or not root.is_dir()
            or manifest_path.is_symlink()
            or content.is_symlink()
        ):
            raise PublicArtifactError("artifact_corrupt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or manifest.get("filename") != _CONTENT
            or manifest.get("content_type") != "video/mp4"
            or not isinstance(manifest.get("size_bytes"), int)
            or manifest["size_bytes"] <= 0
            or not isinstance(manifest.get("sha256"), str)
            or len(manifest["sha256"]) != 64
            or not isinstance(manifest.get("duration_seconds"), (int, float))
            or isinstance(manifest.get("duration_seconds"), bool)
            or manifest["duration_seconds"] <= 0
            or not isinstance(manifest.get("manifest_sha256"), str)
        ):
            raise PublicArtifactError("artifact_corrupt")
        unsigned = dict(manifest)
        expected_manifest_sha = unsigned.pop("manifest_sha256")
        if hashlib.sha256(_canonical(unsigned)).hexdigest() != expected_manifest_sha:
            raise PublicArtifactError("artifact_corrupt")
        info = content.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != manifest["size_bytes"]:
            raise PublicArtifactError("artifact_corrupt")
        if _sha256(content) != manifest["sha256"]:
            raise PublicArtifactError("artifact_corrupt")
        probe = storage.probe_video(content)
        if abs(float(probe.duration_s) - float(manifest["duration_seconds"])) > 0.25:
            raise PublicArtifactError("artifact_corrupt")
        return content, manifest
    except PublicArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, storage.UploadError):
        raise PublicArtifactError("artifact_corrupt") from None


def publish(cdir: Path, source: Path, *, job_id: str) -> tuple[Path, dict[str, Any]]:
    with _publish_lock(cdir):
        existing = load(cdir)
        if existing is not None:
            if existing[1].get("job_id") != job_id:
                raise PublicArtifactError("artifact_job_mismatch")
            return existing
        staging = Path(tempfile.mkdtemp(
            prefix=".public-v1-artifact-", suffix=".staging", dir=cdir
        ))
        try:
            destination = staging / _CONTENT
            with source.open("rb") as input_stream, destination.open("xb") as output:
                shutil.copyfileobj(input_stream, output, length=_CHUNK)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(destination, 0o600)
            digest = _sha256(destination)
            info = destination.stat()
            probe = storage.probe_video(destination)
            unsigned: dict[str, Any] = {
                "version": 1,
                "job_id": job_id,
                "filename": _CONTENT,
                "content_type": "video/mp4",
                "size_bytes": info.st_size,
                "sha256": digest,
                "duration_seconds": round(float(probe.duration_s), 6),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": None,
            }
            manifest = {
                **unsigned,
                "manifest_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
            }
            manifest_path = staging / _MANIFEST
            with manifest_path.open("xb") as stream:
                stream.write(_canonical(manifest) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(manifest_path, 0o600)
            _fsync_directory(staging)
            try:
                os.rename(staging, cdir / _DIRECTORY)
            except FileExistsError:
                loaded = load(cdir)
                if loaded is None:
                    raise PublicArtifactError("artifact_publish_failed") from None
                return loaded
            _fsync_directory(cdir)
            loaded = load(cdir)
            if loaded is None:
                raise PublicArtifactError("artifact_publish_failed")
            return loaded
        finally:
            if staging.exists():
                shutil.rmtree(staging)
