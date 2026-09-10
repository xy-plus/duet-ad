"""Durable, owner-scoped media staging for the public quotation gate."""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

QUOTE_TTL_SECONDS = 24 * 60 * 60
_ID = re.compile(r"^vq_([0-9a-f]{32})$")
_LOGGER = logging.getLogger(__name__)


class QuoteStore:
    def __init__(self, root: Path):
        self.root = root / ".public-quotes"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._sweep = iter(())

    @contextmanager
    def lock(self):
        with (self.root / ".lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def directory(self, quote_id: str) -> Path:
        match = _ID.fullmatch(quote_id)
        if match is None:
            raise FileNotFoundError("quote")
        path = self.root / match.group(1)
        if path.is_symlink():
            raise ValueError("invalid quote directory")
        return path

    def read(self, quote_id: str, owner_id: str) -> dict:
        path = self.directory(quote_id) / "quote.json"
        if path.is_symlink():
            raise ValueError("invalid quote record")
        record = json.loads(path.read_text())
        if record["owner_id"] != owner_id:
            raise FileNotFoundError("quote")
        if record["quote_id"] != quote_id:
            raise ValueError("invalid quote identity")
        return record

    def save(self, record: dict):
        directory = self.directory(record["quote_id"])
        directory.mkdir(mode=0o700, exist_ok=True)
        temporary = directory / f".record-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                json.dump(record, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / "quote.json")
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, quote_id: str, staged: dict, **metadata) -> dict:
        directory = self.directory(quote_id)
        directory.mkdir(mode=0o700, exist_ok=True)
        media_directory = directory / "media"
        media_directory.mkdir(mode=0o700, exist_ok=True)
        frozen = dict(staged)
        for key, name in (("staged_source", "source.mp4"), ("staged_replacement", "replacement")):
            if staged[key] is not None:
                destination = media_directory / name
                os.replace(staged[key], destination)
                os.chmod(destination, 0o600)
                with destination.open("rb") as handle:
                    os.fsync(handle.fileno())
                frozen[key] = name
        if "staged_replacements" in staged:
            names = []
            for index, path in enumerate(staged["staged_replacements"], 1):
                name = "replacement" if index == 1 else f"replacement-{index}"
                if index > 1:
                    destination = media_directory / name
                    os.replace(path, destination)
                    os.chmod(destination, 0o600)
                    with destination.open("rb") as handle:
                        os.fsync(handle.fileno())
                names.append(name)
            frozen["staged_replacements"] = names
        descriptor = os.open(media_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        expires = time.time() + QUOTE_TTL_SECONDS
        record = dict(metadata, quote_id=quote_id, staged=frozen,
                      expires_epoch=expires,
                      expires_at=datetime.fromtimestamp(expires, timezone.utc).isoformat())
        self.save(record)
        return record

    @contextmanager
    def staged_copy(self, record: dict):
        """The existing commit consumes its inputs; keep originals for retries."""
        directory = self.directory(record["quote_id"])
        if (directory / "media").is_symlink():
            raise ValueError("invalid media directory")
        attempt = directory / f"attempt-{uuid.uuid4().hex}"
        attempt.mkdir(mode=0o700)
        staged = dict(record["staged"])
        try:
            for key, name, receipt_key in (("staged_source", "source.mp4", "source"),
                                           ("staged_replacement", "replacement", "replacement_image")):
                if staged[key] is None:
                    continue
                if staged[key] != name:
                    raise ValueError("invalid staged media path")
                source = directory / "media" / name
                if source.is_symlink() or not source.is_file():
                    raise ValueError("invalid staged media")
                destination = attempt / name
                shutil.copyfile(source, destination)
                receipt = staged["input_receipt"][receipt_key]
                with destination.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if destination.stat().st_size != receipt["bytes"] or digest != receipt["sha256"]:
                    raise ValueError("staged media changed")
                staged[key] = destination
            if "staged_replacements" in staged:
                from app.store import replacement_receipts
                receipts = replacement_receipts(staged["input_receipt"])
                names = staged["staged_replacements"]
                expected_names = ["replacement" if index == 1 else f"replacement-{index}" for index in range(1, len(receipts) + 1)]
                if names != expected_names:
                    raise ValueError("invalid staged replacement list")
                copies = []
                for index, (name, receipt) in enumerate(zip(names, receipts), 1):
                    destination = attempt / name
                    if index > 1:
                        source = directory / "media" / name
                        if source.is_symlink() or not source.is_file():
                            raise ValueError("invalid staged media")
                        shutil.copyfile(source, destination)
                        with destination.open("rb") as handle:
                            digest = hashlib.file_digest(handle, "sha256").hexdigest()
                        if destination.stat().st_size != receipt["bytes"] or digest != receipt["sha256"]:
                            raise ValueError("staged media changed")
                    copies.append(destination)
                staged["staged_replacements"] = copies
            yield staged
        finally:
            try:
                shutil.rmtree(attempt)
            except OSError:
                # Cleanup cannot turn a published job into an undispatched job.
                _LOGGER.warning("Could not reclaim quote attempt; deferred to sweep")

    def discard_media(self, record: dict):
        media_directory = self.directory(record["quote_id"]) / "media"
        if media_directory.is_symlink():
            raise ValueError("invalid media directory")
        if media_directory.exists():
            shutil.rmtree(media_directory)

    def cleanup(self):
        """Sweep bounded batches; retain small receipts for idempotent replays."""
        batch = list(itertools.islice(self._sweep, 64))
        if not batch:
            self._sweep = self.root.iterdir()
            batch = list(itertools.islice(self._sweep, 64))
        for directory in batch:
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name) or directory.is_symlink():
                continue
            path = directory / "quote.json"
            try:
                # The caller holds the quote lock, so no attempt is live. These
                # copies are otherwise orphaned by process termination.
                for attempt in itertools.islice(directory.glob("attempt-*"), 64):
                    if re.fullmatch(r"attempt-[0-9a-f]{32}", attempt.name) and not attempt.is_symlink():
                        shutil.rmtree(attempt)
                if not path.exists():
                    # An interrupted initial record write has no issued quote.
                    if directory.stat().st_mtime + QUOTE_TTL_SECONDS <= time.time():
                        shutil.rmtree(directory)
                    continue
                if path.is_symlink():
                    continue
                record = json.loads(path.read_text())
                if record["quote_id"] != f"vq_{directory.name}":
                    continue
                if record["expires_epoch"] <= time.time():
                    self.discard_media(record)
            except (OSError, ValueError, KeyError, TypeError):
                continue
