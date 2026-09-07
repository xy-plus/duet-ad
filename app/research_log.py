from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


class ResearchLog:
    """Unredacted local evidence whose own failures never fail the pipeline."""

    def __init__(self, project_dir: Path) -> None:
        self.root = project_dir / "logs"
        self.trace = self.root / "trace.jsonl"
        self._directory_lock = threading.RLock()
        self._trace_lock = threading.Lock()
        try:
            self._ensure_directory(self.root)
        except Exception as exc:
            self._report(exc)

    @staticmethod
    def _report(exc: Exception) -> None:
        try:
            print(f"research log failure: {exc}", file=sys.stderr, flush=True)
        except Exception:
            pass

    def _ensure_directory(self, path: Path) -> None:
        with self._directory_lock:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(path, 0o700)
                return
            current = self.root
            for part in relative.parts:
                current = current / part
                current.mkdir(exist_ok=True, mode=0o700)
                os.chmod(current, 0o700)

    def _write_bytes(self, path: Path, content: bytes, *, append: bool = False) -> None:
        self._ensure_directory(path.parent)
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab" if append else "wb") as handle:
                descriptor = -1
                handle.write(content)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def event(self, event: str, **fields: Any) -> None:
        try:
            row = {"at": now(), "event": event, **jsonable(fields)}
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            with self._trace_lock:
                self._write_bytes(self.trace, encoded, append=True)
        except Exception as exc:
            self._report(exc)

    _ATTEMPT_RE = re.compile(r"^attempt-(\d+)$")

    def attempt_dir(self, node: str, scope: str, attempt: int) -> Path:
        path = self.root / node / scope / f"attempt-{attempt}"
        try:
            self._ensure_directory(path)
        except Exception as exc:
            self._report(exc)
        return path

    def next_attempt(self, node: str, scope: str) -> int:
        """One past the highest existing attempt number for this node/scope.

        Resumed nodes continue numbering instead of restarting at 1, so the
        failed attempt's original input/output evidence is never overwritten.
        """
        try:
            base = self.root / node / scope
            highest = 0
            if base.is_dir():
                for child in base.iterdir():
                    match = self._ATTEMPT_RE.match(child.name)
                    if match is not None:
                        highest = max(highest, int(match.group(1)))
            return highest + 1
        except Exception as exc:
            self._report(exc)
            return 1

    def write_json(self, path: Path, value: Any) -> None:
        try:
            encoded = json.dumps(
                jsonable(value),
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            self._write_bytes(path, encoded)
        except Exception as exc:
            self._report(exc)

    def write_text(self, path: Path, value: str) -> None:
        try:
            self._write_bytes(path, value.encode("utf-8"))
        except Exception as exc:
            self._report(exc)

    def copy_media(self, destination: Path, paths: Iterable[Path]) -> None:
        try:
            self._ensure_directory(destination)
        except Exception as exc:
            self._report(exc)
            return
        for index, source in enumerate(paths, 1):
            try:
                if not source.is_file():
                    continue
                target = destination / f"{index:03d}-{source.name}"
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.fchmod(descriptor, 0o600)
                    with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
                        descriptor = -1
                        shutil.copyfileobj(input_handle, output_handle)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
            except Exception as exc:
                self._report(exc)

    def error(self, path: Path, exc: Exception) -> None:
        self.write_json(
            path,
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            },
        )
