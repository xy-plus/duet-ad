"""Deterministically derive one receipt-bound, audio-free reference MP4.

The derivation is intentionally local and singular: copy the frozen source
bytes into a private snapshot, remux exactly its sole video stream, validate
the complete timestamp structure, then publish the MP4 and canonical receipt.
There is no provider call, transcode, quality decision, retry, or fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "duet.reference-video"
VERSION = 1
MAX_DURATION_S = 10
_FFMPEG_TIMEOUT_S = 120
_FFPROBE_TIMEOUT_S = 60
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
FFMPEG_PATH = Path("/usr/bin/ffmpeg")
FFPROBE_PATH = Path("/usr/bin/ffprobe")

# Paths are placeholders so the command recorded in every receipt is stable.
# The runtime substitutes private snapshot/candidate paths only.
FFMPEG_COMMAND = (
    "/usr/bin/ffmpeg",
    "-v",
    "error",
    "-nostdin",
    "-fflags",
    "+bitexact",
    "-copyts",
    "-i",
    "{source}",
    "-map",
    "0:v:0",
    "-c:v",
    "copy",
    "-an",
    "-map_metadata",
    "-1",
    "-map_chapters",
    "-1",
    "-fflags",
    "+bitexact",
    "-avoid_negative_ts",
    "disabled",
    "-movflags",
    "+faststart",
    "{output}",
)

_PRESERVED_VIDEO_FIELDS = (
    "codec_name",
    "codec_tag_string",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "time_base",
    "start_pts",
    "start_time",
    "duration_ts",
    "duration",
    "r_frame_rate",
    "avg_frame_rate",
    "frame_count",
    "packet_timeline",
    "frame_timeline",
)


class ReferenceVideoError(RuntimeError):
    """The local reference-video artifact failed its structural contract."""


@dataclass(frozen=True)
class ReferenceVideoResult:
    output_path: Path
    receipt_path: Path
    source_sha256: str
    output_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class _Probe:
    streams: tuple[str, ...]
    format_name: str
    video: dict[str, object]


@dataclass(frozen=True)
class _ToolIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _FrozenTool:
    path: Path
    sha256: str
    identity: _ToolIdentity

    def receipt(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.identity.size,
        }


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ReferenceVideoError("reference_video_receipt_invalid") from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise ReferenceVideoError("reference_video_file_invalid") from None
    return digest.hexdigest()


def _clean_error(stderr: str) -> str:
    text = " ".join(stderr.strip().split())
    return text[-600:] if text else "no diagnostic output"


def _tool_identity(value: os.stat_result) -> _ToolIdentity:
    return _ToolIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
    )


def _tool_is_executable(value: os.stat_result) -> bool:
    execute_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    return stat.S_ISREG(value.st_mode) and bool(value.st_mode & execute_bits)


def _open_tool(path: Path, *, changed: bool) -> tuple[int, _ToolIdentity]:
    code = "reference_video_tool_changed" if changed else "reference_video_tool_invalid"
    if not path.is_absolute():
        raise ReferenceVideoError(code)
    descriptor = -1
    try:
        linked = path.lstat()
        if (
            stat.S_ISLNK(linked.st_mode)
            or not _tool_is_executable(linked)
            or not os.access(path, os.X_OK, follow_symlinks=False)
        ):
            raise ReferenceVideoError(code)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if (
            not _tool_is_executable(opened)
            or _tool_identity(linked) != _tool_identity(opened)
        ):
            raise ReferenceVideoError(code)
        return descriptor, _tool_identity(opened)
    except ReferenceVideoError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise ReferenceVideoError(code) from None


def _verify_open_tool(tool: _FrozenTool, descriptor: int) -> None:
    try:
        opened = os.fstat(descriptor)
        linked = tool.path.lstat()
    except OSError:
        raise ReferenceVideoError("reference_video_tool_changed") from None
    if (
        stat.S_ISLNK(linked.st_mode)
        or not _tool_is_executable(linked)
        or not _tool_is_executable(opened)
        or _tool_identity(linked) != tool.identity
        or _tool_identity(opened) != tool.identity
    ):
        raise ReferenceVideoError("reference_video_tool_changed")


def _freeze_tool(path: Path) -> _FrozenTool:
    descriptor, identity = _open_tool(path, changed=False)
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        tool = _FrozenTool(path=path, sha256=digest.hexdigest(), identity=identity)
        _verify_open_tool(tool, descriptor)
        return tool
    finally:
        os.close(descriptor)


def _run_checked(
    tool: _FrozenTool, argv: list[str], *, timeout: int
) -> subprocess.CompletedProcess[str]:
    if not argv or argv[0] != str(tool.path):
        raise ReferenceVideoError("reference_video_tool_invalid")
    descriptor, identity = _open_tool(tool.path, changed=True)
    if identity != tool.identity:
        os.close(descriptor)
        raise ReferenceVideoError("reference_video_tool_changed")
    try:
        # Execute the already verified inode while retaining the absolute path
        # as argv[0]. This closes the replace-between-check-and-exec window.
        return subprocess.run(
            argv,
            executable=f"/proc/self/fd/{descriptor}",
            pass_fds=(descriptor,),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            _verify_open_tool(tool, descriptor)
        finally:
            os.close(descriptor)


def _root_path(value: str | Path) -> Path:
    try:
        raw = Path(value)
        if raw.is_symlink():
            raise ReferenceVideoError("reference_video_path_invalid")
        root = raw.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ReferenceVideoError("reference_video_path_invalid") from None
    if not root.is_dir():
        raise ReferenceVideoError("reference_video_path_invalid")
    return root


def _relative_path(value: str | Path) -> str:
    try:
        text = os.fspath(value)
    except TypeError:
        raise ReferenceVideoError("reference_video_path_invalid") from None
    if not isinstance(text, str) or not text or "\x00" in text or "\\" in text:
        raise ReferenceVideoError("reference_video_path_invalid")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReferenceVideoError("reference_video_path_invalid")
    return text


def _checked_existing_file(root: Path, relative: str) -> Path:
    current = root
    try:
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.is_symlink():
                raise ReferenceVideoError("reference_video_path_invalid")
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ReferenceVideoError("reference_video_path_invalid") from None
    if not resolved.is_file():
        raise ReferenceVideoError("reference_video_file_invalid")
    return resolved


def _checked_parent(root: Path, relative: str) -> Path:
    current = root
    parts = PurePosixPath(relative).parts[:-1]
    try:
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ReferenceVideoError("reference_video_path_invalid")
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ReferenceVideoError("reference_video_path_invalid") from None
    if not resolved.is_dir():
        raise ReferenceVideoError("reference_video_path_invalid")
    return resolved


def _checked_target(parent: Path, name: str) -> Path:
    target = parent / name
    try:
        if target.is_symlink():
            raise ReferenceVideoError("reference_video_path_invalid")
        if target.exists() and not stat.S_ISREG(
            target.stat(follow_symlinks=False).st_mode
        ):
            raise ReferenceVideoError("reference_video_path_invalid")
    except OSError:
        raise ReferenceVideoError("reference_video_path_invalid") from None
    return target


def _snapshot_source(source: Path, snapshot: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    size = 0
    descriptor = -1
    try:
        descriptor = os.open(source, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReferenceVideoError("reference_video_file_invalid")
        input_handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        with input_handle, snapshot.open("xb") as output_handle:
            for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                output_handle.write(chunk)
            after = os.fstat(input_handle.fileno())
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or size != before.st_size:
            raise ReferenceVideoError("reference_video_source_changed")
    except ReferenceVideoError:
        raise
    except OSError:
        raise ReferenceVideoError("reference_video_file_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), size


def _run_json(tool: _FrozenTool, argv: list[str]) -> dict[str, Any]:
    try:
        result = _run_checked(tool, argv, timeout=_FFPROBE_TIMEOUT_S)
    except FileNotFoundError:
        raise ReferenceVideoError("reference_video_ffprobe_not_found") from None
    except subprocess.TimeoutExpired:
        raise ReferenceVideoError("reference_video_ffprobe_timeout") from None
    if result.returncode != 0:
        raise ReferenceVideoError(
            f"reference_video_ffprobe_failed: {_clean_error(result.stderr)}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ReferenceVideoError("reference_video_ffprobe_invalid") from None
    if not isinstance(payload, dict):
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    return payload


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    return value


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    return value


def _positive_fraction(value: object) -> Fraction:
    text = _nonempty_string(value)
    try:
        fraction = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ReferenceVideoError("reference_video_ffprobe_invalid") from None
    if fraction <= 0:
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    return fraction


def _timeline(
    entries: object,
    *,
    fields: tuple[str, ...],
    start_pts: int,
) -> dict[str, object]:
    if not isinstance(entries, list) or not entries:
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    normalized: list[list[int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReferenceVideoError("reference_video_ffprobe_invalid")
        row = [_integer(entry.get(field)) for field in fields]
        row[0] -= start_pts
        if len(row) > 1 and fields[1] in {"dts", "best_effort_timestamp"}:
            row[1] -= start_pts
        normalized.append(row)
    return {
        "basis": "stream_start_pts",
        "count": len(normalized),
        "sha256": hashlib.sha256(_canonical_json(normalized)).hexdigest(),
    }


def _probe(path: Path, ffprobe: _FrozenTool) -> _Probe:
    inventory = _run_json(
        ffprobe,
        [
            str(ffprobe.path),
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type:format=format_name",
            "-of",
            "json",
            str(path),
        ]
    )
    raw_streams = inventory.get("streams")
    raw_format = inventory.get("format")
    if not isinstance(raw_streams, list) or not isinstance(raw_format, dict):
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    streams: list[str] = []
    video_count = 0
    for item in raw_streams:
        if not isinstance(item, dict):
            raise ReferenceVideoError("reference_video_ffprobe_invalid")
        stream_type = _nonempty_string(item.get("codec_type"))
        streams.append(stream_type)
        video_count += stream_type == "video"
    if video_count != 1:
        raise ReferenceVideoError("reference_video_video_stream_invalid")
    format_name = _nonempty_string(raw_format.get("format_name"))

    packet_payload = _run_json(
        ffprobe,
        [
            str(ffprobe.path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_packets",
            "-show_entries",
            (
                "stream=codec_name,codec_tag_string,profile,level,width,height,"
                "pix_fmt,time_base,start_pts,start_time,duration_ts,duration,"
                "r_frame_rate,avg_frame_rate,nb_frames:packet=pts,dts,duration"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    frame_payload = _run_json(
        ffprobe,
        [
            str(ffprobe.path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pts,best_effort_timestamp,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream_values = packet_payload.get("streams")
    if not isinstance(stream_values, list) or len(stream_values) != 1:
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    stream = stream_values[0]
    if not isinstance(stream, dict):
        raise ReferenceVideoError("reference_video_ffprobe_invalid")

    start_pts = _integer(stream.get("start_pts"))
    duration_ts = _integer(stream.get("duration_ts"))
    time_base_text = _nonempty_string(stream.get("time_base"))
    time_base = _positive_fraction(time_base_text)
    duration_text = _nonempty_string(stream.get("duration"))
    start_time = _nonempty_string(stream.get("start_time"))
    try:
        decimal_duration = Decimal(duration_text)
        decimal_start = Decimal(start_time)
    except InvalidOperation:
        raise ReferenceVideoError("reference_video_ffprobe_invalid") from None
    if (
        duration_ts <= 0
        or not decimal_duration.is_finite()
        or decimal_duration <= 0
        or not decimal_start.is_finite()
        or duration_ts * time_base > MAX_DURATION_S
        or decimal_duration > MAX_DURATION_S
    ):
        raise ReferenceVideoError("reference_video_duration_invalid")
    _positive_fraction(stream.get("r_frame_rate"))
    _positive_fraction(stream.get("avg_frame_rate"))

    packets = _timeline(
        packet_payload.get("packets"),
        fields=("pts", "dts", "duration"),
        start_pts=start_pts,
    )
    frames = _timeline(
        frame_payload.get("frames"),
        fields=("pts", "best_effort_timestamp", "duration"),
        start_pts=start_pts,
    )
    nb_frames = stream.get("nb_frames")
    if nb_frames is not None:
        try:
            declared_frames = int(nb_frames)
        except (TypeError, ValueError):
            raise ReferenceVideoError("reference_video_ffprobe_invalid") from None
        if declared_frames != frames["count"]:
            raise ReferenceVideoError("reference_video_ffprobe_invalid")

    video = {
        "codec_name": _nonempty_string(stream.get("codec_name")),
        "codec_tag_string": _nonempty_string(stream.get("codec_tag_string")),
        "profile": _nonempty_string(stream.get("profile")),
        "level": _integer(stream.get("level")),
        "width": _integer(stream.get("width")),
        "height": _integer(stream.get("height")),
        "pix_fmt": _nonempty_string(stream.get("pix_fmt")),
        "time_base": time_base_text,
        "start_pts": start_pts,
        "start_time": start_time,
        "duration_ts": duration_ts,
        "duration": duration_text,
        "r_frame_rate": _nonempty_string(stream.get("r_frame_rate")),
        "avg_frame_rate": _nonempty_string(stream.get("avg_frame_rate")),
        "frame_count": frames["count"],
        "packet_timeline": packets,
        "frame_timeline": frames,
    }
    if video["width"] <= 0 or video["height"] <= 0:
        raise ReferenceVideoError("reference_video_ffprobe_invalid")
    return _Probe(tuple(streams), format_name, video)


def _run_ffmpeg(source: Path, output: Path, ffmpeg: _FrozenTool) -> None:
    command = (str(ffmpeg.path), *FFMPEG_COMMAND[1:])
    argv = [
        str(source) if item == "{source}" else str(output) if item == "{output}" else item
        for item in command
    ]
    try:
        result = _run_checked(ffmpeg, argv, timeout=_FFMPEG_TIMEOUT_S)
    except FileNotFoundError:
        raise ReferenceVideoError("reference_video_ffmpeg_not_found") from None
    except subprocess.TimeoutExpired:
        raise ReferenceVideoError("reference_video_ffmpeg_timeout") from None
    if result.returncode != 0:
        raise ReferenceVideoError(
            f"reference_video_ffmpeg_failed: {_clean_error(result.stderr)}"
        )


def _sync_file(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise ReferenceVideoError("reference_video_publish_failed") from None


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise ReferenceVideoError("reference_video_publish_failed") from None


def derive_reference_video(
    *,
    root: str | Path,
    source_path: str | Path,
    expected_source_sha256: str,
    output_path: str | Path,
    receipt_path: str | Path,
) -> ReferenceVideoResult:
    """Create one deterministic, audio-free MP4 from an exact frozen segment.

    Artifact paths are strictly root-relative and every existing component must
    be non-symlinked. The output and receipt must share an existing directory;
    validated temporary files are replaced into that directory, receipt last.
    """

    base = _root_path(root)
    source_relative = _relative_path(source_path)
    output_relative = _relative_path(output_path)
    receipt_relative = _relative_path(receipt_path)
    if (
        not isinstance(expected_source_sha256, str)
        or _SHA256_RE.fullmatch(expected_source_sha256) is None
    ):
        raise ReferenceVideoError("reference_video_expected_sha256_invalid")
    if len({source_relative, output_relative, receipt_relative}) != 3:
        raise ReferenceVideoError("reference_video_path_invalid")
    output_posix = PurePosixPath(output_relative)
    receipt_posix = PurePosixPath(receipt_relative)
    if (
        output_posix.suffix != ".mp4"
        or receipt_posix.suffix != ".json"
        or output_posix.parent != receipt_posix.parent
    ):
        raise ReferenceVideoError("reference_video_path_invalid")

    source = _checked_existing_file(base, source_relative)
    output_parent = _checked_parent(base, output_relative)
    receipt_parent = _checked_parent(base, receipt_relative)
    if output_parent != receipt_parent:
        raise ReferenceVideoError("reference_video_path_invalid")
    output = _checked_target(output_parent, output_posix.name)
    receipt = _checked_target(receipt_parent, receipt_posix.name)
    ffmpeg = _freeze_tool(FFMPEG_PATH)
    ffprobe = _freeze_tool(FFPROBE_PATH)

    with tempfile.TemporaryDirectory(
        prefix=".reference-video-", dir=output_parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        snapshot = temporary / "source.snapshot"
        candidate = temporary / "reference.mp4"
        temporary_receipt = temporary / "reference_video.json"

        source_sha256, source_size = _snapshot_source(source, snapshot)
        if source_sha256 != expected_source_sha256:
            raise ReferenceVideoError("reference_video_source_hash_mismatch")
        source_probe = _probe(snapshot, ffprobe)

        _run_ffmpeg(snapshot, candidate, ffmpeg)
        output_probe = _probe(candidate, ffprobe)
        if output_probe.streams != ("video",):
            raise ReferenceVideoError("reference_video_audio_contract_mismatch")
        if "mp4" not in output_probe.format_name.split(","):
            raise ReferenceVideoError("reference_video_container_mismatch")
        if output_probe.video != source_probe.video:
            raise ReferenceVideoError("reference_video_structure_mismatch")

        output_sha256 = _sha256(candidate)
        output_size = candidate.stat().st_size
        payload = {
            "schema": SCHEMA,
            "version": VERSION,
            "contract": {
                "audio_stream_count": 0,
                "container": "mp4",
                "max_duration_s": MAX_DURATION_S,
                "preserved_video_fields": list(_PRESERVED_VIDEO_FIELDS),
                "timestamp_basis": "stream_start_pts",
                "video_mode": "stream_copy",
            },
            "derivation": {
                "argv": [str(ffmpeg.path), *FFMPEG_COMMAND[1:]],
            },
            "tools": {
                "ffmpeg": ffmpeg.receipt(),
                "ffprobe": ffprobe.receipt(),
            },
            "source": {
                "path": source_relative,
                "sha256": source_sha256,
                "size": source_size,
                "streams": list(source_probe.streams),
                "video": source_probe.video,
            },
            "output": {
                "path": output_relative,
                "sha256": output_sha256,
                "size": output_size,
                "streams": list(output_probe.streams),
                "video": output_probe.video,
            },
        }
        receipt_bytes = _canonical_json(payload)
        try:
            with temporary_receipt.open("xb") as handle:
                handle.write(receipt_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            raise ReferenceVideoError("reference_video_publish_failed") from None
        _sync_file(candidate)

        # Recheck final names immediately before the same-filesystem replaces.
        _checked_target(output_parent, output_posix.name)
        _checked_target(receipt_parent, receipt_posix.name)
        try:
            os.replace(candidate, output)
            os.replace(temporary_receipt, receipt)
        except OSError:
            raise ReferenceVideoError("reference_video_publish_failed") from None
        _sync_directory(output_parent)

    return ReferenceVideoResult(
        output_path=output,
        receipt_path=receipt,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
    )
