import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import NamedTuple

ALLOWED_EXT = {".mp4", ".mov", ".webm"}
_CHUNK = 1024 * 1024
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# files 白名单：segments/<正整数N>/work/(keyframes|postprocessed)/<纯文件名>
_SEG_FILE_RE = re.compile(r"^([1-9]\d*)/work/(keyframes|postprocessed)/([^/]+)$")
_META_LOCKS: dict[str, threading.Lock] = {}
_META_LOCKS_GUARD = threading.Lock()


class UploadError(ValueError):
    """上传校验失败（HTTP 层转 422）。"""


class VideoProbe(NamedTuple):
    duration_s: float
    width: int
    height: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_title(filename: str) -> str:
    """原文件名仅作展示：去路径、去控制字符、限 80 字。"""
    base = _CONTROL_RE.sub("", filename.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    stem = Path(base).stem.strip()
    return stem[:80] or "untitled"


def _write_meta(cdir: Path, meta: dict) -> None:
    payload = json.dumps(meta, ensure_ascii=False, indent=2)
    fd, temporary = tempfile.mkstemp(prefix=".meta-", suffix=".json", dir=cdir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cdir / "meta.json")
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _meta_lock(cdir: Path):
    key = str(cdir.resolve())
    with _META_LOCKS_GUARD:
        lock = _META_LOCKS.setdefault(key, threading.Lock())
    with lock:
        yield


def new_conversation(data_dir: Path, note: str, orig_name: str, client_request_id: str = "",
                     voice_mode: str = "keep", target_language: str = "") -> dict:
    cid = uuid.uuid4().hex
    cdir = data_dir / cid
    (cdir / "work").mkdir(parents=True)
    now = _now()
    meta = {
        "schema_version": 2,
        "id": cid,
        "title": note or sanitize_title(orig_name),
        "note": note,
        "status": "queued",
        "error": None,
        "created_at": now,
        "updated_at": now,
        "keyframes": [],
        "prompt": None,
        "voice_mode": voice_mode,
        "duration_s": None,
        "fit_required": None,
        "dialogue_mode": "auto",
        "generation": None,
    }
    if client_request_id:
        meta["client_request_id"] = client_request_id
    if target_language:
        meta["target_language"] = target_language
    _write_meta(cdir, meta)
    return meta


def update_meta(data_dir: Path, cid: str, **changes) -> dict | None:
    """合并写字段并刷新 updated_at；cid 非法或不存在返回 None。"""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None:
            return None
        meta.update(changes)
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def load_meta(data_dir: Path, cid: str) -> dict | None:
    if not _ID_RE.match(cid):
        return None
    p = data_dir / cid / "meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def list_conversations(data_dir: Path) -> list[dict]:
    if not data_dir.is_dir():
        return []
    metas = []
    for cdir in data_dir.iterdir():
        if cdir.is_dir() and _ID_RE.match(cdir.name) and (cdir / "meta.json").is_file():
            metas.append(json.loads((cdir / "meta.json").read_text()))
    metas.sort(key=lambda m: m["created_at"], reverse=True)
    return metas


def remove_conversation(data_dir: Path, cid: str) -> None:
    if _ID_RE.match(cid):
        shutil.rmtree(data_dir / cid, ignore_errors=True)


async def save_upload(cdir: Path, upload, max_bytes: int) -> Path:
    """流式落盘为 source.<ext>，超限即删并报错；不读进内存。"""
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise UploadError(f"unsupported extension: {ext or '(none)'}")
    dest = cdir / f"source{ext}"
    written = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await upload.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadError(f"file exceeds {max_bytes} bytes")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def probe_video(path: Path) -> VideoProbe:
    """一次 ffprobe 得到源视频时长与首个视频流尺寸。"""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration:stream=width,height",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise UploadError(f"ffprobe failed: {e}") from e
    if r.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        payload = json.loads(r.stdout)
        duration = float(payload["format"]["duration"])
        stream = payload["streams"][0]
        width, height = stream["width"], stream["height"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise UploadError("cannot parse video duration or dimensions") from e
    if not isfinite(duration) or duration <= 0:
        raise UploadError("cannot parse video duration")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise UploadError("cannot parse video dimensions")
    return VideoProbe(duration, width, height)


def probe_audio(path: Path) -> bool:
    """ffprobe 探测是否有音轨（-select_streams a，同 video-maker 的 probe_audio 逻辑）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise UploadError(f"ffprobe failed: {e}") from e
    if r.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        return bool(json.loads(r.stdout or "{}").get("streams"))
    except ValueError as e:
        raise UploadError("cannot parse audio probe result") from e


def resolve_file(data_dir: Path, cid: str, name: str) -> Path | None:
    """白名单映射 + resolved-path 防穿越；越界/不存在返回 None。"""
    if not _ID_RE.match(cid):
        return None
    cdir = (data_dir / cid).resolve()
    if name == "source.mp4":
        # 源视频扩展名不定（ALLOWED_EXT 内），取唯一 source.*；缺文件走下方 is_file 判空
        cand = next(iter(sorted(cdir.glob("source.*"))), cdir / "source.mp4")
    elif name == "preview.mp4":
        cand = cdir / "preview.mp4"
    elif name == "generated.mp4":
        cand = cdir / "generated.mp4"
    elif name == "contact_sheet.jpg":
        cand = cdir / "work" / "contact_sheet.jpg"
    elif name.startswith("keyframes/"):
        fn = name[len("keyframes/"):]
        if not fn or fn != Path(fn).name:
            return None
        cand = cdir / "work" / "keyframes" / fn
    elif name.startswith("postprocessed/"):
        fn = name[len("postprocessed/"):]
        if not fn or fn != Path(fn).name:
            return None
        cand = cdir / "work" / "postprocessed" / fn
    elif name.startswith("segments/"):
        m = _SEG_FILE_RE.match(name[len("segments/"):])
        if not m or m.group(3) != Path(m.group(3)).name:
            return None
        cand = cdir / "work" / "segments" / m.group(1) / "work" / m.group(2) / m.group(3)
    else:
        return None
    resolved = cand.resolve()
    if not resolved.is_relative_to(cdir) or not resolved.is_file():
        return None
    return resolved
