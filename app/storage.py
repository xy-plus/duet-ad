import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_EXT = {".mp4", ".mov", ".webm"}
_CHUNK = 1024 * 1024
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class UploadError(ValueError):
    """上传校验失败（HTTP 层转 422）。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_title(filename: str) -> str:
    """原文件名仅作展示：去路径、去控制字符、限 80 字。"""
    base = _CONTROL_RE.sub("", filename.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    stem = Path(base).stem.strip()
    return stem[:80] or "untitled"


def _write_meta(cdir: Path, meta: dict) -> None:
    (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def new_conversation(data_dir: Path, note: str, orig_name: str) -> dict:
    cid = uuid.uuid4().hex
    cdir = data_dir / cid
    (cdir / "work").mkdir(parents=True)
    now = _now()
    meta = {
        "id": cid,
        "title": note or sanitize_title(orig_name),
        "note": note,
        "status": "queued",
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
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


def probe_video(path: Path, max_duration_s: float) -> float:
    """ffprobe 实际探测：打不开或超时即报错；时长超限即报错。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise UploadError(f"ffprobe failed: {e}") from e
    if r.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        duration = float(json.loads(r.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as e:
        raise UploadError("cannot parse video duration") from e
    if duration > max_duration_s:
        raise UploadError(f"duration {duration:.1f}s exceeds {max_duration_s}s")
    return duration


def resolve_file(data_dir: Path, cid: str, name: str) -> Path | None:
    """白名单映射 + resolved-path 防穿越；越界/不存在返回 None。"""
    if not _ID_RE.match(cid):
        return None
    cdir = (data_dir / cid).resolve()
    if name == "preview.mp4":
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
    else:
        return None
    resolved = cand.resolve()
    if not resolved.is_relative_to(cdir) or not resolved.is_file():
        return None
    return resolved
