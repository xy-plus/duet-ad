"""Seedream 后处理编排：HTTP 门控 + 后台逐帧编辑。

门控顺序（仿 seedance.submit 模式）：ENABLE_SEEDREAM_EDIT 关 → 501；会话不存在 → 404；
confirm 非严格 True → 409；四选项至少选一否则 422（非 bool 值 → 422 明确类型错误）；
status != done → 409；meta.postprocess 已在 running → 409；face_hold 被勾选但 cv2 haarcascade
数据不可用 → 503（不静默降级）；上次 done/failed 的 options 与本次不同 → 409（防旧产物
贴新标签）；锁内复查（running / 产物完整）后置 running。
后台任务（BackgroundTasks，独立路径不吃管道闸；可并发，每会话一把锁）：
收集目标帧（单段 work/keyframes/*.png；多段 work/segments/N/work/keyframes/*.png）→ 按勾选选项
构造中文编辑指令（多选项分号连接；face_hold 先 cv2 haarcascade 正面人脸检测，有人脸才做，
无人脸跳过该帧的此选项；无适用选项的帧整帧跳过）→ 逐帧 seedream.edit_image(confirm=True)
产出写 work/postprocessed/<帧名>.png 或 work/segments/N/work/postprocessed/<帧名>.png → 有人脸被
处理时，该帧所属段（或单段）的 prompt 末尾追加动作线（写回 prompt.txt 与 meta 对应 prompt）→
任一帧失败整体 failed（error 指明帧名，已成功帧保留且重跑跳过）→
meta.postprocess = {status: running|done|failed, options, frames, error}（内部字段）。
cv2 检测（cascade 加载 / imread / detectMultiScale）一律 asyncio.to_thread，不堵事件循环。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2

from app import seedream, storage
from app.config import Settings
from app.sanitize import sanitize

OPTION_KEYS = ("change_bg", "face_hold", "remove_subtitle", "remove_brand")

_INSTRUCTIONS = {
    "change_bg": "将图片背景更换为简洁干净的背景，保持主体人物与物品不变",
    "face_hold": "将图片中的人物改为用手捂住脸的造型，其余保持不变",
    "remove_subtitle": "移除图片中的所有字幕、水印和贴纸元素，其余保持不变",
    "remove_brand": "移除图片中的所有品牌标志、logo、商标等版权元素，其余保持不变",
}

# 有人脸被处理时追加到所属段（或单段）prompt 末尾的动作线
FACE_LINE = "图中所有人物在1秒内快速把手放下到一个合理的位置，然后按照正常节奏进行后续剧情"


class PostprocessError(Exception):
    """后处理门控/执行失败（路由层转成 status+detail，同 SubmitError 模式）。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


async def start(
    settings: Settings, cid: str, payload: dict, locks: dict[str, asyncio.Lock]
) -> dict[str, bool]:
    """门控 + 置 running；返回勾选选项（路由层据此调度后台任务）。"""
    if not settings.enable_seedream_edit:
        raise PostprocessError(501, "Seedream edit is disabled.")
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
        raise PostprocessError(404, "not found")
    if payload.get("confirm") is not True:
        raise PostprocessError(409, "confirmation required")
    options = _parse_options(payload)
    if meta.get("status") != "done":
        raise PostprocessError(409, "artifacts not ready")
    if (meta.get("postprocess") or {}).get("status") == "running":
        raise PostprocessError(409, "already running")
    if options.get("face_hold") and await asyncio.to_thread(_load_cascade) is None:
        raise PostprocessError(503, "face detection data unavailable")
    last = meta.get("postprocess") or {}
    if last.get("status") in ("done", "failed") and last.get("options") != options:
        raise PostprocessError(409, "options changed since last run")
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None or (meta.get("postprocess") or {}).get("status") == "running":
            raise PostprocessError(409, "already running")
        _targets(settings.data_dir / cid, meta)  # 产物完整才受理（帧目录缺失 → 409）
        storage.update_meta(settings.data_dir, cid, postprocess={
            "status": "running", "options": options, "frames": [], "error": None,
        })
    return options


async def run_task(
    settings: Settings, cid: str, options: dict[str, bool], lock: asyncio.Lock
) -> None:
    """后台任务：逐帧编辑；任一帧失败 → meta.postprocess failed（已成功帧保留，重跑跳过）。"""
    # data_dir 可能是相对路径（生产默认 "data"）：子进程带 cwd 时相对路径会错位，统一起点解析为绝对
    cdir = (settings.data_dir / cid).resolve()
    frames: list[str] = []
    try:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None:
            raise PostprocessError(404, "not found")
        if options.get("face_hold"):
            cascade = await asyncio.to_thread(_load_cascade)
            if cascade is None:
                raise PostprocessError(503, "face detection data unavailable")
        else:
            cascade = None
        face_segments: set[int | None] = set()
        for seg_index, src, out in _targets(cdir, meta):
            if out.is_file():
                frames.append(_frame_ref(seg_index, out.name))  # 已成功帧保留，重跑不重复扣费
                continue
            has_face = (
                await asyncio.to_thread(_detect_face, src, cascade)
                if options.get("face_hold") else False
            )
            if has_face:
                face_segments.add(seg_index)
            prompt = _build_instruction(options, has_face)
            if not prompt:
                continue  # 该帧无适用选项（face_hold 且无人脸）→ 整帧跳过
            try:
                await seedream.edit_image(settings, cdir, src, prompt, out, lock, True)
            except seedream.SeedreamError as e:
                raise PostprocessError(502, f"frame {out.name} failed: {sanitize(e.detail)}") from None
            except Exception as e:
                raise PostprocessError(502, f"frame {out.name} failed: {sanitize(str(e))}") from None
            frames.append(_frame_ref(seg_index, out.name))
        if face_segments:
            _append_face_line(settings, cid, cdir, meta, face_segments)
        post = {"status": "done", "options": options, "frames": frames, "error": None}
    except PostprocessError as e:
        post = {"status": "failed", "options": options, "frames": frames, "error": e.detail}
    except Exception as e:
        post = {"status": "failed", "options": options, "frames": frames, "error": sanitize(str(e))}
    storage.update_meta(settings.data_dir, cid, postprocess=post)


def _parse_options(payload: dict) -> dict[str, bool]:
    """options 白名单校验：至少选一，否则 422；选项值非 bool → 422 明确类型错误；未知键忽略。"""
    raw = payload.get("options")
    if not isinstance(raw, dict):
        raise PostprocessError(422, "at least one option required")
    options: dict[str, bool] = {}
    for key in OPTION_KEYS:
        value = raw.get(key)
        if value is not None and not isinstance(value, bool):
            raise PostprocessError(422, "options must be booleans")
        options[key] = bool(value)
    if not any(options.values()):
        raise PostprocessError(422, "at least one option required")
    return options


def _targets(cdir: Path, meta: dict) -> list[tuple[int | None, Path, Path]]:
    """收集 (段号|None, 源帧, 目标帧)：单段 = work/keyframes；多段 = work/segments/N/work/keyframes。"""
    segs = meta.get("segments") or []
    if segs:
        out: list[tuple[int | None, Path, Path]] = []
        for seg in segs:
            n = seg.get("index")
            src_dir = cdir / "work" / "segments" / str(n) / "work" / "keyframes"
            dst_dir = cdir / "work" / "segments" / str(n) / "work" / "postprocessed"
            files = (
                sorted(p for p in src_dir.glob("*.png") if p.is_file())
                if src_dir.is_dir() else []
            )
            for p in files:
                out.append((n, p, dst_dir / p.name))
        if not out:
            raise PostprocessError(409, "artifacts not ready")
        return out
    kdir = cdir / "work" / "keyframes"
    files = sorted(p for p in kdir.glob("*.png") if p.is_file()) if kdir.is_dir() else []
    if not files:
        raise PostprocessError(409, "artifacts not ready")
    return [(None, p, cdir / "work" / "postprocessed" / p.name) for p in files]


def _frame_ref(seg_index: int | None, name: str) -> str:
    """frames 列表条目：单段 = 帧名；多段 = segments/N/work/postprocessed/帧名。"""
    return name if seg_index is None else f"segments/{seg_index}/work/postprocessed/{name}"


def _load_cascade():
    """cv2 自带 haarcascade 正面人脸分类器；数据文件缺失/载入失败 → None（face_hold 勾选时 503）。"""
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.is_file():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


def _detect_face(image: Path, cascade=None) -> bool:
    """正面人脸检测；cascade 缺失/图不可解码/无人脸 → False（跳过该帧的此选项）。"""
    if cascade is None:
        return False
    img = cv2.imread(str(image))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bool(数组) 对多元素数组抛 numpy 真值歧义——必须按长度判定（空数组 = 无人脸）
    return len(cascade.detectMultiScale(gray)) > 0


def _build_instruction(options: dict[str, bool], has_face: bool) -> str:
    """多选项合并为一条指令（分号连接）；face_hold 仅当该帧有人脸。"""
    parts = []
    if options.get("change_bg"):
        parts.append(_INSTRUCTIONS["change_bg"])
    if options.get("face_hold") and has_face:
        parts.append(_INSTRUCTIONS["face_hold"])
    if options.get("remove_subtitle"):
        parts.append(_INSTRUCTIONS["remove_subtitle"])
    if options.get("remove_brand"):
        parts.append(_INSTRUCTIONS["remove_brand"])
    return "；".join(parts)


def _append_face_line(
    settings: Settings, cid: str, cdir: Path, meta: dict, face_segments: set
) -> None:
    """有人脸被处理 → 所属段（或单段）prompt 末尾追加动作线，写回 prompt.txt 与 meta。"""
    if None in face_segments:
        prompt = (cdir / "work" / "prompt.txt").read_text(encoding="utf-8")
        if FACE_LINE not in prompt:
            prompt = f"{prompt.rstrip()}\n{FACE_LINE}"
            (cdir / "work" / "prompt.txt").write_text(prompt, encoding="utf-8")
        storage.update_meta(settings.data_dir, cid, prompt=prompt)
    segs = meta.get("segments") or []
    if not segs:
        return
    changed = False
    for seg in segs:
        n = seg.get("index")
        if n not in face_segments:
            continue
        path = cdir / "work" / "segments" / str(n) / "work" / "prompt.txt"
        prompt = path.read_text(encoding="utf-8")
        if FACE_LINE not in prompt:
            prompt = f"{prompt.rstrip()}\n{FACE_LINE}"
            path.write_text(prompt, encoding="utf-8")
            seg["prompt"] = prompt
            changed = True
    if changed:
        storage.update_meta(settings.data_dir, cid, segments=segs)
