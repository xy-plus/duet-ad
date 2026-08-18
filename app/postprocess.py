"""Seedream 后处理编排：HTTP 门控 + 后台并行逐帧编辑。

门控顺序（仿 seedance.submit 模式）：ENABLE_SEEDREAM_EDIT 关 → 501；会话不存在 → 404；
confirm 非严格 True → 409；三选项至少选一否则 422（非 bool 值 → 422 明确类型错误）；
status != done → 409；meta.postprocess 已在 running → 409；上次 done/failed 的 options 与本次
不同 → 409（防旧产物贴新标签，锁定比对只认当前 OPTION_KEYS 共有键——旧会话四键
options 里的废弃键忽略；纯废弃形态放行并清旧产物）；锁内复查（running / 产物完整）后置 running。
后台任务（BackgroundTasks，独立路径不吃管道闸；可并发，每会话一把锁）：
收集目标帧（单段 work/keyframes/*.png；多段 work/segments/N/work/keyframes/*.png）→ 按勾选选项
构造中文编辑指令（多选项分号连接；face_hold 为条件式指令——含人脸则捂脸、不含人脸保持原样，
条件句放最前，所有帧都发编辑请求，不做人脸预判/过滤）→ 未跳过帧 asyncio.gather 并行
seedream.edit_image(confirm=True, size=按帧像素等比放大的 "WxH")，平台级信号量（主进程
seedream_sem，SEEDREAM_CONCURRENCY）限并发 → 产出写 work/postprocessed/<帧名>.png 或
work/segments/N/work/postprocessed/<帧名>.png → 任一帧失败整体 failed（error 列失败帧名；
其余帧照常跑完，已成功帧保留且重跑跳过）→
meta.postprocess = {status: running|done|failed, options, frames, error}（内部字段；
running 期间每成功一帧即写回 frames，前端 2s 轮询据此显示 n/m 实时进度）。
无人脸帧的编辑输出为近似原图（seedream 条件指令已实证能正确区分），直接存 postprocessed 展示；
将来可加输入-输出变化判定过滤（见 OPEN_ISSUE）。
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import cv2

from app import seedream, storage
from app.config import Settings
from app.sanitize import sanitize

OPTION_KEYS = ("face_hold", "remove_subtitle", "remove_brand")

_INSTRUCTIONS = {
    "face_hold": "如果图片中含有人脸：将图片中的人物改为用手捂住脸的造型。如果图片中不含人脸：跳过捂脸处理，仅执行其余修改。",
    "remove_subtitle": "移除图片中的所有字幕、水印和贴纸元素，其余（尺寸、内容等）保持不变",
    "remove_brand": "图片中的所有品牌标志、logo、商标等版权元素改为不侵权的类似视觉效果的等效物，其余（尺寸、内容等）保持不变",
}

# Seedream size 参数下限（像素数）；不传 size 时模型输出 2048 方形（方向可能失真），
# 故按输入帧尺寸等比放大到 ≥ 下限保持宽高比（实测 1440×2560 可用，无需 16 对齐）
_SEEDREAM_MIN_PIXELS = 3_686_400


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
    last = meta.get("postprocess") or {}
    if last.get("status") in ("done", "failed") and not _options_match(last.get("options"), options):
        raise PostprocessError(409, "options changed since last run")
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None or (meta.get("postprocess") or {}).get("status") == "running":
            raise PostprocessError(409, "already running")
        _targets(settings.data_dir / cid, meta)  # 产物完整才受理（帧目录缺失 → 409）
        # 纯废弃形态（旧版只勾 change_bg 等）放行重跑：清除旧产物，强制全帧重编辑防贴错标签；
        # 锁内执行（用锁内重载的 meta），避免「清产物后复查失败毁掉旧产物」与并发 stale 读窗口；
        # 同选项正常重跑不清产物（跳过逻辑依赖已有输出）
        last_in = meta.get("postprocess") or {}
        if last_in.get("status") in ("done", "failed") and _is_pure_legacy(last_in.get("options")):
            _clear_postprocessed(settings.data_dir / cid, meta)
        storage.update_meta(settings.data_dir, cid, postprocess={
            "status": "running", "options": options, "frames": [], "error": None,
        })
    return options


async def run_task(
    settings: Settings, cid: str, options: dict[str, bool], sem: asyncio.Semaphore
) -> None:
    """后台任务：并行逐帧编辑（平台级信号量 sem 限并发）；任一帧失败 → meta.postprocess failed
    （error 列失败帧名，其余帧照常跑完，已成功帧保留，重跑跳过）；
    每成功一帧即写回 frames（status 保持 running），供前端轮询显示实时进度。"""
    # data_dir 可能是相对路径（生产默认 "data"）：子进程带 cwd 时相对路径会错位，统一起点解析为绝对
    cdir = (settings.data_dir / cid).resolve()
    frames: list[str] = []
    try:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None:
            raise PostprocessError(404, "not found")
        todo: list[tuple[int | None, Path, Path]] = []
        for seg_index, src, out in _targets(cdir, meta):
            if out.is_file():
                frames.append(_frame_ref(seg_index, out.name))  # 已成功帧保留，重跑不重复扣费
                _write_progress(settings, cid, options, frames)
                continue
            todo.append((seg_index, src, out))
        # 并发编辑：return_exceptions 收集，全部完成后统一判定（任一失败 → failed；
        # 其余帧自然跑完，保留更多产物；每帧一次编辑，无同 out 并发，不需会话锁串行）
        results = await asyncio.gather(
            *(_edit_one(settings, cdir, cid, src, out, seg_index, options, frames, sem)
              for seg_index, src, out in todo),
            return_exceptions=True,
        )
        errors = [
            _frame_error(out.name, e)
            for (_, _, out), e in zip(todo, results) if isinstance(e, BaseException)
        ]
        if errors:
            raise PostprocessError(502, sanitize("；".join(errors)))
        post = {"status": "done", "options": options, "frames": sorted(frames), "error": None}
    except PostprocessError as e:
        post = {"status": "failed", "options": options, "frames": sorted(frames), "error": e.detail}
    except Exception as e:
        post = {"status": "failed", "options": options, "frames": sorted(frames), "error": sanitize(str(e))}
    storage.update_meta(settings.data_dir, cid, postprocess=post)


async def _edit_one(
    settings: Settings,
    cdir: Path,
    cid: str,
    src: Path,
    out: Path,
    seg_index: int | None,
    options: dict[str, bool],
    frames: list[str],
    sem: asyncio.Semaphore,
) -> None:
    """单帧编辑（gather 成员）：读帧尺寸算 size → 信号量内提交；成功后追加帧名并写回进度。

    run_task 收集阶段已保证各帧 out 互异且未产出，同帧不会被并发重复提交，故编辑不再复用
    会话锁（会话锁跨整个提交持有，共享会串行化所有帧），改传每帧独立锁保持 edit_image 门控形态。
    """
    try:
        async with sem:
            await seedream.edit_image(
                settings, cdir, src, _build_instruction(options), out, asyncio.Lock(), True,
                size=_read_size(src),
            )
    except seedream.SeedreamError as e:
        raise PostprocessError(502, f"frame {out.name} failed: {sanitize(e.detail)}") from None
    except Exception as e:
        raise PostprocessError(502, f"frame {out.name} failed: {sanitize(str(e))}") from None
    frames.append(_frame_ref(seg_index, out.name))
    _write_progress(settings, cid, options, frames)


def _frame_error(name: str, e: BaseException) -> str:
    """失败帧错误文案：_edit_one 已包装成 PostprocessError 的直接用 detail，异常形态兜底脱敏。"""
    if isinstance(e, PostprocessError):
        return e.detail
    return f"frame {name} failed: {sanitize(str(e))}"


def _read_size(src: Path) -> str:
    """读帧像素尺寸 → Seedream size "WxH"；cv2 读不出（非标准图）→ 空串 = 不传 size。"""
    img = cv2.imread(str(src))
    if img is None:
        return ""
    h, w = img.shape[:2]  # cv2.imread shape 为 (高, 宽, 通道)
    return _fit_size(w, h)


def _fit_size(w: int, h: int) -> str:
    """等比放大到 Seedream size 下限（≥3,686,400 像素）保持宽高比：scale = ceil(sqrt(下限/(w*h)))。"""
    scale = math.ceil(math.sqrt(_SEEDREAM_MIN_PIXELS / (w * h)))
    return f"{w * scale}x{h * scale}"


def _options_match(last_options: object, options: dict[str, bool]) -> bool:
    """锁定比对只认当前 OPTION_KEYS 内共有键：旧会话 options 里的废弃键忽略（历史会话可能
    存四键 options）；上次 options 非 dict（如 None）一律视为不一致；
    纯废弃形态（上次在当前键上无任何 True，如旧版只勾 change_bg）视为无锁定放行——
    否则该类会话永久 409 无出口。"""
    if not isinstance(last_options, dict):
        return False
    if _is_pure_legacy(last_options):
        return True
    return all(
        last_options.get(key) == options[key] for key in OPTION_KEYS if key in last_options
    )


def _is_pure_legacy(last_options: object) -> bool:
    """上次 options 为 dict 且当前键无任何 True（旧版只勾 change_bg 等废弃选择）→ 纯废弃形态。"""
    return isinstance(last_options, dict) and not any(
        last_options.get(key) is True for key in OPTION_KEYS if key in last_options
    )


def _clear_postprocessed(cdir: Path, meta: dict) -> None:
    """删除既有 postprocessed 产物（纯废弃形态重跑时旧产物无对应新选项意义，防贴错标签）。"""
    targets = [cdir / "work" / "postprocessed"]
    targets += [
        cdir / "work" / "segments" / str(seg.get("index")) / "work" / "postprocessed"
        for seg in meta.get("segments") or []
    ]
    for d in targets:
        if d.is_dir():
            for p in d.glob("*.png"):
                p.unlink(missing_ok=True)


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


def _write_progress(
    settings: Settings, cid: str, options: dict[str, bool], frames: list[str]
) -> None:
    """逐帧写回进度：frames 累计已完成帧，status 保持 running（run_task 是 running 期间唯一写者）。"""
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": options, "frames": frames, "error": None,
    })


def _build_instruction(options: dict[str, bool]) -> str:
    """多选项合并为一条指令（分号连接）；face_hold 条件句放最前（无人脸帧由 seedream 保持原样）。"""
    parts = []
    if options.get("face_hold"):
        parts.append(_INSTRUCTIONS["face_hold"])
    if options.get("remove_subtitle"):
        parts.append(_INSTRUCTIONS["remove_subtitle"])
    if options.get("remove_brand"):
        parts.append(_INSTRUCTIONS["remove_brand"])
    return "；".join(parts)
