"""MediaKit erase-image 后处理编排：HTTP 门控 + 后台并行逐帧擦除。

门控顺序：ENABLE_MEDIAKIT_ERASE 关 → 501；会话不存在 → 404；
confirm 非严格 True → 409；字幕/品牌选项至少选一否则 422（未知键或非 bool 值 → 422）；
status != done → 409；meta.postprocess 已在 running → 409；上次 done/failed 的 options 与本次
不同 → 409（防旧产物贴新标签，锁定比对只认当前 OPTION_KEYS 共有键——旧状态中的
废弃键忽略；纯废弃形态放行并清旧产物）；锁内复查（running / 产物完整）后置 running。
后台任务（BackgroundTasks，独立路径不吃管道闸；可并发，每会话一把锁）：
收集目标帧（单段 work/keyframes/*.png；多段 work/segments/N/work/keyframes/*.png）→ 按勾选选项
映射文字/图标擦除场景 → 未跳过帧 asyncio.gather 并行
mediakit.erase_image(confirm=True)，进程级信号量（主进程
mediakit_sem，MEDIAKIT_CONCURRENCY）限并发 → 产出写 work/postprocessed/<帧名>.png 或
work/segments/N/work/postprocessed/<帧名>.png → 任一帧失败整体 failed（error 列失败帧名；
其余帧照常跑完，已成功帧保留且重跑跳过）→
meta.postprocess = {status: running|done|failed, options, frames, error}（内部字段；
running 期间每成功一帧即写回 frames，前端 2s 轮询据此显示 n/m 实时进度）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app import mediakit, storage
from app.config import Settings
from app.sanitize import sanitize

OPTION_KEYS = ("remove_subtitle", "remove_brand")
_LEGACY_OPTION_KEYS = frozenset({"change_bg", "face_hold"})
_CLIENT_REFRESH_MESSAGE = "页面版本已更新，请刷新页面后重试。"

_SCENES = {
    "remove_subtitle": mediakit.TEXT_SCENE,
    "remove_brand": mediakit.ICON_SCENE,
}


class PostprocessError(Exception):
    """后处理门控/执行失败（路由层转成 status+detail，同 SubmitError 模式）。"""

    def __init__(self, status: int, detail: str | dict[str, str]) -> None:
        if isinstance(detail, dict):
            if set(detail) != {"code", "message"} or not all(
                isinstance(detail[key], str) and detail[key]
                for key in ("code", "message")
            ):
                raise TypeError("structured postprocess detail must contain safe code and message")
            public_detail: str | dict[str, str] = {
                "code": detail["code"],
                "message": detail["message"],
            }
        elif isinstance(detail, str):
            public_detail = detail
        else:
            raise TypeError("postprocess detail must be a public string or safe structure")
        super().__init__(str(public_detail))
        self.status = status
        self.detail = public_detail


async def start(
    settings: Settings, cid: str, payload: dict, locks: dict[str, asyncio.Lock]
) -> dict[str, bool]:
    """门控 + 置 running；返回勾选选项（路由层据此调度后台任务）。"""
    if not settings.enable_mediakit_erase:
        raise PostprocessError(501, "MediaKit erase is disabled.")
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
        raise PostprocessError(404, "not found")
    if set(payload) != {"confirm", "options"}:
        raise PostprocessError(422, "invalid_postprocess_request")
    if payload.get("confirm") is not True:
        raise PostprocessError(409, "confirmation required")
    options = _parse_options(payload)
    if meta.get("status") != "done":
        raise PostprocessError(409, "artifacts not ready")
    if isinstance(meta.get("generation"), dict) or meta.get("_input_owner"):
        raise PostprocessError(409, "generation_already_started")
    if (meta.get("postprocess") or {}).get("status") == "running":
        raise PostprocessError(409, "already running")
    last = meta.get("postprocess") or {}
    if last.get("status") in ("done", "failed") and not _options_match(last.get("options"), options):
        raise PostprocessError(409, _options_locked_detail())
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None or (meta.get("postprocess") or {}).get("status") == "running":
            raise PostprocessError(409, "already running")
        if isinstance(meta.get("generation"), dict) or meta.get("_input_owner"):
            raise PostprocessError(409, "generation_already_started")
        last_in = meta.get("postprocess") or {}
        if (
            last_in.get("status") in ("done", "failed")
            and not _options_match(last_in.get("options"), options)
        ):
            raise PostprocessError(409, _options_locked_detail())
        _targets(settings.data_dir / cid, meta)  # 产物完整才受理（帧目录缺失 → 409）
        # 纯废弃形态（旧版只勾 change_bg 等）放行重跑：清除旧产物，强制全帧重编辑防贴错标签；
        # 锁内执行（用锁内重载的 meta），避免「清产物后复查失败毁掉旧产物」与并发 stale 读窗口；
        # 同选项正常重跑不清产物（跳过逻辑依赖已有输出）
        if last_in.get("status") in ("done", "failed") and _is_pure_legacy(last_in.get("options")):
            _clear_postprocessed(settings.data_dir / cid, meta)
        storage.update_meta(settings.data_dir, cid, postprocess={
            "status": "running", "options": options, "frames": [], "error": None,
        })
    return options


async def run_task(
    settings: Settings, cid: str, options: dict[str, bool], sem: asyncio.Semaphore
) -> None:
    """后台任务：并行逐帧编辑（进程级信号量 sem 限并发）；任一帧失败 → meta.postprocess failed
    （error 列失败帧名，其余帧照常跑完，已成功帧保留，重跑跳过）；
    父任务被取消（graceful shutdown）→ 写 failed(error=cancelled) 后继续传播取消；
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
    except asyncio.CancelledError:
        # 父任务被取消（uvicorn graceful shutdown）：gather 自身抛 CancelledError（BaseException），
        # 不写终态会永久 running、start 永久 409；先写 failed（update_meta 同步，可安全调用）再传播取消
        post = {"status": "failed", "options": options, "frames": sorted(frames), "error": "cancelled"}
        storage.update_meta(settings.data_dir, cid, postprocess=post)
        raise
    except PostprocessError as e:
        error = e.detail if isinstance(e.detail, str) else e.detail["message"]
        post = {"status": "failed", "options": options, "frames": sorted(frames), "error": error}
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
    """单帧编辑（gather 成员）：信号量内提交；成功后追加帧名并写回进度。

    run_task 收集阶段已保证各帧 out 互异且未产出，同帧不会被并发重复提交；erase_image 不再
    接收并发锁（每帧新建锁退化为自守卫，无实际防护），同 out 仅一次提交由收集不变量 + start
    门控保证。
    """
    try:
        async with sem:
            await mediakit.erase_image(
                settings, cdir, src, out, True, _selected_scenes(options),
            )
    except mediakit.MediaKitError as e:
        raise PostprocessError(502, f"frame {out.name} failed: {sanitize(e.detail)}") from None
    except Exception as e:
        raise PostprocessError(502, f"frame {out.name} failed: {sanitize(str(e))}") from None
    frames.append(_frame_ref(seg_index, out.name))
    _write_progress(settings, cid, options, frames)


def _frame_error(name: str, e: BaseException) -> str:
    """失败帧错误文案：_edit_one 已包装成 PostprocessError 的直接用 detail，异常形态兜底脱敏。"""
    if isinstance(e, PostprocessError):
        return e.detail if isinstance(e.detail, str) else e.detail["message"]
    return f"frame {name} failed: {sanitize(str(e))}"


def _options_match(last_options: object, options: dict[str, bool]) -> bool:
    """锁定比对只认当前 OPTION_KEYS 内共有键：旧状态 options 里的废弃键忽略；
    上次 options 非 dict（如 None）一律视为不一致；
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
    """options 白名单校验：未知键、非 bool 或未选中任何选项均 fail closed 为 422。"""
    raw = payload.get("options")
    if not isinstance(raw, dict):
        raise PostprocessError(422, "at least one option required")
    unknown = sorted(set(raw) - set(OPTION_KEYS))
    if unknown:
        if (
            set(unknown) <= _LEGACY_OPTION_KEYS
            and all(isinstance(value, bool) for value in raw.values())
        ):
            raise PostprocessError(409, _CLIENT_REFRESH_MESSAGE)
        raise PostprocessError(422, f"unknown options: {', '.join(unknown)}")
    options: dict[str, bool] = {}
    for key in OPTION_KEYS:
        value = raw.get(key)
        if value is not None and not isinstance(value, bool):
            raise PostprocessError(422, "options must be booleans")
        options[key] = bool(value)
    if not any(options.values()):
        raise PostprocessError(422, "at least one option required")
    return options


def _options_locked_detail() -> dict[str, str]:
    return {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }


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
    """逐帧写回进度：frames 累计已完成帧，status 保持 running。storage.update_meta 全程同步、
    append 与写回之间无 await，单事件循环内每帧写回是原子块，多协程写者无丢失更新。"""
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": options, "frames": frames, "error": None,
    })


def _selected_scenes(options: dict[str, bool]) -> tuple[str, ...]:
    """按稳定选项顺序生成 MediaKit 擦除场景；双选即两个有回执的阶段。"""
    return tuple(_SCENES[key] for key in OPTION_KEYS if options.get(key))


def generation_keyframes(cdir: Path, meta: dict, originals: list[Path]) -> list[Path]:
    """Resolve the only keyframe set generation is allowed to consume.

    No postprocess state means the user skipped optimization and the original
    keyframes remain authoritative.  Once optimization exists, generation must
    wait for a complete ``done`` set and then consume every corresponding output;
    silently falling back to originals would make the confirmation misleading.
    """
    state = meta.get("postprocess")
    if state is None:
        return originals
    if not isinstance(state, dict) or state.get("status") != "done":
        raise PostprocessError(409, "postprocess_not_ready")
    frame_refs = state.get("frames")
    if not isinstance(frame_refs, list) or any(
        not isinstance(item, str) for item in frame_refs
    ):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    available = set(frame_refs)
    selected: list[Path] = []
    work = cdir.resolve() / "work"
    for original in originals:
        source = original.resolve()
        try:
            relative = source.relative_to(work)
        except ValueError:
            raise PostprocessError(409, "postprocess_artifacts_invalid") from None
        parts = relative.parts
        if len(parts) == 2 and parts[0] == "keyframes":
            output = work / "postprocessed" / source.name
            frame_ref = source.name
        elif (
            len(parts) == 5
            and parts[0] == "segments"
            and parts[1].isdigit()
            and parts[2:4] == ("work", "keyframes")
        ):
            output = work / "segments" / parts[1] / "work" / "postprocessed" / source.name
            frame_ref = f"segments/{parts[1]}/work/postprocessed/{source.name}"
        else:
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        if frame_ref not in available or not output.is_file():
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        selected.append(output)
    if len({path.resolve() for path in selected}) != len(selected):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    return selected
