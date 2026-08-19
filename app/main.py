import asyncio
import hmac
import math
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import downloader, frame_fit, h3, pipeline, postprocess, prepared_input, storage
from app.auth import require_auth
from app.codex_runner import CodexRunner
from app.config import Settings, get_settings

_RATE_LIMIT = 10  # 每 IP 每分钟上传次数
_RATE_WINDOW_S = 60
# 前端幂等键（boot / 内容变更 / 上传成功时轮换，失败重试复用）；空 = 不参与幂等（兼容 curl）
_CLIENT_REQUEST_ID_RE = re.compile(r"^[0-9A-Za-z-]{8,64}$")
_DIALOGUE_MODES = frozenset({"auto", "edit", "custom", "none"})
_FIT_MODES = frozenset({"none", "crop", "pad"})
_GENERATION_ACTIVE = frozenset({"queued", "running"})
_GENERATION_RETRYABLE = frozenset({"failed"})
_GENERATION_RESUMABLE = frozenset({"resume_required"})
_AMBIGUOUS_SUBMIT_ERRORS = frozenset(
    {"state_persist_failed", "submission_unknown", "h3_internal_error"}
)
_KNOWN_TASK_ERRORS = frozenset(
    {
        "ir_query_failed",
        "ir_timeout",
        "h3_query_failed",
        "h3_timeout",
        "download_failed",
        "download_dns_failed",
        "download_peer_unverified",
        "output_write_failed",
        "output_probe_failed",
    }
)


class _SubmitError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _is_read_only(meta: dict) -> bool:
    return meta.get("schema_version") != 2


def _public_lines(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text, start_s, end_s = item.get("text"), item.get("start_s"), item.get("end_s")
        if (
            isinstance(text, str)
            and isinstance(start_s, (int, float))
            and not isinstance(start_s, bool)
            and isinstance(end_s, (int, float))
            and not isinstance(end_s, bool)
        ):
            lines.append({"text": text, "start_s": float(start_s), "end_s": float(end_s)})
    return lines


def _automatic_public_lines(meta: dict) -> list[dict]:
    provenance = meta.get("voice_line_provenance")
    if isinstance(provenance, list):
        return _public_lines([line for line in provenance if isinstance(line, dict) and line.get("kept") is True])
    if _is_read_only(meta):
        return _public_lines(meta.get("voice_lines"))
    return []


def _public_dialogue(meta: dict) -> dict:
    mode = meta.get("dialogue_mode")
    if mode not in _DIALOGUE_MODES:
        mode = "auto"
    automatic = _automatic_public_lines(meta)
    if mode == "auto":
        effective = automatic
    elif mode in {"edit", "custom"}:
        effective = _public_lines(meta.get("voice_lines"))
    else:
        effective = []
    return {"mode": mode, "lines": effective, "auto_lines": automatic}


def _public_generation(meta: dict) -> dict | None:
    generation = meta.get("generation")
    if not isinstance(generation, dict):
        return None
    status = generation.get("status")
    if status == "failed" and generation.get("error") in _KNOWN_TASK_ERRORS:
        status = "resume_required"
    return {
        "status": status,
        "error": generation.get("error"),
        "attempt": generation.get("attempt"),
        "client_request_id": generation.get("client_request_id"),
    }


def _receipt_version(cdir: Path, meta: dict) -> int | None:
    name = meta.get("prepared_input_receipt")
    if not isinstance(name, str) or not name or name != Path(name).name:
        return None
    return prepared_input.RECEIPT_VERSION if (cdir / name).is_file() else None


def _timeouts(settings: Settings) -> h3.Timeouts:
    return h3.Timeouts(
        request_s=settings.h3_request_timeout_s,
        upload_s=settings.h3_upload_timeout_s,
        ir_poll_s=settings.h3_ir_poll_timeout_s,
        h3_poll_s=settings.h3_poll_timeout_s,
        download_s=settings.h3_download_timeout_s,
        poll_interval_s=settings.h3_poll_interval_s,
    )


def _credentials_ready(settings: Settings) -> bool:
    return bool(settings.minimax_api_key.strip() and settings.autodl_art_token.strip())


def _source(cdir: Path) -> Path:
    sources = sorted(cdir.glob("source.*"))
    if len(sources) != 1 or not sources[0].is_file():
        raise _SubmitError(409, "prepared_input_invalid")
    return sources[0]


def _original_keyframes(cdir: Path, meta: dict) -> list[Path]:
    names = meta.get("keyframes")
    if not isinstance(names, list) or not 1 <= len(names) <= 9:
        raise _SubmitError(409, "prepared_input_invalid")
    paths = []
    for name in names:
        if not isinstance(name, str) or name != Path(name).name or Path(name).suffix.lower() != ".png":
            raise _SubmitError(409, "prepared_input_invalid")
        path = cdir / "work" / "keyframes" / name
        if not path.is_file():
            raise _SubmitError(409, "prepared_input_invalid")
        paths.append(path)
    if len({path.name for path in paths}) != len(paths):
        raise _SubmitError(409, "prepared_input_invalid")
    return paths


def _validated_dialogue(meta: dict, payload: dict) -> tuple[dict, ...]:
    mode = payload.get("dialogue_mode")
    if mode not in _DIALOGUE_MODES:
        raise _SubmitError(422, "invalid_dialogue")
    has_lines = "lines" in payload
    duration = meta.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise _SubmitError(409, "prepared_input_invalid")
    filter_enabled = bool(meta.get("vocal_filter_enabled", True))
    try:
        if mode == "auto":
            if has_lines:
                raise _SubmitError(422, "invalid_dialogue")
            raw = meta.get("voice_line_provenance")
            if not isinstance(raw, list):
                if meta.get("voice_lines"):
                    raise _SubmitError(409, "prepared_input_invalid")
                raw = []
            automatic = []
            for item in raw:
                if not isinstance(item, dict) or item.get("kept") is not True:
                    continue
                automatic.append(
                    {
                        "text": item.get("text"),
                        "start_s": item.get("start_s"),
                        "end_s": item.get("end_s"),
                        "classification": item.get("classification"),
                    }
                )
            return prepared_input.prepare_dialogue(
                "auto",
                duration,
                automatic_lines=automatic,
                vocal_filter_enabled=filter_enabled,
            )
        if mode == "none":
            if has_lines:
                raise _SubmitError(422, "invalid_dialogue")
            return prepared_input.prepare_dialogue("none", duration)
        lines = payload.get("lines")
        if not isinstance(lines, list) or not lines:
            raise _SubmitError(422, "invalid_dialogue")
        if any(not isinstance(line, dict) or set(line) != {"text", "start_s", "end_s"} for line in lines):
            raise _SubmitError(422, "invalid_dialogue")
        return prepared_input.prepare_dialogue(
            mode,
            duration,
            supplied_lines=lines,
            vocal_filter_enabled=filter_enabled,
        )
    except prepared_input.PreparedInputError:
        raise _SubmitError(422, "invalid_dialogue") from None


def _validate_submit_payload(meta: dict, payload: dict) -> tuple[str, str, tuple[dict, ...]]:
    if payload.get("confirm") is not True:
        raise _SubmitError(409, "confirmation required")
    allowed = {"confirm", "client_request_id", "dialogue_mode", "lines", "fit_mode"}
    if set(payload) - allowed:
        raise _SubmitError(422, "invalid_submit_request")
    request_id = payload.get("client_request_id")
    if not isinstance(request_id, str) or not _CLIENT_REQUEST_ID_RE.fullmatch(request_id):
        raise _SubmitError(422, "invalid_client_request_id")
    fit_mode = payload.get("fit_mode")
    if fit_mode not in _FIT_MODES:
        raise _SubmitError(422, "invalid_fit_mode")
    if meta.get("fit_required") is True:
        if fit_mode not in {"crop", "pad"}:
            raise _SubmitError(422, "fit_mode_required")
    elif fit_mode != "none":
        raise _SubmitError(422, "fit_mode_not_allowed")
    return request_id, fit_mode, _validated_dialogue(meta, payload)


def _make_h3_request(
    settings: Settings,
    cid: str,
    frozen: prepared_input.PreparedInput,
    client_request_id: str,
) -> h3.H3Request:
    duration = max(1, min(15, math.ceil(frozen.duration_s)))
    return h3.H3Request(
        cid=cid,
        workdir=(settings.data_dir / cid).resolve(),
        client_request_id=client_request_id,
        prompt=frozen.final_prompt.data.decode("utf-8"),
        keyframes=tuple((artifact.path, artifact.data) for artifact in frozen.keyframes),
        voice_texts=frozen.voice_texts,
        voice_receipt=h3.voice_texts_receipt(frozen.voice_texts),
        duration=duration,
        ratio="9:16",
        minimax_api_key=settings.minimax_api_key,
        autodl_token=settings.autodl_art_token,
        timeouts=_timeouts(settings),
    )


def _freeze_submission(
    settings: Settings,
    cid: str,
    meta: dict,
    request_id: str,
    dialogue_mode: str,
    fit_mode: str,
    dialogue: tuple[dict, ...],
) -> h3.H3Request:
    cdir = (settings.data_dir / cid).resolve()
    work = cdir / "work"
    originals = _original_keyframes(cdir, meta)
    if fit_mode == "none":
        keyframes = originals
    else:
        try:
            keyframes = list(frame_fit.fit_frames(originals, work / "h3_frames" / fit_mode, fit_mode))
        except frame_fit.FrameFitError:
            raise _SubmitError(409, "frame_fit_failed") from None
    visual = work / "visual_prompt.txt"
    if not visual.is_file():
        raise _SubmitError(409, "prepared_input_invalid")
    duration = float(meta["duration_s"])
    request_duration = max(1, min(15, math.ceil(duration)))
    engine_request = {
        "context_ir": {"model": h3.IR_MODEL, "duration": request_duration, "ratio": "9:16"},
        "h3": {
            "workflow": h3.H3_WORKFLOW,
            "duration": request_duration,
            "resolution": h3.H3_RESOLUTION,
        },
    }
    try:
        frozen = prepared_input.write_prepared_input(
            root=cdir,
            source=_source(cdir),
            audio=(work / "voice.mp3") if (work / "voice.mp3").is_file() else None,
            keyframes=keyframes,
            visual=visual,
            final=work / "prompt.txt",
            dialogue_mode=dialogue_mode,
            dialogue=dialogue,
            vocal_filter_enabled=bool(meta.get("vocal_filter_enabled", True)),
            duration_s=duration,
            ratio="9:16",
            fit_mode=fit_mode,
            engine_request=engine_request,
        )
    except prepared_input.PreparedInputError:
        raise _SubmitError(409, "prepared_input_invalid") from None
    return _make_h3_request(settings, cid, frozen, request_id)


def _load_h3_request(settings: Settings, cid: str, meta: dict) -> h3.H3Request:
    generation = meta.get("generation")
    request_id = generation.get("client_request_id") if isinstance(generation, dict) else None
    dialogue = meta.get("prepared_dialogue")
    receipt_name = meta.get("prepared_input_receipt")
    if (
        not isinstance(request_id, str)
        or not _CLIENT_REQUEST_ID_RE.fullmatch(request_id)
        or not isinstance(dialogue, list)
        or not isinstance(receipt_name, str)
        or receipt_name != Path(receipt_name).name
    ):
        raise _SubmitError(409, "prepared_input_invalid")
    cdir = (settings.data_dir / cid).resolve()
    try:
        frozen = prepared_input.load_prepared_input(
            cdir,
            cdir / receipt_name,
            expected_dialogue=dialogue,
        )
    except prepared_input.PreparedInputError:
        raise _SubmitError(409, "prepared_input_invalid") from None
    if (
        frozen.dialogue_mode != meta.get("dialogue_mode")
        or frozen.fit_mode != meta.get("fit_mode")
    ):
        raise _SubmitError(409, "prepared_input_invalid")
    return _make_h3_request(settings, cid, frozen, request_id)


def _result_fields(result: h3.H3Result) -> tuple[str, str | None]:
    if result.status == "succeeded":
        return "succeeded", None
    if result.status in {"submission_unknown", "ir_submitting", "h3_submitting"}:
        return "submission_unknown", "submission_unknown"
    if result.status == "ready_for_h3":
        return "resume_required", "ready_for_h3"
    if result.error_code in _KNOWN_TASK_ERRORS:
        return "resume_required", result.error_code
    if result.status in {"ir_running", "h3_running"}:
        return "resume_required", result.status
    if result.status in {"failed", "retryable_failure"}:
        return "failed", result.error_code or "h3_failed"
    if result.status == "not_started":
        return "failed", "h3_state_missing"
    return "failed", "h3_incomplete"


def _finish_generation(settings: Settings, cid: str, request: h3.H3Request, result: h3.H3Result) -> None:
    meta = storage.load_meta(settings.data_dir, cid)
    generation = meta.get("generation") if meta else None
    if not isinstance(generation, dict) or generation.get("client_request_id") != request.client_request_id:
        return
    status, error = _result_fields(result)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={**generation, "status": status, "error": error},
    )


def _generation_error(settings: Settings, cid: str, request: h3.H3Request, code: str) -> None:
    if code in _AMBIGUOUS_SUBMIT_ERRORS or code in _KNOWN_TASK_ERRORS:
        try:
            inspected = h3.inspect(request)
        except Exception:
            if code in _AMBIGUOUS_SUBMIT_ERRORS:
                inspected = h3.H3Result("submission_unknown", None)
            else:
                inspected = h3.H3Result(
                    "retryable_failure",
                    None,
                    retryable=True,
                    error_code=code,
                )
        if inspected.status == "not_started":
            if code in _AMBIGUOUS_SUBMIT_ERRORS:
                inspected = h3.H3Result("submission_unknown", inspected.attempt_id)
            else:
                inspected = h3.H3Result(
                    "retryable_failure",
                    inspected.attempt_id,
                    retryable=True,
                    error_code=code,
                )
        elif code in _AMBIGUOUS_SUBMIT_ERRORS and inspected.status not in {
            "succeeded",
            "submission_unknown",
            "ir_submitting",
            "h3_submitting",
            "ready_for_h3",
            "ir_running",
            "h3_running",
            "failed",
            "retryable_failure",
        }:
            inspected = h3.H3Result("submission_unknown", inspected.attempt_id)
        _finish_generation(settings, cid, request, inspected)
        return
    meta = storage.load_meta(settings.data_dir, cid)
    generation = meta.get("generation") if meta else None
    if isinstance(generation, dict) and generation.get("client_request_id") == request.client_request_id:
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={**generation, "status": "failed", "error": code},
        )


def _run_generation(settings: Settings, cid: str, request: h3.H3Request, retry: bool) -> None:
    meta = storage.load_meta(settings.data_dir, cid)
    generation = meta.get("generation") if meta else None
    if not isinstance(generation, dict) or generation.get("client_request_id") != request.client_request_id:
        return
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={**generation, "status": "running", "error": None},
    )
    try:
        result = h3.retry(request, request.client_request_id) if retry else h3.start(request)
    except h3.H3Error as exc:
        _generation_error(settings, cid, request, exc.code)
        return
    except Exception:
        _generation_error(settings, cid, request, "h3_internal_error")
        return
    _finish_generation(settings, cid, request, result)


def _mark_submission_unknown(settings: Settings, cid: str, generation: dict) -> None:
    """Lock an active paid attempt when it cannot be inspected safely."""
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            **generation,
            "status": "submission_unknown",
            "error": "submission_unknown",
        },
    )


def _resume_generation(settings: Settings, cid: str) -> None:
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None or _is_read_only(meta):
        return
    generation = meta.get("generation")
    if not isinstance(generation, dict) or generation.get("status") not in _GENERATION_ACTIVE:
        return
    if not _credentials_ready(settings):
        _mark_submission_unknown(settings, cid, generation)
        return
    request = None
    try:
        request = _load_h3_request(settings, cid, meta)
        result = h3.resume(request)
    except _SubmitError:
        _mark_submission_unknown(settings, cid, generation)
    except h3.H3Error as exc:
        if request is None:
            _mark_submission_unknown(settings, cid, generation)
        else:
            _generation_error(settings, cid, request, exc.code)
    except Exception:
        if request is not None:
            _generation_error(settings, cid, request, "h3_internal_error")
        else:
            # The persisted generation says a paid attempt may already exist,
            # but we could not rebuild enough immutable input to inspect it.
            # Fail closed: never expose the new-request-id retry path.
            _mark_submission_unknown(settings, cid, generation)
    else:
        _finish_generation(settings, cid, request, result)


class _RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        q = self._hits[ip]
        while q and now - q[0] > _RATE_WINDOW_S:
            q.popleft()
        if len(q) >= _RATE_LIMIT:
            return False
        q.append(now)
        return True


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    app.state.settings = settings
    limiter = _RateLimiter()
    codex_runner = CodexRunner(
        timeout_s=settings.codex_timeout_s, concurrency=settings.codex_concurrency
    )
    # 管道闸：同时处理的会话数上限；拿不到闸的会话保持 queued
    pipeline_sem = threading.Semaphore(settings.codex_concurrency)
    # 创建临界区：幂等查重 + queued 计数 + 建目录必须原子
    create_lock = threading.Lock()
    submit_locks: dict[str, asyncio.Lock] = {}
    postprocess_locks: dict[str, asyncio.Lock] = {}
    # Seedream 后处理并行提交的进程级信号量：单进程内跨会话全局并发上限（SEEDREAM_CONCURRENCY）
    seedream_sem = asyncio.Semaphore(settings.seedream_concurrency)
    app.state.h3_resume_threads = []

    @app.on_event("startup")
    async def resume_h3_generations() -> None:
        if not settings.enable_h3_submit:
            return
        for meta in storage.list_conversations(settings.data_dir):
            generation = meta.get("generation")
            if (
                meta.get("schema_version") == 2
                and isinstance(generation, dict)
                and generation.get("status") in _GENERATION_ACTIVE
            ):
                thread = threading.Thread(
                    target=_resume_generation,
                    args=(settings, meta["id"]),
                    daemon=True,
                    name=f"h3-resume-{meta['id'][:8]}",
                )
                app.state.h3_resume_threads.append(thread)
                thread.start()

    def run_pipeline_gated(cid: str) -> None:
        with pipeline_sem:
            pipeline.run(settings, cid, codex_runner)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.post("/api/login")
    async def login(payload: dict):
        if not hmac.compare_digest(str(payload.get("token", "")), settings.access_token):
            raise HTTPException(status_code=401, detail="invalid token")
        return {"ok": True}

    @app.get("/api/conversations", dependencies=[Depends(require_auth)])
    async def list_conversations():
        return [
            {
                "id": m["id"],
                "title": m["title"],
                "note": m["note"],
                "status": m["status"],
                "created_at": m["created_at"],
                "has_video": (settings.data_dir / m["id"] / "generated.mp4").is_file(),
            }
            for m in storage.list_conversations(settings.data_dir)
        ]

    @app.post("/api/conversations", status_code=201, dependencies=[Depends(require_auth)])
    async def create_conversation(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        file: UploadFile | None = File(None),
        reference_url: str = Form(""),
        note: str = Form(""),
        client_request_id: str = Form(""),
        voice_mode: str = Form("keep"),
        target_language: str = Form(""),
    ):
        ip = request.client.host if request.client else "unknown"
        if not limiter.allow(ip):
            raise HTTPException(status_code=429, detail="too many uploads")
        reference_url = reference_url.strip()
        if (file is None) == (not reference_url):
            raise HTTPException(status_code=400, detail="provide exactly one of file or reference_url")
        client_request_id = client_request_id.strip()
        if client_request_id and not _CLIENT_REQUEST_ID_RE.match(client_request_id):
            raise HTTPException(status_code=400, detail="invalid client_request_id")
        # 口播转换：模式白名单 + 翻译必填目标语言；非 translate 忽略 target_language
        voice_mode = voice_mode.strip()
        if voice_mode not in ("keep", "rewrite", "translate"):
            raise HTTPException(status_code=422, detail=f"invalid voice_mode: {voice_mode}")
        target_language = target_language.strip()
        if voice_mode == "translate" and not target_language:
            raise HTTPException(status_code=422, detail="target_language required for translate")
        with create_lock:
            metas = storage.list_conversations(settings.data_dir)
            if client_request_id:
                for m in metas:
                    if m.get("client_request_id") == client_request_id:
                        # 幂等命中：不建目录、不重复入队，200 返回既有会话
                        response.status_code = 200
                        return {"id": m["id"], "status": m["status"]}
            if sum(1 for m in metas if m["status"] == "queued") >= settings.max_queued:
                raise HTTPException(status_code=429, detail="too many queued tasks")
            meta = storage.new_conversation(
                settings.data_dir,
                note,
                (file.filename or "") if file else reference_url,
                client_request_id,
                voice_mode=voice_mode,
                target_language=target_language if voice_mode == "translate" else "",
            )
        cdir = settings.data_dir / meta["id"]
        try:
            if file is not None:
                dest = await storage.save_upload(cdir, file, settings.max_upload_mb * 1024 * 1024)
            else:
                # 下载最长 download_timeout_s 秒，不能堵事件循环
                dest = await run_in_threadpool(downloader.fetch_reference, reference_url, cdir, settings)
            video = storage.probe_video(dest, min(settings.max_duration_s, 15))
            storage.update_meta(
                settings.data_dir,
                meta["id"],
                duration_s=video.duration_s,
            )
        except (storage.UploadError, downloader.DownloadError) as e:
            storage.remove_conversation(settings.data_dir, meta["id"])
            raise HTTPException(status_code=422, detail=str(e)) from e
        if settings.enable_pipeline:
            background_tasks.add_task(run_pipeline_gated, meta["id"])
        return {"id": meta["id"], "status": "queued"}

    @app.get("/api/conversations/{cid}", dependencies=[Depends(require_auth)])
    async def get_conversation(cid: str):
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None:
            raise HTTPException(status_code=404, detail="not found")
        cdir = settings.data_dir / cid
        return {
            "id": meta["id"],
            "title": meta["title"],
            "note": meta["note"],
            "status": meta["status"],
            "error": meta["error"],
            "created_at": meta["created_at"],
            "updated_at": meta["updated_at"],
            "keyframes": meta.get("keyframes", []),
            "prompt": meta.get("prompt"),
            "segments": meta.get("segments", []),
            "voice_lines": meta.get("voice_lines", []),
            "read_only": _is_read_only(meta),
            "duration_s": meta.get("duration_s"),
            "fit_required": meta.get("fit_required"),
            "fit_mode": meta.get("fit_mode"),
            "dialogue": _public_dialogue(meta),
            "receipt_version": _receipt_version(cdir, meta),
            "generation": _public_generation(meta),
            "has_source": any(cdir.glob("source.*")),
            "has_video": (cdir / "generated.mp4").is_file(),
            "submit_enabled": settings.enable_h3_submit,
            "postprocess": meta.get("postprocess"),
            "postprocess_enabled": settings.enable_seedream_edit,
        }

    @app.get("/api/conversations/{cid}/files/{name:path}", dependencies=[Depends(require_auth)])
    async def get_file(cid: str, name: str):
        path = storage.resolve_file(settings.data_dir, cid, name)
        if path is None:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.post(
        "/api/conversations/{cid}/submit",
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    async def submit_conversation(cid: str, payload: dict, background_tasks: BackgroundTasks):
        if not settings.enable_h3_submit:
            raise HTTPException(status_code=501, detail="H3 submission is disabled.")
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None:
            raise HTTPException(status_code=404, detail="not found")
        if _is_read_only(meta):
            raise HTTPException(status_code=409, detail="read_only")
        try:
            request_id, fit_mode, dialogue = _validate_submit_payload(meta, payload)
        except _SubmitError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        if meta.get("status") != "done":
            raise HTTPException(status_code=409, detail="artifacts not ready")
        if not _credentials_ready(settings):
            raise HTTPException(status_code=503, detail="h3_credentials_missing")

        lock = submit_locks.setdefault(cid, asyncio.Lock())
        async with lock:
            meta = storage.load_meta(settings.data_dir, cid)
            if meta is None:
                raise HTTPException(status_code=404, detail="not found")
            if (settings.data_dir / cid / "generated.mp4").is_file():
                raise HTTPException(status_code=409, detail="already submitted")
            generation = meta.get("generation")
            if isinstance(generation, dict):
                previous_status = generation.get("status")
                if (
                    previous_status == "failed"
                    and generation.get("error") in _KNOWN_TASK_ERRORS
                ):
                    previous_status = "resume_required"
                previous_id = generation.get("client_request_id")
                if previous_status == "submission_unknown":
                    raise HTTPException(status_code=409, detail="submission_outcome_unknown")
                if previous_status not in (
                    _GENERATION_ACTIVE
                    | _GENERATION_RETRYABLE
                    | _GENERATION_RESUMABLE
                    | {"succeeded"}
                ):
                    raise HTTPException(status_code=409, detail="generation_state_invalid")
                if previous_status in _GENERATION_ACTIVE or previous_status == "succeeded":
                    if previous_id == request_id:
                        return {
                            "status": previous_status,
                            "attempt": generation.get("attempt"),
                        }
                    detail = "already submitted" if previous_status == "succeeded" else "generation in progress"
                    raise HTTPException(status_code=409, detail=detail)
                if previous_status in _GENERATION_RETRYABLE and previous_id == request_id:
                    raise HTTPException(status_code=409, detail="new client_request_id required")
                if previous_status in _GENERATION_RESUMABLE:
                    if previous_id != request_id:
                        raise HTTPException(status_code=409, detail="resume_request_id_mismatch")
                    expected_dialogue = meta.get("prepared_dialogue")
                    if (
                        meta.get("dialogue_mode") != payload["dialogue_mode"]
                        or meta.get("fit_mode") != fit_mode
                        or not isinstance(expected_dialogue, list)
                        or expected_dialogue != [dict(line) for line in dialogue]
                    ):
                        raise HTTPException(status_code=409, detail="resume_parameters_changed")
                    previous_attempt = generation.get("attempt")
                    if (
                        isinstance(previous_attempt, bool)
                        or not isinstance(previous_attempt, int)
                        or previous_attempt <= 0
                    ):
                        raise HTTPException(status_code=409, detail="generation_state_invalid")
                    try:
                        request = await asyncio.to_thread(
                            _load_h3_request, settings, cid, meta
                        )
                    except _SubmitError as exc:
                        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
                    except h3.H3Error as exc:
                        raise HTTPException(
                            status_code=503, detail="h3_configuration_invalid"
                        ) from exc
                    storage.update_meta(
                        settings.data_dir,
                        cid,
                        generation={**generation, "status": "queued", "error": None},
                    )
                    background_tasks.add_task(
                        _run_generation, settings, cid, request, False
                    )
                    return {"status": "queued", "attempt": previous_attempt}
            retry = isinstance(generation, dict) and generation.get("status") in _GENERATION_RETRYABLE
            previous_attempt = generation.get("attempt") if isinstance(generation, dict) else 0
            if (
                isinstance(previous_attempt, bool)
                or not isinstance(previous_attempt, int)
                or previous_attempt < 0
            ):
                raise HTTPException(status_code=409, detail="generation_state_invalid")
            attempt = previous_attempt + 1
            dialogue_mode = payload["dialogue_mode"]
            try:
                request = await asyncio.to_thread(
                    _freeze_submission,
                    settings,
                    cid,
                    meta,
                    request_id,
                    dialogue_mode,
                    fit_mode,
                    dialogue,
                )
            except _SubmitError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
            except h3.H3Error as exc:
                raise HTTPException(status_code=503, detail="h3_configuration_invalid") from exc
            bare_lines = [
                {key: line[key] for key in ("text", "start_s", "end_s")}
                for line in dialogue
            ]
            generation = {
                "status": "queued",
                "error": None,
                "attempt": attempt,
                "client_request_id": request_id,
            }
            storage.update_meta(
                settings.data_dir,
                cid,
                dialogue_mode=dialogue_mode,
                voice_lines=bare_lines,
                prompt=request.prompt,
                prepared_dialogue=[dict(line) for line in dialogue],
                prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
                fit_mode=fit_mode,
                generation=generation,
            )
            background_tasks.add_task(_run_generation, settings, cid, request, retry)
        return {"status": "queued", "attempt": attempt}

    @app.post("/api/conversations/{cid}/postprocess", dependencies=[Depends(require_auth)])
    async def postprocess_conversation(
        cid: str, payload: dict, background_tasks: BackgroundTasks
    ):
        if not settings.enable_seedream_edit:
            raise HTTPException(status_code=501, detail="Seedream edit is disabled.")
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None:
            raise HTTPException(status_code=404, detail="not found")
        if _is_read_only(meta):
            raise HTTPException(status_code=409, detail="read_only")
        try:
            options = await postprocess.start(settings, cid, payload, postprocess_locks)
        except postprocess.PostprocessError as e:
            raise HTTPException(status_code=e.status, detail=e.detail) from e
        background_tasks.add_task(
            postprocess.run_task, settings, cid, options, seedream_sem
        )
        return {"status": "running", "frames": []}

    web = Path(__file__).resolve().parent.parent / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app


app = create_app(get_settings())
