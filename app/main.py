import asyncio
import hmac
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import downloader, pipeline, seedance, storage
from app.auth import require_auth
from app.codex_runner import CodexRunner
from app.config import Settings, get_settings

_RATE_LIMIT = 10  # 每 IP 每分钟上传次数
_RATE_WINDOW_S = 60
# 前端幂等键（boot / 内容变更 / 上传成功时轮换，失败重试复用）；空 = 不参与幂等（兼容 curl）
_CLIENT_REQUEST_ID_RE = re.compile(r"^[0-9A-Za-z-]{8,64}$")


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
        voice_mode: str = Form("none"),
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
        if voice_mode not in ("none", "keep", "rewrite", "translate"):
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
            storage.probe_video(dest, settings.max_duration_s)
            if voice_mode != "none" and not storage.probe_audio(dest):
                raise storage.UploadError("no audio track in video")
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
        # 显式键：meta 落盘的提交标记（submitted_at/task_id 等）不外泄，冻结 14 字段契约
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
            "has_source": any(cdir.glob("source.*")),
            "has_video": (cdir / "generated.mp4").is_file(),
            "submit_enabled": settings.enable_seedance_submit,
        }

    @app.get("/api/conversations/{cid}/files/{name:path}", dependencies=[Depends(require_auth)])
    async def get_file(cid: str, name: str):
        path = storage.resolve_file(settings.data_dir, cid, name)
        if path is None:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.post("/api/conversations/{cid}/submit", dependencies=[Depends(require_auth)])
    async def submit_conversation(cid: str, payload: dict):
        try:
            return await seedance.submit(settings, cid, payload, submit_locks)
        except seedance.SubmitError as e:
            raise HTTPException(status_code=e.status, detail=e.detail) from e

    web = Path(__file__).resolve().parent.parent / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app


app = create_app(get_settings())
