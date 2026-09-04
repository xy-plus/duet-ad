from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.pipeline import Pipeline
from app.research_log import ResearchLog
from app.safe_download import ReferenceDownloadError, Resolver, download_reference_url, resolve_addresses
from app.store import (
    ClientRequestIdConflict,
    IdempotencyIndexUnavailable,
    ProjectStateError,
    ProjectStore,
    canonical_json_sha256,
    creation_request_digest,
)


LOGGER = logging.getLogger(__name__)
MIB = 1024 * 1024
REPLACEMENT_MAX_BYTES = 10 * MIB
UPLOAD_CHUNK_BYTES = MIB
CREATION_REQUEST_OVERHEAD_BYTES = 4 * MIB
PROVIDER_ASSET_TTL_S = 24 * 60 * 60
REPLACEMENT_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
PROVIDER_ASSET_SUFFIXES = {
    "video/mp4": ".mp4",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "audio/wav": ".wav",
}
PUBLIC_PROJECT_FILES = {
    "source.mp4": ("inputs/source.mp4", "video/mp4"),
    "generated.mp4": ("output/generated.mp4", "video/mp4"),
}

CAPABILITY = {
    "minimal_creation": {
        "supported": True,
        "version": 1,
        "endpoint": "/api/conversations",
        "encoding": "multipart/form-data",
        "request_field": "generation_request",
        "replacement_image_field": "replacement_image",
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": ["480p", "768p"],
        "defaults": {"fit_mode": "auto", "optimize_image": True, "remove_subtitle": True, "remove_logo": True},
        "dialogue": {"mode": "auto_rewrite", "translation": True},
        "replacement": {
            "supported": True,
            "accept": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": REPLACEMENT_MAX_BYTES,
            "max_instruction_chars": 1000,
        },
    }
}

bearer = HTTPBearer(auto_error=False)


class UploadTooLarge(RuntimeError):
    pass


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> ProjectStore:
    return request.app.state.store


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code, detail={"code": code, "message": message})


async def require_auth(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    expected = _settings(request).access_token
    if credentials is None or not hmac.compare_digest(credentials.credentials, expected):
        raise _error(401, "unauthorized", "登录凭据无效或已过期")


def _read_project(store: ProjectStore, project_id: str) -> dict[str, Any]:
    try:
        return store.read(project_id)
    except (FileNotFoundError, OSError):
        raise _error(404, "project_not_found", "项目不存在") from None
    except ProjectStateError:
        raise _error(500, "invalid_project_state", "项目状态不可读取") from None


def _project_file(store: ProjectStore, project_id: str, relative: str) -> Path:
    try:
        return store.resolve_project_file(project_id, store.path(project_id) / relative)
    except (FileNotFoundError, OSError):
        raise _error(404, "file_not_found", "文件不存在") from None


def _safe_basename(filename: str | None, fallback: str) -> str:
    raw = filename or fallback
    basename = raw.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(character for character in basename if character.isprintable() and character not in {"/", "\\"}).strip()
    basename = basename[:255]
    return fallback if not basename or basename in {".", ".."} else basename


def _declared_upload_size(upload: UploadFile, max_bytes: int, message: str) -> None:
    size = upload.size
    if isinstance(size, int) and not isinstance(size, bool) and size > max_bytes:
        raise _error(413, "upload_too_large", message)


async def _write_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    total = 0
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            while True:
                chunk = await upload.read(min(UPLOAD_CHUNK_BYTES, max_bytes - total + 1))
                if not chunk:
                    break
                if total + len(chunk) > max_bytes:
                    raise UploadTooLarge
                handle.write(chunk)
                total += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        return total
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_generation_request(
    raw: str, internal_fusion_agent: str = "deepseek",
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise _error(400, "invalid_generation_request", "generation_request 必须是 JSON 对象") from None
    if not isinstance(value, dict):
        raise _error(400, "invalid_generation_request", "generation_request 必须是 JSON 对象")
    # The Fusion implementation is an internal deployment choice.  Never let
    # a client-supplied field select a provider; freeze the server choice into
    # the project so retries and resume remain deterministic.
    return {**value, "fusion": {"agent": internal_fusion_agent}}


def _capability() -> dict[str, Any]:
    return copy.deepcopy(CAPABILITY)


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_factory=None,
    reference_transport: httpx.AsyncBaseTransport | None = None,
    reference_resolver: Resolver = resolve_addresses,
) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.store.fail_incomplete_after_restart()
            yield
        finally:
            tasks = tuple(app.state.tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            with app.state.pipeline_workers_lock:
                workers_active = app.state.pipeline_workers
            # asyncio cancellation cannot stop an already-running to_thread
            # call.  Keep the process lock until process exit whenever a task
            # was still registered at shutdown.
            if not tasks and workers_active == 0:
                app.state.store.close()

    app = FastAPI(title="Duet H3 Max R2V Research", lifespan=lifespan)
    app.state.settings = configured
    app.state.store = ProjectStore(
        configured.data_dir / "projects",
        acquire_process_lock=configured.store_lock_enabled,
    )
    app.state.tasks = set()
    app.state.assets: dict[str, dict[str, Any]] = {}
    app.state.assets_lock = threading.Lock()
    app.state.resume_lock = threading.Lock()
    app.state.pipeline_workers = 0
    app.state.pipeline_workers_lock = threading.Lock()

    @app.middleware("http")
    async def reject_oversized_declared_creation_request(
        request: Request, call_next,
    ):
        if request.method == "POST" and request.url.path == "/api/conversations":
            raw_length = request.headers.get("content-length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except ValueError:
                    declared_length = -1
                if declared_length < 0:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "detail": {
                                "code": "invalid_content_length",
                                "message": "Content-Length 无效",
                            }
                        },
                    )
                request_limit = (
                    configured.max_upload_mb * MIB
                    + REPLACEMENT_MAX_BYTES
                    + CREATION_REQUEST_OVERHEAD_BYTES
                )
                if declared_length > request_limit:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": {
                                "code": "request_too_large",
                                "message": "创建请求超过大小限制",
                            }
                        },
                    )
        return await call_next(request)

    def register_asset(project_id: str, path: Path) -> str:
        state = _read_project(app.state.store, project_id)
        try:
            owned_path = app.state.store.resolve_project_file(project_id, path)
        except (FileNotFoundError, OSError):
            raise ValueError("provider asset must be a real file inside its project") from None
        media_type = mimetypes.guess_type(owned_path.name)[0]
        if owned_path.suffix.lower() == ".wav":
            media_type = "audio/wav"
        if state.get("replacement_image_path") == "inputs/replacement-image":
            replacement_path = app.state.store.path(project_id) / "inputs" / "replacement-image"
            if owned_path == replacement_path.resolve(strict=True):
                snapshot = (state.get("creation_input") or {}).get("replacement_image") or {}
                declared_type = snapshot.get("media_type")
                if declared_type in REPLACEMENT_MEDIA_TYPES:
                    media_type = declared_type
        now = time.time()
        asset_id = secrets.token_urlsafe(18)
        asset_name = asset_id + PROVIDER_ASSET_SUFFIXES.get(media_type or "", "")
        token = secrets.token_urlsafe(32)
        with app.state.assets_lock:
            expired = [key for key, record in app.state.assets.items() if record["expires_at"] < now]
            for key in expired:
                app.state.assets.pop(key, None)
            while asset_name in app.state.assets:
                asset_id = secrets.token_urlsafe(18)
                asset_name = asset_id + PROVIDER_ASSET_SUFFIXES.get(media_type or "", "")
            app.state.assets[asset_name] = {
                "project_id": project_id,
                "asset_id": asset_name,
                "token": token,
                "path": owned_path,
                "media_type": media_type or "application/octet-stream",
                "expires_at": now + PROVIDER_ASSET_TTL_S,
            }
        return (
            f"{configured.public_base_url}/internal/provider-assets/"
            f"{quote(project_id, safe='')}/{quote(asset_name, safe='')}?token={quote(token, safe='')}"
        )

    app.state.register_asset = register_asset
    try:
        app.state.pipeline = (
            pipeline_factory(configured, app.state.store, register_asset)
            if pipeline_factory
            else Pipeline(configured, app.state.store, register_asset)
        )
    except BaseException:
        app.state.store.close()
        raise

    def pipeline_entry(project_id: str, operation: str = "run") -> None:
        with app.state.pipeline_workers_lock:
            app.state.pipeline_workers += 1
        try:
            getattr(app.state.pipeline, operation)(project_id)
            state = app.state.store.read(project_id)
            if state["status"] not in {"succeeded", "failed"}:
                app.state.store.mark_failed(
                    project_id,
                    code="pipeline_incomplete",
                    message="后台任务结束但未产生终态",
                )
        except BaseException as exc:
            LOGGER.exception("unhandled background pipeline failure for project %s", project_id)
            try:
                app.state.store.mark_failed(
                    project_id,
                    code="pipeline_failed",
                    message="后台任务执行失败",
                )
            except BaseException:
                LOGGER.exception("could not persist background failure for project %s", project_id)
        finally:
            with app.state.pipeline_workers_lock:
                app.state.pipeline_workers -= 1

    @app.post("/api/login")
    async def login(request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象")
        token = body.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token, configured.access_token):
            raise _error(401, "unauthorized", "口令不正确")
        return {"ok": True}

    @app.get("/api/capabilities", dependencies=[Depends(require_auth)])
    async def capabilities():
        return _capability()

    @app.get("/api/conversations", dependencies=[Depends(require_auth)])
    async def conversations(request: Request):
        return _store(request).list()

    @app.post("/api/conversations", status_code=201, dependencies=[Depends(require_auth)])
    async def create_conversation(
        request: Request,
        response: Response,
        file: UploadFile | None = File(default=None),
        reference_url: str | None = Form(default=None),
        client_request_id: str = Form(default=""),
        generation_request: str = Form(default="{}"),
        replacement_image: UploadFile | None = File(default=None),
    ):
        effective_request = _parse_generation_request(
            generation_request, configured.fusion_agent,
        )
        normalized_reference_url = reference_url.strip() if isinstance(reference_url, str) else ""
        if file is None and not normalized_reference_url:
            raise _error(400, "source_missing", "没有提供视频")

        source_limit = configured.max_upload_mb * MIB
        if file is not None:
            _declared_upload_size(file, source_limit, "源视频超过大小限制")
        replacement_media_type = None
        if replacement_image is not None:
            _declared_upload_size(replacement_image, REPLACEMENT_MAX_BYTES, "替换图片超过大小限制")
            replacement_media_type = (replacement_image.content_type or "").split(";", 1)[0].strip().lower()
            if replacement_media_type not in REPLACEMENT_MEDIA_TYPES:
                raise _error(415, "unsupported_replacement_image", "替换图片类型不受支持")

        try:
            with tempfile.TemporaryDirectory(
                prefix=".create-staging-",
                dir=app.state.store.root,
            ) as staging_name:
                staging = Path(staging_name)
                os.chmod(staging, 0o700)
                staged_source = staging / "source.mp4"
                staged_replacement = staging / "replacement-image"
                if file is not None:
                    source_bytes = await _write_upload(file, staged_source, source_limit)
                    source_snapshot = {
                        "mode": "upload",
                        "filename": _safe_basename(file.filename, "source.mp4"),
                        "bytes": source_bytes,
                    }
                else:
                    source_bytes = await download_reference_url(
                        normalized_reference_url,
                        staged_source,
                        max_bytes=source_limit,
                        timeout_s=configured.request_timeout_s,
                        resolver=reference_resolver,
                        transport=reference_transport,
                    )
                    source_snapshot = {
                        "mode": "link",
                        "reference_url": normalized_reference_url,
                    }
                source_sha256 = _sha256_file(staged_source)

                replacement_snapshot = None
                replacement_receipt = None
                if replacement_image is not None:
                    replacement_bytes = await _write_upload(
                        replacement_image,
                        staged_replacement,
                        REPLACEMENT_MAX_BYTES,
                    )
                    replacement_sha256 = _sha256_file(staged_replacement)
                    replacement_snapshot = {
                        "filename": _safe_basename(
                            replacement_image.filename,
                            "replacement-image",
                        ),
                        "bytes": replacement_bytes,
                        "media_type": replacement_media_type,
                    }
                    replacement_receipt = {
                        "sha256": replacement_sha256,
                        "bytes": replacement_bytes,
                    }

                generation_request_sha256 = canonical_json_sha256(effective_request)
                input_receipt = {
                    "version": 1,
                    "client_request_id": client_request_id,
                    "generation_request_sha256": generation_request_sha256,
                    "source": {
                        "sha256": source_sha256,
                        "bytes": source_bytes,
                    },
                    "replacement_image": replacement_receipt,
                }
                request_digest = creation_request_digest(
                    generation_request_sha256,
                    source_sha256,
                    (
                        replacement_receipt["sha256"]
                        if replacement_receipt is not None
                        else None
                    ),
                )
                project_id, project, state, created = app.state.store.create_from_staging(
                    effective_request,
                    client_request_id,
                    request_digest=request_digest,
                    input_receipt=input_receipt,
                    source_snapshot=source_snapshot,
                    staged_source=staged_source,
                    replacement_snapshot=replacement_snapshot,
                    staged_replacement=(
                        staged_replacement if replacement_image is not None else None
                    ),
                )
        except UploadTooLarge:
            raise _error(413, "upload_too_large", "上传文件超过大小限制") from None
        except ReferenceDownloadError as exc:
            raise _error(exc.status_code, exc.code, exc.message) from None
        except ClientRequestIdConflict:
            raise _error(
                409,
                "client_request_id_conflict",
                "client_request_id 已绑定到不同的创建输入",
            ) from None
        except IdempotencyIndexUnavailable:
            LOGGER.exception("durable client request index is unavailable")
            raise _error(
                503,
                "idempotency_index_unavailable",
                "创建请求幂等记录不可用，已停止创建以避免重复项目",
            ) from None
        except OSError:
            raise _error(500, "input_write_failed", "无法保存上传文件") from None
        except Exception:
            LOGGER.exception("could not commit creation input")
            raise _error(500, "input_read_failed", "无法读取上传文件") from None

        response.status_code = 201 if created else 200
        if not created:
            return state

        log = ResearchLog(project)
        input_dir = project / "inputs"
        source = input_dir / "source.mp4"
        replacement_path = input_dir / "replacement-image"
        attempt = log.attempt_dir("http-create", "project", 1)
        log.write_json(attempt / "input.json", {
            "headers": dict(request.headers),
            "generation_request_raw": generation_request,
            "generation_request": effective_request,
            "client_request_id": client_request_id,
            "reference_url": reference_url,
            "file": {"filename": file.filename, "headers": dict(file.headers)} if file else None,
            "replacement_image": {"filename": replacement_image.filename, "headers": dict(replacement_image.headers)} if replacement_image else None,
        })
        log.copy_media(attempt / "input-media", [source] + ([replacement_path] if replacement_image else []))
        log.write_json(attempt / "output.json", state)
        log.event("node_succeeded", node="http-create", scope="project", attempt=1)
        task = asyncio.create_task(asyncio.to_thread(pipeline_entry, project_id))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return state

    @app.get("/api/conversations/{project_id}", dependencies=[Depends(require_auth)])
    async def conversation(project_id: str, request: Request):
        return _read_project(_store(request), project_id)

    @app.post(
        "/api/conversations/{project_id}/resume-pre-picsart",
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    async def resume_pre_picsart(project_id: str, request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or body.get("confirm") is not True:
            raise _error(400, "confirmation_required", "必须明确确认续跑前置产物")
        store = _store(request)
        with app.state.resume_lock:
            state = _read_project(store, project_id)
            if state.get("status") != "failed":
                raise _error(409, "project_not_failed", "只有失败项目可以续跑前置产物")
            try:
                app.state.pipeline.validate_pre_picsart_resume(project_id)
            except (FileNotFoundError, TypeError, ValueError, OSError):
                raise _error(
                    409,
                    "pre_picsart_resume_not_ready",
                    "项目缺少可安全复用的前置产物，或已存在 Picsart 提交痕迹",
                ) from None
            state = store.update(
                project_id,
                status="queued",
                has_video=False,
                project_progress={
                    "status": "queued",
                    "percent": 0,
                    "stage": "pre_picsart_resume_queued",
                },
                generation={
                    "status": "queued",
                    "stage": "pre_picsart_resume_queued",
                    "error": None,
                },
                error=None,
            )
        task = asyncio.create_task(
            asyncio.to_thread(pipeline_entry, project_id, "resume_pre_picsart")
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return state

    @app.post(
        "/api/conversations/{project_id}/resume-picsart",
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    async def resume_picsart(project_id: str, request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or body.get("confirm") is not True:
            raise _error(400, "confirmation_required", "必须明确确认续跑 Picsart")
        store = _store(request)
        with app.state.resume_lock:
            state = _read_project(store, project_id)
            if state.get("status") != "failed":
                raise _error(409, "project_not_failed", "只有失败项目可以续跑 Picsart")
            try:
                app.state.pipeline.validate_r2v_resume(project_id)
            except (FileNotFoundError, TypeError, ValueError, OSError):
                raise _error(
                    409,
                    "picsart_resume_not_ready",
                    "项目缺少可复用的 Picsart 前置产物",
                ) from None
            state = store.update(
                project_id,
                status="queued",
                has_video=False,
                project_progress={
                    "status": "queued",
                    "percent": 0,
                    "stage": "picsart_resume_queued",
                },
                generation={
                    "status": "queued",
                    "stage": "picsart_resume_queued",
                    "error": None,
                },
                error=None,
            )
        task = asyncio.create_task(
            asyncio.to_thread(pipeline_entry, project_id, "resume_r2v")
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return state

    @app.get("/api/conversations/{project_id}/creation-input/replacement-image", dependencies=[Depends(require_auth)])
    async def replacement(project_id: str, request: Request):
        store = _store(request)
        state = _read_project(store, project_id)
        if state.get("replacement_image_path") != "inputs/replacement-image":
            raise _error(404, "file_not_found", "替换图片不存在")
        path = _project_file(store, project_id, "inputs/replacement-image")
        snapshot = (state.get("creation_input") or {}).get("replacement_image") or {}
        media_type = snapshot.get("media_type") if snapshot.get("media_type") in REPLACEMENT_MEDIA_TYPES else "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.get("/api/conversations/{project_id}/files/{relative:path}", dependencies=[Depends(require_auth)])
    async def project_file(project_id: str, relative: str, request: Request):
        target = PUBLIC_PROJECT_FILES.get(relative)
        if target is None:
            raise _error(404, "file_not_found", "文件不存在")
        store = _store(request)
        state = _read_project(store, project_id)
        if relative == "source.mp4" and state.get("has_source") is not True:
            raise _error(404, "file_not_found", "文件不存在")
        if relative == "generated.mp4" and (
            state.get("status") != "succeeded" or state.get("has_video") is not True
        ):
            raise _error(404, "file_not_found", "文件不存在")
        path = _project_file(store, project_id, target[0])
        return FileResponse(path, media_type=target[1])

    @app.api_route(
        "/internal/provider-assets/{project_id}/{asset_id}",
        methods=["GET", "HEAD"],
    )
    async def provider_asset(project_id: str, asset_id: str, token: str):
        now = time.time()
        with app.state.assets_lock:
            record = app.state.assets.get(asset_id)
            if record is not None and record["expires_at"] < now:
                app.state.assets.pop(asset_id, None)
                record = None
        if (
            record is None
            or record["project_id"] != project_id
            or record["asset_id"] != asset_id
            or not hmac.compare_digest(token, record["token"])
        ):
            raise HTTPException(404)
        try:
            path = app.state.store.resolve_project_file(project_id, record["path"])
        except (FileNotFoundError, OSError):
            raise HTTPException(404) from None
        return FileResponse(path, media_type=record["media_type"])

    web = Path(__file__).resolve().parents[1] / "web"
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def _configuration_error_app(message: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        raise RuntimeError(message)
        yield

    return FastAPI(title="Duet H3 Max R2V Research (misconfigured)", lifespan=lifespan)


try:
    app = create_app()
except (RuntimeError, ValueError) as configuration_error:
    app = _configuration_error_app(str(configuration_error))
