from __future__ import annotations

import asyncio
import copy
import errno
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
import uuid
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
from app.pipeline import SAME_AS_SOURCE_LANGUAGE, Pipeline
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
CLIENT_REQUEST_ID_MAX_CHARS = 64
TARGET_LANGUAGE_MAX_CHARS = 80
REPLACEMENT_INSTRUCTION_MAX_CHARS = 1000
DIALOGUE_BRIEF_MAX_CHARS = 1000
UPLOAD_CHUNK_BYTES = MIB
CREATION_REQUEST_OVERHEAD_BYTES = 4 * MIB
PROVIDER_ASSET_TTL_S = 24 * 60 * 60
REPLACEMENT_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
CREATION_FORM_FIELDS = frozenset({
    "file",
    "reference_url",
    "client_request_id",
    "generation_request",
    "replacement_image",
})
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
        "dialogue": {
            "modes": ["auto_rewrite", "generate"],
            "mode": "auto_rewrite",
            "translation": True,
            "max_brief_chars": DIALOGUE_BRIEF_MAX_CHARS,
            "optional": True,
            "default": {
                "mode": "auto_rewrite",
                "target_language": SAME_AS_SOURCE_LANGUAGE,
            },
        },
        "replacement": {
            "supported": True,
            "accept": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": REPLACEMENT_MAX_BYTES,
            "max_instruction_chars": REPLACEMENT_INSTRUCTION_MAX_CHARS,
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


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _utf16_code_units(value: str) -> int:
    return sum(2 if ord(character) > 0xFFFF else 1 for character in value)


def _normalize_client_request_id(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > CLIENT_REQUEST_ID_MAX_CHARS:
        raise _error(
            400,
            "invalid_client_request_id",
            f"client_request_id 必须是 1 至 {CLIENT_REQUEST_ID_MAX_CHARS} 个字符",
        )
    return normalized


async def _validate_creation_form(request: Request) -> None:
    counts: dict[str, int] = {}
    for field, _value in (await request.form()).multi_items():
        if field not in CREATION_FORM_FIELDS:
            raise _error(422, "invalid_creation_form", f"不支持的创建字段: {field}")
        counts[field] = counts.get(field, 0) + 1
        if counts[field] > 1:
            raise _error(422, "invalid_creation_form", f"创建字段不能重复: {field}")


def _parse_generation_request(
    raw: str | None,
    internal_fusion_agent: str,
) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise _error(
            422,
            "invalid_generation_request",
            "generation_request 必须是唯一 v1 JSON 对象",
        ) from None
    if not isinstance(value, dict) or not (
        {"version", "output", "processing", "replacement_guidance"}
        <= set(value)
        <= {
            "version", "output", "processing", "dialogue", "replacement_guidance",
            "adaptation_prompt",
        }
    ):
        raise _error(
            422,
            "invalid_generation_request",
            "generation_request 必须是唯一 v1 JSON 对象",
        )
    if type(value["version"]) is not int or value["version"] != 1:
        raise _error(422, "invalid_generation_request", "generation_request.version 必须为 1")

    output = value["output"]
    if (
        not isinstance(output, dict)
        or set(output) != {"aspect_ratio", "resolution", "fit_mode"}
        or output.get("aspect_ratio") not in {"16:9", "9:16"}
        or output.get("resolution") not in {"480p", "768p"}
        or output.get("fit_mode") != "auto"
    ):
        raise _error(422, "invalid_generation_request", "generation_request.output 无效")

    processing = value["processing"]
    if (
        not isinstance(processing, dict)
        or set(processing) != {
            "optimize_image", "remove_subtitle", "remove_logo",
        }
        or any(processing[key] is not True for key in processing)
    ):
        raise _error(422, "invalid_generation_request", "generation_request.processing 无效")

    # 兼容老客户端：dialogue 整体可缺省（默认自动改写为原视频语种），
    # 也可只给 target_language（mode 默认 auto_rewrite）；归一化后始终完整。
    dialogue = value.get("dialogue")
    if dialogue is None and "dialogue" not in value:
        dialogue = {"target_language": SAME_AS_SOURCE_LANGUAGE}
    if not isinstance(dialogue, dict):
        raise _error(422, "invalid_generation_request", "generation_request.dialogue 无效")
    dialogue_mode = dialogue.get("mode", "auto_rewrite")
    if dialogue_mode == "auto_rewrite":
        if set(dialogue) not in ({"target_language"}, {"mode", "target_language"}):
            raise _error(422, "invalid_generation_request", "generation_request.dialogue 无效")
    elif dialogue_mode == "generate":
        if set(dialogue) != {"mode", "target_language", "brief"}:
            raise _error(422, "invalid_generation_request", "generation_request.dialogue 无效")
    else:
        raise _error(422, "invalid_generation_request", "generation_request.dialogue 无效")
    target_language = dialogue.get("target_language")
    if not isinstance(target_language, str) or not target_language.strip():
        raise _error(422, "target_language_required", "target_language 不能为空")
    target_language = target_language.strip()
    if _utf16_code_units(target_language) > TARGET_LANGUAGE_MAX_CHARS:
        raise _error(422, "invalid_generation_request", "target_language 过长")
    brief = None
    if dialogue_mode == "generate":
        brief = dialogue.get("brief")
        if not isinstance(brief, str) or not brief.strip():
            raise _error(
                422,
                "invalid_generation_request",
                "generation_request.dialogue.brief 不能为空",
            )
        brief = brief.strip()
        if _utf16_code_units(brief) > DIALOGUE_BRIEF_MAX_CHARS:
            raise _error(
                422,
                "invalid_generation_request",
                "generation_request.dialogue.brief 过长",
            )

    guidance = value["replacement_guidance"]
    normalized_guidance = None
    if guidance is not None:
        if (
            not isinstance(guidance, dict)
            or set(guidance) != {"instruction", "image_field"}
            or guidance.get("image_field") != "replacement_image"
        ):
            raise _error(
                422,
                "invalid_generation_request",
                "replacement_guidance 无效",
            )
        instruction = guidance.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise _error(
                422,
                "invalid_generation_request",
                "replacement_guidance.instruction 不能为空",
            )
        instruction = instruction.strip()
        if _utf16_code_units(instruction) > REPLACEMENT_INSTRUCTION_MAX_CHARS:
            raise _error(
                422,
                "invalid_generation_request",
                "replacement_guidance.instruction 过长",
            )
        normalized_guidance = {
            "instruction": instruction,
            "image_field": "replacement_image",
        }

    adaptation_prompt = value.get("adaptation_prompt")
    if adaptation_prompt is not None:
        if not isinstance(adaptation_prompt, str):
            raise _error(
                422,
                "invalid_generation_request",
                "adaptation_prompt 无效",
            )
        adaptation_prompt = adaptation_prompt.strip()
        if not adaptation_prompt:
            adaptation_prompt = None
        elif _utf16_code_units(adaptation_prompt) > DIALOGUE_BRIEF_MAX_CHARS:
            raise _error(
                422,
                "invalid_generation_request",
                "adaptation_prompt 过长",
            )

    # Fusion is injected only after the complete public request has passed the
    # closed v1 validator; clients cannot select or shadow the internal agent.
    return {
        "version": 1,
        "output": dict(output),
        "processing": dict(processing),
        "dialogue": (
            {
                "mode": "generate",
                "target_language": target_language,
                "brief": brief,
            }
            if dialogue_mode == "generate"
            else {
                "mode": "auto_rewrite",
                "target_language": target_language,
            }
        ),
        "replacement_guidance": normalized_guidance,
        "adaptation_prompt": adaptation_prompt,
        "fusion": {"agent": internal_fusion_agent},
    }


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
            restarted = app.state.store.fail_incomplete_after_restart()
            for item in restarted:
                _auto_resume_after_restart(item)
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

    if configured.public_api_enabled:

        @app.middleware("http")
        async def public_api_request_context(request: Request, call_next):
            path = request.url.path
            if path != "/api/v1" and not path.startswith("/api/v1/"):
                return await call_next(request)
            request.state.public_request_id = f"req_{uuid.uuid4().hex}"
            if request.method == "POST" and path == "/api/v1/video-generations":
                raw_length = request.headers.get("content-length")
                if raw_length is not None:
                    try:
                        declared_length = int(raw_length)
                    except ValueError:
                        declared_length = -1

                    def public_error(status: int, code: str, message: str) -> JSONResponse:
                        return JSONResponse(
                            status_code=status,
                            content={
                                "error": {
                                    "code": code,
                                    "message": message,
                                    "request_id": request.state.public_request_id,
                                }
                            },
                            headers={
                                "Cache-Control": "private, no-store",
                                "X-Request-ID": request.state.public_request_id,
                            },
                        )

                    if declared_length < 0:
                        return public_error(400, "invalid_content_length", "Content-Length 无效")
                    request_limit = (
                        configured.max_upload_mb * MIB
                        + REPLACEMENT_MAX_BYTES
                        + CREATION_REQUEST_OVERHEAD_BYTES
                    )
                    if declared_length > request_limit:
                        return public_error(413, "request_too_large", "创建请求超过大小限制")
            response = await call_next(request)
            response.headers["Cache-Control"] = "private, no-store"
            response.headers["X-Request-ID"] = request.state.public_request_id
            return response

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
        size_bytes = owned_path.stat().st_size
        with owned_path.open("rb") as stream:
            asset_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
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
                "size_bytes": size_bytes,
                "sha256": asset_sha256,
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
            # ENOSPC also breaks the failure write itself; keep retrying on the
            # node backoff schedule so the project never stays "running" after
            # its worker died (fake-alive).
            persist_attempt = 0
            while True:
                try:
                    app.state.store.mark_failed(
                        project_id,
                        code="pipeline_failed",
                        message="后台任务执行失败",
                    )
                    break
                except OSError as persist_exc:
                    if persist_exc.errno != errno.ENOSPC:
                        LOGGER.exception("could not persist background failure for project %s", project_id)
                        break
                    persist_attempt += 1
                    delay = 0.0 if persist_attempt < 3 else min(60.0 * 2 ** (persist_attempt - 3), 1800.0)
                    if delay:
                        time.sleep(delay)
                except BaseException:
                    LOGGER.exception("could not persist background failure for project %s", project_id)
                    break
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

    def start_pipeline(project_id: str, operation: str = "run") -> None:
        task = asyncio.create_task(asyncio.to_thread(pipeline_entry, project_id, operation))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)

    def _auto_resume_after_restart(item: dict[str, Any]) -> None:
        """Requeue a project cut off by a process restart when a resume gate
        passes; otherwise keep it failed with manual-reconciliation guidance
        in the research log.  A restart must never be a dead end."""
        project_id = item.get("project_id")
        if not isinstance(project_id, str):
            return
        log = ResearchLog(app.state.store.path(project_id))
        try:
            with app.state.resume_lock:
                try:
                    state = app.state.store.read(project_id)
                except (FileNotFoundError, OSError, ProjectStateError):
                    return
                if state.get("status") != "failed":
                    return
                operation: str | None = None
                try:
                    app.state.pipeline.validate_r2v_resume(project_id)
                    operation = "resume_r2v"
                except (FileNotFoundError, TypeError, ValueError, OSError):
                    try:
                        app.state.pipeline.validate_pre_picsart_resume(project_id)
                        operation = "resume_pre_picsart"
                    except (FileNotFoundError, TypeError, ValueError, OSError):
                        operation = None
                if operation is None:
                    log.event(
                        "auto_resume_after_restart_blocked",
                        project_id=project_id,
                        downgraded_receipts=item.get("downgraded_receipts"),
                        guidance=(
                            "存在 submission_unknown 回执或缺少可复用前置产物；"
                            "先按 AGENTS.md 与供应商人工对账，确认未建任务/未扣费后"
                            "把对应回执 status 改为 rejected，再调 resume-picsart"
                        ),
                    )
                    return
                stage = (
                    "picsart_resume_queued"
                    if operation == "resume_r2v"
                    else "pre_picsart_resume_queued"
                )
                app.state.store.update(
                    project_id,
                    status="queued",
                    has_video=False,
                    project_progress={
                        "status": "queued",
                        "percent": 0,
                        "stage": stage,
                    },
                    generation={
                        "status": "queued",
                        "stage": stage,
                        "error": None,
                    },
                    error=None,
                )
            log.event(
                "auto_resumed_after_restart",
                project_id=project_id,
                operation=operation,
                downgraded_receipts=item.get("downgraded_receipts"),
            )
            start_pipeline(project_id, operation)
        except Exception:
            LOGGER.exception(
                "auto-resume after restart failed for project %s", project_id,
            )

    @asynccontextmanager
    async def staged_creation(
        *,
        file: UploadFile | None,
        reference_url: str | None,
        client_request_id: str | None,
        generation_request: str | None,
        replacement_image: UploadFile | None,
    ):
        """Validate and stage a creation request, yielding commit inputs.

        Shared by the internal /api/conversations endpoint and the public v1
        API so both enforce the exact same input rules and idempotency store.
        """

        normalized_client_request_id = _normalize_client_request_id(client_request_id)
        effective_request = _parse_generation_request(
            generation_request,
            configured.fusion_agent,
        )
        normalized_reference_url = reference_url.strip() if isinstance(reference_url, str) else ""
        if (file is not None) == bool(normalized_reference_url):
            raise _error(
                400,
                "source_exactly_one_required",
                "file 与 reference_url 必须且只能提供一个",
            )
        has_replacement_guidance = effective_request["replacement_guidance"] is not None
        if (replacement_image is not None) != has_replacement_guidance:
            raise _error(
                422,
                "replacement_pair_required",
                "replacement_image 与 replacement_guidance 必须同时提供",
            )

        source_limit = configured.max_upload_mb * MIB
        if file is not None:
            _declared_upload_size(file, source_limit, "源视频超过大小限制")
        replacement_media_type = None
        if replacement_image is not None:
            _declared_upload_size(replacement_image, REPLACEMENT_MAX_BYTES, "替换图片超过大小限制")
            replacement_media_type = (replacement_image.content_type or "").split(";", 1)[0].strip().lower()
            if replacement_media_type not in REPLACEMENT_MEDIA_TYPES:
                raise _error(415, "unsupported_replacement_image", "替换图片类型不受支持")

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
            source_identity = (
                {
                    "kind": "upload",
                    "sha256": source_sha256,
                    "bytes": source_bytes,
                }
                if file is not None
                else {"kind": "url", "url": normalized_reference_url}
            )

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
                "client_request_id": normalized_client_request_id,
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
            yield {
                "effective_request": effective_request,
                "client_request_id": normalized_client_request_id,
                "request_digest": request_digest,
                "input_receipt": input_receipt,
                "source_snapshot": source_snapshot,
                "staged_source": staged_source,
                "replacement_snapshot": replacement_snapshot,
                "staged_replacement": (
                    staged_replacement if replacement_image is not None else None
                ),
                "source_identity": source_identity,
                "replacement_identity": replacement_receipt,
            }

    @app.post("/api/conversations", status_code=201, dependencies=[Depends(require_auth)])
    async def create_conversation(
        request: Request,
        response: Response,
        file: UploadFile | None = File(default=None),
        reference_url: str | None = Form(default=None),
        client_request_id: str | None = Form(default=None),
        generation_request: str | None = Form(default=None),
        replacement_image: UploadFile | None = File(default=None),
    ):
        await _validate_creation_form(request)
        try:
            async with staged_creation(
                file=file,
                reference_url=reference_url,
                client_request_id=client_request_id,
                generation_request=generation_request,
                replacement_image=replacement_image,
            ) as staged:
                project_id, project, state, created = app.state.store.create_from_staging(
                    staged["effective_request"],
                    staged["client_request_id"],
                    request_digest=staged["request_digest"],
                    input_receipt=staged["input_receipt"],
                    source_snapshot=staged["source_snapshot"],
                    staged_source=staged["staged_source"],
                    replacement_snapshot=staged["replacement_snapshot"],
                    staged_replacement=staged["staged_replacement"],
                )
        except HTTPException:
            raise
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
            "generation_request": staged["effective_request"],
            "client_request_id": staged["client_request_id"],
            "reference_url": reference_url,
            "file": {"filename": file.filename, "headers": dict(file.headers)} if file else None,
            "replacement_image": {"filename": replacement_image.filename, "headers": dict(replacement_image.headers)} if replacement_image else None,
        })
        log.copy_media(attempt / "input-media", [source] + ([replacement_path] if replacement_image else []))
        log.write_json(attempt / "output.json", state)
        log.event("node_succeeded", node="http-create", scope="project", attempt=1)
        start_pipeline(project_id)
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
        try:
            if path.stat().st_size != record["size_bytes"]:
                raise HTTPException(404)
            with path.open("rb") as stream:
                current_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            raise HTTPException(404) from None
        if not hmac.compare_digest(current_sha256, record["sha256"]):
            raise HTTPException(404)
        return FileResponse(path, media_type=record["media_type"])

    if configured.public_api_enabled:
        from app import public_api

        def valid_generated_video(meta: dict[str, Any]) -> bool:
            project_id = meta.get("id")
            if (
                meta.get("status") != "succeeded"
                or meta.get("has_video") is not True
                or not isinstance(project_id, str)
            ):
                return False
            try:
                project = app.state.store.path(project_id)
            except (FileNotFoundError, OSError):
                return False
            return (project / "output" / "generated.mp4").is_file()

        def progress_for(meta: dict[str, Any], has_video: bool) -> int | None:
            if has_video:
                return 100
            progress = meta.get("project_progress")
            percent = progress.get("percent") if isinstance(progress, dict) else None
            if isinstance(percent, bool) or not isinstance(percent, int):
                return None
            return percent

        public_api.install(
            app,
            configured,
            store=app.state.store,
            stage_creation=staged_creation,
            start_pipeline=start_pipeline,
            valid_generated_video=valid_generated_video,
            progress_for=progress_for,
        )

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
