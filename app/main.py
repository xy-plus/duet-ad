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
import re
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
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.pipeline import SAME_AS_SOURCE_LANGUAGE, Pipeline
from app import prompt_review as prompt_review_store
from app.prompt_review import PromptReviewError
from app.research_log import ResearchLog, display_duration_seconds
from app.safe_download import ReferenceDownloadError, Resolver, download_reference_url, resolve_addresses
from app.store import (
    ClientRequestIdConflict,
    IdempotencyIndexUnavailable,
    ProjectStateError,
    ProjectStore,
    creation_request_digest,
    replacement_receipt_digest,
    generation_request_sha256 as hash_generation_request,
)
from app.users import (
    COOKIE_NAME,
    SESSION_LIFETIME_S,
    ResolvedUser,
    User,
    UserAuth,
)


LOGGER = logging.getLogger(__name__)
MIB = 1024 * 1024
REPLACEMENT_MAX_BYTES = 10 * MIB
REPLACEMENT_MAX_IMAGES = 4
CLIENT_REQUEST_ID_MAX_CHARS = 64
TARGET_LANGUAGE_MAX_CHARS = 80
REPLACEMENT_INSTRUCTION_MAX_CHARS = 1000
DIALOGUE_BRIEF_MAX_CHARS = 1000
UPLOAD_CHUNK_BYTES = MIB
CREATION_REQUEST_OVERHEAD_BYTES = 4 * MIB
PROVIDER_ASSET_TTL_S = 24 * 60 * 60
TITLE_MAX_CHARS = 100
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
        "defaults": {"fit_mode": "auto", "optimize_image": True, "remove_subtitle": True, "remove_logo": True, "add_subtitles": False},
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
            "max_images": REPLACEMENT_MAX_IMAGES,
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


def _can_access_project(state: dict[str, Any], user: ResolvedUser) -> bool:
    """superuser 可见全部；普通用户只看到自己的项目。无 owner_id 的存量
    项目（迁移尚未运行，即用户库还没有 superuser）只对 superuser 可见。"""
    if user.is_superuser:
        return True
    owner = state.get("owner_id")
    return isinstance(owner, str) and owner == user.id


def _read_project(store: ProjectStore, project_id: str) -> dict[str, Any]:
    try:
        return store.read(project_id)
    except (FileNotFoundError, OSError):
        raise _error(404, "project_not_found", "项目不存在") from None
    except ProjectStateError:
        raise _error(500, "invalid_project_state", "项目状态不可读取") from None


def _owned_project(
    store: ProjectStore, project_id: str, user: ResolvedUser
) -> dict[str, Any]:
    """读项目并做归属鉴权；无权访问与不存在一样是 404（不泄漏存在性）。"""
    state = _read_project(store, project_id)
    if not _can_access_project(state, user):
        raise _error(404, "project_not_found", "项目不存在")
    return state


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
        if counts[field] > (REPLACEMENT_MAX_IMAGES if field == "replacement_image" else 1):
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
            "add_subtitles", "prompt_review",
        }
    ):
        raise _error(
            422,
            "invalid_generation_request",
            "generation_request 必须是唯一 v1 JSON 对象",
        )
    if type(value["version"]) is not int or value["version"] != 1:
        raise _error(422, "invalid_generation_request", "generation_request.version 必须为 1")
    add_subtitles = value.get("add_subtitles", False)
    if type(add_subtitles) is not bool:
        raise _error(422, "invalid_generation_request", "generation_request.add_subtitles 必须为布尔值")
    prompt_review = value.get("prompt_review", False)
    if type(prompt_review) is not bool:
        raise _error(422, "invalid_generation_request", "generation_request.prompt_review 必须为布尔值")

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
        "add_subtitles": add_subtitles,
        "prompt_review": prompt_review,
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
        # 用户系统先就绪：建表、播种管理员、迁移存量项目归属，再做重启恢复。
        await app.state.auth.setup()
        await app.state.auth.migrate_project_ownership(app.state.store)
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
            await app.state.auth.close()

    app = FastAPI(title="Duet H3 Max R2V Research", lifespan=lifespan)
    app.state.settings = configured
    app.state.auth = UserAuth(configured)
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

    auth: UserAuth = app.state.auth

    async def require_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        account: User | None = Depends(auth.optional_current_user),
    ) -> ResolvedUser:
        # 迁移桥：旧共享口令（老脚本、旧前端 localStorage 里的 token）映射为
        # 管理员身份，JWT 解码失败的 Bearer 也落到这条；口令不匹配再尝试
        # cookie / bearer JWT 双通道（fastapi-users）。
        if credentials is not None and hmac.compare_digest(
            credentials.credentials, configured.access_token
        ):
            return await auth.resolve_legacy_admin()
        if account is not None:
            return ResolvedUser(
                id=str(account.id),
                email=account.email,
                is_superuser=account.is_superuser,
            )
        raise _error(401, "unauthorized", "登录凭据无效或已过期")

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
                            "detail": {                                "code": "invalid_content_length",
                                "message": "Content-Length 无效",
                            }
                        },
                    )
                request_limit = (
                    configured.max_upload_mb * MIB
                    + REPLACEMENT_MAX_BYTES * REPLACEMENT_MAX_IMAGES
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
                        + REPLACEMENT_MAX_BYTES * REPLACEMENT_MAX_IMAGES
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
        creation = state.get("creation_input") or {}
        image_paths = state.get("replacement_image_paths", [state["replacement_image_path"]] if state.get("replacement_image_path") else [])
        snapshots = creation.get("replacement_images", [creation.get("replacement_image") or {}])
        for relative, snapshot in zip(image_paths, snapshots):
            replacement_path = app.state.store.path(project_id) / relative
            if owned_path == replacement_path.resolve(strict=True):
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
            if state["status"] not in {
                "succeeded", "failed", "awaiting_prompt_review",
            }:
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
        # 旧共享口令登录（兼容入口）：校验通过即可，前端随后用口令本身做
        # Bearer 凭据，由 require_user 映射为管理员身份。
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

    # 账号注册（fastapi-users 库路由）：POST /api/auth/register {email, password}
    app.include_router(auth.register_router, prefix="/api/auth")

    @app.post("/api/auth/login")
    async def account_login(
        request: Request,
        user_manager=Depends(auth.get_user_manager),
    ):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象")
        email = body.get("email")
        password = body.get("password")
        if (
            not isinstance(email, str)
            or not email.strip()
            or not isinstance(password, str)
            or not password
        ):
            raise _error(400, "invalid_credentials", "email 与 password 必须为非空字符串")
        user = await user_manager.authenticate(
            OAuth2PasswordRequestForm(
                username=email.strip(), password=password, scope=""
            )
        )
        if user is None or not user.is_active:
            raise _error(401, "unauthorized", "邮箱或密码不正确")
        token = await auth.get_jwt_strategy().write_token(user)
        # 双通道：写 HttpOnly cookie（浏览器）并在 body 返回 bearer token（脚本）。
        response = JSONResponse({
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "is_superuser": user.is_superuser,
            },
        })
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=SESSION_LIFETIME_S,
            path="/",
            httponly=True,
            samesite="lax",
            secure=False,
        )
        return response

    @app.post("/api/auth/logout")
    async def account_logout():
        # JWT 无状态：清 cookie 即登出浏览器通道；bearer token 由客户端丢弃。
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/api/auth/me")
    async def auth_me(user: ResolvedUser = Depends(require_user)):
        return {
            "id": user.id,
            "email": user.email,
            "is_superuser": user.is_superuser,
        }

    @app.get("/api/capabilities")
    async def capabilities(user: ResolvedUser = Depends(require_user)):
        return _capability()

    @app.get("/api/conversations")
    async def conversations(request: Request, user: ResolvedUser = Depends(require_user)):
        rows = [
            state
            for state in _store(request).list()
            if _can_access_project(state, user)
        ]
        # 收藏优先；组内保持 store.list 的 created_at 倒序（稳定排序）。
        rows.sort(key=lambda state: state.get("favorite") is not True)
        return rows

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
        replacement_images: list[UploadFile] | None = None,
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
        images = replacement_images if replacement_images is not None else ([replacement_image] if replacement_image is not None else [])
        if len(images) > REPLACEMENT_MAX_IMAGES:
            raise _error(422, "invalid_creation_form", "最多提供 4 张参考图")
        replacement_image = images[0] if images else None
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
        media_types = []
        for image in images:
            if not hasattr(image, "read") or not hasattr(image, "filename"):
                raise _error(422, "invalid_creation_form", "参考图必须为上传文件")
            _declared_upload_size(image, REPLACEMENT_MAX_BYTES, "替换图片超过大小限制")
            media_type = (image.content_type or "").split(";", 1)[0].strip().lower()
            if media_type not in REPLACEMENT_MEDIA_TYPES:
                raise _error(415, "unsupported_replacement_image", "替换图片类型不受支持")
            media_types.append(media_type)

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
            replacement_snapshots = []
            image_receipts = []
            staged_replacements = []
            for index, (image, media_type) in enumerate(zip(images, media_types), 1):
                image_path = staging / ("replacement-image" if index == 1 else f"replacement-image-{index}")
                replacement_bytes = await _write_upload(
                    image,
                    image_path,
                    REPLACEMENT_MAX_BYTES,
                )
                replacement_sha256 = _sha256_file(image_path)
                replacement_snapshots.append({
                    "filename": _safe_basename(
                        image.filename,
                        "replacement-image",
                    ),
                    "bytes": replacement_bytes,
                    "media_type": media_type,
                })
                image_receipts.append({
                    "sha256": replacement_sha256,
                    "bytes": replacement_bytes,
                })
                staged_replacements.append(image_path)
            replacement_snapshot = replacement_snapshots[0] if images else None
            replacement_receipt = image_receipts[0] if images else None

            generation_request_sha256 = hash_generation_request(effective_request)
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
            if len(images) > 1:
                input_receipt.update(version=2, replacement_images=image_receipts)
            request_digest = creation_request_digest(
                generation_request_sha256,
                source_sha256,
                replacement_receipt_digest(input_receipt),
            )
            yield {
                "effective_request": effective_request,
                "client_request_id": normalized_client_request_id,
                "request_digest": request_digest,
                "input_receipt": input_receipt,
                "source_snapshot": source_snapshot,
                "staged_source": staged_source,
                "replacement_snapshot": replacement_snapshot,
                "replacement_snapshots": replacement_snapshots,
                "staged_replacements": staged_replacements,
                "staged_replacement": (
                    staged_replacement if replacement_image is not None else None
                ),
                "source_identity": source_identity,
                "replacement_identity": image_receipts if len(images) > 1 else replacement_receipt,
            }

    @app.post("/api/conversations", status_code=201)
    async def create_conversation(
        request: Request,
        response: Response,
        file: UploadFile | None = File(default=None),
        reference_url: str | None = Form(default=None),
        client_request_id: str | None = Form(default=None),
        generation_request: str | None = Form(default=None),
        replacement_image: list[UploadFile] | None = File(default=None),
        user: ResolvedUser = Depends(require_user),
    ):
        await _validate_creation_form(request)
        replacement_images = (await request.form()).getlist("replacement_image")
        replacement_image = replacement_images[0] if replacement_images else None
        try:
            async with staged_creation(
                file=file,
                reference_url=reference_url,
                client_request_id=client_request_id,
                generation_request=generation_request,
                replacement_image=replacement_image,
                replacement_images=replacement_images,
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
                    replacement_snapshots=staged["replacement_snapshots"],
                    staged_replacements=staged["staged_replacements"],
                    extra_state={"owner_id": user.id},
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
            # 幂等命中：项目属于谁就只能由谁看到重放结果。
            if not _can_access_project(state, user):
                raise _error(404, "project_not_found", "项目不存在")
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
        log.copy_media(attempt / "input-media", [source] + [project / path for path in state["replacement_image_paths"]])
        log.write_json(attempt / "output.json", state)
        log.event("node_succeeded", node="http-create", scope="project", attempt=1)
        start_pipeline(project_id)
        return state

    @app.get("/api/conversations/{project_id}")
    async def conversation(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        state = _owned_project(_store(request), project_id, user)
        # 终态项目的"总耗时"用 trace 推导（前置节点 + 最后一次 H3 生成，
        # 审核暂停/等待不计入）；记录不全的老项目缺省该字段，前端回退墙钟。
        if state.get("status") in {"succeeded", "failed"}:
            display = display_duration_seconds(_store(request).path(project_id))
            if display is not None:
                state = {**state, "display_duration_s": round(display, 3)}
        return state

    async def _json_object_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象")
        return body

    @app.post("/api/conversations/{project_id}/rename")
    async def rename_conversation(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        body = await _json_object_body(request)
        title = body.get("title")
        if not isinstance(title, str):
            raise _error(422, "invalid_title", "title 必须为字符串")
        title = title.strip()
        if not title:
            raise _error(422, "invalid_title", "项目名称不能为空")
        if _utf16_code_units(title) > TITLE_MAX_CHARS:
            raise _error(
                422, "invalid_title", f"项目名称最长 {TITLE_MAX_CHARS} 个字符"
            )
        store = _store(request)
        _owned_project(store, project_id, user)
        return store.update(project_id, title=title)

    @app.post("/api/conversations/{project_id}/favorite")
    async def favorite_conversation(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        body = await _json_object_body(request)
        favorite = body.get("favorite")
        if type(favorite) is not bool:
            raise _error(422, "invalid_favorite", "favorite 必须为布尔值")
        store = _store(request)
        _owned_project(store, project_id, user)
        return store.update(project_id, favorite=favorite)

    @app.post(
        "/api/conversations/{project_id}/resume-pre-picsart",
        status_code=202,
    )
    async def resume_pre_picsart(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or body.get("confirm") is not True:
            raise _error(400, "confirmation_required", "必须明确确认续跑前置产物")
        store = _store(request)
        with app.state.resume_lock:
            state = _owned_project(store, project_id, user)
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
        "/api/conversations/{project_id}/resume-h3-prep",
        status_code=202,
    )
    async def resume_h3_prep(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or body.get("confirm") is not True:
            raise _error(400, "confirmation_required", "必须明确确认重新准备 H3")
        store = _store(request)
        with app.state.resume_lock:
            state = _owned_project(store, project_id, user)
            if state.get("status") not in {"failed", "succeeded"}:
                raise _error(
                    409,
                    "project_not_resumable",
                    "只有失败或已成功项目可以重新准备 H3",
                )
            try:
                app.state.pipeline.validate_h3_prep_resume(project_id)
            except (FileNotFoundError, TypeError, ValueError, OSError):
                raise _error(
                    409,
                    "h3_prep_resume_not_ready",
                    "项目缺少可安全复用的前置产物",
                ) from None
            if state.get("status") == "succeeded":
                ResearchLog(store.path(project_id)).event(
                    "h3_prep_output_superseded",
                    previous_output=state.get("output_path"),
                    reason="explicit succeeded-project H3 preparation rerun",
                )
            state = store.update(
                project_id,
                status="queued",
                has_video=False,
                project_progress={
                    "status": "queued",
                    "percent": 0,
                    "stage": "h3_prep_resume_queued",
                },
                generation={
                    "status": "queued",
                    "stage": "h3_prep_resume_queued",
                    "error": None,
                },
                error=None,
            )
        task = asyncio.create_task(
            asyncio.to_thread(pipeline_entry, project_id, "resume_h3_prep")
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return state

    @app.post(
        "/api/conversations/{project_id}/resume-picsart",
        status_code=202,
    )
    async def resume_picsart(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or body.get("confirm") is not True:
            raise _error(400, "confirmation_required", "必须明确确认续跑 Picsart")
        store = _store(request)
        with app.state.resume_lock:
            state = _owned_project(store, project_id, user)
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

    def apply_prompt_review_confirmation(
        project_id: str, segments: Any,
    ) -> dict[str, Any]:
        """确认审核后的提示词：重建冻结提示词并把项目排回 Picsart 提交队列。

        内外两个 API 共用；抛 PromptReviewError（code 原样映射到各 API 的
        错误码）。确认成功后由调用方调度 resume_r2v（须在事件循环上）。
        """
        with app.state.resume_lock:
            app.state.pipeline.confirm_prompt_review(project_id, segments)
            return app.state.store.update(
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

    @app.get(
        "/api/conversations/{project_id}/prompt-review",
    )
    async def prompt_review_read(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        store = _store(request)
        state = _owned_project(store, project_id, user)
        segments = prompt_review_store.display_segments(
            store.path(project_id), state,
        )
        return {"status": state.get("status"), "segments": segments}

    @app.post(
        "/api/conversations/{project_id}/prompt-review/confirm",
        status_code=202,
    )
    async def prompt_review_confirm(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or set(body) != {"segments"}:
            raise _error(
                400,
                "invalid_prompt_review_edit",
                '请求体必须是 {"segments":[{"segment_index":N,"prompt":"..."}]}',
            )
        _owned_project(_store(request), project_id, user)
        try:
            state = apply_prompt_review_confirmation(
                project_id, body["segments"],
            )
        except PromptReviewError as exc:
            status = 409 if exc.code in {
                "prompt_review_not_pending",
                "prompt_review_unavailable",
                "prompt_review_stale",
            } else 422
            raise _error(status, exc.code, exc.message) from None
        except (FileNotFoundError, OSError):
            raise _error(404, "conversation_not_found", "会话不存在") from None
        task = asyncio.create_task(
            asyncio.to_thread(pipeline_entry, project_id, "resume_r2v")
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return state

    @app.get(
        "/api/conversations/{project_id}/generation-history",
    )
    async def generation_history_read(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        store = _store(request)
        state = _owned_project(store, project_id, user)
        return {
            "versions": prompt_review_store.generation_history(
                store.path(project_id), state,
            ),
        }

    @app.post(
        "/api/conversations/{project_id}/regenerate",
        status_code=202,
    )
    async def regenerate_with_edited_prompts(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _error(400, "invalid_json", "请求体必须是 JSON 对象") from None
        if not isinstance(body, dict) or set(body) != {"segments"}:
            raise _error(
                400,
                "invalid_prompt_review_edit",
                '请求体必须是 {"segments":[{"segment_index":N,"prompt":"..."}]}',
            )
        try:
            with app.state.resume_lock:
                state = _owned_project(_store(request), project_id, user)
                if state.get("status") != "succeeded":
                    raise _error(
                        409,
                        "regenerate_not_succeeded",
                        "只有已成功项目可以编辑提示词重新生成",
                    )
                prepared = app.state.pipeline.prepare_regenerate_prompts(
                    project_id, body["segments"],
                )
                # 状态迁移原子且先于文件操作：归档窗口内任何轮询都不得看到
                # succeeded + has_video=true（旧视频）的组合。
                state = _store(request).update(
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
                try:
                    app.state.pipeline.execute_regenerate(project_id, prepared)
                except Exception:
                    LOGGER.exception(
                        "regenerate archive/apply failed for project %s",
                        project_id,
                    )
                    _store(request).mark_failed(
                        project_id,
                        code="regenerate_failed",
                        message="重新生成的归档/重写失败，请重试；已归档的历史版本不受影响",
                        stage="regenerate_failed",
                    )
                    raise _error(
                        500, "regenerate_failed", "重新生成准备失败，请重试"
                    ) from None
        except PromptReviewError as exc:
            status = 409 if exc.code in {
                "regenerate_not_succeeded",
                "prompt_review_unavailable",
                "prompt_review_stale",
                "regenerate_unavailable",
            } else 422
            raise _error(status, exc.code, exc.message) from None
        except (FileNotFoundError, OSError):
            raise _error(404, "conversation_not_found", "会话不存在") from None
        task = asyncio.create_task(
            asyncio.to_thread(pipeline_entry, project_id, "resume_r2v")
        )
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return state

    @app.get("/api/conversations/{project_id}/creation-input/replacement-image")
    async def replacement(
        project_id: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        store = _store(request)
        state = _owned_project(store, project_id, user)
        if state.get("replacement_image_path") != "inputs/replacement-image":
            raise _error(404, "file_not_found", "替换图片不存在")
        path = _project_file(store, project_id, "inputs/replacement-image")
        snapshot = (state.get("creation_input") or {}).get("replacement_image") or {}
        media_type = snapshot.get("media_type") if snapshot.get("media_type") in REPLACEMENT_MEDIA_TYPES else "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.get("/api/conversations/{project_id}/creation-input/replacement-images/{index}")
    async def replacement_at(
        project_id: str,
        index: int,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        store = _store(request)
        state = _owned_project(store, project_id, user)
        paths = state.get("replacement_image_paths", [state["replacement_image_path"]] if state.get("replacement_image_path") else [])
        snapshots = (state.get("creation_input") or {}).get("replacement_images", [(state.get("creation_input") or {}).get("replacement_image") or {}])
        if not 1 <= index <= len(paths) or index > len(snapshots):
            raise _error(404, "file_not_found", "参考图不存在")
        expected = "inputs/replacement-image" if index == 1 else f"inputs/replacement-image-{index}"
        if paths[index - 1] != expected:
            raise _error(404, "file_not_found", "参考图不存在")
        path = _project_file(store, project_id, expected)
        media_type = snapshots[index - 1].get("media_type")
        return FileResponse(path, media_type=media_type if media_type in REPLACEMENT_MEDIA_TYPES else "application/octet-stream")

    @app.get("/api/conversations/{project_id}/files/{relative:path}")
    async def project_file(
        project_id: str,
        relative: str,
        request: Request,
        user: ResolvedUser = Depends(require_user),
    ):
        target = PUBLIC_PROJECT_FILES.get(relative)
        store = _store(request)
        state = _owned_project(store, project_id, user)
        if target is None:
            # 历史版本成片：output/history/v{N}/generated.mp4，版本号必须在
            # generation_history 里有正式记录才放行（不开放任意路径）。
            match = re.fullmatch(r"history/v([1-9][0-9]*)/generated\.mp4", relative)
            archived_versions = {
                entry.get("version")
                for entry in (state.get("generation_history") or [])
                if isinstance(entry, dict)
            } if isinstance(state.get("generation_history"), list) else set()
            if match is None or int(match.group(1)) not in archived_versions:
                raise _error(404, "file_not_found", "文件不存在")
            target = (f"output/history/v{int(match.group(1))}/generated.mp4", "video/mp4")
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
            apply_prompt_review_confirmation=apply_prompt_review_confirmation,
        )

    web = Path(__file__).resolve().parents[1] / "web"

    @app.middleware("http")
    async def disable_static_browser_cache(request: Request, call_next):
        # Static assets are served straight from the working tree, so a stale
        # browser copy can pair a new frontend with an old backend (or the
        # reverse after a downgrade). Web responses must revalidate on every
        # load (no-cache): StaticFiles already sends ETag/Last-Modified, so a
        # fresh deploy answers 304 cheaply while a changed file is picked up
        # immediately. /api 与 /internal 保持原语义（后者仍是 no-store）。
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api"):
            return response
        if path.startswith("/internal"):
            response.headers["Cache-Control"] = "private, no-store"
        else:
            response.headers["Cache-Control"] = "private, no-cache"
        return response

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
