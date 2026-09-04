"""Versioned server-to-server API facade for asynchronous video generation."""

from __future__ import annotations

import hashlib
import fcntl
import ipaddress
import json
import logging
import os
import re
import stat
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Request,
    Response,
    Security,
    UploadFile,
)
from fastapi.openapi.utils import get_openapi
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Match

from app import minimal_creation, public_api_auth, public_artifacts, public_credits, storage
from app.config import Settings


_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_JOB_RE = re.compile(r"^vg_([0-9a-f]{32})$")
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}
_CREATE_FIELDS = {
    "source_video",
    "source_video_url",
    "aspect_ratio",
    "resolution",
    "target_language",
    "replacement_image",
    "replacement_instruction",
}
_UNKNOWN_MARKERS = frozenset({"submission_unknown", "h3_submitting"})
_STRICT_LIMIT_RE = re.compile(r"^(?:[1-9]|[1-9][0-9]|100)$")
_CREATE_ERROR_MESSAGES = {
    "idempotency_key_reused": "Idempotency-Key 已绑定到不同输入",
    "invalid_create_request": "创建请求无效",
    "invalid_generation_request": "生成参数无效",
    "invalid_replacement_image": "参考图无法读取，请更换图片",
    "invalid_source_media": "原视频无法读取，请更换视频",
    "invalid_source_video_url": "视频链接不是可访问的公网 HTTPS 地址",
    "queue_full": "当前任务队列已满，请稍后重试",
    "replacement_image_too_large": "参考图超过大小限制",
    "replacement_pair_required": "参考图和替换说明必须同时提供",
    "source_exactly_one_required": "必须且只能提供一个视频来源",
    "source_too_large": "原视频超过大小限制",
    "unsupported_replacement_media_type": "参考图格式不受支持",
    "video_duration_exceeds_h3_limit": "原视频时长超过限制，请裁剪后重试",
}
_LOGGER = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class Progress(BaseModel):
    percent: int | None = Field(default=None, ge=0, le=100)


class Parameters(BaseModel):
    aspect_ratio: Literal["9:16", "16:9"]
    resolution: Literal["768p", "480p"]
    target_language: str | None
    replacement_image: bool


class Billing(BaseModel):
    currency: Literal["CNY"] = "CNY"
    credits_per_cny: Literal[100] = 100
    quoted_credits: Literal[1000] = 1000
    quoted_amount_minor: Literal[1000] = 1000
    price_version: Literal["credits-fixed-1000-v1"] = public_credits.PRICE_VERSION
    settlement_status: Literal["pending", "final"]
    settled_credits: int | None
    settled_amount_minor: int | None


class VideoResult(BaseModel):
    content_url: str
    content_type: Literal["video/mp4"] = "video/mp4"
    size_bytes: int
    sha256: str
    duration_seconds: float
    expires_at: None = None


class Result(BaseModel):
    video: VideoResult


class JobError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed", "submission_unknown"]
    progress: Progress
    parameters: Parameters
    billing: Billing
    result: Result | None
    error: JobError | None
    created_at: str
    updated_at: str
    poll_after_seconds: int = 5


class CreditBalance(BaseModel):
    owner_id: str
    credits_per_cny: Literal[100] = 100
    available_credits: int
    reserved_credits: int
    spent_credits: int


class CreditTransaction(BaseModel):
    id: str
    type: Literal["adjustment", "reserve", "capture", "release"]
    credits: int
    direction: str | None = None
    job_id: str | None = None
    reason: str | None = None
    created_at: str


class CreditTransactions(BaseModel):
    data: list[CreditTransaction]


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["v1"] = "v1"
    endpoint: Literal["/api/v1/video-generations"] = "/api/v1/video-generations"
    encoding: Literal["multipart/form-data"] = "multipart/form-data"
    defaults: dict[str, str | None]
    allowed_output_combinations: list[dict[str, str]]
    source_video: dict[str, Any]
    replacement_image: dict[str, Any]
    target_language: dict[str, Any]
    pricing: dict[str, Any]
    polling: dict[str, int]


class PublicAPIError(ValueError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        field: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.field = field
        self.headers = headers or {}


class _RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def check(self, owner_id: str, bucket: str, limit: int) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(owner_id, bucket)]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = max(1, int(60 - (now - hits[0])))
                raise PublicAPIError(
                    429,
                    "rate_limit_exceeded",
                    "请求过于频繁",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)


class _DownloadLease:
    """One cross-process download slot, held until the response body closes."""

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor
        self._guard = threading.Lock()

    def release(self) -> None:
        with self._guard:
            descriptor = self._descriptor
            if descriptor < 0:
                return
            self._descriptor = -1
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _acquire_download_lease(
    data_dir: Path, owner_id: str, *, slots: int = 2
) -> _DownloadLease | None:
    directory = (
        data_dir
        / ".public-api"
        / "download-leases"
        / hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    flags = (
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for index in range(slots):
        descriptor = os.open(directory / f"slot-{index}.lock", flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("download lease is not a regular file")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(
                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError:
                os.close(descriptor)
                continue
            return _DownloadLease(descriptor)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    return None


def _request_id(request: Request) -> str:
    value = getattr(request.state, "public_request_id", None)
    if isinstance(value, str):
        return value
    value = f"req_{uuid.uuid4().hex}"
    request.state.public_request_id = value
    return value


def _is_public_path(path: str) -> bool:
    return path == "/api/v1" or path.startswith("/api/v1/")


def _error_response(request: Request, exc: PublicAPIError) -> JSONResponse:
    detail: dict[str, Any] = {
        "code": exc.code,
        "message": exc.message,
        "request_id": _request_id(request),
    }
    if exc.field is not None:
        detail["field"] = exc.field
    headers = {"Cache-Control": "private, no-store", **exc.headers}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail},
        headers=headers,
    )


def _public_create_error(status: int, detail: Mapping[str, Any]) -> PublicAPIError:
    code = detail.get("code")
    if not isinstance(code, str) or code not in _CREATE_ERROR_MESSAGES:
        return PublicAPIError(503, "creation_failed", "任务创建暂时失败，请稍后重试")
    field = detail.get("field")
    return PublicAPIError(
        status,
        code,
        _CREATE_ERROR_MESSAGES[code],
        field=field if isinstance(field, str) else None,
    )


def _contains_marker(value: Any, markers: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_marker(key, markers) or _contains_marker(item, markers)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_marker(item, markers) for item in value)
    return isinstance(value, str) and value in markers


def _definitely_failed(meta: Mapping[str, Any]) -> bool:
    generation = meta.get("generation")
    postprocess = meta.get("postprocess")
    prompt_fusion = meta.get("_prompt_fusion")
    return bool(
        meta.get("status") == "failed"
        or (isinstance(generation, Mapping) and generation.get("status") == "failed")
        or (
            isinstance(postprocess, Mapping)
            and postprocess.get("status") == "failed"
        )
        or (
            isinstance(prompt_fusion, Mapping)
            and prompt_fusion.get("status") == "failed"
        )
    )


def _failed_job_error(meta: Mapping[str, Any]) -> JobError:
    pipeline_error = meta.get("error")
    if pipeline_error == "provider_rejected":
        return JobError(
            code="image_rejected_by_provider",
            message="图片未通过供应商审核，请更换图片后重新创建任务",
        )
    if pipeline_error in {"audio_required", "source video has no audio track"}:
        return JobError(
            code="video_audio_required",
            message="视频没有可用音轨，请上传带口播音轨的视频后重新创建任务",
        )
    if pipeline_error == "long_video_duration_exceeded":
        return JobError(
            code="video_duration_exceeds_h3_limit",
            message="视频时长超过限制，请裁剪后重新创建任务",
        )
    if pipeline_error == "invalid_duration":
        return JobError(
            code="video_duration_invalid",
            message="无法确认视频时长，请更换视频后重新创建任务",
        )
    postprocess = meta.get("postprocess")
    if (
        isinstance(postprocess, Mapping)
        and postprocess.get("status") == "failed"
        and postprocess.get("error") == "provider_rejected"
    ):
        return JobError(
            code="image_rejected_by_provider",
            message="图片未通过供应商审核，请更换图片后重新创建任务",
        )
    generation = meta.get("generation")
    generation_errors: list[object] = []
    if isinstance(generation, Mapping) and generation.get("status") == "failed":
        generation_errors.append(generation.get("error"))
        segments = generation.get("segments")
        if isinstance(segments, list):
            generation_errors.extend(
                segment.get("error")
                for segment in segments
                if isinstance(segment, Mapping)
                and segment.get("status") == "failed"
            )
    if "h3_submit_rejected" in generation_errors:
        return JobError(
            code="video_rejected_by_provider",
            message="视频内容未通过供应商审核，请调整素材后重新创建任务",
        )
    if pipeline_error == "long_video_audio_mode_unsupported":
        return JobError(
            code="video_audio_unsupported",
            message="视频音频模式不受支持，请更换视频后重新创建任务",
        )
    return JobError(code="generation_failed", message="视频生成失败")


def _normalize_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接无效", field="source_video_url") from None
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接必须是无凭据的 HTTPS 公网地址", field="source_video_url")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接必须是无凭据的 HTTPS 公网地址", field="source_video_url")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接必须是无凭据的 HTTPS 公网地址", field="source_video_url")
    try:
        parsed_port = parsed.port
    except ValueError:
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接无效", field="source_video_url") from None
    rendered_host = f"[{host}]" if address is not None and address.version == 6 else host
    port = f":{parsed_port}" if parsed_port not in (None, 443) else ""
    return urlunsplit(("https", f"{rendered_host}{port}", parsed.path or "/", parsed.query, ""))


def _idempotency_digest(owner_id: str, value: str) -> str:
    return hashlib.sha256(
        f"{owner_id}\0POST /api/v1/video-generations\0{value}".encode("utf-8")
    ).hexdigest()


def _public_receipt(meta: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = meta.get("_public_api")
    if isinstance(value, Mapping) and value.get("version") == 1:
        return value
    return None


def _load_owned(settings: Settings, owner_id: str, job_id: str) -> tuple[str, dict[str, Any]]:
    match = _JOB_RE.fullmatch(job_id)
    if match is None:
        raise PublicAPIError(404, "job_not_found", "任务不存在")
    cid = match.group(1)
    meta = storage.load_meta(settings.data_dir, cid)
    receipt = _public_receipt(meta or {})
    if meta is None or receipt is None or receipt.get("owner_id") != owner_id or receipt.get("job_id") != job_id:
        raise PublicAPIError(404, "job_not_found", "任务不存在")
    return cid, meta


def _billing(status: str) -> Billing:
    if status == "succeeded":
        return Billing(settlement_status="final", settled_credits=1000, settled_amount_minor=1000)
    if status == "failed":
        return Billing(settlement_status="final", settled_credits=0, settled_amount_minor=0)
    return Billing(settlement_status="pending", settled_credits=None, settled_amount_minor=None)


def install(
    app,
    settings: Settings,
    *,
    create_job: Callable[..., Awaitable[dict[str, object]]],
    valid_generated_video: Callable[[dict[str, Any]], bool],
    progress_for: Callable[[dict[str, Any], bool], int | None],
) -> None:
    """Install public-only routes.  Call only when explicitly enabled."""

    public_api_auth.validate_registry(settings.public_api_clients_file)
    router = APIRouter(prefix="/api/v1")
    bearer = HTTPBearer(auto_error=False)
    limiter = _RateLimiter()

    async def principal(
        credentials: HTTPAuthorizationCredentials | None = Security(bearer),
    ) -> public_api_auth.Principal:
        authorization = None
        if credentials is not None:
            authorization = f"{credentials.scheme} {credentials.credentials}"
        try:
            return await run_in_threadpool(
                public_api_auth.authenticate,
                settings.public_api_clients_file,
                authorization,
            )
        except public_api_auth.PublicAuthError as exc:
            headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
            message = "API Key 无效" if exc.status_code == 401 else "API Key 注册表不可用"
            raise PublicAPIError(exc.status_code, exc.code, message, headers=headers) from None

    async def no_query(request: Request) -> None:
        if request.query_params:
            raise PublicAPIError(
                422,
                "invalid_query_parameters",
                "本接口不接受 Query 参数",
            )

    async def require_multipart(request: Request) -> None:
        content_type = request.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "multipart/form-data":
            raise PublicAPIError(
                415,
                "unsupported_content_type",
                "Content-Type 必须是 multipart/form-data",
                field="Content-Type",
            )

    async def validate_create_parameters(request: Request) -> None:
        form = await request.form()
        for field, allowed, code in (
            ("aspect_ratio", {"9:16", "16:9"}, "invalid_aspect_ratio"),
            ("resolution", {"768p", "480p"}, "invalid_resolution"),
        ):
            if field in form and form.get(field) not in allowed:
                raise PublicAPIError(
                    422,
                    code,
                    f"{field} 无效",
                    field=field,
                )

    async def validate_credit_query(request: Request) -> None:
        if set(request.query_params) - {"limit"} or len(
            request.query_params.getlist("limit")
        ) > 1:
            raise PublicAPIError(
                422,
                "invalid_query_parameters",
                "请求包含未知或重复的 Query 参数",
            )
        values = request.query_params.getlist("limit")
        if values and _STRICT_LIMIT_RE.fullmatch(values[0]) is None:
            raise PublicAPIError(
                422,
                "invalid_limit",
                "limit 必须是 1 到 100 的十进制整数",
                field="limit",
            )

    def project(owner_id: str, job_id: str) -> tuple[Job, Path | None]:
        cid, meta = _load_owned(settings, owner_id, job_id)
        cdir = settings.data_dir / cid
        artifact: tuple[Path, dict[str, Any]] | None
        try:
            artifact = public_artifacts.load(cdir)
            if artifact is not None and artifact[1].get("job_id") != job_id:
                raise public_artifacts.PublicArtifactError(
                    "artifact_job_mismatch"
                )
            if artifact is None and valid_generated_video(meta):
                artifact = public_artifacts.publish(
                    cdir, cdir / "generated.mp4", job_id=job_id
                )
            if artifact is not None:
                public_credits.settle(settings.data_dir, owner_id, job_id, succeeded=True)
                status = "succeeded"
            elif _contains_marker(meta, _UNKNOWN_MARKERS):
                status = "submission_unknown"
            elif _definitely_failed(meta):
                public_credits.settle(settings.data_dir, owner_id, job_id, succeeded=False)
                status = "failed"
            elif meta.get("status") == "queued":
                status = "queued"
            else:
                status = "running"
        except public_artifacts.PublicArtifactError as exc:
            raise PublicAPIError(503, "result_temporarily_unavailable", "成片校验失败，请稍后重试") from exc
        except public_credits.CreditError as exc:
            raise PublicAPIError(503, "billing_state_unavailable", "积分状态暂时不可用") from exc
        receipt = _public_receipt(meta)
        assert receipt is not None
        parameters = receipt.get("parameters")
        if not isinstance(parameters, Mapping):
            raise PublicAPIError(503, "job_state_invalid", "任务状态不可用")
        progress = progress_for(meta, artifact is not None)
        result = None
        artifact_path = None
        if artifact is not None:
            artifact_path, manifest = artifact
            result = Result(video=VideoResult(
                content_url=f"/api/v1/video-generations/{job_id}/content",
                size_bytes=manifest["size_bytes"],
                sha256=manifest["sha256"],
                duration_seconds=float(manifest["duration_seconds"]),
            ))
            progress = 100
        error = None
        if status == "failed":
            error = _failed_job_error(meta)
        elif status == "submission_unknown":
            error = JobError(
                code="submission_outcome_unknown",
                message="供应商提交结果待对账；请继续查询，勿重复创建任务",
                retryable=False,
            )
        try:
            job = Job(
                id=job_id,
                status=status,
                progress=Progress(percent=progress),
                parameters=Parameters(**dict(parameters)),
                billing=_billing(status),
                result=result,
                error=error,
                created_at=str(meta.get("created_at")),
                updated_at=str(meta.get("updated_at")),
            )
        except ValidationError as exc:
            raise PublicAPIError(
                503, "job_state_invalid", "任务状态不可用"
            ) from exc
        return job, artifact_path

    reconciliation_stop = threading.Event()
    reconciliation_thread: threading.Thread | None = None

    def reconcile_all() -> None:
        for meta in storage.list_conversations(settings.data_dir):
            receipt = _public_receipt(meta)
            if receipt is None:
                continue
            owner_id = receipt.get("owner_id")
            job_id = receipt.get("job_id")
            if not isinstance(owner_id, str) or not isinstance(job_id, str):
                continue
            try:
                project(owner_id, job_id)
            except Exception:
                _LOGGER.exception("public API reconciliation failed for %s", job_id)

    def reconciliation_loop() -> None:
        reconcile_all()
        while not reconciliation_stop.wait(5):
            reconcile_all()

    @app.on_event("startup")
    async def start_public_api_reconciliation() -> None:
        nonlocal reconciliation_thread
        reconciliation_thread = threading.Thread(
            target=reconciliation_loop,
            daemon=True,
            name="public-api-reconciliation",
        )
        reconciliation_thread.start()

    @app.on_event("shutdown")
    async def stop_public_api_reconciliation() -> None:
        reconciliation_stop.set()
        if reconciliation_thread is not None:
            reconciliation_thread.join(timeout=2)

    @router.get(
        "/video-generations/capabilities",
        response_model=Capabilities,
        dependencies=[Depends(no_query)],
    )
    async def capabilities(identity: public_api_auth.Principal = Depends(principal)):
        limiter.check(identity.owner_id, "read", 120)
        return Capabilities(
            defaults={"aspect_ratio": "9:16", "resolution": "768p", "target_language": None},
            allowed_output_combinations=[
                {"aspect_ratio": aspect, "resolution": resolution}
                for aspect in ("9:16", "16:9")
                for resolution in ("768p", "480p")
            ],
            source_video={
                "exactly_one_of": ["source_video", "source_video_url"],
                "extensions": [".mp4", ".mov", ".webm"],
                "max_bytes": settings.max_upload_mb * 1024 * 1024,
                "max_duration_seconds": 300,
                "url_scheme": "https",
            },
            replacement_image={
                "paired_with": "replacement_instruction",
                "media_types": list(minimal_creation.REPLACEMENT_MEDIA_TYPES),
                "max_bytes": minimal_creation.REPLACEMENT_MAX_BYTES,
                "max_instruction_chars": minimal_creation.REPLACEMENT_MAX_INSTRUCTION_CHARS,
            },
            target_language={"omitted_means": "same_as_source", "max_chars": 80},
            pricing={
                "credits_per_cny": 100,
                "job_price_credits": 1000,
                "job_price_amount_minor": 1000,
                "currency": "CNY",
                "price_version": public_credits.PRICE_VERSION,
            },
            polling={"recommended_seconds": 5, "maximum_backoff_seconds": 10},
        )

    @router.post(
        "/video-generations",
        response_model=Job,
        status_code=201,
        dependencies=[
            Depends(no_query),
            Depends(require_multipart),
            Depends(validate_create_parameters),
        ],
        responses={400: {"model": ErrorEnvelope}, 402: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}, 429: {"model": ErrorEnvelope}},
    )
    async def create_video_generation(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        identity: public_api_auth.Principal = Depends(principal),
        source_video: Annotated[UploadFile | None, File()] = None,
        source_video_url: Annotated[str | None, Form()] = None,
        aspect_ratio: Annotated[Literal["9:16", "16:9"], Form()] = "9:16",
        resolution: Annotated[Literal["768p", "480p"], Form()] = "768p",
        target_language: Annotated[str | None, Form()] = None,
        replacement_image: Annotated[UploadFile | None, File()] = None,
        replacement_instruction: Annotated[str | None, Form()] = None,
    ):
        limiter.check(identity.owner_id, "create", 10)
        idempotency_key = request.headers.get("Idempotency-Key")
        if not isinstance(idempotency_key, str) or _IDEMPOTENCY_RE.fullmatch(idempotency_key) is None:
            raise PublicAPIError(400, "invalid_idempotency_key", "Idempotency-Key 格式无效", field="Idempotency-Key")
        form = await request.form()
        if set(form) - _CREATE_FIELDS or any(len(form.getlist(key)) != 1 for key in form):
            raise PublicAPIError(422, "invalid_create_request", "请求包含未知或重复字段")
        raw_url = form.get("source_video_url")
        if raw_url is not None and not isinstance(raw_url, str):
            raise PublicAPIError(422, "invalid_source_video_url", "视频链接无效", field="source_video_url")
        url = (raw_url or source_video_url or "").strip()
        if (source_video is None) == (not url):
            raise PublicAPIError(422, "source_exactly_one_required", "source_video 与 source_video_url 必须且只能提供一个")
        normalized_url = _normalize_url(url) if url else None
        if source_video is not None:
            suffix = Path(source_video.filename or "").suffix.lower()
            if suffix not in _VIDEO_SUFFIXES:
                raise PublicAPIError(415, "unsupported_source_media_type", "原视频格式不受支持", field="source_video")
        raw_aspect_ratio = form.get("aspect_ratio", aspect_ratio)
        if raw_aspect_ratio not in {"9:16", "16:9"}:
            raise PublicAPIError(422, "invalid_aspect_ratio", "aspect_ratio 无效", field="aspect_ratio")
        aspect_ratio = str(raw_aspect_ratio)
        raw_resolution = form.get("resolution", resolution)
        if raw_resolution not in {"768p", "480p"}:
            raise PublicAPIError(422, "invalid_resolution", "resolution 无效", field="resolution")
        resolution = str(raw_resolution)
        language = None
        raw_language = form.get("target_language")
        if raw_language is not None:
            if not isinstance(raw_language, str):
                raise PublicAPIError(422, "invalid_target_language", "target_language 无效", field="target_language")
            language = raw_language.strip()
            if not language:
                raise PublicAPIError(422, "invalid_target_language", "target_language 不能为空", field="target_language")
            if minimal_creation.utf16_code_units(language) > 80:
                raise PublicAPIError(422, "target_language_too_long", "target_language 超过长度限制", field="target_language")
        instruction = None
        raw_instruction = form.get("replacement_instruction")
        if raw_instruction is not None:
            if not isinstance(raw_instruction, str):
                raise PublicAPIError(422, "replacement_pair_required", "replacement_instruction 无效", field="replacement_instruction")
            instruction = raw_instruction.strip()
        if (replacement_image is None) != (not instruction):
            raise PublicAPIError(422, "replacement_pair_required", "replacement_image 与 replacement_instruction 必须同时提供")
        if instruction is not None and minimal_creation.utf16_code_units(instruction) > 1000:
            raise PublicAPIError(422, "replacement_instruction_too_long", "replacement_instruction 超过长度限制", field="replacement_instruction")
        generation_request = {
            "version": 1,
            "output": {"aspect_ratio": aspect_ratio, "resolution": resolution, "fit_mode": "auto"},
            "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
            "dialogue": {"mode": "auto_rewrite", "target_language": language or "与原视频相同"},
            "replacement_guidance": None if instruction is None else {"instruction": instruction, "image_field": "replacement_image"},
        }
        idem_digest = _idempotency_digest(identity.owner_id, idempotency_key)
        descriptor = {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "target_language": language,
            "replacement_instruction": instruction,
            "source_kind": "upload" if source_video is not None else "url",
            "source_url": normalized_url,
        }
        public_context = {
            "version": 1,
            "owner_id": identity.owner_id,
            "creator_key_id": identity.key_id,
            "idempotency_digest": idem_digest,
            "request_descriptor": descriptor,
            "parameters": {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "target_language": language,
                "replacement_image": replacement_image is not None,
            },
        }
        internal_items: list[tuple[str, Any]] = []
        if source_video is not None:
            internal_items.append(("file", source_video))
        else:
            internal_items.append(("reference_url", normalized_url))
        internal_items.extend([
            ("client_request_id", idem_digest[:32]),
            ("generation_request", json.dumps(generation_request, ensure_ascii=False)),
        ])
        if replacement_image is not None:
            internal_items.append(("replacement_image", replacement_image))
        internal_response = Response(status_code=201)
        try:
            result = await create_job(
                form=FormData(internal_items),
                response=internal_response,
                background_tasks=background_tasks,
                file=source_video,
                reference_url=normalized_url or "",
                client_request_id=idem_digest[:32],
                generation_request_json=json.dumps(generation_request, ensure_ascii=False),
                replacement_image=replacement_image,
                public_context=public_context,
            )
        except public_credits.CreditError as exc:
            if exc.code == "insufficient_credits":
                raise PublicAPIError(402, "insufficient_credits", "可用积分不足 1000") from None
            raise PublicAPIError(503, "billing_state_unavailable", "积分状态暂时不可用") from None
        except PublicAPIError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = getattr(exc, "detail", None)
            if isinstance(status, int) and isinstance(detail, Mapping):
                raise _public_create_error(status, detail) from None
            raise
        job_id = f"vg_{result['id']}"
        job, _ = await run_in_threadpool(project, identity.owner_id, job_id)
        response.status_code = internal_response.status_code
        response.headers["Location"] = f"/api/v1/video-generations/{job_id}"
        response.headers["Retry-After"] = "5"
        response.headers["Cache-Control"] = "private, no-store"
        return job

    @router.get(
        "/video-generations/{job_id}",
        response_model=Job,
        dependencies=[Depends(no_query)],
    )
    async def get_video_generation(
        job_id: str,
        response: Response,
        identity: public_api_auth.Principal = Depends(principal),
    ):
        limiter.check(identity.owner_id, "read", 120)
        job, _ = await run_in_threadpool(project, identity.owner_id, job_id)
        if job.status in {"queued", "running", "submission_unknown"}:
            response.headers["Retry-After"] = str(job.poll_after_seconds)
        return job

    @router.get(
        "/account/credits",
        response_model=CreditBalance,
        dependencies=[Depends(no_query)],
    )
    async def get_credits(identity: public_api_auth.Principal = Depends(principal)):
        limiter.check(identity.owner_id, "read", 120)
        try:
            current = await run_in_threadpool(public_credits.balance, settings.data_dir, identity.owner_id)
        except public_credits.CreditError as exc:
            raise PublicAPIError(503, "billing_state_unavailable", "积分状态暂时不可用") from exc
        return CreditBalance(
            owner_id=identity.owner_id,
            available_credits=current["available"],
            reserved_credits=current["reserved"],
            spent_credits=current["spent"],
        )

    @router.get(
        "/account/credit-transactions",
        response_model=CreditTransactions,
        dependencies=[Depends(validate_credit_query)],
    )
    async def get_credit_transactions(
        identity: public_api_auth.Principal = Depends(principal),
        limit: int = 50,
    ):
        limiter.check(identity.owner_id, "read", 120)
        try:
            events = await run_in_threadpool(
                lambda: public_credits.recent_events(settings.data_dir, identity.owner_id, limit=limit)
            )
        except public_credits.CreditError as exc:
            code = "invalid_limit" if exc.code == "invalid_limit" else "billing_state_unavailable"
            status = 422 if exc.code == "invalid_limit" else 503
            raise PublicAPIError(status, code, "limit 必须在 1 到 100 之间" if status == 422 else "积分状态暂时不可用") from exc
        return CreditTransactions(data=[CreditTransaction(
            id=item["event_id"], type=item["type"], credits=item["credits"],
            direction=item.get("direction"), job_id=item.get("job_id"),
            reason=item.get("reason"), created_at=item["created_at"],
        ) for item in events])

    def _range_header(value: str | None, size: int) -> tuple[int, int] | None:
        if value is None:
            return None
        if not value.startswith("bytes=") or "," in value:
            raise PublicAPIError(416, "invalid_range", "仅支持单一字节范围", headers={"Content-Range": f"bytes */{size}"})
        spec = value[6:]
        try:
            start_raw, end_raw = spec.split("-", 1)
            if not start_raw:
                length = int(end_raw)
                if length <= 0:
                    raise ValueError
                start = max(0, size - length)
                end = size - 1
            else:
                start = int(start_raw)
                end = size - 1 if not end_raw else int(end_raw)
                if start < 0 or end < start or start >= size:
                    raise ValueError
                end = min(end, size - 1)
        except (ValueError, TypeError):
            raise PublicAPIError(416, "invalid_range", "字节范围无效", headers={"Content-Range": f"bytes */{size}"}) from None
        return start, end

    def _stream(path: Path, start: int, length: int, lease: _DownloadLease):
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        finally:
            lease.release()

    async def content_response(job_id: str, identity: public_api_auth.Principal, range_value: str | None, *, head: bool):
        limiter.check(identity.owner_id, "read", 120)
        job, path = await run_in_threadpool(project, identity.owner_id, job_id)
        if job.status != "succeeded" or path is None or job.result is None:
            raise PublicAPIError(409, "result_not_ready", "成片尚未就绪")
        video = job.result.video
        selected = _range_header(None if head else range_value, video.size_bytes)
        start, end = selected or (0, video.size_bytes - 1)
        length = end - start + 1
        headers = {
            "Accept-Ranges": "bytes",
            "ETag": f'"{video.sha256}"',
            "Content-Length": str(length),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="{job_id}.mp4"',
        }
        status_code = 200
        if selected is not None:
            status_code = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{video.size_bytes}"
        if head:
            return Response(status_code=200, media_type="video/mp4", headers=headers)
        try:
            lease = await run_in_threadpool(
                _acquire_download_lease,
                settings.data_dir,
                identity.owner_id,
            )
        except OSError as exc:
            raise PublicAPIError(
                503,
                "download_state_unavailable",
                "下载状态暂时不可用",
            ) from exc
        if lease is None:
            raise PublicAPIError(429, "download_limit_exceeded", "并发下载数已达上限", headers={"Retry-After": "1"})
        try:
            return StreamingResponse(
                _stream(path, start, length, lease),
                status_code=status_code,
                media_type="video/mp4",
                headers=headers,
                background=BackgroundTask(lease.release),
            )
        except BaseException:
            lease.release()
            raise

    @router.get(
        "/video-generations/{job_id}/content",
        dependencies=[Depends(no_query)],
    )
    async def get_content(
        job_id: str,
        request: Request,
        identity: public_api_auth.Principal = Depends(principal),
    ):
        return await content_response(job_id, identity, request.headers.get("Range"), head=False)

    @router.head(
        "/video-generations/{job_id}/content",
        dependencies=[Depends(no_query)],
    )
    async def head_content(
        job_id: str,
        identity: public_api_auth.Principal = Depends(principal),
    ):
        return await content_response(job_id, identity, None, head=True)

    app.include_router(router)

    @app.get("/api/v1/openapi.json", include_in_schema=False)
    async def public_openapi(request: Request):
        await no_query(request)
        return get_openapi(
            title="Duet Video Generation API",
            version="1.0.0",
            description="Asynchronous server-to-server video generation API.",
            routes=router.routes,
        )

    @app.exception_handler(PublicAPIError)
    async def public_api_error_handler(request: Request, exc: PublicAPIError):
        return _error_response(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def public_http_error_handler(
        request: Request, exc: StarletteHTTPException
    ):
        if not _is_public_path(request.url.path):
            return await http_exception_handler(request, exc)
        if exc.status_code == 400:
            code, message = "invalid_multipart", "multipart 请求格式无效"
        elif exc.status_code == 404:
            code, message = "endpoint_not_found", "接口不存在"
        elif exc.status_code == 405:
            code, message = "method_not_allowed", "HTTP 方法不受支持"
        elif exc.status_code == 413:
            code, message = "request_too_large", "请求正文超过大小限制"
        elif exc.status_code == 415:
            code, message = "unsupported_content_type", "请求 Content-Type 不受支持"
        else:
            code, message = "request_failed", "请求未能完成"
        headers = dict(exc.headers or {})
        if exc.status_code == 405 and "Allow" not in headers:
            allowed = {
                method
                for route in router.routes
                if route.matches(request.scope)[0] is Match.PARTIAL
                for method in (route.methods or set())
            }
            if allowed:
                headers["Allow"] = ", ".join(sorted(allowed))
        return _error_response(
            request,
            PublicAPIError(
                exc.status_code,
                code,
                message,
                headers=headers,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def public_validation_error_handler(
        request: Request, exc: RequestValidationError
    ):
        if not _is_public_path(request.url.path):
            return await request_validation_exception_handler(request, exc)
        errors = exc.errors()
        first = errors[0] if errors else {}
        location = first.get("loc")
        field = None
        if isinstance(location, (list, tuple)) and location:
            field = str(location[-1])
        return _error_response(
            request,
            PublicAPIError(
                422,
                "invalid_request",
                "请求字段不符合 v1 合同",
                field=field,
            ),
        )
