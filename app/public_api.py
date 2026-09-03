"""Versioned server-to-server API facade for asynchronous video generation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
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
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

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


def _request_id(request: Request) -> str:
    value = getattr(request.state, "public_request_id", None)
    if isinstance(value, str):
        return value
    value = f"req_{uuid.uuid4().hex}"
    request.state.public_request_id = value
    return value


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
    return bool(
        meta.get("status") == "failed"
        or (isinstance(generation, Mapping) and generation.get("status") == "failed")
    )


def _normalize_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接无效", field="source_video_url") from None
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise PublicAPIError(422, "invalid_source_video_url", "视频链接必须是无凭据的 HTTPS 公网地址", field="source_video_url")
    host = parsed.hostname.lower()
    port = f":{parsed.port}" if parsed.port not in (None, 443) else ""
    return urlunsplit(("https", f"{host}{port}", parsed.path or "/", parsed.query, ""))


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
    download_locks: dict[str, threading.BoundedSemaphore] = {}
    download_guard = threading.Lock()

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
            error = JobError(code="generation_failed", message="视频生成失败")
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

    @router.get("/video-generations/capabilities", response_model=Capabilities)
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
        url = (source_video_url or "").strip()
        if (source_video is None) == (not url):
            raise PublicAPIError(422, "source_exactly_one_required", "source_video 与 source_video_url 必须且只能提供一个")
        normalized_url = _normalize_url(url) if url else None
        if source_video is not None:
            suffix = Path(source_video.filename or "").suffix.lower()
            if suffix not in _VIDEO_SUFFIXES:
                raise PublicAPIError(415, "unsupported_source_media_type", "原视频格式不受支持", field="source_video")
        language = None
        if target_language is not None:
            language = target_language.strip()
            if not language:
                raise PublicAPIError(422, "invalid_target_language", "target_language 不能为空", field="target_language")
            if minimal_creation.utf16_code_units(language) > 80:
                raise PublicAPIError(422, "target_language_too_long", "target_language 超过长度限制", field="target_language")
        instruction = None
        if replacement_instruction is not None:
            instruction = replacement_instruction.strip()
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
                raise PublicAPIError(status, str(detail.get("code", "invalid_create_request")), str(detail.get("message", "创建请求无效")), field=detail.get("field")) from None
            raise
        job_id = f"vg_{result['id']}"
        job, _ = await run_in_threadpool(project, identity.owner_id, job_id)
        response.status_code = internal_response.status_code
        response.headers["Location"] = f"/api/v1/video-generations/{job_id}"
        response.headers["Retry-After"] = "5"
        response.headers["Cache-Control"] = "private, no-store"
        return job

    @router.get("/video-generations/{job_id}", response_model=Job)
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

    @router.get("/account/credits", response_model=CreditBalance)
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

    @router.get("/account/credit-transactions", response_model=CreditTransactions)
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

    def _stream(path: Path, start: int, length: int, semaphore: threading.BoundedSemaphore):
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
            semaphore.release()

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
        with download_guard:
            semaphore = download_locks.setdefault(identity.owner_id, threading.BoundedSemaphore(2))
        if not semaphore.acquire(blocking=False):
            raise PublicAPIError(429, "download_limit_exceeded", "并发下载数已达上限", headers={"Retry-After": "1"})
        return StreamingResponse(
            _stream(path, start, length, semaphore),
            status_code=status_code,
            media_type="video/mp4",
            headers=headers,
        )

    @router.get("/video-generations/{job_id}/content")
    async def get_content(
        job_id: str,
        request: Request,
        identity: public_api_auth.Principal = Depends(principal),
    ):
        return await content_response(job_id, identity, request.headers.get("Range"), head=False)

    @router.head("/video-generations/{job_id}/content")
    async def head_content(
        job_id: str,
        identity: public_api_auth.Principal = Depends(principal),
    ):
        return await content_response(job_id, identity, None, head=True)

    app.include_router(router)

    @app.get("/api/v1/openapi.json", include_in_schema=False)
    async def public_openapi():
        return get_openapi(
            title="Duet Video Generation API",
            version="1.0.0",
            description="Asynchronous server-to-server video generation API.",
            routes=router.routes,
        )

    @app.exception_handler(PublicAPIError)
    async def public_api_error_handler(request: Request, exc: PublicAPIError):
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def public_validation_error_handler(
        request: Request, exc: RequestValidationError
    ):
        if not request.url.path.startswith("/api/v1/"):
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
