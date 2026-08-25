import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import (
    downloader,
    frame_fit,
    h3,
    image_optimization,
    long_generation,
    long_video,
    pipeline,
    postprocess,
    prepared_input,
    storage,
    voice,
)
from app.auth import require_auth
from app.codex_runner import CodexRunner
from app.config import Settings, get_settings

_RATE_LIMIT = 10  # 每 IP 每分钟上传次数
_RATE_WINDOW_S = 60
# 前端幂等键（boot / 内容变更 / 上传成功时轮换，失败重试复用）；空 = 不参与幂等（兼容 curl）
_CLIENT_REQUEST_ID_RE = re.compile(r"^[0-9A-Za-z-]{8,64}$")
_DIALOGUE_MODES = frozenset({"auto", "edit", "custom", "none"})
_FIT_MODES = frozenset({"none", "crop", "pad"})
_ASPECT_RATIOS = h3.H3_ASPECT_RATIOS
_RESOLUTIONS = h3.H3_RESOLUTIONS
_GENERATION_ACTIVE = frozenset({"queued", "running"})
_GENERATION_RETRYABLE = frozenset({"failed"})
_GENERATION_RESUMABLE = frozenset({"resume_required"})
_AMBIGUOUS_SUBMIT_ERRORS = frozenset(
    {"state_persist_failed", "submission_unknown", "h3_internal_error"}
)
_KNOWN_TASK_ERRORS = frozenset(
    {
        "h3_query_failed",
        "h3_timeout",
        "download_failed",
        "download_dns_failed",
        "download_peer_unverified",
        "output_write_failed",
        "output_probe_failed",
    }
)
_NO_STORE_WEB_PATHS = frozenset({"/", "/index.html", "/app.js", "/styles.css"})
_CLIENT_REFRESH_MESSAGE = "页面版本已更新，请刷新页面后重试。"
_GENERATED_VIDEO_VALIDATION_CACHE_SIZE = 256


def _short_provider_failure_is_recoverable(generation: object) -> bool:
    return (
        isinstance(generation, dict)
        and not isinstance(generation.get("segments"), list)
        and generation.get("status") == "failed"
        and generation.get("error") == "h3_provider_failed"
    )


def _long_provider_failure_is_recoverable(generation: object) -> bool:
    if (
        not isinstance(generation, dict)
        or generation.get("status") != "failed"
        or generation.get("error") != "long_video_segment_failed"
        or not isinstance(generation.get("segments"), list)
    ):
        return False
    return any(
        isinstance(segment, dict)
        and segment.get("status") == "failed"
        and segment.get("error") == "h3_provider_failed"
        for segment in generation["segments"]
    )


class _GeneratedVideoValidationCache:
    """Bound strict local validation without weakening its file bindings."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple, tuple[str, bool]] = OrderedDict()
        self._inflight: dict[tuple, threading.Event] = {}
        self._lock = threading.Lock()

    def get_or_validate(self, identity: tuple, fingerprint: str, fingerprint_fn,
                        validator) -> bool:
        while True:
            with self._lock:
                cached = self._entries.get(identity)
                if cached is not None and cached[0] == fingerprint:
                    self._entries.move_to_end(identity)
                    return cached[1]
                if cached is not None:
                    self._entries.pop(identity)
                event = self._inflight.get(identity)
                if event is None:
                    event = threading.Event()
                    self._inflight[identity] = event
                    break
            event.wait()
            refreshed = fingerprint_fn()
            if refreshed is None:
                return bool(validator())
            fingerprint = refreshed

        try:
            result = bool(validator())
            stable_fingerprint = fingerprint_fn()
            with self._lock:
                if result and stable_fingerprint == fingerprint:
                    self._entries[identity] = (fingerprint, result)
                    self._entries.move_to_end(identity)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
            # A changing artifact makes a successful observation stale. The
            # next request will validate the new state from scratch.
            return result if stable_fingerprint == fingerprint else False
        finally:
            with self._lock:
                current = self._inflight.pop(identity, None)
                if current is not None:
                    current.set()


_GENERATED_VIDEO_VALIDATION_CACHE = _GeneratedVideoValidationCache(
    _GENERATED_VIDEO_VALIDATION_CACHE_SIZE
)


class _SubmitError(RuntimeError):
    def __init__(self, status: int, detail: str | dict[str, str]) -> None:
        if isinstance(detail, dict):
            if set(detail) != {"code", "message"} or not all(
                isinstance(detail[key], str) and detail[key]
                for key in ("code", "message")
            ):
                raise TypeError("structured submit detail must contain safe code and message")
            public_detail: str | dict[str, str] = {
                "code": detail["code"],
                "message": detail["message"],
            }
        elif isinstance(detail, str):
            public_detail = detail
        else:
            raise TypeError("submit detail must be a public string or safe structure")
        super().__init__(str(public_detail))
        self.status = status
        self.detail = public_detail


def _duration_limit_detail(duration_s: float) -> dict:
    return {
        "code": "video_duration_exceeds_h3_limit",
        "message": (
            f"视频时长 {duration_s:.1f} 秒，超过 H3 最大允许时长 "
            f"{long_video.LONG_VIDEO_MAX_S:g} 秒，请裁剪后重新上传。"
        ),
        "actual_duration_s": duration_s,
        "max_duration_s": long_video.LONG_VIDEO_MAX_S,
    }


def _duration_exceeds_h3_limit(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > long_video.LONG_VIDEO_MAX_S
    )


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
        return _public_lines([
            line for line in provenance
            if isinstance(line, dict)
            and line.get("kept") is True
            and not voice.is_unrecognized_text(line.get("text"))
        ])
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


def _navigation_status(meta: dict, *, has_video: bool) -> str:
    analysis = meta.get("status")
    analysis_states = {
        "queued": "analysis_queued",
        "processing": "analysis_processing",
        "failed": "analysis_failed",
    }
    if analysis in analysis_states:
        return analysis_states[analysis]
    if analysis != "done":
        return "analysis_unknown"

    generation = meta.get("generation")
    if not isinstance(generation, dict):
        return "analysis_complete"
    generation_status = _effective_generation_status(generation)
    generation_states = {
        "queued": "generation_queued",
        "running": "generation_running",
        "failed": "generation_failed",
        "submission_unknown": "generation_submission_unknown",
        "resume_required": "generation_resume_required",
    }
    if generation_status in generation_states:
        return generation_states[generation_status]
    if generation_status != "succeeded":
        return "generation_unknown"
    if not has_video:
        return "output_missing"

    postprocess_state = meta.get("postprocess")
    if isinstance(postprocess_state, dict):
        postprocess_states = {
            "running": "postprocessing",
            "failed": "postprocess_failed",
        }
        projected = postprocess_states.get(postprocess_state.get("status"))
        if projected is not None:
            return projected
    return "completed"


def _public_generation(meta: dict, cdir: Path, settings: Settings) -> dict | None:
    generation = meta.get("generation")
    if not isinstance(generation, dict):
        return None
    legacy = _is_legacy_generation_contract(generation)
    status = _effective_generation_status(generation)
    public = {
        "status": status,
        "error": (
            "generation_path_removed"
            if legacy and status == "failed"
            else generation.get("error")
        ),
        "attempt": generation.get("attempt"),
        "client_request_id": generation.get("client_request_id"),
        "stage": "h3" if legacy else generation.get("stage"),
    }
    if isinstance(generation.get("segments"), list):
        public["fast_mode"] = generation.get("fast_mode", False)
        public["segments"] = long_generation.public_segments(generation)
        if _is_long_video(meta) and status == "failed":
            frozen_segments = meta.get("segments")
            if isinstance(frozen_segments, list):
                expected = tuple(range(1, len(frozen_segments) + 1))
                reusable = frozenset()
                receipt = meta.get("frozen_plan_receipt")
                if isinstance(receipt, str):
                    try:
                        plan = long_generation.freeze_plan(
                            cdir,
                            meta,
                            receipt,
                            meta.get("fit_mode"),
                            meta.get("dialogue_mode"),
                            prepare_fit=False,
                        )
                        reusable = long_generation.bound_reusable_segment_indices(
                            settings, meta["id"], plan, generation
                        )
                    except long_generation.LongGenerationError:
                        pass
                public["retry_paid_segment_count"] = len(expected) - len(reusable)
    return public


def _is_long_video(meta: dict) -> bool:
    duration = meta.get("duration_s")
    valid_duration = (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
    )
    has_frozen_long_plan = (
        isinstance(meta.get("long_video_plan_receipt"), str)
        or isinstance(meta.get("segments"), list) and bool(meta["segments"])
        or isinstance(meta.get("generation"), dict)
        and isinstance(meta["generation"].get("segments"), list)
    )
    return (
        valid_duration
        and (
            float(duration) > long_video.SHORT_VIDEO_MAX_S
            or (
                float(duration) > long_video.PREVIOUS_SHORT_VIDEO_MAX_S
                and has_frozen_long_plan
            )
        )
    )


def _generation_semantics(meta: dict) -> tuple[str, str]:
    """Read frozen/recommended values; missing historical fields are exact legacy defaults."""
    aspect_ratio = meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
    resolution = meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
    if aspect_ratio not in _ASPECT_RATIOS or resolution not in _RESOLUTIONS:
        raise _SubmitError(409, "generation_parameters_invalid")
    return aspect_ratio, resolution


def _short_generation_parameters_match(
    meta: dict,
    *,
    dialogue_mode: str,
    dialogue: tuple[dict, ...],
    fit_mode: str,
    aspect_ratio: str,
    resolution: str,
) -> bool:
    """Compare every paid short-video input against its frozen semantics."""
    expected_dialogue = meta.get("prepared_dialogue")
    return (
        meta.get("dialogue_mode") == dialogue_mode
        and meta.get("fit_mode") == fit_mode
        and _generation_semantics(meta) == (aspect_ratio, resolution)
        and isinstance(expected_dialogue, list)
        and expected_dialogue == [dict(line) for line in dialogue]
    )


def _validated_fit_profiles(meta: dict) -> dict[str, dict[str, object]]:
    raw = meta.get("fit_profiles")
    if isinstance(raw, dict) and set(raw) == set(_ASPECT_RATIOS):
        normalized = {}
        for aspect_ratio in ("16:9", "9:16"):
            profile = raw.get(aspect_ratio)
            if (
                not isinstance(profile, dict)
                or set(profile) != {"fit_required", "default_fit_mode"}
                or not isinstance(profile.get("fit_required"), bool)
                or profile.get("default_fit_mode")
                != ("crop" if profile["fit_required"] else "none")
            ):
                raise _SubmitError(409, "generation_parameters_invalid")
            normalized[aspect_ratio] = dict(profile)
        return normalized
    # Historical receipts had only the fixed 9:16 requirement. Preserve that
    # exact interpretation; the alternative profile is conservatively fitted.
    legacy_required = meta.get("fit_required")
    if isinstance(legacy_required, bool):
        return {
            "16:9": {"fit_required": True, "default_fit_mode": "crop"},
            "9:16": {
                "fit_required": legacy_required,
                "default_fit_mode": "crop" if legacy_required else "none",
            },
        }
    raise _SubmitError(409, "fit_requirement_unknown")


def _fit_required(meta: dict, aspect_ratio: str) -> bool:
    return bool(_validated_fit_profiles(meta)[aspect_ratio]["fit_required"])


def _long_fit_required(cdir: Path, meta: dict) -> bool:
    """Resolve legacy-null long-video fit state from its immutable H3 anchors."""
    generation = meta.get("generation")
    if isinstance(generation, dict):
        fit_mode = meta.get("fit_mode")
        if fit_mode in _FIT_MODES:
            return fit_mode != "none"
        raise _SubmitError(409, "fit_requirement_unknown")
    persisted = meta.get("fit_required")
    if isinstance(persisted, bool):
        return persisted
    expected = long_generation.plan_receipt(cdir, meta)
    if expected is None:
        raise _SubmitError(409, "fit_requirement_unknown")
    try:
        plan = long_generation.freeze_plan(
            cdir,
            meta,
            expected,
            "none",
            meta.get("dialogue_mode", "auto"),
            prepare_fit=False,
        )
        return frame_fit.frame_bytes_require_fit(
            [
                anchor
                for segment in plan.segments
                for anchor in (
                    (
                        (segment.first_frame_data,)
                        if segment.join_mode == "hard_cut"
                        else ()
                    )
                    + (segment.last_frame_data,)
                )
            ],
            h3.H3_DEFAULT_ASPECT_RATIO,
        )
    except (frame_fit.FrameFitError, long_generation.LongGenerationError):
        raise _SubmitError(409, "fit_requirement_unknown") from None


def _is_legacy_generation_contract(generation: dict) -> bool:
    return (
        "context_ir_enabled" in generation
        or generation.get("stage") == "context_ir"
    )


def _effective_generation_status(generation: dict) -> str | None:
    status = generation.get("status")
    error = generation.get("error")
    if (
        _is_legacy_generation_contract(generation)
        and status not in {"succeeded", "submission_unknown"}
    ):
        return "failed"
    if status == "failed" and error in _KNOWN_TASK_ERRORS:
        return "resume_required"
    return status


def _source_prompt_snapshot(cdir: Path) -> tuple[str | None, str | None]:
    path = cdir / "work" / "visual_prompt.txt"
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None, None
    if not text.strip() or len(raw) > prepared_input.MAX_FINAL_PROMPT_BYTES:
        return None, None
    return text, hashlib.sha256(raw).hexdigest()


def _replace_source_prompt(
    settings: Settings,
    cid: str,
    meta: dict,
    expected_sha256: str,
    prompt: str,
) -> tuple[str, str, str]:
    cdir = (settings.data_dir / cid).resolve()
    current, current_sha256 = _source_prompt_snapshot(cdir)
    if current is None or current_sha256 is None:
        raise _SubmitError(409, "prepared_input_invalid")
    if expected_sha256 != current_sha256:
        raise _SubmitError(409, {
            "code": "prompt_changed",
            "message": "提示词已更新，请刷新页面后重试。",
        })
    replacement = prompt.strip()
    if (
        not replacement
        or len(replacement.encode("utf-8")) > prepared_input.MAX_FINAL_PROMPT_BYTES
    ):
        raise _SubmitError(422, "invalid_prompt")
    receipt_name = meta.get("prepared_input_receipt")
    if not isinstance(receipt_name, str) or receipt_name != prepared_input.RECEIPT_FILENAME:
        raise _SubmitError(409, "prepared_input_invalid")
    receipt_path = cdir / receipt_name
    try:
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_dialogue = receipt_payload["dialogue"]["lines"]
        frozen = prepared_input.load_prepared_input(
            cdir, receipt_path, expected_dialogue=expected_dialogue
        )
        prepared_input.compose_final_prompt(replacement, frozen.dialogue)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, prepared_input.PreparedInputError):
        raise _SubmitError(409, "prepared_input_invalid") from None

    frozen.visual_prompt.path.write_text(replacement, encoding="utf-8")
    try:
        rewritten = prepared_input.write_prepared_input(
            root=cdir,
            source=frozen.source.path,
            audio=frozen.normalized_audio.path if frozen.normalized_audio else None,
            keyframes=[item.path for item in frozen.keyframes],
            visual=frozen.visual_prompt.path,
            final=frozen.final_prompt.path,
            dialogue_mode=frozen.dialogue_mode,
            dialogue=frozen.dialogue,
            vocal_filter_enabled=frozen.vocal_filter_enabled,
            duration_s=frozen.duration_s,
            ratio=frozen.ratio,
            fit_mode=frozen.fit_mode,
            engine_request=frozen.engine_request,
            receipt_path=frozen.receipt_path,
        )
    except prepared_input.PreparedInputError:
        raise _SubmitError(409, "prepared_input_invalid") from None
    storage.update_meta(settings.data_dir, cid, prompt=rewritten.prompt_text)
    digest = hashlib.sha256(replacement.encode("utf-8")).hexdigest()
    return replacement, digest, rewritten.prompt_text


def _receipt_version(cdir: Path, meta: dict) -> int | None:
    name = meta.get("prepared_input_receipt")
    if not isinstance(name, str) or not name or name != Path(name).name:
        return None
    return prepared_input.RECEIPT_VERSION if (cdir / name).is_file() else None


def _timeouts(settings: Settings) -> h3.Timeouts:
    return h3.Timeouts(
        request_s=settings.h3_request_timeout_s,
        h3_poll_s=settings.h3_poll_timeout_s,
        download_s=settings.h3_download_timeout_s,
        poll_interval_s=settings.h3_poll_interval_s,
        retry_count=settings.retry_count,
        retry_interval_s=settings.retry_interval_s,
    )


def _credentials_ready(settings: Settings) -> bool:
    return bool(settings.autodl_art_token.strip())


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
                if (
                    not isinstance(item, dict)
                    or item.get("kept") is not True
                    or voice.is_unrecognized_text(item.get("text"))
                ):
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


def _validate_submit_payload(
    meta: dict, payload: dict
) -> tuple[str, str, str, str, tuple[dict, ...]]:
    if payload.get("confirm") is not True:
        raise _SubmitError(409, "confirmation required")
    allowed = {
        "confirm",
        "client_request_id",
        "dialogue_mode",
        "lines",
        "fit_mode",
        "aspect_ratio",
        "resolution",
    }
    if set(payload) - allowed:
        raise _SubmitError(422, "invalid_submit_request")
    request_id = payload.get("client_request_id")
    if not isinstance(request_id, str) or not _CLIENT_REQUEST_ID_RE.fullmatch(request_id):
        raise _SubmitError(422, "invalid_client_request_id")
    aspect_ratio = payload.get("aspect_ratio")
    if aspect_ratio not in _ASPECT_RATIOS:
        raise _SubmitError(422, "invalid_aspect_ratio")
    resolution = payload.get("resolution")
    if resolution not in _RESOLUTIONS:
        raise _SubmitError(422, "invalid_resolution")
    fit_mode = payload.get("fit_mode")
    if fit_mode not in _FIT_MODES:
        raise _SubmitError(422, "invalid_fit_mode")
    fit_required = _fit_required(meta, aspect_ratio)
    if fit_required:
        if fit_mode not in {"crop", "pad"}:
            raise _SubmitError(422, "fit_mode_required")
    elif fit_mode != "none":
        raise _SubmitError(422, "fit_mode_not_allowed")
    return (
        request_id,
        fit_mode,
        aspect_ratio,
        resolution,
        _validated_dialogue(meta, payload),
    )


def _validate_long_submit_payload(
    meta: dict, payload: dict,
) -> tuple[str, str, str, str, str, str, bool]:
    if payload.get("confirm") is not True:
        raise _SubmitError(409, "confirmation required")
    allowed = {
        "confirm", "client_request_id", "dialogue_mode", "fit_mode",
        "aspect_ratio", "resolution",
        "expected_plan_receipt",
        "fast_mode",
    }
    required = allowed - {"fast_mode"}
    if not required.issubset(payload) or set(payload) - allowed:
        if "lines" in payload or payload.get("dialogue_mode") in {"edit", "custom"}:
            raise _SubmitError(422, "long_video_audio_mode_unsupported")
        legacy_allowed = required - {
            "expected_plan_receipt", "aspect_ratio", "resolution"
        }
        request_id = payload.get("client_request_id")
        dialogue_mode = payload.get("dialogue_mode")
        fit_mode = payload.get("fit_mode")
        legacy_fit_valid = (
            fit_mode in _FIT_MODES
            and (
                isinstance(meta.get("generation"), dict)
                or (meta.get("fit_required") is True and fit_mode in {"crop", "pad"})
                or (meta.get("fit_required") is False and fit_mode == "none")
            )
        )
        if (
            set(payload) == legacy_allowed
            and isinstance(request_id, str)
            and _CLIENT_REQUEST_ID_RE.fullmatch(request_id)
            and dialogue_mode in {"auto", "none"}
            and meta.get("voice_mode") == "keep"
            and legacy_fit_valid
        ):
            raise _SubmitError(
                409,
                {
                    "code": "client_refresh_required",
                    "message": "页面版本已更新，请刷新页面后重试。",
                },
            )
        raise _SubmitError(422, "invalid_submit_request")
    request_id = payload.get("client_request_id")
    if not isinstance(request_id, str) or not _CLIENT_REQUEST_ID_RE.fullmatch(request_id):
        raise _SubmitError(422, "invalid_client_request_id")
    dialogue_mode = payload.get("dialogue_mode")
    if dialogue_mode not in {"auto", "none"}:
        raise _SubmitError(422, "long_video_audio_mode_unsupported")
    if meta.get("voice_mode") != "keep":
        raise _SubmitError(422, "long_video_audio_mode_unsupported")
    aspect_ratio = payload.get("aspect_ratio")
    if aspect_ratio not in _ASPECT_RATIOS:
        raise _SubmitError(422, "invalid_aspect_ratio")
    resolution = payload.get("resolution")
    if resolution not in _RESOLUTIONS:
        raise _SubmitError(422, "invalid_resolution")
    fit_mode = payload.get("fit_mode")
    if fit_mode not in _FIT_MODES:
        raise _SubmitError(422, "invalid_fit_mode")
    if not isinstance(meta.get("generation"), dict):
        fit_required = _fit_required(meta, aspect_ratio)
        if fit_required:
            if fit_mode not in {"crop", "pad"}:
                raise _SubmitError(422, "fit_mode_required")
        else:
            if fit_mode != "none":
                raise _SubmitError(422, "fit_mode_not_allowed")
    expected = payload.get("expected_plan_receipt")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise _SubmitError(422, "invalid_plan_receipt")
    fast_mode = payload.get("fast_mode", False)
    if not isinstance(fast_mode, bool):
        raise _SubmitError(422, "invalid_fast_mode")
    return (
        request_id, fit_mode, dialogue_mode, expected, aspect_ratio, resolution,
        fast_mode,
    )


def _make_h3_request(
    settings: Settings,
    cid: str,
    frozen: prepared_input.PreparedInput,
    client_request_id: str,
) -> h3.H3Request:
    duration = long_video.provider_duration_s(0.0, frozen.duration_s)
    engine_h3 = frozen.engine_request.get("h3")
    legacy_engine = (
        frozen.ratio == h3.H3_DEFAULT_ASPECT_RATIO
        and isinstance(engine_h3, dict)
        and engine_h3.get("resolution") == h3.H3_RESOLUTION
        and "aspect_ratio" not in engine_h3
        and "provider_resolution" not in engine_h3
    )
    if legacy_engine:
        aspect_ratio = h3.H3_DEFAULT_ASPECT_RATIO
        resolution = h3.H3_DEFAULT_RESOLUTION
    elif (
        frozen.ratio not in _ASPECT_RATIOS
        or not isinstance(engine_h3, dict)
        or engine_h3.get("workflow") not in h3.H3_REFERENCE_WORKFLOWS
        or engine_h3.get("aspect_ratio") != frozen.ratio
        or engine_h3.get("resolution") not in _RESOLUTIONS
        or engine_h3.get("provider_resolution")
        != h3.provider_resolution(frozen.ratio, engine_h3.get("resolution"))
    ):
        raise _SubmitError(409, "prepared_input_invalid")
    else:
        aspect_ratio = frozen.ratio
        resolution = engine_h3["resolution"]
    return h3.H3Request(
        cid=cid,
        workdir=(settings.data_dir / cid).resolve(),
        client_request_id=client_request_id,
        prompt=frozen.final_prompt.data.decode("utf-8"),
        keyframes=tuple((artifact.path, artifact.data) for artifact in frozen.keyframes),
        voice_texts=frozen.voice_texts,
        voice_receipt=h3.voice_texts_receipt(frozen.voice_texts),
        duration=duration,
        autodl_token=settings.autodl_art_token,
        timeouts=_timeouts(settings),
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        workflow=engine_h3.get("workflow", h3.H3_PREVIOUS_WORKFLOW),
    )


def _freeze_submission(
    settings: Settings,
    cid: str,
    meta: dict,
    request_id: str,
    dialogue_mode: str,
    fit_mode: str,
    aspect_ratio: str,
    resolution: str,
    dialogue: tuple[dict, ...],
) -> h3.H3Request:
    cdir = (settings.data_dir / cid).resolve()
    work = cdir / "work"
    originals = _original_keyframes(cdir, meta)
    try:
        originals = postprocess.generation_keyframes(cdir, meta, originals)
    except postprocess.PostprocessError as exc:
        detail = exc.detail if isinstance(exc.detail, str) else exc.detail["code"]
        raise _SubmitError(exc.status, detail) from None
    if fit_mode == "none":
        keyframes = originals
    else:
        try:
            keyframes = list(frame_fit.fit_frames(
                originals,
                work / "h3_frames" / aspect_ratio.replace(":", "x") / fit_mode,
                fit_mode,
                aspect_ratio,
            ))
        except frame_fit.FrameFitError:
            raise _SubmitError(409, "frame_fit_failed") from None
    visual = work / "visual_prompt.txt"
    if not visual.is_file():
        raise _SubmitError(409, "prepared_input_invalid")
    duration = float(meta["duration_s"])
    request_duration = long_video.provider_duration_s(0.0, duration)
    engine_request = {
        "h3": {
            "workflow": h3.H3_WORKFLOW,
            "duration": request_duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "provider_resolution": h3.provider_resolution(
                aspect_ratio, resolution
            ),
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
            ratio=aspect_ratio,
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
    aspect_ratio, resolution = _generation_semantics(meta)
    if (
        frozen.dialogue_mode != meta.get("dialogue_mode")
        or frozen.fit_mode != meta.get("fit_mode")
        or frozen.ratio != aspect_ratio
    ):
        raise _SubmitError(409, "prepared_input_invalid")
    request = _make_h3_request(settings, cid, frozen, request_id)
    if request.resolution != resolution:
        raise _SubmitError(409, "prepared_input_invalid")
    return request


def _claim_first_submission(
    settings: Settings, cid: str, request_id: str,
) -> tuple[dict, object]:
    claimed = storage.claim_submission_input(settings.data_dir, cid, request_id)
    if claimed is None:
        current = storage.load_meta(settings.data_dir, cid)
        detail = (
            "artifacts not ready"
            if current and current.get("status") != "done"
            else "generation in progress"
        )
        raise HTTPException(status_code=409, detail=detail)
    return claimed, claimed["_input_owner"]


def _finish_submission_claim(
    settings: Settings, cid: str, owner: object, **changes,
) -> None:
    if storage.finish_input_claim(
        settings.data_dir, cid, owner, **changes
    ) is None:
        raise HTTPException(status_code=409, detail="generation in progress")


def _result_fields(result: h3.H3Result) -> tuple[str, str | None]:
    if result.status == "succeeded":
        return "succeeded", None
    if result.status in {"submission_unknown", "h3_submitting"}:
        return "submission_unknown", "submission_unknown"
    if result.error_code in _KNOWN_TASK_ERRORS:
        return "resume_required", result.error_code
    if result.status == "h3_running":
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
            "h3_submitting",
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


def _run_generation(
    settings: Settings,
    cid: str,
    request: h3.H3Request,
    action: str,
) -> None:
    if action not in {"start", "resume", "retry"}:
        raise ValueError("invalid generation action")
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
        if action == "start":
            result = h3.start(request)
        elif action == "resume":
            result = h3.resume(request)
        else:
            result = h3.retry(request, request.client_request_id)
        if action == "resume" and result.status == "not_started":
            _mark_submission_unknown(settings, cid, generation)
            return
    except h3.H3Error as exc:
        if action == "resume" and isinstance(exc, h3.ReceiptError):
            _mark_submission_unknown(settings, cid, generation)
            return
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


def _bound_artifact_path(root: Path, binding: object) -> Path | None:
    if not isinstance(binding, dict):
        return None
    relative = binding.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        return None
    return root / relative


def _h3_validation_paths(workdir: Path) -> set[Path]:
    paths = {
        workdir / "generated.mp4",
        workdir / ".h3" / "session.json",
    }
    attempts = workdir / ".h3" / "attempts"
    try:
        paths.update(attempts.glob("*/attempt.json"))
    except OSError:
        paths.add(attempts)
    return paths


def _short_validation_paths(cdir: Path, meta: dict) -> set[Path]:
    paths = _h3_validation_paths(cdir)
    receipt_name = meta.get("prepared_input_receipt")
    if not isinstance(receipt_name, str) or receipt_name != Path(receipt_name).name:
        return paths
    receipt_path = cdir / receipt_name
    paths.add(receipt_path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return paths
    bindings = payload.get("bindings") if isinstance(payload, dict) else None
    if not isinstance(bindings, dict):
        return paths
    candidates = [
        bindings.get("source"),
        bindings.get("normalized_audio"),
        bindings.get("visual_prompt"),
        bindings.get("final_prompt"),
    ]
    keyframes = bindings.get("keyframes")
    if isinstance(keyframes, list):
        candidates.extend(keyframes)
    paths.update(
        path for binding in candidates
        if (path := _bound_artifact_path(cdir, binding)) is not None
    )
    return paths


def _long_validation_paths(cdir: Path, meta: dict) -> set[Path]:
    # Historical single-output attempts above the previous 10-second limit also reach the
    # long-video validator before their strict legacy fallback. Keep those
    # root H3 receipts in the same fingerprint as modern segment receipts.
    paths = _h3_validation_paths(cdir)
    paths.add(cdir / "stitch-receipt.json")
    receipt_name = meta.get("long_video_plan_receipt")
    if not isinstance(receipt_name, str) or receipt_name != Path(receipt_name).name:
        return paths
    receipt_path = cdir / receipt_name
    paths.add(receipt_path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return paths
    if not isinstance(payload, dict):
        return paths
    candidates = [payload.get("source")]
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return paths
    fit_mode = meta.get("fit_mode")
    aspect_ratio = meta.get("aspect_ratio")
    generation = meta.get("generation")
    fast_mode = (
        generation.get("fast_mode", False)
        if isinstance(generation, dict)
        else False
    )
    fit_layout = (
        generation.get("fit_layout")
        if isinstance(generation, dict)
        else None
    )
    if fit_layout == long_generation.FIT_LAYOUT_LEGACY:
        layout_dirs: tuple[str | None, ...] = (None,)
    elif (
        fit_layout == long_generation.FIT_LAYOUT_ASPECT
        and aspect_ratio in _ASPECT_RATIOS
    ):
        layout_dirs = (aspect_ratio.replace(":", "x"),)
    else:
        # Pre-marker generations may have either historical or semantic paths.
        # Fingerprint both; strict freeze_plan later selects exactly one complete
        # layout and rejects missing, mixed, or ambiguous frozen inputs.
        semantic_dir = (
            aspect_ratio if aspect_ratio in _ASPECT_RATIOS
            else h3.H3_DEFAULT_ASPECT_RATIO
        )
        layout_dirs = (None, semantic_dir.replace(":", "x"))
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        candidates.extend([
            raw.get("source"),
            raw.get("visual_prompt"),
            raw.get("final_prompt"),
        ])
        keyframes = raw.get("keyframes")
        if isinstance(keyframes, list):
            candidates.extend(keyframes)
        anchors = raw.get("anchors")
        if isinstance(anchors, list):
            candidates.extend(anchors)
        index = raw.get("index")
        if isinstance(index, int) and not isinstance(index, bool) and index > 0:
            workdir = cdir / "work" / "segments" / str(index)
            paths.update(_h3_validation_paths(workdir))
            tail = workdir / "work" / "generated_last.png"
            if not fast_mode:
                paths.add(tail)
            if fit_mode in {"crop", "pad"}:
                for layout_dir in layout_dirs:
                    fit_root = workdir / "work" / "h3_frames"
                    if layout_dir is not None:
                        fit_root = fit_root / layout_dir
                    fit_root = fit_root / fit_mode
                    for role, anchor_index in (("first", 0), ("end", 1)):
                        if isinstance(anchors, list) and len(anchors) > anchor_index:
                            bound = _bound_artifact_path(cdir, anchors[anchor_index])
                            if bound is not None:
                                paths.add(fit_root / role / bound.name)
                    if not fast_mode:
                        paths.add(fit_root / "continued" / tail.name)
    paths.update(
        path for binding in candidates
        if (path := _bound_artifact_path(cdir, binding)) is not None
    )
    return paths


def _generated_video_validation_fingerprint(cdir: Path, meta: dict) -> str | None:
    """Bind the cache only to fields and files read by strict validation."""
    try:
        generation = meta.get("generation")
        generation_binding = (
            {
                "status": generation.get("status"),
                "attempt": generation.get("attempt"),
                "client_request_id": generation.get("client_request_id"),
                "fit_layout": generation.get("fit_layout"),
                "fast_mode": generation.get("fast_mode", False),
                "segments": generation.get("segments"),
            }
            if isinstance(generation, dict)
            else generation
        )
        if _is_long_video(meta):
            binding = {
                "id": meta.get("id"),
                "duration_s": meta.get("duration_s"),
                "fit_mode": meta.get("fit_mode"),
                "aspect_ratio": meta.get("aspect_ratio"),
                "resolution": meta.get("resolution"),
                "dialogue_mode": meta.get("dialogue_mode"),
                "segments": meta.get("segments"),
                "frozen_plan_receipt": meta.get("frozen_plan_receipt"),
                "long_video_plan_receipt": meta.get("long_video_plan_receipt"),
                "generation": generation_binding,
            }
            paths = _long_validation_paths(cdir, meta)
        else:
            binding = {
                "id": meta.get("id"),
                "duration_s": meta.get("duration_s"),
                "prepared_dialogue": meta.get("prepared_dialogue"),
                "prepared_input_receipt": meta.get("prepared_input_receipt"),
                "dialogue_mode": meta.get("dialogue_mode"),
                "fit_mode": meta.get("fit_mode"),
                "aspect_ratio": meta.get("aspect_ratio"),
                "resolution": meta.get("resolution"),
                "generation": generation_binding,
            }
            paths = _short_validation_paths(cdir, meta)
        binding_bytes = json.dumps(
            binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(binding_bytes)
        for path in sorted(paths, key=lambda candidate: candidate.as_posix()):
            relative = path.relative_to(cdir).as_posix()
            try:
                stat = path.stat()
            except OSError:
                digest.update(relative.encode("utf-8") + b"\0missing\n")
                continue
            digest.update(relative.encode("utf-8"))
            digest.update(
                f"\0{stat.st_mode}\0{stat.st_ino}\0{stat.st_size}"
                f"\0{stat.st_mtime_ns}\0{stat.st_ctime_ns}\n".encode("ascii")
            )
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _has_valid_generated_video(settings: Settings, meta: dict) -> bool:
    generation = meta.get("generation")
    # Pre-H3 legacy conversations have no receipt to validate. Preserve their
    # historical visibility; every receipt-aware generation is fail closed.
    if generation is None:
        cid = meta.get("id")
        return (
            isinstance(cid, str)
            and (settings.data_dir / cid / "generated.mp4").is_file()
        )
    if not isinstance(generation, dict) or generation.get("status") != "succeeded":
        return False
    cid = meta.get("id")
    if not isinstance(cid, str):
        return False
    cdir = (settings.data_dir / cid).resolve()
    fingerprint = _generated_video_validation_fingerprint(cdir, meta)
    if fingerprint is None:
        return _validate_generated_video_uncached(settings, meta)
    identity = (
        str(settings.data_dir.resolve()),
        cid,
    )
    return _GENERATED_VIDEO_VALIDATION_CACHE.get_or_validate(
        identity,
        fingerprint,
        lambda: _generated_video_validation_fingerprint(cdir, meta),
        lambda: _validate_generated_video_uncached(settings, meta),
    )


def _validate_generated_video_uncached(settings: Settings, meta: dict) -> bool:
    generation = meta.get("generation")
    if not isinstance(generation, dict) or generation.get("status") != "succeeded":
        return False
    cid = meta.get("id")
    if not isinstance(cid, str):
        return False
    if _is_long_video(meta):
        expected = meta.get("frozen_plan_receipt")
        if (
            isinstance(expected, str)
            and long_generation.generation_segments_are_valid(
                meta.get("segments"), generation
            )
        ):
            try:
                plan = long_generation.freeze_plan(
                    settings.data_dir / cid,
                    meta,
                    expected,
                    meta.get("fit_mode"),
                    meta.get("dialogue_mode"),
                    prepare_fit=False,
                )
                reusable = long_generation.bound_reusable_segment_indices(
                    settings, cid, plan, generation
                )
                if (
                    reusable == frozenset(item.index for item in plan.segments)
                    and long_generation.stitched_output_is_reusable(
                        plan, meta.get("dialogue_mode")
                    )
                ):
                    return True
            except long_generation.LongGenerationError:
                pass
    else:
        try:
            request = _load_h3_request(settings, cid, meta)
            if h3.output_is_reusable(request):
                return True
        except (_SubmitError, h3.H3Error):
            pass
    return h3.legacy_succeeded_output_is_valid(
        settings.data_dir / cid,
        cid=cid,
        client_request_id=generation.get("client_request_id"),
        attempt=generation.get("attempt"),
        probe_timeout_s=_timeouts(settings).probe_s,
    )


def _resume_generation(settings: Settings, cid: str) -> None:
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None or _is_read_only(meta):
        return
    generation = meta.get("generation")
    if not isinstance(generation, dict):
        return
    recovering_missing_output = (
        generation.get("status") == "succeeded"
        and not _has_valid_generated_video(settings, meta)
    )
    if (
        generation.get("status") not in _GENERATION_ACTIVE
        and not _short_provider_failure_is_recoverable(generation)
        and not recovering_missing_output
    ):
        return
    if _is_legacy_generation_contract(generation):
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "failed",
                "error": "generation_path_removed",
                "stage": "h3",
            },
        )
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
    except h3.ReceiptError:
        _mark_submission_unknown(settings, cid, generation)
    except h3.H3Error as exc:
        if request is None or exc.code == "state_unavailable":
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
        if result.status == "not_started":
            _mark_submission_unknown(settings, cid, generation)
        else:
            _finish_generation(settings, cid, request, result)


def _resume_long_generation(settings: Settings, cid: str) -> None:
    meta = storage.load_meta(settings.data_dir, cid)
    generation = meta.get("generation") if meta else None
    if not isinstance(generation, dict) or not isinstance(generation.get("segments"), list):
        return
    if not long_generation.generation_segments_are_valid(
        meta.get("segments"), generation
    ):
        _mark_submission_unknown(settings, cid, generation)
        return
    expected = meta.get("frozen_plan_receipt")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        storage.update_meta(
            settings.data_dir, cid,
            generation={**generation, "status": "submission_unknown",
                        "error": "submission_unknown"},
        )
        return
    try:
        plan = long_generation.freeze_plan(
            settings.data_dir / cid, meta, expected,
            meta.get("fit_mode"), meta.get("dialogue_mode"),
            prepare_fit=False,
        )
    except Exception:
        storage.update_meta(
            settings.data_dir, cid,
            generation={**generation, "status": "submission_unknown",
                        "error": "submission_unknown"},
        )
        return
    long_generation.run(settings, cid, plan, startup=True)


def _reconcile_stale_submission(settings: Settings, cid: str, owner: object) -> None:
    """Release a half-frozen first submit without creating paid generation evidence."""
    meta = storage.load_submission_claim(settings.data_dir, cid, owner)
    if meta is None:
        return
    cdir = (settings.data_dir / cid).resolve()
    changes = {
        "status": "done",
        "error": "submission_recovery_required",
    }
    receipt = cdir / prepared_input.RECEIPT_FILENAME
    if receipt.is_file():
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            dialogue = payload["dialogue"]["lines"]
            frozen = prepared_input.load_prepared_input(
                cdir, receipt, expected_dialogue=dialogue
            )
            changes.update(
                prompt=frozen.prompt_text,
                prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
                dialogue_mode=frozen.dialogue_mode,
                prepared_dialogue=[dict(line) for line in frozen.dialogue],
                fit_mode=frozen.fit_mode,
                aspect_ratio=frozen.ratio,
                resolution=(
                    frozen.engine_request.get("h3", {}).get("resolution")
                    if isinstance(frozen.engine_request.get("h3"), dict)
                    else None
                ),
            )
        except (
            OSError,
            UnicodeDecodeError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            prepared_input.PreparedInputError,
        ):
            pass
    else:
        expected = long_generation.plan_receipt(cdir, meta)
        if expected is not None:
            try:
                long_generation.freeze_plan(
                    cdir,
                    meta,
                    expected,
                    "none",
                    meta.get("dialogue_mode", "auto"),
                )
                changes["frozen_plan_receipt"] = expected
            except long_generation.LongGenerationError:
                pass
    storage.finish_input_claim(settings.data_dir, cid, owner, **changes)


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

    @app.middleware("http")
    async def prevent_stale_web_contracts(request: Request, call_next):
        response = await call_next(request)
        if request.method in {"GET", "HEAD"} and request.url.path in _NO_STORE_WEB_PATHS:
            response.headers["Cache-Control"] = "no-store"
        return response

    limiter = _RateLimiter()
    codex_runner = CodexRunner(
        timeout_s=settings.codex_timeout_s, concurrency=settings.codex_concurrency
    )
    # 管道闸：同时处理的会话数上限；拿不到闸的会话保持 queued
    pipeline_sem = threading.Semaphore(settings.codex_concurrency)
    # 创建临界区：幂等查重 + queued 计数 + 建目录必须原子
    create_lock = threading.Lock()
    conversation_locks: dict[str, asyncio.Lock] = {}
    submit_locks = conversation_locks
    postprocess_locks = conversation_locks
    # MediaKit 后处理并行提交的进程级信号量：单进程内跨会话全局并发上限
    mediakit_sem = asyncio.Semaphore(settings.mediakit_concurrency)
    seedream_sem = asyncio.Semaphore(settings.seedream_concurrency)
    app.state.h3_resume_threads = []
    app.state.postprocess_recovery_tasks = []

    @app.on_event("startup")
    async def resume_postprocessing() -> None:
        for cid in postprocess.recover_running(settings):
            task = asyncio.create_task(
                postprocess.run_task(
                    settings, cid, mediakit_sem, seedream_sem
                ),
                name=f"postprocess-recover-{cid[:8]}",
            )
            app.state.postprocess_recovery_tasks.append(task)

    @app.on_event("startup")
    async def resume_h3_generations() -> None:
        if not settings.enable_h3_submit:
            return
        for meta in storage.list_conversations(settings.data_dir):
            generation = meta.get("generation")
            if (
                meta.get("schema_version") == 2
                and isinstance(generation, dict)
                and (
                    generation.get("status") in _GENERATION_ACTIVE
                    or _short_provider_failure_is_recoverable(generation)
                    or _long_provider_failure_is_recoverable(generation)
                    or (
                        generation.get("status") == "succeeded"
                        and not _has_valid_generated_video(settings, meta)
                    )
                )
            ):
                target = (
                    _resume_long_generation
                    if isinstance(generation.get("segments"), list)
                    else _resume_generation
                )
                thread = threading.Thread(
                    target=target,
                    args=(settings, meta["id"]),
                    daemon=True,
                    name=f"h3-resume-{meta['id'][:8]}",
                )
                app.state.h3_resume_threads.append(thread)
                thread.start()

    def run_pipeline_gated(cid: str, claimed_owner: object = None) -> None:
        with pipeline_sem:
            if claimed_owner is None:
                pipeline.run(settings, cid, codex_runner)
            else:
                pipeline.run(
                    settings, cid, codex_runner, claimed_owner=claimed_owner
                )

    @app.on_event("startup")
    async def recover_pipeline_inputs() -> None:
        for cid, owner in storage.claim_stale_input_reconciliations(
            settings.data_dir
        ):
            if owner["kind"] == "pipeline":
                pipeline.reconcile_stale_pipeline(settings, cid, owner)
            else:
                _reconcile_stale_submission(settings, cid, owner)
        if not settings.enable_pipeline:
            return
        for cid, owner in storage.claim_stale_pipeline_inputs(settings.data_dir):
            thread = threading.Thread(
                target=run_pipeline_gated,
                args=(cid, owner),
                daemon=True,
                name=f"pipeline-recover-{cid[:8]}",
            )
            thread.start()

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.post("/api/login")
    async def login(payload: dict):
        token = payload.get("token")
        if set(payload) - {"token"} or ("token" in payload and not isinstance(token, str)):
            raise HTTPException(status_code=422, detail="invalid_login_request")
        if not isinstance(token, str) or not hmac.compare_digest(token, settings.access_token):
            raise HTTPException(status_code=401, detail="invalid token")
        return {"ok": True}

    @app.get("/api/conversations", dependencies=[Depends(require_auth)])
    async def list_conversations():
        result = []
        for meta in storage.list_conversations(settings.data_dir):
            has_video = _has_valid_generated_video(settings, meta)
            result.append({
                "id": meta["id"],
                "title": meta["title"],
                "note": meta["note"],
                "status": meta["status"],
                "navigation_status": _navigation_status(meta, has_video=has_video),
                "created_at": meta["created_at"],
                "has_video": has_video,
            })
        return result

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
        form = await request.form()
        allowed_form_fields = {
            "file", "reference_url", "note", "client_request_id",
            "voice_mode", "target_language",
        }
        if set(form) - allowed_form_fields or any(
            len(form.getlist(key)) != 1 for key in form
        ):
            raise HTTPException(status_code=422, detail="invalid_create_request")
        reference_url = reference_url.strip()
        if (file is None) == (not reference_url):
            raise HTTPException(status_code=400, detail="provide exactly one of file or reference_url")
        client_request_id = client_request_id.strip()
        if client_request_id and not _CLIENT_REQUEST_ID_RE.match(client_request_id):
            raise HTTPException(status_code=400, detail="invalid client_request_id")
        # 口播转换：模式白名单 + 翻译必填目标语言；非 translate 忽略 target_language
        voice_mode = voice_mode.strip()
        target_language = target_language.strip()
        if voice_mode == "none" and not target_language:
            raise HTTPException(status_code=409, detail=_CLIENT_REFRESH_MESSAGE)
        if voice_mode not in ("keep", "rewrite", "translate"):
            raise HTTPException(status_code=422, detail=f"invalid voice_mode: {voice_mode}")
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
            video = storage.probe_video(dest)
            if _duration_exceeds_h3_limit(video.duration_s):
                storage.remove_conversation(settings.data_dir, meta["id"])
                raise HTTPException(
                    status_code=422,
                    detail=_duration_limit_detail(video.duration_s),
                )
            if (
                video.duration_s > long_video.SHORT_VIDEO_MAX_S
                and voice_mode != "keep"
            ):
                storage.remove_conversation(settings.data_dir, meta["id"])
                raise HTTPException(
                    status_code=422,
                    detail="long_video_audio_mode_unsupported",
                )
            storage.update_meta(
                settings.data_dir,
                meta["id"],
                duration_s=video.duration_s,
                source_width=video.width,
                source_height=video.height,
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
        has_video = _has_valid_generated_video(settings, meta)
        source_prompt, source_prompt_sha256 = _source_prompt_snapshot(cdir)
        try:
            effective_meta = meta
            if _is_long_video(meta) and not isinstance(meta.get("fit_profiles"), dict):
                effective_meta = {
                    **meta,
                    "fit_required": _long_fit_required(cdir, meta),
                }
            aspect_ratio, resolution = _generation_semantics(effective_meta)
            fit_profiles = _validated_fit_profiles(effective_meta)
            effective_fit_required = fit_profiles[aspect_ratio]["fit_required"]
        except _SubmitError:
            aspect_ratio = resolution = fit_profiles = None
            effective_fit_required = None
        optimization_prompts = image_optimization.public_prompts(meta, settings)
        public_segments = []
        for raw_segment in meta.get("segments", []):
            segment = dict(raw_segment) if isinstance(raw_segment, dict) else raw_segment
            if isinstance(segment, dict) and segment.get("index") in optimization_prompts:
                segment["image_optimization_prompt"] = optimization_prompts[segment["index"]]
            public_segments.append(segment)
        capabilities = {
            "remove_subtitle": bool(settings.enable_mediakit_erase),
            "remove_brand": bool(settings.enable_mediakit_erase),
            "optimize_image": bool(os.environ.get("ARK_API_KEY", "").strip()),
        }
        result = {
            "id": meta["id"],
            "title": meta["title"],
            "note": meta["note"],
            "status": meta["status"],
            "navigation_status": _navigation_status(meta, has_video=has_video),
            "error": meta["error"],
            "created_at": meta["created_at"],
            "updated_at": meta["updated_at"],
            "keyframes": meta.get("keyframes", []),
            "prompt": meta.get("prompt"),
            "source_prompt": source_prompt,
            "source_prompt_sha256": source_prompt_sha256,
            "segments": public_segments,
            "voice_lines": meta.get("voice_lines", []),
            "read_only": _is_read_only(meta),
            "duration_s": meta.get("duration_s"),
            "fit_required": effective_fit_required,
            "fit_mode": meta.get("fit_mode"),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "fit_profiles": fit_profiles,
            "dialogue": _public_dialogue(meta),
            "receipt_version": _receipt_version(cdir, meta),
            "generation": _public_generation(meta, cdir, settings),
            "has_source": any(cdir.glob("source.*")),
            "has_video": has_video,
            "submit_enabled": settings.enable_h3_submit,
            "postprocess": postprocess.public_state(meta.get("postprocess")),
            "postprocess_capabilities": capabilities,
            "postprocess_enabled": any(capabilities.values()),
        }
        if not _is_long_video(meta):
            result["image_optimization_prompt"] = optimization_prompts.get(0)
        if _is_long_video(meta):
            result["plan_receipt"] = long_generation.plan_receipt(cdir, meta)
            segments = meta.get("segments")
            result["segment_count"] = len(segments) if isinstance(segments, list) else 0
        return result

    @app.patch(
        "/api/conversations/{cid}/image-optimization-prompt",
        dependencies=[Depends(require_auth)],
    )
    async def edit_image_optimization_prompt(cid: str, payload: dict):
        required = {"confirm", "segment_index", "expected_sha256", "prompt"}
        if set(payload) != required:
            raise HTTPException(status_code=422, detail="invalid_image_optimization_prompt_request")
        if payload.get("confirm") is not True:
            raise HTTPException(status_code=409, detail="confirmation required")
        segment_index = payload.get("segment_index")
        expected = payload.get("expected_sha256")
        prompt = payload.get("prompt")
        if (
            isinstance(segment_index, bool) or not isinstance(segment_index, int)
            or not isinstance(expected, str) or not isinstance(prompt, str)
        ):
            raise HTTPException(status_code=422, detail="invalid_image_optimization_prompt_request")
        lock = postprocess_locks.setdefault(cid, asyncio.Lock())
        async with lock:
            result: dict[str, str] = {}

            def mutate(meta: dict) -> None:
                frozen = image_optimization.replace(
                    meta, settings, segment_index, expected, prompt
                )
                meta["_image_optimization"] = frozen
                result.update(
                    image_optimization.public_prompts(meta, settings)[segment_index]
                )

            try:
                updated = storage.mutate_meta(settings.data_dir, cid, mutate)
            except image_optimization.ImageOptimizationError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
            if updated is None:
                raise HTTPException(status_code=404, detail="not found")
            return result

    @app.patch(
        "/api/conversations/{cid}/prompt",
        dependencies=[Depends(require_auth)],
    )
    async def edit_source_prompt(cid: str, payload: dict):
        if set(payload) != {"confirm", "expected_sha256", "prompt"}:
            raise HTTPException(status_code=422, detail="invalid_prompt_request")
        if payload.get("confirm") is not True:
            raise HTTPException(status_code=409, detail="confirmation required")
        expected_sha256 = payload.get("expected_sha256")
        prompt = payload.get("prompt")
        if not isinstance(expected_sha256, str) or not isinstance(prompt, str):
            raise HTTPException(status_code=422, detail="invalid_prompt_request")
        lock = submit_locks.setdefault(cid, asyncio.Lock())
        async with lock:
            meta = storage.load_meta(settings.data_dir, cid)
            if meta is None:
                raise HTTPException(status_code=404, detail="not found")
            if _is_read_only(meta):
                raise HTTPException(status_code=409, detail="read_only")
            if meta.get("status") != "done":
                raise HTTPException(status_code=409, detail="artifacts not ready")
            if isinstance(meta.get("generation"), dict) or (
                settings.data_dir / cid / ".h3" / "session.json"
            ).exists():
                raise HTTPException(status_code=409, detail="prompt_frozen")
            try:
                updated, digest, final_prompt = await asyncio.to_thread(
                    _replace_source_prompt,
                    settings,
                    cid,
                    meta,
                    expected_sha256,
                    prompt,
                )
            except _SubmitError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        return {"prompt": updated, "sha256": digest, "final_prompt": final_prompt}

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
        if _is_long_video(meta):
            try:
                effective_meta = {
                    **meta,
                    "fit_required": _long_fit_required(
                        settings.data_dir / cid, meta
                    ),
                }
                (
                    request_id,
                    fit_mode,
                    dialogue_mode,
                    expected_receipt,
                    aspect_ratio,
                    resolution,
                    fast_mode,
                ) = (
                    _validate_long_submit_payload(effective_meta, payload)
                )
            except _SubmitError as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
            if meta.get("status") != "done":
                raise HTTPException(status_code=409, detail="artifacts not ready")
            post_state = meta.get("postprocess")
            if isinstance(post_state, dict) and post_state.get("status") != "done":
                raise HTTPException(status_code=409, detail="postprocess_not_ready")
            if _duration_exceeds_h3_limit(meta.get("duration_s")):
                raise HTTPException(
                    status_code=422,
                    detail=_duration_limit_detail(float(meta["duration_s"])),
                )
            if not _credentials_ready(settings):
                raise HTTPException(status_code=503, detail="h3_credentials_missing")
            lock = submit_locks.setdefault(cid, asyncio.Lock())
            async with lock:
                meta = storage.load_meta(settings.data_dir, cid)
                if meta is None:
                    raise HTTPException(status_code=404, detail="not found")
                post_state = meta.get("postprocess")
                if isinstance(post_state, dict) and post_state.get("status") != "done":
                    raise HTTPException(status_code=409, detail="postprocess_not_ready")
                old = meta.get("generation")
                previous_status = old.get("status") if isinstance(old, dict) else None
                previous_id = old.get("client_request_id") if isinstance(old, dict) else None
                if isinstance(old, dict) and not long_generation.generation_segments_are_valid(
                    meta.get("segments"), old
                ):
                    _mark_submission_unknown(settings, cid, old)
                    raise HTTPException(
                        status_code=409, detail="submission_outcome_unknown"
                    )
                same_parameters = (
                    meta.get("dialogue_mode") == dialogue_mode
                    and meta.get("fit_mode") == fit_mode
                    and meta.get("frozen_plan_receipt") == expected_receipt
                    and _generation_semantics(meta)
                    == (aspect_ratio, resolution)
                    and (
                        old.get("fast_mode", False) == fast_mode
                        if isinstance(old, dict)
                        else True
                    )
                )
                # Active replays are pure idempotent reads.  In particular they
                # must not rewrite fitted inputs or enqueue a second coordinator.
                if previous_status in {"queued", "running"}:
                    if previous_id != request_id:
                        raise HTTPException(status_code=409, detail="generation in progress")
                    if not same_parameters:
                        raise HTTPException(status_code=409, detail="resume_parameters_changed")
                    return {"status": previous_status, "attempt": old.get("attempt")}
                if previous_status == "submission_unknown":
                    raise HTTPException(status_code=409, detail="submission_outcome_unknown")
                stitch_retry = (
                    previous_status == "failed"
                    and old.get("stage") == "stitch"
                    and previous_id == request_id
                    and same_parameters
                )
                if _has_valid_generated_video(settings, meta):
                    if previous_status == "succeeded":
                        if previous_id != request_id or not same_parameters:
                            raise HTTPException(
                                status_code=409, detail="already submitted"
                            )
                        return {
                            "status": "succeeded",
                            "attempt": old.get("attempt"),
                        }
                    if not stitch_retry:
                        raise HTTPException(status_code=409, detail="already submitted")
                if previous_status == "failed" and not same_parameters:
                    raise HTTPException(status_code=409, detail="resume_parameters_changed")
                claim_owner = None
                if not isinstance(old, dict):
                    meta, claim_owner = _claim_first_submission(
                        settings, cid, request_id
                    )
                try:
                    plan = await asyncio.to_thread(
                        long_generation.freeze_plan,
                        settings.data_dir / cid,
                        meta,
                        expected_receipt,
                        fit_mode,
                        dialogue_mode,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                        prepare_fit=not isinstance(old, dict),
                    )
                except long_generation.LongGenerationError as exc:
                    if claim_owner:
                        _finish_submission_claim(settings, cid, claim_owner)
                    if previous_status in {"resume_required", "succeeded"} or (
                        previous_status == "failed"
                        and isinstance(old, dict)
                        and old.get("stage") == "stitch"
                    ):
                        _mark_submission_unknown(settings, cid, old)
                        raise HTTPException(
                            status_code=409,
                            detail="submission_outcome_unknown",
                        ) from exc
                    raise HTTPException(status_code=exc.status, detail=exc.code) from exc
                except BaseException:
                    if claim_owner:
                        _finish_submission_claim(settings, cid, claim_owner)
                    raise
                if previous_status == "resume_required":
                    if previous_id != request_id:
                        raise HTTPException(status_code=409, detail="resume_request_id_mismatch")
                    if not same_parameters:
                        raise HTTPException(status_code=409, detail="resume_parameters_changed")
                    claimed_generation = {
                        **old, "status": "queued", "error": None
                    }
                    claimed = storage.update_meta(
                        settings.data_dir, cid,
                        generation=claimed_generation,
                    )
                    if claimed is None:
                        raise HTTPException(status_code=404, detail="not found")
                    background_tasks.add_task(long_generation.run, settings, cid, plan)
                    return {
                        "status": "queued",
                        "attempt": claimed_generation.get("attempt"),
                    }
                if previous_status == "succeeded":
                    if previous_id != request_id or not same_parameters:
                        raise HTTPException(
                            status_code=409, detail="resume_parameters_changed"
                        )
                    storage.update_meta(
                        settings.data_dir,
                        cid,
                        generation={**old, "status": "queued", "error": None},
                    )
                    background_tasks.add_task(
                        long_generation.run, settings, cid, plan
                    )
                    return {"status": "queued", "attempt": old.get("attempt")}
                if previous_status == "failed" and old.get("stage") == "stitch":
                    if previous_id != request_id or not same_parameters:
                        raise HTTPException(status_code=409, detail="resume_parameters_changed")
                    updated = {**old, "status": "queued", "error": None}
                    storage.update_meta(settings.data_dir, cid, generation=updated)
                    background_tasks.add_task(long_generation.run, settings, cid, plan)
                    return {"status": "queued", "attempt": old.get("attempt")}
                if previous_status == "failed" and previous_id == request_id:
                    raise HTTPException(status_code=409, detail="new client_request_id required")
                previous_attempt = old.get("attempt", 0) if isinstance(old, dict) else 0
                if isinstance(previous_attempt, bool) or not isinstance(previous_attempt, int):
                    raise HTTPException(status_code=409, detail="generation_state_invalid")
                attempt = previous_attempt + 1
                generation = long_generation.initial_generation(
                    settings, cid, plan, request_id, attempt,
                    old if isinstance(old, dict) else None,
                    fast_mode=fast_mode,
                )
                changes = dict(
                    dialogue_mode=dialogue_mode,
                    voice_lines=[],
                    prepared_dialogue=[],
                    fit_mode=fit_mode,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    frozen_plan_receipt=expected_receipt,
                    generation=generation,
                )
                if claim_owner:
                    _finish_submission_claim(
                        settings, cid, claim_owner, **changes
                    )
                else:
                    storage.update_meta(settings.data_dir, cid, **changes)
                background_tasks.add_task(long_generation.run, settings, cid, plan)
            return {"status": "queued", "attempt": attempt}
        try:
            (
                request_id,
                fit_mode,
                aspect_ratio,
                resolution,
                dialogue,
            ) = _validate_submit_payload(meta, payload)
        except _SubmitError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        if meta.get("status") != "done":
            raise HTTPException(status_code=409, detail="artifacts not ready")
        post_state = meta.get("postprocess")
        if isinstance(post_state, dict) and post_state.get("status") != "done":
            raise HTTPException(status_code=409, detail="postprocess_not_ready")
        if _duration_exceeds_h3_limit(meta.get("duration_s")):
            raise HTTPException(
                status_code=422,
                detail=_duration_limit_detail(float(meta["duration_s"])),
            )
        if not _credentials_ready(settings):
            raise HTTPException(status_code=503, detail="h3_credentials_missing")

        lock = submit_locks.setdefault(cid, asyncio.Lock())
        async with lock:
            meta = storage.load_meta(settings.data_dir, cid)
            if meta is None:
                raise HTTPException(status_code=404, detail="not found")
            post_state = meta.get("postprocess")
            if isinstance(post_state, dict) and post_state.get("status") != "done":
                raise HTTPException(status_code=409, detail="postprocess_not_ready")
            generation = meta.get("generation")
            if (
                (settings.data_dir / cid / "generated.mp4").is_file()
                and not (
                    isinstance(generation, dict)
                    and generation.get("status") == "succeeded"
                )
            ):
                raise HTTPException(status_code=409, detail="already submitted")
            previous_status = None
            if isinstance(generation, dict):
                previous_status = _effective_generation_status(generation)
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
                        if not _short_generation_parameters_match(
                            meta,
                            dialogue_mode=payload["dialogue_mode"],
                            dialogue=dialogue,
                            fit_mode=fit_mode,
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail="resume_parameters_changed",
                            )
                        if previous_status == "succeeded":
                            try:
                                request = await asyncio.to_thread(
                                    _load_h3_request, settings, cid, meta
                                )
                            except (_SubmitError, h3.H3Error) as exc:
                                _mark_submission_unknown(settings, cid, generation)
                                raise HTTPException(
                                    status_code=409,
                                    detail="submission_outcome_unknown",
                                ) from exc
                            if h3.output_is_reusable(request):
                                return {
                                    "status": "succeeded",
                                    "attempt": generation.get("attempt"),
                                }
                            storage.update_meta(
                                settings.data_dir,
                                cid,
                                generation={
                                    **generation,
                                    "status": "queued",
                                    "error": None,
                                    "stage": "h3",
                                },
                            )
                            background_tasks.add_task(
                                _run_generation,
                                settings,
                                cid,
                                request,
                                "resume",
                            )
                            return {
                                "status": "queued",
                                "attempt": generation.get("attempt"),
                            }
                        return {
                            "status": previous_status,
                            "attempt": generation.get("attempt"),
                        }
                    detail = "already submitted" if previous_status == "succeeded" else "generation in progress"
                    raise HTTPException(status_code=409, detail=detail)
                if previous_status in _GENERATION_RETRYABLE and previous_id == request_id:
                    raise HTTPException(status_code=409, detail="new client_request_id required")
                legacy_pre_h3 = (
                    previous_status in _GENERATION_RETRYABLE
                    and _is_legacy_generation_contract(generation)
                )
                if legacy_pre_h3 and not h3.legacy_h3_is_provably_unsubmitted(
                    settings.data_dir / cid,
                    cid=cid,
                    attempt=generation.get("attempt"),
                    client_request_id=previous_id,
                ):
                    _mark_submission_unknown(settings, cid, generation)
                    raise HTTPException(
                        status_code=409, detail="submission_outcome_unknown"
                    )
                if (
                    previous_status in _GENERATION_RETRYABLE
                    and not legacy_pre_h3
                    and not _short_generation_parameters_match(
                        meta,
                        dialogue_mode=payload["dialogue_mode"],
                        dialogue=dialogue,
                        fit_mode=fit_mode,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                    )
                ):
                    raise HTTPException(
                        status_code=409, detail="resume_parameters_changed"
                    )
                if previous_status in _GENERATION_RESUMABLE:
                    if previous_id != request_id:
                        raise HTTPException(status_code=409, detail="resume_request_id_mismatch")
                    if not _short_generation_parameters_match(
                        meta,
                        dialogue_mode=payload["dialogue_mode"],
                        dialogue=dialogue,
                        fit_mode=fit_mode,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
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
                        _mark_submission_unknown(settings, cid, generation)
                        raise HTTPException(
                            status_code=409, detail="submission_outcome_unknown"
                        ) from exc
                    except h3.H3Error as exc:
                        _mark_submission_unknown(settings, cid, generation)
                        raise HTTPException(
                            status_code=409, detail="submission_outcome_unknown"
                        ) from exc
                    storage.update_meta(
                        settings.data_dir,
                        cid,
                        generation={
                            **generation,
                            "status": "queued",
                            "error": None,
                            "stage": "h3",
                        },
                    )
                    background_tasks.add_task(
                        _run_generation, settings, cid, request, "resume"
                    )
                    return {"status": "queued", "attempt": previous_attempt}
            action = (
                "retry"
                if isinstance(generation, dict)
                and previous_status in _GENERATION_RETRYABLE
                else "start"
            )
            previous_attempt = generation.get("attempt") if isinstance(generation, dict) else 0
            if (
                isinstance(previous_attempt, bool)
                or not isinstance(previous_attempt, int)
                or previous_attempt < 0
            ):
                raise HTTPException(status_code=409, detail="generation_state_invalid")
            attempt = previous_attempt + 1
            dialogue_mode = payload["dialogue_mode"]
            claim_owner = None
            if not isinstance(generation, dict):
                meta, claim_owner = _claim_first_submission(
                    settings, cid, request_id
                )
            try:
                request = await asyncio.to_thread(
                    _freeze_submission,
                    settings,
                    cid,
                    meta,
                    request_id,
                    dialogue_mode,
                    fit_mode,
                    aspect_ratio,
                    resolution,
                    dialogue,
                )
            except _SubmitError as exc:
                if claim_owner:
                    _finish_submission_claim(settings, cid, claim_owner)
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
            except h3.H3Error as exc:
                if claim_owner:
                    _finish_submission_claim(settings, cid, claim_owner)
                raise HTTPException(status_code=503, detail="h3_configuration_invalid") from exc
            except BaseException:
                if claim_owner:
                    _finish_submission_claim(settings, cid, claim_owner)
                raise
            bare_lines = [
                {key: line[key] for key in ("text", "start_s", "end_s")}
                for line in dialogue
            ]
            generation = {
                "status": "queued",
                "error": None,
                "attempt": attempt,
                "client_request_id": request_id,
                "stage": "h3",
            }
            changes = dict(
                dialogue_mode=dialogue_mode,
                voice_lines=bare_lines,
                prompt=request.prompt,
                prepared_dialogue=[dict(line) for line in dialogue],
                prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
                fit_mode=fit_mode,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                generation=generation,
            )
            if claim_owner:
                _finish_submission_claim(
                    settings, cid, claim_owner, **changes
                )
            else:
                storage.update_meta(settings.data_dir, cid, **changes)
            background_tasks.add_task(_run_generation, settings, cid, request, action)
        return {"status": "queued", "attempt": attempt}

    @app.post("/api/conversations/{cid}/postprocess", dependencies=[Depends(require_auth)])
    async def postprocess_conversation(
        cid: str, payload: dict, background_tasks: BackgroundTasks
    ):
        if not settings.enable_mediakit_erase and not os.environ.get("ARK_API_KEY", "").strip():
            raise HTTPException(status_code=501, detail="MediaKit erase is disabled.")
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None:
            raise HTTPException(status_code=404, detail="not found")
        if _is_read_only(meta):
            raise HTTPException(status_code=409, detail="read_only")
        try:
            await postprocess.start(settings, cid, payload, postprocess_locks)
        except postprocess.PostprocessError as e:
            raise HTTPException(status_code=e.status, detail=e.detail) from e
        background_tasks.add_task(
            postprocess.run_task, settings, cid, mediakit_sem, seedream_sem
        )
        return {"status": "running", "frames": []}

    @app.post(
        "/api/conversations/{cid}/postprocess/segments/{index}/retry",
        dependencies=[Depends(require_auth)],
    )
    async def retry_postprocess_segment(
        cid: str, index: int, payload: dict, background_tasks: BackgroundTasks
    ):
        try:
            await postprocess.retry_segment(
                settings, cid, index, payload, postprocess_locks
            )
        except postprocess.PostprocessError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        background_tasks.add_task(
            postprocess.run_task, settings, cid, mediakit_sem, seedream_sem, {index}
        )
        return {"status": "running", "segment_index": index}

    web = Path(__file__).resolve().parent.parent / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app


app = create_app(get_settings())
