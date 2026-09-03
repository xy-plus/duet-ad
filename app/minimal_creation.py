"""Pure validation and normalization for the minimal creation v1 contract.

HTTP multipart handling, media validation, persistence, and pipeline scheduling are
deliberately outside this module.  Keeping the contract boundary pure makes it
possible to reject a request before any durable or paid side effect occurs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


TARGET_LANGUAGE_MAX_CHARS = 80
REPLACEMENT_MAX_BYTES = 10 * 1024 * 1024
REPLACEMENT_MAX_INSTRUCTION_CHARS = 1_000

ASPECT_RATIOS = ("16:9", "9:16")
RESOLUTIONS = ("480p", "768p")
REPLACEMENT_IMAGE_FIELD = "replacement_image"
REPLACEMENT_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")

_TOP_LEVEL_KEYS = {
    "version",
    "output",
    "processing",
    "dialogue",
    "replacement_guidance",
}
_OUTPUT_KEYS = {"aspect_ratio", "resolution", "fit_mode"}
_PROCESSING_KEYS = {"optimize_image", "remove_subtitle", "remove_logo"}
_DIALOGUE_KEYS = {"mode", "target_language"}
_GUIDANCE_KEYS = {"instruction", "image_field"}


class MinimalCreationError(ValueError):
    """A stable, public-safe v1 contract failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        status_code: int = 422,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.field = field

    def detail(self) -> dict[str, str]:
        result = {"code": self.code, "message": self.message}
        if self.field is not None:
            result["field"] = self.field
        return result


@dataclass(frozen=True, slots=True)
class ParsedGenerationRequest:
    """Normalized public request plus the existing pipeline's processing names."""

    effective_request: dict[str, Any]
    generation_request_sha256: str
    internal_processing: dict[str, bool]


class _DuplicateJsonKey(ValueError):
    pass


def _fail(code: str, message: str, *, field: str | None = None) -> None:
    raise MinimalCreationError(code, message, field=field)


def public_error_detail(exc: MinimalCreationError) -> dict[str, str]:
    """Return the object expected at FastAPI's public ``detail`` boundary."""

    if not isinstance(exc, MinimalCreationError):
        raise TypeError("exc must be MinimalCreationError")
    return exc.detail()


def capability() -> dict[str, Any]:
    """Return a fresh, complete minimal-creation v1 capability object."""

    return {
        "supported": True,
        "version": 1,
        "endpoint": "/api/conversations",
        "encoding": "multipart/form-data",
        "request_field": "generation_request",
        "replacement_image_field": REPLACEMENT_IMAGE_FIELD,
        "aspect_ratios": list(ASPECT_RATIOS),
        "resolutions": list(RESOLUTIONS),
        "defaults": {
            "fit_mode": "auto",
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "auto_rewrite",
            "translation": True,
        },
        "replacement": {
            "supported": True,
            "accept": list(REPLACEMENT_MEDIA_TYPES),
            "max_bytes": REPLACEMENT_MAX_BYTES,
            "max_instruction_chars": REPLACEMENT_MAX_INSTRUCTION_CHARS,
        },
    }


def utf16_code_units(value: str) -> int:
    """Count the same code units as browser ``String.length``."""

    if not isinstance(value, str):
        raise TypeError("value must be str")
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for durable v1 request identity."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        _fail(
            "invalid_generation_request",
            "generation_request 包含无效值",
            field="generation_request",
        )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite number is not JSON")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _decode_json(raw: str | bytes) -> Mapping[str, Any]:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            _fail(
                "invalid_generation_request_json",
                "generation_request 必须是 UTF-8 JSON 对象",
                field="generation_request",
            )
    elif isinstance(raw, str):
        text = raw
    else:
        _fail(
            "invalid_generation_request_json",
            "generation_request 必须是 UTF-8 JSON 对象",
            field="generation_request",
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey:
        _fail(
            "invalid_generation_request",
            "generation_request 不允许重复字段",
            field="generation_request",
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        _fail(
            "invalid_generation_request_json",
            "generation_request 必须是 UTF-8 JSON 对象",
            field="generation_request",
        )
    if not isinstance(value, dict):
        _fail(
            "invalid_generation_request_json",
            "generation_request 必须是 UTF-8 JSON 对象",
            field="generation_request",
        )
    return value


def _exact_object(
    value: Any,
    keys: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(
            "invalid_generation_request",
            "generation_request 结构不符合 v1 合同",
            field=field,
        )
    return value


def _normalize_output(value: Any) -> dict[str, str]:
    output = _exact_object(
        value,
        _OUTPUT_KEYS,
        field="generation_request.output",
    )
    if (
        output["aspect_ratio"] not in ASPECT_RATIOS
        or output["resolution"] not in RESOLUTIONS
        or output["fit_mode"] != "auto"
    ):
        _fail(
            "invalid_output_config",
            "输出参数不符合 v1 合同",
            field="generation_request.output",
        )
    return {
        "aspect_ratio": output["aspect_ratio"],
        "resolution": output["resolution"],
        "fit_mode": output["fit_mode"],
    }


def _normalize_processing(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        _fail(
            "processing_must_be_enabled",
            "图像处理选项必须全部启用",
            field="generation_request.processing",
        )
    if set(value) - _PROCESSING_KEYS:
        _fail(
            "invalid_generation_request",
            "generation_request 结构不符合 v1 合同",
            field="generation_request.processing",
        )
    if set(value) != _PROCESSING_KEYS or any(
        not isinstance(value[key], bool) or value[key] is not True
        for key in _PROCESSING_KEYS
        if key in value
    ):
        _fail(
            "processing_must_be_enabled",
            "图像处理选项必须全部启用",
            field="generation_request.processing",
        )
    return {
        "optimize_image": True,
        "remove_subtitle": True,
        "remove_logo": True,
    }


def _normalize_dialogue(value: Any) -> dict[str, Any]:
    dialogue = _exact_object(
        value,
        _DIALOGUE_KEYS,
        field="generation_request.dialogue",
    )
    if dialogue["mode"] != "auto_rewrite":
        _fail(
            "invalid_generation_request",
            "dialogue.mode 必须为 auto_rewrite",
            field="generation_request.dialogue.mode",
        )
    target_language = dialogue["target_language"]
    if not isinstance(target_language, str) or not target_language.strip():
        _fail(
            "target_language_required",
            "请填写目标语言",
            field="generation_request.dialogue.target_language",
        )
    target_language = target_language.strip()
    if utf16_code_units(target_language) > TARGET_LANGUAGE_MAX_CHARS:
        _fail(
            "target_language_too_long",
            "目标语言超过长度限制",
            field="generation_request.dialogue.target_language",
        )
    return {
        "mode": "auto_rewrite",
        "target_language": target_language,
    }


def _normalize_guidance(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or "instruction" not in value
        or set(value) - _GUIDANCE_KEYS
    ):
        _fail(
            "invalid_generation_request",
            "generation_request 结构不符合 v1 合同",
            field="generation_request.replacement_guidance",
        )
    guidance = value
    instruction = guidance["instruction"]
    if not isinstance(instruction, str):
        _fail(
            "invalid_generation_request",
            "replacement_guidance.instruction 必须是字符串",
            field="generation_request.replacement_guidance.instruction",
        )
    instruction = instruction.strip()
    if not instruction:
        _fail(
            "replacement_instruction_required",
            "请填写替换说明",
            field="generation_request.replacement_guidance.instruction",
        )
    if utf16_code_units(instruction) > REPLACEMENT_MAX_INSTRUCTION_CHARS:
        _fail(
            "replacement_instruction_too_long",
            "替换说明超过长度限制",
            field="generation_request.replacement_guidance.instruction",
        )
    return {
        "instruction": instruction,
        "image_field": REPLACEMENT_IMAGE_FIELD,
    }


def normalize_generation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the canonical public v1 request object."""

    request = _exact_object(value, _TOP_LEVEL_KEYS, field="generation_request")
    version = request["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        _fail(
            "unsupported_generation_request_version",
            "仅支持 generation_request version 1",
            field="generation_request.version",
        )
    return {
        "version": 1,
        "output": _normalize_output(request["output"]),
        "processing": _normalize_processing(request["processing"]),
        "dialogue": _normalize_dialogue(request["dialogue"]),
        "replacement_guidance": _normalize_guidance(
            request["replacement_guidance"]
        ),
    }


def parse_generation_request(raw: str | bytes) -> ParsedGenerationRequest:
    """Parse strict UTF-8 JSON, normalize it, and derive its immutable identity."""

    effective_request = normalize_generation_request(_decode_json(raw))
    processing = effective_request["processing"]
    internal_processing = {
        "optimize_image": processing["optimize_image"],
        "remove_subtitle": processing["remove_subtitle"],
        "remove_watermark": processing["remove_logo"],
    }
    return ParsedGenerationRequest(
        effective_request=effective_request,
        generation_request_sha256=canonical_json_sha256(effective_request),
        internal_processing=internal_processing,
    )


def validate_replacement_pair(
    parsed: ParsedGenerationRequest,
    *,
    replacement_image_present: bool,
) -> None:
    """Enforce the atomic guidance/image presence rule before persistence."""

    if not isinstance(parsed, ParsedGenerationRequest) or not isinstance(
        replacement_image_present, bool
    ):
        raise TypeError("invalid replacement pair arguments")
    guidance_present = parsed.effective_request["replacement_guidance"] is not None
    if guidance_present and not replacement_image_present:
        _fail(
            "replacement_image_required",
            "参考图与替换说明需要一起提供",
            field=REPLACEMENT_IMAGE_FIELD,
        )
    if replacement_image_present and not guidance_present:
        _fail(
            "replacement_guidance_required",
            "参考图与替换说明需要一起提供",
            field="generation_request.replacement_guidance",
        )
