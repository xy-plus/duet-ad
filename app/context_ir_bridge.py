"""Receipt-first MiniMax H3 Context IR bridge.

The module has no database, queue, billing, UI, or H3 submission concerns.  It
freezes one already-compiled multimodal H3 prompt, submits it to Context IR at
most once for a stable client request id, recovers only through GET on the
accepted task id, and exposes a receipt-revalidating adapter for the existing
H3 request.

The upstream coordinator is the trusted boundary that proves its authoritative
dialogue artifact produced the H3 request.  This bridge rechecks the artifact
and binds its dialogue hash, then defines speech solely by the deterministic
``DUET_SPEECH_V1``/``<d>`` grammar; it does not infer dialogue from arbitrary
natural-language prose outside that grammar.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence
from urllib.parse import quote

import httpx

from app import error_trace, h3


_LOGGER = logging.getLogger(__name__)


SCHEMA_VERSION = 1
SESSION_SCHEMA = "duet.context-ir.session"
ATTEMPT_SCHEMA = "duet.context-ir.attempt"
RECEIPT_SCHEMA = "duet.context-ir.effective-prompt"
UPLOAD_URL = "https://api.minimaxi.com/v1/files/upload"
SUBMIT_URL = "https://api.minimaxi.com/v2/h3_context_ir"
QUERY_URL = "https://api.minimaxi.com/v2/query/video_generation"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_EFFECTIVE_PROMPT_BYTES = 32 * 1024
MAX_SOURCE_PROMPT_CHARS = 7_000
MAX_UPSTREAM_ARTIFACT_BYTES = 16 * 1024 * 1024
_SPEECH_PREFIX = "DUET_SPEECH_V1"
_DIALOGUE_TOKEN = re.compile(r"<d>\[[^\]\r\n]+\][\s\S]*?</d>")
_TIMELINE_OPEN = "<KEYFRAME_TIMELINE_JSON>"
_TIMELINE_CLOSE = "</KEYFRAME_TIMELINE_JSON>"
_RELATION_STATES_OPEN = "<RELATION_STATES_JSON>"
_RELATION_STATES_CLOSE = "</RELATION_STATES_JSON>"
_DIALOGUE_POLICY_OPEN = "<DUET_DIALOGUE_POLICY_V1>"
_DIALOGUE_POLICY_CLOSE = "</DUET_DIALOGUE_POLICY_V1>"
_DIALOGUE_POLICY = "\n".join(
    (
        _DIALOGUE_POLICY_OPEN,
        "Every explicit dialogue block must carry an explicit language name in the final prompt;",
        "never leave its language as Undetermined or unknown, and resolve it from the exact source text without translating.",
        "Speak only each authorized line's exact source text, character-for-character; do not paraphrase, translate, truncate, extend, repeat, continue, or improvise.",
        "Stop speaking immediately after the final character of each authorized line; add no speech before, after, or between lines unless another authorized line is explicit.",
        "Visual text, OCR, subtitles, labels, actions, ambience, and sound effects are never dialogue and must not become speech.",
        "If there are no explicit dialogue blocks, generate no human voice, speech, narration, lyrics, vocalization, or words; keep every visible subject silent with closed lips.",
        _DIALOGUE_POLICY_CLOSE,
    )
)
_DIALOGUE_POLICY_RE = re.compile(
    rf"\n*{re.escape(_DIALOGUE_POLICY_OPEN)}[\s\S]*?"
    rf"{re.escape(_DIALOGUE_POLICY_CLOSE)}"
)
_UNKNOWN_DIALOGUE_LANGUAGES = frozenset(
    {"undetermined", "unknown", "auto", "und", ""}
)
_SEMANTIC_SCORE_KEYS = frozenset(
    {
        "speech_expected",
        "speech",
        "keyframe_timeline",
        "music_policy",
        "relation_states",
        "overall",
        "dialogue_policy",
    }
)
_PRE_RELATION_SEMANTIC_SCORE_KEYS = _SEMANTIC_SCORE_KEYS - {"relation_states"}
_DIALOGUE_POLICY_SCORE_KEYS = frozenset(
    {
        "language_explicit",
        "exact_text",
        "stop_after_line",
        "no_repeat_or_improvise",
        "no_extra_speech",
        "overall",
    }
)
_DIALOGUE_PARTS = re.compile(r"^<d>\[([^\]\r\n]+)\]([^\r\n<>]+)</d>$")
_MARKER_PARTS = re.compile(
    r"^DUET_SPEECH_V1 order=([1-9]\d*) mode=(on_screen|off_screen) "
    r"subject=(S[1-9]\d*|none) voice_ref=([1-9]\d*|provider_default) "
    r"lips=(only_subject_speaks|all_visible_subjects_closed) :: "
    r"(.+) says exactly (<d>\[[^\]\r\n]+\][^\r\n<>]+</d>)\.$"
)
_SAFE_ERROR_CODES = frozenset(
    {
        "context_ir_upload_rejected",
        "context_ir_submit_rejected",
        "context_ir_query_failed",
        "context_ir_provider_failed",
        "context_ir_cancelled",
        "context_ir_unknown_status",
        "context_ir_result_invalid",
        "context_ir_result_type_invalid",
        "context_ir_task_mismatch",
        "context_ir_semantic_mismatch",
        "context_ir_poll_timeout",
        "context_ir_query_unknown",
        "context_ir_submission_unknown",
    }
)
_SAFE_PROVIDER_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class ContextIrError(RuntimeError):
    """Stable bridge failure that never includes provider-controlled text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ContextIrContractError(ContextIrError):
    """Caller input or optimized semantics violate the frozen contract."""


class ContextIrReceiptError(ContextIrError):
    """Persisted or caller-supplied receipt does not match frozen input."""


@dataclass(frozen=True, slots=True)
class ContextIrTimeouts:
    request_s: float = 30.0
    poll_total_s: float = 120.0
    poll_interval_s: float = 2.0

    def __post_init__(self) -> None:
        for value in (self.request_s, self.poll_total_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ContextIrContractError("context_ir_timeout_invalid")
        if (
            isinstance(self.poll_interval_s, bool)
            or not isinstance(self.poll_interval_s, (int, float))
            or not math.isfinite(float(self.poll_interval_s))
            or self.poll_interval_s < 0
        ):
            raise ContextIrContractError("context_ir_timeout_invalid")


@dataclass(frozen=True, slots=True)
class FrozenContextIrReference:
    order: int
    type: Literal["image_url", "audio_url"]
    role: Literal["reference_image", "reference_audio"]
    name: str
    mime_type: str
    data: bytes = field(repr=False)
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenContextIrRequest:
    """Secret-bearing runtime input; secrets and media bytes are never persisted."""

    source_h3_request: h3.H3Request = field(repr=False, compare=False)
    skill_plan_sha256: str
    source_prompt: str
    source_prompt_sha256: str
    semantic_contract_sha256: str
    speech_markers: tuple[str, ...]
    dialogue_tokens: tuple[str, ...]
    keyframe_timeline_json: str | None
    references: tuple[FrozenContextIrReference, ...] = field(repr=False)
    references_sha256: str
    voice_texts_sha256: str
    source_h3_request_sha256: str
    upstream_dialogue_sha256: str
    upstream_artifact_path: Path
    upstream_artifact_sha256: str
    upstream_dialogue_sha256_path: tuple[str | int, ...]
    client_request_id: str
    minimax_api_key: str = field(repr=False, compare=False)
    duration: int
    ratio: str
    workdir: Path
    cid: str
    context_ir_attempt_sha256: str
    timeouts: ContextIrTimeouts = ContextIrTimeouts()


@dataclass(frozen=True, slots=True)
class EffectivePromptReceipt:
    receipt_path: Path
    receipt_sha256: str
    cid: str
    attempt_id: str
    client_request_id: str
    source_prompt: str
    source_prompt_sha256: str
    effective_prompt: str
    effective_prompt_sha256: str
    skill_plan_sha256: str
    semantic_contract_sha256: str
    references_sha256: str
    voice_texts_sha256: str
    source_h3_request_sha256: str
    upstream_dialogue_sha256: str
    upstream_artifact_path: Path
    upstream_artifact_sha256: str
    upstream_dialogue_sha256_path: tuple[str | int, ...]
    context_ir_request_sha256: str
    provider_task_id: str
    context_ir_task_sha256: str
    context_ir_attempt_sha256: str
    semantic_score: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ContextIrResult:
    status: Literal[
        "ready",
        "running",
        "query_unknown",
        "submission_unknown",
        "succeeded",
        "failed",
    ]
    attempt_id: str
    provider_task_id: str | None
    effective_prompt: str | None
    source_prompt_sha256: str
    effective_prompt_sha256: str | None
    context_ir_request_sha256: str | None
    context_ir_task_sha256: str | None
    context_ir_attempt_sha256: str
    receipt_path: Path | None
    receipt_sha256: str | None
    error_code: str | None = None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _without_dialogue_policy(prompt: str) -> str:
    """Remove a prior deterministic policy before adding one canonical copy."""
    return _DIALOGUE_POLICY_RE.sub("", prompt).rstrip()


def _with_dialogue_policy(prompt: str) -> str:
    """Make the speech contract explicit for both Context IR and H3."""
    if not isinstance(prompt, str) or not prompt.strip():
        return prompt
    return f"{_without_dialogue_policy(prompt)}\n\n{_DIALOGUE_POLICY}"


def _dialogue_policy_score(
    request: FrozenContextIrRequest, prompt: str,
) -> dict[str, object]:
    """Record before/after dialogue evidence without making a quality gate."""
    expected_texts = tuple(request.source_h3_request.voice_texts)
    tokens = tuple(_DIALOGUE_TOKEN.finditer(prompt))
    parsed: list[tuple[str, str]] = []
    for token in tokens:
        parts = _DIALOGUE_PARTS.fullmatch(token.group(0))
        if parts is not None:
            parsed.append((parts.group(1), parts.group(2)))
    actual_text = "".join(text for _language, text in parsed)
    expected_text = "".join(expected_texts)
    language_explicit = float(
        bool(parsed)
        and all(
            language.strip().lower() not in _UNKNOWN_DIALOGUE_LANGUAGES
            for language, _text in parsed
        )
    ) if expected_texts else float(not tokens)
    exact_text = float(actual_text == expected_text) if expected_texts else float(
        not tokens
    )
    no_extra_speech = exact_text
    policy_present = (
        _DIALOGUE_POLICY_OPEN in prompt
        and _DIALOGUE_POLICY_CLOSE in prompt
    )
    stop_after_line = float(
        policy_present and "Stop speaking immediately" in prompt
    )
    no_repeat_or_improvise = float(
        policy_present
        and "repeat, continue, or improvise" in prompt
    )
    if not expected_texts:
        no_extra_speech = float(not tokens)
    values = (
        language_explicit,
        exact_text,
        stop_after_line,
        no_repeat_or_improvise,
        no_extra_speech,
    )
    return {
        "language_explicit": language_explicit,
        "exact_text": exact_text,
        "stop_after_line": stop_after_line,
        "no_repeat_or_improvise": no_repeat_or_improvise,
        "no_extra_speech": no_extra_speech,
        "overall": sum(values) / len(values),
    }


def _explicit_dialogue_language(
    source_language: str, provider_language: str | None, text: str,
) -> str:
    """Resolve a language label without giving the provider text authority."""
    if source_language.strip().lower() not in _UNKNOWN_DIALOGUE_LANGUAGES:
        return source_language.strip()
    if re.search(r"[\u3400-\u9fff]", text):
        return "Chinese"
    if re.search(r"[A-Za-z]", text):
        return "English"
    if (
        isinstance(provider_language, str)
        and provider_language.strip().lower() not in _UNKNOWN_DIALOGUE_LANGUAGES
    ):
        return provider_language.strip()
    return "OriginalLanguage"


def _bind_exact_dialogue(
    request: FrozenContextIrRequest, prompt: str,
) -> str:
    """Give Context IR prosody authority, never dialogue-text authority."""
    source_parts = []
    for token in request.dialogue_tokens:
        parts = _DIALOGUE_PARTS.fullmatch(token)
        if parts is None:
            raise ContextIrContractError("source_speech_contract_invalid")
        source_parts.append((parts.group(1), parts.group(2)))
    provider_tokens = list(_DIALOGUE_TOKEN.finditer(prompt))
    provider_languages = []
    for token in provider_tokens:
        parts = _DIALOGUE_PARTS.fullmatch(token.group(0))
        provider_languages.append(parts.group(1) if parts is not None else None)

    replacements = []
    for index, (source_language, exact_text) in enumerate(source_parts):
        provider_language = (
            provider_languages[index]
            if index < len(provider_languages)
            else None
        )
        language = _explicit_dialogue_language(
            source_language, provider_language, exact_text,
        )
        replacements.append(f"<d>[{language}]{exact_text}</d>")

    position = 0
    pieces: list[str] = []
    for index, token in enumerate(provider_tokens):
        pieces.append(prompt[position:token.start()])
        if index < len(replacements):
            pieces.append(replacements[index])
        position = token.end()
    pieces.append(prompt[position:])
    compiled = "".join(pieces).rstrip()
    if len(provider_tokens) < len(replacements):
        missing = replacements[len(provider_tokens):]
        compiled = "\n".join((compiled, *missing)).strip()
    return compiled


def _is_current_ref2va(request: FrozenContextIrRequest) -> bool:
    source = request.source_h3_request
    return (
        source.mode == "reference"
        and source.workflow == h3.H3_WORKFLOW
        and source.context_ir_required is True
        and len(source.keyframes) == 9
        and source.reference_audios == ()
    )


def _keyframe_timeline_contract(prompt: str) -> str | None:
    """Extract one canonical timeline block; absence preserves v1 recovery."""
    opening_count = prompt.count(_TIMELINE_OPEN)
    closing_count = prompt.count(_TIMELINE_CLOSE)
    if opening_count == closing_count == 0:
        return None
    if opening_count != 1 or closing_count != 1:
        raise ContextIrContractError("context_ir_semantic_mismatch")
    start = prompt.find(_TIMELINE_OPEN) + len(_TIMELINE_OPEN)
    end = prompt.find(_TIMELINE_CLOSE, start)
    if end < start:
        raise ContextIrContractError("context_ir_semantic_mismatch")
    raw_json = prompt[start:end]
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError:
        raise ContextIrContractError("context_ir_semantic_mismatch") from None
    if not isinstance(value, list) or len(value) != 9:
        raise ContextIrContractError("context_ir_semantic_mismatch")
    frozen: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for order, item in enumerate(value, 1):
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "order", "segment_time_s", "source_scene_id", "transition",
            }
            or item.get("order") != order
            or isinstance(item.get("segment_time_s"), bool)
            or not isinstance(item.get("segment_time_s"), (int, float))
            or not math.isfinite(float(item["segment_time_s"]))
            or float(item["segment_time_s"]) < 0
            or not isinstance(item.get("source_scene_id"), str)
            or not item["source_scene_id"].strip()
            or not isinstance(item.get("transition"), Mapping)
            or set(item["transition"]) != {"type", "at_segment_s"}
        ):
            raise ContextIrContractError("context_ir_semantic_mismatch")
        segment_time_s = float(item["segment_time_s"])
        transition_type = item["transition"].get("type")
        at_segment_s = item["transition"].get("at_segment_s")
        if transition_type not in {"start", "continuous", "hard_cut"}:
            raise ContextIrContractError("context_ir_semantic_mismatch")
        if previous is None:
            if (
                segment_time_s != 0.0
                or transition_type != "start"
                or at_segment_s != 0.0
            ):
                raise ContextIrContractError("context_ir_semantic_mismatch")
        else:
            previous_time_s = float(previous["segment_time_s"])
            if segment_time_s <= previous_time_s or transition_type == "start":
                raise ContextIrContractError("context_ir_semantic_mismatch")
            if transition_type == "hard_cut":
                if (
                    isinstance(at_segment_s, bool)
                    or not isinstance(at_segment_s, (int, float))
                    or not math.isfinite(float(at_segment_s))
                    or not previous_time_s < float(at_segment_s) <= segment_time_s
                    or item["source_scene_id"] == previous["source_scene_id"]
                ):
                    raise ContextIrContractError("context_ir_semantic_mismatch")
            elif (
                at_segment_s is not None
                or item["source_scene_id"] != previous["source_scene_id"]
            ):
                raise ContextIrContractError("context_ir_semantic_mismatch")
        normalized = {
            "order": order,
            "segment_time_s": segment_time_s,
            "source_scene_id": item["source_scene_id"],
            "transition": {
                "type": transition_type,
                "at_segment_s": at_segment_s,
            },
        }
        frozen.append(normalized)
        previous = normalized
    canonical = json.dumps(
        frozen,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    if raw_json != canonical:
        raise ContextIrContractError("context_ir_semantic_mismatch")
    return canonical


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _single_line(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in "<>\r\n\t")
        or any(ord(character) < 32 for character in value)
    ):
        raise ContextIrContractError(code)
    return value


def format_speech_marker(
    *,
    order: int,
    delivery: Literal["on_screen", "off_screen"],
    subject_id: str | None,
    voice_ref: int | None,
    language: str,
    text: str,
) -> str:
    """Compile one route-neutral speech binding from upstream frozen dialogue."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ContextIrContractError("speech_order_invalid")
    language = _single_line(language, "speech_language_invalid")
    text = _single_line(text, "speech_text_invalid")
    if delivery == "on_screen":
        if (
            not isinstance(subject_id, str)
            or re.fullmatch(r"S[1-9]\d*", subject_id) is None
            or isinstance(voice_ref, bool)
            or not isinstance(voice_ref, int)
            or voice_ref < 1
        ):
            raise ContextIrContractError("speech_binding_invalid")
        subject = subject_id
        voice = str(voice_ref)
        lips = "only_subject_speaks"
        speaker = f"<Subject {subject_id[1:]}>({subject_id})"
    elif delivery == "off_screen":
        if subject_id is not None or (
            voice_ref is not None
            and (
                isinstance(voice_ref, bool)
                or not isinstance(voice_ref, int)
                or voice_ref < 1
            )
        ):
            raise ContextIrContractError("speech_binding_invalid")
        subject = "none"
        voice = str(voice_ref) if voice_ref is not None else "provider_default"
        lips = "all_visible_subjects_closed"
        speaker = (
            "The off-screen narrator using the provider default voice"
            if voice_ref is None
            else f"The off-screen narrator using <Audio {voice_ref}>"
        )
    else:
        raise ContextIrContractError("speech_delivery_invalid")
    return (
        f"{_SPEECH_PREFIX} order={order} mode={delivery} subject={subject} "
        f"voice_ref={voice} lips={lips} :: {speaker} says exactly "
        f"<d>[{language}]{text}</d>."
    )


def _speech_contract(
    prompt: str, voice_texts: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ContextIrContractError("source_prompt_invalid")
    markers = tuple(
        line.strip()
        for line in prompt.splitlines()
        if line.lstrip().startswith(_SPEECH_PREFIX)
    )
    tokens = tuple(match.group(0) for match in _DIALOGUE_TOKEN.finditer(prompt))
    expected_texts = tuple(voice_texts)
    if (
        prompt.count(_SPEECH_PREFIX) != len(markers)
        or prompt.count("<d>") != len(tokens)
        or prompt.count("</d>") != len(tokens)
        or len(markers) != len(tokens)
        or len(tokens) != len(expected_texts)
    ):
        raise ContextIrContractError("source_speech_contract_invalid")
    for expected_order, (marker, token, expected_text) in enumerate(
        zip(markers, tokens, expected_texts, strict=True), 1
    ):
        matched = _MARKER_PARTS.fullmatch(marker)
        token_parts = _DIALOGUE_PARTS.fullmatch(token)
        if matched is None or token_parts is None or matched.group(7) != token:
            raise ContextIrContractError("source_speech_contract_invalid")
        order, delivery, subject, raw_voice, lips, _speaker, _token = matched.groups()
        subject_id = None if subject == "none" else subject
        voice_ref = None if raw_voice == "provider_default" else int(raw_voice)
        if (
            int(order) != expected_order
            or token_parts.group(2) != expected_text
            or marker
            != format_speech_marker(
                order=expected_order,
                delivery=delivery,
                subject_id=subject_id,
                voice_ref=voice_ref,
                language=token_parts.group(1),
                text=token_parts.group(2),
            )
            or (
                delivery == "on_screen" and lips != "only_subject_speaks"
            )
            or (
                delivery == "off_screen"
                and lips != "all_visible_subjects_closed"
            )
        ):
            raise ContextIrContractError("source_speech_contract_invalid")
    return markers, tokens


def _ref2va_speech_contract(
    prompt: str, voice_texts: Sequence[str],
) -> tuple[str, ...]:
    """Bind backend-compiled Ref2VA dialogue without a second prompt grammar."""
    tokens = tuple(match.group(0) for match in _DIALOGUE_TOKEN.finditer(prompt))
    if (
        prompt.count("<d>") != len(tokens)
        or prompt.count("</d>") != len(tokens)
        or len(tokens) != len(voice_texts)
    ):
        raise ContextIrContractError("source_speech_contract_invalid")
    for token, text in zip(tokens, voice_texts, strict=True):
        parts = _DIALOGUE_PARTS.fullmatch(token)
        if parts is None or parts.group(2) != text:
            raise ContextIrContractError("source_speech_contract_invalid")
    return tokens


def _fusion_policy_suffix(prompt: str) -> str | None:
    """Return the exact current Fusion contract suffix, if present.

    Current v2 prompts bind timeline, audio, and music as one adjacent tail so
    Context cannot keep the bytes while moving a block to another position.
    The audio/music-only form remains readable for historical fixtures.
    """
    audio_open = "<AUDIO_CONTENT_JSON>"
    audio_close = "</AUDIO_CONTENT_JSON>"
    music_open = "<MUSIC_POLICY>"
    music_close = "</MUSIC_POLICY>"
    music = (
        f"{music_open}\n"
        "non_diegetic_music: N/A\n"
        f"{music_close}"
    )
    policy_tail = f"\n{music}"
    # Historical v1 dialogue may legally contain either marker as text inside
    # its JSON payload. Only the v2 outer marker at the exact prompt tail
    # activates this contract; the v2 loader rejects malformed outer blocks.
    if not prompt.endswith(policy_tail):
        return None
    audio_start = prompt.find(audio_open)
    if audio_start < 0:
        raise ContextIrContractError("source_music_policy_invalid")
    timeline_start = prompt.find(_TIMELINE_OPEN)
    marker_start = timeline_start if timeline_start >= 0 else audio_start
    suffix = prompt[marker_start:]
    if not suffix.endswith(f"{audio_close}\n{music}"):
        raise ContextIrContractError("source_music_policy_invalid")
    if timeline_start >= 0:
        timeline_end = prompt.find(_TIMELINE_CLOSE, timeline_start)
        if (
            timeline_end < timeline_start
            or timeline_end + len(_TIMELINE_CLOSE) + 1 != audio_start
            or prompt[timeline_end + len(_TIMELINE_CLOSE):audio_start] != "\n"
        ):
            raise ContextIrContractError("source_music_policy_invalid")
    prefix = prompt[:marker_start]
    if (
        _TIMELINE_OPEN in prefix
        or _TIMELINE_CLOSE in prefix
        or audio_open in prefix
        or audio_close in prefix
        or music_open in prefix
        or music_close in prefix
    ):
        raise ContextIrContractError("source_music_policy_invalid")
    return suffix


def _relation_states_contract(prompt: str) -> str | None:
    """Extract the one canonical backend-compiled relation authority block."""
    opening_count = prompt.count(_RELATION_STATES_OPEN)
    closing_count = prompt.count(_RELATION_STATES_CLOSE)
    if opening_count == closing_count == 0:
        return None
    if opening_count != 1 or closing_count != 1:
        raise ContextIrContractError("source_relation_states_invalid")
    start = prompt.find(_RELATION_STATES_OPEN)
    data_start = start + len(_RELATION_STATES_OPEN)
    end = prompt.find(_RELATION_STATES_CLOSE, data_start)
    if end < data_start:
        raise ContextIrContractError("source_relation_states_invalid")
    raw = prompt[data_start:end]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ContextIrContractError("source_relation_states_invalid") from None
    try:
        canonical = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ContextIrContractError("source_relation_states_invalid") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"v", "d", "i"}
        or value.get("v") != 1
        or not isinstance(value.get("d"), dict)
        or not isinstance(value.get("i"), list)
        or raw != canonical
    ):
        raise ContextIrContractError("source_relation_states_invalid")
    return f"{_RELATION_STATES_OPEN}{raw}{_RELATION_STATES_CLOSE}"


def _without_relation_states(prompt: str) -> str:
    """Remove model-authored relation blocks before restoring frozen bytes."""
    text = prompt
    while _RELATION_STATES_OPEN in text:
        start = text.find(_RELATION_STATES_OPEN)
        end = text.find(_RELATION_STATES_CLOSE, start)
        if end < 0:
            text = text[:start]
            break
        text = text[:start] + text[end + len(_RELATION_STATES_CLOSE):]
    return text.replace(_RELATION_STATES_CLOSE, "")


def _effective_preserves_fusion_policy(
    effective_prompt: str, expected_suffix: str,
) -> bool:
    if not effective_prompt.endswith(expected_suffix):
        return False
    prefix = effective_prompt[:-len(expected_suffix)]
    return not any(marker in prefix for marker in (
        _TIMELINE_OPEN,
        _TIMELINE_CLOSE,
        "<AUDIO_CONTENT_JSON>",
        "</AUDIO_CONTENT_JSON>",
        "<MUSIC_POLICY>",
        "</MUSIC_POLICY>",
    ))


def _fusion_voice_texts(prompt: str) -> tuple[str, ...] | None:
    """Read the exact deterministic audio block used by current Fusion."""
    suffix = _fusion_policy_suffix(prompt)
    if suffix is None:
        return None
    audio_open = "<AUDIO_CONTENT_JSON>"
    audio_close = "</AUDIO_CONTENT_JSON>"
    start = suffix.find(audio_open)
    end = suffix.find(audio_close, start + len(audio_open))
    if start < 0 or end < 0:
        raise ContextIrContractError("source_music_policy_invalid")
    try:
        lines = json.loads(suffix[start + len(audio_open):end])
    except json.JSONDecodeError:
        raise ContextIrContractError("source_speech_contract_invalid") from None
    if not isinstance(lines, list):
        raise ContextIrContractError("source_speech_contract_invalid")
    texts: list[str] = []
    for line in lines:
        text = line.get("text") if isinstance(line, Mapping) else None
        if not isinstance(text, str) or not text:
            raise ContextIrContractError("source_speech_contract_invalid")
        texts.append(text)
    return tuple(texts)


def _semantic_score(
    request: FrozenContextIrRequest, context_output_prompt: str,
) -> dict[str, object]:
    """Score Context semantics for iteration; never select or stop a chain."""
    expected_voice_texts = tuple(request.source_h3_request.voice_texts)
    if expected_voice_texts:
        speech = float(all(
            text in context_output_prompt for text in expected_voice_texts
        ))
    else:
        speech = float(
            "<d>" not in context_output_prompt
            and "</d>" not in context_output_prompt
        )
    timeline = 1.0
    if request.keyframe_timeline_json is not None:
        try:
            actual_timeline = _keyframe_timeline_contract(
                context_output_prompt
            )
        except ContextIrContractError:
            actual_timeline = None
        timeline = float(actual_timeline == request.keyframe_timeline_json)
    try:
        suffix = _fusion_policy_suffix(request.source_prompt)
    except ContextIrContractError:
        suffix = None
    music_policy = float(
        suffix is None
        or _effective_preserves_fusion_policy(context_output_prompt, suffix)
    )
    try:
        relation_contract = _relation_states_contract(request.source_prompt)
    except ContextIrContractError:
        relation_contract = None
    relations = float(
        relation_contract is None
        or relation_contract in context_output_prompt
    )
    values = (speech, timeline, music_policy, relations)
    return {
        "speech_expected": bool(expected_voice_texts),
        "speech": speech,
        "keyframe_timeline": timeline,
        "music_policy": music_policy,
        "relation_states": relations,
        "overall": sum(values) / len(values),
        "dialogue_policy": _dialogue_policy_score(request, context_output_prompt),
    }


def _compile_effective_prompt(
    request: FrozenContextIrRequest, context_output_prompt: str,
) -> str:
    """Mechanically restore immutable Fusion fields around Context semantics."""
    relation_contract = _relation_states_contract(request.source_prompt)
    context_output_prompt = _without_relation_states(context_output_prompt)
    if _is_current_ref2va(request):
        context_output_prompt = _bind_exact_dialogue(
            request, context_output_prompt,
        )
    suffix = _fusion_policy_suffix(request.source_prompt)
    if suffix is None:
        effective = _with_dialogue_policy(context_output_prompt)
        return (
            f"{effective}\n{relation_contract}"
            if relation_contract is not None else effective
        )
    visual = _without_dialogue_policy(context_output_prompt)
    opening = "<VISUAL>"
    closing = "</VISUAL>"
    if visual.startswith(opening):
        end = visual.find(closing, len(opening))
        if end >= 0:
            visual = visual[len(opening):end]
    else:
        positions = [
            visual.find(marker)
            for marker in (
                _TIMELINE_OPEN,
                "<AUDIO_CONTENT_JSON>",
                "<MUSIC_POLICY>",
            )
            if visual.find(marker) >= 0
        ]
        if positions:
            visual = visual[:min(positions)]
    for marker in (
        opening,
        closing,
        _TIMELINE_OPEN,
        _TIMELINE_CLOSE,
        "<AUDIO_CONTENT_JSON>",
        "</AUDIO_CONTENT_JSON>",
        "<MUSIC_POLICY>",
        "</MUSIC_POLICY>",
    ):
        visual = visual.replace(marker, "")
    visual = visual.strip()
    if not visual:
        visual = _without_dialogue_policy(context_output_prompt.strip())
    relation_suffix = (
        f"\n{relation_contract}" if relation_contract is not None else ""
    )
    return (
        f"<VISUAL>\n{visual}{relation_suffix}\n</VISUAL>\n"
        f"{_DIALOGUE_POLICY}\n{suffix}"
    )


def _mime_type(path: Path, *, audio: bool) -> str:
    suffix = path.suffix.lower()
    types = (
        {".wav": "audio/wav", ".mp3": "audio/mpeg"}
        if audio
        else {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
    )
    try:
        return types[suffix]
    except KeyError:
        raise ContextIrContractError("context_ir_reference_format_invalid") from None


def _reference_manifest(
    references: Sequence[FrozenContextIrReference],
) -> list[dict[str, Any]]:
    return [
        {
            "order": reference.order,
            "type": reference.type,
            "role": reference.role,
            "name": reference.name,
            "mime_type": reference.mime_type,
            "sha256": reference.sha256,
            "size": len(reference.data),
        }
        for reference in references
    ]


def _frame_manifest(frame: h3.FrozenFrame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    path, data = frame
    return {
        "name": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _h3_request_manifest(
    request: h3.H3Request,
    *,
    source_prompt_sha256: str,
    references_sha256: str,
) -> dict[str, Any]:
    """Bind every non-secret H3 field that may affect the provider request."""
    h3.validate_request_authority(request)
    return {
        "cid": request.cid,
        "workdir": str(request.workdir.resolve()),
        "client_request_id": request.client_request_id,
        "source_prompt_sha256": source_prompt_sha256,
        "references_sha256": references_sha256,
        "voice_texts_sha256": request.voice_receipt,
        "duration": request.duration,
        "mode": request.mode,
        "first_frame": _frame_manifest(request.first_frame),
        "last_frame": _frame_manifest(request.last_frame),
        "seed": request.seed,
        "aspect_ratio": request.aspect_ratio,
        "resolution": request.resolution,
        "workflow": request.workflow,
        "skill_plan_sha256": request.skill_plan_sha256,
        "multimodal_compiler_version": request.multimodal_compiler_version,
        "speaker_timing_sha256": request.speaker_timing_sha256,
        **(
            {
                "speaker_timing_authority": (
                    h3.speaker_timing_authority_manifest(request)
                )
            }
            if h3.speaker_timing_authority_manifest(request) is not None
            else {}
        ),
        "on_screen_dialogue_sha256": request.on_screen_dialogue_sha256,
        "audio_required": request.audio_required,
        "context_ir_required": request.context_ir_required,
        "reference_audio_metadata": [
            {
                "order": audio.order,
                "purpose": audio.purpose,
                "format": audio.format,
                "sha256": audio.sha256,
                "duration_s": audio.duration_s,
            }
            for audio in request.reference_audios
        ],
    }


def _verified_upstream_artifact(
    path: Path,
    expected_sha256: str,
    dialogue_sha256: str,
    dialogue_sha256_path: tuple[str | int, ...],
) -> Path:
    if not _is_sha256(expected_sha256):
        raise ContextIrContractError("upstream_artifact_sha256_invalid")
    if (
        not isinstance(dialogue_sha256_path, tuple)
        or not dialogue_sha256_path
        or len(dialogue_sha256_path) > 16
        or any(
            (isinstance(part, str) and not part)
            or (isinstance(part, int) and (isinstance(part, bool) or part < 0))
            or not isinstance(part, (str, int))
            for part in dialogue_sha256_path
        )
    ):
        raise ContextIrContractError("upstream_dialogue_sha256_path_invalid")
    try:
        resolved = Path(path).resolve(strict=True)
        size = resolved.stat().st_size
        if not resolved.is_file() or not 0 < size <= MAX_UPSTREAM_ARTIFACT_BYTES:
            raise OSError
        data = resolved.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        artifact: Any = json.loads(data)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ContextIrContractError("upstream_artifact_invalid") from None
    if digest != expected_sha256:
        raise ContextIrContractError("upstream_artifact_sha256_mismatch")
    try:
        value: Any = artifact
        for part in dialogue_sha256_path:
            if isinstance(part, str) and isinstance(value, Mapping):
                value = value[part]
            elif (
                isinstance(part, int)
                and isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
            ):
                value = value[part]
            else:
                raise KeyError
    except (IndexError, KeyError, TypeError):
        raise ContextIrContractError("upstream_dialogue_sha256_path_invalid") from None
    if value != dialogue_sha256:
        raise ContextIrContractError("upstream_dialogue_sha256_mismatch")
    return resolved


def _source_input_manifest(
    *,
    cid: str,
    source_h3_client_request_id: str,
    skill_plan_sha256: str,
    source_prompt_sha256: str,
    semantic_contract_sha256: str,
    references_sha256: str,
    voice_texts_sha256: str,
    source_h3_request_sha256: str,
    upstream_dialogue_sha256: str,
    upstream_artifact_path: Path,
    upstream_artifact_sha256: str,
    upstream_dialogue_sha256_path: tuple[str | int, ...],
    duration: int,
    ratio: str,
) -> dict[str, Any]:
    return {
        "cid": cid,
        "source_h3_client_request_id": source_h3_client_request_id,
        "skill_plan_sha256": skill_plan_sha256,
        "source_prompt_sha256": source_prompt_sha256,
        "semantic_contract_sha256": semantic_contract_sha256,
        "references_sha256": references_sha256,
        "voice_texts_sha256": voice_texts_sha256,
        "source_h3_request_sha256": source_h3_request_sha256,
        "upstream_dialogue_sha256": upstream_dialogue_sha256,
        "upstream_artifact_path": str(upstream_artifact_path),
        "upstream_artifact_sha256": upstream_artifact_sha256,
        "upstream_dialogue_sha256_path": list(upstream_dialogue_sha256_path),
        "duration": duration,
        "ratio": ratio,
    }


def freeze_context_ir_request(
    *,
    source_h3_request: h3.H3Request,
    upstream_dialogue_sha256: str,
    upstream_artifact_path: Path,
    upstream_artifact_sha256: str,
    upstream_dialogue_sha256_path: tuple[str | int, ...],
    source_prompt_sha256: str,
    minimax_api_key: str,
    timeouts: ContextIrTimeouts = ContextIrTimeouts(),
) -> FrozenContextIrRequest:
    """Bind a coordinator-verified dialogue artifact to the exact H3 request."""
    if not isinstance(source_h3_request, h3.H3Request):
        raise ContextIrContractError("source_h3_request_invalid")
    is_multimodal = h3.is_multimodal_request(source_h3_request)
    is_no_audio_reference = (
        source_h3_request.mode == "reference"
        and source_h3_request.workflow in {None, h3.H3_WORKFLOW}
        and source_h3_request.reference_audios == ()
        and source_h3_request.audio_required is False
    )
    if not is_multimodal and not is_no_audio_reference:
        raise ContextIrContractError("source_h3_request_invalid")
    if not isinstance(minimax_api_key, str) or not minimax_api_key.strip():
        raise ContextIrContractError("context_ir_credential_missing")
    if (
        not _is_sha256(source_prompt_sha256)
        or source_prompt_sha256 != _sha256_text(source_h3_request.prompt)
    ):
        raise ContextIrContractError("source_prompt_sha256_mismatch")
    skill_plan_sha256 = (
        str(source_h3_request.skill_plan_sha256)
        if is_multimodal
        else source_prompt_sha256
    )
    if (
        not _is_sha256(skill_plan_sha256)
        or not _is_sha256(upstream_dialogue_sha256)
        or source_h3_request.voice_receipt
        != h3.voice_texts_receipt(source_h3_request.voice_texts)
    ):
        raise ContextIrContractError("source_speech_receipt_invalid")
    if len(source_h3_request.prompt) > MAX_SOURCE_PROMPT_CHARS:
        raise ContextIrContractError("source_prompt_too_long")
    verified_artifact_path = _verified_upstream_artifact(
        upstream_artifact_path,
        upstream_artifact_sha256,
        upstream_dialogue_sha256,
        upstream_dialogue_sha256_path,
    )
    keyframe_timeline_json = _keyframe_timeline_contract(
        source_h3_request.prompt
    )
    relation_states_contract = _relation_states_contract(
        source_h3_request.prompt
    )
    fusion_policy_suffix = _fusion_policy_suffix(source_h3_request.prompt)
    fusion_voice_texts = _fusion_voice_texts(source_h3_request.prompt)
    if (
        fusion_voice_texts is None
        and source_h3_request.mode == "reference"
        and source_h3_request.workflow == h3.H3_WORKFLOW
    ):
        speech_markers = ()
        dialogue_tokens = _ref2va_speech_contract(
            source_h3_request.prompt, source_h3_request.voice_texts
        )
    elif fusion_voice_texts is None:
        speech_markers, dialogue_tokens = _speech_contract(
            source_h3_request.prompt, source_h3_request.voice_texts
        )
    else:
        if fusion_voice_texts != tuple(source_h3_request.voice_texts):
            raise ContextIrContractError("source_speech_receipt_invalid")
        speech_markers = ()
        dialogue_tokens = fusion_voice_texts

    references: list[FrozenContextIrReference] = []
    for order, (path, data) in enumerate(source_h3_request.keyframes, 1):
        references.append(
            FrozenContextIrReference(
                order=order,
                type="image_url",
                role="reference_image",
                name=path.name,
                mime_type=_mime_type(path, audio=False),
                data=data,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    for audio in source_h3_request.reference_audios:
        references.append(
            FrozenContextIrReference(
                order=audio.order,
                type="audio_url",
                role="reference_audio",
                name=audio.path.name,
                mime_type=_mime_type(audio.path, audio=True),
                data=audio.data,
                sha256=audio.sha256,
            )
        )
    frozen_references = tuple(references)
    references_sha256 = _canonical_sha256(_reference_manifest(frozen_references))
    source_h3_request_sha256 = _canonical_sha256(
        _h3_request_manifest(
            source_h3_request,
            source_prompt_sha256=source_prompt_sha256,
            references_sha256=references_sha256,
        )
    )
    semantic_contract = {
        "speech_markers": list(speech_markers),
        "dialogue_tokens": list(dialogue_tokens),
    }
    if keyframe_timeline_json is not None:
        semantic_contract["keyframe_timeline_json"] = keyframe_timeline_json
    if fusion_policy_suffix is not None:
        semantic_contract["fusion_policy_suffix_sha256"] = _sha256_text(
            fusion_policy_suffix
        )
    if relation_states_contract is not None:
        semantic_contract["relation_states_sha256"] = _sha256_text(
            relation_states_contract
        )
    semantic_contract_sha256 = _canonical_sha256(semantic_contract)
    input_manifest = _source_input_manifest(
        cid=source_h3_request.cid,
        source_h3_client_request_id=source_h3_request.client_request_id,
        skill_plan_sha256=skill_plan_sha256,
        source_prompt_sha256=source_prompt_sha256,
        semantic_contract_sha256=semantic_contract_sha256,
        references_sha256=references_sha256,
        voice_texts_sha256=source_h3_request.voice_receipt,
        source_h3_request_sha256=source_h3_request_sha256,
        upstream_dialogue_sha256=upstream_dialogue_sha256,
        upstream_artifact_path=verified_artifact_path,
        upstream_artifact_sha256=upstream_artifact_sha256,
        upstream_dialogue_sha256_path=upstream_dialogue_sha256_path,
        duration=source_h3_request.duration,
        ratio=source_h3_request.aspect_ratio,
    )
    context_ir_attempt_sha256 = _canonical_sha256(
        {
            "schema": ATTEMPT_SCHEMA,
            "version": SCHEMA_VERSION,
            "client_request_id": source_h3_request.client_request_id,
            "input": input_manifest,
        }
    )
    frozen = FrozenContextIrRequest(
        source_h3_request=source_h3_request,
        skill_plan_sha256=skill_plan_sha256,
        source_prompt=source_h3_request.prompt,
        source_prompt_sha256=source_prompt_sha256,
        semantic_contract_sha256=semantic_contract_sha256,
        speech_markers=speech_markers,
        dialogue_tokens=dialogue_tokens,
        keyframe_timeline_json=keyframe_timeline_json,
        references=frozen_references,
        references_sha256=references_sha256,
        voice_texts_sha256=source_h3_request.voice_receipt,
        source_h3_request_sha256=source_h3_request_sha256,
        upstream_dialogue_sha256=upstream_dialogue_sha256,
        upstream_artifact_path=verified_artifact_path,
        upstream_artifact_sha256=upstream_artifact_sha256,
        upstream_dialogue_sha256_path=upstream_dialogue_sha256_path,
        client_request_id=source_h3_request.client_request_id,
        minimax_api_key=minimax_api_key,
        duration=source_h3_request.duration,
        ratio=source_h3_request.aspect_ratio,
        workdir=source_h3_request.workdir,
        cid=source_h3_request.cid,
        context_ir_attempt_sha256=context_ir_attempt_sha256,
        timeouts=timeouts,
    )
    _assert_existing_request_matches(frozen)
    return frozen


def _validate_frozen_request(request: FrozenContextIrRequest) -> None:
    if not isinstance(request, FrozenContextIrRequest):
        raise ContextIrContractError("context_ir_request_invalid")
    expected = freeze_context_ir_request(
        source_h3_request=request.source_h3_request,
        upstream_dialogue_sha256=request.upstream_dialogue_sha256,
        upstream_artifact_path=request.upstream_artifact_path,
        upstream_artifact_sha256=request.upstream_artifact_sha256,
        upstream_dialogue_sha256_path=request.upstream_dialogue_sha256_path,
        source_prompt_sha256=request.source_prompt_sha256,
        minimax_api_key=request.minimax_api_key,
        timeouts=request.timeouts,
    )
    if expected != request:
        raise ContextIrReceiptError("context_ir_request_receipt_mismatch")


def _state_root(request: FrozenContextIrRequest) -> Path:
    return request.workdir / ".context-ir"


def _attempts_root(request: FrozenContextIrRequest) -> Path:
    return _state_root(request) / "attempts"


def _input_manifest(request: FrozenContextIrRequest) -> dict[str, Any]:
    return _source_input_manifest(
        cid=request.cid,
        source_h3_client_request_id=request.source_h3_request.client_request_id,
        skill_plan_sha256=request.skill_plan_sha256,
        source_prompt_sha256=request.source_prompt_sha256,
        semantic_contract_sha256=request.semantic_contract_sha256,
        references_sha256=request.references_sha256,
        voice_texts_sha256=request.voice_texts_sha256,
        source_h3_request_sha256=request.source_h3_request_sha256,
        upstream_dialogue_sha256=request.upstream_dialogue_sha256,
        upstream_artifact_path=request.upstream_artifact_path,
        upstream_artifact_sha256=request.upstream_artifact_sha256,
        upstream_dialogue_sha256_path=request.upstream_dialogue_sha256_path,
        duration=request.duration,
        ratio=request.ratio,
    )


def _uploaded_reference_state(request: FrozenContextIrRequest) -> list[dict[str, Any]]:
    return [dict(item, file_id=None) for item in _reference_manifest(request.references)]


def _new_state(request: FrozenContextIrRequest, attempt_id: str) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "version": SCHEMA_VERSION,
        "cid": request.cid,
        "attempt_id": attempt_id,
        "client_request_id": request.client_request_id,
        "input": _input_manifest(request),
        "context_ir_attempt_sha256": request.context_ir_attempt_sha256,
        "status": "ready",
        "references": _uploaded_reference_state(request),
        "context_ir_request": None,
        "context_ir_request_sha256": None,
        "provider_task_id": None,
        "context_ir_task_sha256": None,
        "http_status": None,
        "provider_error_code": None,
        "receipt": None,
        "error": None,
    }


_STATE_KEYS = frozenset(
    {
        "schema",
        "version",
        "cid",
        "attempt_id",
        "client_request_id",
        "input",
        "context_ir_attempt_sha256",
        "status",
        "references",
        "context_ir_request",
        "context_ir_request_sha256",
        "provider_task_id",
        "context_ir_task_sha256",
        "http_status",
        "provider_error_code",
        "receipt",
        "error",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ContextIrReceiptError("context_ir_state_invalid") from None
    if not isinstance(value, dict):
        raise ContextIrReceiptError("context_ir_state_invalid")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, create: bool = False) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if create:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            _fsync_directory(path.parent)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                path.unlink()
            except OSError:
                pass
            raise
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attempt_path(request: FrozenContextIrRequest, attempt_id: str) -> Path:
    return _attempts_root(request) / attempt_id / "attempt.json"


def _record_attempt_error(
    request: FrozenContextIrRequest,
    path: Path,
    event: str,
    *,
    call_path: list[str],
    error: BaseException | None = None,
    reason: object | None = None,
) -> None:
    """Persist every diagnostic without replacing earlier attempt evidence."""
    directory = path.parent / "errors"
    sequence = 1
    while (directory / f"context-ir-{event}-{sequence:06d}.json").exists():
        sequence += 1
    destination = directory / f"context-ir-{event}-{sequence:06d}.json"
    kwargs = {
        "call_path": [*call_path, f"attempt:{path.parent.name}"],
        "logger": _LOGGER,
        "secrets": (request.minimax_api_key,),
    }
    if error is not None:
        kwargs["error"] = error
    else:
        kwargs["reason"] = reason
    error_trace.record(destination, **kwargs)
    # Retain the established latest-error location for existing support tools.
    kwargs["logger"] = None
    kwargs["call_path"] = call_path
    error_trace.record(path.with_name("error.json"), **kwargs)


def _matching_attempt_path(request: FrozenContextIrRequest) -> Path | None:
    attempts = _attempts_root(request)
    if not attempts.is_dir():
        return None
    try:
        paths = sorted(attempts.glob("*/attempt.json"), reverse=True)
    except OSError:
        raise ContextIrReceiptError("context_ir_state_invalid") from None
    for path in paths:
        raw = _read_json(path)
        if raw.get("client_request_id") == request.client_request_id:
            return path
    return None


def _assert_existing_request_matches(request: FrozenContextIrRequest) -> None:
    path = _matching_attempt_path(request)
    if path is None:
        return
    state = _read_json(path)
    if (
        state.get("context_ir_attempt_sha256")
        != request.context_ir_attempt_sha256
        or state.get("input") != _input_manifest(request)
    ):
        raise ContextIrReceiptError("context_ir_request_receipt_mismatch")
    _validate_state(request, state, path)


def _request_body_from_state(state: Mapping[str, Any], source_prompt: str) -> dict[str, Any] | None:
    references = state.get("references")
    if not isinstance(references, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("file_id"), str)
        or not item["file_id"]
        for item in references
    ):
        return None
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": _with_dialogue_policy(source_prompt),
    }]
    for item in references:
        url = f"mm_file://{item['file_id']}"
        if item["type"] == "image_url":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": item["role"],
                }
            )
        elif item["type"] == "audio_url":
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": url},
                    "role": item["role"],
                }
            )
        else:
            return None
    input_manifest = state["input"]
    return {
        "model": "MiniMax-H3",
        "content": content,
        "duration": input_manifest["duration"],
        "ratio": input_manifest["ratio"],
    }


def _validate_state(
    request: FrozenContextIrRequest, state: Mapping[str, Any], path: Path
) -> None:
    attempt_id = state.get("attempt_id")
    if (
        set(state) != _STATE_KEYS
        or state.get("schema") != ATTEMPT_SCHEMA
        or state.get("version") != SCHEMA_VERSION
        or state.get("cid") != request.cid
        or not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
        or path != _attempt_path(request, attempt_id)
        or state.get("client_request_id") != request.client_request_id
        or state.get("input") != _input_manifest(request)
        or state.get("context_ir_attempt_sha256")
        != request.context_ir_attempt_sha256
        or state.get("status")
        not in {
            "ready",
            "uploading",
            "ready_to_submit",
            "submitting",
            "polling",
            "query_unknown",
            "submission_unknown",
            "succeeded",
            "failed",
        }
    ):
        raise ContextIrReceiptError("context_ir_request_receipt_mismatch")
    expected_references = _uploaded_reference_state(request)
    stored_references = state.get("references")
    if not isinstance(stored_references, list) or len(stored_references) != len(
        expected_references
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")
    for expected, stored in zip(expected_references, stored_references, strict=True):
        if not isinstance(stored, dict) or set(stored) != set(expected):
            raise ContextIrReceiptError("context_ir_state_invalid")
        if any(stored.get(key) != expected[key] for key in expected if key != "file_id"):
            raise ContextIrReceiptError("context_ir_request_receipt_mismatch")
        file_id = stored.get("file_id")
        if file_id is not None and (
            not isinstance(file_id, str)
            or not file_id.isdecimal()
            or int(file_id) <= 0
        ):
            raise ContextIrReceiptError("context_ir_state_invalid")
    body = _request_body_from_state(state, request.source_prompt)
    stored_body = state.get("context_ir_request")
    stored_request_sha = state.get("context_ir_request_sha256")
    if stored_body is None:
        if stored_request_sha is not None:
            raise ContextIrReceiptError("context_ir_state_invalid")
    elif (
        body is None
        or stored_body != body
        or not _is_sha256(stored_request_sha)
        or stored_request_sha != _canonical_sha256(body)
    ):
        raise ContextIrReceiptError("context_ir_request_receipt_mismatch")
    task_id = state.get("provider_task_id")
    task_sha = state.get("context_ir_task_sha256")
    if task_id is None:
        if task_sha is not None:
            raise ContextIrReceiptError("context_ir_state_invalid")
    elif (
        not isinstance(task_id, str)
        or not task_id.strip()
        or not _is_sha256(task_sha)
        or not isinstance(stored_request_sha, str)
        or task_sha
        != _canonical_sha256(
            {
                "context_ir_request_sha256": stored_request_sha,
                "provider_task_id": task_id,
            }
        )
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")
    http_status = state.get("http_status")
    provider_error_code = state.get("provider_error_code")
    if (
        http_status is not None
        and (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 100 <= http_status <= 599
        )
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")
    if provider_error_code is not None and (
        not isinstance(provider_error_code, str)
        or _SAFE_PROVIDER_ERROR_CODE.fullmatch(provider_error_code) is None
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")
    if (
        provider_error_code is not None and http_status is None
    ) or (
        http_status is not None
        and state.get("status") not in {"failed", "submission_unknown"}
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")
    error = state.get("error")
    if error is not None and error not in _SAFE_ERROR_CODES:
        raise ContextIrReceiptError("context_ir_state_invalid")
    receipt = state.get("receipt")
    if receipt is not None and (
        not isinstance(receipt, dict)
        or set(receipt) != {"path", "sha256"}
        or receipt.get("path") != "receipt.json"
        or not _is_sha256(receipt.get("sha256"))
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")
    status = state["status"]
    all_uploaded = all(item.get("file_id") is not None for item in stored_references)
    references_ready = all_uploaded
    if (
        (
            status == "ready"
            and (
                any(item.get("file_id") is not None for item in stored_references)
                or stored_body is not None
                or task_id is not None
                or error is not None
            )
        )
        or (
            status in {"ready_to_submit", "submitting"}
            and (
                not references_ready
                or stored_body is None
                or task_id is not None
                or error is not None
            )
        )
        or (
            status in {"polling", "query_unknown", "succeeded"}
            and (not references_ready or stored_body is None or task_id is None)
        )
        or (
            status == "query_unknown"
            and error
            not in {
                "context_ir_query_unknown",
                "context_ir_unknown_status",
                "context_ir_poll_timeout",
            }
        )
        or (
            status == "submission_unknown"
            and (task_id is not None or error != "context_ir_submission_unknown")
        )
        or (status == "succeeded" and (receipt is None or error is not None))
        or (status != "succeeded" and receipt is not None)
    ):
        raise ContextIrReceiptError("context_ir_state_invalid")


def _ensure_session_marker(request: FrozenContextIrRequest) -> None:
    marker = _state_root(request) / "session.json"
    payload = {
        "schema": SESSION_SCHEMA,
        "version": SCHEMA_VERSION,
        "cid": request.cid,
    }
    try:
        _atomic_write_json(marker, payload, create=True)
    except FileExistsError:
        if _read_json(marker) != payload:
            raise ContextIrReceiptError("context_ir_session_mismatch") from None


@contextmanager
def _session_lease(request: FrozenContextIrRequest) -> Iterator[None]:
    root = _state_root(request)
    try:
        root.mkdir(parents=True, exist_ok=True)
        _fsync_directory(root.parent)
        descriptor = os.open(root / "session.lock", os.O_RDWR | os.O_CREAT, 0o600)
        _fsync_directory(root)
    except OSError:
        raise ContextIrError("context_ir_state_unavailable") from None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContextIrError("context_ir_session_busy") from None
        _ensure_session_marker(request)
        yield
    finally:
        os.close(descriptor)


def _prepare(request: FrozenContextIrRequest) -> tuple[dict[str, Any], Path]:
    existing = _matching_attempt_path(request)
    if existing is not None:
        state = _read_json(existing)
        _validate_state(request, state, existing)
        return state, existing
    attempts = _attempts_root(request)
    try:
        attempts.mkdir(parents=True, exist_ok=True)
        _fsync_directory(attempts.parent)
        numbers = [
            int(path.name)
            for path in attempts.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        attempt_id = f"{max(numbers, default=0) + 1:06d}"
        directory = attempts / attempt_id
        directory.mkdir()
        _fsync_directory(attempts)
        state = _new_state(request, attempt_id)
        path = directory / "attempt.json"
        _atomic_write_json(path, state, create=True)
        return state, path
    except (OSError, ValueError):
        raise ContextIrError("context_ir_attempt_claim_failed") from None


def _persist(path: Path, state: Mapping[str, Any]) -> None:
    try:
        _atomic_write_json(path, state)
    except OSError as exc:
        raise ContextIrError("context_ir_state_unavailable") from exc


def _parse_json(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        if len(response.content) > MAX_RESPONSE_BYTES:
            return None
        payload = response.json()
    except (UnicodeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _safe_provider_error_code(payload: Mapping[str, Any] | None) -> str | None:
    """Extract only a bounded provider code; never persist provider messages."""
    if payload is None:
        return None
    candidates: list[Any] = []
    error = payload.get("error")
    if isinstance(error, Mapping):
        candidates.append(error.get("code"))
    candidates.extend((payload.get("error_code"), payload.get("code")))
    base_resp = payload.get("base_resp")
    if isinstance(base_resp, Mapping):
        candidates.extend((base_resp.get("code"), base_resp.get("status_code")))
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int):
            if candidate == 0:
                continue
            candidate = str(candidate)
        if (
            isinstance(candidate, str)
            and _SAFE_PROVIDER_ERROR_CODE.fullmatch(candidate) is not None
        ):
            return candidate
    return None


def _normalized_file_id(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    base_resp = payload.get("base_resp")
    file_payload = payload.get("file")
    if (
        not isinstance(base_resp, Mapping)
        or base_resp.get("status_code") not in {0, "0"}
        or not isinstance(file_payload, Mapping)
    ):
        return None
    file_id = file_payload.get("file_id")
    if isinstance(file_id, bool):
        return None
    if isinstance(file_id, int) and file_id > 0:
        return str(file_id)
    if isinstance(file_id, str) and file_id.isdecimal() and int(file_id) > 0:
        return file_id
    return None


def _mark_terminal(
    path: Path,
    state: dict[str, Any],
    *,
    status: Literal["failed", "submission_unknown", "query_unknown"],
    error: str,
    http_status: int | None = None,
    provider_error_code: str | None = None,
) -> None:
    state["status"] = status
    state["error"] = error
    state["http_status"] = http_status
    state["provider_error_code"] = provider_error_code
    _persist(path, state)


def _upload_references(
    request: FrozenContextIrRequest,
    state: dict[str, Any],
    path: Path,
    client: httpx.Client,
) -> bool:
    state["status"] = "uploading"
    state["error"] = None
    _persist(path, state)
    for index, (reference, stored) in enumerate(
        zip(request.references, state["references"], strict=True)
    ):
        if stored["file_id"] is not None:
            continue
        try:
            response = client.post(
                UPLOAD_URL,
                data={"purpose": "video_generation_input"},
                files={
                    "file": (reference.name, reference.data, reference.mime_type),
                },
                headers={"Authorization": f"Bearer {request.minimax_api_key}"},
                timeout=request.timeouts.request_s,
            )
        except httpx.HTTPError as exc:
            _mark_terminal(
                path,
                state,
                status="submission_unknown",
                error="context_ir_submission_unknown",
            )
            _record_attempt_error(
                request,
                path,
                "upload",
                call_path=["generation", "context_ir", "upload", str(index + 1)],
                error=exc,
            )
            return False
        payload = _parse_json(response)
        file_id = _normalized_file_id(payload)
        if not response.is_success or file_id is None:
            base_resp = payload.get("base_resp") if payload is not None else None
            explicit_rejection = (
                response.is_success
                and isinstance(base_resp, Mapping)
                and base_resp.get("status_code") not in {0, "0"}
            )
            error = (
                "context_ir_upload_rejected"
                if explicit_rejection or (not response.is_success and response.status_code < 500)
                else "context_ir_submission_unknown"
            )
            _mark_terminal(
                path,
                state,
                status="submission_unknown" if error.endswith("submission_unknown") else "failed",
                error=error,
                http_status=response.status_code if not response.is_success else None,
                provider_error_code=(
                    _safe_provider_error_code(payload)
                    if not response.is_success
                    else None
                ),
            )
            _record_attempt_error(
                request,
                path,
                "upload",
                call_path=["generation", "context_ir", "upload", str(index + 1)],
                reason={"code": error, "provider": error_trace.provider_response(response, secrets=(request.minimax_api_key,))},
            )
            return False
        state["references"][index]["file_id"] = file_id
        _persist(path, state)
    body = _request_body_from_state(state, request.source_prompt)
    if body is None:
        raise ContextIrReceiptError("context_ir_state_invalid")
    state["context_ir_request"] = body
    state["context_ir_request_sha256"] = _canonical_sha256(body)
    state["status"] = "ready_to_submit"
    _persist(path, state)
    return True


def _task_id(payload: Mapping[str, Any] | None) -> str | None:
    task_id = payload.get("task_id") if payload is not None else None
    return task_id if isinstance(task_id, str) and task_id.strip() else None


def _submit(
    request: FrozenContextIrRequest,
    state: dict[str, Any],
    path: Path,
    client: httpx.Client,
) -> None:
    if state["status"] == "ready":
        if not _upload_references(request, state, path, client):
            return
    if state["status"] != "ready_to_submit":
        return
    state["status"] = "submitting"
    state["error"] = None
    _persist(path, state)
    try:
        response = client.post(
            SUBMIT_URL,
            json=state["context_ir_request"],
            headers={
                "Authorization": f"Bearer {request.minimax_api_key}",
                "Content-Type": "application/json",
            },
            timeout=request.timeouts.request_s,
        )
    except httpx.HTTPError as exc:
        _mark_terminal(
            path,
            state,
            status="submission_unknown",
            error="context_ir_submission_unknown",
        )
        _record_attempt_error(
            request,
            path,
            "submit",
            call_path=["generation", "context_ir", "submit"],
            error=exc,
        )
        return
    payload = _parse_json(response)
    task_id = _task_id(payload)
    if not response.is_success or task_id is None:
        http_status = response.status_code if not response.is_success else None
        provider_error_code = (
            _safe_provider_error_code(payload)
            if not response.is_success
            else None
        )
        if response.status_code >= 500 or response.is_success:
            _mark_terminal(
                path,
                state,
                status="submission_unknown",
                error="context_ir_submission_unknown",
                http_status=http_status,
                provider_error_code=provider_error_code,
            )
        else:
            _mark_terminal(
                path,
                state,
                status="failed",
                error="context_ir_submit_rejected",
                http_status=http_status,
                provider_error_code=provider_error_code,
            )
        _record_attempt_error(
            request,
            path,
            "submit",
            call_path=["generation", "context_ir", "submit"],
            reason={
                "code": state["error"],
                "provider": error_trace.provider_response(response, secrets=(request.minimax_api_key,)),
            },
        )
        return
    state["provider_task_id"] = task_id
    state["context_ir_task_sha256"] = _canonical_sha256(
        {
            "context_ir_request_sha256": state["context_ir_request_sha256"],
            "provider_task_id": task_id,
        }
    )
    state["status"] = "polling"
    _persist(path, state)


def _receipt_payload(
    request: FrozenContextIrRequest,
    state: Mapping[str, Any],
    effective_prompt: str,
    context_output_prompt: str,
    semantic_score: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "version": SCHEMA_VERSION,
        "cid": request.cid,
        "attempt_id": state["attempt_id"],
        "client_request_id": request.client_request_id,
        "source_prompt": request.source_prompt,
        "source_prompt_sha256": request.source_prompt_sha256,
        "effective_prompt": effective_prompt,
        "effective_prompt_sha256": _sha256_text(effective_prompt),
        "context_output_prompt": context_output_prompt,
        "context_output_prompt_sha256": _sha256_text(context_output_prompt),
        "semantic_score": dict(semantic_score),
        "skill_plan_sha256": request.skill_plan_sha256,
        "semantic_contract_sha256": request.semantic_contract_sha256,
        "references_sha256": request.references_sha256,
        "voice_texts_sha256": request.voice_texts_sha256,
        "source_h3_request_sha256": request.source_h3_request_sha256,
        "upstream_dialogue_sha256": request.upstream_dialogue_sha256,
        "upstream_artifact_path": str(request.upstream_artifact_path),
        "upstream_artifact_sha256": request.upstream_artifact_sha256,
        "upstream_dialogue_sha256_path": list(
            request.upstream_dialogue_sha256_path
        ),
        "context_ir_request_sha256": state["context_ir_request_sha256"],
        "provider_task_id": state["provider_task_id"],
        "context_ir_task_sha256": state["context_ir_task_sha256"],
        "context_ir_attempt_sha256": request.context_ir_attempt_sha256,
    }


def _complete(
    request: FrozenContextIrRequest,
    state: dict[str, Any],
    path: Path,
    effective_prompt: str,
) -> None:
    if (
        not isinstance(effective_prompt, str)
        or not effective_prompt.strip()
        or len(effective_prompt.encode("utf-8")) > MAX_EFFECTIVE_PROMPT_BYTES
    ):
        _mark_terminal(
            path,
            state,
            status="failed",
            error="context_ir_result_invalid",
        )
        return
    compiled_prompt = _compile_effective_prompt(request, effective_prompt)
    semantic_score = _semantic_score(request, effective_prompt)
    semantic_score["dialogue_policy"] = _dialogue_policy_score(
        request, compiled_prompt
    )
    payload = _receipt_payload(
        request,
        state,
        compiled_prompt,
        effective_prompt,
        semantic_score,
    )
    receipt_sha256 = _canonical_sha256(payload)
    receipt = dict(payload, receipt_sha256=receipt_sha256)
    receipt_path = path.with_name("receipt.json")
    try:
        _atomic_write_json(receipt_path, receipt, create=True)
    except FileExistsError:
        existing = _read_json(receipt_path)
        if existing != receipt:
            raise ContextIrReceiptError("context_ir_receipt_mismatch") from None
    state["receipt"] = {"path": receipt_path.name, "sha256": receipt_sha256}
    state["status"] = "succeeded"
    state["error"] = None
    _persist(path, state)


def _task_from_payload(
    payload: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    task = payload.get("task") if payload is not None else None
    return task if isinstance(task, Mapping) else None


def _poll_once(
    request: FrozenContextIrRequest,
    state: dict[str, Any],
    path: Path,
    client: httpx.Client,
    last_response: dict[str, object] | None = None,
) -> Literal["continue", "stop"]:
    task_id = state.get("provider_task_id")
    if not isinstance(task_id, str) or not task_id:
        _mark_terminal(
            path,
            state,
            status="submission_unknown",
            error="context_ir_submission_unknown",
        )
        return "stop"
    try:
        response = client.get(
            f"{QUERY_URL}/{quote(task_id, safe='')}",
            headers={"Authorization": f"Bearer {request.minimax_api_key}"},
            timeout=request.timeouts.request_s,
        )
    except httpx.HTTPError as exc:
        _mark_terminal(
            path,
            state,
            status="query_unknown",
            error="context_ir_query_unknown",
        )
        _record_attempt_error(
            request,
            path,
            "poll",
            call_path=["generation", "context_ir", "poll"],
            error=exc,
        )
        return "stop"
    if not response.is_success:
        _mark_terminal(
            path,
            state,
            status="query_unknown",
            error="context_ir_query_unknown",
        )
        _record_attempt_error(
            request,
            path,
            "poll",
            call_path=["generation", "context_ir", "poll"],
            reason={
                "code": "context_ir_query_unknown",
                "provider": error_trace.provider_response(response, secrets=(request.minimax_api_key,)),
            },
        )
        return "stop"
    payload = _parse_json(response)
    task = _task_from_payload(payload)
    if last_response is not None:
        last_response.clear()
        last_response.update(
            provider=error_trace.provider_response(
                response, secrets=(request.minimax_api_key,)
            ),
            task=task,
        )
    if task is None:
        _mark_terminal(
            path,
            state,
            status="query_unknown",
            error="context_ir_query_unknown",
        )
        _record_attempt_error(
            request,
            path,
            "poll-decode",
            call_path=["generation", "context_ir", "poll", "decode"],
            reason={
                "code": "context_ir_query_unknown",
                "provider": error_trace.provider_response(
                    response, secrets=(request.minimax_api_key,)
                ),
            },
        )
        return "stop"
    if task.get("id") != task_id:
        _mark_terminal(
            path,
            state,
            status="failed",
            error="context_ir_task_mismatch",
        )
        _record_attempt_error(
            request,
            path,
            "task-mismatch",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={"code": "context_ir_task_mismatch", "provider_task": task},
        )
        return "stop"
    status = task.get("status")
    if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
        _mark_terminal(
            path,
            state,
            status="query_unknown",
            error="context_ir_unknown_status",
        )
        _record_attempt_error(
            request,
            path,
            "unknown-status",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={"code": "context_ir_unknown_status", "provider_task": task},
        )
        return "stop"
    modality = task.get("modality")
    if (
        task.get("task_type") != "h3_context_ir"
        or (modality is not None and modality != "text")
    ):
        _mark_terminal(
            path,
            state,
            status="failed",
            error="context_ir_result_type_invalid",
        )
        _record_attempt_error(
            request,
            path,
            "result-type",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={"code": "context_ir_result_type_invalid", "provider_task": task},
        )
        return "stop"
    if status in {"queued", "running"}:
        state["status"] = "polling"
        state["error"] = None
        _persist(path, state)
        return "continue"
    if status == "failed":
        _mark_terminal(
            path,
            state,
            status="failed",
            error="context_ir_provider_failed",
        )
        _record_attempt_error(
            request,
            path,
            "provider-failed",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={"code": "context_ir_provider_failed", "provider_task": task},
        )
        return "stop"
    if status == "cancelled":
        _mark_terminal(
            path,
            state,
            status="failed",
            error="context_ir_cancelled",
        )
        _record_attempt_error(
            request,
            path,
            "cancelled",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={
                "code": "context_ir_cancelled",
                "provider": error_trace.provider_response(
                    response, secrets=(request.minimax_api_key,)
                ),
                "provider_task": task,
            },
        )
        return "stop"
    content = task.get("content")
    effective_prompt = content.get("prompt") if isinstance(content, Mapping) else None
    if not isinstance(effective_prompt, str) or not effective_prompt.strip():
        _mark_terminal(
            path,
            state,
            status="query_unknown",
            error="context_ir_query_unknown",
        )
        _record_attempt_error(
            request,
            path,
            "result-missing",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={
                "code": "context_ir_query_unknown",
                "provider": error_trace.provider_response(
                    response, secrets=(request.minimax_api_key,)
                ),
                "provider_task": task,
            },
        )
        return "stop"
    if len(effective_prompt.encode("utf-8")) > MAX_EFFECTIVE_PROMPT_BYTES:
        _mark_terminal(
            path,
            state,
            status="failed",
            error="context_ir_result_invalid",
        )
        _record_attempt_error(
            request,
            path,
            "result-invalid",
            call_path=["generation", "context_ir", "poll", task_id],
            reason={
                "code": "context_ir_result_invalid",
                "provider": error_trace.provider_response(
                    response, secrets=(request.minimax_api_key,)
                ),
                "provider_task": task,
            },
        )
        return "stop"
    _complete(request, state, path, effective_prompt)
    return "stop"


def _result(
    request: FrozenContextIrRequest,
    state: Mapping[str, Any],
    path: Path,
) -> ContextIrResult:
    status = state["status"]
    receipt_path: Path | None = None
    receipt_sha256: str | None = None
    effective_prompt: str | None = None
    effective_prompt_sha256: str | None = None
    if status == "succeeded":
        receipt_path = path.with_name("receipt.json")
        receipt = load_effective_prompt_receipt(request, receipt_path)
        receipt_sha256 = receipt.receipt_sha256
        effective_prompt = receipt.effective_prompt
        effective_prompt_sha256 = receipt.effective_prompt_sha256
    if status in {"ready", "ready_to_submit"}:
        public_status = "ready"
    elif status in {"submitting", "uploading"}:
        public_status = "submission_unknown"
    elif status == "polling":
        public_status = "running"
    else:
        public_status = status
    return ContextIrResult(
        status=public_status,
        attempt_id=str(state["attempt_id"]),
        provider_task_id=state.get("provider_task_id"),
        effective_prompt=effective_prompt,
        source_prompt_sha256=request.source_prompt_sha256,
        effective_prompt_sha256=effective_prompt_sha256,
        context_ir_request_sha256=state.get("context_ir_request_sha256"),
        context_ir_task_sha256=state.get("context_ir_task_sha256"),
        context_ir_attempt_sha256=request.context_ir_attempt_sha256,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        error_code=state.get("error"),
    )


@contextmanager
def _client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    if client is not None:
        yield client
        return
    with httpx.Client(trust_env=False) as owned:
        yield owned


def optimize_h3_prompt(
    request: FrozenContextIrRequest, *, client: httpx.Client | None = None
) -> ContextIrResult:
    """Start or resume exactly one Context IR attempt for this frozen request."""
    _validate_frozen_request(request)
    with _session_lease(request):
        state, path = _prepare(request)
        terminal = {"succeeded", "failed", "submission_unknown"}
        if (
            state["status"] == "failed"
            and state.get("error") in {
                "context_ir_result_invalid",
                "context_ir_semantic_mismatch",
            }
            and isinstance(state.get("provider_task_id"), str)
            and state.get("receipt") is None
        ):
            # Older coordinators could reject a documented provider response
            # before writing a receipt. The paid task id is already frozen, so
            # recovery may only poll that exact task and can never resubmit.
            state["status"] = "polling"
            state["error"] = None
            _persist(path, state)
        if state["status"] in terminal:
            return _result(request, state, path)
        if state["status"] in {"uploading", "submitting"}:
            _mark_terminal(
                path,
                state,
                status="submission_unknown",
                error="context_ir_submission_unknown",
            )
            return _result(request, state, path)
        with _client(client) as active_client:
            if state["status"] in {"ready", "ready_to_submit"}:
                _submit(request, state, path, active_client)
            if state["status"] in terminal:
                return _result(request, state, path)
            if state["status"] == "query_unknown":
                state["status"] = "polling"
                state["error"] = None
                _persist(path, state)
            deadline = time.monotonic() + request.timeouts.poll_total_s
            last_response: dict[str, object] = {}
            while state["status"] == "polling":
                outcome = _poll_once(
                    request, state, path, active_client, last_response
                )
                if outcome == "stop":
                    break
                if time.monotonic() >= deadline:
                    _mark_terminal(
                        path,
                        state,
                        status="query_unknown",
                        error="context_ir_poll_timeout",
                    )
                    _record_attempt_error(
                        request,
                        path,
                        "poll-timeout",
                        call_path=["generation", "context_ir", "poll"],
                        reason={
                            "code": "context_ir_poll_timeout",
                            "last_response": last_response,
                        },
                    )
                    break
                if request.timeouts.poll_interval_s:
                    time.sleep(request.timeouts.poll_interval_s)
        return _result(request, state, path)


_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "cid",
        "attempt_id",
        "client_request_id",
        "source_prompt",
        "source_prompt_sha256",
        "effective_prompt",
        "effective_prompt_sha256",
        "context_output_prompt",
        "context_output_prompt_sha256",
        "semantic_score",
        "skill_plan_sha256",
        "semantic_contract_sha256",
        "references_sha256",
        "voice_texts_sha256",
        "source_h3_request_sha256",
        "upstream_dialogue_sha256",
        "upstream_artifact_path",
        "upstream_artifact_sha256",
        "upstream_dialogue_sha256_path",
        "context_ir_request_sha256",
        "provider_task_id",
        "context_ir_task_sha256",
        "context_ir_attempt_sha256",
        "receipt_sha256",
    }
)


def load_effective_prompt_receipt(
    request: FrozenContextIrRequest, receipt_path: Path
) -> EffectivePromptReceipt:
    """Fail-closed loader for the only prompt allowed to enter H3."""
    _validate_frozen_request(request)
    path = Path(receipt_path).resolve()
    attempts = _attempts_root(request).resolve()
    try:
        relative = path.relative_to(attempts)
    except ValueError:
        raise ContextIrReceiptError("context_ir_receipt_path_invalid") from None
    if len(relative.parts) != 2 or relative.name != "receipt.json":
        raise ContextIrReceiptError("context_ir_receipt_path_invalid")
    raw = _read_json(path)
    legacy_receipt = set(raw) == _RECEIPT_KEYS - {
        "context_output_prompt",
        "context_output_prompt_sha256",
        "semantic_score",
    }
    if set(raw) != _RECEIPT_KEYS and not legacy_receipt:
        raise ContextIrReceiptError("context_ir_receipt_invalid")
    unhashed = dict(raw)
    receipt_sha256 = unhashed.pop("receipt_sha256", None)
    if not _is_sha256(receipt_sha256) or receipt_sha256 != _canonical_sha256(unhashed):
        raise ContextIrReceiptError("context_ir_receipt_invalid")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "version": SCHEMA_VERSION,
        "cid": request.cid,
        "client_request_id": request.client_request_id,
        "source_prompt": request.source_prompt,
        "source_prompt_sha256": request.source_prompt_sha256,
        "skill_plan_sha256": request.skill_plan_sha256,
        "semantic_contract_sha256": request.semantic_contract_sha256,
        "references_sha256": request.references_sha256,
        "voice_texts_sha256": request.voice_texts_sha256,
        "source_h3_request_sha256": request.source_h3_request_sha256,
        "upstream_dialogue_sha256": request.upstream_dialogue_sha256,
        "upstream_artifact_path": str(request.upstream_artifact_path),
        "upstream_artifact_sha256": request.upstream_artifact_sha256,
        "upstream_dialogue_sha256_path": list(
            request.upstream_dialogue_sha256_path
        ),
        "context_ir_attempt_sha256": request.context_ir_attempt_sha256,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ContextIrReceiptError("context_ir_receipt_mismatch")
    effective_prompt = raw.get("effective_prompt")
    if (
        not isinstance(effective_prompt, str)
        or not effective_prompt.strip()
        or raw.get("effective_prompt_sha256") != _sha256_text(effective_prompt)
    ):
        raise ContextIrReceiptError("context_ir_receipt_invalid")
    if raw.get("source_prompt_sha256") != _sha256_text(str(raw.get("source_prompt"))):
        raise ContextIrReceiptError("context_ir_receipt_invalid")
    try:
        source_relations = _relation_states_contract(str(raw.get("source_prompt")))
        effective_relations = _relation_states_contract(effective_prompt)
    except ContextIrContractError:
        raise ContextIrReceiptError("context_ir_receipt_invalid") from None
    if source_relations != effective_relations:
        raise ContextIrReceiptError("context_ir_receipt_mismatch")
    if legacy_receipt:
        semantic_score: Mapping[str, object] = {}
    else:
        context_output_prompt = raw.get("context_output_prompt")
        semantic_score = raw.get("semantic_score")
        if (
            not isinstance(context_output_prompt, str)
            or not context_output_prompt.strip()
            or raw.get("context_output_prompt_sha256")
            != _sha256_text(context_output_prompt)
            or not isinstance(semantic_score, Mapping)
            or set(semantic_score) not in {
                _SEMANTIC_SCORE_KEYS, _PRE_RELATION_SEMANTIC_SCORE_KEYS,
            }
            or not isinstance(semantic_score.get("speech_expected"), bool)
            or any(
                isinstance(semantic_score.get(key), bool)
                or not isinstance(semantic_score.get(key), (int, float))
                or not 0.0 <= float(semantic_score[key]) <= 1.0
                for key in (
                    "speech", "keyframe_timeline", "music_policy", "overall",
                    *(('relation_states',) if 'relation_states' in semantic_score else ()),
                )
            )
            or not isinstance(semantic_score.get("dialogue_policy"), Mapping)
            or set(semantic_score["dialogue_policy"])
            != _DIALOGUE_POLICY_SCORE_KEYS
            or any(
                isinstance(
                    semantic_score["dialogue_policy"].get(key), bool
                )
                or not isinstance(
                    semantic_score["dialogue_policy"].get(key),
                    (int, float),
                )
                or not 0.0 <= float(
                    semantic_score["dialogue_policy"][key]
                ) <= 1.0
                for key in _DIALOGUE_POLICY_SCORE_KEYS
            )
        ):
            raise ContextIrReceiptError("context_ir_receipt_invalid")
    for key in (
        "context_ir_request_sha256",
        "context_ir_task_sha256",
        "context_ir_attempt_sha256",
    ):
        if not _is_sha256(raw.get(key)):
            raise ContextIrReceiptError("context_ir_receipt_invalid")
    provider_task_id = raw.get("provider_task_id")
    if not isinstance(provider_task_id, str) or not provider_task_id:
        raise ContextIrReceiptError("context_ir_receipt_invalid")
    attempt_path = path.with_name("attempt.json")
    state = _read_json(attempt_path)
    _validate_state(request, state, attempt_path)
    if (
        state.get("status") != "succeeded"
        or state.get("receipt")
        != {"path": "receipt.json", "sha256": receipt_sha256}
        or state.get("context_ir_request_sha256")
        != raw.get("context_ir_request_sha256")
        or state.get("context_ir_task_sha256") != raw.get("context_ir_task_sha256")
        or state.get("provider_task_id") != provider_task_id
    ):
        raise ContextIrReceiptError("context_ir_receipt_mismatch")
    return EffectivePromptReceipt(
        receipt_path=path,
        receipt_sha256=receipt_sha256,
        cid=request.cid,
        attempt_id=str(raw["attempt_id"]),
        client_request_id=request.client_request_id,
        source_prompt=request.source_prompt,
        source_prompt_sha256=request.source_prompt_sha256,
        effective_prompt=effective_prompt,
        effective_prompt_sha256=str(raw["effective_prompt_sha256"]),
        skill_plan_sha256=request.skill_plan_sha256,
        semantic_contract_sha256=request.semantic_contract_sha256,
        references_sha256=request.references_sha256,
        voice_texts_sha256=request.voice_texts_sha256,
        source_h3_request_sha256=request.source_h3_request_sha256,
        upstream_dialogue_sha256=request.upstream_dialogue_sha256,
        upstream_artifact_path=request.upstream_artifact_path,
        upstream_artifact_sha256=request.upstream_artifact_sha256,
        upstream_dialogue_sha256_path=request.upstream_dialogue_sha256_path,
        context_ir_request_sha256=str(raw["context_ir_request_sha256"]),
        provider_task_id=provider_task_id,
        context_ir_task_sha256=str(raw["context_ir_task_sha256"]),
        context_ir_attempt_sha256=request.context_ir_attempt_sha256,
        semantic_score=dict(semantic_score),
    )


def apply_effective_prompt(
    request: FrozenContextIrRequest,
    receipt_path: Path | None,
) -> h3.H3Request:
    """Revalidate the frozen request and receipt, then replace only the prompt."""
    _validate_frozen_request(request)
    if receipt_path is None:
        raise ContextIrContractError("context_ir_effective_prompt_unavailable")
    receipt = load_effective_prompt_receipt(request, Path(receipt_path))
    if (
        h3.is_multimodal_request(request.source_h3_request)
        or request.source_h3_request.context_ir_required
    ):
        return replace(
            request.source_h3_request,
            prompt=receipt.effective_prompt,
            context_ir_receipt_path=receipt.receipt_path,
            context_ir_receipt_sha256=receipt.receipt_sha256,
        )
    return replace(request.source_h3_request, prompt=receipt.effective_prompt)
