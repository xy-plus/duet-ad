"""Strict video-maker plan consumer for H3 native image/audio generation.

This module owns semantic validation and deterministic Context-IR compilation.
Provider I/O and recovery remain exclusively in :mod:`app.h3`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from app import context_ir_bridge, h3


COMPILER_VERSION = "duet.h3-context-ir.multimodal-audio.v2"
MAX_PROMPT_CHARS = 7000
_TOP_KEYS = {
    "version",
    "phase",
    "eligible",
    "reason",
    "visual_prompt",
    "dialogue_source_sha256",
    "subjects",
    "audio_refs",
    "speech_bindings",
    "sound_design",
}


class MultimodalContractError(RuntimeError):
    """Stable pre-provider contract failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise MultimodalContractError(code)


def _object(value: Any, keys: set[str], code: str = "skill_plan_shape_invalid") -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(code)
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        _fail("skill_plan_shape_invalid")
    return value


def _positive_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(code)
    return value


def _text(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "<" in value
        or ">" in value
        or any(ord(character) < 32 and character not in "\n\t" for character in value)
    ):
        _fail(code)
    return value


@dataclass(frozen=True, slots=True)
class FrozenVisualInput:
    """Already-read visual semantics and ordered image bytes."""

    prompt: str
    keyframes: h3.FrozenKeyframes
    prompt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.prompt, "visual_prompt_invalid")
        if len(self.prompt) > MAX_PROMPT_CHARS:
            _fail("visual_prompt_invalid")
        if not isinstance(self.keyframes, tuple) or not 1 <= len(self.keyframes) <= 9:
            _fail("keyframes_not_frozen")
        names: list[str] = []
        for frame in self.keyframes:
            try:
                path, data = frame
            except (TypeError, ValueError):
                _fail("keyframes_not_frozen")
            if (
                not isinstance(path, Path)
                or not isinstance(data, bytes)
                or not data
                or path.name != Path(path.name).name
            ):
                _fail("keyframes_not_frozen")
            names.append(path.name)
        if len(names) != len(set(names)):
            _fail("duplicate_keyframe_name")
        object.__setattr__(
            self,
            "prompt_sha256",
            hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class _Subject:
    subject_id: str
    picture_refs: tuple[int, ...]
    voice_ref: int | None


@dataclass(frozen=True, slots=True)
class _Speech:
    order: int
    line_index: int
    start_s: float
    end_s: float
    language: str
    text: str
    delivery: Literal["on_screen", "off_screen_voiceover"]
    subject_id: str | None
    voice_ref: int | None


@dataclass(frozen=True, slots=True)
class _Sound:
    audio_index: int
    description: str
    purpose: Literal["ambience", "effect"]


def _consume_plan(
    skill_plan: Any,
    *,
    visual: FrozenVisualInput,
    reference_audios: h3.FrozenReferenceAudios,
    upstream_dialogue: Sequence[Mapping[str, Any]],
    upstream_dialogue_receipt_sha256: str,
) -> tuple[tuple[_Subject, ...], tuple[_Speech, ...], tuple[_Sound, ...]]:
    plan = _object(skill_plan, _TOP_KEYS)
    if plan["version"] != 2 or plan["phase"] != "multimodal_audio":
        _fail("skill_plan_shape_invalid")
    if plan["eligible"] is not True:
        _fail("skill_plan_ineligible")
    if plan["reason"] is not None or plan["visual_prompt"] != visual.prompt:
        _fail("skill_plan_visual_mismatch")
    if plan["dialogue_source_sha256"] != upstream_dialogue_receipt_sha256:
        _fail("upstream_dialogue_receipt_mismatch")

    subjects: list[_Subject] = []
    used_pictures: set[int] = set()
    for expected, raw in enumerate(_list(plan["subjects"]), 1):
        item = _object(raw, {"subject_id", "picture_refs", "voice_ref"})
        if item["subject_id"] != f"S{expected}":
            _fail("subject_order_invalid")
        pictures = tuple(_positive_int(value, "subject_picture_refs_invalid") for value in _list(item["picture_refs"]))
        if not pictures or pictures != tuple(sorted(set(pictures))):
            _fail("subject_picture_refs_invalid")
        if any(value > len(visual.keyframes) for value in pictures):
            _fail("subject_picture_refs_invalid")
        if used_pictures.intersection(pictures):
            _fail("picture_reused")
        used_pictures.update(pictures)
        subjects.append(
            _Subject(
                item["subject_id"],
                pictures,
                (
                    None
                    if item["voice_ref"] is None
                    else _positive_int(
                        item["voice_ref"], "subject_voice_ref_invalid"
                    )
                ),
            )
        )

    subjects_by_id = {subject.subject_id: subject for subject in subjects}
    subject_ids = set(subjects_by_id)
    dialogue_source = list(upstream_dialogue)
    dialogue: list[_Speech] = []
    for expected, raw in enumerate(_list(plan["speech_bindings"]), 1):
        item = _object(
            raw,
            {
                "line_index", "delivery", "subject_id", "language", "voice_ref",
            },
        )
        line_index = _positive_int(
            item["line_index"], "dialogue_line_index_invalid"
        )
        if line_index != expected:
            _fail("dialogue_order_invalid")
        try:
            source = dialogue_source[line_index - 1]
            text = source["text"]
            start_s = float(source["start_s"])
            end_s = float(source["end_s"])
        except (IndexError, KeyError, TypeError, ValueError):
            _fail("upstream_dialogue_invalid")
        text = _text(text, "upstream_dialogue_invalid")
        if not (0 <= start_s < end_s):
            _fail("upstream_dialogue_invalid")
        delivery = item["delivery"]
        subject_id = item["subject_id"]
        voice_ref = item["voice_ref"]
        if delivery == "on_screen":
            if (
                subject_id not in subject_ids
                or subjects_by_id[subject_id].voice_ref is None
                or voice_ref is not None
            ):
                _fail("dialogue_subject_invalid")
        elif delivery == "off_screen_voiceover":
            if subject_id is not None:
                _fail("dialogue_subject_invalid")
            if voice_ref is not None:
                voice_ref = _positive_int(
                    voice_ref, "dialogue_voice_ref_invalid"
                )
        else:
            _fail("dialogue_delivery_invalid")
        dialogue.append(
            _Speech(
                expected,
                line_index,
                start_s,
                end_s,
                _text(item["language"], "dialogue_language_invalid"),
                text,
                delivery,
                subject_id,
                voice_ref,
            )
        )
    if len(dialogue) != len(dialogue_source):
        _fail("upstream_dialogue_mismatch")

    sound_design = _object(
        plan["sound_design"], {"ambience_refs", "effects"}
    )
    speech = tuple(dialogue)

    sounds: list[_Sound] = []
    for field_name, purpose in (("ambience_refs", "ambience"), ("effects", "effect")):
        for raw in _list(sound_design[field_name]):
            item = _object(raw, {"audio_index", "description"})
            sounds.append(
                _Sound(
                    _positive_int(item["audio_index"], "sound_reference_invalid"),
                    _text(item["description"], "sound_reference_invalid"),
                    purpose,
                )
            )

    audio_refs = _list(plan["audio_refs"])
    if len(audio_refs) != len(reference_audios):
        _fail("skill_audio_refs_mismatch")
    subject_voice_refs = {
        subject.voice_ref: subject.subject_id
        for subject in subjects
        if subject.voice_ref is not None
    }
    if len(subject_voice_refs) != sum(
        subject.voice_ref is not None for subject in subjects
    ):
        _fail("voice_reference_reused")
    narrator_voice_refs = {
        line.voice_ref
        for line in speech
        if line.delivery == "off_screen_voiceover" and line.voice_ref is not None
    }
    claimed_voice_refs = set(subject_voice_refs).union(narrator_voice_refs)
    actual_voice_refs: set[int] = set()
    sound_by_audio = {sound.audio_index: sound for sound in sounds}
    if len(sound_by_audio) != len(sounds):
        _fail("sound_reference_duplicate")
    for expected, raw in enumerate(audio_refs, 1):
        item = _object(raw, {"audio_index", "purpose", "subject_id"})
        if item["audio_index"] != expected:
            _fail("skill_audio_refs_mismatch")
        if item["purpose"] not in {"voice", "ambience", "effect"}:
            _fail("skill_audio_refs_mismatch")
        audio = reference_audios[expected - 1]
        if audio.order != expected or audio.purpose != item["purpose"]:
            _fail("skill_audio_refs_mismatch")
        if item["purpose"] == "voice":
            actual_voice_refs.add(expected)
            expected_subject = subject_voice_refs.get(expected)
            is_narrator = expected in narrator_voice_refs
            if expected_subject is None and not is_narrator:
                _fail("voice_reference_unbound")
            if item["subject_id"] != expected_subject:
                _fail("skill_audio_refs_mismatch")
        else:
            sound = sound_by_audio.get(expected)
            if item["subject_id"] is not None or sound is None or sound.purpose != item["purpose"]:
                _fail("sound_reference_unbound")
    if claimed_voice_refs != actual_voice_refs:
        _fail("voice_reference_unbound")
    if any(sound.audio_index > len(reference_audios) for sound in sounds):
        _fail("sound_reference_unbound")
    return tuple(subjects), speech, tuple(sounds)


def _compile_prompt(
    visual: FrozenVisualInput,
    audios: h3.FrozenReferenceAudios,
    subjects: tuple[_Subject, ...],
    speech: tuple[_Speech, ...],
    sounds: tuple[_Sound, ...],
) -> str:
    definitions = [
        f"<Picture {index}> is ordered visual reference {path.name}."
        for index, (path, _data) in enumerate(visual.keyframes, 1)
    ]
    for subject in subjects:
        pictures = ", ".join(f"<Picture {value}>" for value in subject.picture_refs)
        voice = (
            f"; <Audio {subject.voice_ref}> is this subject's voice conditioning reference"
            if subject.voice_ref is not None
            else "; this visible subject remains silent"
        )
        definitions.append(
            f"<Subject {subject.subject_id[1:]}>({subject.subject_id}) appears only in {pictures}{voice}."
        )
    sound_by_audio = {sound.audio_index: sound for sound in sounds}
    narrator_refs = {line.voice_ref for line in speech if line.subject_id is None and line.voice_ref is not None}
    for audio in audios:
        if audio.purpose in {"ambience", "effect"}:
            sound = sound_by_audio[audio.order]
            definitions.append(
                f"<Audio {audio.order}> is {audio.purpose} conditioning only: {sound.description}."
            )
        elif audio.order in narrator_refs:
            definitions.append(
                f"<Audio {audio.order}> is the off-screen narrator's voice conditioning reference."
            )
    events: list[str] = []
    subjects_by_id = {subject.subject_id: subject for subject in subjects}
    for line in speech:
        if line.delivery == "on_screen":
            speaker = f"<Subject {line.subject_id[1:]}>({line.subject_id})"
            marker_voice_ref = subjects_by_id[str(line.subject_id)].voice_ref
            marker_delivery = "on_screen"
            delivery = (
                " The subject's visible lips articulate exactly this line in sync "
                "with the jointly generated speech."
            )
        elif line.voice_ref is None:
            speaker = "The off-screen narrator using the provider default voice"
            marker_voice_ref = None
            marker_delivery = "off_screen"
            delivery = (
                " This is an off-screen voiceover; every visible person's lips "
                "remain completely closed during it."
            )
        else:
            speaker = f"The off-screen narrator using <Audio {line.voice_ref}>"
            marker_voice_ref = line.voice_ref
            marker_delivery = "off_screen"
            delivery = (
                " This is an off-screen voiceover; every visible person's lips "
                "remain completely closed during it."
            )
        events.extend((
            context_ir_bridge.format_speech_marker(
                order=line.order,
                delivery=marker_delivery,
                subject_id=line.subject_id,
                voice_ref=marker_voice_ref,
                language=line.language,
                text=line.text,
            ),
            f"[{line.order}] [{line.start_s:.3f}-{line.end_s:.3f}s] "
            f"{speaker}.{delivery}",
        ))
    if not events:
        events.append(
            "No audible speech is specified; every visible person's lips remain "
            "completely closed."
        )
    soundscape = [
        f"<Audio {sound.audio_index}> {sound.purpose}: {sound.description}."
        for sound in sounds
    ] or ["No ambience or effects are specified."]
    prompt = (
        "references:\n"
        + "\n".join(definitions)
        + "\n\nsource_visual_semantics:\n"
        + visual.prompt
        + "\n\nspeaking_sequence:\n"
        + "\n".join(events)
        + "\n\noverall_soundscape:\n"
        + "\n".join(soundscape)
        + "\n\nconstraints:\nGenerate picture and native sound jointly. Preserve the exact "
        "1-based bindings and speaking order above. Audio inputs are conditioning references "
        "only, not target tracks or timing locks. Do not invent dialogue, speakers, subtitles, "
        "music, non-diegetic background music, or unlisted sounds."
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        _fail("effective_prompt_too_large")
    return prompt


def build_h3_request(
    *,
    skill_plan: Mapping[str, Any],
    approved_skill_plan_sha256: str,
    upstream_dialogue: Sequence[Mapping[str, Any]],
    upstream_dialogue_receipt_sha256: str,
    upstream_dialogue_content_sha256: str,
    visual: FrozenVisualInput,
    reference_audios: Sequence[h3.FrozenReferenceAudio],
    mode: Literal["multimodal", "multimodal_hd"],
    cid: str,
    workdir: Path,
    client_request_id: str,
    duration: int,
    resolution: str,
    aspect_ratio: str,
    autodl_token: str,
    timeouts: h3.Timeouts = h3.Timeouts(),
    seed: int | None = None,
) -> h3.H3Request:
    """Freeze one approved Skill plan into the existing recoverable H3 request."""
    if not isinstance(visual, FrozenVisualInput):
        _fail("visual_input_invalid")
    plan_sha256 = h3.canonical_json_sha256(skill_plan)
    if approved_skill_plan_sha256 != plan_sha256:
        _fail("skill_plan_approval_mismatch")
    audios = tuple(reference_audios)
    dialogue = tuple(dict(line) for line in upstream_dialogue)
    content_sha256 = h3.canonical_json_sha256(list(dialogue))
    try:
        valid_dialogue_receipt = (
            isinstance(upstream_dialogue_receipt_sha256, str)
            and len(upstream_dialogue_receipt_sha256) == 64
            and int(upstream_dialogue_receipt_sha256, 16) >= 0
            and upstream_dialogue_content_sha256 == content_sha256
        )
    except ValueError:
        valid_dialogue_receipt = False
    if not valid_dialogue_receipt:
        _fail("upstream_dialogue_receipt_mismatch")
    subjects, speech, sounds = _consume_plan(
        skill_plan,
        visual=visual,
        reference_audios=audios,
        upstream_dialogue=dialogue,
        upstream_dialogue_receipt_sha256=(
            upstream_dialogue_receipt_sha256
        ),
    )
    prompt = _compile_prompt(visual, audios, subjects, speech, sounds)
    workflows = {
        "multimodal": h3.H3_MULTIMODAL_WORKFLOW,
        "multimodal_hd": h3.H3_MULTIMODAL_HD_WORKFLOW,
    }
    if mode not in workflows:
        _fail("invalid_multimodal_mode")
    voice_texts = tuple(line.text for line in speech)
    return h3.H3Request(
        cid=cid,
        workdir=Path(workdir),
        client_request_id=client_request_id,
        prompt=prompt,
        keyframes=visual.keyframes,
        voice_texts=voice_texts,
        voice_receipt=h3.voice_texts_receipt(voice_texts),
        duration=duration,
        autodl_token=autodl_token,
        timeouts=timeouts,
        mode="reference",
        seed=seed,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        workflow=workflows[mode],
        reference_audios=audios,
        skill_plan_sha256=plan_sha256,
        multimodal_compiler_version=COMPILER_VERSION,
        upstream_dialogue_receipt_sha256=upstream_dialogue_receipt_sha256,
        audio_required=True,
    )
