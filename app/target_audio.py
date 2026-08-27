"""Route-neutral target-audio planning and receipt-first materialization.

The shared plan freezes dialogue, verified speaker mapping and optional clean
voice references.  It deliberately does not build an H3 prompt: route A may
compile these semantics into Context-IR, while route B materializes complete
target tracks through an injected provider boundary.

``decoded_sha256`` always means SHA-256 of canonical interleaved little-endian
signed 16-bit PCM at the recorded sample rate and channel count.  The injected
probe owns decoding MP3/WAV into that representation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence


RECEIPT_SCHEMA = "duet.target-audio-plan"
RECEIPT_VERSION = 1
RECEIPT_FILENAME = "target_audio_plan.json"

AudioMode = Literal[
    "keep_content", "edit_dialogue", "translate", "add_dialogue", "replace_voice"
]
AudioFormat = Literal["mp3", "wav"]
MaterialRole = Literal["dialogue", "sound_effect", "ambience"]
VoiceStrategyKind = Literal["voice_reference", "target_voice"]
SpeakerKind = Literal["subject", "narrator"]
AudioReferencePurpose = Literal["voice", "ambience", "sound_effect"]
MaterializationStatus = Literal[
    "ready_to_submit",
    "submitting",
    "submission_unknown",
    "processing",
    "failed",
    "validating",
    "output_invalid",
    "succeeded",
]

_AUDIO_MODES = frozenset(
    {"keep_content", "edit_dialogue", "translate", "add_dialogue", "replace_voice"}
)
_AUDIO_FORMATS = frozenset({"mp3", "wav"})
_MATERIAL_ROLES = frozenset({"dialogue", "sound_effect", "ambience"})
_SPEAKER_KINDS = frozenset({"subject", "narrator"})
_REFERENCE_PURPOSES = frozenset({"voice", "ambience", "sound_effect"})
_MATERIALIZATION_RESULTS = frozenset(
    {"succeeded", "processing", "failed", "submission_unknown"}
)
_STORED_STATUSES = frozenset(
    {
        "ready_to_submit",
        "submitting",
        "submission_unknown",
        "processing",
        "failed",
        "validating",
        "output_invalid",
        "succeeded",
    }
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TargetAudioError(RuntimeError):
    """The plan, frozen files, receipt, or materialization is unsafe to use."""


@dataclass(frozen=True)
class AudioProbe:
    """Validated audio metadata returned by an injected ffprobe/decoder adapter."""

    format: AudioFormat
    time_base: int
    duration_pts: int
    sample_rate_hz: int
    channels: int
    decoded_sha256: str
    decoded_format: str = "pcm_s16le"


@dataclass(frozen=True)
class CostEstimate:
    currency: str
    amount_micros: int


@dataclass(frozen=True)
class VoiceStrategy:
    kind: VoiceStrategyKind
    voice_reference_id: str | None = None
    target_voice: str | None = None


@dataclass(frozen=True)
class SpeakerPlan:
    speaker_id: str
    language: str
    voice_strategy: VoiceStrategy
    kind: SpeakerKind = "subject"
    subject_id: str | None = None


@dataclass(frozen=True)
class DialogueLine:
    line_id: str
    order: int
    speaker_id: str
    language: str
    time_base: int
    start_pts: int
    end_pts: int
    text: str


@dataclass(frozen=True)
class FrozenDialogueLine:
    line_id: str
    order: int
    speaker_id: str
    language: str
    time_base: int
    start_pts: int
    end_pts: int
    text: str
    voice_strategy: VoiceStrategy


@dataclass(frozen=True)
class AudioReference:
    reference_id: str
    order: int
    speaker_id: str | None
    path: Path
    purpose: AudioReferencePurpose = "voice"
    description: str | None = None


@dataclass(frozen=True)
class AudioCue:
    cue_id: str
    order: int
    role: Literal["sound_effect", "ambience"]
    time_base: int
    start_pts: int
    end_pts: int
    prompt: str


@dataclass(frozen=True)
class TargetMaterialSpec:
    material_id: str
    order: int
    role: MaterialRole
    format: AudioFormat
    time_base: int
    duration_pts: int


@dataclass(frozen=True)
class MaterializerSpec:
    provider: str
    model: str
    version: str
    prompt: str
    cost: CostEstimate


@dataclass(frozen=True)
class TargetAudioRequest:
    client_request_id: str
    mode: AudioMode
    source_audio: Path | None
    dialogue: tuple[DialogueLine, ...]
    speaker_plan: tuple[SpeakerPlan, ...]
    audio_refs: tuple[AudioReference, ...]
    effects: tuple[AudioCue, ...]
    target_materials: tuple[TargetMaterialSpec, ...]
    speaker_mapping_verified: bool
    speaker_mapping_source: str
    materializer: MaterializerSpec


@dataclass(frozen=True)
class FrozenAudioArtifact:
    path: Path
    relative_path: str
    data: bytes
    format: AudioFormat
    size_bytes: int
    sha256: str
    probe: AudioProbe

    @property
    def time_base(self) -> int:
        return self.probe.time_base

    @property
    def duration_pts(self) -> int:
        return self.probe.duration_pts

    @property
    def decoded_sha256(self) -> str:
        return self.probe.decoded_sha256


@dataclass(frozen=True)
class FrozenAudioReference:
    reference_id: str
    order: int
    purpose: AudioReferencePurpose
    speaker_id: str | None
    description: str | None
    audio: FrozenAudioArtifact

    @property
    def path(self) -> Path:
        return self.audio.path

    @property
    def relative_path(self) -> str:
        return self.audio.relative_path

    @property
    def data(self) -> bytes:
        return self.audio.data

    @property
    def format(self) -> AudioFormat:
        return self.audio.format

    @property
    def size_bytes(self) -> int:
        return self.audio.size_bytes

    @property
    def sha256(self) -> str:
        return self.audio.sha256

    @property
    def probe(self) -> AudioProbe:
        return self.audio.probe

    @property
    def time_base(self) -> int:
        return self.audio.time_base

    @property
    def duration_pts(self) -> int:
        return self.audio.duration_pts

    @property
    def decoded_sha256(self) -> str:
        return self.audio.decoded_sha256


@dataclass(frozen=True)
class TargetAudioMaterial:
    material_id: str
    order: int
    role: MaterialRole
    audio: FrozenAudioArtifact

    @property
    def path(self) -> Path:
        return self.audio.path

    @property
    def relative_path(self) -> str:
        return self.audio.relative_path

    @property
    def data(self) -> bytes:
        return self.audio.data

    @property
    def format(self) -> AudioFormat:
        return self.audio.format

    @property
    def size_bytes(self) -> int:
        return self.audio.size_bytes

    @property
    def sha256(self) -> str:
        return self.audio.sha256

    @property
    def probe(self) -> AudioProbe:
        return self.audio.probe

    @property
    def decoded_sha256(self) -> str:
        return self.audio.decoded_sha256


@dataclass(frozen=True)
class MaterializedOutput:
    material_id: str
    path: Path
    format: AudioFormat
    time_base: int
    duration_pts: int
    size_bytes: int
    sha256: str
    decoded_sha256: str


@dataclass(frozen=True)
class MaterializationResult:
    status: Literal["succeeded", "processing", "failed", "submission_unknown"]
    task_id: str | None = None
    outputs: tuple[MaterializedOutput, ...] = ()


@dataclass(frozen=True)
class MaterializationRequest:
    plan_receipt: str
    client_request_id: str
    mode: AudioMode
    source_audio: FrozenAudioArtifact | None
    dialogue: tuple[FrozenDialogueLine, ...]
    speaker_plan: tuple[SpeakerPlan, ...]
    audio_refs: tuple[FrozenAudioReference, ...]
    effects: tuple[AudioCue, ...]
    target_materials: tuple[TargetMaterialSpec, ...]
    materializer: MaterializerSpec


class Materializer(Protocol):
    def submit(self, request: MaterializationRequest) -> MaterializationResult: ...

    def get(
        self, task_id: str, request: MaterializationRequest
    ) -> MaterializationResult: ...


AudioProbeFn = Callable[[Path], AudioProbe]


@dataclass(frozen=True)
class TargetAudioPlan:
    receipt_path: Path
    plan_receipt: str
    client_request_id: str
    mode: AudioMode
    source_audio: FrozenAudioArtifact | None
    dialogue: tuple[FrozenDialogueLine, ...]
    speaker_plan: tuple[SpeakerPlan, ...]
    audio_refs: tuple[FrozenAudioReference, ...]
    effects: tuple[AudioCue, ...]
    target_material_specs: tuple[TargetMaterialSpec, ...]
    target_materials: tuple[TargetAudioMaterial, ...]
    speaker_mapping_verified: bool
    speaker_mapping_source: str
    materializer: MaterializerSpec
    materialization_status: MaterializationStatus
    task_id: str | None

    def materialization_request(self) -> MaterializationRequest:
        return MaterializationRequest(
            plan_receipt=self.plan_receipt,
            client_request_id=self.client_request_id,
            mode=self.mode,
            source_audio=self.source_audio,
            dialogue=self.dialogue,
            speaker_plan=self.speaker_plan,
            audio_refs=self.audio_refs,
            effects=self.effects,
            target_materials=self.target_material_specs,
            materializer=self.materializer,
        )


@dataclass(frozen=True)
class _NormalizedRequest:
    request: TargetAudioRequest
    source_audio: FrozenAudioArtifact | None
    dialogue: tuple[FrozenDialogueLine, ...]
    audio_refs: tuple[FrozenAudioReference, ...]
    core: dict
    plan_receipt: str


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TargetAudioError(f"target audio value is not canonical JSON: {exc}") from None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetAudioError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: object, label: str) -> str:
    text = _non_empty(value, label)
    if _ID_RE.fullmatch(text) is None:
        raise TargetAudioError(f"{label} is invalid")
    return text


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TargetAudioError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TargetAudioError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolved_root(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink():
        raise TargetAudioError("target audio root must not be a symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        raise TargetAudioError("target audio root is missing") from None
    if not resolved.is_dir():
        raise TargetAudioError("target audio root is missing")
    return resolved


def _lexical_inside(root: Path, path: Path, label: str) -> tuple[Path, str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise TargetAudioError(f"{label} escapes target audio root") from None
    if not relative.parts:
        raise TargetAudioError(f"{label} must be a regular file")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TargetAudioError(f"{label} must not contain a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise TargetAudioError(f"{label} escapes target audio root or is missing") from None
    return resolved, relative.as_posix()


def _read_regular_file(root: Path, path: Path, label: str) -> tuple[Path, str, bytes]:
    resolved, relative = _lexical_inside(root, path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise TargetAudioError(f"{label} is not safely readable: {exc}") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise TargetAudioError(f"{label} must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        raise TargetAudioError(f"{label} is not safely readable: {exc}") from None
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if not data:
        raise TargetAudioError(f"{label} is empty")
    return resolved, relative, data


def _normalize_probe(value: object, path: Path, label: str) -> AudioProbe:
    if not isinstance(value, AudioProbe):
        raise TargetAudioError(f"{label} probe returned an invalid result")
    audio_format = str(value.format).lower()
    if audio_format not in _AUDIO_FORMATS:
        raise TargetAudioError(f"{label} format must be MP3 or WAV")
    suffix = path.suffix.lower().removeprefix(".")
    if suffix != audio_format:
        raise TargetAudioError(f"{label} format does not match its file extension")
    time_base = _integer(value.time_base, f"{label}.time_base", minimum=1)
    duration_pts = _integer(value.duration_pts, f"{label}.duration_pts", minimum=1)
    sample_rate = _integer(value.sample_rate_hz, f"{label}.sample_rate_hz", minimum=1)
    channels = _integer(value.channels, f"{label}.channels", minimum=1)
    if channels > 8:
        raise TargetAudioError(f"{label}.channels is invalid")
    if value.decoded_format != "pcm_s16le":
        raise TargetAudioError(f"{label}.decoded_format must be pcm_s16le")
    decoded = _digest(value.decoded_sha256, f"{label}.decoded_sha256")
    return AudioProbe(
        format=audio_format,
        time_base=time_base,
        duration_pts=duration_pts,
        sample_rate_hz=sample_rate,
        channels=channels,
        decoded_sha256=decoded,
    )


def _freeze_audio(
    root: Path, path: Path, *, label: str, probe: AudioProbeFn
) -> FrozenAudioArtifact:
    resolved, relative, data = _read_regular_file(root, path, label)
    try:
        metadata = _normalize_probe(probe(resolved), resolved, label)
    except TargetAudioError:
        raise
    except Exception as exc:
        raise TargetAudioError(f"{label} probe failed: {type(exc).__name__}") from None
    return FrozenAudioArtifact(
        path=resolved,
        relative_path=relative,
        data=data,
        format=metadata.format,
        size_bytes=len(data),
        sha256=_sha256(data),
        probe=metadata,
    )


def _probe_payload(probe: AudioProbe) -> dict:
    return {
        "format": probe.format,
        "time_base": probe.time_base,
        "duration_pts": probe.duration_pts,
        "sample_rate_hz": probe.sample_rate_hz,
        "channels": probe.channels,
        "decoded_format": probe.decoded_format,
        "decoded_sha256": probe.decoded_sha256,
    }


def _audio_binding(audio: FrozenAudioArtifact) -> dict:
    return {
        "path": audio.relative_path,
        "format": audio.format,
        "size_bytes": audio.size_bytes,
        "sha256": audio.sha256,
        "probe": _probe_payload(audio.probe),
    }


def _strategy_payload(strategy: VoiceStrategy, label: str) -> dict:
    if not isinstance(strategy, VoiceStrategy):
        raise TargetAudioError(f"{label} must be a VoiceStrategy")
    if strategy.kind == "voice_reference":
        reference_id = _identifier(strategy.voice_reference_id, f"{label}.voice_reference_id")
        if strategy.target_voice is not None:
            raise TargetAudioError(f"{label} voice_reference cannot set target_voice")
        normalized = VoiceStrategy("voice_reference", reference_id, None)
    elif strategy.kind == "target_voice":
        target = _non_empty(strategy.target_voice, f"{label}.target_voice")
        if strategy.voice_reference_id is not None:
            raise TargetAudioError(f"{label} target_voice cannot set voice_reference_id")
        normalized = VoiceStrategy("target_voice", None, target)
    else:
        raise TargetAudioError(f"{label}.kind is invalid")
    return {
        "kind": normalized.kind,
        "voice_reference_id": normalized.voice_reference_id,
        "target_voice": normalized.target_voice,
    }


def _strategy_from_payload(payload: object, label: str) -> VoiceStrategy:
    if not isinstance(payload, dict) or set(payload) != {
        "kind", "voice_reference_id", "target_voice"
    }:
        raise TargetAudioError(f"{label} is invalid")
    strategy = VoiceStrategy(
        kind=payload["kind"],
        voice_reference_id=payload["voice_reference_id"],
        target_voice=payload["target_voice"],
    )
    normalized = _strategy_payload(strategy, label)
    return VoiceStrategy(**normalized)


def _normalize_cost(cost: object) -> CostEstimate:
    if not isinstance(cost, CostEstimate):
        raise TargetAudioError("materializer.cost must be a CostEstimate")
    currency = _non_empty(cost.currency, "materializer.cost.currency").upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise TargetAudioError("materializer.cost.currency must be a 3-letter code")
    amount = _integer(
        cost.amount_micros, "materializer.cost.amount_micros", minimum=0
    )
    return CostEstimate(currency, amount)


def _normalize_materializer(value: object) -> MaterializerSpec:
    if not isinstance(value, MaterializerSpec):
        raise TargetAudioError("materializer must be a MaterializerSpec")
    return MaterializerSpec(
        provider=_non_empty(value.provider, "materializer.provider"),
        model=_non_empty(value.model, "materializer.model"),
        version=_non_empty(value.version, "materializer.version"),
        prompt=_non_empty(value.prompt, "materializer.prompt"),
        cost=_normalize_cost(value.cost),
    )


def _materializer_payload(value: MaterializerSpec) -> dict:
    return {
        "provider": value.provider,
        "model": value.model,
        "version": value.version,
        "prompt": value.prompt,
        "prompt_sha256": _sha256(value.prompt.encode("utf-8")),
        "cost": {
            "currency": value.cost.currency,
            "amount_micros": value.cost.amount_micros,
        },
    }


def _normalize_request(
    root: Path, request: TargetAudioRequest, probe: AudioProbeFn
) -> _NormalizedRequest:
    if not isinstance(request, TargetAudioRequest):
        raise TargetAudioError("request must be a TargetAudioRequest")
    client_request_id = _identifier(request.client_request_id, "client_request_id")
    if request.mode not in _AUDIO_MODES:
        raise TargetAudioError("target audio mode is invalid")
    if request.speaker_mapping_verified is not True:
        raise TargetAudioError("speaker mapping must be explicitly verified")
    mapping_source = _non_empty(
        request.speaker_mapping_source, "speaker_mapping_source"
    )
    source_key = re.sub(r"[-\s]+", "_", mapping_source.casefold())
    if (
        source_key in {"asr", "automatic_speech_recognition"}
        or "asr" in source_key and ("coarse" in source_key or "raw" in source_key)
    ):
        raise TargetAudioError("coarse ASR windows are not verified speaker mapping")

    materializer = _normalize_materializer(request.materializer)
    source_audio = (
        None
        if request.source_audio is None
        else _freeze_audio(root, request.source_audio, label="source_audio", probe=probe)
    )

    speakers: list[SpeakerPlan] = []
    speaker_by_id: dict[str, SpeakerPlan] = {}
    subject_ids: set[str] = set()
    for index, speaker in enumerate(request.speaker_plan):
        if not isinstance(speaker, SpeakerPlan):
            raise TargetAudioError(f"speaker_plan[{index}] is invalid")
        speaker_id = _identifier(speaker.speaker_id, f"speaker_plan[{index}].speaker_id")
        if speaker_id in speaker_by_id:
            raise TargetAudioError("speaker_plan contains duplicate speaker_id")
        language = _non_empty(speaker.language, f"speaker_plan[{index}].language")
        if speaker.kind not in _SPEAKER_KINDS:
            raise TargetAudioError(f"speaker_plan[{index}].kind is invalid")
        if speaker.kind == "subject":
            subject_id = _identifier(
                speaker.subject_id, f"speaker_plan[{index}].subject_id"
            )
            if subject_id in subject_ids:
                raise TargetAudioError("speaker_plan contains duplicate subject_id")
            subject_ids.add(subject_id)
        else:
            if speaker.subject_id is not None:
                raise TargetAudioError("narrator must not bind a visual subject_id")
            subject_id = None
        strategy = VoiceStrategy(
            **_strategy_payload(
                speaker.voice_strategy, f"speaker_plan[{index}].voice_strategy"
            )
        )
        normalized = SpeakerPlan(speaker_id, language, strategy, speaker.kind, subject_id)
        speakers.append(normalized)
        speaker_by_id[speaker_id] = normalized

    references = tuple(request.audio_refs)
    if len(references) > 3:
        raise TargetAudioError("H3 accepts at most 3 ordered voice references")
    frozen_refs: list[FrozenAudioReference] = []
    reference_ids: set[str] = set()
    reference_speakers: set[str] = set()
    voice_reference_ids: set[str] = set()
    total_reference_duration = Fraction(0, 1)
    for index, reference in enumerate(references):
        if not isinstance(reference, AudioReference):
            raise TargetAudioError(f"audio_refs[{index}] is invalid")
        if reference.order != index:
            raise TargetAudioError("audio_refs order must match array order")
        reference_id = _identifier(
            reference.reference_id, f"audio_refs[{index}].reference_id"
        )
        if reference.purpose not in _REFERENCE_PURPOSES:
            raise TargetAudioError(f"audio_refs[{index}].purpose is invalid")
        if reference.purpose == "voice":
            speaker_id = _identifier(
                reference.speaker_id, f"audio_refs[{index}].speaker_id"
            )
            if speaker_id not in speaker_by_id:
                raise TargetAudioError("voice reference speaker is missing from speaker_plan")
            if speaker_id in reference_speakers:
                raise TargetAudioError("each speaker may have only one clean voice reference")
            reference_speakers.add(speaker_id)
            voice_reference_ids.add(reference_id)
            if reference.description is not None:
                raise TargetAudioError("voice reference must not set a sound description")
            description = None
        else:
            if reference.speaker_id is not None:
                raise TargetAudioError(
                    "ambience/sound_effect reference must not bind a speaker"
                )
            speaker_id = None
            description = _non_empty(
                reference.description, f"audio_refs[{index}].description"
            )
        if reference_id in reference_ids:
            raise TargetAudioError("audio_refs contains duplicate reference_id")
        audio = _freeze_audio(
            root, reference.path, label=f"audio_refs[{index}]", probe=probe
        )
        duration = Fraction(audio.probe.duration_pts, audio.probe.time_base)
        if duration < 2 or duration > 15:
            raise TargetAudioError("each H3 voice reference must be 2..15 seconds")
        total_reference_duration += duration
        reference_ids.add(reference_id)
        frozen_refs.append(
            FrozenAudioReference(
                reference_id,
                index,
                reference.purpose,
                speaker_id,
                description,
                audio,
            )
        )
    if total_reference_duration > 15:
        raise TargetAudioError("H3 voice reference total duration must be <=15 seconds")

    required_reference_ids = {
        speaker.voice_strategy.voice_reference_id
        for speaker in speakers
        if speaker.voice_strategy.kind == "voice_reference"
    }
    if required_reference_ids != voice_reference_ids:
        raise TargetAudioError(
            "speaker voice_reference strategies must exactly match frozen audio_refs"
        )
    ref_speaker_by_id = {
        item.reference_id: item.speaker_id
        for item in frozen_refs
        if item.purpose == "voice"
    }
    for speaker in speakers:
        reference_id = speaker.voice_strategy.voice_reference_id
        if reference_id is not None and ref_speaker_by_id.get(reference_id) != speaker.speaker_id:
            raise TargetAudioError("voice reference speaker mapping does not match")

    dialogue: list[FrozenDialogueLine] = []
    line_ids: set[str] = set()
    used_speakers: set[str] = set()
    previous_end: Fraction | None = None
    for index, line in enumerate(request.dialogue):
        if not isinstance(line, DialogueLine):
            raise TargetAudioError(f"dialogue[{index}] is invalid")
        if line.order != index:
            raise TargetAudioError("dialogue order must match array order")
        line_id = _identifier(line.line_id, f"dialogue[{index}].line_id")
        if line_id in line_ids:
            raise TargetAudioError("dialogue contains duplicate line_id")
        speaker_id = _identifier(line.speaker_id, f"dialogue[{index}].speaker_id")
        speaker = speaker_by_id.get(speaker_id)
        if speaker is None:
            raise TargetAudioError(f"dialogue[{index}] speaker is missing from speaker_plan")
        language = _non_empty(line.language, f"dialogue[{index}].language")
        text = _non_empty(line.text, f"dialogue[{index}].text")
        time_base = _integer(line.time_base, f"dialogue[{index}].time_base", minimum=1)
        start_pts = _integer(line.start_pts, f"dialogue[{index}].start_pts")
        end_pts = _integer(line.end_pts, f"dialogue[{index}].end_pts")
        if start_pts >= end_pts:
            raise TargetAudioError("dialogue PTS range must be non-empty and half-open")
        start = Fraction(start_pts, time_base)
        end = Fraction(end_pts, time_base)
        if previous_end is not None and start < previous_end:
            raise TargetAudioError("dialogue PTS ranges overlap")
        previous_end = end
        line_ids.add(line_id)
        used_speakers.add(speaker_id)
        dialogue.append(
            FrozenDialogueLine(
                line_id,
                index,
                speaker_id,
                language,
                time_base,
                start_pts,
                end_pts,
                text,
                speaker.voice_strategy,
            )
        )
    if used_speakers != set(speaker_by_id):
        raise TargetAudioError("speaker_plan must exactly map dialogue speakers")

    effects: list[AudioCue] = []
    cue_ids: set[str] = set()
    previous_by_role: dict[str, Fraction] = {}
    for index, cue in enumerate(request.effects):
        if not isinstance(cue, AudioCue):
            raise TargetAudioError(f"effects[{index}] is invalid")
        if cue.order != index:
            raise TargetAudioError("effects order must match array order")
        cue_id = _identifier(cue.cue_id, f"effects[{index}].cue_id")
        if cue_id in cue_ids:
            raise TargetAudioError("effects contains duplicate cue_id")
        if cue.role not in {"sound_effect", "ambience"}:
            raise TargetAudioError(f"effects[{index}].role is invalid")
        time_base = _integer(cue.time_base, f"effects[{index}].time_base", minimum=1)
        start_pts = _integer(cue.start_pts, f"effects[{index}].start_pts")
        end_pts = _integer(cue.end_pts, f"effects[{index}].end_pts")
        if start_pts >= end_pts:
            raise TargetAudioError("effect PTS range must be non-empty and half-open")
        start = Fraction(start_pts, time_base)
        end = Fraction(end_pts, time_base)
        if start < previous_by_role.get(cue.role, Fraction(0, 1)):
            raise TargetAudioError(f"{cue.role} PTS ranges overlap")
        previous_by_role[cue.role] = end
        prompt_text = _non_empty(cue.prompt, f"effects[{index}].prompt")
        cue_ids.add(cue_id)
        effects.append(
            AudioCue(cue_id, index, cue.role, time_base, start_pts, end_pts, prompt_text)
        )

    material_specs = tuple(request.target_materials)
    if not 1 <= len(material_specs) <= 3:
        raise TargetAudioError("target_audio must contain 1..3 material specs")
    normalized_specs: list[TargetMaterialSpec] = []
    material_ids: set[str] = set()
    material_roles: set[str] = set()
    for index, item in enumerate(material_specs):
        if not isinstance(item, TargetMaterialSpec):
            raise TargetAudioError(f"target_materials[{index}] is invalid")
        if item.order != index:
            raise TargetAudioError("target material order must match array order")
        material_id = _identifier(
            item.material_id, f"target_materials[{index}].material_id"
        )
        if material_id in material_ids:
            raise TargetAudioError("target materials contain duplicate material_id")
        if item.role not in _MATERIAL_ROLES:
            raise TargetAudioError(f"target_materials[{index}].role is invalid")
        if item.role in material_roles:
            raise TargetAudioError("target materials contain duplicate role")
        audio_format = str(item.format).lower()
        if audio_format not in _AUDIO_FORMATS:
            raise TargetAudioError("target material format must be MP3 or WAV")
        time_base = _integer(
            item.time_base, f"target_materials[{index}].time_base", minimum=1
        )
        duration_pts = _integer(
            item.duration_pts, f"target_materials[{index}].duration_pts", minimum=1
        )
        material_ids.add(material_id)
        material_roles.add(item.role)
        normalized_specs.append(
            TargetMaterialSpec(
                material_id, index, item.role, audio_format, time_base, duration_pts
            )
        )

    spec_by_role = {item.role: item for item in normalized_specs}
    if dialogue and "dialogue" not in spec_by_role:
        raise TargetAudioError("dialogue has no target dialogue material")
    for line in dialogue:
        spec = spec_by_role["dialogue"]
        if Fraction(line.end_pts, line.time_base) > Fraction(
            spec.duration_pts, spec.time_base
        ):
            raise TargetAudioError("dialogue PTS exceeds target material duration")
    for cue in effects:
        spec = spec_by_role.get(cue.role)
        if spec is None:
            raise TargetAudioError(f"{cue.role} cue has no matching target material")
        if Fraction(cue.end_pts, cue.time_base) > Fraction(
            spec.duration_pts, spec.time_base
        ):
            raise TargetAudioError(f"{cue.role} PTS exceeds target material duration")

    normalized_request = TargetAudioRequest(
        client_request_id=client_request_id,
        mode=request.mode,
        source_audio=source_audio.path if source_audio is not None else None,
        dialogue=tuple(
            DialogueLine(
                item.line_id,
                item.order,
                item.speaker_id,
                item.language,
                item.time_base,
                item.start_pts,
                item.end_pts,
                item.text,
            )
            for item in dialogue
        ),
        speaker_plan=tuple(speakers),
        audio_refs=tuple(
            AudioReference(
                item.reference_id,
                item.order,
                item.speaker_id,
                item.path,
                item.purpose,
                item.description,
            )
            for item in frozen_refs
        ),
        effects=tuple(effects),
        target_materials=tuple(normalized_specs),
        speaker_mapping_verified=True,
        speaker_mapping_source=mapping_source,
        materializer=materializer,
    )

    speaker_payload = [
        {
            "order": index,
            "speaker_id": item.speaker_id,
            "language": item.language,
            "kind": item.kind,
            "subject_id": item.subject_id,
            "voice_strategy": _strategy_payload(
                item.voice_strategy, f"speaker_plan[{index}].voice_strategy"
            ),
        }
        for index, item in enumerate(speakers)
    ]
    line_payload = [
        {
            "line_id": item.line_id,
            "order": item.order,
            "speaker_id": item.speaker_id,
            "language": item.language,
            "time_base": item.time_base,
            "start_pts": item.start_pts,
            "end_pts": item.end_pts,
            "range": "half_open",
            "text": item.text,
            "voice_strategy": _strategy_payload(
                item.voice_strategy, f"dialogue[{index}].voice_strategy"
            ),
        }
        for index, item in enumerate(dialogue)
    ]
    effect_payload = [
        {
            "cue_id": item.cue_id,
            "order": item.order,
            "role": item.role,
            "time_base": item.time_base,
            "start_pts": item.start_pts,
            "end_pts": item.end_pts,
            "range": "half_open",
            "prompt": item.prompt,
        }
        for item in effects
    ]
    material_payload = [
        {
            "material_id": item.material_id,
            "order": item.order,
            "role": item.role,
            "format": item.format,
            "time_base": item.time_base,
            "duration_pts": item.duration_pts,
        }
        for item in normalized_specs
    ]
    core = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "client_request_id": client_request_id,
        "mode": request.mode,
        "inputs": {
            "source_audio": None if source_audio is None else _audio_binding(source_audio),
            "audio_refs": [
                {
                    "reference_id": item.reference_id,
                    "order": item.order,
                    "purpose": item.purpose,
                    "speaker_id": item.speaker_id,
                    "description": item.description,
                    **_audio_binding(item.audio),
                }
                for item in frozen_refs
            ],
        },
        "speaker_map": {
            "verified": True,
            "source": mapping_source,
            "speakers": speaker_payload,
            "sha256": _sha256(_canonical_json(speaker_payload)),
        },
        "script": {
            "range": "half_open_pts",
            "lines": line_payload,
            "sha256": _sha256(_canonical_json(line_payload)),
        },
        "effects": {
            "cues": effect_payload,
            "sha256": _sha256(_canonical_json(effect_payload)),
        },
        "target_material_specs": {
            "materials": material_payload,
            "sha256": _sha256(_canonical_json(material_payload)),
        },
        "materializer": _materializer_payload(materializer),
    }
    plan_receipt = _sha256(_canonical_json(core))
    return _NormalizedRequest(
        request=normalized_request,
        source_audio=source_audio,
        dialogue=tuple(dialogue),
        audio_refs=tuple(frozen_refs),
        core=core,
        plan_receipt=plan_receipt,
    )


def _receipt_path(
    root: Path, receipt_path: Path | None, *, create_parent: bool = True
) -> Path:
    candidate = Path(receipt_path) if receipt_path is not None else root / RECEIPT_FILENAME
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise TargetAudioError("receipt path escapes target audio root") from None
    if not relative.parts:
        raise TargetAudioError("receipt path must name a file")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise TargetAudioError("receipt path must not contain a symlink")
        if not current.exists():
            if not create_parent:
                raise TargetAudioError("target audio receipt is missing")
            current.mkdir(mode=0o700)
        if not current.is_dir():
            raise TargetAudioError("receipt parent must be a directory")
    if candidate.is_symlink():
        raise TargetAudioError("receipt path must not be a symlink")
    return candidate


def _atomic_write_json(path: Path, payload: dict) -> None:
    data = _canonical_json(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise TargetAudioError(f"target audio receipt write failed: {exc}") from None
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _receipt_lease(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    if lock_path.is_symlink():
        raise TargetAudioError("target audio receipt lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise TargetAudioError(f"target audio receipt lock failed: {exc}") from None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_receipt(root: Path, path: Path) -> dict:
    _, _, data = _read_regular_file(root, path, "target audio receipt")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TargetAudioError("target audio receipt is invalid JSON") from None
    if not isinstance(payload, dict):
        raise TargetAudioError("target audio receipt is invalid")
    return payload


def _request_from_core(root: Path, core: dict) -> TargetAudioRequest:
    try:
        if set(core) != {
            "schema",
            "version",
            "client_request_id",
            "mode",
            "inputs",
            "speaker_map",
            "script",
            "effects",
            "target_material_specs",
            "materializer",
        }:
            raise ValueError
        inputs = core["inputs"]
        speaker_map = core["speaker_map"]
        script = core["script"]
        effects = core["effects"]
        targets = core["target_material_specs"]
        materializer = core["materializer"]
        if not isinstance(inputs, dict) or set(inputs) != {"source_audio", "audio_refs"}:
            raise ValueError
        if not isinstance(speaker_map, dict) or set(speaker_map) != {
            "verified", "source", "speakers", "sha256"
        }:
            raise ValueError
        if not isinstance(script, dict) or set(script) != {"range", "lines", "sha256"}:
            raise ValueError
        if not isinstance(effects, dict) or set(effects) != {"cues", "sha256"}:
            raise ValueError
        if not isinstance(targets, dict) or set(targets) != {"materials", "sha256"}:
            raise ValueError
        if not isinstance(materializer, dict) or set(materializer) != {
            "provider", "model", "version", "prompt", "prompt_sha256", "cost"
        }:
            raise ValueError
        if _sha256(_canonical_json(speaker_map["speakers"])) != speaker_map["sha256"]:
            raise ValueError
        if _sha256(_canonical_json(script["lines"])) != script["sha256"]:
            raise ValueError
        if _sha256(_canonical_json(effects["cues"])) != effects["sha256"]:
            raise ValueError
        if _sha256(_canonical_json(targets["materials"])) != targets["sha256"]:
            raise ValueError
        if _sha256(materializer["prompt"].encode("utf-8")) != materializer["prompt_sha256"]:
            raise ValueError
        source_binding = inputs["source_audio"]
        source_path = None if source_binding is None else root / source_binding["path"]
        speakers = []
        for index, item in enumerate(speaker_map["speakers"]):
            if not isinstance(item, dict) or set(item) != {
                "order", "speaker_id", "language", "kind", "subject_id",
                "voice_strategy"
            } or item["order"] != index:
                raise ValueError
            speakers.append(
                SpeakerPlan(
                    item["speaker_id"],
                    item["language"],
                    _strategy_from_payload(
                        item["voice_strategy"], f"speaker_plan[{index}].voice_strategy"
                    ),
                    item["kind"],
                    item["subject_id"],
                )
            )
        lines = []
        for index, item in enumerate(script["lines"]):
            if not isinstance(item, dict) or set(item) != {
                "line_id", "order", "speaker_id", "language", "time_base",
                "start_pts", "end_pts", "range", "text", "voice_strategy",
            } or item["range"] != "half_open":
                raise ValueError
            lines.append(
                DialogueLine(
                    item["line_id"], item["order"], item["speaker_id"],
                    item["language"], item["time_base"], item["start_pts"],
                    item["end_pts"], item["text"],
                )
            )
        cues = []
        for item in effects["cues"]:
            if not isinstance(item, dict) or set(item) != {
                "cue_id", "order", "role", "time_base", "start_pts", "end_pts",
                "range", "prompt",
            } or item["range"] != "half_open":
                raise ValueError
            cues.append(
                AudioCue(
                    item["cue_id"], item["order"], item["role"], item["time_base"],
                    item["start_pts"], item["end_pts"], item["prompt"],
                )
            )
        specs = []
        for item in targets["materials"]:
            if not isinstance(item, dict) or set(item) != {
                "material_id", "order", "role", "format", "time_base", "duration_pts"
            }:
                raise ValueError
            specs.append(TargetMaterialSpec(**item))
        refs = []
        for item in inputs["audio_refs"]:
            if not isinstance(item, dict) or set(item) != {
                "reference_id", "order", "purpose", "speaker_id", "description",
                "path", "format", "size_bytes", "sha256", "probe",
            }:
                raise ValueError
            refs.append(
                AudioReference(
                    item["reference_id"], item["order"], item["speaker_id"],
                    root / item["path"], item["purpose"], item["description"],
                )
            )
        cost = materializer["cost"]
        if not isinstance(cost, dict) or set(cost) != {"currency", "amount_micros"}:
            raise ValueError
        return TargetAudioRequest(
            client_request_id=core["client_request_id"],
            mode=core["mode"],
            source_audio=source_path,
            dialogue=tuple(lines),
            speaker_plan=tuple(speakers),
            audio_refs=tuple(refs),
            effects=tuple(cues),
            target_materials=tuple(specs),
            speaker_mapping_verified=speaker_map["verified"],
            speaker_mapping_source=speaker_map["source"],
            materializer=MaterializerSpec(
                provider=materializer["provider"],
                model=materializer["model"],
                version=materializer["version"],
                prompt=materializer["prompt"],
                cost=CostEstimate(**cost),
            ),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise TargetAudioError("target audio receipt shape is invalid") from None


def _validate_receipt(root: Path, path: Path, probe: AudioProbeFn) -> tuple[dict, _NormalizedRequest]:
    receipt = _read_receipt(root, path)
    if set(receipt) != set(_empty_receipt_keys()):
        raise TargetAudioError("target audio receipt shape is invalid")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("version") != RECEIPT_VERSION:
        raise TargetAudioError("target audio receipt schema/version is incompatible")
    materialization = receipt.get("materialization")
    if not isinstance(materialization, dict) or set(materialization) != {
        "status", "task_id", "outputs", "error"
    }:
        raise TargetAudioError("target audio materialization receipt is invalid")
    if materialization["status"] not in _STORED_STATUSES:
        raise TargetAudioError("target audio materialization status is invalid")
    task_id = materialization["task_id"]
    if task_id is not None:
        _identifier(task_id, "materialization.task_id")
    if not isinstance(materialization["outputs"], list):
        raise TargetAudioError("target audio materialization outputs are invalid")
    if materialization["error"] is not None and not isinstance(materialization["error"], str):
        raise TargetAudioError("target audio materialization error is invalid")
    core = {key: receipt[key] for key in _empty_receipt_keys() if key not in {
        "plan_receipt", "materialization"
    }}
    expected = _sha256(_canonical_json(core))
    if receipt.get("plan_receipt") != expected:
        raise TargetAudioError("target audio plan receipt mismatch")
    request = _request_from_core(root, core)
    normalized = _normalize_request(root, request, probe)
    if normalized.core != core or normalized.plan_receipt != expected:
        raise TargetAudioError("target audio frozen input sha256/probe binding mismatch")
    return receipt, normalized


def _empty_receipt_keys() -> tuple[str, ...]:
    return (
        "schema", "version", "client_request_id", "mode", "inputs", "speaker_map",
        "script", "effects", "target_material_specs", "materializer", "plan_receipt",
        "materialization",
    )


def _plan_from_receipt(
    root: Path,
    path: Path,
    receipt: dict,
    normalized: _NormalizedRequest,
    probe: AudioProbeFn,
) -> TargetAudioPlan:
    state = receipt["materialization"]
    materials: tuple[TargetAudioMaterial, ...] = ()
    if state["status"] == "succeeded":
        materials, canonical_outputs = _load_material_entries(
            root, normalized.request.target_materials, state["outputs"], probe
        )
        if canonical_outputs != state["outputs"]:
            raise TargetAudioError("target audio material receipt mismatch")
    return TargetAudioPlan(
        receipt_path=path,
        plan_receipt=normalized.plan_receipt,
        client_request_id=normalized.request.client_request_id,
        mode=normalized.request.mode,
        source_audio=normalized.source_audio,
        dialogue=normalized.dialogue,
        speaker_plan=normalized.request.speaker_plan,
        audio_refs=normalized.audio_refs,
        effects=normalized.request.effects,
        target_material_specs=normalized.request.target_materials,
        target_materials=materials,
        speaker_mapping_verified=True,
        speaker_mapping_source=normalized.request.speaker_mapping_source,
        materializer=normalized.request.materializer,
        materialization_status=state["status"],
        task_id=state["task_id"],
    )


def _initial_receipt(normalized: _NormalizedRequest) -> dict:
    return {
        **normalized.core,
        "plan_receipt": normalized.plan_receipt,
        "materialization": {
            "status": "ready_to_submit",
            "task_id": None,
            "outputs": [],
            "error": None,
        },
    }


def _freeze_unlocked(
    root: Path,
    path: Path,
    request: TargetAudioRequest,
    probe: AudioProbeFn,
) -> TargetAudioPlan:
    normalized = _normalize_request(root, request, probe)
    if path.exists():
        receipt, existing = _validate_receipt(root, path, probe)
        if existing.plan_receipt != normalized.plan_receipt:
            raise TargetAudioError("existing target audio receipt does not match request")
        return _plan_from_receipt(root, path, receipt, existing, probe)
    receipt = _initial_receipt(normalized)
    _atomic_write_json(path, receipt)
    return _plan_from_receipt(root, path, receipt, normalized, probe)


def freeze_target_audio_plan(
    root: Path,
    request: TargetAudioRequest,
    *,
    probe: AudioProbeFn,
    receipt_path: Path | None = None,
) -> TargetAudioPlan:
    """Freeze route-neutral semantics and input bytes without calling a provider."""

    root = _resolved_root(root)
    path = _receipt_path(root, receipt_path)
    with _receipt_lease(path):
        return _freeze_unlocked(root, path, request, probe)


def load_target_audio_plan(
    root: Path,
    receipt_path: Path | None = None,
    *,
    probe: AudioProbeFn,
) -> TargetAudioPlan:
    """Reload every bound byte and probe fact; any drift fails closed."""

    root = _resolved_root(root)
    path = _receipt_path(root, receipt_path, create_parent=False)
    receipt, normalized = _validate_receipt(root, path, probe)
    return _plan_from_receipt(root, path, receipt, normalized, probe)


def _output_payload(
    output: MaterializedOutput,
    root: Path,
    spec: TargetMaterialSpec,
    index: int,
) -> dict:
    if not isinstance(output, MaterializedOutput):
        raise TargetAudioError("target material output is invalid")
    path = Path(output.path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = Path(os.path.abspath(path)).relative_to(root).as_posix()
    except ValueError:
        raise TargetAudioError("target material output escapes target audio root") from None
    return {
        "material_id": output.material_id,
        "order": index,
        "role": spec.role,
        "path": relative,
        "format": output.format,
        "time_base": output.time_base,
        "duration_pts": output.duration_pts,
        "size_bytes": output.size_bytes,
        "sha256": output.sha256,
        "decoded_sha256": output.decoded_sha256,
    }


def _validate_result(
    result: object, *, expected_task_id: str | None
) -> MaterializationResult:
    if not isinstance(result, MaterializationResult):
        raise TargetAudioError("materializer returned an invalid result")
    if result.status not in _MATERIALIZATION_RESULTS:
        raise TargetAudioError("materializer returned an invalid status")
    task_id = result.task_id
    if task_id is not None:
        task_id = _identifier(task_id, "materializer task_id")
    if expected_task_id is not None and task_id != expected_task_id:
        raise TargetAudioError("materializer task_id changed")
    if result.status != "succeeded" and result.outputs:
        raise TargetAudioError("unmaterialized result must not contain outputs")
    if result.status in {"processing", "failed"} and task_id is None:
        raise TargetAudioError("materializer result has no recoverable task_id")
    return MaterializationResult(result.status, task_id, tuple(result.outputs))


def _load_material_entries(
    root: Path,
    specs: Sequence[TargetMaterialSpec],
    outputs: object,
    probe: AudioProbeFn,
) -> tuple[tuple[TargetAudioMaterial, ...], list[dict]]:
    if not isinstance(outputs, list) or len(outputs) != len(specs):
        raise TargetAudioError("target material count does not match the frozen plan")
    materials: list[TargetAudioMaterial] = []
    canonical: list[dict] = []
    for index, (spec, claim) in enumerate(zip(specs, outputs)):
        base_keys = {
            "material_id", "order", "role", "path", "format", "time_base",
            "duration_pts", "size_bytes", "sha256", "decoded_sha256",
        }
        if not isinstance(claim, dict) or frozenset(claim) not in {
            frozenset(base_keys), frozenset(base_keys | {"probe"})
        }:
            raise TargetAudioError(f"target material[{index}] receipt is invalid")
        if claim["material_id"] != spec.material_id:
            raise TargetAudioError("target material order/id mismatch")
        if claim["order"] != index or claim["role"] != spec.role:
            raise TargetAudioError("target material order/role mismatch")
        if claim["format"] != spec.format:
            raise TargetAudioError("target material format mismatch")
        if claim["time_base"] != spec.time_base or claim["duration_pts"] != spec.duration_pts:
            raise TargetAudioError("target material duration mismatch")
        if not isinstance(claim["path"], str) or Path(claim["path"]).is_absolute():
            raise TargetAudioError("target material path must be project-relative")
        audio = _freeze_audio(
            root, root / claim["path"], label=f"target material[{index}]", probe=probe
        )
        if audio.format != spec.format:
            raise TargetAudioError("target material normalized format mismatch")
        if audio.probe.time_base != spec.time_base or audio.probe.duration_pts != spec.duration_pts:
            raise TargetAudioError("target material probed duration mismatch")
        if claim["size_bytes"] != audio.size_bytes:
            raise TargetAudioError("target material byte size mismatch")
        if claim["sha256"] != audio.sha256:
            raise TargetAudioError("target material sha256 mismatch")
        if claim["decoded_sha256"] != audio.decoded_sha256:
            raise TargetAudioError("target material decoded sha256 mismatch")
        if "probe" in claim and claim["probe"] != _probe_payload(audio.probe):
            raise TargetAudioError("target material probe metadata mismatch")
        material = TargetAudioMaterial(spec.material_id, index, spec.role, audio)
        materials.append(material)
        canonical.append(
            {
                "material_id": spec.material_id,
                "order": index,
                "role": spec.role,
                "path": audio.relative_path,
                "format": spec.format,
                "time_base": spec.time_base,
                "duration_pts": spec.duration_pts,
                "size_bytes": audio.size_bytes,
                "sha256": audio.sha256,
                "decoded_sha256": audio.decoded_sha256,
                "probe": _probe_payload(audio.probe),
            }
        )
    return tuple(materials), canonical


def _validate_materialized_outputs(
    root: Path,
    specs: Sequence[TargetMaterialSpec],
    outputs: Sequence[MaterializedOutput],
    probe: AudioProbeFn,
) -> tuple[tuple[TargetAudioMaterial, ...], list[dict]]:
    if len(outputs) != len(specs):
        raise TargetAudioError("target material count does not match the frozen plan")
    claims = [
        _output_payload(output, root, spec, index)
        for index, (spec, output) in enumerate(zip(specs, outputs))
    ]
    return _load_material_entries(root, specs, claims, probe)


def _persist_materialization(path: Path, receipt: dict, **changes: object) -> None:
    state = dict(receipt["materialization"])
    state.update(changes)
    receipt["materialization"] = state
    _atomic_write_json(path, receipt)


def materialize_target_audio_plan(
    root: Path,
    request: TargetAudioRequest,
    *,
    materializer: Materializer,
    probe: AudioProbeFn,
    receipt_path: Path | None = None,
) -> TargetAudioPlan:
    """Materialize route B once; unknown POSTs never become another submit.

    A persisted task id always selects ``materializer.get``.  A persisted
    ``submission_unknown`` without a task id is terminal and requires external
    reconciliation; this function never guesses by issuing another paid POST.
    """

    root = _resolved_root(root)
    path = _receipt_path(root, receipt_path)
    with _receipt_lease(path):
        plan = _freeze_unlocked(root, path, request, probe)
        receipt, normalized = _validate_receipt(root, path, probe)
        state = receipt["materialization"]
        status = state["status"]
        task_id = state["task_id"]
        frozen_request = plan.materialization_request()

        if status == "succeeded":
            return _plan_from_receipt(root, path, receipt, normalized, probe)
        if status == "submission_unknown" and task_id is None:
            raise TargetAudioError("submission_unknown: refusing another materializer POST")
        if status == "submitting" and task_id is None:
            _persist_materialization(
                path, receipt, status="submission_unknown", error="submission_unknown"
            )
            raise TargetAudioError("submission_unknown: refusing another materializer POST")
        if status == "validating" and task_id is None:
            try:
                _materials, outputs = _load_material_entries(
                    root,
                    normalized.request.target_materials,
                    state["outputs"],
                    probe,
                )
            except TargetAudioError as exc:
                _persist_materialization(
                    path, receipt, status="output_invalid", error=str(exc)
                )
                raise
            _persist_materialization(
                path, receipt, status="succeeded", outputs=outputs, error=None
            )
            return load_target_audio_plan(root, path, probe=probe)
        if task_id is None and status in {"failed", "output_invalid"}:
            raise TargetAudioError("target_audio_not_materialized")

        submitting = task_id is None and status == "ready_to_submit"
        if submitting:
            _persist_materialization(
                path,
                receipt,
                status="submitting",
                task_id=None,
                outputs=[],
                error=None,
            )
        try:
            raw_result = (
                materializer.submit(frozen_request)
                if submitting
                else materializer.get(task_id, frozen_request)
            )
        except Exception as exc:
            if submitting:
                _persist_materialization(
                    path,
                    receipt,
                    status="submission_unknown",
                    task_id=None,
                    outputs=[],
                    error="submission_unknown",
                )
                raise TargetAudioError(
                    "submission_unknown: refusing another materializer POST"
                ) from None
            _persist_materialization(
                path,
                receipt,
                status="processing",
                task_id=task_id,
                error=f"materializer_get_failed:{type(exc).__name__}",
            )
            raise TargetAudioError("target_audio_not_materialized") from None

        returned_task_id: str | None = None
        if isinstance(raw_result, MaterializationResult) and raw_result.task_id is not None:
            try:
                returned_task_id = _identifier(
                    raw_result.task_id, "materializer task_id"
                )
            except TargetAudioError:
                returned_task_id = None
        if submitting and returned_task_id is not None:
            # Persist the query handle before inspecting any other provider field.
            # From this point onward even a malformed response is GET-only.
            task_id = returned_task_id
            _persist_materialization(
                path,
                receipt,
                status="processing",
                task_id=task_id,
                outputs=[],
                error=None,
            )

        try:
            result = _validate_result(raw_result, expected_task_id=task_id)
        except TargetAudioError as exc:
            if task_id is None:
                _persist_materialization(
                    path,
                    receipt,
                    status="submission_unknown",
                    task_id=None,
                    outputs=[],
                    error="submission_unknown",
                )
                raise TargetAudioError(
                    "submission_unknown: materializer response was not safely receipted"
                ) from None
            _persist_materialization(
                path, receipt, status="failed", task_id=task_id, error=str(exc)
            )
            raise

        if result.status == "submission_unknown" and result.task_id is None:
            _persist_materialization(
                path,
                receipt,
                status="submission_unknown",
                task_id=None,
                outputs=[],
                error="submission_unknown",
            )
            raise TargetAudioError("submission_unknown: refusing another materializer POST")
        if result.status != "succeeded":
            _persist_materialization(
                path,
                receipt,
                status=result.status,
                task_id=result.task_id,
                outputs=[],
                error=None if result.status == "processing" else result.status,
            )
            raise TargetAudioError("target_audio_not_materialized")

        try:
            if len(result.outputs) != len(normalized.request.target_materials):
                raise TargetAudioError(
                    "target material count does not match the frozen plan"
                )
            claims = [
                _output_payload(output, root, spec, index)
                for index, (spec, output) in enumerate(
                    zip(normalized.request.target_materials, result.outputs)
                )
            ]
        except TargetAudioError as exc:
            _persist_materialization(
                path,
                receipt,
                status="output_invalid",
                task_id=result.task_id,
                outputs=[],
                error=str(exc),
            )
            raise
        _persist_materialization(
            path,
            receipt,
            status="validating",
            task_id=result.task_id,
            outputs=claims,
            error=None,
        )
        try:
            _materials, outputs = _validate_materialized_outputs(
                root, normalized.request.target_materials, result.outputs, probe
            )
        except TargetAudioError as exc:
            _persist_materialization(
                path,
                receipt,
                status="output_invalid",
                task_id=result.task_id,
                outputs=claims,
                error=str(exc),
            )
            raise
        _persist_materialization(
            path,
            receipt,
            status="succeeded",
            task_id=result.task_id,
            outputs=outputs,
            error=None,
        )
        return load_target_audio_plan(root, path, probe=probe)


__all__ = [
    "AudioCue",
    "AudioProbe",
    "AudioReference",
    "CostEstimate",
    "DialogueLine",
    "FrozenAudioArtifact",
    "FrozenAudioReference",
    "FrozenDialogueLine",
    "MaterializationRequest",
    "MaterializationResult",
    "MaterializedOutput",
    "Materializer",
    "MaterializerSpec",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "RECEIPT_VERSION",
    "SpeakerPlan",
    "TargetAudioError",
    "TargetAudioMaterial",
    "TargetAudioPlan",
    "TargetAudioRequest",
    "TargetMaterialSpec",
    "VoiceStrategy",
    "freeze_target_audio_plan",
    "load_target_audio_plan",
    "materialize_target_audio_plan",
]
