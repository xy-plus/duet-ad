"""Fail-closed orchestration for paid long-video H3 segment generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from app import (
    context_ir_bridge,
    dialogue_delivery as dialogue_delivery_contract,
    frame_fit,
    h3,
    h3_project,
    long_video,
    postprocess,
    prepared_input,
    stitch,
    storage,
)
from app.config import Settings

WORKFLOW = h3.H3_WORKFLOW
_PLAN_WORKFLOWS = h3.H3_REFERENCE_WORKFLOWS | {h3.H3_BOUNDARY_WORKFLOW}
_PIPELINE_NO_BGM = "不要生成背景音乐"
_EPS = 1e-6
FIT_LAYOUT_LEGACY = "legacy-v0"
FIT_LAYOUT_ASPECT = "aspect-v1"
_FIT_LAYOUTS = frozenset({FIT_LAYOUT_LEGACY, FIT_LAYOUT_ASPECT})
_FAST_MODE_WORKERS = 8
AUDIO_ROUTE_SCHEMA = "duet.long-generation.audio-route"
AUDIO_ROUTE_VERSION = 1
H3_NATIVE_AUDIO_ROUTE = {
    "schema": AUDIO_ROUTE_SCHEMA,
    "version": AUDIO_ROUTE_VERSION,
    "mode": "h3_native",
}
PROMPT_FUSION_INPUT_SCHEMA = "duet.video-prompt-fusion-input"
PROMPT_FUSION_OUTPUT_SCHEMA = "duet.video-prompt-fusion-output"
PROMPT_FUSION_VERSION = 1
PROMPT_FUSION_MANIFEST_SCHEMA = "duet.video-prompt-fusion-production"
PROMPT_FUSION_MANIFEST_VERSION = 1
PROMPT_FUSION_SKILL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "skills" / "video-prompt-fusion" / "SKILL.md"
)


class LongGenerationError(RuntimeError):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


@dataclass(frozen=True)
class FrozenPromptFusion:
    input_path: Path
    input_data: bytes
    input_sha256: str
    output_path: Path
    output_data: bytes
    output_sha256: str
    segments: tuple[Mapping, ...]
    final_prompts: tuple[str, ...]


def _fusion_json(path: Path, code: str) -> tuple[bytes, dict]:
    if path.is_symlink():
        raise LongGenerationError(code)
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise LongGenerationError(code) from None
    if not data or not isinstance(value, dict):
        raise LongGenerationError(code)
    return data, value


def _fusion_text_binding(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"text", "sha256"}
        and isinstance(value.get("text"), str)
        and value["text"].strip()
        and value.get("sha256")
        == hashlib.sha256(value["text"].encode("utf-8")).hexdigest()
    )


def _fusion_audio_block(lines_json: str) -> str:
    return (
        f"<AUDIO_CONTENT_JSON>{lines_json}</AUDIO_CONTENT_JSON>"
    )


def _canonical_fusion_prompt(prompt: str, lines_json: str) -> str:
    """Accept one exact audio block and freeze one canonical byte form."""
    opening = "<AUDIO_CONTENT_JSON>"
    closing = "</AUDIO_CONTENT_JSON>"
    if prompt.count(opening) != 1 or prompt.count(closing) != 1:
        raise LongGenerationError("prompt_fusion_output_invalid")
    canonical = _fusion_audio_block(lines_json)
    if canonical in prompt:
        return prompt
    lf_envelope = f"{opening}\n{lines_json}\n{closing}"
    if lf_envelope in prompt:
        return prompt.replace(lf_envelope, canonical, 1)
    raise LongGenerationError("prompt_fusion_output_invalid")


def load_prompt_fusion(
    *, input_path: Path, output_path: Path, root: Path | None = None,
) -> FrozenPromptFusion:
    """Load one project-level fusion result and verify all ordered inputs."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_data, source = _fusion_json(input_path, "prompt_fusion_input_invalid")
    output_data, output = _fusion_json(output_path, "prompt_fusion_output_invalid")
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    project_root = (
        Path(root).resolve()
        if root is not None else input_path.parent.parent.resolve()
    )
    try:
        input_path.relative_to(project_root)
        output_path.relative_to(project_root)
    except ValueError:
        raise LongGenerationError("prompt_fusion_input_invalid") from None
    segments = source.get("segments")
    if (
        input_path.name != h3_project.SKILL_INPUT_FILENAME
        or output_path.name != "h3_prompt_plan.json"
        or set(source) != {"schema", "version", "segments"}
        or source.get("schema") != PROMPT_FUSION_INPUT_SCHEMA
        or source.get("version") != PROMPT_FUSION_VERSION
        or not isinstance(segments, list)
        or not segments
    ):
        raise LongGenerationError("prompt_fusion_input_invalid")
    for index, segment in enumerate(segments, 1):
        if (
            not isinstance(segment, Mapping)
            or set(segment) != {
                "index", "new_keyframes", "old_video_prompt",
                "image_optimization_prompt", "audio_content",
            }
            or segment.get("index") != index
            or not _fusion_text_binding(segment.get("old_video_prompt"))
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        frames = segment.get("new_keyframes")
        prompts = segment.get("image_optimization_prompt")
        if (
            not isinstance(frames, list)
            or len(frames) != 9
            or not isinstance(prompts, list)
            or len(prompts) != 9
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        for order, (frame, prompt) in enumerate(zip(frames, prompts), 1):
            if (
                not isinstance(frame, Mapping)
                or set(frame) != {"order", "path", "sha256"}
                or frame.get("order") != order
                or not isinstance(frame.get("path"), str)
                or not frame["path"]
                or not isinstance(frame.get("sha256"), str)
                or len(frame["sha256"]) != 64
                or not isinstance(prompt, Mapping)
                or set(prompt) != {"order", "text", "sha256"}
                or prompt.get("order") != order
                or not _fusion_text_binding({
                    "text": prompt.get("text"), "sha256": prompt.get("sha256")
                })
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            frame_candidate = project_root / frame["path"]
            if frame_candidate.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            frame_path = frame_candidate.resolve()
            try:
                frame_path.relative_to(project_root)
                frame_data = frame_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if (
                not frame_data
                or hashlib.sha256(frame_data).hexdigest() != frame["sha256"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
        audio = segment.get("audio_content")
        if (
            not isinstance(audio, Mapping)
            or set(audio) != {
                "lines_json", "lines_sha256", "voice_references",
            }
            or not isinstance(audio.get("lines_json"), str)
            or not isinstance(audio.get("voice_references"), list)
            or audio.get("lines_sha256")
            != hashlib.sha256(audio["lines_json"].encode("utf-8")).hexdigest()
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        try:
            lines = json.loads(audio["lines_json"])
        except json.JSONDecodeError:
            raise LongGenerationError("prompt_fusion_input_invalid") from None
        if not isinstance(lines, list):
            raise LongGenerationError("prompt_fusion_input_invalid")
        used_voice_refs: set[int] = set()
        for order, line in enumerate(lines, 1):
            if (
                not isinstance(line, Mapping)
                or set(line) != {
                    "order", "text", "start_s", "end_s", "delivery", "voice_ref",
                }
                or line.get("order") != order
                or not isinstance(line.get("text"), str)
                or not line["text"].strip()
                or line.get("delivery") not in {"on_screen", "off_screen"}
                or line.get("voice_ref") not in {None, 1}
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            if line["voice_ref"] is not None:
                used_voice_refs.add(line["voice_ref"])
        references = audio["voice_references"]
        if len(references) != len(used_voice_refs):
            raise LongGenerationError("prompt_fusion_input_invalid")
        for voice_ref, reference in enumerate(references, 1):
            if (
                not isinstance(reference, Mapping)
                or set(reference) != {"voice_ref", "path", "sha256", "purpose"}
                or reference.get("voice_ref") != voice_ref
                or reference.get("purpose") != "voice"
                or voice_ref not in used_voice_refs
                or not isinstance(reference.get("path"), str)
                or not reference["path"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            reference_candidate = project_root / reference["path"]
            if reference_candidate.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            reference_path = reference_candidate.resolve()
            try:
                reference_path.relative_to(project_root)
                reference_data = reference_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if (
                not reference_data
                or hashlib.sha256(reference_data).hexdigest()
                != reference.get("sha256")
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
    frozen_input_sha256 = hashlib.sha256(input_data).hexdigest()
    output_segments = output.get("segments")
    if (
        set(output) != {"schema", "version", "input_sha256", "segments"}
        or output.get("schema") != PROMPT_FUSION_OUTPUT_SCHEMA
        or output.get("version") != PROMPT_FUSION_VERSION
        or output.get("input_sha256") != frozen_input_sha256
        or not isinstance(output_segments, list)
        or len(output_segments) != len(segments)
    ):
        raise LongGenerationError("prompt_fusion_output_invalid")
    final_prompts: list[str] = []
    for index, segment in enumerate(output_segments, 1):
        if (
            not isinstance(segment, Mapping)
            or set(segment) != {"index", "final_prompt"}
            or segment.get("index") != index
            or not isinstance(segment.get("final_prompt"), str)
            or not segment["final_prompt"].strip()
        ):
            raise LongGenerationError("prompt_fusion_output_invalid")
        final_prompts.append(_canonical_fusion_prompt(
            segment["final_prompt"],
            segments[index - 1]["audio_content"]["lines_json"],
        ))
    return FrozenPromptFusion(
        input_path=input_path,
        input_data=input_data,
        input_sha256=frozen_input_sha256,
        output_path=output_path,
        output_data=output_data,
        output_sha256=hashlib.sha256(output_data).hexdigest(),
        segments=tuple(segments),
        final_prompts=tuple(final_prompts),
    )


def load_prompt_fusion_manifest(
    *, root: Path, skill_source_path: Path,
) -> FrozenPromptFusion:
    """Revalidate the manifest-last fusion production at a paid boundary."""
    root = Path(root).resolve()
    manifest_path = root / "work" / h3_project.SOURCE_FILENAME
    manifest_data, manifest = _fusion_json(
        manifest_path, "prompt_fusion_manifest_invalid"
    )
    if (
        set(manifest) != {
            "schema", "version", "image_acceptance_sha256", "input",
            "output", "skill", "segments",
        }
        or manifest.get("schema") != PROMPT_FUSION_MANIFEST_SCHEMA
        or manifest.get("version") != PROMPT_FUSION_MANIFEST_VERSION
        or not isinstance(manifest.get("image_acceptance_sha256"), str)
        or len(manifest["image_acceptance_sha256"]) != 64
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")

    def artifact(value: object, expected_path: str) -> Path:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or value.get("path") != expected_path
            or not isinstance(value.get("sha256"), str)
        ):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        candidate = root / expected_path
        if candidate.is_symlink():
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        try:
            data = candidate.read_bytes()
        except OSError:
            raise LongGenerationError("prompt_fusion_manifest_invalid") from None
        if not data or hashlib.sha256(data).hexdigest() != value["sha256"]:
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        return candidate

    input_path = artifact(
        manifest["input"], f"work/{h3_project.SKILL_INPUT_FILENAME}"
    )
    output_path = artifact(manifest["output"], "work/h3_prompt_plan.json")
    skill = manifest["skill"]
    if (
        not isinstance(skill, Mapping)
        or set(skill) != {"source_path", "frozen_path", "sha256"}
        or skill.get("source_path") != "skills/video-prompt-fusion/SKILL.md"
        or skill.get("frozen_path") != "work/video_prompt_fusion_skill.md"
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    frozen_skill = root / skill["frozen_path"]
    try:
        source_skill_data = Path(skill_source_path).read_bytes()
        frozen_skill_data = frozen_skill.read_bytes()
    except OSError:
        raise LongGenerationError("prompt_fusion_manifest_invalid") from None
    if (
        frozen_skill.is_symlink()
        or not source_skill_data
        or source_skill_data != frozen_skill_data
        or hashlib.sha256(source_skill_data).hexdigest() != skill.get("sha256")
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    frozen = load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=root,
    )
    segments = manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != len(frozen.final_prompts):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    for index, (binding, prompt) in enumerate(
        zip(segments, frozen.final_prompts), 1
    ):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"index", "final_prompt_sha256"}
            or binding.get("index") != index
            or binding.get("final_prompt_sha256")
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
    return frozen


@dataclass(frozen=True)
class FrozenSegment:
    index: int
    start_s: float
    end_s: float
    chain_id: str
    join_mode: str
    workdir: Path
    first_frame: Path
    first_frame_data: bytes
    last_frame: Path
    last_frame_data: bytes
    prompt: str
    keyframes: tuple[h3.FrozenFrame, ...] = ()
    multimodal: h3_project.FrozenProjectMultimodal | None = None
    prompt_fusion_audio_paths: tuple[Path, ...] = ()
    dialogue: tuple[dict, ...] = ()
    dialogue_sha256: str | None = None


@dataclass(frozen=True)
class FrozenPlan:
    root: Path
    source: Path
    receipt: str
    segments: tuple[FrozenSegment, ...]
    receipt_version: int = long_video.PLAN_RECEIPT_VERSION
    aspect_ratio: str = h3.H3_DEFAULT_ASPECT_RATIO
    resolution: str = h3.H3_DEFAULT_RESOLUTION
    legacy_layout: bool = False
    workflow: str = h3.H3_WORKFLOW
    prompt_fusion: FrozenPromptFusion | None = None


def _segment_duration_s(plan: FrozenPlan, segment: FrozenSegment) -> float:
    """Interpret a segment boundary with its frozen plan receipt version."""
    try:
        return long_video.segment_duration_s(
            segment.start_s,
            segment.end_s,
            receipt_version=plan.receipt_version,
        )
    except long_video.LongVideoError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _is_h3_multimodal_plan(plan: FrozenPlan) -> bool:
    workflows = getattr(h3, "H3_MULTIMODAL_WORKFLOWS", frozenset())
    return isinstance(workflows, (set, frozenset)) and plan.workflow in workflows


def _requires_context_ir(plan: FrozenPlan) -> bool:
    """Current fusion prompts and historical multimodal prompts use Context IR."""
    return plan.prompt_fusion is not None or _is_h3_multimodal_plan(plan)


def _segment_uses_h3_native_audio(
    plan: FrozenPlan, segment: FrozenSegment,
) -> bool:
    if plan.prompt_fusion is not None:
        return bool(segment.prompt_fusion_audio_paths)
    return _is_h3_multimodal_plan(plan)


def _native_audio_segment_indices(plan: FrozenPlan) -> frozenset[int]:
    return frozenset(
        segment.index for segment in plan.segments
        if _segment_uses_h3_native_audio(plan, segment)
    )


def _generation_uses_h3_native_audio(plan: FrozenPlan, generation: Mapping) -> bool:
    expected = H3_NATIVE_AUDIO_ROUTE if _is_h3_multimodal_plan(plan) else None
    actual = generation.get("audio_route")
    if actual != expected:
        raise LongGenerationError("long_video_audio_route_invalid")
    return expected is not None


def _stitch_segments(
    plan: FrozenPlan,
    provider_media: Mapping[int, tuple[str, Mapping[str, object]]] | None = None,
) -> list[stitch.StitchSegment]:
    media = provider_media or {}
    return [
        stitch.StitchSegment(
            item.workdir / "generated.mp4",
            _segment_duration_s(plan, item),
            item.join_mode,
            media.get(item.index, (None, None))[0],
            media.get(item.index, (None, None))[1],
        )
        for item in plan.segments
    ]


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _canonical_digest(value: object) -> str:
    try:
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode()
    except (TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise LongGenerationError("prompt_fusion_input_invalid") from None


def prompt_fusion_image_authority_sha256(meta: Mapping) -> str:
    """Bind manual acceptance, or the exact v4 MediaKit-only receipt."""
    acceptance = meta.get("_image_user_acceptance")
    if (
        isinstance(acceptance, Mapping)
        and isinstance(acceptance.get("sha256"), str)
        and len(acceptance["sha256"]) == 64
    ):
        return acceptance["sha256"]
    private = meta.get("_postprocess_receipt")
    post = meta.get("postprocess")
    options = private.get("options") if isinstance(private, Mapping) else None
    if (
        isinstance(private, Mapping)
        and private.get("version") == 4
        and isinstance(options, Mapping)
        and options.get("optimize_image") is False
        and isinstance(post, Mapping)
        and post.get("status") == "done"
    ):
        return h3.canonical_json_sha256(dict(private))
    raise LongGenerationError("image_acceptance_required")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_prompt_fusion_input(
    *, root: Path, meta: Mapping, plan: FrozenPlan,
    dialogue_mode: str, dialogue_delivery: str,
) -> bytes:
    """Compile the four frozen fusion inputs for all ordered segments."""
    root = Path(root).resolve()
    if plan.root != root or not plan.segments:
        raise LongGenerationError("prompt_fusion_input_invalid")
    raw_segments = meta.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(plan.segments):
        raise LongGenerationError("prompt_fusion_input_invalid")
    optimization = meta.get("_image_optimization")
    optimization_frames = (
        optimization.get("frames") if isinstance(optimization, Mapping) else None
    )
    if not isinstance(optimization_frames, list):
        raise LongGenerationError("prompt_fusion_input_invalid")
    try:
        requested_delivery = dialogue_delivery_contract.parse(dialogue_delivery)
        authoritative_dialogue = tuple(
            line for segment in plan.segments for line in segment.dialogue
        )
        resolved_delivery = dialogue_delivery_contract.resolve(
            requested_delivery, authoritative_dialogue
        ).value
    except ValueError:
        raise LongGenerationError("invalid_dialogue_delivery", 422) from None
    if dialogue_mode != "none" and authoritative_dialogue \
            and resolved_delivery == "on_screen":
        raise LongGenerationError("on_screen_authority_unavailable")

    compiled_segments: list[dict] = []
    optimization_indices = {
        item.get("segment_index")
        for item in optimization_frames if isinstance(item, Mapping)
    }
    top_level_short = len(plan.segments) == 1 and optimization_indices == {0}
    if not top_level_short and optimization_indices != set(
        range(1, len(plan.segments) + 1)
    ):
        raise LongGenerationError("image_optimization_prompt_invalid")
    for index, (raw, segment) in enumerate(
        zip(raw_segments, plan.segments), 1
    ):
        if not isinstance(raw, Mapping) or segment.index != index:
            raise LongGenerationError("prompt_fusion_input_invalid")
        old_prompt = raw.get("visual_prompt")
        if not isinstance(old_prompt, str) or not old_prompt.strip():
            raise LongGenerationError("prompt_fusion_input_invalid")
        if len(segment.keyframes) != 9:
            raise LongGenerationError("postprocess_artifacts_invalid")
        keyframes: list[dict] = []
        for order, (path, data) in enumerate(segment.keyframes, 1):
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if path.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            try:
                current_data = path.read_bytes()
            except OSError:
                raise LongGenerationError(
                    "prompt_fusion_input_invalid"
                ) from None
            if (
                not data
                or current_data != data
                or hashlib.sha256(current_data).hexdigest()
                != hashlib.sha256(data).hexdigest()
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            keyframes.append({
                "order": order,
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
            })

        source_index = 0 if top_level_short else index
        prompts = [
            item for item in optimization_frames
            if isinstance(item, Mapping)
            and item.get("segment_index") == source_index
        ]
        if len(prompts) != 9:
            raise LongGenerationError("image_optimization_prompt_invalid")
        prompt_bindings: list[dict] = []
        for order, item in enumerate(prompts, 1):
            text = item.get("current")
            digest = item.get("sha256")
            if (
                not isinstance(text, str)
                or not text.strip()
                or digest != hashlib.sha256(text.encode("utf-8")).hexdigest()
            ):
                raise LongGenerationError("image_optimization_prompt_invalid")
            prompt_bindings.append({
                "order": order, "text": text, "sha256": digest,
            })

        dialogue = () if dialogue_mode == "none" else segment.dialogue
        lines: list[dict] = []
        for order, line in enumerate(dialogue, 1):
            try:
                text = line["text"]
                start_s = line["start_s"]
                end_s = line["end_s"]
            except (KeyError, TypeError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if not isinstance(text, str) or not text.strip():
                raise LongGenerationError("prompt_fusion_input_invalid")
            lines.append({
                "order": order,
                "text": text,
                "start_s": start_s,
                "end_s": end_s,
                "delivery": resolved_delivery,
                "voice_ref": 1,
            })
        voice_references: list[dict] = []
        if lines:
            audio_path = segment.workdir / "work" / "voice.mp3"
            if len(plan.segments) == 1 and not audio_path.is_file():
                audio_path = root / "work" / "voice.mp3"
            if audio_path.is_symlink():
                raise LongGenerationError("reference_audio_binding_invalid")
            try:
                audio_data = audio_path.read_bytes()
                audio_relative = audio_path.resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                raise LongGenerationError("reference_audio_binding_invalid") from None
            if not audio_data:
                raise LongGenerationError("reference_audio_binding_invalid")
            voice_references.append({
                "voice_ref": 1,
                "path": audio_relative,
                "sha256": hashlib.sha256(audio_data).hexdigest(),
                "purpose": "voice",
            })
        lines_json = json.dumps(
            lines, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
        compiled_segments.append({
            "index": index,
            "new_keyframes": keyframes,
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(old_prompt.encode("utf-8")).hexdigest(),
            },
            "image_optimization_prompt": prompt_bindings,
            "audio_content": {
                "lines_json": lines_json,
                "lines_sha256": hashlib.sha256(
                    lines_json.encode("utf-8")
                ).hexdigest(),
                "voice_references": voice_references,
            },
        })
    return _canonical_json_bytes({
        "schema": PROMPT_FUSION_INPUT_SCHEMA,
        "version": PROMPT_FUSION_VERSION,
        "segments": compiled_segments,
    })


def _publish_fusion_h3_segments(
    *, root: Path, meta: Mapping, base: FrozenPlan,
    fusion: FrozenPromptFusion, dialogue_mode: str,
) -> tuple[list[dict], list[dict]]:
    """Freeze exact fusion prompts; Context IR is the sole later transformer."""
    receipt_segments: list[dict] = []
    updated_segments: list[dict] = []
    raw_segments = meta.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(base.segments):
        raise LongGenerationError("prompt_fusion_input_invalid")
    for source, segment, fusion_source, final_prompt in zip(
        raw_segments, base.segments, fusion.segments, fusion.final_prompts
    ):
        if not isinstance(source, Mapping):
            raise LongGenerationError("prompt_fusion_input_invalid")
        work = segment.workdir / "work"
        visual_path = work / "fusion_prompt.txt"
        _atomic_bytes(visual_path, final_prompt.encode("utf-8"))

        for item in fusion_source["new_keyframes"]:
            source_candidate = root / item["path"]
            if source_candidate.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            source_path = source_candidate.resolve()
            try:
                source_path.relative_to(root)
                source_data = source_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if (
                not source_data
                or hashlib.sha256(source_data).hexdigest() != item["sha256"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")

        audio = fusion_source["audio_content"]
        lines = json.loads(audio["lines_json"])
        for item in audio["voice_references"]:
            source_candidate = root / item["path"]
            if source_candidate.is_symlink():
                raise LongGenerationError("reference_audio_binding_invalid")
            source_path = source_candidate.resolve()
            try:
                source_path.relative_to(root)
                data = source_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("reference_audio_binding_invalid") from None
            if (
                not data
                or hashlib.sha256(data).hexdigest() != item["sha256"]
            ):
                raise LongGenerationError("reference_audio_binding_invalid")
        dialogue = () if dialogue_mode == "none" else segment.dialogue
        if len(lines) != len(dialogue):
            raise LongGenerationError("prompt_fusion_input_invalid")
        for order, (compiled, authoritative) in enumerate(
            zip(lines, dialogue), 1
        ):
            if (
                compiled["order"] != order
                or compiled["text"] != authoritative.get("text")
                or compiled["start_s"] != authoritative.get("start_s")
                or compiled["end_s"] != authoritative.get("end_s")
                or compiled["delivery"] != "off_screen"
                or compiled["voice_ref"] != 1
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
        updated = {**dict(source), "prompt": final_prompt}
        updated_segments.append(updated)
        receipt_segments.append({
            **updated,
            "source_path": root / "work" / str(source["source"]),
            "keyframe_paths": [
                root / "work" / str(path) for path in source["keyframe_paths"]
            ],
            "first_frame_path": root / "work" / str(source["first_frame_path"]),
            "last_frame_path": root / "work" / str(source["last_frame_path"]),
            "visual_prompt_path": work / "visual_prompt.txt",
            "final_prompt_path": visual_path,
            "dialogue": list(dialogue),
        })
    return receipt_segments, updated_segments


def plan_receipt(root: Path, meta: Mapping) -> str | None:
    name = meta.get("long_video_plan_receipt")
    if name != long_video.PLAN_RECEIPT_FILENAME:
        return None
    path = Path(root) / name
    if not path.is_file():
        return None
    try:
        return _digest(path)
    except LongGenerationError:
        return None


def normalize_single_segment_project(
    settings: Settings, cid: str, meta: Mapping,
) -> dict:
    """Adapt a frozen pre-unification v4 N=1 project to segments[1]."""
    root = (settings.data_dir / cid).resolve()
    if (
        meta.get("schema_version") != 2
        or isinstance(meta.get("segments"), list)
        or meta.get("long_video_plan_receipt") is not None
    ):
        return dict(meta)
    private = meta.get("_postprocess_receipt")
    if (
        not isinstance(private, Mapping)
        or private.get("version") != 4
        or not isinstance(private.get("options"), Mapping)
    ):
        return dict(meta)
    try:
        duration = float(meta["duration_s"])
    except (KeyError, TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if not 0 < duration <= long_video.SHORT_VIDEO_MAX_S:
        raise LongGenerationError("long_video_plan_invalid")
    names = meta.get("keyframes")
    if not isinstance(names, list) or len(names) != 9:
        raise LongGenerationError("postprocess_artifacts_invalid")
    originals = [root / "work" / "keyframes" / str(name) for name in names]
    try:
        selected = postprocess.generation_keyframes(
            root, dict(meta), originals, settings=settings,
        )
    except postprocess.PostprocessError as exc:
        detail = exc.detail if isinstance(exc.detail, str) else exc.detail["code"]
        raise LongGenerationError(detail, exc.status) from None
    if len(selected) != 9:
        raise LongGenerationError("postprocess_artifacts_invalid")
    sources = sorted(root.glob("source.*"))
    if len(sources) != 1:
        raise LongGenerationError("long_video_plan_invalid")
    segment_dir = root / "work" / "segments" / "1"
    segment_work = segment_dir / "work"
    if sources[0].is_symlink():
        raise LongGenerationError("long_video_plan_invalid")
    try:
        source_data = sources[0].read_bytes()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None
    if not source_data:
        raise LongGenerationError("long_video_plan_invalid")
    _atomic_bytes(segment_dir / "source.mp4", source_data)
    segment_work.mkdir(parents=True, exist_ok=True)
    visual_path = root / "work" / "visual_prompt.txt"
    try:
        visual = visual_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    frozen_visual = segment_work / "visual_prompt.txt"
    _atomic_bytes(frozen_visual, visual.encode("utf-8"))
    dialogue = []
    raw_lines = meta.get("voice_line_provenance")
    if isinstance(raw_lines, list):
        for raw in raw_lines:
            if not isinstance(raw, Mapping) or raw.get("kept") is not True:
                continue
            line = {
                key: raw[key] for key in ("text", "start_s", "end_s")
                if key in raw
            }
            for optional in ("classification", "language"):
                if optional in raw:
                    line[optional] = raw[optional]
            dialogue.append(line)
    try:
        final = (
            f"{_PIPELINE_NO_BGM}\n"
            + prepared_input.compose_final_prompt(
                long_video.compose_segment_visual_prompt(visual), dialogue,
            )
        )
    except (prepared_input.PreparedInputError, long_video.LongVideoError):
        raise LongGenerationError("long_video_plan_invalid") from None
    final_path = segment_work / "prompt.txt"
    _atomic_bytes(final_path, final.encode("utf-8"))
    global_audio = root / "work" / "voice.mp3"
    if dialogue and global_audio.is_file() and not global_audio.is_symlink():
        _atomic_bytes(segment_work / "voice.mp3", global_audio.read_bytes())
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": round(duration, long_video.BOUNDARY_PRECISION),
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
        "source": "segments/1/source.mp4",
        "keyframes": list(names),
        "keyframe_paths": [
            path.resolve().relative_to(root / "work").as_posix()
            for path in originals
        ],
        "first_frame_path": originals[0].resolve().relative_to(
            root / "work"
        ).as_posix(),
        "last_frame_path": originals[-1].resolve().relative_to(
            root / "work"
        ).as_posix(),
        "visual_prompt": visual,
        "prompt": final,
        "dialogue": dialogue,
        "lines": [line["text"] for line in dialogue],
    }
    receipt_path = long_video.write_plan_receipt(
        root,
        source=sources[0],
        duration_s=duration,
        segments=[{
            **segment,
            "source_path": segment_dir / "source.mp4",
            "keyframe_paths": originals,
            "first_frame_path": originals[0],
            "last_frame_path": originals[-1],
            "visual_prompt_path": frozen_visual,
            "final_prompt_path": final_path,
        }],
        workflow=h3.H3_WORKFLOW,
    )
    updated = storage.update_meta(
        settings.data_dir,
        cid,
        segments=[segment],
        long_video_plan_receipt=receipt_path.name,
    )
    if updated is None:
        raise LongGenerationError("long_video_plan_invalid")
    return updated


def _promote_legacy_segment_multimodal_intent(
    *, root: Path, meta: Mapping, expected_receipt: str,
    previous_bytes: bytes, fit_mode: str, dialogue_mode: str,
    dialogue_delivery: str, aspect_ratio: str, resolution: str,
    settings: Settings | None,
) -> str | None:
    """Preserve the exact historical v2 per-segment multimodal promotion."""
    current_segments = meta.get("segments")
    if not isinstance(current_segments, list) or not current_segments:
        return None
    intent: list[bool] = []
    complete: list[bool] = []
    receipt_segments: list[dict] = []
    authoritative_dialogue: list[dict] = []
    for expected_index, current in enumerate(current_segments, 1):
        if not isinstance(current, Mapping) or current.get("index") != expected_index:
            raise LongGenerationError("long_video_plan_invalid")
        segdir = root / "work" / "segments" / str(expected_index)
        segwork = segdir / "work"
        manifest = segwork / h3_project.SOURCE_FILENAME
        intent.append(any(
            (segwork / name).exists()
            for name in (
                h3_project.SKILL_INPUT_FILENAME,
                "h3_prompt_plan.json",
                h3_project.SOURCE_FILENAME,
            )
        ))
        complete.append(manifest.is_file())
        raw_dialogue = current.get("dialogue")
        if not isinstance(raw_dialogue, list):
            raise LongGenerationError("long_video_plan_invalid")
        authoritative_dialogue.extend(
            dict(line) for line in raw_dialogue if isinstance(line, Mapping)
        )
        try:
            receipt_segments.append({
                **dict(current),
                "source_path": root / "work" / str(current["source"]),
                "keyframe_paths": [
                    root / "work" / str(path)
                    for path in current["keyframe_paths"]
                ],
                "first_frame_path": root / "work" / str(
                    current["first_frame_path"]
                ),
                "last_frame_path": root / "work" / str(
                    current["last_frame_path"]
                ),
                "visual_prompt_path": segwork / "visual_prompt.txt",
                "final_prompt_path": segwork / "prompt.txt",
                "dialogue": [] if dialogue_mode == "none" else raw_dialogue,
                "multimodal_manifest_path": manifest,
            })
        except (KeyError, TypeError):
            raise LongGenerationError("long_video_plan_invalid") from None
    if not any(intent):
        return None
    if not all(complete):
        raise LongGenerationError("long_video_multimodal_incomplete")
    try:
        resolved_delivery = dialogue_delivery_contract.resolve(
            dialogue_delivery_contract.parse(dialogue_delivery),
            tuple(authoritative_dialogue),
        ).value
    except ValueError:
        raise LongGenerationError("invalid_dialogue_delivery", 422) from None
    base = freeze_plan(
        root, meta, expected_receipt, fit_mode, dialogue_mode,
        dialogue_delivery=dialogue_delivery,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        prepare_fit=True,
        settings=settings,
    )
    receipt_path = root / long_video.PLAN_RECEIPT_FILENAME
    try:
        long_video.write_plan_receipt(
            root,
            source=base.source,
            duration_s=float(meta["duration_s"]),
            segments=receipt_segments,
            workflow=h3.H3_MULTIMODAL_WORKFLOW,
            dialogue_mode=dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            resolved_dialogue_delivery=resolved_delivery,
        )
        promoted = _digest(receipt_path)
        freeze_plan(
            root, meta, promoted, fit_mode, dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            prepare_fit=True,
            settings=settings,
        )
        return promoted
    except (
        OSError, KeyError, TypeError, ValueError,
        long_video.LongVideoError, LongGenerationError,
    ) as exc:
        _atomic_bytes(receipt_path, previous_bytes)
        raise LongGenerationError(
            getattr(exc, "code", "long_video_multimodal_invalid")
        ) from None


def finalize_multimodal_plan(
    root: Path,
    meta: Mapping,
    expected_receipt: str,
    fit_mode: str,
    dialogue_mode: str,
    *,
    aspect_ratio: str,
    resolution: str,
    dialogue_delivery: str = "auto",
    settings: Settings | None = None,
) -> str | None:
    """Freeze/consume the one project-level prompt fusion before H3."""
    root = Path(root).resolve()
    if isinstance(meta.get("generation"), Mapping):
        return None
    receipt_path = root / long_video.PLAN_RECEIPT_FILENAME
    try:
        previous_bytes = receipt_path.read_bytes()
        payload = json.loads(previous_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if hashlib.sha256(previous_bytes).hexdigest() != expected_receipt:
        raise LongGenerationError("long_video_plan_changed")
    receipt_version = payload.get("version") if isinstance(payload, Mapping) else None
    if receipt_version == long_video.MULTIMODAL_PLAN_RECEIPT_VERSION:
        if isinstance(payload, Mapping) and payload.get("prompt_fusion") is not None:
            if settings is None:
                raise LongGenerationError("prompt_fusion_manifest_invalid")
            load_prompt_fusion_manifest(
                root=root,
                skill_source_path=PROMPT_FUSION_SKILL_SOURCE,
            )
        return None
    if receipt_version != long_video.PLAN_RECEIPT_VERSION:
        return None
    private_postprocess = meta.get("_postprocess_receipt")
    if (
        not isinstance(private_postprocess, Mapping)
        or private_postprocess.get("version") != 4
    ):
        return _promote_legacy_segment_multimodal_intent(
            root=root,
            meta=meta,
            expected_receipt=expected_receipt,
            previous_bytes=previous_bytes,
            fit_mode=fit_mode,
            dialogue_mode=dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            settings=settings,
        )
    if settings is None:
        raise LongGenerationError("prompt_fusion_refresh_required")

    base = freeze_plan(
        root,
        meta,
        expected_receipt,
        fit_mode,
        dialogue_mode,
        dialogue_delivery=dialogue_delivery,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        prepare_fit=True,
        settings=settings,
    )
    input_data = build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=base,
        dialogue_mode=dialogue_mode,
        dialogue_delivery=dialogue_delivery,
    )
    image_authority_sha256 = prompt_fusion_image_authority_sha256(meta)
    from app import pipeline  # local import: pipeline owns the Codex runner seam

    queued = pipeline.queue_prompt_fusion(
        settings,
        str(meta.get("id")),
        input_data=input_data,
        image_acceptance_sha256=image_authority_sha256,
    )
    if queued == "failed":
        raise LongGenerationError("prompt_fusion_failed")
    if queued != "done":
        raise LongGenerationError("prompt_fusion_refresh_required")
    fusion = load_prompt_fusion_manifest(
        root=root, skill_source_path=pipeline.PROMPT_FUSION_SKILL_MD,
    )
    if fusion.input_data != input_data:
        raise LongGenerationError("prompt_fusion_input_invalid")
    try:
        resolved_delivery = dialogue_delivery_contract.resolve(
            dialogue_delivery_contract.parse(dialogue_delivery),
            tuple(line for segment in base.segments for line in segment.dialogue),
        ).value
    except ValueError:
        raise LongGenerationError("invalid_dialogue_delivery", 422) from None
    receipt_segments, _updated_segments = _publish_fusion_h3_segments(
        root=root,
        meta=meta,
        base=base,
        fusion=fusion,
        dialogue_mode=dialogue_mode,
    )
    try:
        fusion_has_audio = any(
            bool(segment["audio_content"]["voice_references"])
            for segment in fusion.segments
        )
    except (KeyError, TypeError):
        raise LongGenerationError("prompt_fusion_input_invalid") from None
    promoted_workflow = (
        h3.H3_MULTIMODAL_WORKFLOW if fusion_has_audio else h3.H3_WORKFLOW
    )
    try:
        long_video.write_plan_receipt(
            root,
            source=base.source,
            duration_s=float(meta["duration_s"]),
            segments=receipt_segments,
            workflow=promoted_workflow,
            dialogue_mode=dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            resolved_dialogue_delivery=resolved_delivery,
            prompt_fusion_manifest_path=root / "work" / h3_project.SOURCE_FILENAME,
        )
        promoted = _digest(receipt_path)
        promoted_meta = dict(meta)
        freeze_plan(
            root,
            promoted_meta,
            promoted,
            fit_mode,
            dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            prepare_fit=True,
            settings=settings,
        )
        storage.update_meta(
            settings.data_dir,
            str(meta.get("id")),
            long_video_plan_receipt=long_video.PLAN_RECEIPT_FILENAME,
        )
        return promoted
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        long_video.LongVideoError,
        LongGenerationError,
    ) as exc:
        _atomic_bytes(receipt_path, previous_bytes)
        code = getattr(exc, "code", "long_video_multimodal_invalid")
        raise LongGenerationError(code) from None



def _bound_path(root: Path, artifact: object) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise LongGenerationError("long_video_plan_invalid")
    relative, expected = artifact.get("path"), artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise LongGenerationError("long_video_plan_invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None
    if not path.is_file() or _digest(path) != expected:
        raise LongGenerationError("long_video_plan_invalid")
    return path


def _bound_bytes(root: Path, artifact: object) -> tuple[Path, bytes]:
    """Read one receipt-bound artifact once and retain the verified bytes."""
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise LongGenerationError("long_video_plan_invalid")
    relative, expected = artifact.get("path"), artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise LongGenerationError("long_video_plan_invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
        data = path.read_bytes()
    except (ValueError, OSError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if hashlib.sha256(data).hexdigest() != expected:
        raise LongGenerationError("long_video_plan_invalid")
    return path, data


def _relative_to_work(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root / "work").as_posix()
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _fit_anchor(
    path: Path, data: bytes, output: Path, fit_mode: str, aspect_ratio: str,
    *, prepare: bool,
) -> tuple[Path, bytes]:
    if fit_mode == "none":
        return path, data
    try:
        fitted = frame_fit.fit_frame_bytes(
            data, fit_mode, aspect_ratio, label=path.name
        )
    except frame_fit.FrameFitError:
        raise LongGenerationError("frame_fit_failed") from None
    target = output / (path.stem + ".png")
    if prepare:
        output.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_bytes(fitted)
            temporary.replace(target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise LongGenerationError("frame_fit_failed") from None
    else:
        try:
            if target.read_bytes() != fitted:
                raise LongGenerationError("frame_fit_failed")
        except OSError:
            raise LongGenerationError("frame_fit_failed") from None
    return target, fitted


def _persisted_fit_layout(meta: Mapping) -> str | None:
    generation = meta.get("generation")
    if not isinstance(generation, Mapping) or "fit_layout" not in generation:
        return None
    layout = generation.get("fit_layout")
    if layout not in _FIT_LAYOUTS:
        raise LongGenerationError("frame_fit_failed")
    return str(layout)


def _fit_outputs_complete(paths: tuple[Path, Path], aspect_ratio: str) -> bool:
    if not all(path.is_file() for path in paths):
        return False
    try:
        return not frame_fit.frames_require_fit(paths, aspect_ratio)
    except frame_fit.FrameFitError:
        return False


def freeze_plan(root: Path, meta: Mapping, expected_receipt: str, fit_mode: str,
                dialogue_mode: str, *, aspect_ratio: str | None = None,
                resolution: str | None = None,
                dialogue_delivery: str = "auto",
                prepare_fit: bool = True,
                settings: Settings | None = None) -> FrozenPlan:
    """Validate every immutable plan fact and pre-fit every source anchor."""
    root = Path(root).resolve()
    if (aspect_ratio is None) != (resolution is None):
        raise LongGenerationError("invalid_generation_parameters", 422)
    parameters_explicit = aspect_ratio is not None
    persisted_layout = _persisted_fit_layout(meta)
    detect_existing_layout = (
        persisted_layout is None and not prepare_fit and fit_mode != "none"
    )
    if persisted_layout is not None:
        legacy_layout: bool | None = persisted_layout == FIT_LAYOUT_LEGACY
    elif prepare_fit or fit_mode == "none":
        # Current callers explicitly supply semantic parameters and always use
        # the versioned path.  Calls without them are the historical contract.
        # initial_generation persists this choice before any provider POST.
        legacy_layout = not parameters_explicit
    else:
        # Pre-marker attempts are recovered from their complete, decodable
        # frozen files.  Never let fields added after the POST select a path.
        legacy_layout = None
    aspect_ratio = (
        meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
        if aspect_ratio is None else aspect_ratio
    )
    resolution = (
        meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
        if resolution is None else resolution
    )
    if aspect_ratio not in h3.H3_ASPECT_RATIOS:
        raise LongGenerationError("invalid_aspect_ratio", 422)
    if resolution not in h3.H3_RESOLUTIONS:
        raise LongGenerationError("invalid_resolution", 422)
    if fit_mode not in {"none", "crop", "pad"}:
        raise LongGenerationError("frame_fit_failed")
    name = meta.get("long_video_plan_receipt")
    if name != long_video.PLAN_RECEIPT_FILENAME:
        raise LongGenerationError("long_video_plan_invalid")
    receipt_path = root / name
    try:
        receipt_data = receipt_path.read_bytes()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None
    receipt = hashlib.sha256(receipt_data).hexdigest()
    if expected_receipt != receipt:
        raise LongGenerationError("long_video_plan_changed", 409)
    try:
        payload = json.loads(receipt_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    receipt_version = payload.get("version") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "duet.long-video-plan"
        or isinstance(receipt_version, bool)
        or not isinstance(receipt_version, int)
        or receipt_version not in {
            long_video.LEGACY_PLAN_RECEIPT_VERSION,
            long_video.PLAN_RECEIPT_VERSION,
            long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        }
        or payload.get("workflow") not in _PLAN_WORKFLOWS
    ):
        raise LongGenerationError("long_video_plan_invalid")
    source = _bound_path(root, payload.get("source"))
    receipt_workflow = payload["workflow"]
    is_multimodal_receipt = (
        receipt_version == long_video.MULTIMODAL_PLAN_RECEIPT_VERSION
    )
    if (
        is_multimodal_receipt
        and receipt_workflow not in h3.H3_REFERENCE_WORKFLOWS
    ) or (
        not is_multimodal_receipt
        and receipt_workflow in h3.H3_MULTIMODAL_WORKFLOWS
    ):
        raise LongGenerationError("long_video_plan_invalid")
    if is_multimodal_receipt and payload.get("dialogue_mode") != dialogue_mode:
        raise LongGenerationError(
            "long_video_multimodal_dialogue_refresh_required", 409
        )
    if is_multimodal_receipt and "dialogue_delivery" in payload:
        if payload.get("dialogue_delivery") != dialogue_delivery:
            raise LongGenerationError(
                "long_video_multimodal_dialogue_refresh_required", 409
            )
    frozen_fusion = None
    fusion_binding = payload.get("prompt_fusion")
    if fusion_binding is not None:
        if (
            not is_multimodal_receipt
            or not isinstance(fusion_binding, Mapping)
            or set(fusion_binding) != {"path", "sha256"}
            or fusion_binding.get("path") != f"work/{h3_project.SOURCE_FILENAME}"
        ):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        manifest_path = _bound_path(root, fusion_binding)
        frozen_fusion = load_prompt_fusion_manifest(
            root=root,
            skill_source_path=PROMPT_FUSION_SKILL_SOURCE,
        )
        if manifest_path != root / "work" / h3_project.SOURCE_FILENAME \
                or len(frozen_fusion.segments) != len(payload.get("segments", [])):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
    generation = meta.get("generation")
    persisted_workflow = (
        generation.get("workflow") if isinstance(generation, Mapping) else None
    )
    if persisted_workflow is not None and persisted_workflow not in _PLAN_WORKFLOWS:
        raise LongGenerationError("long_video_plan_invalid")
    # Existing attempts without a marker predate reference-mode long video and
    # must resume their boundary receipts GET-only.  Unsubmitted v2 plans are
    # safe to promote because every provider segment is within the current limit.
    workflow = (
        persisted_workflow
        or (receipt_workflow if isinstance(generation, Mapping) else None)
        or (
            h3.H3_WORKFLOW
            if receipt_version == long_video.PLAN_RECEIPT_VERSION
            else receipt_workflow
        )
    )
    try:
        duration = float(payload["video"]["duration_s"])
        meta_duration = float(meta["duration_s"])
    except (KeyError, TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    raw_segments, meta_segments = payload.get("segments"), meta.get("segments")
    minimum_plan_duration = (
        long_video.PREVIOUS_SHORT_VIDEO_MAX_S
        if receipt_version == long_video.LEGACY_PLAN_RECEIPT_VERSION
        else 0.0
    )
    if (
        not math.isfinite(duration)
        or duration <= minimum_plan_duration
        or duration > long_video.LONG_VIDEO_MAX_S
        or abs(duration - meta_duration) > _EPS
        or not isinstance(raw_segments, list)
        or not raw_segments
        or not isinstance(meta_segments, list)
        or len(raw_segments) != len(meta_segments)
    ):
        raise LongGenerationError("long_video_plan_invalid")

    project_selected_paths: dict[int, tuple[Path, ...]] = {}
    if workflow in h3.H3_REFERENCE_WORKFLOWS:
        project_originals: list[Path] = []
        project_counts: list[int] = []
        for raw in raw_segments:
            keys = raw.get("keyframes") if isinstance(raw, Mapping) else None
            if not isinstance(keys, list) or not 1 <= len(keys) <= 9:
                raise LongGenerationError("long_video_plan_invalid")
            originals = [_bound_path(root, artifact) for artifact in keys]
            project_originals.extend(originals)
            project_counts.append(len(originals))
        try:
            selected_project = postprocess.generation_keyframes(
                root, dict(meta), project_originals, settings=settings,
            )
        except postprocess.PostprocessError as exc:
            detail = exc.detail if isinstance(exc.detail, str) else exc.detail["code"]
            raise LongGenerationError(detail, exc.status) from None
        if len(selected_project) != len(project_originals):
            raise LongGenerationError("postprocess_artifacts_invalid")
        cursor = 0
        for index, count in enumerate(project_counts, 1):
            project_selected_paths[index] = tuple(
                selected_project[cursor:cursor + count]
            )
            cursor += count

    frozen: list[FrozenSegment] = []
    authoritative_dialogue_for_delivery: list[dict] = []
    max_provider_duration = (
        long_video.LEGACY_PROVIDER_MAX_DURATION_S
        if len(raw_segments) == 1
        else (
            long_video.SEGMENT_PROVIDER_MAX_DURATION_S
            if receipt_version == long_video.PLAN_RECEIPT_VERSION
            else long_video.LEGACY_PROVIDER_MAX_DURATION_S
        )
    )
    previous_end = 0.0
    previous_chain = None
    for position, (raw, current) in enumerate(zip(raw_segments, meta_segments), 1):
        if not isinstance(raw, dict) or not isinstance(current, dict):
            raise LongGenerationError("long_video_plan_invalid")
        try:
            index = raw["index"]
            start_s, end_s = float(raw["start_s"]), float(raw["end_s"])
            chain_id, join_mode = raw["chain_id"], raw["join_mode"]
        except (KeyError, TypeError, ValueError):
            raise LongGenerationError("long_video_plan_invalid") from None
        try:
            frozen_duration = long_video.segment_duration_s(
                start_s, end_s, receipt_version=receipt_version
            )
        except long_video.LongVideoError:
            raise LongGenerationError("long_video_plan_invalid") from None
        comparable = ("index", "start_s", "end_s", "chain_id", "join_mode")
        if (
            index != position
            or any(current.get(key) != raw.get(key) for key in comparable)
            or not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or abs(start_s - previous_end) > _EPS
            or frozen_duration < long_video.SEGMENT_MIN_S
            or long_video.provider_duration_s(
                start_s, end_s, receipt_version=receipt_version
            )
            > max_provider_duration
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
            or (position == 1 and join_mode != "hard_cut")
            or (join_mode == "continue" and chain_id != previous_chain)
            or (position > 1 and join_mode == "hard_cut" and chain_id == previous_chain)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        if raw.get("source") is None:
            raise LongGenerationError("long_video_plan_invalid")
        segment_source = _bound_path(root, raw["source"])
        keys = raw.get("keyframes")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 9:
            raise LongGenerationError("long_video_plan_invalid")
        bound_keyframes = [_bound_bytes(root, artifact) for artifact in keys]
        keyframe_paths = [path for path, _data in bound_keyframes]
        anchors = raw.get("anchors")
        if (
            not isinstance(anchors, list)
            or len(anchors) != 2
            or [item.get("role") if isinstance(item, dict) else None for item in anchors]
            != ["first", "end"]
        ):
            raise LongGenerationError("long_video_plan_invalid")
        first_source, first_source_data = _bound_bytes(
            root, {k: v for k, v in anchors[0].items() if k != "role"}
        )
        last_source, last_source_data = _bound_bytes(
            root, {k: v for k, v in anchors[1].items() if k != "role"}
        )
        expected_prefix = f"segments/{index}/"
        if (
            current.get("source") != expected_prefix + "source.mp4"
            or segment_source != root / "work" / current["source"]
            or current.get("keyframe_paths") != [
                _relative_to_work(root, path) for path in keyframe_paths
            ]
            or current.get("first_frame_path")
            != _relative_to_work(root, first_source)
            or current.get("last_frame_path")
            != _relative_to_work(root, last_source)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        visual_path, visual_data = _bound_bytes(root, raw.get("visual_prompt"))
        final_path, final_data = _bound_bytes(root, raw.get("final_prompt"))
        try:
            visual = visual_data.decode("utf-8")
            final = final_data.decode("utf-8")
        except UnicodeDecodeError:
            raise LongGenerationError("long_video_plan_invalid") from None
        source_dialogue = current.get("dialogue")
        if isinstance(source_dialogue, list):
            authoritative_dialogue_for_delivery.extend(
                dict(line) for line in source_dialogue
                if isinstance(line, Mapping)
            )
        dialogue = (
            []
            if is_multimodal_receipt and dialogue_mode == "none"
            else source_dialogue
        )
        dialogue_binding = raw.get("dialogue")
        if (
            not isinstance(source_dialogue, list)
            or not isinstance(dialogue, list)
            or not isinstance(dialogue_binding, dict)
            or set(dialogue_binding) != {"count", "sha256"}
            or dialogue_binding.get("count") != len(dialogue)
            or dialogue_binding.get("sha256") != _canonical_digest(dialogue)
            or current.get("visual_prompt") != visual
            or (
                not is_multimodal_receipt
                and current.get("prompt") != final
            )
        ):
            raise LongGenerationError("long_video_plan_invalid")
        if is_multimodal_receipt:
            prompt = final
        else:
            try:
                rebuilt_visual = long_video.compose_segment_visual_prompt(visual)
                auto_prompt = (
                    f"{_PIPELINE_NO_BGM}\n"
                    + prepared_input.compose_final_prompt(
                        rebuilt_visual, source_dialogue
                    )
                )
            except (prepared_input.PreparedInputError, long_video.LongVideoError):
                raise LongGenerationError("long_video_plan_invalid") from None
            if final != auto_prompt:
                raise LongGenerationError("long_video_plan_invalid")
            if dialogue_mode == "none":
                try:
                    rebuilt = prepared_input.compose_final_prompt(
                        long_video.compose_segment_visual_prompt(visual), ()
                    )
                except (prepared_input.PreparedInputError, long_video.LongVideoError):
                    raise LongGenerationError("long_video_plan_invalid") from None
                prompt = f"{_PIPELINE_NO_BGM}\n{rebuilt}"
            else:
                prompt = final
        segdir = root / "work" / "segments" / str(index)
        fit_base = segdir / "work" / "h3_frames"
        legacy_root = fit_base / fit_mode
        aspect_root = fit_base / aspect_ratio.replace(":", "x") / fit_mode
        if detect_existing_layout:
            legacy_paths = (
                legacy_root / "first" / first_source.name,
                legacy_root / "end" / last_source.name,
            )
            aspect_paths = (
                aspect_root / "first" / first_source.name,
                aspect_root / "end" / last_source.name,
            )
            has_legacy = _fit_outputs_complete(legacy_paths, aspect_ratio)
            has_aspect = _fit_outputs_complete(aspect_paths, aspect_ratio)
            if has_legacy == has_aspect:
                raise LongGenerationError("frame_fit_failed")
            if legacy_layout is None:
                legacy_layout = has_legacy
            elif legacy_layout != has_legacy:
                raise LongGenerationError("frame_fit_failed")
        fit_root = legacy_root if legacy_layout else aspect_root
        # Complete all static transformations before the caller can make a POST.
        first, first_data = _fit_anchor(
            first_source, first_source_data, fit_root / "first", fit_mode,
            aspect_ratio,
            prepare=prepare_fit,
        )
        last, last_data = _fit_anchor(
            last_source, last_source_data, fit_root / "end", fit_mode,
            aspect_ratio,
            prepare=prepare_fit,
        )
        frozen_keyframes: tuple[h3.FrozenFrame, ...] = ()
        if workflow in h3.H3_REFERENCE_WORKFLOWS:
            selected_paths = project_selected_paths[index]
            selected: list[h3.FrozenFrame] = []
            original_data = {path.resolve(): data for path, data in bound_keyframes}
            for selected_path in selected_paths:
                resolved = selected_path.resolve()
                data = original_data.get(resolved)
                if data is None:
                    try:
                        data = resolved.read_bytes()
                    except OSError:
                        raise LongGenerationError("postprocess_artifacts_invalid") from None
                    if not data:
                        raise LongGenerationError("postprocess_artifacts_invalid")
                selected.append(
                    _fit_anchor(
                        resolved,
                        data,
                        fit_root / "keyframes",
                        fit_mode,
                        aspect_ratio,
                        prepare=prepare_fit,
                    )
                )
            frozen_keyframes = tuple(selected)
        frozen_multimodal = None
        prompt_fusion_audio_paths: tuple[Path, ...] = ()
        if frozen_fusion is not None:
            if raw.get("multimodal") is not None:
                raise LongGenerationError("long_video_plan_invalid")
            fusion_segment = frozen_fusion.segments[position - 1]
            if final != frozen_fusion.final_prompts[position - 1]:
                raise LongGenerationError("prompt_fusion_output_invalid")
            try:
                fusion_lines = json.loads(
                    fusion_segment["audio_content"]["lines_json"]
                )
                voice_references = fusion_segment["audio_content"][
                    "voice_references"
                ]
            except (KeyError, TypeError, json.JSONDecodeError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if not isinstance(fusion_lines, list) or len(fusion_lines) != len(dialogue):
                raise LongGenerationError("prompt_fusion_input_invalid")
            for line_order, (compiled, authoritative) in enumerate(
                zip(fusion_lines, dialogue), 1
            ):
                if (
                    compiled.get("order") != line_order
                    or compiled.get("text") != authoritative.get("text")
                    or compiled.get("start_s") != authoritative.get("start_s")
                    or compiled.get("end_s") != authoritative.get("end_s")
                    or compiled.get("delivery") != payload.get(
                        "resolved_dialogue_delivery"
                    )
                    or compiled.get("voice_ref") != 1
                ):
                    raise LongGenerationError("prompt_fusion_input_invalid")
            audio_paths: list[Path] = []
            for reference in voice_references:
                audio_paths.append(_bound_path(root, {
                    "path": reference.get("path"),
                    "sha256": reference.get("sha256"),
                }))
            prompt_fusion_audio_paths = tuple(audio_paths)
            if bool(dialogue) != bool(prompt_fusion_audio_paths):
                raise LongGenerationError("prompt_fusion_input_invalid")
        elif is_multimodal_receipt:
            try:
                frozen_multimodal = h3_project.load_bound(
                    root, raw.get("multimodal")
                )
            except h3_project.ProjectMultimodalError as exc:
                raise LongGenerationError(exc.code) from None
            expected_workflow = {
                "multimodal": h3.H3_MULTIMODAL_WORKFLOW,
                "multimodal_hd": h3.H3_MULTIMODAL_HD_WORKFLOW,
            }.get(frozen_multimodal.mode)
            if expected_workflow != workflow:
                raise LongGenerationError("long_video_plan_invalid")
            try:
                # Validate the exact Skill semantics before any caller can
                # prepare or claim a paid H3 attempt.
                h3_project.build_request_from_parts(
                    multimodal=frozen_multimodal,
                    visual_prompt=frozen_multimodal.skill_plan.get(
                        "visual_prompt"
                    ),
                    keyframes=frozen_keyframes,
                    upstream_dialogue=tuple(dict(line) for line in dialogue),
                    upstream_dialogue_receipt_sha256=dialogue_binding["sha256"],
                    source_sha256=_digest(segdir / "source.mp4"),
                    source_duration_s=end_s - start_s,
                    cid=f"freeze-segment-{index}",
                    workdir=segdir / "work" / "h3-native",
                    client_request_id=f"freeze-segment-{index}",
                    duration=long_video.provider_duration_s(
                        start_s, end_s, receipt_version=receipt_version
                    ),
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    autodl_token="freeze-validation-only",
                )
            except (h3.H3Error, h3_project.ProjectMultimodalError):
                raise LongGenerationError("long_video_multimodal_invalid") from None
        elif raw.get("multimodal") is not None:
            raise LongGenerationError("long_video_plan_invalid")
        frozen.append(FrozenSegment(
            index=index,
            start_s=start_s,
            end_s=end_s,
            chain_id=chain_id,
            join_mode=join_mode,
            workdir=segdir,
            first_frame=first,
            first_frame_data=first_data,
            last_frame=last,
            last_frame_data=last_data,
            prompt=prompt,
            keyframes=frozen_keyframes,
            multimodal=frozen_multimodal,
            prompt_fusion_audio_paths=prompt_fusion_audio_paths,
            dialogue=tuple(dict(line) for line in dialogue),
            dialogue_sha256=dialogue_binding["sha256"],
        ))
        previous_end, previous_chain = end_s, chain_id
    if abs(previous_end - duration) > _EPS:
        raise LongGenerationError("long_video_plan_invalid")
    if is_multimodal_receipt and "dialogue_delivery" in payload:
        try:
            resolved_delivery = dialogue_delivery_contract.resolve(
                dialogue_delivery_contract.parse(payload["dialogue_delivery"]),
                tuple(authoritative_dialogue_for_delivery),
            ).value
        except ValueError:
            raise LongGenerationError("long_video_plan_invalid") from None
        if payload.get("resolved_dialogue_delivery") != resolved_delivery:
            raise LongGenerationError("long_video_plan_invalid")
    assert legacy_layout is not None
    return FrozenPlan(
        root=root,
        source=source,
        receipt=receipt,
        segments=tuple(frozen),
        receipt_version=receipt_version,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        legacy_layout=legacy_layout,
        workflow=workflow,
        prompt_fusion=frozen_fusion,
    )


def child_request_id(parent_id: str, receipt: str, index: int) -> str:
    digest = hashlib.sha256(f"{parent_id}\0{receipt}\0{index}".encode()).hexdigest()
    return f"long-{digest[:59]}"  # 64 bytes, deterministic, provider-safe.


def _extract_last_frame(video: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", str(video),
         "-vf", "reverse", "-frames:v", "1", str(temporary)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise LongGenerationError("long_video_tail_frame_failed")
    temporary.replace(output)
    return output


def _bind_h3_operational_roots(
    settings,
    plan: FrozenPlan,
    request: h3.H3Request,
) -> h3.H3Request:
    if not h3.is_multimodal_request(request):
        return request
    return replace(
        request,
        gateway_storage_root=settings.h3_gateway_storage_root,
        speaker_timing_authority_root=(
            plan.root
            if request.speaker_timing_authority_version in {0, 1}
            else None
        ),
    )


def _request(settings, cid: str, plan: FrozenPlan, segment: FrozenSegment,
             parent_id: str, fit_mode: str, *, frozen_child_id: str | None = None,
             prepare_inputs: bool = True, fast_mode: bool = False,
             context_ir_binding: object = None) -> h3.H3Request:
    if plan.prompt_fusion is not None:
        try:
            native_audio = bool(segment.prompt_fusion_audio_paths)
            segment_workflow = (
                h3.H3_MULTIMODAL_WORKFLOW if native_audio else h3.H3_WORKFLOW
            )
            reference_audios = (
                h3.freeze_reference_audios(tuple(
                    (path, "voice")
                    for path in segment.prompt_fusion_audio_paths
                ))
                if native_audio else ()
            )
            fusion_prompt_sha256 = hashlib.sha256(
                segment.prompt.encode("utf-8")
            ).hexdigest()
            multimodal_fields = (
                {
                    "multimodal_compiler_version": "video-prompt-fusion-v1",
                    "audio_required": True,
                }
                if native_audio else {}
            )
            source_request = h3.H3Request(
                cid=f"{cid}-segment-{segment.index}",
                workdir=segment.workdir,
                client_request_id=(
                    frozen_child_id
                    or child_request_id(parent_id, plan.receipt, segment.index)
                ),
                prompt=segment.prompt,
                keyframes=segment.keyframes,
                voice_texts=(),
                voice_receipt=h3.voice_texts_receipt(()),
                duration=long_video.provider_duration_s(
                    segment.start_s,
                    segment.end_s,
                    receipt_version=plan.receipt_version,
                ),
                autodl_token=settings.autodl_art_token,
                timeouts=h3.Timeouts(
                    request_s=settings.h3_request_timeout_s,
                    h3_poll_s=settings.h3_poll_timeout_s,
                    download_s=settings.h3_download_timeout_s,
                    poll_interval_s=settings.h3_poll_interval_s,
                    retry_count=settings.retry_count,
                    retry_interval_s=settings.retry_interval_s,
                ),
                mode="reference",
                aspect_ratio=plan.aspect_ratio,
                resolution=plan.resolution,
                workflow=segment_workflow,
                reference_audios=reference_audios,
                skill_plan_sha256=fusion_prompt_sha256,
                upstream_dialogue_receipt_sha256=segment.dialogue_sha256,
                context_ir_required=True,
                **multimodal_fields,
            )
            if context_ir_binding is None:
                return source_request
            context = _freeze_segment_context_ir(
                settings, plan, segment, source_request
            )
            return _bind_h3_operational_roots(
                settings,
                plan,
                h3_project.apply_bound_context_ir(context, context_ir_binding),
            )
        except (h3.H3Error, h3_project.ProjectMultimodalError) as exc:
            raise LongGenerationError(
                getattr(exc, "code", "long_video_multimodal_invalid")
            ) from None
    if plan.workflow in h3.H3_MULTIMODAL_WORKFLOWS:
        if segment.multimodal is None:
            raise LongGenerationError("long_video_multimodal_invalid")
        try:
            source_request = h3_project.build_request_from_parts(
                multimodal=segment.multimodal,
                visual_prompt=segment.multimodal.skill_plan.get("visual_prompt"),
                keyframes=segment.keyframes,
                upstream_dialogue=segment.dialogue,
                upstream_dialogue_receipt_sha256=(
                    segment.dialogue_sha256 or ""
                ),
                source_sha256=_digest(segment.workdir / "source.mp4"),
                source_duration_s=segment.end_s - segment.start_s,
                cid=f"{cid}-segment-{segment.index}",
                workdir=segment.workdir,
                client_request_id=(
                    frozen_child_id
                    or child_request_id(parent_id, plan.receipt, segment.index)
                ),
                duration=long_video.provider_duration_s(
                    segment.start_s,
                    segment.end_s,
                    receipt_version=plan.receipt_version,
                ),
                resolution=plan.resolution,
                aspect_ratio=plan.aspect_ratio,
                autodl_token=settings.autodl_art_token,
                timeouts=h3.Timeouts(
                    request_s=settings.h3_request_timeout_s,
                    h3_poll_s=settings.h3_poll_timeout_s,
                    download_s=settings.h3_download_timeout_s,
                    poll_interval_s=settings.h3_poll_interval_s,
                    retry_count=settings.retry_count,
                    retry_interval_s=settings.retry_interval_s,
                ),
            )
            if context_ir_binding is None:
                return source_request
            context = _freeze_segment_context_ir(
                settings, plan, segment, source_request
            )
            return _bind_h3_operational_roots(
                settings,
                plan,
                h3_project.apply_bound_context_ir(context, context_ir_binding),
            )
        except (h3.H3Error, h3_project.ProjectMultimodalError) as exc:
            raise LongGenerationError(
                getattr(exc, "code", "long_video_multimodal_invalid")
            ) from None
    if plan.workflow in h3.H3_REFERENCE_WORKFLOWS:
        return h3.H3Request(
            cid=f"{cid}-segment-{segment.index}",
            workdir=segment.workdir,
            client_request_id=(
                frozen_child_id
                or child_request_id(parent_id, plan.receipt, segment.index)
            ),
            prompt=segment.prompt,
            keyframes=segment.keyframes,
            voice_texts=(),
            voice_receipt=h3.voice_texts_receipt(()),
            duration=long_video.provider_duration_s(
                segment.start_s,
                segment.end_s,
                receipt_version=plan.receipt_version,
            ),
            autodl_token=settings.autodl_art_token,
            timeouts=h3.Timeouts(
                request_s=settings.h3_request_timeout_s,
                h3_poll_s=settings.h3_poll_timeout_s,
                download_s=settings.h3_download_timeout_s,
                poll_interval_s=settings.h3_poll_interval_s,
                retry_count=settings.retry_count,
                retry_interval_s=settings.retry_interval_s,
            ),
            mode="reference",
            aspect_ratio=plan.aspect_ratio,
            resolution=plan.resolution,
            workflow=plan.workflow,
        )
    first, first_data = segment.first_frame, segment.first_frame_data
    if segment.join_mode == "continue":
        upstream = plan.segments[segment.index - 2]
        if fast_mode:
            # The upstream end anchor is already receipt-bound and fitted by
            # freeze_plan.  Reuse those exact immutable bytes; no generated
            # output is read and no duplicate anchor file is created.
            first, first_data = upstream.last_frame, upstream.last_frame_data
        else:
            tail = upstream.workdir / "work" / "generated_last.png"
            # A paid start belongs to the current parent attempt and must refresh
            # the dependency; resume may only reuse the already-frozen tail.
            if prepare_inputs:
                tail = _extract_last_frame(upstream.workdir / "generated.mp4", tail)
            elif not tail.is_file():
                raise LongGenerationError("long_video_tail_frame_missing")
            try:
                tail_data = tail.read_bytes()
            except OSError:
                raise LongGenerationError("long_video_tail_frame_missing") from None
            continued = segment.workdir / "work" / "h3_frames"
            if not plan.legacy_layout:
                continued = continued / plan.aspect_ratio.replace(":", "x")
            continued = continued / fit_mode / "continued"
            first, first_data = _fit_anchor(
                tail, tail_data, continued, fit_mode, plan.aspect_ratio,
                prepare=prepare_inputs,
            )
    return h3.H3Request(
        cid=f"{cid}-segment-{segment.index}",
        workdir=segment.workdir,
        client_request_id=(
            frozen_child_id
            or child_request_id(parent_id, plan.receipt, segment.index)
        ),
        prompt=segment.prompt,
        keyframes=(),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        duration=long_video.provider_duration_s(
            segment.start_s,
            segment.end_s,
            receipt_version=plan.receipt_version,
        ),
        autodl_token=settings.autodl_art_token,
        timeouts=h3.Timeouts(
            request_s=settings.h3_request_timeout_s,
            h3_poll_s=settings.h3_poll_timeout_s,
            download_s=settings.h3_download_timeout_s,
            poll_interval_s=settings.h3_poll_interval_s,
            retry_count=settings.retry_count,
            retry_interval_s=settings.retry_interval_s,
        ),
        mode="boundary",
        first_frame=(first, first_data),
        last_frame=(segment.last_frame, segment.last_frame_data),
        aspect_ratio=plan.aspect_ratio,
        resolution=plan.resolution,
        workflow=h3.H3_BOUNDARY_WORKFLOW,
    )


def _revalidate_speaker_authority(
    plan: FrozenPlan, segment: FrozenSegment, request: h3.H3Request,
) -> None:
    if (
        plan.workflow not in h3.H3_MULTIMODAL_WORKFLOWS
        or plan.prompt_fusion is not None
    ):
        return
    try:
        h3_project.revalidate_production_authority(
            plan.root,
            segment.workdir / "work",
            request,
            expected_production_sha256=(
                hashlib.sha256(
                    segment.multimodal.speaker_timing_production_data
                ).hexdigest()
                if segment.multimodal is not None
                and segment.multimodal.speaker_timing_production_data is not None
                else None
            ),
        )
    except h3_project.ProjectMultimodalError as exc:
        raise LongGenerationError(exc.code) from None


def _freeze_segment_context_ir(
    settings,
    plan: FrozenPlan,
    segment: FrozenSegment,
    source_request: h3.H3Request,
) -> context_ir_bridge.FrozenContextIrRequest:
    if not getattr(settings, "minimax_api_key", ""):
        raise LongGenerationError("context_ir_credential_missing", 503)
    try:
        return h3_project.freeze_context_ir(
            source_request=source_request,
            upstream_dialogue_sha256=segment.dialogue_sha256 or "",
            upstream_artifact_path=(
                plan.root / long_video.PLAN_RECEIPT_FILENAME
            ),
            upstream_artifact_sha256=plan.receipt,
            upstream_dialogue_sha256_path=(
                "segments", segment.index - 1, "dialogue", "sha256"
            ),
            minimax_api_key=settings.minimax_api_key,
            request_timeout_s=settings.h3_request_timeout_s,
            poll_timeout_s=settings.h3_poll_timeout_s,
            poll_interval_s=settings.h3_poll_interval_s,
        )
    except h3_project.ProjectMultimodalError as exc:
        raise LongGenerationError(exc.code) from None


def _optimize_segment_context_ir(
    settings,
    plan: FrozenPlan,
    segment: FrozenSegment,
    source_request: h3.H3Request,
) -> tuple[h3.H3Request | None, dict[str, object], str, str | None]:
    context = _freeze_segment_context_ir(settings, plan, segment, source_request)
    try:
        result = context_ir_bridge.optimize_h3_prompt(context)
        binding = h3_project.context_ir_binding(result)
        if result.status == "succeeded":
            return (
                _bind_h3_operational_roots(
                    settings,
                    plan,
                    h3_project.apply_bound_context_ir(context, binding),
                ),
                binding,
                "succeeded",
                None,
            )
    except (context_ir_bridge.ContextIrError,
            h3_project.ProjectMultimodalError) as exc:
        raise LongGenerationError(getattr(exc, "code", "context_ir_invalid")) from None
    if result.status == "submission_unknown":
        return None, binding, "submission_unknown", "submission_unknown"
    if result.status in {"running", "query_unknown"}:
        return (
            None,
            binding,
            "resume_required",
            result.error_code or "context_ir_query_unknown",
        )
    return None, binding, "failed", result.error_code or "context_ir_failed"


def _context_ir_may_progress(
    state: Mapping,
    source_request: h3.H3Request,
    *,
    allow_create: bool = False,
) -> bool:
    binding = state.get("context_ir")
    if binding is None:
        return (
            allow_create
            and state.get("child_request_id") is None
            and state.get("h3_attempt_id") is None
        )
    if not isinstance(binding, Mapping):
        return False
    return h3_project.context_ir_progress_binding_matches(
        source_request, binding
    )


def public_segments(generation: Mapping) -> list[dict]:
    result = []
    for item in generation.get("segments", []):
        if isinstance(item, dict):
            result.append({key: item.get(key) for key in
                           ("index", "chain_id", "join_mode", "status", "attempt", "error")})
    return result


def _result_status(result: h3.H3Result) -> tuple[str, str | None]:
    if result.status == "succeeded":
        return "succeeded", None
    if result.status in {"submission_unknown", "h3_submitting"}:
        return "submission_unknown", "submission_unknown"
    if result.status == "h3_running" or result.error_code in {
        "h3_query_failed", "h3_timeout", "download_failed", "download_dns_failed",
        "download_peer_unverified", "output_write_failed", "output_probe_failed",
    }:
        return "resume_required", result.error_code or result.status
    return "failed", result.error_code or "h3_failed"


def _exact_h3_attempt_id(value: object, fallback: object = None) -> str | None:
    for candidate in (value, fallback):
        if (
            isinstance(candidate, str)
            and len(candidate) == 6
            and candidate.isdigit()
        ):
            return candidate
    return None


def _result_state(
    result: h3.H3Result,
    fallback_attempt_id: object = None,
) -> tuple[str, str | None, str | None]:
    status, error = _result_status(result)
    return status, error, _exact_h3_attempt_id(
        result.attempt_id, fallback_attempt_id
    )


def generation_segments_are_valid(
    expected_segments: object,
    generation: Mapping,
) -> bool:
    """Validate the complete ordered persisted coordinator state."""
    raw = generation.get("segments")
    if "fast_mode" in generation and not isinstance(generation.get("fast_mode"), bool):
        return False
    if not isinstance(expected_segments, (list, tuple)) or not isinstance(raw, list):
        return False
    if len(raw) != len(expected_segments) or not raw:
        return False
    legacy_keys = {
        "index", "chain_id", "join_mode", "status", "attempt", "error",
        "child_request_id",
    }
    native_keys = legacy_keys | {"h3_attempt_id", "context_ir"}
    context_keys = legacy_keys | {"context_ir"}
    audio_route = generation.get("audio_route")
    native_required = audio_route == H3_NATIVE_AUDIO_ROUTE
    if audio_route is None:
        allowed_keys = (legacy_keys, context_keys)
    elif native_required:
        allowed_keys = (native_keys, context_keys)
    else:
        return False
    statuses = {
        "not_started", "queued", "running", "resume_required", "succeeded",
        "failed", "submission_unknown",
    }
    native_items = 0
    for position, (expected, item) in enumerate(zip(expected_segments, raw), 1):
        if (
            not isinstance(item, dict)
            or set(item) not in allowed_keys
        ):
            return False
        if isinstance(expected, FrozenSegment):
            expected_index = expected.index
            expected_chain = expected.chain_id
            expected_join = expected.join_mode
        elif isinstance(expected, Mapping):
            expected_index = expected.get("index")
            expected_chain = expected.get("chain_id")
            expected_join = expected.get("join_mode")
        else:
            return False
        attempt = item.get("attempt")
        child_id = item.get("child_request_id")
        h3_attempt_id = item.get("h3_attempt_id")
        context_ir = item.get("context_ir")
        context_required = "context_ir" in item
        item_native = "h3_attempt_id" in item
        native_items += int(item_native)
        if (
            expected_index != position
            or item.get("index") != expected_index
            or item.get("chain_id") != expected_chain
            or item.get("join_mode") != expected_join
            or item.get("status") not in statuses
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 0
            or (
                item.get("error") is not None
                and not isinstance(item.get("error"), str)
            )
            or (
                child_id is not None
                and (not isinstance(child_id, str) or not child_id)
            )
            or (
                h3_attempt_id is not None
                and (
                    not isinstance(h3_attempt_id, str)
                    or len(h3_attempt_id) != 6
                    or not h3_attempt_id.isdigit()
                )
            )
            or (
                context_required
                and item.get("status") == "succeeded"
                and (
                    (item_native and h3_attempt_id is None)
                    or not isinstance(context_ir, Mapping)
                    or context_ir.get("status") != "succeeded"
                )
            )
            or (
                context_required
                and context_ir is not None
                and not isinstance(context_ir, Mapping)
            )
        ):
            return False
    return bool(native_items) == native_required


def bound_reusable_segment_indices(
    settings,
    cid: str,
    plan: FrozenPlan,
    generation: Mapping,
) -> frozenset[int]:
    """Single source of truth for paid-count and execution reuse decisions."""
    segments = generation.get("segments")
    expected = tuple(item.index for item in plan.segments)
    if not generation_segments_are_valid(plan.segments, generation):
        return frozenset()
    meta = storage.load_meta(settings.data_dir, cid)
    fit_mode = meta.get("fit_mode") if isinstance(meta, dict) else None
    if fit_mode not in {"none", "crop", "pad"}:
        return frozenset()
    fast_mode = generation.get("fast_mode", False)
    by_index = {
        item.get("index"): item for item in segments or [] if isinstance(item, dict)
    }
    reusable: set[int] = set()

    def valid(index: int) -> bool:
        item = by_index.get(index)
        segment = plan.segments[index - 1]
        if not isinstance(item, dict):
            return False
        attempt = item.get("attempt")
        child_id = item.get("child_request_id")
        status_can_revalidate = item.get("status") == "succeeded" or (
            item.get("status") == "failed"
            and item.get("error") == "long_video_segment_output_invalid"
        )
        if (
            not status_can_revalidate
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(child_id, str)
            or not child_id
            or (
                not fast_mode
                and
                segment.join_mode == "continue"
                and segment.index - 1 not in reusable
            )
        ):
            return False
        try:
            request = _request(
                settings, cid, plan, segment,
                str(generation.get("client_request_id") or "frozen-parent"),
                fit_mode,
                frozen_child_id=child_id,
                prepare_inputs=False,
                fast_mode=fast_mode,
                context_ir_binding=item.get("context_ir"),
            )
            if _segment_uses_h3_native_audio(plan, segment):
                inspected = h3.inspect(request)
                if (
                    inspected.status != "succeeded"
                    or inspected.attempt_id != item.get("h3_attempt_id")
                ):
                    return False
            return h3.output_is_reusable(
                request,
                expected_duration_s=_segment_duration_s(plan, segment),
                allow_provider_duration_ceiling=True,
            )
        except (OSError, h3.H3Error, LongGenerationError, ValueError):
            return False

    for index in expected:
        if valid(index):
            reusable.add(index)
    return frozenset(reusable)


def bound_h3_native_media(
    settings,
    cid: str,
    plan: FrozenPlan,
    generation: Mapping,
) -> dict[int, tuple[str, Mapping[str, object]]]:
    """Bind each native-audio segment to its exact successful H3 attempt."""
    if not _generation_uses_h3_native_audio(plan, generation):
        raise LongGenerationError("long_video_audio_route_invalid")
    if not generation_segments_are_valid(plan.segments, generation):
        raise LongGenerationError("long_video_h3_native_audio_invalid")
    meta = storage.load_meta(settings.data_dir, cid)
    fit_mode = meta.get("fit_mode") if isinstance(meta, Mapping) else None
    parent_id = generation.get("client_request_id")
    fast_mode = generation.get("fast_mode", False)
    is_multimodal_request = getattr(h3, "is_multimodal_request", None)
    if fit_mode not in {"none", "crop", "pad"} or not isinstance(parent_id, str):
        raise LongGenerationError("long_video_h3_native_audio_invalid")
    if not callable(is_multimodal_request):
        raise LongGenerationError("long_video_h3_native_audio_invalid")
    states = {
        item.get("index"): item
        for item in generation.get("segments", [])
        if isinstance(item, Mapping)
    }
    result: dict[int, tuple[str, Mapping[str, object]]] = {}
    for segment in plan.segments:
        state = states.get(segment.index)
        if not isinstance(state, Mapping) or state.get("status") != "succeeded":
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        if not _segment_uses_h3_native_audio(plan, segment):
            if state.get("h3_attempt_id") is not None:
                raise LongGenerationError("long_video_h3_native_audio_invalid")
            continue
        attempt_id = state.get("h3_attempt_id")
        child_id = state.get("child_request_id")
        if (
            _exact_h3_attempt_id(attempt_id) is None
            or not isinstance(child_id, str)
            or not child_id
        ):
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        try:
            request = _request(
                settings,
                cid,
                plan,
                segment,
                parent_id,
                fit_mode,
                frozen_child_id=child_id,
                prepare_inputs=False,
                fast_mode=fast_mode,
                context_ir_binding=state.get("context_ir"),
            )
            if is_multimodal_request(request) is not True:
                raise LongGenerationError("long_video_h3_native_audio_invalid")
            timeline = h3.load_media_timeline_receipt(request, attempt_id)
        except (OSError, TypeError, ValueError, h3.H3Error, LongGenerationError):
            raise LongGenerationError("long_video_h3_native_audio_invalid") from None
        if not isinstance(timeline.get("audio"), Mapping):
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        result[segment.index] = (attempt_id, timeline)
    return result


def initial_generation(settings, cid: str, plan: FrozenPlan, parent_id: str, attempt: int,
                       old: Mapping | None = None, *, fast_mode: bool = False) -> dict:
    if not isinstance(fast_mode, bool):
        raise LongGenerationError("invalid_fast_mode", 422)
    raw_old_segments = (old or {}).get("segments", [])
    old_segments = raw_old_segments if isinstance(raw_old_segments, list) else []
    reusable = bound_reusable_segment_indices(
        settings, cid, plan, old or {"segments": []}
    )
    old_by_index = {
        item.get("index"): item for item in old_segments
        if isinstance(item, dict)
    }
    items = []
    native_audio_indices = _native_audio_segment_indices(plan)
    context_ir_required = _requires_context_ir(plan)
    for segment in plan.segments:
        prior = old_by_index.get(segment.index, {})
        succeeded = segment.index in reusable
        item = {
            "index": segment.index,
            "chain_id": segment.chain_id,
            "join_mode": segment.join_mode,
            "status": "succeeded" if succeeded else "not_started",
            "attempt": prior.get("attempt", 0) if succeeded else int(prior.get("attempt", 0) or 0),
            "error": None,
            "child_request_id": prior.get("child_request_id") if succeeded else None,
        }
        if segment.index in native_audio_indices:
            item["h3_attempt_id"] = (
                prior.get("h3_attempt_id") if succeeded else None
            )
        if context_ir_required:
            item["context_ir"] = (
                prior.get("context_ir") if succeeded else None
            )
        items.append(item)
    generation = {
        "status": "queued",
        "error": None,
        "attempt": attempt,
        "client_request_id": parent_id,
        "stage": "h3",
        "fit_layout": (
            FIT_LAYOUT_LEGACY if plan.legacy_layout else FIT_LAYOUT_ASPECT
        ),
        "fast_mode": fast_mode,
        "workflow": plan.workflow,
        "segments": items,
    }
    if native_audio_indices:
        generation["audio_route"] = dict(H3_NATIVE_AUDIO_ROUTE)
    return generation


def _stitch(
    settings,
    cid: str,
    plan: FrozenPlan,
    dialogue_mode: str,
    *,
    generation: Mapping | None = None,
) -> None:
    provider_media = None
    if _is_h3_multimodal_plan(plan):
        if generation is None:
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        provider_media = bound_h3_native_media(settings, cid, plan, generation)
        audio_mode: stitch.AudioMode = "provider_generated"
    else:
        if generation is not None:
            _generation_uses_h3_native_audio(plan, generation)
        if dialogue_mode not in {"auto", "none"}:
            raise LongGenerationError("invalid_dialogue_mode", 422)
        audio_mode = "keep" if dialogue_mode == "auto" else "mute"
    stitch.stitch_video(
        segments=_stitch_segments(plan, provider_media),
        source_video=plan.source,
        output=plan.root / "generated.mp4",
        audio_mode=audio_mode,
    )
    reusable = (
        stitched_output_is_reusable(
            plan,
            dialogue_mode,
            generation=generation,
            provider_media=provider_media,
        )
        if provider_media is not None
        else stitched_output_is_reusable(plan, dialogue_mode)
    )
    if not reusable:
        raise LongGenerationError("long_video_stitch_output_invalid")


def stitched_output_is_reusable(
    plan: FrozenPlan,
    dialogue_mode: str,
    *,
    generation: Mapping | None = None,
    provider_media: Mapping[int, tuple[str, Mapping[str, object]]] | None = None,
) -> bool:
    """Validate legacy output or native output with exact attempt evidence."""
    if dialogue_mode not in {"auto", "none"}:
        return False
    output = plan.root / "generated.mp4"
    receipt_path = plan.root / stitch.RECEIPT_FILENAME
    native_audio = _is_h3_multimodal_plan(plan)
    if native_audio:
        native_indices = _native_audio_segment_indices(plan)
        if (
            not isinstance(generation, Mapping)
            or not generation
            or not isinstance(provider_media, Mapping)
            or not provider_media
            or not generation_segments_are_valid(plan.segments, generation)
        ):
            return False
        try:
            if not _generation_uses_h3_native_audio(plan, generation):
                return False
        except LongGenerationError:
            return False
        states = {
            item.get("index"): item
            for item in generation.get("segments", [])
            if isinstance(item, Mapping)
        }
        if set(provider_media) != set(native_indices):
            return False
        for segment in plan.segments:
            state = states.get(segment.index)
            media = provider_media.get(segment.index)
            if not isinstance(state, Mapping) or state.get("status") != "succeeded":
                return False
            if segment.index not in native_indices:
                if media is not None or state.get("h3_attempt_id") is not None:
                    return False
                continue
            if (
                not isinstance(media, tuple)
                or len(media) != 2
                or media[0] != state.get("h3_attempt_id")
                or not isinstance(media[1], Mapping)
            ):
                return False
    else:
        if provider_media is not None:
            return False
        if generation is not None:
            try:
                if _generation_uses_h3_native_audio(plan, generation):
                    return False
            except LongGenerationError:
                return False
    audio_mode: stitch.AudioMode = (
        "provider_generated"
        if native_audio
        else ("keep" if dialogue_mode == "auto" else "mute")
    )
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "version", "segments", "audio", "output"}
            or payload.get("schema") != "duet.stitch"
            or payload.get("version") != (2 if native_audio else 1)
        ):
            return False
        stitch_segments = _stitch_segments(plan, provider_media)
        budgets = stitch._frame_budgets(stitch_segments)
        expected_segments = [
            {
                "index": index,
                "path": str((item.workdir / "generated.mp4").resolve()),
                "sha256": stitch._sha256(item.workdir / "generated.mp4"),
                "target_duration_s": stitch_segments[index - 1].target_duration_s,
                "output_frames": budgets[index - 1],
                "join_mode": item.join_mode,
            }
            for index, item in enumerate(plan.segments, 1)
        ]
        if payload.get("segments") != expected_segments:
            return False
        source_info = stitch._probe(plan.source)
        expected_audio = {
            "mode": audio_mode,
            "source": str(plan.source.resolve()),
            "source_sha256": stitch._sha256(plan.source),
            "source_has_audio": source_info.has_audio,
        }
        if native_audio:
            actual_audio = payload.get("audio")
            if not isinstance(actual_audio, dict):
                return False
            expected_audio["provider_segments"] = [
                stitch._segment_audio_binding(segment, index)
                for index, segment in enumerate(stitch_segments, 1)
            ]
            expected_audio["edl"] = {
                "schema": "duet.av-edl",
                "version": 1,
                "fps": stitch.FPS,
                "interval": "integer-half-open",
            }
        if payload.get("audio") != expected_audio:
            return False
        output_receipt = payload.get("output")
        stat = output.stat()
        if (
            not output.is_file()
            or stat.st_size <= 0
            or not isinstance(output_receipt, dict)
            or set(output_receipt)
            != {"name", "sha256", "size", "duration_s", "fps"}
            or output_receipt.get("name") != "generated.mp4"
            or output_receipt.get("size") != stat.st_size
            or output_receipt.get("sha256") != stitch._sha256(output)
            or output_receipt.get("fps") != stitch.FPS
        ):
            return False
        expected_duration = sum(
            item.target_duration_s for item in stitch_segments
        )
        output_info = stitch._validate_output(
            output, expected_duration, audio_mode, source_info.has_audio
        )
        receipt_duration = float(output_receipt.get("duration_s"))
        return (
            math.isfinite(receipt_duration)
            and abs(receipt_duration - output_info.duration_s) <= _EPS
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        LongGenerationError,
        stitch.StitchError,
    ):
        return False


def run(settings, cid: str, plan: FrozenPlan, *, startup: bool = False) -> None:
    """Drive frozen serial chains or fast fan-out; only this coordinator writes meta."""
    meta = storage.load_meta(settings.data_dir, cid)
    if not meta or not isinstance(meta.get("generation"), dict):
        return
    generation = meta["generation"]
    if not generation_segments_are_valid(plan.segments, generation):
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    try:
        _generation_uses_h3_native_audio(plan, generation)
    except LongGenerationError:
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    context_ir_required = _requires_context_ir(plan)
    parent_id = generation.get("client_request_id")
    fast_mode = generation.get("fast_mode", False)
    fit_mode = meta.get("fit_mode")
    dialogue_mode = meta.get("dialogue_mode")
    states = {item["index"]: dict(item) for item in generation.get("segments", [])}
    native_audio_indices = _native_audio_segment_indices(plan)
    if any(
        ("context_ir" in state) != context_ir_required
        for state in states.values()
    ) or any(
        ("h3_attempt_id" in state) != (index in native_audio_indices)
        for index, state in states.items()
    ):
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    if (
        not isinstance(parent_id, str)
        or fit_mode not in {"none", "crop", "pad"}
        or meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
        != plan.aspect_ratio
        or meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
        != plan.resolution
    ):
        return

    def persist(status: str | None = None, error: str | None = None, stage: str = "h3") -> None:
        nonlocal generation
        ordered = [states[item.index] for item in plan.segments]
        generation = {**generation, "segments": ordered,
                      "status": status or generation.get("status"), "error": error, "stage": stage}
        storage.update_meta(settings.data_dir, cid, generation=generation)

    def exact_generation() -> dict:
        return {
            **generation,
            "segments": [states[item.index] for item in plan.segments],
        }

    def parallel_update(segments, operation) -> None:
        with ThreadPoolExecutor(
            max_workers=min(_FAST_MODE_WORKERS, len(segments))
        ) as pool:
            futures = {pool.submit(operation, segment): segment for segment in segments}
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    segment = futures.pop(future)
                    status, error, attempt_id = future.result()
                    changes = {"status": status, "error": error}
                    if segment.index in native_audio_indices:
                        changes["h3_attempt_id"] = attempt_id
                    states[segment.index].update(changes)
                    persist("running", None)

    reusable = bound_reusable_segment_indices(settings, cid, plan, generation)
    for index, state in states.items():
        if (
            index in reusable
            and state.get("status") == "failed"
            and state.get("error") == "long_video_segment_output_invalid"
        ):
            state.update(status="succeeded", error=None)
        elif state.get("status") == "succeeded" and index not in reusable:
            state.update(
                status="failed",
                error="long_video_segment_output_invalid",
            )

    if startup:
        recoverable = []
        recovered_provider_failure = False
        for segment in plan.segments:
            state = states[segment.index]
            provider_failed = (
                state.get("status") == "failed"
                and state.get("error") == "h3_provider_failed"
            )
            recovered_provider_failure = (
                recovered_provider_failure or provider_failed
            )
            if (
                state.get("status") not in {"queued", "running", "resume_required"}
                and not provider_failed
            ):
                continue
            recoverable.append(segment)

        def recover(segment: FrozenSegment):
            state = states[segment.index]
            try:
                context_binding = state.get("context_ir")
                request = _request(
                    settings, cid, plan, segment, parent_id, fit_mode,
                    prepare_inputs=False,
                    fast_mode=fast_mode,
                    frozen_child_id=state.get("child_request_id"),
                    context_ir_binding=(
                        context_binding
                        if isinstance(context_binding, Mapping)
                        and context_binding.get("status") == "succeeded"
                        else None
                    ),
                )
                if context_ir_required and not (
                    isinstance(context_binding, Mapping)
                    and context_binding.get("status") == "succeeded"
                ):
                    if not isinstance(context_binding, Mapping):
                        if (
                            state.get("child_request_id") is not None
                            or state.get("h3_attempt_id") is not None
                        ):
                            return (
                                "submission_unknown",
                                "submission_unknown",
                                _exact_h3_attempt_id(state.get("h3_attempt_id")),
                            )
                        return (
                            "resume_required",
                            "context_ir_resume_required",
                            _exact_h3_attempt_id(state.get("h3_attempt_id")),
                        )
                    if not _context_ir_may_progress(state, request):
                        return (
                            "submission_unknown",
                            "submission_unknown",
                            _exact_h3_attempt_id(state.get("h3_attempt_id")),
                        )
                    final_request, updated_binding, status, error = (
                        _optimize_segment_context_ir(
                            settings, plan, segment, request
                        )
                    )
                    state["context_ir"] = updated_binding
                    if status == "succeeded" and final_request is not None:
                        return (
                            "resume_required",
                            "context_ir_ready",
                            _exact_h3_attempt_id(state.get("h3_attempt_id")),
                        )
                    return (
                        status,
                        error,
                        _exact_h3_attempt_id(state.get("h3_attempt_id")),
                    )
                _revalidate_speaker_authority(plan, segment, request)
                result = h3.resume(request)
                if result.status == "not_started":
                    return (
                        (
                            "queued",
                            None,
                            _exact_h3_attempt_id(
                                result.attempt_id, state.get("h3_attempt_id")
                            ),
                        )
                        if (
                            fast_mode
                            and state.get("status") == "queued"
                            and isinstance(result.attempt_id, str)
                            and bool(result.attempt_id)
                        )
                        else (
                            "submission_unknown",
                            "submission_unknown",
                            _exact_h3_attempt_id(
                                result.attempt_id, state.get("h3_attempt_id")
                            ),
                        )
                    )
                status = _result_state(result, state.get("h3_attempt_id"))
                if (
                    fast_mode
                    and status[0] == "succeeded"
                    and not h3.output_is_reusable(
                        request,
                        expected_duration_s=_segment_duration_s(plan, segment),
                        allow_provider_duration_ceiling=True,
                    )
                ):
                    return (
                        "failed",
                        "long_video_segment_output_invalid",
                        status[2],
                    )
                return status
            except Exception:
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(state.get("h3_attempt_id")),
                )

        if recoverable and fast_mode:
            parallel_update(recoverable, recover)
        elif recoverable:
            for segment in recoverable:
                status, error, attempt_id = recover(segment)
                changes = {"status": status, "error": error}
                if segment.index in native_audio_indices:
                    changes["h3_attempt_id"] = attempt_id
                states[segment.index].update(changes)
        if any(item.get("status") == "submission_unknown" for item in states.values()):
            persist("submission_unknown", "submission_unknown")
            return
        elif all(item.get("status") == "succeeded" for item in states.values()):
            try:
                _stitch(
                    settings,
                    cid,
                    plan,
                    dialogue_mode,
                    generation=exact_generation(),
                )
            except LongGenerationError as exc:
                persist("failed", exc.code, "stitch")
            except Exception:
                persist("failed", "long_video_stitch_failed", "stitch")
            else:
                persist("succeeded", None, "stitch")
            return
        elif any(item.get("status") == "failed" for item in states.values()):
            persist("failed", "long_video_segment_failed")
            return
        if fast_mode or not recovered_provider_failure:
            # General startup remains GET-only. A prepared child still awaits
            # explicit confirmation unless this is the narrow serial provider-
            # failure continuation handled below.
            persist("resume_required", "long_video_resume_required")
            return
        # The serial chain was already authorized and stopped only at a
        # provider-declared, non-billable failure. Once that exact attempt is
        # recovered, continue its already-frozen downstream without a click.
        persist("running", None)

    if fast_mode:
        # Phase 1: construct every immutable request before creating any paid
        # attempt. A local validation failure therefore guarantees zero POSTs.
        requests: dict[int, h3.H3Request] = {}
        try:
            for segment in plan.segments:
                state = states[segment.index]
                if state.get("status") == "succeeded":
                    continue
                if state.get("status") not in {
                    "not_started", "queued", "running", "resume_required",
                }:
                    continue
                child_id = state.get("child_request_id")
                if not isinstance(child_id, str) or not child_id:
                    child_id = child_request_id(parent_id, plan.receipt, segment.index)
                requests[segment.index] = _request(
                    settings, cid, plan, segment, parent_id, fit_mode,
                    frozen_child_id=child_id,
                    prepare_inputs=False,
                    fast_mode=True,
                )

            if context_ir_required:
                context_targets = [
                    segment for segment in plan.segments
                    if segment.index in requests
                ]

                def prepare_context(segment: FrozenSegment):
                    state = states[segment.index]
                    binding = state.get("context_ir")
                    source_request = requests[segment.index]
                    if (
                        isinstance(binding, Mapping)
                        and binding.get("status") == "succeeded"
                    ):
                        context = _freeze_segment_context_ir(
                            settings, plan, segment, source_request
                        )
                        return (
                            _bind_h3_operational_roots(
                                settings,
                                plan,
                                h3_project.apply_bound_context_ir(context, binding),
                            ),
                            binding,
                            "succeeded",
                            None,
                        )
                    if not _context_ir_may_progress(
                        state,
                        source_request,
                        allow_create=state.get("status") == "not_started",
                    ):
                        return (
                            None,
                            binding,
                            "submission_unknown",
                            "submission_unknown",
                        )
                    return _optimize_segment_context_ir(
                        settings, plan, segment, source_request
                    )

                with ThreadPoolExecutor(
                    max_workers=min(_FAST_MODE_WORKERS, len(context_targets))
                ) as pool:
                    futures = {
                        pool.submit(prepare_context, segment): segment
                        for segment in context_targets
                    }
                    for future in futures:
                        segment = futures[future]
                        final_request, binding, status, error = future.result()
                        states[segment.index]["context_ir"] = binding
                        if status == "succeeded" and final_request is not None:
                            requests[segment.index] = final_request
                        else:
                            states[segment.index].update(
                                status=status, error=error
                            )
                persist("running", None, "context_ir_native")
                context_statuses = {
                    states[segment.index].get("status")
                    for segment in context_targets
                    if states[segment.index].get("context_ir", {}).get("status")
                    != "succeeded"
                }
                if "submission_unknown" in context_statuses:
                    persist("submission_unknown", "submission_unknown", "context_ir_native")
                    return
                if "resume_required" in context_statuses:
                    persist("resume_required", "context_ir_resume_required", "context_ir_native")
                    return
                if context_statuses:
                    persist("failed", "long_video_context_ir_failed", "context_ir_native")
                    return

            # Phase 2: persist every unpaid child receipt before the first POST.
            for segment in plan.segments:
                state = states[segment.index]
                if state.get("status") != "not_started":
                    continue
                _revalidate_speaker_authority(
                    plan, segment, requests[segment.index]
                )
                result = h3.prepare(requests[segment.index])
                if result.status == "not_started":
                    prepared_status, prepared_error = "queued", None
                elif result.status == "h3_running":
                    prepared_status, prepared_error = "running", None
                elif result.status == "succeeded":
                    if not h3.output_is_reusable(
                        requests[segment.index],
                        expected_duration_s=_segment_duration_s(plan, segment),
                        allow_provider_duration_ceiling=True,
                    ):
                        prepared_status, prepared_error = (
                            "failed", "long_video_segment_output_invalid"
                        )
                    else:
                        prepared_status, prepared_error = "succeeded", None
                else:
                    prepared_status, prepared_error = _result_status(result)
                changes = dict(
                    status=prepared_status,
                    attempt=int(state.get("attempt", 0) or 0) + 1,
                    error=prepared_error,
                    child_request_id=requests[segment.index].client_request_id,
                )
                if segment.index in native_audio_indices:
                    changes["h3_attempt_id"] = _exact_h3_attempt_id(
                        result.attempt_id, state.get("h3_attempt_id")
                    )
                state.update(changes)
            persist("running", None)
        except (h3.H3Error, LongGenerationError, OSError, ValueError) as exc:
            code = exc.code if isinstance(exc, (h3.H3Error, LongGenerationError)) else "long_video_request_invalid"
            failed_index = next(
                (
                    segment.index for segment in plan.segments
                    if states[segment.index].get("status") == "not_started"
                ),
                None,
            )
            if failed_index is not None:
                states[failed_index].update(status="failed", error=code)
            persist("failed", "long_video_segment_failed")
            return

        prepared_statuses = {item.get("status") for item in states.values()}
        if "submission_unknown" in prepared_statuses:
            persist("submission_unknown", "submission_unknown")
            return
        if "failed" in prepared_statuses:
            persist("failed", "long_video_segment_failed")
            return

        def submit_one(segment: FrozenSegment):
            previous_attempt = states[segment.index].get("h3_attempt_id")
            try:
                _revalidate_speaker_authority(
                    plan, segment, requests[segment.index]
                )
                result = h3.submit(requests[segment.index])
                if result.status == "h3_running":
                    return (
                        "running",
                        None,
                        _exact_h3_attempt_id(result.attempt_id, previous_attempt),
                    )
                status = _result_state(result, previous_attempt)
                if status[0] == "succeeded" and not h3.output_is_reusable(
                    requests[segment.index],
                    expected_duration_s=_segment_duration_s(plan, segment),
                    allow_provider_duration_ceiling=True,
                ):
                    return (
                        "failed",
                        "long_video_segment_output_invalid",
                        status[2],
                    )
                return status
            except h3.H3Error as exc:
                if exc.code == "attempt_not_prepared":
                    return (
                        "submission_unknown",
                        "submission_unknown",
                        _exact_h3_attempt_id(previous_attempt),
                    )
                try:
                    inspected = h3.inspect(requests[segment.index])
                    status = _result_state(inspected, previous_attempt)
                except Exception:
                    status = (
                        "submission_unknown",
                        "submission_unknown",
                        _exact_h3_attempt_id(previous_attempt),
                    )
                if status[0] == "failed" and exc.code in {
                    "submission_unknown", "state_persist_failed", "h3_internal_error",
                }:
                    return "submission_unknown", "submission_unknown", status[2]
                return status
            except Exception:
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )

        # Phase 3: fan out only the short POST boundary. No worker waits for a
        # provider result, so every queued child is submitted before polling.
        to_submit = [
            segment for segment in plan.segments
            if states[segment.index].get("status") == "queued"
        ]
        if to_submit:
            parallel_update(to_submit, submit_one)

        def poll_one(segment: FrozenSegment):
            request = requests[segment.index]
            previous_attempt = states[segment.index].get("h3_attempt_id")
            try:
                _revalidate_speaker_authority(plan, segment, request)
                result = h3.resume(request)
                if result.status == "not_started":
                    return (
                        "submission_unknown",
                        "submission_unknown",
                        _exact_h3_attempt_id(result.attempt_id, previous_attempt),
                    )
                status = _result_state(result, previous_attempt)
                if status[0] == "succeeded" and not h3.output_is_reusable(
                    request,
                    expected_duration_s=_segment_duration_s(plan, segment),
                    allow_provider_duration_ceiling=True,
                ):
                    return (
                        "failed",
                        "long_video_segment_output_invalid",
                        status[2],
                    )
                return status
            except h3.H3Error as exc:
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                ) if exc.code in {
                    "submission_unknown", "state_persist_failed", "h3_internal_error",
                } else (
                    "resume_required",
                    exc.code,
                    _exact_h3_attempt_id(previous_attempt),
                )
            except Exception:
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )

        # Phase 4: bounded long-lived GET polling. Unknown children never get a
        # second POST, while known siblings are still allowed to finish.
        to_poll = [
            segment for segment in plan.segments
            if states[segment.index].get("status") in {"running", "resume_required"}
        ]
        if to_poll:
            parallel_update(to_poll, poll_one)

        statuses = {item.get("status") for item in states.values()}
        if statuses == {"succeeded"}:
            try:
                _stitch(
                    settings,
                    cid,
                    plan,
                    dialogue_mode,
                    generation=exact_generation(),
                )
            except LongGenerationError as exc:
                persist("failed", exc.code, "stitch")
            except Exception:
                persist("failed", "long_video_stitch_failed", "stitch")
            else:
                persist("succeeded", None, "stitch")
        elif "submission_unknown" in statuses:
            persist("submission_unknown", "submission_unknown")
        elif "resume_required" in statuses or "running" in statuses:
            persist("resume_required", "long_video_resume_required")
        else:
            persist("failed", "long_video_segment_failed")
        return

    chains: dict[str, list[FrozenSegment]] = {}
    for segment in plan.segments:
        chains.setdefault(segment.chain_id, []).append(segment)

    attempted_indices: set[int] = set()

    def ready() -> list[FrozenSegment]:
        candidates = []
        for chain in chains.values():
            for segment in chain:
                state = states[segment.index]
                if state.get("status") == "succeeded":
                    continue
                if state.get("status") in {"not_started", "queued", "resume_required"}:
                    if segment.index in attempted_indices:
                        break
                    prior = [states[item.index].get("status") for item in chain if item.index < segment.index]
                    if all(value == "succeeded" for value in prior):
                        candidates.append(segment)
                    break
                break
        return candidates

    def worker(segment: FrozenSegment, action: str):
        if (
            action == "start"
            and plan.receipt_version == long_video.LEGACY_PLAN_RECEIPT_VERSION
            and long_video.provider_duration_s(
                segment.start_s,
                segment.end_s,
                receipt_version=plan.receipt_version,
            )
            > long_video.PREVIOUS_SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            return None, ("failed", "long_video_legacy_plan_read_only", None)
        existing_child_id = states[segment.index].get("child_request_id")
        previous_attempt = states[segment.index].get("h3_attempt_id")
        try:
            request = _request(
                settings, cid, plan, segment, parent_id, fit_mode,
                prepare_inputs=action == "start",
            )
            h3_action = action
            if context_ir_required:
                context_binding = states[segment.index].get("context_ir")
                if (
                    isinstance(context_binding, Mapping)
                    and context_binding.get("status") == "succeeded"
                ):
                    context = _freeze_segment_context_ir(
                        settings, plan, segment, request
                    )
                    request = _bind_h3_operational_roots(
                        settings,
                        plan,
                        h3_project.apply_bound_context_ir(
                            context, context_binding
                        ),
                    )
                else:
                    if not _context_ir_may_progress(
                        states[segment.index],
                        request,
                        allow_create=action == "start",
                    ):
                        return (
                            states[segment.index].get("child_request_id"),
                            (
                                "submission_unknown",
                                "submission_unknown",
                                _exact_h3_attempt_id(previous_attempt),
                            ),
                        )
                    request, context_binding, status, error = (
                        _optimize_segment_context_ir(
                            settings, plan, segment, request
                        )
                    )
                    states[segment.index]["context_ir"] = context_binding
                    if status != "succeeded" or request is None:
                        return (
                            states[segment.index].get("child_request_id"),
                            (
                                status,
                                error,
                                _exact_h3_attempt_id(previous_attempt),
                            ),
                        )
                    if action == "resume":
                        h3_action = "start"
        except LongGenerationError as exc:
            if action == "resume":
                return existing_child_id, (
                    "submission_unknown", "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )
            return None, (
                "failed", exc.code, _exact_h3_attempt_id(previous_attempt)
            )
        except Exception:
            if action == "resume":
                return existing_child_id, (
                    "submission_unknown", "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )
            return None, (
                "failed", "long_video_request_invalid",
                _exact_h3_attempt_id(previous_attempt),
            )
        try:
            _revalidate_speaker_authority(plan, segment, request)
            result = h3.start(request) if h3_action == "start" else h3.resume(request)
            if h3_action == "resume" and result.status == "not_started":
                return request.client_request_id, (
                    "submission_unknown", "submission_unknown",
                    _exact_h3_attempt_id(result.attempt_id, previous_attempt),
                )
            status = _result_state(result, previous_attempt)
            if status[0] == "succeeded" and not h3.output_is_reusable(
                request,
                expected_duration_s=_segment_duration_s(plan, segment),
                allow_provider_duration_ceiling=True,
            ):
                status = (
                    "failed",
                    "long_video_segment_output_invalid",
                    status[2],
                )
            return request.client_request_id, status
        except h3.H3Error as exc:
            try:
                inspected = h3.inspect(request)
                status = _result_state(inspected, previous_attempt)
            except Exception:
                status = (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )
            if status[0] == "failed" and exc.code in {"submission_unknown", "state_persist_failed", "h3_internal_error"}:
                status = (
                    "submission_unknown", "submission_unknown", status[2]
                )
            return request.client_request_id, status
        except Exception:
            return request.client_request_id, (
                "submission_unknown",
                "submission_unknown",
                _exact_h3_attempt_id(previous_attempt),
            )

    active = {}
    locked = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            if not locked:
                active_chains = {segment.chain_id for segment in active.values()}
                for segment in ready():
                    if len(active) >= 2:
                        break
                    if segment.chain_id in active_chains:
                        continue
                    state = states[segment.index]
                    is_new_child = state.get("status") == "not_started"
                    action = "start" if is_new_child else "resume"
                    state["status"], state["error"] = "running", None
                    if is_new_child:
                        state["attempt"] = int(state.get("attempt", 0) or 0) + 1
                    attempted_indices.add(segment.index)
                    persist("running", None)
                    future = pool.submit(worker, segment, action)
                    active[future] = segment
                    active_chains.add(segment.chain_id)
            if not active:
                break
            done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                segment = active.pop(future)
                child_id, (status, error, attempt_id) = future.result()
                changes = {
                    "status": status,
                    "error": error,
                    "child_request_id": child_id,
                }
                if segment.index in native_audio_indices:
                    changes["h3_attempt_id"] = attempt_id
                states[segment.index].update(changes)
                if status == "submission_unknown":
                    locked = True
                persist("submission_unknown" if locked else "running",
                        "submission_unknown" if locked else None)

    if locked:
        persist("submission_unknown", "submission_unknown")
        return
    statuses = {item.get("status") for item in states.values()}
    if statuses == {"succeeded"}:
        try:
            _stitch(
                settings,
                cid,
                plan,
                dialogue_mode,
                generation=exact_generation(),
            )
        except LongGenerationError as exc:
            persist("failed", exc.code, "stitch")
        except Exception:
            persist("failed", "long_video_stitch_failed", "stitch")
        else:
            persist("succeeded", None, "stitch")
    elif "resume_required" in statuses:
        persist("resume_required", "long_video_resume_required")
    else:
        persist("failed", "long_video_segment_failed")
