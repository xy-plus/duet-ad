from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from app import media
from app.config import Settings
from app.fusion_schema import fusion_prompt_schema, normalize_fusion_prompt
from app.object_replace_schema import (
    MAX_REFERENCE_AUDIO,
    OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index,
    normalize_object_replace,
    object_replace_schema,
)
from app.providers import (
    CodexClient,
    DeepSeekClient,
    MediaKitClient,
    PicsartClient,
    ProviderFailure,
    ProviderResult,
)
from app.research_log import ResearchLog
from app.store import ProjectStore
from app.voice_catalog import VoiceLibrary, load_voice_library
from app.video_maker_schemas import (
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
    normalize_visual_prompt,
    project_index_schema,
)


VIDEO_SKILL = Path(__file__).resolve().parents[1] / "skills" / "video-maker" / "SKILL.md"
R2V_SKILL = Path(__file__).resolve().parents[1] / "skills" / "h3-max-r2v-prompt" / "SKILL.md"
OBJECT_REPLACE_SKILL = (
    Path(__file__).resolve().parents[1] / "skills" / "object-replace" / "SKILL.md"
)
SAME_AS_SOURCE_LANGUAGE = "与原视频相同"
MASTER_EMPTY_ASR_SIGNAL = 51 / 256
TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines"],
    "additionalProperties": False,
}

PICSART_RECEIPT_VERSION = 1
H3_PREPARED_ARTIFACTS_VERSION = 2
PICSART_PROMPT_MAX_CHARS = 50_000
PICSART_REFERENCE_MIN_DURATION_S = 2
PICSART_REFERENCE_MAX_DURATION_S = 15
PICSART_GENERATION_MIN_DURATION_S = 5
PICSART_GENERATION_MAX_DURATION_S = 15
PICSART_RESOLUTIONS = frozenset({"480p", "768p"})
PICSART_ASPECT_RATIOS = frozenset({"16:9", "9:16"})


@dataclass(frozen=True, slots=True)
class PreparedPicsartRequest:
    """Immutable boundary between local preparation and paid provider work."""

    project_id: str
    segment_index: int
    scope: str
    run_identity: str
    artifact_suffix: str
    final_prompt: str
    muted: Path
    provider_replacement: Path | None
    voice_references: tuple[Path, ...]
    video_url: str | None
    image_url: str | None
    audio_urls: tuple[str, ...]
    provider_duration: int
    resolution: str
    aspect_ratio: str
    target_duration_s: float
    request_sha256: str
    existing_task_id: str | None = None

    def submit_input(self) -> dict[str, Any]:
        if self.existing_task_id is not None:
            raise ValueError("an existing Picsart task has no submit input")
        if self.video_url is None:
            raise ValueError("prepared Picsart video URL is missing")
        return {
            "workflow": PicsartClient.WORKFLOW,
            "prompt": self.final_prompt,
            "reference_video_urls": [self.video_url],
            "reference_image_urls": (
                [self.image_url] if self.image_url is not None else []
            ),
            "reference_audio_urls": list(self.audio_urls),
            "resolution": self.resolution.upper(),
            "duration": self.provider_duration,
            "aspect_ratio": self.aspect_ratio,
            "prompt_expansion_mode": "quality",
        }


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        store: ProjectStore,
        register_asset: Callable[[str, Path], str],
        *,
        codex: CodexClient | None = None,
        deepseek: DeepSeekClient | None = None,
        mediakit: MediaKitClient | None = None,
        picsart: PicsartClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.register_asset = register_asset
        self.codex = codex or CodexClient(settings)
        self.deepseek = deepseek or DeepSeekClient(settings)
        self.mediakit = mediakit or MediaKitClient(settings)
        self.picsart = picsart or PicsartClient(settings)
        self.voice_library: VoiceLibrary = load_voice_library()
        self._mediakit_gate = threading.BoundedSemaphore(
            settings.mediakit_concurrency,
        )
        # ProjectStore.update() is atomic, but a receipt ledger change is a
        # read-modify-write operation assembled here.  Serialize that whole
        # operation so parallel segment transitions cannot overwrite each
        # other before the paid POST boundary.
        self._picsart_receipt_lock = threading.Lock()

    def _erase_with_limit(
        self,
        source: Path,
        destination: Path,
        stages: list[str],
    ) -> ProviderResult[Path]:
        with self._mediakit_gate:
            return self.mediakit.erase(source, destination, stages)

    def _progress(
        self,
        project_id: str,
        percent: int,
        stage: str,
        *,
        segments: list[dict[str, Any]] | None = None,
    ) -> None:
        changes: dict[str, Any] = {
            "status": "running",
            "project_progress": {
                "status": "running",
                "percent": percent,
                "stage": stage,
            },
            "generation": {"status": "running", "stage": stage, "error": None},
        }
        if segments is not None:
            changes["segments"] = segments
        self.store.update(project_id, **changes)

    def _node(
        self,
        log: ResearchLog,
        node: str,
        scope: str,
        inputs: Any,
        input_media: list[Path],
        operation: Callable[[], Any],
        output_media: Callable[[Any], list[Path]] | None = None,
        *,
        transform: Callable[[Any], Any] | None = None,
        retry_if: Callable[[Exception], bool] | None = None,
    ) -> Any:
        """Run one node at most twice and keep transport evidence per attempt."""
        last_error: Exception | None = None
        for attempt in (1, 2):
            directory = log.attempt_dir(node, scope, attempt)
            log.write_json(directory / "input.json", inputs)
            log.copy_media(directory / "input-media", input_media)
            log.event("node_started", node=node, scope=scope, attempt=attempt)
            provider_record: dict[str, Any] | None = None
            provider_media: list[Path] = []
            try:
                raw = operation()
                if isinstance(raw, ProviderResult):
                    result = raw.value
                    provider_record = raw.record
                    provider_media = list(raw.evidence_media)
                else:
                    result = raw
                if transform is not None:
                    result = transform(result)
                logged_output: Any = result
                if provider_record is not None:
                    logged_output = {
                        "value": result,
                        "provider_record": provider_record,
                    }
                log.write_json(directory / "output.json", logged_output)
                media_paths = list(provider_media)
                if output_media is not None:
                    try:
                        media_paths.extend(output_media(result))
                    except Exception as exc:
                        log.event(
                            "research_log_media_selection_failed",
                            node=node,
                            scope=scope,
                            attempt=attempt,
                            error=str(exc),
                        )
                log.copy_media(directory / "output-media", media_paths)
                event = (
                    "node_skipped"
                    if provider_record is not None
                    and provider_record.get("outcome") == "skipped"
                    else "node_succeeded"
                )
                log.event(event, node=node, scope=scope, attempt=attempt)
                return result
            except ProviderFailure as exc:
                last_error = exc
                log.write_json(
                    directory / "output.json",
                    {"provider_record": exc.record},
                )
                log.copy_media(directory / "output-media", exc.evidence_media)
                log.error(directory / "error.json", exc)
                self._write_process_evidence(log, directory, exc)
                log.event(
                    "node_failed",
                    node=node,
                    scope=scope,
                    attempt=attempt,
                    error=str(exc),
                )
            except Exception as exc:
                last_error = exc
                if provider_record is not None:
                    log.write_json(
                        directory / "output.json",
                        {"provider_record": provider_record},
                    )
                    log.copy_media(directory / "output-media", provider_media)
                log.error(directory / "error.json", exc)
                self._write_process_evidence(log, directory, exc)
                log.event(
                    "node_failed",
                    node=node,
                    scope=scope,
                    attempt=attempt,
                    error=str(exc),
                )
            should_retry = attempt == 1 and (
                retry_if(last_error)
                if retry_if is not None
                else not node.endswith("-submit")
            )
            if not should_retry:
                break
            if self.settings.retry_delay_s:
                time.sleep(self.settings.retry_delay_s)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _write_process_evidence(
        log: ResearchLog, directory: Path, exc: Exception,
    ) -> None:
        cause = exc.cause if isinstance(exc, ProviderFailure) else exc
        stdout = getattr(cause, "stdout", None)
        stderr = getattr(cause, "stderr", None)
        if isinstance(stdout, str):
            log.write_text(directory / "stdout.log", stdout)
        if isinstance(stderr, str):
            log.write_text(directory / "stderr.log", stderr)

    @staticmethod
    def _stored_input_paths(
        project: Path,
        state: dict[str, Any],
    ) -> tuple[Path, Path | None]:
        """Resolve only the two canonical project-owned input locations."""
        if state.get("source_path") != "inputs/source.mp4":
            raise ValueError("stored source path must be inputs/source.mp4")
        replacement_path = state.get("replacement_image_path")
        if replacement_path not in {None, "inputs/replacement-image"}:
            raise ValueError(
                "stored replacement path must be inputs/replacement-image"
            )
        project_root = project.resolve(strict=True)
        source_path = project / "inputs" / "source.mp4"
        source = source_path.resolve(strict=True)
        if (
            project_root not in source.parents
            or source_path.is_symlink()
            or not source.is_file()
        ):
            raise ValueError("stored source is outside the project")
        replacement: Path | None = None
        if replacement_path is not None:
            candidate = project / "inputs" / "replacement-image"
            replacement = candidate.resolve(strict=True)
            if (
                project_root not in replacement.parents
                or candidate.is_symlink()
                or not replacement.is_file()
            ):
                raise ValueError("stored replacement image is outside the project")
        return source, replacement

    def run(self, project_id: str) -> None:
        project = self.store.path(project_id)
        log = ResearchLog(project)
        state = self.store.read(project_id)
        source, replacement = self._stored_input_paths(project, state)
        request = state.get("effective_request") or {}
        try:
            self._progress(project_id, 2, "probe")
            info = self._node(
                log,
                "probe",
                "project",
                {"source": source},
                [source],
                lambda: media.probe(source),
            )
            duration_s = float(info["duration_s"])
            self.store.update(project_id, duration_s=duration_s, has_source=True)

            self._progress(project_id, 5, "source-4fps")
            source_frames = project / "work" / "source-4fps"
            source_manifest = self._node(
                log,
                "source-frames-4fps",
                "project",
                {"source": source, "fps": 4, "output": source_frames},
                [source],
                lambda: media.extract_frames_4fps(source, source_frames),
                lambda value: self._manifest_media(source_frames, value),
            )
            duration_s = float(source_manifest.get("duration_seconds") or duration_s)
            self.store.update(project_id, duration_s=duration_s)

            yamnet, asr_lines, spoken_lines, voice_decisions = (
                self._analyze_audio_dialogue(
                    project_id,
                    project,
                    source,
                    has_audio=bool(info.get("has_audio")),
                    log=log,
                )
            )
            self.store.update(
                project_id,
                voice_lines=spoken_lines,
                asr_voice_lines=asr_lines,
                voice_decisions=voice_decisions,
                vocal_analysis=yamnet,
            )

            self._progress(project_id, 18, "scene-detection")
            scenes = self._node(
                log,
                "scene-detection",
                "project",
                {"source": source, "threshold": 27.0},
                [source],
                lambda: media.detect_scenes(source, duration_s),
            )
            segments = self._node(
                log,
                "segment-planning",
                "project",
                {
                    "scenes": scenes,
                    "duration_s": duration_s,
                    "dialogue": spoken_lines,
                    "provider_window_s": [4, 10],
                },
                [],
                lambda: media.plan_segments(scenes, duration_s, spoken_lines),
            )
            selections: dict[int, list[dict[str, Any]]] = {}
            for segment in segments:
                index = int(segment["index"])
                scope = self._segment_scope(index)
                selections[index] = self._node(
                    log,
                    "keyframe-selection-exact-nine",
                    scope,
                    {"segment": segment, "scenes": scenes, "count": 9},
                    [],
                    lambda segment=segment: media.select_segment_keyframes(
                        scenes, segment,
                    ),
                )
            self.store.update(
                project_id,
                segment_count=len(segments),
                scenes=scenes,
                segments=segments,
            )

            self._progress(project_id, 24, "video-maker", segments=segments)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(segments)),
            ) as pool:
                futures = [
                    pool.submit(
                        self._prepare_segment,
                        project_id,
                        project,
                        source,
                        segment,
                        selections[int(segment["index"])],
                        request,
                        log,
                    )
                    for segment in segments
                ]
                prepared = [future.result() for future in futures]
            prepared.sort(key=lambda item: int(item["index"]))
            self._continue_after_prepared(
                project_id,
                project,
                prepared,
                spoken_lines,
                replacement,
                request,
                log,
            )
        except Exception as exc:
            log.event("pipeline_failed", error=str(exc))
            current = self.store.read(project_id)
            percent = max(
                1,
                int((current.get("project_progress") or {}).get("percent") or 1),
            )
            self.store.update(
                project_id,
                status="failed",
                project_progress={
                    "status": "failed",
                    "percent": min(percent, 99),
                    "stage": "pipeline_failed",
                },
                generation={
                    "status": "failed",
                    "stage": "pipeline_failed",
                    "error": "pipeline_failed",
                },
                error={
                    "code": "pipeline_failed",
                    "message": "流水线执行失败",
                },
            )

    def _fusion_client_for_request(
        self, request: dict[str, Any],
    ) -> tuple[str, CodexClient | DeepSeekClient]:
        configured_fusion = request.get("fusion")
        # Legacy projects without a frozen internal choice used DeepSeek.
        agent = (
            configured_fusion.get("agent")
            if isinstance(configured_fusion, dict)
            else "deepseek"
        )
        if agent == "codex":
            return agent, self.codex
        if agent == "deepseek":
            return agent, self.deepseek
        raise ValueError("frozen fusion agent is invalid")

    @staticmethod
    def _stored_voice_reference(
        project: Path, state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, Path | None]:
        selection = state.get("voice_selection")
        if selection is None:
            # Backward compatibility for projects created before voice selection.
            return None, None
        if not isinstance(selection, dict):
            raise TypeError("stored voice selection is invalid")
        status = selection.get("status")
        if status == "not_needed":
            return selection, None
        if status != "selected":
            raise ValueError("stored voice selection status is invalid")
        if selection.get("asset_path") != "work/voice-reference.wav":
            raise ValueError("stored voice reference path is invalid")
        expected_sha256 = selection.get("asset_sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("stored voice reference checksum is invalid")
        voice_reference = (project / "work" / "voice-reference.wav").resolve(strict=True)
        project_root = project.resolve(strict=True)
        if project_root not in voice_reference.parents or not voice_reference.is_file():
            raise ValueError("stored voice reference is outside the project")
        if Pipeline._sha256(voice_reference) != expected_sha256:
            raise ValueError("stored voice reference checksum does not match")
        return selection, voice_reference

    @staticmethod
    def _stored_voice_references(
        project: Path, state: dict[str, Any],
    ) -> list[Path]:
        """Load the frozen Audio 1..N list, with legacy Audio 1 fallback."""
        if "voice_references" not in state:
            _selection, legacy = Pipeline._stored_voice_reference(project, state)
            return [legacy] if legacy is not None else []

        references = state["voice_references"]
        if not isinstance(references, list) or len(references) > MAX_REFERENCE_AUDIO:
            raise ValueError("stored voice references are invalid")
        project_root = project.resolve(strict=True)
        paths: list[Path] = []
        for slot, reference in enumerate(references, 1):
            if not isinstance(reference, dict):
                raise ValueError("stored voice reference is invalid")
            expected_path = f"work/voice-references/audio-{slot:02d}.wav"
            expected_sha256 = reference.get("asset_sha256")
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError("stored voice reference checksum is invalid")
            resolved = (project / expected_path).resolve(strict=True)
            if project_root not in resolved.parents or not resolved.is_file():
                raise ValueError("stored voice reference is outside the project")
            if Pipeline._sha256(resolved) != expected_sha256:
                raise ValueError("stored voice reference checksum does not match")
            paths.append(resolved)
        return paths

    def _materialize_voice_references(
        self,
        project: Path,
        patch: dict[str, dict[str, dict[str, Any]]],
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[Path]]:
        selected_ids: list[str] = []
        people = patch.get("people")
        records = people.values() if isinstance(people, dict) else ()
        for person in records:
            voice_id = person.get("voice_id") if isinstance(person, dict) else None
            if (
                isinstance(voice_id, str)
                and voice_id in self.voice_library.files
                and voice_id not in selected_ids
                and len(selected_ids) < MAX_REFERENCE_AUDIO
            ):
                selected_ids.append(voice_id)

        directory = project / "work" / "voice-references"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        metadata_by_id: dict[str, dict[str, Any]] = {}
        references: list[dict[str, Any]] = []
        paths: list[Path] = []
        for slot, voice_id in enumerate(selected_ids, 1):
            voice = self.voice_library.voice(voice_id)
            source = self.voice_library.files[voice_id]
            relative = f"work/voice-references/audio-{slot:02d}.wav"
            destination = project / relative
            temporary = directory / f".audio-{slot:02d}.wav.tmp"
            try:
                shutil.copyfile(source, temporary)
                temporary.chmod(0o600)
                os.replace(temporary, destination)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            if self._sha256(destination) != voice["sha256"]:
                raise ValueError("materialized voice reference checksum does not match")
            metadata = {
                "audio_reference": f"Audio {slot}",
                "voice_id": voice_id,
                "asset_path": relative,
                "asset_sha256": voice["sha256"],
                "features": self.voice_library.model_line(voice_id),
            }
            metadata_by_id[voice_id] = metadata
            references.append(metadata)
            paths.append(destination)
        return metadata_by_id, references, paths

    def _run_object_replace(
        self,
        project: Path,
        prepared: list[dict[str, Any]],
        translated_lines: list[dict[str, Any]],
        element_index: dict[str, Any],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[Path]]:
        user_instruction = str(
            ((request.get("replacement_guidance") or {}).get("instruction"))
            or ""
        ).strip()
        target_language = str(
            ((request.get("dialogue") or {}).get("target_language"))
            or SAME_AS_SOURCE_LANGUAGE
        ).strip()
        voice_ids = [voice["voice_id"] for voice in self.voice_library.voices]
        payload = {
            "global_element_index": element_index,
            "segment_visual_prompts": [
                {
                    "segment_index": int(item["index"]),
                    "visual_prompt": item["visual_prompt"],
                }
                for item in sorted(prepared, key=lambda value: int(value["index"]))
            ],
            "target_language": target_language,
            "translated_dialogue": translated_lines,
            "user_replacement_instruction": user_instruction,
            "has_user_image": replacement is not None,
            "user_replacement_image": (
                {"name": "Image 1", "sha256": self._sha256(replacement)}
                if replacement is not None else None
            ),
            "voice_catalog_sha256": self.voice_library.digest,
            "voice_catalog_lines": self.voice_library.model_lines(),
        }
        output_schema, input_sha256 = object_replace_schema(payload, voice_ids)
        materialized: dict[str, Any] = {}

        def normalize_and_merge(value: object) -> dict[str, Any]:
            patch = normalize_object_replace(
                value,
                input_sha256=input_sha256,
                element_index=element_index,
                voice_ids=voice_ids,
                has_user_image=replacement is not None,
                user_instruction=user_instruction,
            )
            metadata_by_id, references, paths = self._materialize_voice_references(
                project, patch,
            )
            materialized["references"] = references
            materialized["paths"] = paths
            return merge_object_replace_index(
                element_index, patch, metadata_by_id,
            )

        skill = OBJECT_REPLACE_SKILL.read_text(encoding="utf-8")
        enhanced_index = self._node(
            log,
            "object-replace",
            "project",
            {
                "agent": "deepseek",
                "skill": skill,
                "input_sha256": input_sha256,
                "payload": payload,
                "output_schema": output_schema,
            },
            [replacement] if replacement is not None else [],
            lambda: self.deepseek.complete_structured(
                skill,
                payload,
                output_schema,
                [replacement] if replacement is not None else [],
            ),
            transform=normalize_and_merge,
            output_media=lambda _value: list(materialized.get("paths") or []),
        )
        references = list(materialized.get("references") or [])
        paths = list(materialized.get("paths") or [])
        metadata = {
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": input_sha256,
            "voice_catalog_sha256": self.voice_library.digest,
        }
        return enhanced_index, metadata, references, paths

    @staticmethod
    def _project_artifact_receipt(
        project: Path,
        path: Path,
        relative_path: str,
    ) -> dict[str, Any]:
        expected = project / relative_path
        project_root = project.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if (
            resolved != expected.resolve(strict=True)
            or project_root not in resolved.parents
            or path.is_symlink()
            or not resolved.is_file()
        ):
            raise ValueError("prepared H3 artifact path is invalid")
        size_bytes = resolved.stat().st_size
        if size_bytes < 1:
            raise ValueError("prepared H3 artifact is empty")
        return {
            "path": relative_path,
            "size_bytes": size_bytes,
            "sha256": Pipeline._sha256(resolved),
        }

    def _freeze_h3_prepared_artifacts(
        self,
        project_id: str,
        project: Path,
        h3_inputs: list[tuple[dict[str, Any], str, Path, str]],
        *,
        replacement: Path | None = None,
        request: dict[str, Any] | None = None,
    ) -> None:
        segments: dict[str, dict[str, Any]] = {}
        for item, final_prompt, muted, fusion_input_sha256 in sorted(
            h3_inputs, key=lambda value: int(value[0]["index"]),
        ):
            index = int(item["index"])
            if str(final_prompt).strip() != final_prompt:
                raise ValueError("prepared H3 prompt is not canonical")
            base = f"work/segments/{index:04d}"
            segments[str(index)] = {
                "segment_index": index,
                "source_video": self._project_artifact_receipt(
                    project, item["source"], f"{base}/source.mp4",
                ),
                "muted_video": self._project_artifact_receipt(
                    project, muted, f"{base}/source-muted.mp4",
                ),
                "prompt": self._project_artifact_receipt(
                    project, muted.parent / "h3_prompt.txt", f"{base}/h3_prompt.txt",
                ),
                "fusion_input_sha256": fusion_input_sha256,
                "segment_metadata_sha256": self._json_sha256({
                    "index": item.get("index"),
                    "start_s": item.get("start_s"),
                    "end_s": item.get("end_s"),
                    "duration_s": item.get("duration_s"),
                    "join_mode": item.get("join_mode", "hard_cut"),
                }),
            }
        state = self.store.read(project_id)
        enhanced_index = state.get("enhanced_element_index")
        voice_references = state.get("voice_references")
        object_replace = state.get("object_replace")
        if (
            not isinstance(enhanced_index, dict)
            or not isinstance(voice_references, list)
            or not isinstance(object_replace, dict)
            or not isinstance(object_replace.get("input_sha256"), str)
            or len(object_replace["input_sha256"]) != 64
        ):
            raise ValueError("H3 preparation binding state is invalid")
        if replacement is None and state.get("replacement_image_path") is not None:
            replacement = project / "inputs" / "replacement-image"
        replacement_receipt = (
            self._project_artifact_receipt(
                project, replacement, "inputs/replacement-image",
            )
            if replacement is not None
            else None
        )
        effective_request = request if isinstance(request, dict) else (
            state.get("effective_request") or {}
        )
        output_request = (
            effective_request.get("output") or {}
            if isinstance(effective_request, dict)
            else {}
        )
        self.store.update(
            project_id,
            h3_prepared_artifacts={
                "version": H3_PREPARED_ARTIFACTS_VERSION,
                "bindings": {
                    "enhanced_element_index_sha256": self._json_sha256(
                        enhanced_index,
                    ),
                    "voice_references_sha256": self._json_sha256(
                        voice_references,
                    ),
                    "object_replace_input_sha256": object_replace[
                        "input_sha256"
                    ],
                    "replacement_image": replacement_receipt,
                    "output_request_sha256": self._json_sha256(output_request),
                },
                "segments": segments,
            },
        )

    def _run_h3_segments(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        voice_reference_paths: list[Path],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> list[Path]:
        """Cross both free-work barriers before scheduling any paid submit."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(prepared)),
        ) as pool:
            prepare_futures = [
                pool.submit(
                    self._prepare_h3_segment,
                    project,
                    item,
                    enhanced_element_index,
                    request,
                    log,
                )
                for item in prepared
            ]
            h3_inputs = [future.result() for future in prepare_futures]
            self._freeze_h3_prepared_artifacts(
                project_id,
                project,
                h3_inputs,
                replacement=replacement,
                request=request,
            )
            preflight_futures = [
                pool.submit(
                    self._prepare_picsart_segment,
                    project_id,
                    project,
                    item,
                    voice_reference_paths,
                    replacement,
                    request,
                    log,
                    final_prompt=final_prompt,
                    muted=muted,
                )
                for item, final_prompt, muted, _fusion_input_sha256 in h3_inputs
            ]
            picsart_requests = [
                future.result() for future in preflight_futures
            ]
            self._begin_submission_batch(picsart_requests)
            submit_futures = [
                pool.submit(
                    self._generate_picsart_segment,
                    picsart_request,
                    item,
                    log,
                )
                for (item, _final_prompt, _muted, _fusion_hash), picsart_request
                in zip(h3_inputs, picsart_requests)
            ]
            return [future.result() for future in submit_futures]

    def _continue_after_prepared(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        spoken_lines: list[dict[str, Any]],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> None:
        visual_prompts = "\n\n".join(
            f"Segment {item['index']}: {item['visual_prompt']}"
            for item in prepared
        )
        target_language = str(
            ((request.get("dialogue") or {}).get("target_language"))
            or SAME_AS_SOURCE_LANGUAGE
        ).strip()
        self._progress(project_id, 43, "translation")
        translated_lines = self._node(
            log,
            "deepseek-translation",
            "project",
            {
                "target_language": target_language,
                "source_lines": spoken_lines,
                "visual_prompts": visual_prompts,
            },
            [],
            lambda: self._translation_request(
                spoken_lines, target_language, visual_prompts,
            ),
            transform=lambda value: self._normalize_translation(
                value, spoken_lines,
            ),
        )
        for item in prepared:
            item["dialogue"] = self._node(
                log,
                "dialogue-localize",
                self._segment_scope(int(item["index"])),
                {"translated_lines": translated_lines, "segment": item},
                [],
                lambda item=item: media.localize_dialogue(
                    translated_lines, item, segments=prepared,
                ),
            )

        index_request, index_images = self._project_index_request(
            project,
            prepared,
            self._generation_config(request),
        )
        index_request_path = project / "work" / "project_index_request.json"
        index_request_path.write_text(
            json.dumps(index_request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_schema = project_index_schema([
            segment["segment_index"] for segment in index_request["segments"]
        ])
        self._progress(project_id, 50, "video-maker-project-index")
        skill = VIDEO_SKILL.read_text(encoding="utf-8")
        element_index = self._node(
            log,
            "video-maker-project-index",
            "project",
            index_request,
            index_images,
            lambda: self.deepseek.complete_structured(
                skill
                + (
                    "\n\n严格执行项目级索引阶段；像素帧是视觉事实的唯一依据；"
                    "segments[].visual_prompt 只用于汇总逐段已发现对象并检查跨段覆盖，"
                    "不得用它替代或推翻像素证据；最终填写本次注入的 Schema。"
                ),
                index_request,
                index_schema,
                index_images,
            ),
            transform=normalize_project_index,
        )
        element_path = project / "work" / "element_index.json"
        element_path.write_text(
            json.dumps(element_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        public_segments = [self._public_segment(item) for item in prepared]
        self.store.update(
            project_id,
            segments=public_segments,
            element_index=element_index,
            translated_voice_lines=translated_lines,
        )

        self._progress(
            project_id, 53, "object-replace", segments=public_segments,
        )
        (
            enhanced_element_index,
            object_replace,
            voice_references,
            voice_reference_paths,
        ) = self._run_object_replace(
            project,
            prepared,
            translated_lines,
            element_index,
            replacement,
            request,
            log,
        )
        enhanced_path = project / "work" / "enhanced_element_index.json"
        enhanced_temporary = project / "work" / ".enhanced_element_index.json.tmp"
        try:
            enhanced_temporary.write_text(
                json.dumps(
                    enhanced_element_index,
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            enhanced_temporary.chmod(0o600)
            os.replace(enhanced_temporary, enhanced_path)
        finally:
            try:
                enhanced_temporary.unlink()
            except FileNotFoundError:
                pass
        self.store.update(
            project_id,
            enhanced_element_index=enhanced_element_index,
            object_replace=object_replace,
            voice_references=voice_references,
        )

        # This is the public-input barrier.  No paid segment branch starts
        # before every visual prompt, translation, localization, the single
        # project index and project-wide replacement/voice plan have completed.
        self._progress(
            project_id, 57, "h3-max-r2v", segments=public_segments,
        )
        outputs = self._run_h3_segments(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            voice_reference_paths,
            replacement,
            request,
            log,
        )

        final = project / "output" / "generated.mp4"
        self._progress(
            project_id,
            95,
            "stitch",
            segments=[self._public_segment(item) for item in prepared],
        )
        self._node(
            log,
            "stitch",
            "project",
            {
                "segments": outputs,
                "target_durations": [item["duration_s"] for item in prepared],
                "join_modes": [
                    item.get("join_mode", "hard_cut") for item in prepared
                ],
                "output": final,
            },
            outputs,
            lambda: media.concat(
                outputs,
                final,
                target_durations=[item["duration_s"] for item in prepared],
                join_modes=[
                    item.get("join_mode", "hard_cut") for item in prepared
                ],
            ),
            lambda _value: [final],
        )
        self.store.update(
            project_id,
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={
                "status": "succeeded",
                "percent": 100,
                "stage": "completed",
            },
            generation={
                "status": "succeeded",
                "stage": "completed",
                "error": None,
            },
            segments=[
                {**self._public_segment(item), "status": "succeeded"}
                for item in prepared
            ],
            error=None,
        )
        log.event("pipeline_succeeded", output=final)

    @staticmethod
    def _validated_stored_segments(
        state: dict[str, Any],
        *,
        missing_message: str,
    ) -> list[dict[str, Any]]:
        raw_segments = state.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(missing_message)
        segment_count = state.get("segment_count")
        if (
            isinstance(segment_count, bool)
            or not isinstance(segment_count, int)
            or segment_count < 1
            or segment_count != len(raw_segments)
        ):
            raise ValueError("stored segment count is invalid")
        segments: list[dict[str, Any]] = []
        indices: list[int] = []
        for raw in raw_segments:
            if not isinstance(raw, dict):
                raise TypeError("stored segment must be an object")
            index = raw.get("index")
            duration = raw.get("duration_s")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) * 24 < 1
            ):
                raise ValueError("stored segment metadata is invalid")
            join_mode = raw.get("join_mode", "hard_cut")
            if join_mode not in {"hard_cut", "continue"}:
                raise ValueError("stored segment join mode is invalid")
            indices.append(index)
            segments.append({**raw, "join_mode": join_mode})
        if sorted(indices) != list(range(1, segment_count + 1)):
            raise ValueError(
                "stored segment indices must be unique and contiguous"
            )
        segments.sort(key=lambda item: int(item["index"]))
        if segments[0]["join_mode"] != "hard_cut":
            raise ValueError("first stored segment must use a hard cut")
        return segments

    def _load_pre_picsart_resume(
        self,
        project_id: str,
    ) -> tuple[
        Path,
        Path,
        Path | None,
        dict[str, Any],
        list[dict[str, Any]],
        dict[int, list[dict[str, Any]]],
        list[dict[str, Any]],
    ]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        status = state.get("status")
        stage = (state.get("project_progress") or {}).get("stage")
        if status != "failed" and not (
            status == "queued" and stage == "pre_picsart_resume_queued"
        ):
            raise ValueError("only failed projects can resume preparation")
        logs = project / "logs"
        if (
            state.get("picsart_submission_receipts") is not None
            or any(path.is_dir() for path in logs.glob("picsart*-submit"))
        ):
            raise ValueError("project has Picsart submission evidence")
        source, replacement = self._stored_input_paths(project, state)
        raw_segments = self._validated_stored_segments(
            state, missing_message="stored segments are missing",
        )
        segments: list[dict[str, Any]] = []
        selections: dict[int, list[dict[str, Any]]] = {}
        for raw in raw_segments:
            segment = dict(raw)
            index = segment["index"]
            scope = self._segment_scope(index)
            directory = project / "work" / "segments" / f"{index:04d}"
            selection_path = directory / "keyframe_selection.json"
            if not selection_path.is_file():
                raise FileNotFoundError("reusable keyframe selection is missing")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if not isinstance(selection, list) or len(selection) != 9:
                raise ValueError("reusable keyframe selection must contain 9 items")
            if not (directory / "source.mp4").is_file():
                raise FileNotFoundError("reusable segment source is missing")
            if len(list((directory / "keyframes").glob("*.png"))) != 9:
                raise ValueError("reusable segment must contain exactly 9 keyframes")
            if not isinstance(
                self._completed_node_value(
                    project, "segment-frames-4fps", scope,
                ),
                dict,
            ):
                raise TypeError("reusable segment manifest is invalid")
            segments.append(segment)
            selections[index] = selection
        spoken_lines = state.get("voice_lines") or []
        if not isinstance(spoken_lines, list) or not all(
            isinstance(line, dict) for line in spoken_lines
        ):
            raise TypeError("stored spoken dialogue is invalid")
        return (
            project,
            source,
            replacement,
            state.get("effective_request") or {},
            segments,
            selections,
            [dict(line) for line in spoken_lines],
        )

    def validate_pre_picsart_resume(self, project_id: str) -> None:
        self._load_pre_picsart_resume(project_id)

    def resume_pre_picsart(self, project_id: str) -> None:
        (
            project,
            source,
            replacement,
            request,
            segments,
            selections,
            spoken_lines,
        ) = self._load_pre_picsart_resume(project_id)
        log = ResearchLog(project)
        log.event("pre_picsart_resume_started")
        try:
            self._progress(
                project_id, 24, "video-maker-resume", segments=segments,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(segments)),
            ) as pool:
                futures = [
                    pool.submit(
                        self._prepare_segment,
                        project_id,
                        project,
                        source,
                        segment,
                        selections[int(segment["index"])],
                        request,
                        log,
                        resume=True,
                    )
                    for segment in segments
                ]
                prepared = [future.result() for future in futures]
            prepared.sort(key=lambda item: int(item["index"]))
            self._continue_after_prepared(
                project_id,
                project,
                prepared,
                spoken_lines,
                replacement,
                request,
                log,
            )
            log.event("pre_picsart_resume_succeeded")
        except Exception as exc:
            log.event("pre_picsart_resume_failed", error=str(exc))
            current = self.store.read(project_id)
            percent = max(
                24,
                int((current.get("project_progress") or {}).get("percent") or 24),
            )
            self.store.update(
                project_id,
                status="failed",
                project_progress={
                    "status": "failed",
                    "percent": min(percent, 99),
                    "stage": "pre_picsart_resume_failed",
                },
                generation={
                    "status": "failed",
                    "stage": "pre_picsart_resume_failed",
                    "error": "pre_picsart_resume_failed",
                },
                error={
                    "code": "pre_picsart_resume_failed",
                    "message": "前置产物续跑失败",
                },
            )

    @staticmethod
    def _resume_prefix_key(prefix: str) -> int:
        if prefix == "picsart-resume":
            return 1
        marker = "picsart-resume-"
        if prefix.startswith(marker):
            try:
                return int(prefix[len(marker):])
            except ValueError:
                return 0
        return 0

    def _resume_prefixes(self, project: Path) -> list[str]:
        prefixes: set[str] = set()
        for suffix in ("-submit", "-poll"):
            prefixes.update(
                path.name.removesuffix(suffix)
                for path in (project / "logs").glob(
                    f"picsart-resume*{suffix}"
                )
                if path.is_dir()
            )
        return sorted(prefixes, key=self._resume_prefix_key, reverse=True)

    def _next_resume_identity(
        self,
        project: Path,
        state: dict[str, Any],
    ) -> tuple[str, str]:
        prefixes = self._resume_prefixes(project)
        receipts = self._validated_submission_receipts(state)
        if receipts is not None:
            prefixes.extend(
                receipt["run_identity"]
                for records in receipts["segments"].values()
                for receipt in records
                if receipt["run_identity"].startswith("picsart-resume")
            )
        number = max((self._resume_prefix_key(value) for value in prefixes), default=0) + 1
        if number == 1:
            return "picsart-resume", "-resume"
        return f"picsart-resume-{number:02d}", f"-resume-{number:02d}"

    @staticmethod
    def _validated_submission_receipts(
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw = state.get("picsart_submission_receipts")
        if raw is None:
            return None
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "segments"}
            or raw.get("version") != PICSART_RECEIPT_VERSION
            or isinstance(raw.get("version"), bool)
            or not isinstance(raw.get("segments"), dict)
        ):
            raise ValueError("stored Picsart receipt ledger is invalid")
        normalized: dict[str, list[dict[str, Any]]] = {}
        allowed_statuses = {
            "submission_started", "accepted", "unknown", "rejected",
            "provider_failed",
        }
        for segment_key, records in raw["segments"].items():
            if (
                not isinstance(segment_key, str)
                or not segment_key.isdecimal()
                or str(int(segment_key)) != segment_key
                or int(segment_key) < 1
                or not isinstance(records, list)
                or not records
            ):
                raise ValueError("stored Picsart segment receipts are invalid")
            seen_runs: set[str] = set()
            copied: list[dict[str, Any]] = []
            for offset, receipt in enumerate(records):
                if not isinstance(receipt, dict) or set(receipt) != {
                    "segment_index", "run_identity", "request_sha256",
                    "status", "task_id",
                }:
                    raise ValueError("stored Picsart receipt is invalid")
                run_identity = receipt["run_identity"]
                digest = receipt["request_sha256"]
                status = receipt["status"]
                task_id = receipt["task_id"]
                if (
                    receipt["segment_index"] != int(segment_key)
                    or isinstance(receipt["segment_index"], bool)
                    or not isinstance(run_identity, str)
                    or not run_identity
                    or run_identity != run_identity.strip()
                    or run_identity in seen_runs
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or status not in allowed_statuses
                    or (
                        status in {"accepted", "provider_failed"}
                        and (not isinstance(task_id, str) or not task_id.strip())
                    )
                    or (
                        status not in {"accepted", "provider_failed"}
                        and task_id is not None
                    )
                    or (
                        offset < len(records) - 1
                        and status not in {"rejected", "provider_failed"}
                    )
                ):
                    raise ValueError("stored Picsart receipt fields are invalid")
                seen_runs.add(run_identity)
                copied.append(dict(receipt))
            normalized[segment_key] = copied
        return {"version": PICSART_RECEIPT_VERSION, "segments": normalized}

    def _begin_submission_batch(
        self,
        requests: list[PreparedPicsartRequest],
    ) -> None:
        pending = [item for item in requests if item.existing_task_id is None]
        if not pending:
            return
        project_ids = {item.project_id for item in pending}
        segment_indices = [item.segment_index for item in pending]
        if len(project_ids) != 1 or len(segment_indices) != len(set(segment_indices)):
            raise ValueError("prepared Picsart submission batch is invalid")
        project_id = pending[0].project_id
        with self._picsart_receipt_lock:
            state = self.store.read(project_id)
            ledger = self._validated_submission_receipts(state) or {
                "version": PICSART_RECEIPT_VERSION,
                "segments": {},
            }
            for prepared in sorted(pending, key=lambda item: item.segment_index):
                key = str(prepared.segment_index)
                records = ledger["segments"].setdefault(key, [])
                if any(
                    receipt["run_identity"] == prepared.run_identity
                    for receipt in records
                ):
                    raise ValueError("Picsart run identity was already used")
                if records and records[-1]["status"] not in {
                    "rejected", "provider_failed",
                }:
                    raise ValueError(
                        "Picsart submission is not proven safe to retry"
                    )
                records.append({
                    "segment_index": prepared.segment_index,
                    "run_identity": prepared.run_identity,
                    "request_sha256": prepared.request_sha256,
                    "status": "submission_started",
                    "task_id": None,
                })
            self.store.update(
                project_id,
                picsart_submission_receipts=ledger,
            )

    def _transition_submission_receipt(
        self,
        prepared: PreparedPicsartRequest,
        *,
        status: str,
        task_id: str | None = None,
    ) -> None:
        if status not in {"accepted", "unknown", "rejected"}:
            raise ValueError("invalid Picsart submission receipt transition")
        if status == "accepted":
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("accepted Picsart receipt requires a task id")
        elif task_id is not None:
            raise ValueError("non-accepted Picsart receipt cannot have a task id")
        with self._picsart_receipt_lock:
            state = self.store.read(prepared.project_id)
            ledger = self._validated_submission_receipts(state)
            if ledger is None:
                raise ValueError("Picsart submission receipt ledger is missing")
            records = ledger["segments"].get(str(prepared.segment_index)) or []
            matches = [
                receipt for receipt in records
                if receipt["run_identity"] == prepared.run_identity
            ]
            if len(matches) != 1:
                raise ValueError("Picsart submission receipt identity is missing")
            receipt = matches[0]
            if (
                receipt["request_sha256"] != prepared.request_sha256
                or receipt["status"] != "submission_started"
            ):
                raise ValueError("Picsart submission receipt transition is unsafe")
            receipt["status"] = status
            receipt["task_id"] = task_id
            self.store.update(
                prepared.project_id,
                picsart_submission_receipts=ledger,
            )

    def _mark_submission_task_failed(
        self,
        project_id: str,
        segment_index: int,
        task_id: str,
    ) -> None:
        with self._picsart_receipt_lock:
            state = self.store.read(project_id)
            ledger = self._validated_submission_receipts(state)
            if ledger is None:
                return
            records = ledger["segments"].get(str(segment_index)) or []
            # A ledger may first appear when a different segment is resumed
            # from a mixed legacy/new project. This segment's accepted task
            # can still be owned solely by the legacy research log, so there
            # is no state receipt to transition here.
            if not records:
                return
            matches = [
                receipt for receipt in records
                if receipt["status"] == "accepted"
                and receipt["task_id"] == task_id
            ]
            if len(matches) != 1:
                raise ValueError("accepted Picsart task receipt is missing")
            matches[0]["status"] = "provider_failed"
            self.store.update(
                project_id,
                picsart_submission_receipts=ledger,
            )

    @classmethod
    def _state_submission_task(
        cls,
        state: dict[str, Any],
        segment_index: int,
    ) -> tuple[bool, str | None]:
        ledger = cls._validated_submission_receipts(state)
        if ledger is None:
            return False, None
        records = ledger["segments"].get(str(segment_index)) or []
        if not records:
            # A ledger can first appear during a mixed legacy resume where a
            # different segment needed a new POST.  Absence for this segment
            # does not erase its older accepted-task evidence.
            return False, None
        receipt = records[-1]
        if receipt["status"] in {"submission_started", "unknown"}:
            raise ValueError(
                "Picsart submission outcome is unknown; reconcile it before resume"
            )
        if receipt["status"] == "accepted":
            return True, receipt["task_id"]
        return True, None

    def _picsart_receipt_prefixes(self, project: Path) -> list[str]:
        prefixes = self._resume_prefixes(project)
        if (project / "logs" / "picsart-submit").is_dir():
            prefixes.append("picsart")
        return prefixes

    @staticmethod
    def _poll_record_is_terminal_failure(record: dict[str, Any]) -> bool:
        for exchange in record.get("polls") or []:
            if not isinstance(exchange, dict):
                continue
            response = exchange.get("response")
            if not isinstance(response, dict):
                continue
            status_code = response.get("status_code")
            body = response.get("body")
            if status_code == 200:
                envelope = body.get("response") if isinstance(body, dict) else None
                if (
                    isinstance(envelope, dict)
                    and str(envelope.get("status") or "").upper() == "FAILED"
                ):
                    return True
                continue
            # The workflow result endpoint also reports a completed task
            # failure as this exact 422 envelope.  No other HTTP error proves
            # that an accepted paid task is terminal.
            if (
                status_code == 422
                and isinstance(body, dict)
                and str(body.get("status") or "").upper() == "ERROR"
                and body.get("reason") == f"{PicsartClient.WORKFLOW}_failed"
            ):
                return True
        return False

    def _terminal_resume_task_ids(
        self, project: Path, scope: str,
    ) -> set[str]:
        terminal_task_ids: set[str] = set()
        for prefix in self._picsart_receipt_prefixes(project):
            poll_root = project / "logs" / f"{prefix}-poll" / scope
            for poll_output in poll_root.glob("attempt-*/output.json"):
                try:
                    poll_logged = json.loads(
                        poll_output.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                poll_record = (
                    poll_logged.get("provider_record")
                    if isinstance(poll_logged, dict)
                    else None
                )
                task_id = (
                    poll_record.get("task_id")
                    if isinstance(poll_record, dict)
                    else None
                )
                if (
                    isinstance(task_id, str)
                    and task_id
                    and self._poll_record_is_terminal_failure(poll_record)
                ):
                    terminal_task_ids.add(task_id)
        return terminal_task_ids

    def _active_resume_task(
        self, project: Path, scope: str,
    ) -> str | None:
        terminal_task_ids = self._terminal_resume_task_ids(project, scope)

        for prefix in self._picsart_receipt_prefixes(project):
            submit_root = project / "logs" / f"{prefix}-submit" / scope
            outputs = sorted(
                submit_root.glob("attempt-*/output.json"), reverse=True,
            )
            for output in outputs:
                try:
                    logged = json.loads(output.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                record = logged.get("provider_record") if isinstance(logged, dict) else None
                task_id = record.get("task_id") if isinstance(record, dict) else None
                if not isinstance(task_id, str) or not task_id:
                    continue
                if task_id not in terminal_task_ids:
                    return task_id
        return None

    def _has_retryable_submit_evidence(
        self, project: Path, scope: str,
    ) -> bool:
        terminal_task_ids = self._terminal_resume_task_ids(project, scope)
        for prefix in self._picsart_receipt_prefixes(project):
            submit_root = project / "logs" / f"{prefix}-submit" / scope
            for output in submit_root.glob("attempt-*/output.json"):
                try:
                    logged = json.loads(output.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                record = (
                    logged.get("provider_record")
                    if isinstance(logged, dict)
                    else None
                )
                if not isinstance(record, dict):
                    continue
                task_id = record.get("task_id")
                if isinstance(task_id, str) and task_id in terminal_task_ids:
                    return True
                response = record.get("response")
                status_code = (
                    response.get("status_code")
                    if isinstance(response, dict)
                    else None
                )
                if (
                    record.get("outcome") == "rejected"
                    and status_code in PicsartClient.SAFE_REJECT_STATUS_CODES
                ):
                    return True
        return False

    def _has_unknown_submit_receipt(
        self, project: Path, scope: str,
    ) -> bool:
        for prefix in self._picsart_receipt_prefixes(project):
            submit_root = project / "logs" / f"{prefix}-submit" / scope
            for attempt in submit_root.glob("attempt-*"):
                if not attempt.is_dir():
                    continue
                output = attempt / "output.json"
                try:
                    logged = json.loads(output.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    # Input evidence is written before the paid POST. A
                    # missing or corrupt result cannot prove rejection.
                    return True
                record = (
                    logged.get("provider_record")
                    if isinstance(logged, dict)
                    else None
                )
                if not isinstance(record, dict):
                    return True
                task_id = record.get("task_id")
                if isinstance(task_id, str) and task_id.strip():
                    continue
                # Explicit rejection is the only no-task-id result that
                # makes another paid POST safe.
                response = record.get("response")
                status_code = (
                    response.get("status_code")
                    if isinstance(response, dict)
                    else None
                )
                if (
                    record.get("outcome") != "rejected"
                    or status_code not in PicsartClient.SAFE_REJECT_STATUS_CODES
                ):
                    return True
        return False

    def _completed_resume_output(
        self, project: Path, segment_dir: Path, scope: str,
    ) -> Path | None:
        for prefix in self._picsart_receipt_prefixes(project):
            number = self._resume_prefix_key(prefix)
            suffix = (
                "" if prefix == "picsart"
                else "-resume" if number == 1
                else f"-resume-{number:02d}"
            )
            output = segment_dir / f"generated{suffix}.mp4"
            trim_root = (
                project / "logs"
                / (
                    "trim-provider-output-24fps"
                    if prefix == "picsart"
                    else f"{prefix}-trim-provider-output-24fps"
                )
                / scope
            )
            if output.is_file() and any(trim_root.glob("attempt-*/output.json")):
                if output.stat().st_size < 1:
                    raise ValueError("reusable generated segment is empty")
                info = media.probe(output)
                duration = info.get("duration_s") if isinstance(info, dict) else None
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(float(duration))
                    or float(duration) <= 0
                ):
                    raise ValueError("reusable generated segment is invalid")
                return output
        return None

    @staticmethod
    def _validate_artifact_receipt(
        project: Path,
        receipt: object,
        expected_path: str,
    ) -> Path:
        if not isinstance(receipt, dict) or set(receipt) != {
            "path", "size_bytes", "sha256",
        }:
            raise ValueError("stored H3 artifact receipt is invalid")
        size_bytes = receipt.get("size_bytes")
        digest = receipt.get("sha256")
        if (
            receipt.get("path") != expected_path
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 1
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("stored H3 artifact receipt fields are invalid")
        candidate = project / expected_path
        resolved = candidate.resolve(strict=True)
        project_root = project.resolve(strict=True)
        if (
            project_root not in resolved.parents
            or candidate.is_symlink()
            or not resolved.is_file()
            or resolved.stat().st_size != size_bytes
            or Pipeline._sha256(resolved) != digest
        ):
            raise ValueError("stored H3 artifact receipt does not match")
        return resolved

    @classmethod
    def _validated_h3_prepared_artifacts(
        cls,
        project: Path,
        state: dict[str, Any],
        segments: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]] | None:
        raw = state.get("h3_prepared_artifacts")
        if raw is None:
            return None
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "bindings", "segments"}
            or raw.get("version") != H3_PREPARED_ARTIFACTS_VERSION
            or isinstance(raw.get("version"), bool)
            or not isinstance(raw.get("bindings"), dict)
            or not isinstance(raw.get("segments"), dict)
        ):
            raise ValueError("stored H3 preparation manifest is invalid")
        bindings = raw["bindings"]
        if set(bindings) != {
            "enhanced_element_index_sha256", "voice_references_sha256",
            "object_replace_input_sha256", "replacement_image",
            "output_request_sha256",
        }:
            raise ValueError("stored H3 preparation bindings are invalid")
        enhanced_index = state.get("enhanced_element_index")
        voice_references = state.get("voice_references")
        object_replace = state.get("object_replace")
        effective_request = state.get("effective_request") or {}
        output_request = (
            effective_request.get("output") or {}
            if isinstance(effective_request, dict)
            else {}
        )
        if (
            not isinstance(enhanced_index, dict)
            or not isinstance(voice_references, list)
            or not isinstance(object_replace, dict)
            or bindings.get("enhanced_element_index_sha256")
            != cls._json_sha256(enhanced_index)
            or bindings.get("voice_references_sha256")
            != cls._json_sha256(voice_references)
            or bindings.get("object_replace_input_sha256")
            != object_replace.get("input_sha256")
            or bindings.get("output_request_sha256")
            != cls._json_sha256(output_request)
        ):
            raise ValueError("stored H3 preparation bindings do not match")
        replacement_receipt = bindings.get("replacement_image")
        has_replacement = state.get("replacement_image_path") is not None
        if has_replacement != (replacement_receipt is not None):
            raise ValueError("stored H3 preparation replacement binding does not match")
        if replacement_receipt is not None:
            cls._validate_artifact_receipt(
                project, replacement_receipt, "inputs/replacement-image",
            )
        expected_keys = {str(item["index"]) for item in segments}
        if set(raw["segments"]) != expected_keys:
            raise ValueError("stored H3 preparation segments do not match")
        normalized: dict[str, dict[str, Any]] = {}
        for item in segments:
            index = item["index"]
            entry = raw["segments"].get(str(index))
            if not isinstance(entry, dict) or set(entry) != {
                "segment_index", "source_video", "muted_video", "prompt",
                "fusion_input_sha256", "segment_metadata_sha256",
            }:
                raise ValueError("stored H3 preparation segment is invalid")
            fusion_digest = entry.get("fusion_input_sha256")
            metadata_digest = entry.get("segment_metadata_sha256")
            expected_metadata_digest = cls._json_sha256({
                "index": item.get("index"),
                "start_s": item.get("start_s"),
                "end_s": item.get("end_s"),
                "duration_s": item.get("duration_s"),
                "join_mode": item.get("join_mode", "hard_cut"),
            })
            if (
                entry.get("segment_index") != index
                or isinstance(entry.get("segment_index"), bool)
                or not isinstance(fusion_digest, str)
                or len(fusion_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in fusion_digest
                )
                or metadata_digest != expected_metadata_digest
            ):
                raise ValueError("stored H3 preparation segment fields are invalid")
            base = f"work/segments/{index:04d}"
            cls._validate_artifact_receipt(
                project, entry["source_video"], f"{base}/source.mp4",
            )
            cls._validate_artifact_receipt(
                project, entry["muted_video"], f"{base}/source-muted.mp4",
            )
            cls._validate_artifact_receipt(
                project, entry["prompt"], f"{base}/h3_prompt.txt",
            )
            normalized[str(index)] = dict(entry)
        return normalized

    def _load_r2v_resume(
        self, project_id: str,
    ) -> tuple[
        Path, list[dict[str, Any]], Path | None, dict[str, Any], list[Path],
    ]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        status = state.get("status")
        stage = (state.get("project_progress") or {}).get("stage")
        if status != "failed" and not (
            status == "queued" and stage == "picsart_resume_queued"
        ):
            raise ValueError("only a failed or explicitly queued project can resume Picsart")
        _source_input, replacement = self._stored_input_paths(project, state)
        raw_segments = self._validated_stored_segments(
            state, missing_message="project has no reusable segments",
        )
        manifest = self._validated_h3_prepared_artifacts(
            project, state, raw_segments,
        )
        if manifest is None and any(
            key in state
            for key in (
                "object_replace", "enhanced_element_index", "voice_references",
            )
        ):
            raise ValueError(
                "new H3 preparation state is missing its artifact manifest"
            )
        state_receipts = self._validated_submission_receipts(state)
        if state_receipts is not None and not set(
            state_receipts["segments"]
        ).issubset({str(item["index"]) for item in raw_segments}):
            raise ValueError("stored Picsart receipts reference unknown segments")

        prepared: list[dict[str, Any]] = []
        for raw in raw_segments:
            item = dict(raw)
            index = item["index"]
            segment_dir = project / "work" / "segments" / f"{index:04d}"
            source = segment_dir / "source.mp4"
            muted = segment_dir / "source-muted.mp4"
            prompt_path = segment_dir / "h3_prompt.txt"
            for required in (source, muted, prompt_path):
                if not required.is_file():
                    raise FileNotFoundError(
                        f"reusable segment artifact is missing: {required.name}"
                    )
                if required.stat().st_size < 1:
                    raise ValueError(
                        f"reusable segment artifact is empty: {required.name}"
                    )
            if manifest is None:
                self._project_artifact_receipt(
                    project,
                    source,
                    f"work/segments/{index:04d}/source.mp4",
                )
                self._project_artifact_receipt(
                    project,
                    muted,
                    f"work/segments/{index:04d}/source-muted.mp4",
                )
                self._project_artifact_receipt(
                    project,
                    prompt_path,
                    f"work/segments/{index:04d}/h3_prompt.txt",
                )
                self._completed_node_value(
                    project, "mute-reference-video", self._segment_scope(index),
                )
                self._completed_node_value(
                    project, "h3-max-r2v-prompt", self._segment_scope(index),
                )
            self._validate_reference_video(muted)
            final_prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not final_prompt:
                raise ValueError("reusable H3 prompt is empty")
            resume_output = self._completed_resume_output(
                project, segment_dir, self._segment_scope(index),
            )
            scope = self._segment_scope(index)
            has_state_receipt, resume_task_id = self._state_submission_task(
                state, index,
            )
            if (
                has_state_receipt
                and isinstance(resume_task_id, str)
                and resume_task_id
                and resume_task_id
                in self._terminal_resume_task_ids(project, scope)
            ):
                # A previous runtime may have persisted the accepted receipt
                # before it learned this provider-specific terminal envelope.
                # Reconcile only the same task id under an explicitly
                # confirmed resume request, then allow this segment alone to
                # obtain a new submission receipt.
                self._mark_submission_task_failed(
                    project_id, index, resume_task_id,
                )
                resume_task_id = None
            if not has_state_receipt:
                resume_task_id = self._active_resume_task(
                    project, scope,
                )
                if (
                    resume_task_id is None
                    and self._has_unknown_submit_receipt(
                        project, scope,
                    )
                ):
                    raise ValueError(
                        "Picsart submission outcome is unknown; reconcile it before resume"
                    )
                if (
                    resume_task_id is None
                    and resume_output is None
                    and not self._has_retryable_submit_evidence(
                        project, scope,
                    )
                ):
                    raise ValueError(
                        "Picsart submission outcome is unproven; use preparation resume "
                        "or reconcile it before a new paid submit"
                    )
            item.update({
                "source": source,
                "resume_muted": muted,
                "resume_prompt": final_prompt,
                "resume_task_id": resume_task_id,
                "resume_output": resume_output,
                "status": "running",
            })
            prepared.append(item)
        prepared.sort(key=lambda item: int(item["index"]))
        voice_references = self._stored_voice_references(project, state)
        return (
            project,
            prepared,
            replacement,
            state.get("effective_request") or {},
            voice_references,
        )

    def validate_r2v_resume(self, project_id: str) -> None:
        self._load_r2v_resume(project_id)

    def resume_r2v(self, project_id: str) -> None:
        project, prepared, replacement, request, voice_references = (
            self._load_r2v_resume(project_id)
        )
        log = ResearchLog(project)
        node_prefix, artifact_suffix = self._next_resume_identity(
            project, self.store.read(project_id),
        )
        log.event("picsart_resume_started")
        try:
            self._progress(
                project_id,
                57,
                "picsart-resume",
                segments=[self._public_segment(item) for item in prepared],
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(prepared)),
            ) as pool:
                preflight_futures: list[
                    concurrent.futures.Future[PreparedPicsartRequest] | None
                ] = []
                for item in prepared:
                    reusable = item.get("resume_output")
                    if isinstance(reusable, Path):
                        item["final_prompt"] = item["resume_prompt"]
                        item["task_id"] = item.get("resume_task_id")
                        item["status"] = "succeeded"
                        preflight_futures.append(None)
                        continue
                    preflight_futures.append(pool.submit(
                        self._prepare_picsart_segment,
                        project_id,
                        project,
                        item,
                        voice_references,
                        replacement,
                        request,
                        log,
                        final_prompt=item["resume_prompt"],
                        muted=item["resume_muted"],
                        node_prefix=node_prefix,
                        artifact_suffix=artifact_suffix,
                        existing_task_id=item.get("resume_task_id"),
                    ))
                picsart_requests = [
                    None if future is None else future.result()
                    for future in preflight_futures
                ]
                self._begin_submission_batch([
                    prepared_request for prepared_request in picsart_requests
                    if prepared_request is not None
                ])
                futures: list[concurrent.futures.Future[Path] | None] = []
                for item, prepared_request in zip(prepared, picsart_requests):
                    if prepared_request is None:
                        futures.append(None)
                        continue
                    futures.append(pool.submit(
                        self._generate_picsart_segment,
                        prepared_request,
                        item,
                        log,
                    ))
                outputs = [
                    item["resume_output"] if future is None else future.result()
                    for item, future in zip(prepared, futures)
                ]

            final = project / "output" / "generated.mp4"
            self._progress(
                project_id,
                95,
                "stitch-resume",
                segments=[self._public_segment(item) for item in prepared],
            )
            self._node(
                log,
                "stitch-resume",
                "project",
                {
                    "segments": outputs,
                    "target_durations": [item["duration_s"] for item in prepared],
                    "join_modes": [
                        item.get("join_mode", "hard_cut") for item in prepared
                    ],
                    "output": final,
                },
                outputs,
                lambda: media.concat(
                    outputs,
                    final,
                    target_durations=[item["duration_s"] for item in prepared],
                    join_modes=[
                        item.get("join_mode", "hard_cut") for item in prepared
                    ],
                ),
                lambda _value: [final],
            )
            self.store.update(
                project_id,
                status="succeeded",
                has_video=True,
                output_path="output/generated.mp4",
                project_progress={
                    "status": "succeeded",
                    "percent": 100,
                    "stage": "completed",
                },
                generation={
                    "status": "succeeded",
                    "stage": "completed",
                    "error": None,
                },
                segments=[
                    {**self._public_segment(item), "status": "succeeded"}
                    for item in prepared
                ],
                error=None,
            )
            log.event("picsart_resume_succeeded", output=final)
        except Exception as exc:
            log.event("picsart_resume_failed", error=str(exc))
            self.store.update(
                project_id,
                status="failed",
                project_progress={
                    "status": "failed",
                    "percent": 57,
                    "stage": "picsart_resume_failed",
                },
                generation={
                    "status": "failed",
                    "stage": "picsart_resume_failed",
                    "error": "picsart_resume_failed",
                },
                segments=[self._public_segment(item) for item in prepared],
                error={
                    "code": "picsart_resume_failed",
                    "message": "Picsart 后续链路执行失败",
                },
            )

    def _analyze_audio_dialogue(
        self,
        project_id: str,
        project: Path,
        source: Path,
        *,
        has_audio: bool,
        log: ResearchLog,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        audio = project / "work" / "source.wav"
        asr_output = project / "work" / "voice_lines.json"
        if not has_audio:
            reason = "source_has_no_audio"
            self._progress(project_id, 8, "audio")
            self._node(
                log,
                "extract-audio",
                "project",
                {
                    "source": source,
                    "output": audio,
                    "timeline": "video-v0",
                    "has_audio": False,
                },
                [source],
                lambda: self._skipped_result(None, reason),
            )
            self._progress(project_id, 11, "yamnet")
            yamnet = self._node(
                log,
                "yamnet",
                "project",
                {"audio": None, "reason": reason},
                [],
                lambda: self._skipped_result(
                    {"windows": [], "has_bgm": False}, reason,
                ),
            )
            self._progress(project_id, 14, "whisper-asr")
            asr_lines = self._node(
                log,
                "whisper-asr",
                "project",
                {"audio": None, "reason": reason},
                [],
                lambda: self._skipped_result([], reason),
            )
            spoken_lines, voice_decisions = self._node(
                log,
                "spoken-dialogue-filter",
                "project",
                {"asr_lines": [], "yamnet": yamnet, "reason": reason},
                [],
                lambda: self._skipped_result(([], []), reason),
            )
            return yamnet, asr_lines, spoken_lines, voice_decisions

        self._progress(project_id, 8, "audio")
        self._node(
            log,
            "extract-audio",
            "project",
            {"source": source, "output": audio, "timeline": "video-v0"},
            [source],
            lambda: media.extract_audio(source, audio),
            lambda _value: [audio],
        )

        self._progress(project_id, 11, "yamnet")
        yamnet = self._node(
            log,
            "yamnet",
            "project",
            {"audio": audio, "model": self.settings.yamnet_model},
            [audio, self.settings.yamnet_model],
            lambda: media.yamnet_analyze(audio, self.settings.yamnet_model),
        )

        vocal_expected = self._yamnet_has_vocal_signal(yamnet)

        def transcribe_once() -> ProviderResult[list[dict[str, Any]]]:
            lines, record = media.transcribe(
                audio,
                asr_output,
                self.settings.asr_cli,
                self.settings.asr_model,
            )
            return ProviderResult(
                value=lines,
                record=record,
                evidence_media=[asr_output],
            )

        self._progress(project_id, 14, "whisper-asr")
        asr_lines = self._node(
            log,
            "whisper-asr",
            "project",
            {
                "audio": audio,
                "cli": self.settings.asr_cli,
                "model": self.settings.asr_model,
                "vocal_signal_observed": vocal_expected,
            },
            [audio, self.settings.asr_model],
            transcribe_once,
        )
        if vocal_expected and not asr_lines:
            log.event(
                "empty_asr_accepted",
                node="whisper-asr",
                scope="project",
                attempts=1,
            )
        spoken_lines, voice_decisions = self._node(
            log,
            "spoken-dialogue-filter",
            "project",
            {"asr_lines": asr_lines, "yamnet": yamnet},
            [],
            lambda: media.filter_spoken_lines(asr_lines, yamnet),
        )
        return yamnet, asr_lines, spoken_lines, voice_decisions

    @staticmethod
    def _completed_node_value(project: Path, node: str, scope: str) -> Any:
        attempts = project / "logs" / node / scope
        for directory in sorted(attempts.glob("attempt-*"), reverse=True):
            output = directory / "output.json"
            if not output.is_file() or (directory / "error.json").exists():
                continue
            value = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "value" in value:
                return value["value"]
            return value
        raise FileNotFoundError(f"completed node output is missing: {node}/{scope}")

    def _prepare_segment(
        self,
        project_id: str,
        project: Path,
        source: Path,
        segment: dict[str, Any],
        selection: list[dict[str, Any]],
        request: dict[str, Any],
        log: ResearchLog,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        del project_id
        index = int(segment["index"])
        scope = self._segment_scope(index)
        directory = project / "work" / "segments" / f"{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        segment_source = directory / "source.mp4"
        segment_frames = directory / "source-4fps"
        raw_dir = directory / "keyframes"
        if resume:
            if not segment_source.is_file():
                raise FileNotFoundError("reusable segment source is missing")
            segment_manifest = self._completed_node_value(
                project, "segment-frames-4fps", scope,
            )
            if not isinstance(segment_manifest, dict):
                raise TypeError("reusable segment manifest is invalid")
            frames = sorted(raw_dir.glob("*.png"))
            if len(frames) != 9:
                raise ValueError("reusable segment must contain exactly 9 keyframes")
            log.event("segment_preparation_reused", scope=scope)
        else:
            self._node(
                log,
                "segment-cut",
                scope,
                {"source": source, **segment},
                [source],
                lambda: media.cut_segment(
                    source,
                    float(segment["start_s"]),
                    float(segment["duration_s"]),
                    segment_source,
                ),
                lambda _value: [segment_source],
            )
            segment_manifest = self._node(
                log,
                "segment-frames-4fps",
                scope,
                {"source": segment_source, "fps": 4, "output": segment_frames},
                [segment_source],
                lambda: media.extract_frames_4fps(segment_source, segment_frames),
                lambda value: self._manifest_media(segment_frames, value),
            )
            frames = self._node(
                log,
                "keyframes-exact-nine",
                scope,
                {"source": source, "selection": selection, "count": 9},
                [source],
                lambda: media.materialize_keyframes(source, selection, raw_dir),
                lambda paths: list(paths),
            )
            (directory / "keyframe_selection.json").write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        processing = request.get("processing") or {}
        generation_config = self._generation_config(request)
        stages: list[str] = []
        if processing.get("remove_subtitle"):
            stages.append("full_screen_text_erase")
        if processing.get("remove_logo"):
            stages.append("full_screen_icon_erase")
        use_mediakit = (
            self.settings.enable_mediakit
            and processing.get("optimize_image", True)
            and bool(stages)
        )
        authoritative_frames = list(frames)
        if use_mediakit:
            optimized = directory / "optimized-keyframes"
            optimized.mkdir(exist_ok=True)
            authoritative_frames = []
            for order, frame in enumerate(frames, 1):
                target = optimized / frame.name
                if resume and target.is_file():
                    authoritative_frames.append(target)
                    log.event(
                        "mediakit_artifact_reused",
                        scope=f"{scope}-frame-{order:02d}",
                        output=target,
                    )
                    continue
                erase_source = frame
                remaining_stages = stages
                node = "mediakit-erase"
                if resume:
                    node = "mediakit-erase-resume"
                    for completed in range(1, len(stages)):
                        checkpoint = target.with_name(
                            f"{target.stem}.mediakit-stage-{completed:02d}{target.suffix}"
                        )
                        if checkpoint.is_file():
                            erase_source = checkpoint
                            remaining_stages = stages[completed:]
                result = self._node(
                    log,
                    node,
                    f"{scope}-frame-{order:02d}",
                    {"source": erase_source, "scenes": remaining_stages},
                    [erase_source],
                    lambda erase_source=erase_source,
                    target=target,
                    remaining_stages=remaining_stages: self._erase_with_limit(
                        erase_source, target, remaining_stages,
                    ),
                    lambda path: [path],
                )
                authoritative_frames.append(result)

        proxy_dir = directory / "analysis-keyframes"
        proxy_dir.mkdir(exist_ok=True)

        def make_proxies() -> list[Path]:
            proxies: list[Path] = []
            for frame in authoritative_frames:
                target = proxy_dir / frame.name
                proxies.append(media.half_resolution_png(frame, target))
            return proxies

        reusable_proxies = sorted(proxy_dir.glob("*.png")) if resume else []
        if len(reusable_proxies) == 9:
            analysis_frames = reusable_proxies
            log.event("analysis_proxies_reused", scope=scope)
        else:
            analysis_frames = self._node(
                log,
                "video-maker-analysis-proxies-resume"
                if resume else "video-maker-analysis-proxies",
                scope,
                {"frames": authoritative_frames, "scale": "half-resolution"},
                authoritative_frames,
                make_proxies,
                lambda paths: list(paths),
            )
        visual_payload = {
            "generation_config": generation_config,
            "segment": {
                key: value
                for key, value in segment.items()
                if key not in {"visual_prompt", "dialogue"}
            },
            "manifest": segment_manifest,
            "frames": [
                {
                    "frame_order": order,
                    "path": path.relative_to(project).as_posix(),
                    "sha256": self._sha256(path),
                }
                for order, path in enumerate(analysis_frames, 1)
            ],
        }
        visual_path = directory / "visual_prompt.txt"
        visual_prompt = visual_path.read_text(encoding="utf-8").strip() if resume and visual_path.is_file() else ""
        if visual_prompt:
            log.event("video_maker_artifact_reused", scope=scope)
        else:
            skill = VIDEO_SKILL.read_text(encoding="utf-8")
            visual_prompt = self._node(
                log,
                "video-maker-resume" if resume else "video-maker",
                scope,
                visual_payload,
                analysis_frames,
                lambda: self.deepseek.complete_structured(
                    skill
                    + "\n\n本次只执行逐段视觉分析；最终输出严格遵循本次注入的 JSON Schema。",
                    visual_payload,
                    VISUAL_PROMPT_SCHEMA,
                    analysis_frames,
                ),
                transform=normalize_visual_prompt,
            )
            visual_path.write_text(visual_prompt, encoding="utf-8")
        return {
            **segment,
            "source": segment_source,
            "keyframe_paths": authoritative_frames,
            "analysis_paths": analysis_frames,
            "visual_prompt": visual_prompt,
            "status": "running",
        }

    def _translation_request(
        self,
        lines: list[dict[str, Any]],
        target_language: str,
        visual_prompts: str,
    ) -> ProviderResult[Any]:
        if not lines:
            return ProviderResult(
                value={"lines": []},
                record={"outcome": "skipped", "reason": "no_spoken_dialogue"},
                evidence_media=[],
            )
        if target_language == SAME_AS_SOURCE_LANGUAGE:
            return ProviderResult(
                value={
                    "lines": [{"text": str(line["text"])} for line in lines],
                },
                record={"outcome": "skipped", "reason": "same_language"},
                evidence_media=[],
            )
        instruction = (
            f"把台词翻译并自然改写为{target_language}。保持输入行顺序；"
            "输出行数必须与输入完全一致，lines 每项只含非空 text。"
            "最终输出严格遵循本次注入的 JSON Schema。"
            "视频描述仅用于语境。"
        )
        return self.deepseek.complete_structured(
            instruction,
            {"lines": lines, "video_prompts": visual_prompts},
            TRANSLATION_SCHEMA,
        )

    @staticmethod
    def _normalize_translation(
        value: Any, source_lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Structured output owns the ideal shape. Do not turn translation
        # quality into a second local gate: unusable/missing entries fall back
        # to their source text, and surplus entries are ignored.
        parsed = value.get("lines") if isinstance(value, dict) else None
        candidates = parsed if isinstance(parsed, list) else []
        translated: list[dict[str, Any]] = []
        for index, source in enumerate(source_lines):
            candidate = candidates[index] if index < len(candidates) else None
            text = candidate.get("text") if isinstance(candidate, dict) else None
            if not isinstance(text, str) or not text.strip():
                text = str(source.get("text") or "")
            translated.append({
                **source,
                "text": text.strip(),
            })
        return translated

    @staticmethod
    def _project_index_request(
        project: Path,
        prepared: list[dict[str, Any]],
        generation_config: dict[str, bool],
    ) -> tuple[dict[str, Any], list[Path]]:
        segments: list[dict[str, Any]] = []
        images: list[Path] = []
        for item in sorted(prepared, key=lambda value: int(value["index"])):
            frames = []
            for order, path in enumerate(item["analysis_paths"], 1):
                frames.append({
                    "frame_order": order,
                    "path": path.relative_to(project).as_posix(),
                    "sha256": Pipeline._sha256(path),
                })
                images.append(path)
            segments.append({
                "segment_index": int(item["index"]),
                "visual_prompt": item["visual_prompt"],
                "frames": frames,
            })
        return {
            "phase": "project_index",
            "generation_config": generation_config,
            "segments": segments,
        }, images

    @staticmethod
    def _one_line(value: object, fallback: str = "unspecified") -> str:
        if not isinstance(value, str):
            return fallback
        normalized = " ".join(value.split())
        return normalized or fallback

    @staticmethod
    def _appears_in_segment(record: dict[str, Any], segment_index: int) -> bool:
        occurrences = record.get("occurrences")
        if not isinstance(occurrences, list):
            # If occurrence metadata is unusable, including the record is less
            # destructive than blocking the project or silently losing it.
            return True
        return any(
            isinstance(occurrence, dict)
            and occurrence.get("segment_index") == segment_index
            for occurrence in occurrences
        )

    @classmethod
    def _fusion_authority_block(
        cls, element_index: dict[str, Any], segment_index: int,
    ) -> str:
        lines = [
            "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN",
            (
                "This block is the sole authority. Ignore any conflicting "
                "replacement or voice statement elsewhere in the prompt."
            ),
        ]
        replaced_ids: set[str] = set()
        for category in ("people", "entities", "scenes"):
            records = element_index.get(category)
            if not isinstance(records, dict):
                continue
            for stable_id, record in records.items():
                if not isinstance(stable_id, str) or not isinstance(record, dict):
                    continue
                if isinstance(record.get("replacement"), dict):
                    replaced_ids.add(stable_id)
                if not cls._appears_in_segment(record, segment_index):
                    continue
                source = cls._one_line(record.get("source_visual_description"))
                replacement = record.get("replacement")
                if isinstance(replacement, dict):
                    if category == "people":
                        preserve_text = (
                            "source role, action, pose, position, and continuity; "
                            "only explicitly stated gender, race, overall style, and "
                            "primary clothing color"
                        )
                    elif category == "entities":
                        preserve_text = (
                            "source function, action interface, scale, contact, and "
                            "relation semantics"
                        )
                    else:
                        preserve_text = (
                            "source scene type, color tone, composition, camera "
                            "position and angle, shot movement, focal length, and lighting"
                        )
                    target = cls._one_line(
                        replacement.get("target_visual_description")
                    )
                    reference_text = (
                        "Image 1"
                        if replacement.get("source") == "image_1"
                        else "generated variant"
                    )
                    line = (
                        f"- {stable_id} | REPLACE | source: {source} | target: "
                        f"{target} | reference: {reference_text} | preserve: "
                        f"{preserve_text}"
                    )
                else:
                    preserve = record.get("preserve")
                    preserve_items = (
                        [
                            item for item in preserve
                            if isinstance(item, str) and item.strip()
                        ]
                        if isinstance(preserve, list)
                        else []
                    )
                    preserve_text = "; ".join(
                        cls._one_line(item) for item in preserve_items
                    ) or "source action, geometry, scale, and continuity"
                    line = (
                        f"- {stable_id} | KEEP | source: {source} | preserve: "
                        f"{preserve_text}"
                    )
                if category == "people":
                    voice = record.get("voice")
                    if isinstance(voice, dict) and isinstance(
                        voice.get("audio_reference"), str
                    ) and voice["audio_reference"].strip():
                        line += (
                            f" | voice: {cls._one_line(voice['audio_reference'])}"
                            f" | voice features: {cls._one_line(voice.get('features'))}"
                            " | use this voice only when this stable person speaks; "
                            "ignore the reference audio words"
                        )
                    if isinstance(replacement, dict):
                        line += (
                            " | person policy: different face; same explicit gender, "
                            "race, overall style, and clothing color; visibly different "
                            "clothing design; do not infer missing traits"
                        )
                elif category == "scenes" and isinstance(replacement, dict):
                    line += (
                        " | scene policy: same scene type with a clearly different "
                        "layout and details; preserve color tone, composition, camera, "
                        "shot movement, focal length, and lighting"
                    )
                elif category == "entities" and isinstance(replacement, dict):
                    line += (
                        " | entity policy: preserve function, action interface, scale, "
                        "contact, and relation semantics"
                    )
                lines.append(line)

        relations = element_index.get("relations")
        relation_items = relations.items() if isinstance(relations, dict) else ()
        for stable_id, relation in relation_items:
            if not isinstance(stable_id, str) or not isinstance(relation, dict):
                continue
            if not cls._appears_in_segment(relation, segment_index):
                continue
            subject = cls._one_line(relation.get("subject_key"))
            predicate = cls._one_line(relation.get("predicate"))
            object_key = cls._one_line(relation.get("object_key"))
            preserve = relation.get("preserve")
            preserve_items = (
                [item for item in preserve if isinstance(item, str) and item.strip()]
                if isinstance(preserve, list)
                else []
            )
            preserve_text = "; ".join(
                cls._one_line(item) for item in preserve_items
            ) or "source relation state and geometry"
            replace_together = relation.get("replace_together")
            replace_together_text = (
                str(replace_together).lower()
                if isinstance(replace_together, bool)
                else "unspecified"
            )
            if subject in replaced_ids or object_key in replaced_ids:
                lines.append(
                    f"- {stable_id} | RELATION | endpoints: {subject} -> "
                    f"{object_key} | preserve: interaction type, contact, relative "
                    "position, and motion continuity | replace together: "
                    f"{replace_together_text}"
                )
            else:
                lines.append(
                    f"- {stable_id} | RELATION | {subject} {predicate} {object_key} "
                    f"| preserve: {preserve_text} | replace together: "
                    f"{replace_together_text}"
                )
        lines.append("END AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN")
        return "\n".join(lines)

    def _prepare_h3_segment(
        self,
        project: Path,
        item: dict[str, Any],
        element_index: dict[str, Any],
        request: dict[str, Any],
        log: ResearchLog,
    ) -> tuple[dict[str, Any], str, Path, str]:
        index = int(item["index"])
        scope = self._segment_scope(index)
        try:
            skill = R2V_SKILL.read_text(encoding="utf-8")
            fusion_payload = {
                "segment_index": index,
                "visual_prompt": item["visual_prompt"],
                "global_element_index": element_index,
                "translated_dialogue": item["dialogue"],
                "authoritative_constraints": self._fusion_authority_block(
                    element_index, index,
                ),
            }
            fusion_schema, fusion_input_sha256 = fusion_prompt_schema(
                fusion_payload,
            )
            fusion_agent, fusion_client = self._fusion_client_for_request(request)
            prompt = self._node(
                log,
                "h3-max-r2v-prompt",
                scope,
                {
                    "agent": fusion_agent,
                    "skill": skill,
                    "input_sha256": fusion_input_sha256,
                    "payload": fusion_payload,
                    "output_schema": fusion_schema,
                },
                [],
                lambda: fusion_client.complete_structured(
                    skill,
                    fusion_payload,
                    fusion_schema,
                    [],
                ),
                transform=lambda value: normalize_fusion_prompt(
                    value,
                    input_sha256=fusion_input_sha256,
                    segment_index=index,
                    authoritative_constraints=fusion_payload[
                        "authoritative_constraints"
                    ],
                ),
            )
            dialogue = "\n".join(
                f"[{line['start_s']:.3f}-{line['end_s']:.3f}s] {line['text']}"
                for line in item["dialogue"]
            )
            final_prompt = prompt + (
                "\n\nDialogue requirements: Speak the following lines naturally in "
                "the target language. Do not speak the original-language dialogue. "
                "Do not render captions or subtitles. Finish every line within the "
                "frozen source-segment time window. Do not add dialogue in padded "
                "time.\n" + dialogue
                if dialogue
                else "\n\nDo not add captions, subtitles, or new dialogue."
            )
            segment_dir = item["source"].parent
            (segment_dir / "h3_prompt.txt").write_text(
                final_prompt, encoding="utf-8",
            )
            muted = segment_dir / "source-muted.mp4"
            self._node(
                log,
                "mute-reference-video",
                scope,
                {"source": item["source"], "output": muted},
                [item["source"]],
                lambda: media.mute(item["source"], muted),
                lambda _value: [muted],
            )
            return item, final_prompt, muted, fusion_input_sha256
        except Exception:
            item["status"] = "failed"
            raise

    @staticmethod
    def _validated_provider_asset_url(value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("prepared provider asset URL is invalid")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
            raise ValueError("prepared provider asset URL must be public HTTPS")
        return value

    @staticmethod
    def _validate_reference_video(path: Path) -> None:
        info = media.probe(path)
        duration = info.get("duration_s") if isinstance(info, dict) else None
        width = info.get("width") if isinstance(info, dict) else None
        height = info.get("height") if isinstance(info, dict) else None
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not PICSART_REFERENCE_MIN_DURATION_S
            <= float(duration)
            <= PICSART_REFERENCE_MAX_DURATION_S
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
        ):
            raise ValueError("prepared Picsart reference video is invalid")

    def _prepare_picsart_segment(
        self,
        project_id: str,
        project: Path,
        item: dict[str, Any],
        voice_references: list[Path],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
        *,
        final_prompt: str,
        muted: Path,
        node_prefix: str = "picsart",
        artifact_suffix: str = "",
        existing_task_id: str | None = None,
    ) -> PreparedPicsartRequest:
        index = item.get("index")
        duration = item.get("duration_s")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 1
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("Picsart segment metadata is invalid")
        scope = self._segment_scope(index)
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        muted_receipt = self._project_artifact_receipt(
            project,
            muted,
            f"work/segments/{index:04d}/source-muted.mp4",
        )
        self._validate_reference_video(muted)
        if not isinstance(final_prompt, str):
            raise ValueError("prepared Picsart prompt is invalid")
        final_prompt = final_prompt.strip()
        if not final_prompt or len(final_prompt) > PICSART_PROMPT_MAX_CHARS:
            raise ValueError("prepared Picsart prompt is invalid")
        if len(voice_references) > MAX_REFERENCE_AUDIO:
            raise ValueError("H3 accepts at most three reference voices")
        project_root = project.resolve(strict=True)
        resolved_voice_references: list[Path] = []
        for slot, path in enumerate(voice_references, 1):
            resolved = path.resolve(strict=True)
            valid_paths = {
                (project / "work" / "voice-reference.wav").resolve(),
                (
                    project / "work" / "voice-references"
                    / f"audio-{slot:02d}.wav"
                ).resolve(),
            }
            if (
                project_root not in resolved.parents
                or resolved not in valid_paths
                or path.is_symlink()
                or not resolved.is_file()
                or resolved.stat().st_size < 1
            ):
                raise ValueError("H3 reference voice file is invalid")
            resolved_voice_references.append(resolved)
        if len(resolved_voice_references) != len(set(resolved_voice_references)):
            raise ValueError("H3 reference voice files must be unique")
        if existing_task_id is not None and (
            not isinstance(existing_task_id, str)
            or not existing_task_id
            or existing_task_id != existing_task_id.strip()
        ):
            raise ValueError("stored Picsart task id is invalid")
        output_request = request.get("output") or {}
        if not isinstance(output_request, dict):
            raise ValueError("stored Picsart output request is invalid")
        resolution = output_request.get("resolution", "768p")
        aspect_ratio = output_request.get("aspect_ratio", "adaptive")
        if not isinstance(resolution, str) or resolution not in PICSART_RESOLUTIONS:
            raise ValueError("Picsart resolution is unsupported")
        if (
            not isinstance(aspect_ratio, str)
            or aspect_ratio not in PICSART_ASPECT_RATIOS
        ):
            raise ValueError("Picsart aspect ratio is unsupported")
        provider_duration = max(
            PICSART_GENERATION_MIN_DURATION_S,
            math.ceil(float(duration)),
        )
        if not (
            PICSART_GENERATION_MIN_DURATION_S
            <= provider_duration
            <= PICSART_GENERATION_MAX_DURATION_S
        ):
            raise ValueError("Picsart generation duration is unsupported")

        provider_replacement: Path | None = None
        video_url: str | None = None
        image_url: str | None = None
        audio_urls: list[str] = []
        if existing_task_id is None:
            if replacement is not None:
                canonical_replacement = project / "inputs" / "replacement-image"
                resolved_replacement = replacement.resolve(strict=True)
                if (
                    resolved_replacement
                    != canonical_replacement.resolve(strict=True)
                    or project_root not in resolved_replacement.parents
                    or replacement.is_symlink()
                ):
                    raise ValueError("replacement image path is invalid")
                provider_replacement = (
                    segment_dir / f"provider-reference{artifact_suffix}.png"
                )
                self._node(
                    log,
                    f"{node_prefix}-reference-image",
                    scope,
                    {"source": replacement, "output": provider_replacement},
                    [replacement],
                    lambda: media.provider_reference_png(
                        replacement, provider_replacement,
                    ),
                    lambda _value: [provider_replacement],
                )
                provider_bytes = provider_replacement.read_bytes()
                if (
                    not provider_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                    or len(provider_bytes) < 9
                ):
                    raise ValueError("prepared provider reference is not PNG")
            video_url = self._validated_provider_asset_url(
                self.register_asset(project_id, muted)
            )
            if provider_replacement is not None:
                image_url = self._validated_provider_asset_url(
                    self.register_asset(project_id, provider_replacement)
                )
            audio_urls = [
                self._validated_provider_asset_url(
                    self.register_asset(project_id, voice_reference)
                )
                for voice_reference in resolved_voice_references
            ]
            if len(audio_urls) != len(set(audio_urls)):
                raise ValueError("prepared Picsart audio URLs must be unique")
        request_payload = (
            {
                "existing_task_id": existing_task_id,
                "segment_index": index,
                "muted_video": muted_receipt,
            }
            if existing_task_id is not None
            else {
                "workflow": PicsartClient.WORKFLOW,
                "prompt": final_prompt,
                "reference_video_urls": [video_url],
                "reference_image_urls": [image_url] if image_url else [],
                "reference_audio_urls": audio_urls,
                "resolution": resolution.upper(),
                "duration": provider_duration,
                "aspect_ratio": aspect_ratio,
                "prompt_expansion_mode": "quality",
            }
        )
        prepared = PreparedPicsartRequest(
            project_id=project_id,
            segment_index=index,
            scope=scope,
            run_identity=node_prefix,
            artifact_suffix=artifact_suffix,
            final_prompt=final_prompt,
            muted=muted.resolve(strict=True),
            provider_replacement=(
                provider_replacement.resolve(strict=True)
                if provider_replacement is not None else None
            ),
            voice_references=tuple(resolved_voice_references),
            video_url=video_url,
            image_url=image_url,
            audio_urls=tuple(audio_urls),
            provider_duration=provider_duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            target_duration_s=float(duration),
            request_sha256=self._json_sha256(request_payload),
            existing_task_id=existing_task_id,
        )
        log.event(
            "picsart_preflight_succeeded",
            segment_index=index,
            run_identity=node_prefix,
            request_sha256=prepared.request_sha256,
            reuses_task=existing_task_id is not None,
        )
        return prepared

    def _submit_prepared_picsart(
        self,
        prepared: PreparedPicsartRequest,
    ) -> ProviderResult[str]:
        if prepared.existing_task_id is not None or prepared.video_url is None:
            raise ValueError("prepared Picsart request is not submittable")
        try:
            result = self.picsart.submit(
                prompt=prepared.final_prompt,
                video_url=prepared.video_url,
                image_url=prepared.image_url,
                audio_urls=list(prepared.audio_urls),
                duration=prepared.provider_duration,
                resolution=prepared.resolution,
                aspect_ratio=prepared.aspect_ratio,
            )
            if (
                not isinstance(result, ProviderResult)
                or not isinstance(result.value, str)
                or not result.value.strip()
            ):
                raise ValueError("Picsart submit returned no task id")
            self._transition_submission_receipt(
                prepared, status="accepted", task_id=result.value,
            )
            return result
        except ProviderFailure as exc:
            outcome = exc.record.get("outcome")
            self._transition_submission_receipt(
                prepared,
                status="rejected" if outcome == "rejected" else "unknown",
            )
            raise
        except BaseException:
            try:
                self._transition_submission_receipt(
                    prepared, status="unknown",
                )
            except BaseException:
                # The pre-POST submission_started receipt remains fail-closed.
                pass
            raise

    def _generate_picsart_segment(
        self,
        prepared: PreparedPicsartRequest,
        item: dict[str, Any],
        log: ResearchLog,
    ) -> Path:
        index = prepared.segment_index
        scope = prepared.scope
        segment_dir = prepared.muted.parent
        node_prefix = prepared.run_identity
        artifact_suffix = prepared.artifact_suffix
        try:
            task_id = prepared.existing_task_id
            if task_id is None:
                task_id = self._node(
                    log,
                    f"{node_prefix}-submit",
                    scope,
                    prepared.submit_input(),
                    [prepared.muted]
                    + (
                        [prepared.provider_replacement]
                        if prepared.provider_replacement is not None else []
                    )
                    + list(prepared.voice_references),
                    lambda: self._submit_prepared_picsart(prepared),
                    retry_if=lambda _exc: False,
                )
            else:
                log.event(
                    "picsart_task_reused",
                    segment_index=index,
                    task_id=task_id,
                )

            def poll_once() -> ProviderResult[str]:
                try:
                    return self.picsart.poll(task_id)
                except ProviderFailure as exc:
                    if self._poll_record_is_terminal_failure(exc.record):
                        self._mark_submission_task_failed(
                            prepared.project_id, index, task_id,
                        )
                    raise

            generated_url = self._node(
                log,
                f"{node_prefix}-poll",
                scope,
                {"task_id": task_id},
                [],
                poll_once,
                retry_if=lambda exc: not (
                    isinstance(exc, ProviderFailure)
                    and self._poll_record_is_terminal_failure(exc.record)
                ),
            )
            downloaded = segment_dir / f"h3-provider{artifact_suffix}.mp4"
            self._node(
                log,
                f"{node_prefix}-download",
                scope,
                {"url": generated_url, "task_id": task_id},
                [],
                lambda: self.picsart.download(generated_url, downloaded),
                lambda path: [path],
            )
            trimmed = segment_dir / f"generated{artifact_suffix}.mp4"
            trim_node = (
                "trim-provider-output-24fps"
                if node_prefix == "picsart"
                else f"{node_prefix}-trim-provider-output-24fps"
            )
            self._node(
                log,
                trim_node,
                scope,
                {
                    "source": downloaded,
                    "target_duration_s": prepared.target_duration_s,
                },
                [downloaded],
                lambda: media.trim_24fps(
                    downloaded, prepared.target_duration_s, trimmed,
                ),
                lambda _value: [trimmed],
            )
            item["final_prompt"] = prepared.final_prompt
            item["task_id"] = task_id
            item["status"] = "succeeded"
            return trimmed
        except Exception:
            item["status"] = "failed"
            raise

    def _generation_config(self, request: dict[str, Any]) -> dict[str, bool]:
        processing = request.get("processing") or {}
        enabled = self.settings.enable_mediakit and bool(
            processing.get("optimize_image", True)
        )
        return {
            "remove_subtitle": enabled and bool(
                processing.get("remove_subtitle")
            ),
            "remove_watermark": enabled and bool(
                processing.get("remove_logo")
            ),
        }

    @staticmethod
    def _skipped_result(value: Any, reason: str) -> ProviderResult[Any]:
        return ProviderResult(
            value=value,
            record={"outcome": "skipped", "reason": reason},
            evidence_media=[],
        )

    @staticmethod
    def _yamnet_has_vocal_signal(analysis: dict[str, Any]) -> bool:
        return any(
            max(float(window.get("spoken") or 0), float(window.get("sung") or 0))
            >= MASTER_EMPTY_ASR_SIGNAL
            for window in analysis.get("windows") or []
        )

    @staticmethod
    def _manifest_media(directory: Path, manifest: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for frame in manifest.get("frames") or []:
            name = frame.get("file") if isinstance(frame, dict) else None
            if isinstance(name, str):
                paths.append(directory / name)
        return paths

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json_sha256(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _segment_scope(index: int) -> str:
        return f"segment-{index:04d}"

    @staticmethod
    def _public_segment(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": item["index"],
            "start_s": item["start_s"],
            "end_s": item["end_s"],
            "duration_s": item["duration_s"],
            "chain_id": item.get("chain_id"),
            "join_mode": item.get("join_mode"),
            "status": item.get("status", "running"),
            "visual_prompt": item.get("visual_prompt"),
            "final_prompt": item.get("final_prompt"),
            "dialogue": item.get("dialogue", []),
        }
