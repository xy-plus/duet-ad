from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from app import media
from app import face_mask
from app.config import Settings
from app.fusion_schema import fusion_prompt_schema, normalize_fusion_prompt
from app.object_replace_schema import (
    MAX_REFERENCE_AUDIO,
    OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index,
    normalize_object_replace,
    object_replace_planning_context,
    object_replace_schema,
)
from app.replacement_sheet import (
    build_replacement_sheet_spec,
    finalize_replacement_sheet,
    materialize_replacement_sheet_inputs,
    replacement_sheet_spec_sha256,
    validate_replacement_sheet_spec,
)
from app.providers import (
    AutodlClient,
    CodexClient,
    CodexProcessError,
    DeepSeekClient,
    MediaKitClient,
    PicsartClient,
    ProviderFailure,
    ProviderResult,
)
from app.seedream import (
    SEEDREAM_OUTPUT_HEIGHT,
    SEEDREAM_OUTPUT_SIZE,
    SEEDREAM_OUTPUT_WIDTH,
    SeedreamClient,
    _validated_image_media_type,
    reset_attempt as _seedream_reset_attempt,
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
DIALOGUE_WRITER_SKILL = (
    Path(__file__).resolve().parents[1] / "skills" / "dialogue-writer" / "SKILL.md"
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
DIALOGUE_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["segment_index", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def _validate_translation_shape(value: dict[str, Any]) -> None:
    """Transport-shape gate for the JSON repair loop; content stays lenient."""
    lines = value.get("lines")
    if not isinstance(lines, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str)
        for item in lines
    ):
        raise ValueError("translation output is invalid")


def _validate_dialogue_generation_shape(value: dict[str, Any]) -> None:
    """Transport-shape gate for the JSON repair loop; content stays lenient."""
    lines = value.get("lines")
    if not isinstance(lines, list) or any(
        not isinstance(item, dict)
        or type(item.get("segment_index")) is not int
        or not isinstance(item.get("text"), str)
        for item in lines
    ):
        raise ValueError("dialogue generation output is invalid")

PICSART_RECEIPT_VERSION = 2
H3_PREPARED_ARTIFACTS_VERSION = 6
REPLACEMENT_SHEET_VERSION = 4
PICSART_PROMPT_MAX_CHARS = 50_000
PICSART_REFERENCE_MIN_DURATION_S = 2
PICSART_REFERENCE_MAX_DURATION_S = 15
PICSART_GENERATION_MIN_DURATION_S = 5
PICSART_GENERATION_MAX_DURATION_S = 15
PICSART_RESOLUTIONS = frozenset({"480p", "768p"})
PICSART_ASPECT_RATIOS = frozenset({"16:9", "9:16"})
# Provider-side fetch/validation failure markers classify why a terminal H3
# task failure happened; every terminal failure is resubmitted under a fresh
# run identity (a resubmit also means fresh asset URLs, sidestepping the
# 24h provider asset TTL), so there is no resubmit cap.
_PICSART_FETCH_FAILURE_MARKERS = (
    "failed to download",
    "unreachable",
    "validationerror",
)
# Face masking keeps H3 identity takeover working, so it retries a few times
# before degrading to the unmasked reference (never blocking the render).
FACE_MASK_MAX_ATTEMPTS = 3


class PicsartTaskHungError(RuntimeError):
    """A submitted Picsart task stayed non-terminal past the hang limit."""


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
    provider_images: tuple[Path, ...]
    voice_references: tuple[Path, ...]
    video_url: str | None
    image_urls: tuple[str, ...]
    audio_urls: tuple[str, ...]
    provider_duration: int
    resolution: str
    aspect_ratio: str
    target_duration_s: float
    request_sha256: str
    preparation_sha256: str | None
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
            "reference_image_urls": list(self.image_urls),
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
        seedream: SeedreamClient | None = None,
        picsart: PicsartClient | None = None,
        autodl: AutodlClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.register_asset = register_asset
        self.codex = codex or CodexClient(settings)
        self.deepseek = deepseek or DeepSeekClient(settings)
        self.mediakit = mediakit or MediaKitClient(settings)
        self.seedream = seedream or SeedreamClient(settings)
        self.picsart = picsart or PicsartClient(settings)
        self.autodl = autodl or AutodlClient(settings)
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
        """Run one node, retrying retryable failures until they succeed.

        Attempts 2 and 3 rerun immediately; from attempt 4 on the wait backs
        off exponentially from one minute (capped at thirty minutes). Nodes
        with an explicit ``retry_if`` keep their own gate (paid submissions
        with receipt semantics). Transport evidence is kept per attempt.
        Resumed nodes continue from the highest existing attempt number so
        the failed attempts' evidence is never overwritten.
        """
        last_error: Exception | None = None
        attempt = log.next_attempt(node, scope) - 1
        moderation_fingerprint: str | None = None
        moderation_streak = 0
        while True:
            attempt += 1
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
            should_retry = (
                retry_if(last_error)
                if retry_if is not None
                else self._retryable_node_error(last_error)
            )
            if not should_retry:
                break
            if isinstance(last_error, ProviderFailure):
                moderation_text = (
                    json.dumps(
                        last_error.record, ensure_ascii=False, default=str,
                    )
                    + str(last_error)
                ).lower()
                if any(
                    marker in moderation_text
                    for marker in self._NODE_FATAL_MARKERS
                ):
                    fingerprint = hashlib.sha256(
                        moderation_text.encode("utf-8")
                    ).hexdigest()[:16]
                    moderation_streak = (
                        moderation_streak + 1
                        if fingerprint == moderation_fingerprint
                        else 1
                    )
                    moderation_fingerprint = fingerprint
                    if moderation_streak >= 5:
                        log.event(
                            "node_moderation_persistent",
                            node=node,
                            scope=scope,
                            attempt=attempt,
                            fingerprint=fingerprint,
                            consecutive=moderation_streak,
                        )
            delay = self._node_retry_delay(attempt)
            if delay:
                log.event(
                    "node_retry_wait",
                    node=node,
                    scope=scope,
                    attempt=attempt,
                    delay_s=delay,
                )
                self._retry_sleep(delay)
        assert last_error is not None
        raise last_error

    _NODE_FATAL_MARKERS = (
        "image_rejected",
        "content policy",
        "moderation",
        "审核",
    )

    @classmethod
    def _retryable_node_error(cls, exc: Exception) -> bool:
        """Provider, network, subprocess and transient model-output failures
        retry; input and business validation errors don't.  Moderation
        rejections also retry with backoff instead of failing the project."""
        if isinstance(exc, ProviderFailure):
            return True
        if isinstance(exc, (
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
            media.CommandError,
            CodexProcessError,
        )):
            return True
        # Any ValueError/MediaError from node execution (model-output or
        # media-consumption validation) is treated as transient: retrying
        # lets nondeterministic generation self-heal.  AssertionError covers
        # defensive leftovers.  TypeError/KeyError/AttributeError stay fatal
        # (programmer-bug semantics).
        if isinstance(exc, (ValueError, media.MediaError, AssertionError)):
            return True
        return False

    @staticmethod
    def _node_retry_delay(failed_attempt: int) -> float:
        """Attempts 2-3 rerun immediately; from attempt 4 on, back off
        exponentially from one minute, capped at thirty minutes."""
        if failed_attempt < 3:
            return 0.0
        return min(60.0 * 2 ** (failed_attempt - 3), 1800.0)

    def _retry_sleep(self, seconds: float) -> None:
        time.sleep(seconds)

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

    def _erase_source_subtitles(
        self,
        project_id: str,
        project: Path,
        source: Path,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> Path:
        """Erase burned-in subtitles from the source before any downstream use."""
        processing = request.get("processing") if isinstance(request, dict) else None
        if not (
            isinstance(processing, dict)
            and processing.get("remove_subtitle") is True
        ):
            return source
        erased = project / "work" / "source-nosubs.mp4"

        def operation() -> ProviderResult[Path]:
            video_url = self.register_asset(project_id, source)
            submitted = self.mediakit.submit_video_subtitle_erase(video_url)
            record: dict[str, Any] = {
                "submit": submitted.record,
                "video_url": video_url,
            }
            polled = self.mediakit.poll_task(submitted.value)
            record["poll"] = polled.record
            downloaded = self.mediakit.download(polled.value, erased)
            record["download"] = downloaded.record
            return ProviderResult(
                erased, record, list(downloaded.evidence_media),
            )

        self._progress(project_id, 3, "erase-source-subtitles")
        self._node(
            log,
            "erase-source-subtitles",
            "project",
            {
                "source": source,
                "erase_mode": self.settings.mediakit_video_erase_mode,
                "output": erased,
            },
            [source],
            operation,
            lambda value: [value],
        )
        return erased

    def run(self, project_id: str) -> None:
        project = self.store.path(project_id)
        log = ResearchLog(project)
        state = self.store.read(project_id)
        source, replacement = self._stored_input_paths(project, state)
        request = self._validated_effective_request(
            state.get("effective_request")
        )
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

            source = self._erase_source_subtitles(
                project_id, project, source, request, log,
            )

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
            self._validated_join_modes(segments)
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
            code, message = self._failure_code(exc)
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
                    "error": code,
                },
                error={
                    "code": code,
                    "message": message,
                },
            )

    @staticmethod
    def _failure_code(exc: Exception) -> tuple[str, str]:
        """Deterministic input/business failures surface an actionable code
        instead of the generic pipeline_failed."""
        if isinstance(exc, media.MediaError):
            return str(exc), "输入视频不满足处理要求"
        cause = exc.cause if isinstance(exc, ProviderFailure) else exc
        if getattr(cause, "code", None) == "provider_rejected":
            return "provider_rejected", "参考图未通过供应商审核"
        if (
            isinstance(exc, ProviderFailure)
            and exc.record.get("outcome") == "rejected"
        ):
            return "picsart_submit_rejected", "视频提交未通过供应商审核"
        return "pipeline_failed", "流水线执行失败"

    @staticmethod
    def _validated_effective_request(value: object) -> dict[str, Any]:
        # 中间层不做形态校验：是我们自己写下的状态，读到啥用啥。
        if not isinstance(value, dict):
            raise ValueError("stored effective request is invalid")
        return value

    @staticmethod
    def _validated_join_modes(items: list[dict[str, Any]]) -> list[str]:
        if not items:
            raise ValueError("segment list is empty")
        modes: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("segment metadata is invalid")
            mode = item.get("join_mode")
            if mode not in {"hard_cut", "continue"}:
                raise ValueError("segment join mode is invalid")
            modes.append(mode)
        return modes

    def _fusion_client_for_request(
        self, request: dict[str, Any],
    ) -> tuple[str, CodexClient | DeepSeekClient]:
        request = self._validated_effective_request(request)
        configured_fusion = request["fusion"]
        if not isinstance(configured_fusion, dict):
            raise ValueError("frozen fusion configuration is invalid")
        agent = configured_fusion.get("agent")
        if agent == "codex":
            return agent, self.codex
        if agent == "deepseek":
            return agent, self.deepseek
        raise ValueError("frozen fusion agent is invalid")

    @staticmethod
    def _stored_voice_references(
        project: Path, state: dict[str, Any],
    ) -> list[Path]:
        """Load the frozen Audio 1..N list."""
        references = state.get("voice_references")
        if not isinstance(references, list) or len(references) > MAX_REFERENCE_AUDIO:
            raise ValueError("stored voice references are invalid")
        project_root = project.resolve(strict=True)
        paths: list[Path] = []
        for reference in references:
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("asset_path"), str)
                or not reference["asset_path"]
                or not isinstance(reference.get("voice_id"), str)
                or not reference["voice_id"]
                or not isinstance(reference.get("features"), str)
                or not reference["features"]
            ):
                raise ValueError("stored voice reference is invalid")
            candidate = project / reference["asset_path"]
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("stored voice reference is outside the project")
            resolved = candidate.resolve(strict=True)
            if project_root not in resolved.parents:
                raise ValueError("stored voice reference is outside the project")
            if Pipeline._sha256(resolved) != reference.get("asset_sha256"):
                raise ValueError("stored voice reference checksum does not match")
            paths.append(resolved)
        return paths

    def _materialize_voice_references(
        self,
        project: Path,
        patch: dict[str, dict[str, dict[str, Any]]],
        fallback_voice_id: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[Path], bool]:
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
        fallback_used = False
        if not selected_ids and fallback_voice_id is not None:
            selected_ids.append(fallback_voice_id)
            fallback_used = True

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
        return metadata_by_id, references, paths, fallback_used

    def _materialize_canonical_user_reference(
        self,
        project: Path,
        replacement: Path,
        log: ResearchLog,
    ) -> Path:
        destination = (
            project / "work" / "global-reference-sheet"
            / "user-reference.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.chmod(0o700)
        self._node(
            log,
            "replacement-sheet-user-reference",
            "project",
            {"source": replacement, "output": destination},
            [replacement],
            lambda: media.provider_reference_png(replacement, destination),
            output_media=lambda _value: [destination],
        )
        payload = destination.read_bytes()
        if (
            not payload.startswith(b"\x89PNG\r\n\x1a\n")
            or len(payload) < 9
        ):
            raise ValueError("prepared provider reference is not PNG")
        self._project_artifact_receipt(
            project,
            destination,
            "work/global-reference-sheet/user-reference.png",
        )
        return destination

    def _run_object_replace(
        self,
        project: Path,
        prepared: list[dict[str, Any]],
        translated_lines: list[dict[str, Any]],
        element_index: dict[str, Any],
        user_reference: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[Path]]:
        request = self._validated_effective_request(request)
        guidance = request["replacement_guidance"]
        user_instruction = (
            guidance["instruction"] if guidance is not None else ""
        )
        target_language = request["dialogue"]["target_language"]
        voice_ids = [voice["voice_id"] for voice in self.voice_library.voices]
        planning_context = object_replace_planning_context(element_index)
        payload = {
            **planning_context,
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
            "has_user_image": user_reference is not None,
            "user_replacement_image": (
                {"name": "Image 1", "sha256": self._sha256(user_reference)}
                if user_reference is not None else None
            ),
            "voice_catalog_sha256": self.voice_library.digest,
            "voice_catalog_lines": self.voice_library.model_lines(),
        }
        output_schema, input_sha256 = object_replace_schema(payload, voice_ids)
        # Dialogue lines need a TTS voice even when the model deliberately
        # leaves every person voiceless (e.g. only hands on camera); pick a
        # deterministic catalog fallback instead of failing later.
        fallback_voice_id = (
            self.voice_library.default_voice(target_language)["voice_id"]
            if translated_lines
            else None
        )
        materialized: dict[str, Any] = {}

        def normalize_and_merge(value: object) -> dict[str, Any]:
            patch = normalize_object_replace(
                value,
                input_sha256=input_sha256,
                element_index=element_index,
                voice_ids=voice_ids,
                has_user_image=user_reference is not None,
                user_instruction=user_instruction,
                dialogue_line_count=len(translated_lines),
            )
            (
                metadata_by_id, references, paths, fallback_used,
            ) = self._materialize_voice_references(
                project, patch, fallback_voice_id=fallback_voice_id,
            )
            materialized["references"] = references
            materialized["paths"] = paths
            materialized["fallback_used"] = fallback_used
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
            [user_reference] if user_reference is not None else [],
            lambda: self.deepseek.complete_structured(
                skill,
                payload,
                output_schema,
                [user_reference] if user_reference is not None else [],
                validator=lambda value: normalize_object_replace(
                    value,
                    input_sha256=input_sha256,
                    element_index=element_index,
                    voice_ids=voice_ids,
                    has_user_image=user_reference is not None,
                    user_instruction=user_instruction,
                    dialogue_line_count=len(translated_lines),
                ),
            ),
            transform=normalize_and_merge,
            output_media=lambda _value: list(materialized.get("paths") or []),
        )
        references = list(materialized.get("references") or [])
        paths = list(materialized.get("paths") or [])
        fallback_used = bool(materialized.get("fallback_used"))
        if fallback_used:
            log.event(
                "voice_reference_default_fallback",
                voice_id=fallback_voice_id,
                reason=(
                    "dialogue requires TTS but no person was assigned a voice"
                ),
            )
        metadata = {
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": input_sha256,
            "voice_catalog_sha256": self.voice_library.digest,
            "voice_reference_fallback": (
                {
                    "voice_id": fallback_voice_id,
                    "reason": (
                        "dialogue requires TTS but no person was assigned a voice"
                    ),
                }
                if fallback_used
                else None
            ),
        }
        return enhanced_index, metadata, references, paths

    def _run_dialogue_audio(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        voice_reference_paths: list[Path],
        log: ResearchLog,
    ) -> Path | None:
        """Synthesize every dialogue line and assemble the full speech track."""
        translated_lines = self.store.read(project_id).get(
            "translated_voice_lines"
        )
        if not isinstance(translated_lines, list) or any(
            not isinstance(line, dict) for line in translated_lines
        ):
            raise ValueError("stored translated dialogue is invalid")
        if not translated_lines:
            return None
        dialogue_voices = enhanced_element_index.get("dialogue_voices")
        people = enhanced_element_index.get("people")
        # 不做形态校验：音色归属缺失/不齐时按无归属处理，落到默认音色。
        if not isinstance(dialogue_voices, list):
            dialogue_voices = []
        if len(dialogue_voices) < len(translated_lines):
            dialogue_voices = dialogue_voices + [None] * (
                len(translated_lines) - len(dialogue_voices)
            )
        if not isinstance(people, dict):
            people = {}
        valid_voices = {
            path.resolve(strict=True) for path in voice_reference_paths
        }
        default_voice = (
            voice_reference_paths[0].resolve(strict=True)
            if voice_reference_paths
            else None
        )
        directory = project / "work" / "dialogue-audio"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        line_audios: list[dict[str, Any]] = []
        for offset, (line, speaker) in enumerate(
            zip(translated_lines, dialogue_voices), 1
        ):
            text = line.get("text")
            try:
                start_s = float(line["start_s"])
                end_s = float(line["end_s"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "stored translated dialogue is invalid"
                ) from None
            if (
                not isinstance(text, str)
                or not text.strip()
                or not math.isfinite(start_s)
                or not math.isfinite(end_s)
                or not 0 <= start_s < end_s
            ):
                raise ValueError("stored translated dialogue is invalid")
            voice_path: Path | None = None
            record = people.get(speaker) if isinstance(speaker, str) else None
            voice = record.get("voice") if isinstance(record, dict) else None
            asset_path = (
                voice.get("asset_path") if isinstance(voice, dict) else None
            )
            if isinstance(asset_path, str):
                candidate = (project / asset_path).resolve()
                if candidate in valid_voices:
                    voice_path = candidate
            if voice_path is None:
                # Unattributed lines (narration) and voiceless people fall
                # back to the Audio 1 voice.
                voice_path = default_voice
            if voice_path is None:
                # Defensive: fresh runs always materialize at least the
                # catalog fallback voice in _run_object_replace.
                raise ValueError("dialogue TTS requires a voice reference")
            destination = directory / f"line-{offset:04d}.wav"

            def operation(
                text: str = text,
                voice_path: Path = voice_path,
                destination: Path = destination,
            ) -> ProviderResult[Path]:
                submitted = self.autodl.submit_tts(
                    text=text, voice_path=voice_path,
                )
                record: dict[str, Any] = {"submit": submitted.record}
                polled = self.autodl.poll(submitted.value)
                record["poll"] = polled.record
                downloaded = self.autodl.download(polled.value, destination)
                record["download"] = downloaded.record
                return ProviderResult(
                    destination, record, list(downloaded.evidence_media),
                )

            self._node(
                log,
                "dialogue-tts-line",
                "project",
                {
                    "line_index": offset,
                    "text": text,
                    "start_s": start_s,
                    "end_s": end_s,
                    "speaker": speaker,
                    "voice": voice_path,
                    "output": destination,
                },
                [voice_path],
                operation,
                lambda value: [value],
            )
            line_audios.append(
                {"audio": destination, "start_s": start_s, "end_s": end_s}
            )
        full_track = directory / "full-track.wav"
        total_duration = sum(float(item["duration_s"]) for item in prepared)

        def assemble() -> dict[str, Any]:
            return media.assemble_dialogue_track(
                line_audios, total_duration, full_track,
            )

        self._node(
            log,
            "assemble-dialogue-track",
            "project",
            {
                "lines": [
                    {**entry, "audio": entry["audio"].name}
                    for entry in line_audios
                ],
                "total_duration_s": total_duration,
                "output": full_track,
            },
            [entry["audio"] for entry in line_audios],
            assemble,
            output_media=lambda _value: [full_track],
        )
        return full_track

    @staticmethod
    def _enhanced_index_uses_user_image(
        enhanced_element_index: dict[str, Any],
    ) -> bool:
        """Derive the effective user-image binding from the frozen index."""

        for category in ("people", "entities", "scenes"):
            records = enhanced_element_index.get(category)
            if not isinstance(records, dict):
                continue
            for record in records.values():
                replacement = (
                    record.get("replacement")
                    if isinstance(record, dict)
                    else None
                )
                if (
                    isinstance(replacement, dict)
                    and replacement.get("source") == "image_1"
                ):
                    return True
        return False

    @staticmethod
    def _enhanced_index_has_replacement(
        enhanced_element_index: dict[str, Any],
    ) -> bool:
        """Return whether any globally indexed visual element is replaced."""

        for category in ("people", "entities", "scenes"):
            records = enhanced_element_index.get(category)
            if not isinstance(records, dict):
                continue
            for record in records.values():
                if isinstance(record, dict) and record.get("replacement"):
                    return True
        return False

    def _run_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        object_replace: dict[str, Any],
        replacement: Path | None,
        log: ResearchLog,
        *,
        user_reference: Path | None = None,
    ) -> tuple[dict[str, Any], Path | None]:
        """Render the one project-wide appearance sheet from the frozen index."""

        if object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION:
            raise ValueError(
                "replacement sheet requires the current Object Replace output"
            )
        spec = build_replacement_sheet_spec(enhanced_element_index)
        user_image_present = (
            replacement is not None
            and self._enhanced_index_uses_user_image(enhanced_element_index)
        )
        existing = self.store.read(project_id).get("replacement_sheet")
        if (
            isinstance(existing, dict)
            and existing.get("version") != REPLACEMENT_SHEET_VERSION
        ):
            raise ValueError("stored replacement sheet version is unsupported")
        if isinstance(existing, dict) and existing.get("status") in {
            "not_needed", "succeeded",
        }:
            if existing.get("spec_sha256") != spec["spec_sha256"]:
                raise ValueError(
                    "frozen replacement sheet does not match the current global index"
                )
            existing_path = self._validated_stored_replacement_sheet(
                project, self.store.read(project_id),
            )
            return existing, existing_path
        if isinstance(existing, dict) and existing.get("status") == "prepared":
            if existing.get("spec_sha256") != spec["spec_sha256"]:
                raise ValueError(
                    "prepared replacement sheet does not match the current global index"
                )
            raise ValueError(
                "prepared replacement sheet requires receipt-safe replay"
            )
        # The replacement sheet is always H3 Image 1 so the identity anchor
        # carries the primary reference weight; the user image, when present,
        # follows as Image 2.
        image_reference = "Image 1"
        h3_reference_images: list[dict[str, Any]] = []
        state = self.store.read(project_id)
        voice_references = state.get("voice_references")
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
        if not isinstance(voice_references, list):
            raise ValueError("stored voice references are invalid")
        base_bundle: dict[str, Any] = {
            "version": REPLACEMENT_SHEET_VERSION,
            "status": "not_needed" if not spec["tiles"] else "prepared",
            "spec": spec,
            "spec_sha256": spec["spec_sha256"],
            "user_image_present": user_image_present,
            "h3_image_reference": image_reference if spec["tiles"] else None,
            "h3_reference_images": h3_reference_images,
            "upstream_bindings": self._replacement_sheet_upstream_bindings(
                project,
                prepared,
                enhanced_element_index,
                object_replace,
                voice_references,
                effective_request,
                replacement,
            ),
            "seedream": None,
            "output": None,
        }
        if not spec["tiles"]:
            self.store.update(project_id, replacement_sheet=base_bundle)
            return base_bundle, None

        directory = project / "work" / "global-reference-sheet"
        inputs = self._node(
            log,
            "replacement-sheet-inputs",
            "project",
            {
                "spec": spec,
            },
            [],
            lambda: materialize_replacement_sheet_inputs(
                directory,
                spec,
                user_image_present=user_image_present,
            ),
            output_media=lambda value: [value["canvas_path"]],
        )
        canvas = inputs["canvas_path"]
        prompt_path = directory / "prompt.txt"
        spec_path = directory / "spec.json"
        if user_image_present and user_reference is None:
            assert replacement is not None
            user_reference = self._materialize_canonical_user_reference(
                project, replacement, log,
            )
        canonical_replacement = user_reference if user_image_present else None
        if canonical_replacement is not None:
            canonical_bytes = canonical_replacement.read_bytes()
            if (
                not canonical_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                or len(canonical_bytes) < 9
            ):
                raise ValueError("prepared provider reference is not PNG")
            canonical_receipt = self._project_artifact_receipt(
                project,
                canonical_replacement,
                "work/global-reference-sheet/user-reference.png",
            )
            h3_reference_images.append({
                "image_reference": "Image 2",
                "role": "user_replacement",
                **canonical_receipt,
            })
        seedream_inputs = [
            canvas,
            *([canonical_replacement] if canonical_replacement is not None else []),
        ]
        input_receipts = [
            self._project_artifact_receipt(
                project,
                path,
                path.relative_to(project).as_posix(),
            )
            for path in seedream_inputs
        ]
        base_bundle["h3_reference_images"] = h3_reference_images
        seedream_state = {
            "model": self.settings.seedream_model,
            "endpoint": (
                self.settings.seedream_api_base + "/images/generations"
            ),
            "prompt": self._project_artifact_receipt(
                project, prompt_path, "work/global-reference-sheet/prompt.txt",
            ),
            "spec": self._project_artifact_receipt(
                project, spec_path, "work/global-reference-sheet/spec.json",
            ),
            "inputs": input_receipts,
            "request_sha256": self._json_sha256({
                "version": 1,
                "endpoint": (
                    self.settings.seedream_api_base + "/images/generations"
                ),
                "model": self.settings.seedream_model,
                "prompt_sha256": self._sha256(prompt_path),
                "input_sha256": [
                    receipt["sha256"] for receipt in input_receipts
                ],
                "size": SEEDREAM_OUTPUT_SIZE,
                "response_format": "b64_json",
                "watermark": False,
            }),
            "receipt": None,
        }
        prepared_bundle = {**base_bundle, "seedream": seedream_state}
        # Persist the exact global-index-derived plan before the paid image POST.
        self.store.update(project_id, replacement_sheet=prepared_bundle)

        provider_output = directory / "provider-output.bin"
        receipt_path = directory / "seedream-receipt.json"
        self._seedream_generate_never_fail(
            project_id,
            project,
            log,
            "seedream-reference-submit",
            {
                "model": self.settings.seedream_model,
                "spec_sha256": spec["spec_sha256"],
                "prompt_sha256": self._sha256(prompt_path),
                "input_receipts": input_receipts,
            },
            seedream_inputs,
            prompt_path=prompt_path,
            provider_output=provider_output,
            receipt_path=receipt_path,
            seedream_state=seedream_state,
            sheet_bundle=prepared_bundle,
        )

        return self._finish_replacement_sheet(
            project_id,
            project,
            prepared_bundle,
            provider_output,
            receipt_path,
            log,
            node="replacement-sheet-finalize",
        )

    def _finish_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        prepared_bundle: dict[str, Any],
        provider_output: Path,
        receipt_path: Path,
        log: ResearchLog,
        *,
        node: str,
    ) -> tuple[dict[str, Any], Path]:
        spec = prepared_bundle["spec"]
        final_path = project / "work" / "global-reference-sheet" / "reference-sheet.png"
        self._node(
            log,
            node,
            "project",
            {
                "provider_output": provider_output,
                "spec_sha256": spec["spec_sha256"],
                "output": final_path,
            },
            [provider_output],
            lambda: finalize_replacement_sheet(
                provider_output,
                spec,
                final_path,
            ),
            output_media=lambda _value: [final_path],
        )
        seedream_state = prepared_bundle["seedream"]
        completed_bundle = {
            **prepared_bundle,
            "status": "succeeded",
            "seedream": {
                **seedream_state,
                "receipt": self._project_artifact_receipt(
                    project,
                    receipt_path,
                    "work/global-reference-sheet/seedream-receipt.json",
                ),
            },
            "output": self._project_artifact_receipt(
                project,
                final_path,
                "work/global-reference-sheet/reference-sheet.png",
            ),
        }
        completed_bundle["h3_reference_images"] = [
            {
                "image_reference": prepared_bundle["h3_image_reference"],
                "role": "global_replacement_sheet",
                **completed_bundle["output"],
                "spec_sha256": spec["spec_sha256"],
            },
            *prepared_bundle["h3_reference_images"],
        ]
        self.store.update(project_id, replacement_sheet=completed_bundle)
        return completed_bundle, final_path

    _SEEDREAM_REGENERATABLE_CODES = frozenset({
        "provider_rejected", "submission_unknown", "provider_protocol_error",
    })

    def _seedream_generate_never_fail(
        self,
        project_id: str,
        project: Path,
        log: ResearchLog,
        node: str,
        inputs_summary: dict[str, Any],
        images: list[Path],
        *,
        prompt_path: Path,
        provider_output: Path,
        receipt_path: Path,
        seedream_state: dict[str, Any],
        sheet_bundle: dict[str, Any],
    ) -> None:
        """Submit the frozen Seedream sheet request without ever failing the
        project: submission-unknown/protocol outcomes reset the attempt
        artifacts and re-POST; content rejections additionally perturb the
        prompt and re-freeze its receipt before re-POSTing."""
        base_prompt = prompt_path.read_text(encoding="utf-8")
        regenerations = 0
        while True:
            prompt = prompt_path.read_text(encoding="utf-8")
            try:
                self._node(
                    log,
                    node,
                    "project",
                    {**inputs_summary, "regeneration": regenerations},
                    images,
                    lambda prompt=prompt: self.seedream.generate(
                        prompt=prompt,
                        images=images,
                        destination=provider_output,
                        receipt_path=receipt_path,
                    ),
                    output_media=lambda path: [path],
                    retry_if=lambda _exc: False,
                )
                return
            except ProviderFailure as exc:
                code = getattr(exc.cause, "code", None)
                if code not in self._SEEDREAM_REGENERATABLE_CODES:
                    raise
                regenerations += 1
                removed = _seedream_reset_attempt(receipt_path, provider_output)
                if code == "provider_rejected":
                    prompt_path.write_text(
                        base_prompt
                        + f"\n\nRegeneration {regenerations}: produce a fresh "
                        "variation of the same specification.",
                        encoding="utf-8",
                    )
                    seedream_state["prompt"] = self._project_artifact_receipt(
                        project,
                        prompt_path,
                        "work/global-reference-sheet/prompt.txt",
                    )
                    seedream_state["request_sha256"] = self._json_sha256({
                        "version": 1,
                        "endpoint": seedream_state["endpoint"],
                        "model": seedream_state["model"],
                        "prompt_sha256": seedream_state["prompt"]["sha256"],
                        "input_sha256": [
                            receipt["sha256"]
                            for receipt in seedream_state["inputs"]
                        ],
                        "size": SEEDREAM_OUTPUT_SIZE,
                        "response_format": "b64_json",
                        "watermark": False,
                    })
                    self.store.update(
                        project_id, replacement_sheet=sheet_bundle,
                    )
                log.event(
                    "seedream_submission_regenerated",
                    node=node,
                    reason=code,
                    regeneration=regenerations,
                    removed_artifacts=removed,
                    error=str(exc),
                )
                self._retry_sleep(min(60.0 * regenerations, 1800.0))

    def _resume_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        frozen_sheet: dict[str, Any],
        recovery_mode: str,
        log: ResearchLog,
    ) -> tuple[dict[str, Any], Path | None]:
        """Resume only the frozen replacement-sheet checkpoint."""

        current = self.store.read(project_id)
        if current.get("replacement_sheet") != frozen_sheet:
            raise ValueError("stored replacement sheet changed during resume")
        validated_path = self._validated_stored_replacement_sheet(
            project,
            current,
            allow_prepared=True,
        )
        status = frozen_sheet.get("status")
        if status in {"not_needed", "succeeded"}:
            if recovery_mode != "reuse":
                raise ValueError("stored replacement sheet recovery mode is invalid")
            return frozen_sheet, validated_path
        if status != "prepared" or recovery_mode not in {
            "generate", "replay", "regenerate",
        }:
            raise ValueError("stored replacement sheet recovery mode is invalid")
        if self._validated_prepared_seedream_attempt(
            project, frozen_sheet,
        ) != recovery_mode:
            raise ValueError("Seedream attempt evidence changed during resume")

        seedream = frozen_sheet["seedream"]
        prompt_path = project / seedream["prompt"]["path"]
        images = [
            project / receipt["path"]
            for receipt in seedream["inputs"]
        ]
        directory = project / "work" / "global-reference-sheet"
        provider_output = directory / "provider-output.bin"
        receipt_path = directory / "seedream-receipt.json"
        if recovery_mode == "replay":
            self._node(
                log,
                "seedream-reference-replay",
                "project",
                {
                    "model": seedream["model"],
                    "endpoint": seedream["endpoint"],
                    "request_sha256": seedream["request_sha256"],
                    "spec_sha256": frozen_sheet["spec_sha256"],
                    "input_receipts": seedream["inputs"],
                },
                images,
                lambda: self.seedream.replay_received(
                    destination=provider_output,
                    receipt_path=receipt_path,
                ),
                output_media=lambda path: [path],
                retry_if=lambda _exc: False,
            )
        else:
            if recovery_mode == "regenerate":
                # A rejected/unknown terminal receipt cannot be replayed;
                # downgrade it: drop the attempt artifacts and re-POST.
                try:
                    prior = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    prior = None
                removed = _seedream_reset_attempt(receipt_path, provider_output)
                log.event(
                    "seedream_receipt_reset",
                    prior_status=(
                        prior.get("status") if isinstance(prior, dict) else None
                    ),
                    removed_artifacts=removed,
                )
            self._seedream_generate_never_fail(
                project_id,
                project,
                log,
                "seedream-reference-submit-resume",
                {
                    "model": seedream["model"],
                    "endpoint": seedream["endpoint"],
                    "request_sha256": seedream["request_sha256"],
                    "spec_sha256": frozen_sheet["spec_sha256"],
                    "input_receipts": seedream["inputs"],
                },
                images,
                prompt_path=prompt_path,
                provider_output=provider_output,
                receipt_path=receipt_path,
                seedream_state=seedream,
                sheet_bundle=frozen_sheet,
            )
        return self._finish_replacement_sheet(
            project_id,
            project,
            frozen_sheet,
            provider_output,
            receipt_path,
            log,
            node="replacement-sheet-finalize-resume",
        )

    @classmethod
    def _replacement_sheet_upstream_bindings(
        cls,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        object_replace: dict[str, Any],
        voice_references: list[dict[str, Any]],
        effective_request: dict[str, Any],
        replacement: Path | None,
    ) -> dict[str, Any]:
        effective_request = cls._validated_effective_request(
            effective_request
        )
        segment_bindings: list[dict[str, Any]] = []
        source_receipts: list[dict[str, Any]] = []
        sorted_prepared = sorted(
            prepared, key=lambda value: int(value["index"])
        )
        cls._validated_join_modes(sorted_prepared)
        for item in sorted_prepared:
            index = int(item["index"])
            source = item.get("source")
            if not isinstance(source, Path):
                raise ValueError(
                    "replacement sheet segment source is not frozen"
                )
            segment_bindings.append({
                "index": index,
                "start_s": item.get("start_s"),
                "end_s": item.get("end_s"),
                "duration_s": item.get("duration_s"),
                "chain_id": item.get("chain_id"),
                "join_mode": item["join_mode"],
                "visual_prompt": item.get("visual_prompt"),
                "dialogue": item.get("dialogue", []),
            })
            source_receipts.append(cls._project_artifact_receipt(
                project,
                source,
                f"work/segments/{index:04d}/source.mp4",
            ))
        replacement_receipt = (
            cls._project_artifact_receipt(
                project,
                replacement,
                "inputs/replacement-image",
            )
            if replacement is not None
            else None
        )
        return {
            "segments_sha256": cls._json_sha256(segment_bindings),
            "segment_sources": source_receipts,
            "enhanced_element_index_sha256": cls._json_sha256(
                enhanced_element_index
            ),
            "object_replace_sha256": cls._json_sha256(object_replace),
            "voice_references_sha256": cls._json_sha256(voice_references),
            "effective_request_sha256": cls._json_sha256(effective_request),
            "replacement_image": replacement_receipt,
        }

    @staticmethod
    def _muted_artifact_name(muted: Path) -> str:
        return muted.name

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
        replacement_sheet: dict[str, Any] | None = None,
        request: dict[str, Any] | None = None,
    ) -> None:
        segments: dict[str, dict[str, Any]] = {}
        sorted_h3_inputs = sorted(
            h3_inputs, key=lambda value: int(value[0]["index"]),
        )
        self._validated_join_modes([value[0] for value in sorted_h3_inputs])
        for item, final_prompt, muted, fusion_input_sha256 in sorted_h3_inputs:
            index = int(item["index"])
            base = f"work/segments/{index:04d}"
            segments[str(index)] = {
                "segment_index": index,
                "source_video": self._project_artifact_receipt(
                    project, item["source"], f"{base}/source.mp4",
                ),
                "muted_video": self._project_artifact_receipt(
                    project, muted,
                    f"{base}/{self._muted_artifact_name(muted)}",
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
                    "join_mode": item["join_mode"],
                }),
            }
        state = self.store.read(project_id)
        enhanced_index = state.get("enhanced_element_index")
        voice_references = state.get("voice_references")
        object_replace = state.get("object_replace")
        frozen_replacement_sheet = (
            replacement_sheet
            if isinstance(replacement_sheet, dict)
            else state.get("replacement_sheet")
        )
        if (
            not isinstance(enhanced_index, dict)
            or not isinstance(voice_references, list)
            or not isinstance(object_replace, dict)
            or not isinstance(frozen_replacement_sheet, dict)
            or not isinstance(object_replace.get("input_sha256"), str)
            or len(object_replace["input_sha256"]) != 64
        ):
            raise ValueError("H3 preparation binding state is invalid")
        if object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION:
            raise ValueError(
                f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} requires "
                f"Object Replace v{OBJECT_REPLACE_OUTPUT_VERSION}"
            )
        if frozen_replacement_sheet.get("version") != REPLACEMENT_SHEET_VERSION:
            raise ValueError(
                f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} requires "
                f"replacement sheet v{REPLACEMENT_SHEET_VERSION}"
            )
        if state.get("replacement_sheet") != frozen_replacement_sheet:
            raise ValueError(
                "H3 preparation replacement sheet is not the stored sheet"
            )
        self._validated_stored_replacement_sheet(project, state)
        if replacement is None and state.get("replacement_image_path") is not None:
            replacement = project / "inputs" / "replacement-image"
        replacement_receipt = (
            self._project_artifact_receipt(
                project, replacement, "inputs/replacement-image",
            )
            if replacement is not None
            else None
        )
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
        if request is not None and request != effective_request:
            raise ValueError(
                "H3 preparation request is not the stored effective request"
            )
        output_request = effective_request["output"]
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
                    "replacement_sheet_sha256": self._json_sha256(
                        frozen_replacement_sheet,
                    ),
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
        replacement_sheet: dict[str, Any],
        replacement_sheet_path: Path | None,
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
                    replacement_sheet,
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
                replacement_sheet=replacement_sheet,
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
                    replacement_sheet_path,
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
        request = self._validated_effective_request(request)
        visual_prompts = "\n\n".join(
            f"Segment {item['index']}: {item['visual_prompt']}"
            for item in prepared
        )
        target_language = request["dialogue"]["target_language"]
        if request["dialogue"]["mode"] == "generate":
            brief = request["dialogue"]["brief"]
            self._progress(project_id, 43, "dialogue-generation")
            translated_lines = self._node(
                log,
                "deepseek-dialogue-generation",
                "project",
                self._dialogue_generation_payload(
                    brief, target_language, prepared,
                ),
                [],
                lambda: self._dialogue_generation_request(
                    brief, target_language, prepared,
                ),
                transform=lambda value: self._normalize_generated_dialogue(
                    value, prepared,
                ),
            )
        else:
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
                validator=normalize_project_index,
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

        user_reference = (
            self._materialize_canonical_user_reference(
                project, replacement, log,
            )
            if replacement is not None
            else None
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
            user_reference,
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

        self._progress(
            project_id, 55, "seedream-reference-sheet", segments=public_segments,
        )
        replacement_sheet, replacement_sheet_path = self._run_replacement_sheet(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            object_replace,
            replacement,
            log,
            user_reference=user_reference,
        )

        self._continue_after_replacement_sheet(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            voice_reference_paths,
            replacement,
            replacement_sheet,
            replacement_sheet_path,
            request,
            log,
        )

    def _continue_after_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        voice_reference_paths: list[Path],
        replacement: Path | None,
        replacement_sheet: dict[str, Any],
        replacement_sheet_path: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> None:
        """Run only Fusion, H3, and local stitching after the sheet checkpoint."""

        public_segments = [self._public_segment(item) for item in prepared]

        # This is the public-input barrier.  No paid segment branch starts
        # before every visual prompt, translation, localization, the single
        # project index and project-wide replacement/voice plan have completed.
        self._progress(project_id, 56, "dialogue-tts", segments=public_segments)
        self._run_dialogue_audio(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            voice_reference_paths,
            log,
        )
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
            replacement_sheet,
            replacement_sheet_path,
            request,
            log,
        )

        final = project / "output" / "generated.mp4"
        join_modes = self._validated_join_modes(prepared)
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
                "join_modes": join_modes,
                "output": final,
            },
            outputs,
            lambda: media.concat(
                outputs,
                final,
                target_durations=[item["duration_s"] for item in prepared],
                join_modes=join_modes,
            ),
            lambda _value: [final],
        )
        self._replace_dialogue_audio_track(project, prepared, final, log)
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

    def _replace_dialogue_audio_track(
        self,
        project: Path,
        prepared: list[dict[str, Any]],
        final: Path,
        log: ResearchLog,
    ) -> None:
        """Overwrite the stitched video with the assembled TTS speech track."""
        if not any(item.get("dialogue") for item in prepared):
            return
        full_track = project / "work" / "dialogue-audio" / "full-track.wav"
        if not full_track.is_file():
            # Degrade: keep the stitched video with its generated audio
            # instead of failing the whole project.
            log.event(
                "dialogue_track_missing_degraded",
                output=final,
                impact=(
                    "assembled TTS dialogue track is missing; the stitched "
                    "video keeps its generated audio"
                ),
            )
            return
        self._node(
            log,
            "replace-audio-track",
            "project",
            {"video": final, "audio": full_track, "output": final},
            [final, full_track],
            lambda: media.replace_audio_track(final, full_track, final),
            lambda _value: [final],
        )

    @staticmethod
    def _validated_stored_segments(
        state: dict[str, Any],
        *,
        missing_message: str,
    ) -> list[dict[str, Any]]:
        raw_segments = state.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(missing_message)
        segments: list[dict[str, Any]] = []
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
            join_mode = raw.get("join_mode")
            if join_mode not in {"hard_cut", "continue"}:
                raise ValueError("stored segment join mode is invalid")
            segments.append(dict(raw))
        segments.sort(key=lambda item: int(item["index"]))
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
        dict[str, Any] | None,
    ]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
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
        if state.get("h3_prepared_artifacts") is not None:
            raise ValueError(
                "project already has frozen H3 artifacts; use Picsart resume"
            )
        source, replacement = self._stored_input_paths(project, state)
        raw_segments = self._validated_stored_segments(
            state, missing_message="stored segments are missing",
        )
        frozen_sheet = state.get("replacement_sheet")
        if frozen_sheet is not None:
            if not isinstance(frozen_sheet, dict):
                raise ValueError("stored replacement sheet state is invalid")
            sheet_path = self._validated_stored_replacement_sheet(
                project,
                state,
                allow_prepared=True,
            )
            sheet_status = frozen_sheet.get("status")
            recovery_mode = (
                "reuse"
                if sheet_status in {"not_needed", "succeeded"}
                else self._validated_prepared_seedream_attempt(
                    project, frozen_sheet,
                )
            )
            voice_reference_paths = self._stored_voice_references(
                project, state,
            )
            prepared_segments: list[dict[str, Any]] = []
            for raw in raw_segments:
                visual_prompt = raw.get("visual_prompt")
                dialogue = raw.get("dialogue")
                if not isinstance(visual_prompt, str) or not isinstance(
                    dialogue, list,
                ) or any(not isinstance(line, dict) for line in dialogue):
                    raise ValueError(
                        "stored replacement sheet segment content is invalid"
                    )
                index = int(raw["index"])
                prepared_segments.append({
                    **raw,
                    "source": (
                        project / "work" / "segments"
                        / f"{index:04d}" / "source.mp4"
                    ),
                    "status": "running",
                })
            enhanced_element_index = state.get("enhanced_element_index")
            if not isinstance(enhanced_element_index, dict):
                raise ValueError("stored enhanced element index is invalid")
            return (
                project,
                source,
                replacement,
                effective_request,
                prepared_segments,
                {},
                [],
                {
                    "replacement_sheet": frozen_sheet,
                    "replacement_sheet_path": sheet_path,
                    "recovery_mode": recovery_mode,
                    "enhanced_element_index": enhanced_element_index,
                    "voice_reference_paths": voice_reference_paths,
                },
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
            if not isinstance(selection, list):
                raise ValueError("reusable keyframe selection is invalid")
            if not (directory / "source.mp4").is_file():
                raise FileNotFoundError("reusable segment source is missing")
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
            effective_request,
            segments,
            selections,
            [dict(line) for line in spoken_lines],
            None,
        )

    def _backfill_resume_voice_fallback(
        self,
        project_id: str,
        project: Path,
        request: dict[str, Any],
        prepared_segments: list[dict[str, Any]],
        replacement: Path | None,
        checkpoint: dict[str, Any],
        log: ResearchLog,
    ) -> None:
        """Backfill the catalog default voice for projects frozen before the
        object-replace fallback existed: dialogue waits for TTS but the
        frozen plan materialized zero voice references.  The frozen sheet
        stays byte-identical; only its stored upstream bindings are
        recomputed so the added reference is honestly reflected."""
        if checkpoint["voice_reference_paths"]:
            return
        state = self.store.read(project_id)
        translated_lines = state.get("translated_voice_lines")
        if (
            not isinstance(translated_lines, list)
            or not translated_lines
            or any(not isinstance(line, dict) for line in translated_lines)
        ):
            return
        reason = (
            "resume: dialogue requires TTS but the frozen plan assigned "
            "no voice"
        )
        fallback_voice_id = self.voice_library.default_voice(
            request["dialogue"]["target_language"],
        )["voice_id"]
        (
            _metadata_by_id, references, paths, fallback_used,
        ) = self._materialize_voice_references(
            project, {"people": {}}, fallback_voice_id=fallback_voice_id,
        )
        if not fallback_used or len(references) != 1:
            raise ValueError("voice reference fallback did not materialize")
        log.event(
            "voice_reference_default_fallback",
            voice_id=fallback_voice_id,
            reason=reason,
        )
        object_replace = state.get("object_replace")
        if not isinstance(object_replace, dict):
            raise ValueError("stored object-replace state is invalid")
        updated_object_replace = {
            **object_replace,
            "voice_reference_fallback": {
                "voice_id": fallback_voice_id,
                "reason": reason,
            },
        }
        updated_sheet = {
            **checkpoint["replacement_sheet"],
            "upstream_bindings": self._replacement_sheet_upstream_bindings(
                project,
                prepared_segments,
                checkpoint["enhanced_element_index"],
                updated_object_replace,
                references,
                request,
                replacement,
            ),
        }
        self.store.update(
            project_id,
            voice_references=references,
            object_replace=updated_object_replace,
            replacement_sheet=updated_sheet,
        )
        checkpoint["replacement_sheet"] = updated_sheet
        checkpoint["voice_reference_paths"] = paths

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
            checkpoint,
        ) = self._load_pre_picsart_resume(project_id)
        log = ResearchLog(project)
        log.event("pre_picsart_resume_started")
        try:
            if checkpoint is not None:
                self._backfill_resume_voice_fallback(
                    project_id,
                    project,
                    request,
                    segments,
                    replacement,
                    checkpoint,
                    log,
                )
                self._progress(
                    project_id,
                    55,
                    "seedream-reference-sheet-resume",
                    segments=[self._public_segment(item) for item in segments],
                )
                replacement_sheet, replacement_sheet_path = (
                    self._resume_replacement_sheet(
                        project_id,
                        project,
                        checkpoint["replacement_sheet"],
                        checkpoint["recovery_mode"],
                        log,
                    )
                )
                self._continue_after_replacement_sheet(
                    project_id,
                    project,
                    segments,
                    checkpoint["enhanced_element_index"],
                    checkpoint["voice_reference_paths"],
                    replacement,
                    replacement_sheet,
                    replacement_sheet_path,
                    request,
                    log,
                )
                log.event("pre_picsart_resume_succeeded")
                return
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

    @classmethod
    def _validated_submission_receipts(
        cls,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw = state.get("picsart_submission_receipts")
        if raw is None:
            return None
        version = raw.get("version") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "segments"}
            or isinstance(version, bool)
            or version != PICSART_RECEIPT_VERSION
            or not isinstance(raw.get("segments"), dict)
        ):
            raise ValueError("stored Picsart receipt ledger is invalid")
        prepared_manifest = state.get("h3_prepared_artifacts")
        prepared_version = (
            prepared_manifest.get("version")
            if isinstance(prepared_manifest, dict)
            else None
        )
        if prepared_version != H3_PREPARED_ARTIFACTS_VERSION:
            raise ValueError(
                "Picsart receipt v2 requires the current H3 preparation manifest"
            )
        preparation_sha256 = cls._json_sha256(prepared_manifest)
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
                expected_receipt_keys = {
                    "segment_index", "run_identity", "request_sha256",
                    "preparation_sha256", "status", "task_id",
                }
                if (
                    not isinstance(receipt, dict)
                    or set(receipt) != expected_receipt_keys
                ):
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
                    or receipt.get("preparation_sha256")
                    != preparation_sha256
                ):
                    raise ValueError("stored Picsart receipt fields are invalid")
                seen_runs.add(run_identity)
                copied.append(dict(receipt))
            normalized[segment_key] = copied
        return {"version": version, "segments": normalized}

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
            manifest = state.get("h3_prepared_artifacts")
            manifest_version = (
                manifest.get("version") if isinstance(manifest, dict) else None
            )
            if manifest_version != H3_PREPARED_ARTIFACTS_VERSION:
                raise ValueError(
                    "Picsart submission requires the current H3 preparation manifest"
                )
            ledger_version = PICSART_RECEIPT_VERSION
            preparation_sha256 = self._json_sha256(manifest)
            if any(
                prepared.preparation_sha256 != preparation_sha256
                for prepared in pending
            ):
                raise ValueError(
                    "prepared Picsart requests do not match H3 preparation"
                )
            ledger = self._validated_submission_receipts(state) or {
                "version": ledger_version,
                "segments": {},
            }
            if ledger["version"] != ledger_version:
                raise ValueError(
                    "Picsart receipt ledger does not match H3 preparation"
                )
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
                receipt = {
                    "segment_index": prepared.segment_index,
                    "run_identity": prepared.run_identity,
                    "request_sha256": prepared.request_sha256,
                    "status": "submission_started",
                    "task_id": None,
                }
                receipt["preparation_sha256"] = preparation_sha256
                records.append(receipt)
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
                or (
                    ledger["version"] == PICSART_RECEIPT_VERSION
                    and receipt.get("preparation_sha256")
                    != prepared.preparation_sha256
                )
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
                raise ValueError("Picsart submission receipt ledger is missing")
            records = ledger["segments"].get(str(segment_index)) or []
            if not records:
                raise ValueError("accepted Picsart task receipt is missing")
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
            raise ValueError("Picsart segment receipt is missing")
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

    @staticmethod
    def _poll_failure_is_asset_fetch(record: dict[str, Any]) -> bool:
        """True when a terminal task failure is a provider-side asset fetch
        or validation error (safe to resubmit with the same frozen inputs)."""

        texts: list[str] = []
        for exchange in record.get("polls") or []:
            if not isinstance(exchange, dict):
                continue
            response = exchange.get("response")
            if not isinstance(response, dict):
                continue
            body = response.get("body")
            if not isinstance(body, dict):
                continue
            envelope = body.get("response")
            candidates = envelope if isinstance(envelope, dict) else body
            error = candidates.get("error")
            if isinstance(error, dict):
                error = error.get("message")
            for value in (error, candidates.get("message"), candidates.get("reason")):
                if isinstance(value, str):
                    texts.append(value)
        haystack = "\n".join(texts).lower()
        return any(
            marker in haystack for marker in _PICSART_FETCH_FAILURE_MARKERS
        )

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
        version = raw.get("version") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "bindings", "segments"}
            or isinstance(version, bool)
            or version != H3_PREPARED_ARTIFACTS_VERSION
            or not isinstance(raw.get("bindings"), dict)
            or not isinstance(raw.get("segments"), dict)
        ):
            raise ValueError("stored H3 preparation manifest is invalid")
        bindings = raw["bindings"]
        expected_binding_keys = {
            "enhanced_element_index_sha256", "voice_references_sha256",
            "object_replace_input_sha256", "replacement_image",
            "output_request_sha256",
        }
        expected_binding_keys.add("replacement_sheet_sha256")
        if set(bindings) != expected_binding_keys:
            raise ValueError("stored H3 preparation bindings are invalid")
        enhanced_index = state.get("enhanced_element_index")
        voice_references = state.get("voice_references")
        object_replace = state.get("object_replace")
        effective_request = cls._validated_effective_request(
            state.get("effective_request")
        )
        output_request = effective_request["output"]
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
        if object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION:
            raise ValueError(
                "stored H3 preparation Object Replace version does not match"
            )
        replacement_receipt = bindings.get("replacement_image")
        has_replacement = state.get("replacement_image_path") is not None
        if has_replacement != (replacement_receipt is not None):
            raise ValueError("stored H3 preparation replacement binding does not match")
        if replacement_receipt is not None:
            cls._validate_artifact_receipt(
                project, replacement_receipt, "inputs/replacement-image",
            )
        replacement_sheet = state.get("replacement_sheet")
        if (
            not isinstance(replacement_sheet, dict)
            or bindings.get("replacement_sheet_sha256")
            != cls._json_sha256(replacement_sheet)
        ):
            raise ValueError(
                "stored H3 preparation replacement sheet binding does not match"
            )
        if replacement_sheet.get("version") != REPLACEMENT_SHEET_VERSION:
            raise ValueError(
                "stored H3 preparation replacement sheet version does not match"
            )
        cls._validated_stored_replacement_sheet(project, state)
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
                "join_mode": item["join_mode"],
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
            muted_receipt = entry["muted_video"]
            muted_path = (
                muted_receipt.get("path")
                if isinstance(muted_receipt, dict)
                else None
            )
            if not isinstance(muted_path, str):
                raise ValueError("stored H3 artifact receipt fields are invalid")
            cls._validate_artifact_receipt(project, muted_receipt, muted_path)
            cls._validate_artifact_receipt(
                project, entry["prompt"], f"{base}/h3_prompt.txt",
            )
            normalized[str(index)] = dict(entry)
        return normalized

    @classmethod
    def _validated_stored_replacement_sheet(
        cls,
        project: Path,
        state: dict[str, Any],
        *,
        allow_prepared: bool = False,
    ) -> Path | None:
        """Validate the frozen project-wide Image N binding and its checkpoint."""

        raw = state.get("replacement_sheet")
        if raw is None:
            return None
        expected_keys = {
            "version", "status", "spec", "spec_sha256",
            "user_image_present", "h3_image_reference",
            "h3_reference_images", "upstream_bindings", "seedream", "output",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_keys
            or raw.get("version") != REPLACEMENT_SHEET_VERSION
            or isinstance(raw.get("version"), bool)
            or not isinstance(raw.get("user_image_present"), bool)
            or raw["user_image_present"]
            != (
                state.get("replacement_image_path") is not None
                and cls._enhanced_index_uses_user_image(
                    state.get("enhanced_element_index")
                    if isinstance(state.get("enhanced_element_index"), dict)
                    else {}
                )
            )
            or not isinstance(raw.get("spec"), dict)
            or not isinstance(state.get("enhanced_element_index"), dict)
            or not isinstance(raw.get("h3_reference_images"), list)
        ):
            raise ValueError("stored replacement sheet state is invalid")
        try:
            validate_replacement_sheet_spec(raw["spec"], allow_empty=True)
        except ValueError as exc:
            raise ValueError("stored replacement sheet state is invalid") from exc
        if (
            raw.get("spec_sha256") != raw["spec"].get("spec_sha256")
            or raw.get("spec_sha256")
            != replacement_sheet_spec_sha256(raw["spec"])
            or raw["spec"].get("enhanced_element_index_sha256")
            != cls._json_sha256(state["enhanced_element_index"])
        ):
            raise ValueError("stored replacement sheet state is invalid")

        stored_segments = cls._validated_stored_segments(
            state,
            missing_message="stored replacement sheet segments are missing",
        )
        prepared_for_binding = [
            {
                **segment,
                "source": (
                    project / "work" / "segments"
                    / f"{int(segment['index']):04d}" / "source.mp4"
                ),
            }
            for segment in stored_segments
        ]
        object_replace = state.get("object_replace")
        voice_references = state.get("voice_references")
        effective_request = cls._validated_effective_request(
            state.get("effective_request")
        )
        if (
            not isinstance(object_replace, dict)
            or object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION
            or not isinstance(voice_references, list)
        ):
            raise ValueError("stored replacement sheet upstream state is invalid")
        replacement = (
            project / "inputs" / "replacement-image"
            if state.get("replacement_image_path") is not None
            else None
        )
        expected_upstream_bindings = cls._replacement_sheet_upstream_bindings(
            project,
            prepared_for_binding,
            state["enhanced_element_index"],
            object_replace,
            voice_references,
            effective_request,
            replacement,
        )
        if raw.get("upstream_bindings") != expected_upstream_bindings:
            raise ValueError(
                "stored replacement sheet upstream bindings do not match"
            )

        expected_spec = build_replacement_sheet_spec(
            state["enhanced_element_index"],
        )
        if raw["spec"] != expected_spec:
            raise ValueError(
                "stored replacement sheet tiles do not match the global index"
            )

        status = raw.get("status")
        tiles = raw["spec"].get("tiles")
        if not isinstance(tiles, list):
            raise ValueError("stored replacement sheet tiles are invalid")
        if status == "not_needed":
            if (
                tiles
                or raw.get("h3_image_reference") is not None
                or raw.get("seedream") is not None
                or raw.get("output") is not None
                or raw["h3_reference_images"] != []
            ):
                raise ValueError("stored replacement sheet skip state is invalid")
            return None
        if status not in {"prepared", "succeeded"} or not tiles:
            raise ValueError("stored replacement sheet is not reusable")

        expected_references: list[dict[str, Any]] = []
        if raw["user_image_present"]:
            canonical_user = (
                project / "work" / "global-reference-sheet"
                / "user-reference.png"
            )
            canonical_receipt = cls._project_artifact_receipt(
                project,
                canonical_user,
                "work/global-reference-sheet/user-reference.png",
            )
            canonical_bytes = canonical_user.read_bytes()
            if (
                not canonical_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                or len(canonical_bytes) < 9
            ):
                raise ValueError("stored user reference is not PNG")
            expected_references.append({
                "image_reference": "Image 2",
                "role": "user_replacement",
                **canonical_receipt,
            })

        # The sheet is always H3 Image 1; the user image follows as Image 2.
        expected_image_reference = "Image 1"
        if raw.get("h3_image_reference") != expected_image_reference:
            raise ValueError("stored H3 image reference order is invalid")

        seedream = raw.get("seedream")
        if not isinstance(seedream, dict) or set(seedream) != {
            "model", "endpoint", "prompt", "spec", "inputs",
            "request_sha256", "receipt",
        }:
            raise ValueError("stored Seedream state is invalid")
        model = seedream.get("model")
        endpoint = seedream.get("endpoint")
        model_characters = frozenset(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        if (
            not isinstance(model, str)
            or not 1 <= len(model) <= 128
            or not model[0].isalnum()
            or not model.isascii()
            or any(character not in model_characters for character in model)
            or not isinstance(endpoint, str)
            or not endpoint.startswith("https://")
            or not endpoint.isascii()
            or any(not 0x21 <= ord(character) <= 0x7E for character in endpoint)
            or not isinstance(seedream.get("inputs"), list)
        ):
            raise ValueError("stored Seedream fields are invalid")
        prompt_path = cls._validate_artifact_receipt(
            project,
            seedream["prompt"],
            "work/global-reference-sheet/prompt.txt",
        )
        spec_path = cls._validate_artifact_receipt(
            project,
            seedream["spec"],
            "work/global-reference-sheet/spec.json",
        )
        try:
            persisted_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored replacement sheet spec is invalid") from exc
        if persisted_spec != raw["spec"]:
            raise ValueError("stored replacement sheet spec does not match")
        allowed_inputs = [
            "work/global-reference-sheet/canvas.png",
            *(
                ["work/global-reference-sheet/user-reference.png"]
                if raw["user_image_present"]
                else []
            ),
        ]
        if len(seedream["inputs"]) != len(allowed_inputs):
            raise ValueError("stored Seedream input order is invalid")
        for receipt, relative_path in zip(seedream["inputs"], allowed_inputs):
            cls._validate_artifact_receipt(project, receipt, relative_path)
        prompt_sha256 = seedream["prompt"].get("sha256")
        input_sha256 = [
            receipt.get("sha256")
            if isinstance(receipt, dict)
            else None
            for receipt in seedream["inputs"]
        ]
        request_binding = {
            "version": 1,
            "endpoint": seedream["endpoint"],
            "model": seedream["model"],
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "size": SEEDREAM_OUTPUT_SIZE,
            "response_format": "b64_json",
            "watermark": False,
        }
        request_sha256 = seedream.get("request_sha256")
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("stored Seedream prompt is invalid") from exc
        if (
            not prompt_text.strip()
            or not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in request_sha256
            )
            or cls._json_sha256(request_binding) != request_sha256
        ):
            raise ValueError("stored Seedream request binding is invalid")

        if status == "prepared":
            if (
                not allow_prepared
                or raw.get("output") is not None
                or seedream.get("receipt") is not None
                or raw["h3_reference_images"] != expected_references
            ):
                raise ValueError("stored replacement sheet is not reusable")
            return None

        output = raw.get("output")
        output_path = cls._validate_artifact_receipt(
            project,
            output,
            "work/global-reference-sheet/reference-sheet.png",
        )
        output_bytes = output_path.read_bytes()
        try:
            output_media_type = _validated_image_media_type(output_bytes)
        except ValueError as exc:
            raise ValueError(
                "stored replacement sheet is not a valid 2048x2048 image"
            ) from exc
        if output_media_type != "image/png":
            raise ValueError("stored replacement sheet is not PNG")
        expected_references.insert(0, {
            "image_reference": expected_image_reference,
            "role": "global_replacement_sheet",
            **output,
            "spec_sha256": raw["spec_sha256"],
        })
        if raw["h3_reference_images"] != expected_references:
            raise ValueError("stored H3 image reference order is invalid")
        receipt_path = cls._validate_artifact_receipt(
            project,
            seedream["receipt"],
            "work/global-reference-sheet/seedream-receipt.json",
        )
        try:
            seedream_receipt = json.loads(
                receipt_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored Seedream receipt is invalid") from exc
        endpoint = (
            seedream_receipt.get("endpoint")
            if isinstance(seedream_receipt, dict)
            else None
        )
        receipt_request_sha256 = (
            seedream_receipt.get("request_sha256")
            if isinstance(seedream_receipt, dict)
            else None
        )
        if (
            not isinstance(seedream_receipt, dict)
            or seedream_receipt.get("version") != 1
            or seedream_receipt.get("provider") != "ark_seedream"
            or seedream_receipt.get("status") != "succeeded"
            or endpoint != seedream["endpoint"]
            or seedream_receipt.get("model") != seedream["model"]
            or seedream_receipt.get("prompt_sha256") != prompt_sha256
            or seedream_receipt.get("input_sha256") != input_sha256
            or seedream_receipt.get("size") != SEEDREAM_OUTPUT_SIZE
            or receipt_request_sha256 != request_sha256
        ):
            raise ValueError(
                "stored Seedream receipt does not match its frozen request"
            )
        return output_path

    def _validated_prepared_seedream_attempt(
        self,
        project: Path,
        replacement_sheet: dict[str, Any],
    ) -> str:
        """Return generate/replay/regenerate after a read-only paid-attempt audit."""

        seedream = replacement_sheet.get("seedream")
        if not isinstance(seedream, dict):
            raise ValueError("stored Seedream state is invalid")
        directory = project / "work" / "global-reference-sheet"
        receipt_path = directory / "seedream-receipt.json"
        claim_path = directory / "seedream-receipt.json.post-claim"
        marker_path = directory / "seedream-receipt.json.response.json"
        result_path = directory / "seedream-receipt.json.result"
        provider_output = directory / "provider-output.bin"
        final_output = directory / "reference-sheet.png"

        if not receipt_path.exists():
            if any(
                path.exists()
                for path in (
                    claim_path,
                    marker_path,
                    result_path,
                    provider_output,
                    final_output,
                )
            ):
                raise ValueError(
                    "Seedream attempt evidence exists without its receipt"
                )
            expected_endpoint = (
                self.settings.seedream_api_base + "/images/generations"
            )
            if (
                seedream.get("endpoint") != expected_endpoint
                or seedream.get("model") != self.settings.seedream_model
                or not self.settings.seedream_api_key
            ):
                raise ValueError(
                    "current Seedream configuration does not match the frozen request"
                )
            return "generate"

        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("stored Seedream receipt path is invalid")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored Seedream receipt is invalid") from exc
        input_sha256 = [
            value.get("sha256") if isinstance(value, dict) else None
            for value in seedream.get("inputs", [])
        ]
        if (
            not isinstance(receipt, dict)
            or receipt.get("version") != 1
            or receipt.get("provider") != "ark_seedream"
            or receipt.get("endpoint") != seedream.get("endpoint")
            or receipt.get("model") != seedream.get("model")
            or receipt.get("prompt_sha256")
            != seedream.get("prompt", {}).get("sha256")
            or receipt.get("input_sha256") != input_sha256
            or receipt.get("size") != SEEDREAM_OUTPUT_SIZE
            or receipt.get("request_sha256")
            != seedream.get("request_sha256")
            or receipt.get("status")
            not in {
                "submitting", "response_received", "succeeded",
                "rejected", "submission_unknown",
            }
        ):
            raise ValueError(
                "stored Seedream receipt does not match its frozen request"
            )

        if receipt.get("status") in {"rejected", "submission_unknown"}:
            # A terminally rejected/unknown attempt has no recoverable
            # response; the caller resets its artifacts and re-POSTs the
            # frozen request instead of blocking the resume paths.
            return "regenerate"

        if claim_path.is_symlink() or not claim_path.is_file():
            raise ValueError("stored Seedream POST claim is missing")
        try:
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored Seedream POST claim is invalid") from exc
        if (
            not isinstance(claim, dict)
            or claim.get("version") != 1
            or claim.get("kind") != "seedream_paid_post"
            or claim.get("request_sha256") != seedream.get("request_sha256")
        ):
            raise ValueError("stored Seedream POST claim is invalid")

        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError("stored Seedream response is not locally recoverable")
        try:
            result = result_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                "stored Seedream response is not locally recoverable"
            ) from exc
        try:
            media_type = _validated_image_media_type(result)
        except ValueError as exc:
            raise ValueError(
                "stored Seedream result is not a complete image"
            ) from exc
        result_sha256 = hashlib.sha256(result).hexdigest()

        if marker_path.exists():
            if marker_path.is_symlink() or not marker_path.is_file():
                raise ValueError("stored Seedream response marker is invalid")
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("stored Seedream response marker is invalid") from exc
            if (
                not isinstance(marker, dict)
                or marker.get("version") != 1
                or marker.get("kind") != "seedream_response"
                or marker.get("request_sha256")
                != seedream.get("request_sha256")
                or marker.get("result_file") != result_path.name
                or marker.get("result_bytes") != len(result)
                or marker.get("result_sha256") != result_sha256
                or marker.get("media_type") != media_type
                or marker.get("width") != SEEDREAM_OUTPUT_WIDTH
                or marker.get("height") != SEEDREAM_OUTPUT_HEIGHT
            ):
                raise ValueError("stored Seedream response marker is invalid")
        elif receipt.get("status") != "submitting":
            raise ValueError("stored Seedream response marker is missing")

        if (
            receipt.get("result_bytes") not in {None, len(result)}
            or receipt.get("result_sha256") not in {None, result_sha256}
        ):
            raise ValueError("stored Seedream receipt result binding is invalid")
        if receipt.get("status") == "succeeded" and (
            receipt.get("result_file") != result_path.name
            or receipt.get("result_bytes") != len(result)
            or receipt.get("result_sha256") != result_sha256
            or receipt.get("response_marker_file") != marker_path.name
            or receipt.get("output_sha256") != result_sha256
        ):
            raise ValueError("stored Seedream succeeded receipt is invalid")
        return "replay"

    def _load_r2v_resume(
        self, project_id: str,
    ) -> tuple[
        Path, list[dict[str, Any]], Path | None, Path | None,
        dict[str, Any], list[Path],
    ]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
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
        if manifest is None:
            raise ValueError(
                "project is missing the current H3 preparation manifest"
            )
        replacement_sheet = self._validated_stored_replacement_sheet(
            project, state,
        )
        voice_references = self._stored_voice_references(project, state)
        state_receipts = self._validated_submission_receipts(state)
        manifest_version = (
            (state.get("h3_prepared_artifacts") or {}).get("version")
            if isinstance(state.get("h3_prepared_artifacts"), dict)
            else None
        )
        if manifest_version != H3_PREPARED_ARTIFACTS_VERSION:
            raise ValueError("stored H3 preparation manifest is invalid")
        current_first_submit_resume = False
        expected_receipt_segments = {
            str(item["index"]) for item in raw_segments
        }
        if state_receipts is None:
            has_submit_directory = any(
                path.is_dir()
                for path in (project / "logs").glob("picsart*-submit")
            )
            if has_submit_directory:
                raise ValueError(
                    f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} has "
                    "Picsart submission evidence without its receipt ledger"
                )
            # The current manifest is frozen before global Picsart preflight.
            # A preflight failure therefore legitimately leaves neither a
            # ledger nor a submit directory. This is the only state in which
            # Picsart resume may perform the first paid submission.
            current_first_submit_resume = True
        elif set(state_receipts["segments"]) != expected_receipt_segments:
            raise ValueError(
                f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} requires a "
                "complete Picsart receipt ledger; reconcile it before any paid submit"
            )
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
            muted = project / manifest[str(index)]["muted_video"]["path"]
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
            elif not has_state_receipt and not current_first_submit_resume:
                raise ValueError("Picsart segment receipt is missing")
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
        return (
            project,
            prepared,
            replacement,
            replacement_sheet,
            effective_request,
            voice_references,
        )

    def validate_r2v_resume(self, project_id: str) -> None:
        self._load_r2v_resume(project_id)

    def resume_r2v(self, project_id: str) -> None:
        (
            project,
            prepared,
            replacement,
            replacement_sheet,
            request,
            voice_references,
        ) = (
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
                        replacement_sheet,
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
            join_modes = self._validated_join_modes(prepared)
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
                    "join_modes": join_modes,
                    "output": final,
                },
                outputs,
                lambda: media.concat(
                    outputs,
                    final,
                    target_durations=[item["duration_s"] for item in prepared],
                    join_modes=join_modes,
                ),
                lambda _value: [final],
            )
            self._replace_dialogue_audio_track(project, prepared, final, log)
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
        numbered = [
            directory
            for directory in attempts.glob("attempt-*")
            if directory.name.rsplit("-", 1)[-1].isdigit()
        ]
        numbered.sort(
            key=lambda directory: int(directory.name.rsplit("-", 1)[-1]),
            reverse=True,
        )
        for directory in numbered:
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

        request = self._validated_effective_request(request)
        processing = request["processing"]
        generation_config = self._generation_config(request)
        stages: list[str] = []
        if processing["remove_subtitle"]:
            stages.append("full_screen_text_erase")
        if processing["remove_logo"]:
            stages.append("full_screen_icon_erase")
        use_mediakit = (
            self.settings.enable_mediakit
            and processing["optimize_image"]
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
                    validator=normalize_visual_prompt,
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
            validator=_validate_translation_shape,
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
    def _dialogue_generation_payload(
        brief: str,
        target_language: str,
        prepared: list[dict[str, Any]],
    ) -> dict[str, Any]:
        segments = [
            {
                "index": int(item["index"]),
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
                "visual_prompt": str(item["visual_prompt"]),
            }
            for item in prepared
        ]
        duration_s = segments[-1]["end_s"] if segments else 0.0
        return {
            "brief": brief,
            "target_language": target_language,
            "duration_s": duration_s,
            "segments": segments,
        }

    def _dialogue_generation_request(
        self,
        brief: str,
        target_language: str,
        prepared: list[dict[str, Any]],
    ) -> ProviderResult[Any]:
        skill = DIALOGUE_WRITER_SKILL.read_text(encoding="utf-8")
        if target_language == SAME_AS_SOURCE_LANGUAGE:
            language_instruction = (
                "台词语言使用源视频画面的原始语言；"
                "target_language 的占位值只表示语言不变，没有源台词可抄。"
            )
        else:
            language_instruction = f"台词语言为{target_language}。"
        instruction = (
            skill
            + "\n\n本次执行口播台词生成；"
            + language_instruction
            + "最终输出严格遵循本次注入的 JSON Schema。"
        )
        return self.deepseek.complete_structured(
            instruction,
            self._dialogue_generation_payload(brief, target_language, prepared),
            DIALOGUE_GENERATION_SCHEMA,
            validator=_validate_dialogue_generation_shape,
        )

    @staticmethod
    def _normalize_generated_dialogue(
        value: Any, prepared: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Same contract as translated_lines: [{start_s, end_s, text}] in
        # absolute source time. Unusable entries are dropped instead of
        # blocking the project; multiple lines inside one segment split that
        # segment's window by text length.
        parsed = value.get("lines") if isinstance(value, dict) else None
        candidates = parsed if isinstance(parsed, list) else []
        by_segment: dict[int, list[str]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            segment_index = candidate.get("segment_index")
            text = candidate.get("text")
            if (
                type(segment_index) is not int
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue
            by_segment.setdefault(segment_index, []).append(text.strip())
        generated: list[dict[str, Any]] = []
        for item in prepared:
            texts = by_segment.get(int(item["index"]))
            if not texts:
                continue
            start = float(item["start_s"])
            end = float(item["end_s"])
            total_chars = sum(len(text) for text in texts)
            for position, text in enumerate(texts):
                line_start = (
                    start
                    if position == 0
                    else generated[-1]["end_s"]
                )
                line_end = (
                    end
                    if position == len(texts) - 1
                    else start
                    + (end - start)
                    * sum(len(text) for text in texts[: position + 1])
                    / total_chars
                )
                generated.append({
                    "start_s": line_start,
                    "end_s": line_end,
                    "text": text,
                })
        return generated

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
        if not isinstance(occurrences, list) or not occurrences:
            # If occurrence metadata is unusable, including the record is less
            # destructive than blocking the project or silently losing it.
            return True
        return any(
            isinstance(occurrence, dict)
            and occurrence.get("segment_index") == segment_index
            for occurrence in occurrences
        )

    @classmethod
    def _segment_has_replaced_person(
        cls, element_index: dict[str, Any], segment_index: int
    ) -> bool:
        people = element_index.get("people")
        if not isinstance(people, dict):
            return False
        return any(
            isinstance(record, dict)
            and isinstance(record.get("replacement"), dict)
            and cls._appears_in_segment(record, segment_index)
            for record in people.values()
        )

    @classmethod
    def _fusion_authority_block(
        cls,
        element_index: dict[str, Any],
        segment_index: int,
        replacement_sheet: dict[str, Any] | None = None,
    ) -> str:
        lines = [
            "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN",
            (
                "This block is the sole authority. Ignore any conflicting "
                "replacement or voice statement elsewhere in the prompt."
            ),
        ]
        sheet_reference: str | None = None
        sheet_tiles: dict[str, dict[str, Any]] = {}
        if isinstance(replacement_sheet, dict):
            candidate_reference = replacement_sheet.get("h3_image_reference")
            spec = replacement_sheet.get("spec")
            tiles = spec.get("tiles") if isinstance(spec, dict) else None
            if isinstance(candidate_reference, str) and candidate_reference.strip():
                sheet_reference = cls._one_line(candidate_reference)
            if isinstance(tiles, list):
                sheet_tiles = {
                    tile["stable_key"]: tile
                    for tile in tiles
                    if isinstance(tile, dict)
                    and isinstance(tile.get("stable_key"), str)
                }
        if sheet_reference is not None and sheet_tiles:
            lines.append(
                f"The single global replacement appearance sheet is {sheet_reference}. "
                "For every REPLACE item, copy the new identity or design only from "
                "its bound tile; never borrow another tile or the source Video 1 "
                "appearance. The same tile is authoritative in every segment."
            )
            if replacement_sheet.get("user_image_present") is True:
                lines.append(
                    "The user reference image is Image 2. It already shaped its "
                    "bound tile during sheet creation and adds no further "
                    "constraint; never copy from Image 2 directly."
                )
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
                    tile = sheet_tiles.get(stable_id)
                    if sheet_reference is not None and isinstance(tile, dict):
                        reference_text = (
                            f"{sheet_reference}, {cls._one_line(tile.get('tile_id'))} "
                            f"(row {tile.get('row')}, column {tile.get('column')})"
                        )
                        if (
                            replacement.get("source") == "image_1"
                            and replacement_sheet.get("user_image_present") is True
                        ):
                            reference_text += (
                                "; this tile was designed from the user image "
                                "Image 2, but the tile is the final appearance "
                                "authority"
                            )
                    else:
                        reference_text = (
                            "Image 1"
                            if replacement.get("source") == "image_1"
                            else "generated variant"
                        )
                    line = (
                        f"- {stable_id} | REPLACE | target: "
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
                            " | person policy: replace with a clearly different "
                            "identity; the new person must differ visibly from the "
                            "source in at least two of: hair length, hair texture, "
                            "hair color, face shape, facial hair or other prominent "
                            "facial features; keep only explicitly stated gender "
                            "and age group; when clothing is visible, change its "
                            "design; do not infer unseen traits"
                        )
                elif category == "scenes" and isinstance(replacement, dict):
                    line += (
                        " | scene policy: same scene type, use, and layout but a "
                        "clearly different instance and visible design details; preserve "
                        "color tone, composition, camera, shot movement, focal length, "
                        "and lighting"
                    )
                elif category == "entities" and isinstance(replacement, dict):
                    line += (
                        " | entity policy: preserve function, action interface, scale, "
                        "contact, and relation semantics; visibly change material, "
                        "color, breed, outline, structure, or form as specified by target"
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
        replacement_sheet: dict[str, Any],
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
                "h3_reference_images": replacement_sheet.get(
                    "h3_reference_images", []
                ),
                "replacement_sheet": {
                    "spec_sha256": replacement_sheet.get("spec_sha256"),
                    "h3_image_reference": replacement_sheet.get(
                        "h3_image_reference"
                    ),
                    "tiles": (
                        (replacement_sheet.get("spec") or {}).get("tiles", [])
                        if isinstance(replacement_sheet.get("spec"), dict)
                        else []
                    ),
                },
                "authoritative_constraints": self._fusion_authority_block(
                    element_index, index, replacement_sheet,
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
                    validator=lambda value: normalize_fusion_prompt(
                        value,
                        input_sha256=fusion_input_sha256,
                        segment_index=index,
                        authoritative_constraints=fusion_payload[
                            "authoritative_constraints"
                        ],
                    ),
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
            processing = self._validated_effective_request(request)["processing"]
            face_block = ""
            sheet_reference = replacement_sheet.get("h3_image_reference")
            if (
                isinstance(sheet_reference, str)
                and sheet_reference.strip()
                and self._segment_has_replaced_person(element_index, index)
            ):
                face_block = (
                    "FACE REPLACEMENT IS MANDATORY AND TAKES PRIORITY OVER "
                    "EVERYTHING ELSE. Every person in Video 1 must appear with "
                    f"the face from their tile in {sheet_reference}: face "
                    "shape, eyes, nose, mouth, skin tone, and hairstyle all "
                    "come from that tile, never from Video 1. Keeping the "
                    "source face is a failure, even if everything else is "
                    "correct.\n\n"
                )
            overlay_targets = []
            if processing["remove_subtitle"]:
                overlay_targets.append(
                    "burned-in subtitles, captions, and on-screen text"
                )
            if processing["remove_logo"]:
                overlay_targets.append("watermarks and logos")
            overlay_block = ""
            if overlay_targets:
                overlay_block = (
                    "\n\nOverlay removal: Video 1 may contain "
                    + " and ".join(overlay_targets)
                    + ". Remove all of them from the generated video and "
                    "reconstruct the occluded background naturally. Do not "
                    "reproduce, translate, or re-render any source on-screen "
                    "text, watermark, or logo."
                )
            dialogue = "\n".join(
                f"[{line['start_s']:.3f}-{line['end_s']:.3f}s] {line['text']}"
                for line in item["dialogue"]
            )
            final_prompt = face_block + prompt + overlay_block + (
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
            if self._segment_has_replaced_person(element_index, index):
                # Mask every source face so H3 must take the face from its
                # tile.  Masking is load-bearing for identity takeover, so it
                # retries a few times; persistent failures degrade to the
                # unmasked reference instead of blocking the render.  Only a
                # missing detection model (a configuration error) stays fatal.
                masked = segment_dir / "source-faces-masked.mp4"
                mask_failures = 0

                def _mask_retry(exc: Exception) -> bool:
                    nonlocal mask_failures
                    mask_failures += 1
                    if (
                        isinstance(exc, face_mask.FaceMaskError)
                        and "model is missing" in str(exc)
                    ):
                        return False
                    return mask_failures < FACE_MASK_MAX_ATTEMPTS

                try:
                    self._node(
                        log,
                        "mask-reference-faces",
                        scope,
                        {"source": muted, "output": masked},
                        [muted],
                        lambda: face_mask.mask_faces(muted, masked),
                        lambda _value: [masked],
                        retry_if=_mask_retry,
                    )
                except Exception as exc:
                    if (
                        isinstance(exc, face_mask.FaceMaskError)
                        and "model is missing" in str(exc)
                    ):
                        raise
                    log.event(
                        "face_mask_degraded",
                        scope=scope,
                        attempts=mask_failures,
                        error=str(exc),
                        impact=(
                            "source faces stay visible in the reference "
                            "video; H3 identity takeover is weakened"
                        ),
                    )
                else:
                    muted = masked
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
        replacement_sheet: Path | None,
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
            f"work/segments/{index:04d}/{self._muted_artifact_name(muted)}",
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
            valid_paths = {(
                project / "work" / "voice-references"
                / f"audio-{slot:02d}.wav"
            ).resolve()}
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
        request = self._validated_effective_request(request)
        output_request = request["output"]
        resolution = output_request["resolution"]
        aspect_ratio = output_request["aspect_ratio"]
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

        provider_images: list[Path] = []
        video_url: str | None = None
        image_urls: list[str] = []
        audio_urls: list[str] = []
        binding_state = self.store.read(project_id)
        prepared_manifest = binding_state.get("h3_prepared_artifacts")
        prepared_manifest_version = (
            prepared_manifest.get("version")
            if isinstance(prepared_manifest, dict)
            else None
        )
        if prepared_manifest_version == H3_PREPARED_ARTIFACTS_VERSION:
            preparation_sha256 = self._json_sha256(prepared_manifest)
        else:
            raise ValueError("stored H3 preparation manifest is invalid")
        if existing_task_id is None:
            stored_state = binding_state
            stored_sheet = stored_state.get("replacement_sheet")
            if not isinstance(stored_sheet, dict):
                raise ValueError("stored replacement sheet state is invalid")
            validated_sheet = self._validated_stored_replacement_sheet(
                project, stored_state,
            )
            supplied_sheet = (
                replacement_sheet.resolve(strict=True)
                if replacement_sheet is not None
                else None
            )
            if (
                supplied_sheet != validated_sheet
                or stored_sheet["user_image_present"]
                != (
                    replacement is not None
                    and self._enhanced_index_uses_user_image(
                        stored_state.get("enhanced_element_index")
                        if isinstance(
                            stored_state.get("enhanced_element_index"),
                            dict,
                        )
                        else {}
                    )
                )
            ):
                raise ValueError(
                    "prepared Picsart image binding does not match"
                )
            provider_images.extend(
                (project / reference["path"]).resolve(strict=True)
                for reference in stored_sheet["h3_reference_images"]
            )
            if len(provider_images) != len(set(provider_images)):
                raise ValueError("prepared Picsart image files must be unique")
            video_url = self._validated_provider_asset_url(
                self.register_asset(project_id, muted)
            )
            image_urls = [
                self._validated_provider_asset_url(
                    self.register_asset(project_id, provider_image)
                )
                for provider_image in provider_images
            ]
            if len(image_urls) != len(set(image_urls)):
                raise ValueError("prepared Picsart image URLs must be unique")
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
                "reference_image_urls": image_urls,
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
            provider_images=tuple(provider_images),
            voice_references=tuple(resolved_voice_references),
            video_url=video_url,
            image_urls=tuple(image_urls),
            audio_urls=tuple(audio_urls),
            provider_duration=provider_duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            target_duration_s=float(duration),
            request_sha256=self._json_sha256(request_payload),
            preparation_sha256=preparation_sha256,
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
                image_urls=list(prepared.image_urls),
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
            current = prepared
            task_id = prepared.existing_task_id
            resubmits = 0
            while True:
                # Submit/poll node names carry the run identity, so every
                # (re)submission keeps its own complete receipt/log chain.
                submit_prefix = current.run_identity
                task_started = time.monotonic()
                if task_id is None:
                    task_id = self._node(
                        log,
                        f"{submit_prefix}-submit",
                        scope,
                        current.submit_input(),
                        [current.muted]
                        + list(current.provider_images)
                        + list(current.voice_references),
                        lambda current=current: self._submit_prepared_picsart(current),
                        retry_if=lambda _exc: False,
                    )
                else:
                    log.event(
                        "picsart_task_reused",
                        segment_index=index,
                        task_id=task_id,
                    )

                def poll_once(
                    task_id: str = task_id,
                    current: PreparedPicsartRequest = current,
                    task_started: float = task_started,
                ) -> ProviderResult[str]:
                    if (
                        time.monotonic() - task_started
                        > self.settings.picsart_task_hang_limit_s
                    ):
                        raise PicsartTaskHungError(
                            f"Picsart task {task_id} stayed non-terminal past "
                            f"{self.settings.picsart_task_hang_limit_s}s"
                        )
                    try:
                        return self.picsart.poll(task_id)
                    except ProviderFailure as exc:
                        if self._poll_record_is_terminal_failure(exc.record):
                            self._mark_submission_task_failed(
                                current.project_id, index, task_id,
                            )
                        raise

                try:
                    generated_url = self._node(
                        log,
                        f"{submit_prefix}-poll",
                        scope,
                        {"task_id": task_id},
                        [],
                        poll_once,
                        retry_if=lambda exc: not (
                            isinstance(exc, PicsartTaskHungError)
                            or (
                                isinstance(exc, ProviderFailure)
                                and self._poll_record_is_terminal_failure(
                                    exc.record
                                )
                            )
                        ),
                    )
                except (ProviderFailure, PicsartTaskHungError) as exc:
                    hung = isinstance(exc, PicsartTaskHungError)
                    if not hung and not self._poll_record_is_terminal_failure(
                        exc.record
                    ):
                        raise
                    if hung:
                        self._mark_submission_task_failed(
                            current.project_id, index, task_id,
                        )
                        reason = "task_hung"
                        error_text = str(exc)
                    else:
                        reason = (
                            "asset_fetch"
                            if self._poll_failure_is_asset_fetch(exc.record)
                            else "terminal_failure"
                        )
                        error_text = str(exc)
                    # Any terminal task failure (or a task hung past the hang
                    # limit) resubmits the same frozen segment under a fresh
                    # run identity — fresh asset URLs included — with no cap.
                    resubmits += 1
                    log.event(
                        "picsart_terminal_failure_resubmit",
                        segment_index=index,
                        resubmit=resubmits,
                        failed_task_id=task_id,
                        reason=reason,
                        error=error_text,
                    )
                    current = replace(
                        prepared,
                        run_identity=f"{node_prefix}-resubmit-{resubmits:02d}",
                        existing_task_id=None,
                    )
                    self._begin_submission_batch([current])
                    task_id = None
                    continue
                break
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
        request = self._validated_effective_request(request)
        processing = request["processing"]
        enabled = self.settings.enable_mediakit and bool(
            processing["optimize_image"]
        )
        return {
            "remove_subtitle": enabled and bool(
                processing["remove_subtitle"]
            ),
            "remove_watermark": enabled and bool(
                processing["remove_logo"]
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
            "join_mode": item["join_mode"],
            "status": item.get("status", "running"),
            "visual_prompt": item.get("visual_prompt"),
            "final_prompt": item.get("final_prompt"),
            "dialogue": item.get("dialogue", []),
        }
