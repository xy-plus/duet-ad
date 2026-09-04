from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app import media
from app.config import Settings
from app.fusion_schema import fusion_prompt_schema, normalize_fusion_prompt
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
from app.video_maker_schemas import (
    PROJECT_INDEX_SCHEMA,
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
    normalize_visual_prompt,
)


VIDEO_SKILL = Path(__file__).resolve().parents[1] / "skills" / "video-maker" / "SKILL.md"
R2V_SKILL = Path(__file__).resolve().parents[1] / "skills" / "h3-max-r2v-prompt" / "SKILL.md"
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
        self._mediakit_gate = threading.BoundedSemaphore(
            settings.mediakit_concurrency,
        )

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
                retry_if(last_error) if retry_if is not None else True
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

    def run(self, project_id: str) -> None:
        project = self.store.path(project_id)
        log = ResearchLog(project)
        state = self.store.read(project_id)
        source = project / state["source_path"]
        replacement = (
            project / state["replacement_image_path"]
            if state.get("replacement_image_path")
            else None
        )
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
                + "\n\n严格执行项目级索引阶段；只根据输入帧填写本次注入的 Schema。",
                index_request,
                PROJECT_INDEX_SCHEMA,
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

        # This is the public-input barrier.  No paid segment branch starts
        # before every visual prompt, translation, localization and the
        # single project index above have completed.
        self._progress(
            project_id, 57, "h3-max-r2v", segments=public_segments,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(prepared)),
        ) as pool:
            futures = [
                pool.submit(
                    self._generate_segment,
                    project_id,
                    project,
                    item,
                    element_index,
                    replacement,
                    request,
                    log,
                )
                for item in prepared
            ]
            outputs = [future.result() for future in futures]

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
        if any(path.is_dir() for path in logs.glob("picsart*-submit")):
            raise ValueError("project has Picsart submission evidence")
        source_path = state.get("source_path")
        if not isinstance(source_path, str):
            raise TypeError("stored source path is invalid")
        source = project / source_path
        if not source.is_file():
            raise FileNotFoundError("stored source is missing")
        replacement = (
            project / state["replacement_image_path"]
            if state.get("replacement_image_path")
            else None
        )
        if replacement is not None and not replacement.is_file():
            raise FileNotFoundError("replacement image is missing")
        raw_segments = state.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("stored segments are missing")
        segments: list[dict[str, Any]] = []
        selections: dict[int, list[dict[str, Any]]] = {}
        for raw in raw_segments:
            if not isinstance(raw, dict):
                raise TypeError("stored segment must be an object")
            segment = dict(raw)
            index = int(segment["index"])
            if index < 1 or float(segment["duration_s"]) <= 0:
                raise ValueError("stored segment metadata is invalid")
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
        prefixes = {
            path.name.removesuffix("-submit")
            for path in (project / "logs").glob("picsart-resume*-submit")
            if path.is_dir()
        }
        return sorted(prefixes, key=self._resume_prefix_key, reverse=True)

    def _next_resume_identity(self, project: Path) -> tuple[str, str]:
        prefixes = self._resume_prefixes(project)
        number = max((self._resume_prefix_key(value) for value in prefixes), default=0) + 1
        if number == 1:
            return "picsart-resume", "-resume"
        return f"picsart-resume-{number:02d}", f"-resume-{number:02d}"

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
            if isinstance(status_code, int) and 400 <= status_code < 500:
                return True
            body = response.get("body")
            envelope = body.get("response") if isinstance(body, dict) else None
            if isinstance(envelope, dict) and str(envelope.get("status") or "").upper() == "FAILED":
                return True
        return False

    def _active_resume_task(
        self, project: Path, scope: str,
    ) -> str | None:
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
                return output
        return None

    def _load_r2v_resume(
        self, project_id: str,
    ) -> tuple[Path, list[dict[str, Any]], Path | None, dict[str, Any]]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        if state.get("status") not in {"failed", "queued"}:
            raise ValueError("only a failed or explicitly queued project can resume Picsart")
        raw_segments = state.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("project has no reusable segments")
        replacement = (
            project / state["replacement_image_path"]
            if state.get("replacement_image_path")
            else None
        )
        if replacement is not None and not replacement.is_file():
            raise FileNotFoundError("replacement image is missing")

        prepared: list[dict[str, Any]] = []
        for raw in raw_segments:
            if not isinstance(raw, dict):
                raise TypeError("stored segment must be an object")
            item = dict(raw)
            index = int(item["index"])
            duration_s = float(item["duration_s"])
            if index < 1 or duration_s <= 0:
                raise ValueError("stored segment metadata is invalid")
            segment_dir = project / "work" / "segments" / f"{index:04d}"
            source = segment_dir / "source.mp4"
            muted = segment_dir / "source-muted.mp4"
            prompt_path = segment_dir / "h3_prompt.txt"
            for required in (source, muted, prompt_path):
                if not required.is_file():
                    raise FileNotFoundError(
                        f"reusable segment artifact is missing: {required.name}"
                    )
            final_prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not final_prompt:
                raise ValueError("reusable H3 prompt is empty")
            item.update({
                "source": source,
                "resume_muted": muted,
                "resume_prompt": final_prompt,
                "resume_task_id": self._active_resume_task(
                    project, self._segment_scope(index),
                ),
                "resume_output": self._completed_resume_output(
                    project, segment_dir, self._segment_scope(index),
                ),
                "status": "running",
            })
            prepared.append(item)
        prepared.sort(key=lambda item: int(item["index"]))
        return project, prepared, replacement, state.get("effective_request") or {}

    def validate_r2v_resume(self, project_id: str) -> None:
        self._load_r2v_resume(project_id)

    def resume_r2v(self, project_id: str) -> None:
        project, prepared, replacement, request = self._load_r2v_resume(project_id)
        log = ResearchLog(project)
        node_prefix, artifact_suffix = self._next_resume_identity(project)
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
                futures: list[concurrent.futures.Future[Path] | None] = []
                for item in prepared:
                    reusable = item.get("resume_output")
                    if isinstance(reusable, Path):
                        item["final_prompt"] = item["resume_prompt"]
                        item["task_id"] = item.get("resume_task_id")
                        item["status"] = "succeeded"
                        futures.append(None)
                        continue
                    futures.append(pool.submit(
                        self._generate_picsart_segment,
                        project_id,
                        item,
                        replacement,
                        request,
                        log,
                        final_prompt=item["resume_prompt"],
                        muted=item["resume_muted"],
                        node_prefix=node_prefix,
                        artifact_suffix=artifact_suffix,
                        existing_task_id=item.get("resume_task_id"),
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

        asr_calls = 0
        vocal_expected = self._yamnet_has_vocal_signal(yamnet)

        def transcribe_once() -> ProviderResult[list[dict[str, Any]]]:
            nonlocal asr_calls
            asr_calls += 1
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

        def apply_empty_asr_semantics(
            lines: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            if vocal_expected and not lines and asr_calls == 1:
                raise RuntimeError("empty ASR despite YAMNet vocal signal")
            return lines

        self._progress(project_id, 14, "whisper-asr")
        asr_lines = self._node(
            log,
            "whisper-asr",
            "project",
            {
                "audio": audio,
                "cli": self.settings.asr_cli,
                "model": self.settings.asr_model,
                "retry_empty_when_vocal": vocal_expected,
            },
            [audio, self.settings.asr_model],
            transcribe_once,
            transform=apply_empty_asr_semantics,
        )
        if vocal_expected and not asr_lines:
            log.event(
                "empty_asr_accepted",
                node="whisper-asr",
                scope="project",
                attempts=asr_calls,
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
                    + "\n\n本次只执行逐段视觉分析；通过唯一 submit_result 调用填写注入的 Schema。",
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
            "通过唯一一次 submit_result 函数调用提交完整结果。"
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
        if not isinstance(value, dict) or set(value) != {"lines"}:
            raise ValueError("translation response must be a closed object")
        parsed = value["lines"]
        if not isinstance(parsed, list) or len(parsed) != len(source_lines):
            raise ValueError("translation line count must match the source")
        translated: list[dict[str, Any]] = []
        for index, source in enumerate(source_lines):
            candidate = parsed[index]
            if not isinstance(candidate, dict) or set(candidate) != {"text"}:
                raise ValueError("translation line must contain only text")
            text = candidate["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("translation text must be non-empty")
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
            segments.append({"segment_index": int(item["index"]), "frames": frames})
        return {
            "phase": "project_index",
            "generation_config": generation_config,
            "segments": segments,
        }, images

    def _generate_segment(
        self,
        project_id: str,
        project: Path,
        item: dict[str, Any],
        element_index: dict[str, Any],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> Path:
        del project
        index = int(item["index"])
        scope = self._segment_scope(index)
        try:
            user_instruction = str(
                ((request.get("replacement_guidance") or {}).get("instruction"))
                or ""
            )
            skill = R2V_SKILL.read_text(encoding="utf-8")
            fusion_payload = {
                "segment_index": index,
                "visual_prompt": item["visual_prompt"],
                "global_element_index": element_index,
                "translated_dialogue": item["dialogue"],
                "user_replacement_instruction": user_instruction,
                "has_user_image": replacement is not None,
                "user_replacement_image": (
                    {"name": "Image 1", "sha256": self._sha256(replacement)}
                    if replacement is not None else None
                ),
            }
            fusion_schema, fusion_input_sha256 = fusion_prompt_schema(
                fusion_payload,
            )
            configured_fusion = request.get("fusion")
            # Legacy projects without a frozen internal choice used DeepSeek.
            # New requests always freeze the server choice at API ingress.
            fusion_agent = (
                configured_fusion.get("agent")
                if isinstance(configured_fusion, dict)
                else "deepseek"
            )
            if fusion_agent == "codex":
                fusion_client = self.codex
            elif fusion_agent == "deepseek":
                fusion_client = self.deepseek
            else:
                raise ValueError("frozen fusion agent is invalid")
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
                [replacement] if replacement else [],
                lambda: fusion_client.complete_structured(
                    skill,
                    fusion_payload,
                    fusion_schema,
                    [replacement] if replacement else [],
                ),
                transform=lambda value: normalize_fusion_prompt(
                    value,
                    input_sha256=fusion_input_sha256,
                    segment_index=index,
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
            return self._generate_picsart_segment(
                project_id,
                item,
                replacement,
                request,
                log,
                final_prompt=final_prompt,
                muted=muted,
            )
        except Exception:
            item["status"] = "failed"
            raise

    def _generate_picsart_segment(
        self,
        project_id: str,
        item: dict[str, Any],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
        *,
        final_prompt: str,
        muted: Path,
        node_prefix: str = "picsart",
        artifact_suffix: str = "",
        existing_task_id: str | None = None,
    ) -> Path:
        index = int(item["index"])
        scope = self._segment_scope(index)
        segment_dir = muted.parent
        try:
            task_id = existing_task_id
            if task_id is None:
                video_url = self.register_asset(project_id, muted)
                provider_replacement = None
                if replacement is not None:
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
                image_url = (
                    self.register_asset(project_id, provider_replacement)
                    if provider_replacement is not None
                    else None
                )
                output_request = request.get("output") or {}
                provider_duration = max(5, math.ceil(float(item["duration_s"])))
                submit_input = {
                    "workflow": PicsartClient.WORKFLOW,
                    "prompt": final_prompt,
                    "reference_video_urls": [video_url],
                    "reference_image_urls": [image_url] if image_url else [],
                    "resolution": str(
                        output_request.get("resolution", "768p")
                    ).upper(),
                    "duration": provider_duration,
                    "aspect_ratio": output_request.get(
                        "aspect_ratio", "adaptive"
                    ),
                    "prompt_expansion_mode": "quality",
                }
                task_id = self._node(
                    log,
                    f"{node_prefix}-submit",
                    scope,
                    submit_input,
                    [muted]
                    + ([provider_replacement] if provider_replacement else []),
                    lambda: self.picsart.submit(
                        prompt=final_prompt,
                        video_url=video_url,
                        image_url=image_url,
                        duration=provider_duration,
                        resolution=output_request.get("resolution", "768p"),
                        aspect_ratio=output_request.get(
                            "aspect_ratio", "adaptive"
                        ),
                    ),
                )
            else:
                log.event(
                    "picsart_task_reused",
                    segment_index=index,
                    task_id=task_id,
                )
            generated_url = self._node(
                log,
                f"{node_prefix}-poll",
                scope,
                {"task_id": task_id},
                [],
                lambda: self.picsart.poll(task_id),
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
                {"source": downloaded, "target_duration_s": item["duration_s"]},
                [downloaded],
                lambda: media.trim_24fps(
                    downloaded, float(item["duration_s"]), trimmed,
                ),
                lambda _value: [trimmed],
            )
            item["final_prompt"] = final_prompt
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
        return hashlib.sha256(path.read_bytes()).hexdigest()

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
