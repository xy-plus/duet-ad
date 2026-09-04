from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from typing import Any

from app import media
from app.config import Settings
from app.fusion_schema import FUSION_OUTPUT_NAME
from app.pipeline import Pipeline, SAME_AS_SOURCE_LANGUAGE
from app.providers import ProviderFailure, ProviderResult
from app.research_log import ResearchLog
from app.store import ProjectStore
from app.video_maker_schemas import (
    PROJECT_INDEX_SCHEMA,
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
)


def _settings(path: Path, **changes: Any) -> Settings:
    values = {
        "access_token": "test-token",
        "data_dir": path,
        "public_base_url": "https://pipeline.test",
        "retry_delay_s": 0,
        "enable_mediakit": False,
        "provider_poll_interval_s": 0,
    }
    values.update(changes)
    return Settings(**values)


class _UnusedMediaKit:
    def erase(self, *_args: Any, **_kwargs: Any) -> ProviderResult[Path]:
        raise AssertionError("MediaKit is disabled in this test")


class _FlowMediaKit:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[list[str]] = []

    def erase(
        self, source: Path, destination: Path, scenes: list[str],
    ) -> ProviderResult[Path]:
        destination.write_bytes(source.read_bytes() + b"-processed")
        with self.lock:
            self.calls.append(list(scenes))
        return ProviderResult(
            destination,
            {"stages": list(scenes), "raw": "mediakit-response"},
            [destination],
        )


class _FlowDeepSeek:
    def __init__(self) -> None:
        self.visual_barrier = threading.Barrier(2)
        self.branch_barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.structured_calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.common_ready = False

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        with self.lock:
            self.structured_calls.append({
                "instruction": instruction,
                "payload": payload,
                "schema": schema,
                "images": list(images or []),
            })
        if schema == VISUAL_PROMPT_SCHEMA:
            index = int(payload["segment"]["index"])
            with self.lock:
                self.events.append(f"visual-start-{index}")
            self.visual_barrier.wait(timeout=3)
            return ProviderResult(
                {"prompt": f"visual-{index}"},
                {"response": {"tool": "submit_result", "segment": index}},
            )
        if "input_sha256" in schema.get("properties", {}):
            index = int(payload["segment_index"])
            with self.lock:
                assert self.common_ready
                assert "translation" in self.events
                self.events.append(f"branch-start-{index}")
            self.branch_barrier.wait(timeout=3)
            return ProviderResult(
                {
                    "schema": FUSION_OUTPUT_NAME,
                    "version": 1,
                    "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                    "segment_index": index,
                    "prompt": f"r2v prompt {index}",
                },
                {"response": {"tool": "submit_result", "segment": index}},
            )
        assert schema == PROJECT_INDEX_SCHEMA
        with self.lock:
            self.events.append("project-index")
            self.common_ready = True
        raw = {
            "people": [{
                "key": "local-person",
                "source_visual_description": "a visible person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replaceable": ["face", "clothing"],
                "preserve": ["motion"],
            }],
            "entities": [],
            "scenes": [{
                "key": "local-scene",
                "source_visual_description": "a visible kitchen",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replaceable": ["layout"],
                "preserve": ["lighting"],
            }],
            "relations": [],
        }
        return ProviderResult(
            raw,
            {"response": {"tool": "submit_result", "phase": "project_index"}},
        )

    def complete(
        self,
        instruction: str,
        payload: Any,
        images: list[Path] | None = None,
        *,
        json_mode: bool = False,
    ) -> ProviderResult[str]:
        del instruction, images
        if json_mode:
            with self.lock:
                self.events.append("translation")
            return ProviderResult(
                json.dumps({"lines": [{"text": "translated line"}]}),
                {"response": {"raw": "translation-response"}},
            )
        raise AssertionError("plain DeepSeek completion is translation-only")


class _FlowPicsart:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.submits: list[dict[str, Any]] = []
        self.poll_counts: dict[str, int] = {}

    def submit(self, **kwargs: Any) -> ProviderResult[str]:
        with self.lock:
            task_id = f"task-{len(self.submits) + 1}"
            self.submits.append(dict(kwargs))
        return ProviderResult(
            task_id,
            {"outcome": "accepted", "task_id": task_id, "request": kwargs},
        )

    def poll(self, task_id: str) -> ProviderResult[str]:
        with self.lock:
            count = self.poll_counts.get(task_id, 0) + 1
            self.poll_counts[task_id] = count
        if task_id == "task-1" and count == 1:
            raise ProviderFailure(
                {"task_id": task_id, "polls": [{"status": "network-error"}]},
                [],
                OSError("temporary poll failure"),
            )
        return ProviderResult(
            f"https://result.invalid/{task_id}.mp4",
            {"task_id": task_id, "polls": [{"status": "COMPLETED"}]},
        )

    def download(self, url: str, destination: Path) -> ProviderResult[Path]:
        destination.write_bytes(b"provider-video")
        return ProviderResult(
            destination,
            {"request": {"url": url}, "response": {"status_code": 200}},
            [destination],
        )


def _install_fake_media(monkeypatch: Any, events: list[str]) -> None:
    def probe(_source: Path) -> dict[str, Any]:
        return {"duration_s": 9.2, "width": 1280, "height": 720, "has_audio": True}

    def frames(_source: Path, destination: Path) -> dict[str, Any]:
        destination.mkdir(parents=True, exist_ok=True)
        frame = destination / "frame-0001.png"
        frame.write_bytes(b"frame")
        return {
            "duration_seconds": 9.2,
            "frames": [{"index": 1, "time_seconds": 0.0, "file": frame.name}],
        }

    def extract_audio(_source: Path, destination: Path) -> dict[str, Any]:
        destination.write_bytes(b"audio")
        return {"stdout": "", "stderr": ""}

    def transcribe(
        _audio: Path, destination: Path, _cli: Path, _model: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        lines = [{"text": "source line", "start_s": 0.5, "end_s": 1.5}]
        destination.write_text(json.dumps(lines), encoding="utf-8")
        return lines, {"provider_raw": {"transcription": ["raw line"]}}

    def filter_spoken(
        lines: list[dict[str, Any]], _analysis: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return lines, [{"classification": "spoken", "kept": True}]

    def planned(
        _scenes: list[dict[str, Any]],
        _duration_s: float,
        dialogue: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        assert [line["text"] for line in dialogue] == ["source line"]
        return [
            {
                "index": 1,
                "start_s": 0.0,
                "end_s": 4.2,
                "duration_s": 4.2,
                "chain_id": "chain-1",
                "join_mode": "hard_cut",
                "scene_indices": [1],
                "source_cut_timeline": [],
            },
            {
                "index": 2,
                "start_s": 4.2,
                "end_s": 9.2,
                "duration_s": 5.0,
                "chain_id": "chain-2",
                "join_mode": "hard_cut",
                "scene_indices": [2],
                "source_cut_timeline": [],
            },
        ]

    def selection(_scenes: Any, segment: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"order": order, "decode_frame_index": order + int(segment["index"]) * 10}
            for order in range(1, 10)
        ]

    def cut(
        _source: Path,
        _start_s: float,
        _duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.write_bytes(b"segment")
        return {"stdout": "", "stderr": ""}

    def materialize(
        _source: Path,
        _selection: list[dict[str, Any]],
        destination: Path,
    ) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for order in range(1, 10):
            path = destination / f"{order:02d}.png"
            path.write_bytes(f"frame-{order}".encode())
            paths.append(path)
        return paths

    def half(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes() + b"-half")
        return destination

    def provider_reference(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes() + b"-provider-reference")
        return destination

    def localize(
        lines: list[dict[str, Any]],
        segment: dict[str, Any],
        *,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del segments
        localized = []
        for line in lines:
            start = max(float(line["start_s"]), float(segment["start_s"]))
            end = min(float(line["end_s"]), float(segment["end_s"]))
            if start < end:
                localized.append({
                    "text": line["text"],
                    "start_s": start - float(segment["start_s"]),
                    "end_s": end - float(segment["start_s"]),
                })
        return localized

    def mute(source: Path, destination: Path) -> dict[str, Any]:
        destination.write_bytes(source.read_bytes())
        return {"stdout": "", "stderr": ""}

    def trim(source: Path, duration_s: float, destination: Path) -> dict[str, Any]:
        events.append(f"trim-{duration_s}")
        destination.write_bytes(source.read_bytes())
        return {"fps": 24, "duration_s": duration_s}

    def concat(
        paths: list[Path],
        destination: Path,
        *,
        target_durations: list[float] | None = None,
        join_modes: list[str] | None = None,
    ) -> dict[str, Any]:
        assert target_durations == [4.2, 5.0]
        assert join_modes == ["hard_cut", "hard_cut"]
        destination.write_bytes(b"".join(path.read_bytes() for path in paths))
        return {"fps": 24}

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(media, "extract_frames_4fps", frames, raising=False)
    monkeypatch.setattr(media, "extract_audio", extract_audio)
    monkeypatch.setattr(
        media,
        "yamnet_analyze",
        lambda _audio, _model: {
            "windows": [{
                "start_ms": 0,
                "end_ms": 2000,
                "spoken": 0.9,
                "sung": 0.0,
                "music": 0.0,
            }],
            "has_bgm": False,
        },
    )
    monkeypatch.setattr(media, "transcribe", transcribe)
    monkeypatch.setattr(media, "filter_spoken_lines", filter_spoken, raising=False)
    monkeypatch.setattr(
        media,
        "detect_scenes",
        lambda _source, _duration: [{"index": 1, "frames": [{}]}],
    )
    monkeypatch.setattr(media, "plan_segments", planned)
    monkeypatch.setattr(media, "select_segment_keyframes", selection, raising=False)
    monkeypatch.setattr(media, "cut_segment", cut)
    monkeypatch.setattr(media, "materialize_keyframes", materialize, raising=False)
    monkeypatch.setattr(media, "half_resolution_png", half, raising=False)
    monkeypatch.setattr(
        media, "provider_reference_png", provider_reference, raising=False,
    )
    monkeypatch.setattr(media, "localize_dialogue", localize, raising=False)
    monkeypatch.setattr(media, "mute", mute)
    monkeypatch.setattr(media, "trim_24fps", trim, raising=False)
    monkeypatch.setattr(media, "concat", concat)


def test_full_pipeline_uses_clean_structured_index_barrier_and_split_poll_retry(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = {
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {"target_language": "日语"},
        "output": {"resolution": "768p", "aspect_ratio": "16:9"},
    }
    project_id, project, _state = store.create(request, "request-1")
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"source")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path=None,
    )
    media_events: list[str] = []
    _install_fake_media(monkeypatch, media_events)
    deepseek = _FlowDeepSeek()
    picsart = _FlowPicsart()
    mediakit = _FlowMediaKit()
    pipeline = Pipeline(
        _settings(tmp_path, enable_mediakit=True),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.parent.name}/{path.name}",
        deepseek=deepseek,
        mediakit=mediakit,
        picsart=picsart,
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert state["project_progress"] == {
        "status": "succeeded",
        "percent": 100,
        "stage": "completed",
    }
    assert len(picsart.submits) == 2
    assert len(mediakit.calls) == 18
    assert all(
        call == ["full_screen_text_erase", "full_screen_icon_erase"]
        for call in mediakit.calls
    )
    assert sorted(call["duration"] for call in picsart.submits) == [5, 5]
    assert picsart.poll_counts["task-1"] == 2
    assert picsart.poll_counts["task-2"] == 1
    assert media_events == ["trim-4.2", "trim-5.0"] or sorted(media_events) == [
        "trim-4.2",
        "trim-5.0",
    ]

    visual_calls = [
        call for call in deepseek.structured_calls
        if call["schema"] == VISUAL_PROMPT_SCHEMA
    ]
    index_calls = [
        call for call in deepseek.structured_calls
        if call["schema"] == PROJECT_INDEX_SCHEMA
    ]
    fusion_calls = [
        call for call in deepseek.structured_calls
        if "input_sha256" in call["schema"].get("properties", {})
    ]
    assert len(visual_calls) == 2
    assert len(index_calls) == 1
    assert len(fusion_calls) == 2
    assert sorted(call["payload"]["segment_index"] for call in fusion_calls) == [1, 2]
    assert all(call["schema"]["additionalProperties"] is False for call in fusion_calls)
    assert all(
        call["payload"]["generation_config"] == {
            "remove_subtitle": True,
            "remove_watermark": True,
        }
        for call in visual_calls
    )
    index_payload = index_calls[0]["payload"]
    assert "visual_prompt" not in json.dumps(index_payload)
    assert index_payload["generation_config"] == {
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    for segment in index_payload["segments"]:
        assert set(segment) == {"segment_index", "frames"}
        assert len(segment["frames"]) == 9
        assert all(
            set(frame) == {"frame_order", "path", "sha256"}
            for frame in segment["frames"]
        )
    assert set(state["element_index"]) == {
        "people", "entities", "scenes", "relations",
    }
    serialized_index = json.dumps(state["element_index"])
    for forbidden in ("provider_record", '"id"', "subject_id", "object_id"):
        assert forbidden not in serialized_index

    translation_output = json.loads(
        (project / "logs" / "deepseek-translation" / "project" / "attempt-1" / "output.json").read_text(encoding="utf-8")
    )
    assert translation_output["value"][0]["text"] == "translated line"
    assert translation_output["provider_record"]["response"]["raw"] == "translation-response"

    poll_root = project / "logs" / "picsart-poll"
    retry_scopes = [
        scope for scope in poll_root.iterdir()
        if (scope / "attempt-2").is_dir()
    ]
    assert len(retry_scopes) == 1
    assert (retry_scopes[0] / "attempt-1" / "output.json").is_file()
    assert (retry_scopes[0] / "attempt-1" / "error.json").is_file()
    assert stat.S_IMODE(
        (retry_scopes[0] / "attempt-1" / "output.json").stat().st_mode
    ) == 0o600
    assert not any(
        (scope / "attempt-2").exists()
        for scope in (project / "logs" / "picsart-submit").iterdir()
    )


def test_h3_prompt_fusion_selects_codex_and_keeps_schema_binding(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "codex-fusion")
    segment_dir = project / "work" / "segments" / "segment-001"
    segment_dir.mkdir(parents=True)
    source = segment_dir / "source.mp4"
    source.write_bytes(b"source")
    output = segment_dir / "generated.mp4"
    calls: list[dict[str, Any]] = []

    class FakeCodex:
        def complete_structured(
            self,
            instruction: str,
            payload: dict[str, Any],
            schema: dict[str, Any],
            images: list[Path],
        ) -> ProviderResult[dict[str, Any]]:
            calls.append({
                "instruction": instruction,
                "payload": payload,
                "schema": schema,
                "images": images,
            })
            return ProviderResult({
                "schema": FUSION_OUTPUT_NAME,
                "version": 1,
                "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                "segment_index": 1,
                "prompt": "replace all people and all scenes while preserving camera and light",
            }, {"provider": "codex", "account": "acct2"})

    class RejectDeepSeek:
        def complete_structured(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("DeepSeek must not run for a Codex-frozen project")

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.name}",
        codex=FakeCodex(),
        deepseek=RejectDeepSeek(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        media, "mute", lambda input_path, target: target.write_bytes(input_path.read_bytes()),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_picsart_segment",
        lambda *_args, **_kwargs: output,
    )

    result = pipeline._generate_segment(
        project_id,
        project,
        {
            "index": 1,
            "source": source,
            "visual_prompt": "a person in a kitchen",
            "dialogue": [{"start_s": 0.0, "end_s": 1.0, "text": "你好"}],
        },
        {"people": {"person-01": {}}, "scenes": {"scene-01": {}}},
        None,
        {
            "fusion": {"agent": "codex"},
            "replacement_guidance": None,
        },
        ResearchLog(project),
    )

    assert result == output
    assert len(calls) == 1
    assert calls[0]["payload"]["segment_index"] == 1
    assert calls[0]["schema"]["additionalProperties"] is False
    prompt = (segment_dir / "h3_prompt.txt").read_text(encoding="utf-8")
    assert prompt.startswith("replace all people and all scenes")
    assert "[0.000-1.000s] 你好" in prompt
def test_resume_r2v_reuses_prompt_and_muted_segments_without_upstream_calls(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = {"output": {"resolution": "480p", "aspect_ratio": "9:16"}}
    project_id, project, _state = store.create(request, "resume-only")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    segments = [
        {
            "index": 1,
            "start_s": 0.0,
            "end_s": 4.2,
            "duration_s": 4.2,
            "chain_id": "chain-1",
            "join_mode": "hard_cut",
            "dialogue": [],
            "visual_prompt": "visual-1",
            "status": "running",
        },
        {
            "index": 2,
            "start_s": 4.2,
            "end_s": 9.2,
            "duration_s": 5.0,
            "chain_id": "chain-2",
            "join_mode": "hard_cut",
            "dialogue": [],
            "visual_prompt": "visual-2",
            "status": "running",
        },
    ]
    for segment in segments:
        segment_dir = project / "work" / "segments" / f"{segment['index']:04d}"
        segment_dir.mkdir(parents=True)
        (segment_dir / "source.mp4").write_bytes(b"source")
        (segment_dir / "source-muted.mp4").write_bytes(b"muted")
        (segment_dir / "h3_prompt.txt").write_text(
            f"frozen prompt {segment['index']}", encoding="utf-8",
        )
    store.update(
        project_id,
        status="failed",
        replacement_image_path="inputs/replacement-image",
        segments=segments,
        project_progress={"status": "failed", "percent": 57, "stage": "pipeline_failed"},
        generation={"status": "failed", "stage": "pipeline_failed", "error": "failed"},
    )
    accepted = (
        project / "logs" / "picsart-submit"
        / "segment-0001" / "attempt-1"
    )
    accepted.mkdir(parents=True)
    (accepted / "output.json").write_text(json.dumps({
        "value": "existing-task",
        "provider_record": {
            "outcome": "accepted", "task_id": "existing-task",
        },
    }), encoding="utf-8")
    failed = (
        project / "logs" / "picsart-submit"
        / "segment-0002" / "attempt-1"
    )
    failed.mkdir(parents=True)
    (failed / "output.json").write_text(json.dumps({
        "value": "failed-task",
        "provider_record": {
            "outcome": "accepted", "task_id": "failed-task",
        },
    }), encoding="utf-8")
    terminal = (
        project / "logs" / "picsart-poll"
        / "segment-0002" / "attempt-1"
    )
    terminal.mkdir(parents=True)
    (terminal / "output.json").write_text(json.dumps({
        "provider_record": {
            "task_id": "failed-task",
            "polls": [{"response": {"status_code": 422, "body": {}}}],
        },
    }), encoding="utf-8")
    events: list[str] = []
    _install_fake_media(monkeypatch, events)
    picsart = _FlowPicsart()
    registered: list[Path] = []
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or f"https://assets.invalid/{path.name}"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
    )

    pipeline.validate_r2v_resume(project_id)
    pipeline.resume_r2v(project_id)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert state["has_video"] is True
    assert len(picsart.submits) == 1
    assert [call["prompt"] for call in picsart.submits] == ["frozen prompt 2"]
    assert picsart.poll_counts["existing-task"] == 1
    assert "failed-task" not in picsart.poll_counts
    assert {path.name for path in registered} == {
        "source-muted.mp4", "provider-reference-resume.png",
    }
    assert (project / "output" / "generated.mp4").is_file()
    assert (project / "logs" / "picsart-resume-submit").is_dir()
    assert not (
        project / "logs" / "picsart-resume-submit" / "segment-0001"
    ).exists()
    assert not (project / "logs" / "h3-max-r2v-prompt").exists()


def test_same_language_translation_is_an_explicit_skipped_attempt(
    tmp_path: Path,
) -> None:
    class NoCallDeepSeek:
        def complete(self, *_args: Any, **_kwargs: Any) -> ProviderResult[str]:
            raise AssertionError("same-language translation must not call DeepSeek")

    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "request-2")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=NoCallDeepSeek(),
        mediakit=_UnusedMediaKit(),
        picsart=object(),
    )
    log = ResearchLog(project)
    source = [{"text": "原文", "start_s": 0.0, "end_s": 1.0}]

    translated = pipeline._node(
        log,
        "deepseek-translation",
        "project",
        {"target_language": SAME_AS_SOURCE_LANGUAGE, "source_lines": source},
        [],
        lambda: pipeline._translation_request(
            source, SAME_AS_SOURCE_LANGUAGE, "visual context",
        ),
        transform=lambda value: pipeline._normalize_translation(value, source),
    )

    assert translated == source
    output = json.loads(
        (project / "logs" / "deepseek-translation" / "project" / "attempt-1" / "output.json").read_text(encoding="utf-8")
    )
    assert output["provider_record"] == {
        "outcome": "skipped",
        "reason": "same_language",
    }
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"event": "node_skipped"' in trace


def test_no_audio_skips_audio_analysis_without_creating_fake_media(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "request-no-audio")
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"silent-video")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    for name in (
        "extract_audio",
        "yamnet_analyze",
        "transcribe",
        "filter_spoken_lines",
    ):
        monkeypatch.setattr(
            media,
            name,
            lambda *_args, name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"{name} must not run without an audio track")
            ),
        )
    log = ResearchLog(project)

    yamnet, asr_lines, spoken_lines, decisions = (
        pipeline._analyze_audio_dialogue(
            project_id,
            project,
            source,
            has_audio=False,
            log=log,
        )
    )

    assert yamnet == {"windows": [], "has_bgm": False}
    assert asr_lines == []
    assert spoken_lines == []
    assert decisions == []
    assert not (project / "work" / "source.wav").exists()
    assert not (project / "work" / "voice_lines.json").exists()
    rows = [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    skipped = [row for row in rows if row["event"] == "node_skipped"]
    assert [row["node"] for row in skipped] == [
        "extract-audio",
        "yamnet",
        "whisper-asr",
        "spoken-dialogue-filter",
    ]
    for node in (row["node"] for row in skipped):
        output = json.loads(
            (
                project / "logs" / node / "project" / "attempt-1" / "output.json"
            ).read_text(encoding="utf-8")
        )
        assert output["provider_record"] == {
            "outcome": "skipped",
            "reason": "source_has_no_audio",
        }
        assert not (
            project / "logs" / node / "project" / "attempt-2"
        ).exists()


def test_project_index_normalizer_mechanically_maps_ids_without_metadata() -> None:
    normalized = normalize_project_index({
        "people": [{
            "key": "p",
            "source_visual_description": "person",
            "occurrences": [{"segment_index": 0, "frame_orders": [1]}],
            "replaceable": [],
            "preserve": [],
        }],
        "entities": [{
            "key": "e",
            "source_visual_description": "cup",
            "occurrences": [{"segment_index": 0, "frame_orders": [1]}],
            "replaceable": [],
            "preserve": [],
        }],
        "scenes": [{
            "key": "s",
            "source_visual_description": "kitchen",
            "occurrences": [{"segment_index": 0, "frame_orders": [1]}],
            "replaceable": [],
            "preserve": [],
        }],
        "relations": [{
            "key": "r",
            "subject_key": "p",
            "predicate": "holds",
            "object_key": "e",
            "occurrences": [{
                "segment_index": 0,
                "frames": [{"frame_order": 1, "state": "held", "geometry": "hand"}],
            }],
            "preserve": [],
            "replace_together": False,
        }],
    })

    assert list(normalized["people"]) == ["person-01"]
    assert list(normalized["entities"]) == ["entity-01"]
    assert list(normalized["scenes"]) == ["scene-01"]
    assert list(normalized["relations"]) == ["relation-01"]
    relation = normalized["relations"]["relation-01"]
    assert relation["subject_key"] == "person-01"
    assert relation["object_key"] == "entity-01"
    serialized = json.dumps(normalized)
    assert '"id"' not in serialized
    assert "provider_record" not in serialized
    assert "subject_id" not in serialized


def test_node_records_both_failed_attempts_and_process_streams(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create({}, "request-3")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise media.CommandError(["fake"], 1, f"stdout-{calls}", f"stderr-{calls}")

    try:
        pipeline._node(log, "command", "project", {}, [], fail)
    except media.CommandError:
        pass
    else:
        raise AssertionError("the second command failure must escape")

    assert calls == 2
    for attempt in (1, 2):
        root = project / "logs" / "command" / "project" / f"attempt-{attempt}"
        assert (root / "error.json").is_file()
        assert (root / "stdout.log").read_text(encoding="utf-8") == f"stdout-{attempt}"
        assert (root / "stderr.log").read_text(encoding="utf-8") == f"stderr-{attempt}"
        assert stat.S_IMODE((root / "stdout.log").stat().st_mode) == 0o600
        assert stat.S_IMODE((root / "stderr.log").stat().st_mode) == 0o600


def test_pipeline_keeps_raw_failure_in_logs_but_not_public_state(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "request-public-error")
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"source")
    store.update(project_id, source_path="inputs/source.mp4", has_source=True)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _source: (_ for _ in ()).throw(
            RuntimeError("private provider response and /private/path")
        ),
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    assert state["status"] == "failed"
    assert state["generation"]["error"] == "pipeline_failed"
    assert state["error"] == {
        "code": "pipeline_failed",
        "message": "流水线执行失败",
    }
    assert "private provider response" not in json.dumps(state)
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "private provider response" in trace


def test_submit_node_retries_once_without_a_task_id(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create({}, "request-4")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    unknown_calls = 0

    def unknown_then_accepted() -> ProviderResult[str]:
        nonlocal unknown_calls
        unknown_calls += 1
        if unknown_calls == 1:
            raise ProviderFailure(
                {"outcome": "unknown", "request": {"method": "POST"}},
                [],
                OSError("connection ended before response"),
            )
        return ProviderResult("task-kept", {"outcome": "accepted"})

    assert pipeline._node(
        log,
        "picsart-submit",
        "unknown",
        {},
        [],
        unknown_then_accepted,
    ) == "task-kept"
    assert unknown_calls == 2

    rejected_calls = 0

    def rejected() -> None:
        nonlocal rejected_calls
        rejected_calls += 1
        raise ProviderFailure(
            {"outcome": "rejected", "response": {"status_code": 400}},
            [],
            RuntimeError("bad request"),
        )

    try:
        pipeline._node(
            log,
            "picsart-submit",
            "rejected",
            {},
            [],
            rejected,
        )
    except ProviderFailure:
        pass
    else:
        raise AssertionError("a rejected submit must escape")
    assert rejected_calls == 2
    assert (
        project / "logs" / "picsart-submit" / "rejected" / "attempt-2"
    ).is_dir()
