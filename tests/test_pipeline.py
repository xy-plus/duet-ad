from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from app import media
from app.config import Settings
from app.fusion_schema import AUTHORITY_BEGIN, AUTHORITY_END, FUSION_OUTPUT_NAME
from app.object_replace_schema import OBJECT_REPLACE_OUTPUT_NAME
from app.pipeline import (
    Pipeline,
    PreparedPicsartRequest,
    SAME_AS_SOURCE_LANGUAGE,
    TRANSLATION_SCHEMA,
)
from app.providers import ProviderFailure, ProviderResult
from app.research_log import ResearchLog
from app.store import ProjectStore
from app.voice_catalog import load_voice_library
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
                {"response": {"api": "responses", "segment": index}},
            )
        output_name = (
            ((schema.get("properties") or {}).get("schema") or {}).get("enum")
            or [None]
        )[0]
        if output_name == OBJECT_REPLACE_OUTPUT_NAME:
            with self.lock:
                assert self.common_ready
                assert "translation" in self.events
                self.events.append("object-replace")
            category_schemas = {
                category: schema["properties"][category]["properties"]
                for category in ("people", "entities", "scenes")
            }
            people_schema = category_schemas["people"]
            voice_ids = next(iter(people_schema.values()))["properties"][
                "voice_id"
            ]["anyOf"][0]["enum"]
            people = {}
            for offset, stable_id in enumerate(people_schema):
                matches_user_request = offset == 0
                people[stable_id] = {
                    "matches_user_request": matches_user_request,
                    "replacement": {
                        "source": (
                            "image_1"
                            if matches_user_request
                            else "generated_variant"
                        ),
                        "target_visual_description": (
                            f"replacement appearance for {stable_id}"
                        ),
                    },
                    "voice_id": voice_ids[offset],
                }
            return ProviderResult(
                {
                    "schema": OBJECT_REPLACE_OUTPUT_NAME,
                    "version": 1,
                    "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                    "people": people,
                    "entities": {},
                    "scenes": {
                        stable_id: {
                            "matches_user_request": False,
                            "replacement": {
                                "source": "generated_variant",
                                "target_visual_description": (
                                    f"replacement layout for {stable_id}"
                                ),
                            },
                        }
                        for stable_id in category_schemas["scenes"]
                    },
                },
                {"response": {"api": "responses", "phase": "object-replace"}},
            )
        if output_name == FUSION_OUTPUT_NAME:
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
                    "prompt": (
                        f"r2v cinematic motion prompt {index}; preserve timing, "
                        "camera, lighting, contact, and continuity"
                    ),
                },
                {"response": {"api": "responses", "segment": index}},
            )
        if schema == TRANSLATION_SCHEMA:
            with self.lock:
                self.events.append("translation")
            return ProviderResult(
                {"lines": [{"text": "translated line"}]},
                {"response": {"api": "responses", "phase": "translation"}},
            )
        assert schema == PROJECT_INDEX_SCHEMA
        with self.lock:
            self.events.append("project-index")
            self.common_ready = True
        raw = {
            "people": [
                {
                    "key": "local-person-a",
                    "source_visual_description": "the main visible person",
                    "occurrences": [
                        {"segment_index": 1, "frame_orders": [1]},
                        {"segment_index": 2, "frame_orders": [1]},
                    ],
                    "replaceable": ["face", "clothing"],
                    "preserve": ["motion"],
                },
                {
                    "key": "local-person-b",
                    "source_visual_description": "a second visible person",
                    "occurrences": [
                        {"segment_index": 1, "frame_orders": [1]},
                    ],
                    "replaceable": ["face", "clothing"],
                    "preserve": ["motion"],
                },
                {
                    "key": "local-person-c",
                    "source_visual_description": "a third visible person",
                    "occurrences": [
                        {"segment_index": 2, "frame_orders": [1]},
                    ],
                    "replaceable": ["face", "clothing"],
                    "preserve": ["motion"],
                },
            ],
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
            {"response": {"api": "responses", "phase": "project_index"}},
        )


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
        destination.write_bytes(
            b"\x89PNG\r\n\x1a\n" + source.read_bytes()
        )
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
        "replacement_guidance": {
            "instruction": "把主角替换成用户参考形象",
            "image_field": "replacement_image",
        },
        "output": {"resolution": "768p", "aspect_ratio": "16:9"},
    }
    project_id, project, _state = store.create(request, "request-1")
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"source")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
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
        if (
            ((call["schema"].get("properties") or {}).get("schema") or {}).get("enum")
            == [FUSION_OUTPUT_NAME]
        )
    ]
    object_replace_calls = [
        call for call in deepseek.structured_calls
        if (
            ((call["schema"].get("properties") or {}).get("schema") or {}).get("enum")
            == [OBJECT_REPLACE_OUTPUT_NAME]
        )
    ]
    translation_calls = [
        call for call in deepseek.structured_calls
        if call["schema"] == TRANSLATION_SCHEMA
    ]
    assert len(visual_calls) == 2
    assert len(index_calls) == 1
    assert len(fusion_calls) == 2
    assert len(object_replace_calls) == 1
    assert len(translation_calls) == 1
    assert all(
        "submit_result" not in call["instruction"]
        for call in deepseek.structured_calls
    )
    assert all("JSON Schema" in call["instruction"] for call in visual_calls)
    assert "JSON Schema" in translation_calls[0]["instruction"]
    assert translation_calls[0]["schema"]["additionalProperties"] is False
    assert translation_calls[0]["schema"]["properties"]["lines"]["items"]["additionalProperties"] is False
    assert sorted(call["payload"]["segment_index"] for call in fusion_calls) == [1, 2]
    object_replace_call = object_replace_calls[0]
    compact_catalog = object_replace_call["payload"]["voice_catalog_lines"]
    assert len(compact_catalog.splitlines()) == 30
    assert len(compact_catalog) < 3500
    assert "{" not in compact_catalog
    assert all(" - " in line for line in compact_catalog.splitlines())
    assert object_replace_call["payload"]["user_replacement_instruction"] == (
        "把主角替换成用户参考形象"
    )
    assert object_replace_call["payload"]["has_user_image"] is True
    assert object_replace_call["payload"]["user_replacement_image"] == {
        "name": "Image 1",
        "sha256": Pipeline._sha256(replacement),
    }
    assert object_replace_call["images"] == [replacement]
    assert deepseek.events.index("project-index") < deepseek.events.index(
        "object-replace"
    )
    assert deepseek.events.index("object-replace") < min(
        deepseek.events.index("branch-start-1"),
        deepseek.events.index("branch-start-2"),
    )
    assert all(call["schema"]["additionalProperties"] is False for call in fusion_calls)
    assert all(call["images"] == [] for call in fusion_calls)
    assert all(
        set(call["payload"]) == {
            "segment_index",
            "visual_prompt",
            "global_element_index",
            "translated_dialogue",
            "authoritative_constraints",
        }
        for call in fusion_calls
    )
    assert all(
        call["payload"]["authoritative_constraints"].startswith(
            "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN\n"
        )
        for call in fusion_calls
    )
    assert all(
        "user_replacement_instruction" not in json.dumps(
            call["payload"], ensure_ascii=False,
        )
        and "user_replacement_image" not in json.dumps(
            call["payload"], ensure_ascii=False,
        )
        and "selected_voice_reference" not in json.dumps(
            call["payload"], ensure_ascii=False,
        )
        for call in fusion_calls
    )
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
    raw_index = state["element_index"]
    enhanced_index = state["enhanced_element_index"]
    assert set(raw_index) == {
        "people", "entities", "scenes", "relations",
    }
    assert list(raw_index["people"]) == [
        "person-01", "person-02", "person-03",
    ]
    assert json.loads(
        (project / "work" / "element_index.json").read_text(encoding="utf-8")
    ) == raw_index
    assert json.loads(
        (project / "work" / "enhanced_element_index.json").read_text(
            encoding="utf-8",
        )
    ) == enhanced_index
    assert enhanced_index != raw_index
    assert all(
        "replacement" not in record and "voice" not in record
        for category in ("people", "entities", "scenes")
        for record in raw_index[category].values()
    )
    assert enhanced_index["people"]["person-01"]["replacement"]["source"] == (
        "image_1"
    )
    assert state["object_replace"]["version"] == 1
    assert [reference["audio_reference"] for reference in state["voice_references"]] == [
        "Audio 1", "Audio 2", "Audio 3",
    ]
    assert all(
        (project / reference["asset_path"]).is_file()
        for reference in state["voice_references"]
    )
    expected_audio_urls = [
        f"https://assets.invalid/voice-references/audio-{slot:02d}.wav"
        for slot in range(1, 4)
    ]
    assert all(
        call["audio_urls"] == expected_audio_urls for call in picsart.submits
    )
    serialized_index = json.dumps(raw_index)
    for forbidden in ("provider_record", '"id"', "subject_id", "object_id"):
        assert forbidden not in serialized_index

    for call in fusion_calls:
        segment_index = call["payload"]["segment_index"]
        prompt = (
            project / "work" / "segments" / f"{segment_index:04d}"
            / "h3_prompt.txt"
        ).read_text(encoding="utf-8")
        segment_audio: list[str] = []
        requires_image_1 = False
        for category in ("people", "entities", "scenes"):
            for stable_id, record in enhanced_index[category].items():
                if not any(
                    occurrence["segment_index"] == segment_index
                    for occurrence in record["occurrences"]
                ):
                    continue
                replacement_plan = record["replacement"]
                if replacement_plan is not None:
                    assert stable_id in prompt
                    requires_image_1 = (
                        requires_image_1
                        or replacement_plan["source"] == "image_1"
                    )
                if category == "people":
                    audio_reference = record["voice"]["audio_reference"]
                    if audio_reference not in segment_audio:
                        segment_audio.append(audio_reference)
        assert requires_image_1 is True
        assert "Image 1" in prompt
        assert segment_audio
        assert all(reference in prompt for reference in segment_audio)
        assert prompt.count(AUTHORITY_BEGIN) == 1
        assert prompt.count(AUTHORITY_END) == 1
        for stable_id, person in enhanced_index["people"].items():
            if not any(
                occurrence["segment_index"] == segment_index
                for occurrence in person["occurrences"]
            ):
                continue
            authority_line = next(
                line for line in prompt.splitlines()
                if line.startswith(f"- {stable_id} | REPLACE |")
            )
            assert (
                f"| voice: {person['voice']['audio_reference']} |"
                in authority_line
            )

    translation_output = json.loads(
        (project / "logs" / "deepseek-translation" / "project" / "attempt-1" / "output.json").read_text(encoding="utf-8")
    )
    assert translation_output["value"][0]["text"] == "translated line"
    assert translation_output["provider_record"]["response"] == {
        "api": "responses", "phase": "translation",
    }

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
    calls: list[dict[str, Any]] = []
    library = load_voice_library()
    voice = library.voices[0]
    voice_directory = project / "work" / "voice-references"
    voice_directory.mkdir(parents=True)
    voice_reference = voice_directory / "audio-01.wav"
    voice_reference.write_bytes(library.files[voice["voice_id"]].read_bytes())
    enhanced_index = {
        "people": {
            "person-01": {
                "source_visual_description": "a visible person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replaceable": ["face", "clothing"],
                "preserve": ["motion"],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different visible person",
                },
                "voice": {
                    "audio_reference": "Audio 1",
                    "voice_id": voice["voice_id"],
                    "asset_path": "work/voice-references/audio-01.wav",
                    "asset_sha256": voice["sha256"],
                    "features": library.model_line(voice["voice_id"]),
                },
            },
        },
        "entities": {},
        "scenes": {
            "scene-01": {
                "source_visual_description": "a visible kitchen",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replaceable": ["layout"],
                "preserve": ["lighting"],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a replacement kitchen",
                },
            },
        },
        "relations": {},
    }

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
                "prompt": "Preserve the source motion, camera, light, and timing",
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
    result_item, result_prompt, result_muted, fusion_input_sha256 = (
        pipeline._prepare_h3_segment(
        project,
        {
            "index": 1,
            "source": source,
            "visual_prompt": "a person in a kitchen",
            "dialogue": [{"start_s": 0.0, "end_s": 1.0, "text": "你好"}],
        },
        enhanced_index,
        {
            "fusion": {"agent": "codex"},
            "replacement_guidance": None,
        },
        ResearchLog(project),
        )
    )

    assert result_item["index"] == 1
    assert result_prompt.startswith("Preserve the source motion")
    assert result_muted == segment_dir / "source-muted.mp4"
    assert len(fusion_input_sha256) == 64
    assert len(calls) == 1
    assert calls[0]["payload"]["segment_index"] == 1
    assert calls[0]["payload"]["global_element_index"] == enhanced_index
    assert calls[0]["schema"]["additionalProperties"] is False
    assert calls[0]["images"] == []
    prompt = (segment_dir / "h3_prompt.txt").read_text(encoding="utf-8")
    assert prompt.startswith("Preserve the source motion")
    assert "Audio 1" in prompt
    assert "[0.000-1.000s] 你好" in prompt


def test_h3_barrier_never_submits_if_any_fusion_preparation_fails(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "h3-barrier")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    submits: list[int] = []

    def prepare(
        _project: Path,
        item: dict[str, Any],
        *_args: Any,
    ) -> tuple[dict[str, Any], str, Path]:
        if item["index"] == 2:
            raise ValueError("fusion contract rejected segment 2")
        return item, "prompt", project / "work" / "muted.mp4"

    def submit(_project_id: str, item: dict[str, Any], *_args: Any, **_kwargs: Any) -> Path:
        submits.append(item["index"])
        return project / "output" / "unexpected.mp4"

    monkeypatch.setattr(pipeline, "_prepare_h3_segment", prepare)
    monkeypatch.setattr(pipeline, "_generate_picsart_segment", submit)

    with pytest.raises(ValueError, match="segment 2"):
        pipeline._run_h3_segments(
            project_id,
            project,
            [{"index": 1}, {"index": 2}],
            {},
            [],
            None,
            {},
            ResearchLog(project),
        )
    assert submits == []


@pytest.mark.parametrize("voice_mode", ["none", "legacy", "multi"])
def test_resume_r2v_reuses_prompt_and_muted_segments_without_upstream_calls(
    tmp_path: Path, monkeypatch: Any, voice_mode: str,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = {"output": {"resolution": "480p", "aspect_ratio": "9:16"}}
    project_id, project, _state = store.create(request, "resume-only")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
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
        mute_log = (
            project / "logs" / "mute-reference-video"
            / f"segment-{segment['index']:04d}" / "attempt-1"
        )
        mute_log.mkdir(parents=True)
        (mute_log / "output.json").write_text("{}", encoding="utf-8")
        fusion_log = (
            project / "logs" / "h3-max-r2v-prompt"
            / f"segment-{segment['index']:04d}" / "attempt-1"
        )
        fusion_log.mkdir(parents=True)
        (fusion_log / "output.json").write_text(
            json.dumps(f"frozen prompt {segment['index']}"),
            encoding="utf-8",
        )
    voice_state: dict[str, Any] = {}
    if voice_mode == "legacy":
        library = load_voice_library()
        voice = library.voices[0]
        voice_reference = project / "work" / "voice-reference.wav"
        voice_reference.write_bytes(library.files[voice["voice_id"]].read_bytes())
        voice_state = {
            "voice_selection": {
                "status": "selected",
                "voice_id": voice["voice_id"],
                "catalog_sha256": library.digest,
                "selection_input_sha256": "0" * 64,
                "asset_path": "work/voice-reference.wav",
                "asset_sha256": voice["sha256"],
                "features": dict(voice["features"]),
                "source": dict(voice["source"]),
            }
        }
    elif voice_mode == "multi":
        library = load_voice_library()
        voice_directory = project / "work" / "voice-references"
        voice_directory.mkdir(parents=True)
        references = []
        for slot, voice in enumerate(library.voices[:3], 1):
            asset_path = f"work/voice-references/audio-{slot:02d}.wav"
            destination = project / asset_path
            destination.write_bytes(
                library.files[voice["voice_id"]].read_bytes(),
            )
            references.append({
                "audio_reference": f"Audio {slot}",
                "voice_id": voice["voice_id"],
                "asset_path": asset_path,
                "asset_sha256": voice["sha256"],
                "features": library.model_line(voice["voice_id"]),
            })
            voice_state = {
                "voice_references": references,
                "object_replace": {"input_sha256": "0" * 64},
                "enhanced_element_index": {
                "people": {
                    f"person-{slot:02d}": {"voice": dict(reference)}
                    for slot, reference in enumerate(references, 1)
                },
                "entities": {},
                "scenes": {},
                "relations": {},
            },
        }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
        segment_count=2,
        segments=segments,
        project_progress={"status": "failed", "percent": 57, "stage": "pipeline_failed"},
        generation={"status": "failed", "stage": "pipeline_failed", "error": "failed"},
        **voice_state,
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
            "polls": [{"response": {
                "status_code": 200,
                "body": {"response": {"status": "FAILED"}},
            }}],
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
    if voice_mode == "multi":
        pipeline._freeze_h3_prepared_artifacts(
            project_id,
            project,
            [
                (
                    dict(
                        segment,
                        source=(
                            project / "work" / "segments"
                            / f"{segment['index']:04d}" / "source.mp4"
                        ),
                    ),
                    f"frozen prompt {segment['index']}",
                    (
                        project / "work" / "segments"
                        / f"{segment['index']:04d}" / "source-muted.mp4"
                    ),
                    f"{segment['index']}" * 64,
                )
                for segment in segments
            ],
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
    expected_registered = {
        "source-muted.mp4", "provider-reference-resume.png",
    }
    expected_audio_urls: list[str] = []
    if voice_mode == "legacy":
        expected_registered.add("voice-reference.wav")
        expected_audio_urls = ["https://assets.invalid/voice-reference.wav"]
    elif voice_mode == "multi":
        expected_registered.update({
            "audio-01.wav", "audio-02.wav", "audio-03.wav",
        })
        expected_audio_urls = [
            f"https://assets.invalid/audio-{slot:02d}.wav"
            for slot in range(1, 4)
        ]
    assert {path.name for path in registered} == expected_registered
    assert picsart.submits[0]["audio_urls"] == expected_audio_urls
    assert (project / "output" / "generated.mp4").is_file()
    assert (project / "logs" / "picsart-resume-submit").is_dir()
    assert not (
        project / "logs" / "picsart-resume-submit" / "segment-0001"
    ).exists()
    assert not any(
        (scope / "attempt-2").exists()
        for scope in (project / "logs" / "h3-max-r2v-prompt").iterdir()
    )


def test_resume_r2v_blocks_an_unknown_submit_receipt(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "unknown-submit")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "chain_id": "chain-1",
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "frozen",
        "status": "failed",
    }
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    (segment_dir / "source.mp4").write_bytes(b"source")
    (segment_dir / "source-muted.mp4").write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("frozen", encoding="utf-8")
    mute_log = (
        project / "logs" / "mute-reference-video"
        / "segment-0001" / "attempt-1"
    )
    mute_log.mkdir(parents=True)
    (mute_log / "output.json").write_text("{}", encoding="utf-8")
    fusion_log = (
        project / "logs" / "h3-max-r2v-prompt"
        / "segment-0001" / "attempt-1"
    )
    fusion_log.mkdir(parents=True)
    (fusion_log / "output.json").write_text(
        json.dumps("frozen"), encoding="utf-8",
    )
    receipt = (
        project / "logs" / "picsart-submit" / "segment-0001" / "attempt-1"
    )
    receipt.mkdir(parents=True)
    (receipt / "output.json").write_text(
        json.dumps({
            "provider_record": {
                "outcome": "unknown",
                "request": {"method": "POST"},
            },
        }),
        encoding="utf-8",
    )
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    picsart = _FlowPicsart()
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
    )

    with pytest.raises(ValueError, match="outcome is unknown"):
        pipeline.validate_r2v_resume(project_id)
    assert picsart.submits == []


@pytest.mark.parametrize("result", [None, "{broken", json.dumps({})])
def test_legacy_submit_attempt_without_a_proven_result_is_unknown(
    tmp_path: Path,
    result: str | None,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create({}, "uncertain-legacy-submit")
    attempt = (
        project / "logs" / "picsart-submit"
        / "segment-0001" / "attempt-1"
    )
    attempt.mkdir(parents=True)
    (attempt / "input.json").write_text("{}", encoding="utf-8")
    if result is not None:
        (attempt / "output.json").write_text(result, encoding="utf-8")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )

    assert pipeline._has_unknown_submit_receipt(
        project, "segment-0001",
    ) is True


def test_legacy_submit_attempt_with_explicit_rejection_is_retryable(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create({}, "rejected-legacy-submit")
    attempt = (
        project / "logs" / "picsart-submit"
        / "segment-0001" / "attempt-1"
    )
    attempt.mkdir(parents=True)
    (attempt / "output.json").write_text(json.dumps({
        "provider_record": {
            "outcome": "rejected",
            "task_id": None,
            "response": {"status_code": 400},
        },
    }), encoding="utf-8")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )

    assert pipeline._has_unknown_submit_receipt(
        project, "segment-0001",
    ) is False


def test_legacy_server_error_labeled_rejected_remains_unknown(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create({}, "legacy-500")
    attempt = (
        project / "logs" / "picsart-submit"
        / "segment-0001" / "attempt-1"
    )
    attempt.mkdir(parents=True)
    (attempt / "output.json").write_text(json.dumps({
        "provider_record": {
            "outcome": "rejected",
            "task_id": None,
            "response": {"status_code": 500},
        },
    }), encoding="utf-8")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )

    assert pipeline._has_unknown_submit_receipt(
        project, "segment-0001",
    ) is True


def test_mixed_ledger_keeps_legacy_accepted_task_for_unlisted_segment(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = {"output": {"resolution": "768p", "aspect_ratio": "16:9"}}
    project_id, project, _state = store.create(request, "mixed-receipts")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segments = []
    for index in (1, 2):
        segment = {
            "index": index,
            "start_s": float(index - 1) * 5,
            "end_s": float(index) * 5,
            "duration_s": 5.0,
            "dialogue": [],
            "visual_prompt": f"visual-{index}",
            "status": "failed",
        }
        segments.append(segment)
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        segment_dir.mkdir(parents=True)
        (segment_dir / "source.mp4").write_bytes(b"source")
        (segment_dir / "source-muted.mp4").write_bytes(b"muted")
        (segment_dir / "h3_prompt.txt").write_text(
            f"prompt-{index}", encoding="utf-8",
        )
        for node, value in (
            ("mute-reference-video", {}),
            ("h3-max-r2v-prompt", f"prompt-{index}"),
        ):
            evidence = (
                project / "logs" / node
                / f"segment-{index:04d}" / "attempt-1"
            )
            evidence.mkdir(parents=True)
            (evidence / "output.json").write_text(
                json.dumps(value), encoding="utf-8",
            )
    accepted = (
        project / "logs" / "picsart-submit"
        / "segment-0001" / "attempt-1"
    )
    accepted.mkdir(parents=True)
    (accepted / "output.json").write_text(json.dumps({
        "value": "legacy-task-1",
        "provider_record": {
            "outcome": "accepted", "task_id": "legacy-task-1",
        },
    }), encoding="utf-8")
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=2,
        segments=segments,
        picsart_submission_receipts={
            "version": 1,
            "segments": {
                "2": [{
                    "segment_index": 2,
                    "run_identity": "picsart-resume",
                    "request_sha256": "c" * 64,
                    "status": "provider_failed",
                    "task_id": "new-task-2",
                }],
            },
        },
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )

    _project, prepared, _replacement, _request, _voices = (
        pipeline._load_r2v_resume(project_id)
    )

    assert prepared[0]["resume_task_id"] == "legacy-task-1"
    assert prepared[1]["resume_task_id"] is None


def test_mixed_ledger_legacy_task_failure_needs_no_state_transition(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create({}, "mixed-task-failure")
    ledger = {
        "version": 1,
        "segments": {
            "2": [{
                "segment_index": 2,
                "run_identity": "picsart-resume",
                "request_sha256": "c" * 64,
                "status": "provider_failed",
                "task_id": "new-task-2",
            }],
        },
    }
    store.update(project_id, picsart_submission_receipts=ledger)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )

    pipeline._mark_submission_task_failed(project_id, 1, "legacy-task-1")

    assert store.read(project_id)["picsart_submission_receipts"] == ledger


def test_same_language_translation_is_an_explicit_skipped_attempt(
    tmp_path: Path,
) -> None:
    class NoCallDeepSeek:
        def complete_structured(self, *_args: Any, **_kwargs: Any) -> ProviderResult[Any]:
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


@pytest.mark.parametrize(
    "value",
    [
        {"lines": []},
        {"lines": [{"text": "   "}]},
        {"lines": "not-a-list"},
        {},
    ],
)
def test_translation_quality_is_not_a_local_blocker(value: Any) -> None:
    source = [{"text": "原文", "start_s": 0.0, "end_s": 1.0}]

    assert Pipeline._normalize_translation(value, source) == source


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


def test_nullable_object_replace_plan_is_compiled_without_a_quality_gate(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create({}, "nullable-plan")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    patch = {
        "people": {
            "person-01": {
                "matches_user_request": False,
                "replacement": None,
                "voice_id": None,
            },
        },
        "entities": {},
        "scenes": {
            "scene-01": {
                "matches_user_request": False,
                "replacement": None,
            },
        },
    }

    metadata, references, paths = pipeline._materialize_voice_references(
        store.path(project_id), patch,
    )
    authority = pipeline._fusion_authority_block({
        "people": {
            "person-01": {
                "occurrences": [{"segment_index": 1}],
                "replacement": None,
                "voice": None,
            },
        },
        "entities": {},
        "scenes": {
            "scene-01": {
                "occurrences": [{"segment_index": 1}],
                "replacement": None,
            },
        },
        "relations": {},
    }, 1)

    assert metadata == {}
    assert references == []
    assert paths == []
    assert "person-01 | KEEP" in authority
    assert "scene-01 | KEEP" in authority
    assert "Audio " not in authority


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


def test_submit_node_never_retries_without_an_explicit_safe_policy(
    tmp_path: Path,
) -> None:
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

    def unknown() -> ProviderResult[str]:
        nonlocal unknown_calls
        unknown_calls += 1
        raise ProviderFailure(
            {"outcome": "unknown", "request": {"method": "POST"}},
            [],
            OSError("connection ended before response"),
        )

    with pytest.raises(ProviderFailure):
        pipeline._node(
            log,
            "picsart-submit",
            "unknown",
            {},
            [],
            unknown,
        )
    assert unknown_calls == 1
    assert not (
        project / "logs" / "picsart-submit" / "unknown" / "attempt-2"
    ).exists()

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
    assert rejected_calls == 1
    assert not (
        project / "logs" / "picsart-submit" / "rejected" / "attempt-2"
    ).exists()


def test_pre_picsart_resume_rejects_state_receipt_without_log_evidence(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create({}, "state-only-receipt")
    store.update(
        project_id,
        status="failed",
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        picsart_submission_receipts={"version": 1, "segments": {}},
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )

    with pytest.raises(ValueError, match="submission evidence"):
        pipeline.validate_pre_picsart_resume(project_id)


@pytest.mark.parametrize("failure", ["reference_png", "asset_register"])
def test_global_picsart_preflight_failure_keeps_every_segment_zero_submit(
    tmp_path: Path,
    monkeypatch: Any,
    failure: str,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = {"output": {"resolution": "768p", "aspect_ratio": "16:9"}}
    project_id, project, _state = store.create(request, f"preflight-{failure}")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    store.update(
        project_id,
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={"input_sha256": "a" * 64},
    )
    items: list[dict[str, Any]] = []
    for index in (1, 2):
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        segment_dir.mkdir(parents=True)
        source = segment_dir / "source.mp4"
        source.write_bytes(f"source-{index}".encode())
        items.append({
            "index": index,
            "start_s": float(index - 1) * 5,
            "end_s": float(index) * 5,
            "duration_s": 5.0,
            "source": source,
            "status": "running",
        })

    def prepare(
        _project: Path,
        item: dict[str, Any],
        *_args: Any,
    ) -> tuple[dict[str, Any], str, Path, str]:
        segment_dir = item["source"].parent
        prompt = f"frozen prompt {item['index']}"
        (segment_dir / "h3_prompt.txt").write_text(prompt, encoding="utf-8")
        muted = segment_dir / "source-muted.mp4"
        muted.write_bytes(f"muted-{item['index']}".encode())
        return item, prompt, muted, f"{item['index']}" * 64

    def provider_reference(_source: Path, destination: Path) -> Path:
        if failure == "reference_png" and destination.parent.name == "0002":
            raise ValueError("segment 2 image conversion failed")
        destination.write_bytes(b"\x89PNG\r\n\x1a\nreference")
        return destination

    registered: list[Path] = []

    def register(_project_id: str, path: Path) -> str:
        registered.append(path)
        if (
            failure == "asset_register"
            and path.parent.name == "0002"
            and path.name == "source-muted.mp4"
        ):
            raise ValueError("segment 2 asset registration failed")
        return f"https://assets.invalid/{path.parent.name}/{path.name}"

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    monkeypatch.setattr(media, "provider_reference_png", provider_reference)
    picsart = _FlowPicsart()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        register,
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
    )
    monkeypatch.setattr(pipeline, "_prepare_h3_segment", prepare)

    with pytest.raises(ValueError, match="segment 2"):
        pipeline._run_h3_segments(
            project_id,
            project,
            items,
            {},
            [],
            replacement if failure == "reference_png" else None,
            request,
            ResearchLog(project),
        )

    assert picsart.submits == []
    assert not (project / "logs" / "picsart-submit").exists()
    assert "h3_prepared_artifacts" in store.read(project_id)
    assert registered or failure == "reference_png"


def _prepared_submit(project_id: str, project: Path) -> PreparedPicsartRequest:
    muted = project / "work" / "segments" / "0001" / "source-muted.mp4"
    muted.parent.mkdir(parents=True, exist_ok=True)
    muted.write_bytes(b"muted")
    payload = {
        "workflow": "minimax/h3-max/reference-to-video",
        "prompt": "frozen prompt",
        "reference_video_urls": ["https://assets.invalid/video.mp4"],
        "reference_image_urls": [],
        "reference_audio_urls": [],
        "resolution": "768P",
        "duration": 5,
        "aspect_ratio": "16:9",
        "prompt_expansion_mode": "quality",
    }
    return PreparedPicsartRequest(
        project_id=project_id,
        segment_index=1,
        scope="segment-0001",
        run_identity="picsart",
        artifact_suffix="",
        final_prompt="frozen prompt",
        muted=muted,
        provider_replacement=None,
        voice_references=(),
        video_url="https://assets.invalid/video.mp4",
        image_url=None,
        audio_urls=(),
        provider_duration=5,
        resolution="768p",
        aspect_ratio="16:9",
        target_duration_s=5.0,
        request_sha256=Pipeline._json_sha256(payload),
    )


def test_submit_receipt_is_accepted_without_research_log_evidence(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "accepted-state-receipt")

    class AcceptedPicsart:
        def submit(self, **_kwargs: Any) -> ProviderResult[str]:
            return ProviderResult(
                "paid-task",
                {"outcome": "accepted", "task_id": "paid-task"},
            )

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=AcceptedPicsart(),
    )
    prepared = _prepared_submit(project_id, project)
    pipeline._begin_submission_batch([prepared])

    result = pipeline._submit_prepared_picsart(prepared)

    assert result.value == "paid-task"
    assert not (project / "logs" / "picsart-submit").exists()
    has_state_receipt, task_id = pipeline._state_submission_task(
        store.read(project_id), 1,
    )
    assert has_state_receipt is True
    assert task_id == "paid-task"


def test_submit_response_then_accepted_state_write_failure_becomes_unknown(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "accepted-write-crash")
    calls = 0

    class AcceptedPicsart:
        def submit(self, **_kwargs: Any) -> ProviderResult[str]:
            nonlocal calls
            calls += 1
            return ProviderResult(
                "paid-task",
                {"outcome": "accepted", "task_id": "paid-task"},
            )

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=AcceptedPicsart(),
    )
    prepared = _prepared_submit(project_id, project)
    pipeline._begin_submission_batch([prepared])
    original_update = store.update
    failed_accepted_write = False

    def update(project: str, **changes: Any) -> dict[str, Any]:
        nonlocal failed_accepted_write
        ledger = changes.get("picsart_submission_receipts")
        records = (
            ledger.get("segments", {}).get("1", [])
            if isinstance(ledger, dict) else []
        )
        if (
            not failed_accepted_write
            and records
            and records[-1].get("status") == "accepted"
        ):
            failed_accepted_write = True
            raise OSError("simulated crash before accepted receipt commit")
        return original_update(project, **changes)

    monkeypatch.setattr(store, "update", update)

    with pytest.raises(OSError, match="simulated crash"):
        pipeline._submit_prepared_picsart(prepared)

    assert calls == 1
    receipt = store.read(project_id)["picsart_submission_receipts"]["segments"]["1"][-1]
    assert receipt["status"] == "unknown"
    assert receipt["task_id"] is None
    with pytest.raises(ValueError, match="outcome is unknown"):
        pipeline._state_submission_task(store.read(project_id), 1)


def test_poll_http_4xx_is_not_provider_task_terminal_failure() -> None:
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {"status_code": 404, "body": {}}}],
    }) is False
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {
            "status_code": 500,
            "body": {"response": {"status": "FAILED"}},
        }}],
    }) is False
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {
            "status_code": 200,
            "body": {"response": {"status": "FAILED"}},
        }}],
    }) is True


@pytest.mark.parametrize(
    "segments",
    [
        [
            {"index": 1, "duration_s": 5.0},
            {"index": 1, "duration_s": 5.0},
        ],
        [{"index": 1, "duration_s": float("nan")}],
        [{"index": 1, "duration_s": 0.001}],
        [{"index": 1, "duration_s": 5.0, "join_mode": "bogus"}],
        [{"index": 1, "duration_s": 5.0, "join_mode": "continue"}],
    ],
)
def test_resume_invalid_segment_structure_fails_before_any_submit(
    tmp_path: Path,
    segments: list[dict[str, Any]],
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "invalid-resume-segments")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=len(segments),
        segments=segments,
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    picsart = _FlowPicsart()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
    )

    with pytest.raises(ValueError, match="stored segment"):
        pipeline.validate_r2v_resume(project_id)
    assert picsart.submits == []


def test_h3_manifest_rejects_new_voice_mapping_with_old_prompt_binding(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = {"output": {"resolution": "768p", "aspect_ratio": "16:9"}}
    project_id, project, _state = store.create(request, "voice-binding")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    prompt_path = segment_dir / "h3_prompt.txt"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    prompt_path.write_text("old prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "dialogue": [],
        "visual_prompt": "visual",
        "status": "failed",
    }
    library = load_voice_library()
    voice_directory = project / "work" / "voice-references"
    voice_directory.mkdir(parents=True)

    def reference(voice: dict[str, Any]) -> dict[str, Any]:
        return {
            "audio_reference": "Audio 1",
            "voice_id": voice["voice_id"],
            "asset_path": "work/voice-references/audio-01.wav",
            "asset_sha256": voice["sha256"],
            "features": library.model_line(voice["voice_id"]),
        }

    first = reference(library.voices[0])
    audio = voice_directory / "audio-01.wav"
    audio.write_bytes(library.files[first["voice_id"]].read_bytes())
    first_index = {
        "people": {"person-01": {"voice": dict(first)}},
        "entities": {}, "scenes": {}, "relations": {},
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        enhanced_element_index=first_index,
        voice_references=[first],
        object_replace={"input_sha256": "a" * 64},
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "old prompt", muted, "b" * 64)],
    )
    second = reference(library.voices[1])
    audio.write_bytes(library.files[second["voice_id"]].read_bytes())
    second_index = {
        "people": {"person-01": {"voice": dict(second)}},
        "entities": {}, "scenes": {}, "relations": {},
    }
    store.update(
        project_id,
        enhanced_element_index=second_index,
        voice_references=[second],
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )

    with pytest.raises(ValueError, match="preparation bindings do not match"):
        pipeline.validate_r2v_resume(project_id)


def test_h3_manifest_rejects_a_changed_replacement_image(
    tmp_path: Path,
) -> None:
    request = {"output": {"resolution": "768p", "aspect_ratio": "16:9"}}
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "replacement-binding")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"original-image")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "visual",
        "status": "failed",
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
        segment_count=1,
        segments=[segment],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={"input_sha256": "a" * 64},
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "prompt", muted, "b" * 64)],
        replacement=replacement,
        request=request,
    )
    replacement.write_bytes(b"changed-image")

    with pytest.raises(ValueError, match="artifact receipt does not match"):
        pipeline.validate_r2v_resume(project_id)


def test_new_object_replace_state_cannot_use_legacy_manifest_fallback(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "new-state-no-manifest")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    (segment_dir / "source.mp4").write_bytes(b"source")
    (segment_dir / "source-muted.mp4").write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("stale prompt", encoding="utf-8")
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[{"index": 1, "duration_s": 5.0}],
        object_replace={"input_sha256": "a" * 64},
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )

    with pytest.raises(ValueError, match="missing its artifact manifest"):
        pipeline.validate_r2v_resume(project_id)
