import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import codex_output_schemas, pipeline
from app.codex_runner import CodexError, CodexOutputValidationError
from app.config import Settings


def _png() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _closed(schema: object) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            assert set(schema.get("required", ())) == set(schema.get("properties", ()))
        for value in schema.values():
            _closed(value)
    elif isinstance(schema, list):
        for value in schema:
            _closed(value)


def test_visual_and_dialogue_schemas_are_closed_and_do_not_echo_backend_fields():
    schemas = [
        codex_output_schemas.VISUAL_PROMPT_SCHEMA,
        codex_output_schemas.dialogue_lines_schema(line_count=2),
    ]
    for schema in schemas:
        _closed(schema)
        rendered = json.dumps(schema, ensure_ascii=False)
        for forbidden in ("segment_index", "frame_order", "start_s", "end_s", "line_id"):
            assert f'"{forbidden}"' not in rendered


def test_visual_prompt_input_carries_exact_render_switches():
    prompt = pipeline._codex_prompt(
        Path("/tmp/visual-contract-test"),
        visual_only=True,
        render_options={
            "remove_subtitle": True,
            "remove_watermark": False,
        },
    )
    assert (
        'generation_config={"remove_subtitle":true,"remove_watermark":false}'
        in prompt
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": 1},
        {"prompt": "valid", "extra": True},
    ],
)
def test_visual_prompt_normalizer_rejects_missing_extra_and_wrong_types(payload):
    with pytest.raises(CodexOutputValidationError):
        codex_output_schemas.normalize_visual_prompt(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"lines": [{"content": "one"}]},
        {"lines": [{"content": "one"}, {"content": "two"}, {"content": "three"}]},
        {"lines": [{"content": "one"}, {"content": 2}]},
        {"lines": [{"content": "one"}, {"content": "two", "start_s": 1}]},
    ],
)
def test_dialogue_normalizer_rejects_count_shape_and_type_drift(payload):
    source = [
        {"line_id": "line-1", "text": "source one", "start_s": 0.0, "end_s": 1.0},
        {"line_id": "line-2", "text": "source two", "start_s": 1.0, "end_s": 2.0},
    ]
    with pytest.raises(CodexOutputValidationError):
        codex_output_schemas.normalize_dialogue_lines(payload, source_lines=source)


def test_dialogue_normalizer_mechanically_preserves_identity_order_and_timecodes():
    source = [
        {"line_id": "line-1", "text": "source one", "start_s": 0.0, "end_s": 1.0},
        {"line_id": "line-2", "text": "source two", "start_s": 1.0, "end_s": 2.0},
    ]
    transformed = codex_output_schemas.normalize_dialogue_lines(
        {"lines": [
            {"content": "  target one\n"},
            {"content": "target two"},
        ]},
        source_lines=source,
    )
    assert transformed == [
        {"line_id": "line-1", "text": "  target one\n", "start_s": 0.0, "end_s": 1.0},
        {"line_id": "line-2", "text": "target two", "start_s": 1.0, "end_s": 2.0},
    ]


def test_dialogue_content_is_not_stripped_by_timeline_validation():
    content = "  complete model line\n"
    decision = {
        "text": content,
        "start_s": 0.0,
        "end_s": 1.0,
        "classification": "spoken",
        "provenance": "asr",
        "kept": True,
    }

    lines, decisions, warnings = pipeline._normalize_voice_timeline(
        [decision], 1.0,
    )

    assert lines == [{"text": content, "start_s": 0.0, "end_s": 1.0}]
    assert decisions == [decision]
    assert warnings == []


class _VisualRunner:
    def __init__(self, payload: object, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls = []

    def run_isolated_until_output(
        self, stage, prompt, *, session_dir, output_path, max_output_bytes,
        validate_output, output_schema,
    ):
        self.calls.append((Path(stage), prompt, Path(session_dir), output_schema))
        # A model-side business-file write is not authoritative.
        (Path(stage) / "work" / "prompt.txt").write_text(
            "direct write must be ignored", encoding="utf-8",
        )
        if self.fail:
            raise CodexError("structured visual failed", retryable=True)
        return validate_output(json.dumps(self.payload).encode("utf-8"))


def test_visual_uses_final_answer_only_and_publishes_against_backend_frozen_frames(tmp_path):
    cdir = tmp_path / "conversation"
    work = cdir / "work"
    work.mkdir(parents=True)
    frozen = tuple(_png() for _ in range(3))
    runner = _VisualRunner({"prompt": "authoritative visual"})

    pipeline._run_visual_codex(
        runner,
        cdir,
        "visual request",
        work,
        isolate_dialogue=True,
        skill_bytes=b"skill",
        frozen_keyframes=frozen,
    )

    assert (work / "prompt.txt").read_text(encoding="utf-8") == "authoritative visual"
    assert [(work / "keyframes" / f"{index:02d}.png").read_bytes() for index in range(1, 4)] == list(frozen)
    assert runner.calls[0][3] == codex_output_schemas.VISUAL_PROMPT_SCHEMA


def test_visual_retry_exhaustion_is_phase_local_and_publishes_nothing(tmp_path):
    cdir = tmp_path / "conversation"
    work = cdir / "work"
    work.mkdir(parents=True)
    frozen = tuple(_png() for _ in range(3))
    runner = _VisualRunner({"prompt": "unused"}, fail=True)
    settings = Settings(
        access_token="test", data_dir=tmp_path / "data", retry_count=2,
        retry_interval_s=0,
    )

    with pytest.raises(CodexError, match="structured visual failed"):
        pipeline._run_visual_with_retry(
            settings,
            runner,
            cdir,
            "visual request",
            work,
            isolate_dialogue=True,
            step="segment 1 visual codex",
            frozen_keyframes=frozen,
            skill_bytes=b"skill",
        )

    assert len(runner.calls) == 3
    assert not (work / "prompt.txt").exists()
    assert [(work / "keyframes" / f"{index:02d}.png").read_bytes() for index in range(1, 4)] == list(frozen)


class _DialogueRunner:
    def __init__(self, payload: object, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls = 0

    def run_isolated_until_output(
        self, stage, prompt, *, session_dir, output_path, max_output_bytes,
        validate_output, output_schema,
    ):
        self.calls += 1
        Path(output_path).write_text(
            json.dumps({"lines": [{"content": "direct write"}]}), encoding="utf-8",
        )
        if self.fail:
            raise CodexError("dialogue phase failed", retryable=True)
        return validate_output(json.dumps(self.payload).encode("utf-8"))


def test_dialogue_final_answer_publishes_text_with_backend_identity_and_timing(tmp_path):
    work = tmp_path / "conversation" / "work"
    work.mkdir(parents=True)
    source = [
        {"line_id": "line-1", "text": "source one", "start_s": 0.0, "end_s": 1.0},
        {"line_id": "line-2", "text": "source two", "start_s": 1.0, "end_s": 2.0},
    ]
    runner = _DialogueRunner({"lines": [
        {"content": "target one"}, {"content": "target two"},
    ]})

    result = pipeline._run_voice_attempt(runner, work, "rewrite", source)

    assert result == [
        {"line_id": "line-1", "text": "target one", "start_s": 0.0, "end_s": 1.0},
        {"line_id": "line-2", "text": "target two", "start_s": 1.0, "end_s": 2.0},
    ]
    assert not (work / "voice_lines.json").exists()


def test_dialogue_retry_exhaustion_does_not_repeat_local_phase(tmp_path):
    work = tmp_path / "conversation" / "work"
    work.mkdir(parents=True)
    source = [{"text": "source", "start_s": 0.0, "end_s": 1.0}]
    runner = _DialogueRunner({"lines": [{"content": "unused"}]}, fail=True)
    settings = Settings(
        access_token="test", data_dir=tmp_path / "data", retry_count=2,
        retry_interval_s=0,
    )

    with pytest.raises(CodexError, match="dialogue phase failed"):
        pipeline._transform_voice_with_retry(
            settings, runner, work, "rewrite", source,
        )

    assert runner.calls == 3
    assert source == [{"text": "source", "start_s": 0.0, "end_s": 1.0}]
    assert not (work / "voice_lines.json").exists()


def test_keep_transcription_is_local_and_never_calls_model(tmp_path, monkeypatch):
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    cli.write_bytes(b"cli")
    model.write_bytes(b"model")
    work = tmp_path / "conversation" / "work"
    work.mkdir(parents=True)
    (work / "voice.mp3").write_bytes(b"audio")
    settings = Settings(
        access_token="test", data_dir=tmp_path / "data", asr_cli=cli,
        asr_model=model,
    )
    expected = [{"text": "local", "start_s": 0.0, "end_s": 1.0}]
    monkeypatch.setattr(pipeline.asr, "transcribe", lambda *_args, **_kwargs: expected)

    class NoModel:
        def __getattr__(self, _name):
            raise AssertionError("keep must not call a model")

    assert pipeline._transcribe_voice_attempt(
        settings, NoModel(), work, "", 1.0, "keep",
    ) == expected
