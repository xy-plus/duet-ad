"""voice-over（幕后口播）模式：两种及以上音色 → 单一音色画外音、不对口型。

判定在 _run_dialogue_audio 逐行音色解析完成后进行（len(set(line_voices))
>= 2），统一为 default_voice_type 并把判定证据落 state.dialogue_voice_over；
H3 冻结提示词 dialogue 块、H3 提交 reference_audio_urls、Context IR 蒸馏
输入与素材、冻结清单的 dialogue_track 回执全部消费持久化值，不重新判定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app import media
from app.doubao_voices import load_doubao_voice_catalog
from app.object_replace_schema import OBJECT_REPLACE_OUTPUT_VERSION
from app.pipeline import Pipeline
from app.research_log import ResearchLog
from app.store import ProjectStore

from test_pipeline import (
    _dialogue_audio_fixture,
    _effective_request,
    _FlowDoubaoTTS,
    _FlowPicsart,
    _materialize_test_replacement_sheet,
    _prepare_h3_person_fixture,
    _settings,
)

_ADAM = "en_male_adam-imitation_uranus_bigtts"
_INFORMATIVE = "en_female_authoritative-informative_uranus_bigtts"


def _voice_over_block(
    enabled: bool,
    resolved: list[str],
    language_label: str = "英语",
) -> dict[str, Any]:
    default = load_doubao_voice_catalog().default_voice_type(language_label)
    return {
        "enabled": enabled,
        "resolved_voices": list(resolved),
        "voice_type": default if enabled else None,
        "language_label": language_label,
    }


def test_two_distinct_voices_trigger_voice_over_mode(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    tts = _FlowDoubaoTTS()
    (
        store, project_id, project, pipeline,
        prepared, enhanced_index,
    ) = _dialogue_audio_fixture(tmp_path, monkeypatch, tts)
    # 两行解析出两种已标定音色（无画面/方舟证据时保留继承绑定）。
    enhanced_index["dialogue_voices"] = [_ADAM, _INFORMATIVE]

    pipeline._run_dialogue_audio(
        project_id, project, prepared, enhanced_index, "英语",
        ResearchLog(project),
    )

    library = load_doubao_voice_catalog()
    default = library.default_voice_type("英语")
    # 全部行统一为该语种已标定默认音色。
    assert [call["voice_type"] for call in tts.tts_calls] == [default, default]
    # 模式与判定证据持久化。
    assert store.read(project_id)["dialogue_voice_over"] == {
        "enabled": True,
        "resolved_voices": [_ADAM, _INFORMATIVE],
        "voice_type": default,
        "language_label": "英语",
    }
    # section_id 自然只有一个说话人 section。
    sections = store.read(project_id)["dialogue_tts"]["sections"]
    assert len(sections) == 1
    assert sections[0]["voice_type"] == default
    # 两行音色不同 → 输入哈希不同行各自重新合成（fixture 全新项目，两行
    # 均已合成即覆盖"音色变化行自动重新合成"路径）。
    assert [call["text"] for call in tts.tts_calls] == ["第一句", "第二句"]


def test_single_voice_project_stays_lip_sync_mode(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    tts = _FlowDoubaoTTS()
    (
        store, project_id, project, pipeline,
        prepared, enhanced_index,
    ) = _dialogue_audio_fixture(tmp_path, monkeypatch, tts)
    enhanced_index["dialogue_voices"] = [_ADAM, _ADAM]

    pipeline._run_dialogue_audio(
        project_id, project, prepared, enhanced_index, "英语",
        ResearchLog(project),
    )

    assert store.read(project_id)["dialogue_voice_over"] == {
        "enabled": False,
        "resolved_voices": [_ADAM, _ADAM],
        "voice_type": None,
        "language_label": "英语",
    }
    assert [call["voice_type"] for call in tts.tts_calls] == [_ADAM, _ADAM]


def _voice_over_prompt_fixture(
    tmp_path: Path, monkeypatch: Any, enabled: bool,
) -> tuple[Pipeline, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    pipeline, project, item, enhanced_index, replacement_sheet = (
        _prepare_h3_person_fixture(tmp_path, monkeypatch)
    )
    item["dialogue"] = [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}]
    resolved = [_ADAM, _INFORMATIVE] if enabled else [_ADAM, _ADAM]
    pipeline.store.update(
        project.name,
        dialogue_voice_over=_voice_over_block(enabled, resolved),
    )
    return pipeline, project, item, enhanced_index, replacement_sheet


def test_voice_over_frozen_prompt_uses_narrator_variant(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet = (
        _voice_over_prompt_fixture(tmp_path, monkeypatch, enabled=True)
    )

    _item, prompt, _muted, _sha = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
        None,
    )

    assert "lip-sync" not in prompt
    assert "Audio 1" not in prompt
    assert "single-voice narrator voice-over" in prompt
    assert "Do not generate any speech beyond the narration" in prompt
    assert "[0.000-2.000s] 第一句" in prompt


def test_single_voice_frozen_prompt_keeps_lip_sync_variant(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet = (
        _voice_over_prompt_fixture(tmp_path, monkeypatch, enabled=False)
    )

    _item, prompt, _muted, _sha = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
        None,
    )

    assert "lip-sync precisely to Audio 1" in prompt
    assert "voice-over" not in prompt


def _freeze_prompt_and_manifest(
    tmp_path: Path,
    monkeypatch: Any,
    enabled: bool,
) -> tuple[ProjectStore, str, Path, Pipeline, Path, Any]:
    """单段带台词项目：冻结 H3 清单并返回 preflight 所需的全部句柄。"""
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "voice-over-h3")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    dialogue_track = segment_dir / "dialogue-track.wav"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    dialogue_track.write_bytes(b"dialogue-slice")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        "visual_prompt": "visual",
        "status": "running",
    }
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[Pipeline._public_segment(segment)],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        dialogue_voice_over=_voice_over_block(
            enabled, [_ADAM, _INFORMATIVE] if enabled else [_ADAM, _ADAM],
        ),
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.name}",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0, "width": 1280, "height": 720,
            "has_audio": False,
        },
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, source=source, analysis_paths=[])],
            store.read(project_id)["enhanced_element_index"],
        )
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "prompt", muted, "b" * 64)],
        replacement_sheet=replacement_sheet,
        request=request,
    )
    return (
        store, project_id, project, pipeline, muted, replacement_sheet_path
    )


def test_voice_over_manifest_binds_no_dialogue_track_and_sends_no_audio(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, project, pipeline, muted, replacement_sheet_path = (
        _freeze_prompt_and_manifest(tmp_path, monkeypatch, enabled=True)
    )

    # voice-over 段不传 Audio 1：清单 dialogue_track 回执为 None（与无台词
    # 段同形），校验与 resume 装载一致通过。
    manifest = store.read(project_id)["h3_prepared_artifacts"]
    assert manifest["segments"]["1"]["dialogue_track"] is None
    Pipeline._validated_h3_prepared_artifacts(
        project, store.read(project_id), store.read(project_id)["segments"],
    )

    prepared = pipeline._prepare_picsart_segment(
        project_id,
        project,
        {
            "index": 1,
            "duration_s": 5.0,
            "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        },
        None,
        replacement_sheet_path,
        _effective_request(),
        ResearchLog(project),
        final_prompt="prompt",
        muted=muted,
    )

    assert prepared.dialogue_track is None
    assert prepared.audio_urls == ()


def test_single_voice_manifest_binds_dialogue_track_and_sends_audio(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, project, pipeline, muted, replacement_sheet_path = (
        _freeze_prompt_and_manifest(tmp_path, monkeypatch, enabled=False)
    )

    manifest = store.read(project_id)["h3_prepared_artifacts"]
    assert manifest["segments"]["1"]["dialogue_track"]["path"] == (
        "work/segments/0001/dialogue-track.wav"
    )

    prepared = pipeline._prepare_picsart_segment(
        project_id,
        project,
        {
            "index": 1,
            "duration_s": 5.0,
            "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        },
        None,
        replacement_sheet_path,
        _effective_request(),
        ResearchLog(project),
        final_prompt="prompt",
        muted=muted,
    )

    assert prepared.audio_urls == (
        "https://assets.invalid/dialogue-track.wav",
    )


def test_persisted_mode_is_consumed_without_recompute(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """resume 路径读持久化值：冻结后即便改 state 里的音色绑定，提示词
    装配仍按持久化的 voice-over 模式走（不重新判定）。"""
    pipeline, project, item, enhanced_index, replacement_sheet = (
        _voice_over_prompt_fixture(tmp_path, monkeypatch, enabled=True)
    )
    # 冻结后篡改音色来源输入：_prepare_h3_segment 仍按持久化模式装配。
    pipeline.store.update(
        project.name,
        enhanced_element_index={
            **enhanced_index,
            "dialogue_voices": [_ADAM, _ADAM],
        },
    )

    _item, prompt, _muted, _sha = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
        None,
    )

    assert "single-voice narrator voice-over" in prompt
    assert "lip-sync" not in prompt


@pytest.mark.parametrize(
    "block",
    [
        # 模式与音色集矛盾：声明 voice-over 但只解析出一种音色。
        {
            "enabled": True,
            "resolved_voices": [_ADAM, _ADAM],
            "voice_type": _ADAM,
            "language_label": "英语",
        },
        # 声明非 voice-over 但解析出两种音色。
        {
            "enabled": False,
            "resolved_voices": [_ADAM, _INFORMATIVE],
            "voice_type": None,
            "language_label": "英语",
        },
        # voice-over 缺统一音色。
        {
            "enabled": True,
            "resolved_voices": [_ADAM, _INFORMATIVE],
            "voice_type": None,
            "language_label": "英语",
        },
        # 非 voice-over 携带统一音色。
        {
            "enabled": False,
            "resolved_voices": [_ADAM],
            "voice_type": _ADAM,
            "language_label": "英语",
        },
        # 键集不齐。
        {"enabled": True},
        # resolved_voices 形态非法。
        {
            "enabled": True,
            "resolved_voices": [_ADAM, ""],
            "voice_type": _ADAM,
            "language_label": "英语",
        },
    ],
)
def test_voice_over_state_inconsistent_fails_closed(block: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="voice-over|voice_over"):
        Pipeline._validated_dialogue_voice_over(
            {"dialogue_voice_over": block}
        )


def test_voice_over_state_absent_means_lip_sync() -> None:
    # 旧项目（字段不存在，2026-10-03 前冻结）：按既有对口型行为续跑。
    assert Pipeline._validated_dialogue_voice_over({}) is False


def test_context_ir_distilled_input_voice_over_variant() -> None:
    from app.pipeline import _context_ir_input_text
    from test_context_ir import _builder_index, _builder_sheet

    text = _context_ir_input_text(
        visual_prompt="segment one visual",
        dialogue_lines=[{"start_s": 0.0, "end_s": 2.1, "text": "Line A."}],
        target_language="马来语",
        replacement_sheet=_builder_sheet(),
        element_index=_builder_index(),
        segment_index=1,
        usage_paragraphs=[],
        voice_over=True,
    )

    assert "Audio 1：" not in text
    assert "对口型" not in text
    assert "幕后口播" in text
    assert "[0.000-2.100s] Line A." in text


def test_context_ir_distilled_input_lip_sync_default() -> None:
    from app.pipeline import _context_ir_input_text
    from test_context_ir import _builder_index, _builder_sheet

    text = _context_ir_input_text(
        visual_prompt="segment one visual",
        dialogue_lines=[{"start_s": 0.0, "end_s": 2.1, "text": "Line A."}],
        target_language="马来语",
        replacement_sheet=_builder_sheet(),
        element_index=_builder_index(),
        segment_index=1,
        usage_paragraphs=[],
        voice_over=False,
    )

    assert "Audio 1：" in text
    assert "对口型" in text
