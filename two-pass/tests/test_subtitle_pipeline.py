from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import media, subtitles
from app.pipeline import Pipeline
from app.research_log import ResearchLog


@pytest.fixture
def context(tmp_path, monkeypatch):
    state = {"translated_voice_lines": [{"start_s": 0, "end_s": 2, "text": "你好"}]}
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.store = SimpleNamespace(
        read=lambda _: state, update=lambda _, **changes: state.update(changes),
    )
    final = tmp_path / "output" / "generated.mp4"
    final.parent.mkdir()
    final.write_bytes(b"clean")
    audio = tmp_path / "work" / "dialogue-audio"
    audio.mkdir(parents=True)
    for name in ["full-track.wav", "line-0001.wav"]:
        (audio / name).write_bytes(b"audio")
    monkeypatch.setattr(media, "probe", lambda _: {"duration_s": 2})
    cues = Mock(return_value=state["translated_voice_lines"])
    monkeypatch.setattr(subtitles, "build_subtitle_cues", cues)
    def burn(source, lines, output):
        assert source.read_bytes() == b"clean"
        output.write_bytes(b"subtitled")
        return {"count": len(lines)}
    monkeypatch.setattr(subtitles, "burn_subtitles", burn)
    return pipeline, state, tmp_path, final, cues


@pytest.mark.parametrize("frozen_request", [{}, {"add_subtitles": False}])
def test_disabled_legacy_request_never_reads_tts(context, frozen_request):
    pipeline, state, project, final, cues = context
    state.clear()
    pipeline._add_subtitles("id", project, final, frozen_request, ResearchLog(project))
    assert final.read_bytes() == b"clean"
    assert not final.with_name("generated-clean.mp4").exists()
    cues.assert_not_called()


def test_empty_dialogue_skips_with_reason(context):
    pipeline, state, project, final, cues = context
    state["translated_voice_lines"] = []
    pipeline._add_subtitles("id", project, final, {"add_subtitles": True}, ResearchLog(project))
    assert final.read_bytes() == b"clean"
    cues.assert_not_called()
    assert '"reason": "no_dialogue"' in (
        project / "logs/subtitles/project/attempt-1/output.json"
    ).read_text()


@pytest.mark.parametrize("missing", ["full-track.wav", "line-0001.wav"])
def test_missing_tts_fails_once_without_publishing(context, missing):
    pipeline, state, project, final, cues = context
    (project / "work/dialogue-audio" / missing).unlink()
    with pytest.raises(ValueError, match="缺少 TTS 音频"):
        pipeline._add_subtitles("id", project, final, {"add_subtitles": True}, ResearchLog(project))
    assert final.read_bytes() == b"clean"
    assert len(list((project / "logs/subtitles/project").glob("attempt-*"))) == 1
    cues.assert_not_called()


@pytest.mark.parametrize("resume,fail_first", [(False, False), (True, False), (True, True)])
def test_finalization_orders_subtitles_after_audio_and_preserves_public_path(context, monkeypatch, resume, fail_first):
    pipeline, state, project, final, cues = context
    order = []
    prepared = [{"index": 0, "duration_s": 2, "join_mode": "hard_cut",
                 "resume_output": project / "segment.mp4", "resume_prompt": "frozen"}]
    prepared[0]["resume_output"].write_bytes(b"segment")
    monkeypatch.setattr(pipeline, "_public_segment", lambda item: dict(item))
    monkeypatch.setattr(pipeline, "_run_dialogue_audio", lambda *args: order.append("tts"))
    monkeypatch.setattr(pipeline, "_run_h3_segments", lambda *args: [prepared[0]["resume_output"]])
    monkeypatch.setattr(pipeline, "_replace_dialogue_audio_track", lambda *args: order.append("audio"))
    def concat(inputs, output, **kwargs):
        order.append("stitch")
        output.write_bytes(b"clean")
    monkeypatch.setattr(media, "concat", concat)
    burn = subtitles.burn_subtitles
    def tracked_burn(*args):
        order.append("subtitles")
        if fail_first and order.count("subtitles") == 1:
            args[2].write_bytes(b"partial-failed-render")
            raise ValueError("local renderer failed")
        return burn(*args)
    monkeypatch.setattr(subtitles, "burn_subtitles", tracked_burn)
    request = {"add_subtitles": True, "dialogue": {"target_language": "中文"}}
    if resume:
        monkeypatch.setattr(pipeline, "_load_r2v_resume", lambda _: (project, prepared, None, None, request, [], "pass2", {}))
        monkeypatch.setattr(pipeline, "_next_resume_identity", lambda *args: ("resume", "resume"))
        monkeypatch.setattr(pipeline, "_begin_submission_batch", lambda requests: requests == [] or pytest.fail("paid batch"))
        monkeypatch.setattr(pipeline, "_prepare_picsart_segment", lambda *args, **kwargs: pytest.fail("provider preflight"))
        monkeypatch.setattr(pipeline, "_generate_picsart_segment", lambda *args: pytest.fail("paid submit"))
        monkeypatch.setattr(pipeline, "_persist_failure", lambda *args, **kwargs: state.update(status="failed"))
        if fail_first:
            pipeline.resume_r2v("id")
            assert state["status"] == "failed"
            assert final.read_bytes() == b"clean"
            assert order.count("subtitles") == 1
        pipeline.resume_r2v("id")
    else:
        pipeline._continue_after_replacement_sheet("id", project, prepared, {}, None, {}, None, request, ResearchLog(project))
    assert order[-3:] == ["stitch", "audio", "subtitles"]
    assert state["status"] == "succeeded"
    assert state["output_path"] == "output/generated.mp4"
    assert final.read_bytes() == b"subtitled"
    assert final.with_name("generated-clean.mp4").read_bytes() == b"clean"
