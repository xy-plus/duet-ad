"""口播转换三模式：参数校验、无音轨兼容、meta 落盘和幂等。"""
import json
import subprocess

import pytest
from conftest import AUTH

from app import storage


@pytest.fixture
def video_with_audio(tmp_path):
    """1 秒真实视频 + sine 音轨（video_1s fixture 本身无音轨，正好互补）。"""
    p = tmp_path / "with_audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-pix_fmt", "yuv420p", "-shortest", str(p),
        ],
        check=True, capture_output=True,
    )
    return p


def _post(client, video, data=None, rid=None):
    data = dict(data or {})
    if rid is not None:
        data["client_request_id"] = rid
    with open(video, "rb") as f:
        return client.post("/api/conversations", headers=AUTH,
                           files={"file": ("clip.mp4", f, "video/mp4")},
                           data=data)


def _meta(settings, cid):
    return json.loads((settings.data_dir / cid / "meta.json").read_text())


def test_voice_mode_invalid_422(client, video_1s, settings):
    r = _post(client, video_1s, {"voice_mode": "scream"})
    assert r.status_code == 422
    assert "voice_mode" in r.json()["detail"]
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_translate_requires_target_language_422(client, video_1s, settings):
    for lang in ("", "   "):
        r = _post(client, video_1s, {"voice_mode": "translate", "target_language": lang})
        assert r.status_code == 422, lang
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_voice_mode_no_audio_is_valid_empty_dialogue_input(client, video_1s, settings):
    """无音轨视频合法：后续 auto 模式生成空台词，不把画面文字变成发声。"""
    r = _post(client, video_1s, {"voice_mode": "keep"})
    assert r.status_code == 201
    meta = _meta(settings, r.json()["id"])
    assert meta["voice_mode"] == "keep"
    assert meta["dialogue_mode"] == "auto"


def test_voice_mode_none_is_retired(client, video_1s, settings):
    """旧页面的 none 不再接受，明确提示刷新；最终无台词由提交 dialogue_mode=none 表达。"""
    r = _post(client, video_1s, {"voice_mode": "none"})
    assert r.status_code == 409
    assert r.json() == {"detail": "页面版本已更新，请刷新页面后重试。"}
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_voice_mode_default_meta_is_keep(client, video_1s, settings):
    """不带 voice_mode 时默认保留自动听写结果。"""
    r = _post(client, video_1s)
    assert r.status_code == 201
    assert _meta(settings, r.json()["id"])["voice_mode"] == "keep"


def test_voice_mode_translate_with_audio_ok(client, video_with_audio, settings):
    r = _post(client, video_with_audio,
              {"voice_mode": "translate", "target_language": "日语"})
    assert r.status_code == 201
    meta = _meta(settings, r.json()["id"])
    assert meta["voice_mode"] == "translate"
    assert meta["target_language"] == "日语"


def test_voice_mode_keep_with_audio_ok(client, video_with_audio, settings):
    """keep 成功且非 translate 不落 target_language。"""
    r = _post(client, video_with_audio, {"voice_mode": "keep"})
    assert r.status_code == 201
    meta = _meta(settings, r.json()["id"])
    assert meta["voice_mode"] == "keep"
    assert "target_language" not in meta


def test_voice_mode_keep_drops_target_language(client, video_with_audio, settings):
    """非 translate 模式显式带 target_language → 201 且被丢弃。"""
    r = _post(client, video_with_audio,
              {"voice_mode": "keep", "target_language": "日语"})
    assert r.status_code == 201
    assert "target_language" not in _meta(settings, r.json()["id"])


def test_voice_mode_rewrite_with_audio_ok(client, video_with_audio, settings):
    r = _post(client, video_with_audio, {"voice_mode": "rewrite"})
    assert r.status_code == 201
    assert _meta(settings, r.json()["id"])["voice_mode"] == "rewrite"


def test_voice_mode_idempotent_dedup(client, video_with_audio, settings):
    """同 client_request_id + 口播参数二次提交：200 返回既有会话，只建一个目录。"""
    r1 = _post(client, video_with_audio, {"voice_mode": "keep"}, rid="req-voice-01")
    r2 = _post(client, video_with_audio, {"voice_mode": "keep"}, rid="req-voice-01")
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.json()["id"] == r1.json()["id"]
    assert len(list(settings.data_dir.iterdir())) == 1
    assert _meta(settings, r1.json()["id"])["voice_mode"] == "keep"


def test_probe_audio_detects_presence(video_with_audio, video_1s):
    assert storage.probe_audio(video_with_audio) is True
    assert storage.probe_audio(video_1s) is False


def test_probe_audio_ffprobe_failure_raises(monkeypatch, tmp_path):
    def boom(*a, **kw):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr("app.storage.subprocess.run", boom)
    with pytest.raises(storage.UploadError):
        storage.probe_audio(tmp_path / "x.mp4")


def test_new_conversation_voice_defaults(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    assert meta["voice_mode"] == "keep"
    assert "target_language" not in meta


def test_new_conversation_voice_explicit(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4",
                                    voice_mode="translate", target_language="日语")
    assert meta["voice_mode"] == "translate"
    assert meta["target_language"] == "日语"
