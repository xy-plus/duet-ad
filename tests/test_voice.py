"""任务 T2：口播链路纯函数（extract_audio 抽音轨 / validate_voice_lines 白名单校验）。"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app import storage, voice
from app.pipeline import PipelineError


@pytest.fixture
def video_with_audio(tmp_path):
    """ffmpeg 合成 1 秒带音轨样例视频（testsrc 画面 + 440Hz 正弦音）。"""
    p = tmp_path / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-shortest", "-pix_fmt", "yuv420p", str(p),
        ],
        check=True, capture_output=True,
    )
    return p


def _conv(tmp_path, video):
    """按会话目录布局摆放 source.mp4。"""
    cdir = tmp_path / "conv"
    (cdir / "work").mkdir(parents=True)
    shutil.copy(video, cdir / "source.mp4")
    return cdir


# ---------- extract_audio ----------


class TestExtractAudio:
    def test_extracts_voice_mp3(self, tmp_path, video_with_audio):
        cdir = _conv(tmp_path, video_with_audio)
        out = voice.extract_audio(cdir)
        assert out == cdir / "work" / "voice.mp3"
        assert out.is_file() and out.stat().st_size > 0

    def test_no_audio_track_returns_none(self, tmp_path, video_1s):
        cdir = _conv(tmp_path, video_1s)
        assert voice.extract_audio(cdir) is None
        assert not (cdir / "work" / "voice.mp3").exists()

    def test_missing_source(self, tmp_path):
        cdir = tmp_path / "conv"
        (cdir / "work").mkdir(parents=True)
        with pytest.raises(PipelineError, match="source video missing"):
            voice.extract_audio(cdir)

    def test_missing_ffmpeg(self, tmp_path, video_1s, monkeypatch):
        cdir = _conv(tmp_path, video_1s)
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(PipelineError, match="ffmpeg"):
            voice.extract_audio(cdir)

    def test_ffmpeg_failure(self, tmp_path, video_with_audio, monkeypatch):
        cdir = _conv(tmp_path, video_with_audio)
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="codec broken")
        monkeypatch.setattr(voice.subprocess, "run", lambda *a, **kw: fake)
        with pytest.raises(PipelineError, match="exit 1"):
            voice.extract_audio(cdir)

    def test_probe_failure_defers_to_ffmpeg(self, tmp_path, video_with_audio, monkeypatch):
        """storage.probe_audio 探测失败（UploadError）→ 交 ffmpeg 裁决，不误判无音轨。"""
        cdir = _conv(tmp_path, video_with_audio)

        def fake_probe(_path):
            raise storage.UploadError("ffprobe failed")

        monkeypatch.setattr(voice.storage, "probe_audio", fake_probe)
        out = voice.extract_audio(cdir)
        assert out == cdir / "work" / "voice.mp3"
        assert out.is_file() and out.stat().st_size > 0


# ---------- validate_voice_lines ----------

VALID_LINES = [
    {"text": "第一句。", "start_s": 0.0, "end_s": 2.0},
    {"text": "第二句。", "start_s": 2.0, "end_s": 4.5},
    {"text": "收尾。", "start_s": 4.5, "end_s": 10.0},
]


class TestValidateVoiceLines:
    def test_valid(self):
        lines = voice.validate_voice_lines(json.dumps(VALID_LINES).encode(), 10.0)
        assert lines == VALID_LINES

    def test_valid_equal_start_and_float_slack(self):
        """start_s 相等允许（单调不减）；end_s 尾部浮点误差允许。"""
        data = [
            {"text": "a", "start_s": 0.0, "end_s": 5.0},
            {"text": "b", "start_s": 5.0, "end_s": 10.005},
        ]
        lines = voice.validate_voice_lines(json.dumps(data).encode(), 10.0)
        assert [l["text"] for l in lines] == ["a", "b"]

    def test_extra_keys_stripped(self):
        """返回项只保留白名单字段，agent 附加字段不进 meta。"""
        data = [{"text": "x", "start_s": 0.0, "end_s": 1.0, "note": "junk"}]
        lines = voice.validate_voice_lines(json.dumps(data).encode(), 10.0)
        assert lines == [{"text": "x", "start_s": 0.0, "end_s": 1.0}]

    def test_not_utf8(self):
        with pytest.raises(PipelineError, match="UTF-8"):
            voice.validate_voice_lines(b"\xff\xfe\x00", 10.0)

    def test_not_json(self):
        with pytest.raises(PipelineError, match="JSON"):
            voice.validate_voice_lines(b"not json", 10.0)

    @pytest.mark.parametrize("payload", [b"{}", b"[]", b'"str"'])
    def test_not_a_nonempty_list(self, payload):
        with pytest.raises(PipelineError, match="array"):
            voice.validate_voice_lines(payload, 10.0)

    def test_item_not_object(self):
        with pytest.raises(PipelineError, match=r"voice_lines\[0\]"):
            voice.validate_voice_lines(b"[1]", 10.0)

    def test_error_names_failing_index(self):
        """错误信息指明第几项：第二项 start>end。"""
        data = [
            {"text": "ok", "start_s": 0.0, "end_s": 1.0},
            {"text": "bad", "start_s": 5.0, "end_s": 1.0},
        ]
        with pytest.raises(PipelineError, match=r"voice_lines\[1\]"):
            voice.validate_voice_lines(json.dumps(data).encode(), 10.0)

    @pytest.mark.parametrize(
        "item,msg",
        [
            ({"start_s": 0.0, "end_s": 1.0}, "text"),          # 缺 text
            ({"text": "", "start_s": 0.0, "end_s": 1.0}, "text"),  # text 空
            ({"text": "   ", "start_s": 0.0, "end_s": 1.0}, "text"),  # 全空白
            ({"text": 123, "start_s": 0.0, "end_s": 1.0}, "text"),  # text 非字符串
            ({"text": "x", "end_s": 1.0}, "start_s"),           # 缺 start_s
            ({"text": "x", "start_s": 0.0}, "end_s"),           # 缺 end_s
            ({"text": "x", "start_s": "0.0", "end_s": 1.0}, "number"),  # 非 number
            ({"text": "x", "start_s": True, "end_s": 1.0}, "number"),   # bool 不算 number
            ({"text": "x", "start_s": -1.0, "end_s": 1.0}, "start_s"),  # 负时间
            ({"text": "x", "start_s": 2.0, "end_s": 1.0}, "start_s"),   # start ≥ end
            ({"text": "x", "start_s": 0.0, "end_s": 11.0}, "duration"),  # 超时长
        ],
    )
    def test_invalid_items(self, item, msg):
        with pytest.raises(PipelineError, match=msg):
            voice.validate_voice_lines(json.dumps([item]).encode(), 10.0)

    def test_start_s_out_of_order(self):
        data = [
            {"text": "b", "start_s": 2.0, "end_s": 3.0},
            {"text": "a", "start_s": 1.0, "end_s": 2.0},
        ]
        with pytest.raises(PipelineError, match=r"voice_lines\[1\]"):
            voice.validate_voice_lines(json.dumps(data).encode(), 10.0)

    def test_raw_too_large(self):
        """raw 超 32KB 上限 → PipelineError（错误带实际大小与上限）。"""
        raw = b"[" + b" " * (voice.MAX_VOICE_LINES_BYTES + 1) + b"]"
        with pytest.raises(PipelineError, match="exceeds 32768 bytes"):
            voice.validate_voice_lines(raw, 10.0)

    def test_text_too_long(self):
        data = [{"text": "字" * 501, "start_s": 0.0, "end_s": 1.0}]
        with pytest.raises(PipelineError, match=r"voice_lines\[0\].text.*500"):
            voice.validate_voice_lines(json.dumps(data).encode(), 10.0)

    def test_text_at_limit_ok(self):
        data = [{"text": "字" * 500, "start_s": 0.0, "end_s": 1.0}]
        lines = voice.validate_voice_lines(json.dumps(data).encode(), 10.0)
        assert len(lines[0]["text"]) == 500

    def test_too_many_items(self):
        data = [
            {"text": f"第{i}句", "start_s": i * 0.01, "end_s": i * 0.01 + 0.005}
            for i in range(201)
        ]
        with pytest.raises(PipelineError, match="exceeds 200 items"):
            voice.validate_voice_lines(json.dumps(data).encode(), 10.0)
