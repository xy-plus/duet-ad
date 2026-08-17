"""T4 拆段流水线：scenes 接入、拆段并发、句子归属、切段精度、detail 14 字段契约。"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import pipeline, storage
from app.codex_runner import CodexError, CodexRunner
from app.main import create_app

# 1×1 真实 PNG（validate_work_dir 会用 cv2 解码校验）
_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

SEGMENTS = [
    {"index": 1, "start_s": 0.0, "end_s": 8.0},
    {"index": 2, "start_s": 8.0, "end_s": 16.0},
    {"index": 3, "start_s": 16.0, "end_s": 24.0},
]

SEGMENTS_4 = [
    {"index": 1, "start_s": 0.0, "end_s": 6.0},
    {"index": 2, "start_s": 6.0, "end_s": 12.0},
    {"index": 3, "start_s": 12.0, "end_s": 18.0},
    {"index": 4, "start_s": 18.0, "end_s": 24.0},
]


def _write_valid_package(work: Path, frames: int = 3, prompt: str = "分段桩产物"):
    """按约定文件名造一套合法产物（同 test_pipeline 的桩约定）。"""
    kdir = work / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(1, frames + 1):
        name = f"{i:02d}.png"
        (kdir / name).write_bytes(_PX_PNG)
        names.append(name)
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    return names


# ---------- 句子归属 ----------


def test_attribute_lines_basic():
    lines = [
        {"text": "第一段。", "start_s": 3.0, "end_s": 3.5},
        {"text": "第三段。", "start_s": 17.5, "end_s": 18.0},
    ]
    got = pipeline.attribute_lines(lines, SEGMENTS)
    assert [l["text"] for l in got[1]] == ["第一段。"]
    assert got[2] == []
    assert [l["text"] for l in got[3]] == ["第三段。"]


def test_attribute_lines_start_at_boundary_belongs_to_later_segment():
    """start_s 恰在段边界 → 归后段（与 scenes.py 的 [start,end) 口径一致）。"""
    lines = [{"text": "边界句。", "start_s": 8.0, "end_s": 9.0}]
    got = pipeline.attribute_lines(lines, SEGMENTS)
    assert got[1] == []
    assert [l["text"] for l in got[2]] == ["边界句。"]


def test_attribute_lines_tail_within_epsilon_belongs_to_last_segment():
    """start_s 恰在末段终点或 0.01s 浮点误差内 → 归末段；超出容差 → 不归任何段。"""
    lines = [
        {"text": "终点句。", "start_s": 24.0, "end_s": 24.5},
        {"text": "误差句。", "start_s": 24.005, "end_s": 24.5},
        {"text": "越界句。", "start_s": 24.5, "end_s": 24.9},
    ]
    got = pipeline.attribute_lines(lines, SEGMENTS)
    assert [l["text"] for l in got[3]] == ["终点句。", "误差句。"]
    assert got[1] == [] and got[2] == []
    assert "越界句。" not in [l["text"] for l in got[3]]


# ---------- 后端前缀（不依赖 codex 写） ----------


def test_prefix_no_bgm_prepends_line(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("正文", encoding="utf-8")
    got = pipeline._prefix_no_bgm("正文", prompt_path)
    assert got == pipeline.NO_BGM_LINE + "\n正文"
    assert prompt_path.read_text(encoding="utf-8") == got


def test_prefix_no_bgm_rejects_oversize(tmp_path):
    """前缀后超 MAX_PROMPT_BYTES → PipelineError（校验在前缀插入前，机械操作须自查）。"""
    prompt_path = tmp_path / "prompt.txt"
    big = "x" * (pipeline.MAX_PROMPT_BYTES - 1)  # 加前缀必然超限
    with pytest.raises(pipeline.PipelineError, match="prefix"):
        pipeline._prefix_no_bgm(big, prompt_path)


# ---------- ffmpeg 切段精度 ----------


def test_cut_segment_duration_within_tolerance(tmp_path):
    """ffmpeg -ss 在 -i 前重编码切段：切出时长与段边界误差 <0.1s（真 subprocess）。"""
    video = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=20:size=320x240:rate=10",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True, capture_output=True,
    )
    segdir = tmp_path / "seg1"
    pipeline._cut_segment(video, 5.0, 12.0, segdir)
    out = segdir / "source.mp4"
    assert out.is_file()
    assert abs(pipeline._probe_duration(out) - 7.0) < pipeline.CUT_DURATION_TOLERANCE_S


# ---------- detail 14 字段契约 ----------


def test_detail_segments_voice_lines_defaults(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post(
            "/api/conversations", headers=AUTH,
            files={"file": ("clip.mp4", f, "video/mp4")},
        )
    cid = r.json()["id"]
    body = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert body["segments"] == []
    assert body["voice_lines"] == []


def test_detail_exposes_segments_and_voice_lines(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        meta = storage.new_conversation(settings.data_dir, note="", orig_name="a.mp4")
        segs = [
            {"index": 1, "start_s": 0.0, "end_s": 8.0,
             "keyframes": ["01.png"], "prompt": "不要生成背景音乐\np", "lines": ["台词。"]},
        ]
        lines = [{"text": "台词。", "start_s": 0.0, "end_s": 1.0}]
        storage.update_meta(settings.data_dir, meta["id"], segments=segs, voice_lines=lines)
        r = c.get(f"/api/conversations/{meta['id']}", headers=AUTH)
    body = r.json()
    assert body["segments"] == segs
    assert body["voice_lines"] == lines


# ---------- 拆段编排（桩 codex，monkeypatch _run_cmd/_cut_segment） ----------


def _make_segment_conversation(settings, segments, voice_mode="none"):
    """建会话 + 落 source 占位文件 + 设定 voice_mode；返回 meta。"""
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    (settings.data_dir / meta["id"] / "source.mp4").write_bytes(b"fake-video")
    if voice_mode != "none":
        storage.update_meta(settings.data_dir, meta["id"], voice_mode=voice_mode)
    return meta


def _fake_cmd_segments(calls, segments):
    """假 _run_cmd：extract 写空 manifest；scenes 写给定 segments 的 scenes.json。"""

    def fake_cmd(argv, *, timeout, step, cwd=None):
        calls["cmd"].append({"argv": list(argv), "timeout": timeout, "step": step, "cwd": cwd})
        if step == "extract":
            out = Path(argv[argv.index("--out-dir") + 1])
            (out / "contact_sheet.jpg").write_bytes(b"sheet")
            (out / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 24.0, "scenes": [], "segments": segments}),
                encoding="utf-8",
            )

    return fake_cmd


def _fake_cut(source, start_s, end_s, segdir):
    segdir.mkdir(parents=True, exist_ok=True)
    (segdir / "source.mp4").write_bytes(b"fake-video")


def test_run_single_segment_no_prefix_and_no_segments_key(tmp_path, monkeypatch):
    """segments 空（≤20s）：现有流程原样——prompt 无前缀、meta 不写 segments。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, [])
    cid = meta["id"]
    calls = {"cmd": [], "codex": []}

    def fake_codex(self, workdir, prompt):
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, []))
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert "segments" not in m
    assert m["prompt"] == "分段桩产物"  # 单段模式不加前缀
    (codex_call,) = calls["codex"]
    assert "分段模式" not in codex_call["prompt"]

    # scenes 步 argv 契约：venv python、scenes.py、--work-dir、300s 超时
    scenes = calls["cmd"][1]
    assert scenes["step"] == "scenes"
    assert scenes["argv"][0] == sys.executable
    assert str(pipeline.SCENES_SCRIPT) in scenes["argv"]
    assert "--work-dir" in scenes["argv"]
    assert scenes["timeout"] == 300


def test_run_segment_codex_cwd_and_prompt(tmp_path, monkeypatch):
    """分段 codex 调用：cwd=会话目录（与单段一致）；prompt 含 SKILL.md、分段模式、
    work/segments/N/、全片 scenes.json、该段台词与硬性禁令。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    cdir = settings.data_dir / cid
    calls = {"cmd": [], "codex": []}

    def fake_codex(self, workdir, prompt):
        m = re.search(r"work/segments/(\d+)/", prompt)
        assert m, "segment index missing from prompt"
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert len(calls["codex"]) == 3
    assert {c["workdir"] for c in calls["codex"]} == {cdir}  # 与单段模式同一 cwd
    by_index = {
        re.search(r"work/segments/(\d+)/", c["prompt"]).group(1): c
        for c in calls["codex"]
    }
    prompt = by_index["2"]["prompt"]
    for needle in (
        str(pipeline.SKILL_MD),
        "分段模式",
        "work/segments/2/",
        "work/scenes.json",
        "voice_lines.json",
        sys.executable,
        "禁止联网",
        "环境变量",
    ):
        assert needle in prompt, needle
    assert "voice.mp3" not in prompt
    # scripts 只拷一份到会话目录；段目录不复制 scripts/scenes.json；无口播不写段台词文件
    assert (cdir / "scripts" / "crop_image.py").is_file()
    segdir = cdir / "work" / "segments" / "2"
    assert not (segdir / "scripts").exists()
    assert not (segdir / "scenes.json").exists()
    assert not (segdir / "voice_lines.json").exists()


def test_run_segment_failure_marks_overall_failed(tmp_path, monkeypatch):
    """任一段失败 → 整体 failed，error 指明段号。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": []}

    def fake_codex(self, workdir, prompt):
        if "work/segments/2/" in prompt:
            raise CodexError("codex exit 3: segment crashed")
        m = re.search(r"work/segments/(\d+)/", prompt)
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "failed"
    assert "segment 2" in m["error"]


def test_run_segments_voice_lines_written_per_segment_including_empty(tmp_path, monkeypatch):
    """口播模式：每段都写 voice_lines.json（无台词的段写空数组）；meta.segments 逐段带 lines。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS, voice_mode="keep")
    cid = meta["id"]
    cdir = settings.data_dir / cid
    lines = [{"text": "第一段。", "start_s": 3.0, "end_s": 3.5}]

    def fake_voice_step(settings, cid, cdir, work, runner, voice_mode, target_language):
        return lines

    calls = {"cmd": []}

    def fake_codex(self, workdir, prompt):
        m = re.search(r"work/segments/(\d+)/", prompt)
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))

    monkeypatch.setattr(pipeline, "_voice_step", fake_voice_step)
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    for n, expected in ((1, lines), (2, []), (3, [])):
        segdir = cdir / "work" / "segments" / str(n)
        assert json.loads((segdir / "voice_lines.json").read_text(encoding="utf-8")) == expected
    assert [s["lines"] for s in m["segments"]] == [["第一段。"], [], []]


def test_run_segments_processed_concurrently(tmp_path, monkeypatch):
    """各段在 ThreadPoolExecutor 里并发处理（codex 调用出现重叠）。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": []}
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_codex(self, workdir, prompt):
        nonlocal active, max_active
        m = re.search(r"work/segments/(\d+)/", prompt)
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.3)
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))
        with lock:
            active -= 1

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert max_active >= 2  # 串行执行时该值为 1


def test_run_scenes_detection_failure_falls_back_to_single_segment(tmp_path, monkeypatch):
    """scenes.py 检测不出场景（如无硬切的单场景视频）→ 回退单段模式并留痕，照常 done。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, [])
    cid = meta["id"]

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            out = Path(argv[argv.index("--out-dir") + 1])
            (out / "contact_sheet.jpg").write_bytes(b"sheet")
            (out / "manifest.json").write_text("{}")
        elif step == "scenes":
            raise pipeline.PipelineError("scenes exit 1: 未检测到任何场景")

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert "segments" not in m
    assert "fallback" in m["scenes_note"]  # 回退留痕（内部字段）


def test_run_scenes_invalid_segments_falls_back_to_single_segment(tmp_path, monkeypatch):
    """scenes.json 的 segments 违反结构不变量（长度/连续性/覆盖）→ 回退单段模式并留痕。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, [])
    cid = meta["id"]
    bad = [
        {"index": 1, "start_s": 0.0, "end_s": 3.0},  # <4s 违规
        {"index": 2, "start_s": 3.0, "end_s": 24.0},
    ]

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            out = Path(argv[argv.index("--out-dir") + 1])
            (out / "contact_sheet.jpg").write_bytes(b"sheet")
            (out / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 24.0, "scenes": [], "segments": bad}),
                encoding="utf-8",
            )

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert "segments" not in m
    assert "invalid" in m["scenes_note"]


def test_run_translate_target_language_in_segment_prompt(tmp_path, monkeypatch):
    """voice_mode=translate：目标语言由后端写进分段 prompt（codex 不从台词反推）。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS, voice_mode="translate")
    storage.update_meta(settings.data_dir, meta["id"], target_language="日语")
    cid = meta["id"]
    calls = {"cmd": [], "codex": []}

    def fake_voice_step(settings, cid, cdir, work, runner, voice_mode, target_language):
        return [{"text": "你好。", "start_s": 3.0, "end_s": 3.5}]

    def fake_codex(self, workdir, prompt):
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        m = re.search(r"work/segments/(\d+)/", prompt)
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))

    monkeypatch.setattr(pipeline, "_voice_step", fake_voice_step)
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert len(calls["codex"]) == 3
    for call in calls["codex"]:
        assert "提示词与台词使用目标语言：日语" in call["prompt"]


def test_run_voice_lines_dropped_recorded(tmp_path, monkeypatch):
    """越界台词（超出末段终点+容差）不归段，但计数落 meta.voice_lines_dropped（内部字段）。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS, voice_mode="keep")
    cid = meta["id"]
    lines = [
        {"text": "第一段。", "start_s": 3.0, "end_s": 3.5},
        {"text": "越界句。", "start_s": 30.0, "end_s": 30.5},
    ]

    def fake_voice_step(settings, cid, cdir, work, runner, voice_mode, target_language):
        return lines

    calls = {"cmd": []}

    def fake_codex(self, workdir, prompt):
        m = re.search(r"work/segments/(\d+)/", prompt)
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))

    monkeypatch.setattr(pipeline, "_voice_step", fake_voice_step)
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["voice_lines_dropped"] == 1
    assert [s["lines"] for s in m["segments"]] == [["第一段。"], [], []]


def test_run_segment_workers_capped_at_half_codex_concurrency(tmp_path, monkeypatch):
    """段并发上限 = codex_concurrency//2：一条长视频不得占满全部 codex 槽饿死其他会话。"""
    settings = make_settings(tmp_path, codex_concurrency=4)
    meta = _make_segment_conversation(settings, SEGMENTS_4)
    cid = meta["id"]
    calls = {"cmd": []}
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_codex(self, workdir, prompt):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.3)
        m = re.search(r"work/segments/(\d+)/", prompt)
        _write_valid_package(Path(workdir) / "work" / "segments" / m.group(1))
        with lock:
            active -= 1

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS_4))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert max_active == 2  # 4 槽 → 2 并发段；不设上限则为 4


def test_run_segment_cut_failure_marks_overall_failed(tmp_path, monkeypatch):
    """ffmpeg 切段非零退出 → 整体 failed，error 指明段号与切段步。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            out = Path(argv[argv.index("--out-dir") + 1])
            (out / "contact_sheet.jpg").write_bytes(b"sheet")
            (out / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 24.0, "scenes": [], "segments": SEGMENTS}),
                encoding="utf-8",
            )
        else:
            raise pipeline.PipelineError(f"{step} exit 1: codec missing")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", lambda self, workdir, prompt: None)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "failed"
    assert "segment 1" in m["error"]
    assert "segment cut" in m["error"]


# ---------- 拆段 e2e：真 subprocess 全链路 + 桩 codex ----------


def _make_scene_video_24s(path: Path) -> Path:
    """3 段 8 秒纯色硬切拼接 + sine 音轨（24s，>20s 触发拆段）。"""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:r=10:d=8",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=10:d=8",
            "-f", "lavfi", "-i", "color=c=green:s=320x240:r=10:d=8",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=24",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]",
            "-map", "[outv]", "-map", "3:a",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def _write_stub_codex_segments(bin_dir: Path, frames: int = 3) -> Path:
    """桩 codex 兼处理两种调用：ASR 写 3 句台词；分段调用按 prompt 里的段号写 work/segments/N/ 产物。"""
    stub = bin_dir / "codex"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""\
            import json, re, shutil, sys
            from pathlib import Path

            argv = sys.argv[1:]
            workdir = Path(argv[argv.index("-C") + 1])
            out = Path(argv[argv.index("-o") + 1])
            prompt = argv[-1]
            if "voice.mp3" in prompt:
                (workdir / "work" / "voice_lines.json").write_text(
                    json.dumps([
                        {{"text": "第一句。", "start_s": 3.0, "end_s": 3.5}},
                        {{"text": "边界句。", "start_s": 8.0, "end_s": 8.5}},
                        {{"text": "第三句。", "start_s": 17.5, "end_s": 18.0}},
                    ]),
                    encoding="utf-8",
                )
                out.write_text("asr done", encoding="utf-8")
                raise SystemExit(0)
            m = re.search(r"work/segments/(\\d+)/", prompt)
            assert m, "segment index missing from prompt"
            segdir = workdir / "work" / "segments" / m.group(1)
            kdir = segdir / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(segdir.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in segment dir"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (segdir / "prompt.txt").write_text("分段桩产物", encoding="utf-8")
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_run_multi_segment_full_pipeline(tmp_path, monkeypatch):
    """真 subprocess 全链路：>20s 多场景视频 → 场景检测拆 3 段 → 并发处理 → meta.segments。"""
    video = _make_scene_video_24s(tmp_path / "scene.mp4")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex_segments(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    shutil.copy(video, settings.data_dir / meta["id"] / "source.mp4")
    storage.update_meta(settings.data_dir, meta["id"], voice_mode="keep")
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    segs = m["segments"]
    assert [s["index"] for s in segs] == [1, 2, 3]
    for seg, (start, end) in zip(segs, [(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)]):
        assert abs(seg["start_s"] - start) < 0.001
        assert abs(seg["end_s"] - end) < 0.001
        assert seg["keyframes"] == ["01.png", "02.png", "03.png"]
        assert seg["prompt"] == pipeline.NO_BGM_LINE + "\n分段桩产物"
    # 顶层 keyframes/prompt 保持空值，不重复写
    assert m["keyframes"] == [] and m["prompt"] is None
    # 句子归属：3.0s→段1；8.0s 恰在边界→段2；17.5s→段3
    assert [s["lines"] for s in segs] == [["第一句。"], ["边界句。"], ["第三句。"]]
    assert [l["text"] for l in m["voice_lines"]] == ["第一句。", "边界句。", "第三句。"]

    for n, texts in ((1, ["第一句。"]), (2, ["边界句。"]), (3, ["第三句。"])):
        segdir = cdir / "work" / "segments" / str(n)
        lines = json.loads((segdir / "voice_lines.json").read_text(encoding="utf-8"))
        assert [l["text"] for l in lines] == texts
        assert (segdir / "prompt.txt").read_text(encoding="utf-8") == (
            pipeline.NO_BGM_LINE + "\n分段桩产物"
        )
        assert (segdir / "manifest.json").is_file()  # 该段抽帧产物
        duration = pipeline._probe_duration(segdir / "source.mp4")
        assert abs(duration - 8.0) < pipeline.CUT_DURATION_TOLERANCE_S  # 切段时长准确
    assert not (cdir / "work" / "keyframes").exists()  # 多段模式不产出顶层 keyframes
    assert (cdir / "scripts" / "crop_image.py").is_file()  # scripts 拷进会话目录（一份）
    assert (cdir / "work" / "scenes.json").is_file()  # 全片场景清单留在 work/
    assert not (cdir / "work" / "segments" / "2" / "scenes.json").exists()
