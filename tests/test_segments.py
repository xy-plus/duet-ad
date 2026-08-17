"""T4 拆段流水线：scenes 接入、拆段并发、句子归属、切段精度、detail 14 字段契约。"""

import base64
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

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
    """分段 codex 调用：cwd=段目录；prompt 含 SKILL.md、分段模式、scenes.json、该段台词与硬性禁令。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": [], "codex": []}

    def fake_codex(self, workdir, prompt):
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        _write_valid_package(Path(workdir))  # 分段模式产物在段目录（cwd）内

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert len(calls["codex"]) == 3
    # 段间并发，调用顺序不定：按 workdir 取第二段的调用
    by_workdir = {c["workdir"]: c for c in calls["codex"]}
    call = by_workdir[settings.data_dir / cid / "work" / "segments" / "2"]
    prompt = call["prompt"]
    for needle in (
        str(pipeline.SKILL_MD),
        "分段模式",
        "work/segments/2/",
        "scenes.json",
        "voice_lines.json",
        sys.executable,
        "禁止联网",
        "环境变量",
    ):
        assert needle in prompt, needle
    assert "voice.mp3" not in prompt


def test_run_segment_failure_marks_overall_failed(tmp_path, monkeypatch):
    """任一段失败 → 整体 failed，error 指明段号。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": []}

    def fake_codex(self, workdir, prompt):
        if str(workdir).endswith(f"segments{os.sep}2"):
            raise CodexError("codex exit 3: segment crashed")
        _write_valid_package(Path(workdir))

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "failed"
    assert "segment 2" in m["error"]


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
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.3)
        _write_valid_package(Path(workdir))
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
    """scenes.py 检测不出场景（如无硬切的单场景视频）→ 回退单段模式，照常 done。"""
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
    """桩 codex 兼处理三种调用：ASR 写 3 句台词；分段调用在 cwd（段目录）内产产物。"""
    stub = bin_dir / "codex"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""\
            import json, shutil, sys
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
            kdir = workdir / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(workdir.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in segment dir"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (workdir / "prompt.txt").write_text("分段桩产物", encoding="utf-8")
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
        assert (segdir / "scenes.json").is_file()  # 全片场景清单副本
        assert (segdir / "scripts" / "crop_image.py").is_file()
        assert (segdir / "manifest.json").is_file()  # 该段抽帧产物
        duration = pipeline._probe_duration(segdir / "source.mp4")
        assert abs(duration - 8.0) < pipeline.CUT_DURATION_TOLERANCE_S  # 切段时长准确
    assert not (cdir / "work" / "keyframes").exists()  # 多段模式不产出顶层 keyframes
