"""T4 拆段流水线：scenes 接入、拆段并发、句子归属、切段精度、detail 16 字段契约。"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import long_video, pipeline, storage, vocal
from app.codex_runner import CodexError, CodexRunner
from app.main import create_app


@pytest.fixture(autouse=True)
def _stub_image_postprocess_codex(monkeypatch):
    def generate(_runner, segments, mode, **_kwargs):
        indices = [segment["index"] for segment in segments]
        continuity = None if indices == [0] else {
            "version": 1,
            "segment_indices": indices,
            "elements": [],
        }
        return continuity, {
            index: (
                f"Codex 生成的 {mode} 图片二次编辑提示词"
                + (f"-{index}" if indices != [0] else "")
            )
            for index in indices
        }

    monkeypatch.setattr(
        pipeline.image_optimization, "generate_project_prompts", generate
    )

# 1×1 真实 PNG（validate_work_dir 会用 cv2 解码校验）
_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_ANCHOR_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAkAAAAQCAIAAABLKsIUAAAAMklEQVQYGXXBAQEAAAABIJZ33QJV5ChyFDmKHEWOIkeRo8hR5ChyFDmKHEWOIkeRo8gxBaoX4WKqZ68AAAAASUVORK5CYII="
)


def _solid_png(value: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((240, 320, 3), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


_SOURCE_FIRST_PNG = _solid_png(40)
_SOURCE_LAST_PNG = _solid_png(80)

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


def _write_valid_package(work: Path, frames: int = 9, prompt: str = "分段桩产物"):
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


def test_sub_four_second_input_reaches_real_failed_terminal_before_extraction(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "short", "source.mp4")
    cid = meta["id"]
    (settings.data_dir / cid / "source.mp4").write_bytes(b"source")
    storage.update_meta(
        settings.data_dir,
        cid,
        dialogue_mode="none",
        voice_mode="none",
        duration_s=2.5,
    )
    monkeypatch.setattr(
        storage,
        "probe_video",
        lambda _path: storage.VideoProbe(2.5, 320, 240),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_cmd",
        lambda *_args, **_kwargs: pytest.fail(
            "sub-four-second input must fail before extraction"
        ),
    )

    pipeline.run(settings, cid, None)

    stored = storage.load_meta(settings.data_dir, cid)
    assert stored["status"] == "failed"
    assert stored["error"] == "long_video_duration_below_provider_minimum"
    assert (
        settings.data_dir / cid / "work" / "errors" / "pipeline.json"
    ).is_file()


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


# ---------- 后端前缀（仅多段 BGM 行，不依赖 codex 写） ----------


def test_prefix_multi_segment_bgm_only(tmp_path):
    """多段模式只加 BGM 行，不再注入人脸动作 workaround。"""
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("正文", encoding="utf-8")
    got = pipeline._apply_no_bgm_prefix("正文", prompt_path, enabled=True)
    assert got == pipeline.NO_BGM_LINE + "\n正文"
    assert prompt_path.read_text(encoding="utf-8") == got


def test_prefix_single_segment_is_identity_and_preserves_person_semantics(tmp_path):
    """单段模式无前缀，原 prompt 中普通人物语义逐字保留。"""
    prompt_path = tmp_path / "prompt.txt"
    original = "普通人物抬手展示产品，随后自然放下。"
    got = pipeline._apply_no_bgm_prefix(original, prompt_path, enabled=False)
    assert got == original
    assert prompt_path.read_text(encoding="utf-8") == got


def test_prefix_rejects_oversize(tmp_path):
    """BGM 前缀后超 MAX_PROMPT_BYTES → PipelineError。"""
    prompt_path = tmp_path / "prompt.txt"
    big = "x" * (pipeline.MAX_PROMPT_BYTES - 1)  # 加前缀必然超限
    with pytest.raises(pipeline.PipelineError, match="prefix"):
        pipeline._apply_no_bgm_prefix(big, prompt_path, enabled=True)


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


# ---------- detail 16 字段契约 ----------


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
    return storage.update_meta(settings.data_dir, meta["id"], voice_mode=voice_mode)


def _fake_cmd_segments(calls, segments):
    """假 _run_cmd：extract 写空 manifest；scenes 写给定 segments 的 scenes.json。"""

    def fake_cmd(argv, *, timeout, step, cwd=None):
        calls["cmd"].append({"argv": list(argv), "timeout": timeout, "step": step, "cwd": cwd})
        if step == "extract":
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "contact_sheet.jpg").write_bytes(b"sheet")
            (out / "manifest.json").write_text(
                json.dumps({"duration_seconds": 24.0})
            )
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


def test_new_input_long_video_keeps_segments_and_writes_bound_plan_receipt(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path, codex_concurrency=2)
    meta = _make_segment_conversation(settings, [])
    cid = meta["id"]
    cdir = settings.data_dir / cid
    storage.update_meta(
        settings.data_dir,
        cid,
        duration_s=24.0,
        dialogue_mode="auto",
        voice_mode="keep",
    )
    line = {"text": "局部台词", "start_s": 16.5, "end_s": 17.5}
    calls = {"cmd": [], "codex": []}
    image_calls = []
    visual_call_count = 0

    def fake_voice_step(*args, **kwargs):
        storage.update_meta(
            settings.data_dir,
            cid,
            voice_line_provenance=[{
                **line,
                "classification": "spoken",
                "provenance": "asr",
                "kept": True,
            }],
        )
        return [line]

    def fake_codex(self, workdir, prompt):
        nonlocal visual_call_count
        visual_call_count += 1
        calls["codex"].append(Path(workdir))
        staged_work = Path(workdir) / "work"
        keyframes = staged_work / "keyframes"
        keyframes.mkdir(parents=True, exist_ok=True)
        for order, source_frame in enumerate(
            sorted(staged_work.glob("*_frame_*.png")), 1
        ):
            shutil.copy(source_frame, keyframes / f"{order:02d}.png")
        (staged_work / "prompt.txt").write_text(
            f"局部动作-{visual_call_count}", encoding="utf-8"
        )

    def fake_isolated_until_output(
        self, workdir, prompt, *, session_dir, output_path,
        max_output_bytes, validate_output,
    ):
        raw = json.dumps({
            "people": {},
            "entities": {},
            "scenes": {},
            "relations": {},
        }).encode("utf-8")
        assert len(raw) <= max_output_bytes
        return validate_output(raw)

    def fake_image_project(
        _runner, segments, mode, *, session_dir, expected_version,
        element_index_path, skill_bytes, phase_retry_count,
        phase_retry_interval_s,
    ):
        assert expected_version == 4
        assert Path(element_index_path).is_file()
        assert skill_bytes
        indices = [segment["index"] for segment in segments]
        image_calls.append((Path(session_dir), indices))
        return {
            "version": 1,
            "segment_indices": indices,
            "elements": [],
        }, {
            index: f"图片二次编辑提示词-{index}-{mode}"
            for index in indices
        }

    base_fake_cmd = _fake_cmd_segments(calls, SEGMENTS)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        base_fake_cmd(argv, timeout=timeout, step=step, cwd=cwd)
        if step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps(
                    {
                        "duration_s": 24.0,
                        "scenes": [
                            {"index": seg["index"], "start_s": seg["start_s"],
                             "end_s": seg["end_s"], "frames": []}
                            for seg in SEGMENTS
                        ],
                        "segments": SEGMENTS,
                    }
                ),
                encoding="utf-8",
            )
        elif step.startswith("segment ") and step.endswith(" extract"):
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            frame_data = [
                _SOURCE_FIRST_PNG,
                *(_solid_png(value) for value in range(45, 80, 5)),
                _SOURCE_LAST_PNG,
            ]
            frame_times = [round(7.75 * index / 8, 3) for index in range(9)]
            frame_names = [
                f"{index:03d}_frame_{time_s:07.3f}s.png"
                for index, time_s in enumerate(frame_times, 1)
            ]
            for name, data in zip(frame_names, frame_data):
                (out / name).write_bytes(data)
            (out / "manifest.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "index": index,
                                "time_seconds": time_s,
                                "file": name,
                            }
                            for index, (time_s, name) in enumerate(
                                zip(frame_times, frame_names), 1
                            )
                        ]
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(pipeline, "_voice_step", fake_voice_step)
    monkeypatch.setattr(
        storage, "probe_video", lambda _path: storage.VideoProbe(24.0, 320, 240)
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        CodexRunner, "run_isolated_until_output", fake_isolated_until_output,
    )
    monkeypatch.setattr(
        pipeline.image_optimization,
        "generate_project_prompts",
        fake_image_project,
    )

    pipeline.run(settings, cid, CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, cid)
    assert stored["status"] == "done", stored["error"]
    assert stored["_image_continuity"]["segment_indices"] == [1, 2, 3]
    assert (stored["aspect_ratio"], stored["resolution"], stored["fit_mode"]) == (
        "16:9", "480p", "crop"
    )
    assert stored["fit_profiles"] == {
        "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        "9:16": {"fit_required": True, "default_fit_mode": "crop"},
    }
    assert [(s["start_s"], s["end_s"]) for s in stored["segments"]] == [
        (0.0, 8.0),
        (8.0, 16.0),
        (16.0, 24.0),
    ]
    assert [s["chain_id"] for s in stored["segments"]] == [
        "chain-001",
        "chain-002",
        "chain-003",
    ]
    assert image_calls == [(cdir, [1, 2, 3])]
    frozen_prompts = stored["_image_optimization"]["segments"]
    assert [item["current"] for item in frozen_prompts] == [
        f"图片二次编辑提示词-{index}-independent_parallel"
        for index in (1, 2, 3)
    ]
    assert all(
        item["current"] != stored["segments"][item["segment_index"] - 1]["prompt"]
        for item in frozen_prompts
    )
    assert [s["join_mode"] for s in stored["segments"]] == ["hard_cut"] * 3
    assert stored["segments"][2]["dialogue"] == [
        {
            "text": "局部台词", "start_s": 0.5, "end_s": 1.5,
            "classification": "spoken",
        }
    ]
    assert stored["segments"][0]["source"] == "segments/1/source.mp4"
    assert stored["segments"][0]["keyframe_paths"][0].startswith(
        "segments/1/work/keyframes/"
    )
    assert stored["segments"][0]["first_frame_path"] == (
        "segments/1/work/anchors/first.png"
    )
    assert stored["segments"][0]["last_frame_path"] == (
        "segments/1/work/anchors/last.png"
    )
    continuity = long_video.build_continuity_block()
    for segment in stored["segments"]:
        assert continuity in segment["prompt"]
        assert f"局部动作-{segment['index']}" in segment["prompt"]
    assert stored["long_video_plan_receipt"] == long_video.PLAN_RECEIPT_FILENAME
    receipt = json.loads((cdir / long_video.PLAN_RECEIPT_FILENAME).read_text(encoding="utf-8"))
    assert receipt["source"]["sha256"]
    assert receipt["workflow"] == "minimax_h3_lightx2v_v5_15s"
    assert [anchor["role"] for anchor in receipt["segments"][0]["anchors"]] == [
        "first",
        "end",
    ]
    assert receipt["segments"][0]["anchors"][0]["sha256"] == hashlib.sha256(
        _SOURCE_FIRST_PNG
    ).hexdigest()
    assert receipt["segments"][0]["anchors"][1]["sha256"] == hashlib.sha256(
        _SOURCE_LAST_PNG
    ).hexdigest()
    assert receipt["segments"][0]["anchors"][0]["sha256"] == (
        receipt["segments"][0]["keyframes"][0]["sha256"]
    )
    assert [s["chain_id"] for s in receipt["segments"]] == [
        "chain-001",
        "chain-002",
        "chain-003",
    ]


def test_new_input_over_300_seconds_fails_before_any_pipeline_operation(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, [])
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        duration_s=300.001,
        dialogue_mode="auto",
        voice_mode="keep",
    )
    calls = []

    def must_not_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("pipeline operation ran for oversized input")

    monkeypatch.setattr(pipeline, "_run_cmd", must_not_run)
    monkeypatch.setattr(
        storage,
        "probe_video",
        lambda _path: storage.VideoProbe(300.001, 320, 240),
    )
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "long_video_duration_exceeded"
    assert calls == []


def test_run_single_segment_has_no_prefix_and_no_segments_key(tmp_path, monkeypatch):
    """旧 scenes 契约显式返回空 segments：prompt 原样保留、meta 不写 segments。"""
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
    assert m["prompt"] == "分段桩产物"
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
    """分段 visual Codex 在一次性 /tmp stage 中运行；prompt 与单段逐字相同，
    stage 内仍使用 SKILL.md 的 work/ 路径，且不暴露「分段模式」/ scenes.json。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    cdir = settings.data_dir / cid
    calls = {"cmd": [], "codex": []}

    def fake_codex(self, workdir, prompt):
        stage = Path(workdir)
        calls["codex"].append({
            "workdir": stage,
            "prompt": prompt,
            "has_scenes": (stage / "work" / "scenes.json").exists(),
            "has_voice_lines": (stage / "work" / "voice_lines.json").exists(),
        })
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert len(calls["codex"]) == 3
    assert all(
        call["workdir"].parent == Path("/tmp")
        and call["workdir"].name.startswith("duet-visual-")
        for call in calls["codex"]
    )
    assert all(not call["has_scenes"] for call in calls["codex"])
    assert all(not call["has_voice_lines"] for call in calls["codex"])
    prompt = calls["codex"][0]["prompt"]
    assert prompt == pipeline._codex_prompt(cdir / "work" / "segments" / "2", "")
    for needle in (
        "SKILL.md",
        "输入在 work/",
        sys.executable,
        "当前隔离目录",
        "禁止联网",
        "环境变量",
    ):
        assert needle in prompt, needle
    for banned in ("分段模式", "scenes.json"):
        assert banned not in prompt
    assert "voice.mp3" not in prompt
    # 持久段目录保留 scripts，但不写 scenes.json 或无口播台词。
    for n in (1, 2, 3):
        segdir = cdir / "work" / "segments" / str(n)
        assert (segdir / "scripts" / "crop_image.py").is_file()
        assert not (segdir / "scenes.json").exists()
        assert not (segdir / "work" / "scenes.json").exists()
    segdir = cdir / "work" / "segments" / "2"
    assert not (segdir / "work" / "voice_lines.json").exists()


def test_run_segment_failure_marks_overall_failed(tmp_path, monkeypatch):
    """任一段失败 → 整体 failed，error 指明段号。"""
    settings = make_settings(tmp_path, codex_concurrency=2)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": []}

    call_count = 0

    def fake_codex(self, workdir, prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise CodexError("codex exit 3: segment crashed")
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "failed"
    assert m["error"] == "pipeline_failed"


def test_run_segment_zero_exit_without_outputs_reports_codex_stage(tmp_path, monkeypatch):
    """段 Codex 返回 0 但未填充预建帧位时，解码校验精确指向首个空帧。"""
    settings = make_settings(tmp_path, codex_concurrency=2)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": []}

    call_count = 0

    def fake_codex(self, workdir, prompt):
        nonlocal call_count
        call_count += 1
        if call_count != 2:
            _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "failed"
    assert m["error"] == "pipeline_failed"


def test_run_segment_codex_error_salvages_complete_outputs(tmp_path, monkeypatch):
    """段 Codex 报错但完整产物已落盘时，逐段收养且整体成功。"""
    settings = make_settings(tmp_path)
    meta = _make_segment_conversation(settings, SEGMENTS)
    cid = meta["id"]
    calls = {"cmd": []}

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")
        raise CodexError("codex timed out after 600s")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert len(m["segments"]) == len(SEGMENTS)


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
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_voice_step", fake_voice_step)
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_cmd_segments(calls, SEGMENTS))
    monkeypatch.setattr(pipeline, "_cut_segment", _fake_cut)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, cid, CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    for n, expected in ((1, lines), (2, []), (3, [])):
        segdir = cdir / "work" / "segments" / str(n)
        assert json.loads(
            (segdir / "work" / "voice_lines.json").read_text(encoding="utf-8")
        ) == expected
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
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.3)
        _write_valid_package(Path(workdir) / "work")
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
        _write_valid_package(Path(workdir) / "work")

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
        _write_valid_package(Path(workdir) / "work")

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
        _write_valid_package(Path(workdir) / "work")
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
    assert m["error"] == "pipeline_failed"


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


def _write_stub_codex_segments(bin_dir: Path, frames: int = 9) -> Path:
    """桩 codex 兼处理两种调用：ASR 写 3 句台词；视觉阶段在隔离 cwd 中
    按 SKILL.md 字面路径写 work/ 下的 9 帧和 prompt。"""
    stub = bin_dir / "codex"
    stub.write_text(
        "#!/usr/bin/python3\n"
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
            # 分段调用：cwd 即段目录；按 SKILL.md 字面执行——输入/产物都在 workdir/work/
            wdir = workdir / "work"
            kdir = wdir / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(wdir.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in segment dir"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (wdir / "prompt.txt").write_text("分段桩产物", encoding="utf-8")
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


@pytest.fixture
def segment_voice_stub_bin():
    """voice 外层 bwrap 会隐藏 /tmp 和仓库，桩程序放在仍可见的位置。"""
    with tempfile.TemporaryDirectory(prefix="duet-segment-stub-", dir="/var/tmp") as raw:
        yield Path(raw)


def test_run_multi_segment_full_pipeline(tmp_path, monkeypatch, segment_voice_stub_bin):
    """真 subprocess 全链路：>20s 多场景视频 → 场景检测拆 3 段 → 并发处理 → meta.segments。

    桩 codex 按 SKILL.md 字面执行（输入/产物均 workdir/work/，与单段 codex 逐字相同）→
    白名单校验通过；这是「段目录嵌套 work/，SKILL.md 逐字适用」的 e2e 形状用例。"""
    video = _make_scene_video_24s(tmp_path / "scene.mp4")
    bin_dir = segment_voice_stub_bin
    _write_stub_codex_segments(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    shutil.copy(video, settings.data_dir / meta["id"] / "source.mp4")
    storage.update_meta(settings.data_dir, meta["id"], voice_mode="keep")
    cid = meta["id"]
    cdir = settings.data_dir / cid

    # 桩台词不是音轨里的真口播（音轨是 sine），真模型会全滤掉；本测试验证的是
    # 台词按 start_s 归段，声学过滤由 test_vocal.py 与 test_pipeline.py 覆盖。
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 24_000, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )
    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    segs = m["segments"]
    assert [s["index"] for s in segs] == [1, 2, 3]
    for seg, (start, end) in zip(segs, [(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)]):
        assert abs(seg["start_s"] - start) < 0.001
        assert abs(seg["end_s"] - end) < 0.001
        assert seg["keyframes"] == [f"{index:02d}.png" for index in range(1, 10)]
        assert seg["prompt"] == pipeline.NO_BGM_LINE + "\n分段桩产物"
    # 顶层 keyframes/prompt 保持空值，不重复写
    assert m["keyframes"] == [] and m["prompt"] is None
    # 句子归属：3.0s→段1；8.0s 恰在边界→段2；17.5s→段3
    assert [s["lines"] for s in segs] == [["第一句。"], ["边界句。"], ["第三句。"]]
    assert [l["text"] for l in m["voice_lines"]] == ["第一句。", "边界句。", "第三句。"]

    for n, texts in ((1, ["第一句。"]), (2, ["边界句。"]), (3, ["第三句。"])):
        segdir = cdir / "work" / "segments" / str(n)
        lines = json.loads(
            (segdir / "work" / "voice_lines.json").read_text(encoding="utf-8")
        )
        assert [l["text"] for l in lines] == texts
        assert (segdir / "work" / "prompt.txt").read_text(encoding="utf-8") == (
            pipeline.NO_BGM_LINE + "\n分段桩产物"
        )
        assert (segdir / "work" / "manifest.json").is_file()  # 该段抽帧产物
        duration = pipeline._probe_duration(segdir / "source.mp4")
        assert abs(duration - 8.0) < pipeline.CUT_DURATION_TOLERANCE_S  # 切段时长准确
    assert not (cdir / "work" / "keyframes").exists()  # 多段模式不产出顶层 keyframes
    assert (cdir / "work" / "scenes.json").is_file()  # 全片场景清单留在 work/
    for n in (1, 2, 3):
        segdir = cdir / "work" / "segments" / str(n)
        assert (segdir / "scripts" / "crop_image.py").is_file()  # scripts 逐段拷入（段目录根）
        assert not (segdir / "codex_last_message.txt").exists()
        assert not (segdir / "scenes.json").exists()  # scenes.json 不拷入段目录
        assert not (segdir / "work" / "scenes.json").exists()
