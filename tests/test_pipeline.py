"""任务 B：处理流水线（extract --fps 4 → codex 沙箱 → 白名单校验 → meta 落盘）。"""
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import codex_runner, long_generation, long_video, pipeline, prepared_input, storage, vocal, voice
from app.codex_runner import CodexError, CodexRunner
from app.main import create_app

ROOT = Path(pipeline.__file__).resolve().parent.parent
EXTRACT_SCRIPT = ROOT / "skills" / "video-maker" / "scripts" / "extract_keyframes.py"
CROP_SCRIPT = ROOT / "skills" / "video-maker" / "scripts" / "crop_image.py"

PROMPT_TEXT = "生成一支 15 秒、9:16 竖屏、720p、写实手机实拍风格的清洁短视频。"

# 1×1 真实 PNG（validate_work_dir 会用 cv2 解码校验，占位字节过不了）
_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _write_valid_package(work: Path, frames: int = 3, prompt: str = PROMPT_TEXT):
    """按约定文件名造一套合法产物，返回关键帧文件名列表。"""
    kdir = work / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(1, frames + 1):
        name = f"{i:02d}.png"
        (kdir / name).write_bytes(_PX_PNG)
        names.append(name)
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    return names


def _make_conversation(settings, video_1s):
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    shutil.copy(video_1s, settings.data_dir / meta["id"] / "source.mp4")
    # 本文件既有用例模拟旧 voice_mode 会话；新 prepared-input 用例会显式补
    # dialogue_mode + duration_s + 新 voice_mode。
    return storage.update_meta(settings.data_dir, meta["id"], voice_mode="none")


def test_long_scene_bounds_normalize_detector_millisecond_rounding(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps(
            {
                "duration_s": 36.733,
                "scenes": [
                    {"start_s": 0.0, "end_s": 20.467},
                    {"start_s": 20.467, "end_s": 36.733},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert pipeline._scene_bounds_for_long_plan(work, 36.733333) == [
        {"start_s": 0.0, "end_s": 20.467},
        {"start_s": 20.467, "end_s": 36.733333},
    ]


def test_long_scene_bounds_include_the_one_millisecond_limit(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps({"scenes": [{"start_s": 0.0, "end_s": 36.733}]}),
        encoding="utf-8",
    )

    assert pipeline._scene_bounds_for_long_plan(work, 36.734) == [
        {"start_s": 0.0, "end_s": 36.734}
    ]


def test_long_scene_bounds_reject_just_over_one_millisecond(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps({"scenes": [{"start_s": 0.0, "end_s": 36.733}]}),
        encoding="utf-8",
    )

    bounds = pipeline._scene_bounds_for_long_plan(work, 36.734001)
    with pytest.raises(long_video.LongVideoError) as caught:
        long_video.plan_segments(36.734001, bounds, [])
    assert caught.value.code == "long_video_invalid_scenes"


def test_long_scene_bounds_do_not_hide_a_real_terminal_gap(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {"start_s": 0.0, "end_s": 20.467},
                    {"start_s": 20.467, "end_s": 36.7},
                ],
            }
        ),
        encoding="utf-8",
    )

    bounds = pipeline._scene_bounds_for_long_plan(work, 36.733333)
    with pytest.raises(long_video.LongVideoError) as caught:
        long_video.plan_segments(36.733333, bounds, [])
    assert caught.value.code == "long_video_invalid_scenes"


def test_run_converges_container_duration_to_visual_manifest_timeline(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    storage.update_meta(
        settings.data_dir, meta["id"], duration_s=16.787007,
        dialogue_mode="auto", voice_mode="keep",
    )
    captured = {}

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 16.766667}), encoding="utf-8"
            )
            (work / "contact_sheet.jpg").write_bytes(b"sheet")

    def fake_write(root, *, source, duration_s, segments, workflow):
        captured["duration_s"] = duration_s
        captured["segments"] = segments
        path = root / long_video.PLAN_RECEIPT_FILENAME
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: storage.VideoProbe(16.766667, 1080, 1920),
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(pipeline, "_voice_step", lambda *_a, **_kw: [])
    monkeypatch.setattr(pipeline, "_detect_segments", lambda *_a, **_kw: [])
    (cdir / "work" / "scenes.json").write_text(
        json.dumps({
            "duration_s": 16.767,
            "scenes": [{"start_s": 0.0, "end_s": 16.767}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline, "_process_segment",
        lambda _settings, _work, _source, seg, _runner, lines, _lang,
               new_input_contract: {
            **seg, "keyframes": [], "prompt": "p", "dialogue": lines or [],
        },
    )
    monkeypatch.setattr(long_video, "write_plan_receipt", fake_write)

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["duration_s"] == 16.766667
    assert captured["duration_s"] == 16.766667
    assert captured["segments"][-1]["end_s"] == 16.766667


def test_run_does_not_reprobe_or_rewrite_frozen_generation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    original_generation = {"status": "succeeded", "attempt": 1}
    storage.update_meta(
        settings.data_dir, meta["id"], status="done", duration_s=9.5,
        generation=original_generation,
    )
    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: (_ for _ in ()).throw(AssertionError("must stay frozen")),
    )

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["duration_s"] == 9.5
    assert stored["generation"] == original_generation


def test_run_rechecks_300_second_gate_after_manifest(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    storage.update_meta(settings.data_dir, meta["id"], duration_s=300.0)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        assert step == "extract"
        work = Path(argv[argv.index("--out-dir") + 1])
        (work / "manifest.json").write_text(
            json.dumps({"duration_seconds": 300.001}), encoding="utf-8"
        )

    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: storage.VideoProbe(300.0, 320, 240),
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "long_video_duration_exceeded"


@pytest.fixture
def fake_steps(monkeypatch):
    """mock 掉 extract 子进程与 codex；返回调用记录。"""
    calls = {"cmd": [], "codex": []}

    def fake_cmd(argv, *, timeout, step, cwd=None):
        calls["cmd"].append({"argv": list(argv), "timeout": timeout, "step": step, "cwd": cwd})
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 1.0})
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def fake_codex(self, workdir, prompt):
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    return calls


@pytest.fixture(autouse=True)
def fake_vocal_analysis(monkeypatch):
    """旧口播流水线测试只验证编排；声学算法由 test_vocal.py 和专门集成测试覆盖。"""
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 1000, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )


# ---------- 产物白名单校验 ----------


class TestValidateWorkDir:
    def test_valid(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        names = _write_valid_package(work, frames=3)
        got_names, prompt = pipeline.validate_work_dir(work)
        assert got_names == names
        assert prompt == PROMPT_TEXT

    def test_zero_keyframes(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work, frames=0)
        with pytest.raises(pipeline.PipelineError, match="keyframe"):
            pipeline.validate_work_dir(work)

    def test_ten_keyframes(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work, frames=10)
        with pytest.raises(pipeline.PipelineError, match="keyframe"):
            pipeline.validate_work_dir(work)

    def test_prompt_missing(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        (work / "prompt.txt").unlink()
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_prompt_empty(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work, prompt="  \n ")
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_prompt_too_large(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        (work / "prompt.txt").write_bytes(b"x" * (32 * 1024 + 1))
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_non_png_not_counted(self, tmp_path):
        """keyframes/ 里的非 PNG 文件不计入帧数。"""
        work = tmp_path / "work"
        work.mkdir()
        names = _write_valid_package(work, frames=2)
        (work / "keyframes" / "notes.txt").write_text("x", encoding="utf-8")
        got_names, _ = pipeline.validate_work_dir(work)
        assert got_names == names


# ---------- _run_cmd：子进程包装 ----------


class TestRunCmd:
    def test_timeout(self, monkeypatch):
        def slow(argv, **kw):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kw["timeout"])

        monkeypatch.setattr(pipeline.subprocess, "run", slow)
        with pytest.raises(pipeline.PipelineError, match="timed out"):
            pipeline._run_cmd(["whatever"], timeout=1, step="extract")

    def test_missing_executable(self, monkeypatch):
        def nope(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        monkeypatch.setattr(pipeline.subprocess, "run", nope)
        with pytest.raises(pipeline.PipelineError, match="not found"):
            pipeline._run_cmd(["no-such-bin"], timeout=1, step="extract")

    def test_stderr_scrubbed_and_truncated(self, monkeypatch):
        stderr = (
            "PATH=/home/xy/.local/bin:/usr/bin\n"
            "ARK_API_KEY=supersecretvalue\n"
            + "y" * 1200 + "\nreal error line\n"
        )
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
        monkeypatch.setattr(
            pipeline.subprocess, "run", lambda *a, **kw: fake
        )
        with pytest.raises(pipeline.PipelineError) as exc_info:
            pipeline._run_cmd(["x"], timeout=1, step="extract")
        msg = str(exc_info.value)
        assert "real error line" in msg
        assert "ARK_API_KEY" not in msg and "supersecretvalue" not in msg
        assert "PATH=" not in msg
        assert len(msg) <= 560  # 500 截断 + 步骤/退出码前缀


# ---------- CodexRunner ----------


@pytest.fixture
def captured_codex(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append({"argv": list(argv), "kw": kw})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    return calls


class TestCodexRunner:
    def test_argv_sandbox(self, captured_codex, tmp_path):
        runner = CodexRunner(timeout_s=600, concurrency=1)
        runner.run(tmp_path, "提示词")
        (call,) = captured_codex
        argv, kw = call["argv"], call["kw"]

        assert argv[:2] == ["codex", "exec"]
        assert argv[argv.index("-C") + 1] == str(tmp_path)
        assert argv[argv.index("-s") + 1] == "workspace-write"
        assert "use_legacy_landlock" not in argv
        assert "--skip-git-repo-check" in argv
        assert "--ephemeral" in argv
        assert argv[argv.index("--color") + 1] == "never"
        assert argv[argv.index("-o") + 1] == str(tmp_path / "codex_last_message.txt")
        configs = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
        assert 'model_reasoning_effort="medium"' in configs
        assert "sandbox_workspace_write.network_access=false" in configs
        assert 'shell_environment_policy.inherit="core"' in configs
        assert any(
            c.startswith("shell_environment_policy.exclude=")
            and "*KEY*" in c and "*TOKEN*" in c and "*SECRET*" in c and "*PASSWORD*" in c
            for c in configs
        )
        assert not any("dangerously-bypass" in a for a in argv)
        assert argv[-1] == "提示词"
        assert kw["timeout"] == 600
        assert kw.get("shell") is not True
        assert kw["capture_output"] is True and kw["text"] is True

    def test_env_scrubbed(self, captured_codex, monkeypatch, tmp_path):
        """调起 codex 的进程环境不得携带秘密变量；PATH/HOME 保留。"""
        monkeypatch.setenv("ARK_API_KEY", "topsecret")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "topsecret")
        monkeypatch.setenv("MY_DB_PASSWORD", "topsecret")
        monkeypatch.setenv("SAFE_VAR", "ok")
        CodexRunner(timeout_s=1, concurrency=1).run(tmp_path, "p")
        env = captured_codex[0]["kw"]["env"]
        assert env is not None
        for key in env:
            assert not re.search(r"KEY|TOKEN|SECRET|PASSWORD", key, re.IGNORECASE), key
        assert env["SAFE_VAR"] == "ok"
        assert "PATH" in env and "HOME" in env

    def test_timeout(self, monkeypatch):
        def slow(argv, **kw):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kw["timeout"])

        monkeypatch.setattr(codex_runner.subprocess, "run", slow)
        with pytest.raises(CodexError, match="timed out") as caught:
            CodexRunner(timeout_s=7, concurrency=1).run(Path("/wd"), "p")
        assert caught.value.retryable is True

    def test_nonzero_stderr_scrubbed(self, monkeypatch):
        stderr = (
            "PATH=/usr/bin\nAWS_SECRET_ACCESS_KEY=abc123\n" + "z" * 1200 + "\nreal codex failure\n"
        )
        fake = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr=stderr)
        monkeypatch.setattr(codex_runner.subprocess, "run", lambda *a, **kw: fake)
        with pytest.raises(CodexError) as exc_info:
            CodexRunner(timeout_s=1, concurrency=1).run(Path("/wd"), "p")
        msg = str(exc_info.value)
        assert "real codex failure" in msg
        assert "abc123" not in msg and "AWS_SECRET_ACCESS_KEY" not in msg
        assert len(msg) <= 560
        assert exc_info.value.retryable is True

    def test_missing_codex_binary(self, monkeypatch):
        def nope(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        monkeypatch.setattr(codex_runner.subprocess, "run", nope)
        with pytest.raises(CodexError, match="codex") as caught:
            CodexRunner(timeout_s=1, concurrency=1).run(Path("/wd"), "p")
        assert caught.value.retryable is False

    def test_concurrency_serialized(self, monkeypatch):
        runner = CodexRunner(timeout_s=30, concurrency=1)
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_run(argv, **kw):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
        threads = [
            threading.Thread(target=runner.run, args=(Path("/wd"), "p")) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert max_active == 1

    def test_voice_run_stages_only_audio_inputs_behind_outer_bwrap(
        self, monkeypatch, tmp_path
    ):
        """音频 agent 的可见输入只有 voice.mp3 与最小 manifest；输出不直接落主 work。"""
        cdir = tmp_path / "conversation"
        work = cdir / "work"
        frames = work / "frames"
        frames.mkdir(parents=True)
        (cdir / "source.mp4").write_text("SOURCE_VISUAL_SECRET", encoding="utf-8")
        (frames / "000001.png").write_text("FRAME_VISUAL_SECRET", encoding="utf-8")
        (work / "contact_sheet.jpg").write_text("OCR_FAKE_DIALOGUE", encoding="utf-8")
        (work / "visual_prompt.txt").write_text("OCR_FAKE_DIALOGUE", encoding="utf-8")
        (work / "voice.mp3").write_bytes(b"audio-only")
        (work / "manifest.json").write_text(
            json.dumps({"duration_seconds": 99, "frames": ["000001.png"]}),
            encoding="utf-8",
        )

        calls = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs))
            stage = Path(argv[argv.index("-C") + 1])
            visible = {
                path.relative_to(stage).as_posix()
                for path in stage.rglob("*")
                if path.is_file()
            }
            assert visible == {"work/voice.mp3", "work/manifest.json"}
            assert json.loads((stage / "work" / "manifest.json").read_text()) == {
                "duration_seconds": 1.25
            }
            (stage / "work" / "voice_lines.json").write_text(
                json.dumps([{"text": "真实口播", "start_s": 0.0, "end_s": 1.0}]),
                encoding="utf-8",
            )
            (stage / "work" / "rogue.txt").write_text("ignore me", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setenv("ARK_API_KEY", "must-not-leak")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-leak")
        monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
        lines = CodexRunner(timeout_s=7, concurrency=1).run_voice(
            work,
            "voice prompt",
            duration_s=1.25,
            validate_output=lambda raw: voice.validate_voice_lines(raw, 1.25),
        )

        assert lines == [{"text": "真实口播", "start_s": 0.0, "end_s": 1.0}]
        assert not (work / "voice_lines.json").exists()
        assert not (work / "rogue.txt").exists()
        (argv, kwargs), = calls
        assert Path(argv[0]).name == "bwrap"
        assert any(argv[index:index + 2] == ["--tmpfs", "/tmp"] for index in range(len(argv)))
        assert "--bind" in argv and "--chdir" in argv
        inner = argv.index("codex")
        assert argv[inner:inner + 2] == ["codex", "exec"]
        assert argv[argv.index("-s", inner) + 1] == "workspace-write"
        assert "sandbox_workspace_write.network_access=false" in argv
        assert kwargs["timeout"] == 7
        assert all(
            secret not in kwargs["env"]
            for secret in ("ARK_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        )

    def test_voice_run_fails_closed_without_bwrap(self, monkeypatch, tmp_path):
        work = tmp_path / "conversation" / "work"
        work.mkdir(parents=True)
        (work / "voice.mp3").write_bytes(b"audio")
        monkeypatch.setattr(codex_runner.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            codex_runner.subprocess,
            "run",
            lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("must not run")),
        )

        with pytest.raises(CodexError, match="bwrap"):
            CodexRunner(timeout_s=1, concurrency=1).run_voice(
                work,
                "voice prompt",
                duration_s=1.0,
                validate_output=lambda raw: raw,
            )

    def test_voice_run_rejects_symlinked_audio(self, tmp_path):
        work = tmp_path / "conversation" / "work"
        work.mkdir(parents=True)
        outside = tmp_path / "outside.mp3"
        outside.write_bytes(b"audio")
        (work / "voice.mp3").symlink_to(outside)

        with pytest.raises(CodexError, match="voice.mp3"):
            CodexRunner(timeout_s=1, concurrency=1).run_voice(
                work,
                "voice prompt",
                duration_s=1.0,
                validate_output=lambda raw: raw,
            )

    def test_real_nonpaid_voice_sandbox_probe_blocks_session_and_repo(self, tmp_path):
        """真实 bwrap + `codex sandbox` 探针，不调用模型/API。"""
        if not shutil.which("bwrap") or not shutil.which("codex"):
            pytest.skip("bwrap/codex unavailable")
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        (cdir / "source.mp4").write_text("outside", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="duet-voice-probe-", dir="/tmp") as raw_stage:
            stage = Path(raw_stage)
            (stage / "work").mkdir()
            (stage / "work" / "voice.mp3").write_bytes(b"audio")
            (stage / "work" / "manifest.json").write_text(
                json.dumps({"duration_seconds": 1.0}), encoding="utf-8"
            )
            script = (
                f"test ! -r {str(cdir / 'source.mp4')!r} && "
                f"test ! -r {str(ROOT / 'app' / 'pipeline.py')!r} && "
                "test -r work/voice.mp3 && test -r work/manifest.json && "
                "printf '[]' > work/voice_lines.json"
            )
            inner = [
                "codex", "sandbox", "-P", ":workspace", "-C", str(stage),
                "bash", "-c", script,
            ]
            argv = codex_runner._voice_outer_argv(stage, cdir, inner)
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=20)

            assert proc.returncode == 0, proc.stderr
            assert (stage / "work" / "voice_lines.json").read_text() == "[]"


# ---------- 流水线编排（状态机） ----------


def test_run_done(tmp_path, video_1s, fake_steps):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    done = storage.load_meta(settings.data_dir, cid)
    assert done["status"] == "done" and done["error"] is None
    assert done["keyframes"] == ["01.png", "02.png", "03.png"]
    # 单段模式不添加任何 workaround 前缀，meta 与磁盘同步
    assert done["prompt"] == PROMPT_TEXT
    assert (cdir / "work" / "prompt.txt").read_text(encoding="utf-8") == done["prompt"]
    assert not (cdir / "preview.mp4").exists()  # 新契约不再生成占位预览

    # extract 调用契约：venv python 绝对路径、argv 列表、--fps 4、120s 超时
    extract = fake_steps["cmd"][0]
    assert extract["step"] == "extract"
    assert extract["argv"][0] == sys.executable
    assert str(EXTRACT_SCRIPT) in extract["argv"]
    assert "--fps" in extract["argv"] and "4" in extract["argv"]
    assert extract["timeout"] == 120

    # codex 运行前 skill 的 scripts/ 拷进会话目录（crop_image.py 相对引用）
    assert (cdir / "scripts" / "crop_image.py").read_bytes() == CROP_SCRIPT.read_bytes()
    assert (cdir / "scripts" / "extract_keyframes.py").is_file()

    # codex：工作目录=会话目录；prompt 指向 SKILL.md 且含硬性禁令
    (codex_call,) = fake_steps["codex"]
    assert codex_call["workdir"] == cdir
    prompt = codex_call["prompt"]
    for needle in (
        str(pipeline.SKILL_MD),
        "work/",
        sys.executable,
        str(cdir),
        "禁止联网",
        "环境变量",
    ):
        assert needle in prompt, needle


def test_run_status_sequence_processing_then_done(tmp_path, video_1s, fake_steps, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    seen = []
    orig = storage.update_meta

    def recording(data_dir, cid, **changes):
        if "status" in changes:
            seen.append(changes["status"])
        return orig(data_dir, cid, **changes)

    monkeypatch.setattr(storage, "update_meta", recording)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    assert seen == ["processing", "done"]


def test_run_extract_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def boom(argv, *, timeout, step, cwd=None):
        raise pipeline.PipelineError(f"{step} exit 1: codec missing")

    monkeypatch.setattr(pipeline, "_run_cmd", boom)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "extract" in m["error"]


def test_run_codex_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def bad_codex(self, workdir, prompt):
        raise CodexError("codex exit 2: agent crashed")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", bad_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "codex" in m["error"]


def test_run_codex_timeout(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def slow_codex(self, workdir, prompt):
        raise CodexError("codex timed out after 600s")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", slow_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "timed out" in m["error"]


def test_run_visual_retries_invalid_output_and_transient_timeout(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    calls = 0

    def flaky_codex(self, workdir, prompt):
        nonlocal calls
        calls += 1
        work = Path(workdir) / "work"
        assert not (work / "keyframes").exists()
        assert not (work / "prompt.txt").exists()
        if calls == 1:
            (work / "keyframes").mkdir()
            (work / "keyframes" / "stale.txt").write_text("stale")
            return
        if calls == 2:
            raise CodexError("codex timed out after 600s", retryable=True)
        _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", flaky_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert calls == 3
    assert not (settings.data_dir / meta["id"] / "work" / "keyframes" / "stale.txt").exists()


def test_run_codex_timeout_salvages_complete_output(tmp_path, video_1s, monkeypatch):
    """codex 超时被杀但产物已完整落盘 → 收养为 done。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def slow_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")  # 被杀前产物已写完
        raise CodexError("codex timed out after 600s")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", slow_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]
    assert m["prompt"] == PROMPT_TEXT


def test_run_validation_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def noop_codex(self, workdir, prompt):
        pass  # 一个产物都不写

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", noop_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert m["error"] == (
        "codex visual output invalid: required keyframes/prompt artifacts "
        "are missing or invalid"
    )
    assert "keyframe count 0" not in m["error"]


# ---------- 口播步（ASR，抽帧之后） ----------

VOICE_LINES = [
    {"text": "第一句。", "start_s": 0.0, "end_s": 0.5},
    {"text": "第二句。", "start_s": 0.5, "end_s": 1.0},
]


def _fake_extract_ok(argv, *, timeout, step, cwd=None):
    """extract/scenes 假子进程：写 manifest（含 duration_seconds，口播步要读）与空拆段 scenes.json。"""
    if step == "extract":
        work = Path(argv[argv.index("--out-dir") + 1])
        (work / "contact_sheet.jpg").write_bytes(b"sheet")
        (work / "manifest.json").write_text(
            json.dumps({"duration_seconds": 1.0}), encoding="utf-8"
        )
    elif step == "scenes":
        work = Path(argv[argv.index("--work-dir") + 1])
        (work / "scenes.json").write_text(
            json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
        )


def _set_voice_mode(settings, meta, voice_mode, target_language=""):
    changes = {"voice_mode": voice_mode}
    if target_language:
        changes["target_language"] = target_language
    storage.update_meta(settings.data_dir, meta["id"], **changes)


def _no_codex(self, workdir, prompt):
    raise AssertionError("codex must not run")


def test_run_voice_none_skips_asr(tmp_path, video_1s, fake_steps):
    """voice_mode=none（默认）不跑口播步：codex 只被调一次、无 voice 产物。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert "voice_lines" not in m
    (codex_call,) = fake_steps["codex"]
    assert "voice.mp3" not in codex_call["prompt"] and "听写" not in codex_call["prompt"]
    assert not (cdir / "work" / "voice.mp3").exists()


def test_run_voice_none_does_not_call_vocal_analyze(tmp_path, video_1s, fake_steps, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    called = []

    def unexpected(audio):
        called.append(audio)
        raise AssertionError("vocal.analyze must not run when voice_mode=none")

    monkeypatch.setattr(vocal, "analyze", unexpected)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    assert called == []
    assert storage.load_meta(settings.data_dir, meta["id"])["status"] == "done"


def test_run_voice_vocal_filter_records_bgm_and_dropped_count(
    tmp_path, video_1s, monkeypatch
):
    """句级过滤：spoken 与 sung 都保留（sung = 吟唱/唱词型台词），只丢 None 假转录。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    lines = [
        {"text": "口播", "start_s": 0.0, "end_s": 0.3},
        {"text": "唱歌", "start_s": 0.3, "end_s": 0.6},
        {"text": "幻觉", "start_s": 0.6, "end_s": 0.9},
    ]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps(lines), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[
                vocal.VocalWindow(0, 300, sung=0.0, spoken=0.3, music=0.2),
                vocal.VocalWindow(300, 600, sung=0.1, spoken=0.01, music=0.2),
                vocal.VocalWindow(600, 900, sung=0.01, spoken=0.01, music=0.2),
            ],
            has_bgm=True,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["voice_lines"] == [lines[0], lines[1]]
    assert stored["has_bgm"] is True
    assert stored["voice_lines_vocal_dropped"] == 1
    assert stored["vocal_filter_enabled"] is True
    assert stored["voice_line_provenance"] == [
        {**lines[0], "classification": "spoken", "provenance": "asr", "kept": True},
        {**lines[1], "classification": "sung", "provenance": "asr", "kept": True},
        {**lines[2], "classification": None, "provenance": "asr", "kept": False},
    ]


def test_run_voice_vocal_filter_off_bypasses_but_records_decisions(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    lines = [
        {"text": "口播", "start_s": 0.0, "end_s": 0.4},
        {"text": "幻觉", "start_s": 0.4, "end_s": 0.9},
    ]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps(lines), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setenv("VOCAL_FILTER", "off")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[
                vocal.VocalWindow(0, 400, sung=0.0, spoken=0.3, music=0.2),
                vocal.VocalWindow(400, 900, sung=0.01, spoken=0.01, music=0.2),
            ],
            has_bgm=True,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == lines
    assert stored["vocal_filter_enabled"] is False
    assert stored["voice_line_provenance"] == [
        {**lines[0], "classification": "spoken", "provenance": "asr", "kept": True},
        {**lines[1], "classification": None, "provenance": "asr", "kept": True},
    ]


def test_run_voice_vocal_failure_fails_pipeline(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        if "voice.mp3" in prompt:
            (Path(workdir) / "work" / "voice_lines.json").write_text(
                json.dumps([{"text": "口播", "start_s": 0.0, "end_s": 0.5}]),
                encoding="utf-8",
            )

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _audio: (_ for _ in ()).throw(
        vocal.VocalError("模型校验失败")
    ))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert "vocal classification unavailable" in stored["error"]
    assert "模型校验失败" in stored["error"]


def test_run_voice_audio_longer_than_container(tmp_path, video_1s, monkeypatch):
    """音频长 36ms 时 ASR 先按音频通过，再把最终台词裁回视频时间轴。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    line = {"text": "台词", "start_s": 0.224, "end_s": 27.936}  # end_s 超容器 27.9 但等于音频时长

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([line]), encoding="utf-8")
        else:
            _write_valid_package(work)

    def fake_extract_ok(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 27.9}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 27.9, "scenes": [], "segments": []})
            )

    monkeypatch.setattr(pipeline, "_run_cmd", fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 27.936)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 30_000, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["voice_lines"] == [
        {"text": "台词", "start_s": 0.224, "end_s": 27.9}
    ]
    assert any("video duration 27.900s" in item for item in stored["voice_warnings"])


def test_prepared_duration_has_no_project_upper_bound():
    assert pipeline._prepared_durations({"duration_s": 3600.1}) == (3600.1, 3601)


def test_run_voice_keep_runs_asr_and_stores_lines(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    cdir = settings.data_dir / meta["id"]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    codex_calls = []

    def fake_codex(self, workdir, prompt):
        codex_calls.append({"workdir": Path(workdir), "prompt": prompt})
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:  # ASR 调用
            (work / "voice_lines.json").write_text(json.dumps(VOICE_LINES), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done" and m["error"] is None
    assert m["voice_lines"] == VOICE_LINES
    assert (cdir / "work" / "voice.mp3").read_bytes() == b"mp3-bytes"

    # 两次 codex 调用：ASR 在前（抽帧之后）、video-maker 在后；ASR 不带 SKILL.md
    assert len(codex_calls) == 2
    (asr_call, maker_call) = codex_calls
    asr_prompt = asr_call["prompt"]
    assert "work/voice.mp3" in asr_prompt
    assert "work/manifest.json" in asr_prompt
    assert "1.000" in asr_prompt  # 时长数字直传 prompt
    assert "原文保持" in asr_prompt
    assert str(pipeline.SKILL_MD) not in asr_prompt
    for needle in ("禁止联网", "环境变量"):
        assert needle in asr_prompt, needle
    assert sys.executable not in asr_prompt
    assert str(cdir) not in asr_prompt
    assert asr_call["workdir"] != cdir
    assert asr_call["workdir"].parent == Path("/tmp")
    assert str(pipeline.SKILL_MD) in maker_call["prompt"]


def test_run_voice_agent_cannot_see_visual_or_ocr_inputs(
    tmp_path, video_1s, monkeypatch
):
    """独立污染反例：原会话有 OCR 假台词和视觉输入，ASR stage 仍只有音频输入。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    cdir = settings.data_dir / meta["id"]
    work = cdir / "work"
    work.mkdir(exist_ok=True)
    (work / "visual_prompt.txt").write_text("画面字：买一送一（不要朗读）", encoding="utf-8")
    (work / "frame_note.txt").write_text("OCR_FAKE_DIALOGUE", encoding="utf-8")
    asr_seen = []
    real_line = {"text": "Jumpa geng sekelapa.", "start_s": 0.0, "end_s": 0.8}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        stage_work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            visible = {
                path.relative_to(Path(workdir)).as_posix()
                for path in Path(workdir).rglob("*")
                if path.is_file()
            }
            asr_seen.append((Path(workdir), visible))
            assert visible == {"work/voice.mp3", "work/manifest.json"}
            assert "OCR_FAKE_DIALOGUE" not in "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in Path(workdir).rglob("*")
                if path.is_file()
            )
            (stage_work / "voice_lines.json").write_text(
                json.dumps([real_line]), encoding="utf-8"
            )
        else:
            _write_valid_package(stage_work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [real_line]
    assert len(asr_seen) == 1
    assert asr_seen[0][0] != cdir


def test_run_voice_translate_prompt_has_target_language(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "translate", target_language="英文")
    calls = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        calls.append(prompt)
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps(VOICE_LINES), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert m["voice_lines"] == VOICE_LINES
    assert "翻译成英文" in calls[0]
    # 目标语言由后端注入 maker prompt（codex 不从台词反推）
    assert "提示词与台词使用目标语言：英文" in calls[1]


def test_run_voice_rewrite_prompt_has_rule_and_lines(tmp_path, video_1s, monkeypatch):
    """rewrite 模式：ASR prompt 含洗稿规则（句数/句序/时间边界不变）且 voice_lines 落 meta。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "rewrite")
    calls = []
    rewritten_lines = [
        {"text": "直接使用金刚刷。", "start_s": 0.0, "end_s": 1.0}
    ]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        calls.append(prompt)
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps(rewritten_lines), encoding="utf-8"
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert m["voice_lines"] == [
        {"text": "直接使用金刚刷。", "start_s": 0.0, "end_s": 1.0}
    ]
    assert m["voice_text_normalizations"] == []
    asr_prompt = calls[0]
    assert "洗稿" in asr_prompt
    assert "句数不变" in asr_prompt
    assert "句序不变" in asr_prompt
    assert "时间边界不变" in asr_prompt
    assert "通用称呼" in asr_prompt


def test_run_voice_mode_unknown_fails(tmp_path, video_1s, monkeypatch):
    """绕过入口校验直改 meta 的非法 voice_mode → failed 且 error 含 unknown voice_mode。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "dub")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "unknown voice_mode" in m["error"]


def test_run_voice_translate_whitespace_target_fails(tmp_path, video_1s, monkeypatch):
    """target_language 为纯空白串 → 视为缺失 failed，不生成「翻译成   」prompt。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "translate", target_language="   ")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "target_language" in m["error"]


def test_run_voice_translate_requires_target_language(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "translate")  # 契约要求 translate 必带 target_language
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "target_language" in m["error"]


def test_run_voice_no_audio_track_fails(tmp_path, video_1s, monkeypatch):
    """无音轨兜底：extract_audio 探测返回 None → failed（上传校验只查时长，不查音轨）。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "audio" in m["error"]


def test_run_dialogue_auto_no_audio_is_valid_and_writes_prepared_receipt(
    tmp_path, video_1s, monkeypatch
):
    """新 H3 auto：无音轨等价于空台词，视觉产物完成后必须冻结 receipt。"""
    from app import prepared_input

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", lambda _cdir: None)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["voice_line_provenance"] == []
    assert stored["vocal_filter_enabled"] is True
    assert stored["prepared_input_receipt"] == prepared_input.RECEIPT_FILENAME
    receipt_path = cdir / prepared_input.RECEIPT_FILENAME
    assert receipt_path.is_file()
    loaded = prepared_input.load_prepared_input(
        cdir, receipt_path, expected_dialogue=()
    )
    assert loaded.dialogue_mode == "auto"
    assert loaded.normalized_audio is None
    assert loaded.voice_texts == ()


def test_run_dialogue_auto_ignores_external_lines_and_isolates_visual_codex(
    tmp_path, video_1s, monkeypatch
):
    """auto 只收养 _voice_step：外部/OCR 文字不能进入唯一发声块。"""
    from app import prepared_input

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        voice_lines=[{"text": "外部伪台词", "start_s": 0.0, "end_s": 0.8}],
        ratio="9:16",
        fit_mode="crop",
    )
    asr_line = {"text": "真实口播。", "start_s": 0.1, "end_s": 0.8}
    maker_saw_voice_file = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([asr_line]), encoding="utf-8"
            )
            return
        maker_saw_voice_file.append((work / "voice_lines.json").exists())
        assert "最终台词由后端" in prompt
        _write_valid_package(work, prompt="画面包装上可见 OCR ONLY。")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert maker_saw_voice_file == [False]
    assert stored["voice_lines"] == [asr_line]
    assert '说出台词："真实口播。"，嘴型与画面同步' in stored["prompt"]
    assert '说出台词："OCR ONLY"' not in stored["prompt"]
    assert "外部伪台词" not in stored["prompt"]
    expected = prepared_input.prepare_dialogue(
        "auto",
        duration_s=10,
        automatic_lines=[{**asr_line, "classification": "spoken"}],
    )
    loaded = prepared_input.load_prepared_input(
        cdir,
        cdir / prepared_input.RECEIPT_FILENAME,
        expected_dialogue=expected,
    )
    assert loaded.normalized_audio.data == b"normalized-audio"
    assert loaded.voice_texts == ("真实口播。",)


@pytest.mark.parametrize(
    ("voice_mode", "target_language", "expected"),
    [
        ("rewrite", "", "洗稿"),
        ("translate", "日语", "翻译成日语"),
    ],
)
def test_run_dialogue_auto_preserves_requested_voice_processing_mode(
    tmp_path, video_1s, monkeypatch, voice_mode, target_language, expected
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode=voice_mode,
        target_language=target_language,
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )
    asr_prompts = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            asr_prompts.append(prompt)
            (work / "voice_lines.json").write_text(
                json.dumps([{"text": "台词。", "start_s": 0.1, "end_s": 0.8}]),
                encoding="utf-8",
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert len(asr_prompts) == 1
    assert expected in asr_prompts[0]


def test_run_dialogue_auto_routes_explicit_empty_12_4s_scene_result_to_long_plan(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=12.4,
        ratio="9:16",
        fit_mode="pad",
    )
    line = {"text": "第十一秒台词。", "start_s": 11.0, "end_s": 12.0}

    def fake_steps(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 12.4}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 12.4, "scenes": [], "segments": []}),
                encoding="utf-8",
            )
        elif step.startswith("segment ") and step.endswith(" extract"):
            work = Path(argv[argv.index("--out-dir") + 1])
            work.mkdir(parents=True, exist_ok=True)
            (work / "001_frame_000.000s.png").write_bytes(b"source-first")
            (work / "002_frame_012.400s.png").write_bytes(b"source-last")
            (work / "manifest.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {"index": 1, "time_seconds": 0.0,
                             "file": "001_frame_000.000s.png"},
                            {"index": 2, "time_seconds": 12.4,
                             "file": "002_frame_012.400s.png"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([line]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(
        pipeline,
        "_cut_segment",
        lambda _source, _start, _end, segdir: (
            segdir.mkdir(parents=True, exist_ok=True),
            (segdir / "source.mp4").write_bytes(b"segment"),
        ),
    )
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 12.4)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 12_400, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [line]
    receipt = json.loads((cdir / "long_video_plan.json").read_text(encoding="utf-8"))
    assert receipt["video"]["duration_s"] == 12.4
    assert len(receipt["segments"]) == 1
    assert math.ceil(
        receipt["segments"][0]["end_s"] - receipt["segments"][0]["start_s"]
    ) == 13
    assert stored["segments"][0]["dialogue"] == [line]
    digest = hashlib.sha256((cdir / "long_video_plan.json").read_bytes()).hexdigest()
    frozen = long_generation.freeze_plan(cdir, stored, digest, "none", "auto")
    assert [math.ceil(item.end_s - item.start_s) for item in frozen.segments] == [13]


def test_run_dialogue_auto_clips_mp3_encoder_tail_to_video_timeline(
    tmp_path, video_1s, monkeypatch
):
    """线上复现：10.080s MP3 不能把 10.000s 视频 receipt 的台词时间轴撑长。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )
    asr_line = {"text": "完整十秒口播。", "start_s": 0.0, "end_s": 10.08}
    normalized = {"text": "完整十秒口播。", "start_s": 0.0, "end_s": 10.0}

    def fake_steps(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 10.0}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 10.0, "scenes": [], "segments": []}),
                encoding="utf-8",
            )

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([asr_line]), encoding="utf-8"
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 10.08)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 10_080, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [normalized]
    assert stored["voice_line_provenance"] == [
        {
            **normalized,
            "classification": "spoken",
            "provenance": "asr",
            "kept": True,
            "asr_start_s": 0.0,
            "asr_end_s": 10.08,
            "time_adjustment": "clipped_to_video_duration",
        }
    ]
    assert any("video duration 10.000s" in item for item in stored["voice_warnings"])
    receipt = json.loads((cdir / "prepared_input.json").read_text(encoding="utf-8"))
    assert receipt["video"]["duration_s"] == 10.0
    assert receipt["dialogue"]["lines"][0]["end_s"] == 10.0


def test_run_voice_drops_lines_starting_in_mp3_only_tail(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    kept = {"text": "视频内台词", "start_s": 9.5, "end_s": 9.9}
    tail = {"text": "编码尾部伪句", "start_s": 10.0, "end_s": 10.08}

    def fake_steps(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 10.0}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 10.0, "scenes": [], "segments": []}),
                encoding="utf-8",
            )

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([kept, tail]), encoding="utf-8"
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 10.08)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 10_080, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [kept]
    assert stored["voice_line_provenance"][1] == {
        **tail,
        "classification": "spoken",
        "provenance": "asr",
        "kept": False,
        "drop_reason": "starts_at_or_after_video_duration",
    }
    assert any("dropped 1" in item for item in stored["voice_warnings"])


@pytest.mark.parametrize(
    "lines",
    [
        [{"text": "Jumpa geng sekelapa.", "start_s": 0.0, "end_s": 2.5}],
        [
            {
                "text": "¿Tu perro tiene nudos y demasiado pelo suelto?",
                "start_s": 0.05,
                "end_s": 3.35,
            },
            {
                "text": "Este peine de doble diente es la solución.",
                "start_s": 3.84,
                "end_s": 6.65,
            },
            {
                "text": "Desenreda suavemente y sin tirones.",
                "start_s": 7.09,
                "end_s": 9.32,
            },
        ],
        [
            {
                "text": "Finally found the cheapest cheese squishy!",
                "start_s": 0.2,
                "end_s": 2.0,
            }
        ],
    ],
)
def test_voice_timeline_normalization_preserves_11_12_13_text_and_order(lines):
    decisions = [
        {
            **line,
            "classification": "spoken" if index % 2 == 0 else "sung",
            "provenance": "asr",
            "kept": True,
        }
        for index, line in enumerate(lines)
    ]

    normalized, normalized_decisions, warnings = pipeline._normalize_voice_timeline(
        decisions, 10.0
    )

    assert normalized == lines
    assert normalized_decisions == decisions
    assert warnings == []


def test_single_voice_line_uses_strong_track_evidence_when_asr_timestamp_misses():
    """temp/11 形态：唯一台词文本正确，但 ASR 区间早于真实口播，不能被误删。"""
    line = {"text": "Jumpa geng sekelapa.", "start_s": 0.0, "end_s": 2.5}
    analysis = vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 975, sung=0.0, spoken=0.015625, music=0.0),
            vocal.VocalWindow(4875, 5850, sung=0.0, spoken=0.33203125, music=0.0),
        ],
        has_bgm=False,
    )
    bgm_only = vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 975, sung=0.0, spoken=0.05859375, music=0.2),
        ],
        has_bgm=True,
    )

    assert pipeline._classify_voice_line(line, analysis, only_line=True) == "spoken"
    assert pipeline._classify_voice_line(line, bgm_only, only_line=True) is None


def test_voice_prompt_requires_multilingual_transcript_and_forbids_placeholders(tmp_path):
    prompt = pipeline._voice_prompt(tmp_path, "keep", "", 10.08)
    assert "自动识别实际语言" in prompt
    assert "[无法辨识]" in prompt
    assert "[inaudible]" in prompt
    assert "输出空数组" in prompt


def test_run_voice_placeholder_retries_then_continues_without_fake_dialogue(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    calls = {"voice": 0}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            calls["voice"] += 1
            (work / "voice_lines.json").write_text(
                json.dumps([
                    {"text": "[无法辨识]", "start_s": 0.0, "end_s": 0.9}
                ]),
                encoding="utf-8",
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 1.0)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(True))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert calls["voice"] == 3
    assert stored["voice_lines"] == []
    assert "[无法辨识]" not in stored["prompt"]
    assert any("占位符" in warning for warning in stored["voice_warnings"])


def test_run_voice_codex_timeout_salvages_complete_lines(tmp_path, video_1s, monkeypatch):
    """ASR 的 codex 超时被杀但 voice_lines.json 已完整 → 收养，继续 video-maker。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    calls = {"asr": 0, "maker": 0}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            calls["asr"] += 1
            (work / "voice_lines.json").write_text(json.dumps(VOICE_LINES), encoding="utf-8")
            raise CodexError("codex timed out after 600s")
        calls["maker"] += 1
        _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done" and m["error"] is None
    assert m["voice_lines"] == VOICE_LINES
    assert calls == {"asr": 1, "maker": 1}


def test_run_voice_codex_failure_no_product(tmp_path, video_1s, monkeypatch):
    """ASR 失败且无完整产物 → 报原始 CodexError。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def bad_codex(self, workdir, prompt):
        if "voice.mp3" in prompt:  # 只有 ASR 调用失败；video-maker 正常则验证断言只针对 ASR
            raise CodexError("codex timed out after 600s")
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", bad_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "timed out" in m["error"]


def test_run_voice_validation_failure(tmp_path, video_1s, monkeypatch):
    """ASR 返回 0 但产物非法 → failed，错误归因于 Codex 输出阶段。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def bad_codex(self, workdir, prompt):
        (Path(workdir) / "work" / "voice_lines.json").write_bytes(b"not json")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", bad_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert m["error"] == (
        "codex voice output invalid: required voice_lines artifact "
        "is missing or invalid"
    )


def test_run_voice_missing_output_reports_codex_stage(tmp_path, video_1s, monkeypatch):
    """ASR 返回 0 但没写产物时，不把底层文件缺失误报成输入视频问题。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", lambda self, workdir, prompt: None)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert m["error"].startswith("codex voice output invalid:")
    assert "voice_lines.json missing" not in m["error"]


# ---------- HTTP 接线 ----------


def test_post_triggers_pipeline_and_detail_filled(tmp_path, video_1s, fake_steps):
    settings = make_settings(tmp_path, enable_pipeline=True)
    with TestClient(create_app(settings)) as c:
        with open(video_1s, "rb") as f:
            r = c.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert r.status_code == 201
        cid = r.json()["id"]
        r = c.get(f"/api/conversations/{cid}", headers=AUTH)
    body = r.json()
    assert body["status"] == "done"
    assert body["keyframes"] == ["01.png", "02.png", "03.png"]
    assert body["prompt"] == prepared_input.compose_final_prompt(PROMPT_TEXT, ())
    assert "has_preview" not in body
    assert body["error"] is None


def test_done_fit_requirement_uses_actual_keyframes_not_source_dimensions(
    tmp_path, video_1s, fake_steps, monkeypatch
):
    settings = make_settings(tmp_path, enable_pipeline=True)
    monkeypatch.setattr(storage, "probe_video", lambda *_args: storage.VideoProbe(1.0, 90, 160))

    with TestClient(create_app(settings)) as client:
        with open(video_1s, "rb") as file:
            created = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", file, "video/mp4")},
            )
        detail = client.get(
            f"/api/conversations/{created.json()['id']}", headers=AUTH
        ).json()

    assert detail["status"] == "done"
    assert detail["fit_required"] is True  # fake Codex 产出 1x1 关键帧


def test_pipeline_off_by_default(client, video_1s, monkeypatch):
    """Settings 直建（旧测试路径）默认不触发流水线，保持 queued。"""
    called = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: called.append(1))
    with open(video_1s, "rb") as f:
        r = client.post(
            "/api/conversations", headers=AUTH, files={"file": ("clip.mp4", f, "video/mp4")}
        )
    assert r.status_code == 201
    assert called == []
    r = client.get(f"/api/conversations/{r.json()['id']}", headers=AUTH)
    assert r.json()["status"] == "queued"


# ---------- config 新字段 ----------


def test_config_pipeline_fields(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("ACCESS_TOKEN", "t")
    monkeypatch.setenv("CODEX_TIMEOUT_S", "42")
    monkeypatch.setenv("CODEX_CONCURRENCY", "3")
    monkeypatch.setenv("MAX_QUEUED", "7")
    monkeypatch.setenv("ENABLE_PIPELINE", "0")
    s = get_settings()
    assert s.codex_timeout_s == 42
    assert s.codex_concurrency == 3
    assert s.max_queued == 7
    assert s.enable_pipeline is False

    monkeypatch.delenv("CODEX_TIMEOUT_S")
    monkeypatch.delenv("CODEX_CONCURRENCY")
    monkeypatch.delenv("MAX_QUEUED")
    monkeypatch.delenv("ENABLE_PIPELINE")
    s = get_settings()
    assert s.codex_timeout_s == 1800
    assert s.codex_concurrency == 10
    assert s.max_queued == 100
    assert s.enable_pipeline is True  # 生产路径默认开


# ---------- storage.update_meta ----------


def test_update_meta(tmp_path):
    meta = storage.new_conversation(tmp_path, "", "a.mp4")
    updated = storage.update_meta(tmp_path, meta["id"], status="done", keyframes=["k.png"])
    assert updated["status"] == "done" and updated["keyframes"] == ["k.png"]
    assert updated["updated_at"] >= meta["updated_at"]
    assert storage.load_meta(tmp_path, meta["id"])["status"] == "done"
    assert storage.update_meta(tmp_path, "0" * 32, status="x") is None
    assert storage.update_meta(tmp_path, "..", status="x") is None


# ---------- 假 codex 桩：全编排真实子进程 e2e（无 mock） ----------


def _write_stub_codex(bin_dir: Path, frames: int) -> Path:
    """生成一个按新契约直产合法产物的假 codex：从 work/ 抽好的帧里挑 frames 张复制进 keyframes/。"""
    stub = bin_dir / "codex"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""\
            import shutil, sys
            from pathlib import Path

            argv = sys.argv[1:]
            workdir = Path(argv[argv.index("-C") + 1])
            out = Path(argv[argv.index("-o") + 1])
            work = workdir / "work"
            kdir = work / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(work.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in work/"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (work / "prompt.txt").write_text({PROMPT_TEXT!r} + "（桩产物）", encoding="utf-8")
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _write_stub_codex_voice(bin_dir: Path, frames: int) -> Path:
    """桩 codex 兼处理两种调用：ASR 调用写 voice_lines.json，video-maker 调用挑帧写 prompt。"""
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
            work = workdir / "work"
            if "voice.mp3" in argv[-1]:
                (work / "voice_lines.json").write_text(
                    json.dumps([{{"text": "你好，世界。", "start_s": 0.0, "end_s": 1.0}}]),
                    encoding="utf-8",
                )
                out.write_text("asr done", encoding="utf-8")
                raise SystemExit(0)
            kdir = work / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(work.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in work/"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (work / "prompt.txt").write_text({PROMPT_TEXT!r} + "（桩产物）", encoding="utf-8")
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


@pytest.fixture
def voice_stub_bin():
    """外层 voice bwrap 会隐藏 /tmp 和仓库，桩程序必须放在仍可见的位置。"""
    with tempfile.TemporaryDirectory(prefix="duet-codex-stub-", dir="/var/tmp") as raw:
        yield Path(raw)


def test_full_pipeline_voice_with_stub_codex(tmp_path, monkeypatch, voice_stub_bin):
    """真 subprocess 全链路含口播：extract → ffmpeg 抽音轨 → 桩 codex ASR → 桩 codex 选帧 → done。"""
    video = tmp_path / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-shortest", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True, capture_output=True,
    )
    bin_dir = voice_stub_bin
    _write_stub_codex_voice(bin_dir, 3)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video)
    _set_voice_mode(settings, meta, "keep")
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["voice_lines"] == [{"text": "你好，世界。", "start_s": 0.0, "end_s": 1.0}]
    assert (cdir / "work" / "voice.mp3").is_file()
    assert (cdir / "work" / "voice_lines.json").is_file()
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]


def test_full_pipeline_with_stub_codex(tmp_path, video_1s, monkeypatch):
    """真 subprocess 全链路：extract --fps 4 → 桩 codex → 校验 → done（不再生成 preview）。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex(bin_dir, 3)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]
    assert "15 秒" in m["prompt"]
    assert (cdir / "codex_last_message.txt").is_file()
    assert (cdir / "work" / "contact_sheet.jpg").is_file()  # 1s×4fps=5 帧，单页联系表
    assert (cdir / "work" / "manifest.json").is_file()
    assert (cdir / "scripts" / "crop_image.py").is_file()  # skill 脚本已拷进会话目录
    assert not (cdir / "preview.mp4").exists()


def test_full_pipeline_relative_data_dir(tmp_path, video_1s, monkeypatch):
    """回归：DATA_DIR 为相对路径（生产默认 "data"）时流水线也必须成功。

    子进程带 cwd 时相对 data_dir 会错位，run() 入口须先把会话目录解析为绝对路径。
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex(bin_dir, 3)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)

    settings = make_settings(tmp_path, data_dir=Path("data"))
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]
    assert (tmp_path / "data" / cid / "work" / "prompt.txt").is_file()


def _spoken_analysis(spoken: bool) -> vocal.VocalAnalysis:
    """声学预判桩：spoken=True 表示音轨含人声（12 窗口 spoken≥0.2 的形态）。"""
    return vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 30_000, sung=0.0, spoken=0.3 if spoken else 0.01, music=0.0),
        ],
        has_bgm=False,
    )


def _sung_analysis(sung: float) -> vocal.VocalAnalysis:
    """声学预判桩：纯唱/吟唱音轨（spoken≈0，sung 给定为强或弱）。"""
    return vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 30_000, sung=sung, spoken=0.01, music=0.8),
        ],
        has_bgm=True,
    )


def test_run_voice_empty_lines_with_sung_retries_then_warns(tmp_path, video_1s, monkeypatch):
    """真实量化边界 51/256 算人声；三次空则明确警告并按用户确认继续无台词。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _sung_analysis(0.19921875))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["voice_warnings"] == [
        "voice_lines.json empty after automatic retries despite vocal evidence; continuing without dialogue"
    ]


def test_run_voice_empty_lines_with_weak_sung_passes(tmp_path, video_1s, monkeypatch):
    """弱 sung（纯 BGM 里蹭出的 <0.2 唱分）+ 听写为空：仍属合法「无台词」。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _sung_analysis(0.059))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []


def test_run_voice_empty_lines_with_spoken_retries_and_succeeds(tmp_path, video_1s, monkeypatch):
    """音轨有人声但 codex 第一次输出空数组（随机摆烂）：重试一次听出台词 → done。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    line = {"text": "台词", "start_s": 0.0, "end_s": 0.9}
    codex_calls = []
    voice_stages = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        codex_calls.append(prompt)
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            visible_before = {
                path.relative_to(Path(workdir)).as_posix()
                for path in Path(workdir).rglob("*")
                if path.is_file()
            }
            assert visible_before == {"work/voice.mp3", "work/manifest.json"}
            voice_stages.append(Path(workdir))
            (work / "voice_lines.json").write_text(
                json.dumps([] if len(codex_calls) == 1 else [line]), encoding="utf-8"
            )
            (work / "stale-from-attempt.txt").write_text("must not survive", encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(True))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [line]
    # voice 调用 = 第一次听写 + 重试共 2 次（prompt 步另有 1 次非 voice 调用）
    assert sum(1 for p in codex_calls if "voice.mp3" in p) == 2
    assert len(voice_stages) == 2 and voice_stages[0] != voice_stages[1]


def test_run_voice_empty_lines_with_spoken_retry_still_empty_warns(tmp_path, video_1s, monkeypatch):
    """音轨有人声、重试后仍空：明确 warning 后继续，且不会伪造台词。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(True))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["voice_warnings"]


def test_run_voice_empty_lines_without_spoken_passes(tmp_path, video_1s, monkeypatch):
    """音轨无人声（纯 BGM/静音）且听写为空：合法「无台词」，done 且 voice_lines=[]。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(False))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
