"""任务 B：处理流水线（extract --fps 4 → codex 沙箱 → 白名单校验 → meta 落盘）。"""
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

from app import codex_runner, pipeline, storage, vocal, voice
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
    return meta


@pytest.fixture
def fake_steps(monkeypatch):
    """mock 掉 extract 子进程与 codex；返回调用记录。"""
    calls = {"cmd": [], "codex": []}

    def fake_cmd(argv, *, timeout, step, cwd=None):
        calls["cmd"].append({"argv": list(argv), "timeout": timeout, "step": step, "cwd": cwd})
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
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
        with pytest.raises(CodexError, match="timed out"):
            CodexRunner(timeout_s=7, concurrency=1).run(Path("/wd"), "p")

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

    def test_missing_codex_binary(self, monkeypatch):
        def nope(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        monkeypatch.setattr(codex_runner.subprocess, "run", nope)
        with pytest.raises(CodexError, match="codex"):
            CodexRunner(timeout_s=1, concurrency=1).run(Path("/wd"), "p")

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
    # 单段模式：条件动作行机械加在 prompt 开头（无 BGM 行），meta 与磁盘同步
    assert done["prompt"] == pipeline.FACE_HOLD_CONDITION_LINE + "\n" + PROMPT_TEXT
    assert done["prompt"].splitlines()[0] == pipeline.FACE_HOLD_CONDITION_LINE
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
    assert m["prompt"] == pipeline.FACE_HOLD_CONDITION_LINE + "\n" + PROMPT_TEXT


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
    assert "keyframe" in m["error"]


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
    assert stored["voice_lines"] == [lines[0]]
    assert stored["has_bgm"] is True
    assert stored["voice_lines_vocal_dropped"] == 2


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
    """音视频不同长（音频流比容器长 36ms，常态）：台词 end_s 写到音频末尾应通过——
    校验基准是 voice.mp3 实际时长，不是容器时长（否则全部此类视频都 failed）。"""
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
    assert stored["voice_lines"] == [line]


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
    for needle in (sys.executable, str(cdir), "禁止联网", "环境变量"):
        assert needle in asr_prompt, needle
    assert str(pipeline.SKILL_MD) in maker_call["prompt"]


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
    asr_prompt = calls[0]
    assert "洗稿" in asr_prompt
    assert "句数不变" in asr_prompt
    assert "句序不变" in asr_prompt
    assert "时间边界不变" in asr_prompt


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
    """ASR 产物过不了白名单 → failed，错误指明 voice_lines。"""
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
    assert "voice_lines" in m["error"]


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
    assert body["prompt"] == pipeline.FACE_HOLD_CONDITION_LINE + "\n" + PROMPT_TEXT
    assert "has_preview" not in body
    assert body["error"] is None


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
        f"#!{sys.executable}\n"
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


def test_full_pipeline_voice_with_stub_codex(tmp_path, monkeypatch):
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
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
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
