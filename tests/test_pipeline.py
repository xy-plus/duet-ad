"""任务 B：处理流水线（extract → codex 沙箱 → 白名单校验 → ffmpeg 占位预览）。"""
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

from app import codex_runner, pipeline, storage
from app.codex_runner import CodexError, CodexRunner
from app.main import create_app

ROOT = Path(pipeline.__file__).resolve().parent.parent
EXTRACT_SCRIPT = ROOT / "skills" / "seedance-cleaning-video-maker" / "scripts" / "extract_keyframes.py"
SEEDANCE_SCRIPT = ROOT / "skills" / "seedance-cleaning-video-maker" / "scripts" / "seedance_task.py"

PROMPT_TEXT = "生成一支 15 秒、9:16 竖屏、720p、写实手机实拍风格的清洁短视频。"


def _write_valid_package(work: Path, frames: int = 3, prompt: str = PROMPT_TEXT):
    """按约定文件名造一套合法产物，返回关键帧文件名列表。"""
    kdir = work / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(1, frames + 1):
        name = f"{i:02d}_keyframe_{i / 10:.3f}s.png"
        (kdir / name).write_bytes(b"\x89PNG")
        names.append(name)
    (work / "seedance_prompt.txt").write_text(prompt, encoding="utf-8")
    (work / "shot_timeline.md").write_text("# 分镜时间线\n", encoding="utf-8")
    req = {
        "model": "doubao-seedance-2-0-260128",
        "content": [{"type": "text", "text": prompt}] + [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AA=="},
                "role": "reference_image",
            }
            for _ in names
        ],
        "ratio": "9:16",
        "duration": 15,
        "resolution": "720p",
        "generate_audio": True,
        "watermark": False,
    }
    (work / "api_request.json").write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
    return names


def _make_conversation(settings, video_1s):
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    shutil.copy(video_1s, settings.data_dir / meta["id"] / "source.mp4")
    return meta


@pytest.fixture
def fake_steps(monkeypatch):
    """mock 掉 extract/ffmpeg 子进程与 codex；返回调用记录。"""
    calls = {"cmd": [], "codex": []}

    def fake_cmd(argv, *, timeout, step, cwd=None):
        calls["cmd"].append({"argv": list(argv), "timeout": timeout, "step": step, "cwd": cwd})
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "preview":
            Path(argv[-1]).write_bytes(b"mp4")

    def fake_codex(self, workdir, prompt):
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    return calls


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
        (work / "seedance_prompt.txt").unlink()
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
        (work / "seedance_prompt.txt").write_bytes(b"x" * (32 * 1024 + 1))
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_timeline_missing(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        (work / "shot_timeline.md").unlink()
        with pytest.raises(pipeline.PipelineError, match="shot_timeline"):
            pipeline.validate_work_dir(work)

    def test_api_request_invalid_json(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        (work / "api_request.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(pipeline.PipelineError, match="api_request"):
            pipeline.validate_work_dir(work)

    def test_api_request_two_texts(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        req = json.loads((work / "api_request.json").read_text())
        req["content"].insert(1, {"type": "text", "text": "extra"})
        (work / "api_request.json").write_text(json.dumps(req), encoding="utf-8")
        with pytest.raises(pipeline.PipelineError, match="text"):
            pipeline.validate_work_dir(work)

    def test_api_request_no_text(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        req = json.loads((work / "api_request.json").read_text())
        req["content"] = [i for i in req["content"] if i["type"] != "text"]
        (work / "api_request.json").write_text(json.dumps(req), encoding="utf-8")
        with pytest.raises(pipeline.PipelineError, match="text"):
            pipeline.validate_work_dir(work)

    def test_api_request_ten_images(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        req = json.loads((work / "api_request.json").read_text())
        img = req["content"][1]
        req["content"] = req["content"][:1] + [img] * 10
        (work / "api_request.json").write_text(json.dumps(req), encoding="utf-8")
        with pytest.raises(pipeline.PipelineError, match="image"):
            pipeline.validate_work_dir(work)

    def test_api_request_unknown_item_type(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        req = json.loads((work / "api_request.json").read_text())
        req["content"].append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}})
        (work / "api_request.json").write_text(json.dumps(req), encoding="utf-8")
        with pytest.raises(pipeline.PipelineError, match="content"):
            pipeline.validate_work_dir(work)

    @pytest.mark.parametrize("bad_key", ["authorization", "Token", "API_KEY", "secret"])
    @pytest.mark.parametrize(
        "where",
        ["top", "nested_dict", "nested_list", "inside_content"],
    )
    def test_api_request_secret_keys_rejected(self, tmp_path, bad_key, where):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        req = json.loads((work / "api_request.json").read_text())
        if where == "top":
            req[bad_key] = "x"
        elif where == "nested_dict":
            req["extra"] = {"deep": {bad_key: "x"}}
        elif where == "nested_list":
            req["extra"] = [[{bad_key: "x"}]]
        else:
            req["content"][0][bad_key] = "x"
        (work / "api_request.json").write_text(json.dumps(req), encoding="utf-8")
        with pytest.raises(pipeline.PipelineError, match="api_request"):
            pipeline.validate_work_dir(work)

    def test_secret_words_in_values_are_ok(self, tmp_path):
        """只扫字段名，不扫值：URL/文本里出现 token 字样不误伤。"""
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        req = json.loads((work / "api_request.json").read_text())
        req["content"][1]["image_url"]["url"] = "data:image/png;base64,tokenlikevalue"
        (work / "api_request.json").write_text(json.dumps(req), encoding="utf-8")
        pipeline.validate_work_dir(work)


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
    assert done["keyframes"] == [
        "01_keyframe_0.100s.png",
        "02_keyframe_0.200s.png",
        "03_keyframe_0.300s.png",
    ]
    assert done["prompt"] == PROMPT_TEXT
    assert (cdir / "preview.mp4").read_bytes() == b"mp4"

    # extract 调用契约：venv python 绝对路径、argv 列表、40 帧、inspect 前缀、120s 超时
    extract = fake_steps["cmd"][0]
    assert extract["step"] == "extract"
    assert extract["argv"][0] == sys.executable
    assert str(EXTRACT_SCRIPT) in extract["argv"]
    assert "40" in extract["argv"] and "inspect" in extract["argv"]
    assert extract["timeout"] == 120

    # codex：工作目录=会话目录；prompt 含产物约定与禁令
    (codex_call,) = fake_steps["codex"]
    assert codex_call["workdir"] == cdir
    prompt = codex_call["prompt"]
    for needle in (
        "seedance_prompt.txt",
        "shot_timeline.md",
        "api_request.json",
        "keyframes",
        "--dry-run",
        "禁止联网",
        "环境变量",
    ):
        assert needle in prompt, needle

    # ffmpeg 预览契约：720x1280、25fps、120s 超时、argv 列表
    preview = fake_steps["cmd"][1]
    assert preview["step"] == "preview"
    assert preview["argv"][0] == "ffmpeg"
    joined = " ".join(preview["argv"])
    assert "720:1280" in joined and "fps=25" in joined
    assert preview["timeout"] == 120


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
    assert not (settings.data_dir / meta["id"] / "preview.mp4").exists()


def test_run_codex_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")

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

    def slow_codex(self, workdir, prompt):
        raise CodexError("codex timed out after 600s")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", slow_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "timed out" in m["error"]


def test_run_validation_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")

    def noop_codex(self, workdir, prompt):
        pass  # 一个产物都不写

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", noop_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "keyframe" in m["error"]


def test_run_preview_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "preview":
            raise pipeline.PipelineError("preview exit 1: encoder missing")

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "preview" in m["error"]


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
    assert body["keyframes"] == [
        "01_keyframe_0.100s.png",
        "02_keyframe_0.200s.png",
        "03_keyframe_0.300s.png",
    ]
    assert body["prompt"] == PROMPT_TEXT
    assert body["has_contact_sheet"] is True
    assert body["has_preview"] is True
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
    assert s.codex_timeout_s == 600
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


def _write_stub_codex(bin_dir: Path, times: str) -> Path:
    """生成一个按约定文件名直产合法产物的假 codex 可执行文件。"""
    stub = bin_dir / "codex"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""\
            import subprocess, sys
            from pathlib import Path

            argv = sys.argv[1:]
            workdir = Path(argv[argv.index("-C") + 1])
            out = Path(argv[argv.index("-o") + 1])
            work = workdir / "work"
            source = next(workdir.glob("source.*"))
            subprocess.run([{sys.executable!r}, {str(EXTRACT_SCRIPT)!r}, str(source),
                            "--out-dir", str(work / "keyframes"), "--times", {times!r},
                            "--prefix", "keyframe", "--columns", "3"], check=True)
            frames = sorted((work / "keyframes").glob("*.png"))
            assert 1 <= len(frames) <= 9
            (work / "shot_timeline.md").write_text(
                "# 分镜时间线\\n\\n0.0-15.0 秒：单场景清洁演示。\\n", encoding="utf-8")
            (work / "seedance_prompt.txt").write_text(
                {PROMPT_TEXT!r} + "（桩产物）", encoding="utf-8")
            subprocess.run([{sys.executable!r}, {str(SEEDANCE_SCRIPT)!r}, "create", "--dry-run",
                            "--prompt-file", str(work / "seedance_prompt.txt"),
                            "--ref-images", *[str(p) for p in frames],
                            "--model", "doubao-seedance-2-0-260128", "--ratio", "9:16",
                            "--duration", "15", "--resolution", "720p",
                            "--generate-audio", "--no-watermark",
                            "--payload-out", str(work / "api_request.json")], check=True)
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_full_pipeline_with_stub_codex(tmp_path, video_1s, monkeypatch):
    """真 subprocess 全链路：extract → 桩 codex → 校验 → ffmpeg → done。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex(bin_dir, "0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert len(m["keyframes"]) == 9
    assert all(re.match(r"^\d{2}_keyframe_.*\.png$", n) for n in m["keyframes"])
    assert "15 秒" in m["prompt"]
    assert (cdir / "codex_last_message.txt").is_file()
    assert (cdir / "work" / "contact_sheet.jpg").is_file()
    assert (cdir / "work" / "manifest.json").is_file()
    assert (cdir / "work" / "shot_timeline.md").is_file()

    req = json.loads((cdir / "work" / "api_request.json").read_text())
    types = [i["type"] for i in req["content"]]
    assert types.count("text") == 1 and types.count("image_url") == 9

    info = json.loads(
        subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                "-of", "json", str(cdir / "preview.mp4"),
            ],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    stream = info["streams"][0]
    assert (stream["width"], stream["height"]) == (720, 1280)
    assert stream["avg_frame_rate"] == "25/1"
    assert abs(float(info["format"]["duration"]) - 15.0) < 0.2


def test_full_pipeline_relative_data_dir(tmp_path, video_1s, monkeypatch):
    """回归：DATA_DIR 为相对路径（生产默认 "data"）时流水线也必须成功。

    _render_preview 以 keyframes/ 为 cwd 跑 ffmpeg，相对 dest 会解析到错误位置。
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex(bin_dir, "0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)

    settings = make_settings(tmp_path, data_dir=Path("data"))
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert (tmp_path / "data" / cid / "preview.mp4").is_file()
