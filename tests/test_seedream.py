"""seedream_task.py（argv 脚本）与 seedream.py（门控层）的测试。"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import make_settings

from app import seedream, seedream_task
from app.config import get_settings

PNG = b"\x89PNG\r\n\x1a\n"
SECRET = "sk-live-abcdef123456"
EDIT_URL = "https://ark.cn-beijing.volces.com/api/v1/images/edits"


def _png(tmp_path, name="in.png"):
    p = tmp_path / name
    p.write_bytes(PNG + b"fake-image-data")
    return p


# ---------- 参数校验与默认值 ----------

def test_edit_missing_required_args(tmp_path):
    png = _png(tmp_path)
    cases = [
        ["edit", "--prompt", "p", "--out", str(tmp_path / "o.png")],
        ["edit", "--image", str(png), "--out", str(tmp_path / "o.png")],
        ["edit", "--image", str(png), "--prompt", "p"],
    ]
    for argv in cases:
        with pytest.raises(SystemExit) as e:
            seedream_task.parse_args(argv)
        assert e.value.code == 2, argv


def test_edit_arg_defaults(tmp_path):
    png = _png(tmp_path)
    args = seedream_task.parse_args(
        ["edit", "--image", str(png), "--prompt", "p", "--out", str(tmp_path / "o.png")]
    )
    assert args.model == "doubao-seedream-5-0-pro-260628"
    assert args.poll_interval == 5
    assert args.poll_timeout == 600
    assert args.request_timeout == 120


def test_edit_bad_timeouts(tmp_path):
    png = _png(tmp_path)
    for flag in ("--poll-interval", "--poll-timeout", "--request-timeout"):
        argv = ["edit", "--image", str(png), "--prompt", "p",
                "--out", str(tmp_path / "o.png"), flag, "-1"]
        with pytest.raises(SystemExit, match="positive"):
            seedream_task.main(argv)


# ---------- dry-run：不联网、不需要 key、校验入参 ----------

def test_dry_run_no_key_no_network(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    out = tmp_path / "edited.png"

    def boom(*a, **k):
        raise AssertionError("dry-run must not touch the network")

    monkeypatch.setattr(seedream_task.urllib.request, "urlopen", boom)
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "戴上眼镜",
         "--out", str(out), "--dry-run"]
    )
    assert seedream_task.run_edit(args) == 0
    assert '"dry_run": true' in capsys.readouterr().out
    assert not out.exists()


def test_dry_run_missing_image(tmp_path):
    args = seedream_task.parse_args(
        ["edit", "--image", str(tmp_path / "nope.png"), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--dry-run"]
    )
    with pytest.raises(SystemExit, match="does not exist"):
        seedream_task.run_edit(args)


def test_dry_run_non_png_rejected(tmp_path):
    img = tmp_path / "in.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0junk")
    args = seedream_task.parse_args(
        ["edit", "--image", str(img), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--dry-run"]
    )
    with pytest.raises(SystemExit, match="PNG"):
        seedream_task.run_edit(args)


def test_dry_run_empty_prompt(tmp_path):
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "   ",
         "--out", str(tmp_path / "o.png"), "--dry-run"]
    )
    with pytest.raises(SystemExit, match="Prompt is empty"):
        seedream_task.run_edit(args)


# ---------- 机械门控与密钥 ----------

def test_live_without_confirm_blocked(tmp_path, monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png")]
    )
    with pytest.raises(SystemExit, match="confirm-submit"):
        seedream_task.run_edit(args)


def test_live_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    with pytest.raises(SystemExit, match="ARK_API_KEY"):
        seedream_task.run_edit(args)


def test_script_error_exit_nonzero_no_key_leak(tmp_path):
    """真实子进程跑脚本：校验失败 → 退出码 1，stderr 不含密钥字面值。"""
    env = {**os.environ, "ARK_API_KEY": SECRET}
    r = subprocess.run(
        [sys.executable, str(seedream_task.__file__), "edit",
         "--image", "missing.png", "--prompt", "p", "--out", "o.png",
         "--confirm-submit"],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert r.returncode == 1
    assert SECRET not in r.stdout and SECRET not in r.stderr
    assert "does not exist" in r.stderr


# ---------- 真实提交全链路（monkeypatch urllib） ----------

class FakeResponse:
    """JSON 响应假实现；bytes 直接原样返回（下载分支）。"""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_ark(monkeypatch, responses):
    """urlopen 换成依次吐出 responses 的假实现，记录 (request, kwargs)。"""
    calls = []
    it = iter(responses)

    def fake_urlopen(request, **kw):
        calls.append((request, kw))
        payload = next(it)
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)

    monkeypatch.setattr(seedream_task.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(seedream_task.time, "sleep", lambda s: None)
    return calls


def test_full_chain_success_b64(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    out = tmp_path / "edited.png"
    state = tmp_path / "task.json"
    img = PNG + b"edited-image-bytes"
    b64 = base64.b64encode(img).decode("ascii")
    calls = _fake_ark(monkeypatch, [
        {"request_id": "req-123", "status": "queued"},
        {"request_id": "req-123", "status": "processing"},
        {"request_id": "req-123", "status": "succeeded",
         "content": [{"type": "image_url", "b64_json": b64}]},
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "戴上眼镜",
         "--out", str(out), "--confirm-submit", "--state-file", str(state)]
    )
    assert seedream_task.run_edit(args) == 0

    # POST：multipart 构造 + 认证头
    (req, kw), = calls[:1]
    assert req.full_url == EDIT_URL
    assert req.get_header("Authorization") == f"Bearer {SECRET}"
    assert kw["timeout"] == 120
    assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
    body = req.data
    assert b'name="model"' in body
    assert b"doubao-seedream-5-0-pro-260628" in body
    assert b'name="prompt"' in body and "戴上眼镜".encode("utf-8") in body
    assert b'name="image"' in body
    assert b'filename="in.png"' in body
    assert b"image/png" in body
    assert PNG in body
    assert b'name="response_format"' in body and b"b64_json" in body
    assert b'name="watermark"' in body and b"false" in body

    # 轮询：GET {EDIT_URL}/{request_id}
    assert [r.full_url for r, _ in calls[1:]] == [f"{EDIT_URL}/req-123"] * 2
    # b64 解码落盘 + state-file 写最新任务
    assert out.read_bytes() == img
    task = json.loads(state.read_text(encoding="utf-8"))
    assert task["status"] == "succeeded"
    assert "req-123" in capsys.readouterr().out


def test_sync_response_saves_without_poll(tmp_path, monkeypatch):
    """提交即带图（同步风格 data[] 响应，无 id）→ 不轮询直接写盘。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    out = tmp_path / "edited.png"
    img = PNG + b"sync-bytes"
    b64 = base64.b64encode(img).decode("ascii")
    calls = _fake_ark(monkeypatch, [{"created": 1758200023, "data": [{"b64_json": b64}]}])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(out), "--confirm-submit"]
    )
    assert seedream_task.run_edit(args) == 0
    assert len(calls) == 1
    assert out.read_bytes() == img


def test_succeeded_url_download(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    out = tmp_path / "edited.png"
    img = PNG + b"url-downloaded"
    calls = _fake_ark(monkeypatch, [
        {"request_id": "req-9", "status": "running"},
        {"request_id": "req-9", "status": "succeeded",
         "content": {"image_url": {"url": "https://tos.example/edited.png"}}},
        img,
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(out), "--confirm-submit"]
    )
    assert seedream_task.run_edit(args) == 0
    assert calls[2][0] == "https://tos.example/edited.png"
    assert out.read_bytes() == img


def test_failed_status_exit_1(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    _fake_ark(monkeypatch, [
        {"request_id": "req-f", "status": "running"},
        {"request_id": "req-f", "status": "failed"},
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    assert seedream_task.run_edit(args) == 1


@pytest.mark.parametrize("status", ["cancelled", "expired"])
def test_initial_cancelled_expired_exit_1(tmp_path, monkeypatch, status):
    """提交即终态（取消/过期）→ 退出码 1，不再轮询。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    calls = _fake_ark(monkeypatch, [{"request_id": "req-c", "status": status}])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    assert seedream_task.run_edit(args) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["cancelled", "expired"])
def test_poll_terminal_cancelled_expired_exit_1(tmp_path, monkeypatch, status):
    """轮询中变取消/过期 → 退出码 1，不白等 poll-timeout。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    _fake_ark(monkeypatch, [
        {"request_id": "req-c", "status": "running"},
        {"request_id": "req-c", "status": status},
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    assert seedream_task.run_edit(args) == 1


def test_poll_timeout_nonzero_exit(tmp_path, monkeypatch, capsys):
    """轮询超时：cli 把 TimeoutError 转成退出码 1，stderr 不泄密钥。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    _fake_ark(monkeypatch, [
        {"request_id": "req-t", "status": "running"},
    ])
    clock = iter([0.0, 5.0])
    monkeypatch.setattr(seedream_task.time, "monotonic", lambda: next(clock))
    rc = seedream_task.cli(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit",
         "--poll-timeout", "1"]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "did not finish" in err
    assert SECRET not in err


def test_succeeded_without_image_data_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    _fake_ark(monkeypatch, [
        {"request_id": "req-x", "status": "succeeded"},  # 成功但缺图
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    with pytest.raises(RuntimeError, match="image"):
        seedream_task.run_edit(args)


# ---------- seedream.py 门控 ----------

def _no_subprocess(*a, **k):
    raise AssertionError("subprocess.run must not be called in this branch")


class FakeEdit:
    """模拟 seedream_task.py：dry-run 打印 dry_run 摘要；真实提交写 --out。"""

    def __init__(self, rc=0, stderr="", write_out=True, dry_stdout='{"dry_run": true}', delay=0.0):
        self.rc = rc
        self.stderr = stderr
        self.write_out = write_out
        self.dry_stdout = dry_stdout
        self.delay = delay
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if "--dry-run" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=self.dry_stdout, stderr="")
        if self.delay:
            time.sleep(self.delay)
        if self.rc == 0 and self.write_out:
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(PNG + b"edited")
        return subprocess.CompletedProcess(argv, self.rc, stdout="out", stderr=self.stderr)

    @property
    def real_calls(self):
        return [c for c in self.calls if "--dry-run" not in c[0]]


def _edit(settings, cdir, image, prompt, out, lock, confirm=True):
    return asyncio.run(seedream.edit_image(settings, cdir, image, prompt, out, lock, confirm))


def test_seedream_error_shape():
    err = seedream.SeedreamError(409, "x")
    assert isinstance(err, Exception)
    assert err.status == 409 and err.detail == "x"


def test_disabled_501(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)  # 默认 enable_seedream_edit=False
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    for confirm in (False, True):  # 开关最优先，不看 confirm
        with pytest.raises(seedream.SeedreamError) as e:
            _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock(), confirm=confirm)
        assert e.value.status == 501
        assert e.value.detail == "Seedream edit is disabled."


def test_confirm_required_409(tmp_path, monkeypatch):
    """confirm 严格为 True 才放行（同 seedance payload.confirm 语义），位置在锁与入参校验之前。"""
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    lock = asyncio.Lock()
    for confirm in (False, "true", 1):
        with pytest.raises(seedream.SeedreamError) as e:
            _edit(settings, cdir, _png(tmp_path), "p", out, lock, confirm=confirm)
        assert e.value.status == 409, confirm
        assert e.value.detail == "confirmation required"
    # confirm 门控先于入参校验：图缺失也先报 confirmation required
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, tmp_path / "missing.png", "p", out, lock, confirm=False)
    assert e.value.status == 409 and e.value.detail == "confirmation required"


def test_bad_input_409(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    lock = asyncio.Lock()
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, tmp_path / "missing.png", "p", out, lock)
    assert e.value.status == 409 and e.value.detail == "invalid edit request"
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "   ", out, lock)
    assert e.value.status == 409 and e.value.detail == "invalid edit request"


def test_dryrun_failed_409(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setenv("ARK_API_KEY", SECRET)

    def dry_fails(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="Image is not a PNG file.")

    monkeypatch.setattr(subprocess, "run", dry_fails)
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock())
    assert e.value.status == 409
    assert e.value.detail == "invalid edit request"


def test_dryrun_no_marker_409(tmp_path, monkeypatch):
    """dry-run 退出码 0 但没打印 dry_run 标记，同样视为预检未通过。"""
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    fake = FakeEdit(dry_stdout="")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock())
    assert e.value.status == 409
    assert fake.real_calls == []  # 未进入真实提交


def test_missing_ark_key_503(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    fake = FakeEdit()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock())
    assert e.value.status == 503
    assert e.value.detail == "ARK_API_KEY not configured"
    assert fake.real_calls == [] and len(fake.calls) == 1  # dry-run 预检后才查 key


def test_edit_success(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True,
                             seedream_model="doubao-seedream-custom")
    cdir = tmp_path / "c"
    cdir.mkdir()
    image = _png(tmp_path)
    out = cdir / "edited.png"
    fake = FakeEdit()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("ARK_API_KEY", SECRET)

    result = _edit(settings, cdir, image, "戴上眼镜", out, asyncio.Lock())
    assert result == out
    assert out.read_bytes() == PNG + b"edited"

    # dry-run 预检 argv 契约：列表、cwd、120s、无 shell、env 继承、不带 --confirm-submit；
    # 带 --model，保证预检即真实请求形态
    (dargv, dkw), = fake.calls[:1]
    assert isinstance(dargv, list)
    assert dkw["cwd"] == cdir
    assert dkw["timeout"] == 120
    assert dkw.get("shell") is not True
    assert dkw.get("env") is None
    assert "--dry-run" in dargv
    assert "--confirm-submit" not in dargv
    assert dargv[dargv.index("--model") + 1] == "doubao-seedream-custom"

    # 真实提交 argv 契约：列表、cwd、960s、无 shell、env 继承、机械门控 flag
    (argv, kw), = fake.real_calls
    assert isinstance(argv, list)
    assert kw["cwd"] == cdir
    assert kw["timeout"] == 960
    assert kw.get("shell") is not True
    assert kw.get("env") is None
    assert "--confirm-submit" in argv
    assert argv[argv.index("--image") + 1] == str(image)
    assert argv[argv.index("--prompt") + 1] == "戴上眼镜"
    assert argv[argv.index("--out") + 1] == str(out)
    assert argv[argv.index("--model") + 1] == "doubao-seedream-custom"
    assert "--dry-run" not in argv

    # 幂等：out 已存在 → 409
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, image, "戴上眼镜", out, asyncio.Lock())
    assert e.value.status == 409 and e.value.detail == "already edited"


def test_edit_failure_502_sanitized(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    err = (f"Authorization: Bearer {SECRET}\n"
           f"request failed with api_key={SECRET}\n"
           "plain failure line")
    monkeypatch.setattr(subprocess, "run", FakeEdit(rc=1, stderr=err))
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock())
    assert e.value.status == 502
    detail = e.value.detail
    assert len(detail) <= 300
    assert SECRET not in detail
    assert "Authorization" not in detail
    assert "api_key" not in detail
    assert "plain failure line" in detail
    assert not out.exists()  # 未误产出


def test_edit_timeout_502(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    real = FakeEdit()

    def fake(argv, **kwargs):
        if "--dry-run" in argv:
            return real(argv, **kwargs)
        raise subprocess.TimeoutExpired(argv, 960)

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock())
    assert e.value.status == 502 and e.value.detail == "seedream task timed out"


def test_edit_runner_unavailable_502(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    real = FakeEdit()

    def fake(argv, **kwargs):
        if "--dry-run" in argv:
            return real(argv, **kwargs)
        raise OSError("interpreter gone")

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(seedream.SeedreamError) as e:
        _edit(settings, cdir, _png(tmp_path), "p", out, asyncio.Lock())
    assert e.value.status == 502 and e.value.detail == "seedream runner unavailable"


def test_concurrent_edit_single_task(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    out = cdir / "edited.png"
    monkeypatch.setenv("ARK_API_KEY", "sk-x")
    fake = FakeEdit(delay=0.3)
    monkeypatch.setattr(subprocess, "run", fake)
    lock = asyncio.Lock()

    async def run_both():
        return await asyncio.gather(
            seedream.edit_image(settings, cdir, _png(tmp_path), "p", out, lock, True),
            seedream.edit_image(settings, cdir, _png(tmp_path), "p", out, lock, True),
            return_exceptions=True,
        )

    results = asyncio.run(run_both())
    oks = [r for r in results if isinstance(r, Path)]
    errs = [r for r in results if isinstance(r, seedream.SeedreamError)]
    assert len(oks) == 1 and oks[0] == out
    assert len(errs) == 1 and errs[0].status == 409
    assert errs[0].detail == "already edited"
    assert len(fake.real_calls) == 1  # 一次确认 = 一个任务


# ---------- config ----------

def test_settings_seedream_defaults(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.enable_seedream_edit is False
    assert settings.seedream_model == "doubao-seedream-5-0-pro-260628"


def test_get_settings_seedream_env(monkeypatch):
    monkeypatch.setenv("ENABLE_SEEDREAM_EDIT", "true")
    monkeypatch.setenv("SEEDREAM_MODEL", "doubao-seedream-custom")
    settings = get_settings()
    assert settings.enable_seedream_edit is True
    assert settings.seedream_model == "doubao-seedream-custom"
