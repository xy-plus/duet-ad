"""seedream_task.py（argv 脚本）与 seedream.py（门控层）的测试。"""

import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import pytest

from conftest import make_settings

from app import seedream, seedream_task
from app.config import get_settings

PNG = b"\x89PNG\r\n\x1a\n"
SECRET = "sk-live-abcdef123456"
EDIT_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"


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
    assert args.request_timeout == 300  # 实测同步响应 60s+，默认留余量
    assert not hasattr(args, "poll_interval")  # 同步为唯一形态，轮询参数已删
    assert not hasattr(args, "poll_timeout")
    assert not hasattr(args, "state_file")  # 无消费方，轮询时代残余已删


def test_edit_bad_request_timeout(tmp_path):
    png = _png(tmp_path)
    argv = ["edit", "--image", str(png), "--prompt", "p",
            "--out", str(tmp_path / "o.png"), "--request-timeout", "-1"]
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


def test_dry_run_size_in_summary(tmp_path, capsys):
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--size", "1440x2560", "--dry-run"]
    )
    assert seedream_task.run_edit(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["size"] == "1440x2560"


def test_dry_run_no_size_omits_summary_key(tmp_path, capsys):
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--dry-run"]
    )
    assert seedream_task.run_edit(args) == 0
    assert "size" not in json.loads(capsys.readouterr().out)


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


# ---------- 真实提交全链路（monkeypatch urllib，同步 JSON 契约） ----------

class FakeResponse:
    """JSON 响应假实现。"""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
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
    return calls


def test_full_chain_success_b64(tmp_path, monkeypatch, capsys):
    """实测契约：单次同步 POST，JSON body（image 为 data URI 数组），b64_json 解码落盘。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    out = tmp_path / "edited.png"
    src = PNG + b"fake-image-data"
    img = PNG + b"edited-image-bytes"
    b64 = base64.b64encode(img).decode("ascii")
    calls = _fake_ark(monkeypatch, [
        {"model": "doubao-seedream-5-0-pro-260628", "created": 1758200023,
         "data": [{"b64_json": b64}]},
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "戴上眼镜",
         "--out", str(out), "--confirm-submit"]
    )
    assert seedream_task.run_edit(args) == 0

    # 唯一一次请求：POST images/generations，同步返回，无轮询
    (req, kw), = calls
    assert req.full_url == EDIT_URL
    assert req.get_header("Authorization") == f"Bearer {SECRET}"
    assert kw["timeout"] == 300
    assert req.get_header("Content-type") == "application/json"

    # JSON body：中文 ensure_ascii=False、image 为 data URI 字符串数组
    assert "戴上眼镜".encode("utf-8") in req.data
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "doubao-seedream-5-0-pro-260628"
    assert body["prompt"] == "戴上眼镜"
    assert body["image"] == [f"data:image/png;base64,{base64.b64encode(src).decode('ascii')}"]
    assert body["response_format"] == "b64_json"
    assert body["watermark"] is False
    assert "size" not in body  # 不传 size：请求体无该键（模型默认 2048 方形）

    # b64 解码落盘 + stdout 打响应摘要（不塞图字节）
    assert out.read_bytes() == img
    assert "image_bytes" in capsys.readouterr().out


def test_full_chain_size_in_request_body(tmp_path, monkeypatch, capsys):
    """--size "WxH" 时请求体含 size 键（真实提交透传，保持输入帧宽高比）。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    out = tmp_path / "edited.png"
    img = PNG + b"edited-image-bytes"
    b64 = base64.b64encode(img).decode("ascii")
    calls = _fake_ark(monkeypatch, [
        {"model": "doubao-seedream-5-0-pro-260628", "created": 1758200023,
         "data": [{"b64_json": b64}]},
    ])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "戴上眼镜",
         "--out", str(out), "--confirm-submit", "--size", "1440x2560"]
    )
    assert seedream_task.run_edit(args) == 0
    (req, _), = calls
    body = json.loads(req.data.decode("utf-8"))
    assert body["size"] == "1440x2560"
    assert out.read_bytes() == img


@pytest.mark.parametrize("payload", [
    {"model": "doubao-seedream-5-0-pro-260628", "data": []},
    {"model": "doubao-seedream-5-0-pro-260628", "data": [{}]},
    {"model": "doubao-seedream-5-0-pro-260628", "data": [{"b64_json": ""}]},
    {"model": "doubao-seedream-5-0-pro-260628", "data": [{"url": "https://tos.example/edited.png"}]},
    {},
])
def test_missing_or_empty_b64_fails(tmp_path, monkeypatch, payload):
    """b64_json 缺失或为空 → RuntimeError（契约恒 b64_json，无 url 兜底）。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    calls = _fake_ark(monkeypatch, [payload])
    args = seedream_task.parse_args(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    with pytest.raises(RuntimeError, match="b64_json"):
        seedream_task.run_edit(args)
    assert len(calls) == 1  # 无兜底下载请求


def test_bad_b64_exit_1_no_out(tmp_path, monkeypatch, capsys):
    """非法 b64 → 硬错误退出码 1，不落盘、不泄密钥。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    out = tmp_path / "o.png"
    _fake_ark(monkeypatch, [{"data": [{"b64_json": "!!not-base64!!"}]}])
    rc = seedream_task.cli(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(out), "--confirm-submit"]
    )
    assert rc == 1
    err_text = capsys.readouterr().err
    assert "base64" in err_text
    assert SECRET not in err_text
    assert not out.exists()


def test_http_error_exit_nonzero_no_key_leak(tmp_path, monkeypatch, capsys):
    """HTTP 错误走既有报错路径：退出码 1，stderr 不含密钥字面值。"""
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    err = urllib.error.HTTPError(
        EDIT_URL, 400, "bad request", {}, io.BytesIO(b'{"error": "bad request"}')
    )
    _fake_ark(monkeypatch, [err])
    rc = seedream_task.cli(
        ["edit", "--image", str(_png(tmp_path)), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--confirm-submit"]
    )
    assert rc == 1
    err_text = capsys.readouterr().err
    assert "Ark HTTP 400" in err_text
    assert SECRET not in err_text


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


def _edit(settings, cdir, image, prompt, out, lock, confirm=True, size=""):
    return asyncio.run(
        seedream.edit_image(settings, cdir, image, prompt, out, lock, confirm, size=size)
    )


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

    # 真实提交 argv 契约：列表、cwd、600s（300 请求 + 余量）、无 shell、env 继承、机械门控 flag
    (argv, kw), = fake.real_calls
    assert isinstance(argv, list)
    assert kw["cwd"] == cdir
    assert kw["timeout"] == 600
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


def test_edit_size_in_argv(tmp_path, monkeypatch):
    """size 非空：dry-run 预检与真实提交 argv 都带 --size；size 空串（默认）：argv 不带 --size。"""
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    image = _png(tmp_path)
    fake = FakeEdit()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("ARK_API_KEY", SECRET)

    _edit(settings, cdir, image, "p", cdir / "a.png", asyncio.Lock(), size="1440x2560")
    for argv, _ in fake.calls[:2]:  # dry-run + 真实提交
        assert argv[argv.index("--size") + 1] == "1440x2560"

    _edit(settings, cdir, image, "p", cdir / "b.png", asyncio.Lock())
    assert all("--size" not in argv for argv, _ in fake.calls[2:])


def test_edit_relative_paths_resolved(tmp_path, monkeypatch):
    """相对 cdir/image/out（生产 data_dir=相对 "data" 的形态）→ 入口统一转绝对，
    子进程 cwd 与 argv 路径不再错位（回归：生产 invalid edit request）。"""
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cdir = tmp_path / "c"
    cdir.mkdir()
    image = _png(tmp_path)
    out = cdir / "edited.png"
    fake = FakeEdit()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("ARK_API_KEY", SECRET)
    monkeypatch.chdir(tmp_path)  # 服务 cwd 形态

    rel_cdir = Path("c")
    rel_image = Path("in.png")
    rel_out = Path("c") / "edited.png"
    result = _edit(settings, rel_cdir, rel_image, "戴上眼镜", rel_out, asyncio.Lock())
    assert result == cdir / "edited.png"  # 返回绝对
    assert out.read_bytes() == PNG + b"edited"
    (dargv, dkw), = fake.calls[:1]
    assert Path(dargv[dargv.index("--image") + 1]).is_absolute()
    assert Path(dargv[dargv.index("--out") + 1]).is_absolute()
    assert Path(dkw["cwd"]).is_absolute()


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
        raise subprocess.TimeoutExpired(argv, 600)

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


def test_settings_seedream_concurrency_default(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.seedream_concurrency == 10


def test_get_settings_seedream_concurrency_env(monkeypatch):
    monkeypatch.setenv("SEEDREAM_CONCURRENCY", "3")
    assert get_settings().seedream_concurrency == 3
