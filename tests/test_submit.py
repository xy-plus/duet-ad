import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import seedance, storage
from app.main import create_app

PROMPT = "把房间打扫干净的视频"
REVIEWED = {
    "model": "doubao-seedance-2-0-260128",
    "ratio": "9:16",
    "duration": 15,
    "resolution": "720p",
    "generate_audio": True,
    "watermark": False,
    "content": [
        {"type": "text", "text": PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,AAAA"},
            "role": "reference_image",
        },
    ],
}


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(tmp_path, enable_seedance_submit=True)
    with TestClient(create_app(settings)) as c:
        yield settings, c


def _make_conv(settings, status="done", has_video=False, with_work=True):
    """造一个会话；默认 status=done 且 work/ 下有 prompt+keyframes+评审 payload。"""
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    cid = meta["id"]
    cdir = settings.data_dir / cid
    meta["status"] = status
    if has_video:
        meta["has_video"] = True
    (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if with_work:
        (cdir / "work" / "keyframes").mkdir(parents=True)
        # 与 pipeline 真实产物布局一致：关键帧 PNG 与 contact_sheet/manifest 同目录
        (cdir / "work" / "keyframes" / "01_keyframe_0.330s.png").write_bytes(b"img1")
        (cdir / "work" / "keyframes" / "contact_sheet.jpg").write_bytes(b"sheet")
        (cdir / "work" / "keyframes" / "manifest.json").write_text("{}", encoding="utf-8")
        (cdir / "work" / "seedance_prompt.txt").write_text(PROMPT, encoding="utf-8")
        (cdir / "work" / "api_request.json").write_text(
            json.dumps(REVIEWED, ensure_ascii=False), encoding="utf-8"
        )
    return cid


def _no_subprocess(*a, **k):
    raise AssertionError("subprocess.run must not be called in this branch")


class FakeSubmit:
    """模拟 seedance_task.py：dry-run 写 recheck payload；真实提交写 task.json + generated.mp4。"""

    def __init__(self, rebuilt=None, rc=0, stderr="", task_id="task-123"):
        self.rebuilt = REVIEWED if rebuilt is None else rebuilt
        self.rc = rc
        self.stderr = stderr
        self.task_id = task_id
        self.calls = []

    def __call__(self, argv, **kwargs):
        cwd = Path(kwargs["cwd"])
        self.calls.append((list(argv), kwargs))
        if "--dry-run" in argv:
            out = cwd / argv[argv.index("--payload-out") + 1]
            out.write_text(json.dumps(self.rebuilt, ensure_ascii=False), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="dry", stderr="")
        if self.rc == 0:
            (cwd / "work" / "task.json").write_text(
                json.dumps({"id": self.task_id, "status": "succeeded"}), encoding="utf-8"
            )
            (cwd / "generated.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(argv, self.rc, stdout="out", stderr=self.stderr)

    @property
    def real_calls(self):
        return [c for c in self.calls if "--dry-run" not in c[0]]


# ---------- 矩阵 1：开关关闭 → 501（不看 confirm、不看会话是否存在） ----------

def test_disabled_501(client):
    r = client.post(f"/api/conversations/{'0' * 32}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 501
    assert r.json() == {"detail": "Seedance submission is disabled."}
    r = client.post(f"/api/conversations/{'0' * 32}/submit", headers=AUTH, json={})
    assert r.status_code == 501


def test_requires_auth(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    assert c.post(f"/api/conversations/{cid}/submit", json={"confirm": True}).status_code == 401


# ---------- 矩阵 2：会话不存在 → 404 ----------

def test_404_when_enabled(enabled):
    _, c = enabled
    r = c.post(f"/api/conversations/{'0' * 32}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 404
    assert r.json() == {"detail": "not found"}


# ---------- 矩阵 3：confirm 不是 true → 409 ----------

def test_confirm_required_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    for body in ({}, {"confirm": False}, {"confirm": "true"}, {"confirm": 1}):
        r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body)
        assert r.status_code == 409, body
        assert r.json() == {"detail": "confirmation required"}


# ---------- 矩阵 4：status != done → 409 ----------

def test_not_done_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, status="queued")
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}


# ---------- 矩阵 5：已 has_video → 409 ----------

def test_already_submitted_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, has_video=True)
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 409
    assert r.json() == {"detail": "already submitted"}


# ---------- 矩阵 6：dry-run 复核不一致 → 409 ----------

_CHANGED_PAYLOADS = [
    {**REVIEWED, "ratio": "16:9"},                       # 标量被改
    {**REVIEWED, "generate_audio": False},               # 布尔被改
    {**REVIEWED, "content": [dict(REVIEWED["content"][0], text="改过的 prompt"),
                             REVIEWED["content"][1]]},   # text 被改
    {**REVIEWED, "content": [*REVIEWED["content"],
                             REVIEWED["content"][1]]},   # content 长度变
]


@pytest.mark.parametrize("rebuilt", _CHANGED_PAYLOADS)
def test_payload_changed_409(enabled, monkeypatch, rebuilt):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeSubmit(rebuilt=rebuilt)
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("ARK_API_KEY", "sk-test")
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 409
    assert r.json() == {"detail": "payload changed since review"}
    assert fake.real_calls == []  # 未进入真实提交
    assert len(fake.calls) == 1   # 只跑了 dry-run 复核


def test_payload_review_missing_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, with_work=False)  # 无 api_request.json
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    monkeypatch.setenv("ARK_API_KEY", "sk-test")
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 409
    assert r.json() == {"detail": "payload changed since review"}


def test_payload_dryrun_failed_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)

    def dry_fails(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="Prompt is empty.")

    monkeypatch.setattr(subprocess, "run", dry_fails)
    monkeypatch.setenv("ARK_API_KEY", "sk-test")
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 409
    assert r.json() == {"detail": "payload changed since review"}


# ---------- 矩阵 7：无 ARK_API_KEY → 503 ----------

def test_missing_ark_key_503(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeSubmit()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 503
    assert r.json() == {"detail": "ARK_API_KEY not configured"}
    assert fake.real_calls == [] and len(fake.calls) == 1  # dry-run 复核后才查 key


# ---------- 矩阵 8a：成功 → 200 + meta 落盘 + 契约不破 ----------

def test_submit_success_200(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeSubmit(task_id="cgt-abc123")
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("ARK_API_KEY", "sk-test-secret")

    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 200
    assert r.json() == {"status": "succeeded", "video": "generated.mp4"}

    # 真实提交 argv 契约：列表、cwd、1800s、无 shell、env 继承
    (argv, kw), = fake.real_calls
    assert isinstance(argv, list)
    assert kw["cwd"] == cdir
    assert kw["timeout"] == 1800
    assert kw.get("shell") is not True
    assert kw.get("env") is None
    for flag in ("--confirm-submit", "--wait",
                 "--state-file", "work/task.json",
                 "--download", "generated.mp4",
                 "--prompt-file", "work/seedance_prompt.txt",
                 "--ref-images", "work/keyframes/01_keyframe_0.330s.png",
                 "--model", REVIEWED["model"]):
        assert flag in argv
    # 回归：keyframes/ 里的非关键帧产物不得进 ref-images
    assert not any("contact_sheet" in a or "manifest" in a for a in argv)

    # meta 落盘：has_video / submitted_at / task_id；密钥不进任何写盘文件
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["has_video"] is True
    assert meta["task_id"] == "cgt-abc123"
    assert meta["submitted_at"]
    assert "sk-test-secret" not in (cdir / "meta.json").read_text(encoding="utf-8")
    assert not (cdir / "work" / "recheck_payload.json").exists()  # 复核临时文件已清理

    # detail 冻结契约仍 13 字段（meta 新增字段不外泄），has_video 按文件翻真
    d = c.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert len(d) == 13
    assert "task_id" not in d
    assert d["has_video"] is True

    # 幂等：再提交一次 → 409
    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 409
    assert r.json() == {"detail": "already submitted"}


# ---------- 矩阵 8b：脚本非零 → 502，detail 脱敏 ----------

def test_submit_failure_502_sanitized(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    secret = "sk-live-abcdef123456"
    monkeypatch.setenv("ARK_API_KEY", secret)
    err = (f"Authorization: Bearer {secret}\n"
           f"request failed with api_key={secret}\n"
           "plain failure line")
    fake = FakeSubmit(rc=1, stderr=err)
    monkeypatch.setattr(subprocess, "run", fake)

    r = c.post(f"/api/conversations/{cid}/submit", headers=AUTH, json={"confirm": True})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert len(detail) <= 300
    assert secret not in detail
    assert "Authorization" not in detail
    assert "api_key" not in detail
    assert "plain failure line" in detail
    assert not storage.load_meta(settings.data_dir, cid).get("has_video")  # 未误标记


# ---------- 矩阵 9：并发双击 → 锁内重查，只产生一个任务 ----------

def test_concurrent_submit_single_task(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedance_submit=True)
    cid = _make_conv(settings)
    monkeypatch.setenv("ARK_API_KEY", "sk-x")
    real_calls = []

    def fake(argv, **kwargs):
        cwd = Path(kwargs["cwd"])
        if "--dry-run" in argv:
            out = cwd / argv[argv.index("--payload-out") + 1]
            out.write_text(json.dumps(REVIEWED, ensure_ascii=False), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")
        real_calls.append(argv)
        time.sleep(0.3)  # 让第二个请求抵达锁
        (cwd / "work" / "task.json").write_text(json.dumps({"id": "t-1"}), encoding="utf-8")
        (cwd / "generated.mp4").write_bytes(b"v")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake)
    locks = {}

    async def run_both():
        return await asyncio.gather(
            seedance.submit(settings, cid, {"confirm": True}, locks),
            seedance.submit(settings, cid, {"confirm": True}, locks),
            return_exceptions=True,
        )

    results = asyncio.run(run_both())
    oks = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, seedance.SubmitError)]
    assert len(oks) == 1 and oks[0]["status"] == "succeeded"
    assert len(errs) == 1 and errs[0].status == 409
    assert len(real_calls) == 1  # 一次确认 = 一个任务
    assert storage.load_meta(settings.data_dir, cid)["has_video"] is True
