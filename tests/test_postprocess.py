"""后处理编排：HTTP 门控、MediaKit 场景映射、失败保留和并发限流。"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import mediakit, postprocess, storage
from app.main import create_app

PNG = b"\x89PNG\r\n\x1a\n"

OPTIONS_SUB = {"remove_subtitle": True, "remove_brand": False}
OPTIONS_BRAND = {"remove_subtitle": False, "remove_brand": True}


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    with TestClient(create_app(settings)) as c:
        yield settings, c


def _make_conv(settings, status="done", segments=False):
    """造会话：单段 = work/keyframes + work/prompt.txt；多段 = meta.segments + 段目录产物。"""
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    cid = meta["id"]
    cdir = settings.data_dir / cid
    if segments:
        segs = [
            {"index": 1, "start_s": 0.0, "end_s": 8.0,
             "keyframes": ["01.png"], "prompt": "段一提示词", "lines": ["台词。"]},
            {"index": 2, "start_s": 8.0, "end_s": 16.0,
             "keyframes": ["01.png", "02.png"], "prompt": "段二提示词", "lines": []},
        ]
        meta["segments"] = segs
        for seg in segs:
            segdir = cdir / "work" / "segments" / str(seg["index"])
            (segdir / "work" / "keyframes").mkdir(parents=True)
            for name in seg["keyframes"]:
                (segdir / "work" / "keyframes" / name).write_bytes(PNG)
            (segdir / "work" / "prompt.txt").write_text(seg["prompt"], encoding="utf-8")
    else:
        (cdir / "work" / "keyframes").mkdir(parents=True)
        for i in (1, 2):
            (cdir / "work" / "keyframes" / f"{i:02d}.png").write_bytes(PNG)
        (cdir / "work" / "prompt.txt").write_text("单段提示词", encoding="utf-8")
        meta["keyframes"] = ["01.png", "02.png"]
        meta["prompt"] = "单段提示词"
    meta["status"] = status
    (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cid


class FakeEdit:
    """桩 mediakit.erase_image：记录场景并写 out；按 fail 名单抛 MediaKitError。"""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = list(fail)

    async def __call__(self, settings, cdir, image, out, confirm, scenes):
        self.calls.append({
            "image": image.name, "out": out, "confirm": confirm, "scenes": scenes,
        })
        if image.name in self.fail:
            raise mediakit.MediaKitError(502, "stub failure")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(PNG + b"edited")
        return out


def _post(c, cid, options, confirm=True):
    return c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH,
                  json={"options": options, "confirm": confirm})


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


# ---------- 门控矩阵 ----------

def test_requires_auth(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")})
    cid = r.json()["id"]
    assert client.post(f"/api/conversations/{cid}/postprocess",
                       json={"options": OPTIONS_SUB, "confirm": True}).status_code == 401


def test_disabled_501(client):
    # 开关最优先（默认关）：不看 confirm、不看会话是否存在、不看选项
    r = client.post(f"/api/conversations/{'0' * 32}/postprocess", headers=AUTH,
                    json={"options": OPTIONS_SUB, "confirm": True})
    assert r.status_code == 501
    assert r.json() == {"detail": "MediaKit erase is disabled."}


def test_404_when_enabled(enabled):
    _, c = enabled
    r = _post(c, "0" * 32, OPTIONS_SUB)
    assert r.status_code == 404
    assert r.json() == {"detail": "not found"}


def test_confirm_required_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    for body in ({"options": OPTIONS_SUB, "confirm": False},
                 {"options": OPTIONS_SUB, "confirm": "true"}, {"options": OPTIONS_SUB, "confirm": 1}):
        r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH, json=body)
        assert r.status_code == 409, body
        assert r.json() == {"detail": "confirmation required"}


def test_no_options_422(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    empty = {"remove_subtitle": False, "remove_brand": False}
    for body in ({"options": {}, "confirm": True},
                 {"options": empty, "confirm": True}):
        r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH, json=body)
        assert r.status_code == 422, body
        assert r.json() == {"detail": "at least one option required"}


def test_options_non_bool_422(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH,
               json={"options": {"remove_subtitle": "yes"}, "confirm": True})
    assert r.status_code == 422
    assert r.json() == {"detail": "options must be booleans"}


@pytest.mark.parametrize(
    ("legacy_key", "legacy_value"),
    [("change_bg", True), ("change_bg", False), ("face_hold", True), ("face_hold", False)],
)
def test_known_stale_postprocess_options_require_refresh_without_side_effects(
    enabled, monkeypatch, legacy_key, legacy_value
):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    before = _file_snapshot(cdir)
    calls = []
    monkeypatch.setattr(
        postprocess.mediakit, "erase_image",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    r = _post(c, cid, {**OPTIONS_SUB, legacy_key: legacy_value})

    assert r.status_code == 409
    assert r.json() == {"detail": "页面版本已更新，请刷新页面后重试。"}
    assert calls == []
    assert _file_snapshot(cdir) == before


@pytest.mark.parametrize(
    ("options", "detail"),
    [
        ({**OPTIONS_SUB, "future_option": True}, "unknown options: future_option"),
        (
            {**OPTIONS_SUB, "face_hold": True, "future_option": True},
            "unknown options: face_hold, future_option",
        ),
    ],
)
def test_other_unknown_postprocess_option_remains_fail_closed(
    enabled, options, detail
):
    settings, c = enabled
    cid = _make_conv(settings)
    r = _post(c, cid, options)
    assert r.status_code == 422
    assert r.json() == {"detail": detail}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"confirm": True},
        {"options": OPTIONS_SUB},
        {"confirm": True, "options": OPTIONS_SUB, "unexpected": True},
    ],
)
def test_invalid_postprocess_top_level_shape_is_rejected_before_write_or_provider(
    enabled, monkeypatch, body
):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    before = _file_snapshot(cdir)
    calls = []
    monkeypatch.setattr(
        postprocess.mediakit, "erase_image",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    r = c.post(
        f"/api/conversations/{cid}/postprocess",
        headers=AUTH,
        json=body,
    )

    assert r.status_code == 422
    assert r.json() == {"detail": "invalid_postprocess_request"}
    assert calls == []
    assert _file_snapshot(cdir) == before


def test_not_done_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, status="queued")
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}


def test_already_running_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": OPTIONS_SUB, "frames": [], "error": None,
    })
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "already running"}


def test_artifacts_gone_409(enabled, monkeypatch):
    """status=done 但帧目录缺失 → 409 artifacts not ready，不写 meta.postprocess。"""
    settings, c = enabled
    cid = _make_conv(settings)
    (settings.data_dir / cid / "work" / "keyframes").rename(
        settings.data_dir / cid / "work" / "gone")
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}
    assert storage.load_meta(settings.data_dir, cid).get("postprocess") is None


def test_postprocess_cannot_start_after_generation_input_is_frozen(enabled):
    settings, client = enabled
    cid = _make_conv(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "queued",
            "client_request_id": "already-frozen",
        },
    )

    response = _post(client, cid, OPTIONS_SUB)

    assert response.status_code == 409
    assert response.json() == {"detail": "generation_already_started"}
    assert storage.load_meta(settings.data_dir, cid).get("postprocess") is None


# ---------- 单段全链路 ----------

def test_single_segment_full_chain(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    options = {"remove_subtitle": True, "remove_brand": True}
    r = _post(c, cid, options)
    assert r.status_code == 200
    assert r.json() == {"status": "running", "frames": []}  # 受理即返回，进度走 detail 轮询

    # 每帧按稳定顺序执行文字擦除、图标擦除；confirm 恒 True
    assert sorted(call["image"] for call in fake.calls) == ["01.png", "02.png"]
    for call in fake.calls:
        assert call["scenes"] == (mediakit.TEXT_SCENE, mediakit.ICON_SCENE)
        assert call["confirm"] is True

    # 产出：work/postprocessed/<帧名>.png（与源帧同目录层级）
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "postprocessed" / "02.png").is_file()

    # meta.postprocess done + frames；detail 中保留后处理状态
    meta = storage.load_meta(settings.data_dir, cid)
    pp = meta["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == options
    assert pp["frames"] == ["01.png", "02.png"]
    assert pp["error"] is None

    d = c.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert d["postprocess"] == pp
    assert d["postprocess_enabled"] is True


# ---------- 多段全链路 ----------

def test_multi_segment_full_chain(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200

    assert sorted(call["image"] for call in fake.calls) == ["01.png", "01.png", "02.png"]
    assert (cdir / "work" / "segments" / "1" / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "segments" / "2" / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "segments" / "2" / "work" / "postprocessed" / "02.png").is_file()
    assert not (cdir / "work" / "postprocessed").exists()

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    # frames 为全形路径（与 files 白名单同形），前端按 segments/N/work/postprocessed/ 前缀过滤展示
    assert pp["frames"] == [
        "segments/1/work/postprocessed/01.png",
        "segments/2/work/postprocessed/01.png",
        "segments/2/work/postprocessed/02.png",
    ]


# ---------- 擦除场景 ----------

def test_subtitle_option_maps_to_text_scene(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert all(call["scenes"] == (mediakit.TEXT_SCENE,) for call in fake.calls)


# ---------- 失败处理 ----------

def test_frame_failure_marks_failed_keeps_successes(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    monkeypatch.setattr(postprocess.mediakit, "erase_image", FakeEdit(fail=["02.png"]))

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200  # 受理成功；结果走 detail

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert "02.png" in pp["error"]
    assert pp["frames"] == ["01.png"]  # 已成功帧保留并记录
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()  # 已成功帧落盘保留
    assert not (cdir / "work" / "postprocessed" / "02.png").exists()

    d = c.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert d["postprocess"]["status"] == "failed"
    assert d["postprocess"]["error"] == pp["error"]


def test_rerun_skips_existing_outputs(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "postprocessed").mkdir()
    (cdir / "work" / "postprocessed" / "01.png").write_bytes(b"kept")
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200

    assert [call["image"] for call in fake.calls] == ["02.png"]  # 已有优化图的帧不重复扣费
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["frames"] == ["01.png", "02.png"]
    assert (cdir / "work" / "postprocessed" / "01.png").read_bytes() == b"kept"


# ---------- 并发：每会话一把锁 ----------

def test_concurrent_start_single_runner(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    locks = {}

    async def run_both():
        return await asyncio.gather(
            postprocess.start(settings, cid, {"options": OPTIONS_SUB, "confirm": True}, locks),
            postprocess.start(settings, cid, {"options": OPTIONS_SUB, "confirm": True}, locks),
            return_exceptions=True,
        )

    results = asyncio.run(run_both())
    oks = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, postprocess.PostprocessError)]
    assert len(oks) == 1 and len(errs) == 1
    assert errs[0].status == 409 and errs[0].detail == "already running"
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "running"


def test_options_lock_is_rechecked_inside_lock_without_writing(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    before = _file_snapshot(cdir)
    initial = storage.load_meta(settings.data_dir, cid)
    locked = {
        **initial,
        "postprocess": {
            "status": "done",
            "options": OPTIONS_BRAND,
            "frames": [],
            "error": None,
        },
    }
    reads = iter((initial, locked))
    monkeypatch.setattr(postprocess.storage, "load_meta", lambda *_args: next(reads))

    with pytest.raises(postprocess.PostprocessError) as caught:
        asyncio.run(postprocess.start(
            settings,
            cid,
            {"confirm": True, "options": OPTIONS_SUB},
            {},
        ))

    assert caught.value.status == 409
    assert caught.value.detail == {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }
    assert _file_snapshot(cdir) == before


# ---------- files 接口（前端取图路径） ----------

def test_files_endpoint_serves_postprocessed(enabled):
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    (cdir / "work" / "postprocessed").mkdir()
    (cdir / "work" / "postprocessed" / "01.png").write_bytes(b"opt")

    r = c.get(f"/api/conversations/{cid}/files/postprocessed/01.png", headers=AUTH)
    assert r.status_code == 200 and r.content == b"opt"
    r = c.get(f"/api/conversations/{cid}/files/segments/2/work/keyframes/01.png", headers=AUTH)
    assert r.status_code == 200 and r.content == PNG
    r = c.get(f"/api/conversations/{cid}/files/segments/2/work/postprocessed/01.png", headers=AUTH)
    assert r.status_code == 404  # 磁盘上不存在
    # 穿越一律 404：%2F 编码斜杠会绕过 HTTP 客户端的路径归一化，直击服务端白名单
    for name in ("segments/2/work/postprocessed/..%2Fkeyframes/01.png",
                 "postprocessed/..%2Fmeta.json"):
        r = c.get(f"/api/conversations/{cid}/files/{name}", headers=AUTH)
        assert r.status_code == 404, name


def test_rerun_different_options_409(enabled, monkeypatch):
    """上次 done 的 options 与本次不同 → 409（防旧产物贴新标签）；同选项重跑照常跳过已有图。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    other = {"remove_subtitle": True, "remove_brand": True}
    r = _post(c, cid, other)
    assert r.status_code == 409
    assert r.json() == {"detail": {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }}
    assert len(fake.calls) == 2  # 未产生新编辑

    # 同选项重跑：跳过已有优化图，正常 200 done
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert len(fake.calls) == 2  # 全部帧已存在，无新编辑
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "done"


def test_legacy_change_bg_in_meta_options_rerun_no_409(enabled, monkeypatch):
    """旧会话 meta 含已废弃 change_bg，新请求当前两键且共有键一致 →
    锁定比对只认当前 OPTION_KEYS 共有键，重跑不 409，正常覆盖为新两键 options。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    # 历史会话：meta 里存有 change_bg 时代的废弃键
    legacy = {"change_bg": True, "remove_subtitle": True, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "done", "options": legacy, "frames": ["01.png", "02.png"], "error": None,
    })
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    for name in ("01.png", "02.png"):
        (cdir / "work" / "postprocessed" / name).write_bytes(PNG + b"legacy")

    # 新请求两键与旧状态中的当前键一致；change_bg 忽略，不比
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert len(fake.calls) == 0  # 已有优化图，全部跳过

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == OPTIONS_SUB  # 覆盖为两键契约

    # 共有键真变了仍然 409：兼容比对不放松锁定
    r = _post(c, cid, OPTIONS_BRAND)
    assert r.status_code == 409
    assert r.json() == {"detail": {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }}


def test_legacy_pure_change_bg_rerun_clears_artifacts_and_reedits(enabled, monkeypatch):
    """旧会话「只勾 change_bg」（当前两键全 False 的纯废弃形态）→ 放行重跑：
    旧产物清除、全帧强制重编辑（防旧 change_bg 产物贴新选项标签）。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": ["01.png", "02.png"], "error": "x",
    })
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    for name in ("01.png", "02.png"):
        (cdir / "work" / "postprocessed" / name).write_bytes(PNG + b"legacy")

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert len(fake.calls) == 2  # 旧产物已清，两帧全部重新编辑
    assert not (cdir / "work" / "postprocessed" / "01.png").exists() or \
        (cdir / "work" / "postprocessed" / "01.png").read_bytes() == PNG + b"edited"


def test_legacy_pure_change_bg_any_new_option_no_409(enabled, monkeypatch):
    """纯废弃形态下任何合法新选项（≥1 True）都不 409——永久死锁回归。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": [], "error": "x",
    })
    r = _post(c, cid, OPTIONS_BRAND)
    assert r.status_code == 200


def test_legacy_pure_change_bg_multi_segment_clears_artifacts(enabled, monkeypatch):
    """多段会话纯废弃形态重跑 → 各段 postprocessed 旧产物同样清除。"""
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": ["segments/1/work/postprocessed/01.png"], "error": "x",
    })
    for n in (1, 2):
        d = cdir / "work" / "segments" / str(n) / "work" / "postprocessed"
        d.mkdir(parents=True)
        (d / "01.png").write_bytes(PNG + b"legacy")

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    for n in (1, 2):
        d = cdir / "work" / "segments" / str(n) / "work" / "postprocessed"
        assert not (d / "01.png").exists() or (d / "01.png").read_bytes() == PNG + b"edited"
    assert len(fake.calls) == 3  # 段1两帧 + 段2一帧全量重编辑


# ---------- 并行提交：进程级信号量限流与失败语义 ----------

class SlowEdit:
    """慢速桩：记录 MediaKit 帧级并发峰值；按 fail 名单抛错。"""

    def __init__(self, fail=(), delay=0.05):
        self.fail = list(fail)
        self.delay = delay  # float（统一延时）或 {帧名: 秒}（打乱完成顺序）
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, settings, cdir, image, out, confirm, scenes):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append({"image": image.name, "scenes": scenes})
        try:
            if isinstance(self.delay, dict):
                await asyncio.sleep(self.delay.get(image.name, 0.0))
            else:
                await asyncio.sleep(self.delay)
            if image.name in self.fail:
                raise mediakit.MediaKitError(502, "stub failure")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(PNG + b"edited")
            return out
        finally:
            self.active -= 1


def _add_frames(settings, cid, names):
    for name in names:
        (settings.data_dir / cid / "work" / "keyframes" / name).write_bytes(PNG)


def test_parallel_edits_respect_process_semaphore(tmp_path, monkeypatch):
    """5 帧、进程并发上限 2：编辑并行提交且活跃数峰值恰为 2（信号量限流）。"""
    settings = make_settings(tmp_path, enable_mediakit_erase=True, mediakit_concurrency=2)
    cid = _make_conv(settings)
    _add_frames(settings, cid, ["03.png", "04.png", "05.png"])
    slow = SlowEdit(delay=0.05)
    monkeypatch.setattr(postprocess.mediakit, "erase_image", slow)

    with TestClient(create_app(settings)) as c:
        assert _post(c, cid, OPTIONS_SUB).status_code == 200

    assert slow.max_active == 2
    assert slow.max_active <= settings.mediakit_concurrency
    # 帧到达顺序不定（线程池并发读尺寸），断言按集合：每帧恰一次
    assert sorted(call["image"] for call in slow.calls) == \
        ["01.png", "02.png", "03.png", "04.png", "05.png"]
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["frames"] == ["01.png", "02.png", "03.png", "04.png", "05.png"]


def test_parallel_frame_failure_waits_for_rest(enabled, monkeypatch):
    """并发下任一帧失败 → 整体 failed（error 指明帧名），其余帧照常跑完、成功帧全保留；
    frames 终序为目标顺序，与完成顺序无关。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    _add_frames(settings, cid, ["03.png"])
    # 打乱完成顺序：01 最慢、03 次之、02 最快且失败
    slow = SlowEdit(fail=["02.png"], delay={"01.png": 0.08, "02.png": 0.01, "03.png": 0.04})
    monkeypatch.setattr(postprocess.mediakit, "erase_image", slow)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert "02.png" in pp["error"]
    assert pp["frames"] == ["01.png", "03.png"]  # 完成顺序 02→03→01，终序按目标顺序
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "postprocessed" / "03.png").is_file()
    assert not (cdir / "work" / "postprocessed" / "02.png").exists()


# ---------- 取消：父任务取消写 failed 终态 ----------

def test_run_task_cancelled_writes_failed(tmp_path, monkeypatch):
    """父任务被取消（uvicorn graceful shutdown）：CancelledError 是 BaseException，run_task 须在
    继续传播前把 meta.postprocess 写成 failed——否则永久 running、start 永久 409 拒重跑。"""
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": OPTIONS_SUB, "frames": [], "error": None,
    })

    async def hang(*a, **k):
        await asyncio.Event().wait()  # 被取消时才结束的挂起桩

    monkeypatch.setattr(postprocess, "_edit_one", hang)
    sem = asyncio.Semaphore(10)

    async def drive():
        task = asyncio.create_task(postprocess.run_task(settings, cid, OPTIONS_SUB, sem))
        await asyncio.sleep(0.05)  # 让出至进入 gather
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert "cancelled" in pp["error"]
    assert pp["options"] == OPTIONS_SUB
