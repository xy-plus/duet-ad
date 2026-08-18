"""T5b 后处理编排：HTTP 门控矩阵、全链路（桩 edit_image）、face_hold 条件指令、失败处理、
并行提交（信号量限流）、输出尺寸。"""

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import postprocess, seedream, storage
from app.main import create_app

PNG = b"\x89PNG\r\n\x1a\n"

FACE_HOLD = ("如果图片中含有人脸：将图片中的人物改为用手捂住脸的造型。"
             "如果图片中不含人脸：跳过捂脸处理，仅执行其余修改。")
REMOVE_SUBTITLE = "移除图片中的所有字幕、水印和贴纸元素，其余（尺寸、内容等）保持不变"
REMOVE_BRAND = ("图片中的所有品牌标志、logo、商标等版权元素改为不侵权的类似视觉效果的等效物，"
                "其余（尺寸、内容等）保持不变")

OPTIONS_SUB = {"face_hold": False, "remove_subtitle": True, "remove_brand": False}
FACE_ONLY = {"face_hold": True, "remove_subtitle": False, "remove_brand": False}


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
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
    """桩 seedream.edit_image：记录调用（含 size）并写 out；按 fail 名单抛 SeedreamError。"""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = list(fail)

    async def __call__(self, settings, cdir, image, prompt, out, confirm, size=""):
        self.calls.append({
            "image": image.name, "prompt": prompt, "out": out, "confirm": confirm, "size": size,
        })
        if image.name in self.fail:
            raise seedream.SeedreamError(502, "stub failure")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(PNG + b"edited")
        return out


def _post(c, cid, options, confirm=True):
    return c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH,
                  json={"options": options, "confirm": confirm})


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
    assert r.json() == {"detail": "Seedream edit is disabled."}


def test_404_when_enabled(enabled):
    _, c = enabled
    r = _post(c, "0" * 32, OPTIONS_SUB)
    assert r.status_code == 404
    assert r.json() == {"detail": "not found"}


def test_confirm_required_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    monkeypatch.setattr(postprocess.seedream, "edit_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    for body in ({}, {"options": OPTIONS_SUB}, {"options": OPTIONS_SUB, "confirm": False},
                 {"options": OPTIONS_SUB, "confirm": "true"}, {"options": OPTIONS_SUB, "confirm": 1}):
        r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH, json=body)
        assert r.status_code == 409, body
        assert r.json() == {"detail": "confirmation required"}


def test_no_options_422(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    empty = {"face_hold": False, "remove_subtitle": False, "remove_brand": False}
    for body in ({"confirm": True},
                 {"options": {}, "confirm": True},
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


def test_not_done_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, status="queued")
    monkeypatch.setattr(postprocess.seedream, "edit_image",
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
    monkeypatch.setattr(postprocess.seedream, "edit_image",
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
    monkeypatch.setattr(postprocess.seedream, "edit_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}
    assert storage.load_meta(settings.data_dir, cid).get("postprocess") is None


# ---------- 单段全链路 ----------

def test_single_segment_full_chain(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    options = {"face_hold": False,
               "remove_subtitle": True, "remove_brand": True}
    r = _post(c, cid, options)
    assert r.status_code == 200
    assert r.json() == {"status": "running", "frames": []}  # 受理即返回，进度走 detail 轮询

    # 每帧一条合并指令（分号连接），confirm 恒 True（到达顺序不定：线程池并发读尺寸）
    assert sorted(call["image"] for call in fake.calls) == ["01.png", "02.png"]
    expected = f"{REMOVE_SUBTITLE}；{REMOVE_BRAND}"
    for call in fake.calls:
        assert call["prompt"] == expected
        assert call["confirm"] is True

    # 产出：work/postprocessed/<帧名>.png（与源帧同目录层级）
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "postprocessed" / "02.png").is_file()

    # meta.postprocess done + frames；detail 契约 16 字段
    meta = storage.load_meta(settings.data_dir, cid)
    pp = meta["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == options
    assert pp["frames"] == ["01.png", "02.png"]
    assert pp["error"] is None

    d = c.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert d["postprocess"] == pp
    assert d["postprocess_enabled"] is True
    assert len(d) == 16


# ---------- 多段全链路 ----------

def test_multi_segment_full_chain(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

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


# ---------- face_hold 条件指令 ----------

def test_face_hold_conditional_instruction_all_frames_single(enabled, monkeypatch):
    """face_hold 单独勾选：所有帧都发编辑请求，指令为条件式文案（不预判人脸、不跳帧）。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 200

    # 每帧一条条件指令（含人脸遮挡、无人脸保持原样），无帧被跳过
    assert sorted(call["image"] for call in fake.calls) == ["01.png", "02.png"]
    for call in fake.calls:
        assert call["prompt"] == FACE_HOLD

    # 不再向后追加动作线（条件动作行由流水线机械加进 prompt，见 pipeline）
    prompt = (cdir / "work" / "prompt.txt").read_text(encoding="utf-8")
    assert prompt == "单段提示词"
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["prompt"] == "单段提示词"
    assert meta["postprocess"]["frames"] == ["01.png", "02.png"]


def test_face_hold_merges_conditional_first(enabled, monkeypatch):
    """face_hold 与其他选项合并：条件句放最前，其余选项照旧分号连接。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    options = {"face_hold": True, "remove_subtitle": True, "remove_brand": False}
    r = _post(c, cid, options)
    assert r.status_code == 200
    assert len(fake.calls) == 2
    assert all(call["prompt"] == f"{FACE_HOLD}；{REMOVE_SUBTITLE}" for call in fake.calls)


def test_face_hold_all_frames_multi_segment(enabled, monkeypatch):
    """多段模式 face_hold：每段每帧都编辑、指令条件式；段 prompt 不被追加动作线。"""
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 200

    assert sorted(call["image"] for call in fake.calls) == ["01.png", "01.png", "02.png"]
    assert all(call["prompt"] == FACE_HOLD for call in fake.calls)

    seg1_p = (cdir / "work" / "segments" / "1" / "work" / "prompt.txt").read_text(encoding="utf-8")
    seg2_p = (cdir / "work" / "segments" / "2" / "work" / "prompt.txt").read_text(encoding="utf-8")
    assert seg1_p == "段一提示词"
    assert seg2_p == "段二提示词"
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["segments"][0]["prompt"] == seg1_p
    assert meta["segments"][1]["prompt"] == seg2_p
    assert meta["postprocess"]["frames"] == [
        "segments/1/work/postprocessed/01.png",
        "segments/2/work/postprocessed/01.png",
        "segments/2/work/postprocessed/02.png",
    ]


def test_no_face_hold_instruction_omits_face_hold(enabled, monkeypatch):
    """未勾选 face_hold：指令不含条件句，其余选项照常合并。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert all(call["prompt"] == REMOVE_SUBTITLE for call in fake.calls)


# ---------- 失败处理 ----------

def test_frame_failure_marks_failed_keeps_successes(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    monkeypatch.setattr(postprocess.seedream, "edit_image", FakeEdit(fail=["02.png"]))

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
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200

    assert [call["image"] for call in fake.calls] == ["02.png"]  # 已有优化图的帧不重复扣费
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["frames"] == ["01.png", "02.png"]
    assert (cdir / "work" / "postprocessed" / "01.png").read_bytes() == b"kept"


# ---------- 并发：每会话一把锁 ----------

def test_concurrent_start_single_runner(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_seedream_edit=True)
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
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    other = {"face_hold": False, "remove_subtitle": True, "remove_brand": True}
    r = _post(c, cid, other)
    assert r.status_code == 409
    assert r.json() == {"detail": "options changed since last run"}
    assert len(fake.calls) == 2  # 未产生新编辑

    # 同选项重跑：跳过已有优化图，正常 200 done
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert len(fake.calls) == 2  # 全部帧已存在，无新编辑
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "done"


def test_legacy_change_bg_in_meta_options_rerun_no_409(enabled, monkeypatch):
    """旧会话 meta 存四键 options（含已废弃 change_bg），新请求三键且共有键一致 →
    锁定比对只认当前 OPTION_KEYS 共有键，重跑不 409，正常覆盖为新三键 options。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    # 历史会话：meta 里存的是四键 options（change_bg 时代产物）
    legacy = {"change_bg": True, "face_hold": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "done", "options": legacy, "frames": ["01.png", "02.png"], "error": None,
    })
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    for name in ("01.png", "02.png"):
        (cdir / "work" / "postprocessed" / name).write_bytes(PNG + b"legacy")

    # 新请求三键（face_hold 等共有键与旧一致；change_bg 忽略，不比）
    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 200
    assert len(fake.calls) == 0  # 已有优化图，全部跳过

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == FACE_ONLY  # 覆盖为三键契约

    # 共有键真变了仍然 409：兼容比对不放松锁定
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "options changed since last run"}


def test_legacy_pure_change_bg_rerun_clears_artifacts_and_reedits(enabled, monkeypatch):
    """旧会话「只勾 change_bg」（当前三键全 False 的纯废弃形态）→ 放行重跑：
    旧产物清除、全帧强制重编辑（防旧 change_bg 产物贴新三键标签）。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    legacy = {"change_bg": True, "face_hold": False, "remove_subtitle": False, "remove_brand": False}
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
    """纯废弃形态下任何合法新三键（≥1 True）都不 409——永久死锁回归。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)
    legacy = {"change_bg": True, "face_hold": False, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": [], "error": "x",
    })
    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 200


def test_legacy_pure_change_bg_multi_segment_clears_artifacts(enabled, monkeypatch):
    """多段会话纯废弃形态重跑 → 各段 postprocessed 旧产物同样清除。"""
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    legacy = {"change_bg": True, "face_hold": False, "remove_subtitle": False, "remove_brand": False}
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


# ---------- 输出尺寸：_fit_size 与 run_task 透传 ----------

def test_fit_size_min_pixels_floor():
    """面积已在合法区间 [3,686,400, 4,624,220]（1920×1920 恰下限、2048×2048 区间内）不缩放。"""
    assert postprocess._fit_size(1920, 1920) == "1920x1920"
    assert postprocess._fit_size(2048, 2048) == "2048x2048"


@pytest.mark.parametrize("w,h,expected", [
    (720, 1280, "1440x2560"),   # 竖屏 9:16 帧（scale 恰 2，恰好下限）
    (1280, 720, "2560x1440"),   # 横屏 16:9 帧
    (1080, 1920, "1440x2560"),  # 1080p 竖屏：浮点 scale≈1.333 恰下限（ceil 版 2160×3840=8.29MP 实测 400 超上限）
    (1920, 1080, "2560x1440"),  # 1080p 横屏
    (640, 480, "2217x1663"),    # 非整数 scale（sqrt(12)≈3.464，round 后 3,686,871 达标）
    (4096, 4096, "1920x1920"),  # 超大图缩到下限
])
def test_fit_size_scales_up_keeping_aspect_ratio(w, h, expected):
    """输出面积落在 Ark 合法区间 [3,686,400, 4,624,220] 且保持宽高比（浮点 scale 非整数倍）。"""
    assert postprocess._fit_size(w, h) == expected


def _write_real_png(path, w, h):
    """写真实可读 PNG（cv2.imwrite）：run_task 读尺寸走真实 cv2 路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


def test_run_task_passes_fitted_size(enabled, monkeypatch):
    """run_task 用 cv2 读帧像素尺寸（imread shape 高x宽），等比放大后的 "WxH" 传给 edit_image；
    线程池并发读尺寸，到达顺序不定——断言按帧名配对，不断言顺序。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    _write_real_png(cdir / "work" / "keyframes" / "01.png", 720, 1280)
    _write_real_png(cdir / "work" / "keyframes" / "02.png", 1280, 720)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    assert {call["image"]: call["size"] for call in fake.calls} == {
        "01.png": "1440x2560", "02.png": "2560x1440",
    }


def test_run_task_unreadable_frame_omits_size(enabled, monkeypatch):
    """cv2 读不出的帧（非标准 PNG）→ size 空串 = 不传 size，不阻断编辑。"""
    settings, c = enabled
    cid = _make_conv(settings)  # helper 写的是魔数 PNG，cv2.imread 返回 None
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    assert sorted(call["image"] for call in fake.calls) == ["01.png", "02.png"]
    assert all(call["size"] == "" for call in fake.calls)


def test_edit_one_reads_size_in_thread_pool(tmp_path, monkeypatch):
    """imread 移出事件循环：_read_size 经 asyncio.to_thread 在线程池执行（线程 ≠ 驱动协程所在
    线程），同步 cv2 读图不阻塞 loop、不占信号量槽；返回空串时 size 透传空串（降级不传 size）。"""
    import threading
    settings = make_settings(tmp_path, enable_seedream_edit=True)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    src = cdir / "work" / "keyframes" / "01.png"
    out = cdir / "work" / "postprocessed" / "01.png"
    driver_tid = threading.get_ident()
    seen = {}

    def fake_read_size(path):
        seen["tid"] = threading.get_ident()
        seen["path"] = path
        return ""

    monkeypatch.setattr(postprocess, "_read_size", fake_read_size)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)
    sem = asyncio.Semaphore(1)
    frames: list[str] = []
    asyncio.run(postprocess._edit_one(
        settings, cdir, cid, src, out, None, OPTIONS_SUB, frames, sem))
    assert seen["path"] == src
    assert seen["tid"] != driver_tid  # 在线程池执行，不阻塞事件循环
    assert frames == ["01.png"]
    assert [call["size"] for call in fake.calls] == [""]


# ---------- 并行提交：进程级信号量限流与失败语义 ----------

class SlowEdit:
    """慢速桩：asyncio.sleep 模拟真实耗时；记录并发活跃数与峰值；按 fail 名单抛 SeedreamError。"""

    def __init__(self, fail=(), delay=0.05):
        self.fail = list(fail)
        self.delay = delay  # float（统一延时）或 {帧名: 秒}（打乱完成顺序）
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, settings, cdir, image, prompt, out, confirm, size=""):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append({"image": image.name, "size": size})
        try:
            if isinstance(self.delay, dict):
                await asyncio.sleep(self.delay.get(image.name, 0.0))
            else:
                await asyncio.sleep(self.delay)
            if image.name in self.fail:
                raise seedream.SeedreamError(502, "stub failure")
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
    settings = make_settings(tmp_path, enable_seedream_edit=True, seedream_concurrency=2)
    cid = _make_conv(settings)
    _add_frames(settings, cid, ["03.png", "04.png", "05.png"])
    slow = SlowEdit(delay=0.05)
    monkeypatch.setattr(postprocess.seedream, "edit_image", slow)

    with TestClient(create_app(settings)) as c:
        assert _post(c, cid, OPTIONS_SUB).status_code == 200

    assert slow.max_active == 2
    assert slow.max_active <= settings.seedream_concurrency
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
    monkeypatch.setattr(postprocess.seedream, "edit_image", slow)

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
    settings = make_settings(tmp_path, enable_seedream_edit=True)
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
