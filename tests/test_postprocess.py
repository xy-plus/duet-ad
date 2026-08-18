"""T5b 后处理编排：HTTP 门控矩阵、全链路（桩 edit_image）、人脸规则、失败处理。"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import postprocess, seedream, storage
from app.main import create_app

PNG = b"\x89PNG\r\n\x1a\n"

CHANGE_BG = "将图片背景更换为简洁干净的背景，保持主体人物与物品不变"
FACE_HOLD = "将图片中的人物改为用手捂住脸的造型，其余保持不变"
REMOVE_SUBTITLE = "移除图片中的所有字幕、水印和贴纸元素，其余保持不变"

OPTIONS_BG = {"change_bg": True, "face_hold": False, "remove_subtitle": False, "remove_brand": False}
FACE_ONLY = {"change_bg": False, "face_hold": True, "remove_subtitle": False, "remove_brand": False}


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
    """桩 seedream.edit_image：记录调用并写 out；按 fail 名单抛 SeedreamError。"""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = list(fail)

    async def __call__(self, settings, cdir, image, prompt, out, lock, confirm):
        self.calls.append({
            "image": image.name, "prompt": prompt, "out": out, "confirm": confirm,
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
                       json={"options": OPTIONS_BG, "confirm": True}).status_code == 401


def test_disabled_501(client):
    # 开关最优先（默认关）：不看 confirm、不看会话是否存在、不看选项
    r = client.post(f"/api/conversations/{'0' * 32}/postprocess", headers=AUTH,
                    json={"options": OPTIONS_BG, "confirm": True})
    assert r.status_code == 501
    assert r.json() == {"detail": "Seedream edit is disabled."}


def test_404_when_enabled(enabled):
    _, c = enabled
    r = _post(c, "0" * 32, OPTIONS_BG)
    assert r.status_code == 404
    assert r.json() == {"detail": "not found"}


def test_confirm_required_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    monkeypatch.setattr(postprocess.seedream, "edit_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    for body in ({}, {"options": OPTIONS_BG}, {"options": OPTIONS_BG, "confirm": False},
                 {"options": OPTIONS_BG, "confirm": "true"}, {"options": OPTIONS_BG, "confirm": 1}):
        r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH, json=body)
        assert r.status_code == 409, body
        assert r.json() == {"detail": "confirmation required"}


def test_no_options_422(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    empty = {"change_bg": False, "face_hold": False, "remove_subtitle": False, "remove_brand": False}
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
               json={"options": {"change_bg": "yes"}, "confirm": True})
    assert r.status_code == 422
    assert r.json() == {"detail": "options must be booleans"}


def test_not_done_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, status="queued")
    monkeypatch.setattr(postprocess.seedream, "edit_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_BG)
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}


def test_already_running_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": OPTIONS_BG, "frames": [], "error": None,
    })
    monkeypatch.setattr(postprocess.seedream, "edit_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_BG)
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
    r = _post(c, cid, OPTIONS_BG)
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

    options = {"change_bg": True, "face_hold": False,
               "remove_subtitle": True, "remove_brand": False}
    r = _post(c, cid, options)
    assert r.status_code == 200
    assert r.json() == {"status": "running", "frames": []}  # 受理即返回，进度走 detail 轮询

    # 每帧一条合并指令（分号连接），confirm 恒 True
    assert [call["image"] for call in fake.calls] == ["01.png", "02.png"]
    expected = f"{CHANGE_BG}；{REMOVE_SUBTITLE}"
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

    r = _post(c, cid, OPTIONS_BG)
    assert r.status_code == 200

    assert [call["image"] for call in fake.calls] == ["01.png", "01.png", "02.png"]
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


# ---------- face_hold 人脸规则 ----------

def test_face_hold_with_face_appends_prompt_line_single(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)
    monkeypatch.setattr(postprocess, "_detect_face",
                        lambda image, cascade=None: image.name == "01.png")

    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 200

    # 有人脸帧带 face_hold 指令；无人脸帧无适用选项 → 整帧跳过不编辑
    assert [call["image"] for call in fake.calls] == ["01.png"]
    assert fake.calls[0]["prompt"] == FACE_HOLD

    # 有人脸被处理 → prompt.txt 末尾追加一行，meta.prompt 同步
    prompt = (cdir / "work" / "prompt.txt").read_text(encoding="utf-8")
    assert prompt == "单段提示词\n" + postprocess.FACE_LINE
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["prompt"] == prompt
    assert meta["postprocess"]["frames"] == ["01.png"]


def test_face_hold_no_face_skips_option(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)
    monkeypatch.setattr(postprocess, "_detect_face", lambda image, cascade=None: False)

    options = {"change_bg": True, "face_hold": True, "remove_subtitle": False, "remove_brand": False}
    r = _post(c, cid, options)
    assert r.status_code == 200
    # 无人脸：该帧跳过 face_hold 选项，其余选项照常
    assert all(call["prompt"] == CHANGE_BG for call in fake.calls)
    assert len(fake.calls) == 2
    meta = storage.load_meta(settings.data_dir, cid)
    assert postprocess.FACE_LINE not in meta["prompt"]  # 不追加动作线


def test_face_hold_appends_per_segment(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)
    # 只有段 2 的 02.png 有人脸
    monkeypatch.setattr(postprocess, "_detect_face",
                        lambda image, cascade=None: str(image).endswith("segments/2/work/keyframes/02.png"))

    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 200

    seg1_p = (cdir / "work" / "segments" / "1" / "work" / "prompt.txt").read_text(encoding="utf-8")
    seg2_p = (cdir / "work" / "segments" / "2" / "work" / "prompt.txt").read_text(encoding="utf-8")
    assert seg1_p == "段一提示词"  # 段 1 无人脸：不动
    assert seg2_p == "段二提示词\n" + postprocess.FACE_LINE
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["segments"][0]["prompt"] == seg1_p
    assert meta["segments"][1]["prompt"] == seg2_p
    assert meta["postprocess"]["frames"] == ["segments/2/work/postprocessed/02.png"]


def test_no_face_hold_never_detects(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)
    monkeypatch.setattr(postprocess, "_detect_face",
                        lambda *a, **k: pytest.fail("face detection must not run"))

    r = _post(c, cid, OPTIONS_BG)
    assert r.status_code == 200
    assert all(call["prompt"] == CHANGE_BG for call in fake.calls)


# ---------- 失败处理 ----------

def test_frame_failure_marks_failed_keeps_successes(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    monkeypatch.setattr(postprocess.seedream, "edit_image", FakeEdit(fail=["02.png"]))

    r = _post(c, cid, OPTIONS_BG)
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

    r = _post(c, cid, OPTIONS_BG)
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
            postprocess.start(settings, cid, {"options": OPTIONS_BG, "confirm": True}, locks),
            postprocess.start(settings, cid, {"options": OPTIONS_BG, "confirm": True}, locks),
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


# ---------- 人脸检测单元（真 cv2 管线） ----------

class FakeCascade:
    def __init__(self, faces):
        self._faces = list(faces)

    def detectMultiScale(self, gray):
        return self._faces


def test_detect_face_real_cv2_pipeline(tmp_path):
    import cv2
    import numpy as np
    img = tmp_path / "face.png"
    cv2.imwrite(str(img), np.zeros((48, 48, 3), dtype=np.uint8))
    assert postprocess._detect_face(img, FakeCascade([[10, 10, 20, 20]])) is True
    assert postprocess._detect_face(img, FakeCascade([])) is False


def test_detect_face_undecodable_and_missing_cascade(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert postprocess._detect_face(bad, FakeCascade([[0, 0, 1, 1]])) is False
    assert postprocess._detect_face(bad, None) is False


def test_detect_face_real_cascade_on_synthetic(tmp_path):
    """真 cascade 数据 + 真 cv2 管线端到端：纯色合成图无人脸 → False（4.x 环境能力回归守卫）。"""
    import cv2
    import numpy as np
    img = tmp_path / "synth.png"
    cv2.imwrite(str(img), np.zeros((96, 96, 3), dtype=np.uint8))
    cascade = postprocess._load_cascade()
    assert cascade is not None
    assert postprocess._detect_face(img, cascade) is False


def test_load_cascade_missing_file_returns_none(tmp_path, monkeypatch):
    import cv2
    monkeypatch.setattr(cv2.data, "haarcascades", str(tmp_path / "no-cascade-dir"))
    assert postprocess._load_cascade() is None


def test_load_cascade_loads_with_real_data():
    """cv2 4.x 自带 haarcascade 数据：加载成功非 None（face_hold 可用）。"""
    assert postprocess._load_cascade() is not None


def test_face_hold_no_cascade_503(enabled, monkeypatch):
    """cv2 数据目录无 haarcascade → 勾选 face_hold 在门控直接 503，不做静默降级。"""
    settings, c = enabled
    cid = _make_conv(settings)
    monkeypatch.setattr(postprocess.cv2.data, "haarcascades",
                        str(settings.data_dir / "no-cascade"))
    monkeypatch.setattr(postprocess.seedream, "edit_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, FACE_ONLY)
    assert r.status_code == 503
    assert r.json() == {"detail": "face detection data unavailable"}
    assert storage.load_meta(settings.data_dir, cid).get("postprocess") is None


def test_rerun_different_options_409(enabled, monkeypatch):
    """上次 done 的 options 与本次不同 → 409（防旧产物贴新标签）；同选项重跑照常跳过已有图。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.seedream, "edit_image", fake)

    assert _post(c, cid, OPTIONS_BG).status_code == 200
    other = {"change_bg": True, "face_hold": False, "remove_subtitle": False, "remove_brand": True}
    r = _post(c, cid, other)
    assert r.status_code == 409
    assert r.json() == {"detail": "options changed since last run"}
    assert len(fake.calls) == 2  # 未产生新编辑

    # 同选项重跑：跳过已有优化图，正常 200 done
    r = _post(c, cid, OPTIONS_BG)
    assert r.status_code == 200
    assert len(fake.calls) == 2  # 全部帧已存在，无新编辑
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "done"
