"""后处理编排：HTTP 门控、MediaKit 场景映射、失败保留和并发限流。"""

import asyncio
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import image_optimization, mediakit, postprocess, storage
from app.main import create_app

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

OPTIONS_SUB = {"remove_subtitle": True, "remove_brand": False}
OPTIONS_BRAND = {"remove_subtitle": False, "remove_brand": True}


def _solid_png(path: Path, bgr: tuple[int, int, int], *, local_bgr=None) -> None:
    image = np.full((100, 100, 3), bgr, dtype=np.uint8)
    if local_bgr is not None:
        image[:5, :5] = local_bgr
    assert cv2.imwrite(str(path), image)


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
    prompts = (
        {seg["index"]: f"第 {seg['index']} 段 Codex 图片优化提示词" for seg in meta["segments"]}
        if segments else {0: "当前视频 Codex 图片优化提示词"}
    )
    meta.update(image_optimization.freeze_prompts(settings, meta, prompts))
    (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cid


def test_v4_generation_keyframes_require_an_intact_verified_output_receipt(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    source = cdir / "work" / "keyframes" / "01.png"
    output_dir = cdir / "work" / "postprocessed"
    output_dir.mkdir()
    output = output_dir / "01.png"
    output.write_bytes(source.read_bytes())
    schedule = {"version": 4, "scenes": []}
    optimization = {
        "version": 4,
        "plan_sha256": "a" * 64,
        "continuity_sha256": "b" * 64,
        "scene_anchor_schedule": schedule,
    }
    monkeypatch.setattr(
        postprocess.image_optimization, "receipt", lambda _meta: optimization
    )
    monkeypatch.setattr(
        postprocess.image_optimization,
        "dual_target_plan_receipt",
        lambda _meta: {"version": 4, "person_plans": [], "scene_plans": []},
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={
            "status": "done", "options": {"remove_subtitle": False,
            "remove_brand": False, "optimize_image": True},
            "frames": ["01.png"], "segments": [], "error": None,
        },
    )
    meta = storage.load_meta(settings.data_dir, cid)
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(cdir, meta, [source])

    source_metric = postprocess._area_weighted_palette_metric(source)
    contract = {
        "area_weighted_warm_cool_family": source_metric["warm_cool_family"],
        "saturation_style": source_metric["saturation_style"],
    }
    palette_payload = {
        "version": 1,
        "algorithm": postprocess._PALETTE_METRIC_ALGORITHM,
        "thresholds": postprocess._PALETTE_METRIC_THRESHOLDS,
        "frames": [{
            "segment_index": 0,
            "frame_index": 1,
            "contract": contract,
            "source": source_metric,
            "output": postprocess._area_weighted_palette_metric(output),
        }],
    }
    palette_metrics = {
        **palette_payload, "sha256": postprocess._receipt_sha256(palette_payload),
    }
    source_palette_payload = {
        "version": 1,
        "plan_sha256": optimization["plan_sha256"],
        "continuity_sha256": optimization["continuity_sha256"],
        "metrics": {
            **{
                key: value for key, value in palette_metrics.items()
                if key != "sha256"
            },
            "frames": [{
                key: value for key, value in palette_metrics["frames"][0].items()
                if key != "output"
            }],
        },
    }
    source_palette_receipt = {
        **source_palette_payload,
        "sha256": postprocess._receipt_sha256(source_palette_payload),
    }
    semantic_payload = {
        "version": 1,
        "plan_sha256": optimization["plan_sha256"],
        "continuity_sha256": optimization["continuity_sha256"],
        "label": "bootstrap",
        "pack_bindings": [],
        "metrics_sha256": palette_metrics["sha256"],
        "verdict": {"passed": True},
    }
    semantic_receipt = {
        **semantic_payload, "sha256": postprocess._receipt_sha256(semantic_payload),
    }
    postprocess._write_json_receipt(
        postprocess._semantic_receipt_path(cdir, "bootstrap"), semantic_receipt,
    )
    postprocess._write_json_receipt(
        cdir / "work" / ".postprocess-private" / "scene-anchors" / "palette-source.json",
        source_palette_receipt,
    )
    payload = {
        "version": 1,
        "plan_sha256": optimization["plan_sha256"],
        "continuity_sha256": optimization["continuity_sha256"],
        "scene_anchor_schedule_sha256": postprocess._receipt_sha256(schedule),
        "semantic_receipts": [{"label": "bootstrap", "sha256": semantic_receipt["sha256"]}],
        "source_palette_receipt_sha256": source_palette_receipt["sha256"],
        "palette_metrics": palette_metrics,
        "palette_metrics_sha256": palette_metrics["sha256"],
        "frames": [{
            "segment_index": 0,
            "frame_name": "01.png",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }],
        "verdict": {"passed": True},
    }
    receipt = {**payload, "sha256": postprocess._receipt_sha256(payload)}
    storage.update_meta(settings.data_dir, cid, _image_verification=receipt)
    meta = storage.load_meta(settings.data_dir, cid)
    assert postprocess.generation_keyframes(cdir, meta, [source]) == [output]
    output.write_bytes(PNG + b"drift")
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(cdir, meta, [source])


def test_v4_postprocess_global_pack_failure_prevents_layout_and_fanout(
    tmp_path, monkeypatch,
):
    """The v4 product coordinator cannot reach a dependent after global pack fail."""
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    cdir.mkdir()
    calls = []
    private = {
        "options": {"remove_subtitle": False, "remove_brand": False},
        "plan_sha256": "a" * 64,
        "continuity_sha256": "b" * 64,
    }
    metric = {
        "version": 1,
        "algorithm": postprocess._PALETTE_METRIC_ALGORITHM,
        "thresholds": postprocess._PALETTE_METRIC_THRESHOLDS,
        "frames": [],
    }
    metric["sha256"] = postprocess._receipt_sha256(metric)
    monkeypatch.setattr(postprocess, "_v4_frozen_plan", lambda *_args: {"segments": []})
    monkeypatch.setattr(postprocess, "_v4_frame_sources", lambda *_args: {})
    monkeypatch.setattr(postprocess, "_v4_palette_metrics", lambda *_args: metric)

    async def bootstrap(*_args, **_kwargs):
        calls.append("global-anchor")
        return {}, []

    async def fail_global_pack(*_args, **_kwargs):
        calls.append("global-pack")
        raise postprocess.PostprocessError(409, "image_reference_pack_failed")

    async def forbidden_layout(*_args, **_kwargs):
        pytest.fail("layout must not run after global semantic failure")

    async def forbidden_fanout(*_args, **_kwargs):
        pytest.fail("fanout must not run after global semantic failure")

    monkeypatch.setattr(postprocess, "_v4_bootstrap_scene_anchors", bootstrap)
    monkeypatch.setattr(postprocess, "_v4_verify_bootstrap_packs", fail_global_pack)
    monkeypatch.setattr(postprocess, "_v4_generate_layout_anchors", forbidden_layout)
    monkeypatch.setattr(postprocess, "_v4_fan_out", forbidden_fanout)

    with pytest.raises(postprocess.PostprocessError, match="image_reference_pack_failed"):
        asyncio.run(postprocess._run_v4_task(
            settings, "cid", cdir, {}, private, {}, asyncio.Semaphore(1), object(), object(),
        ))
    assert calls == ["global-anchor", "global-pack"]


def test_palette_metric_uses_area_weighted_lab_b_star_and_allows_local_change(tmp_path):
    yellow = tmp_path / "yellow.png"
    blue = tmp_path / "blue.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _solid_png(yellow, (0, 255, 255))
    _solid_png(blue, (255, 0, 0))
    _solid_png(source, (127, 127, 127))
    _solid_png(output, (127, 127, 127), local_bgr=(0, 0, 255))

    warm = postprocess._area_weighted_palette_metric(yellow)
    cool = postprocess._area_weighted_palette_metric(blue)
    assert warm["warm_cool_family"] == "warm"
    assert cool["warm_cool_family"] == "cool"
    assert warm["mean_lab_b_star"] > 0 > cool["mean_lab_b_star"]

    plan = {"segments": [{
        "segment_index": 0,
        "frame_constraints": [{
            "frame_index": 1,
            "dominant_palette_contract": {
                "area_weighted_warm_cool_family": "balanced",
                "saturation_style": "muted",
            },
        }],
    }]}
    metrics = postprocess._v4_palette_metrics(
        plan, {(0, 1): source}, {(0, 1): output},
    )
    assert metrics["frames"][0]["source"]["warm_cool_family"] == "balanced"
    assert metrics["frames"][0]["output"]["warm_cool_family"] == "balanced"


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

    # 严格阶段屏障：全段文字擦除完成后才开始全段图标擦除。
    assert [call["scenes"] for call in fake.calls] == [
        (mediakit.TEXT_SCENE,), (mediakit.TEXT_SCENE,),
        (mediakit.ICON_SCENE,), (mediakit.ICON_SCENE,),
    ]
    assert all(call["confirm"] is True for call in fake.calls)

    # 产出：work/postprocessed/<帧名>.png（与源帧同目录层级）
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "postprocessed" / "02.png").is_file()

    # meta.postprocess done + frames；detail 中保留后处理状态
    meta = storage.load_meta(settings.data_dir, cid)
    pp = meta["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == {**options, "optimize_image": False}
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
    assert pp["error"] == "segment_failed"
    assert "02.png" in pp["segments"][0]["error"]
    assert pp["frames"] == []  # 整段完整前不发布 canonical
    private = cdir / "work" / ".postprocess-private" / "0" / "text"
    assert (private / "01.png").is_file()
    assert not (cdir / "work" / "postprocessed").exists()

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
    assert r.status_code == 409
    assert r.json() == {"detail": "postprocess_canonical_conflict"}
    assert fake.calls == []
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
    oks = [r for r in results if r is None]
    errs = [r for r in results if isinstance(r, postprocess.PostprocessError)]
    assert len(oks) == 1 and len(errs) == 1
    assert errs[0].status == 409 and errs[0].detail == "already running"
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "running"


def test_options_lock_is_rechecked_inside_lock_without_writing(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
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
    (cdir / "meta.json").write_text(json.dumps(locked), encoding="utf-8")
    before = _file_snapshot(cdir)

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
    assert pp["options"] == {**OPTIONS_SUB, "optimize_image": False}

    # 共有键真变了仍然 409：兼容比对不放松锁定
    r = _post(c, cid, OPTIONS_BRAND)
    assert r.status_code == 409
    assert r.json() == {"detail": {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }}


def test_legacy_pure_change_bg_rerun_clears_artifacts_and_reedits(enabled, monkeypatch):
    """任何 failed 状态都只能走分段重试，包括旧 change_bg 状态。"""
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
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "postprocess_segment_retry_required"
    assert len(fake.calls) == 0


def test_legacy_pure_change_bg_any_new_option_requires_segment_retry(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": [], "error": "x",
    })
    r = _post(c, cid, OPTIONS_BRAND)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "postprocess_segment_retry_required"
    assert fake.calls == []


def test_legacy_pure_change_bg_multi_segment_clears_artifacts(enabled, monkeypatch):
    """多段 failed 旧状态也不得由普通 start 重建。"""
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
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "postprocess_segment_retry_required"
    for n in (1, 2):
        d = cdir / "work" / "segments" / str(n) / "work" / "postprocessed"
        assert (d / "01.png").read_bytes() == PNG + b"legacy"
    assert len(fake.calls) == 0


def test_failed_start_preserves_meta_revision_and_never_calls_provider(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    post = storage.load_meta(settings.data_dir, cid)["postprocess"]
    post.update(status="failed", error="segment_failed")
    post["segments"][0].update(status="failed", revision=7, error="provider_rejected")
    storage.update_meta(settings.data_dir, cid, postprocess=post)
    before = storage.load_meta(settings.data_dir, cid)
    calls_before = len(fake.calls)

    response = _post(c, cid, OPTIONS_SUB)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "postprocess_segment_retry_required"
    assert storage.load_meta(settings.data_dir, cid) == before
    assert len(fake.calls) == calls_before


def test_corrupt_existing_canonical_is_not_reused(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    canonical = settings.data_dir / cid / "work" / "postprocessed" / "01.png"
    canonical.write_bytes(b"not-a-png")
    calls_before = len(fake.calls)

    response = _post(c, cid, OPTIONS_SUB)

    assert response.status_code == 409
    assert response.json() == {"detail": "postprocess_canonical_conflict"}
    assert len(fake.calls) == calls_before


def test_publish_fsyncs_staged_directory_before_parent(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    staged = tmp_path / "private" / "01.png"
    canonical = tmp_path / "postprocessed" / "01.png"
    source.write_bytes(PNG)
    staged.parent.mkdir()
    staged.write_bytes(PNG)
    synced = []
    monkeypatch.setattr(postprocess, "_fsync_dir", lambda path: synced.append(path))

    postprocess._publish_segment([staged], [(source, canonical)])

    assert synced == [
        tmp_path / ".postprocessed.publishing",
        tmp_path,
    ]
    assert canonical.read_bytes() == PNG


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
    assert pp["error"] == "segment_failed"
    assert "02.png" in pp["segments"][0]["error"]
    assert pp["frames"] == []  # 段内有失败帧时不发布任何 canonical
    private = cdir / "work" / ".postprocess-private" / "0" / "text"
    assert (private / "01.png").is_file()
    assert (private / "03.png").is_file()
    assert not (private / "02.png").exists()


# ---------- 取消：父任务取消写 failed 终态 ----------

def test_run_task_cancelled_writes_failed(tmp_path, monkeypatch):
    """父任务被取消（uvicorn graceful shutdown）：CancelledError 是 BaseException，run_task 须在
    继续传播前把 meta.postprocess 写成 failed——否则永久 running、start 永久 409 拒重跑。"""
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    asyncio.run(postprocess.start(
        settings, cid, {"confirm": True, "options": OPTIONS_SUB}, {}
    ))

    async def hang(*a, **k):
        await asyncio.Event().wait()  # 被取消时才结束的挂起桩

    monkeypatch.setattr(postprocess, "_mediakit_stage", hang)
    sem = asyncio.Semaphore(10)

    async def drive():
        task = asyncio.create_task(postprocess.run_task(settings, cid, sem))
        await asyncio.sleep(0.05)  # 让出至进入 gather
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert "cancelled" in pp["error"]
    assert pp["options"] == {**OPTIONS_SUB, "optimize_image": False}


def test_parallel_segment_updates_use_atomic_storage_mutation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings, segments=True)
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": {**OPTIONS_SUB, "optimize_image": False},
        "frames": [], "error": None,
        "segments": [
            postprocess._segment_state(1, 1),
            postprocess._segment_state(2, 2),
        ],
    })
    public_load_calls = 0

    def forbidden_stale_load(*_args):
        nonlocal public_load_calls
        public_load_calls += 1
        raise AssertionError("_update_segment must not perform a lock-outside load")

    monkeypatch.setattr(postprocess.storage, "load_meta", forbidden_stale_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(
            lambda args: postprocess._update_segment(
                settings, cid, args[0], stage=args[1], completed_frames=args[2]
            ),
            ((1, "brand", 1), (2, "seedream", 2)),
        ))
    current = storage._load_meta_unlocked(settings.data_dir, cid)["postprocess"]
    by_index = {item["index"]: item for item in current["segments"]}
    assert public_load_calls == 0
    assert (by_index[1]["stage"], by_index[1]["completed_frames"]) == ("brand", 1)
    assert (by_index[2]["stage"], by_index[2]["completed_frames"]) == ("seedream", 2)


def test_public_state_maps_untrusted_status_stage_and_error_to_safe_closed_values():
    public = postprocess.public_state({
        "status": "provider-secret-status", "options": OPTIONS_SUB,
        "frames": [], "error": "request_id=req-secret-123",
        "segments": [{
            "index": 1, "status": "remote-running", "stage": "task_id=secret",
            "completed_frames": 0, "total_frames": 1, "revision": 1,
            "error": "provider task_id=secret-456",
        }, {
            "index": 2, "status": "failed", "stage": "brand",
            "completed_frames": 0, "total_frames": 1, "revision": 2,
            "error": "frame 02.png failed: provider request req-secret",
        }],
    })
    assert public["status"] == "failed"
    assert public["error"] == "postprocess_failed"
    assert public["segments"][0] == {
        "index": 1, "status": "failed", "stage": "unknown",
        "completed_frames": 0, "total_frames": 1, "revision": 1,
        "error": "postprocess_failed",
    }
    assert public["segments"][1]["error"] == "frame 02.png failed"
    assert "secret" not in json.dumps(public)


@pytest.mark.parametrize("segments", [
    [{"index": 0, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None},
     {"index": 1, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None}],
    [{"index": 1, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None},
     {"index": 3, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None}],
    [{"index": 0, "status": "running", "stage": "queued", "completed_frames": 2,
      "total_frames": 1, "revision": 1, "error": None}],
    [{"index": True, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None}],
])
def test_public_state_fails_closed_for_invalid_segment_collection(segments):
    public = postprocess.public_state({
        "status": "running", "options": {**OPTIONS_SUB, "optimize_image": False},
        "frames": ["01.png"], "error": None, "segments": segments,
    })
    assert public["status"] == "failed"
    assert public["error"] == "postprocess_receipt_invalid"
    assert public["segments"] == []
    assert public["frames"] == []
