"""提示词审核/编辑（prompt_review）的端到端契约测试。

覆盖：请求校验（封闭键集合 + 布尔归一化 + 指纹兼容）、状态机（新状态
awaiting_prompt_review 不参与重启恢复）、流水线暂停/确认重建、历史项目
主体剥离器、内部两个端点、对外 v1 两个端点。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import prompt_review
from app.main import create_app
from app.pipeline import Pipeline
from app.prompt_review import PromptReviewError
from app.research_log import ResearchLog
from app.store import ProjectStore, generation_request_sha256

from test_pipeline import (  # noqa: E402
    _effective_request,
    _FlowDeepSeek,
    _FlowDoubaoTTS,
    _FlowMediaKit,
    _FlowPicsart,
    _FlowSeedream,
    _install_fake_media,
    _settings,
)
from test_public_api import _public_setup  # noqa: E402


GENERATION_REQUEST = {
    "version": 1,
    "output": {"aspect_ratio": "16:9", "resolution": "480p", "fit_mode": "auto"},
    "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
    "dialogue": {"mode": "auto_rewrite", "target_language": "日语"},
    "replacement_guidance": None,
}

NEW_PREFIX = (
    "This is an editing task on Video 1, not a generation task: keep every "
    "shot exactly as in Video 1.\n\n"
    "Solid black (0,0,0) blocks fully conceal the people to replace.\n\n"
    "PERSON REPLACEMENT IS MANDATORY AND TAKES PRIORITY OVER SOURCE IDENTITY.\n\n"
)
OLD_PREFIX = (
    "This is an editing task on Video 1, not a generation task: keep every "
    "shot exactly as in Video 1.\n\n"
    "Dark gray (64,64,64) blocks fully conceal the people to replace.\n\n"
    "FACE REPLACEMENT IS MANDATORY AND TAKES PRIORITY OVER SOURCE IDENTITY.\n\n"
)
SUFFIX = (
    "\n\nOverlay removal: Video 1 may contain burned-in subtitles."
    "\n\nDialogue requirements: Audio 1 is this segment's track.\n"
)
BODY = "Replace the host with the reference person while keeping the action."


# ---------- 请求校验与指纹 ----------

@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_prompt_review_json_rejects_non_boolean(value):
    from fastapi import HTTPException
    from app.main import _parse_generation_request
    with pytest.raises(HTTPException) as error:
        _parse_generation_request(
            json.dumps({**GENERATION_REQUEST, "prompt_review": value}), "internal",
        )
    assert error.value.status_code == 422


@pytest.mark.parametrize("value", [True, False])
def test_prompt_review_json_normalization_and_legacy_default(value):
    from app.main import _parse_generation_request
    parsed = _parse_generation_request(
        json.dumps({**GENERATION_REQUEST, "prompt_review": value}), "internal",
    )
    assert parsed["prompt_review"] is value
    assert _parse_generation_request(
        json.dumps(GENERATION_REQUEST), "internal",
    )["prompt_review"] is False


def test_prompt_review_unknown_key_still_rejected():
    from fastapi import HTTPException
    from app.main import _parse_generation_request
    with pytest.raises(HTTPException) as error:
        _parse_generation_request(
            json.dumps({**GENERATION_REQUEST, "prompt_reviewx": True}), "internal",
        )
    assert error.value.status_code == 422


def test_prompt_review_fingerprint_preserves_legacy_and_binds_enabled():
    legacy = generation_request_sha256(GENERATION_REQUEST)
    assert generation_request_sha256(
        {**GENERATION_REQUEST, "prompt_review": False}
    ) == legacy
    assert generation_request_sha256(
        {**GENERATION_REQUEST, "prompt_review": True}
    ) != legacy


# ---------- 状态机 ----------

def test_awaiting_status_survives_restart_recovery(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create(
        _effective_request(), "awaiting-restart",
    )
    store.update(
        project_id,
        status="awaiting_prompt_review",
        project_progress={
            "status": "awaiting_prompt_review",
            "percent": 57,
            "stage": "awaiting_prompt_review",
        },
        generation={
            "status": "awaiting_prompt_review",
            "stage": "awaiting_prompt_review",
            "error": None,
        },
    )
    restarted = store.fail_incomplete_after_restart()
    assert restarted == []
    state = store.read(project_id)
    assert state["status"] == "awaiting_prompt_review"
    assert state["project_progress"]["stage"] == "awaiting_prompt_review"


# ---------- 历史项目主体剥离器 ----------

@pytest.mark.parametrize("prefix", [NEW_PREFIX, OLD_PREFIX])
def test_split_frozen_prompt_body_strips_known_injection(prefix):
    text = prefix + BODY + SUFFIX
    assert prompt_review.split_frozen_prompt_body(text) == BODY


def test_split_frozen_prompt_body_without_injection():
    assert prompt_review.split_frozen_prompt_body(BODY) == BODY
    text = BODY + "\n\nDo not add captions, subtitles, or new dialogue."
    assert prompt_review.split_frozen_prompt_body(text) == BODY


def test_split_frozen_prompt_body_fail_closed():
    assert prompt_review.split_frozen_prompt_body("") is None
    assert prompt_review.split_frozen_prompt_body("   \n\n  ") is None
    # 注入块开头文案出现在"主体"里说明块顺序未识别：宁可不展示。
    assert prompt_review.split_frozen_prompt_body(
        BODY + "\n\n" + NEW_PREFIX + "tail"
    ) is None
    assert prompt_review.split_frozen_prompt_body(None) is None


# ---------- 流水线：暂停与确认 ----------

def _create_review_project(tmp_path, monkeypatch, *, review: bool):
    store = ProjectStore(tmp_path / "projects")
    request = {**_effective_request(
        target_language="日语",
        replacement_instruction="把主角替换成用户参考形象",
    ), "prompt_review": review}
    project_id, project, _state = store.create(request, "request-review")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    (project / "inputs" / "replacement-image").write_bytes(b"replacement")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
    )
    _install_fake_media(monkeypatch, [])
    deepseek = _FlowDeepSeek()
    picsart = _FlowPicsart()

    def register(_project_id: str, path: Path) -> str:
        return f"https://assets.invalid/{path.parent.name}/{path.name}"

    pipeline = Pipeline(
        _settings(tmp_path, enable_mediakit=True),
        store,
        register,
        deepseek=deepseek,
        mediakit=_FlowMediaKit(),
        picsart=picsart,
        seedream=_FlowSeedream(deepseek.events),
        tts=_FlowDoubaoTTS(),
    )
    return store, pipeline, project_id, project, picsart


def _read_review(project: Path) -> dict:
    review = prompt_review.read_review(project)
    assert review is not None
    return review


def test_run_pauses_before_any_paid_submit_when_prompt_review_enabled(
    tmp_path, monkeypatch,
):
    store, pipeline, project_id, project, picsart = _create_review_project(
        tmp_path, monkeypatch, review=True,
    )
    pipeline.run(project_id)

    state = store.read(project_id)
    assert state["status"] == "awaiting_prompt_review"
    assert state["project_progress"] == {
        "status": "awaiting_prompt_review",
        "percent": 57,
        "stage": "awaiting_prompt_review",
    }
    assert picsart.submits == []
    assert not any(
        path.is_dir() for path in (project / "logs").glob("picsart*-submit")
    )

    review = _read_review(project)
    assert [segment["segment_index"] for segment in review["segments"]] == [1, 2]
    for segment in review["segments"]:
        index = segment["segment_index"]
        frozen = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt"
        ).read_text(encoding="utf-8")
        assert segment["prefix"] + segment["body"] + segment["suffix"] == frozen
        assert segment["body"].strip()
        # 展示给用户的 body 不含机械注入块。
        assert "editing task on Video 1" not in segment["body"]
        assert "Overlay removal:" not in segment["body"]
        assert "Dialogue requirements:" not in segment["body"]
    # 直通展示视图读到的就是 DS 主体。
    display = prompt_review.display_segments(project, store.read(project_id))
    assert [item["prompt"] for item in display] == [
        segment["body"] for segment in review["segments"]
    ]


def test_straight_through_run_still_writes_review_file(tmp_path, monkeypatch):
    store, pipeline, project_id, project, picsart = _create_review_project(
        tmp_path, monkeypatch, review=False,
    )
    pipeline.run(project_id)

    assert store.read(project_id)["status"] == "succeeded"
    assert len(picsart.submits) == 2
    review = _read_review(project)
    for segment in review["segments"]:
        index = segment["segment_index"]
        frozen = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt"
        ).read_text(encoding="utf-8")
        assert segment["prefix"] + segment["body"] + segment["suffix"] == frozen


def test_confirm_rebuilds_frozen_prompts_and_resume_completes(
    tmp_path, monkeypatch,
):
    store, pipeline, project_id, project, picsart = _create_review_project(
        tmp_path, monkeypatch, review=True,
    )
    pipeline.run(project_id)
    review = _read_review(project)
    edits = [
        {"segment_index": 1, "prompt": "edited body one"},
        {"segment_index": 2, "prompt": review["segments"][1]["body"]},
    ]
    pipeline.confirm_prompt_review(project_id, edits)

    state = store.read(project_id)
    assert state["status"] == "awaiting_prompt_review"  # 状态迁移由调用方负责
    manifest = state["h3_prepared_artifacts"]
    for segment in review["segments"]:
        index = segment["segment_index"]
        body = "edited body one" if index == 1 else segment["body"]
        expected = segment["prefix"] + body + segment["suffix"]
        prompt_path = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt"
        )
        assert prompt_path.read_text(encoding="utf-8") == expected
        receipt = manifest["segments"][str(index)]["prompt"]
        assert receipt["sha256"] == hashlib.sha256(
            expected.encode("utf-8")
        ).hexdigest()
        assert receipt["size_bytes"] == len(expected.encode("utf-8"))
    # 审核记录更新为确认后的主体（完成后只读展示用）。
    updated = _read_review(project)
    assert updated["segments"][0]["body"] == "edited body one"
    assert updated["segments"][1]["body"] == review["segments"][1]["body"]

    # 等价 resume-picsart 入口：排回队列后逐字重放重建的提示词继续。
    store.update(
        project_id,
        status="queued",
        has_video=False,
        project_progress={
            "status": "queued", "percent": 0, "stage": "picsart_resume_queued",
        },
        generation={
            "status": "queued", "stage": "picsart_resume_queued", "error": None,
        },
        error=None,
    )
    pipeline.resume_r2v(project_id)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert len(picsart.submits) == 2
    submitted = sorted(
        submit["prompt"] for submit in picsart.submits
    )
    assert any("edited body one" in prompt for prompt in submitted)
    for submit in picsart.submits:
        # 注入的前/尾段由后端机械重建，确认后仍然完整（该流程台词为空，
        # 尾部是 "Do not add captions" 分支）。
        assert "editing task on Video 1" in submit["prompt"]
        assert (
            "Dialogue requirements:" in submit["prompt"]
            or "Do not add captions" in submit["prompt"]
        )


def test_confirm_rejects_invalid_edits(tmp_path, monkeypatch):
    store, pipeline, project_id, project, _picsart = _create_review_project(
        tmp_path, monkeypatch, review=True,
    )
    pipeline.run(project_id)
    review = _read_review(project)
    bodies = [
        {"segment_index": s["segment_index"], "prompt": s["body"]}
        for s in review["segments"]
    ]

    def rejected(edits, code):
        with pytest.raises(PromptReviewError) as error:
            pipeline.confirm_prompt_review(project_id, edits)
        assert error.value.code == code

    rejected("not-a-list", "invalid_prompt_review_edit")
    rejected([], "invalid_prompt_review_edit")
    rejected(bodies[:1], "invalid_prompt_review_edit")  # 段集合不一致
    rejected(
        [*bodies, {"segment_index": 99, "prompt": "x"}],
        "invalid_prompt_review_edit",
    )
    rejected(
        [{"segment_index": 1, "prompt": "x"}, {"segment_index": 1, "prompt": "y"}],
        "invalid_prompt_review_edit",
    )
    rejected(
        [{**bodies[0], "prompt": "  \n "}, bodies[1]],
        "invalid_prompt_review_edit",
    )
    rejected(
        [{**bodies[0], "extra": True}, bodies[1]],
        "invalid_prompt_review_edit",
    )
    rejected(
        [{**bodies[0], "prompt": "x" * 60_000}, bodies[1]],
        "invalid_prompt_review_edit",
    )
    # 全部拒绝后冻结提示词原样不动。
    for segment in review["segments"]:
        index = segment["segment_index"]
        frozen = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt"
        ).read_text(encoding="utf-8")
        assert segment["prefix"] + segment["body"] + segment["suffix"] == frozen


def test_confirm_requires_awaiting_status(tmp_path, monkeypatch):
    store, pipeline, project_id, _project, _picsart = _create_review_project(
        tmp_path, monkeypatch, review=False,
    )
    pipeline.run(project_id)
    assert store.read(project_id)["status"] == "succeeded"
    with pytest.raises(PromptReviewError) as error:
        pipeline.confirm_prompt_review(
            project_id, [{"segment_index": 1, "prompt": "x"}],
        )
    assert error.value.code == "prompt_review_not_pending"


# ---------- 内部端点 ----------

AUTH = {"Authorization": "Bearer test-secret"}


def _endpoint_settings(tmp_path: Path):
    from app.config import Settings
    return Settings(
        access_token="test-secret",
        multiref_mode="composed",
        data_dir=tmp_path / "data",
        public_base_url="https://studio.example",
        store_lock_enabled=False,
    )


class _PromptReviewPipeline:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.release = threading.Event()
        self.confirmed: list[tuple[str, list]] = []
        self.resumed: list[str] = []

    def run(self, project_id: str) -> None:
        self.release.wait(timeout=10)

    def confirm_prompt_review(self, project_id: str, segments) -> None:
        state = self.store.read(project_id)
        if state.get("status") != "awaiting_prompt_review":
            raise PromptReviewError(
                "prompt_review_not_pending", "项目不在等待提示词审核状态"
            )
        self.confirmed.append((project_id, segments))

    def resume_r2v(self, project_id: str) -> None:
        self.resumed.append(project_id)
        project = self.store.path(project_id)
        output = project / "output" / "generated.mp4"
        output.write_bytes(b"resumed-video")
        self.store.update(
            project_id,
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={
                "status": "succeeded", "percent": 100, "stage": "completed",
            },
            generation={"status": "succeeded", "stage": "completed", "error": None},
        )


def _prompt_review_factory(_settings, store, _register_asset):
    return _PromptReviewPipeline(store)


def _create_project(app, request=None) -> str:
    project_id, _project, _state = app.state.store.create(
        request or _effective_request(), f"endpoint-{time.time_ns()}",
    )
    return project_id


def _mark_awaiting(app, project_id: str, segments) -> None:
    store = app.state.store
    project = store.path(project_id)
    prompt_review.write_review(project, [
        {
            "segment_index": index,
            "prefix": f"prefix-{index}\n\n",
            "body": body,
            "suffix": "\n\nDialogue requirements: tail",
        }
        for index, body in segments
    ])
    store.update(
        project_id,
        status="awaiting_prompt_review",
        project_progress={
            "status": "awaiting_prompt_review",
            "percent": 57,
            "stage": "awaiting_prompt_review",
        },
        generation={
            "status": "awaiting_prompt_review",
            "stage": "awaiting_prompt_review",
            "error": None,
        },
    )


def _wait_status(client: TestClient, project_id: str, status: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = client.get(
            f"/api/conversations/{project_id}", headers=AUTH,
        ).json()
        if state["status"] == status:
            return state
        time.sleep(0.01)
    raise AssertionError(f"project did not reach {status}")


def test_internal_prompt_review_read_and_confirm(tmp_path):
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_prompt_review_factory,
    )
    try:
        with TestClient(app) as client:
            project_id = _create_project(app)
            _mark_awaiting(app, project_id, [(1, "body one"), (2, "body two")])

            read = client.get(
                f"/api/conversations/{project_id}/prompt-review", headers=AUTH,
            )
            assert read.status_code == 200
            assert read.json() == {
                "status": "awaiting_prompt_review",
                "segments": [
                    {"segment_index": 1, "prompt": "body one"},
                    {"segment_index": 2, "prompt": "body two"},
                ],
            }

            confirm = client.post(
                f"/api/conversations/{project_id}/prompt-review/confirm",
                headers=AUTH,
                json={"segments": [
                    {"segment_index": 1, "prompt": "edited one"},
                    {"segment_index": 2, "prompt": "body two"},
                ]},
            )
            assert confirm.status_code == 202
            assert confirm.json()["status"] == "queued"
            assert app.state.pipeline.confirmed == [(
                project_id,
                [
                    {"segment_index": 1, "prompt": "edited one"},
                    {"segment_index": 2, "prompt": "body two"},
                ],
            )]
            state = _wait_status(client, project_id, "succeeded")
            assert state["generation"]["status"] == "succeeded"
            assert app.state.pipeline.resumed == [project_id]

            # 已确认继续的项目再次 confirm 被状态门禁拒绝。
            again = client.post(
                f"/api/conversations/{project_id}/prompt-review/confirm",
                headers=AUTH,
                json={"segments": [
                    {"segment_index": 1, "prompt": "edited one"},
                    {"segment_index": 2, "prompt": "body two"},
                ]},
            )
            assert again.status_code == 409
            assert again.json()["detail"]["code"] == "prompt_review_not_pending"
    finally:
        app.state.pipeline.release.set()


def test_internal_prompt_review_confirm_gates(tmp_path):
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_prompt_review_factory,
    )
    try:
        with TestClient(app) as client:
            project_id = _create_project(app)  # queued：不在审核暂停状态
            confirm = client.post(
                f"/api/conversations/{project_id}/prompt-review/confirm",
                headers=AUTH,
                json={"segments": [{"segment_index": 1, "prompt": "x"}]},
            )
            assert confirm.status_code == 409
            assert confirm.json()["detail"]["code"] == "prompt_review_not_pending"

            for body, expected in (
                (None, 400),
                ({}, 400),
                ({"segments": [], "extra": 1}, 400),
            ):
                rejected = client.post(
                    f"/api/conversations/{project_id}/prompt-review/confirm",
                    headers=AUTH,
                    json=body,
                )
                assert rejected.status_code == expected

            missing = client.get(
                "/api/conversations/" + "0" * 32 + "/prompt-review", headers=AUTH,
            )
            assert missing.status_code in {404, 422}
            assert client.get(
                f"/api/conversations/{project_id}/prompt-review",
            ).status_code == 401
    finally:
        app.state.pipeline.release.set()


def test_internal_prompt_review_legacy_split_and_empty(tmp_path):
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_prompt_review_factory,
    )
    try:
        with TestClient(app) as client:
            store = app.state.store
            # 历史项目：没有 prompt-review.json，只有冻结清单 + h3_prompt.txt。
            project_id = _create_project(app)
            project = store.path(project_id)
            segment_dir = project / "work" / "segments" / "0001"
            segment_dir.mkdir(parents=True)
            legacy_text = OLD_PREFIX + BODY + SUFFIX
            prompt_path = segment_dir / "h3_prompt.txt"
            prompt_path.write_text(legacy_text, encoding="utf-8")
            receipt = Pipeline._project_artifact_receipt(
                project, prompt_path, "work/segments/0001/h3_prompt.txt",
            )
            store.update(
                project_id,
                status="succeeded",
                has_video=True,
                h3_prepared_artifacts={
                    "version": 5,
                    "bindings": {},
                    "segments": {"1": {"prompt": receipt}},
                },
                project_progress={
                    "status": "succeeded", "percent": 100, "stage": "completed",
                },
            )
            read = client.get(
                f"/api/conversations/{project_id}/prompt-review", headers=AUTH,
            )
            assert read.status_code == 200
            assert read.json() == {
                "status": "succeeded",
                "segments": [{"segment_index": 1, "prompt": BODY}],
            }

            # 剥不出注入边界的历史项目返回空 segments，不展示注入内容。
            # （只有注入块、没有可识别主体的冻结提示词即属于此类。）
            unsplittable = _create_project(app)
            other = store.path(unsplittable)
            other_dir = other / "work" / "segments" / "0001"
            other_dir.mkdir(parents=True)
            unknown_text = OLD_PREFIX + "\n\nDo not add captions, subtitles, or new dialogue."
            other_prompt = other_dir / "h3_prompt.txt"
            other_prompt.write_text(unknown_text, encoding="utf-8")
            other_receipt = Pipeline._project_artifact_receipt(
                other, other_prompt, "work/segments/0001/h3_prompt.txt",
            )
            store.update(
                unsplittable,
                status="succeeded",
                has_video=True,
                h3_prepared_artifacts={
                    "version": 5,
                    "bindings": {},
                    "segments": {"1": {"prompt": other_receipt}},
                },
                project_progress={
                    "status": "succeeded", "percent": 100, "stage": "completed",
                },
            )
            read = client.get(
                f"/api/conversations/{unsplittable}/prompt-review", headers=AUTH,
            )
            assert read.json()["segments"] == []

            # 完全没有提示词痕迹的项目同样返回空。
            bare = _create_project(app)
            read = client.get(
                f"/api/conversations/{bare}/prompt-review", headers=AUTH,
            )
            assert read.json()["segments"] == []
    finally:
        app.state.pipeline.release.set()


# ---------- 对外 v1 端点 ----------

@pytest.fixture
def video_1s(tmp_path):
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def _v1_create(client, auth, video, *, key="review001", **data):
    payload = {"aspect_ratio": "9:16", "resolution": "768p", **data}
    with video.open("rb") as stream:
        return client.post(
            "/api/v1/video-generations",
            headers={**auth, "Idempotency-Key": key},
            data=payload,
            files={"source_video": ("source.mp4", stream, "video/mp4")},
        )


def test_v1_prompt_review_create_roundtrip_and_idempotency(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=_prompt_review_factory)
    try:
        with TestClient(app) as client:
            created = _v1_create(client, auth, video_1s, prompt_review="true")
            assert created.status_code == 201
            pid = created.json()["id"][3:]
            state = app.state.store.read(pid)
            assert state["effective_request"]["prompt_review"] is True
            replay = _v1_create(client, auth, video_1s, prompt_review="1")
            assert replay.status_code == 200
            changed = _v1_create(client, auth, video_1s, prompt_review="false")
            assert changed.status_code == 409

            default = _v1_create(client, auth, video_1s, key="review002")
            assert default.status_code == 201
            pid2 = default.json()["id"][3:]
            assert app.state.store.read(pid2)["effective_request"]["prompt_review"] is False
            # 缺省与显式 false 同指纹（前向兼容旧幂等记录）。
            false_replay = _v1_create(
                client, auth, video_1s, key="review002", prompt_review="false",
            )
            assert false_replay.status_code == 200
            invalid = _v1_create(
                client, auth, video_1s, key="review003", prompt_review="maybe",
            )
            assert invalid.status_code == 422
    finally:
        app.state.pipeline.release.set()


def test_v1_prompt_review_endpoints(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=_prompt_review_factory)
    try:
        with TestClient(app) as client:
            created = _v1_create(client, auth, video_1s, prompt_review="true")
            assert created.status_code == 201
            job_id = created.json()["id"]
            pid = job_id[3:]

            # queued：不在审核暂停状态，读与确认都是 409。
            read = client.get(
                f"/api/v1/video-generations/{job_id}/prompt-review", headers=auth,
            )
            assert read.status_code == 409
            assert read.json()["error"]["code"] == "prompt_review_not_available"

            _mark_awaiting(app, pid, [(1, "body one"), (2, "body two")])
            read = client.get(
                f"/api/v1/video-generations/{job_id}/prompt-review", headers=auth,
            )
            assert read.status_code == 200
            assert read.json() == {
                "status": "awaiting_prompt_review",
                "segments": [
                    {"segment_index": 1, "prompt": "body one"},
                    {"segment_index": 2, "prompt": "body two"},
                ],
            }
            job = client.get(
                f"/api/v1/video-generations/{job_id}", headers=auth,
            ).json()
            assert job["status"] == "awaiting_prompt_review"

            confirm = client.post(
                f"/api/v1/video-generations/{job_id}/prompt-review/confirm",
                headers=auth,
                json={"segments": [
                    {"segment_index": 1, "prompt": "edited one"},
                    {"segment_index": 2, "prompt": "body two"},
                ]},
            )
            assert confirm.status_code == 202
            assert confirm.json()["status"] in {"queued", "running", "succeeded"}
            assert app.state.pipeline.confirmed == [(
                pid,
                [
                    {"segment_index": 1, "prompt": "edited one"},
                    {"segment_index": 2, "prompt": "body two"},
                ],
            )]

            again = client.post(
                f"/api/v1/video-generations/{job_id}/prompt-review/confirm",
                headers=auth,
                json={"segments": [
                    {"segment_index": 1, "prompt": "edited one"},
                    {"segment_index": 2, "prompt": "body two"},
                ]},
            )
            assert again.status_code == 409
            assert again.json()["error"]["code"] == "prompt_review_not_pending"

            for bad in (
                {"segments": [{"segment_index": 1}]},
                {"segments": [{"segment_index": "1", "prompt": "x"}]},
                {"segments": [{"segment_index": 1, "prompt": ""}]},
                {"segments": [], },
                {"segments": [{"segment_index": 1, "prompt": "x", "x": 1}]},
            ):
                rejected = client.post(
                    f"/api/v1/video-generations/{job_id}/prompt-review/confirm",
                    headers=auth,
                    json=bad,
                )
                assert rejected.status_code == 422
    finally:
        app.state.pipeline.release.set()


def test_v1_prompt_review_auth_and_ownership(tmp_path, video_1s):
    from app import public_api_auth, public_credits
    settings, auth_one, _ = _public_setup(tmp_path)
    key_two = public_api_auth.create_client_key(
        settings.public_api_clients_file, "partner_two", key_id="key00002",
    )
    public_credits.adjust(
        settings.data_dir, "partner_two", 5_000,
        reason="test grant", idempotency_key="grant0002",
    )
    auth_two = {"Authorization": f"Bearer {key_two}"}
    app = create_app(settings, pipeline_factory=_prompt_review_factory)
    try:
        with TestClient(app) as client:
            created = _v1_create(client, auth_one, video_1s, prompt_review="true")
            job_id = created.json()["id"]
            pid = job_id[3:]
            _mark_awaiting(app, pid, [(1, "body one")])

            assert client.get(
                f"/api/v1/video-generations/{job_id}/prompt-review",
            ).status_code == 401
            assert client.get(
                f"/api/v1/video-generations/{job_id}/prompt-review",
                headers=auth_two,
            ).status_code == 404
            assert client.post(
                f"/api/v1/video-generations/{job_id}/prompt-review/confirm",
                headers=auth_two,
                json={"segments": [{"segment_index": 1, "prompt": "x"}]},
            ).status_code == 404
            assert client.get(
                "/api/v1/video-generations/vg_"
                + "0" * 32
                + "/prompt-review",
                headers=auth_one,
            ).status_code == 404
    finally:
        app.state.pipeline.release.set()


def test_v1_prompt_review_succeeded_readonly(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=_prompt_review_factory)
    try:
        with TestClient(app) as client:
            created = _v1_create(client, auth, video_1s)
            job_id = created.json()["id"]
            pid = job_id[3:]
            store = app.state.store
            project = store.path(pid)
            prompt_review.write_review(project, [{
                "segment_index": 1,
                "prefix": "prefix\n\n",
                "body": "frozen body",
                "suffix": "\n\nDialogue requirements: tail",
            }])
            store.update(
                pid,
                status="succeeded",
                has_video=True,
                project_progress={
                    "status": "succeeded", "percent": 100, "stage": "completed",
                },
            )
            read = client.get(
                f"/api/v1/video-generations/{job_id}/prompt-review", headers=auth,
            )
            assert read.status_code == 200
            assert read.json() == {
                "status": "succeeded",
                "segments": [{"segment_index": 1, "prompt": "frozen body"}],
            }
            confirm = client.post(
                f"/api/v1/video-generations/{job_id}/prompt-review/confirm",
                headers=auth,
                json={"segments": [{"segment_index": 1, "prompt": "x"}]},
            )
            assert confirm.status_code == 409
    finally:
        app.state.pipeline.release.set()
