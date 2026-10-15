"""提示词审核/编辑（prompt_review）的端到端契约测试。

覆盖：请求校验（封闭键集合 + 布尔归一化 + 指纹兼容）、状态机（新状态
awaiting_prompt_review 不参与重启恢复）、流水线暂停/确认重建、历史项目
主体剥离器、内部两个端点、对外 v1 两个端点。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
    review = prompt_review.read_review(project, "pass2")
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
    # 两遍出片：审核只在 pass2（人物遍）冻结后暂停——pass1（物品遍）已
    # 付费出片完成。
    assert state["project_progress"] == {
        "status": "awaiting_prompt_review",
        "percent": 76,
        "stage": "awaiting_prompt_review",
    }
    assert state["prompt_review_pending_pass"] == "pass2"
    assert len(picsart.submits) == 2
    assert (project / "logs" / "picsart-pass1-submit").is_dir()
    assert not (project / "logs" / "picsart-submit").exists()

    review = _read_review(project)
    assert [segment["segment_index"] for segment in review["segments"]] == [1, 2]
    # 落盘携带原视频时间区间（前端下拉框"分段 N（Xs ~ Ys）"用）。
    assert [
        (segment["start_s"], segment["end_s"]) for segment in review["segments"]
    ] == [(0.0, 4.2), (4.2, 9.2)]
    for segment in review["segments"]:
        index = segment["segment_index"]
        frozen = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt-pass2.txt"
        ).read_text(encoding="utf-8")
        assert (
            segment["prefix"] + segment["editable"] + segment["hidden"]
            + segment["suffix"] == frozen
        )
        assert segment["editable"].strip()
        # 展示给用户的 editable 不含机械注入块，也不含权威计划块。
        assert "editing task on Video 1" not in segment["editable"]
        assert "Overlay removal:" not in segment["editable"]
        assert "Dialogue requirements:" not in segment["editable"]
        assert (
            prompt_review.HIDDEN_BLOCK_START not in segment["editable"]
        )
    # 直通展示视图读到的就是可编辑主体前半（含时间区间）。
    display = prompt_review.display_segments(project, store.read(project_id))
    assert display == [
        {
            "segment_index": segment["segment_index"],
            "prompt": segment["editable"],
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
        }
        for segment in review["segments"]
    ]


def test_straight_through_run_still_writes_review_file(tmp_path, monkeypatch):
    store, pipeline, project_id, project, picsart = _create_review_project(
        tmp_path, monkeypatch, review=False,
    )
    pipeline.run(project_id)

    assert store.read(project_id)["status"] == "succeeded"
    # 两遍出片：2 段 × 2 遍共 4 次提交；审核文件按 pass 各落一份。
    assert len(picsart.submits) == 4
    assert prompt_review.read_review(project, "pass1") is not None
    review = _read_review(project)
    for segment in review["segments"]:
        index = segment["segment_index"]
        frozen = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt-pass2.txt"
        ).read_text(encoding="utf-8")
        assert (
            segment["prefix"] + segment["editable"] + segment["hidden"]
            + segment["suffix"] == frozen
        )


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
        {"segment_index": 2, "prompt": review["segments"][1]["editable"]},
    ]
    pipeline.confirm_prompt_review(project_id, edits)

    state = store.read(project_id)
    assert state["status"] == "awaiting_prompt_review"  # 状态迁移由调用方负责
    manifest = state["h3_prepared_artifacts_pass2"]
    for segment in review["segments"]:
        index = segment["segment_index"]
        body = "edited body one" if index == 1 else segment["editable"]
        expected = (
            segment["prefix"] + body + segment["hidden"] + segment["suffix"]
        )
        prompt_path = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt-pass2.txt"
        )
        assert prompt_path.read_text(encoding="utf-8") == expected
        receipt = manifest["segments"][str(index)]["prompt"]
        assert receipt["sha256"] == hashlib.sha256(
            expected.encode("utf-8")
        ).hexdigest()
        assert receipt["size_bytes"] == len(expected.encode("utf-8"))
    # 审核记录更新为确认后的主体（完成后只读展示用）。
    updated = _read_review(project)
    assert updated["segments"][0]["editable"] == "edited body one"
    assert updated["segments"][1]["editable"] == review["segments"][1]["editable"]

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
    # pass1 的 2 次提交已在暂停前完成；confirm 后 resume 只重放 pass2。
    assert len(picsart.submits) == 4
    pass2_submitted = sorted(
        submit["prompt"] for submit in picsart.submits
        if "edited body one" in submit["prompt"]
        or "Audio 1" in submit["prompt"]
        or "people are not replaced in this pass" not in submit["prompt"]
    )
    assert any("edited body one" in prompt for prompt in pass2_submitted)
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
        {"segment_index": s["segment_index"], "prompt": s["editable"]}
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
    # 空白主体拒绝/放行（取决于 hidden 块）由
    # test_confirm_empty_editable_allowed_only_with_hidden_block 覆盖。
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
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt-pass2.txt"
        ).read_text(encoding="utf-8")
        assert (
            segment["prefix"] + segment["editable"] + segment["hidden"]
            + segment["suffix"] == frozen
        )


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


# ---------- 权威计划块（hidden：BEGIN 整行起到 body 末尾） ----------

HIDDEN_BLOCK = (
    prompt_review.HIDDEN_BLOCK_START + "\n"
    "object plan details\n"
    "END AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN"
)


def test_split_body_begin_block_cases():
    # 没有 BEGIN 行：整段可编辑，hidden 为空（与拆分前逐字节一致）。
    assert prompt_review.split_body("only editable") == ("only editable", "")
    # BEGIN 在 body 中间。
    editable, hidden = prompt_review.split_body("editable text\n\n" + HIDDEN_BLOCK)
    assert editable == "editable text\n\n"
    assert hidden == HIDDEN_BLOCK
    # BEGIN 在 body 开头：可编辑部分为空，整块 hidden。
    assert prompt_review.split_body(HIDDEN_BLOCK) == ("", HIDDEN_BLOCK)
    # 行内出现而非整行：不算标记。
    inline = "see " + prompt_review.HIDDEN_BLOCK_START + " here"
    assert prompt_review.split_body(inline) == (inline, "")


def _synthetic_awaiting_project(tmp_path, parts):
    """parts: 每段 (prefix, editable, hidden, suffix) 四元组。"""
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "synthetic-review",
    )
    records = []
    manifest_segments = {}
    for index, (prefix, editable, hidden, suffix) in enumerate(parts, 1):
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        segment_dir.mkdir(parents=True)
        prompt_path = segment_dir / "h3_prompt-pass2.txt"
        prompt_path.write_text(
            prefix + editable + hidden + suffix, encoding="utf-8",
        )
        manifest_segments[str(index)] = {
            "prompt": Pipeline._project_artifact_receipt(
                project, prompt_path, f"work/segments/{index:04d}/h3_prompt-pass2.txt",
            ),
        }
        records.append({
            "segment_index": index,
            "prefix": prefix,
            "editable": editable,
            "hidden": hidden,
            "suffix": suffix,
        })
    prompt_review.write_review(project, records, "pass2")
    store.update(
        project_id,
        status="awaiting_prompt_review",
        prompt_review_pending_pass="pass2",
        h3_prepared_artifacts_pass2={
            "version": 5, "bindings": {}, "segments": manifest_segments,
        },
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
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _pid, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    return store, pipeline, project_id, project


def test_confirm_hidden_block_verbatim_and_user_input_cannot_forge_it(tmp_path):
    store, pipeline, project_id, project = _synthetic_awaiting_project(
        tmp_path, [("prefix\n\n", "original editable\n\n", HIDDEN_BLOCK, "\n\nsuffix")],
    )
    forged = (
        "edited editable\n\n"
        + prompt_review.HIDDEN_BLOCK_START + "\nforged plan\n"
        "END AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN"
    )
    pipeline.confirm_prompt_review(
        project_id, [{"segment_index": 1, "prompt": forged}],
    )
    rebuilt = (
        project / "work" / "segments" / "0001" / "h3_prompt-pass2.txt"
    ).read_text(encoding="utf-8")
    # 重建 = prefix + edited + hidden（逐字取自落盘）+ suffix；用户伪造的
    # BEGIN 文本只落在可编辑区，真实 hidden 不受任何影响。
    assert rebuilt == "prefix\n\n" + forged + HIDDEN_BLOCK + "\n\nsuffix"
    assert rebuilt.count(prompt_review.HIDDEN_BLOCK_START) == 2
    review = _read_review(project)
    assert review["segments"][0]["editable"] == forged
    assert review["segments"][0]["hidden"] == HIDDEN_BLOCK
    display = prompt_review.display_segments(project, store.read(project_id))
    assert display == [
        {"segment_index": 1, "prompt": forged, "start_s": None, "end_s": None},
    ]


def test_confirm_empty_editable_allowed_only_with_hidden_block(tmp_path):
    _store, pipeline, project_id, project = _synthetic_awaiting_project(
        tmp_path,
        [
            ("p1\n\n", "editable one", HIDDEN_BLOCK, "\n\ns1"),
            ("p2\n\n", "editable two", "", "\n\ns2"),
        ],
    )
    with pytest.raises(PromptReviewError) as error:
        pipeline.confirm_prompt_review(project_id, [
            {"segment_index": 1, "prompt": "x"},
            {"segment_index": 2, "prompt": "  "},
        ])
    assert error.value.code == "invalid_prompt_review_edit"
    pipeline.confirm_prompt_review(project_id, [
        {"segment_index": 1, "prompt": ""},
        {"segment_index": 2, "prompt": "edited two"},
    ])
    rebuilt = (
        project / "work" / "segments" / "0001" / "h3_prompt-pass2.txt"
    ).read_text(encoding="utf-8")
    assert rebuilt == "p1\n\n" + HIDDEN_BLOCK + "\n\ns1"


def test_confirm_stale_check_uses_four_part_formula(tmp_path):
    _store, pipeline, project_id, project = _synthetic_awaiting_project(
        tmp_path, [("p\n\n", "editable", HIDDEN_BLOCK, "\n\ns")],
    )
    # 冻结文件里 hidden 被删：与 prefix+editable+hidden+suffix 不一致 → stale。
    prompt_path = project / "work" / "segments" / "0001" / "h3_prompt-pass2.txt"
    prompt_path.write_text("p\n\neditable\n\ns", encoding="utf-8")
    with pytest.raises(PromptReviewError) as error:
        pipeline.confirm_prompt_review(
            project_id, [{"segment_index": 1, "prompt": "x"}],
        )
    assert error.value.code == "prompt_review_stale"


def test_legacy_split_truncates_hidden_block(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "legacy-hidden",
    )
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    prompt_path = segment_dir / "h3_prompt-pass2.txt"
    prompt_path.write_text(
        NEW_PREFIX + "editable text\n\n" + HIDDEN_BLOCK + SUFFIX,
        encoding="utf-8",
    )
    receipt = Pipeline._project_artifact_receipt(
        project, prompt_path, "work/segments/0001/h3_prompt-pass2.txt",
    )
    store.update(
        project_id,
        h3_prepared_artifacts_pass2={
            "version": 5, "bindings": {}, "segments": {"1": {"prompt": receipt}},
        },
    )
    display = prompt_review.display_segments(project, store.read(project_id))
    assert display == [
        {
            "segment_index": 1,
            "prompt": "editable text\n\n",
            "start_s": None,
            "end_s": None,
        },
    ]


def test_review_dirty_time_values_fall_back_to_null(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(_effective_request(), "dirty-times")
    # 脏时间（负数、区间颠倒/相等、非数、inf、布尔）在落盘时归一化为 null。
    for start, end in [
        (-1, 5), (5, 5), (9, 8), ("0", 8), (None, None),
        (float("inf"), 9), (float("nan"), 9), (True, 9),
    ]:
        prompt_review.write_review(project, [{
            "segment_index": 1, "prefix": "p", "editable": "e", "hidden": "",
            "suffix": "s", "start_s": start, "end_s": end,
        }], "pass2")
        review = prompt_review.read_review(project, "pass2")
        assert review["segments"][0]["start_s"] is None
        assert review["segments"][0]["end_s"] is None
    # 绕过 write_review 手写的脏文件在读取时同样归一化，不进响应。
    (project / "work" / "prompt-review-pass2.json").write_text(
        json.dumps({
            "version": prompt_review.PROMPT_REVIEW_VERSION,
            "segments": [{
                "segment_index": 1,
                "prefix": "p",
                "prefix_sha256": prompt_review._sha256_text("p"),
                "editable": "e",
                "hidden": "",
                "suffix": "s",
                "suffix_sha256": prompt_review._sha256_text("s"),
                "start_s": -3,
                "end_s": "junk",
            }],
        }),
        encoding="utf-8",
    )
    review = prompt_review.read_review(project, "pass2")
    assert review["segments"][0]["start_s"] is None
    assert review["segments"][0]["end_s"] is None
    # 合法时间（含整数）原样保留并出现在展示视图里。
    prompt_review.write_review(project, [{
        "segment_index": 1, "prefix": "p", "editable": "e", "hidden": "",
        "suffix": "s", "start_s": 0, "end_s": 8,
    }], "pass2")
    display = prompt_review.display_segments(project, store.read(project_id))
    assert display == [
        {"segment_index": 1, "prompt": "e", "start_s": 0.0, "end_s": 8.0},
    ]


def test_confirm_preserves_segment_time_range(tmp_path):
    _store, pipeline, project_id, project = _synthetic_awaiting_project(
        tmp_path, [("p\n\n", "editable", HIDDEN_BLOCK, "\n\ns")],
    )
    review_path = project / "work" / "prompt-review-pass2.json"
    raw = json.loads(review_path.read_text(encoding="utf-8"))
    raw["segments"][0]["start_s"] = 1.5
    raw["segments"][0]["end_s"] = 9.0
    review_path.write_text(json.dumps(raw), encoding="utf-8")
    pipeline.confirm_prompt_review(
        project_id, [{"segment_index": 1, "prompt": "edited"}],
    )
    updated = _read_review(project)
    assert updated["segments"][0]["start_s"] == 1.5
    assert updated["segments"][0]["end_s"] == 9.0


def test_old_shape_review_file_falls_back_to_legacy_split(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "old-review-shape",
    )
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    text = NEW_PREFIX + "legacy editable" + SUFFIX
    prompt_path = segment_dir / "h3_prompt-pass2.txt"
    prompt_path.write_text(text, encoding="utf-8")
    receipt = Pipeline._project_artifact_receipt(
        project, prompt_path, "work/segments/0001/h3_prompt-pass2.txt",
    )
    store.update(
        project_id,
        status="awaiting_prompt_review",
        h3_prepared_artifacts_pass2={
            "version": 5, "bindings": {}, "segments": {"1": {"prompt": receipt}},
        },
        project_progress={
            "status": "awaiting_prompt_review",
            "percent": 57,
            "stage": "awaiting_prompt_review",
        },
    )
    # 旧版（body 字段、version 1）审核记录不再被读取，展示回退到清单剥离。
    (project / "work" / "prompt-review-pass2.json").write_text(
        json.dumps({
            "version": 1,
            "segments": [{
                "segment_index": 1,
                "prefix": "x", "prefix_sha256": "x",
                "body": "old body",
                "suffix": "x", "suffix_sha256": "x",
            }],
        }),
        encoding="utf-8",
    )
    assert prompt_review.read_review(project, "pass2") is None
    display = prompt_review.display_segments(project, store.read(project_id))
    assert display == [
        {"segment_index": 1, "prompt": "legacy editable", "start_s": None, "end_s": None},
    ]
    # 旧形状记录不能 confirm（无法保证 hidden 完整）。用独立目录避免
    # ProjectStore 进程锁冲突。
    _store2, pipeline, _pid, _project = _synthetic_awaiting_project(
        tmp_path / "b", [("p\n\n", "e", "", "\n\ns")],
    )
    (_project / "work" / "prompt-review-pass2.json").write_text(
        json.dumps({"version": 1, "segments": []}), encoding="utf-8",
    )
    with pytest.raises(PromptReviewError) as error:
        pipeline.confirm_prompt_review(
            _pid, [{"segment_index": 1, "prompt": "x"}],
        )
    assert error.value.code == "prompt_review_unavailable"


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
    def __init__(self, store: ProjectStore, video: Path | None = None) -> None:
        self.store = store
        # v1 测试中 succeeded 会触发 public_artifacts.publish → 真 ffprobe
        # 校验成片，假字节必然失败；传入真实视频文件时写成片用真内容。
        self.video = video
        self.release = threading.Event()
        self.confirmed: list[tuple[str, list]] = []
        self.resumed: list[str] = []

    def run(self, project_id: str) -> None:
        # 超时只是防挂死的兜底：-n auto 高负载下事件循环/线程池排队可能
        # 超过 10s，超时误触发会把项目标 failed 造成偶发失败。
        self.release.wait(timeout=60)

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
        if self.video is not None:
            shutil.copyfile(self.video, output)
        else:
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


def _prompt_review_factory_with_video(video: Path):
    def factory(_settings, store, _register_asset):
        return _PromptReviewPipeline(store, video=video)

    return factory


def _create_project(app, request=None) -> str:
    project_id, _project, _state = app.state.store.create(
        request or _effective_request(), f"endpoint-{time.time_ns()}",
    )
    return project_id


def _mark_awaiting(app, project_id: str, segments) -> None:
    """segments: (index, body) 或 (index, body, start_s, end_s) 元组列表。"""
    store = app.state.store
    project = store.path(project_id)
    prompt_review.write_review(project, [
        {
            "segment_index": segment[0],
            "prefix": f"prefix-{segment[0]}\n\n",
            "editable": segment[1],
            "hidden": "",
            "suffix": "\n\nDialogue requirements: tail",
            "start_s": segment[2] if len(segment) > 2 else None,
            "end_s": segment[3] if len(segment) > 2 else None,
        }
        for segment in segments
    ], "pass2")
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
            _mark_awaiting(app, project_id, [
                (1, "body one", 0.0, 8.0),
                (2, "body two", 8.5, 16.0),
            ])

            read = client.get(
                f"/api/conversations/{project_id}/prompt-review", headers=AUTH,
            )
            assert read.status_code == 200
            assert read.json() == {
                "status": "awaiting_prompt_review",
                "segments": [
                    {
                        "segment_index": 1, "prompt": "body one",
                        "start_s": 0.0, "end_s": 8.0,
                    },
                    {
                        "segment_index": 2, "prompt": "body two",
                        "start_s": 8.5, "end_s": 16.0,
                    },
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
            # 历史项目：没有 prompt-review.json，只有冻结清单 + h3_prompt-pass2.txt。
            project_id = _create_project(app)
            project = store.path(project_id)
            segment_dir = project / "work" / "segments" / "0001"
            segment_dir.mkdir(parents=True)
            legacy_text = OLD_PREFIX + BODY + SUFFIX
            prompt_path = segment_dir / "h3_prompt-pass2.txt"
            prompt_path.write_text(legacy_text, encoding="utf-8")
            receipt = Pipeline._project_artifact_receipt(
                project, prompt_path, "work/segments/0001/h3_prompt-pass2.txt",
            )
            store.update(
                project_id,
                status="succeeded",
                has_video=True,
                h3_prepared_artifacts_pass2={
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
                "segments": [
                    {"segment_index": 1, "prompt": BODY, "start_s": None, "end_s": None},
                ],
            }

            # 剥不出注入边界的历史项目返回空 segments，不展示注入内容。
            # （只有注入块、没有可识别主体的冻结提示词即属于此类。）
            unsplittable = _create_project(app)
            other = store.path(unsplittable)
            other_dir = other / "work" / "segments" / "0001"
            other_dir.mkdir(parents=True)
            unknown_text = OLD_PREFIX + "\n\nDo not add captions, subtitles, or new dialogue."
            other_prompt = other_dir / "h3_prompt-pass2.txt"
            other_prompt.write_text(unknown_text, encoding="utf-8")
            other_receipt = Pipeline._project_artifact_receipt(
                other, other_prompt, "work/segments/0001/h3_prompt-pass2.txt",
            )
            store.update(
                unsplittable,
                status="succeeded",
                has_video=True,
                h3_prepared_artifacts_pass2={
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
    # confirm 后假 resume 直接把项目标 succeeded：v1 confirm 端点随即读取
    # Job 视图会 publish 成片（真 ffprobe 校验），必须给假 pipeline 真视频。
    app = create_app(
        settings,
        pipeline_factory=_prompt_review_factory_with_video(video_1s),
    )
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

            _mark_awaiting(app, pid, [
                (1, "body one", 0.0, 8.0),
                (2, "body two", 8.0, 16.0),
            ])
            read = client.get(
                f"/api/v1/video-generations/{job_id}/prompt-review", headers=auth,
            )
            assert read.status_code == 200
            assert read.json() == {
                "status": "awaiting_prompt_review",
                "segments": [
                    {
                        "segment_index": 1, "prompt": "body one",
                        "start_s": 0.0, "end_s": 8.0,
                    },
                    {
                        "segment_index": 2, "prompt": "body two",
                        "start_s": 8.0, "end_s": 16.0,
                    },
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
                "editable": "frozen body",
                "hidden": "",
                "suffix": "\n\nDialogue requirements: tail",
            }], "pass2")
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
                "segments": [
                    {"segment_index": 1, "prompt": "frozen body", "start_s": None, "end_s": None},
                ],
            }
            confirm = client.post(
                f"/api/v1/video-generations/{job_id}/prompt-review/confirm",
                headers=auth,
                json={"segments": [{"segment_index": 1, "prompt": "x"}]},
            )
            assert confirm.status_code == 409
    finally:
        app.state.pipeline.release.set()


# ---------- 成功后编辑提示词重新生成 + 视频版本历史 ----------


def test_regenerate_archives_current_version_and_resubmits(tmp_path, monkeypatch):
    store, pipeline, project_id, project, picsart = _create_review_project(
        tmp_path, monkeypatch, review=False,
    )
    pipeline.run(project_id)
    assert store.read(project_id)["status"] == "succeeded"
    # 两遍出片：2 段 × 2 遍。
    assert len(picsart.submits) == 4
    review = _read_review(project)
    original_finals = {
        segment["segment_index"]: (
            project / "work" / "segments" / f"{segment['segment_index']:04d}"
            / "h3_prompt-pass2.txt"
        ).read_text(encoding="utf-8")
        for segment in review["segments"]
    }
    edits = [
        {"segment_index": 1, "prompt": "regenerated body one"},
        {"segment_index": 2, "prompt": review["segments"][1]["editable"]},
    ]
    pipeline.regenerate_with_edited_prompts(project_id, edits)

    # 归档：v1 完整（成片 + 分段成片 + 各段冻结提示词全文 + submit 证据）。
    archive = project / "output" / "history" / "v1"
    assert (archive / "generated.mp4").is_file()
    assert (archive / "generated.mp4").read_bytes() == (
        # 归档复制的是首次成功的成片；当前 output/generated.mp4 尚未被覆盖。
        project / "output" / "generated.mp4"
    ).read_bytes()
    for index in (1, 2):
        archived_prompt = (
            archive / "prompts" / f"segment-{index:04d}.txt"
        ).read_text(encoding="utf-8")
        assert archived_prompt == original_finals[index]
        assert (archive / "segments" / f"segment-{index:04d}.mp4").is_file()
    assert (archive / "submit-logs" / "picsart-submit").is_dir()
    assert not (project / "logs" / "picsart-submit").exists()
    # 归档只收 pass2 的 submit 证据；pass1 证据留在原位（pass1 账本保留）。
    assert (project / "logs" / "picsart-pass1-submit").is_dir()
    # 分段成片被移走（强制新版本全新提交，而不是复用旧输出）。
    assert not (
        project / "work" / "segments" / "0001" / "generated.mp4"
    ).exists()

    state = store.read(project_id)
    history = state["generation_history"]
    assert len(history) == 1
    entry = history[0]
    assert entry["version"] == 1
    assert entry["completed_at"]
    assert entry["video"]["path"] == "output/history/v1/generated.mp4"
    assert entry["video"]["sha256"] == hashlib.sha256(
        (archive / "generated.mp4").read_bytes()
    ).hexdigest()
    assert [
        (segment["segment_index"], segment["prompt_sha256"])
        for segment in entry["segments"]
    ] == [
        (index, hashlib.sha256(original_finals[index].encode("utf-8")).hexdigest())
        for index in (1, 2)
    ]
    # pass2 账本清零（重提新提交）；pass1 账本保留（pass1 产物是当版本的
    # 组成部分）。清单 prompt 回执与重写后的提示词一致。
    assert state["picsart_submission_receipts_pass2"] is None
    assert state["picsart_submission_receipts_pass1"] is not None
    manifest = state["h3_prepared_artifacts_pass2"]
    for segment in review["segments"]:
        index = segment["segment_index"]
        body = "regenerated body one" if index == 1 else segment["editable"]
        expected = (
            segment["prefix"] + body + segment["hidden"] + segment["suffix"]
        )
        prompt_path = (
            project / "work" / "segments" / f"{index:04d}" / "h3_prompt-pass2.txt"
        )
        assert prompt_path.read_text(encoding="utf-8") == expected
        receipt = manifest["segments"][str(index)]["prompt"]
        assert receipt["sha256"] == hashlib.sha256(
            expected.encode("utf-8")
        ).hexdigest()
    updated = _read_review(project)
    assert updated["segments"][0]["editable"] == "regenerated body one"

    # 等价 resume-picsart 入口重提：全新提交（旧账本已清、旧成片已归档）。
    # 状态序列：翻转后 queued + has_video=false（窗口期看不到旧成功态）。
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
    window_state = store.read(project_id)
    assert window_state["status"] == "queued"
    assert window_state["has_video"] is False
    pipeline.resume_r2v(project_id)
    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert state["has_video"] is True
    # 首次 2 段 × 2 遍 + 重新生成的 pass2 2 段（pass1 不重提）。
    assert len(picsart.submits) == 6
    assert any(
        "regenerated body one" in submit["prompt"] for submit in picsart.submits[4:]
    )

    # 二次 regenerate：v2 归档、v1 原样保留。
    second_review = _read_review(project)
    pipeline.regenerate_with_edited_prompts(project_id, [
        {"segment_index": 1, "prompt": "second regen body"},
        {"segment_index": 2, "prompt": second_review["segments"][1]["editable"]},
    ])
    archive_v2 = project / "output" / "history" / "v2"
    assert (archive_v2 / "generated.mp4").is_file()
    assert (
        archive_v2 / "prompts" / "segment-0001.txt"
    ).read_text(encoding="utf-8") == (
        second_review["segments"][0]["prefix"] + "regenerated body one"
        + second_review["segments"][0]["hidden"]
        + second_review["segments"][0]["suffix"]
    )
    assert (archive / "generated.mp4").is_file()  # v1 仍在
    state = store.read(project_id)
    assert [item["version"] for item in state["generation_history"]] == [1, 2]


def test_regenerate_gates_and_edit_validation(tmp_path, monkeypatch):
    # 非 succeeded：409 语义。
    store, pipeline, project_id, project, _picsart = _create_review_project(
        tmp_path, monkeypatch, review=True,
    )
    pipeline.run(project_id)
    assert store.read(project_id)["status"] == "awaiting_prompt_review"
    with pytest.raises(PromptReviewError) as error:
        pipeline.regenerate_with_edited_prompts(
            project_id, [{"segment_index": 1, "prompt": "x"}],
        )
    assert error.value.code == "regenerate_not_succeeded"

    # succeeded 但没有 review 文件：409 unavailable。
    _store, pipeline2, project_id2, _project2 = _synthetic_awaiting_project(
        tmp_path / "b", [("p\n\n", "editable", HIDDEN_BLOCK, "\n\ns")],
    )
    store2 = pipeline2.store
    (_project2 / "work" / "prompt-review-pass2.json").unlink()
    store2.update(
        project_id2,
        status="succeeded",
        has_video=True,
        project_progress={
            "status": "succeeded", "percent": 100, "stage": "completed",
        },
    )
    with pytest.raises(PromptReviewError) as error:
        pipeline2.regenerate_with_edited_prompts(
            project_id2, [{"segment_index": 1, "prompt": "x"}],
        )
    assert error.value.code == "prompt_review_unavailable"

    # 段集合不符 / hidden 篡改防护：校验先于归档，坏请求不动任何文件。
    store3, pipeline3, project_id3, project3 = _synthetic_awaiting_project(
        tmp_path / "c", [("p\n\n", "editable", HIDDEN_BLOCK, "\n\ns")],
    )
    store3.update(
        project_id3,
        status="succeeded",
        has_video=True,
        project_progress={
            "status": "succeeded", "percent": 100, "stage": "completed",
        },
    )
    with pytest.raises(PromptReviewError) as error:
        pipeline3.regenerate_with_edited_prompts(
            project_id3,
            [{"segment_index": 1, "prompt": "x"}, {"segment_index": 2, "prompt": "y"}],
        )
    assert error.value.code == "invalid_prompt_review_edit"
    assert not (project3 / "output" / "history").exists()
    forged = "edited\n\n" + prompt_review.HIDDEN_BLOCK_START + "\nforged\nEND"
    with pytest.raises(PromptReviewError):
        # 段集合对但编辑超长：还是 422；且归档未发生。
        pipeline3.regenerate_with_edited_prompts(
            project_id3, [{"segment_index": 1, "prompt": "x" * 60_000}],
        )
    assert not (project3 / "output" / "history").exists()
    assert "generation_history" not in store3.read(project_id3)


def test_generation_history_reader(tmp_path, monkeypatch):
    store, pipeline, project_id, project, picsart = _create_review_project(
        tmp_path, monkeypatch, review=False,
    )
    pipeline.run(project_id)
    state = store.read(project_id)
    # 从未 regenerate：只有当前版本。
    versions = prompt_review.generation_history(project, state)
    assert len(versions) == 1
    assert versions[0]["current"] is True
    assert versions[0]["video_file"] == "generated.mp4"
    assert versions[0]["version"] == 1
    assert [s["segment_index"] for s in versions[0]["segments"]] == [1, 2]

    review = _read_review(project)
    pipeline.regenerate_with_edited_prompts(project_id, [
        {"segment_index": 1, "prompt": "regen one"},
        {"segment_index": 2, "prompt": review["segments"][1]["editable"]},
    ])
    state = store.read(project_id)
    versions = prompt_review.generation_history(project, state)
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["current"] is False
    assert versions[0]["video_file"] == "history/v1/generated.mp4"
    assert versions[1]["current"] is True
    assert versions[1]["video_file"] == "generated.mp4"
    # 归档版本的提示词从归档 final prompt 剥出 editable（不含注入/权威块）。
    archived = versions[0]["segments"]
    assert [s["segment_index"] for s in archived] == [1, 2]
    for segment in archived:
        assert "editing task on Video 1" not in segment["prompt"]
        assert prompt_review.HIDDEN_BLOCK_START not in segment["prompt"]
        assert segment["start_s"] is not None and segment["end_s"] is not None
    # 归档可编辑部分 = 当时的 editable。
    assert archived[0]["prompt"] == review["segments"][0]["editable"]

    # 损坏的 generation_history：fail closed 回空历史（只藏不炸）。
    store.update(project_id, generation_history=[{"version": "v1"}])
    assert prompt_review.generation_history(
        project, store.read(project_id),
    ) == []


def test_generation_history_field_roundtrips_through_store(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create(
        _effective_request(), "history-roundtrip",
    )
    entry = {
        "version": 1,
        "completed_at": "2026-09-21T00:00:00+00:00",
        "video": {
            "path": "output/history/v1/generated.mp4",
            "size_bytes": 10,
            "sha256": "a" * 64,
        },
        "segments": [],
    }
    store.update(project_id, generation_history=[entry])
    assert store.read(project_id)["generation_history"] == [entry]


# ---------- regenerate / generation-history 内部端点 ----------


class _RegeneratePipeline(_PromptReviewPipeline):
    def __init__(self, store: ProjectStore) -> None:
        super().__init__(store)
        self.regenerated: list[tuple[str, list]] = []

    def prepare_regenerate_prompts(self, project_id: str, segments, *, pass_name: str):
        assert pass_name == "pass2"
        state = self.store.read(project_id)
        if state.get("status") != "succeeded":
            raise PromptReviewError(
                "regenerate_not_succeeded", "只有已成功项目可以编辑提示词重新生成"
            )
        return {"segments": segments}

    def execute_regenerate(self, project_id: str, prepared) -> None:
        self.regenerated.append((project_id, prepared["segments"]))


def _regenerate_factory(_settings, store, _register_asset):
    return _RegeneratePipeline(store)


def _mark_succeeded(app, project_id: str, segments) -> None:
    _mark_awaiting(app, project_id, segments)
    store = app.state.store
    project = store.path(project_id)
    output = project / "output" / "generated.mp4"
    output.write_bytes(b"current-video")
    store.update(
        project_id,
        status="succeeded",
        has_video=True,
        output_path="output/generated.mp4",
        project_progress={
            "status": "succeeded", "percent": 100, "stage": "completed",
        },
        generation={"status": "succeeded", "stage": "completed", "error": None},
    )


def test_internal_regenerate_endpoint(tmp_path):
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_regenerate_factory,
    )
    try:
        with TestClient(app) as client:
            queued = _create_project(app)
            rejected = client.post(
                f"/api/conversations/{queued}/regenerate",
                headers=AUTH,
                json={"segments": [{"segment_index": 1, "prompt": "x"}]},
            )
            assert rejected.status_code == 409
            assert rejected.json()["detail"]["code"] == "regenerate_not_succeeded"

            project_id = _create_project(app)
            _mark_succeeded(app, project_id, [(1, "body one", 0.0, 8.0)])
            regenerate = client.post(
                f"/api/conversations/{project_id}/regenerate",
                headers=AUTH,
                json={"segments": [{"segment_index": 1, "prompt": "edited one"}]},
            )
            assert regenerate.status_code == 202
            assert regenerate.json()["status"] == "queued"
            assert app.state.pipeline.regenerated == [(
                project_id, [{"segment_index": 1, "prompt": "edited one"}],
            )]
            state = _wait_status(client, project_id, "succeeded")
            assert state["has_video"] is True
            assert app.state.pipeline.resumed == [project_id]

            # 回到 succeeded 后可以再次 regenerate。
            again = client.post(
                f"/api/conversations/{project_id}/regenerate",
                headers=AUTH,
                json={"segments": [{"segment_index": 1, "prompt": "edited twice"}]},
            )
            assert again.status_code == 202
            assert len(app.state.pipeline.regenerated) == 2

            bad_body = client.post(
                f"/api/conversations/{project_id}/regenerate",
                headers=AUTH,
                json={"segments": [], "extra": 1},
            )
            assert bad_body.status_code == 400
            assert client.post(
                f"/api/conversations/{project_id}/regenerate",
            ).status_code == 401
    finally:
        app.state.pipeline.release.set()


def test_internal_generation_history_endpoint_and_files_route(tmp_path):
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_regenerate_factory,
    )
    try:
        with TestClient(app) as client:
            store = app.state.store
            project_id = _create_project(app)
            _mark_succeeded(app, project_id, [(1, "current body", 0.0, 8.0)])
            project = store.path(project_id)

            # 空历史：只有当前版本。
            read = client.get(
                f"/api/conversations/{project_id}/generation-history", headers=AUTH,
            )
            assert read.status_code == 200
            versions = read.json()["versions"]
            assert len(versions) == 1
            assert versions[0]["current"] is True
            assert versions[0]["video_file"] == "generated.mp4"
            assert versions[0]["segments"][0]["prompt"] == "current body"

            # 手工构造一个归档版本（v1），验证多版本 + editable 剥离 + files 路由。
            archive = project / "output" / "history" / "v1"
            (archive / "prompts").mkdir(parents=True)
            (archive / "generated.mp4").write_bytes(b"archived-video")
            archived_prompt = NEW_PREFIX + "archived editable\n\n" + HIDDEN_BLOCK + SUFFIX
            prompt_path = archive / "prompts" / "segment-0001.txt"
            prompt_path.write_text(archived_prompt, encoding="utf-8")
            store.update(
                project_id,
                generation_history=[{
                    "version": 1,
                    "completed_at": "2026-09-20T10:00:00+00:00",
                    "video": {
                        "path": "output/history/v1/generated.mp4",
                        "size_bytes": len(b"archived-video"),
                        "sha256": hashlib.sha256(b"archived-video").hexdigest(),
                    },
                    "segments": [{
                        "segment_index": 1,
                        "prompt_sha256": hashlib.sha256(
                            archived_prompt.encode("utf-8")
                        ).hexdigest(),
                        "prompt_path": "output/history/v1/prompts/segment-0001.txt",
                        "start_s": 0.0,
                        "end_s": 8.0,
                    }],
                }],
            )
            read = client.get(
                f"/api/conversations/{project_id}/generation-history", headers=AUTH,
            )
            versions = read.json()["versions"]
            assert [v["version"] for v in versions] == [1, 2]
            assert versions[0]["current"] is False
            assert versions[0]["completed_at"] == "2026-09-20T10:00:00+00:00"
            assert versions[0]["video_file"] == "history/v1/generated.mp4"
            assert versions[0]["segments"] == [{
                "segment_index": 1,
                "prompt": "archived editable\n\n",
                "start_s": 0.0,
                "end_s": 8.0,
            }]
            assert versions[1]["current"] is True

            served = client.get(
                f"/api/conversations/{project_id}/files/history/v1/generated.mp4",
                headers=AUTH,
            )
            assert served.status_code == 200
            assert served.content == b"archived-video"
            assert client.get(
                f"/api/conversations/{project_id}/files/history/v9/generated.mp4",
                headers=AUTH,
            ).status_code == 404
            assert client.get(
                f"/api/conversations/{project_id}/files/history/v1/prompts/segment-0001.txt",
                headers=AUTH,
            ).status_code == 404
            # 没有历史记录的项目不放行 history 路径。
            bare = _create_project(app)
            assert client.get(
                f"/api/conversations/{bare}/files/history/v1/generated.mp4",
                headers=AUTH,
            ).status_code == 404
    finally:
        app.state.pipeline.release.set()


# ---------- regenerate 状态窗口（原子翻转）与归档回滚 ----------


class _BlockingRegeneratePipeline(_RegeneratePipeline):
    """execute 时断言状态已原子翻转（同步端点里阻塞会卡死事件循环，所以
    在 execute 内部读 store 验证时序，而不是从并发 HTTP 请求观察）。"""

    def __init__(self, store: ProjectStore) -> None:
        super().__init__(store)
        self.window_states: list[tuple[str, bool]] = []

    def execute_regenerate(self, project_id: str, prepared) -> None:
        state = self.store.read(project_id)
        self.window_states.append((state["status"], state["has_video"]))
        super().execute_regenerate(project_id, prepared)


def _blocking_regenerate_factory(_settings, store, _register_asset):
    return _BlockingRegeneratePipeline(store)


def test_regenerate_state_flip_precedes_archive_window(tmp_path):
    """状态迁移原子且先于归档：execute 执行时读到的已是 queued+无视频。"""
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_blocking_regenerate_factory,
    )
    try:
        with TestClient(app) as client:
            project_id = _create_project(app)
            _mark_succeeded(app, project_id, [(1, "body one", 0.0, 8.0)])
            # 翻转前：succeeded + 旧片可读。
            assert client.get(
                f"/api/conversations/{project_id}/files/generated.mp4",
                headers=AUTH,
            ).status_code == 200
            response = client.post(
                f"/api/conversations/{project_id}/regenerate",
                headers=AUTH,
                json={"segments": [{"segment_index": 1, "prompt": "edited"}]},
            )
            assert response.status_code == 202
            # execute（归档/重写）执行期间观察到的一定是翻转后的状态。
            assert app.state.pipeline.window_states == [("queued", False)]

            # 翻转后的窗口态：旧片即使文件还在盘上也不可读（404）。
            project_id2 = _create_project(app)
            _mark_succeeded(app, project_id2, [(1, "body", 0.0, 8.0)])
            app.state.store.update(
                project_id2,
                status="queued",
                has_video=False,
                project_progress={
                    "status": "queued", "percent": 0,
                    "stage": "picsart_resume_queued",
                },
            )
            served = client.get(
                f"/api/conversations/{project_id2}/files/generated.mp4",
                headers=AUTH,
            )
            assert served.status_code == 404
    finally:
        app.state.pipeline.release.set()


class _FailingRegeneratePipeline(_RegeneratePipeline):
    def execute_regenerate(self, project_id: str, prepared) -> None:
        raise OSError("simulated disk full")


def _failing_regenerate_factory(_settings, store, _register_asset):
    return _FailingRegeneratePipeline(store)


def test_regenerate_archive_failure_marks_failed_not_fake_success(tmp_path):
    """归档失败：项目标 failed（带明确 error），不得停在假成功态。"""
    app = create_app(
        _endpoint_settings(tmp_path), pipeline_factory=_failing_regenerate_factory,
    )
    try:
        with TestClient(app) as client:
            project_id = _create_project(app)
            _mark_succeeded(app, project_id, [(1, "body one", 0.0, 8.0)])
            response = client.post(
                f"/api/conversations/{project_id}/regenerate",
                headers=AUTH,
                json={"segments": [{"segment_index": 1, "prompt": "edited"}]},
            )
            assert response.status_code == 500
            assert response.json()["detail"]["code"] == "regenerate_failed"
            state = app.state.store.read(project_id)
            assert state["status"] == "failed"
            assert state["error"]["code"] == "regenerate_failed"
            assert state["has_video"] is False
    finally:
        app.state.pipeline.release.set()


def test_regenerate_archive_failure_rolls_back_files(tmp_path, monkeypatch):
    """归档中途失败：已移动的分段成片搬回原位，半成品归档目录删除。"""
    store, pipeline, project_id, project, _picsart = _create_review_project(
        tmp_path, monkeypatch, review=False,
    )
    pipeline.run(project_id)
    assert store.read(project_id)["status"] == "succeeded"
    review = _read_review(project)
    edits = [
        {"segment_index": s["segment_index"], "prompt": s["editable"]}
        for s in review["segments"]
    ]
    segment_outputs = [
        project / "work" / "segments" / f"{index:04d}" / "generated.mp4"
        for index in (1, 2)
    ]
    assert all(path.is_file() for path in segment_outputs)

    # 第一阶段：复制即失败，什么都没移动。
    def boom_copy2(*_args, **_kwargs):
        raise OSError("simulated ENOSPC")

    with monkeypatch.context() as m:
        m.setattr(shutil, "copy2", boom_copy2)
        with pytest.raises(OSError, match="ENOSPC"):
            pipeline.regenerate_with_edited_prompts(project_id, edits)
    assert not (project / "output" / "history").exists()
    assert all(path.is_file() for path in segment_outputs)

    # 第二阶段：第二次移动失败，第一次移动必须回滚。
    original_replace = os.replace
    moved: list = []
    failed = {"done": False}

    def flaky_replace(source, destination):
        if moved and not failed["done"]:
            failed["done"] = True
            raise OSError("simulated ENOSPC")
        moved.append((source, destination))
        return original_replace(source, destination)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", flaky_replace)
        with pytest.raises(OSError, match="ENOSPC"):
            pipeline.regenerate_with_edited_prompts(project_id, edits)
    assert not (project / "output" / "history" / "v1").exists()
    assert all(path.is_file() for path in segment_outputs)

    state = store.read(project_id)
    # 两次失败都不留痕迹：无历史记录、pass2 账本未清、清单回执仍指向原提示词。
    assert "generation_history" not in state
    assert state["picsart_submission_receipts_pass2"] is not None
    assert state["status"] == "succeeded"

    # 回滚干净后可以正常 regenerate（不挡重试）。
    pipeline.regenerate_with_edited_prompts(project_id, edits)
    assert (
        project / "output" / "history" / "v1" / "generated.mp4"
    ).is_file()
    assert len(store.read(project_id)["generation_history"]) == 1
