"""两次出片（two-pass H3）管线：pass1 物品白块换物、pass2 人物黑块换人。

pass1：仅物品组 SAM 检测 + 白块遮罩 → 静音出片 generated-pass1.mp4
（不挂 Audio 1，suffix 用无台词变体）；pass2：以 generated-pass1.mp4 为源，
仅人物组（person, hair）检测 + 黑块遮罩 → 挂 dialogue-track.wav 出最终成片。
两遍各自冻结清单（v8，含 "pass" 字段）与回执账本（v3，按 pass 分键），
pass2 清单的 bindings 钉 pass1_manifest_sha256 + 各段 pass1 产物回执。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app import media
from app import object_detect
from app import object_mask
from app import prompt_review
from app.pipeline import (
    H3_PREPARED_ARTIFACTS_VERSION,
    PICSART_RECEIPT_VERSION,
    Pipeline,
)
from app.prompt_review import PromptReviewError
from app.research_log import ResearchLog
from app.store import ProjectStore

from test_pipeline import (
    _effective_request,
    _FlowPicsart,
    _materialize_test_replacement_sheet,
    _prepare_h3_object_mask_fixture,
    _prepare_h3_person_fixture,
    _run_h3_with_picsart,
    _settings,
)


def _enhanced_index_with_person_and_object() -> dict[str, Any]:
    return {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {
            "entity-01": {
                "source_visual_description": "the red scarf",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "user_request",
                    "source": "image_1",
                    "detect_label": "scarf",
                    "target_visual_description": "a blue silk scarf",
                },
            },
        },
        "scenes": {},
        "relations": {},
    }


def test_element_index_view_for_pass1_projects_people_to_keep() -> None:
    index = _enhanced_index_with_person_and_object()

    view = Pipeline._element_index_view_for_pass(index, "pass1")

    person = view["people"]["person-01"]
    assert "replacement" not in person
    # 投影为 KEEP 的同时声明人物保持原样（注入 preserve 文案进权威块）。
    assert any("not replaced in this pass" in item for item in person["preserve"])
    # 物品在 pass1 照常 REPLACE。
    assert "replacement" in view["entities"]["entity-01"]
    # 视图是拷贝：原始索引不被改写。
    assert "replacement" in index["people"]["person-01"]
    assert "preserve" not in index["people"]["person-01"]
    authority = Pipeline._fusion_authority_block(view, 1)
    assert "- person-01 | KEEP |" in authority
    assert "- entity-01 | REPLACE |" in authority


def test_element_index_view_for_pass2_projects_entities_scenes_to_keep() -> None:
    index = _enhanced_index_with_person_and_object()
    index["scenes"] = {
        "scene-01": {
            "source_visual_description": "the studio",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replacement": {
                "reason": "automatic",
                "source": "generated_variant",
                "target_visual_description": "a different studio",
            },
        },
    }

    view = Pipeline._element_index_view_for_pass(index, "pass2")

    entity = view["entities"]["entity-01"]
    scene = view["scenes"]["scene-01"]
    assert "replacement" not in entity
    assert "replacement" not in scene
    # 措辞声明 Video 1 中已是替换好的新物品/场景，外观必须原样保留。
    for record in (entity, scene):
        assert any(
            "already the replaced" in item and "as-is" in item
            for item in record["preserve"]
        )
    # 人物在 pass2 照常 REPLACE。
    assert "replacement" in view["people"]["person-01"]
    assert "replacement" in index["entities"]["entity-01"]
    authority = Pipeline._fusion_authority_block(view, 1)
    assert "- entity-01 | KEEP |" in authority
    assert "- scene-01 | KEEP |" in authority
    assert "- person-01 | REPLACE |" in authority


def test_element_index_view_rejects_unknown_pass() -> None:
    with pytest.raises(ValueError, match="pass"):
        Pipeline._element_index_view_for_pass(
            _enhanced_index_with_person_and_object(), "pass3",
        )


def test_pass1_prompt_masks_only_objects_without_face_block_or_dialogue(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet, object_calls, object_kwargs = (
        _prepare_h3_object_mask_fixture(tmp_path, monkeypatch)
    )
    # 段内有台词：pass1 也必须用无台词尾（静音出片不挂 Audio 1）。
    item["dialogue"] = [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}]
    segment_dir = project / "work" / "segments" / "0001"

    _item, prompt, muted, _fusion_sha256 = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
        pass_name="pass1",
    )

    # 仅物品组检测：人物组为空；遮罩产物沿用物品白块文件名。
    assert object_calls == [(
        segment_dir / "source-muted.mp4",
        segment_dir / "source-objects-masked.mp4",
        ["the red scarf around the neck"],
    )]
    assert object_kwargs[0]["person_targets"] == []
    assert muted == segment_dir / "source-objects-masked.mp4"
    # pass1 不注入人物替换块、不含人物黑块文案。
    assert "PERSON REPLACEMENT" not in prompt
    assert "(0,0,0)" not in prompt
    assert "white" in prompt.lower()
    # 权威块：人物投影为 KEEP、物品照常 REPLACE。
    assert "- person-01 | KEEP |" in prompt
    assert "- entity-01 | REPLACE |" in prompt
    # 静音出片：无台词尾变体，不提 Audio 1。
    assert "Dialogue requirements" not in prompt
    assert "Do not add captions, subtitles, or new dialogue." in prompt
    frozen = (
        project / "work" / "segments" / "0001" / "h3_prompt-pass1.txt"
    ).read_text(encoding="utf-8")
    assert frozen == prompt


def test_pass2_prompt_masks_only_persons_with_face_block_and_dialogue(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet, object_calls, object_kwargs = (
        _prepare_h3_object_mask_fixture(tmp_path, monkeypatch)
    )
    item["dialogue"] = [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}]
    segment_dir = project / "work" / "segments" / "0001"

    _item, prompt, muted, _fusion_sha256 = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
        pass_name="pass2",
    )

    # 仅人物组检测：物品组为空；遮罩产物是 pass2 专名。
    assert object_calls == [(
        segment_dir / "source-pass2-muted.mp4",
        segment_dir / "source-pass2-persons-masked.mp4",
        [],
    )]
    assert object_kwargs[0]["person_targets"] == ["person", "hair"]
    assert muted == segment_dir / "source-pass2-persons-masked.mp4"
    # 人物黑块文案不提物品块；人物替换块在 pass2 注入。
    assert "Solid black (0,0,0) blocks fully conceal the people" in prompt
    assert "white" not in prompt.lower()
    assert "PERSON REPLACEMENT IS MANDATORY" in prompt
    # 权威块：物品投影为 KEEP 且声明已是替换好的新物品；人物照常 REPLACE。
    assert "- entity-01 | KEEP |" in prompt
    assert "already the replaced" in prompt
    assert "- person-01 | REPLACE |" in prompt
    # pass2 挂 Audio 1：Dialogue requirements 尾。
    assert "Dialogue requirements: Audio 1" in prompt
    frozen = (
        project / "work" / "segments" / "0001" / "h3_prompt-pass2.txt"
    ).read_text(encoding="utf-8")
    assert frozen == prompt


def test_two_pass_full_run_produces_pass1_and_pass2_artifacts(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    picsart = _FlowPicsart()
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    # 双清单：v8 + pass 字段；旧键不再存在。
    assert "h3_prepared_artifacts" not in state
    for pass_name in ("pass1", "pass2"):
        manifest = state[f"h3_prepared_artifacts_{pass_name}"]
        assert manifest["version"] == H3_PREPARED_ARTIFACTS_VERSION
        assert manifest["pass"] == pass_name
        ledger = state[f"picsart_submission_receipts_{pass_name}"]
        assert ledger["version"] == PICSART_RECEIPT_VERSION
        assert ledger["pass"] == pass_name
    # pass2 清单钉 pass1 清单哈希与各段 pass1 产物回执。
    pass2_bindings = state["h3_prepared_artifacts_pass2"]["bindings"]
    assert pass2_bindings["pass1_manifest_sha256"] == Pipeline._json_sha256(
        state["h3_prepared_artifacts_pass1"]
    )
    assert set(pass2_bindings["pass1_outputs"]) == {"1", "2"}
    for index in (1, 2):
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        assert (segment_dir / "generated-pass1.mp4").is_file()
        assert (segment_dir / "h3_prompt-pass1.txt").is_file()
        assert (segment_dir / "h3_prompt-pass2.txt").is_file()
        receipt = pass2_bindings["pass1_outputs"][str(index)]
        assert receipt["path"] == f"work/segments/{index:04d}/generated-pass1.mp4"
    # 4 次提交 = 2 段 × 2 遍；pass1 一律不挂 Audio 1，pass2 按台词挂。
    assert len(picsart.submits) == 4
    # 以权威块的 pass 投影措辞分类（无台词的 pass2 段也用无台词尾，不能
    # 按 suffix 分类）。
    pass1_submits = [
        call for call in picsart.submits
        if "people are not replaced in this pass" in call["prompt"]
    ]
    pass2_submits = [
        call for call in picsart.submits
        if "people are not replaced in this pass" not in call["prompt"]
    ]
    assert len(pass1_submits) == 2
    assert all(call["audio_urls"] == [] for call in pass1_submits)
    assert len(pass2_submits) == 2
    # 全流程 fixture 只有段 1 有台词。
    assert sorted(
        len(call["audio_urls"]) for call in pass2_submits
    ) == [0, 1]


def _frozen_two_pass_project(
    tmp_path: Path, monkeypatch: Any,
) -> tuple[ProjectStore, str, Path]:
    """跑通两遍全流程的 failed 项目脚手架（供冻结/账本/门禁测试）。"""
    store, project_id, project = _run_h3_with_picsart(
        tmp_path, monkeypatch, _FlowPicsart(),
    )
    store.update(project_id, status="failed")
    return store, project_id, project


def test_pass2_manifest_validation_fails_when_pass1_output_changes(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, project = _frozen_two_pass_project(tmp_path, monkeypatch)
    state = store.read(project_id)
    segments = state["segments"]

    # 哈希链完好时校验通过。
    Pipeline._validated_h3_prepared_artifacts(
        project, state, segments, "pass2",
    )

    # 改动任一 pass1 产物 → pass2 清单校验失败。
    (project / "work" / "segments" / "0001" / "generated-pass1.mp4").write_bytes(
        b"tampered-pass1-output"
    )
    with pytest.raises(ValueError, match="does not match"):
        Pipeline._validated_h3_prepared_artifacts(
            project, store.read(project_id), segments, "pass2",
        )


def test_receipt_ledgers_are_isolated_per_pass(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    _store, project_id, project = _frozen_two_pass_project(
        tmp_path, monkeypatch,
    )
    state = store_read = _store.read(project_id)
    del store_read
    ledger1 = state["picsart_submission_receipts_pass1"]
    ledger2 = state["picsart_submission_receipts_pass2"]
    sha1 = Pipeline._json_sha256(state["h3_prepared_artifacts_pass1"])
    sha2 = Pipeline._json_sha256(state["h3_prepared_artifacts_pass2"])
    assert sha1 != sha2
    for receipt in ledger1["segments"]["1"]:
        assert receipt["preparation_sha256"] == sha1
        assert receipt["run_identity"].startswith("picsart-pass1")
    for receipt in ledger2["segments"]["1"]:
        assert receipt["preparation_sha256"] == sha2
        assert receipt["run_identity"].startswith("picsart")
        assert not receipt["run_identity"].startswith("picsart-pass1")


def test_prompt_review_pauses_only_before_pass2(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    request = _effective_request(
        target_language="日语",
        replacement_instruction="把主角替换成用户参考形象",
    )
    request["prompt_review"] = True
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "two-pass-review")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    (project / "inputs" / "replacement-image").write_bytes(b"replacement")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
    )
    from test_pipeline import (
        _FlowDeepSeek,
        _FlowDoubaoTTS,
        _FlowMediaKit,
        _FlowSeedream,
        _install_fake_media,
    )
    _install_fake_media(monkeypatch, [])
    deepseek = _FlowDeepSeek()
    picsart = _FlowPicsart()
    pipeline = Pipeline(
        _settings(tmp_path, enable_mediakit=True),
        store,
        lambda _project_id, path: (
            f"https://assets.invalid/{path.parent.name}/{path.name}"
        ),
        deepseek=deepseek,
        mediakit=_FlowMediaKit(),
        picsart=picsart,
        seedream=_FlowSeedream(deepseek.events),
        tts=_FlowDoubaoTTS(),
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    # pass1 不暂停：已付费提交完成；暂停发生在 pass2 冻结后、提交前。
    assert state["status"] == "awaiting_prompt_review"
    assert state.get("prompt_review_pending_pass") == "pass2"
    assert len(picsart.submits) == 2  # 全部属于 pass1
    assert state["picsart_submission_receipts_pass1"] is not None
    assert "picsart_submission_receipts_pass2" not in state
    assert state["h3_prepared_artifacts_pass2"]["pass"] == "pass2"
    # 审核文件按 pass 区分。
    assert prompt_review.read_review(project, "pass1") is not None
    assert prompt_review.read_review(project, "pass2") is not None
    assert not (project / "work" / "prompt-review.json").exists()


def test_regenerate_rejects_pass1_prompt_edits(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, _project = _frozen_two_pass_project(tmp_path, monkeypatch)
    store.update(project_id, status="succeeded", has_video=True)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )

    with pytest.raises(PromptReviewError, match="pass1"):
        pipeline.prepare_regenerate_prompts(
            project_id,
            [{"segment_index": 1, "prompt": "edited"}],
            pass_name="pass1",
        )


def test_resume_routes_to_latest_unfinished_pass(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, project = _frozen_two_pass_project(tmp_path, monkeypatch)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )

    # 两遍清单都在 → 续 pass2。
    *_, pass_name, _extras = pipeline._load_r2v_resume(project_id)
    assert pass_name == "pass2"

    # 只有 pass1 清单（pass2 尚未冻结）→ 续 pass1。
    store.update(
        project_id,
        h3_prepared_artifacts_pass2=None,
        picsart_submission_receipts_pass2=None,
    )
    *_, pass_name, _extras = pipeline._load_r2v_resume(project_id)
    assert pass_name == "pass1"


def test_resume_h3_prep_pass2_keeps_pass1_frozen(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, project = _frozen_two_pass_project(tmp_path, monkeypatch)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    before = store.read(project_id)
    pass1_manifest = before["h3_prepared_artifacts_pass1"]

    loaded = pipeline._load_h3_prep_resume(project_id, pass_name="pass2")
    prepared = loaded[1]
    pipeline._invalidate_h3_prepared_artifacts(
        project_id, project, prepared, ResearchLog(project), pass_name="pass2",
    )

    after = store.read(project_id)
    # pass1 清单与账本原样保留；pass2 清零。
    assert after["h3_prepared_artifacts_pass1"] == pass1_manifest
    assert after["picsart_submission_receipts_pass1"] is not None
    assert after["h3_prepared_artifacts_pass2"] is None
    assert after["picsart_submission_receipts_pass2"] is None
    segment_dir = project / "work" / "segments" / "0001"
    # pass1 产物不动；pass2 派生产物移除。
    assert (segment_dir / "generated-pass1.mp4").is_file()
    assert (segment_dir / "h3_prompt-pass1.txt").is_file()
    assert not (segment_dir / "h3_prompt-pass2.txt").exists()
    assert not (segment_dir / "source-pass2.mp4").exists()


def test_resume_h3_prep_pass1_cascades_to_pass2(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store, project_id, project = _frozen_two_pass_project(tmp_path, monkeypatch)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    loaded = pipeline._load_h3_prep_resume(project_id, pass_name="pass1")
    prepared = loaded[1]
    pipeline._invalidate_h3_prepared_artifacts(
        project_id, project, prepared, ResearchLog(project), pass_name="pass1",
    )

    after = store.read(project_id)
    assert after["h3_prepared_artifacts_pass1"] is None
    assert after["h3_prepared_artifacts_pass2"] is None
    assert after["picsart_submission_receipts_pass1"] is None
    assert after["picsart_submission_receipts_pass2"] is None
    segment_dir = project / "work" / "segments" / "0001"
    assert not (segment_dir / "h3_prompt-pass1.txt").exists()
    assert not (segment_dir / "h3_prompt-pass2.txt").exists()


def test_lean_prune_never_targets_pass1_products(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """ARTIFACT_RETENTION=lean 的裁剪目标不含 work/segments/NNNN 下的
    pass1 产物（regenerate-pass2 要复用；pass2 清单 bindings 也引用它们）。"""
    from app import artifact_retention

    store, project_id, project = _frozen_two_pass_project(tmp_path, monkeypatch)
    # lean 裁剪只接受 succeeded 项目。
    store.update(project_id, status="succeeded")
    state = store.read(project_id)

    summary = artifact_retention.prune_after_success(
        project, state=state, log=ResearchLog(project),
    )

    assert summary.get("pruned") is True
    for index in (1, 2):
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        assert (segment_dir / "generated-pass1.mp4").is_file()
        assert (segment_dir / "h3_prompt-pass1.txt").is_file()
        assert (segment_dir / "h3_prompt-pass2.txt").is_file()
        assert (segment_dir / "generated.mp4").is_file()
