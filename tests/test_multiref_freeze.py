from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.object_replace_schema import OBJECT_REPLACE_OUTPUT_VERSION
from app.pipeline import (
    Pipeline,
    REPLACEMENT_SHEET_VERSION,
)
from app.store import ProjectStore
from test_pipeline import (
    _FlowAutodlTTS,
    _FlowDeepSeek,
    _FlowMediaKit,
    _FlowPicsart,
    _FlowSeedream,
    _effective_request as _pipeline_request,
    _install_fake_media,
    _settings as _pipeline_settings,
)


def _request() -> dict[str, Any]:
    return {
        "version": 1,
        "output": {"aspect_ratio": "16:9", "resolution": "768p", "fit_mode": "auto"},
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {"mode": "auto_rewrite", "target_language": "same_as_source"},
        "replacement_guidance": {"instruction": "replace the product"},
        "fusion_agent": "deepseek",
    }


def _settings(path: Path) -> Settings:
    return Settings(
        access_token="test-token",
        data_dir=path,
        public_base_url="https://pipeline.test",
        enable_mediakit=False,
        provider_poll_interval_s=0,
    )


def _project_with_replacements(
    tmp_path: Path,
    *,
    count: int,
) -> tuple[ProjectStore, str, Path, dict[str, Any]]:
    store = ProjectStore(tmp_path / "projects")
    request = _request()
    project_id, project, _state = store.create(request, "multiref-freeze")
    source = project / "work" / "segments" / "0001" / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    values = ["inputs/replacement-image", *[
        f"inputs/replacement-image-{order}" for order in range(2, count + 1)
    ]]
    for order, relative_path in enumerate(values, 1):
        (project / relative_path).write_bytes(f"replacement-{order}".encode())
    item = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "visual_prompt": "visual",
        "dialogue": [],
        "source": source,
    }
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path=values[0],
        replacement_image_paths=values,
        enhanced_element_index={"people": {}, "entities": {}, "scenes": {}, "relations": {}},
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        replacement_sheet={"version": REPLACEMENT_SHEET_VERSION},
    )
    return store, project_id, project, item


def test_replacement_sheet_bindings_include_ordered_images_only_for_multi(
    tmp_path: Path,
) -> None:
    store, project_id, project, item = _project_with_replacements(
        tmp_path, count=3,
    )
    state = store.read(project_id)
    bindings = Pipeline._replacement_sheet_upstream_bindings(
        project,
        [item],
        state["enhanced_element_index"],
        state["object_replace"],
        state["voice_references"],
        state["effective_request"],
        project / "inputs" / "replacement-image",
    )

    assert bindings["replacement_image"]["path"] == "inputs/replacement-image"
    assert [entry["path"] for entry in bindings["replacement_images"]] == [
        "inputs/replacement-image",
        "inputs/replacement-image-2",
        "inputs/replacement-image-3",
    ]
    assert bindings["replacement_images"][0] == bindings["replacement_image"]

    store, project_id, project, item = _project_with_replacements(
        tmp_path / "single", count=1,
    )
    state = store.read(project_id)
    single = Pipeline._replacement_sheet_upstream_bindings(
        project,
        [item],
        state["enhanced_element_index"],
        state["object_replace"],
        state["voice_references"],
        state["effective_request"],
        project / "inputs" / "replacement-image",
    )
    assert "replacement_images" not in single


def test_h3_manifest_rejects_multi_reference_order_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, project_id, project, item = _project_with_replacements(
        tmp_path, count=3,
    )
    muted = project / "work" / "segments" / "0001" / "source-muted.mp4"
    muted.write_bytes(b"muted")
    (muted.parent / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        Pipeline,
        "_validated_stored_replacement_sheet",
        classmethod(lambda _cls, *_args, **_kwargs: None),
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(item, "prompt", muted, "b" * 64)],
        replacement=project / "inputs" / "replacement-image",
        replacement_sheet={"version": REPLACEMENT_SHEET_VERSION},
        request=_request(),
    )

    state = store.read(project_id)
    bindings = state["h3_prepared_artifacts"]["bindings"]
    assert [entry["path"] for entry in bindings["replacement_images"]] == [
        "inputs/replacement-image",
        "inputs/replacement-image-2",
        "inputs/replacement-image-3",
    ]
    assert bindings["replacement_image"] == bindings["replacement_images"][0]
    assert Pipeline._validated_h3_prepared_artifacts(project, state, [item])

    bindings["replacement_images"][1], bindings["replacement_images"][2] = (
        bindings["replacement_images"][2],
        bindings["replacement_images"][1],
    )
    store.update(project_id, h3_prepared_artifacts=state["h3_prepared_artifacts"])
    with pytest.raises(ValueError, match="replacement bindings do not match"):
        Pipeline._validated_h3_prepared_artifacts(
            project, store.read(project_id), [item],
        )


def test_pipeline_multi_reference_inputs_are_frozen_and_r2v_resume_reuses_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _pipeline_request(
        target_language="日语",
        replacement_instruction="按用户参考图替换对应物品",
    )
    project_id, project, _state = store.create(request, "multiref-run")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    replacement_paths = [
        "inputs/replacement-image",
        "inputs/replacement-image-2",
        "inputs/replacement-image-3",
        "inputs/replacement-image-4",
    ]
    for order, relative_path in enumerate(replacement_paths, 1):
        (project / relative_path).write_bytes(
            f"replacement-{order}".encode("ascii"),
        )
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path=replacement_paths[0],
        replacement_image_paths=replacement_paths,
    )
    media_events: list[str] = []
    _install_fake_media(monkeypatch, media_events)
    deepseek = _FlowDeepSeek()
    seedream = _FlowSeedream(deepseek.events)
    picsart = _FlowPicsart()
    registered: list[Path] = []
    pipeline = Pipeline(
        _pipeline_settings(tmp_path, enable_mediakit=True),
        store,
        lambda _project_id, path: (
            registered.append(path)
            or f"https://assets.invalid/{path.parent.name}/{path.name}"
        ),
        deepseek=deepseek,
        mediakit=_FlowMediaKit(),
        picsart=picsart,
        seedream=seedream,
        autodl=_FlowAutodlTTS(),
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    frozen_references = state["replacement_sheet"]["h3_reference_images"]
    expected_h3_paths = [
        project / "work" / "global-reference-sheet" / "reference-sheet.png",
        project / "work" / "global-reference-sheet" / "user-reference.png",
        project / "work" / "global-reference-sheet" / "user-reference-2.png",
        project / "work" / "global-reference-sheet" / "user-reference-3.png",
        project / "work" / "global-reference-sheet" / "user-reference-4.png",
    ]
    assert state["status"] == "succeeded"
    assert [reference["image_reference"] for reference in frozen_references] == [
        "Image 1", "Image 2", "Image 3", "Image 4", "Image 5",
    ]
    assert [reference["path"] for reference in frozen_references[1:]] == [
        "work/global-reference-sheet/user-reference.png",
        "work/global-reference-sheet/user-reference-2.png",
        "work/global-reference-sheet/user-reference-3.png",
        "work/global-reference-sheet/user-reference-4.png",
    ]
    assert len(seedream.calls) == 1
    assert [path.name for path in seedream.calls[0]["images"]] == [
        "canvas.png",
        "user-reference.png",
        "user-reference-2.png",
        "user-reference-3.png",
        "user-reference-4.png",
    ]
    fusion_calls = [
        call for call in deepseek.structured_calls
        if call["payload"].get("segment_index") in {1, 2}
        and "h3_reference_images" in call["payload"]
    ]
    assert len(fusion_calls) == 2
    assert all(
        [path.resolve() for path in call["images"]]
        == [path.resolve() for path in expected_h3_paths]
        for call in fusion_calls
    )
    assert len(picsart.submits) == 2
    expected_image_urls = [
        f"https://assets.invalid/global-reference-sheet/{path.name}"
        for path in expected_h3_paths
    ]
    assert all(
        call["image_urls"] == expected_image_urls
        for call in picsart.submits
    )
    manifest = state["h3_prepared_artifacts"]
    assert [entry["path"] for entry in manifest["bindings"]["replacement_images"]] == replacement_paths

    frozen_prompt = (
        project / "work" / "segments" / "0001" / "h3_prompt.txt"
    ).read_bytes()
    call_counts = (len(deepseek.structured_calls), len(seedream.calls), len(picsart.submits))
    store.update(
        project_id,
        status="failed",
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "retry",
        },
    )

    pipeline.validate_r2v_resume(project_id)
    pipeline.resume_r2v(project_id)

    assert (
        len(deepseek.structured_calls), len(seedream.calls), len(picsart.submits),
    ) == call_counts
    assert (
        project / "work" / "segments" / "0001" / "h3_prompt.txt"
    ).read_bytes() == frozen_prompt
    assert store.read(project_id)["status"] == "succeeded"
