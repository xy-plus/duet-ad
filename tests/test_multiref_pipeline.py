import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.pipeline import Pipeline


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("resume", [False, True])
def test_ordered_references_reach_every_visual_stage(tmp_path, monkeypatch, count, resume):
    pipeline = object.__new__(Pipeline)
    # direct 模式：逐图直传语义（本测试文件覆盖的就是 direct 链路）。
    pipeline.settings = Settings(
        access_token="test",
        multiref_mode="direct",
        public_base_url="https://multiref-pipeline.test",
    )
    monkeypatch.setattr(pipeline, "_generation_config", lambda request: {})
    originals = []
    for order in range(1, count + 1):
        path = tmp_path / "inputs" / ("replacement-image" if order == 1 else f"replacement-image-{order}")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"original-{order}".encode())
        originals.append(path)
    (tmp_path / "state.json").write_text(json.dumps({
        "replacement_image_path": "inputs/replacement-image",
        "replacement_image_paths": [path.relative_to(tmp_path).as_posix() for path in originals],
    }))
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"source-frame")
    events = []
    monkeypatch.setattr(pipeline, "_prepare_segment_materials", lambda *args, **kw: {
        "index": args[3]["index"], "analysis_paths": [frame],
        "segment_manifest": {}, "keyframe_selection": [{"source_time_s": 2.75}],
    })
    def prepass(*args):
        events.append("identity")
        return {"status": "complete", "persons": [], "observations": []}
    monkeypatch.setattr(pipeline, "_person_identity_prepass", prepass)
    monkeypatch.setattr(pipeline, "_materialize_canonical_user_reference", lambda project, path, log, order: path)
    seen = []
    def node(log, name, scope, payload, images, operation, **kw):
        assert events == ["identity"]
        seen.append((payload, list(images)))
        return "source visual description"
    monkeypatch.setattr(pipeline, "_node", node)
    for order in (1, 2):
        (tmp_path / "work" / "segments" / f"{order:04d}").mkdir(parents=True)
    request = {"replacement_guidance": {"instruction": "图1替换包；其他图片按其内容解释"}}
    prepared = pipeline._prepare_project_segments("p", tmp_path, tmp_path / "source", [
        {"index": 2}, {"index": 1},
    ], {1: [], 2: []}, request, SimpleNamespace(event=lambda *a, **kw: None), resume=resume)
    assert len(seen) == 2
    for payload, images in seen:
        assert images == originals + [frame]
        assert payload["user_replacement_instruction"] == request["replacement_guidance"]["instruction"]
        assert [item["reference_id"] for item in payload["user_reference_images"]] == [f"image_{i}" for i in range(1, count + 1)]
        assert payload["frames"][0]["image_reference"] == f"Image {count + 1}"
        assert payload["frames"][0]["source_time_s"] == 2.75
    payload, images = pipeline._project_index_request(tmp_path, prepared, {}, user_replacement_instruction=request["replacement_guidance"]["instruction"])
    assert images == originals + [frame, frame]
    assert len(payload["segments"]) == 2
    assert all(len(segment["frames"]) == 1 for segment in payload["segments"])
    assert [segment["frames"][0]["image_reference"] for segment in payload["segments"]] == [f"Image {count + 1}", f"Image {count + 2}"]
    assert payload["user_replacement_instruction"] == request["replacement_guidance"]["instruction"]


def test_fusion_uses_each_keys_original_reference_set_without_merging_keys():
    entries = {
        "bag-a": {"source": "image_1", "reference_image_ids": ["image_1", "image_3"]},
        "bag-b": {"source": "image_1", "reference_image_ids": ["image_2"]},
        "bag-c": {"source": "image_1", "reference_image_ids": ["image_2"]},
        "lamp": {"source": "generated_variant"},
    }
    index = {"entities": {key: {
        "source_visual_description": key, "replacement": {
            **replacement, "target_visual_description": key,
        }, "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
    } for key, replacement in entries.items()}}
    sheet = {"h3_image_reference": "Image 1", "user_image_present": True,
             "spec": {"tiles": [{"stable_key": key, "tile_id": f"TILE_{i}", "row": 1, "column": i}
                                for i, key in enumerate(entries, 1)]}}
    block = Pipeline._fusion_authority_block(index, 1, sheet)
    lines = {key: next(line for line in block.splitlines() if line.startswith(f"- {key} | REPLACE")) for key in entries}
    assert "original user references Image 2, Image 4 are appearance authority" in lines["bag-a"]
    assert "original user references Image 3 are appearance authority" in lines["bag-b"]
    assert "original user references Image 3 are appearance authority" in lines["bag-c"]
    assert "original user references" not in lines["lamp"]
    assert "TILE_4" in lines["lamp"]
    assert "never copy from Image 2 directly" not in block
