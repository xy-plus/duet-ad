from pathlib import Path
from types import SimpleNamespace
import json
import threading

import pytest

from app.pipeline import Pipeline
from app import person_identity


@pytest.mark.parametrize("resume", [False, True])
def test_project_frame_barrier_precedes_every_agent(tmp_path, monkeypatch, resume):
    pipeline = object.__new__(Pipeline)
    events = []
    barrier = threading.Barrier(2)

    def materialize(*args, **kwargs):
        segment = args[3]
        assert kwargs["resume"] is resume
        barrier.wait(timeout=3)
        events.append(("frames", segment["index"]))
        return segment.copy()

    def prepass(project, source, materials, log):
        assert sorted(events) == [("frames", 1), ("frames", 2)]
        events.append(("identity", 0))
        return {"status": "complete", "persons": [{"canonical_id": "p1"}],
                "observations": [{"segment_index": 2, "source_time_s": 7.25,
                                  "observation_id": "o1", "bbox": [1, 2, 3, 4]}]}

    def analyze(project, item, request, log, **kwargs):
        assert ("identity", 0) in events
        assert item["person_identity"]["persons"] == [{"canonical_id": "p1"}]
        if item["index"] == 2:
            assert item["person_identity"]["observations"][0]["source_time_s"] == 7.25
        else:
            assert item["person_identity"]["observations"] == []
        return item

    monkeypatch.setattr(pipeline, "_prepare_segment_materials", materialize)
    monkeypatch.setattr(pipeline, "_person_identity_prepass", prepass)
    monkeypatch.setattr(pipeline, "_analyze_prepared_segment", analyze)
    result = pipeline._prepare_project_segments(
        "p", tmp_path, tmp_path / "source", [{"index": 2}, {"index": 1}],
        {1: [], 2: []}, {}, object(), resume=resume,
    )
    assert [item["index"] for item in result] == [1, 2]


def test_identity_cache_binds_source_frames_and_model_configuration(tmp_path, monkeypatch):
    pipeline = object.__new__(Pipeline)
    pipeline.store = SimpleNamespace(read=lambda _: {"scenes": [
        {"index": 1, "start_s": 0, "end_s": 9}]})
    (tmp_path / "work" / "source-4fps").mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    source_frame = tmp_path / "work" / "source-4fps" / "f.png"
    source_frame.write_bytes(b"source-frame")
    (source_frame.parent / "manifest.json").write_text(json.dumps({
        "frames": [{"file": "f.png", "time_seconds": 0.25, "index": 1}]}))
    calls = []
    configuration = {"model_sha256": "a"}
    monkeypatch.setattr(person_identity, "identity_configuration_fingerprint", lambda: configuration.copy())
    def analyze(frames, *, source):
        calls.append(frames)
        assert frames[0]["scene_id"] == "SCENE_01"
        assert frames[1]["source_time_s"] == 0.37
        return {"status": "skipped", "reason": "person_identity_model_unavailable_or_invalid", "persons": [], "observations": []}
    monkeypatch.setattr(person_identity, "analyze_person_identity", analyze)
    def node(log, name, scope, inputs, media, operation, **kwargs):
        assert kwargs["retry_if"](RuntimeError()) is False
        return operation()
    monkeypatch.setattr(pipeline, "_node", node)
    events = []
    log = SimpleNamespace(event=lambda *args, **kwargs: events.append((args, kwargs)))
    materials = [{"index": 1, "raw_keyframe_paths": [frame], "keyframe_selection": [
        {"source_time_s": 0.37, "source_scene_id": "SCENE_01", "decode_index": 7}]}]
    for _ in range(2):
        result = pipeline._person_identity_prepass(tmp_path, source, materials, log)
        assert result["reason"] == "person_identity_model_unavailable_or_invalid"
    assert len(calls) == 1
    source.write_bytes(b"different-source")
    pipeline._person_identity_prepass(tmp_path, source, materials, log)
    configuration["model_sha256"] = "b"
    pipeline._person_identity_prepass(tmp_path, source, materials, log)
    materials[0]["keyframe_selection"][0]["decode_index"] = 8
    pipeline._person_identity_prepass(tmp_path, source, materials, log)
    assert len(calls) == 4
    assert any(event[0] == ("person_identity_prepass_reused",) for event in events)


def test_index_payload_retains_exact_time_identity_and_bbox(tmp_path):
    frame = tmp_path / "analysis.png"
    frame.write_bytes(b"pixel-proxy")
    evidence = {"status": "complete", "persons": [
        {"canonical_id": "person-1", "intervals": [{"start_s": 2.25, "end_s": 3.0}],
         "observation_ids": ["face-1"]}], "observations": [
        {"observation_id": "face-1", "canonical_id": "person-1", "kind": "keyframe",
         "segment_index": 2, "frame_order": 1, "source_time_s": 2.375,
         "bbox": [10, 20, 30, 40]}]}
    payload, images = Pipeline._project_index_request(tmp_path, [{
        "index": 2, "visual_prompt": "person", "analysis_paths": [frame],
        "keyframe_selection": [{"source_time_s": 2.375, "source_scene_id": "SCENE_02",
                                "decode_index": 57}], "person_identity": evidence,
    }], {})
    assert images == [frame]
    assert payload["person_identity"] == evidence
    assert payload["segments"][0]["frames"][0]["source_time_s"] == 2.375
    assert payload["segments"][0]["frames"][0]["decode_index"] == 57
