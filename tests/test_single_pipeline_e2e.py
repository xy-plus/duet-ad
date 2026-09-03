import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app import (
    context_ir_bridge,
    h3,
    h3_project,
    image_optimization,
    long_generation,
    long_video,
    pipeline,
    scenes as scene_planner,
    seedream,
    storage,
)
from conftest import make_settings


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _source_scene(index: int, frame_indices: range) -> dict:
    indices = list(frame_indices)
    return {
        "index": index,
        "start_decode_frame_index": indices[0],
        "end_decode_frame_index": indices[-1] + 1,
        "start_s": indices[0] / 10,
        "end_s": (indices[-1] + 1) / 10,
        "frames": [
            {
                "decode_frame_index": frame_index,
                "pts": frame_index,
                "pts_origin": 0,
                "time_base_num": 1,
                "time_base_den": 10,
                "source_time_s": frame_index / 10,
            }
            for frame_index in indices
        ],
    }


def _transition_skeleton(segment: dict, keyframes_dir: Path) -> list[dict]:
    skeleton = []
    for frame_index, source in enumerate(segment["keyframe_sources"], 1):
        transition = source["transition"]["type"]
        frozen_transition = (
            "start" if transition == "start"
            else "hard_cut" if transition == "hard_cut"
            else "same_camera"
        )
        frame = keyframes_dir / f"{frame_index:02d}.png"
        skeleton.append({
            "segment_index": segment["index"],
            "frame_index": frame_index,
            "frame_name": frame.name,
            "source_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            "source_transition_from_previous": frozen_transition,
            "source_transition_evidence_sha256": hashlib.sha256(
                _canonical(source)
            ).hexdigest(),
        })
    return skeleton


class _FailOnProviderCall:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fail(self, *args, **kwargs):
        self.calls.append("sync")
        pytest.fail("zero-provider regression reached a provider client")

    async def fail_async(self, *args, **kwargs):
        self.calls.append("async")
        pytest.fail("zero-provider regression reached a provider client")

    get = fail
    post = fail
    request = fail


def test_n2_single_pipeline_stays_local_from_exact3_to_h3_request(
    tmp_path: Path,
    video_1s: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FailOnProviderCall()
    monkeypatch.setattr(seedream, "edit", provider.fail_async)
    monkeypatch.setattr(h3, "start", provider.fail)
    monkeypatch.setattr(h3, "submit", provider.fail)

    settings = replace(
        make_settings(tmp_path),
        autodl_art_token="offline-test-token",
    )
    created = storage.new_conversation(
        settings.data_dir, "single-pipeline-e2e", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    source = root / "source.mp4"
    source.write_bytes(video_1s.read_bytes())
    work = root / "work"

    source_scenes = [
        _source_scene(1, range(0, 5)),
        _source_scene(2, range(5, 10)),
    ]
    segments = [
        {
            "index": 1,
            "start_s": 0.0,
            "end_s": 0.5,
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
        },
        {
            "index": 2,
            "start_s": 0.5,
            "end_s": 1.0,
            "chain_id": "chain-002",
            "join_mode": "hard_cut",
        },
    ]
    segment_metas = []
    frozen_keyframes: list[tuple[tuple[Path, bytes], ...]] = []
    for segment, scene in zip(segments, source_scenes):
        selection = scene_planner.select_segment_keyframes([scene], segment)
        segment_work = work / "segments" / str(segment["index"]) / "work"
        segment_work.mkdir(parents=True)
        names, receipt, frozen = pipeline._materialize_backend_keyframes(
            source, segment_work, selection
        )

        assert names == [f"{order:02d}.png" for order in range(1, 4)]
        assert len(receipt["keyframes"]) == len(frozen) == 3
        assert all(not item["repeated"] for item in receipt["keyframes"])
        by_decode_index = {
            item["decode_frame_index"]: item["sha256"]
            for item in receipt["keyframes"]
            if not item["repeated"]
        }
        assert all(
            item["sha256"] == by_decode_index[item["decode_frame_index"]]
            for item in receipt["keyframes"]
            if item["repeated"]
        )
        assert json.loads(
            (segment_work / "keyframe_sampling.json").read_text(encoding="utf-8")
        ) == receipt
        receipt["version"] = 3
        receipt["keyframe_count"] = 3

        segment_metas.append({
            **segment,
            "keyframes": names,
            "keyframe_sampling": receipt,
        })
        frozen_keyframes.append(tuple(
            (
                segment_work / "keyframes" / name,
                (segment_work / "keyframes" / name).read_bytes(),
            )
            for name in names
        ))

    bound = pipeline._bind_keyframe_source_timeline(
        work, segments, segment_metas, source_scenes
    )
    assert [len(item["keyframe_sources"]) for item in bound] == [3, 3]
    assert all(
        len({frame["source_time_s"] for frame in item["keyframe_sources"]}) == 3
        for item in bound
    )

    image_segments = []
    frame_inventory = []
    for segment in bound:
        keyframes_dir = (
            work / "segments" / str(segment["index"]) / "work" / "keyframes"
        )
        skeleton = _transition_skeleton(segment, keyframes_dir)
        image_segments.append({
            "index": segment["index"],
            "chain_id": segment["chain_id"],
            "join_mode": segment["join_mode"],
            "keyframes_dir": keyframes_dir,
            "transition_skeleton": skeleton,
        })
        frame_inventory.extend(skeleton)

    compiler_scores: list[float] = []
    compile_semantic_plan = image_optimization.compile_semantic_plan

    def observe_semantic_score(*args, **kwargs):
        plan, diagnostics = compile_semantic_plan(*args, **kwargs)
        compiler_scores.append(diagnostics["score"])
        return plan, diagnostics

    monkeypatch.setattr(
        image_optimization, "compile_semantic_plan", observe_semantic_score
    )

    class SparseSemanticSkill:
        calls = 0

        @classmethod
        def schema_value(cls, schema: dict):
            if "enum" in schema:
                return schema["enum"][0]
            schema_type = schema.get("type")
            if schema_type == "object":
                return {
                    key: cls.schema_value(schema["properties"][key])
                    for key in schema.get("required", [])
                }
            if schema_type == "array":
                item_schema = schema["items"]
                key_schema = item_schema.get("properties", {}).get("key", {})
                if "enum" in key_schema:
                    return [
                        {**cls.schema_value(item_schema), "key": key}
                        for key in key_schema["enum"]
                    ]
                return [
                    cls.schema_value(item_schema)
                    for _index in range(schema.get("minItems", 0))
                ]
            if schema_type == "integer":
                return schema.get("minimum", 1)
            if schema_type == "number":
                return schema.get("minimum", 0.0)
            if schema_type == "boolean":
                return False
            return "observed"

        def run_isolated_until_output(
            self, cwd: Path, _prompt: str, *, session_dir: Path,
            output_path: Path, max_output_bytes: int, validate_output,
            output_schema: dict,
        ):
            del cwd, output_path, max_output_bytes
            self.calls += 1
            assert session_dir == root
            return validate_output(_canonical(self.schema_value(output_schema)))

    semantic_skill = SparseSemanticSkill()
    image_plan, image_prompts = image_optimization.generate_project_prompts(
        semantic_skill,
        image_segments,
        settings.seedream_edit_mode,
        session_dir=root,
        expected_version=4,
        skill_bytes=b"frozen image-postprocess skill",
        generation_config={
            "remove_subtitle": False,
            "remove_watermark": False,
        },
    )
    assert semantic_skill.calls == 3
    assert compiler_scores == [1.0]
    assert image_plan["eligible"] is True
    assert set(image_prompts) == {1, 2}
    assert all(set(prompts) == set(range(1, 4)) for prompts in image_prompts.values())

    execution_inputs = image_optimization.freeze_execution_inputs(
        image_plan,
        revision=1,
        profile={"id": "single-pipeline-e2e", "revision": 1},
        model=settings.seedream_model,
        frame_inventory=frame_inventory,
    )
    frozen_image = image_optimization.freeze_frame_prompts(
        settings,
        execution_inputs,
        image_prompts,
        plan=image_plan,
    )
    dialogue_sha256 = hashlib.sha256(b"[]\n").hexdigest()
    frozen_segments = tuple(
        long_generation.FrozenSegment(
            index=segment["index"],
            start_s=segment["start_s"],
            end_s=segment["end_s"],
            chain_id=segment["chain_id"],
            join_mode=segment["join_mode"],
            workdir=work / "segments" / str(segment["index"]),
            first_frame=frozen_keyframes[position][0][0],
            first_frame_data=frozen_keyframes[position][0][1],
            last_frame=frozen_keyframes[position][-1][0],
            last_frame_data=frozen_keyframes[position][-1][1],
            prompt=f"source visual prompt {segment['index']}",
            keyframes=frozen_keyframes[position],
            keyframe_sources=tuple(segment["keyframe_sources"]),
            dialogue=(),
            dialogue_sha256=dialogue_sha256,
        )
        for position, segment in enumerate(bound)
    )
    base_plan = long_generation.FrozenPlan(
        root=root,
        source=source,
        receipt="f" * 64,
        segments=frozen_segments,
        receipt_version=long_video.THREE_FRAME_VISUAL_PLAN_RECEIPT_VERSION,
        workflow=h3.H3_WORKFLOW,
    )
    acceptance_sha256 = "a" * 64
    meta_segments = [
        {**segment, "visual_prompt": f"source visual prompt {segment['index']}"}
        for segment in bound
    ]
    storage.update_meta(
        settings.data_dir,
        cid,
        segments=meta_segments,
        _image_user_acceptance={
            "version": 1,
            "sha256": acceptance_sha256,
        },
        **image_optimization.freeze_continuity(
            image_plan, frame_counts={1: 3, 2: 3}
        ),
        **frozen_image,
    )
    current_meta = storage.load_meta(settings.data_dir, cid)
    fusion_input = long_generation.build_prompt_fusion_input(
        root=root,
        meta=current_meta,
        plan=base_plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    )

    class VisualOnlyFusionSkill:
        calls = 0

        def run(self, _cwd: Path, _prompt: str) -> None:
            raise AssertionError("Fusion Skill must run in its isolated stage")

        def run_isolated_until_output(
            self,
            cwd: Path,
            _prompt: str,
            *,
            session_dir: Path,
            output_path: Path,
            max_output_bytes: int,
            validate_output,
            output_schema: dict,
        ) -> bytes:
            del max_output_bytes, output_schema
            self.calls += 1
            assert session_dir == root
            input_data = (cwd / "work" / "multimodal_input.json").read_bytes()
            payload = json.loads(input_data)
            assert output_path == cwd / "work" / "h3_prompt_plan.json"
            return validate_output(_canonical({
                "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
                "version": long_generation.PROMPT_FUSION_VERSION,
                "input_sha256": hashlib.sha256(input_data).hexdigest(),
                "segments": [
                    {
                            "index": segment["index"],
                            "visual": [
                                f"segment {segment['index']} local visual motion"
                                for _frame in segment["new_keyframes"]
                            ],
                        }
                        for segment in payload["segments"]
                    ],
                }))

    fusion_skill = VisualOnlyFusionSkill()
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=fusion_input,
        image_acceptance_sha256=acceptance_sha256,
    ) == "queued"
    assert pipeline.produce_prompt_fusion(settings, cid, fusion_skill) == "done"
    assert fusion_skill.calls == 1

    raw_fusion_output = json.loads(
        (work / "h3_prompt_plan.json").read_text(encoding="utf-8")
    )
    assert all(
        set(segment) == {"index", "visual", "relation_states"}
        and len(segment["visual"]) == 3
        for segment in raw_fusion_output["segments"]
    )
    fusion = long_generation.load_prompt_fusion_manifest(
        root=root,
        skill_source_path=pipeline.PROMPT_FUSION_SKILL_MD,
    )
    assert len(fusion.final_prompts) == 2
    ref2va_sections = (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )
    for index, prompt in enumerate(fusion.final_prompts, 1):
        assert [prompt.count(section) for section in ref2va_sections] == [1] * 6
        assert f"segment {index} local visual motion" in prompt
        assert "<Picture 1>=[Shot 1]@00:00.000." in prompt
        assert "<Picture 3>=[Shot 1]@00:00.400." in prompt
        assert "<Picture 1>: segment" in prompt
        assert "[Shot 2]" not in prompt
        assert "<Audio " not in prompt

    plan = replace(
        base_plan,
        segments=tuple(
            replace(segment, prompt=prompt)
            for segment, prompt in zip(base_plan.segments, fusion.final_prompts)
        ),
        prompt_fusion=fusion,
        workflow=h3.H3_WORKFLOW,
    )
    upstream_artifact = work / "prepared_input.json"
    upstream_data = _canonical({"dialogue": {"sha256": dialogue_sha256}})
    upstream_artifact.write_bytes(upstream_data)
    final_requests = []
    for segment in plan.segments:
        source_request = long_generation._request(
            settings,
            cid,
            plan,
            segment,
            "parent-request",
            "none",
        )
        frozen_context = context_ir_bridge.freeze_context_ir_request(
            source_h3_request=source_request,
            context_ir_keyframes=tuple(
                (path, image_optimization.half_resolution_png(data))
                for path, data in source_request.keyframes
            ),
            upstream_dialogue_sha256=dialogue_sha256,
            upstream_artifact_path=upstream_artifact,
            upstream_artifact_sha256=hashlib.sha256(upstream_data).hexdigest(),
            upstream_dialogue_sha256_path=("dialogue", "sha256"),
            source_prompt_sha256=hashlib.sha256(
                source_request.prompt.encode("utf-8")
            ).hexdigest(),
            minimax_api_key="offline-test-key",
        )
        task_id = f"context-task-{segment.index}"
        uploaded_files = 0

        def context_handler(request: httpx.Request) -> httpx.Response:
            nonlocal uploaded_files
            if request.url.path == "/v1/files/upload":
                uploaded_files += 1
                return httpx.Response(200, json={
                    "file": {"file_id": str(427752006353317 + uploaded_files)},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                })
            if request.url.path == "/v2/h3_context_ir":
                return httpx.Response(200, json={"task_id": task_id})
            if request.url.path == f"/v2/query/video_generation/{task_id}":
                return httpx.Response(200, json={
                    "task": {
                        "id": task_id,
                        "task_type": "h3_context_ir",
                        "status": "succeeded",
                        "modality": "text",
                        "content": {"prompt": source_request.prompt},
                    },
                })
            raise AssertionError(
                f"unexpected request: {request.method} {request.url}"
            )

        with httpx.Client(
            transport=httpx.MockTransport(context_handler)
        ) as context_client:
            result = context_ir_bridge.optimize_h3_prompt(
                frozen_context, client=context_client
            )
        assert result.status == "succeeded"
        assert result.effective_prompt == source_request.prompt
        assert result.provider_task_id == task_id
        final_requests.append(h3_project.apply_bound_context_ir(
            frozen_context, h3_project.context_ir_binding(result)
        ))

    assert {request.workflow for request in final_requests} == {h3.H3_WORKFLOW}
    assert all(request.reference_audios == () for request in final_requests)
    assert all(request.audio_required is False for request in final_requests)
    assert all(request.context_ir_required is True for request in final_requests)
    assert provider.calls == []
