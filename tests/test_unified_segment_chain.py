import hashlib
import json
from pathlib import Path

import pytest

from app import h3_project, long_generation, long_video, main, pipeline, storage
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


def _current_v4_meta(*, segments: list[dict] | None) -> dict:
    meta = {
        "schema_version": 2,
        "duration_s": 14.5 if segments is None else 28.0,
        "status": "done",
        "_postprocess_receipt": {
            "version": 4,
            "options": {"optimize_image": True},
        },
        "_image_user_acceptance": {"version": 1, "sha256": "a" * 64},
    }
    if segments is not None:
        meta["segments"] = segments
        meta["long_video_plan_receipt"] = "long-video-plan.json"
    return meta


def test_current_v4_n1_and_n2_use_the_same_segment_coordinator() -> None:
    single = _current_v4_meta(segments=None)
    multiple = _current_v4_meta(
        segments=[
            {"index": 1, "start_s": 0.0, "end_s": 14.0},
            {"index": 2, "start_s": 14.0, "end_s": 28.0},
        ]
    )

    assert main._uses_segment_coordinator(single) is True
    assert main._uses_segment_coordinator(multiple) is True


def test_prompt_fusion_output_is_the_only_prompt_authority(tmp_path: Path) -> None:
    old_prompt = "OLD_VISUAL_PROMPT_MUST_NOT_REACH_CONTEXT_OR_H3"
    final_prompt = (
        "FUSED_PROMPT_FROM_OPTIMIZED_IMAGES"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>"
    )
    for order in range(1, 10):
        (tmp_path / f"{order:02d}.png").write_bytes(f"frame-{order}".encode())
    fusion_input = {
        "schema": "duet.video-prompt-fusion-input",
        "version": 1,
        "segments": [{
            "index": 1,
            "new_keyframes": [
                {
                    "order": order,
                    "path": f"{order:02d}.png",
                    "sha256": hashlib.sha256(f"frame-{order}".encode()).hexdigest(),
                }
                for order in range(1, 10)
            ],
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(old_prompt.encode("utf-8")).hexdigest(),
            },
            "image_optimization_prompt": [{
                "order": order,
                "text": "replace person and scene",
                "sha256": hashlib.sha256(b"replace person and scene").hexdigest(),
            } for order in range(1, 10)],
            "audio_content": {
                "lines_json": "[]",
                "voice_references": [],
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
            },
        }],
    }
    input_data = _canonical(fusion_input)
    fusion_output = {
        "schema": "duet.video-prompt-fusion-output",
        "version": 1,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{"index": 1, "final_prompt": final_prompt}],
    }
    input_path = tmp_path / "multimodal_input.json"
    output_path = tmp_path / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical(fusion_output))

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path,
        output_path=output_path,
        root=tmp_path,
    )

    assert frozen.final_prompts == (final_prompt,)
    assert old_prompt not in frozen.final_prompts


_CID_9533_LINES_JSON = (
    '[{"order":1,"text":"تول جمالا بها القادم وقتك يا لدى عوصة اللوري تاب '
    'البدري يبريزها بجمال يا شعوري","start_s":0.0,"end_s":14.44,'
    '"delivery":"off_screen","voice_ref":1}]'
)


def _write_9533_fusion_fixture(
    root: Path, *, audio_envelope: str,
) -> tuple[Path, Path, str]:
    (root / "work").mkdir(parents=True, exist_ok=True)
    input_payload = json.loads(_fusion_input(root, 1).decode("utf-8"))
    voice = root / "work" / "segments" / "1" / "work" / "voice.mp3"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"9533-frozen-normalized-voice")
    input_payload["segments"][0]["audio_content"] = {
        "lines_json": _CID_9533_LINES_JSON,
        "lines_sha256": hashlib.sha256(
            _CID_9533_LINES_JSON.encode("utf-8")
        ).hexdigest(),
        "voice_references": [{
            "voice_ref": 1,
            "path": "work/segments/1/work/voice.mp3",
            "sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
            "purpose": "voice",
        }],
    }
    input_data = _canonical(input_payload)
    input_path = root / "work" / "multimodal_input.json"
    output_path = root / "work" / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    final_prompt = f"<VISUAL>9533 fused visual</VISUAL>\n{audio_envelope}"
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{"index": 1, "final_prompt": final_prompt}],
    }))
    return input_path, output_path, final_prompt


def test_9533_lf_audio_envelope_is_canonicalized_without_rewriting_json(
    tmp_path: Path,
) -> None:
    envelope = (
        "<AUDIO_CONTENT_JSON>\n"
        f"{_CID_9533_LINES_JSON}\n"
        "</AUDIO_CONTENT_JSON>"
    )
    input_path, output_path, _raw_prompt = _write_9533_fusion_fixture(
        tmp_path, audio_envelope=envelope,
    )

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=tmp_path,
    )

    canonical_block = (
        f"<AUDIO_CONTENT_JSON>{_CID_9533_LINES_JSON}"
        "</AUDIO_CONTENT_JSON>"
    )
    assert frozen.final_prompts == (
        f"<VISUAL>9533 fused visual</VISUAL>\n{canonical_block}",
    )
    assert frozen.output_data == output_path.read_bytes()


@pytest.mark.parametrize(
    "audio_envelope",
    [
        (
            "<AUDIO_CONTENT_JSON> " + _CID_9533_LINES_JSON
            + " </AUDIO_CONTENT_JSON>"
        ),
        (
            "<AUDIO_CONTENT_JSON>\r\n" + _CID_9533_LINES_JSON
            + "\r\n</AUDIO_CONTENT_JSON>"
        ),
        (
            "<AUDIO_CONTENT_JSON>"
            + _CID_9533_LINES_JSON.replace("off_screen", "on_screen")
            + "</AUDIO_CONTENT_JSON>"
        ),
        (
            "<AUDIO_CONTENT_JSON>" + _CID_9533_LINES_JSON
            + "</AUDIO_CONTENT_JSON><AUDIO_CONTENT_JSON>"
            + _CID_9533_LINES_JSON + "</AUDIO_CONTENT_JSON>"
        ),
    ],
    ids=("space", "crlf", "json-rewrite", "duplicate-block"),
)
def test_prompt_fusion_rejects_noncanonical_audio_envelopes(
    tmp_path: Path, audio_envelope: str,
) -> None:
    input_path, output_path, _raw_prompt = _write_9533_fusion_fixture(
        tmp_path, audio_envelope=audio_envelope,
    )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_output_invalid",
    ):
        long_generation.load_prompt_fusion(
            input_path=input_path, output_path=output_path, root=tmp_path,
        )


def test_binding_skill_production_seams_are_absent() -> None:
    assert not hasattr(pipeline, "queue_multimodal_binding")
    assert not hasattr(pipeline, "produce_multimodal_binding")
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "queue_multimodal_binding" not in source
    assert "produce_multimodal_binding" not in source


def _fusion_input(root: Path, segment_count: int) -> bytes:
    segments = []
    for index in range(1, segment_count + 1):
        frames = []
        prompts = []
        for order in range(1, 10):
            relative = f"work/segment-{index}-{order:02d}.png"
            data = f"segment-{index}-frame-{order}".encode()
            (root / relative).write_bytes(data)
            prompt = f"segment {index} optimized image {order}"
            frames.append({
                "order": order,
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            prompts.append({
                "order": order,
                "text": prompt,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
        old_prompt = f"old visual prompt {index}"
        segments.append({
            "index": index,
            "new_keyframes": frames,
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(
                    old_prompt.encode("utf-8")
                ).hexdigest(),
            },
            "image_optimization_prompt": prompts,
            "audio_content": {
                "lines_json": "[]",
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "voice_references": [],
            },
        })
    return _canonical({
        "schema": long_generation.PROMPT_FUSION_INPUT_SCHEMA,
        "version": long_generation.PROMPT_FUSION_VERSION,
        "segments": segments,
    })


@pytest.mark.parametrize("segment_count", [1, 2])
def test_project_prompt_fusion_runs_once_and_publishes_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segment_count: int,
) -> None:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={
            "version": 1,
            "sha256": acceptance_sha256,
        },
    )
    input_data = _fusion_input(root, segment_count)
    skill_path = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_path.write_text("strict project prompt fusion", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_path)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_path
    )
    calls: list[Path] = []

    class Runner:
        def run(self, cwd: Path, prompt: str) -> None:
            calls.append(cwd)
            assert "Binding" in prompt
            frozen_input = (cwd / "work" / "multimodal_input.json").read_bytes()
            payload = json.loads(frozen_input.decode("utf-8"))
            output = {
                "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
                "version": long_generation.PROMPT_FUSION_VERSION,
                "input_sha256": hashlib.sha256(frozen_input).hexdigest(),
                "segments": [{
                    "index": item["index"],
                    "final_prompt": (
                        f"fused prompt {item['index']}"
                        "<AUDIO_CONTENT_JSON>"
                        f"{item['audio_content']['lines_json']}"
                        "</AUDIO_CONTENT_JSON>"
                    ),
                } for item in payload["segments"]],
            }
            (cwd / "work" / "h3_prompt_plan.json").write_bytes(
                _canonical(output)
            )

    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
    ) == "queued"
    assert pipeline.produce_prompt_fusion(settings, cid, Runner()) == "done"
    assert calls == [root]
    frozen = long_generation.load_prompt_fusion_manifest(
        root=root,
        skill_source_path=skill_path,
    )
    assert frozen.final_prompts == tuple(
        f"fused prompt {index}<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>"
        for index in range(1, segment_count + 1)
    )
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
    ) == "done"
    assert pipeline.produce_prompt_fusion(settings, cid, Runner()) == "done"
    assert calls == [root]
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta is not None
    assert main._public_prompt_fusion(meta, root) == {
        "status": "done",
        "error": None,
        "segments": [{
            "index": index,
            "status": "done",
            "final_prompt": (
                f"fused prompt {index}"
                "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>"
            ),
            "error": None,
        } for index in range(1, segment_count + 1)],
    }


def _failed_lf_prompt_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, str, Path, Path, Path]:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion-receipt-recovery", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={
            "version": 1,
            "sha256": acceptance_sha256,
        },
    )
    skill_path = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_path.write_text("strict frozen prompt fusion", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_path)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_path
    )
    input_data = _fusion_input(root, 1)
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
    ) == "queued"
    frozen_skill = root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME
    frozen_skill.write_bytes(skill_path.read_bytes())
    output = root / "work" / "h3_prompt_plan.json"
    output.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "final_prompt": (
                "<VISUAL>fused</VISUAL>\n"
                "<AUDIO_CONTENT_JSON>\n[]\n</AUDIO_CONTENT_JSON>"
            ),
        }],
    }))
    state = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={
            **state,
            "status": "failed",
            "error": "prompt_fusion_output_invalid",
            "manifest_sha256": None,
        },
    )
    return settings, cid, root, skill_path, output


def test_failed_lf_output_can_publish_receipt_without_rerunning_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    assert pipeline.finalize_prompt_fusion_receipt(settings, cid) == "done"

    assert hashlib.sha256(output.read_bytes()).hexdigest() == raw_output_sha256
    frozen = long_generation.load_prompt_fusion_manifest(
        root=root, skill_source_path=skill_path,
    )
    assert frozen.final_prompts == (
        "<VISUAL>fused</VISUAL>\n"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>",
    )
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["_prompt_fusion"]["status"] == "done"
    assert meta["_prompt_fusion"]["error"] is None
    assert meta["_prompt_fusion"]["recovered_error"] == (
        "prompt_fusion_output_invalid"
    )
    assert meta["_prompt_fusion"]["manifest_sha256"] == hashlib.sha256(
        (root / "work" / h3_project.SOURCE_FILENAME).read_bytes()
    ).hexdigest()
    manifest = json.loads(
        (root / "work" / h3_project.SOURCE_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["output"]["sha256"] == raw_output_sha256
    assert manifest["skill"]["sha256"] == hashlib.sha256(
        (root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME).read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "drift", ["input", "output", "skill", "acceptance", "generation", "h3"],
)
def test_receipt_only_finalization_rejects_any_frozen_authority_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    if drift == "input":
        (root / "work" / h3_project.SKILL_INPUT_FILENAME).write_bytes(b"{}")
    elif drift == "output":
        output.write_bytes(b"{}")
    elif drift == "skill":
        (root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME).write_bytes(
            b"drifted skill"
        )
    elif drift == "acceptance":
        storage.update_meta(
            settings.data_dir,
            cid,
            _image_user_acceptance={"version": 1, "sha256": "b" * 64},
        )
    elif drift == "generation":
        storage.update_meta(
            settings.data_dir, cid, generation={"status": "not_started"},
        )
    else:
        (root / "work" / "segments" / "1" / ".h3").mkdir(
            parents=True, exist_ok=True,
        )

    with pytest.raises(pipeline.PipelineError):
        pipeline.finalize_prompt_fusion_receipt(settings, cid)

    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "failed"


def test_frozen_current_v4_projects_publish_a_stable_n1_fusion_shape(
    tmp_path: Path,
) -> None:
    meta = _current_v4_meta(segments=None)
    meta["_prompt_fusion"] = {"status": "running"}

    assert main._public_prompt_fusion(meta, tmp_path) == {
        "status": "running",
        "error": None,
        "segments": [{
            "index": 1,
            "status": "running",
            "final_prompt": None,
            "error": None,
        }],
    }


def test_validation_fingerprint_binds_immutable_fusion_artifact_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / ("a" * 32)
    work = root / "work"
    work.mkdir(parents=True)
    voice_path = work / "voice.mp3"
    voice_path.write_bytes(b"frozen-voice")
    input_data = _canonical({
        "segments": [{
            "audio_content": {
                "voice_references": [{
                    "path": "work/voice.mp3",
                    "sha256": hashlib.sha256(
                        voice_path.read_bytes()
                    ).hexdigest(),
                }],
            },
        }],
    })
    output_data = b'{"output":"fused"}\n'
    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(output_data)
    manifest = {
        "input": {
            "path": f"work/{h3_project.SKILL_INPUT_FILENAME}",
            "sha256": hashlib.sha256(input_data).hexdigest(),
        },
        "output": {
            "path": "work/h3_prompt_plan.json",
            "sha256": hashlib.sha256(output_data).hexdigest(),
        },
    }
    manifest_data = _canonical(manifest)
    manifest_path = work / h3_project.SOURCE_FILENAME
    manifest_path.write_bytes(manifest_data)
    plan = {
        "schema": "duet.long-video-plan",
        "version": long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        "prompt_fusion": {
            "path": f"work/{h3_project.SOURCE_FILENAME}",
            "sha256": hashlib.sha256(manifest_data).hexdigest(),
        },
        "segments": [],
    }
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(_canonical(plan))
    meta = {
        "id": root.name,
        "duration_s": 28.0,
        "segments": [{"index": 1}],
        "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        "_prompt_fusion": {"status": "done"},
    }

    paths = main._long_validation_paths(root, meta)
    assert {manifest_path, input_path, output_path, voice_path} <= paths
    before_entries = main._prompt_fusion_fingerprint_entries(root, meta)
    before = main._generated_video_validation_fingerprint(root, meta)
    del meta["_prompt_fusion"]
    assert main._prompt_fusion_fingerprint_entries(root, meta) == before_entries
    assert main._generated_video_validation_fingerprint(root, meta) == before

    output_path.write_bytes(b'{"output":"DRIFT"}\n')

    after_entries = main._prompt_fusion_fingerprint_entries(root, meta)
    after = main._generated_video_validation_fingerprint(root, meta)
    assert after_entries != before_entries
    assert after != before

    output_path.write_bytes(output_data)
    before_voice = main._generated_video_validation_fingerprint(root, meta)
    voice_path.write_bytes(b"drifted-voice")
    assert main._generated_video_validation_fingerprint(root, meta) != before_voice


def test_warm_validation_cache_invalidates_when_fusion_voice_bytes_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    cid = "c" * 32
    root = settings.data_dir / cid
    work = root / "work"
    work.mkdir(parents=True)
    voice = work / "voice.mp3"
    voice.write_bytes(b"voice-one")
    input_data = _canonical({
        "segments": [{"audio_content": {"voice_references": [{
            "path": "work/voice.mp3",
            "sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
        }]}}],
    })
    output_data = b'{"output":"fused"}\n'
    (work / h3_project.SKILL_INPUT_FILENAME).write_bytes(input_data)
    (work / "h3_prompt_plan.json").write_bytes(output_data)
    manifest_data = _canonical({
        "input": {
            "path": f"work/{h3_project.SKILL_INPUT_FILENAME}",
            "sha256": hashlib.sha256(input_data).hexdigest(),
        },
        "output": {
            "path": "work/h3_prompt_plan.json",
            "sha256": hashlib.sha256(output_data).hexdigest(),
        },
    })
    (work / h3_project.SOURCE_FILENAME).write_bytes(manifest_data)
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(_canonical({
        "schema": "duet.long-video-plan",
        "version": long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        "prompt_fusion": {
            "path": f"work/{h3_project.SOURCE_FILENAME}",
            "sha256": hashlib.sha256(manifest_data).hexdigest(),
        },
        "segments": [],
    }))
    meta = {
        "id": cid,
        "duration_s": 28.0,
        "segments": [{"index": 1}],
        "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        "generation": {"status": "succeeded"},
    }
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(main, "_validate_generated_video_uncached", validate)
    assert main._has_valid_generated_video(settings, meta) is True
    assert main._has_valid_generated_video(settings, meta) is True
    assert calls == 1
    voice.write_bytes(b"voice-two")
    assert main._has_valid_generated_video(settings, meta) is True
    assert calls == 2


@pytest.mark.parametrize("dialogue_mode", ["auto", "none"])
def test_legacy_v2_plan_does_not_bootstrap_prompt_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialogue_mode: str,
) -> None:
    root = tmp_path / ("b" * 32)
    root.mkdir()
    receipt_data = _canonical({
        "schema": "duet.long-video-plan",
        "version": long_video.PLAN_RECEIPT_VERSION,
        "workflow": "legacy-read-only",
    })
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(receipt_data)
    monkeypatch.setattr(
        pipeline,
        "queue_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy v2 may not enter current prompt fusion"
        ),
    )

    assert long_generation.finalize_multimodal_plan(
        root,
        {
            "id": root.name,
            "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        },
        hashlib.sha256(receipt_data).hexdigest(),
        "none",
        dialogue_mode,
        aspect_ratio="9:16",
        resolution="768p",
        settings=make_settings(tmp_path),
    ) is None
