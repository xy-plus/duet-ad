import hashlib
import json
import subprocess
import threading

import pytest
from fastapi.testclient import TestClient

from app import dialogue_timing, pipeline, storage
from app.main import create_app
from conftest import make_settings


SOURCE_DATA = b"source-video-bytes"
SKILL_DATA = b"frozen-video-maker-skill"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _producer_input(*, cut_before: int | None = None) -> tuple[dict, dict[str, bytes]]:
    frame_data = {
        f"speaker-visibility-frames/{order:06d}.png": f"frame-{order}".encode()
        for order in range(1, 9)
    }
    frame_data["speaker-visibility-contact-sheets/000001.png"] = b"sheet-1"
    frame_data["keyframes/01.png"] = b"identity-1"
    frame_data["keyframes/02.png"] = b"identity-2"
    frame_data["scenes.json"] = b"frozen-scenes"
    frames = [
        {
            "order": order,
            "path": path,
            "sha256": _sha(data),
            "pts": order - 1,
            "cut_before": order == cut_before,
        }
        for order, (path, data) in enumerate(
            (
                (path, data) for path, data in frame_data.items()
                if path.startswith("speaker-visibility-frames/")
            ),
            1,
        )
    ]
    return (
        {
            "schema": "duet.speaker-visibility-input",
            "version": 1,
            "phase": "speaker_visibility",
            "source": {
                "sha256": _sha(SOURCE_DATA),
                "duration_pts": 8,
                "time_base": {"numerator": 1, "denominator": 8},
            },
            "sampling": {
                "algorithm": "decoded_pts_nearest_v1",
                "cadence_fps": 8,
                "max_unobserved_gap_pts": 1,
                "endpoint_shrink_intervals": 1,
            },
            "decoded_frame_pts": list(range(8)),
            "cut_pts": [4] if cut_before == 5 else [],
            "cut_source": {
                "path": "scenes.json",
                "sha256": _sha(frame_data["scenes.json"]),
            },
            "frames": frames,
            "contact_sheets": [{
                "order": 1,
                "path": "speaker-visibility-contact-sheets/000001.png",
                "sha256": _sha(frame_data[
                    "speaker-visibility-contact-sheets/000001.png"
                ]),
                "frame_orders": list(range(1, 9)),
            }],
            "persons": [
                {"person_id": "PERSON_01", "identity_refs": [{
                    "path": "keyframes/01.png",
                    "sha256": _sha(frame_data["keyframes/01.png"]),
                }]},
                {"person_id": "PERSON_02", "identity_refs": [{
                    "path": "keyframes/02.png",
                    "sha256": _sha(frame_data["keyframes/02.png"]),
                }]},
            ],
            "on_screen_subjects": ["S1"],
        },
        frame_data,
    )


def _skill_output(
    producer_input: dict,
    *,
    person_id: str = "PERSON_01",
    lip_gap_orders: tuple[int, ...] = (),
) -> dict:
    return {
        "schema": "duet.speaker-visibility-output",
        "version": 1,
        "phase": "speaker_visibility",
        "input_sha256": _sha(_json_bytes(producer_input)),
        "subject_person_mapping": [
            {"subject_id": "S1", "person_id": person_id},
        ],
        "frames": [
            {
                "order": order,
                "visible_person_ids": ["PERSON_01"],
                "lip_verifiable_person_ids": (
                    [] if order in lip_gap_orders else ["PERSON_01"]
                ),
            }
            for order in range(1, 9)
        ],
    }


def _freeze(output: dict, *, cut_before: int | None = None):
    producer_input, frame_data = _producer_input(cut_before=cut_before)
    return dialogue_timing.freeze_speaker_visibility(
        producer_input_data=_json_bytes(producer_input),
        skill_output_data=_json_bytes(output),
        source_data=SOURCE_DATA,
        frame_data=frame_data,
        skill_data=SKILL_DATA,
    )


def test_ordinary_on_screen_project_can_produce_existing_timing_schema():
    producer_input, frame_data = _producer_input()
    output = _skill_output(producer_input)

    production = _freeze(output)

    assert production.speaker_timing["schema"] == "duet.speaker-timing"
    assert production.speaker_timing["version"] == 1
    assert production.speaker_timing["speakers"][0]["windows"] == [{
        "kind": "lip_verifiable",
        "status": "verified",
        "start_pts": 1,
        "end_pts": 6,
        "evidence_keyframes": [2, 3, 4, 5, 6, 7],
    }]
    assert production.receipt == {
        "schema": "duet.speaker-timing-production",
        "version": 1,
        "source_sha256": _sha(SOURCE_DATA),
        "source_duration_pts": 8,
        "source_time_base": {"numerator": 1, "denominator": 8},
        "sampling": producer_input["sampling"],
        "dense_frame_inventory_sha256": dialogue_timing.canonical_sha256(
            producer_input["decoded_frame_pts"]
        ),
        "cut_inventory_sha256": dialogue_timing.canonical_sha256(
            producer_input["cut_pts"]
        ),
        "cut_source_sha256": _sha(frame_data["scenes.json"]),
        "sample_inventory_sha256": dialogue_timing.canonical_sha256(
            producer_input["frames"]
        ),
        "contact_sheet_inventory_sha256": dialogue_timing.canonical_sha256(
            producer_input["contact_sheets"]
        ),
        "subject_mapping_sha256": dialogue_timing.canonical_sha256(
            [{"subject_id": "S1", "person_id": "PERSON_01"}]
        ),
        "artifacts": {
            "producer_input": {
                "path": "speaker_visibility_input.json",
                "sha256": _sha(_json_bytes(producer_input)),
            },
            "raw_output": {
                "path": "speaker_visibility_output.json",
                "sha256": _sha(_json_bytes(output)),
            },
            "skill": {
                "path": "speaker_visibility_skill.md",
                "sha256": _sha(SKILL_DATA),
            },
            "speaker_timing": {
                "path": "speaker_timing.json",
                "sha256": _sha(_json_bytes(production.speaker_timing)),
                "canonical_sha256": dialogue_timing.canonical_sha256(
                    production.speaker_timing
                ),
            },
        },
    }
    frozen = dialogue_timing.freeze_speaker_timing(
        production.speaker_timing,
        source_sha256=_sha(SOURCE_DATA),
        keyframe_sha256s=tuple(
            frame["sha256"] for frame in producer_input["frames"]
        ),
        source_duration_s=1.0,
    )
    dialogue_timing.require_authoritative_window(
        frozen,
        subject_id="S1",
        start_s=0.25,
        end_s=0.625,
    )
    # Dialogue timing is deliberately absent from both the Skill input and output.
    assert b"dialogue" not in _json_bytes(producer_input)
    assert b"start_s" not in _json_bytes(output)


def test_wrong_speaker_mapping_cannot_borrow_another_persons_visible_frames():
    producer_input, _frame_data = _producer_input()

    with pytest.raises(
        dialogue_timing.DialogueTimingError,
        match="speaker_visibility_subject_unverified",
    ):
        _freeze(_skill_output(producer_input, person_id="PERSON_02"))


@pytest.mark.parametrize("break_kind", ["unknown", "cut"])
def test_unverified_gap_or_cut_is_not_interpolated_into_a_window(break_kind):
    producer_input, _frame_data = _producer_input(
        cut_before=5 if break_kind == "cut" else None
    )
    output = _skill_output(
        producer_input,
        lip_gap_orders=(5,) if break_kind == "unknown" else (),
    )
    production = _freeze(output, cut_before=5 if break_kind == "cut" else None)
    frozen = dialogue_timing.freeze_speaker_timing(
        production.speaker_timing,
        source_sha256=_sha(SOURCE_DATA),
        keyframe_sha256s=tuple(
            frame["sha256"] for frame in producer_input["frames"]
        ),
        source_duration_s=1.0,
    )

    with pytest.raises(
        dialogue_timing.DialogueTimingError,
        match="dialogue_outside_speaker_lip_window",
    ):
        dialogue_timing.require_authoritative_window(
            frozen,
            subject_id="S1",
            start_s=0.25,
            end_s=0.75,
        )


def _write_pipeline_plan(root, *, delivery: str) -> None:
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    plan = {
        "speech_bindings": [{
            "delivery": delivery,
            "subject_id": "S1" if delivery == "on_screen" else None,
        }],
    }
    plan_data = _json_bytes(plan)
    (work / "h3_prompt_plan.json").write_bytes(plan_data)
    (work / "multimodal_input.json").write_bytes(_json_bytes({"version": 1}))
    (work / "h3_multimodal_source.json").write_bytes(_json_bytes({
        "version": 2,
        "skill_plan": {
            "path": "h3_prompt_plan.json",
            "sha256": _sha(plan_data),
        },
        "multimodal_input": {
            "path": "multimodal_input.json",
            "sha256": _sha((work / "multimodal_input.json").read_bytes()),
        },
    }))


def test_offscreen_project_skips_before_probe_extract_or_skill(tmp_path):
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "offscreen", "source.mp4"
    )
    root = settings.data_dir / created["id"]
    (root / "source.mp4").write_bytes(SOURCE_DATA)
    _write_pipeline_plan(root, delivery="off_screen_voiceover")

    class Runner:
        def run(self, *_args):
            raise AssertionError("Skill must not run")

    def probe(*_args):
        raise AssertionError("ffprobe must not run")

    assert pipeline.produce_speaker_timing(
        settings, created["id"], Runner(), timeline_probe=probe
    ) == "skipped"
    assert not (root / "work" / "speaker_visibility_input.json").exists()


def test_segment_person_inventory_uses_project_level_reference_authority(
    tmp_path, monkeypatch,
):
    root = tmp_path / "long-project"
    segment_work = root / "work" / "segments" / "1" / "work"
    identity = root / "work" / "segments" / "2" / "work" / "keyframes" / "03.png"
    identity.parent.mkdir(parents=True)
    identity.write_bytes(b"segment-two-authoritative-identity")
    segment_work.mkdir(parents=True)
    monkeypatch.setattr(
        pipeline.image_optimization,
        "dual_target_plan_receipt",
        lambda _meta: {
            "person_plans": [{
                "id": "PERSON_01",
                "reference": {"segment_index": 2, "frame_index": 3},
                "observable_segments": [1, 2],
            }],
        },
    )

    persons, media = pipeline._speaker_visibility_person_inventory(
        {}, root, segment_work, segment_index=1,
    )

    assert persons == [{
        "person_id": "PERSON_01",
        "identity_refs": [{
            "path": "keyframes/speaker-visibility-identities/PERSON_01.png",
            "sha256": _sha(identity.read_bytes()),
        }],
    }]
    assert media == {
        "keyframes/speaker-visibility-identities/PERSON_01.png": identity.read_bytes(),
    }


def test_startup_recovers_only_exact_queued_speaker_jobs_and_locks_running_unknown(
    tmp_path,
):
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "speaker-recovery", "source.mp4"
    )
    root = settings.data_dir / created["id"]
    (root / "source.mp4").write_bytes(SOURCE_DATA)
    _write_pipeline_plan(root, delivery="on_screen")

    assert pipeline.queue_speaker_timing(settings, created["id"]) == "queued"
    queued = storage.load_meta(settings.data_dir, created["id"])[
        "_speaker_timing_producer"
    ]
    assert queued["status"] == "queued"
    assert queued["jobs"] == [{
        "scope": "short",
        "segment_index": 0,
        "status": "queued",
        "source": {
            "path": "source.mp4",
            "sha256": _sha(SOURCE_DATA),
        },
        "manifest": {
            "path": "work/h3_multimodal_source.json",
            "sha256": _sha(
                (root / "work" / "h3_multimodal_source.json").read_bytes()
            ),
        },
    }]
    assert pipeline.recover_speaker_timing_jobs(settings) == [created["id"]]

    producer_input = root / "work" / "speaker_visibility_input.json"
    frozen_skill = root / "work" / "speaker_visibility_skill.md"
    producer_input.write_bytes(b"exact-running-input")
    frozen_skill.write_bytes(b"exact-running-skill")
    running_job = {
        **queued["jobs"][0],
        "status": "running",
        "producer_input": {
            "path": "work/speaker_visibility_input.json",
            "sha256": _sha(producer_input.read_bytes()),
        },
        "skill": {
            "path": "work/speaker_visibility_skill.md",
            "sha256": _sha(frozen_skill.read_bytes()),
        },
    }
    running_job["job_receipt_sha256"] = dialogue_timing.canonical_sha256({
        key: running_job[key]
        for key in (
            "scope", "segment_index", "source", "manifest",
            "producer_input", "skill",
        )
    })
    running = {
        **queued,
        "status": "running",
        "jobs": [running_job],
    }
    storage.update_meta(
        settings.data_dir, created["id"],
        _speaker_timing_producer=running,
    )
    assert pipeline.recover_speaker_timing_jobs(settings) == []
    locked = storage.load_meta(settings.data_dir, created["id"])[
        "_speaker_timing_producer"
    ]
    assert locked["status"] == "submission_unknown"
    assert locked["jobs"][0]["status"] == "submission_unknown"


def test_app_startup_runs_exact_queued_speaker_job_without_provider(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_pipeline=False)
    created = storage.new_conversation(
        settings.data_dir, "startup speaker", "source.mp4"
    )
    root = settings.data_dir / created["id"]
    (root / "source.mp4").write_bytes(SOURCE_DATA)
    _write_pipeline_plan(root, delivery="on_screen")
    assert pipeline.queue_speaker_timing(settings, created["id"]) == "queued"
    ran = threading.Event()
    monkeypatch.setattr(
        pipeline,
        "produce_speaker_timing",
        lambda received_settings, received_cid, _runner: (
            ran.set()
            if received_settings is settings and received_cid == created["id"]
            else pytest.fail("startup recovered wrong speaker job")
        ),
    )

    with TestClient(create_app(settings)):
        assert ran.wait(2)


def test_on_screen_background_worker_writes_production_receipt(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "on-screen", "source.mp4"
    )
    root = settings.data_dir / created["id"]
    work = root / "work"
    (root / "source.mp4").write_bytes(SOURCE_DATA)
    _write_pipeline_plan(root, delivery="on_screen")
    identity = work / "keyframes" / "01.png"
    identity.parent.mkdir()
    identity.write_bytes(b"identity")
    scenes = work / "scenes.json"
    scenes.write_bytes(b"frozen-scenes")

    monkeypatch.setattr(
        pipeline,
        "_speaker_visibility_person_inventory",
        lambda _meta, _root, _work, **_kwargs: ([{
            "person_id": "PERSON_01",
            "identity_refs": [{
                "path": "keyframes/01.png",
                "sha256": _sha(identity.read_bytes()),
            }],
        }], {"keyframes/01.png": identity.read_bytes()}),
    )
    monkeypatch.setattr(pipeline.h3_project, "freeze_optional", lambda *_args: object())

    def probe(_source, _scenes):
        return {
            "time_base": {"numerator": 1, "denominator": 8},
            "duration_pts": 8,
            "decoded_frame_pts": list(range(8)),
            "cut_pts": [],
            "scenes_sha256": _sha(scenes.read_bytes()),
        }

    def extract(_source, target_work, selected):
        media = {}
        frames = []
        frame_dir = target_work / "speaker-visibility-frames"
        frame_dir.mkdir()
        for order, (_index, pts) in enumerate(selected, 1):
            path = frame_dir / f"{order:06d}.png"
            path.write_bytes(f"sample-{order}".encode())
            relative = path.relative_to(target_work).as_posix()
            media[relative] = path.read_bytes()
            frames.append({
                "order": order, "path": relative,
                "sha256": _sha(path.read_bytes()), "pts": pts,
                "cut_before": False,
            })
        sheet = target_work / "speaker-visibility-contact-sheets" / "000001.png"
        sheet.parent.mkdir()
        sheet.write_bytes(b"sheet")
        relative = sheet.relative_to(target_work).as_posix()
        media[relative] = sheet.read_bytes()
        return frames, [{
            "order": 1, "path": relative, "sha256": _sha(sheet.read_bytes()),
            "frame_orders": list(range(1, 9)),
        }], media

    class Runner:
        calls = 0

        def run(self, cdir, _prompt):
            self.calls += 1
            raw = (cdir / "work" / "speaker_visibility_input.json").read_bytes()
            value = json.loads(raw)
            output = _skill_output(value)
            (cdir / "work" / "speaker_visibility_output.json").write_bytes(
                _json_bytes(output)
            )

    runner = Runner()
    assert pipeline.produce_speaker_timing(
        settings, created["id"], runner,
        timeline_probe=probe, sample_extractor=extract,
    ) == "done"
    assert runner.calls == 1
    receipt = json.loads(
        (work / "speaker_timing_production.json").read_text(encoding="utf-8")
    )
    assert receipt["sampling"]["cadence_fps"] == 8
    assert receipt["dense_frame_inventory_sha256"] == dialogue_timing.canonical_sha256(
        list(range(8))
    )
    assert b"dialogue" not in (work / "speaker_visibility_input.json").read_bytes()
    manifest = json.loads(
        (work / "h3_multimodal_source.json").read_text(encoding="utf-8")
    )
    assert manifest["speaker_timing_producer"]["path"] == (
        "speaker_timing_production.json"
    )


def test_long_worker_uses_segment_source_and_cross_segment_identity_authority(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    created = storage.new_conversation(settings.data_dir, "long speaker", "source.mp4")
    root = settings.data_dir / created["id"]
    first = root / "work" / "segments" / "1"
    second = root / "work" / "segments" / "2"
    first.joinpath("source.mp4").parent.mkdir(parents=True)
    first.joinpath("source.mp4").write_bytes(SOURCE_DATA)
    second.joinpath("source.mp4").parent.mkdir(parents=True)
    second.joinpath("source.mp4").write_bytes(b"second-segment")
    _write_pipeline_plan(first, delivery="on_screen")
    _write_pipeline_plan(second, delivery="off_screen_voiceover")
    scenes = root / "work" / "scenes.json"
    scenes.write_text(json.dumps({
        "scenes": [
            {"index": 1, "start_s": 0.0, "end_s": 1.0},
            {"index": 2, "start_s": 1.0, "end_s": 2.0},
        ],
    }), encoding="utf-8")
    identity = second / "work" / "keyframes" / "01.png"
    identity.parent.mkdir(parents=True)
    identity.write_bytes(b"identity-from-segment-two")
    storage.update_meta(
        settings.data_dir, created["id"], duration_s=2.0,
        segments=[
            {"index": 1, "start_s": 0.0, "end_s": 1.0},
            {"index": 2, "start_s": 1.0, "end_s": 2.0},
        ],
    )
    monkeypatch.setattr(
        pipeline.image_optimization,
        "dual_target_plan_receipt",
        lambda _meta: {"person_plans": [{
            "id": "PERSON_01",
            "reference": {"segment_index": 2, "frame_index": 1},
            "observable_segments": [1],
        }]},
    )
    monkeypatch.setattr(pipeline.h3_project, "freeze_optional", lambda *_args: object())

    def probe(source, local_scenes):
        assert source == first / "source.mp4"
        assert json.loads(local_scenes.read_text(encoding="utf-8"))[
            "source_sha256"
        ] == _sha(scenes.read_bytes())
        return {
            "time_base": {"numerator": 1, "denominator": 8},
            "duration_pts": 8,
            "decoded_frame_pts": list(range(8)),
            "cut_pts": [],
            "scenes_sha256": _sha(local_scenes.read_bytes()),
        }

    def extract(_source, target_work, selected):
        frames, media = [], {}
        for order, (_dense_index, pts) in enumerate(selected, 1):
            path = target_work / "speaker-visibility-frames" / f"{order:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"long-sample-{order}".encode())
            relative = path.relative_to(target_work).as_posix()
            media[relative] = path.read_bytes()
            frames.append({
                "order": order, "path": relative,
                "sha256": _sha(path.read_bytes()), "pts": pts,
                "cut_before": False,
            })
        sheet = target_work / "speaker-visibility-contact-sheets" / "000001.png"
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_bytes(b"long-sheet")
        relative = sheet.relative_to(target_work).as_posix()
        media[relative] = sheet.read_bytes()
        return frames, [{
            "order": 1, "path": relative, "sha256": _sha(sheet.read_bytes()),
            "frame_orders": list(range(1, 9)),
        }], media

    class Runner:
        def run(self, scope_root, _prompt):
            assert scope_root == first
            raw = first.joinpath(
                "work", "speaker_visibility_input.json"
            ).read_bytes()
            first.joinpath(
                "work", "speaker_visibility_output.json"
            ).write_bytes(_json_bytes(_skill_output(json.loads(raw))))

    assert pipeline.queue_speaker_timing(settings, created["id"]) == "queued"
    queued = storage.load_meta(settings.data_dir, created["id"])[
        "_speaker_timing_producer"
    ]
    assert [job["scope"] for job in queued["jobs"]] == ["segment:1"]
    assert queued["jobs"][0]["source"]["path"] == "work/segments/1/source.mp4"
    result = pipeline.produce_speaker_timing(
        settings, created["id"], Runner(),
        timeline_probe=probe, sample_extractor=extract,
    )
    assert result == "done", storage.load_meta(settings.data_dir, created["id"])[
        "_speaker_timing_producer"
    ]["error"]
    produced = json.loads(first.joinpath(
        "work", "speaker_visibility_input.json"
    ).read_text(encoding="utf-8"))
    assert produced["persons"][0]["identity_refs"][0]["path"] == (
        "keyframes/speaker-visibility-identities/PERSON_01.png"
    )
    assert not second.joinpath("work", "speaker_visibility_input.json").exists()


def test_real_sampler_selects_eight_real_decoded_pts_per_second(tmp_path):
    source = tmp_path / "source.mp4"
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "color=c=blue:s=64x64:r=24:d=1", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
        capture_output=True,
    )
    (work / "scenes.json").write_text(
        json.dumps({
            "scenes": [{"index": 1, "start_s": 0.0, "end_s": 1.0}],
        }),
        encoding="utf-8",
    )

    timeline = pipeline._probe_speaker_visibility_timeline(
        source, work / "scenes.json"
    )
    selected = pipeline._selected_speaker_visibility_frames(timeline)
    frames, sheets, media = pipeline._extract_speaker_visibility_samples(
        source, work, selected
    )

    assert len(frames) == 8
    assert [frame["pts"] for frame in frames] == [pts for _index, pts in selected]
    assert sheets[0]["frame_orders"] == list(range(1, 9))
    assert set(media) == {
        *(frame["path"] for frame in frames),
        sheets[0]["path"],
    }
