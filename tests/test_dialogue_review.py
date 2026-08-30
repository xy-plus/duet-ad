import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dialogue_review, pipeline, storage
from app.main import create_app
from conftest import AUTH, make_settings


LINES = [
    {"text": "第一句", "start_s": 0.2, "end_s": 0.9},
    {"text": "第二句", "start_s": 1.0, "end_s": 1.8},
]


def _waiting(settings):
    meta = storage.new_conversation(
        settings.data_dir,
        "review",
        "clip.mp4",
        dialogue_review_policy=dialogue_review.REVIEW_REQUIRED,
    )
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        duration_s=6.0,
        source_width=320,
        source_height=240,
        dialogue_mode="auto",
    )
    owner_meta = storage.claim_pipeline_input(settings.data_dir, meta["id"])
    waiting = storage.record_dialogue_analysis(
        settings.data_dir,
        meta["id"],
        owner_meta["_input_owner"],
        policy=dialogue_review.REVIEW_REQUIRED,
        outcome="recognized",
        machine_lines=LINES,
    )
    assert waiting is not None
    return waiting


def _commit_payload(review, *, request_id="dialogue-review-0001", lines=None):
    return {
        "confirm": True,
        "client_request_id": request_id,
        "expected_revision": review["revision"],
        "expected_sha256": review["sha256"],
        "lines": review["lines"] if lines is None else lines,
    }


def test_capability_and_create_default_do_not_pause(
    tmp_path, video_1s,
):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        capability = client.get("/api/capabilities", headers=AUTH).json()
        assert capability["dialogue_review"] == dialogue_review.capability()
        with open(video_1s, "rb") as stream:
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", stream, "video/mp4")},
                data={"client_request_id": "dialogue-default-0001"},
            )
    assert response.status_code == 201
    meta = storage.load_meta(settings.data_dir, response.json()["id"])
    assert meta["dialogue_review_policy"] == dialogue_review.AUTO_CONTINUE


def test_review_policy_is_strict_and_only_allowed_for_auto(
    tmp_path, video_1s,
):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        for index, data in enumerate((
            {"dialogue_review_policy": "unknown"},
            {
                "dialogue_review_policy": dialogue_review.REVIEW_REQUIRED,
                "dialogue_mode": "none",
            },
        )):
            data["client_request_id"] = f"dialogue-invalid-{index:04d}"
            with open(video_1s, "rb") as stream:
                response = client.post(
                    "/api/conversations",
                    headers=AUTH,
                    files={"file": ("clip.mp4", stream, "video/mp4")},
                    data=data,
                )
            assert response.status_code == 422
    assert not settings.data_dir.exists() or not list(settings.data_dir.iterdir())


def test_create_request_id_rejects_dialogue_intent_drift(tmp_path, video_1s):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        with open(video_1s, "rb") as stream:
            created = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", stream, "video/mp4")},
                data={
                    "client_request_id": "dialogue-intent-0001",
                    "dialogue_mode": "auto",
                    "dialogue_review_policy": dialogue_review.REVIEW_REQUIRED,
                },
            )
        assert created.status_code == 201
        with open(video_1s, "rb") as stream:
            conflict = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", stream, "video/mp4")},
                data={
                    "client_request_id": "dialogue-intent-0001",
                    "dialogue_mode": "none",
                    "dialogue_review_policy": dialogue_review.AUTO_CONTINUE,
                },
            )
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "client_request_id_dialogue_review_policy_conflict"
    }


def test_analysis_wait_releases_owner_but_auto_freeze_keeps_it(tmp_path):
    settings = make_settings(tmp_path)
    waiting = _waiting(settings)
    assert waiting["status"] == "processing"
    assert waiting["_input_owner"] is None
    assert waiting["dialogue_review"]["status"] == "waiting"
    assert waiting["dialogue_review"]["revision"] == 1
    assert waiting["dialogue_review"]["sha256"] == dialogue_review.lines_sha256(
        LINES
    )

    auto = storage.new_conversation(settings.data_dir, "auto", "clip.mp4")
    storage.update_meta(settings.data_dir, auto["id"], duration_s=3.0)
    claimed = storage.claim_pipeline_input(settings.data_dir, auto["id"])
    frozen = storage.record_dialogue_analysis(
        settings.data_dir,
        auto["id"],
        claimed["_input_owner"],
        policy=dialogue_review.AUTO_CONTINUE,
        outcome="no_audio",
        machine_lines=[],
    )
    assert frozen["dialogue_review"]["status"] == "frozen"
    assert frozen["dialogue_review"]["frozen_by"] == "automatic"
    assert frozen["_input_owner"] == claimed["_input_owner"]


def test_commit_cas_freezes_revision_and_resumes_exact_operation(tmp_path):
    settings = make_settings(tmp_path)
    waiting = _waiting(settings)
    review = waiting["dialogue_review"]
    edited = [{"text": "改过的台词", "start_s": 0.1, "end_s": 1.2}]
    committed, replay = storage.commit_dialogue_review(
        settings.data_dir,
        waiting["id"],
        request_id="dialogue-review-0001",
        expected_revision=review["revision"],
        expected_sha256=review["sha256"],
        lines=edited,
    )
    assert not replay
    assert committed["id"] == waiting["id"]
    assert committed["dialogue_mode"] == "edit"
    assert committed["voice_lines"] == edited
    assert committed["dialogue_review"]["revision"] == 2
    assert committed["dialogue_review"]["frozen_by"] == "user"
    assert committed["_dialogue_review_continuation"] == "queued"

    replayed, replay = storage.commit_dialogue_review(
        settings.data_dir,
        waiting["id"],
        request_id="dialogue-review-0001",
        expected_revision=review["revision"],
        expected_sha256=review["sha256"],
        lines=edited,
    )
    assert replay and replayed["dialogue_review"]["revision"] == 2

    claimed = storage.claim_queued_dialogue_review_continuation(
        settings.data_dir, waiting["id"]
    )
    assert claimed["_input_owner"]["kind"] == "pipeline"
    assert claimed["_dialogue_review_continuation"] == "running"
    assert storage.claim_queued_dialogue_review_continuation(
        settings.data_dir, waiting["id"]
    ) is None
    finished = storage.finish_input_claim(
        settings.data_dir,
        waiting["id"],
        claimed["_input_owner"],
        status="done",
    )
    assert "_dialogue_review_continuation" not in finished


def test_commit_rejects_stale_cas_and_concurrent_second_writer(tmp_path):
    settings = make_settings(tmp_path)
    waiting = _waiting(settings)
    review = waiting["dialogue_review"]
    with pytest.raises(dialogue_review.DialogueReviewError) as stale:
        storage.commit_dialogue_review(
            settings.data_dir,
            waiting["id"],
            request_id="dialogue-review-stale",
            expected_revision=review["revision"],
            expected_sha256="0" * 64,
            lines=LINES,
        )
    assert stale.value.code == "dialogue_review_conflict"

    results = []

    def commit(request_id, text):
        try:
            result = storage.commit_dialogue_review(
                settings.data_dir,
                waiting["id"],
                request_id=request_id,
                expected_revision=review["revision"],
                expected_sha256=review["sha256"],
                lines=[{"text": text, "start_s": 0.1, "end_s": 1.0}],
            )
            results.append(("ok", result[0]["dialogue_review"]["sha256"]))
        except dialogue_review.DialogueReviewError as exc:
            results.append(("error", exc.code))

    threads = [
        threading.Thread(target=commit, args=("dialogue-writer-0001", "甲")),
        threading.Thread(target=commit, args=("dialogue-writer-0002", "乙")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(result[0] for result in results) == ["error", "ok"]
    assert [result[1] for result in results if result[0] == "error"] == [
        "dialogue_review_read_only"
    ]


def test_commit_api_empty_draft_is_explicit_none_and_read_only(tmp_path):
    settings = make_settings(tmp_path)
    waiting = _waiting(settings)
    review = waiting["dialogue_review"]
    with TestClient(create_app(settings)) as client:
        detail = client.get(
            f"/api/conversations/{waiting['id']}", headers=AUTH
        ).json()
        assert detail["navigation_status"] == "waiting_for_dialogue_review"
        assert detail["dialogue_review"]["editable"] is True
        response = client.post(
            f"/api/conversations/{waiting['id']}/dialogue-review/commit",
            headers=AUTH,
            json=_commit_payload(review, lines=[]),
        )
        assert response.status_code == 200
        assert response.json()["dialogue_review"]["status"] == "frozen"
        assert response.json()["dialogue_review"]["editable"] is False
    meta = storage.load_meta(settings.data_dir, waiting["id"])
    assert meta["dialogue_mode"] == "none"
    assert meta["voice_lines"] == []
    assert meta["_dialogue_review_continuation"] == "queued"


def test_commit_api_strict_payload_and_timestamp_validation(tmp_path):
    settings = make_settings(tmp_path)
    waiting = _waiting(settings)
    review = waiting["dialogue_review"]
    with TestClient(create_app(settings)) as client:
        invalid = _commit_payload(review)
        invalid["extra"] = True
        assert client.post(
            f"/api/conversations/{waiting['id']}/dialogue-review/commit",
            headers=AUTH,
            json=invalid,
        ).status_code == 422
        invalid = _commit_payload(
            review,
            lines=[{"text": "越界", "start_s": 2.0, "end_s": 7.0}],
        )
        response = client.post(
            f"/api/conversations/{waiting['id']}/dialogue-review/commit",
            headers=AUTH,
            json=invalid,
        )
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid_dialogue_review_lines"}


def test_startup_claim_recovers_only_committed_queued_review(tmp_path):
    settings = make_settings(tmp_path)
    waiting = _waiting(settings)
    review = waiting["dialogue_review"]
    storage.commit_dialogue_review(
        settings.data_dir,
        waiting["id"],
        request_id="dialogue-recover-0001",
        expected_revision=review["revision"],
        expected_sha256=review["sha256"],
        lines=LINES,
    )
    untouched = _waiting(settings)
    recovered = storage.claim_queued_dialogue_review_continuations(
        settings.data_dir
    )
    assert [cid for cid, _owner in recovered] == [waiting["id"]]
    assert storage.load_meta(settings.data_dir, untouched["id"])[
        "dialogue_review"
    ]["status"] == "waiting"


def test_outcomes_and_public_state_never_invent_optional_asr_fields():
    state = dialogue_review.analysis_state(
        dialogue_review.REVIEW_REQUIRED,
        "vocal_unrecognized",
        LINES,
    )
    public = dialogue_review.public_state(state)
    assert public["outcome"] == "vocal_unrecognized"
    assert set(public["lines"][0]) == {"text", "start_s", "end_s"}
    assert not ({"language", "speaker", "confidence"} & set(public["lines"][0]))


def test_pipeline_pauses_immediately_after_asr_and_resume_skips_asr(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir,
        "review",
        "clip.mp4",
        dialogue_review_policy=dialogue_review.REVIEW_REQUIRED,
    )
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"source")
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        duration_s=6.0,
        source_width=320,
        source_height=240,
        dialogue_mode="auto",
    )

    class FakeMilestone:
        def public_summary(self):
            return {"version": 1, "skills": {}}

        def read_bytes(self, _name):
            return b"skill"

    def fake_extract(argv, *, timeout, step, cwd=None):
        assert step == "extract"
        work = Path(argv[argv.index("--out-dir") + 1])
        (work / "manifest.json").write_text(
            '{"duration_seconds":6.0}', encoding="utf-8"
        )

    asr_calls = []

    def fake_voice(settings_arg, cid, *_args, **_kwargs):
        asr_calls.append(cid)
        storage.update_meta(
            settings_arg.data_dir,
            cid,
            voice_lines=LINES,
            voice_line_provenance=[
                {**line, "kept": True, "classification": "spoken"}
                for line in LINES
            ],
            voice_analysis_outcome="recognized",
        )
        return LINES

    downstream_calls = []
    monkeypatch.setattr(
        storage, "probe_video", lambda _path: storage.VideoProbe(6.0, 320, 240)
    )
    monkeypatch.setattr(pipeline.skill_milestone, "ensure", lambda *_a, **_k: FakeMilestone())
    monkeypatch.setattr(pipeline, "_run_cmd", fake_extract)
    monkeypatch.setattr(pipeline, "_voice_step", fake_voice)
    monkeypatch.setattr(
        pipeline,
        "_detect_segments",
        lambda *_a, **_k: downstream_calls.append("detect") or [],
    )

    pipeline.run(settings, meta["id"], object())
    waiting = storage.load_meta(settings.data_dir, meta["id"])
    assert asr_calls == [meta["id"]]
    assert downstream_calls == []
    assert waiting["dialogue_review"]["status"] == "waiting"

    automatic = storage.new_conversation(
        settings.data_dir, "automatic", "clip.mp4"
    )
    automatic_dir = settings.data_dir / automatic["id"]
    (automatic_dir / "source.mp4").write_bytes(b"source")
    storage.update_meta(
        settings.data_dir,
        automatic["id"],
        duration_s=6.0,
        source_width=320,
        source_height=240,
        dialogue_mode="auto",
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_segments",
        lambda *_a, **_k: downstream_calls.append("auto-detect")
        or (_ for _ in ()).throw(pipeline.PipelineError("auto-downstream-reached")),
    )
    pipeline.run(settings, automatic["id"], object())
    automatic_meta = storage.load_meta(settings.data_dir, automatic["id"])
    assert automatic_meta["dialogue_review"]["status"] == "frozen"
    assert automatic_meta["dialogue_review"]["frozen_by"] == "automatic"
    assert automatic_meta["error"] == "auto-downstream-reached"

    failed_asr = storage.new_conversation(
        settings.data_dir, "failed-asr", "clip.mp4"
    )
    failed_asr_dir = settings.data_dir / failed_asr["id"]
    (failed_asr_dir / "source.mp4").write_bytes(b"source")
    storage.update_meta(
        settings.data_dir,
        failed_asr["id"],
        duration_s=6.0,
        source_width=320,
        source_height=240,
        dialogue_mode="auto",
    )
    monkeypatch.setattr(
        pipeline,
        "_voice_step",
        lambda *_a, **_k: (_ for _ in ()).throw(
            pipeline.PipelineError("asr_failed")
        ),
    )
    pipeline.run(settings, failed_asr["id"], object())
    failed_asr_meta = storage.load_meta(settings.data_dir, failed_asr["id"])
    assert failed_asr_meta["status"] == "failed"
    assert failed_asr_meta["error"] == "asr_failed"
    assert "dialogue_review" not in failed_asr_meta

    review = waiting["dialogue_review"]
    storage.commit_dialogue_review(
        settings.data_dir,
        meta["id"],
        request_id="dialogue-resume-0001",
        expected_revision=review["revision"],
        expected_sha256=review["sha256"],
        lines=LINES,
    )
    resumed = storage.claim_queued_dialogue_review_continuation(
        settings.data_dir, meta["id"]
    )
    monkeypatch.setattr(
        pipeline,
        "_voice_step",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("ASR must not rerun after review freeze")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_segments",
        lambda *_a, **_k: (_ for _ in ()).throw(
            pipeline.PipelineError("downstream-reached")
        ),
    )
    pipeline.run(
        settings,
        meta["id"],
        object(),
        claimed_owner=resumed["_input_owner"],
    )
    failed = storage.load_meta(settings.data_dir, meta["id"])
    assert failed["error"] == "downstream-reached"
    assert asr_calls == [meta["id"], automatic["id"]]
