from copy import deepcopy

import pytest

from app import storage
from app.project_progress import (
    PROGRESS_FLOOR_FIELD,
    aggregate_project_progress,
)


def _v1_meta(**changes):
    meta = {
        "status": "queued",
        "effective_request": {"version": 1},
        "input_receipt": {"version": 1},
        PROGRESS_FLOOR_FIELD: 0,
    }
    meta.update(changes)
    return meta


def test_queued_projection_is_exact_and_pure():
    meta = _v1_meta()
    original = deepcopy(meta)

    progress = aggregate_project_progress(meta, has_video=False)

    assert progress == {"percent": 0, "status": "queued"}
    assert set(progress) == {"percent", "status"}
    assert meta == original


def test_verified_video_is_the_only_success_authority():
    contradictory = _v1_meta(
        status="failed",
        generation={"status": "failed", "stage": "h3"},
        _project_progress_floor=88,
    )

    assert aggregate_project_progress(contradictory, has_video=True) == {
        "percent": 100,
        "status": "succeeded",
    }
    assert aggregate_project_progress(
        {"status": "done", "generation": {"status": "succeeded"}},
        has_video=False,
    ) == {"percent": 99, "status": "failed"}


def test_v1_floor_prevents_recovery_requeue_from_going_backwards():
    before_recovery = _v1_meta(
        status="done",
        generation={
            "status": "running",
            "segments": [
                {"status": "succeeded"},
                {"status": "succeeded"},
                {"status": "running"},
            ],
        },
    )
    confirmed = aggregate_project_progress(before_recovery, has_video=False)
    recovered = _v1_meta(
        status="queued",
        _project_progress_floor=confirmed["percent"],
    )

    assert confirmed["status"] == "running"
    assert aggregate_project_progress(recovered, has_video=False) == confirmed


def test_failure_retains_the_v1_confirmed_floor():
    meta = _v1_meta(
        status="done",
        postprocess={"status": "failed"},
        _project_progress_floor=73,
    )

    assert aggregate_project_progress(meta, has_video=False) == {
        "percent": 73,
        "status": "failed",
    }


@pytest.mark.parametrize(
    ("meta", "expected_status", "percent_range"),
    [
        ({"status": "queued"}, "queued", range(0, 1)),
        ({"status": "processing"}, "running", range(1, 100)),
        ({"status": "done"}, "running", range(1, 100)),
        (
            {"status": "done", "generation": {"status": "failed"}},
            "failed",
            range(0, 100),
        ),
    ],
)
def test_legacy_projects_receive_a_bounded_coarse_projection(
    meta, expected_status, percent_range
):
    progress = aggregate_project_progress(meta, has_video=False)

    assert progress["status"] == expected_status
    assert progress["percent"] in percent_range
    assert set(progress) == {"percent", "status"}


def test_floor_is_v1_only_and_cannot_publish_100_without_a_video():
    legacy = {"status": "queued", PROGRESS_FLOOR_FIELD: 91}
    malformed_v1_floor = _v1_meta(_project_progress_floor=True)
    excessive_v1_floor = _v1_meta(_project_progress_floor=1000)

    assert aggregate_project_progress(legacy, has_video=False) == {
        "percent": 0,
        "status": "queued",
    }
    assert aggregate_project_progress(malformed_v1_floor, has_video=False) == {
        "percent": 0,
        "status": "queued",
    }
    assert aggregate_project_progress(excessive_v1_floor, has_video=False) == {
        "percent": 99,
        "status": "running",
    }


def test_internal_detail_never_escapes_the_public_object():
    meta = _v1_meta(
        status="done",
        postprocess={
            "status": "done",
            "segments": [{"status": "done", "model": "private-model"}],
        },
        _prompt_fusion={"status": "running", "retry_count": 4},
        generation={
            "status": "running",
            "stage": "h3",
            "provider": "private-provider",
            "task_id": "paid-task-id",
            "segments": [{"status": "running"}],
        },
    )

    progress = aggregate_project_progress(meta, has_video=False)

    assert progress == {"percent": 85, "status": "running"}
    assert set(progress) == {"percent", "status"}


def test_malformed_historical_status_values_still_project_to_valid_output():
    progress = aggregate_project_progress(
        {
            "status": [],
            "postprocess": {"status": []},
            "_prompt_fusion": {"status": []},
            "generation": {"status": []},
        },
        has_video=False,
    )

    assert progress == {"percent": 85, "status": "running"}


def test_storage_persists_v1_floor_across_internal_state_rollback(tmp_path):
    cid = "a" * 32
    cdir = tmp_path / cid
    cdir.mkdir()
    (cdir / "work").mkdir()
    storage._write_meta(cdir, _v1_meta(id=cid))

    advanced = storage.update_meta(
        tmp_path,
        cid,
        status="done",
        generation={
            "status": "running",
            "segments": [
                {"status": "succeeded"},
                {"status": "succeeded"},
                {"status": "running"},
            ],
        },
    )
    assert advanced is not None
    confirmed = aggregate_project_progress(advanced, has_video=False)
    assert 1 <= confirmed["percent"] <= 99

    rolled_back = storage.update_meta(
        tmp_path,
        cid,
        status="queued",
        generation=None,
        _project_progress_floor=0,
    )
    assert rolled_back is not None
    assert rolled_back[PROGRESS_FLOOR_FIELD] == confirmed["percent"]
    assert aggregate_project_progress(rolled_back, has_video=False) == confirmed
    assert storage.load_meta(tmp_path, cid) == rolled_back


def test_storage_floor_is_v1_only_and_never_persists_100(tmp_path):
    legacy = storage.new_conversation(tmp_path, "legacy", "legacy.mp4")
    legacy = storage.update_meta(
        tmp_path,
        legacy["id"],
        status="done",
        generation={"status": "running"},
    )
    assert legacy is not None
    assert PROGRESS_FLOOR_FIELD not in legacy

    cid = "b" * 32
    cdir = tmp_path / cid
    cdir.mkdir()
    (cdir / "work").mkdir()
    storage._write_meta(cdir, _v1_meta(id=cid))
    terminal = storage.update_meta(
        tmp_path,
        cid,
        status="done",
        generation={"status": "succeeded", "stage": "stitch"},
    )

    assert terminal is not None
    assert terminal[PROGRESS_FLOOR_FIELD] == 99
    assert aggregate_project_progress(terminal, has_video=True) == {
        "percent": 100,
        "status": "succeeded",
    }
