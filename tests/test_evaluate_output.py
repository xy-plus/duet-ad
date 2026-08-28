"""Pure-offline output scoring contract tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app import vocal
from scripts import evaluate_output


def _frame(
    time_s: float,
    *,
    cut: float = 0.01,
    topology: float = 0.1,
    motion: float = 0.01,
    scale: float | None = 1.0,
    scale_support: float = 1.0,
) -> evaluate_output.FrameMetric:
    return evaluate_output.FrameMetric(
        time_s=time_s,
        cut_score=cut,
        topology_residual=topology,
        motion_residual=motion,
        scale=scale,
        scale_support=scale_support,
        scale_inliers=10,
        scale_tracks=10,
        topology_grid_row=1,
        topology_grid_column=2,
        topology_grid_score=topology,
    )


def test_visual_summary_reports_scores_without_a_verdict():
    reference = [
        _frame(0.5),
        _frame(1.0, cut=0.9),
        _frame(1.5),
        _frame(2.0),
    ]
    candidate = [
        _frame(0.5, topology=0.2, motion=0.02, scale=1.01),
        _frame(1.08, cut=0.8, topology=0.9, motion=0.5, scale=0.8),
        _frame(1.5, topology=0.3, motion=0.03, scale=1.02),
        _frame(2.0, topology=0.4, motion=0.04, scale=0.99),
    ]

    score, evidence = evaluate_output.summarize_visual(reference, candidate)

    assert score["hard_cut_offset_ms_mean"] == pytest.approx(80.0)
    assert score["hard_cut_offset_ms_max"] == pytest.approx(80.0)
    # The intended cut neighbourhood is excluded from within-shot continuity.
    assert score["topology_residual_p95"] == pytest.approx(0.39)
    assert score["motion_residual_p95"] == pytest.approx(0.039)
    assert score["scale_step_abs_log_pct_p95"] is not None
    assert evidence["hard_cut_matches"][0]["candidate_peak_time_s"] == 1.08
    assert not ({"pass", "fail", "status", "decision"} & set(score))


def test_music_statistics_reuse_vocal_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _path: vocal.VocalAnalysis(
            windows=[
                vocal.VocalWindow(0, 975, sung=0.0, spoken=0.0, music=0.1),
                vocal.VocalWindow(975, 1950, sung=0.0, spoken=0.0, music=0.3),
            ],
            has_bgm=True,
        ),
    )

    score, evidence = evaluate_output.analyze_music(tmp_path / "candidate.mp4")

    assert score == {
        "music_score_mean": pytest.approx(0.2),
        "music_score_p95": pytest.approx(0.29),
        "music_score_max": pytest.approx(0.3),
        "music_window_ratio_at_calibrated_floor": pytest.approx(1.0),
    }
    assert evidence["analyzer"] == "app.vocal.YAMNet"
    assert evidence["window_count"] == 2
    assert evidence["music_score_floor"] == vocal.MUSIC_SCORE_MIN
    assert "has_bgm" not in evidence


def test_music_failure_is_evidence_not_a_quality_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _path: (_ for _ in ()).throw(vocal.VocalError("no audio stream")),
    )

    score, evidence = evaluate_output.analyze_music(tmp_path / "silent.mp4")

    assert all(value is None for value in score.values())
    assert evidence["error"] == "no audio stream"
    assert "status" not in evidence


def test_evaluate_top_level_is_only_score_and_evidence(monkeypatch, tmp_path):
    reference = tmp_path / "reference.mp4"
    candidate = tmp_path / "candidate.mp4"
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(
        evaluate_output,
        "analyze_visual",
        lambda _reference, _candidate: ({"metric": 1.0}, {"peak": 0.5}),
    )
    monkeypatch.setattr(
        evaluate_output,
        "analyze_music",
        lambda _candidate: ({"music_score_mean": 0.2}, {"window_count": 1}),
    )
    monkeypatch.setattr(
        evaluate_output,
        "probe_media",
        lambda _path: {
            "video": {"start_s": 0.04, "duration_s": 1.0},
            "audio": {"start_s": 0.0, "duration_s": 1.0},
        },
    )

    payload = evaluate_output.evaluate(reference, candidate)

    assert set(payload) == {"score", "evidence"}
    assert payload["score"]["media_timing"] == {
        "av_start_offset_ms": pytest.approx(-40.0),
        "av_end_offset_ms": pytest.approx(-40.0),
    }
    banned = {"pass", "fail", "status", "decision", "accepted", "rejected"}
    assert not banned.intersection(_all_mapping_keys(payload))


def test_sidecar_write_is_atomic_and_does_not_touch_meta(tmp_path):
    project_meta = tmp_path / "meta.json"
    project_meta.write_bytes(b'{"status":"done"}\n')
    before = project_meta.read_bytes()
    sidecar = tmp_path / "generated.evaluation.json"
    payload = {"score": {"x": 1.0}, "evidence": {"x": 2.0}}

    evaluate_output.write_sidecar(sidecar, payload)

    assert project_meta.read_bytes() == before
    assert json.loads(sidecar.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob("*.tmp")) == []


def test_sidecar_name_cannot_be_project_meta(tmp_path):
    with pytest.raises(ValueError, match=r"\.evaluation\.json"):
        evaluate_output.write_sidecar(
            tmp_path / "meta.json",
            {"score": {}, "evidence": {}},
        )


@pytest.mark.parametrize("reserved_key", ["pass", "fail", "status", "decision"])
def test_sidecar_rejects_reserved_verdict_keys(tmp_path, reserved_key):
    with pytest.raises(ValueError, match="verdict keys"):
        evaluate_output.write_sidecar(
            tmp_path / "generated.evaluation.json",
            {"score": {reserved_key: 1.0}, "evidence": {}},
        )


def test_script_has_no_network_provider_or_generation_imports():
    tree = ast.parse(Path(evaluate_output.__file__).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert not imports.intersection(
        {
            "requests",
            "httpx",
            "urllib",
            "socket",
            "app.h3",
            "app.pipeline",
            "app.long_generation",
            "app.seedream",
        }
    )
    assert imports.intersection({"app", "app.vocal"}) == {"app"}


def _all_mapping_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_mapping_keys(child))
        return keys
    if isinstance(value, list):
        keys = set()
        for child in value:
            keys.update(_all_mapping_keys(child))
        return keys
    return set()
