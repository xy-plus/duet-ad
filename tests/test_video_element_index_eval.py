"""Offline evaluator for real video-maker project-index executions.

The helpers in this module do not participate in runtime control flow.  They
execute the isolated project-index phase only when called explicitly, preserve
the frozen inputs, and return continuous review dimensions rather than a
pass/fail or release decision.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app import pipeline
from app.codex_runner import CodexRunner


DIMENSIONS = (
    "project_element_coverage",
    "stable_key_consistency",
    "occurrence_accuracy",
    "replaceable_preserve_separation",
    "fragment_non_promotion",
    "frozen_readonly_prompt_isolation",
)

_NEUTRAL_KEY_RE = re.compile(r"^(person|entity|scene)-[0-9]{2,}$")


def _f1(true_positive: float, false_positive: float, false_negative: float) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 100.0
    return 100.0 * 2 * true_positive / denominator


@dataclass(frozen=True)
class ReviewCounts:
    coverage_tp: float
    coverage_fp: float
    coverage_fn: float
    stable_pair_tp: float
    stable_pair_fp: float
    stable_pair_fn: float
    occurrence_tp: float
    occurrence_fp: float
    occurrence_fn: float
    separation_credit: float
    separation_total: float
    fragment_promotions: float
    fragment_traps: float
    frozen_hashes_unchanged: float
    frozen_hashes_total: float
    request_entries_correct: float
    request_entries_total: float
    prompt_absent_credit: float
    stable_key_identity_credit: float = 0.0
    stable_key_identity_total: float = 0.0


def score_review(counts: ReviewCounts) -> dict[str, float]:
    """Return six independent 0..100 dimensions; never derive pass/fail."""
    separation = (
        100.0
        if counts.separation_total == 0
        else 100.0 * counts.separation_credit / counts.separation_total
    )
    fragment = 100.0 * (
        1.0
        - min(
            1.0,
            counts.fragment_promotions / max(1.0, counts.fragment_traps),
        )
    )
    immutable = (
        100.0
        if counts.frozen_hashes_total == 0
        else 100.0
        * counts.frozen_hashes_unchanged
        / counts.frozen_hashes_total
    )
    request = (
        100.0
        if counts.request_entries_total == 0
        else 100.0
        * counts.request_entries_correct
        / counts.request_entries_total
    )
    isolation = (immutable + request + 100.0 * counts.prompt_absent_credit) / 3.0
    stable_key = _f1(
        counts.stable_pair_tp,
        counts.stable_pair_fp,
        counts.stable_pair_fn,
    )
    if counts.stable_key_identity_total:
        stable_key = min(
            stable_key,
            100.0
            * counts.stable_key_identity_credit
            / counts.stable_key_identity_total,
        )
    scores = {
        "project_element_coverage": _f1(
            counts.coverage_tp, counts.coverage_fp, counts.coverage_fn
        ),
        "stable_key_consistency": stable_key,
        "occurrence_accuracy": _f1(
            counts.occurrence_tp, counts.occurrence_fp, counts.occurrence_fn
        ),
        "replaceable_preserve_separation": separation,
        "fragment_non_promotion": fragment,
        "frozen_readonly_prompt_isolation": isolation,
    }
    return {name: round(max(0.0, min(100.0, value)), 2) for name, value in scores.items()}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _EvidenceRunner:
    def __init__(self, evidence_dir: Path, *, timeout_s: int) -> None:
        self._evidence_dir = evidence_dir
        self._delegate = CodexRunner(timeout_s=timeout_s, concurrency=1)

    def run_isolated(
        self,
        workdir: Path,
        prompt: str,
        *,
        session_dir: Path,
        writable_paths: tuple[Path, ...] = (),
    ) -> None:
        inventory = sorted(
            path.relative_to(workdir).as_posix()
            for path in workdir.rglob("*")
            if path.is_file()
        )
        (self._evidence_dir / "staged-input-inventory.json").write_text(
            json.dumps(
                {
                    "files": inventory,
                    "prompt_txt_present": "work/prompt.txt" in inventory,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            workdir / "work/project_index_request.json",
            self._evidence_dir / "project_index_request.json",
        )
        self._delegate.run_isolated(
            workdir,
            prompt,
            session_dir=session_dir,
            writable_paths=writable_paths,
        )
        shutil.copy2(
            workdir / "work/element_index.json",
            self._evidence_dir / "raw-element_index.json",
        )


def run_real_project_index(
    segment_directories: list[Path],
    evidence_dir: Path,
    *,
    skill_bytes: bytes,
    timeout_s: int = 1800,
) -> Path:
    """Execute project_index against absolute frozen PNG directories and Skill bytes."""
    if not segment_directories or not evidence_dir.is_absolute():
        raise ValueError("absolute evidence directory and at least one segment are required")
    if any(not directory.is_absolute() for directory in segment_directories):
        raise ValueError("segment directories must be absolute")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    session_dir = evidence_dir / "session"
    (session_dir / "work").mkdir(parents=True, exist_ok=True)
    frame_paths = {
        segment_index: sorted(directory.glob("*.png"))
        for segment_index, directory in enumerate(segment_directories, 1)
    }
    if any(not paths for paths in frame_paths.values()):
        raise ValueError("every segment directory must contain frozen PNG frames")
    before = {
        f"{segment_index}/{frame_order:02d}": _digest(path)
        for segment_index, paths in frame_paths.items()
        for frame_order, path in enumerate(paths, 1)
    }
    (evidence_dir / "frozen-input-before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = pipeline._generate_project_element_index(
        _EvidenceRunner(evidence_dir, timeout_s=timeout_s),
        session_dir,
        frame_paths,
        skill_bytes=skill_bytes,
    )
    after = {
        f"{segment_index}/{frame_order:02d}": _digest(path)
        for segment_index, paths in frame_paths.items()
        for frame_order, path in enumerate(paths, 1)
    }
    (evidence_dir / "frozen-input-after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("frozen input bytes changed during offline evaluation")
    return result


def index_summary(index_path: Path) -> dict[str, object]:
    """Produce a compact human-review summary from a real index artifact."""
    value = json.loads(index_path.read_text(encoding="utf-8"))
    summary: dict[str, object] = {}
    for category in ("people", "entities", "scenes"):
        items = value.get(category, {})
        if not isinstance(items, dict):
            summary[category] = {"invalid": True}
            continue
        summary[category] = {
            key: {
                "description": item.get("source_visual_description"),
                "occurrences": item.get("occurrences"),
                "replaceable": item.get("replaceable"),
                "preserve": item.get("preserve"),
            }
            for key, item in items.items()
            if isinstance(item, dict)
        }
    return summary


def audit_key_identity(index_path: Path) -> dict[str, object]:
    """Measure whether project keys are neutral, immutable binding IDs.

    This is a continuous review signal.  It intentionally does not decide
    whether an execution may continue or be released.
    """
    value = json.loads(index_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for category in ("people", "entities", "scenes"):
        items = value.get(category, {})
        if not isinstance(items, dict):
            continue
        prefix = {
            "people": "person",
            "entities": "entity",
            "scenes": "scene",
        }[category]
        for key in items:
            neutral = isinstance(key, str) and bool(
                _NEUTRAL_KEY_RE.fullmatch(key)
            ) and key.startswith(f"{prefix}-")
            rows.append({
                "category": category,
                "key": key,
                "neutral": neutral,
            })
    total = len(rows)
    neutral = sum(1 for row in rows if row["neutral"])
    return {
        "total_keys": total,
        "neutral_keys": neutral,
        "neutral_ratio": 100.0 if total == 0 else round(100.0 * neutral / total, 2),
        "violations": [row for row in rows if not row["neutral"]],
    }


def write_run_manifest(
    evidence_dir: Path,
    *,
    skill_sha256: str,
    case_name: str,
    split: str,
    frame_directories: list[Path],
    index_path: Path,
    scores: dict[str, float],
    review_notes: list[str],
) -> Path:
    """Persist review evidence without changing runtime or making a decision."""
    payload = {
        "case": case_name,
        "split": split,
        "skill_sha256": skill_sha256,
        "frame_directories": [str(path) for path in frame_directories],
        "index_path": str(index_path),
        "index_sha256": _digest(index_path),
        "scores": scores,
        "review_notes": review_notes,
    }
    target = evidence_dir / "review.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def test_score_review_is_continuous_and_dimensioned():
    scores = score_review(
        ReviewCounts(
            coverage_tp=7.5,
            coverage_fp=1,
            coverage_fn=2,
            stable_pair_tp=12,
            stable_pair_fp=1,
            stable_pair_fn=3,
            occurrence_tp=31,
            occurrence_fp=4,
            occurrence_fn=5,
            separation_credit=17,
            separation_total=20,
            fragment_promotions=1,
            fragment_traps=8,
            frozen_hashes_unchanged=27,
            frozen_hashes_total=27,
            request_entries_correct=27,
            request_entries_total=27,
            prompt_absent_credit=1,
        )
    )

    assert tuple(scores) == DIMENSIONS
    assert all(0.0 < value <= 100.0 for value in scores.values())
    assert any(value != round(value) for value in scores.values())


def test_score_review_has_no_pass_fail_result():
    scores = score_review(
        ReviewCounts(
            coverage_tp=0,
            coverage_fp=4,
            coverage_fn=9,
            stable_pair_tp=0,
            stable_pair_fp=3,
            stable_pair_fn=8,
            occurrence_tp=0,
            occurrence_fp=5,
            occurrence_fn=20,
            separation_credit=0,
            separation_total=6,
            fragment_promotions=5,
            fragment_traps=5,
            frozen_hashes_unchanged=27,
            frozen_hashes_total=27,
            request_entries_correct=27,
            request_entries_total=27,
            prompt_absent_credit=1,
        )
    )

    assert set(scores) == set(DIMENSIONS)
    assert "pass" not in scores
    assert "fail" not in scores


def test_score_review_includes_neutral_key_identity_without_a_gate():
    scores = score_review(
        ReviewCounts(
            coverage_tp=1,
            coverage_fp=0,
            coverage_fn=0,
            stable_pair_tp=10,
            stable_pair_fp=0,
            stable_pair_fn=0,
            occurrence_tp=1,
            occurrence_fp=0,
            occurrence_fn=0,
            separation_credit=1,
            separation_total=1,
            fragment_promotions=0,
            fragment_traps=1,
            frozen_hashes_unchanged=1,
            frozen_hashes_total=1,
            request_entries_correct=1,
            request_entries_total=1,
            prompt_absent_credit=1,
            stable_key_identity_credit=1,
            stable_key_identity_total=2,
        )
    )

    assert scores["stable_key_consistency"] == 50.0
    assert "pass" not in scores
    assert "fail" not in scores


def test_audit_key_identity_is_continuous_and_category_scoped(tmp_path):
    path = tmp_path / "element_index.json"
    path.write_text(
        json.dumps(
            {
                "people": {"person-01": {}},
                "entities": {"entity-01": {}, "red-object": {}},
                "scenes": {"scene-01": {}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = audit_key_identity(path)

    assert audit["total_keys"] == 4
    assert audit["neutral_keys"] == 3
    assert audit["neutral_ratio"] == 75.0
    assert audit["violations"] == [
        {"category": "entities", "key": "red-object", "neutral": False}
    ]
