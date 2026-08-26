import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import codex_runner, image_optimization
from app.codex_runner import CodexRunner
from conftest import make_settings


def _png(value: int = 127) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((8, 6, 3), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _elements() -> list[dict]:
    return [
        {
            "id": "PERSON_01",
            "kind": "person",
            "source": "反复出现的深发女性",
            "replacement": "椭圆脸、自然直眉的新人物",
            "segments": [1, 2],
        },
        {
            "id": "SCENE_01",
            "kind": "scene",
            "source": "两段重复出现的厨房",
            "replacement": "同类但不重复的暖灰厨房",
            "segments": [1, 2],
        },
    ]


def _multi_output() -> dict:
    return {
        "version": 1,
        "segment_indices": [1, 2],
        "global_elements": _elements(),
        "segment_prompts": [
            {"segment_index": 1, "prompt": "第一段真实 Seedream 提示词"},
            {"segment_index": 2, "prompt": "第二段真实 Seedream 提示词"},
        ],
    }


class _ProjectRunner:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls = []

    def run_isolated(self, workdir, prompt, *, session_dir):
        workdir = Path(workdir)
        self.calls.append({
            "files": sorted(
                str(path.relative_to(workdir))
                for path in workdir.rglob("*") if path.is_file()
            ),
            "request": json.loads(
                (workdir / "work" / "request.json").read_text(encoding="utf-8")
            ),
            "prompt": prompt,
            "session_dir": Path(session_dir),
        })
        (workdir / "work" / "image_optimization.json").write_text(
            json.dumps(self.output, ensure_ascii=False), encoding="utf-8"
        )


def _segments(session: Path) -> list[dict]:
    first = session / "work" / "segments" / "1" / "work" / "keyframes"
    second = session / "work" / "segments" / "2" / "work" / "keyframes"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "01.png").write_bytes(_png(10))
    (second / "01.png").write_bytes(_png(20))
    return [
        {
            "index": 1,
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
            "keyframes_dir": first,
        },
        {
            "index": 2,
            "chain_id": "chain-001",
            "join_mode": "continue",
            "keyframes_dir": second,
        },
    ]


def test_skill_is_single_project_level_prompt_compiler():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    assert "name: image-postprocess" in skill
    assert "为了降低素材重复度" in skill
    assert "同性别呈现" in skill and "长相略有不同" in skill
    assert "同类型但不重复" in skill
    assert "同一个新元素或同一套新设计" in skill
    assert "work/image_optimization.json" in skill
    assert "global_elements" in skill and "segment_prompts" in skill
    assert "真实提交给 Seedream" in skill
    assert "第一张输入是唯一目标帧" in skill
    assert "锚点" in skill and "绝不作为构图或画面内容参考" in skill
    assert "保持第一张图片的近距离俯拍机位" in skill
    assert "手持玩具位于前景" in skill
    assert "猫位于后方" in skill
    assert "不得把动物归为人物" in skill
    assert len(skill.encode("utf-8")) < 12 * 1024
    assert not Path("skills/image-continuity/SKILL.md").exists()


def test_one_project_call_returns_global_map_and_all_real_prompts(tmp_path):
    session = tmp_path / "session"
    segments = _segments(session)
    (session / "work" / "prompt.txt").write_text("H3 SECRET", encoding="utf-8")
    runner = _ProjectRunner(_multi_output())

    continuity, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "anchor_consistency",
        session_dir=session,
    )

    assert len(runner.calls) == 1
    assert continuity == {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": _elements(),
    }
    assert prompts == {
        1: "第一段真实 Seedream 提示词",
        2: "第二段真实 Seedream 提示词",
    }
    assert runner.calls[0]["request"] == {
        "edit_mode": "anchor_consistency",
        "segments": [
            {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
            {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
        ],
    }
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/request.json",
        "work/segments/1/keyframes/01.png",
        "work/segments/2/keyframes/01.png",
    ]
    assert "H3 SECRET" not in runner.calls[0]["prompt"]
    assert runner.calls[0]["session_dir"] == session.resolve()


def test_short_project_uses_same_skill_and_has_no_global_map(tmp_path):
    session = tmp_path / "session"
    keyframes = session / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    runner = _ProjectRunner({
        "version": 1,
        "segment_indices": [0],
        "global_elements": [],
        "segment_prompts": [
            {"segment_index": 0, "prompt": "短视频真实 Seedream 提示词"}
        ],
    })

    continuity, prompts = image_optimization.generate_project_prompts(
        runner,
        [{
            "index": 0,
            "chain_id": "short-000",
            "join_mode": "hard_cut",
            "keyframes_dir": keyframes,
        }],
        "independent_parallel",
        session_dir=session,
    )

    assert continuity is None
    assert prompts == {0: "短视频真实 Seedream 提示词"}
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/request.json",
        "work/segments/0/keyframes/01.png",
    ]


@pytest.mark.parametrize(
    "output",
    [
        {},
        {**_multi_output(), "extra": True},
        {**_multi_output(), "segment_indices": [2, 1]},
        {**_multi_output(), "segment_prompts": [{"segment_index": 1, "prompt": "x"}]},
        {
            **_multi_output(),
            "segment_prompts": [
                {"segment_index": 2, "prompt": "x"},
                {"segment_index": 1, "prompt": "y"},
            ],
        },
        {
            **_multi_output(),
            "global_elements": [{
                "id": "SUBJECT_01",
                "kind": "person",
                "source": "猫",
                "replacement": "另一只猫",
                "segments": [1, 2],
            }],
        },
        {
            **_multi_output(),
            "segment_prompts": [
                {"segment_index": 1, "prompt": " "},
                {"segment_index": 2, "prompt": "x"},
            ],
        },
        {
            **_multi_output(),
            "segment_prompts": [
                {"segment_index": 1, "prompt": "x" * (32 * 1024 + 1)},
                {"segment_index": 2, "prompt": "x"},
            ],
        },
    ],
)
def test_project_output_rejects_noncanonical_shapes(tmp_path, output):
    session = tmp_path / "session"
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.generate_project_prompts(
            _ProjectRunner(output),
            _segments(session),
            "anchor_consistency",
            session_dir=session,
        )


def test_short_project_rejects_global_elements(tmp_path):
    keyframes = tmp_path / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    output = {
        "version": 1,
        "segment_indices": [0],
        "global_elements": _elements(),
        "segment_prompts": [{"segment_index": 0, "prompt": "x"}],
    }
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.generate_project_prompts(
            _ProjectRunner(output),
            [{
                "index": 0,
                "chain_id": "short-000",
                "join_mode": "hard_cut",
                "keyframes_dir": keyframes,
            }],
            "anchor_consistency",
            session_dir=tmp_path,
        )


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_project_generation_rejects_unknown_mode(tmp_path, mode):
    with pytest.raises(ValueError, match="edit mode"):
        image_optimization.generate_project_prompts(
            _ProjectRunner({}), [], mode, session_dir=tmp_path
        )


def test_continuity_receipt_is_private_deterministic_and_exact():
    plan = {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": _elements(),
    }
    frozen = image_optimization.freeze_continuity(plan)
    receipt = frozen["_image_continuity"]
    assert set(receipt) == {"version", "segment_indices", "elements", "sha256"}
    assert image_optimization.continuity_receipt(frozen) == receipt
    tampered = json.loads(json.dumps(frozen))
    tampered["_image_continuity"]["elements"][0]["replacement"] = "changed"
    assert image_optimization.continuity_receipt(tampered) is None


def test_generic_isolated_runner_wraps_stage_and_discards_last_message(tmp_path, monkeypatch):
    runner = CodexRunner(timeout_s=1, concurrency=1)
    captured = []
    monkeypatch.setattr(codex_runner, "_resolve_bwrap", lambda: Path("/bin/true"))

    def inspect(workdir, prompt):
        captured.append(runner.build_argv(workdir, prompt))

    monkeypatch.setattr(runner, "run", inspect)
    with tempfile.TemporaryDirectory(prefix="duet-isolated-test-", dir="/tmp") as raw:
        stage = Path(raw).resolve()
        runner.run_isolated(stage, "execute skill", session_dir=tmp_path)

    argv = captured[0]
    assert argv[0] == "/bin/true"
    assert "--tmpfs" in argv and str(codex_runner._CHECKOUT_BOUNDARY) in argv
    assert "/dev/null" in argv
    assert not any(
        str(tmp_path) in part and part.endswith("codex_last_message.txt")
        for part in argv
    )


def test_freeze_uses_only_codex_outputs_and_has_no_template(tmp_path):
    settings = make_settings(tmp_path, seedream_edit_mode="independent_parallel")
    meta = {
        "schema_version": 2,
        "status": "processing",
        "segments": [
            {"index": 1, "prompt": "H3 ONE"},
            {"index": 2, "prompt": "H3 TWO"},
        ],
    }
    frozen = image_optimization.freeze_prompts(
        settings, meta, {1: "Codex image one", 2: "Codex image two"}
    )["_image_optimization"]

    assert frozen["version"] == 2
    assert set(frozen) == {"version", "model", "edit_mode", "segments"}
    assert [item["current"] for item in frozen["segments"]] == [
        "Codex image one", "Codex image two",
    ]
    assert "H3 ONE" not in json.dumps(frozen)
    assert image_optimization.receipt(
        {**meta, "status": "done", "_image_optimization": frozen}
    ) == frozen


def test_freeze_requires_exact_prompt_for_every_segment(tmp_path):
    settings = make_settings(tmp_path)
    meta = {
        "schema_version": 2,
        "status": "processing",
        "segments": [{"index": 1}, {"index": 2}],
    }
    with pytest.raises(ValueError, match="prompt segments"):
        image_optimization.freeze_prompts(settings, meta, {1: "only one"})


def test_freeze_rejects_boolean_segment_key(tmp_path):
    settings = make_settings(tmp_path)
    meta = {
        "schema_version": 2,
        "status": "processing",
        "segments": [{"index": 1}],
    }
    with pytest.raises(ValueError, match="prompt segments"):
        image_optimization.freeze_prompts(
            settings, meta, {True: "not segment one"}
        )
