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


def test_skill_keeps_anchor_element_only_and_forbids_frame_narration():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    assert "name: image-postprocess" in skill
    assert "图片二次编辑" in skill
    assert "不代表" in skill and "一定" in skill
    assert "原元素 → 新元素" in skill
    assert "其他输入图片只是元素锚点" in skill
    assert "锚点不提供内容、构图、动作、机位、景别、光线或位置参考" in skill
    assert "仅当目标帧本身存在某元素时" in skill
    assert "不得从锚点向目标帧新增" in skill
    assert "第一张输入是唯一目标帧" in skill
    assert "work/continuity.json" in skill
    assert "跨段冻结约束" in skill
    assert "不得重新设计" in skill
    assert "保持第一张图片的近距离俯拍机位" in skill
    assert "手持玩具位于前景" in skill
    assert "猫位于后方" in skill
    assert "不得写某一帧特有的空间描述" in skill
    assert "不得包含“原镜头语义参考”" in skill
    assert "同性别呈现" in skill
    assert "长相略有不同" in skill
    assert "同类型但不重复的真实场景" in skill
    assert "同一个新人物或同一套新设计" in skill
    assert "必须逐字以下段开头" in skill
    assert "work/image_optimization_prompt.txt" in skill
    assert "视频生成提示词" in skill and "不得读取" in skill
    assert len(skill.encode("utf-8")) < 12 * 1024


def test_continuity_skill_is_compact_and_never_describes_frame_layout():
    skill = Path("skills/image-continuity/SKILL.md").read_text(encoding="utf-8")
    assert "name: image-continuity" in skill
    assert "全局元素映射" in skill
    assert "人物、服装、场景、道具和核心商品" in skill
    assert "SUBJECT/subject" in skill
    assert "不得把动物归为 `PERSON`" in skill
    assert "不得描述" in skill and "构图" in skill and "位置" in skill
    assert "continue" in skill and "hard_cut" in skill
    assert "work/continuity.json" in skill
    assert len(skill.encode("utf-8")) < 10 * 1024


class _ContinuityRunner:
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
            "request": json.loads((workdir / "work" / "request.json").read_text()),
            "session_dir": Path(session_dir),
        })
        (workdir / "work" / "continuity.json").write_text(
            json.dumps(self.output, ensure_ascii=False), encoding="utf-8"
        )


def _continuity_plan():
    return {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": [
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
                "replacement": "暖灰墙面和浅橡木柜体",
                "segments": [1, 2],
            },
        ],
    }


def test_global_continuity_generation_sees_all_segments_but_no_other_session_files(tmp_path):
    session = tmp_path / "session"
    first = session / "work" / "segments" / "1" / "work" / "keyframes"
    second = session / "work" / "segments" / "2" / "work" / "keyframes"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "01.png").write_bytes(_png(10))
    (second / "01.png").write_bytes(_png(20))
    (session / "work" / "prompt.txt").write_text("H3 SECRET", encoding="utf-8")
    runner = _ContinuityRunner(_continuity_plan())

    plan = image_optimization.generate_continuity_plan(
        runner,
        [
            {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut", "keyframes_dir": first},
            {"index": 2, "chain_id": "chain-001", "join_mode": "continue", "keyframes_dir": second},
        ],
        session_dir=session,
    )

    assert plan == _continuity_plan()
    assert runner.calls[0]["request"] == {
        "segments": [
            {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
            {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
        ]
    }
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/request.json",
        "work/segments/1/keyframes/01.png",
        "work/segments/2/keyframes/01.png",
    ]
    assert runner.calls[0]["session_dir"] == session.resolve()


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"version": 1, "segment_indices": [1, 2], "elements": [], "extra": True},
        {"version": 1, "segment_indices": [1, 2], "elements": [{"id": "PERSON_01"}]},
        {
            "version": 1,
            "segment_indices": [1, 2],
            "elements": [{
                "id": "PERSON_01", "kind": "person", "source": "x",
                "replacement": "y", "segments": [2, 1],
            }],
        },
        {
            "version": 1,
            "segment_indices": [1, 2],
            "elements": [{
                "id": "SCENE_01", "kind": "person", "source": "x",
                "replacement": "y", "segments": [1, 2],
            }],
        },
        {
            "version": 1,
            "segment_indices": [1, 2],
            "elements": [{
                "id": "SUBJECT_01", "kind": "person", "source": "猫",
                "replacement": "另一只猫", "segments": [1, 2],
            }],
        },
    ],
)
def test_global_continuity_rejects_noncanonical_output(tmp_path, output):
    session = tmp_path / "session"
    first = session / "work" / "segments" / "1" / "work" / "keyframes"
    second = session / "work" / "segments" / "2" / "work" / "keyframes"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "01.png").write_bytes(_png())
    (second / "01.png").write_bytes(_png())
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.generate_continuity_plan(
            _ContinuityRunner(output),
            [
                {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut", "keyframes_dir": first},
                {"index": 2, "chain_id": "chain-001", "join_mode": "continue", "keyframes_dir": second},
            ],
            session_dir=session,
        )


class _InspectingRunner:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = []

    def run_isolated(self, workdir, prompt, *, session_dir):
        workdir = Path(workdir)
        files = sorted(
            str(path.relative_to(workdir))
            for path in workdir.rglob("*")
            if path.is_file()
        )
        request = json.loads((workdir / "work" / "request.json").read_text())
        self.calls.append({
            "files": files,
            "prompt": prompt,
            "request": request,
            "continuity": (
                json.loads((workdir / "work" / "continuity.json").read_text())
                if (workdir / "work" / "continuity.json").is_file() else None
            ),
            "session_dir": Path(session_dir),
        })
        (workdir / "work" / "image_optimization_prompt.txt").write_text(
            self.output, encoding="utf-8"
        )


def test_codex_prompt_generation_sees_only_skill_frames_and_mode(tmp_path):
    session = tmp_path / "session"
    keyframes = session / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png(10))
    (keyframes / "02.png").write_bytes(_png(20))
    (session / "work" / "prompt.txt").write_text("H3 SECRET PROMPT", encoding="utf-8")
    runner = _InspectingRunner("  Seedream 最终真实提示词  \n")

    result = image_optimization.generate_prompt(
        runner,
        keyframes,
        "anchor_consistency",
        session_dir=session,
    )

    assert result == "Seedream 最终真实提示词"
    assert len(runner.calls) == 1
    assert runner.calls[0]["request"] == {"edit_mode": "anchor_consistency"}
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/keyframes/01.png",
        "work/keyframes/02.png",
        "work/request.json",
    ]
    assert "H3 SECRET PROMPT" not in runner.calls[0]["prompt"]
    assert runner.calls[0]["session_dir"] == session.resolve()


def test_segment_prompt_receives_only_its_global_elements(tmp_path):
    session = tmp_path / "session"
    keyframes = session / "work" / "segments" / "2" / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    runner = _InspectingRunner("真实分段提示词")

    continuity = {
        "version": 1,
        "segment_indices": [1, 2, 3],
        "elements": [
            {
                "id": "OUTFIT_01", "kind": "outfit", "source": "外套",
                "replacement": "灰色针织外套", "segments": [1, 3],
            },
            {
                "id": "PERSON_01", "kind": "person", "source": "人物",
                "replacement": "椭圆脸新人物", "segments": [1, 2],
            },
            {
                "id": "SCENE_01", "kind": "scene", "source": "厨房",
                "replacement": "暖灰厨房", "segments": [2, 3],
            },
        ],
    }
    result = image_optimization.generate_prompt(
        runner,
        keyframes,
        "anchor_consistency",
        session_dir=session,
        segment_index=2,
        continuity=continuity,
    )

    assert result == "真实分段提示词"
    assert runner.calls[0]["request"] == {
        "edit_mode": "anchor_consistency", "segment_index": 2,
    }
    assert runner.calls[0]["continuity"] == {
        "version": 1,
        "segment_index": 2,
        "elements": continuity["elements"][1:],
    }
    assert "work/continuity.json" in runner.calls[0]["files"]


def test_continuity_receipt_is_private_deterministic_and_exact():
    frozen = image_optimization.freeze_continuity(_continuity_plan())
    assert set(frozen) == {"_image_continuity"}
    receipt = frozen["_image_continuity"]
    assert set(receipt) == {"version", "segment_indices", "elements", "sha256"}
    assert receipt["version"] == 1
    assert receipt["elements"] == _continuity_plan()["elements"]
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
    assert not any(str(tmp_path) in part and part.endswith("codex_last_message.txt") for part in argv)


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_codex_prompt_generation_rejects_unknown_mode(tmp_path, mode):
    keyframes = tmp_path / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    with pytest.raises(ValueError, match="edit mode"):
        image_optimization.generate_prompt(
            _InspectingRunner("x"), keyframes, mode, session_dir=tmp_path
        )


@pytest.mark.parametrize("output", ["", " \n", "x" * (32 * 1024 + 1)])
def test_codex_prompt_generation_rejects_invalid_output(tmp_path, output):
    keyframes = tmp_path / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    with pytest.raises(ValueError, match="output"):
        image_optimization.generate_prompt(
            _InspectingRunner(output),
            keyframes,
            "independent_parallel",
            session_dir=tmp_path,
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
    assert image_optimization.receipt({**meta, "status": "done", "_image_optimization": frozen}) == frozen


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
    meta = {"schema_version": 2, "status": "processing", "segments": [{"index": 1}]}
    with pytest.raises(ValueError, match="prompt segments"):
        image_optimization.freeze_prompts(settings, meta, {True: "not segment one"})
