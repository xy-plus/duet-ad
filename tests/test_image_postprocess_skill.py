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
            "id": "OUTFIT_01",
            "kind": "outfit",
            "source": "反复出现的深蓝休闲上衣",
            "replacement": "同色同风格的不同剪裁上衣",
            "segments": [1, 2],
        },
        {
            "id": "PERSON_01",
            "kind": "person",
            "source": "反复出现的深发女性",
            "replacement": "椭圆脸、自然直眉的新人物",
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
    assert "性别呈现" in skill and "长相略有不同" in skill
    assert "同类型但不重复" in skill
    assert "同一个新设计" in skill
    assert "work/image_optimization.json" in skill
    assert "global_elements" in skill and "segment_prompts" in skill
    assert "真实提交给 Seedream" in skill
    assert "恰好两句话" in skill
    assert "参考图只认目标身份或设计，不参考构图" in skill
    assert len(skill.encode("utf-8")) < 12 * 1024
    assert not Path("skills/image-continuity/SKILL.md").exists()


def test_skill_prompts_are_short_single_target_edits_with_locked_relations():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "Unicode 字符数" in skill
    assert "不超过 120 个字符" in skill
    assert "恰好两句话" in skill
    assert "生成前自行计数" in skill
    assert "超长必须重写，不能截断" in skill
    assert "32KB" not in skill

    assert "先检查项目全部片段的全部冻结关键帧" in skill
    assert "人物外观" in skill and "背景" in skill
    assert "严格二选一" in skill
    assert "人物脸部和服装" in skill and "稳定同一性" in skill
    assert "优先选择人物外观" in skill
    assert "选择人物外观时不得修改背景" in skill
    assert "选择背景时不得修改人物或服装" in skill

    assert "玩具、商品、道具、宠物永不作为替换目标" in skill
    assert "构图、机位、光线、人物动作" in skill
    assert "目标人物的头部位置、大小、朝向和画面裁切" in skill
    assert "最危险的手与功能道具" in skill
    assert "接触、插入或对齐关系" in skill
    assert "只点名一个最危险的手与功能道具关系" in skill
    assert "先删冗余词" in skill
    assert "不得省略上述锁定项" in skill
    assert "其余一切不变、真实摄影、无文字或 Logo" in skill

    assert "人物目标的第一句只写新面孔与同色同风格不同款服装" in skill
    assert "避免堆砌长相细节" in skill
    assert "不得写任何保持项" in skill
    assert "第二句明确只有人物外观和服装表面可变" in skill
    assert "禁止画质优化、风格化、全场景复述或道具改款" in skill
    assert "independent_parallel" in skill and "不写参考图角色说明" in skill
    assert "anchor_consistency" in skill
    assert "参考图只认目标身份或设计，不参考构图" in skill

    assert "只有真正被替换且跨段重复的 PERSON、OUTFIT、SCENE" in skill
    assert "global_elements 只允许 PERSON、OUTFIT、SCENE" in skill
    assert "同一人物跨段重复且选择人物时" in skill
    assert "所有安全可见的相关段都选择人物" in skill
    assert "相关段不得改背景" in skill
    assert "人物不安全或不可见时，整个相关组件统一选择背景" in skill
    assert "PERSON、OUTFIT 的 replacement 逐字复用" in skill
    assert "相同替换描述必须跨段逐字复用" in skill
    assert "所有 ID 按完整字符串字典序输出" in skill
    assert "三部分语义" not in skill
    assert "四大段" not in skill
    assert "模式说明保持简短" not in skill
    assert "人与物/物与物的握持、接触、插入、对齐、连接、遮挡、前后顺序" not in skill
    assert "陀螺" not in skill and "发射器" not in skill


def test_skill_rejects_continuity_risk_before_deduplication():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "第一优先级是最终视频连续性和现实合理性" in skill
    assert "任何扭曲、元素突变、物理或接触关系异常均一票否决" in skill
    assert "人物或背景替换仅是第二优先级的去重手段" in skill
    assert "只有通过第一层才评价" in skill
    assert "人物替换风险高时，整个相关组件必须项目级统一回退背景" in skill
    assert "不得为完成换人牺牲连续性" in skill


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
