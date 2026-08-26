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


def test_independent_skill_is_compact_and_declares_optional_secondary_editing():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    assert "name: image-postprocess" in skill
    assert "可能选择" in skill
    assert "图片二次编辑" in skill
    assert "不代表" in skill and "一定" in skill
    assert "work/image_optimization_prompt.txt" in skill
    assert "prompt.txt" in skill and "不得读取" in skill
    assert len(skill.encode("utf-8")) < 12 * 1024


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
