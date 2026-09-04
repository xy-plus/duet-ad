from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

import app.deepseek_runner as deepseek_runner
import eval_video_prompt_fusion_artifact as fusion_eval
import run_real_image_postprocess_eval as image_eval


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_deepseek_model_config_matches_runner_request_contract() -> None:
    config_path = Path(__file__).with_name("skill_iteration_model_config.v1.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["schema"] == "duet.skill-iteration-model-config"
    assert config["version"] == 1
    assert set(config) == {
        "schema", "version", "provider", "model", "temperature",
        "max_tokens", "thinking", "transport", "structured_output",
        "multimodal",
    }
    assert config["provider"] == "deepseek"
    assert config["model"] == deepseek_runner.MODEL
    assert config["temperature"] == 0
    assert config["max_tokens"] == deepseek_runner._MAX_OUTPUT_TOKENS
    assert config["thinking"] == {"type": "disabled"}
    assert config["transport"] == {
        "method": "POST",
        "endpoint": deepseek_runner.CHAT_URL,
        "stream": False,
        "content_type": "application/json",
    }
    assert config["structured_output"] == {
        "function_name": "submit_result",
        "tool_type": "function",
        "strict": True,
        "tool_choice_type": "function",
    }
    assert config["multimodal"] == {"image_detail": "low"}

    staged = deepseek_runner._StagedInput(
        path="SKILL.md", sha256="a" * 64, data=b"skill", image_mime=None,
    )
    payload, _omitted = deepseek_runner._payload(
        prompt="prompt", inputs=(staged,), output_schema={
            "type": "object", "properties": {}, "required": [],
        },
    )
    assert payload["model"] == config["model"]
    assert payload["temperature"] == config["temperature"]
    assert payload["max_tokens"] == config["max_tokens"]
    assert payload["thinking"] == config["thinking"]
    assert payload["stream"] is config["transport"]["stream"]
    assert payload["tools"][0]["type"] == config["structured_output"]["tool_type"]
    function = payload["tools"][0]["function"]
    assert function["name"] == config["structured_output"]["function_name"]
    assert function["strict"] is config["structured_output"]["strict"]
    assert payload["tool_choice"]["type"] == config["structured_output"]["tool_choice_type"]
    assert payload["tool_choice"]["function"]["name"] == function["name"]


def _image_source(
    root: Path,
    topology: list[tuple[str, str]],
    *,
    include_sampling: bool = True,
) -> Path:
    segments = []
    for index, (chain_id, join_mode) in enumerate(topology, 1):
        work = root / "work" / "segments" / str(index) / "work"
        keyframes = work / "keyframes"
        keyframes.mkdir(parents=True)
        (keyframes / "01.png").write_bytes(f"frame-{index}".encode())
        if include_sampling:
            _write_json(
                work / "keyframe_sampling.json",
                {
                    "keyframes": [{
                        "order": 1,
                        "source_time_s": float(index),
                        "source_scene_id": "SCENE_SHARED",
                        "transition": {
                            "type": "start",
                            "at_s": float(index),
                        },
                    }],
                },
            )
        segments.append({
            "index": index,
            "chain_id": chain_id,
            "join_mode": join_mode,
        })
    _write_json(
        root / "long_video_plan.json",
        {
            "schema": "duet.long-video-plan",
            "version": 5,
            "segments": segments,
        },
    )
    return root.resolve(strict=True)


def _fusion_input(root: Path) -> tuple[Path, bytes]:
    frames = []
    for order in range(1, 10):
        relative = Path("work") / "frames" / f"{order:02d}.png"
        data = f"fusion-frame-{order}".encode()
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        frames.append({
            "order": order,
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "segment_time_s": float(order - 1),
            "source_scene_id": "SCENE_01",
            "transition": (
                {"type": "start", "at_segment_s": 0.0}
                if order == 1
                else {"type": "continuous", "at_segment_s": None}
            ),
        })
    old_prompt = "frozen old visual"
    frame_prompt = "frozen image optimization"
    value = {
        "schema": "duet.video-prompt-fusion-input",
        "version": 2,
        "segments": [{
            "index": 1,
            "new_keyframes": frames,
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(old_prompt.encode()).hexdigest(),
            },
            "image_optimization_prompt": [{
                "order": order,
                "text": frame_prompt,
                "sha256": hashlib.sha256(frame_prompt.encode()).hexdigest(),
            } for order in range(1, 10)],
            "audio_content": {
                "lines_json": "[]",
                "voice_references": [],
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "music_policy": "forbid",
            },
        }],
    }
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    path = root / "work" / "multimodal_input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, raw


def test_image_segments_use_frozen_multichain_topology(tmp_path: Path) -> None:
    root = _image_source(
        tmp_path / "case",
        [
            ("chain-001", "hard_cut"),
            ("chain-001", "continue"),
            ("chain-002", "hard_cut"),
        ],
    )

    specs, _paths = image_eval._segments(root)

    assert [
        (spec["index"], spec["chain_id"], spec["join_mode"])
        for spec in specs
    ] == [
        (1, "chain-001", "hard_cut"),
        (2, "chain-001", "continue"),
        (3, "chain-002", "hard_cut"),
    ]
    assert specs[1]["transition_skeleton"][0][
        "source_transition_from_previous"
    ] == "same_camera"
    assert specs[2]["transition_skeleton"][0][
        "source_transition_from_previous"
    ] == "hard_cut"


def test_image_segments_reject_receipt_segment_mismatch(tmp_path: Path) -> None:
    root = _image_source(
        tmp_path / "case",
        [("chain-001", "hard_cut"), ("chain-001", "continue")],
        include_sampling=False,
    )
    _write_json(
        root / "long_video_plan.json",
        {
            "schema": "duet.long-video-plan",
            "version": 5,
            "segments": [{
                "index": 1,
                "chain_id": "chain-001",
                "join_mode": "hard_cut",
            }],
        },
    )

    with pytest.raises(ValueError, match="segments do not match"):
        image_eval._segments(root)


def test_fusion_output_count_is_bound_to_each_hard_cut_interval() -> None:
    output = json.dumps({
        "schema": "duet.video-prompt-fusion-output",
        "version": 2,
        "input_sha256": "a" * 64,
        "segments": [{"index": 1, "visual": ["only one"]}],
    }).encode()

    with pytest.raises(ValueError, match="segment is invalid"):
        fusion_eval._validate_fusion_output(
            output,
            input_sha256="a" * 64,
            expected_visual_counts=[2],
        )


@pytest.mark.parametrize("module", [image_eval, fusion_eval])
def test_deepseek_runner_requires_explicit_absolute_credential(
    module,
) -> None:
    kwargs = {"timeout_s": 30, "credential_file": None}
    if module is image_eval:
        kwargs["concurrency"] = 1
    with pytest.raises(ValueError, match="explicit credential"):
        module._build_runner("deepseek", **kwargs)

    kwargs["credential_file"] = Path("relative-secret.env")
    with pytest.raises(ValueError, match="absolute"):
        module._build_runner("deepseek", **kwargs)


def test_image_run_case_uses_deepseek_and_forwards_replacement_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _image_source(
        tmp_path / "case",
        [("chain-001", "hard_cut"), ("chain-002", "hard_cut")],
        include_sampling=False,
    )
    element_index = tmp_path / "element-index.json"
    skill = tmp_path / "SKILL.md"
    credential = tmp_path / "deepseek.env"
    reference = tmp_path / "replacement.png"
    evidence = tmp_path / "evidence"
    element_index.write_text("{}\n", encoding="utf-8")
    skill.write_text("# frozen skill\n", encoding="utf-8")
    secret = "sk-super-secret-never-copy"
    credential.write_text(f"DEEPSEEK_API_KEY={secret}\n", encoding="utf-8")
    reference.write_bytes(b"replacement-image")
    captured: dict[str, object] = {}
    fake_runner = object()

    def deepseek_factory(**kwargs):
        captured["runner_kwargs"] = kwargs
        return fake_runner

    def generate(runner, specs, edit_mode, **kwargs):
        captured.update(
            runner=runner,
            specs=specs,
            edit_mode=edit_mode,
            generate_kwargs=kwargs,
        )
        return {"version": 4}, {1: {"01.png": "prompt"}}

    monkeypatch.setattr(image_eval, "DeepSeekRunner", deepseek_factory)
    monkeypatch.setattr(
        image_eval.image_optimization, "generate_project_prompts", generate
    )
    monkeypatch.setattr(
        image_eval,
        "evaluate",
        lambda *_args, **_kwargs: {
            "descriptive_mean": 4.0,
            "scores": {},
            "output": {"sha256": "a" * 64},
        },
    )

    report = image_eval.run_case(
        root,
        element_index.resolve(),
        skill.resolve(),
        evidence.resolve(),
        timeout_s=45,
        credential_file=credential.resolve(),
        user_reference_image=reference.resolve(),
        user_replacement_prompt="Replace the product only",
    )

    assert captured["runner_kwargs"] == {
        "timeout_s": 45,
        "concurrency": 2,
        "credential_file": credential.resolve(),
    }
    assert captured["runner"] is fake_runner
    assert captured["edit_mode"] == "independent_parallel"
    generate_kwargs = captured["generate_kwargs"]
    assert generate_kwargs["generation_config"] == {
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    assert generate_kwargs["user_reference_image_path"] == reference.resolve()
    assert generate_kwargs["user_replacement_prompt"] == "Replace the product only"
    assert report["experiment"]["skill_sha256"] == hashlib.sha256(
        skill.read_bytes()
    ).hexdigest()
    assert all(
        secret.encode() not in path.read_bytes()
        for path in evidence.rglob("*")
        if path.is_file()
    )


def test_fusion_real_run_uses_deepseek_without_copying_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    input_path, input_data = _fusion_input(source_root)
    skill = tmp_path / "fusion-SKILL.md"
    credential = tmp_path / "deepseek.env"
    evidence = tmp_path / "evidence"
    skill.write_text("# frozen fusion skill\n", encoding="utf-8")
    secret = "sk-another-secret-never-copy"
    credential.write_text(f"DEEPSEEK_API_KEY={secret}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeRunner:
        def run_isolated_until_output(
            self,
            workdir: Path,
            prompt: str,
            *,
            session_dir: Path,
            output_path: Path,
            max_output_bytes: int,
            validate_output,
            output_schema,
        ):
            captured.update(
                stage=workdir,
                prompt=prompt,
                session_dir=session_dir,
                max_output_bytes=max_output_bytes,
                output_schema=output_schema,
            )
            frozen = json.loads(
                (workdir / "work" / "multimodal_input.json").read_text(
                    encoding="utf-8"
                )
            )
            for segment in frozen["segments"]:
                for frame in segment["new_keyframes"]:
                    assert hashlib.sha256(
                        (workdir / frame["path"]).read_bytes()
                    ).hexdigest() == frame["sha256"]
            data = json.dumps(
                {
                    "schema": "duet.video-prompt-fusion-output",
                    "version": 2,
                    "input_sha256": hashlib.sha256(input_data).hexdigest(),
                    "segments": [{
                        "index": segment["index"],
                        "visual": ["visual"] * (
                            1 + sum(
                                frame["transition"]["type"] == "hard_cut"
                                for frame in segment["new_keyframes"][1:]
                            )
                        ),
                    } for segment in frozen["segments"]],
                },
                separators=(",", ":"),
            ).encode()
            output_path.write_bytes(data)
            return validate_output(data)

    def deepseek_factory(**kwargs):
        captured["runner_kwargs"] = kwargs
        return FakeRunner()

    monkeypatch.setattr(fusion_eval, "DeepSeekRunner", deepseek_factory)

    artifact = fusion_eval.run_real_fusion(
        input_path.resolve(),
        skill.resolve(),
        evidence.resolve(),
        timeout_s=60,
        credential_file=credential.resolve(),
    )

    assert captured["runner_kwargs"] == {
        "timeout_s": 60,
        "concurrency": 1,
        "credential_file": credential.resolve(),
    }
    assert (evidence / "frozen_input.json").read_bytes() == input_data
    assert (evidence / "skill.md").read_bytes() == skill.read_bytes()
    assert (evidence / "artifact_sha256.txt").read_text(
        encoding="ascii"
    ).strip() == hashlib.sha256(artifact.read_bytes()).hexdigest()
    inventory = fusion_eval.build_inventory(input_path, artifact)
    assert inventory["segments"][0]["expected_visual_count"] == 1
    assert inventory["segments"][0]["actual_visual_count"] == 1
    assert not Path(captured["stage"]).exists()
    assert all(
        secret.encode() not in path.read_bytes()
        for path in evidence.rglob("*")
        if path.is_file()
    )


def test_codex_remains_an_explicit_runner_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def codex_factory(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(image_eval, "CodexRunner", codex_factory)
    monkeypatch.setattr(fusion_eval, "CodexRunner", codex_factory)

    image_eval._build_runner(
        "codex", timeout_s=12, concurrency=3, credential_file=None
    )
    fusion_eval._build_runner(
        "codex", timeout_s=13, credential_file=None
    )

    assert captured == [
        {"timeout_s": 12, "concurrency": 3},
        {"timeout_s": 13, "concurrency": 1},
    ]


@pytest.mark.parametrize("module", [image_eval, fusion_eval])
def test_eval_cli_path_parser_rejects_relative_paths(module) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="path must be absolute"):
        module._absolute_path("relative/input.json")


def test_image_cli_defaults_to_deepseek_and_forwards_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def run_case(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {"descriptive_mean": 4.0, "scores": {}, "output": {}, "experiment": {}}

    monkeypatch.setattr(image_eval, "run_case", run_case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-real-image-eval",
            "--source-root", str(tmp_path / "source"),
            "--element-index", str(tmp_path / "element.json"),
            "--skill", str(tmp_path / "SKILL.md"),
            "--evidence-dir", str(tmp_path / "evidence"),
            "--credential-file", str(tmp_path / "deepseek.env"),
            "--user-reference-image", str(tmp_path / "reference.png"),
            "--user-replacement-prompt", "Replace only the product",
        ],
    )

    image_eval.main()

    assert captured["kwargs"]["runner_name"] == "deepseek"
    assert captured["kwargs"]["credential_file"] == tmp_path / "deepseek.env"
    assert captured["kwargs"]["user_reference_image"] == tmp_path / "reference.png"
    assert captured["kwargs"]["user_replacement_prompt"] == "Replace only the product"
    assert "secret" not in capsys.readouterr().out.lower()


def test_fusion_cli_defaults_to_deepseek_for_skill_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "evidence" / "h3_prompt_plan.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "report.json"
    captured: dict[str, object] = {}

    def run_real_fusion(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return artifact

    monkeypatch.setattr(fusion_eval, "run_real_fusion", run_real_fusion)
    monkeypatch.setattr(
        fusion_eval,
        "build_inventory",
        lambda input_path, artifact_path: {
            "input": str(input_path),
            "artifact": str(artifact_path),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fusion-eval",
            "--input", str(tmp_path / "input.json"),
            "--skill", str(tmp_path / "SKILL.md"),
            "--evidence-dir", str(tmp_path / "evidence"),
            "--report", str(report),
            "--credential-file", str(tmp_path / "deepseek.env"),
        ],
    )

    fusion_eval.main()

    assert captured["kwargs"]["runner_name"] == "deepseek"
    assert captured["kwargs"]["credential_file"] == tmp_path / "deepseek.env"
    assert json.loads(report.read_text(encoding="utf-8"))["artifact"] == str(
        artifact
    )
