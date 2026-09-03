from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest

from app import codex_output_schemas, main
from app.codex_runner import CodexError, CodexOutputValidationError
from app.config import Settings
from app.deepseek_runner import (
    MODEL,
    DeepSeekRunner,
    _empty_object_transport_schema,
    _load_api_key,
)


def _credential(tmp_path: Path) -> Path:
    path = tmp_path / "deepseek.env"
    path.write_text("export ANTHROPIC_AUTH_TOKEN='sk-test-credential'\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _envelope(arguments: object) -> dict:
    return {
        "id": "response-test",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call-test",
                    "type": "function",
                    "function": {
                        "name": "submit_result",
                        "arguments": json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":"),
                        ),
                    },
                }],
            },
        }],
    }


def _stage() -> tuple[tempfile.TemporaryDirectory, Path]:
    owner = tempfile.TemporaryDirectory(prefix="duet-deepseek-test-", dir="/tmp")
    stage = Path(owner.name).resolve(strict=True)
    (stage / "work" / "keyframes").mkdir(parents=True)
    (stage / "SKILL.md").write_text("# Frozen skill\nRead request.json.\n", encoding="utf-8")
    (stage / "work" / "request.json").write_text(
        '{"phase":"visual"}\n', encoding="utf-8",
    )
    (stage / "work" / "keyframes" / "01.png").write_bytes(
        b"\x89PNG\r\n\x1a\nexact-image-bytes"
    )
    return owner, stage


def test_credential_loader_accepts_the_existing_shell_file_shape(tmp_path: Path) -> None:
    assert _load_api_key(_credential(tmp_path)) == "sk-test-credential"
    ambiguous = tmp_path / "ambiguous.env"
    ambiguous.write_text(
        "DEEPSEEK_API_KEY=sk-first-value\n"
        "ANTHROPIC_AUTH_TOKEN=sk-second-value\n",
        encoding="utf-8",
    )
    ambiguous.chmod(0o600)
    with pytest.raises(CodexError, match="ambiguous"):
        _load_api_key(ambiguous)
    exposed = tmp_path / "exposed.env"
    exposed.write_text("DEEPSEEK_API_KEY=sk-exposed-value\n", encoding="utf-8")
    exposed.chmod(0o644)
    with pytest.raises(CodexError, match="invalid"):
        _load_api_key(exposed)
    linked = tmp_path / "linked.env"
    linked.symlink_to(_credential(tmp_path))
    with pytest.raises(CodexError, match="invalid"):
        _load_api_key(linked)


def test_visual_uses_one_strict_function_and_publishes_only_validated_json(
    tmp_path: Path,
) -> None:
    observed: dict = {}

    def handle(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer sk-test-credential"
        return httpx.Response(200, json=_envelope({"prompt": "完整自然语言提示词"}))

    owner, stage = _stage()
    try:
        output = stage / "work" / "visual_prompt.json"
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )
        result = runner.run_isolated_until_output(
            stage,
            "严格执行当前目录 SKILL.md",
            session_dir=tmp_path,
            output_path=output,
            max_output_bytes=4096,
            validate_output=lambda raw: codex_output_schemas.normalize_visual_prompt(
                json.loads(raw)
            ),
            output_schema=codex_output_schemas.VISUAL_PROMPT_SCHEMA,
        )
        assert result == "完整自然语言提示词"
        assert json.loads(output.read_text(encoding="utf-8")) == {
            "prompt": "完整自然语言提示词"
        }
    finally:
        owner.cleanup()

    assert observed["model"] == MODEL
    assert observed["thinking"] == {"type": "disabled"}
    assert observed["temperature"] == 0
    assert observed["tool_choice"]["function"]["name"] == "submit_result"
    function = observed["tools"][0]["function"]
    assert function["strict"] is True
    assert function["parameters"] == codex_output_schemas.VISUAL_PROMPT_SCHEMA
    assert "parallel_tool_calls" not in observed
    content = observed["messages"][1]["content"]
    invocation = json.loads(content[0]["text"])
    assert invocation["input_order"] == [
        "SKILL.md", "work/keyframes/01.png", "work/request.json",
    ]
    assert [part["type"] for part in content] == [
        "text", "text", "text", "image_url", "text",
    ]
    assert json.loads(content[1]["text"])["utf8_content"].startswith("# Frozen skill")
    assert content[3]["image_url"]["url"].startswith("data:image/png;base64,")


def test_prompt_fusion_sends_exactly_three_visuals_per_segment(
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    schema = codex_output_schemas.prompt_fusion_schema(
        input_sha256=digest, segment_count=2, visual_max_chars=426,
    )
    visuals = [f"frame {index}" for index in range(1, 4)]
    result = {
        "schema": "duet.video-prompt-fusion-output",
        "version": 3,
        "input_sha256": digest,
        "segments": [
            {"index": 1, "visual": visuals},
            {"index": 2, "visual": visuals},
        ],
    }

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"][0]["function"]["parameters"] == schema
        return httpx.Response(200, json=_envelope(result))

    owner, stage = _stage()
    try:
        output = stage / "work" / "h3_prompt_plan.json"
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )
        produced = runner.run_isolated_until_output(
            stage,
            "严格执行当前目录 SKILL.md",
            session_dir=tmp_path,
            output_path=output,
            max_output_bytes=4096,
            validate_output=lambda raw: json.loads(raw),
            output_schema=schema,
        )
        assert produced == result
        assert json.loads(output.read_text(encoding="utf-8")) == result
    finally:
        owner.cleanup()


def test_global_plan_omits_only_frozen_empty_objects_then_restores_them(
    tmp_path: Path,
) -> None:
    schema = codex_output_schemas.global_plan_schema(stable_keys={
        "people": (),
        "entities": ("entity-01",),
        "scenes": ("scene-01",),
        "relations": (),
    })
    model_result = {
        "entities": {
            "entity-01": {
                "description": "toy",
                "owner": "none",
                "association": "foreground",
                "persistence": "visible",
            }
        },
        "scenes": {
            "scene-01": {
                "source_scene": "room",
                "replacement_scene": "room",
                "semantic_change": "none",
                "geometry_change": "none",
                "depth_change": "none",
                "layout_change": "none",
                "local_color_change": "none",
            }
        },
    }

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        parameters = payload["tools"][0]["function"]["parameters"]
        assert set(parameters["properties"]) == {"entities", "scenes"}
        assert set(parameters["required"]) == {"entities", "scenes"}
        assert payload["parallel_tool_calls"] is False
        return httpx.Response(200, json=_envelope(model_result))

    owner, stage = _stage()
    try:
        output = stage / "work" / "global_plan.json"
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )
        result = runner.run_isolated_until_output(
            stage,
            "global plan",
            session_dir=tmp_path,
            output_path=output,
            max_output_bytes=65536,
            validate_output=lambda raw: codex_output_schemas.normalize_global_plan(
                json.loads(raw),
                stable_keys={
                    "people": (), "entities": ("entity-01",),
                    "scenes": ("scene-01",), "relations": (),
                },
            ),
            output_schema=schema,
        )
        assert result["people"] == {}
        assert result["relations"] == {}
        assert json.loads(output.read_text(encoding="utf-8"))["people"] == {}
    finally:
        owner.cleanup()


def test_parallel_tool_calls_is_not_disabled_for_project_index(tmp_path: Path) -> None:
    observed: dict = {}

    def handle(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope({
            "people": [],
            "entities": [],
            "scenes": [{
                "key": "scene-01",
                "source_visual_description": "room",
                "occurrences": [{"segment_index": 0, "frame_orders": [1]}],
                "replaceable": ["background"],
                "preserve": ["layout"],
            }],
            "relations": [],
        }))

    owner, stage = _stage()
    try:
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )
        runner.run_isolated_until_output(
            stage,
            "project index",
            session_dir=tmp_path,
            output_path=stage / "work" / "index.json",
            max_output_bytes=65536,
            validate_output=lambda raw: codex_output_schemas.normalize_project_index(
                json.loads(raw)
            ),
            output_schema=codex_output_schemas.PROJECT_INDEX_SCHEMA,
        )
    finally:
        owner.cleanup()
    assert "parallel_tool_calls" not in observed


def test_local_validator_failure_never_publishes_business_output(tmp_path: Path) -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope({"prompt": "model value"}))

    owner, stage = _stage()
    try:
        output = stage / "work" / "visual_prompt.json"
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )

        def reject(_raw: bytes):
            raise CodexOutputValidationError("visual_prompt_json_invalid", "/prompt")

        with pytest.raises(CodexError, match="local validation") as caught:
            runner.run_isolated_until_output(
                stage,
                "visual",
                session_dir=tmp_path,
                output_path=output,
                max_output_bytes=4096,
                validate_output=reject,
                output_schema=codex_output_schemas.VISUAL_PROMPT_SCHEMA,
            )
        assert caught.value.retryable is True
        assert not output.exists()
        traces = list((tmp_path / "work" / "errors").glob("*.json"))
        assert len(traces) == 1
        trace_text = traces[0].read_text(encoding="utf-8")
        assert "visual_prompt_json_invalid" in trace_text
        assert "model value" not in trace_text
        assert "sk-test-credential" not in trace_text
    finally:
        owner.cleanup()


def test_symlinked_stage_input_is_rejected_before_http(tmp_path: Path) -> None:
    calls = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_envelope({"prompt": "unused"}))

    owner, stage = _stage()
    try:
        (stage / "work" / "linked.json").symlink_to(stage / "work" / "request.json")
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )
        with pytest.raises(CodexError, match="symbolic link"):
            runner.run_isolated_until_output(
                stage,
                "visual",
                session_dir=tmp_path,
                output_path=stage / "work" / "visual_prompt.json",
                max_output_bytes=4096,
                validate_output=lambda raw: raw,
                output_schema=codex_output_schemas.VISUAL_PROMPT_SCHEMA,
            )
        assert calls == 0
    finally:
        owner.cleanup()


def test_empty_object_adapter_does_not_remove_nonempty_or_array_categories() -> None:
    schema, omitted = _empty_object_transport_schema(
        codex_output_schemas.PROJECT_INDEX_SCHEMA
    )
    assert omitted == ()
    assert schema == codex_output_schemas.PROJECT_INDEX_SCHEMA


def test_oversized_schema_is_rejected_before_transport() -> None:
    with pytest.raises(CodexError, match="schema exceeds transport capacity"):
        _empty_object_transport_schema({
            "type": "object",
            "properties": {
                f"field-{index:05d}": {"type": "string", "description": "x" * 128}
                for index in range(2048)
            },
            "required": [],
            "additionalProperties": False,
        })


def test_legacy_direct_write_interfaces_fail_closed(tmp_path: Path) -> None:
    runner = DeepSeekRunner(
        timeout_s=30,
        concurrency=1,
        credential_file=_credential(tmp_path),
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    with pytest.raises(CodexError, match="schema-constrained"):
        runner.run(tmp_path, "prompt")
    with pytest.raises(CodexError, match="schema-constrained"):
        runner.run_isolated(tmp_path, "prompt", session_dir=tmp_path)


def test_create_app_routes_the_pipeline_to_deepseek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class Runner:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(main, "DeepSeekRunner", Runner)
    credential = _credential(tmp_path)
    main.create_app(Settings(
        access_token="token",
        data_dir=tmp_path,
        codex_timeout_s=321,
        codex_concurrency=7,
        deepseek_credential_file=credential,
    ))
    assert captured == {
        "timeout_s": 321,
        "concurrency": 7,
        "credential_file": credential,
    }
    with pytest.raises(ValueError, match="must be absolute"):
        Settings(
            access_token="token",
            data_dir=tmp_path,
            deepseek_credential_file=Path("relative.env"),
        )


def test_provider_rejection_is_short_and_does_not_persist_raw_body(tmp_path: Path) -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-request-id": "../../unsafe header"},
            json={
                "error": {
                    "code": "RateLimitExceeded",
                    "message": "private provider explanation",
                }
            },
        )

    owner, stage = _stage()
    try:
        runner = DeepSeekRunner(
            timeout_s=30,
            concurrency=1,
            credential_file=_credential(tmp_path),
            transport=httpx.MockTransport(handle),
        )
        with pytest.raises(CodexError, match="HTTP 429: RateLimitExceeded") as caught:
            runner.run_isolated_until_output(
                stage,
                "visual",
                session_dir=tmp_path,
                output_path=stage / "work" / "visual_prompt.json",
                max_output_bytes=4096,
                validate_output=lambda raw: raw,
                output_schema=codex_output_schemas.VISUAL_PROMPT_SCHEMA,
            )
        assert caught.value.retryable is True
        trace = next((tmp_path / "work" / "errors").glob("*.json")).read_text(
            encoding="utf-8"
        )
        assert "RateLimitExceeded" in trace
        assert "private provider explanation" not in trace
        assert "../../unsafe header" not in trace
    finally:
        owner.cleanup()


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_DEEPSEEK_TRANSPORT") != "1",
    reason="explicit opt-in real DeepSeek request",
)
def test_real_deepseek_accepts_three_ordered_images_and_production_visual_schema() -> None:
    source = Path(
        "/home/xy/duet-ad1/data/test-instances/three-skill-preview-3211/data/"
        "f54011007d654e55ad03aeef85fe801f/work/segments/1/work/keyframes/01.png"
    )
    assert source.is_file()
    with tempfile.TemporaryDirectory(prefix="duet-deepseek-real-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        frames = stage / "work" / "keyframes"
        frames.mkdir(parents=True)
        shutil.copyfile(
            "/home/xy/duet-ad1/.worktree/deepseek-codex-transport-r1/"
            "skills/video-maker/SKILL.md",
            stage / "SKILL.md",
        )
        for order in range(1, 4):
            shutil.copyfile(source, frames / f"{order:02d}.png")
        (stage / "work" / "request.json").write_text(
            json.dumps({
                "phase": "visual",
                "frames": [f"work/keyframes/{order:02d}.png" for order in range(1, 4)],
            }, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        result = DeepSeekRunner(
            timeout_s=300,
            concurrency=1,
            credential_file=Path("/home/xy/.config/claude/deepseek.env"),
        ).run_isolated_until_output(
            stage,
            "严格执行当前目录 SKILL.md；观察三张有序关键帧并填写输出 Schema。",
            session_dir=stage,
            output_path=stage / "work" / "visual_prompt.json",
            max_output_bytes=32768,
            validate_output=lambda raw: codex_output_schemas.normalize_visual_prompt(
                json.loads(raw)
            ),
            output_schema=codex_output_schemas.VISUAL_PROMPT_SCHEMA,
        )
        assert isinstance(result, str) and result.strip()


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_DEEPSEEK_TRANSPORT") != "1",
    reason="explicit opt-in real DeepSeek request",
)
def test_real_deepseek_accepts_adapted_global_plan_schema() -> None:
    stable_keys = {
        "people": (),
        "entities": ("entity-01",),
        "scenes": ("scene-01",),
        "relations": (),
    }
    with tempfile.TemporaryDirectory(prefix="duet-deepseek-global-real-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        (stage / "work").mkdir()
        (stage / "SKILL.md").write_text(
            "# Frozen global planner\nFill every requested field with concise text.\n",
            encoding="utf-8",
        )
        (stage / "work" / "request.json").write_text(
            '{"phase":"global_plan"}\n', encoding="utf-8",
        )
        result = DeepSeekRunner(
            timeout_s=300,
            concurrency=1,
            credential_file=Path("/home/xy/.config/claude/deepseek.env"),
        ).run_isolated_until_output(
            stage,
            "Return the complete global plan through submit_result.",
            session_dir=stage,
            output_path=stage / "work" / "global_plan.json",
            max_output_bytes=131072,
            validate_output=lambda value: json.loads(value),
            output_schema=codex_output_schemas.global_plan_schema(
                stable_keys=stable_keys,
            ),
        )
        assert result["people"] == {}
        assert result["relations"] == {}
        assert set(result["entities"]) == {"entity-01"}
        assert set(result["scenes"]) == {"scene-01"}
