"""h3-context-ir 节点（MiniMax 官方 Context IR 提示词优化）单测。

契约依据 duet-video-v2 生产实测（docs/agent/reference/minimax-context-ir.md）：
素材走 /v1/files/upload → mm_file:// 引用；查询端点返回账户最近任务列表、
不按 task_id 过滤，客户端按 items[].id 自取。覆盖：config 解析校验、上传
（purpose/尺寸校验/file_id 提取）、submit 契约与 outcome 分类、poll 列表语义、
节点素材顺序（Image/Video/Audio 编号映射）、无台词段无 Audio、duration 取整、
幂等跳过、fail-open 兜底、确定性 4xx fail-closed、vendor failed 换任务重提、
1026 审核映射。
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from app.config import MINIMAX_CONTEXT_IR_API_BASE, Settings
from app.pipeline import (
    CONTEXT_IR_INPUT_MAX_CHARS,
    CONTEXT_IR_MAX_ATTEMPTS,
    ContextIRInputTooLongError,
    ContextIRSanitizeError,
    Pipeline,
    _context_ir_input_text,
    _extract_usage_paragraphs,
)
from app.providers import (
    ContextIRModerationError,
    ContextIRTaskFailedError,
    MiniMaxContextIRClient,
    ProviderFailure,
    ProviderResult,
)
from app.research_log import ResearchLog
from app.store import ProjectStore


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://context-ir.test",
        "minimax_context_ir_enabled": True,
        "minimax_context_ir_api_key": "sk-test",
        "minimax_context_ir_api_base": "https://minimax.test",
        "request_timeout_s": 1.0,
        "provider_poll_interval_s": 0.0,
        "provider_poll_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def _client(tmp_path: Path, handler, **overrides: object) -> MiniMaxContextIRClient:
    client = MiniMaxContextIRClient(
        _settings(tmp_path, **overrides),
        transport=httpx.MockTransport(handler),
    )
    client.poll_interval_s = 0.0
    client.poll_timeout_s = 5.0
    return client


def _png_bytes(size: tuple[int, int] = (512, 512)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- config

class TestConfig:
    def test_defaults_disabled(self, tmp_path: Path) -> None:
        settings = Settings(
            access_token="t",
            data_dir=tmp_path,
            public_base_url="https://context-ir.test",
        )
        assert settings.minimax_context_ir_enabled is False
        assert settings.minimax_context_ir_api_key is None
        assert settings.minimax_context_ir_api_base == MINIMAX_CONTEXT_IR_API_BASE

    def test_enabled_requires_key(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="MINIMAX_CONTEXT_IR_API_KEY"):
            Settings(
                access_token="t",
                data_dir=tmp_path,
                public_base_url="https://context-ir.test",
                minimax_context_ir_enabled=True,
            )

    def test_key_must_be_header_safe_ascii(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="MINIMAX_CONTEXT_IR_API_KEY"):
            Settings(
                access_token="t",
                data_dir=tmp_path,
                public_base_url="https://context-ir.test",
                minimax_context_ir_api_key="sk-中文",
            )

    def test_base_must_be_https(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="MINIMAX_CONTEXT_IR_API_BASE"):
            Settings(
                access_token="t",
                data_dir=tmp_path,
                public_base_url="https://context-ir.test",
                minimax_context_ir_api_base="http://minimax.test",
            )

    def test_base_trailing_slash_stripped(self, tmp_path: Path) -> None:
        settings = Settings(
            access_token="t",
            data_dir=tmp_path,
            public_base_url="https://context-ir.test",
            minimax_context_ir_api_base="https://minimax.test/",
        )
        assert settings.minimax_context_ir_api_base == "https://minimax.test"


# ---------------------------------------------------------------- upload

class TestUpload:
    def test_upload_returns_mm_file_url(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = request.content
            return httpx.Response(200, json={
                "file": {"file_id": 12345},
                "base_resp": {"status_code": 0},
            })

        image = tmp_path / "tile.png"
        image.write_bytes(_png_bytes())
        result = _client(tmp_path, handler).upload(image, kind="image")
        assert result.value == "mm_file://12345"
        assert seen["path"] == "/v1/files/upload"
        assert b"video_generation_input" in seen["body"]
        assert result.record["file_id"] == "12345"

    def test_image_dimension_validation(self, tmp_path: Path) -> None:
        small = tmp_path / "small.png"
        small.write_bytes(_png_bytes((100, 512)))
        client = _client(tmp_path, lambda request: httpx.Response(200))
        with pytest.raises(ValueError, match="256-5760"):
            client.upload(small, kind="image")

    def test_video_and_audio_skip_dimension_check(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"file": {"file_id": "v1"}})

        video = tmp_path / "muted.mp4"
        video.write_bytes(b"fake-video")
        result = _client(tmp_path, handler).upload(video, kind="video")
        assert result.value == "mm_file://v1"

    def test_upload_4xx_is_rejected(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad file"})

        video = tmp_path / "muted.mp4"
        video.write_bytes(b"fake-video")
        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).upload(video, kind="video")
        assert caught.value.record["outcome"] == "rejected"

    def test_upload_missing_file_id_fails(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"base_resp": {"status_code": 0}})

        video = tmp_path / "muted.mp4"
        video.write_bytes(b"fake-video")
        with pytest.raises(ProviderFailure):
            _client(tmp_path, handler).upload(video, kind="video")


# ---------------------------------------------------------------- submit

def _submit_kwargs() -> dict[str, Any]:
    return {
        "prompt": "the final prompt",
        "image_urls": ["mm_file://img1", "mm_file://img2"],
        "video_url": "mm_file://vid1",
        "audio_url": "mm_file://aud1",
        "duration": 9,
        "ratio": "9:16",
    }


class TestSubmit:
    def test_content_order_and_roles(self, tmp_path: Path) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"task_id": "task-1"})

        result = _client(tmp_path, handler).submit(**_submit_kwargs())
        assert result.value == "task-1"
        assert result.record["outcome"] == "accepted"
        body = seen[0]
        assert body["model"] == "MiniMax-H3"
        assert body["duration"] == 9
        assert body["ratio"] == "9:16"
        assert [element["type"] for element in body["content"]] == [
            "text", "image_url", "image_url", "video_url", "audio_url",
        ]
        assert body["content"][0]["text"] == "the final prompt"
        assert [element.get("role") for element in body["content"][1:]] == [
            "reference_image", "reference_image",
            "reference_video", "reference_audio",
        ]
        assert body["content"][1]["image_url"]["url"] == "mm_file://img1"
        assert body["content"][3]["video_url"]["url"] == "mm_file://vid1"
        assert body["content"][4]["audio_url"]["url"] == "mm_file://aud1"

    def test_no_audio_omits_element(self, tmp_path: Path) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"task_id": "task-1"})

        kwargs = {**_submit_kwargs(), "audio_url": None}
        _client(tmp_path, handler).submit(**kwargs)
        assert [element["type"] for element in seen[0]["content"]] == [
            "text", "image_url", "image_url", "video_url",
        ]

    def test_public_https_urls_refused(self, tmp_path: Path) -> None:
        client = _client(tmp_path, lambda request: httpx.Response(200))
        with pytest.raises(ValueError, match="mm_file"):
            client.submit(**{**_submit_kwargs(), "video_url": "https://x.test/v.mp4"})

    def test_deterministic_400_is_rejected(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "prompt too long"})

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "rejected"

    def test_402_insufficient_balance_is_rejected(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(402, json={
                "type": "error",
                "error": {"type": "insufficient_balance_error",
                          "message": "balance", "http_code": "402"},
            })

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "rejected"

    def test_moderation_1026_mapped(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={
                "error": {"code": "1026",
                          "message": "input new_sensitive, input text sensitive"},
            })

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).submit(**_submit_kwargs())
        assert isinstance(caught.value.cause, ContextIRModerationError)
        assert "内容审核" in str(caught.value.cause)
        assert caught.value.record["outcome"] == "rejected"
        assert caught.value.record["moderation"] is True

    def test_5xx_is_unknown(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "unknown"

    def test_200_without_task_id_is_unknown(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "unknown"

    @pytest.mark.parametrize("duration", [True, 0, 3, 16, "9"])
    def test_duration_validation(self, tmp_path: Path, duration: object) -> None:
        client = _client(tmp_path, lambda request: httpx.Response(200))
        with pytest.raises(ValueError, match="duration"):
            client.submit(**{**_submit_kwargs(), "duration": duration})

    def test_ratio_validation(self, tmp_path: Path) -> None:
        client = _client(tmp_path, lambda request: httpx.Response(200))
        with pytest.raises(ValueError, match="ratio"):
            client.submit(**{**_submit_kwargs(), "ratio": "3:2"})


# ---------------------------------------------------------------- poll

def _items_payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"items": list(items)}


class TestPoll:
    def test_succeeded_returns_prompt_and_usage(self, tmp_path: Path) -> None:
        payload = _items_payload(
            {"id": "other-task", "status": "running"},
            {
                "id": "task-1",
                "status": "succeeded",
                "content": {"prompt": "enhanced prompt"},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v2/query/video_generation"
            assert request.url.params["task_id"] == "task-1"
            return httpx.Response(200, json=payload)

        result = _client(tmp_path, handler).poll("task-1")
        assert result.value == {
            "prompt": "enhanced prompt",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    def test_task_absent_from_list_means_queued(self, tmp_path: Path) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                # 目标不在列表 = 仍在排队，绝不取 items[0]
                return httpx.Response(200, json=_items_payload(
                    {"id": "other", "status": "failed"},
                ))
            return httpx.Response(200, json=_items_payload(
                {"id": "task-1", "status": "succeeded",
                 "content": {"prompt": "done"}, "usage": {"total_tokens": 7}},
            ))

        result = _client(tmp_path, handler).poll("task-1")
        assert result.value["prompt"] == "done"
        assert calls["count"] == 2

    def test_unknown_status_is_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_items_payload(
                {"id": "task-1", "status": "mystery"},
            ))

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).poll("task-1")
        assert "未知任务状态" in str(caught.value.cause)

    def test_vendor_failed_raises_typed_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_items_payload(
                {"id": "task-1", "status": "failed",
                 "error": {"code": "5001", "message": "generation failed"}},
            ))

        with pytest.raises(ProviderFailure) as caught:
            _client(tmp_path, handler).poll("task-1")
        assert isinstance(caught.value.cause, ContextIRTaskFailedError)
        assert "5001" in str(caught.value.cause)

    def test_non_list_items_is_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"task": {"id": "task-1"}})

        with pytest.raises(ProviderFailure):
            _client(tmp_path, handler).poll("task-1")

    def test_succeeded_without_prompt_is_failure(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_items_payload(
                {"id": "task-1", "status": "succeeded"},
            ))

        with pytest.raises(ProviderFailure):
            _client(tmp_path, handler).poll("task-1")


# ---------------------------------------------------------------- node

def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_TEST_VISUAL_PROMPT = "本段为竖屏自拍近景。第 1 时点她举起装置。当台词提到【wanita】时她微笑。"


def _node_project(tmp_path: Path, *, dialogue: bool) -> tuple[Pipeline, str, Path, dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        {"output": {"aspect_ratio": "9:16"},
         "dialogue": {"target_language": "马来语"}},
        "context-ir-test",
    )
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    muted = segment_dir / "source-muted.mp4"
    muted.write_bytes(b"muted-video")
    (segment_dir / "h3_prompt.txt").write_text("original prompt", encoding="utf-8")
    if dialogue:
        (segment_dir / "dialogue-track.wav").write_bytes(b"dialogue-audio")
    sheet_dir = project / "work" / "global-reference-sheet"
    sheet_dir.mkdir(parents=True)
    (sheet_dir / "reference-sheet.png").write_bytes(b"sheet")
    (sheet_dir / "user.png").write_bytes(b"user")
    replacement_sheet = {
        "h3_reference_images": [
            {"path": "work/global-reference-sheet/reference-sheet.png",
             "role": "global_replacement_sheet", "image_reference": "Image 1"},
            {"path": "work/global-reference-sheet/user.png",
             "role": "user_replacement", "image_reference": "Image 2"},
        ],
        "spec": {
            "tiles": [
                {"tile_id": "TILE_01", "stable_key": "person-01",
                 "category": "people", "row": 1, "column": 1,
                 "source": "generated_variant"},
                {"tile_id": "TILE_03", "stable_key": "entity-01",
                 "category": "entities", "row": 2, "column": 1,
                 "source": "image_1", "reference_image_ids": ["image_1"]},
            ],
        },
    }
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/unused",
        codex=object(),
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=object(),
        tts=object(),
    )
    item: dict[str, Any] = {
        "index": 1,
        "duration_s": 8.03,
        "visual_prompt": _TEST_VISUAL_PROMPT,
        "h3_mask_profile": {
            "objects_masked": True,
            "person_blocked": True,
            "object_blocked": True,
            "object_semi_transparent": True,
        },
        "dialogue": (
            [{"start_s": 0.0, "end_s": 2.1, "text": "Lelaki boleh."}]
            if dialogue else []
        ),
    }
    element_index: dict[str, Any] = {
        "people": {
            "person-01": {
                "occurrences": [{"segment_index": 1, "frame_orders": [1, 2, 3]}],
                "replacement": {
                    "source": "generated_variant",
                    "appearance_description": (
                        "一位约30岁的马来／东南亚裔男性：方圆脸，眉毛浓黑"
                    ),
                },
            },
        },
        "entities": {
            "entity-01": {
                "occurrences": [{"segment_index": 1, "frame_orders": [1, 2]}],
                "replacement": {"source": "image_1"},
            },
            "entity-keep": {
                "occurrences": [{"segment_index": 1, "frame_orders": [3]}],
                "replacement": None,
            },
        },
    }
    return pipeline, project_id, project, item, replacement_sheet, element_index, muted


def _run_node(
    pipeline: Pipeline,
    project_id: str,
    project: Path,
    item: dict[str, Any],
    replacement_sheet: dict[str, Any],
    element_index: dict[str, Any],
    muted: Path,
) -> dict[str, Any]:
    return pipeline._context_ir_enhance_segment(
        project_id,
        project,
        item,
        muted,
        replacement_sheet,
        element_index,
        {"output": {"aspect_ratio": "9:16"},
         "dialogue": {"target_language": "马来语"},
         "processing": {"remove_subtitle": True, "remove_logo": True}},
        ResearchLog(project),
    )


class _StubContextIR:
    """节点级测试替身：记录调用并按脚本应答，不走 HTTP。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.uploads: list[tuple[str, str]] = []
        self.submits: list[dict[str, Any]] = []
        self.polls: list[str] = []

    def upload(self, path: Path, *, kind: str) -> ProviderResult[str]:
        self.uploads.append((path.name, kind))
        return ProviderResult(f"mm_file://{path.name}", {"file_id": path.name})

    def submit(self, **kwargs: Any) -> Any:
        self.submits.append(kwargs)
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    def poll(self, task_id: str) -> Any:
        self.polls.append(task_id)
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _submit_ok(task_id: str) -> ProviderResult[str]:
    return ProviderResult(task_id, {"outcome": "accepted", "task_id": task_id})


def _poll_ok(prompt: str) -> ProviderResult[dict[str, Any]]:
    return ProviderResult(
        {"prompt": prompt, "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        {"task_id": "t", "polls": [], "usage": {}},
    )


class TestNode:
    def test_success_overwrites_prompt_and_writes_receipt(
        self, tmp_path: Path,
    ) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok("enhanced prompt")])
        pipeline.context_ir = stub

        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)

        assert outcome["status"] == "succeeded"
        segment_dir = muted.parent
        assert (segment_dir / "h3_prompt.txt").read_text().startswith("enhanced prompt")
        assert (segment_dir / "h3_prompt_origin.txt").read_text() == "original prompt"
        receipt = json.loads((segment_dir / "context_ir.json").read_text())
        assert receipt["status"] == "succeeded"
        assert receipt["task_id"] == "task-1"
        assert receipt["origin_prompt_sha256"] == _text_sha256("original prompt")
        assert receipt["optimized_prompt_sha256"] == _text_sha256(
            (segment_dir / "h3_prompt.txt").read_text()
        )
        assert receipt["usage"]["prompt_tokens"] == 1
        assert receipt["distilled_input_sha256"] == _text_sha256(
            stub.submits[0]["prompt"]
        )
        # 上传顺序 = Image 1..N → Video 1 → Audio 1（编号顺序）
        assert stub.uploads == [
            ("reference-sheet.png", "image"),
            ("user.png", "image"),
            ("source-muted.mp4", "video"),
            ("dialogue-track.wav", "audio"),
        ]
        submit = stub.submits[0]
        assert submit["duration"] == 9  # max(5, ceil(8.03))
        assert submit["ratio"] == "9:16"
        # 发给 IR 的是蒸馏文本：含视觉正文逐字与绑定映射，不是冻结 prompt 全文
        assert _TEST_VISUAL_PROMPT in submit["prompt"]
        assert "person-01（一位约30岁的马来／东南亚裔男性） → Image 1 TILE_01" in submit["prompt"]
        assert "entity-01 → Image 2" in submit["prompt"]
        # 分镜归属小节：只含被替换元素、按当前段过滤
        assert "[分镜归属]" in submit["prompt"]
        assert "时点 1：person-01、entity-01" in submit["prompt"]
        assert "时点 3：person-01" in submit["prompt"]  # entity-keep 无 replacement 不进
        assert "Lelaki boleh." in submit["prompt"]
        assert "马来语" in submit["prompt"]
        assert "original prompt" not in submit["prompt"]

    def test_no_dialogue_segment_has_no_audio(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=False,
        )
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok("enhanced")])
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "succeeded"
        assert stub.submits[0]["audio_url"] is None
        assert [kind for _name, kind in stub.uploads] == ["image", "image", "video"]

    def test_disabled_skips_without_http(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        pipeline.settings = _settings(tmp_path, minimax_context_ir_enabled=False)
        stub = _StubContextIR([])
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "disabled"
        assert stub.submits == [] and stub.uploads == []
        assert (muted.parent / "h3_prompt.txt").read_text() == "original prompt"

    def test_idempotent_skip_when_receipt_matches(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        segment_dir = muted.parent
        (segment_dir / "h3_prompt_origin.txt").write_text(
            "original prompt", encoding="utf-8",
        )
        (segment_dir / "h3_prompt.txt").write_text("enhanced prompt", encoding="utf-8")
        (segment_dir / "context_ir.json").write_text(json.dumps({
            "status": "succeeded",
            "origin_prompt_sha256": _text_sha256("original prompt"),
            "optimized_prompt_sha256": _text_sha256("enhanced prompt"),
        }))
        stub = _StubContextIR([])
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "skipped"
        assert stub.submits == [] and stub.uploads == []

    def test_origin_change_invalidates_idempotency(self, tmp_path: Path) -> None:
        """原文变化（regenerate/resume-h3-prep 重建提示词）→ 幂等键失效重跑。"""
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        segment_dir = muted.parent
        (segment_dir / "h3_prompt_origin.txt").write_text(
            "rebuilt original", encoding="utf-8",  # 原文已变
        )
        (segment_dir / "h3_prompt.txt").write_text("enhanced prompt", encoding="utf-8")
        (segment_dir / "context_ir.json").write_text(json.dumps({
            "status": "succeeded",
            "origin_prompt_sha256": _text_sha256("original prompt"),  # 旧原文
            "optimized_prompt_sha256": _text_sha256("enhanced prompt"),
        }))
        stub = _StubContextIR([_submit_ok("task-2"), _poll_ok("re-enhanced")])
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "succeeded"
        assert len(stub.submits) == 1  # 重跑了

    def test_origin_backup_not_overwritten_on_second_run(
        self, tmp_path: Path,
    ) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok("enhanced prompt")])
        pipeline.context_ir = stub
        _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        # 第二次运行（resume 场景）：失效重冻会删掉 context_ir.json（幂等
        # 指纹随之消失），源 prompt 必须取 origin 备份而非已覆写文本。
        (muted.parent / "context_ir.json").unlink()
        stub2 = _StubContextIR([_submit_ok("task-2"), _poll_ok("re-enhanced")])
        pipeline.context_ir = stub2
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "succeeded"
        assert _TEST_VISUAL_PROMPT in stub2.submits[0]["prompt"]
        assert (muted.parent / "h3_prompt_origin.txt").read_text() == "original prompt"
        assert (muted.parent / "h3_prompt.txt").read_text().startswith("re-enhanced")

    def test_transient_failures_fall_back_to_original(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        failure = ProviderFailure(
            {"outcome": "unknown"}, [], ConnectionError("boom"),
        )
        stub = _StubContextIR([failure] * CONTEXT_IR_MAX_ATTEMPTS)
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "fallback_original"
        assert len(stub.submits) == CONTEXT_IR_MAX_ATTEMPTS
        assert (muted.parent / "h3_prompt.txt").read_text() == "original prompt"
        receipt = json.loads((muted.parent / "context_ir.json").read_text())
        assert receipt["status"] == "fallback_original"
        assert "boom" in receipt["reason"]

    def test_fallback_restores_overwritten_prompt(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        segment_dir = muted.parent
        # 半成品现场：上一轮已覆写 prompt 但未写回执（如进程崩溃）。
        (segment_dir / "h3_prompt_origin.txt").write_text("original prompt", encoding="utf-8")
        (segment_dir / "h3_prompt.txt").write_text("partially enhanced", encoding="utf-8")
        failure = ProviderFailure(
            {"outcome": "unknown"}, [], ConnectionError("boom"),
        )
        stub = _StubContextIR([failure] * CONTEXT_IR_MAX_ATTEMPTS)
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "fallback_original"
        assert (segment_dir / "h3_prompt.txt").read_text() == "original prompt"

    def test_deterministic_400_fails_closed(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        failure = ProviderFailure(
            {"outcome": "rejected",
             "response": {"status_code": 400}},
            [], ValueError("prompt too long"),
        )
        stub = _StubContextIR([failure])
        pipeline.context_ir = stub
        with pytest.raises(ProviderFailure):
            _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert len(stub.submits) == 1  # 不重试
        assert not (muted.parent / "context_ir.json").exists()

    def test_moderation_fails_closed_with_clear_message(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        failure = ProviderFailure(
            {"outcome": "rejected", "moderation": True,
             "response": {"status_code": 400}},
            [], ContextIRModerationError("提示词或素材未通过 MiniMax 内容审核"),
        )
        stub = _StubContextIR([failure])
        pipeline.context_ir = stub
        with pytest.raises(ProviderFailure, match="内容审核"):
            _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert len(stub.submits) == 1

    def test_vendor_failed_resubmits_with_fresh_task(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        vendor_failed = ProviderFailure(
            {"task_id": "task-1", "polls": []}, [],
            ContextIRTaskFailedError("task failed"),
        )
        stub = _StubContextIR([
            _submit_ok("task-1"), vendor_failed,
            _submit_ok("task-2"), _poll_ok("enhanced"),
        ])
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "succeeded"
        assert len(stub.submits) == 2
        assert (muted.parent / "h3_prompt.txt").read_text().startswith("enhanced")

    def test_poll_transport_error_keeps_task_and_retries(
        self, tmp_path: Path,
    ) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        transport_error = ProviderFailure(
            {"task_id": "task-1", "polls": []}, [], TimeoutError("poll timeout"),
        )
        stub = _StubContextIR([
            _submit_ok("task-1"), transport_error, _poll_ok("enhanced"),
        ])
        pipeline.context_ir = stub
        outcome = _run_node(pipeline, project_id, project, item, sheet, element_index, muted)
        assert outcome["status"] == "succeeded"
        assert len(stub.submits) == 1  # 已建任务只轮询不重 POST、不重传素材
        assert len(stub.uploads) == 4  # 上传只在首次尝试发生
        assert stub.polls == ["task-1", "task-1"]


# ---------------------------------------------------------------- distilled input builder

def _builder_sheet() -> dict[str, Any]:
    return {
        "h3_reference_images": [
            {"role": "global_replacement_sheet", "image_reference": "Image 1"},
            {"role": "user_replacement", "image_reference": "Image 2"},
            {"role": "user_replacement", "image_reference": "Image 3"},
        ],
        "spec": {
            "tiles": [
                {"tile_id": "TILE_01", "stable_key": "person-01",
                 "category": "people", "row": 1, "column": 1,
                 "source": "generated_variant"},
                {"tile_id": "TILE_02", "stable_key": "person-02",
                 "category": "people", "row": 1, "column": 2,
                 "source": "generated_variant"},
                {"tile_id": "TILE_03", "stable_key": "entity-01",
                 "category": "entities", "row": 2, "column": 1,
                 "source": "image_1",
                 "reference_image_ids": ["image_1", "image_2"]},
            ],
        },
    }


def _builder_index() -> dict[str, Any]:
    return {
        "people": {
            "person-01": {
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1, 2]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replacement": {
                    "source": "generated_variant",
                    "appearance_description": "一位约30岁的马来／东南亚裔男性：方圆脸",
                },
            },
            "person-02": {
                "occurrences": [{"segment_index": 1, "frame_orders": [3]}],
                "replacement": {
                    "source": "generated_variant",
                    "appearance_description": "一位约27岁的马来／东南亚裔女性：鹅蛋脸",
                },
            },
        },
        "entities": {
            "entity-01": {
                "occurrences": [{"segment_index": 1, "frame_orders": [1, 3]}],
                "replacement": {"source": "image_1"},
            },
        },
    }


class TestDistilledInput:
    def test_visual_prompt_verbatim_and_sections(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[{"start_s": 0.0, "end_s": 2.1, "text": "Line A."}],
            target_language="马来语",
            replacement_sheet=_builder_sheet(),
            element_index=_builder_index(),
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        # 视觉正文逐字包含
        assert _TEST_VISUAL_PROMPT in text
        # 小节结构
        for section in ("[任务]", "[视觉正文]", "[素材]", "[替换绑定]",
                        "[台词]", "[保留与禁令]"):
            assert section in text
        # 绑定映射：图版 tile 指向 Image 1，用户图 image_i → Image i+1
        assert "person-01（一位约30岁的马来／东南亚裔男性） → Image 1 TILE_01（图版 r1c1）" in text
        assert "entity-01 → Image 2 / Image 3（用户参考图）" in text
        # 台词逐字 + 语种 + 对口型机械要求
        assert "[0.000-2.100s] Line A." in text
        assert "马来语" in text
        assert "Audio 1" in text and "对口型" in text
        # 保留规则要点
        assert "机位" in text and "身份" in text
        # 目标长度
        assert len(text) <= 4000

    def test_no_dialogue_omits_audio(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=_builder_index(),
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "Audio 1：" not in text
        assert "无台词" in text
        assert "对口型" not in text

    def test_hard_limit_fails_closed(self) -> None:
        with pytest.raises(ContextIRInputTooLongError):
            _context_ir_input_text(
                visual_prompt="长" * (CONTEXT_IR_INPUT_MAX_CHARS + 1),
                dialogue_lines=[],
                target_language=None,
                replacement_sheet=_builder_sheet(),
                element_index=_builder_index(),
                segment_index=1,
                usage_paragraphs=[],
                voice_over=False,
            )

    def test_empty_visual_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="visual_prompt"):
            _context_ir_input_text(
                visual_prompt="  ",
                dialogue_lines=[],
                target_language=None,
                replacement_sheet=_builder_sheet(),
                element_index=_builder_index(),
                segment_index=1,
                usage_paragraphs=[],
                voice_over=False,
            )


class TestDistilledAttribution:
    """缺陷 1 修复：绑定行身份锚 + [分镜归属] 小节。"""

    def test_identity_anchor_and_attribution_section(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=_builder_index(),
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "[分镜归属]（时点编号与视觉正文一致）" in text
        # 段1：person-01 时点 1、2；person-02 时点 3；entity-01 时点 1、3
        assert "- 时点 1：person-01、entity-01" in text
        assert "- 时点 2：person-01" in text
        assert "- 时点 3：person-02、entity-01" in text
        # 身份锚只取 object-replace 输出前缀，不编造
        assert "person-01（一位约30岁的马来／东南亚裔男性）" in text
        assert "person-02（一位约27岁的马来／东南亚裔女性）" in text

    def test_attribution_filtered_by_segment(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=_builder_index(),
            segment_index=2,
            usage_paragraphs=[],
            voice_over=False,
        )
        # 段2 只有 person-01 时点 1
        assert "[分镜归属]" in text
        assert "- 时点 1：person-01" in text
        assert "时点 2" not in text and "时点 3" not in text

    def test_anchor_absent_falls_back_to_plain_key(self) -> None:
        index = _builder_index()
        index["people"]["person-01"]["replacement"]["appearance_description"] = ""
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=index,
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "- person-01 → Image 1 TILE_01" in text


class TestSanitizeOutput:
    """缺陷 2 修复：<d> 内 <scenetrans> 机械清洗。"""

    def test_scenetrans_stripped_words_untouched(self) -> None:
        from app.pipeline import _context_ir_sanitize_output

        raw = ('Shot 3: <d>[Malay] Tekan perlahan<scenetrans></d> cut. '
               '<d>[Malay] <scenetrans>dan tahan</d>')
        cleaned, removed = _context_ir_sanitize_output(raw)
        assert removed == 2
        assert "<scenetrans>" not in cleaned
        assert "<d>[Malay] Tekan perlahan</d>" in cleaned
        assert "<d>[Malay] dan tahan</d>" in cleaned

    def test_scenetrans_outside_dialogue_untouched(self) -> None:
        from app.pipeline import _context_ir_sanitize_output

        raw = "cut marker <scenetrans> here. <d>[Malay] Bersih.</d>"
        cleaned, removed = _context_ir_sanitize_output(raw)
        assert removed == 0
        assert cleaned == raw

    def test_empty_dialogue_after_clean_raises(self) -> None:
        from app.pipeline import _context_ir_sanitize_output

        with pytest.raises(ContextIRSanitizeError):
            _context_ir_sanitize_output("<d>[Malay] <scenetrans></d>")

    def test_clean_text_passthrough(self) -> None:
        from app.pipeline import _context_ir_sanitize_output

        raw = "<d>[Malay] Cuba sendiri.</d>"
        cleaned, removed = _context_ir_sanitize_output(raw)
        assert cleaned == raw and removed == 0


class TestNodeSanitization:
    def test_scenetrans_cleaned_and_audited_in_receipt(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        raw = ("intro <d>[Malay] Tekan<scenetrans></d> mid "
               "<d>[Malay] <scenetrans>tahan.</d> end")
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok(raw)])
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "succeeded"
        written = (muted.parent / "h3_prompt.txt").read_text()
        assert "<scenetrans>" not in written
        assert "Tekan" in written and "tahan." in written
        receipt = json.loads((muted.parent / "context_ir.json").read_text())
        assert receipt["sanitized_scenetrans_removed"] == 2
        assert "<scenetrans>" in receipt["raw_enhanced_prompt"]  # 清洗前入审计
        assert receipt["optimized_prompt_sha256"] == _text_sha256(written)

    def test_empty_dialogue_after_clean_falls_back(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        stub = _StubContextIR([
            _submit_ok("task-1"), _poll_ok("<d>[Malay] <scenetrans></d>"),
        ])
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "fallback_original"
        assert (muted.parent / "h3_prompt.txt").read_text() == "original prompt"


# ---------------------------------------------------------------- five fixed blocks

def _five_block_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "visual_prompt": _TEST_VISUAL_PROMPT,
        "dialogue_lines": [{"start_s": 0.0, "end_s": 2.1, "text": "Line A."}],
        "target_language": "马来语",
        "replacement_sheet": _builder_sheet(),
        "element_index": _builder_index(),
        "segment_index": 1,
        "mask_profile": {
            "objects_masked": True,
            "person_blocked": True,
            "object_blocked": True,
            "object_semi_transparent": True,
        },
        "processing": {"remove_subtitle": True, "remove_logo": True},
        "usage_paragraphs": [],
        "voice_over": False,
    }
    kwargs.update(overrides)
    return kwargs


class TestFiveFixedBlocks:
    """固定五块（editing/mask/face/overlay/dialogue）一块不丢的逐块断言。"""

    def test_editing_block(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs())
        assert "[任务]" in text
        assert "替换编辑" in text
        assert "仅替换下方绑定的元素" in text

    def test_mask_block_full_semantics(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs())
        assert "[遮挡块]" in text
        # 纯黑块完全遮盖人物（脸/发/体/服装）
        assert "纯黑 (0,0,0) 块完全遮盖待替换人物" in text
        assert "脸、头发、身体与服装" in text
        # 半透白块 + 透出内容仅作参考
        assert "半透明白 (255,255,255) 块遮盖待替换物品" in text
        assert "仅作位置、动作与构图参考" in text
        # 两条禁令
        assert "禁止复原、增强、锐化或照抄任何块下可见内容" in text
        assert "禁止画出或还原任何块遮盖的内容" in text
        # 物品按参考图重建全语义
        for phrase in ("identity、形状、尺寸、比例与朝向一致", "物理属性",
                       "被持有、佩戴或使用", "改形、改尺寸或改用途"):
            assert phrase in text

    def test_mask_block_only_object_variant(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs(
            mask_profile={
                "objects_masked": True,
                "person_blocked": False,
                "object_blocked": True,
                "object_semi_transparent": False,
            },
        ))
        assert "[遮挡块]" in text
        assert "纯黑" not in text  # 无人物块不提人物
        assert "白色块遮盖待替换物品" in text
        assert "半透明" not in text

    def test_mask_block_absent_when_unmasked(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs(
            mask_profile={"objects_masked": False,
                          "person_blocked": False, "object_blocked": False,
                          "object_semi_transparent": True},
        ))
        assert "[遮挡块]" not in text
        text_none = _context_ir_input_text(**_five_block_kwargs(mask_profile=None))
        assert "[遮挡块]" not in text_none

    def test_face_block_explicit_priority(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs())
        assert "人物替换是强制的且优先于源身份" in text
        assert "唯一权威" in text
        assert "源人物外观的任何保留都是失败" in text

    def test_face_block_absent_without_replaced_person(self) -> None:
        index = _builder_index()
        index["people"] = {}
        text = _context_ir_input_text(**_five_block_kwargs(element_index=index))
        assert "人物替换是强制的且优先于源身份" not in text

    def test_overlay_block(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs())
        assert "[画面文字]" in text
        assert "烧录字幕" in text and "水印与 logo" in text
        assert "全部擦除并自然补全背景" in text
        assert "禁止复现、翻译或重绘任何源画面文字" in text

    def test_overlay_block_respects_flags(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs(
            processing={"remove_subtitle": False, "remove_logo": True},
        ))
        assert "烧录字幕" not in text
        assert "水印与 logo" in text
        text_off = _context_ir_input_text(**_five_block_kwargs(processing={}))
        assert "[画面文字]" not in text_off

    def test_dialogue_block(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs())
        assert "[台词]" in text
        assert "[0.000-2.100s] Line A." in text
        assert "对口型" in text
        assert "不得跨镜拆分" in text

    def test_five_blocks_length_target(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs())
        assert len(text) <= 4500

    def test_node_submit_carries_mask_and_overlay(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok("enhanced")])
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "succeeded"
        prompt = stub.submits[0]["prompt"]
        assert "[遮挡块]" in prompt
        assert "半透明白 (255,255,255) 块遮盖待替换物品" in prompt
        assert "[画面文字]" in prompt
        assert "人物替换是强制的且优先于源身份" in prompt


# ---------------------------------------------------------------- appended backend constraints

class TestAppendBackendConstraints:
    """第六轮：IR 不吸收的约束由节点机械追加（五块不丢的必然推论）。"""

    def test_mask_through_appended_only_for_semi_transparent_masked(self) -> None:
        from app.pipeline import _context_ir_append_backend_constraints

        text, appended = _context_ir_append_backend_constraints(
            "body",
            mask_profile={"objects_masked": True, "object_blocked": True,
                          "person_blocked": True, "object_semi_transparent": True},
            processing={},
            usage_paragraphs=[],
        )
        assert len(appended) == 1
        assert "serves ONLY as position, motion, and composition reference" in text
        assert "never reconstruct, enhance, sharpen, or copy" in text
        assert "# Backend-mandated constraints (not absorbed by Context IR):" in text

        # 非 masked / 非半透 / 无物品块：不追加
        for profile in (
            {"objects_masked": False, "object_blocked": True,
             "object_semi_transparent": True},
            {"objects_masked": True, "object_blocked": True,
             "object_semi_transparent": False},
            {"objects_masked": True, "object_blocked": False,
             "object_semi_transparent": True},
            None,
        ):
            _t, appended_off = _context_ir_append_backend_constraints(
                "body", mask_profile=profile, processing={},
                usage_paragraphs=[],
            )
            assert appended_off == []

    def test_overlay_appended_by_flags_regardless_of_dialogue(self) -> None:
        from app.pipeline import _context_ir_append_backend_constraints

        text, appended = _context_ir_append_backend_constraints(
            "body",
            mask_profile=None,
            processing={"remove_subtitle": True, "remove_logo": True},
            usage_paragraphs=[],
        )
        assert len(appended) == 1
        assert "burned-in subtitles, captions, and on-screen text" in text
        assert "watermarks and logos" in text
        assert "Remove all of them from the generated video" in text
        assert "Do not reproduce, translate, or re-render" in text

        _t, only_logo = _context_ir_append_backend_constraints(
            "body", mask_profile=None,
            processing={"remove_subtitle": False, "remove_logo": True},
            usage_paragraphs=[],
        )
        assert len(only_logo) == 1
        assert "burned-in subtitles" not in only_logo[0]

        _t2, off = _context_ir_append_backend_constraints(
            "body", mask_profile=None, processing={},
            usage_paragraphs=[],
        )
        assert off == [] and _t2 == "body"

    def test_no_constraints_returns_text_untouched(self) -> None:
        from app.pipeline import _context_ir_append_backend_constraints

        text, appended = _context_ir_append_backend_constraints(
            "body", mask_profile=None, processing=None,
            usage_paragraphs=[],
        )
        assert text == "body" and appended == []

    def test_node_combines_sanitize_and_append(self, tmp_path: Path) -> None:
        pipeline, project_id, project, item, sheet, element_index, muted = _node_project(
            tmp_path, dialogue=True,
        )
        raw = ("intro <d>[Malay] Tekan<scenetrans></d> mid "
               "<d>[Malay] <scenetrans>tahan.</d> end")
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok(raw)])
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "succeeded"
        written = (muted.parent / "h3_prompt.txt").read_text()
        assert "<scenetrans>" not in written
        # 节点 fixture：masked+半透+双 overlay 开关 → 两条都追加
        assert "# Backend-mandated constraints (not absorbed by Context IR):" in written
        assert "serves ONLY as position, motion, and composition reference" in written
        assert "burned-in subtitles, captions, and on-screen text" in written
        # 追加尾部在清洗后文本之后（组合顺序）
        assert written.index("tahan.") < written.index("# Backend-mandated")
        receipt = json.loads((muted.parent / "context_ir.json").read_text())
        assert len(receipt["appended_constraints"]) == 2
        assert "<scenetrans>" in receipt["raw_enhanced_prompt"]
        assert "<scenetrans>" not in receipt["appended_constraints"][0]


# ---------------------------------------------------------------- keep-props section

class TestKeepProps:
    """方案 A：[保留道具] 小节——KEEP 道具结构化存在，防 IR 张冠李戴
    （根因：7d4a3a9b 段3 entity-03 被归并给 <Subject 3> 渲染废片）。"""

    def _index_with_keep(self) -> dict[str, Any]:
        index = _builder_index()
        index["entities"]["entity-03"] = {
            "occurrences": [
                {"segment_index": 3, "frame_orders": [4, 5]},
                {"segment_index": 1, "frame_orders": [2]},
            ],
            "replacement": None,
            "source_visual_description": (
                "白色薄片状睫毛工具：横向长条、带细柄，被演示者捏住横在眼前。"
            ),
        }
        return index

    def test_keep_props_section_present(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=self._index_with_keep(),
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "[保留道具]" in text
        # 简述取 source_visual_description 首句，时点归属本段 frame_orders
        assert "entity-03（白色薄片状睫毛工具）：保留道具，时点 2" in text
        assert "原样保留，不得替换、改形、改色、改名或归并为任何绑定元素/Subject" in text
        # 总禁令
        assert "未列入 [替换绑定] 的物体（含以上保留道具）一律原样保留" in text

    def test_keep_props_filtered_by_segment(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=self._index_with_keep(),
            segment_index=2,  # entity-03 不在段2
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "[保留道具]" not in text

    def test_no_keep_props_no_section(self) -> None:
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=_builder_index(),  # 全部有 replacement
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "[保留道具]" not in text

    def test_missing_description_falls_back_to_key_only(self) -> None:
        index = _builder_index()
        index["entities"]["entity-09"] = {
            "occurrences": [{"segment_index": 1, "frame_orders": [7]}],
            "replacement": None,
        }
        text = _context_ir_input_text(
            visual_prompt=_TEST_VISUAL_PROMPT,
            dialogue_lines=[],
            target_language=None,
            replacement_sheet=_builder_sheet(),
            element_index=index,
            segment_index=1,
            usage_paragraphs=[],
            voice_over=False,
        )
        assert "- entity-09：保留道具，时点 7" in text


# ---------------------------------------------------------------- freeze-before-IR order

def _run_full_flow_with_ir(tmp_path: Path, monkeypatch, ir_stub):
    """完整跑通 run()：context-ir 启用 + stub 客户端；返回运行期对象供断言。"""
    from test_pipeline import (
        _effective_request,
        _FlowDeepSeek,
        _FlowDoubaoTTS,
        _FlowMediaKit,
        _FlowPicsart,
        _FlowSeedream,
        _install_fake_media,
        _settings as _tp_settings,
    )

    store = ProjectStore(tmp_path / "projects")
    request = _effective_request(
        target_language="日语", replacement_instruction="把主角替换成用户参考形象",
    )
    project_id, project, _state = store.create(request, "context-ir-order")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    (project / "inputs" / "replacement-image").write_bytes(b"replacement")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
    )
    _install_fake_media(monkeypatch, [])
    deepseek = _FlowDeepSeek()
    picsart = _FlowPicsart()
    pipeline = Pipeline(
        _tp_settings(
            tmp_path,
            enable_mediakit=True,
            minimax_context_ir_enabled=True,
            minimax_context_ir_api_key="sk-test",
            minimax_context_ir_api_base="https://minimax.test",
        ),
        store,
        lambda _pid, path: f"https://assets.invalid/{path.parent.name}/{path.name}",
        deepseek=deepseek,
        mediakit=_FlowMediaKit(),
        picsart=picsart,
        seedream=_FlowSeedream(deepseek.events),
        tts=_FlowDoubaoTTS(),
        context_ir=ir_stub,
    )
    return store, project_id, project, pipeline, picsart


class TestFreezeBeforeIrOrder:
    """2026-09-30 顺序调整：prepare → 冻结（IR 前原文）→（审核暂停）→
    Context IR（提交前优化层）→ 提交 H3。"""

    def test_freeze_hashes_original_prompt_and_chain_verifies(
        self, tmp_path, monkeypatch,
    ) -> None:
        stub = _StubContextIR([
            _submit_ok("ir-1"), _poll_ok("IR-增强文本-1"),
            _submit_ok("ir-2"), _poll_ok("IR-增强文本-2"),
        ])
        store, project_id, project, pipeline, picsart = _run_full_flow_with_ir(
            tmp_path, monkeypatch, stub,
        )
        pipeline.run(project_id)

        state = store.read(project_id)
        assert state["status"] == "succeeded"
        assert len(stub.submits) == 2  # IR 真跑了两段
        manifest = state["h3_prepared_artifacts"]["segments"]
        for key in ("1", "2"):
            segment_dir = project / "work" / "segments" / f"000{key}"
            origin = (segment_dir / "h3_prompt_origin.txt").read_text(encoding="utf-8")
            current = (segment_dir / "h3_prompt.txt").read_text(encoding="utf-8")
            frozen_sha = manifest[key]["prompt"]["sha256"]
            # 顺序断言：冻结哈希 == IR 前原文哈希（不是 IR 输出）
            assert frozen_sha == _text_sha256(origin)
            assert frozen_sha != _text_sha256(current)
            # 提交侧校验链：origin==冻结 && 回执.origin==冻结 && 当前==optimized
            receipt = json.loads((segment_dir / "context_ir.json").read_text())
            assert receipt["origin_prompt_sha256"] == frozen_sha
            assert receipt["optimized_prompt_sha256"] == _text_sha256(current)
            assert current.startswith("IR-增强文本")
        # 提交给 Picsart 的 prompt 就是覆写后的增强文本（preflight 会 strip）
        submitted_prompts = {call["prompt"] for call in picsart.submits}
        assert submitted_prompts == {
            (project / "work" / "segments" / f"000{key}" / "h3_prompt.txt")
            .read_text(encoding="utf-8").strip()
            for key in ("1", "2")
        }
        # 审核记录是 IR 前文本（editable 含我方原文 body 的开头）
        review = json.loads(
            (project / "work" / "prompt-review.json").read_text(encoding="utf-8")
        )
        for segment in review["segments"]:
            assert "IR-增强文本" not in segment["editable"]
        # manifest 链校验（resume 门禁用的同一函数）必须直接通过
        pipeline._validated_h3_prepared_artifacts(
            project, state, state["segments"],
        )

    def test_resume_picsart_replays_ir_output_via_idempotency(
        self, tmp_path, monkeypatch,
    ) -> None:
        """resume-picsart 的逐字重放靠幂等键复用旧 IR 输出：不重复付费调用，
        原文未变 → IR 不重跑、不重提 Picsart。"""
        stub = _StubContextIR([
            _submit_ok("ir-1"), _poll_ok("IR-增强文本-1"),
            _submit_ok("ir-2"), _poll_ok("IR-增强文本-2"),
        ])
        store, project_id, project, pipeline, picsart = _run_full_flow_with_ir(
            tmp_path, monkeypatch, stub,
        )
        pipeline.run(project_id)
        assert store.read(project_id)["status"] == "succeeded"
        assert len(picsart.submits) == 2

        # 制造 resume 场景：项目 failed，段2 分段成片被清掉需要重建；
        # 段1 成片可复用（该段连 IR 阶段都不应参与）。
        store.update(project_id, status="failed")
        (project / "work" / "segments" / "0002" / "generated.mp4").unlink()
        pipeline.resume_r2v(project_id)

        state = store.read(project_id)
        assert state["status"] == "succeeded"
        # 幂等命中：IR 没有重复付费调用，Picsart 也没有重提
        assert len(stub.submits) == 2
        assert len(picsart.submits) == 2
        # h3_prompt.txt 仍是逐字重放的旧 IR 输出
        for key in ("1", "2"):
            current = (
                project / "work" / "segments" / f"000{key}" / "h3_prompt.txt"
            ).read_text(encoding="utf-8")
            assert current.startswith("IR-增强文本")


class TestFrozenChainHelper:
    def _make_chain(self, segment_dir: Path, *, tamper: bool = False) -> str:
        segment_dir.mkdir(parents=True)
        original = "我方原始提示词"
        enhanced = "IR enhanced text"
        frozen_sha = _text_sha256(original)
        (segment_dir / "h3_prompt_origin.txt").write_text(original, encoding="utf-8")
        (segment_dir / "h3_prompt.txt").write_text(
            enhanced if not tamper else enhanced + "tampered", encoding="utf-8",
        )
        (segment_dir / "context_ir.json").write_text(json.dumps({
            "status": "succeeded",
            "origin_prompt_sha256": frozen_sha,
            "optimized_prompt_sha256": _text_sha256(enhanced),
        }))
        return frozen_sha

    def test_chain_valid(self, tmp_path: Path) -> None:
        from app.pipeline import _context_ir_frozen_chain

        segment_dir = tmp_path / "seg"
        frozen_sha = self._make_chain(segment_dir)
        assert _context_ir_frozen_chain(segment_dir, frozen_sha)

    def test_chain_broken_on_tampered_current(self, tmp_path: Path) -> None:
        from app.pipeline import _context_ir_frozen_chain

        segment_dir = tmp_path / "seg"
        frozen_sha = self._make_chain(segment_dir, tamper=True)
        assert not _context_ir_frozen_chain(segment_dir, frozen_sha)

    def test_chain_broken_without_receipt(self, tmp_path: Path) -> None:
        from app.pipeline import _context_ir_frozen_chain

        segment_dir = tmp_path / "seg"
        frozen_sha = self._make_chain(segment_dir)
        (segment_dir / "context_ir.json").unlink()
        assert not _context_ir_frozen_chain(segment_dir, frozen_sha)


class TestRebuildFrozenPromptsIrChain:
    def test_rebuild_accepts_ir_enhanced_current(self, tmp_path: Path) -> None:
        """regenerate/confirm 的 stale 校验：当前文件是 IR 增强文本时，
        IR 链成立即视为未变化（重建会覆盖并失效旧 IR 产物）。"""
        segment_dir = tmp_path / "work" / "segments" / "0001"
        segment_dir.mkdir(parents=True)
        original = "prefixEDIThiddenSUFFIX"
        enhanced = "IR enhanced"
        (segment_dir / "h3_prompt_origin.txt").write_text(original, encoding="utf-8")
        (segment_dir / "h3_prompt.txt").write_text(enhanced, encoding="utf-8")
        (segment_dir / "context_ir.json").write_text(json.dumps({
            "status": "succeeded",
            "origin_prompt_sha256": _text_sha256(original),
            "optimized_prompt_sha256": _text_sha256(enhanced),
        }))
        frozen = {1: {
            "prefix": "prefix", "editable": "EDIT", "hidden": "hidden",
            "suffix": "SUFFIX", "start_s": 0.0, "end_s": 1.0,
        }}
        rebuilt = Pipeline._rebuild_frozen_prompts(
            tmp_path, frozen, {1: "EDIT-2"},
        )
        assert rebuilt[1] == "prefixEDIT-2hiddenSUFFIX"

    def test_rebuild_rejects_truly_changed_prompt(self, tmp_path: Path) -> None:
        from app.prompt_review import PromptReviewError

        segment_dir = tmp_path / "work" / "segments" / "0001"
        segment_dir.mkdir(parents=True)
        (segment_dir / "h3_prompt.txt").write_text("被篡改", encoding="utf-8")
        frozen = {1: {
            "prefix": "prefix", "editable": "EDIT", "hidden": "hidden",
            "suffix": "SUFFIX", "start_s": 0.0, "end_s": 1.0,
        }}
        with pytest.raises(PromptReviewError):
            Pipeline._rebuild_frozen_prompts(tmp_path, frozen, {1: "X"})


# ---------------------------------------------------------------- usage paragraphs (081e3436 根因修复)

_USAGE_P1 = (
    "【属性与用法】粉色电动加热睫毛夹是有自身结构和使用方式的小电器，"
    "不是一张贴图。前端是可开合的夹头：两片粉色压板对合。"
)
_USAGE_P2 = (
    "【属性与用法】车载湿巾是有自身包装结构的消耗品，不是一张贴图。"
    "抽取口带硬盖，单张抽出。"
)


class TestExtractUsageParagraphs:
    """方案 A/B 共用：从冻结原文确定性提取【属性与用法】段落。"""

    def test_single_paragraph_until_blank_line(self) -> None:
        origin = f"前文块。\n\n{_USAGE_P1}\n\n后文块。"
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1]

    def test_multiple_paragraphs(self) -> None:
        origin = f"{_USAGE_P1}\n\n{_USAGE_P2}\n\n尾块。"
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1, _USAGE_P2]

    def test_no_paragraph_returns_empty(self) -> None:
        assert _extract_usage_paragraphs("【动作复述】没有属性段。\n\n尾块。") == []
        assert _extract_usage_paragraphs("") == []

    def test_paragraph_ends_at_english_block_header(self) -> None:
        origin = (
            f"{_USAGE_P1}\n"
            "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN\n"
            "END AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN"
        )
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1]

    def test_paragraph_ends_at_next_chinese_block_header(self) -> None:
        origin = f"{_USAGE_P1}\n【保持不变的内容】其余不变。"
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1]

    def test_multi_line_paragraph_joined(self) -> None:
        origin = f"{_USAGE_P1}\n续行仍在同一段。\n\n尾块。"
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1 + "\n续行仍在同一段。"]

    def test_blank_line_terminates_paragraph(self) -> None:
        origin = f"{_USAGE_P1}\n\n这行已不属于该段。\n\n尾块。"
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1]

    def test_header_only_marker_line_kept(self) -> None:
        # 只有标记没有内容的行也照实提取（存在判据=包含标记，不另设特例）。
        assert _extract_usage_paragraphs("【属性与用法】\n\n尾块。") == ["【属性与用法】"]
        assert _extract_usage_paragraphs("属性与用法（某商品）：\n\n尾块。") == [
            "属性与用法（某商品）："
        ]

    def test_non_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            _extract_usage_paragraphs(None)  # type: ignore[arg-type]

    def test_parenthesized_header_variant(self) -> None:
        # 081e3436 实测：DeepSeek 输出非确定，段头出现过多变体——
        # 带（产品名）后缀且无方括号不封口（resume 重跑实际产出）。
        variant = _USAGE_P1.replace(
            "【属性与用法】", "属性与用法（粉色电动加热睫毛夹）：", 1,
        )
        assert _extract_usage_paragraphs(variant) == [variant]
        mixed = f"{variant}\n\n{_USAGE_P2}"
        assert _extract_usage_paragraphs(mixed) == [variant, _USAGE_P2]

    def test_prose_mention_is_not_a_paragraph(self) -> None:
        # 行首附近提及"属性与用法"但紧随的不是块装饰符：是正文提及，不是段头。
        origin = f"他说属性与用法很重要，但不是段头。\n\n{_USAGE_P1}"
        assert _extract_usage_paragraphs(origin) == [_USAGE_P1]


class TestRequireFusionUsageParagraphs:
    """fusion 节点硬门禁：段内有绑定用户图的商品时正文必须含【属性与用法】。"""

    def test_missing_paragraph_rejected(self) -> None:
        from app.pipeline import _require_fusion_usage_paragraphs

        with pytest.raises(ValueError, match="属性与用法"):
            _require_fusion_usage_paragraphs(
                "只有动作复述，没有属性段。", user_bound_product=True,
            )

    def test_both_header_variants_accepted(self) -> None:
        from app.pipeline import _require_fusion_usage_paragraphs

        _require_fusion_usage_paragraphs(_USAGE_P1, user_bound_product=True)
        _require_fusion_usage_paragraphs(
            _USAGE_P1.replace(
                "【属性与用法】", "属性与用法（粉色电动加热睫毛夹）：", 1,
            ),
            user_bound_product=True,
        )

    def test_no_user_bound_product_not_gated(self) -> None:
        from app.pipeline import _require_fusion_usage_paragraphs

        _require_fusion_usage_paragraphs(
            "纯换人项目正文。", user_bound_product=False,
        )


class TestUsageSectionDistilledInput:
    """方案 A：蒸馏输入新增 [属性与用法] 小节（[替换绑定] 之后）。"""

    def test_section_present_after_binding(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs(
            usage_paragraphs=[_USAGE_P1],
        ))
        assert "[属性与用法]" in text
        assert _USAGE_P1 in text
        # 位置：[替换绑定] 之后、[分镜归属] 之前
        assert text.index("[替换绑定]") < text.index("[属性与用法]")
        assert text.index("[属性与用法]") < text.index("[分镜归属]")

    def test_multiple_paragraphs_all_included(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs(
            usage_paragraphs=[_USAGE_P1, _USAGE_P2],
        ))
        assert _USAGE_P1 in text and _USAGE_P2 in text

    def test_paragraph_truncated_at_800_chars(self) -> None:
        long_paragraph = "【属性与用法】" + "长" * 1000
        text = _context_ir_input_text(**_five_block_kwargs(
            usage_paragraphs=[long_paragraph],
        ))
        assert long_paragraph not in text
        assert ("【属性与用法】" + "长" * (800 - len("【属性与用法】"))) in text

    def test_empty_list_no_section(self) -> None:
        text = _context_ir_input_text(**_five_block_kwargs(usage_paragraphs=[]))
        assert "[属性与用法]" not in text

    def test_invalid_usage_paragraphs_rejected(self) -> None:
        for bad in (None, "not-a-list", [""], ["  "], [None], [123]):
            with pytest.raises(ValueError):
                _context_ir_input_text(**_five_block_kwargs(usage_paragraphs=bad))

    def test_hard_limit_still_enforced_with_usage(self) -> None:
        with pytest.raises(ContextIRInputTooLongError):
            _context_ir_input_text(**_five_block_kwargs(
                visual_prompt="长" * 6500,
                usage_paragraphs=["【属性与用法】" + "长" * 700],
            ))


class TestUsageAppendBackendConstraints:
    """方案 B：IR 成功路径机械追加【属性与用法】逐字段落 + 形态硬约束。"""

    def test_paragraphs_verbatim_and_form_constraint(self) -> None:
        from app.pipeline import (
            _CONTEXT_IR_APPEND_FORM_AUTHORITY,
            _context_ir_append_backend_constraints,
        )

        text, appended = _context_ir_append_backend_constraints(
            "body",
            mask_profile=None,
            processing={},
            usage_paragraphs=[_USAGE_P1, _USAGE_P2],
        )
        assert _USAGE_P1 in text and _USAGE_P2 in text
        # 段落逐字（不加 bullet 前缀）
        assert f"\n{_USAGE_P1}\n" in text
        assert _CONTEXT_IR_APPEND_FORM_AUTHORITY in text
        assert "唯一权威" in text and "机械钢丝款" in text
        assert "# Backend-mandated constraints (not absorbed by Context IR):" in text
        # 记账：两段 + 一句硬约束
        assert appended == [
            _USAGE_P1, _USAGE_P2, _CONTEXT_IR_APPEND_FORM_AUTHORITY,
        ]

    def test_empty_paragraphs_no_form_constraint(self) -> None:
        from app.pipeline import _context_ir_append_backend_constraints

        text, appended = _context_ir_append_backend_constraints(
            "body", mask_profile=None, processing={}, usage_paragraphs=[],
        )
        assert text == "body" and appended == []

    def test_usage_combined_with_existing_sentences(self) -> None:
        from app.pipeline import (
            _CONTEXT_IR_APPEND_FORM_AUTHORITY,
            _context_ir_append_backend_constraints,
        )

        text, appended = _context_ir_append_backend_constraints(
            "body",
            mask_profile={"objects_masked": True, "object_blocked": True,
                          "person_blocked": True, "object_semi_transparent": True},
            processing={"remove_subtitle": True, "remove_logo": False},
            usage_paragraphs=[_USAGE_P1],
        )
        assert len(appended) == 4  # mask 透出 + overlay + 段落 + 硬约束
        assert appended[2] == _USAGE_P1
        assert appended[3] == _CONTEXT_IR_APPEND_FORM_AUTHORITY
        # 顺序：既有句子（bullet）在前，段落在后
        assert text.index("serves ONLY as position") < text.index(_USAGE_P1)


class TestUsageNodePaths:
    """节点级：成功路径追加 / fallback 与禁用路径不追加。"""

    _ORIGIN_WITH_USAGE = f"original prompt 前文。\n\n{_USAGE_P1}\n\n后文块。"

    def _project_with_usage(self, tmp_path: Path, *, dialogue: bool):
        pipeline, project_id, project, item, sheet, element_index, muted = (
            _node_project(tmp_path, dialogue=dialogue)
        )
        (muted.parent / "h3_prompt.txt").write_text(
            self._ORIGIN_WITH_USAGE, encoding="utf-8",
        )
        return pipeline, project_id, project, item, sheet, element_index, muted

    def test_success_path_appends_usage_and_form_constraint(
        self, tmp_path: Path,
    ) -> None:
        from app.pipeline import _CONTEXT_IR_APPEND_FORM_AUTHORITY

        pipeline, project_id, project, item, sheet, element_index, muted = (
            self._project_with_usage(tmp_path, dialogue=True)
        )
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok("enhanced prompt")])
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "succeeded"
        written = (muted.parent / "h3_prompt.txt").read_text()
        # 最终 prompt 含逐字段落 + 形态硬约束，且在清洗后文本之后
        assert _USAGE_P1 in written
        assert _CONTEXT_IR_APPEND_FORM_AUTHORITY in written
        assert written.index("enhanced prompt") < written.index(_USAGE_P1)
        # 蒸馏输入含 [属性与用法] 小节（逐字段落）
        assert "[属性与用法]" in stub.submits[0]["prompt"]
        assert _USAGE_P1 in stub.submits[0]["prompt"]
        # 回执记账覆盖段落与硬约束
        receipt = json.loads((muted.parent / "context_ir.json").read_text())
        assert _USAGE_P1 in receipt["appended_constraints"]
        assert _CONTEXT_IR_APPEND_FORM_AUTHORITY in receipt["appended_constraints"]

    def test_fallback_path_no_append(self, tmp_path: Path) -> None:
        from app.pipeline import _CONTEXT_IR_APPEND_FORM_AUTHORITY

        pipeline, project_id, project, item, sheet, element_index, muted = (
            self._project_with_usage(tmp_path, dialogue=True)
        )
        failure = ProviderFailure(
            {"outcome": "unknown"}, [], ConnectionError("boom"),
        )
        stub = _StubContextIR([failure] * CONTEXT_IR_MAX_ATTEMPTS)
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "fallback_original"
        written = (muted.parent / "h3_prompt.txt").read_text()
        # 恢复原文：段落只出现一次（原文自带），无机械追加的硬约束
        assert written == self._ORIGIN_WITH_USAGE
        assert written.count(_USAGE_P1) == 1
        assert _CONTEXT_IR_APPEND_FORM_AUTHORITY not in written

    def test_disabled_path_no_append(self, tmp_path: Path) -> None:
        from app.pipeline import _CONTEXT_IR_APPEND_FORM_AUTHORITY

        pipeline, project_id, project, item, sheet, element_index, muted = (
            self._project_with_usage(tmp_path, dialogue=True)
        )
        pipeline.settings = _settings(tmp_path, minimax_context_ir_enabled=False)
        pipeline.context_ir = _StubContextIR([])
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "disabled"
        written = (muted.parent / "h3_prompt.txt").read_text()
        assert written == self._ORIGIN_WITH_USAGE
        assert _CONTEXT_IR_APPEND_FORM_AUTHORITY not in written

    def test_no_usage_paragraphs_no_append(self, tmp_path: Path) -> None:
        """非商品项目（skill 未写属性段）：成功路径也不追加硬约束。"""
        from app.pipeline import _CONTEXT_IR_APPEND_FORM_AUTHORITY

        pipeline, project_id, project, item, sheet, element_index, muted = (
            _node_project(tmp_path, dialogue=True)
        )
        stub = _StubContextIR([_submit_ok("task-1"), _poll_ok("enhanced prompt")])
        pipeline.context_ir = stub
        outcome = _run_node(
            pipeline, project_id, project, item, sheet, element_index, muted,
        )
        assert outcome["status"] == "succeeded"
        written = (muted.parent / "h3_prompt.txt").read_text()
        assert _CONTEXT_IR_APPEND_FORM_AUTHORITY not in written
        assert "[属性与用法]" not in stub.submits[0]["prompt"]
