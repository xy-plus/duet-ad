"""展示用总耗时（display_duration_s）的计算口径测试。

口径：总耗时 = 前置节点耗时 + 最后一次 H3（seedance）生成耗时。
- 前置：首次提交批次开始前，每个 (node, scope) 最早一次完成的执行时长之和。
- 最后一次生成：最后一个提交批次首个 node_started 到其后最后一个节点终止事件。
审核暂停、两次生成间隔、用户等待（无节点事件）一律不计入；trace 缺失/没有
提交批次时返回 None，详情接口缺省字段、前端回退墙钟。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.research_log import display_duration_seconds

from test_api import settings as api_settings  # noqa: E402
from test_api import successful_factory  # noqa: E402
from test_prompt_review import _create_project  # noqa: E402


def _row(at: datetime, event: str, node: str, scope: str = "project", attempt: int = 1) -> str:
    return json.dumps({
        "at": at.isoformat(),
        "event": event,
        "node": node,
        "scope": scope,
        "attempt": attempt,
    })


def _write_trace(project: Path, rows: list[str]) -> None:
    logs = project / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "trace.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


T0 = datetime(2026, 9, 20, 0, 0, 0, tzinfo=timezone.utc)


def _at(**kwargs) -> datetime:
    return T0 + timedelta(**kwargs)


def _basic_trace() -> list[str]:
    """前置 130s（probe 10s + video-maker 60s + fusion 60s），首批生成 190s。"""
    return [
        _row(_at(seconds=0), "node_started", "probe"),
        _row(_at(seconds=10), "node_succeeded", "probe"),
        _row(_at(seconds=10), "node_started", "video-maker"),
        _row(_at(seconds=70), "node_succeeded", "video-maker"),
        _row(_at(seconds=70), "node_started", "h3-max-r2v-prompt", "segment-0001"),
        _row(_at(seconds=130), "node_succeeded", "h3-max-r2v-prompt", "segment-0001"),
        # 首批 H3：submit 00:02:20 开始
        _row(_at(seconds=140), "node_started", "seedance-submit", "segment-0001"),
        _row(_at(seconds=145), "node_succeeded", "seedance-submit", "segment-0001"),
        _row(_at(seconds=300), "node_started", "stitch"),
        _row(_at(seconds=330), "node_succeeded", "stitch"),
    ]


def test_display_duration_basic_pre_plus_last_generation(tmp_path):
    _write_trace(tmp_path, _basic_trace())
    # pre = 10+60+60 = 130；gen = 00:02:20 → 00:05:30 = 190
    assert display_duration_seconds(tmp_path) == 320.0


def test_display_duration_regenerate_uses_last_batch_and_excludes_gap(tmp_path):
    rows = _basic_trace()
    # 审核暂停/用户等待 2 小时（无任何节点事件），然后重生成批次。
    rows += [
        _row(_at(hours=2), "node_started", "seedance-resume-submit", "segment-0001"),
        _row(_at(hours=2, seconds=10), "node_succeeded", "seedance-resume-submit", "segment-0001"),
        _row(_at(hours=2, minutes=2), "node_started", "stitch-resume"),
        _row(_at(hours=2, minutes=3), "node_succeeded", "stitch-resume"),
    ]
    _write_trace(tmp_path, rows)
    # pre = 130（不变）；gen = 02:00:00 → 02:03:00 = 180；2 小时间隔不计入
    assert display_duration_seconds(tmp_path) == 310.0


def test_display_duration_counts_each_node_once(tmp_path):
    rows = [
        # 第一次执行失败 10s，重试成功 60s：同一 (node, scope) 只计最早完成的。
        _row(_at(seconds=0), "node_started", "video-maker"),
        _row(_at(seconds=10), "node_failed", "video-maker"),
        _row(_at(seconds=10), "node_started", "video-maker", attempt=2),
        _row(_at(seconds=70), "node_succeeded", "video-maker", attempt=2),
        _row(_at(seconds=70), "node_started", "seedance-submit", "segment-0001"),
        _row(_at(seconds=80), "node_succeeded", "seedance-submit", "segment-0001"),
    ]
    _write_trace(tmp_path, rows)
    # pre = 10（失败那次最早完成）；gen = 70→80 = 10
    assert display_duration_seconds(tmp_path) == 20.0


def test_display_duration_fallback_when_uncomputable(tmp_path):
    assert display_duration_seconds(tmp_path) is None  # 没有 trace
    _write_trace(tmp_path, [
        _row(_at(seconds=0), "node_started", "probe"),
        _row(_at(seconds=10), "node_succeeded", "probe"),
        "not json",
        _row(_at(seconds=20), "node_succeeded", "probe"),  # 重复行不影响
    ])
    assert display_duration_seconds(tmp_path) is None  # 没有任何提交批次


def test_conversation_detail_carries_display_duration(tmp_path):
    app = create_app(api_settings(tmp_path), pipeline_factory=successful_factory)
    with TestClient(app) as client:
        store = app.state.store
        project_id = _create_project(app)
        project = store.path(project_id)
        _write_trace(project, _basic_trace())
        store.update(
            project_id,
            status="succeeded",
            has_video=True,
            project_progress={
                "status": "succeeded", "percent": 100, "stage": "completed",
            },
        )
        detail = client.get(
            f"/api/conversations/{project_id}", headers={"Authorization": "Bearer test-secret"},
        ).json()
        assert detail["display_duration_s"] == 320.0

        # 记录不全的老项目：缺省该字段（前端回退墙钟），不报错。
        legacy = _create_project(app)
        store.update(
            legacy,
            status="succeeded",
            has_video=True,
            project_progress={
                "status": "succeeded", "percent": 100, "stage": "completed",
            },
        )
        detail = client.get(
            f"/api/conversations/{legacy}", headers={"Authorization": "Bearer test-secret"},
        ).json()
        assert "display_duration_s" not in detail

        # 非终态项目不携带该字段。
        running = _create_project(app)
        detail = client.get(
            f"/api/conversations/{running}", headers={"Authorization": "Bearer test-secret"},
        ).json()
        assert "display_duration_s" not in detail
