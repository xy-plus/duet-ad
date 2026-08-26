import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / ".deploy" / "systemd" / "duet-ad1.service"
SMOKE = ROOT / ".deploy" / "smoke-h3.sh"
RUNBOOK = ROOT / ".deploy" / "runbook.md"
LONG_VIDEO_BEHAVIOR = (
    ROOT / "docs" / "human" / "features" / "conversation-task"
    / "behaviors" / "long-video.md"
)


def test_user_service_preserves_codex_bwrap_sandbox_compatibility():
    """Host hardening must not pre-empt the agent's own user/mount/net sandbox."""
    source = UNIT.read_text(encoding="utf-8")
    for required in (
        "NoNewPrivileges=true",
        "LockPersonality=true",
        "RestrictSUIDSGID=true",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "UMask=0077",
    ):
        assert required in source

    for incompatible in (
        "PrivateTmp=",
        "ProtectSystem=",
        "ProtectControlGroups=",
        "ProtectKernelTunables=",
    ):
        assert incompatible not in source


def test_runbook_checks_audio_and_visual_durations_without_full_audio_claim():
    runbook = RUNBOOK.read_text(encoding="utf-8")
    behavior = LONG_VIDEO_BEHAVIOR.read_text(encoding="utf-8")
    for text in (runbook, behavior):
        assert "复用完整源音轨" not in text
        assert "长于画面时裁剪、短于画面时补静音，画面时长不变" in text
    assert "stream=codec_name,channels,duration,duration_ts,time_base" in runbook


@pytest.mark.parametrize(
    ("duration_s", "receipt_version", "plan_receipt", "segment_count", "accepted"),
    [
        (10, 1, None, None, True),
        (15, 1, None, None, True),
        (15, None, None, None, False),
        (10, None, None, None, False),
        (30, None, "b" * 64, 2, True),
        (30, None, "b" * 64, None, False),
    ],
)
def test_paid_smoke_binds_the_matching_short_or_long_receipt(
    tmp_path, duration_s, receipt_version, plan_receipt, segment_count, accepted
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if "--form" in args:
    print(json.dumps({"id": "a" * 32, "status": "queued"}))
elif "--data" in args:
    payload = args[args.index("--data") + 1]
    Path(os.environ["MOCK_PAYLOAD"]).write_text(payload, encoding="utf-8")
    print(json.dumps({"status": "queued"}))
else:
    count_path = Path(os.environ["MOCK_COUNT"])
    count = int(count_path.read_text() or "0") if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    if count == 0:
        print(json.dumps({
            "status": "done",
            "read_only": False,
            "duration_s": int(os.environ["MOCK_DURATION"]),
            "receipt_version": (
                int(os.environ["MOCK_RECEIPT_VERSION"])
                if os.environ.get("MOCK_RECEIPT_VERSION") else None
            ),
            "plan_receipt": os.environ.get("MOCK_PLAN_RECEIPT") or None,
            "segment_count": (
                int(os.environ["MOCK_SEGMENT_COUNT"])
                if os.environ.get("MOCK_SEGMENT_COUNT") else None
            ),
            "fit_required": False,
            "aspect_ratio": "9:16",
            "resolution": "480p",
        }))
    else:
        print(json.dumps({
            "generation": {"status": "succeeded", "attempt": 1},
            "has_video": True,
        }))
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    payload_path = tmp_path / "payload.json"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUN_PAID_SMOKE": "1",
        "ACCESS_TOKEN": "test-only-secret",
        "POLL_INTERVAL_S": "0",
        "MOCK_COUNT": str(tmp_path / "count"),
        "MOCK_PAYLOAD": str(payload_path),
        "MOCK_DURATION": str(duration_s),
        "MOCK_RECEIPT_VERSION": "" if receipt_version is None else str(receipt_version),
        "MOCK_PLAN_RECEIPT": plan_receipt or "",
        "MOCK_SEGMENT_COUNT": "" if segment_count is None else str(segment_count),
    }

    result = subprocess.run(
        ["bash", str(SMOKE), str(video)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert (result.returncode == 0) is accepted, result.stderr
    assert "test-only-secret" not in result.stdout + result.stderr
    if not accepted:
        assert not payload_path.exists()
        return
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["confirm"] is True
    assert payload["dialogue_mode"] == "auto"
    assert payload["fit_mode"] == "none"
    assert payload["aspect_ratio"] == "9:16"
    assert payload["resolution"] == "480p"
    if plan_receipt is None:
        assert "expected_plan_receipt" not in payload
    else:
        assert payload["expected_plan_receipt"] == plan_receipt
    assert "cid=" + "a" * 32 in result.stdout
    assert "client_request_id=" in result.stdout
    assert f"segment_count={segment_count or 1}" in result.stdout
