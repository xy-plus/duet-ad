#!/usr/bin/env python3
"""Seedance 2.5 回放 c57d9d6c（两段）——smoke / whitelist / submit / poll / download。

用法（worktree 根目录下）：
  /home/xy/duet-ad1/.venv/bin/python /home/xy/duet-ad1-h3-max-r2v/.worktree/seedance25-r2v/scripts/seedance25_replay_c57d9d6c.py smoke
  ... prepare | whitelist | submit-seg1 | submit-seg2 | status

状态机落盘 /home/xy/duet-ad1-h3-max-r2v/data-verify/seedance25-replay-c57d9d6c-20260929/replay-state.json，
可重复执行：已完成的步骤跳过。回执 JSON 落盘前剥掉 Authorization 头；
三个 key 从 recon duet.env 逐键读取，不打印、不写盘。
本脚本与产物不进 git（data-verify 回放目录已加入主树 .gitignore）。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPLAY = Path(
    "/home/xy/duet-ad1-h3-max-r2v/data-verify/seedance25-replay-c57d9d6c-20260929"
)
PROJECT = Path(
    "/home/xy/duet-ad1-h3-max-r2v/data/projects/c57d9d6c4c924c1a90b7fff35bdc6d64"
)
WORKTREE = Path("/home/xy/duet-ad1-h3-max-r2v/.worktree/seedance25-r2v")
DUET_ENV = Path(
    "/home/xy/duet-ad1-h3-max-r2v/data-verify/seedance25-recon-20260929/duet.env"
)
STATE_PATH = REPLAY / "replay-state.json"
PUBLIC_BASE = "https://8.166.140.227:3261"
MODEL = "dreamina-seedance-2-5-260628"
RATIO = "9:16"
RESOLUTION = "720p"
GROUP_NAME = "seedance25-replay-c57d9d6c"

sys.path.insert(0, str(WORKTREE))

from app.byteplus_assets import BytePlusArkAssetClient  # noqa: E402
from app.config import Settings  # noqa: E402
from app.providers import SeedanceClient  # noqa: E402

_SECRET_KEYS = (
    "SEEDANCE_INTL_API_KEY",
    "BYTEPLUS_ACCESS_KEY_ID",
    "BYTEPLUS_SECRET_ACCESS_KEY",
)


def _load_secrets() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in DUET_ENV.read_text(encoding="utf-8").splitlines():
        for key in _SECRET_KEYS:
            if line.startswith(f"{key}="):
                values[key] = line.split("=", 1)[1].strip()
    missing = [key for key in _SECRET_KEYS if not values.get(key)]
    if missing:
        raise SystemExit(f"missing keys in duet.env: {missing}")
    return values


def _settings(secrets: dict[str, str]) -> Settings:
    return Settings(
        access_token="replay-script",
        data_dir=REPLAY,
        public_base_url=PUBLIC_BASE,
        seedance_api_key=secrets["SEEDANCE_INTL_API_KEY"],
        seedance_proxy="http://127.0.0.1:7897",
        byteplus_access_key_id=secrets["BYTEPLUS_ACCESS_KEY_ID"],
        byteplus_secret_access_key=secrets["BYTEPLUS_SECRET_ACCESS_KEY"],
        seedance_poll_interval_s=30.0,
    )


def _load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _sanitize(value):
    """回执落盘前剥掉 Authorization 头（Bearer key 不写入磁盘）。"""
    if isinstance(value, dict):
        return {
            key: ("***" if key.lower() == "authorization" else _sanitize(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _save_receipt(name: str, record) -> None:
    path = REPLAY / "receipts" / name
    path.write_text(
        json.dumps(_sanitize(record), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )


def _intake_url(filename: str) -> str:
    token = (REPLAY / ".intake-token").read_text(encoding="utf-8").strip()
    return f"{PUBLIC_BASE}/intake/{token}/{filename}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> tuple[int, int]:
    import subprocess

    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    width, height = out.split(",")
    return int(width), int(height)


def _ensure_group(client: BytePlusArkAssetClient, state: dict) -> str:
    group_id = state.get("group_id")
    if group_id:
        return group_id
    group_id = client.create_asset_group(name=GROUP_NAME)
    state["group_id"] = group_id
    _save_state(state)
    return group_id


def _whitelist_one(
    client: BytePlusArkAssetClient,
    state: dict,
    *,
    filename: str,
    media_type: str,
) -> str:
    staged = REPLAY / "intake" / (REPLAY / ".intake-token").read_text().strip() / filename
    digest = _sha256(staged)
    assets = state.setdefault("whitelist", {})
    existing = assets.get(digest)
    if existing and existing.get("status") == "Active":
        print(f"  reuse asset://{existing['asset_id']} ({filename})")
        return f"asset://{existing['asset_id']}"
    group_id = _ensure_group(client, state)
    asset_id = client.create_asset(
        group_id=group_id,
        url=_intake_url(filename),
        name=f"{GROUP_NAME}-{digest[:12]}",
        asset_type=media_type,
    )
    print(f"  CreateAsset {filename} -> {asset_id}, waiting Active...")
    status = client.wait_asset_active(
        asset_id=asset_id,
        expected_asset_type=media_type,
        interval_s=5.0,
        timeout_s=900.0,
    )
    record = {
        "asset_id": asset_id,
        "group_id": group_id,
        "status": status.status,
        "name": filename,
        "media_type": media_type,
        "error_code": status.error_code,
        "error_message": status.error_message,
    }
    assets[digest] = record
    _save_state(state)
    if status.status != "Active":
        raise SystemExit(
            f"whitelist failed for {filename}: {status.error_code} "
            f"{status.error_message}"
        )
    print(f"  Active asset://{asset_id} ({filename})")
    return f"asset://{asset_id}"


# 当年 H3 的图片清单与顺序（logs/picsart-submit input.json 与 state
# h3_reference_images 一致）：图版 @Image 1 + 用户图 @Image 2/3/4。
IMAGE_FILES = ["reference-sheet.png", "user-reference.png", "user-reference-2.png", "user-reference-3.png"]
SEGMENTS = {
    "1": {
        "masked": "seg1-masked.mp4",
        "audio": "seg1-dialogue.wav",
        "prompt_source": PROJECT / "work/segments/0001/h3_prompt.txt",
        "duration": 9,
        "source_seconds": 8.066667,
    },
    "2": {
        "masked": "seg2-masked.mp4",
        "audio": "seg2-dialogue.wav",
        "prompt_source": PROJECT / "work/segments/0002/h3_prompt.txt",
        "duration": 7,
        "source_seconds": 6.033333,
    },
}

_REF_PATTERNS = (
    (re.compile(r"\bVideo 1\b"), "@Video 1"),
    (re.compile(r"\bImage (\d+)\b"), r"@Image \1"),
    (re.compile(r"\bAudio 1\b"), "@Audio 1"),
)


def _adapt_prompt(segment: str) -> str:
    source = SEGMENTS[segment]["prompt_source"].read_text(encoding="utf-8")
    counts_before = {
        label: len(pattern.findall(source)) for pattern, label in (
            (re.compile(r"\bVideo 1\b"), "Video 1"),
            (re.compile(r"\bImage \d+\b"), "Image N"),
            (re.compile(r"\bAudio 1\b"), "Audio 1"),
        )
    }
    adapted = source
    replaced = 0
    for pattern, replacement in _REF_PATTERNS:
        adapted, count = pattern.subn(replacement, adapted)
        replaced += count
    expected = sum(counts_before.values())
    assert replaced == expected, (replaced, expected, counts_before)
    assert " Video 1" not in adapted and " Audio 1" not in adapted
    assert not re.search(r"(?<!@)\bImage \d+\b", adapted)
    destination = REPLAY / f"seg{segment}-prompt-adapted.txt"
    destination.write_text(adapted, encoding="utf-8")
    print(
        f"seg{segment}: adapted prompt {len(adapted)} chars, "
        f"{replaced} refs -> @ form ({counts_before})"
    )
    return adapted


def cmd_smoke() -> None:
    """免费控制面 smoke：小 png 走 CreateAsset 全流程，验证签名与跨境回拉。"""
    state = _load_state()
    if state.get("smoke", {}).get("status") == "Active":
        print("smoke already done:", state["smoke"])
        return
    secrets = _load_secrets()
    settings = _settings(secrets)
    client = BytePlusArkAssetClient(settings)
    group_id = _ensure_group(client, state)
    print("asset group:", group_id)
    asset_id = client.create_asset(
        group_id=group_id,
        url=_intake_url("user-reference-3.png"),
        name=f"{GROUP_NAME}-smoke",
        asset_type="Image",
    )
    print("smoke CreateAsset ->", asset_id, "; polling GetAsset ...")
    status = client.wait_asset_active(
        asset_id=asset_id,
        expected_asset_type="Image",
        interval_s=5.0,
        timeout_s=900.0,
    )
    print("smoke GetAsset final:", status.status, status.error_code, status.error_message)
    state["smoke"] = {
        "asset_id": asset_id,
        "status": status.status,
        "error_code": status.error_code,
        "error_message": status.error_message,
    }
    _save_state(state)
    if status.status != "Active":
        raise SystemExit("smoke FAILED: cross-border pull did not succeed; stop here")
    client.delete_asset(asset_id=asset_id, expected_asset_type="Image")
    print("smoke DeleteAsset done; cross-border pull OK")


def cmd_prepare() -> None:
    for segment in ("1", "2"):
        _adapt_prompt(segment)


def cmd_whitelist() -> None:
    secrets = _load_secrets()
    client = BytePlusArkAssetClient(_settings(secrets))
    state = _load_state()
    print("whitelisting images ...")
    for filename in IMAGE_FILES:
        _whitelist_one(client, state, filename=filename, media_type="Image")
    for segment in ("1", "2"):
        print(f"whitelisting segment {segment} ...")
        _whitelist_one(
            client, state,
            filename=SEGMENTS[segment]["masked"], media_type="Video",
        )
        _whitelist_one(
            client, state,
            filename=SEGMENTS[segment]["audio"], media_type="Audio",
        )
    print("all assets Active")


def _segment_refs(client_state: dict) -> dict[str, str]:
    """state 里的 whitelist 记录 -> 每个 intake 文件的 asset:// 引用。"""
    by_name = {
        record["name"]: f"asset://{record['asset_id']}"
        for record in client_state.get("whitelist", {}).values()
        if record.get("status") == "Active"
    }
    return by_name


def cmd_submit(segment: str) -> None:
    state = _load_state()
    seg_state = state.setdefault("segments", {}).setdefault(segment, {})
    if seg_state.get("downloaded"):
        print(f"seg{segment} already downloaded:", seg_state["downloaded"])
        return
    secrets = _load_secrets()
    settings = _settings(secrets)
    client = SeedanceClient(settings)
    refs = _segment_refs(state)
    seg = SEGMENTS[segment]
    needed = [seg["masked"], *IMAGE_FILES, seg["audio"]]
    missing = [name for name in needed if name not in refs]
    if missing:
        raise SystemExit(f"whitelist incomplete, run whitelist first: {missing}")
    adapted = _adapt_prompt(segment)

    if not seg_state.get("task_id"):
        # 与管线同一规则（事实：回放 seg1 首次提交 auto 被误判为 extend 而
        # 失败）：请求画幅与源一致 → edit + adaptive；不一致 → reference +
        # 显式 ratio。本项目源 720x1280(9:16) == 请求 9:16 → edit/adaptive。
        probe = _probe_video(REPLAY / "intake" / (REPLAY / ".intake-token").read_text().strip() / seg["masked"])
        source_ratio = "16:9" if probe[0] >= probe[1] else "9:16"
        if source_ratio == RATIO:
            task_type, ratio = "edit", "adaptive"
            # edit 的输出时长跟随输入视频，显式 duration 会被供应商 400
            # 拒绝（seg1 第二次提交实测）。
            duration = None
        else:
            task_type, ratio = "reference", RATIO
            duration = seg["duration"]
        result = client.submit(
            prompt=adapted,
            video_url=refs[seg["masked"]],
            image_urls=[refs[name] for name in IMAGE_FILES],
            audio_urls=[refs[seg["audio"]]],
            duration=duration,
            resolution=RESOLUTION,
            aspect_ratio=ratio,
            task_type=task_type,
        )
        seg_state["task_id"] = result.value
        _save_receipt(f"seg{segment}-submit.json", result.record)
        _save_state(state)
        print(f"seg{segment} submitted, task id:", result.value)
    else:
        print(f"seg{segment} reusing task id:", seg_state["task_id"])

    print(f"seg{segment} polling (30s interval) ...")
    poll = client.poll(seg_state["task_id"])
    _save_receipt(f"seg{segment}-poll-final.json", poll.record)
    print("seg{segment} final video url:", poll.value[:80])
    destination = REPLAY / f"seg{segment}-generated.mp4"
    downloaded = client.download(poll.value, destination)
    _save_receipt(f"seg{segment}-download.json", downloaded.record)
    seg_state["downloaded"] = str(destination)
    seg_state["video_url"] = poll.value
    _save_state(state)
    print(f"seg{segment} downloaded ->", destination)


def cmd_status() -> None:
    state = _load_state()
    print(json.dumps(state, ensure_ascii=False, indent=1))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "smoke":
        cmd_smoke()
    elif command == "prepare":
        cmd_prepare()
    elif command == "whitelist":
        cmd_whitelist()
    elif command == "submit-seg1":
        cmd_submit("1")
    elif command == "submit-seg2":
        cmd_submit("2")
    elif command == "status":
        cmd_status()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
