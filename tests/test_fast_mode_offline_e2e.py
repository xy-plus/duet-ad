from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import h3, long_generation, long_video, prepared_input, stitch, storage
from app.config import Settings
from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "web" / "app.js"
ACCESS_TOKEN = "offline-access-token"
AUTH = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
SEGMENT_COUNT = 2

pytestmark = pytest.mark.skipif(
    not shutil.which("node")
    or not shutil.which("ffmpeg")
    or not shutil.which("ffprobe"),
    reason="node, ffmpeg, and ffprobe are required",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_video(path: Path, *, duration_s: float, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=54x96:r=12:d={duration_s}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _make_png(path: Path, value: int) -> None:
    image = np.full((160, 90, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _make_frozen_long_plan(settings: Settings) -> tuple[str, str]:
    meta = storage.new_conversation(
        settings.data_dir,
        "offline fast mode E2E",
        "source.mp4",
    )
    cid = meta["id"]
    root = settings.data_dir / cid
    source = root / "source.mp4"
    _make_video(source, duration_s=11, color="black")

    receipt_segments = []
    public_segments = []
    bounds = ((0.0, 5.5), (5.5, 11.0))
    for index, (start_s, end_s) in enumerate(bounds, 1):
        join_mode = "hard_cut" if index == 1 else "continue"
        segdir = root / "work" / "segments" / str(index)
        segment_source = segdir / "source.mp4"
        _make_video(
            segment_source,
            duration_s=end_s - start_s,
            color="navy" if index == 1 else "teal",
        )
        work = segdir / "work"
        keyframe = work / "keyframes" / "01.png"
        first_frame = work / "anchors" / "first.png"
        last_frame = work / "anchors" / "last.png"
        _make_png(keyframe, 20 + index)
        _make_png(first_frame, 40 + index)
        _make_png(last_frame, 60 + index)

        visual_text = f"第{index}段局部动作"
        visual_prompt = work / "visual_prompt.txt"
        visual_prompt.write_text(visual_text, encoding="utf-8")
        prompt_text = "不要生成背景音乐\n" + prepared_input.compose_final_prompt(
            long_video.compose_segment_visual_prompt(visual_text), ()
        )
        final_prompt = work / "prompt.txt"
        final_prompt.write_text(prompt_text, encoding="utf-8")

        public = {
            "index": index,
            "start_s": start_s,
            "end_s": end_s,
            "chain_id": "chain-001",
            "join_mode": join_mode,
            "source": f"segments/{index}/source.mp4",
            "keyframes": ["01.png"],
            "keyframe_paths": [
                f"segments/{index}/work/keyframes/01.png"
            ],
            "first_frame_path": (
                f"segments/{index}/work/anchors/first.png"
            ),
            "last_frame_path": f"segments/{index}/work/anchors/last.png",
            "visual_prompt": visual_text,
            "prompt": prompt_text,
            "dialogue": [],
            "lines": [],
        }
        public_segments.append(public)
        receipt_segments.append(
            {
                **public,
                "source_path": segment_source,
                "keyframe_paths": [keyframe],
                "first_frame_path": first_frame,
                "last_frame_path": last_frame,
                "visual_prompt_path": visual_prompt,
                "final_prompt_path": final_prompt,
            }
        )

    receipt_path = long_video.write_plan_receipt(
        root,
        source=source,
        duration_s=11,
        segments=receipt_segments,
        workflow=h3.H3_BOUNDARY_WORKFLOW,
    )
    receipt = _sha256(receipt_path)
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        duration_s=11,
        voice_mode="keep",
        fit_required=False,
        segments=public_segments,
        long_video_plan_receipt=receipt_path.name,
    )
    return cid, receipt


def _web_fast_mode_payload(cid: str, receipt: str) -> dict:
    script = r"""
const contract = require(process.argv[1]);
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.listeners = {};
    this.checked = false;
    this.type = "";
    this.textContent = "";
    this.className = "";
    this.id = "";
  }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  dispatchEvent(event) { this.listeners[event.type](event); }
  querySelector(selector) {
    const matches = (candidate) => selector === "input"
      && candidate.tagName === "INPUT";
    for (const child of this.children) {
      if (matches(child)) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
}
global.document = {createElement: (tag) => new FakeElement(tag)};
const detail = {
  id: process.argv[2],
  duration_s: 11,
  segment_count: 2,
  plan_receipt: process.argv[3],
};
const draft = {fastMode: false};
const field = contract.fastModeField(detail, draft);
const checkbox = field.querySelector("input");
checkbox.checked = true;
checkbox.dispatchEvent({type: "change"});
const payload = contract.buildSubmitPayload({
  clientRequestId: "offline-fast-parent-001",
  dialogueMode: "none",
  fitRequired: false,
  isLong: true,
  fastMode: draft.fastMode,
  planReceipt: detail.plan_receipt,
  aspectRatio: "9:16",
  resolution: "768p",
});
process.stdout.write(JSON.stringify({draftFastMode: draft.fastMode, payload}));
"""
    completed = subprocess.run(
        [shutil.which("node"), "-e", script, str(APP_JS), cid, receipt],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["draftFastMode"] is True
    assert result["payload"]["fast_mode"] is True
    return result["payload"]


class _PublicNetworkStream:
    def get_extra_info(self, name):
        return ("93.184.216.34", 443) if name == "server_addr" else None


class _OfflineProvider:
    def __init__(self, root: Path, output: bytes) -> None:
        self.root = root
        self.output = output
        self.lock = threading.Lock()
        self.post_barrier = threading.Barrier(SEGMENT_COUNT)
        self.posts: list[dict] = []
        self.result_gets: list[str] = []
        self.downloads: list[str] = []
        self.active_posts = 0
        self.max_active_posts = 0
        self.post_exited = 0
        self.all_prepared_at_first_post = False

    def _assert_all_attempts_prepared(self) -> None:
        for index in range(1, SEGMENT_COUNT + 1):
            attempt_path = (
                self.root
                / "work"
                / "segments"
                / str(index)
                / ".h3"
                / "attempts"
                / "000001"
                / "attempt.json"
            )
            state = json.loads(attempt_path.read_text(encoding="utf-8"))
            assert state["attempt_id"] == "000001"
            assert len(state["input_receipt"]) == 64
            assert state["status"] in {"ready_to_submit", "h3_submitting"}
            assert state["h3"]["status"] in {"ready", "submitting"}
            assert "task_id" not in state["h3"]
        self.all_prepared_at_first_post = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith(
            ("/minimax_h3_lightx2v_v5", "/minimax_h3_lightx2v")
        ):
            assert request.url.host == "autodl.art"
            assert request.headers["authorization"] == "offline-art-token"
            body = json.loads(request.content)
            with self.lock:
                task_id = f"offline-task-{len(self.posts) + 1}"
                first_post = not self.posts
                self.posts.append({"task_id": task_id, "body": body})
                self.active_posts += 1
                self.max_active_posts = max(
                    self.max_active_posts, self.active_posts
                )
            if first_post:
                self._assert_all_attempts_prepared()
            try:
                self.post_barrier.wait(timeout=10)
            finally:
                with self.lock:
                    self.active_posts -= 1
                    self.post_exited += 1
            return httpx.Response(200, json={"data": {"task_id": task_id}})

        if request.method == "GET" and "/result/" in path:
            task_id = path.rsplit("/", 1)[-1]
            with self.lock:
                assert self.post_exited == SEGMENT_COUNT
                assert task_id in {item["task_id"] for item in self.posts}
                self.result_gets.append(task_id)
            return httpx.Response(
                200,
                json={
                    "data": {
                        "status": "SUCCESS",
                        "results": [
                            {
                                "url": (
                                    "https://offline-download.invalid/"
                                    f"{task_id}.mp4"
                                )
                            }
                        ],
                    }
                },
            )

        if request.method == "GET" and request.url.host == "offline-download.invalid":
            task_id = Path(path).stem
            with self.lock:
                assert task_id in {item["task_id"] for item in self.posts}
                self.downloads.append(task_id)
            return httpx.Response(
                200,
                content=self.output,
                extensions={"network_stream": _PublicNetworkStream()},
            )
        raise AssertionError(f"unexpected offline provider request: {request}")


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,codec_name,pix_fmt,avg_frame_rate,duration:"
                "format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_fast_mode_http_to_real_stitched_video_is_fully_offline(
    tmp_path, monkeypatch
):
    settings = Settings(
        access_token=ACCESS_TOKEN,
        data_dir=tmp_path / "data",
        enable_pipeline=False,
        enable_h3_submit=True,
        autodl_art_token="offline-art-token",
        h3_request_timeout_s=5,
        h3_poll_timeout_s=5,
        h3_download_timeout_s=5,
        h3_poll_interval_s=0,
        retry_count=0,
        retry_interval_s=0,
    )
    cid, plan_receipt = _make_frozen_long_plan(settings)
    root = settings.data_dir / cid
    payload = _web_fast_mode_payload(cid, plan_receipt)

    provider_video = tmp_path / "provider-output.mp4"
    _make_video(provider_video, duration_s=6, color="red")
    provider = _OfflineProvider(root, provider_video.read_bytes())

    @contextmanager
    def offline_client(client):
        assert client is None
        with httpx.Client(
            transport=httpx.MockTransport(provider), trust_env=False
        ) as owned:
            yield owned

    def offline_getaddrinfo(host, port, *_args, **_kwargs):
        assert host == "offline-download.invalid"
        assert port == 443
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 443),
            )
        ]

    monkeypatch.setattr(h3, "_client", offline_client)
    monkeypatch.setattr(h3.socket, "getaddrinfo", offline_getaddrinfo)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/submit",
            headers=AUTH,
            json=payload,
        )
        assert response.status_code == 202
        assert response.json() == {"status": "queued", "attempt": 1}
        detail = client.get(
            f"/api/conversations/{cid}", headers=AUTH
        ).json()

    assert provider.all_prepared_at_first_post is True
    assert len(provider.posts) == SEGMENT_COUNT
    assert provider.max_active_posts == SEGMENT_COUNT
    task_ids = {item["task_id"] for item in provider.posts}
    assert len(task_ids) == SEGMENT_COUNT
    assert set(provider.result_gets) == task_ids
    assert len(provider.result_gets) == SEGMENT_COUNT
    assert set(provider.downloads) == task_ids
    assert len(provider.downloads) == SEGMENT_COUNT

    posted_by_segment = {
        index: next(
            item["body"]
            for item in provider.posts
            if f"第{index}段局部动作" in item["body"]["prompt"]
        )
        for index in range(1, SEGMENT_COUNT + 1)
    }
    assert (
        posted_by_segment[1]["last_frame"]
        == posted_by_segment[2]["first_frame"]
    )

    assert detail["generation"]["fast_mode"] is True
    assert detail["generation"]["status"] == "succeeded"
    assert detail["has_video"] is True
    public_generation = json.dumps(detail["generation"], sort_keys=True)
    assert "child_request_id" not in public_generation
    assert "task_id" not in public_generation
    assert "offline-task-" not in public_generation

    segment_paths = []
    persisted_task_ids = set()
    for index in range(1, SEGMENT_COUNT + 1):
        segment_root = root / "work" / "segments" / str(index)
        session = json.loads(
            (segment_root / ".h3" / "session.json").read_text(encoding="utf-8")
        )
        assert session == {
            "schema_version": h3.SCHEMA_VERSION,
            "cid": f"{cid}-segment-{index}",
        }
        attempt = json.loads(
            (
                segment_root
                / ".h3"
                / "attempts"
                / "000001"
                / "attempt.json"
            ).read_text(encoding="utf-8")
        )
        assert attempt["attempt_id"] == "000001"
        assert attempt["status"] == "succeeded"
        assert len(attempt["input_receipt"]) == 64
        assert attempt["h3"]["status"] == "succeeded"
        task_id = attempt["h3"]["task_id"]
        persisted_task_ids.add(task_id)
        assert attempt["h3"]["receipt"]["task_id"] == task_id
        segment_output = segment_root / "generated.mp4"
        output_receipt = attempt["h3"]["output"]
        assert output_receipt == {
            "name": "generated.mp4",
            "sha256": _sha256(segment_output),
            "size": segment_output.stat().st_size,
        }
        segment_paths.append(segment_output)
    assert persisted_task_ids == task_ids

    output = root / "generated.mp4"
    probe = _probe(output)
    video_stream = next(
        item for item in probe["streams"] if item["codec_type"] == "video"
    )
    assert video_stream["codec_name"] == "h264"
    assert video_stream["pix_fmt"] == "yuv420p"
    assert video_stream["avg_frame_rate"] == "24/1"
    duration_s = float(video_stream.get("duration") or probe["format"]["duration"])
    assert duration_s == pytest.approx(11, abs=1 / 24 + 1e-6)

    stitch_receipt = json.loads(
        (root / stitch.RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    assert stitch_receipt["schema"] == "duet.stitch"
    assert stitch_receipt["version"] == 1
    assert [item["index"] for item in stitch_receipt["segments"]] == [1, 2]
    assert [item["join_mode"] for item in stitch_receipt["segments"]] == [
        "hard_cut",
        "continue",
    ]
    assert [item["sha256"] for item in stitch_receipt["segments"]] == [
        _sha256(path) for path in segment_paths
    ]
    assert stitch_receipt["audio"]["mode"] == "mute"
    assert stitch_receipt["output"] == {
        "name": "generated.mp4",
        "sha256": _sha256(output),
        "size": output.stat().st_size,
        "duration_s": pytest.approx(11, abs=1 / 24 + 1e-6),
        "fps": 24,
    }

    final_meta = storage.load_meta(settings.data_dir, cid)
    frozen = long_generation.freeze_plan(
        root,
        final_meta,
        plan_receipt,
        "none",
        "none",
        aspect_ratio="9:16",
        resolution="768p",
        prepare_fit=False,
    )
    assert long_generation.stitched_output_is_reusable(frozen, "none")
