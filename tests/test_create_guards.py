"""上传防呆：client_request_id 幂等 + queued 上限 + 管道闸语义。"""
import json
import threading
import time

from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import pipeline, storage
from app.main import create_app

RID = "req-0001-abcd"


def _post(client, video_1s, rid=None):
    data = {"client_request_id": rid} if rid is not None else {}
    with open(video_1s, "rb") as f:
        return client.post("/api/conversations", headers=AUTH,
                           files={"file": ("clip.mp4", f, "video/mp4")},
                           data=data)


def test_same_client_request_id_dedup(tmp_path, video_1s, monkeypatch):
    """同 id 二次提交：200 返回既有会话，不建目录、不重复入队。"""
    settings = make_settings(tmp_path, enable_pipeline=True)
    called = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: called.append(1))
    with TestClient(create_app(settings)) as c:
        r1 = _post(c, video_1s, RID)
        assert r1.status_code == 201
        r2 = _post(c, video_1s, RID)
        assert r2.status_code == 200
        assert r2.json()["id"] == r1.json()["id"]
    assert called == [1]  # 只入队一次
    dirs = list(settings.data_dir.iterdir())
    assert len(dirs) == 1
    meta = json.loads((dirs[0] / "meta.json").read_text())
    assert meta["client_request_id"] == RID


def test_different_client_request_ids_create_each(client, video_1s, settings):
    r1 = _post(client, video_1s, RID)
    r2 = _post(client, video_1s, RID + "-2")
    assert r1.status_code == 201 and r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]
    assert len(list(settings.data_dir.iterdir())) == 2


def test_invalid_client_request_id_400(client, video_1s, settings):
    for bad in ("short", "bad_chars!!", "x" * 65):
        assert _post(client, video_1s, bad).status_code == 400, bad
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_same_client_request_id_concurrent(tmp_path, video_1s):
    """同 id 双线程并发：恰建一个目录、恰一个 201。"""
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        results = []

        def post():
            results.append(_post(c, video_1s, RID))

        threads = [threading.Thread(target=post) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert sorted(r.status_code for r in results) == [200, 201]
    assert len({r.json()["id"] for r in results}) == 1
    assert len(list(settings.data_dir.iterdir())) == 1


def test_queue_cap_429(tmp_path, video_1s):
    """queued 数达上限即 429（detail 与 IP 限流区分）；非 queued 不占额度。"""
    settings = make_settings(tmp_path, max_queued=2)
    with TestClient(create_app(settings)) as c:
        r1 = _post(c, video_1s)
        r2 = _post(c, video_1s)
        assert r1.status_code == 201 and r2.status_code == 201
        r3 = _post(c, video_1s)
        assert r3.status_code == 429
        assert r3.json()["detail"] == "too many queued tasks"
        # queued 之外的状态不占额度：一个转 done 后即可再建
        storage.update_meta(settings.data_dir, r1.json()["id"], status="done")
        assert _post(c, video_1s).status_code == 201


def test_pipeline_gate_keeps_extra_queued(tmp_path, video_1s, monkeypatch):
    """闸占满时，超额会话的后台任务进不了 pipeline（保持 queued 语义）。"""
    settings = make_settings(tmp_path, enable_pipeline=True, codex_concurrency=1)
    entered = []
    first_in = threading.Event()
    release = threading.Event()

    def fake_run(s, cid, runner, *, claimed_owner=None):
        assert claimed_owner is not None
        entered.append(cid)
        first_in.set()
        release.wait(10)

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(create_app(settings)) as c:
        t1 = threading.Thread(target=lambda: _post(c, video_1s))
        t1.start()
        assert first_in.wait(5)
        t2 = threading.Thread(target=lambda: _post(c, video_1s))
        t2.start()
        time.sleep(0.5)  # 给第二个后台任务到闸前的时间
        assert len(entered) == 1  # 仍卡在管道闸上
        release.set()
        t1.join()
        t2.join()
    assert len(entered) == 2
