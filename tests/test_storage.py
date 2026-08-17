import json

from app import storage


def test_new_conversation_layout(tmp_path):
    meta = storage.new_conversation(tmp_path, note="我的笔记", orig_name="x.mp4")
    cdir = tmp_path / meta["id"]
    assert cdir.is_dir()
    assert (cdir / "work").is_dir()
    saved = json.loads((cdir / "meta.json").read_text())
    for key in ("id", "title", "note", "status", "error", "created_at", "updated_at"):
        assert key in saved
    assert saved["status"] == "queued"
    assert saved["error"] is None
    assert saved["title"] == "我的笔记"
    assert saved["note"] == "我的笔记"
    assert len(saved["id"]) == 32


def test_title_falls_back_to_sanitized_filename(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="../../etc/\x00pass\\wd.mp4")
    assert "/" not in meta["title"] and "\x00" not in meta["title"]
    assert meta["title"] == "wd"
    meta = storage.new_conversation(tmp_path, note="", orig_name="a" * 200 + ".mp4")
    assert len(meta["title"]) <= 80


def test_load_meta_rejects_bad_id(tmp_path):
    assert storage.load_meta(tmp_path, "..") is None
    assert storage.load_meta(tmp_path, "a" * 31) is None
    assert storage.load_meta(tmp_path, "g" * 32) is None
    assert storage.load_meta(tmp_path, "0" * 32) is None  # 合法格式但不存在


def test_list_conversations(tmp_path):
    a = storage.new_conversation(tmp_path, note="first", orig_name="a.mp4")
    b = storage.new_conversation(tmp_path, note="second", orig_name="b.mp4")
    items = storage.list_conversations(tmp_path)
    assert {m["id"] for m in items} == {a["id"], b["id"]}
    assert storage.list_conversations(tmp_path / "empty") == []


def test_resolve_file_whitelist(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cdir = tmp_path / meta["id"]
    (cdir / "source.mov").write_bytes(b"s")
    (cdir / "preview.mp4").write_bytes(b"p")
    (cdir / "generated.mp4").write_bytes(b"g")
    (cdir / "work" / "contact_sheet.jpg").write_bytes(b"c")
    (cdir / "work" / "keyframes").mkdir()
    (cdir / "work" / "keyframes" / "k01.jpg").write_bytes(b"k")

    cid = meta["id"]
    assert storage.resolve_file(tmp_path, cid, "source.mp4") == (cdir / "source.mov").resolve()
    assert storage.resolve_file(tmp_path, cid, "preview.mp4") == (cdir / "preview.mp4").resolve()
    assert storage.resolve_file(tmp_path, cid, "generated.mp4") == (cdir / "generated.mp4").resolve()
    assert storage.resolve_file(tmp_path, cid, "contact_sheet.jpg") == (cdir / "work" / "contact_sheet.jpg").resolve()
    assert storage.resolve_file(tmp_path, cid, "keyframes/k01.jpg") == (cdir / "work" / "keyframes" / "k01.jpg").resolve()


def test_resolve_file_rejects_traversal_and_unknown(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cid = meta["id"]
    assert storage.resolve_file(tmp_path, cid, "../meta.json") is None
    assert storage.resolve_file(tmp_path, cid, "keyframes/../meta.json") is None
    assert storage.resolve_file(tmp_path, cid, "keyframes/sub/dir.jpg") is None
    assert storage.resolve_file(tmp_path, cid, "keyframes/") is None
    assert storage.resolve_file(tmp_path, cid, "meta.json") is None
    assert storage.resolve_file(tmp_path, cid, "preview.exe") is None
    assert storage.resolve_file(tmp_path, "..", "preview.mp4") is None


def test_resolve_file_missing_on_disk(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    assert storage.resolve_file(tmp_path, meta["id"], "preview.mp4") is None
    assert storage.resolve_file(tmp_path, meta["id"], "source.mp4") is None
    assert storage.resolve_file(tmp_path, meta["id"], "keyframes/nope.jpg") is None


def test_resolve_file_postprocessed_and_segments(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cid = meta["id"]
    cdir = tmp_path / cid
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    (cdir / "work" / "postprocessed" / "01.png").write_bytes(b"p")
    (cdir / "work" / "segments" / "2" / "keyframes").mkdir(parents=True)
    (cdir / "work" / "segments" / "2" / "keyframes" / "01.png").write_bytes(b"k")
    (cdir / "work" / "segments" / "2" / "postprocessed").mkdir(parents=True)
    (cdir / "work" / "segments" / "2" / "postprocessed" / "01.png").write_bytes(b"s")

    assert storage.resolve_file(tmp_path, cid, "postprocessed/01.png") == \
        (cdir / "work" / "postprocessed" / "01.png").resolve()
    assert storage.resolve_file(tmp_path, cid, "segments/2/keyframes/01.png") == \
        (cdir / "work" / "segments" / "2" / "keyframes" / "01.png").resolve()
    assert storage.resolve_file(tmp_path, cid, "segments/2/postprocessed/01.png") == \
        (cdir / "work" / "segments" / "2" / "postprocessed" / "01.png").resolve()


def test_resolve_file_rejects_bad_segments_and_postprocessed(tmp_path):
    meta = storage.new_conversation(tmp_path, note="", orig_name="a.mp4")
    cid = meta["id"]
    # N 必须为正整数、fn 必须是纯文件名、目录必须是白名单两类；穿越/越界一律 None
    for name in ("segments/0/keyframes/a.png", "segments/x/keyframes/a.png",
                 "segments/1/meta.json", "segments/1/keyframes/",
                 "segments/1/keyframes/../prompt.txt",
                 "segments/1/postprocessed/../keyframes/a.png",
                 "segments/1/keyframes/sub/a.png",
                 "postprocessed/", "postprocessed/../meta.json",
                 "postprocessed/sub/a.png"):
        assert storage.resolve_file(tmp_path, cid, name) is None, name
    # 合法格式但磁盘上不存在 → None
    assert storage.resolve_file(tmp_path, cid, "postprocessed/nope.png") is None
    assert storage.resolve_file(tmp_path, cid, "segments/1/keyframes/nope.png") is None
