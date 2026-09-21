"""资产库（app/assets.py + /api/assets + material_sha256 创建）测试。

覆盖：素材按 sha256 去重与 used_in 排序、owner 隔离、sha 越权/未知/非法 422、
material_sha256 复用创建、上传+选库混合、成片列表含历史版本与缺文件跳过、
前端资产库/选图弹窗的契约接线。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


ACCESS_TOKEN = "test-secret"
LEGACY = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
PNG_A = b"\x89PNG-fake-a" + b"a" * 64
PNG_B = b"\x89PNG-fake-b" + b"b" * 64
SHA_A = hashlib.sha256(PNG_A).hexdigest()
SHA_B = hashlib.sha256(PNG_B).hexdigest()

GENERATION_BASE = {
    "version": 1,
    "output": {"aspect_ratio": "16:9", "resolution": "480p", "fit_mode": "auto"},
    "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
    "dialogue": {"mode": "auto_rewrite", "target_language": "日语"},
}


class IdlePipeline:
    def run(self, _project_id: str) -> None:
        return None


class SuccessfulPipeline:
    def __init__(self, store):
        self.store = store

    def run(self, project_id: str) -> None:
        project = self.store.path(project_id)
        (project / "output" / "generated.mp4").write_bytes(b"generated-video")
        self.store.update(
            project_id,
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={"status": "succeeded", "percent": 100, "stage": "completed"},
        )


def idle_factory(_settings, _store, _register_asset):
    return IdlePipeline()


def successful_factory(_settings, store, _register_asset):
    return SuccessfulPipeline(store)


def make_app(tmp_path: Path, factory=idle_factory, **overrides):
    settings = Settings(
        access_token=ACCESS_TOKEN,
        multiref_mode="composed",
        data_dir=tmp_path / "data",
        public_base_url="https://studio.example",
        store_lock_enabled=False,
        initial_admin_email="admin@duet.local",
        initial_admin_password="admin-secret-pass",
        **overrides,
    )
    return create_app(settings, pipeline_factory=factory)


def bearer(client: TestClient, email: str, password: str = "user-pass-123") -> dict:
    client.post("/api/auth/register", json={"email": email, "password": password})
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def wait_for_status(client: TestClient, headers: dict, project_id: str, status: str) -> dict:
    """pipeline 在后台线程里跑，断言终态前必须先等它落盘。"""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/conversations/{project_id}", headers=headers)
        assert response.status_code == 200
        if response.json()["status"] == status:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"project {project_id} did not reach {status}")


def create_project(
    client: TestClient,
    headers: dict,
    request_id: str,
    *,
    uploads: list[tuple[str, bytes, str]] | None = None,
    materials: list[str] | None = None,
    with_guidance: bool | None = None,
) -> dict:
    uploads = uploads or []
    materials = materials or []
    if with_guidance is None:
        with_guidance = bool(uploads or materials)
    request = dict(GENERATION_BASE)
    request["replacement_guidance"] = (
        {"instruction": "替换产品", "image_field": "replacement_image"}
        if with_guidance else None
    )
    data = {
        "generation_request": json.dumps(request, ensure_ascii=False),
        "client_request_id": request_id,
    }
    if materials:
        data["material_sha256"] = materials
    files = [("file", ("source.mp4", b"source-" + request_id.encode(), "video/mp4"))]
    files += [("replacement_image", upload) for upload in uploads]
    response = client.post("/api/conversations", headers=headers, data=data, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_materials_dedupe_and_used_in_ordering(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        alice = bearer(client, "alice@duet.local")
        older = create_project(
            client, alice, "mat-old",
            uploads=[("a.png", PNG_A, "image/png"), ("b.png", PNG_B, "image/png")],
        )
        newer = create_project(client, alice, "mat-new", uploads=[("a2.png", PNG_A, "image/png")])
        assets = client.get("/api/assets", headers=alice).json()
        materials = {item["sha256"]: item for item in assets["materials"]}
        # 同图多项目只出现一次；预览指向最近一次使用的项目（列表按时间倒序）
        assert set(materials) == {SHA_A, SHA_B}
        entry = materials[SHA_A]
        assert entry["use_count"] == 2
        assert [use["project_id"] for use in entry["used_in"]] == [newer["id"], older["id"]]
        assert entry["preview_url"] == (
            f"/api/conversations/{newer['id']}/creation-input/replacement-images/1"
        )
        assert entry["bytes"] == len(PNG_A)
        assert entry["media_type"] == "image/png"
        # 素材排序：最近使用在前（SHA_A 最新用在 newer，SHA_B 只在 older）
        assert [item["sha256"] for item in assets["materials"]] == [SHA_A, SHA_B]
        # 预览路由本身有归属鉴权且可读
        preview = client.get(entry["preview_url"], headers=alice)
        assert preview.status_code == 200
        assert preview.content == PNG_A


def test_assets_owner_isolation(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        alice = bearer(client, "alice@duet.local")
        bob = bearer(client, "bob@duet.local")
        create_project(client, alice, "iso-mat", uploads=[("a.png", PNG_A, "image/png")])
        bob_assets = client.get("/api/assets", headers=bob).json()
        assert bob_assets["materials"] == []
        assert bob_assets["videos"] == []
        # superuser（播种管理员）全见
        admin = bearer(client, "admin@duet.local", "admin-secret-pass")
        admin_assets = client.get("/api/assets", headers=admin).json()
        assert [item["sha256"] for item in admin_assets["materials"]] == [SHA_A]
        # 旧口令（映射管理员）也全见
        legacy_assets = client.get("/api/assets", headers=LEGACY).json()
        assert len(legacy_assets["materials"]) == 1
        # 未登录 401（无 cookie 的全新客户端）
        with TestClient(app) as anonymous:
            assert anonymous.get("/api/assets").status_code == 401


def test_create_with_material_sha256_reuses_library_file(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        alice = bearer(client, "alice@duet.local")
        source = create_project(client, alice, "mat-src", uploads=[("a.png", PNG_A, "image/png")])
        reused = create_project(client, alice, "mat-reuse", materials=[SHA_A])
        # 复用落盘：receipt/路径结构与上传一致，内容逐字节相同
        assert reused["input_receipt"]["replacement_image"] == {
            "sha256": SHA_A,
            "bytes": len(PNG_A),
        }
        assert reused["replacement_image_paths"] == ["inputs/replacement-image"]
        copied = (
            tmp_path / "data" / "projects" / reused["id"] / "inputs" / "replacement-image"
        ).read_bytes()
        assert copied == PNG_A
        # 文件名沿用了源项目的快照
        assert reused["creation_input"]["replacement_image"]["filename"] == "a.png"
        # 新项目也进入素材聚合（同一 sha use_count 变为 2）
        assets = client.get("/api/assets", headers=alice).json()
        assert assets["materials"][0]["use_count"] == 2
        assert source["id"] != reused["id"]


def test_create_mixed_upload_and_library_material(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        alice = bearer(client, "alice@duet.local")
        create_project(client, alice, "mix-src", uploads=[("a.png", PNG_A, "image/png")])
        mixed = create_project(
            client, alice, "mix-new",
            uploads=[("b.png", PNG_B, "image/png")],
            materials=[SHA_A],
        )
        # 上传在前、库选在后；两张图走 v2 receipt
        assert mixed["replacement_image_paths"] == [
            "inputs/replacement-image", "inputs/replacement-image-2",
        ]
        receipts = mixed["input_receipt"]["replacement_images"]
        assert [item["sha256"] for item in receipts] == [SHA_B, SHA_A]
        assert mixed["input_receipt"]["version"] == 2


def test_material_sha256_validation_and_authorization(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        alice = bearer(client, "alice@duet.local")
        bob = bearer(client, "bob@duet.local")
        create_project(client, alice, "guard-src", uploads=[("a.png", PNG_A, "image/png")])

        def attempt(headers, sha, request_id):
            response = client.post(
                "/api/conversations",
                headers=headers,
                data={
                    "generation_request": json.dumps({
                        **GENERATION_BASE,
                        "replacement_guidance": {
                            "instruction": "替换产品",
                            "image_field": "replacement_image",
                        },
                    }),
                    "client_request_id": request_id,
                    "material_sha256": sha,
                },
                files={"file": ("source.mp4", b"x-" + request_id.encode(), "video/mp4")},
            )
            return response

        # 非法格式 422 invalid_material
        bad = attempt(alice, "not-a-sha", "guard-bad-format")
        assert bad.status_code == 422
        assert bad.json()["detail"]["code"] == "invalid_material"
        # 未知 sha 422 material_not_found
        unknown = attempt(alice, "0" * 64, "guard-unknown")
        assert unknown.status_code == 422
        assert unknown.json()["detail"]["code"] == "material_not_found"
        # 越权：sha 属于他人项目 → 同样的 422（不泄漏存在性）
        cross = attempt(bob, SHA_A, "guard-cross-user")
        assert cross.status_code == 422
        assert cross.json()["detail"]["code"] == "material_not_found"
        # 只有素材没有替换说明 → 422 replacement_pair_required
        no_guidance = client.post(
            "/api/conversations",
            headers=alice,
            data={
                "generation_request": json.dumps({
                    **GENERATION_BASE, "replacement_guidance": None,
                }),
                "client_request_id": "guard-no-guidance",
                "material_sha256": SHA_A,
            },
            files={"file": ("source.mp4", b"x-no-guidance", "video/mp4")},
        )
        assert no_guidance.status_code == 422
        assert no_guidance.json()["detail"]["code"] == "replacement_pair_required"
        # 超出 4 张上限（4 上传 + 1 库选）→ 422
        overflow = client.post(
            "/api/conversations",
            headers=alice,
            data={
                "generation_request": json.dumps({
                    **GENERATION_BASE,
                    "replacement_guidance": {
                        "instruction": "替换产品", "image_field": "replacement_image",
                    },
                }),
                "client_request_id": "guard-overflow",
                "material_sha256": SHA_A,
            },
            files=[("file", ("source.mp4", b"x-overflow", "video/mp4"))] + [
                ("replacement_image", (f"u{i}.png", b"\x89PNG" + bytes([i]) * 8, "image/png"))
                for i in range(4)
            ],
        )
        assert overflow.status_code == 422


def test_videos_include_history_versions_and_skip_missing_files(tmp_path: Path):
    app = make_app(tmp_path, factory=successful_factory)
    with TestClient(app) as client:
        alice = bearer(client, "alice@duet.local")
        project = create_project(client, alice, "vid-project", with_guidance=False)
        project_id = project["id"]
        wait_for_status(client, alice, project_id, "succeeded")
        # 造一个历史版本：归档文件 + generation_history 记录
        history_dir = tmp_path / "data" / "projects" / project_id / "output" / "history" / "v1"
        history_dir.mkdir(parents=True)
        (history_dir / "generated.mp4").write_bytes(b"old-video")
        app.state.store.update(
            project_id,
            generation_history=[{
                "version": 1,
                "completed_at": "2026-09-20T00:00:00+00:00",
                "video": {"path": "output/history/v1/generated.mp4"},
                "segments": [],
            }],
        )
        # 另一个 succeeded 但文件被裁剪掉的项目：应跳过
        ghost = create_project(client, alice, "vid-ghost", with_guidance=False)
        wait_for_status(client, alice, ghost["id"], "succeeded")
        (tmp_path / "data" / "projects" / ghost["id"] / "output" / "generated.mp4").unlink()

        videos = client.get("/api/assets", headers=alice).json()["videos"]
        by_version = {(v["project_id"], v["version"]): v for v in videos}
        current = by_version[(project_id, None)]
        assert current["file_url"] == f"/api/conversations/{project_id}/files/generated.mp4"
        archived = by_version[(project_id, 1)]
        assert archived["completed_at"] == "2026-09-20T00:00:00+00:00"
        assert archived["file_url"] == (
            f"/api/conversations/{project_id}/files/history/v1/generated.mp4"
        )
        assert all(v["project_id"] != ghost["id"] for v in videos)
        # 两个 file_url 都可经现有 files 路由取回
        for entry in (current, archived):
            response = client.get(entry["file_url"], headers=alice)
            assert response.status_code == 200
        # 当前版排在历史版之前（completed_at 更新）
        pair = [v for v in videos if v["project_id"] == project_id]
        assert [v["version"] for v in pair] == [None, 1]


def test_frontend_assets_wiring_contract():
    root = Path(__file__).resolve().parents[1]
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    source = (root / "web" / "app.js").read_text(encoding="utf-8")
    # 顶部入口与资产库视图
    assert 'id="assets-btn"' in html
    assert 'id="assets-view"' in html
    assert 'id="assets-materials"' in html and 'id="assets-videos"' in html
    # 素材选择弹窗（复用 modal-backdrop/modal-card 体系）
    assert 'id="material-pick-btn"' in html
    assert 'id="material-modal" class="modal-backdrop"' in html
    assert 'id="material-modal-grid"' in html
    assert 'id="material-modal-confirm"' in html
    # 选图提交只传 sha 引用
    assert 'fd.append("material_sha256", image.sha256)' in source
    assert 'image.kind === "library"' in source
    # 资产数据来自 /api/assets
    assert 'apiJSON("/api/assets")' in source
    # 素材库来源标记
    assert '素材库' in source
