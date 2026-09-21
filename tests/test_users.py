"""用户系统（app/users.py + main.py 接入）测试。

覆盖：注册/登录/登出/错误密码 401、cookie+bearer 双通道、旧共享口令映射
管理员、owner 隔离（普通用户看不到他人项目的任何端点）、superuser 全见、
存量项目归属迁移，以及 rename / favorite 的校验与权限。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import ProjectStore


ACCESS_TOKEN = "test-secret"
ADMIN_EMAIL = "admin@duet.local"
ADMIN_PASSWORD = "admin-secret-pass"
LEGACY = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

GENERATION_REQUEST = {
    "version": 1,
    "output": {"aspect_ratio": "16:9", "resolution": "480p", "fit_mode": "auto"},
    "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
    "dialogue": {"mode": "auto_rewrite", "target_language": "日语"},
    "replacement_guidance": None,
}


class IdlePipeline:
    """立即返回的占位 pipeline；pipeline_entry 会把项目标成失败终态，与本组
    测试断言无关（只关心鉴权与 state 字段）。"""

    def run(self, _project_id: str) -> None:
        return None


def idle_factory(_settings, _store, _register_asset):
    return IdlePipeline()


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "access_token": ACCESS_TOKEN,
        "multiref_mode": "composed",
        "data_dir": tmp_path / "data",
        "public_base_url": "https://studio.example",
        "store_lock_enabled": False,
        "initial_admin_email": ADMIN_EMAIL,
        "initial_admin_password": ADMIN_PASSWORD,
    }
    values.update(overrides)
    return Settings(**values)


def make_app(tmp_path: Path, **overrides):
    return create_app(make_settings(tmp_path, **overrides), pipeline_factory=idle_factory)


def register(client: TestClient, email: str, password: str = "user-pass-123"):
    return client.post("/api/auth/register", json={"email": email, "password": password})


def login(client: TestClient, email: str, password: str):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def bearer(client: TestClient, email: str, password: str = "user-pass-123") -> dict:
    response = login(client, email, password)
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def create_project(client: TestClient, headers: dict, request_id: str) -> dict:
    response = client.post(
        "/api/conversations",
        headers=headers,
        data={
            "generation_request": json.dumps(GENERATION_REQUEST, ensure_ascii=False),
            "client_request_id": request_id,
        },
        files={"file": ("source.mp4", b"x", "video/mp4")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_register_login_logout_and_wrong_password(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        response = register(client, "alice@duet.local")
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == "alice@duet.local"
        assert body["is_superuser"] is False
        # 重复注册 → 库默认 400 REGISTER_USER_ALREADY_EXISTS
        assert register(client, "alice@duet.local").status_code == 400
        # 错误密码 → 401
        assert login(client, "alice@duet.local", "wrong-pass").status_code == 401
        assert login(client, "nobody@duet.local", "wrong-pass").status_code == 401
        # 正确登录：返回 bearer token 与用户信息，同时写 cookie
        response = login(client, "alice@duet.local", "user-pass-123")
        assert response.status_code == 200
        assert response.json()["token_type"] == "bearer"
        assert response.json()["access_token"]
        assert response.json()["user"]["email"] == "alice@duet.local"
        assert "duet_auth" in response.cookies
        # 登出：cookie 被清除，纯 cookie 通道随即失效
        assert client.post("/api/auth/logout").json() == {"ok": True}
        assert client.get("/api/auth/me").status_code == 401
        assert client.get("/api/conversations").status_code == 401


def test_cookie_and_bearer_dual_channels(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "bob@duet.local")
        response = login(client, "bob@duet.local", "user-pass-123")
        token = response.json()["access_token"]
        cookie = response.cookies["duet_auth"]
        # 纯 bearer（清掉 cookie 后只带 Authorization 头）
        client.cookies.clear()
        assert client.get(
            "/api/conversations", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 200
        # 纯 cookie（不带 Authorization 头）
        client.cookies.set("duet_auth", cookie)
        assert client.get("/api/conversations").status_code == 200
        # 坏 token 两通道都拒
        client.cookies.clear()
        assert client.get(
            "/api/conversations", headers={"Authorization": "Bearer not-a-jwt"}
        ).status_code == 401
        client.cookies.set("duet_auth", "not-a-jwt")
        assert client.get("/api/conversations").status_code == 401


def test_legacy_token_maps_to_admin(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "carol@duet.local")
        carol = bearer(client, "carol@duet.local")
        created = create_project(client, carol, "legacy-bridge-project")
        # 旧共享口令 → 播种的管理员身份
        me = client.get("/api/auth/me", headers=LEGACY)
        assert me.status_code == 200
        assert me.json()["email"] == ADMIN_EMAIL
        assert me.json()["is_superuser"] is True
        # 管理员可见普通用户的项目
        listing = client.get("/api/conversations", headers=LEGACY).json()
        assert any(item["id"] == created["id"] for item in listing)
        # /api/login 兼容入口行为不变
        assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
        assert client.post("/api/login", json={"token": ACCESS_TOKEN}).json() == {"ok": True}


def test_owner_isolation_across_all_endpoints(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "alice@duet.local")
        register(client, "bob@duet.local")
        alice = bearer(client, "alice@duet.local")
        bob = bearer(client, "bob@duet.local")
        project = create_project(client, alice, "alice-project")
        project_id = project["id"]
        # 新建项目 owner_id = 当前用户
        me = client.get("/api/auth/me", headers=alice).json()
        detail = client.get(f"/api/conversations/{project_id}", headers=alice)
        assert detail.status_code == 200
        assert detail.json()["owner_id"] == me["id"]

        bob_listing = client.get("/api/conversations", headers=bob).json()
        assert all(item["id"] != project_id for item in bob_listing)
        endpoints = [
            ("GET", f"/api/conversations/{project_id}", None),
            ("GET", f"/api/conversations/{project_id}/files/source.mp4", None),
            ("GET", f"/api/conversations/{project_id}/prompt-review", None),
            ("POST", f"/api/conversations/{project_id}/prompt-review/confirm", {"segments": []}),
            ("GET", f"/api/conversations/{project_id}/generation-history", None),
            ("POST", f"/api/conversations/{project_id}/regenerate", {"segments": []}),
            ("POST", f"/api/conversations/{project_id}/resume-picsart", {"confirm": True}),
            ("POST", f"/api/conversations/{project_id}/resume-pre-picsart", {"confirm": True}),
            ("POST", f"/api/conversations/{project_id}/resume-h3-prep", {"confirm": True}),
            ("POST", f"/api/conversations/{project_id}/rename", {"title": "x"}),
            ("POST", f"/api/conversations/{project_id}/favorite", {"favorite": True}),
            ("GET", f"/api/conversations/{project_id}/creation-input/replacement-image", None),
            ("GET", f"/api/conversations/{project_id}/creation-input/replacement-images/1", None),
        ]
        for method, path, body in endpoints:
            response = client.request(method, path, headers=bob, json=body)
            assert response.status_code == 404, f"{method} {path} → {response.status_code}"
            assert response.json()["detail"]["code"] == "project_not_found"
        # 无权访问不泄漏存在性：与真正不存在的项目响应一致
        missing = client.get(f"/api/conversations/{'0' * 32}", headers=bob)
        assert missing.status_code == 404


def test_superuser_sees_everything(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "alice@duet.local")
        alice = bearer(client, "alice@duet.local")
        project = create_project(client, alice, "alice-supervisor-project")
        admin = bearer(client, ADMIN_EMAIL, ADMIN_PASSWORD)
        listing = client.get("/api/conversations", headers=admin).json()
        assert any(item["id"] == project["id"] for item in listing)
        assert client.get(f"/api/conversations/{project['id']}", headers=admin).status_code == 200
        # superuser 可以改名/收藏他人项目
        renamed = client.post(
            f"/api/conversations/{project['id']}/rename",
            headers=admin,
            json={"title": "管理员改的名"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "管理员改的名"


def test_ownerless_projects_migrate_to_first_superuser(tmp_path: Path):
    projects_root = tmp_path / "data" / "projects"
    store = ProjectStore(projects_root, acquire_process_lock=False)
    project_id, _project, _state = store.create({}, "legacy-ownerless")
    # 终态项目不被重启恢复逻辑触碰，updated_at 才能证明迁移本身不改时间戳
    state = store.update(
        project_id,
        has_video=True,
        output_path="output/generated.mp4",
        project_progress={"status": "succeeded", "percent": 100, "stage": "completed"},
    )
    original_updated_at = state["updated_at"]
    store.close()

    app = make_app(tmp_path)
    with TestClient(app) as client:
        me = client.get("/api/auth/me", headers=LEGACY).json()
        migrated = client.get(f"/api/conversations/{project_id}", headers=LEGACY).json()
        assert migrated["owner_id"] == me["id"]
        # 归属迁移不刷新 updated_at
        assert migrated["updated_at"] == original_updated_at


def test_ownerless_projects_hidden_from_regular_users_until_migrated(tmp_path: Path):
    app = make_app(tmp_path, initial_admin_email=None, initial_admin_password=None)
    project_id, _project, _state = app.state.store.create({}, "unmigrated")
    with TestClient(app) as client:
        # 没有 superuser：迁移跳过，普通用户看不到无主项目
        register(client, "dave@duet.local")
        dave = bearer(client, "dave@duet.local")
        listing = client.get("/api/conversations", headers=dave).json()
        assert all(item["id"] != project_id for item in listing)
        assert client.get(f"/api/conversations/{project_id}", headers=dave).status_code == 404
        # 旧口令（合成管理员）仍可访问
        assert client.get(f"/api/conversations/{project_id}", headers=LEGACY).status_code == 200


def test_rename_validation_and_effect(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "alice@duet.local")
        alice = bearer(client, "alice@duet.local")
        project = create_project(client, alice, "rename-target")
        project_id = project["id"]
        assert project["title"]

        renamed = client.post(
            f"/api/conversations/{project_id}/rename",
            headers=alice,
            json={"title": "  夏日新品宣传片  "},
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "夏日新品宣传片"
        # 详情与列表同步生效
        assert client.get(
            f"/api/conversations/{project_id}", headers=alice
        ).json()["title"] == "夏日新品宣传片"
        listing = client.get("/api/conversations", headers=alice).json()
        assert listing[0]["title"] == "夏日新品宣传片"

        for body in (
            {"title": ""},
            {"title": "   "},
            {"title": "x" * 101},
            {"title": 42},
            {"name": "缺少 title 字段"},
        ):
            response = client.post(
                f"/api/conversations/{project_id}/rename", headers=alice, json=body
            )
            assert response.status_code == 422, body
        assert client.post(
            f"/api/conversations/{project_id}/rename", headers=alice, content=b"{"
        ).status_code == 400


def test_favorite_toggle_validation_and_list_ordering(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "alice@duet.local")
        alice = bearer(client, "alice@duet.local")
        first = create_project(client, alice, "fav-first")
        second = create_project(client, alice, "fav-second")
        third = create_project(client, alice, "fav-third")
        # 缺省未收藏
        detail = client.get(f"/api/conversations/{first['id']}", headers=alice).json()
        assert detail.get("favorite") in (None, False)

        toggled = client.post(
            f"/api/conversations/{first['id']}/favorite",
            headers=alice,
            json={"favorite": True},
        )
        assert toggled.status_code == 200
        assert toggled.json()["favorite"] is True

        listing = client.get("/api/conversations", headers=alice).json()
        assert [item["id"] for item in listing] == [first["id"], third["id"], second["id"]]
        assert listing[0]["favorite"] is True

        untoggled = client.post(
            f"/api/conversations/{first['id']}/favorite",
            headers=alice,
            json={"favorite": False},
        )
        assert untoggled.json()["favorite"] is False
        listing = client.get("/api/conversations", headers=alice).json()
        assert [item["id"] for item in listing] == [third["id"], second["id"], first["id"]]

        for body in ({}, {"favorite": "true"}, {"favorite": 1}):
            response = client.post(
                f"/api/conversations/{first['id']}/favorite", headers=alice, json=body
            )
            assert response.status_code == 422, body


def test_favorite_and_rename_require_ownership(tmp_path: Path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        register(client, "alice@duet.local")
        register(client, "bob@duet.local")
        alice = bearer(client, "alice@duet.local")
        bob = bearer(client, "bob@duet.local")
        project = create_project(client, alice, "ownership-guard")
        assert client.post(
            f"/api/conversations/{project['id']}/rename", headers=bob, json={"title": "劫持"}
        ).status_code == 404
        assert client.post(
            f"/api/conversations/{project['id']}/favorite", headers=bob, json={"favorite": True}
        ).status_code == 404
        # 未被篡改
        detail = client.get(f"/api/conversations/{project['id']}", headers=alice).json()
        assert detail["title"] != "劫持"
        assert detail.get("favorite") in (None, False)


def test_admin_seed_only_when_store_is_empty(tmp_path: Path):
    settings = make_settings(tmp_path)
    # 配置不完整（只给邮箱）直接拒绝启动配置
    with pytest.raises(ValueError, match="INITIAL_ADMIN"):
        make_settings(tmp_path, initial_admin_password=None)
    app = create_app(settings, pipeline_factory=idle_factory)
    with TestClient(app):
        pass
    # 第二次启动：用户库非空，不重复播种（管理员仍是唯一用户）
    app2 = create_app(settings, pipeline_factory=idle_factory)
    with TestClient(app2) as client:
        assert register(client, ADMIN_EMAIL, "other-pass").status_code == 400
        assert login(client, ADMIN_EMAIL, ADMIN_PASSWORD).status_code == 200
