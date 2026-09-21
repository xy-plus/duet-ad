"""内部用户系统：fastapi-users（SQLAlchemy + aiosqlite + SQLite）标准件适配层。

- 账号体系：邮箱 + 密码（库默认 User 表，argon2 哈希），开放自助注册。
- 双通道认证：Bearer JWT（API 客户端/脚本）与 Cookie（浏览器，
  HttpOnly + sameSite=lax）共用同一 JWT 策略；两通道同时出现时显式
  Authorization 头优先，避免罐里旧 cookie 抢道。
- 用户库文件：settings.users_db_path（显式配置，默认 <DATA_DIR>/users.db）。
- 迁移桥：旧共享口令（ACCESS_TOKEN）继续有效，映射为首个 superuser；
  用户库还没有 superuser 时映射为合成管理员 LEGACY_ADMIN，保证老脚本不间断。
- 存量项目归属迁移：启动时把无 owner_id 的项目归给首个 superuser（落盘一次，
  不改 updated_at）；没有 superuser 时跳过，下次启动再迁移。

路由（在 app/main.py 注册，错误风格与内部 API 一致）：
POST /api/auth/register（库注册路由）、POST /api/auth/login（邮箱+密码，
成功同时写 cookie 并返回 bearer token）、POST /api/auth/logout（清 cookie；
JWT 无状态，bearer 端由客户端丢弃）、GET /api/auth/me（当前用户，
legacy 口令也返回管理员身份）。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import Depends
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings
from app.store import ProjectStore


LOGGER = logging.getLogger(__name__)

# 会话有效期：cookie max_age 与 JWT lifetime 一致。
SESSION_LIFETIME_S = 7 * 24 * 3600
COOKIE_NAME = "duet_auth"
# 旧共享口令在用户库尚无 superuser 时映射到的合成管理员 id；只出现在
# 新建项目的 owner_id 与 /api/auth/me 响应里。播种真实管理员后旧口令自动
# 改映射到它，合成 id 拥有的存量项目仍只对 superuser 可见（不会丢）。
LEGACY_ADMIN_ID = "legacy-shared-token"


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    pass


class UserRead(schemas.BaseUser[uuid.UUID]):
    # 内部工具允许 admin@duet.local 这类保留域邮箱：库默认 EmailStr 的
    # email-validator 会拒绝 .local，这里放宽为非空字符串。
    email: str


class UserCreate(schemas.BaseUserCreate):
    email: str


@dataclass(frozen=True)
class ResolvedUser:
    """端点统一使用的当前用户身份（库用户或旧口令映射的管理员）。"""

    id: str
    email: str
    is_superuser: bool


def can_access_project(state: dict[str, Any], user: ResolvedUser) -> bool:
    """superuser 可见全部；普通用户只看到自己的项目。无 owner_id 的存量
    项目（迁移尚未运行，即用户库还没有 superuser）只对 superuser 可见。"""
    if user.is_superuser:
        return True
    owner = state.get("owner_id")
    return isinstance(owner, str) and owner == user.id


class UserAuth:
    """一个应用实例一套用户系统：引擎、fastapi-users 装配与迁移都在这里。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{settings.users_db_path}"
        )
        self._session_maker = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        secret = settings.users_jwt_secret

        async def get_session():
            async with self._session_maker() as session:
                yield session

        async def get_user_db(session: AsyncSession = Depends(get_session)):
            yield SQLAlchemyUserDatabase(session, User)

        class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
            reset_password_token_secret = secret
            verification_token_secret = secret

        self._manager_class = UserManager

        async def get_user_manager(user_db=Depends(get_user_db)):
            yield UserManager(user_db)

        self.get_user_manager = get_user_manager

        def get_jwt_strategy() -> JWTStrategy:
            return JWTStrategy(secret=secret, lifetime_seconds=SESSION_LIFETIME_S)

        self.get_jwt_strategy = get_jwt_strategy
        # cookie_secure=False：TLS 在 caddy 终止，测试走 http://testserver；
        # HttpOnly + sameSite=lax 已挡住 JS 读取与跨站携带。
        self.cookie_transport = CookieTransport(
            cookie_name=COOKIE_NAME,
            cookie_max_age=SESSION_LIFETIME_S,
            cookie_secure=False,
            cookie_samesite="lax",
        )
        self.cookie_backend = AuthenticationBackend(
            "cookie", self.cookie_transport, get_jwt_strategy
        )
        self.bearer_backend = AuthenticationBackend(
            "bearer", BearerTransport(tokenUrl="/api/auth/login"), get_jwt_strategy
        )
        # 通道顺序有意义：显式 Authorization 头优先于 cookie（同一浏览器里
        # 脚本/旧前端带头请求不应被罐里的旧 cookie 抢道）；纯浏览器请求走 cookie。
        self.fastapi_users = FastAPIUsers[User, uuid.UUID](
            get_user_manager, [self.bearer_backend, self.cookie_backend]
        )
        self.register_router = self.fastapi_users.get_register_router(
            UserRead, UserCreate
        )
        # optional=True：缺失/非法凭据返回 None，由 require_user 统一判 401
        # （旧口令通道在 fastapi-users 之外）。
        self.optional_current_user = self.fastapi_users.current_user(
            optional=True, active=True
        )

    async def setup(self) -> None:
        """建表；首次启动（用户库为空）且配置了 INITIAL_ADMIN_* 时播种管理员。"""
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        email = self._settings.initial_admin_email
        if email is None:
            return
        async with self._session_maker() as session:
            existing = await session.execute(select(User.id).limit(1))
            if existing.first() is not None:
                return
            manager = self._manager_class(SQLAlchemyUserDatabase(session, User))
            await manager.create(
                UserCreate(
                    email=email,
                    password=self._settings.initial_admin_password,
                    is_superuser=True,
                    is_verified=True,
                ),
                safe=False,
            )
            LOGGER.info("seeded initial admin user %s", email)

    async def first_superuser(self) -> User | None:
        async with self._session_maker() as session:
            result = await session.execute(
                select(User)
                .where(User.is_superuser.is_(True))
                .order_by(User.email)
                .limit(1)
            )
            return result.scalars().first()

    async def resolve_legacy_admin(self) -> ResolvedUser:
        """旧共享口令 → 首个 superuser；还没有 superuser 时用合成管理员。"""
        admin = await self.first_superuser()
        if admin is not None:
            return ResolvedUser(
                id=str(admin.id), email=admin.email, is_superuser=True
            )
        return ResolvedUser(
            id=LEGACY_ADMIN_ID, email="legacy-admin", is_superuser=True
        )

    async def migrate_project_ownership(self, store: ProjectStore) -> int:
        """把无 owner_id 的存量项目归给首个 superuser（落盘一次）。

        用 store.write 而不是 store.update：归属迁移不刷新 updated_at，
        避免列表里的时间展示被迁移动作污染。没有 superuser 时跳过，
        下次启动再迁移。返回迁移数量。
        """
        admin = await self.first_superuser()
        if admin is None:
            return 0
        owner = str(admin.id)
        migrated = 0
        for state in store.list():
            if state.get("owner_id"):
                continue
            store.write(state["id"], {**state, "owner_id": owner})
            migrated += 1
        if migrated:
            LOGGER.info(
                "migrated %d ownerless projects to admin %s", migrated, admin.email
            )
        return migrated

    async def close(self) -> None:
        await self._engine.dispose()
