"""提供异步数据库配置、引擎和会话生命周期管理。"""

# 启用延迟解析类型注解。
from __future__ import annotations

# 导入异步迭代器类型和单实例缓存装饰器。
from collections.abc import AsyncIterator
from functools import lru_cache

# 导入配置模型与 SQLAlchemy 异步数据库组件。
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# 导入统一的命名配置段读取接口。
from config import get_section


class DatabaseSettings(BaseModel):
    """定义异步数据库连接池配置。"""

    # 数据库连接地址必须是非空字符串。
    url: str = Field(min_length=1)
    # 设置连接池常驻连接数并确保至少保留一个连接。
    pool_size: int = Field(default=10, ge=1)
    # 限制连接池允许临时扩展的连接数量。
    max_overflow: int = Field(default=20, ge=0)
    # 设置等待可用连接的最长秒数。
    pool_timeout: int = Field(default=30, ge=1)
    # 定期回收长时间存活的连接，降低服务端断连风险。
    pool_recycle: int = Field(default=1800, ge=1)
    # 默认关闭 SQL 输出，避免生产日志泄露查询细节。
    echo: bool = False


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """读取并校验数据库配置。

    Returns:
        缓存的数据库配置对象。

    Raises:
        RuntimeError: 数据库配置段不存在或未启用。
        pydantic.ValidationError: 数据库配置内容未通过校验。
    """
    # 通过统一入口读取 database 配置段并缓存校验结果。
    return get_section("database", DatabaseSettings)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """按当前数据库配置创建异步 SQLAlchemy 引擎。

    Returns:
        缓存的异步数据库引擎。
    """
    # 延迟读取数据库配置，避免未使用数据库时创建资源。
    settings = get_database_settings()
    # 使用连接池参数构建异步引擎，并在取连接前检测连接有效性。
    return create_async_engine(
        settings.url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout,
        pool_recycle=settings.pool_recycle,
        pool_pre_ping=True,
        echo=settings.echo,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """创建绑定当前数据库引擎的异步会话工厂。

    Returns:
        缓存的异步会话工厂。
    """
    # 绑定缓存引擎，并保留提交后对象属性供调用方继续读取。
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """为单次调用提供自动关闭的异步数据库会话。

    Yields:
        新建的异步数据库会话。
    """
    # 从会话工厂创建会话，并在请求结束后自动关闭。
    async with get_session_factory()() as session:
        # 将当前会话交给 FastAPI 依赖调用方使用。
        yield session


async def dispose_engine() -> None:
    """释放已创建的数据库引擎并清除相关缓存。

    Returns:
        None.
    """
    # 仅在引擎已经创建时释放资源，避免关闭流程反向初始化引擎。
    if get_engine.cache_info().currsize:
        # 释放连接池持有的全部数据库连接。
        await get_engine().dispose()
    # 清除会话工厂缓存，防止继续引用已释放引擎。
    get_session_factory.cache_clear()
    # 清除引擎缓存，允许后续按最新配置重新创建。
    get_engine.cache_clear()
    # 清除数据库配置缓存，与资源生命周期保持一致。
    get_database_settings.cache_clear()
