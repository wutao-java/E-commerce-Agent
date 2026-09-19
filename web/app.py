"""创建并配置 FastAPI 应用实例。"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import CustomerServiceAgent
from config import configure_logging, get_settings
from config.database import dispose_engine
from web.routers.chat import AgentProvider, create_chat_router
from web.routers.health import router as health_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """初始化日志，关闭时释放已创建的数据库资源。"""

    settings = get_settings()
    configure_logging(
        settings.server.log_level,
        settings.server.log_file,
    )
    started_at = time.perf_counter()
    logger.info(
        "Application started service=%s host=%s port=%d log_level=%s file_logging=%s",
        settings.server.name,
        settings.server.host,
        settings.server.port,
        settings.server.log_level.upper(),
        bool(settings.server.log_file),
    )

    try:
        yield
    finally:
        logger.info("Application shutdown started service=%s", settings.server.name)
        await dispose_engine()
        logger.info(
            "Application stopped service=%s uptime_seconds=%.2f",
            settings.server.name,
            time.perf_counter() - started_at,
        )


def create_app(agent_provider: AgentProvider | None = None) -> FastAPI:
    """创建 FastAPI 应用并注册全部路由。"""

    settings = get_settings()

    application = FastAPI(
        title=settings.server.name,
        lifespan=lifespan,
    )

    if agent_provider is None:
        # 默认在应用生命周期内复用同一 Agent，以保留进程内会话计数。
        default_agent = CustomerServiceAgent()
        agent_provider = lambda: default_agent

    application.include_router(health_router)
    application.include_router(
        create_chat_router(agent_provider)
    )

    @application.exception_handler(Exception)
    async def system_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """记录未捕获错误，但不把内部异常详情返回给客户端。"""

        logger.error(
            "Unhandled request error method=%s path=%s error_type=%s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


app = create_app()
