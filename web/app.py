"""创建并配置 FastAPI 应用实例。"""

import logging
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
    """管理应用日志和外部资源生命周期。"""

    settings = get_settings()
    configure_logging(
        settings.server.log_level,
        settings.server.log_file,
    )

    try:
        yield
    finally:
        await dispose_engine()


def create_app(agent_provider: AgentProvider | None = None) -> FastAPI:
    """创建 FastAPI 应用并注册全部路由。"""

    settings = get_settings()

    application = FastAPI(
        title=settings.server.name,
        lifespan=lifespan,
    )

    if agent_provider is None:
        default_agent = CustomerServiceAgent()
        agent_provider = lambda: default_agent

    application.include_router(health_router)
    application.include_router(
        create_chat_router(agent_provider)
    )

    @application.exception_handler(Exception)
    async def system_exception_handler(_request: Request,exc: Exception) -> JSONResponse:
        """处理未捕获的系统异常。"""

        logger.exception(
            "Unhandled request error",
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


app = create_app()