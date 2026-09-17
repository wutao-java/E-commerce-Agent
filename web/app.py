"""创建并配置 FastAPI 应用实例。"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import CustomerServiceAgent
from config import configure_logging, get_settings
from config.database import dispose_engine
from config.rag import get_rag_settings
from web.routers.chat import AgentProvider, create_chat_router
from web.routers.health import router as health_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时准备待审核知识快照，关闭时释放已创建的数据库资源。"""

    settings = get_settings()
    configure_logging(
        settings.server.log_level,
        settings.server.log_file,
    )

    try:
        _app.state.rag_prepared_releases = []
        if get_rag_settings().prepare_on_start:
            try:
                from rag.local_ingestion import prepare_startup

                _app.state.rag_prepared_releases = await prepare_startup()
            except Exception as exc:
                # 准备失败不得自动发布，也不影响当前客服服务与旧知识版本。
                _app.state.rag_preparation_error = str(exc)
                logger.exception("RAG startup preparation failed; active release is unchanged")
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
        # 默认在应用生命周期内复用同一 Agent，以保留进程内会话计数。
        default_agent = CustomerServiceAgent()
        agent_provider = lambda: default_agent

    application.include_router(health_router)
    application.include_router(
        create_chat_router(agent_provider)
    )

    @application.exception_handler(Exception)
    async def system_exception_handler(_request: Request,exc: Exception) -> JSONResponse:
        """记录未捕获错误，但不把内部异常详情返回给客户端。"""

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
