"""创建并配置 FastAPI 应用实例。"""

# 导入日志和异步生命周期上下文管理能力。
import logging
from contextlib import asynccontextmanager

# 导入 FastAPI 应用、请求和 JSON 响应类型。
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 导入配置、资源清理和健康检查路由。
from config import configure_logging, get_settings
from config.database import dispose_engine
from web.routers.health import router as health_router


# 创建当前模块专用日志记录器。
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """管理应用启动时的日志配置和关闭时的资源释放。

    Args:
        _app: 当前 FastAPI 应用实例。

    Yields:
        应用运行阶段的控制权。
    """
    # 读取当前服务日志配置。
    settings = get_settings()
    # 在开始处理请求前完成日志系统初始化。
    configure_logging(settings.server.log_level, settings.server.log_file)
    # 确保应用退出路径始终进入资源清理阶段。
    try:
        # 将控制权交还 FastAPI 运行请求处理。
        yield
    # 无论正常关闭还是异常退出都释放数据库资源。
    finally:
        # 关闭已创建的数据库连接池并清除缓存。
        await dispose_engine()


def create_app() -> FastAPI:
    """创建包含路由和全局异常处理器的 FastAPI 应用。

    Returns:
        配置完成的 FastAPI 应用实例。
    """
    # 读取服务名称等应用级配置。
    settings = get_settings()
    # 创建绑定统一生命周期管理器的 FastAPI 应用。
    application = FastAPI(title=settings.server.name, lifespan=lifespan)
    # 注册无需业务鉴权的系统健康检查路由。
    application.include_router(health_router)

    @application.exception_handler(Exception)
    async def system_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        """记录未处理异常并返回统一的服务端错误响应。

        Args:
            _request: 触发异常的 HTTP 请求。
            exc: 请求处理过程中未捕获的异常。

        Returns:
            不暴露内部错误细节的 HTTP 500 JSON 响应。
        """
        # 在服务端记录完整异常堆栈供排障使用。
        logger.exception("Unhandled request error", exc_info=exc)
        # 对客户端隐藏内部实现和异常细节。
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # 返回完成全部路由与异常处理配置的应用。
    return application


# 在模块加载时创建供 ASGI 服务器使用的应用实例。
app = create_app()
