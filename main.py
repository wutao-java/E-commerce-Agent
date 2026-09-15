"""电子商务 Agent 服务的命令行启动入口。"""

# 导入 ASGI 服务器运行器。
import uvicorn

# 导入统一的应用配置读取接口。
from config import get_settings


def main() -> None:
    """读取服务配置并启动 Uvicorn 服务器。

    Returns:
        None.
    """
    # 读取服务监听地址、端口和日志级别。
    settings = get_settings()
    # 使用模块路径启动 FastAPI 应用。
    uvicorn.run(
        "web.app:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
    )


# 仅在直接执行本脚本时启动服务，作为模块导入时不产生副作用。
if __name__ == "__main__":
    # 调用统一的命令行入口。
    main()
