"""提供服务健康状态检查接口。"""

# 导入 FastAPI 路由器。
from fastapi import APIRouter

# 导入应用配置读取接口。
from config import get_settings


# 创建系统接口路由器。
router = APIRouter()


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """返回服务健康状态和服务名称。

    Returns:
        包含健康状态与服务名称的响应字典。
    """
    # 读取当前配置中的服务名称。
    settings = get_settings()
    # 返回固定健康状态及实际服务名称。
    return {"status": "ok", "service": settings.server.name}
