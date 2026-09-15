"""对外暴露系统健康检查路由。"""

# 导入默认的系统健康检查路由器。
from .health import router

# 明确限制路由包的公共导出符号。
__all__ = ["router"]
