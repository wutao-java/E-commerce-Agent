"""对外暴露 FastAPI 应用实例及其工厂函数。"""

# 导入已创建的 ASGI 应用及应用工厂。
from .app import app, create_app

# 明确限制 Web 包的公共导出符号。
__all__ = ["app", "create_app"]
