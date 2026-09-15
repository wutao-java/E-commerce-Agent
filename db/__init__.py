"""对外暴露数据库声明式模型基类。"""

# 导入 ORM 实体共享的声明式基类。
from .base import Base

# 明确限制数据库包的公共导出符号。
__all__ = ["Base"]
