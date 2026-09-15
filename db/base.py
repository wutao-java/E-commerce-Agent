"""定义 SQLAlchemy 声明式模型基类。"""

# 导入 SQLAlchemy 2.x 声明式基类。
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """作为所有 ORM 实体模型的声明式基类。"""

    # 基类仅承载统一元数据，不增加额外行为。
    pass
