"""定义实时业务事实查询的内部契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


BusinessFactKind = Literal[
    "order",
    "logistics",
    "product",
    "unknown",
]


class BusinessFactNeed(BaseModel):
    """描述本轮是否需要查询实时业务事实。"""

    kind: BusinessFactKind
    requires_realtime: bool
    order_no: str | None = None
    sku: str | None = None
    reason: str


class BusinessFactResult(BaseModel):
    """保存业务后端返回的可信事实及查询状态。"""

    source: str = "ecommerce_backend"
    found: bool
    summary: str
    facts: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None