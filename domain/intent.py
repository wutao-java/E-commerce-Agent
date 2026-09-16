"""定义客服意图识别的内部业务模型。"""

from typing import Literal

from pydantic import BaseModel, Field

IntentSource = Literal["rules", "classifier", "rules_fallback"]
Intent = Literal[
    "general_chat",
    "promotion_consult",
    "product_consult",
    "order_query",
    "refund_request",
    "complaint",
    "unknown",
]


class IntentResult(BaseModel):
    """保存经过校验的意图识别结果。"""

    intent: Intent
    source: IntentSource
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: list[str] = Field(default_factory=list)
    explanation: str
