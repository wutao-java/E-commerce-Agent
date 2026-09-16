"""定义聊天接口的请求和响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ReasoningView = Literal["default", "off", "summary", "teaching"]  # 限定推理内容的展示模式。
IntentSource = Literal["rules", "classifier", "rules_fallback"]  # 限定意图识别结果的来源。
Intent = Literal[
    "general_chat",  # 通用闲聊。
    "promotion_consult",  # 优惠活动咨询。
    "product_consult",  # 商品信息咨询。
    "order_query",  # 订单信息查询。
    "refund_request",  # 退款申请。
    "complaint",  # 用户投诉。
    "unknown",  # 无法识别的意图。
]


class ChatRequest(BaseModel):
    """定义客服聊天请求。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(
        min_length=1,
        description="当前对话会话 ID",
    )
    runtime_user_id: str = Field(
        min_length=1,
        description="系统侧提供的用户 ID",
    )
    runtime_nickname: str | None = Field(
        default=None,
        description="系统侧提供的用户昵称",
    )
    runtime_member_level: str | None = Field(
        default=None,
        description="系统侧提供的会员等级",
    )
    runtime_risk_level: str | None = Field(
        default=None,
        description="系统侧提供的风险等级",
    )
    user_message: str = Field(
        min_length=1,
        description="用户输入的原始消息",
    )
    reasoning_view: ReasoningView = Field(
        default="default",
        description="调试后台请求的公开推理展示级别",
    )
    debug: bool = Field(
        default=True,
        description="是否启用调试观察信息",
    )
    runtime_context: dict[str, Any] | None = Field(
        default=None,
        description="页面等运行时上下文",
    )


class IntentResult(BaseModel):  # 声明意图识别结果的数据模型。
    """定义系统可读的意图识别结果。"""

    intent: Intent  # 保存识别出的意图类型。
    source: IntentSource  # 保存意图识别结果的来源。
    confidence: float = Field(ge=0.0, le=1.0)  # 保存取值范围为 0 到 1 的置信度。
    matched_keywords: list[str] = Field(default_factory=list)  # 保存命中的关键词，默认使用空列表。
    explanation: str  # 保存意图识别结果的说明。


class ChatResponse(BaseModel):
    """定义带结构化意图的客服聊天响应。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    reasoning_summary: list[str]
    session_state: dict[str, Any]
