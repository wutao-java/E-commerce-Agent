"""定义聊天接口的请求和响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from domain import CostSummary, Intent, IntentResult

# 请求可以携带展示偏好，但当前路由尚未用它改变 Agent 的返回内容。
ReasoningView = Literal["default", "off", "summary", "teaching"]


class ChatRequest(BaseModel):
    """校验 HTTP 输入；runtime_* 字段目前由调用方在请求中提供。"""

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


class ChatResponse(BaseModel):
    """暴露回答、结构化意图、成本摘要与本轮处理状态。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    cost_summary: CostSummary
    reasoning_summary: list[str]
    session_state: dict[str, Any]
