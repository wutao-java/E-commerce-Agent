"""定义聊天接口的请求和响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from domain import Intent, IntentResult

ReasoningView = Literal["default", "off", "summary", "teaching"]


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


class ChatResponse(BaseModel):
    """定义带结构化意图的客服聊天响应。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    reasoning_summary: list[str]
    session_state: dict[str, Any]
