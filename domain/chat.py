"""定义客服 Agent 的内部输入和输出契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .intent import Intent, IntentResult


class ChatCommand(BaseModel):
    """表示一次已经通过接口校验的客服请求。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str
    runtime_user_id: str
    runtime_nickname: str | None = None
    runtime_member_level: str | None = None
    runtime_risk_level: str | None = None
    user_message: str
    runtime_context: dict[str, Any] | None = None


class ChatResult(BaseModel):
    """表示客服 Agent 与传输协议无关的处理结果。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    reasoning_summary: list[str]
    session_state: dict[str, Any]
