"""定义客服 Agent 的内部输入和输出契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .cost import CostSummary
from .intent import Intent, IntentResult
from .rag import Citation


class ChatCommand(BaseModel):
    """表示与 HTTP 协议解耦的客服输入，供 Agent 编排使用。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str
    runtime_user_id: str
    runtime_nickname: str | None = None
    runtime_member_level: str | None = None
    runtime_risk_level: str | None = None
    user_message: str
    runtime_context: dict[str, Any] | None = None
    runtime_account_id: int | None = None


class ChatResult(BaseModel):
    """保存回答、意图、成本摘要和本轮状态，供接口层转换。"""

    session_id: str
    answer: str
    intent: Intent
    intent_result: IntentResult
    cost_summary: CostSummary
    reasoning_summary: list[str]
    session_state: dict[str, Any]
    citations: list[Citation] | None = None
