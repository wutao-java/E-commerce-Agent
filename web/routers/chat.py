"""提供 Agent 聊天和能力声明接口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
import os
import secrets
from fastapi import Header
from fastapi import APIRouter, HTTPException, status
from config.capabilities import load_agent_capabilities
from config.rag import get_rag_settings
from domain import ChatCommand, ChatResult
from web.schema import ChatRequest, ChatResponse


class ChatAgent(Protocol):
    """定义聊天路由所依赖的最小 Agent 能力。"""

    def chat(self, command: ChatCommand) -> ChatResult:
        """处理聊天请求。"""


AgentProvider = Callable[[], ChatAgent]


def create_chat_router(agent_provider: AgentProvider) -> APIRouter:
    """通过注入 Agent 提供者创建路由，便于应用复用实例和测试替换。"""

    router = APIRouter(tags=["agent"])

    @router.get("/capabilities")
    def capabilities() -> dict:
        """返回当前 Agent 能力边界。"""
        capabilities = load_agent_capabilities().copy()
        capabilities["features"] = {**capabilities["features"], "rag_citations": get_rag_settings().enabled}
        return capabilities

    @router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
    def chat(request: ChatRequest, x_agent_service_token: str | None = Header(default=None)) -> ChatResponse:
        """将 HTTP 请求转换为内部命令，再把处理结果校验为响应。"""

        expected = os.getenv("AGENT_SERVICE_TOKEN", "")
        if not expected:
               raise HTTPException(status_code=503, detail="Agent 网关令牌未配置")
        if not x_agent_service_token or not secrets.compare_digest(
            x_agent_service_token, expected):
            raise HTTPException(status_code=401, detail="未经授权的 Agent 调用")
        try:
            # 当前只转交 Agent 所需字段；展示级别和 debug 尚未参与处理。
            command = ChatCommand(
                session_id=request.session_id,
                runtime_user_id=request.runtime_user_id,
                runtime_nickname=request.runtime_nickname,
                runtime_member_level=request.runtime_member_level,
                runtime_risk_level=request.runtime_risk_level,
                runtime_account_id=request.runtime_account_id,
                user_message=request.user_message,
                runtime_context=request.runtime_context,
            )
            result = agent_provider().chat(command)
            return ChatResponse.model_validate(result.model_dump())
        except RuntimeError as exc:
            # 业务流程抛出的 RuntimeError 统一映射为 503，其余异常交给全局处理。
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    return router
