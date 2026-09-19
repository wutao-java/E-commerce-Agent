"""提供 Agent 聊天和能力声明接口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Protocol

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config.capabilities import load_agent_capabilities
from config.rag import get_rag_settings
from domain import ChatCommand, ChatResult
from security.jwt import decode_token, get_jwt_settings
from web.schema import ChatRequest, ChatResponse


class ChatAgent(Protocol):
    """定义聊天路由所依赖的最小 Agent 能力。"""

    def chat(self, command: ChatCommand) -> ChatResult:
        """处理聊天请求。"""


AgentProvider = Callable[[], ChatAgent]
bearer_scheme = HTTPBearer(auto_error=False)


def require_agent_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> dict[str, Any]:
    """验证内部 Bearer JWT，并返回可信声明。"""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Agent Bearer 凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = decode_token(credentials.credentials)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent JWT 公钥未配置",
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Agent Bearer 凭证",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    scopes = claims.get("scope", "")
    granted = set(scopes if isinstance(scopes, list) else str(scopes).split())
    if get_jwt_settings().required_scope not in granted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent Bearer 凭证权限不足",
        )
    try:
        int(claims["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Agent Bearer 凭证缺少有效用户身份",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return claims


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
    def chat(
        request: ChatRequest,
        principal: Annotated[dict[str, Any], Depends(require_agent_principal)],
    ) -> ChatResponse:
        """将 HTTP 请求转换为内部命令，再把处理结果校验为响应。"""

        try:
            account_id = int(principal["sub"])
            command = ChatCommand(
                session_id=request.session_id,
                runtime_user_id=str(
                    principal.get("business_user_id") or f"U{account_id}"
                ),
                runtime_nickname=principal.get("nickname"),
                runtime_member_level=principal.get("member_level"),
                runtime_risk_level=principal.get("risk_level"),
                runtime_account_id=account_id,
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
