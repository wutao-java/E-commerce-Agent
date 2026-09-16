# 说明当前模块负责客服 Agent 的基础编排。
"""提供客服 Agent 的最小编排逻辑。"""

# 延迟解析类型注解，避免运行时立即求值。
from __future__ import annotations

# 导入可调用对象的类型定义。
from collections.abc import Callable
# 导入线程锁，保护共享会话计数。
from threading import Lock
from typing import Any
from model import build_messages, call_chat_model, extract_assistant_message
from web.schema import ChatRequest, ChatResponse
# 定义接收消息列表并返回模型响应字典的函数类型。
ModelCall = Callable[[list[dict[str, str]]],dict[str, Any],]


# 定义客服 Agent 的业务编排类。
class CustomerServiceAgent:
    """编排客服请求、模型调用和响应组装。"""

    def __init__(self, model_call: ModelCall = call_chat_model, ) -> None:
        # 保存模型调用函数供聊天流程使用。
        self._model_call = model_call
        # 保存各会话已处理的消息数量。
        self._message_count_by_session: dict[str, int] = {}
        # 创建互斥锁以保证计数更新的线程安全。
        self._session_lock = Lock()

    # 接收聊天请求并返回客服响应。
    def chat(self, request: ChatRequest) -> ChatResponse:
        """处理一次客服聊天请求。"""

        # 递增当前会话的消息计数并取得新值。
        message_count = self._increment_message_count(
            # 将请求中的会话标识传给计数方法。
            request.session_id,
        )

        messages = build_messages(request.user_message)
        model_response = self._model_call(messages)
        answer = extract_assistant_message(model_response)

        return ChatResponse(
            # 回传本次请求所属的会话标识。
            session_id=request.session_id,
            # 写入模型生成的客服答复。
            answer=answer,
            session_state={
                # 标记当前 Agent 的版本。
                "agent_version": "lesson-02-chat-service",
                # 返回当前会话累计处理的消息数。
                "message_count": message_count,
                # 汇总本次请求携带的运行时用户上下文。
                "runtime_context": {
                    # 写入运行时用户标识。
                    "user_id": request.runtime_user_id,
                    # 写入运行时用户昵称。
                    "nickname": request.runtime_nickname,
                    # 写入运行时会员等级。
                    "member_level": request.runtime_member_level,
                    # 写入运行时风险等级。
                    "risk_level": request.runtime_risk_level,
                    # 将页面上下文规范化为普通字典。
                    "page_context": dict(
                        request.runtime_context or {}  # 上下文缺失时使用空字典。
                    ),
                },
            },
        )

    def _increment_message_count(self, session_id: str, ) -> int:
        """安全地递增单进程会话消息计数。"""

        # 加锁以避免并发请求造成计数竞争。
        with self._session_lock:
            # 读取旧计数并递增一次。
            message_count = (self._message_count_by_session.get(session_id, 0) + 1)
            # 保存更新后的会话计数。
            self._message_count_by_session[session_id] = message_count
            return message_count

