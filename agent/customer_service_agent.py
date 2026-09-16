"""编排客服请求、意图识别和回答生成。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from domain import ChatCommand, ChatResult
from llm import call_chat_model, classify_intent_with_model, compose_grounded_answer

from .fallback_answers import build_fallback_answer
from .intent_service import ClassifierCall, classify_intent

ModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]


class CustomerServiceAgent:
    """协调客服能力并返回与传输协议无关的处理结果。"""

    def __init__(
        self,
        model_call: ModelCall = call_chat_model,
        classifier_call: ClassifierCall = classify_intent_with_model,
    ) -> None:
        self._model_call = model_call
        self._classifier_call = classifier_call
        self._message_count_by_session: dict[str, int] = {}
        self._session_lock = Lock()

    def chat(self, command: ChatCommand) -> ChatResult:
        """处理一次客服聊天命令。"""

        message_count = self._increment_message_count(command.session_id)
        intent_result = classify_intent(
            command.user_message,
            classifier_call=self._classifier_call,
        )
        fallback_answer = build_fallback_answer(intent_result)
        model_answer = compose_grounded_answer(
            user_message=command.user_message,
            deterministic_answer=fallback_answer,
            intent_result=intent_result,
            model_call=self._model_call,
        )

        return ChatResult(
            session_id=command.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            reasoning_summary=[
                "API 层校验 ChatRequest，并转换为内部 ChatCommand。",
                "使用规则优先、分类模型兜底的方式识别粗意图。",
                "分类结果只用于分拣，不执行退款、赔偿或人工流转。",
                "内部 ChatResult 由 API 层转换为 ChatResponse。",
            ],
            session_state={
                "agent_version": "lesson-04-intent-structured-output",
                "message_count": message_count,
                "runtime_context": {
                    "user_id": command.runtime_user_id,
                    "nickname": command.runtime_nickname,
                    "member_level": command.runtime_member_level,
                    "risk_level": command.runtime_risk_level,
                    "page_context": dict(command.runtime_context or {}),
                },
                "model_answer": model_answer.model_dump(),
                "next_gap": (
                    "系统知道消息大类，不代表已经知道下一步该怎么处理。"
                ),
            },
        )

    def _increment_message_count(self, session_id: str) -> int:
        """安全地递增单进程会话消息计数。"""

        with self._session_lock:
            message_count = self._message_count_by_session.get(session_id, 0) + 1
            self._message_count_by_session[session_id] = message_count
            return message_count
