"""编排客服请求、意图识别和回答生成。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from domain import ChatCommand, ChatResult
from llm import call_chat_model, classify_intent_with_model, compose_grounded_answer

from .fallback_answers import build_fallback_answer
from .intent_service import ClassifierCall, classify_intent
from prompts import FULL_POLICY_DOCUMENTS, build_full_context_messages, detect_context_conflicts,estimate_tokens

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

        conflicts = detect_context_conflicts(command.user_message)
        messages = build_full_context_messages(
            command,
            intent_result,
            FULL_POLICY_DOCUMENTS,
            conflicts,
        )
        prompt_text = "\n".join(message["content"] for message in messages)

        model_answer = compose_grounded_answer(
            messages=messages,
            deterministic_answer=fallback_answer,
            model_call=self._model_call,
        )

        return ChatResult(
            session_id=command.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            reasoning_summary=[
                "后端沿用第 04 课的结构化粗意图识别。",
                "system prompt 写明客服身份、事实优先级和回答边界。",
                (
                    f"本轮把 {len(FULL_POLICY_DOCUMENTS)} 份规则文档"
                    "全量注入 Prompt。"
                ),
                (
                    f"本轮检测到 {len(conflicts)} 条上下文冲突线索；"
                    "这些线索只用于观察，不会自动裁决规则。"
                ),
            ],
            session_state={
                "agent_version": "lesson-05-prompt-boundary-full-context",
                "message_count": message_count,
                "runtime_context": {
                    "user_id": command.runtime_user_id,
                    "nickname": command.runtime_nickname,
                    "member_level": command.runtime_member_level,
                    "risk_level": command.runtime_risk_level,
                    "page_context": dict(command.runtime_context or {}),
                },
                "model_answer": model_answer.model_dump(),
                "prompt_boundary": {
                    "mode": (
                        "system_prompt_fact_priority_and_refusal_rules"
                    ),
                    "fact_priority": [
                        "runtime_facts",
                        "current_policy_documents",
                        "legacy_documents",
                        "user_claims",
                        "model_general_knowledge",
                    ],
                    "boundary_rule_count": 4,
                },
                "prompt_context": {
                    "mode": "full_document_injection",
                    "document_count": len(FULL_POLICY_DOCUMENTS),
                    "document_ids": [
                        document.doc_id
                        for document in FULL_POLICY_DOCUMENTS
                    ],
                    "estimated_prompt_tokens": estimate_tokens(
                        prompt_text
                    ),
                    "conflict_count": len(conflicts),
                    "conflicts": [
                        conflict.model_dump()
                        for conflict in conflicts
                    ],
                },
                "next_gap": (
                    "system prompt 能限制乱承诺，全量 Prompt 能让规则"
                    "进入模型，但当前规则和历史规则仍会相互干扰。"
                ),
            },
        )

    def _increment_message_count(self, session_id: str) -> int:
        """安全地递增单进程会话消息计数。"""

        with self._session_lock:
            message_count = self._message_count_by_session.get(session_id, 0) + 1
            self._message_count_by_session[session_id] = message_count
            return message_count
