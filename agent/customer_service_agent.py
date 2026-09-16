"""编排客服请求、意图识别和回答生成。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from cost.observer import build_cost_summary
from domain import ChatCommand, ChatResult
from llm import (
    call_chat_model,
    classify_intent_with_model,
    compose_grounded_answer,
)
from prompts import (
    load_prompt_registry,
    render_prompt_template,
    select_prompt_fragments,
)

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
        self._cost_event_count_by_session: dict[str, int] = {}
        self._session_lock = Lock()

    def chat(self, command: ChatCommand) -> ChatResult:
        """按粗意图选择 Prompt 片段并生成客服回答。"""

        # 计数只记录当前进程内该会话的调用次数，不代表持久化会话状态。
        message_count = self._increment_message_count(
            command.session_id
        )

        intent_result = classify_intent(
            command.user_message,
            classifier_call=self._classifier_call,
        )
        # 先准备不依赖外部事实的安全话术，供回答模型不可用时使用。
        fallback_answer = build_fallback_answer(intent_result)

        # 只将匹配本轮意图的启用片段送入模型，避免每轮注入全部规则。
        registry = load_prompt_registry()
        fragments = select_prompt_fragments(
            intent_result.intent,
            registry,
        )
        messages = render_prompt_template(
            command,
            intent_result,
            fragments,
        )

        model_answer = compose_grounded_answer(
            messages=messages,
            deterministic_answer=fallback_answer,
            model_call=self._model_call,
        )

        # 只观察回答模型实际收到的 Prompt；独立意图分类调用不计入本轮摘要。
        cost_summary = build_cost_summary(
            messages,
            model_answer.answer,
            model_answer.usage,
        )
        with self._session_lock:
            event_count = (
                self._cost_event_count_by_session.get(command.session_id, 0)
                + 1
            )
            self._cost_event_count_by_session[command.session_id] = event_count

        # 只保留事件计数；latest 随本轮响应返回，不在内存里无限累积历史。
        cost_event = {
            "message_count": message_count,
            "intent": intent_result.intent,
            "selected_fragment_ids": [
                fragment.fragment_id
                for fragment in fragments
            ],
            "cost_summary": cost_summary.model_dump(),
        }

        # 这些摘要描述实际处理步骤，并非暴露模型的内部推理过程。
        return ChatResult(
            session_id=command.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            cost_summary=cost_summary,
            reasoning_summary=[
                (
                    "后端先识别粗意图，再从 Prompt Registry "
                    "选择当前问题需要的片段。"
                ),
                (
                    f"本轮加载 {len(fragments)} 个 Prompt 片段，"
                    "并按 priority 从高到低渲染。"
                ),
                (
                    "这一版仍然是 Prompt 方案，只是把整面规则墙"
                    "拆成可管理的片段。"
                ),
                (
                    f"本轮回答模型 token 来源为 {cost_summary.token_source}，"
                    f"总 token 为 {cost_summary.total_tokens}。"
                ),
            ],
            session_state={
                "agent_version": "lesson-07-token-cost-observation",
                "message_count": message_count,
                "runtime_context": {
                    "user_id": command.runtime_user_id,
                    "nickname": command.runtime_nickname,
                    "member_level": command.runtime_member_level,
                    "risk_level": command.runtime_risk_level,
                    "page_context": dict(
                        command.runtime_context or {}
                    ),
                },
                "model_answer": model_answer.model_dump(),
                "prompt_registry": {
                    "template": "customer_service_v1",
                    "selected_fragment_ids": [
                        fragment.fragment_id
                        for fragment in fragments
                    ],
                    "selected_fragment_count": len(fragments),
                    "priorities": [
                        fragment.priority
                        for fragment in fragments
                    ],
                    "disabled_fragment_ids": [
                        fragment.fragment_id
                        for fragment in registry
                        if not fragment.enabled
                    ],
                },
                "cost_log": {
                    "event_count": event_count,
                    "latest": cost_event,
                },
                "next_gap": (
                    "Prompt 片段更好维护，但每轮仍有 token 成本；"
                    "后续需要减少无关上下文。"
                ),
            },
        )

    def _increment_message_count(self, session_id: str) -> int:
        """安全地递增单进程会话消息计数。"""

        with self._session_lock:
            message_count = (
                self._message_count_by_session.get(session_id, 0)
                + 1
            )
            self._message_count_by_session[session_id] = message_count
            return message_count
