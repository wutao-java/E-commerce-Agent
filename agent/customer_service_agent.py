"""编排客服请求、意图识别和回答生成。"""

from __future__ import annotations

from collections.abc import Callable
import logging
from threading import Lock
import time
from typing import Any

from config.rag import get_rag_settings
from cost.observer import build_cost_summary
from domain import ChatCommand, ChatResult
from domain.rag import Citation
from llm import (
    GroundedAnswerResult,
    call_chat_model,
    classify_intent_with_model,
    compose_grounded_answer,
)
from prompts import (
    load_prompt_registry,
    render_prompt_template,
    select_prompt_fragments,
)
from rag.prompting import build_citations, render_rag_messages
from rag.service import CourseRagService

from .fallback_answers import build_fallback_answer
from .intent_service import ClassifierCall, classify_intent


ModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]
logger = logging.getLogger(__name__)


class CustomerServiceAgent:
    """协调客服能力并返回与传输协议无关的处理结果。"""

    def __init__(
        self,
        model_call: ModelCall = call_chat_model,
        classifier_call: ClassifierCall = classify_intent_with_model,
        rag_service: CourseRagService | None = None,
    ) -> None:
        self._model_call = model_call
        self._classifier_call = classifier_call
        self._rag_service = rag_service or (CourseRagService() if get_rag_settings().enabled else None)
        self._message_count_by_session: dict[str, int] = {}
        self._cost_event_count_by_session: dict[str, int] = {}
        self._session_lock = Lock()

    def chat(self, command: ChatCommand) -> ChatResult:
        """按粗意图选择 Prompt 片段并生成客服回答。"""

        started_at = time.perf_counter()
        # 计数只记录当前进程内该会话的调用次数，不代表持久化会话状态。
        message_count = self._increment_message_count(
            command.session_id
        )
        logger.info(
            "Agent chat started session_id=%s message_count=%d rag_enabled=%s",
            command.session_id,
            message_count,
            self._rag_service is not None,
        )

        intent_result = classify_intent(
            command.user_message,
            classifier_call=self._classifier_call,
        )
        logger.info(
            "Intent classified session_id=%s intent=%s source=%s confidence=%.2f",
            command.session_id,
            intent_result.intent,
            intent_result.source,
            intent_result.confidence,
        )
        # 先准备不依赖外部事实的安全话术，供回答模型不可用时使用。
        fallback_answer = build_fallback_answer(intent_result)

        # 只将匹配本轮意图的启用片段送入模型，避免每轮注入全部规则。
        registry = load_prompt_registry()
        fragments = select_prompt_fragments(
            intent_result.intent,
            registry,
        )
        logger.info(
            "Prompt fragments selected session_id=%s fragment_count=%d fragment_ids=%s",
            command.session_id,
            len(fragments),
            [fragment.fragment_id for fragment in fragments],
        )
        citations: list[Citation] | None = None
        rag_state: dict[str, Any] | None = None
        if self._rag_service is None:
            messages = render_prompt_template(command, intent_result, fragments)
            model_answer = compose_grounded_answer(
                messages=messages,
                deterministic_answer=fallback_answer,
                model_call=self._model_call,
            )
        else:
            # 课程证据独立于 Prompt Registry 的活动规则；仅保留公共身份和安全约束。
            fragments = [fragment for fragment in fragments if "all" in fragment.applies_to]
            try:
                _plan, hits, rag_state = self._rag_service.retrieve(command, intent_result.intent)
            except RuntimeError as exc:
                logger.warning(
                    "RAG retrieval failed session_id=%s error_type=%s",
                    command.session_id,
                    type(exc).__name__,
                    exc_info=True,
                )
                hits = []
                rag_state = {"mode": "milvus_hybrid_retrieval", "fallback_reason": "retrieval_unavailable"}
            logger.info(
                "RAG retrieval completed session_id=%s hit_count=%d cache_hit=%s fallback_reason=%s",
                command.session_id,
                len(hits),
                rag_state.get("cache_hit", False),
                rag_state.get("fallback_reason"),
            )
            if hits:
                messages = render_rag_messages(
                    command, intent_result, hits, fragments,
                    self._rag_service.settings.reference_date().isoformat(),
                )
                model_answer = compose_grounded_answer(
                    messages=messages,
                    deterministic_answer="课程回答模型暂时不可用，不能依据检索结果给出确定结论。",
                    model_call=self._model_call,
                )
                citations = build_citations(hits) if model_answer.used_model else []
            else:
                answer = (
                    fallback_answer if intent_result.intent == "general_chat" else
                    "当前课程知识没有可靠依据，或检索服务暂时不可用；不能给出确定的规则结论。"
                )
                messages = [{"role": "system", "content": "课程知识不足时不得编造规则。"},
                            {"role": "user", "content": command.user_message}]
                model_answer = GroundedAnswerResult(answer=answer, fallback_reason="no_reliable_knowledge")
                citations = []

        logger.info(
            "Answer composed session_id=%s used_model=%s fallback_reason=%s",
            command.session_id,
            model_answer.used_model,
            model_answer.fallback_reason,
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
        result = ChatResult(
            session_id=command.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            cost_summary=cost_summary,
            reasoning_summary=([
                "先识别意图，再从隔离的课程知识库执行改写、混合召回和重排。",
                f"本轮返回 {len(citations or [])} 条可追溯的课程知识引用。",
                "没有可靠知识或检索服务不可用时不生成规则结论。",
            ] if rag_state is not None else [
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
            ]),
            session_state={
                "agent_version": "lesson-16-course-rag-milvus" if rag_state is not None else "lesson-07-token-cost-observation",
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
                **({"rag": rag_state} if rag_state is not None else {}),
                "next_gap": (
                    "课程知识仅供隔离练习，不是现时商城业务事实。"
                    if rag_state is not None else
                    "Prompt 片段更好维护，但每轮仍有 token 成本；"
                    "后续需要减少无关上下文。"
                ),
            },
            citations=citations,
        )
        logger.info(
            "Agent chat completed session_id=%s intent=%s used_model=%s "
            "total_tokens=%d duration_ms=%.2f",
            command.session_id,
            intent_result.intent,
            model_answer.used_model,
            cost_summary.total_tokens,
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def _increment_message_count(self, session_id: str) -> int:
        """安全地递增单进程会话消息计数。"""

        with self._session_lock:
            message_count = (
                self._message_count_by_session.get(session_id, 0)
                + 1
            )
            self._message_count_by_session[session_id] = message_count
            return message_count
