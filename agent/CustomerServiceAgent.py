# 延迟解析类型注解，避免运行时立即求值。
from __future__ import annotations

# 导入可调用对象的类型定义。
from collections.abc import Callable
from threading import Lock
from typing import Any
from model import call_chat_model, classify_intent_with_model,compose_grounded_answer
from web.schema import ChatRequest, ChatResponse, IntentResult
from .intent_rules import plan_intent_by_rules

# 定义接收消息列表并返回模型响应字典的函数类型。
ModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]
ClassifierCall = Callable[[str], IntentResult | None]

def classify_intent(user_message: str,classifier_call: ClassifierCall = classify_intent_with_model) -> IntentResult:
    """按照规则优先、分类模型兜底的顺序识别意图。"""

    rule_result = plan_intent_by_rules(user_message)

    if rule_result is not None:
        return rule_result

    model_result = classifier_call(user_message)

    if model_result is not None:
        return model_result

    return IntentResult(
        intent="unknown",
        source="rules_fallback",
        confidence=0.3,
        matched_keywords=[],
        explanation=(
            "规则没有高置信命中，分类模型也不可用或输出无效，"
            "先标记为 unknown。"
        ),
    )


def build_answer(intent_result: IntentResult) -> str:
    """根据粗意图生成不越过业务边界的安全回答。"""

    if intent_result.intent == "complaint":
        return (
            "我已经先把这条消息识别为投诉类问题。"
            "当前版本还没有接入人工流转和赔偿处理，"
            "不能直接承诺处理结果。"
        )

    if intent_result.intent == "refund_request":
        return (
            "我已经先把这条消息识别为退款或售后类问题。"
            "当前版本还没有接入售后规则和订单状态，"
            "不能直接判断是否可退。"
        )

    if intent_result.intent == "order_query":
        return (
            "我已经先把这条消息识别为订单或物流查询。"
            "当前版本还没有接入订单工具，不能编造物流节点。"
        )

    if intent_result.intent == "promotion_consult":
        return (
            "我已经先把这条消息识别为优惠活动咨询。"
            "当前版本还没有接入活动规则，不能承诺具体优惠。"
        )

    if intent_result.intent == "product_consult":
        return (
            "我已经先把这条消息识别为商品咨询。"
            "当前版本还没有接入产品知识库，不能编造商品卖点。"
        )

    if intent_result.intent == "general_chat":
        return (
            "你好，我是小哲电商客服 Agent。"
            "现在我已经能把用户问题先分到一个粗意图里。"
        )

    return "我还不能确定这条消息属于哪类客服问题，只能先标记为 unknown。"


# 定义客服 Agent 的业务编排类。
class CustomerServiceAgent:
    """编排客服请求、模型调用和响应组装。"""

    def __init__(self, model_call: ModelCall = call_chat_model,classifier_call: ClassifierCall = classify_intent_with_model) -> None:
        # 保存模型调用函数供聊天流程使用。
        self._model_call = model_call
        # 保存规则不确定时使用的分类调用函数。
        self._classifier_call = classifier_call
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

        # 规则优先，规则不确定时调用分类模型。
        intent_result = classify_intent(
            request.user_message,
            classifier_call=self._classifier_call,
        )

        # 先生成不会越过业务边界的确定性回答。
        deterministic_answer = build_answer(intent_result)

        # 再让回答模型根据结构化意图组织客服话术。
        model_answer = compose_grounded_answer(
            user_message=request.user_message,
            deterministic_answer=deterministic_answer,
            intent_result=intent_result,
            model_call=self._model_call,
        )

        # 推理总结
        reasoning_summary = [
            "后端接收 ChatRequest，保持 user_message 与 runtime_* 分离。",
            "使用规则优先、分类模型兜底的方式识别粗意图。",
            "分类结果只用于分拣，不执行退款、赔偿或人工流转。",
            "IntentResult 经过 Pydantic 校验后进入 ChatResponse。",
        ]
        session_state = {
            "agent_version": "lesson-04-intent-structured-output",
            "message_count": message_count,
            "runtime_context": {
                "user_id": request.runtime_user_id,
                "nickname": request.runtime_nickname,
                "member_level": request.runtime_member_level,
                "risk_level": request.runtime_risk_level,
                "page_context": dict(request.runtime_context or {}),
            },
            "model_answer": model_answer.model_dump(),
            "next_gap": (
                "系统知道消息大类，不代表已经知道下一步该怎么处理。"
            ),
        }

        return ChatResponse(
            session_id=request.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            reasoning_summary=reasoning_summary,
            session_state=session_state,
        )

    def _increment_message_count(self, session_id: str) -> int:
        """安全地递增单进程会话消息计数。"""

        # 加锁以避免并发请求造成计数竞争。
        with self._session_lock:
            # 读取旧计数并递增一次。
            message_count = (self._message_count_by_session.get(session_id, 0) + 1)
            # 保存更新后的会话计数。
            self._message_count_by_session[session_id] = message_count
            return message_count
