"""封装基于结构化意图的客服回答生成。"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any
from pydantic import BaseModel
from web.schema import IntentResult
from .llm_client import call_chat_model, extract_assistant_message


AnswerModelCall = Callable[[list[dict[str, str]]],dict[str, Any]]

class GroundedAnswerResult(BaseModel):
    """记录最终回答是否由真实模型生成。"""

    answer: str
    used_model: bool = False
    fallback_reason: str | None = None


def compose_grounded_answer(
    *,
    user_message: str,
    deterministic_answer: str,
    intent_result: IntentResult,
    model_call: AnswerModelCall = call_chat_model,
) -> GroundedAnswerResult:
    """生成受意图约束的回答，模型失败时使用安全话术。"""

    payload = {
        "user_message": user_message,
        "intent_result": intent_result.model_dump(mode="json"),
        "fallback_answer": deterministic_answer,
        "lesson_boundary": (
            "当前只完成粗意图识别，不能承诺订单、规则、"
            "退款、赔偿或人工流转结果。"
        ),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是小哲电商公司的客服 Agent。"
                "只能根据结构化 intent_result 和课程边界回复；"
                "不得编造订单、物流、活动规则、退款资格、"
                "赔偿或人工流转结果。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    try:
        model_response = model_call(messages)
        answer = extract_assistant_message(model_response)
    except RuntimeError:
        return GroundedAnswerResult(
            answer=deterministic_answer,
            fallback_reason="model_unavailable",
        )

    return GroundedAnswerResult(
        answer=answer,
        used_model=True,
    )