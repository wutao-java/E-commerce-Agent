"""封装基于完整 Prompt 上下文的客服回答生成。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .client import call_chat_model, extract_assistant_message

AnswerModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]


class GroundedAnswerResult(BaseModel):
    """记录实际返回的回答、模型使用情况和兜底原因。"""

    answer: str
    used_model: bool = False
    fallback_reason: str | None = None


def compose_grounded_answer(
    *,
    messages: list[dict[str, str]],
    deterministic_answer: str,
    model_call: AnswerModelCall = call_chat_model,
) -> GroundedAnswerResult:
    """用完整 Prompt 调用模型；运行时失败则返回预先生成的安全话术。"""

    try:
        model_response = model_call(messages)
        answer = extract_assistant_message(model_response)
    except RuntimeError:
        # 模型请求失败或回答为空时，上层仍能返回确定性的安全话术。
        return GroundedAnswerResult(
            answer=deterministic_answer,
            fallback_reason="model_unavailable",
        )

    return GroundedAnswerResult(
        answer=answer,
        used_model=True,
    )
