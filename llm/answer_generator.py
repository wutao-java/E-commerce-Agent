"""封装基于完整 Prompt 上下文的客服回答生成。"""

from __future__ import annotations

from collections.abc import Callable
import logging
import time
from typing import Any

from pydantic import BaseModel

from cost.observer import parse_model_usage
from domain import TokenUsage

from .client import call_chat_model, extract_assistant_message

AnswerModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]
logger = logging.getLogger(__name__)


class GroundedAnswerResult(BaseModel):
    """记录回答、模型用量、实际调用情况和兜底原因。"""

    answer: str
    usage: TokenUsage | None = None
    used_model: bool = False
    fallback_reason: str | None = None


def compose_grounded_answer(
    *,
    messages: list[dict[str, str]],
    deterministic_answer: str,
    model_call: AnswerModelCall = call_chat_model,
) -> GroundedAnswerResult:
    """用完整 Prompt 调用模型；运行时失败则返回预先生成的安全话术。"""

    started_at = time.perf_counter()
    try:
        model_response = model_call(messages)
        answer = extract_assistant_message(model_response)
    except RuntimeError as exc:
        # 模型请求失败或回答为空时，上层仍能返回确定性的安全话术。
        logger.warning(
            "Answer model fallback reason=model_unavailable error_type=%s duration_ms=%.2f",
            type(exc).__name__,
            (time.perf_counter() - started_at) * 1000,
        )
        return GroundedAnswerResult(
            answer=deterministic_answer,
            fallback_reason="model_unavailable",
        )

    result = GroundedAnswerResult(
        answer=answer,
        # 模型客户端已返回完整响应，这里只提取可选 usage，不改变调用协议。
        usage=parse_model_usage(model_response),
        used_model=True,
    )
    logger.info(
        "Answer model completed usage_available=%s duration_ms=%.2f",
        result.usage is not None,
        (time.perf_counter() - started_at) * 1000,
    )
    return result
