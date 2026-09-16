"""封装轻量模型的结构化意图分类。"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, get_args

from pydantic import ValidationError

from config.llm import get_llm_settings
from domain import Intent, IntentResult

from .client import call_chat_model, extract_assistant_message

ClassifierModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]


def build_classifier_messages(user_message: str) -> list[dict[str, str]]:
    """构造只允许返回粗意图 JSON 的分类消息。"""

    allowed_intents = ", ".join(get_args(Intent))
    return [
        {
            "role": "system",
            "content": (
                "你是小哲电商客服 Agent 的轻量意图分类器。"
                "只输出 JSON，不要输出 Markdown。"
                "你只能判断用户消息的大类，不能执行工具、"
                "批准退款或承诺售后动作。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请把下面用户消息分成一个粗意图。\n"
                f"允许的 intent 只能是：{allowed_intents}\n"
                "输出 JSON 格式："
                '{"intent": string, "confidence": number, '
                '"explanation": string}\n\n'
                f"用户消息：{user_message}"
            ),
        },
    ]


def parse_classifier_json(content: str) -> dict[str, Any] | None:
    """解析纯 JSON 或 Markdown 代码块中的 JSON。"""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def call_classifier_model(messages: list[dict[str, str]]) -> dict[str, Any]:
    """使用独立配置的轻量模型完成意图分类。"""

    settings = get_llm_settings()
    return call_chat_model(messages, model=settings.classifier_model)


def classify_intent_with_model(user_message: str,model_call: ClassifierModelCall = call_classifier_model) -> IntentResult | None:
    """调用分类模型并校验结构化输出。"""

    try:
        model_response = model_call(build_classifier_messages(user_message))
        content = extract_assistant_message(model_response)
        payload = parse_classifier_json(content)
        if payload is None:
            return None

        return IntentResult(
            intent=payload.get("intent", "unknown"),
            source="classifier",
            confidence=float(payload.get("confidence", 0.7)),
            matched_keywords=[],
            explanation=str(
                payload.get("explanation")
                or "分类模型给出粗意图兜底。"
            ),
        )
    except (
        RuntimeError,
        TypeError,
        ValueError,
        ValidationError,
    ):
        return None
