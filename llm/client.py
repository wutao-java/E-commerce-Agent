"""封装 OpenAI-compatible 聊天模型调用。"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from config.llm import get_llm_settings


logger = logging.getLogger(__name__)

DEFAULT_USER_MESSAGE = "我在小哲电商看到降噪耳机有活动，请问优惠能和会员券一起用吗？"

PLACEHOLDER_API_KEYS = {
    "",
    "你的模型平台 Key",
    "your-api-key",
    "YOUR_API_KEY",
}


def build_messages(user_message: str = DEFAULT_USER_MESSAGE) -> list[dict[str, str]]:
    """构造发送给客服模型的消息列表。"""

    system_message = (
        "你是小哲电商公司的客服 Agent。需要用客服语气回答用户问题，"
        "但不能承诺具体优惠、退款、发货、赔偿或人工处理结果。"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _api_key_is_missing(api_key: str | None) -> bool:
    """判断 API Key 是否缺失或仍为占位值。"""

    return api_key is None or api_key.strip() in PLACEHOLDER_API_KEYS


def call_chat_model(
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """按 OpenAI-compatible 协议调用模型，确保响应是 JSON 对象。"""

    settings = get_llm_settings()
    # 显式传参优先于配置，便于调用方针对单次请求替换连接或模型。
    resolved_api_key = (
        api_key
        if api_key is not None
        else settings.api_key.get_secret_value()
    )
    resolved_model = model if model is not None else settings.model
    if _api_key_is_missing(resolved_api_key):
        logger.error("LLM request rejected model=%s reason=missing_api_key", resolved_model)
        raise RuntimeError(
            "缺少有效的 AGENT_OPENAI_API_KEY，无法调用聊天模型。"
        )

    resolved_base_url = (
        base_url if base_url is not None else settings.base_url
    ).rstrip("/")
    request_kwargs = {
        "headers": {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
        "json": {
            "model": resolved_model,
            "messages": messages,
        },
        "timeout": settings.timeout_seconds,
    }

    # 将传输错误归一为运行时错误，供上层决定是否使用安全话术。
    started_at = time.perf_counter()
    logger.info(
        "LLM request started model=%s message_count=%d timeout_seconds=%s",
        resolved_model,
        len(messages),
        settings.timeout_seconds,
    )
    try:
        if http_client is None:
            response = httpx.post(
                f"{resolved_base_url}/chat/completions",
                **request_kwargs,
            )
        else:
            response = http_client.post(
                f"{resolved_base_url}/chat/completions",
                **request_kwargs,
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        error_response = getattr(exc, "response", None)
        logger.warning(
            "LLM request failed model=%s status_code=%s error_type=%s duration_ms=%.2f",
            resolved_model,
            getattr(error_response, "status_code", None),
            type(exc).__name__,
            (time.perf_counter() - started_at) * 1000,
            exc_info=True,
        )
        raise RuntimeError(
            "模型接口调用失败，请检查 Base URL、模型名称和 API Key。"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning(
            "LLM response invalid model=%s reason=invalid_json duration_ms=%.2f",
            resolved_model,
            (time.perf_counter() - started_at) * 1000,
        )
        raise RuntimeError("模型接口返回的不是合法 JSON。") from exc

    if not isinstance(payload, dict):
        logger.warning(
            "LLM response invalid model=%s reason=non_object duration_ms=%.2f",
            resolved_model,
            (time.perf_counter() - started_at) * 1000,
        )
        raise RuntimeError("模型接口返回的 JSON 不是对象结构。")

    logger.info(
        "LLM request completed model=%s status_code=%d duration_ms=%.2f",
        resolved_model,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
    )
    return payload


def extract_assistant_message(model_response: dict[str, Any]) -> str:
    """只接受首个 choice 中非空的 assistant 文本。"""

    try:
        content = model_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "模型响应中没有有效的 assistant message。"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型响应中的 assistant message 为空。")

    return content
