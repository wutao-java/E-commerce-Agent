"""封装 OpenAI-compatible 聊天模型调用。"""

from __future__ import annotations

from typing import Any

import httpx

from config.llm import get_llm_settings

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


def call_chat_model(messages: list[dict[str, str]],
                    *,
                    api_key: str | None = None,
                    base_url: str | None = None,
                    model: str | None = None,
                    http_client: httpx.Client | None = None,
                    ) -> dict[str, Any]:
    """调用聊天模型并返回原始响应。"""

    settings = get_llm_settings()

    resolved_api_key = (
        api_key
        if api_key is not None
        else settings.api_key.get_secret_value()
    )
    if _api_key_is_missing(resolved_api_key):
        raise RuntimeError(
            "缺少有效的 AGENT_OPENAI_API_KEY，无法调用聊天模型。"
        )

    resolved_base_url = (
        base_url if base_url is not None else settings.base_url
    ).rstrip("/")
    resolved_model = model if model is not None else settings.model

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
        raise RuntimeError(
            "模型接口调用失败，请检查 Base URL、模型名称和 API Key。"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("模型接口返回的不是合法 JSON。") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("模型接口返回的 JSON 不是对象结构。")

    return payload


def extract_assistant_message(model_response: dict[str, Any]) -> str:
    """从模型响应中提取 assistant message。"""

    try:
        content = model_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "模型响应中没有有效的 assistant message。"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型响应中的 assistant message 为空。")

    return content
