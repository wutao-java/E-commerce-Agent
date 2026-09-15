"""第 01 课模型消息测试。"""

import json
from typing import Any

import httpx
import pytest

from scripts.lesson_01_messages import (
    build_messages,
    call_chat_model,
    extract_assistant_message,
)



def test_build_messages_contains_system_and_user_messages() -> None:
    """客服消息必须包含系统边界和用户原始问题。"""

    messages = build_messages("会员券可以和活动一起使用吗？")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "客服 Agent" in messages[0]["content"]
    assert "不能承诺" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "会员券可以和活动一起使用吗？",
    }


def test_call_chat_model_sends_openai_compatible_request() -> None:
    """模型调用必须发送正确的地址、认证信息和 messages。"""

    messages = build_messages("你好")

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)

        assert str(request.url) == "https://model.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert payload == {
            "model": "test-model",
            "messages": messages,
        }

        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "你好，请问有什么可以帮助你？",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handle_request)

    with httpx.Client(transport=transport) as client:
        response = call_chat_model(
            messages,
            api_key="test-api-key",
            base_url="https://model.example/v1/",
            model="test-model",
            http_client=client,
        )

    assert extract_assistant_message(response) == "你好，请问有什么可以帮助你？"


def test_call_chat_model_rejects_missing_api_key() -> None:
    """缺少 API Key 时必须在发送网络请求前失败。"""

    with pytest.raises(RuntimeError, match="AGENT_OPENAI_API_KEY"):
        call_chat_model(
            build_messages("你好"),
            api_key="",
        )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_extract_assistant_message_rejects_invalid_response(
    response: dict[str, Any],
) -> None:
    """模型响应缺少有效 assistant content 时不能视为调用成功。"""

    with pytest.raises(RuntimeError, match="assistant message"):
        extract_assistant_message(response)