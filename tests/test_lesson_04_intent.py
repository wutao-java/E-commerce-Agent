"""第 04 课结构化意图识别完整测试。"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent import CustomerServiceAgent
from agent.intent_service import classify_intent
from domain import ChatCommand, IntentResult
from llm import classify_intent_with_model
from web.app import create_app


def model_response(content: str) -> dict[str, Any]:
    """构造 OpenAI-compatible 模型响应。"""

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


def unexpected_classifier(_message: str) -> IntentResult | None:
    """规则命中时，分类模型不应被调用。"""

    raise AssertionError("高置信规则命中时不应调用分类模型")


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("我要投诉，还要申请退款。", "complaint"),
        ("物流太慢了，我要退款。", "refund_request"),
        ("我的快递到哪了？", "order_query"),
        ("现在有什么优惠活动？", "promotion_consult"),
        ("请推荐一款耳机。", "product_consult"),
        ("你好，请问在吗？", "general_chat"),
    ],
)
def test_high_confidence_rules_skip_classifier(
    message: str,
    expected_intent: str,
) -> None:
    """高置信规则应直接返回，不调用分类模型。"""

    result = classify_intent(
        message,
        classifier_call=unexpected_classifier,
    )

    assert result.intent == expected_intent
    assert result.source == "rules"
    assert result.confidence == 0.95
    assert result.matched_keywords


def test_uncertain_rule_uses_classifier_and_supports_unknown() -> None:
    """规则不确定时调用分类模型，模型无结果时返回 unknown。"""

    received_messages: list[str] = []

    def classifier(message: str) -> IntentResult:
        received_messages.append(message)
        return IntentResult(
            intent="product_consult",
            source="classifier",
            confidence=0.88,
            explanation="分类模型识别为商品咨询。",
        )

    classified = classify_intent(
        "这款耳机怎么样？",
        classifier_call=classifier,
    )
    unknown = classify_intent(
        "今天天气怎么样？",
        classifier_call=lambda _message: None,
    )

    assert classified.intent == "product_consult"
    assert classified.source == "classifier"
    assert received_messages == ["这款耳机怎么样？"]
    assert unknown.intent == "unknown"
    assert unknown.source == "rules_fallback"
    assert unknown.confidence == 0.3


def test_classifier_model_returns_valid_intent_result() -> None:
    """分类模型的 JSON 输出必须转换为 IntentResult。"""

    def classifier_model(
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        assert [message["role"] for message in messages] == [
            "system",
            "user",
        ]
        assert "允许的 intent" in messages[1]["content"]

        return model_response(
            "```json\n"
            '{"intent":"product_consult","confidence":0.86,'
            '"explanation":"用户正在咨询商品。"}'
            "\n```"
        )

    result = classify_intent_with_model(
        "这款耳机怎么样？",
        model_call=classifier_model,
    )

    assert result is not None
    assert result.intent == "product_consult"
    assert result.source == "classifier"
    assert result.confidence == 0.86
    assert result.matched_keywords == []


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        (
            '{"intent":"unsupported","confidence":0.9,'
            '"explanation":"非法意图。"}'
        ),
        (
            '{"intent":"product_consult","confidence":1.5,'
            '"explanation":"置信度越界。"}'
        ),
    ],
)
def test_classifier_rejects_invalid_output(content: str) -> None:
    """非法 JSON、非法意图和越界置信度不能进入业务响应。"""

    result = classify_intent_with_model(
        "测试消息",
        model_call=lambda _messages: model_response(content),
    )

    assert result is None


def test_answer_model_failure_uses_safe_fallback() -> None:
    """回答模型不可用时返回确定性安全话术。"""

    def unavailable_model(_messages: list[dict[str, str]]):
        raise RuntimeError("模型暂时不可用")

    agent = CustomerServiceAgent(
        model_call=unavailable_model,
        classifier_call=unexpected_classifier,
    )
    response = agent.chat(
        ChatCommand(
            session_id="lesson04-fallback",
            runtime_user_id="U1001",
            user_message="物流太慢了，我要退款。",
        )
    )

    assert response.intent == "refund_request"
    assert response.intent_result.source == "rules"
    assert "不能直接判断是否可退" in response.answer
    assert response.session_state["model_answer"]["used_model"] is False
    assert (
        response.session_state["model_answer"]["fallback_reason"]
        == "model_unavailable"
    )


def test_chat_api_returns_structured_intent_and_capabilities(
    agent_auth_headers: dict[str, str],
) -> None:
    """聊天接口返回结构化意图，并保留对应能力声明。"""

    def answer_model(
        _messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        return model_response("这是受约束的客服回答。")

    agent = CustomerServiceAgent(
        model_call=answer_model,
        classifier_call=unexpected_classifier,
    )
    application = create_app(agent_provider=lambda: agent)
    payload = {
        "session_id": "lesson04-api",
        "runtime_user_id": "U1001",
        "user_message": "我想申请退款。",
    }

    with TestClient(application, headers=agent_auth_headers) as client:
        first_response = client.post("/chat", json=payload)
        second_response = client.post("/chat", json=payload)
        capabilities_response = client.get("/capabilities")

    assert first_response.status_code == 200
    assert first_response.json()["intent"] == "refund_request"
    assert first_response.json()["intent_result"]["source"] == "rules"
    assert first_response.json()["session_state"]["message_count"] == 1
    assert second_response.json()["session_state"]["message_count"] == 2

    assert capabilities_response.status_code == 200
    capabilities = capabilities_response.json()
    assert capabilities["features"]["structured_intent"] is True
    assert capabilities["features"]["tool_calls"] is False
