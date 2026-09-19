"""第 02 课聊天接口契约测试。"""

from collections.abc import Iterator
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from config import clear_settings_cache
from domain import ChatCommand, ChatResult, CostSummary, IntentResult
from web.app import create_app


def valid_chat_payload() -> dict[str, Any]:
    """构造合法聊天请求。"""

    return {
        "session_id": "session-001",
        "runtime_user_id": "U1001",
        "runtime_nickname": "张三",
        "runtime_member_level": "gold",
        "runtime_risk_level": "low",
        "user_message": "你好，我想了解一下当前有什么优惠活动？",
        "runtime_context": {
            "page": "product-detail",
            "product_id": 1,
        },
    }


class StubAgent:
    """测试使用的客服 Agent，不访问真实模型。"""

    def __init__(self) -> None:
        self.received_request: Any | None = None

    def chat(self, request: ChatCommand) -> ChatResult:
        self.received_request = request

        intent_result = IntentResult(
            intent="general_chat",
            source="rules",
            confidence=0.95,
            matched_keywords=["你好"],
            explanation="测试意图结果。",
        )
        return ChatResult(
            session_id=request.session_id,
            answer="这是测试客服回答。",
            intent=intent_result.intent,
            intent_result=intent_result,
            cost_summary=CostSummary(
                prompt_tokens=100,
                answer_tokens=20,
                total_tokens=120,
                token_source="model_usage",
                estimated_input_cost_cny=0.0001,
                estimated_output_cost_cny=0.00004,
                estimated_total_cost_cny=0.00014,
                context_chars=320,
                pricing_note="测试数据",
            ),
            reasoning_summary=["这是测试执行摘要。"],
            session_state={
                "agent_version": "lesson-02-chat-service",
                "message_count": 1,
                "runtime_context": {
                    "user_id": request.runtime_user_id,
                    "nickname": request.runtime_nickname,
                    "member_level": request.runtime_member_level,
                    "risk_level": request.runtime_risk_level,
                    "page_context": request.runtime_context or {},
                },
            },
        )


class FailingAgent:
    """模拟模型服务不可用。"""

    def chat(self, _request: ChatCommand) -> ChatResult:
        raise RuntimeError("模型服务暂时不可用")


@pytest.fixture
def stub_agent() -> StubAgent:
    return StubAgent()


@pytest.fixture
def client(
    stub_agent: StubAgent,
    agent_auth_headers: dict[str, str],
) -> Iterator[TestClient]:
    application = create_app(agent_provider=lambda: stub_agent)

    with TestClient(application, headers=agent_auth_headers) as test_client:
        yield test_client


def test_chat_returns_stable_response_contract(
    client: TestClient,
    stub_agent: StubAgent,
) -> None:
    """合法请求必须返回稳定的聊天响应结构。"""

    response = client.post("/chat", json=valid_chat_payload())

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-001",
        "answer": "这是测试客服回答。",
        "intent": "general_chat",
        "intent_result": {
            "intent": "general_chat",
            "source": "rules",
            "confidence": 0.95,
            "matched_keywords": ["你好"],
            "explanation": "测试意图结果。",
        },
        "cost_summary": {
            "prompt_tokens": 100,
            "answer_tokens": 20,
            "total_tokens": 120,
            "token_source": "model_usage",
            "usage_details": {},
            "estimated_input_cost_cny": 0.0001,
            "estimated_output_cost_cny": 0.00004,
            "estimated_total_cost_cny": 0.00014,
            "context_chars": 320,
            "pricing_note": "测试数据",
        },
        "reasoning_summary": ["这是测试执行摘要。"],
        "session_state": {
            "agent_version": "lesson-02-chat-service",
            "message_count": 1,
            "runtime_context": {
                "user_id": "U1001",
                "nickname": "张三",
                "member_level": "gold",
                "risk_level": "low",
                "page_context": {
                    "page": "product-detail",
                    "product_id": 1,
                },
            },
        },
    }

    assert stub_agent.received_request.user_message == (
        "你好，我想了解一下当前有什么优惠活动？"
    )


def test_chat_uses_identity_from_jwt(
    client: TestClient,
    stub_agent: StubAgent,
    agent_auth_headers: dict[str, str],
) -> None:
    payload = valid_chat_payload()
    payload.update({
        "runtime_account_id": 9999,
        "runtime_user_id": "spoofed-user",
        "runtime_nickname": "伪造用户",
        "runtime_member_level": "vip",
        "runtime_risk_level": "unknown",
    })

    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert stub_agent.received_request.runtime_account_id == 1001
    assert stub_agent.received_request.runtime_user_id == "U1001"
    assert stub_agent.received_request.runtime_nickname == "张三"
    assert stub_agent.received_request.runtime_member_level == "gold"
    assert stub_agent.received_request.runtime_risk_level == "low"
    assert stub_agent.received_request.access_token == (
        agent_auth_headers["Authorization"].removeprefix("Bearer ")
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("session_id", ""),
        ("user_message", ""),
    ],
)
def test_chat_rejects_empty_required_fields(
    client: TestClient,
    field: str,
    invalid_value: str,
) -> None:
    """关键请求字段不能为空。"""

    payload = valid_chat_payload()
    payload[field] = invalid_value

    response = client.post("/chat", json=payload)

    assert response.status_code == 422


def test_chat_maps_agent_runtime_error_to_service_unavailable(
    agent_auth_headers: dict[str, str],
) -> None:
    """模型配置或调用失败时返回 503。"""

    application = create_app(agent_provider=lambda: FailingAgent())

    with TestClient(application, headers=agent_auth_headers) as client:
        response = client.post("/chat", json=valid_chat_payload())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "模型服务暂时不可用",
    }


def test_chat_requires_bearer_token() -> None:
    application = create_app(agent_provider=StubAgent)

    with TestClient(application) as client:
        response = client.post("/chat", json=valid_chat_payload())

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_chat_rejects_wrong_audience(
    issue_agent_token: Callable[..., str],
) -> None:
    token = issue_agent_token(aud=["another-service"])
    application = create_app(agent_provider=StubAgent)

    with TestClient(application) as client:
        response = client.post(
            "/chat",
            json=valid_chat_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 401


def test_chat_rejects_missing_scope(
    issue_agent_token: Callable[..., str],
) -> None:
    token = issue_agent_token(scope="agent:facts:read")
    application = create_app(agent_provider=StubAgent)

    with TestClient(application) as client:
        response = client.post(
            "/chat",
            json=valid_chat_payload(),
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 403


def test_capabilities_describes_lesson_02_boundary(
    client: TestClient,
) -> None:
    """能力接口必须明确当前已经开放和尚未开放的能力。"""

    response = client.get("/capabilities")

    assert response.status_code == 200

    payload = response.json()
    assert payload["schema_version"] == "agent_capabilities_v1"
    assert payload["endpoints"]["chat"] is True
    assert payload["features"]["runtime_context"] is True
    assert payload["features"]["tool_calls"] is False
    assert payload["features"]["memory"] is False


def test_capabilities_reflects_realtime_fact_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("START_INTEGRATIONS", "true")
    clear_settings_cache()
    application = create_app(agent_provider=StubAgent)

    with TestClient(application) as client:
        response = client.get("/capabilities")

    assert response.status_code == 200
    assert response.json()["lesson"]["number"] == 17
    assert response.json()["features"]["realtime_business_facts"] is True
