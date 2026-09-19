"""验证第 17 课实时业务事实链路。"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agent import CustomerServiceAgent
from agent.business_fact_planning import detect_business_fact_need
from config.ecommerce import EcommerceSettings
from domain import BusinessFactNeed, BusinessFactResult, ChatCommand
from integrations import EcommerceClient
from services import BusinessFactService


ORDER_NO = "SO20260420103000001-a1000001"
ACCESS_TOKEN = "lesson-17-secret-token"


def chat_command(message: str) -> ChatCommand:
    return ChatCommand(
        session_id="lesson-17",
        runtime_user_id="U1001",
        user_message=message,
        access_token=ACCESS_TOKEN,
    )


class StubBusinessFactService:
    def __init__(self, result: BusinessFactResult) -> None:
        self.result = result
        self.calls: list[tuple[BusinessFactNeed, str, str | None]] = []

    def lookup(
        self,
        need: BusinessFactNeed,
        *,
        user_message: str,
        access_token: str | None,
    ) -> BusinessFactResult:
        self.calls.append((need, user_message, access_token))
        return self.result


class FailingRagService:
    def retrieve(self, *_args: Any) -> None:
        pytest.fail("实时业务事实问题不应进入 RAG")


def test_detects_supported_realtime_business_facts() -> None:
    logistics = detect_business_fact_need(
        f"请查一下订单 {ORDER_NO} 的物流状态"
    )
    product = detect_business_fact_need("DIGI-AUD-02 现在多少钱，还有库存吗？")
    refund = detect_business_fact_need(f"{ORDER_NO} 的退款进度到哪了？")
    created_order = detect_business_fact_need(
        "查询订单 EC20260919203045A1B2C3"
    )

    assert logistics.kind == "logistics"
    assert logistics.order_no == ORDER_NO
    assert product.kind == "product"
    assert product.sku == "DIGI-AUD-02"
    assert refund.kind == "unknown"
    assert refund.requires_realtime is True
    assert created_order.order_no == "EC20260919203045A1B2C3"


def test_ecommerce_client_forwards_bearer_and_unwraps_data() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "orderNo": ORDER_NO,
                    "status": "SHIPPED",
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = EcommerceClient(
            EcommerceSettings(base_url="http://commerce.test"),
            http_client=http_client,
        )
        order = client.get_order(ORDER_NO, ACCESS_TOKEN)

    assert order == {"orderNo": ORDER_NO, "status": "SHIPPED"}
    assert requests[0].headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert requests[0].url.path.endswith(ORDER_NO)


def test_business_fact_service_builds_logistics_summary() -> None:
    class StubClient:
        def get_order(self, order_no: str, access_token: str) -> dict[str, Any]:
            assert order_no == ORDER_NO
            assert access_token == ACCESS_TOKEN
            return {
                "orderNo": ORDER_NO,
                "status": "SHIPPED",
                "paymentStatus": "PAID",
                "fulfillmentStatus": "SHIPPED",
                "trackingNo": "SFCOURSE1001",
                "logisticsEvents": [],
            }

    result = BusinessFactService(StubClient()).lookup(
        BusinessFactNeed(
            kind="logistics",
            requires_realtime=True,
            order_no=ORDER_NO,
            reason="物流状态必须实时查询。",
        ),
        user_message=f"{ORDER_NO} 的物流到哪了？",
        access_token=ACCESS_TOKEN,
    )

    assert result.found is True
    assert "SFCOURSE1001" in result.summary
    assert result.facts["order"]["status"] == "SHIPPED"


def test_found_business_fact_calls_model_with_verified_facts_only() -> None:
    fact_service = StubBusinessFactService(BusinessFactResult(
        found=True,
        summary=f"订单 {ORDER_NO} 已发货。",
        facts={
            "order": {
                "orderNo": ORDER_NO,
                "status": "SHIPPED",
                "trackingNo": "SFCOURSE1001",
            },
        },
    ))
    received_messages: list[dict[str, str]] = []

    def model_call(messages: list[dict[str, str]]) -> dict[str, Any]:
        received_messages.extend(messages)
        return {
            "choices": [{"message": {"content": "您的订单已发货，运单号为 SFCOURSE1001。"}}],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 12,
                "total_tokens": 92,
            },
        }

    agent = CustomerServiceAgent(
        model_call=model_call,
        classifier_call=lambda _message: None,
        rag_service=FailingRagService(),
        business_fact_service=fact_service,
    )
    result = agent.chat(chat_command(f"{ORDER_NO} 的物流到哪了？"))

    prompt_payload = json.loads(received_messages[1]["content"])
    assert result.answer == "您的订单已发货，运单号为 SFCOURSE1001。"
    assert result.session_state["model_answer"]["used_model"] is True
    assert prompt_payload["verified_facts"]["order"]["status"] == "SHIPPED"
    assert ACCESS_TOKEN not in json.dumps(received_messages, ensure_ascii=False)
    assert "facts" not in result.session_state["business_facts"]["result"]
    assert fact_service.calls[0][2] == ACCESS_TOKEN


def test_unavailable_business_fact_skips_model_and_rag() -> None:
    fact_service = StubBusinessFactService(BusinessFactResult(
        found=False,
        summary="没有查到该订单，或该订单不属于当前登录用户。",
        failure_reason="not_found",
    ))
    agent = CustomerServiceAgent(
        model_call=lambda _messages: pytest.fail("事实不可用时不应调用模型"),
        classifier_call=lambda _message: None,
        rag_service=FailingRagService(),
        business_fact_service=fact_service,
    )

    result = agent.chat(chat_command(f"查询订单 {ORDER_NO}"))

    assert result.answer == "没有查到该订单，或该订单不属于当前登录用户。"
    assert result.citations == []
    assert result.session_state["model_answer"]["used_model"] is False
    assert result.session_state["model_answer"]["fallback_reason"] == "not_found"


def test_business_fact_model_failure_uses_deterministic_summary() -> None:
    summary = f"订单 {ORDER_NO} 当前状态为 SHIPPED。"
    fact_service = StubBusinessFactService(BusinessFactResult(
        found=True,
        summary=summary,
        facts={"order": {"orderNo": ORDER_NO, "status": "SHIPPED"}},
    ))

    def fail_model(_messages: list[dict[str, str]]) -> dict[str, Any]:
        raise RuntimeError("模型不可用")

    agent = CustomerServiceAgent(
        model_call=fail_model,
        classifier_call=lambda _message: None,
        business_fact_service=fact_service,
    )

    result = agent.chat(chat_command(f"查询订单 {ORDER_NO}"))

    assert result.answer == summary
    assert result.session_state["model_answer"]["used_model"] is False
    assert result.session_state["model_answer"]["fallback_reason"] == "model_unavailable"


def test_access_token_is_excluded_from_chat_command_output() -> None:
    command = chat_command("你好")

    assert ACCESS_TOKEN not in repr(command)
    assert "access_token" not in command.model_dump()
