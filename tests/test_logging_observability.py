"""验证关键执行步骤可观测，且日志不泄露请求正文或密钥。"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent import CustomerServiceAgent
from config.rag import RagSettings
from domain import ChatCommand, IntentResult
from llm.answer_generator import compose_grounded_answer
from llm.client import call_chat_model
from llm.intent_classifier import classify_intent_with_model
from rag.service import CourseRagService
from web.app import create_app


SENSITIVE_MESSAGE = "sensitive-message-must-not-be-logged"


def _model_response(answer: str = "测试回答") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": answer}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _classifier_result() -> IntentResult:
    return IntentResult(
        intent="general_chat",
        source="classifier",
        confidence=0.8,
        explanation="测试分类结果",
    )


def test_agent_logs_key_steps_without_message_content(
    caplog,
    monkeypatch,
) -> None:
    """Agent 应记录开始、意图和完成态，但不记录用户原话。"""

    monkeypatch.setattr(
        "agent.customer_service_agent.get_rag_settings",
        lambda: RagSettings(enabled=False),
    )
    agent = CustomerServiceAgent(
        model_call=lambda _messages: _model_response(),
        classifier_call=lambda _message: _classifier_result(),
    )

    with caplog.at_level(logging.INFO):
        agent.chat(
            ChatCommand(
                session_id="logging-session",
                runtime_user_id="logging-user",
                user_message=SENSITIVE_MESSAGE,
            )
        )

    log_text = caplog.text
    assert "Agent chat started" in log_text
    assert "Intent classified" in log_text
    assert "Agent chat completed" in log_text
    assert "session_id=logging-session" in log_text
    assert SENSITIVE_MESSAGE not in log_text


def test_llm_client_logs_call_metadata_without_payload_or_key(caplog) -> None:
    """模型调用日志仅包含模型、状态和耗时等元数据。"""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_model_response())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with caplog.at_level(logging.INFO):
            call_chat_model(
                [{"role": "user", "content": SENSITIVE_MESSAGE}],
                api_key="secret-key-must-not-be-logged",
                model="logging-model",
                base_url="https://llm.example.test/v1",
                http_client=client,
            )

    log_text = caplog.text
    assert "LLM request started" in log_text
    assert "LLM request completed" in log_text
    assert "model=logging-model" in log_text
    assert SENSITIVE_MESSAGE not in log_text
    assert "secret-key-must-not-be-logged" not in log_text


def test_chat_route_logs_success_without_message_content(
    monkeypatch,
    agent_auth_headers: dict[str, str],
) -> None:
    """聊天接口应记录会话级结果，不记录请求正文。"""

    monkeypatch.setattr(
        "agent.customer_service_agent.get_rag_settings",
        lambda: RagSettings(enabled=False),
    )
    agent = CustomerServiceAgent(
        model_call=lambda _messages: _model_response(),
        classifier_call=lambda _message: _classifier_result(),
    )
    with patch("web.routers.chat.logger") as route_logger:
        with TestClient(create_app(agent_provider=lambda: agent)) as client:
            response = client.post(
                "/chat",
                headers=agent_auth_headers,
                json={
                    "session_id": "route-logging-session",
                    "runtime_user_id": "logging-user",
                    "user_message": SENSITIVE_MESSAGE,
                },
            )

    assert response.status_code == 200
    templates = [call.args[0] for call in route_logger.info.call_args_list]
    assert "Chat request received session_id=%s message_length=%d" in templates
    assert (
        "Chat request completed session_id=%s intent=%s total_tokens=%d "
        "citation_count=%d duration_ms=%.2f"
    ) in templates
    assert SENSITIVE_MESSAGE not in str(route_logger.mock_calls)


def test_chat_route_logs_authentication_rejection() -> None:
    """鉴权拒绝应可见，但不得记录提交的令牌值。"""

    with patch("web.routers.chat.logger") as route_logger:
        with TestClient(create_app(agent_provider=lambda: None)) as client:
            response = client.post(
                "/chat",
                json={
                    "session_id": "rejected-session",
                    "runtime_user_id": "logging-user",
                    "user_message": SENSITIVE_MESSAGE,
                },
            )

    assert response.status_code == 401
    route_logger.warning.assert_called_once_with(
        "Agent authentication rejected reason=missing_bearer",
    )


def test_model_fallbacks_are_logged_without_message_content(caplog) -> None:
    """分类与回答模型降级必须可见，且不得输出模型输入。"""

    def unavailable_model(_messages):
        raise RuntimeError("测试模型不可用")

    with caplog.at_level(logging.WARNING):
        intent = classify_intent_with_model(
            SENSITIVE_MESSAGE,
            model_call=unavailable_model,
        )
        answer = compose_grounded_answer(
            messages=[{"role": "user", "content": SENSITIVE_MESSAGE}],
            deterministic_answer="安全兜底回答",
            model_call=unavailable_model,
        )

    assert intent is None
    assert answer.fallback_reason == "model_unavailable"
    assert "Intent classifier fallback reason=model_error" in caplog.text
    assert "Answer model fallback reason=model_unavailable" in caplog.text
    assert SENSITIVE_MESSAGE not in caplog.text


class _FakeEmbedding:
    def embed(self, _query: str) -> list[float]:
        return [1.0, 0.0]


class _FakeStore:
    def collection_name(self, version: str) -> str:
        return f"course_rag_{version}"

    def verify(self, _version: str, _ids: set[str]) -> None:
        return None

    def search(
        self,
        _version: str,
        _vector: list[float],
        _allowed_ids: set[str],
        _limit: int,
    ) -> list[tuple[str, float]]:
        return []


def test_rag_logs_retrieval_summary_without_query_content(caplog) -> None:
    """RAG 应记录检索阶段和数量摘要，不记录原始查询。"""

    service = CourseRagService(
        settings=RagSettings(enabled=True, as_of_date="2026-03-15"),
        embedding=_FakeEmbedding(),
        store=_FakeStore(),
    )
    with caplog.at_level(logging.INFO):
        service.retrieve(
            ChatCommand(
                session_id="rag-logging-session",
                runtime_user_id="logging-user",
                user_message=SENSITIVE_MESSAGE,
            ),
            "promotion_consult",
        )

    assert "RAG retrieval started" in caplog.text
    assert "RAG retrieval completed" in caplog.text
    assert "session_id=rag-logging-session" in caplog.text
    assert SENSITIVE_MESSAGE not in caplog.text
