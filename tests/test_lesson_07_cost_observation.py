"""第 07 课 token 用量解析与成本观察测试。"""

from typing import Any

from fastapi.testclient import TestClient

from agent import CustomerServiceAgent
from config.llm import get_llm_settings
from cost.observer import build_cost_summary, parse_model_usage
from domain import ChatCommand
from web.app import create_app


def model_response(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造回答模型响应，不访问真实模型服务。"""

    response = {
        "choices": [
            {"message": {"content": "是否叠加请以结算页为准。"}}
        ]
    }
    if usage is not None:
        response["usage"] = usage
    return response


def test_platform_usage_takes_priority_and_keeps_details() -> None:
    """平台计量优先于字符估算，reasoning 和缓存仅作为明细透传。"""

    usage = parse_model_usage(model_response({
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 8},
        "prompt_tokens_details": {"cached_tokens": 20},
        "prompt_cache_hit_tokens": 20,
    }))
    assert usage is not None

    summary = build_cost_summary(
        [{"role": "user", "content": "会员价"}],
        "请以结算页为准",
        usage,
    )
    settings = get_llm_settings()
    assert summary.token_source == "model_usage"
    assert (summary.prompt_tokens, summary.answer_tokens, summary.total_tokens) == (
        120, 30, 150
    )
    assert summary.usage_details == {
        "reasoning_tokens": 8,
        "cached_tokens": 20,
        "prompt_cache_hit_tokens": 20,
    }
    assert summary.estimated_input_cost_cny == round(
        120 / 1000 * settings.input_cny_per_1k, 6
    )
    assert summary.estimated_output_cost_cny == round(
        30 / 1000 * settings.output_cny_per_1k, 6
    )


def test_usage_aliases_and_missing_usage() -> None:
    """兼容输入输出别名；缺失有效用量时才启用本地估算。"""

    usage = parse_model_usage(model_response({
        "input_tokens": 40,
        "output_tokens": 10,
    }))
    assert usage is not None
    assert usage.total_tokens == 50
    assert parse_model_usage(model_response()) is None
    assert parse_model_usage(model_response({"prompt_tokens": "invalid"})) is None

    summary = build_cost_summary(
        [{"role": "user", "content": "会员价"}],
        "请以结算页为准",
    )
    assert summary.token_source == "local_estimate"
    assert summary.prompt_tokens > 0
    assert summary.answer_tokens > 0
    assert summary.total_tokens == summary.prompt_tokens + summary.answer_tokens


def test_failed_model_keeps_safe_answer_and_marks_estimate() -> None:
    """模型调用失败仍返回安全话术，不把估算成本当真实账单。"""

    def failing_model(_messages: list[dict[str, str]]) -> dict[str, Any]:
        raise RuntimeError("模型暂时不可用")

    agent = CustomerServiceAgent(model_call=failing_model)
    result = agent.chat(ChatCommand(
        session_id="lesson07-fallback",
        runtime_user_id="U1001",
        user_message="会员价还能叠加会员券吗？",
    ))

    assert result.session_state["model_answer"]["used_model"] is False
    assert result.cost_summary.token_source == "local_estimate"
    assert "不代表实际产生费用" in result.cost_summary.pricing_note


def test_chat_exposes_cost_and_counts_session_events(
    agent_auth_headers: dict[str, str],
) -> None:
    """HTTP 响应暴露成本，同一进程内同一会话事件数递增。"""

    agent = CustomerServiceAgent(model_call=lambda _messages: model_response({
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }))
    request = {
        "session_id": "lesson07-observation",
        "runtime_user_id": "U1001",
        "user_message": "会员价还能叠加会员券吗？",
    }
    with TestClient(
        create_app(agent_provider=lambda: agent),
        headers=agent_auth_headers,
    ) as client:
        capabilities = client.get("/capabilities")
        first = client.post("/chat", json=request)
        second = client.post("/chat", json=request)

    assert capabilities.json()["features"]["cost_summary"] is True
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cost_summary"]["token_source"] == "model_usage"
    assert first.json()["session_state"]["cost_log"]["event_count"] == 1
    assert second.json()["session_state"]["cost_log"]["event_count"] == 2
    assert second.json()["session_state"]["cost_log"]["latest"]["cost_summary"] == (
        second.json()["cost_summary"]
    )
