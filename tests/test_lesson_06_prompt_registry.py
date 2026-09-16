"""第 06 课 Prompt Registry 测试。"""

from typing import Any

from agent import CustomerServiceAgent
from domain import ChatCommand, IntentResult
from prompts import (
    load_prompt_registry,
    render_prompt_template,
    select_prompt_fragments,
)


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


def test_registry_selects_enabled_fragments_by_intent() -> None:
    """优惠意图只选择通用和当前优惠片段。"""

    registry = load_prompt_registry()
    selected = select_prompt_fragments(
        "promotion_consult",
        registry,
    )

    assert len(registry) == 6
    assert [fragment.fragment_id for fragment in selected] == [
        "identity-boundary",
        "fact-priority",
        "promotion-current-audio-rule",
        "answer-style",
    ]
    assert [fragment.priority for fragment in selected] == [
        100,
        90,
        80,
        40,
    ]


def test_template_excludes_disabled_legacy_fragment() -> None:
    """关闭的历史活动片段不能进入模型 Prompt。"""

    registry = load_prompt_registry()
    selected = select_prompt_fragments(
        "promotion_consult",
        registry,
    )
    messages = render_prompt_template(
        ChatCommand(
            session_id="lesson06-template",
            runtime_user_id="U1001",
            runtime_member_level="gold",
            user_message="会员价还能叠加会员券吗？",
        ),
        IntentResult(
            intent="promotion_consult",
            source="rules",
            confidence=0.95,
            matched_keywords=["会员价", "券"],
            explanation="用户在询问优惠或活动。",
        ),
        selected,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
    ]
    assert "promotion-current-audio-rule" in messages[0]["content"]
    assert "legacy-promo-2024-note" not in messages[0]["content"]
    assert "member_level: gold" in messages[1]["content"]


def test_agent_exposes_prompt_registry_state() -> None:
    """Agent 应公开本轮片段选择结果并保留模型观察信息。"""

    received_messages: list[dict[str, str]] = []

    def answer_model(
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        received_messages.extend(messages)
        return model_response("是否可叠加请以结算页实时展示为准。")

    def unexpected_classifier(_message: str):
        raise AssertionError("高置信规则命中时不应调用分类模型")

    agent = CustomerServiceAgent(
        model_call=answer_model,
        classifier_call=unexpected_classifier,
    )
    result = agent.chat(
        ChatCommand(
            session_id="lesson06-agent",
            runtime_user_id="U1001",
            runtime_member_level="gold",
            user_message="降噪耳机会员价还能叠加会员券吗？",
        )
    )

    prompt_registry = result.session_state["prompt_registry"]

    assert result.intent == "promotion_consult"
    assert result.session_state["agent_version"] == (
        "lesson-07-token-cost-observation"
    )
    assert prompt_registry["selected_fragment_count"] == 4
    assert "promotion-current-audio-rule" in (
        prompt_registry["selected_fragment_ids"]
    )
    assert "legacy-promo-2024-note" in (
        prompt_registry["disabled_fragment_ids"]
    )
    assert prompt_registry["priorities"] == [100, 90, 80, 40]
    assert result.session_state["model_answer"]["used_model"] is True
    assert "legacy-promo-2024-note" not in received_messages[0]["content"]
