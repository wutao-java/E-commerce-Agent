"""集中管理 Prompt Registry 的加载、选择和模板渲染。"""

from __future__ import annotations

import json
from pathlib import Path

from domain import ChatCommand, Intent, IntentResult, PromptFragment


PROMPT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "prompt_registry.json"
)


def load_prompt_registry() -> list[PromptFragment]:
    """每轮从 JSON 读取并校验片段，修改文件后无需重启即可生效。"""

    with PROMPT_REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if not isinstance(registry, list):
        raise RuntimeError("prompt_registry.json 必须是 JSON 数组。")

    return [
        PromptFragment.model_validate(item)
        for item in registry
    ]


def select_prompt_fragments(
    intent: Intent,
    registry: list[PromptFragment],
) -> list[PromptFragment]:
    """按启用状态和意图选择片段，并按优先级降序排列。"""

    # applies_to 含 all 的公共约束会与本轮意图专属片段一起加载。
    selected = [
        fragment
        for fragment in registry
        if fragment.enabled
        and (
            "all" in fragment.applies_to
            or intent in fragment.applies_to
        )
    ]

    return sorted(
        selected,
        key=lambda fragment: fragment.priority,
        reverse=True,
    )


def render_prompt_template(
    command: ChatCommand,
    intent_result: IntentResult,
    fragments: list[PromptFragment],
) -> list[dict[str, str]]:
    """把选中的片段和请求中的运行时上下文渲染为模型 messages。"""

    # 片段顺序来自优先级排序，便于模型按高优先级约束处理冲突。
    fragment_text = "\n\n".join(
        (
            f"[{fragment.fragment_id} | priority={fragment.priority}]\n"
            f"{fragment.content}"
        )
        for fragment in fragments
    )

    system_message = (
        "你是小哲电商公司的客服 Agent。当前版本使用 Prompt "
        "Template 和 Prompt Registry 管理规则片段。\n"
        "请严格按照下列片段顺序回答；高优先级片段覆盖"
        "低优先级片段。\n\n"
        f"{fragment_text}"
    )

    # 当前运行时字段来自请求，并在同一 user 消息中与用户原话分段呈现。
    user_message = (
        "小哲电商系统确认的当前用户事实：\n"
        f"- user_id: {command.runtime_user_id}\n"
        f"- nickname: {command.runtime_nickname or '未提供'}\n"
        f"- member_level: {command.runtime_member_level or '未提供'}\n"
        f"- risk_level: {command.runtime_risk_level or '未提供'}\n\n"
        f"粗意图：{intent_result.intent}\n"
        f"粗意图说明：{intent_result.explanation}\n\n"
        "用户原话：\n"
        f"{command.user_message}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
