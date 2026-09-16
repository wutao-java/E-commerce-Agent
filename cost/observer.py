"""标准化模型 usage，并估算没有 usage 时的 token 与费用。"""

from __future__ import annotations

from typing import Any

from config.llm import get_llm_settings
from domain import CostSummary, TokenUsage


def estimate_tokens(text: str) -> int:
    """按字符粗估 token，仅用于平台未返回 usage 的观察场景。"""

    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii_chars // 2)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> tuple[int, int]:
    """返回回答模型输入的估算 token 数与原始字符数。"""

    text = "\n".join(
        f"{message['role']}:{message['content']}"
        for message in messages
    )
    return estimate_tokens(text), len(text)


def _read_usage_int(usage: dict[str, Any], *names: str) -> int | None:
    """读取不同平台字段中的非负整数。"""

    for name in names:
        value = usage.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
    return None


def parse_model_usage(payload: dict[str, Any]) -> TokenUsage | None:
    """将兼容接口中的 usage 转换成统一的输入、输出与总用量。"""

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = _read_usage_int(usage, "prompt_tokens", "input_tokens")
    answer_tokens = _read_usage_int(
        usage, "completion_tokens", "output_tokens", "answer_tokens"
    )
    total_tokens = _read_usage_int(usage, "total_tokens")
    if answer_tokens is None and prompt_tokens is not None and total_tokens is not None:
        answer_tokens = max(0, total_tokens - prompt_tokens)
    if prompt_tokens is None and answer_tokens is not None and total_tokens is not None:
        prompt_tokens = max(0, total_tokens - answer_tokens)
    if prompt_tokens is None or answer_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = prompt_tokens + answer_tokens

    # reasoning 和缓存明细只用于观察，不额外计入平台给出的总 token。
    details: dict[str, int] = {}
    completion_details = usage.get("completion_tokens_details")
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = _read_usage_int(completion_details, "reasoning_tokens")
        if reasoning_tokens is not None:
            details["reasoning_tokens"] = reasoning_tokens
    if isinstance(prompt_details, dict):
        cached_tokens = _read_usage_int(prompt_details, "cached_tokens")
        if cached_tokens is not None:
            details["cached_tokens"] = cached_tokens
    for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        count = _read_usage_int(usage, name)
        if count is not None:
            details[name] = count

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        total_tokens=total_tokens,
        details=details,
    )


def build_cost_summary(
    messages: list[dict[str, str]],
    answer: str,
    usage: TokenUsage | None = None,
) -> CostSummary:
    """优先采用平台计量，缺失时粗估回答模型本轮成本。"""

    estimated_prompt_tokens, context_chars = estimate_messages_tokens(messages)
    if usage is None:
        prompt_tokens = estimated_prompt_tokens
        answer_tokens = estimate_tokens(answer)
        total_tokens = prompt_tokens + answer_tokens
        token_source = "local_estimate"
        usage_details: dict[str, int] = {}
        pricing_note = (
            "平台未返回有效 usage，token 和金额仅用于趋势观察；"
            "模型调用失败时也不代表实际产生费用。"
        )
    else:
        prompt_tokens = usage.prompt_tokens
        answer_tokens = usage.answer_tokens
        total_tokens = usage.total_tokens
        token_source = "model_usage"
        usage_details = usage.details
        pricing_note = (
            "token 采用模型平台 usage；金额按配置单价估算，"
            "实际账单以模型平台为准。"
        )

    settings = get_llm_settings()
    # 输入与输出价格可能不同，分别估算后再合计。
    input_cost = prompt_tokens / 1000 * settings.input_cny_per_1k
    output_cost = answer_tokens / 1000 * settings.output_cny_per_1k
    return CostSummary(
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        total_tokens=total_tokens,
        token_source=token_source,
        usage_details=usage_details,
        estimated_input_cost_cny=round(input_cost, 6),
        estimated_output_cost_cny=round(output_cost, 6),
        estimated_total_cost_cny=round(input_cost + output_cost, 6),
        context_chars=context_chars,
        pricing_note=pricing_note,
    )
