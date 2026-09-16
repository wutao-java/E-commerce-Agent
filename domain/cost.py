"""定义回答模型的 token 用量与成本观察结果。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """保存平台返回的输入、输出 token 数及可选明细。"""

    prompt_tokens: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    details: dict[str, int] = Field(default_factory=dict)


class CostSummary(BaseModel):
    """展示回答模型本轮的用量来源与估算金额。"""

    prompt_tokens: int
    answer_tokens: int
    total_tokens: int
    token_source: Literal["model_usage", "local_estimate"]
    usage_details: dict[str, int] = Field(default_factory=dict)
    estimated_input_cost_cny: float
    estimated_output_cost_cny: float
    estimated_total_cost_cny: float
    context_chars: int
    pricing_note: str
