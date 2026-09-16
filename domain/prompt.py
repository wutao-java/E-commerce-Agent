"""定义 Prompt Registry 片段模型。"""

from pydantic import BaseModel, Field


class PromptFragment(BaseModel):
    """表示 Prompt Registry 中一个可选择的规则片段。"""

    fragment_id: str
    title: str
    priority: int
    enabled: bool = True
    applies_to: list[str]
    tags: list[str] = Field(default_factory=list)
    content: str
