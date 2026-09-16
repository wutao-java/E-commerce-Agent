"""定义 Prompt Registry 片段模型。"""

from pydantic import BaseModel, Field


class PromptFragment(BaseModel):
    """描述片段内容、适用意图和排序优先级，供本轮筛选使用。"""

    fragment_id: str
    title: str
    priority: int
    enabled: bool = True
    applies_to: list[str]
    tags: list[str] = Field(default_factory=list)
    content: str
