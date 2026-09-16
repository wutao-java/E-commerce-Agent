"""定义 Prompt 规则文档和上下文冲突模型。"""

from typing import Literal

from pydantic import BaseModel


class PolicyDocument(BaseModel):
    """表示注入模型上下文的一份规则文档。"""

    doc_id: str
    title: str
    status: Literal["current", "legacy", "draft"]
    keywords: list[str]
    body: str


class ContextConflict(BaseModel):
    """表示全量上下文中可观察到的一条冲突线索。"""

    topic: str
    newer_doc_id: str
    older_doc_id: str
    reason: str