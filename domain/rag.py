"""课程知识、检索结果与来源引用的内部契约。"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source_path: str
    title: str
    metadata: dict[str, Any]
    body: str


class KnowledgeChunk(BaseModel):
    chunk_id: str
    title: str
    source_path: str
    section: str
    topic: str
    status: str
    keywords: list[str]
    text: str
    parent_text: str
    valid_from: date | None = None
    valid_until: date | None = None


class KnowledgeHit(BaseModel):
    chunk: KnowledgeChunk
    score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    sources: list[str] = Field(default_factory=list)
    rerank_reasons: list[str] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    original_query: str
    rewritten_query: str
    scene: str
    allowed_topics: list[str]
    keyword_terms: list[str]
    realtime: bool


class Citation(BaseModel):
    citation_id: str
    source_title: str
    source_path: str
    section: str
    chunk_id: str
    score: float
    snippet: str
