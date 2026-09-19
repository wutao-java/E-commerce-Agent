"""构建本地知识快照，并用内容指纹隔离 Milvus 索引版本。"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from pydantic import BaseModel

from config.rag import get_rag_settings
from domain.rag import KnowledgeChunk
from rag.knowledge_base import build_knowledge_chunks, load_source_documents


class KnowledgeIndex(BaseModel):
    """内存中的课程知识索引快照。

    Attributes:
        version: 由知识内容及切片、向量配置共同生成的版本指纹。
        chunks_by_id: 知识片段 ID 到片段实体的映射。
        inverted_index: 小写关键词到相关知识片段 ID 的倒排表。
    """

    version: str
    chunks_by_id: dict[str, KnowledgeChunk]
    inverted_index: dict[str, list[str]]


@lru_cache(maxsize=1)
def get_knowledge_index() -> KnowledgeIndex:
    """构建并缓存当前配置对应的课程知识索引快照。

    Returns:
        包含版本指纹、知识片段映射和关键词倒排表的索引。

    Raises:
        RuntimeError: 课程知识目录中没有可用片段。
    """
    settings = get_rag_settings()
    chunks = build_knowledge_chunks(load_source_documents(), settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("课程知识目录为空。")
    # embedding 或切片配置变化时生成新版本，避免复用不兼容的 Milvus collection。
    fingerprint = {
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "embedding_model": settings.embedding_model,
        "embedding_base_url": settings.embedding_base_url,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    digest = hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    inverted: dict[str, list[str]] = {}
    for chunk in chunks:
        for keyword in chunk.keywords:
            inverted.setdefault(keyword.lower(), []).append(chunk.chunk_id)
    return KnowledgeIndex(
        version=digest,
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
        inverted_index=inverted,
    )


def rebuild_knowledge_index() -> KnowledgeIndex:
    """清除进程内快照并重新构建课程知识索引。"""
    get_knowledge_index.cache_clear()
    return get_knowledge_index()


def embedding_text(chunk: KnowledgeChunk) -> str:
    """拼接用于向量化的标题、章节、关键词和正文。"""
    return "\n".join((chunk.title, chunk.section, " ".join(chunk.keywords), chunk.text))
