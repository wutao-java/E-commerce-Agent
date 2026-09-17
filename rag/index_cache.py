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
    version: str
    chunks_by_id: dict[str, KnowledgeChunk]
    inverted_index: dict[str, list[str]]


@lru_cache(maxsize=1)
def get_knowledge_index() -> KnowledgeIndex:
    settings = get_rag_settings()
    chunks = build_knowledge_chunks(load_source_documents(), settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("课程知识目录为空。")
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
    get_knowledge_index.cache_clear()
    return get_knowledge_index()


def embedding_text(chunk: KnowledgeChunk) -> str:
    return "\n".join((chunk.title, chunk.section, " ".join(chunk.keywords), chunk.text))
