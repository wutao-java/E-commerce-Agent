"""合并 Milvus 向量候选与精确关键词候选。"""

from __future__ import annotations

from config.rag import RagSettings
from domain.rag import KnowledgeChunk, KnowledgeHit, RetrievalPlan
from rag.embeddings import EmbeddingClient
from rag.milvus_store import CourseMilvusStore


def vector_retrieve(
    plan: RetrievalPlan, version: str, allowed: dict[str, KnowledgeChunk],
    embedding: EmbeddingClient, store: CourseMilvusStore, settings: RagSettings,
) -> list[KnowledgeHit]:
    """在预过滤的知识范围内执行 Milvus 向量召回。

    返回结果会应用最小向量分数阈值，并将分数限制到 ``[0, 1]``。
    """
    vector = embedding.embed(plan.rewritten_query)
    matches = store.search(version, vector, set(allowed), settings.candidate_k)
    return [
        KnowledgeHit(chunk=allowed[chunk_id], score=max(0.0, min(1.0, score)),
                     vector_score=max(0.0, min(1.0, score)), sources=["vector"])
        for chunk_id, score in matches
        if chunk_id in allowed and score >= settings.min_vector_score
    ]


def keyword_retrieve(
    plan: RetrievalPlan, allowed: dict[str, KnowledgeChunk],
    inverted_index: dict[str, list[str]], limit: int,
) -> list[KnowledgeHit]:
    """通过关键词倒排表召回并计算精确匹配分数。

    分数为命中关键词数占计划关键词总数的比例，结果最多返回 ``limit`` 条。
    """
    hits: list[KnowledgeHit] = []
    # 先通过倒排表缩小候选范围，再检查关键词是否出现在章节正文中。
    candidate_ids = {
        chunk_id for term in plan.keyword_terms
        for chunk_id in inverted_index.get(term.lower(), [])
    }
    for chunk_id in candidate_ids & allowed.keys():
        chunk = allowed[chunk_id]
        text = f"{chunk.section} {chunk.text}".lower()
        matched = [term for term in plan.keyword_terms if term.lower() in text or term in chunk.keywords]
        if matched:
            score = min(1.0, len(matched) / max(len(plan.keyword_terms), 1))
            hits.append(KnowledgeHit(chunk=chunk, score=score, keyword_score=score, sources=["keyword"]))
    return sorted(hits, key=lambda hit: hit.keyword_score, reverse=True)[:limit]


def merge_hybrid_hits(vector_hits: list[KnowledgeHit], keyword_hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """按知识片段 ID 合并向量和关键词双路召回结果。

    同一片段命中两路时分别保留两路最高分，并去重记录召回来源。
    """
    merged: dict[str, KnowledgeHit] = {}
    for hit in [*vector_hits, *keyword_hits]:
        existing = merged.get(hit.chunk.chunk_id)
        if existing is None:
            merged[hit.chunk.chunk_id] = hit
        else:
            merged[hit.chunk.chunk_id] = existing.model_copy(update={
                "vector_score": max(existing.vector_score, hit.vector_score),
                "keyword_score": max(existing.keyword_score, hit.keyword_score),
                "sources": list(dict.fromkeys([*existing.sources, *hit.sources])),
            })
    return list(merged.values())
