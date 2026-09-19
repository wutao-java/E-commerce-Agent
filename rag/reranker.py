"""为课程候选提供轻量排序和可选商业 reranker。"""

from __future__ import annotations

import httpx

from config.rag import RagSettings
from domain.rag import KnowledgeHit, RetrievalPlan


def rerank_lightweight(plan: RetrievalPlan, candidates: list[KnowledgeHit]) -> list[KnowledgeHit]:
    """使用双路命中、场景匹配和资料状态对候选进行本地重排。

    Args:
        plan: 当前检索计划。
        candidates: 合并后的向量与关键词候选。

    Returns:
        按综合分数降序排列的新命中列表。
    """
    ranked: list[KnowledgeHit] = []
    for hit in candidates:
        both = hit.vector_score > 0 and hit.keyword_score > 0
        score = max(hit.vector_score, hit.keyword_score * 0.7)
        reasons = ["向量与关键词双路命中"] if both else ["单路召回"]
        if both:
            score += 0.08
        if hit.chunk.topic == plan.scene:
            score += 0.03
            reasons.append("知识场景匹配")
        if hit.chunk.status in {"expired", "historical"}:
            score -= 0.15
            reasons.append("历史资料降权")
        ranked.append(hit.model_copy(update={
            "score": round(max(0.0, min(1.0, score)), 3),
            "rerank_reasons": reasons,
        }))
    return sorted(ranked, key=lambda hit: hit.score, reverse=True)


def rerank_candidates(plan: RetrievalPlan, candidates: list[KnowledgeHit], settings: RagSettings) -> list[KnowledgeHit]:
    """优先调用商业 reranker，未配置或调用失败时使用本地重排。

    Args:
        plan: 当前检索计划。
        candidates: 待精排的候选知识。
        settings: 包含 reranker 地址、模型和密钥的 RAG 配置。

    Returns:
        按相关性降序排列的候选列表。
    """
    if not candidates or not settings.rerank_model or not settings.rerank_base_url:
        return rerank_lightweight(plan, candidates)
    key = settings.rerank_api_key.get_secret_value()
    if not key:
        return rerank_lightweight(plan, candidates)
    try:
        response = httpx.post(
            f"{settings.rerank_base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": settings.rerank_model,
                "query": plan.rewritten_query,
                "documents": [hit.chunk.parent_text for hit in candidates],
                "top_n": len(candidates),
            },
            timeout=20,
        )
        response.raise_for_status()
        results = response.json()["results"]
        ranked = []
        for item in results:
            index = item["index"]
            # 忽略服务端返回的非法下标，避免错误候选污染结果。
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            score = max(0.0, min(1.0, float(item.get("relevance_score", item.get("score", 0)))))
            ranked.append(candidates[index].model_copy(update={
                "score": score, "rerank_reasons": ["商业 reranker 精排"],
            }))
        if ranked:
            return sorted(ranked, key=lambda hit: hit.score, reverse=True)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        # 精排是可选增强能力，失败时保留本地确定性排序作为降级路径。
        pass
    return rerank_lightweight(plan, candidates)
