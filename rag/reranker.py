"""对双路召回结果进行百炼重排，并按父块去重返回证据。"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import httpx

from rag.embeddings import BailianEmbeddings
from rag.milvus_store import Evidence, MilvusKnowledgeStore


class BailianReranker:
    """调用百炼 qwen3-rerank，为候选子块计算相对相关性。"""

    def __init__(self, *, api_key: str, endpoint: str, http_client: httpx.Client | None = None) -> None:
        if not api_key.strip() or not endpoint.strip():
            raise ValueError("Reranker key and endpoint are required")
        self.api_key = api_key
        self.endpoint = endpoint
        self.http_client = http_client

    def rerank(self, query: str, hits: list[Evidence]) -> list[Evidence]:
        """按模型返回的原始索引映射候选，拒绝遗漏或重复结果。"""

        if not hits:
            return []
        try:
            response = (self.http_client or httpx).post(
                self.endpoint, headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "qwen3-rerank", "query": query,
                      "documents": [hit.child_text for hit in hits], "top_n": len(hits)},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Bailian rerank request failed") from exc
        if not isinstance(payload, dict) or payload.get("code") or not isinstance(payload.get("results"), list):
            raise ValueError("Invalid rerank response")
        # 不把模型分数作为跨请求阈值；这里只使用本次排序结果。
        indexes: set[int] = set()
        ordered: list[Evidence] = []
        for item in payload["results"]:
            if not isinstance(item, dict):
                raise ValueError("Invalid rerank result")
            index, score = item.get("index"), item.get("relevance_score")
            if type(index) is not int or not 0 <= index < len(hits) or index in indexes or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError("Invalid rerank index or score")
            indexes.add(index)
            ordered.append(replace(hits[index], score=float(score)))
        if len(ordered) != len(hits):
            raise ValueError("Reranker omitted candidates")
        return ordered


def retrieve(
    *, query: str, kb_id: str, store: MilvusKnowledgeStore,
    embeddings: BailianEmbeddings, reranker: BailianReranker,
    at: datetime, version_ids: list[str], topic: str | None = None, audience: str = "all",
    candidates: int = 30, parent_limit: int = 4,
) -> list[Evidence]:
    """查询向量化、混合召回、重排后返回不同父块的可引用证据。"""
    if not query.strip() or parent_limit < 1:
        raise ValueError("Query and positive parent limit are required")
    if not version_ids:
        return []
    # 模型切换或回滚后，先验证查询向量与当前集合来自同一模型。
    store.assert_model(kb_id, embeddings)
    vector = embeddings.embed([query], text_type="query")[0]
    hits = store.search(kb_id=kb_id, dense=vector.dense, sparse=vector.sparse,
                        at=at, version_ids=version_ids, topic=topic, audience=audience, candidates=candidates)
    ranked = reranker.rerank(query, hits)
    # 多个子块命中同一父块时只保留最靠前的一条引用。
    seen: set[str] = set()
    evidence: list[Evidence] = []
    for hit in ranked:
        if hit.parent_id not in seen:
            evidence.append(hit)
            seen.add(hit.parent_id)
        if len(evidence) == parent_limit:
            break
    return evidence
