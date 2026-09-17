"""组合第 08–16 课的隔离知识检索流程。"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from pymilvus.exceptions import MilvusException

from config.rag import RagSettings, get_rag_settings
from domain import ChatCommand, Intent
from domain.rag import KnowledgeHit, RetrievalPlan
from rag.embeddings import EmbeddingClient
from rag.index_cache import embedding_text, get_knowledge_index
from rag.knowledge_base import is_available
from rag.milvus_store import CourseMilvusStore
from rag.planning import asks_for_history, pre_retrieval_plan
from rag.reranker import rerank_candidates
from rag.retrieval import keyword_retrieve, merge_hybrid_hits, vector_retrieve


logger = logging.getLogger(__name__)


class CourseRagService:
    def __init__(
        self, settings: RagSettings | None = None,
        embedding: EmbeddingClient | None = None,
        store: CourseMilvusStore | None = None,
    ) -> None:
        self.settings = settings or get_rag_settings()
        self.embedding = embedding or EmbeddingClient(self.settings)
        self.store = store or CourseMilvusStore(self.settings)
        self._verified_version: str | None = None
        self._cache: OrderedDict[str, tuple[float, list[KnowledgeHit]]] = OrderedDict()
        self._lock = Lock()

    def _cache_key(self, version: str, plan: RetrievalPlan) -> str:
        payload = [version, plan.scene, plan.rewritten_query, plan.allowed_topics,
                   plan.keyword_terms, self.settings.reference_date().isoformat()]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()

    def retrieve(self, command: ChatCommand, intent: Intent) -> tuple[RetrievalPlan, list[KnowledgeHit], dict[str, Any]]:
        index = get_knowledge_index()
        chunks = list(index.chunks_by_id.values())
        plan = pre_retrieval_plan(command, intent, chunks)
        debug: dict[str, Any] = {
            "mode": "milvus_hybrid_retrieval", "index_version": index.version,
            "collection": self.store.collection_name(index.version),
            "rewrite": {"original_query": plan.original_query, "rewritten_query": plan.rewritten_query},
            "scene": plan.scene, "allowed_topics": plan.allowed_topics,
            "cache_hit": False, "cache_scope": "retrieval_hits_only",
        }
        if plan.realtime or intent == "general_chat":
            debug["fallback_reason"] = "realtime_query" if plan.realtime else "general_chat"
            return plan, [], debug

        historical = asks_for_history(plan.original_query)
        allowed = {
            chunk.chunk_id: chunk for chunk in chunks
            if chunk.topic in plan.allowed_topics
            and is_available(chunk, self.settings.reference_date(), historical)
        }
        if not allowed:
            debug["fallback_reason"] = "no_available_knowledge"
            return plan, [], debug

        key = self._cache_key(index.version, plan)
        if not historical:
            with self._lock:
                cached = self._cache.get(key)
                if cached and time.monotonic() - cached[0] < 300:
                    self._cache.move_to_end(key)
                    debug["cache_hit"] = True
                    debug["matched_chunk_ids"] = [hit.chunk.chunk_id for hit in cached[1]]
                    return plan, cached[1], debug
                self._cache.pop(key, None)

        try:
            if self._verified_version != index.version:
                self.store.verify(index.version, set(index.chunks_by_id))
                self._verified_version = index.version
            vector_hits = vector_retrieve(plan, index.version, allowed, self.embedding, self.store, self.settings)
        except (MilvusException, ValueError) as exc:
            logger.warning("Course RAG retrieval unavailable: %s", exc)
            raise RuntimeError("课程 Milvus 或 embedding 检索不可用。") from exc
        keyword_hits = keyword_retrieve(plan, allowed, index.inverted_index, self.settings.candidate_k)
        candidates = merge_hybrid_hits(vector_hits, keyword_hits)
        ranked = rerank_candidates(plan, candidates, self.settings)
        reliable = [hit for hit in ranked if hit.score >= self.settings.low_confidence_score][:self.settings.top_k]
        debug.update({
            "vector_chunk_ids": [hit.chunk.chunk_id for hit in vector_hits],
            "keyword_chunk_ids": [hit.chunk.chunk_id for hit in keyword_hits],
            "candidate_chunk_ids": [hit.chunk.chunk_id for hit in candidates],
            "matched_chunk_ids": [hit.chunk.chunk_id for hit in reliable],
            "top_score": ranked[0].score if ranked else 0.0,
            "low_confidence": not reliable,
        })
        if reliable and not historical:
            with self._lock:
                self._cache[key] = (time.monotonic(), reliable)
                self._cache.move_to_end(key)
                if len(self._cache) > 256:
                    self._cache.popitem(last=False)
        return plan, reliable, debug

    def publish(self) -> tuple[str, int]:
        index = get_knowledge_index()
        chunks = list(index.chunks_by_id.values())
        vectors = self.embedding.embed_many([embedding_text(chunk) for chunk in chunks])
        name = self.store.publish(index.version, chunks, vectors)
        self._verified_version = index.version
        with self._lock:
            self._cache.clear()
        return name, len(chunks)
