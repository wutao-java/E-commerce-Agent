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


def _log_retrieval_completed(
    session_id: str,
    debug: dict[str, Any],
    hit_count: int,
    started_at: float,
) -> None:
    """记录不包含查询正文和知识正文的检索结果摘要。"""

    logger.info(
        "RAG retrieval completed session_id=%s scene=%s hit_count=%d "
        "cache_hit=%s fallback_reason=%s duration_ms=%.2f",
        session_id,
        debug["scene"],
        hit_count,
        debug["cache_hit"],
        debug.get("fallback_reason"),
        (time.perf_counter() - started_at) * 1000,
    )


class CourseRagService:
    """编排课程知识索引校验、混合召回、重排和结果缓存。"""

    def __init__(
        self, settings: RagSettings | None = None,
        embedding: EmbeddingClient | None = None,
        store: CourseMilvusStore | None = None,
    ) -> None:
        """初始化服务，并支持注入 embedding 与存储依赖以便测试。"""
        self.settings = settings or get_rag_settings()
        self.embedding = embedding or EmbeddingClient(self.settings)
        self.store = store or CourseMilvusStore(self.settings)
        self._verified_version: str | None = None
        self._cache: OrderedDict[str, tuple[float, list[KnowledgeHit]]] = OrderedDict()
        self._lock = Lock()

    def _cache_key(self, version: str, plan: RetrievalPlan) -> str:
        """生成隔离索引版本、检索条件和参考日期的缓存键。"""
        payload = [version, plan.scene, plan.rewritten_query, plan.allowed_topics,
                   plan.keyword_terms, self.settings.reference_date().isoformat()]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()

    def retrieve(self, command: ChatCommand, intent: Intent) -> tuple[RetrievalPlan, list[KnowledgeHit], dict[str, Any]]:
        """执行一次课程知识检索。

        Args:
            command: 当前聊天命令及运行时用户信息。
            intent: 上游识别出的用户意图。

        Returns:
            检索计划、通过置信度阈值的知识命中，以及调试信息。

        Raises:
            RuntimeError: Milvus 索引或 embedding 服务不可用。
        """
        started_at = time.perf_counter()
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
        logger.info(
            "RAG retrieval started session_id=%s intent=%s scene=%s "
            "index_version=%s realtime=%s",
            command.session_id,
            intent,
            plan.scene,
            index.version,
            plan.realtime,
        )
        # 实时业务查询和闲聊不应使用静态课程知识作答。
        if plan.realtime or intent == "general_chat":
            debug["fallback_reason"] = "realtime_query" if plan.realtime else "general_chat"
            _log_retrieval_completed(command.session_id, debug, 0, started_at)
            return plan, [], debug

        historical = asks_for_history(plan.original_query)
        allowed = {
            chunk.chunk_id: chunk for chunk in chunks
            if chunk.topic in plan.allowed_topics
            and is_available(chunk, self.settings.reference_date(), historical)
        }
        if not allowed:
            debug["fallback_reason"] = "no_available_knowledge"
            _log_retrieval_completed(command.session_id, debug, 0, started_at)
            return plan, [], debug

        key = self._cache_key(index.version, plan)
        # 历史查询不缓存，避免与常规有效知识查询共享过期资料结果。
        if not historical:
            with self._lock:
                cached = self._cache.get(key)
                if cached and time.monotonic() - cached[0] < 300:
                    self._cache.move_to_end(key)
                    debug["cache_hit"] = True
                    debug["matched_chunk_ids"] = [hit.chunk.chunk_id for hit in cached[1]]
                    _log_retrieval_completed(
                        command.session_id,
                        debug,
                        len(cached[1]),
                        started_at,
                    )
                    return plan, cached[1], debug
                self._cache.pop(key, None)

        try:
            # 每个索引版本在进程生命周期内只执行一次完整性校验。
            if self._verified_version != index.version:
                self.store.verify(index.version, set(index.chunks_by_id))
                self._verified_version = index.version
            vector_hits = vector_retrieve(plan, index.version, allowed, self.embedding, self.store, self.settings)
        except RuntimeError as exc:
            logger.warning(
                "RAG retrieval failed session_id=%s scene=%s error_type=%s duration_ms=%.2f",
                command.session_id,
                plan.scene,
                type(exc).__name__,
                (time.perf_counter() - started_at) * 1000,
                exc_info=True,
            )
            raise
        except (MilvusException, ValueError) as exc:
            logger.warning(
                "RAG retrieval failed session_id=%s scene=%s error_type=%s duration_ms=%.2f",
                command.session_id,
                plan.scene,
                type(exc).__name__,
                (time.perf_counter() - started_at) * 1000,
                exc_info=True,
            )
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
        _log_retrieval_completed(
            command.session_id,
            debug,
            len(reliable),
            started_at,
        )
        return plan, reliable, debug

    def publish(self) -> tuple[str, int]:
        """向量化当前知识快照并发布对应版本的 Milvus 索引。

        Returns:
            已发布的 collection 名称和知识片段数量。
        """
        started_at = time.perf_counter()
        index = get_knowledge_index()
        chunks = list(index.chunks_by_id.values())
        logger.info(
            "RAG index publish started index_version=%s chunk_count=%d",
            index.version,
            len(chunks),
        )
        try:
            vectors = self.embedding.embed_many([embedding_text(chunk) for chunk in chunks])
            name = self.store.publish(index.version, chunks, vectors)
        except Exception as exc:
            logger.error(
                "RAG index publish failed index_version=%s error_type=%s duration_ms=%.2f",
                index.version,
                type(exc).__name__,
                (time.perf_counter() - started_at) * 1000,
                exc_info=True,
            )
            raise
        self._verified_version = index.version
        with self._lock:
            self._cache.clear()
        logger.info(
            "RAG index publish completed index_version=%s collection=%s "
            "chunk_count=%d duration_ms=%.2f",
            index.version,
            name,
            len(chunks),
            (time.perf_counter() - started_at) * 1000,
        )
        return name, len(chunks)
