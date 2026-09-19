"""验证隔离课程知识检索的核心链路。"""

from __future__ import annotations

import json
import os
from datetime import date
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from agent import CustomerServiceAgent
from config.rag import RagSettings
from domain import ChatCommand
from domain.rag import KnowledgeChunk
from domain.rag import KnowledgeHit
from rag.embeddings import EmbeddingClient
from rag.index_cache import get_knowledge_index
from rag.knowledge_base import build_knowledge_chunks, is_available, load_source_documents
from rag.milvus_store import CourseMilvusStore
from rag.service import CourseRagService
from rag.reranker import rerank_lightweight
from web.app import create_app


class FakeEmbedding:
    def embed(self, _query: str) -> list[float]:
        return [1.0, 0.0]


class FakeStore:
    def __init__(self) -> None:
        self.calls = 0

    def collection_name(self, version: str) -> str:
        return f"course_rag_{version}"

    def verify(self, _version: str, _ids: set[str]) -> None:
        return None

    def search(self, _version: str, _vector: list[float], allowed_ids: set[str], _limit: int) -> list[tuple[str, float]]:
        self.calls += 1
        target = "promotion-current-audio-offer"
        return [(target, 0.86)] if target in allowed_ids else []


def course_command(message: str) -> ChatCommand:
    return ChatCommand(session_id="course-demo", runtime_user_id="U1001", user_message=message)


def test_embedding_client_caches_repeated_queries() -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embedding = EmbeddingClient(RagSettings(embedding_api_key="course-key"), http_client=client)
        assert embedding.embed("耳机") == embedding.embed("耳机")
    assert calls == ["/v1/embeddings"]


def test_embedding_client_limits_batch_size_for_compatible_providers() -> None:
    batch_sizes: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        batch_sizes.append(len(inputs))
        return httpx.Response(200, json={
            "data": [
                {"index": index, "embedding": [float(index), 0.0]}
                for index in range(len(inputs))
            ],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embedding = EmbeddingClient(RagSettings(embedding_api_key="course-key"), http_client=client)
        vectors = embedding.embed_many([f"知识片段 {index}" for index in range(21)])

    assert len(vectors) == 21
    assert batch_sizes == [10, 10, 1]


def test_course_markdown_keeps_ids_and_validity() -> None:
    documents = load_source_documents()
    chunks = build_knowledge_chunks(documents)
    assert len(documents) == 8
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert "promotion-current-audio-offer" in get_knowledge_index().inverted_index["会员价"]
    promotion = next(chunk for chunk in chunks if chunk.chunk_id == "promotion-current-audio-offer")
    assert promotion.valid_until == date(2026, 3, 31)
    assert is_available(promotion, date(2026, 3, 15))
    assert not is_available(promotion, date(2026, 9, 17))


def test_hybrid_retrieval_rewrite_cache_and_realtime_boundary() -> None:
    settings = RagSettings(enabled=True, as_of_date="2026-03-15", low_confidence_score=0.5)
    store = FakeStore()
    service = CourseRagService(settings=settings, embedding=FakeEmbedding(), store=store)
    command = course_command("金卡会员买耳麦活动，会员价可以叠券吗？")

    plan, hits, first = service.retrieve(command, "promotion_consult")
    assert "耳机" in plan.rewritten_query
    assert "优惠券" in plan.rewritten_query
    assert hits[0].chunk.chunk_id == "promotion-current-audio-offer"
    assert first["cache_hit"] is False
    assert first["vector_chunk_ids"] == ["promotion-current-audio-offer"]

    _, cached_hits, second = service.retrieve(command, "promotion_consult")
    assert cached_hits[0].chunk.chunk_id == hits[0].chunk.chunk_id
    assert second["cache_hit"] is True
    assert store.calls == 1

    _, realtime_hits, realtime = service.retrieve(course_command("我的订单物流到哪了？"), "order_query")
    assert realtime_hits == []
    assert realtime["fallback_reason"] == "realtime_query"
    assert store.calls == 1


def test_course_mode_returns_only_actual_citations(
    agent_auth_headers: dict[str, str],
) -> None:
    settings = RagSettings(enabled=True, as_of_date="2026-03-15", low_confidence_score=0.5)
    service = CourseRagService(settings=settings, embedding=FakeEmbedding(), store=FakeStore())
    agent = CustomerServiceAgent(
        rag_service=service,
        classifier_call=lambda _message: None,
        model_call=lambda _messages: {"choices": [{"message": {"content": "课程资料显示不可以叠加。"}}]},
    )
    with TestClient(
        create_app(agent_provider=lambda: agent),
        headers=agent_auth_headers,
    ) as client:
        response = client.post("/chat", json={
            "session_id": "course-demo", "runtime_user_id": "U1001",
            "user_message": "金卡会员买耳机活动，会员价可以叠券吗？",
        })
    assert response.status_code == 200
    data = response.json()
    assert data["citations"][0]["chunk_id"] == "promotion-current-audio-offer"
    assert data["session_state"]["rag"]["index_version"] == get_knowledge_index().version
    assert "cost_summary" in data


def test_course_mode_without_evidence_does_not_call_answer_model() -> None:
    settings = RagSettings(enabled=True, as_of_date="2026-09-17")
    service = CourseRagService(settings=settings, embedding=FakeEmbedding(), store=FakeStore())
    agent = CustomerServiceAgent(
        rag_service=service,
        classifier_call=lambda _message: None,
        model_call=lambda _messages: pytest.fail("没有证据时不应调用模型"),
    )
    result = agent.chat(course_command("我的订单物流到哪了？"))
    assert result.citations == []
    assert "rag" not in result.session_state
    assert result.session_state["business_facts"]["result"]["failure_reason"] == (
        "integration_disabled"
    )


def test_weak_neighbor_is_not_returned_as_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RagSettings(enabled=True, as_of_date="2026-03-15", low_confidence_score=0.5)
    service = CourseRagService(settings=settings, embedding=FakeEmbedding(), store=FakeStore())
    original_rerank = rerank_lightweight

    def weak_neighbor(plan, candidates):
        ranked = original_rerank(plan, candidates)
        return [*ranked, KnowledgeHit(chunk=ranked[0].chunk, score=0.2)]

    monkeypatch.setattr("rag.service.rerank_candidates", lambda plan, candidates, _settings: weak_neighbor(plan, candidates))
    _, hits, _ = service.retrieve(course_command("金卡会员买耳机活动，能叠券吗？"), "promotion_consult")
    assert hits
    assert all(hit.score >= settings.low_confidence_score for hit in hits)
    assert all(hit.score != 0.2 for hit in hits)


@pytest.mark.skipif(os.getenv("COURSE_RAG_VERIFY_MILVUS") != "1", reason="需要本地 Milvus Standalone")
def test_real_milvus_collection_protocol() -> None:
    settings = RagSettings(collection_prefix=f"course_verify_{uuid4().hex[:10]}")
    store = CourseMilvusStore(settings)
    chunk = KnowledgeChunk(
        chunk_id="temporary-example", title="验证", source_path="temporary.md", section="验证",
        topic="faq", status="active", keywords=["验证"], text="验证", parent_text="验证",
    )
    name = store.collection_name("test")
    try:
        store.publish("test", [chunk], [[1.0, 0.0]])
        store.publish("test", [chunk], [[1.0, 0.0]])
        matches = store.search("test", [1.0, 0.0], {chunk.chunk_id}, 1)
        assert matches[0][0] == chunk.chunk_id
    finally:
        if store._client().has_collection(name):
            store._client().drop_collection(name)
