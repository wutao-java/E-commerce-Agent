"""离线验证父子分块、双路向量、混合检索和重排的关键契约。"""

from datetime import datetime, timezone

import httpx
import pytest
from pymilvus import DataType

from rag.chunking import chunk_document
from rag.document_processing import DocumentBlock, ParsedDocument
from rag.embeddings import BailianEmbeddings
from rag.milvus_store import MilvusKnowledgeStore
from rag.reranker import BailianReranker, retrieve
from dataclasses import replace


def sample_document() -> ParsedDocument:
    """构造跨两个章节、包含表格和实际页码的解析结果。"""

    return ParsedDocument(
        source_name="refund.pdf", source_format="pdf", sha256="a" * 64,
        parser_version="test", page_count=2, warnings=[],
        blocks=[
            DocumentBlock(kind="heading", text="退款", heading_path=["退款"], page_start=1),
            DocumentBlock(kind="paragraph", text="七天内可退货。", heading_path=["退款"], page_start=1),
            DocumentBlock(kind="paragraph", text="需保留包装。", heading_path=["退款"], page_start=1),
            DocumentBlock(kind="table", text="|条件|结果|\n|---|---|\n|拆封|人工审核|", heading_path=["退款"], page_start=2),
            DocumentBlock(kind="heading", text="运费", heading_path=["运费"], page_start=2),
            DocumentBlock(kind="paragraph", text="质量问题运费由平台承担。", heading_path=["运费"], page_start=2),
        ],
    )


def test_chunk_boundaries_overlap_and_provenance() -> None:
    """子块只在父块内重叠，引用与版本信息保持可追溯。"""

    chunks = chunk_document(
        sample_document(), kb_id="shop", doc_id="refund", doc_version_id="v1",
        topic="after_sale", audience="all", max_child_chars=39,
        overlap_chars=15,
    )
    refund = [chunk for chunk in chunks if chunk.title == "退款"]
    shipping = [chunk for chunk in chunks if chunk.title == "运费"]
    assert len(refund) >= 2
    assert len({chunk.parent_id for chunk in refund}) == 1
    assert shipping[0].parent_id != refund[0].parent_id
    assert "七天内可退货" in refund[0].parent_text
    assert "拆封|人工审核" in refund[0].parent_text
    assert "运费由平台承担" not in refund[0].parent_text
    assert any("七天内可退货" in chunk.child_text and "需保留包装" in chunk.child_text for chunk in refund)
    assert all("运费由平台承担" not in chunk.child_text for chunk in refund)
    assert [chunk.page_start for chunk in shipping] == [2]
    assert all(chunk.doc_version_id == "v1" and chunk.source_name == "refund.pdf" for chunk in chunks)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunk_document(
        sample_document(), kb_id="shop", doc_id="refund", doc_version_id="v1",
        topic="after_sale", audience="all", max_child_chars=39, overlap_chars=15,
    )]


def test_oversized_atomic_rule_requires_review() -> None:
    """单个语义块超预算时拒绝强行截断。"""

    document = sample_document()
    document.blocks[1].text = "条件" * 100
    with pytest.raises(ValueError, match="atomic"):
        chunk_document(document, kb_id="shop", doc_id="refund", doc_version_id="v1",
                       topic="after_sale", audience="all", max_child_chars=30, overlap_chars=5)


def test_bailian_requires_both_vectors_and_uses_correct_text_type() -> None:
    """查询使用 query 模式，缺少模型稀疏向量时拒绝结果。"""

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = __import__("json").loads(request.content)
        requests.append(data)
        return httpx.Response(200, json={"output": {"embeddings": [
            {"text_index": 0, "embedding": [0.1, 0.2],
             "sparse_embedding": [{"index": 3, "value": 0.5}]}
        ]}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        model = BailianEmbeddings(api_key="test", dimension=2,
                                  endpoint="https://example.test/embed", http_client=http_client)
        vector = model.embed(["会员券"], text_type="query")[0]
    assert vector.dense == [0.1, 0.2]
    assert vector.sparse == {3: 0.5}
    assert requests[0]["parameters"] == {"text_type": "query", "dimension": 2, "output_type": "dense&sparse"}
    assert requests[0]["input"] == {"texts": ["会员券"]}

    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"output": {"embeddings": [
            {"text_index": 0, "embedding": [0.1, 0.2]}
        ]}})
    )) as http_client:
        with pytest.raises(ValueError, match="sparse"):
            BailianEmbeddings(api_key="test", dimension=2,
                              endpoint="https://example.test/embed", http_client=http_client).embed(
                ["会员券"], text_type="document"
            )


def test_embedding_requires_dual_capable_model_and_workspace_endpoint() -> None:
    """没有工作空间地址或选择不支持双路的 flash 时提前失败。"""

    with pytest.raises(ValueError, match="endpoint"):
        BailianEmbeddings(api_key="test")
    with pytest.raises(ValueError, match="flash"):
        BailianEmbeddings(api_key="test", model="qwen3.7-text-embedding-flash",
                          endpoint="https://example.test/embed")


class FakeSearchClient:
    """模拟 Milvus 命中结果，避免单元测试依赖外部服务。"""

    def __init__(self) -> None:
        self.reqs = None
        self.collection_name = None

    def hybrid_search(self, collection_name, reqs, ranker, limit, output_fields, **kwargs):
        self.collection_name = collection_name
        self.reqs = reqs
        return [[{"chunk_id": "v1:p0001:c0001", "distance": 0.9, "entity": {
            "kb_id": "shop", "doc_id": "refund", "doc_version_id": "v1",
            "parent_id": "v1:p0001", "child_text": "七天内可退", "parent_text": "退款规则",
            "source_name": "refund.pdf", "title": "退款", "heading_path": '["退款"]',
            "page_start": 1, "page_end": 1,
        }}]]


def test_hybrid_search_filters_both_paths_and_returns_citations() -> None:
    """稠密、稀疏检索共享有效期过滤，并返回来源和父块原文。"""

    client = FakeSearchClient()
    store = MilvusKnowledgeStore(client=client, dimension=2)
    results = store.search(
        kb_id="shop", dense=[0.1, 0.2], sparse={3: 0.5},
        at=datetime(2026, 1, 1, tzinfo=timezone.utc), topic="after_sale", version_ids=["v1"],
    )
    assert client.collection_name == "rag_shop"
    assert len(client.reqs) == 2
    assert client.reqs[0].anns_field == "dense_vector"
    assert client.reqs[1].anns_field == "sparse_vector"
    assert client.reqs[0].expr == client.reqs[1].expr
    assert "valid_from_ms" in client.reqs[0].expr
    assert "topic" in client.reqs[0].expr
    assert "doc_version_id in {version_ids}" in client.reqs[0].expr
    assert client.reqs[0].expr_params["version_ids"] == ["v1"]
    assert results[0].source_name == "refund.pdf"
    assert results[0].page_start == 1
    assert results[0].parent_text == "退款规则"


def test_reranker_returns_distinct_parents_in_rank_order() -> None:
    """重排保留模型顺序，最终证据按父块去重。"""

    first = MilvusKnowledgeStore(client=FakeSearchClient(), dimension=2).search(
        kb_id="shop", dense=[0.1, 0.2], sparse={3: 0.5}, at=datetime.now(timezone.utc), version_ids=["v1"],
    )[0]
    hits = [first, replace(first, chunk_id="second", child_text="重复父块"),
            replace(first, chunk_id="third", parent_id="another", child_text="另一个规则")]

    def handler(request: httpx.Request) -> httpx.Response:
        assert __import__("json").loads(request.content)["documents"] == [h.child_text for h in hits]
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": score} for i, score in ((1, 0.9), (2, 0.8), (0, 0.2))
        ]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        reranker = BailianReranker(api_key="test", endpoint="https://example.test/reranks", http_client=http_client)
        ranked = reranker.rerank("退款", hits)
        class RetrievalStore:
            def assert_model(self, kb_id, embeddings):
                assert kb_id == "shop"

            def search(self, **kwargs):
                return hits

        class QueryEmbeddings:
            def embed(self, texts, *, text_type):
                assert text_type == "query"
                from rag.embeddings import DualVector
                return [DualVector([0.1, 0.2], {3: 0.5})]

        evidence = retrieve(query="退款", kb_id="shop", store=RetrievalStore(),
                            embeddings=QueryEmbeddings(), reranker=reranker,
                            at=datetime.now(timezone.utc), version_ids=["v1"], parent_limit=2)
    assert [hit.chunk_id for hit in ranked] == ["second", "third", first.chunk_id]
    assert ranked[0].score == 0.9
    assert [hit.chunk_id for hit in evidence] == ["second", "third"]


def test_milvus_schema_and_dual_indexes_are_explicit() -> None:
    """集合显式定义父子字段、来源字段与两种向量索引。"""

    class FakeClient:
        def __init__(self):
            self.rows = []
            self.created = 0

        def has_collection(self, collection):
            return bool(self.created)

        def create_collection(self, collection_name, schema, index_params):
            self.schema = schema
            self.index_params = index_params
            self.created += 1

        def describe_collection(self, collection_name):
            return {"description": "model=qwen3.7-text-embedding;dim=2"}

        def query(self, collection_name, filter, filter_params, output_fields, **kwargs):
            version_id = filter_params["version_id"]
            return [{"count(*)": sum(row["doc_version_id"] == version_id for row in self.rows)}]

        def get(self, collection_name, ids, output_fields, **kwargs):
            return [{field: row[field] for field in output_fields}
                    for row in self.rows if row["chunk_id"] in ids]

        def insert(self, collection_name, data):
            self.rows.extend(data)

        def flush(self, collection_name):
            pass

        def get_collection_stats(self, collection_name):
            return {"row_count": len(self.rows)}

        def load_collection(self, collection_name):
            pass

    class FakeEmbeddings:
        model = "qwen3.7-text-embedding"
        dimension = 2

        def __init__(self):
            self.calls = 0

        def embed(self, texts, *, text_type):
            from rag.embeddings import DualVector
            self.calls += 1
            return [DualVector([0.1, 0.2], {3: 0.5}) for _ in texts]

    from rag.chunking import chunk_document
    chunks = chunk_document(sample_document(), kb_id="shop", doc_id="refund",
                            doc_version_id="v1", topic="after_sale", max_child_chars=900)
    client = FakeClient()
    store = MilvusKnowledgeStore(client=client, dimension=2)
    embeddings = FakeEmbeddings()
    assert store.stage("shop", "r1", chunks, embeddings) == "rag_shop"
    fields = {field.name: field for field in client.schema.fields}
    assert all(field.description for field in fields.values())
    assert fields["child_text"].description == "用于向量检索的子块正文"
    assert fields["parent_text"].description == "命中后用于补充上下文的父块正文"
    assert fields["sparse_vector"].dtype == DataType.SPARSE_FLOAT_VECTOR
    assert fields["dense_vector"].dtype == DataType.FLOAT_VECTOR
    assert {"doc_version_id", "parent_id", "parent_text", "child_text", "valid_to_ms", "sha256"} <= fields.keys()
    assert {index.field_name for index in client.index_params} == {"dense_vector", "sparse_vector"}
    assert client.rows[0]["parent_text"] and client.rows[0]["sparse_vector"] == {3: 0.5}
    assert store.stage("shop", "r2", chunks, embeddings) == "rag_shop"
    assert client.created == 1 and embeddings.calls == 1 and len(client.rows) == len(chunks)
    with pytest.raises(ValueError, match="Stored document version differs"):
        store.stage("shop", "changed", [replace(chunks[0], child_text="被修改的旧版本"), *chunks[1:]], embeddings)
    assert embeddings.calls == 1
    new_chunks = [replace(chunk, chunk_id=chunk.chunk_id.replace("v1", "v2"),
                          parent_id=chunk.parent_id.replace("v1", "v2"), doc_version_id="v2") for chunk in chunks]
    store.stage("shop", "r3", chunks + new_chunks, embeddings)
    assert client.created == 1 and embeddings.calls == 2 and len(client.rows) == 2 * len(chunks)


def test_query_rejects_model_mismatch_before_search() -> None:
    """线上集合与查询 embedding 模型不一致时阻止检索。"""

    class FakeClient:
        def describe_collection(self, name):
            return {"description": "model=qwen3.7-text-embedding;dim=1024"}

    store = MilvusKnowledgeStore(client=FakeClient(), dimension=1024)
    with pytest.raises(ValueError, match="does not match"):
        store.assert_model("shop", BailianEmbeddings(api_key="test", model="other-model",
                                                     endpoint="https://example.test/embed"))
