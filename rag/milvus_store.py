"""每个知识库使用一个集合，按不可变文档版本执行混合检索。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from rag.chunking import KnowledgeChunk
from rag.embeddings import BailianEmbeddings


# 知识库 ID 会进入 Milvus 集合名，先限制字符集，避免非法名称。
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_OUTPUT_FIELDS = [
    "kb_id", "doc_id", "doc_version_id", "parent_id", "child_text", "parent_text",
    "source_name", "title", "heading_path", "page_start", "page_end",
]
_FIELD_DESCRIPTIONS = {
    "chunk_id": "子块唯一标识；集合主键",
    "kb_id": "知识库标识",
    "doc_id": "文档的稳定标识",
    "doc_version_id": "不可变文档版本标识",
    "parent_id": "子块所属父块标识",
    "child_text": "用于向量检索的子块正文",
    "parent_text": "命中后用于补充上下文的父块正文",
    "source_name": "原始文件名",
    "source_format": "原始文件格式，如 md、docx、pdf",
    "sha256": "原始文件的 SHA-256 摘要",
    "parser_version": "文档解析器版本",
    "title": "所属章节标题；无标题时为文件名",
    "heading_path": "JSON 编码的章节标题路径",
    "topic": "业务主题，用于检索过滤",
    "audience": "适用受众；all 表示所有人",
    "child_index": "子块在父块内的序号，从 1 开始",
    "page_start": "原件起始页码；0 表示无可靠页码",
    "page_end": "原件结束页码；0 表示无可靠页码",
    "valid_from_ms": "生效起点，UTC Unix 毫秒，包含此时刻",
    "valid_to_ms": "失效边界，UTC Unix 毫秒，不包含此时刻",
    "dense_vector": "子块文本的稠密语义向量",
    "sparse_vector": "子块文本的模型稀疏向量",
}


def _name(value: str) -> str:
    """校验知识库和发布标识中的字符。"""

    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("Knowledge base and release IDs must contain only ASCII letters, digits or underscore (max 40)")
    return value


@dataclass(frozen=True)
class Evidence:
    """命中子块及其父块上下文，附带回答引用所需的原文位置。"""

    chunk_id: str
    kb_id: str
    doc_id: str
    doc_version_id: str
    parent_id: str
    child_text: str
    parent_text: str
    source_name: str
    title: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    score: float


class MilvusKnowledgeStore:
    """负责知识库增量入库与按生效版本执行稠密/稀疏联合召回。"""

    def __init__(self, *, client: MilvusClient, dimension: int = 1024) -> None:
        self.client = client
        self.dimension = dimension

    @staticmethod
    def collection_name(kb_id: str, release_id: str) -> str:
        """同一知识库的不同发布批次共用一个集合。"""

        _name(release_id)
        return f"rag_{_name(kb_id)}"

    def stage(self, kb_id: str, release_id: str, chunks: list[KnowledgeChunk], embeddings: BailianEmbeddings) -> str:
        """校验已有版本，只向知识库集合写入尚未入库的文档版本。"""
        collection = self.collection_name(kb_id, release_id)
        if not chunks or any(chunk.kb_id != kb_id for chunk in chunks):
            raise ValueError("Snapshot must contain only chunks from its knowledge base")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs in snapshot")
        if embeddings.dimension != self.dimension:
            raise ValueError("Embedding dimension differs from collection schema")
        if not self.client.has_collection(collection):
            # 模型与维度固定在集合描述中；切换模型须单独重建索引。
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                                description=f"model={embeddings.model};dim={self.dimension}")
            for field, length in (
                ("chunk_id", 180), ("kb_id", 64), ("doc_id", 128), ("doc_version_id", 128),
                ("parent_id", 160), ("child_text", 8192), ("parent_text", 65535),
                ("source_name", 512), ("source_format", 12), ("sha256", 64),
                ("parser_version", 80), ("title", 512), ("heading_path", 2048),
                ("topic", 128), ("audience", 128),
            ):
                schema.add_field(field_name=field, datatype=DataType.VARCHAR, max_length=length,
                                 is_primary=(field == "chunk_id"), description=_FIELD_DESCRIPTIONS[field])
            for field in ("child_index", "page_start", "page_end", "valid_from_ms", "valid_to_ms"):
                schema.add_field(field_name=field, datatype=DataType.INT64, description=_FIELD_DESCRIPTIONS[field])
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension,
                             description=_FIELD_DESCRIPTIONS["dense_vector"])
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR,
                             description=_FIELD_DESCRIPTIONS["sparse_vector"])
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                              params={"M": 16, "efConstruction": 200})
            indexes.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            self.client.create_collection(collection_name=collection, schema=schema, index_params=indexes)
        elif self.client.describe_collection(collection).get("description") != f"model={embeddings.model};dim={self.dimension}":
            raise ValueError("Embedding model or dimension differs from the knowledge base collection")

        versions: dict[str, list[KnowledgeChunk]] = {}
        for chunk in chunks:
            versions.setdefault(chunk.doc_version_id, []).append(chunk)
        for version_id, version_chunks in versions.items():
            count = self._version_count(collection, version_id)
            expected_rows = []
            for chunk in version_chunks:
                row = dict(vars(chunk))
                row["heading_path"] = json.dumps(chunk.heading_path, ensure_ascii=False)
                row["page_start"] = chunk.page_start or 0
                row["page_end"] = chunk.page_end or 0
                for field in ("child_text", "parent_text", "heading_path", "title", "source_name"):
                    if len(row[field].encode("utf-8")) > {"child_text": 8192, "parent_text": 65535,
                        "heading_path": 2048, "title": 512, "source_name": 512}[field]:
                        raise ValueError(f"Milvus VARCHAR field exceeds byte limit: {field}")
                expected_rows.append(row)
            if count:
                if count != len(expected_rows):
                    raise ValueError(f"Stored chunk count differs for document version {version_id}")
                actual = []
                for offset in range(0, len(expected_rows), 1000):
                    batch = expected_rows[offset:offset + 1000]
                    actual.extend(self.client.get(collection_name=collection,
                                                  ids=[row["chunk_id"] for row in batch],
                                                  output_fields=list(batch[0])))
                by_id = {row["chunk_id"]: row for row in actual}
                if len(by_id) != count or any(by_id.get(row["chunk_id"]) != row for row in expected_rows):
                    raise ValueError(f"Stored document version differs from input: {version_id}")
                continue
            for offset in range(0, len(expected_rows), 10):
                batch = expected_rows[offset:offset + 10]
                vectors = embeddings.embed([row["child_text"] for row in batch], text_type="document")
                rows = [dict(row, dense_vector=vector.dense, sparse_vector=vector.sparse)
                        for row, vector in zip(batch, vectors, strict=True)]
                self.client.insert(collection_name=collection, data=rows)
        self.client.flush(collection_name=collection)
        self.client.load_collection(collection_name=collection)
        self.verify_versions(kb_id, versions.keys(), counts={key: len(value) for key, value in versions.items()})
        return collection

    def _version_count(self, collection: str, version_id: str) -> int:
        result = self.client.query(collection_name=collection,
                                   filter="doc_version_id == {version_id}",
                                   filter_params={"version_id": version_id},
                                   output_fields=["count(*)"], consistency_level="Strong")
        return int(result[0]["count(*)"])

    def verify_versions(self, kb_id: str, version_ids, *, counts: dict[str, int] | None = None) -> None:
        """发布前确认版本实体存在；已知分块数时同时核对完整性。"""
        collection = self.collection_name(kb_id, "verified")
        if not self.client.has_collection(collection) or not version_ids:
            raise ValueError("Knowledge base collection or document versions are missing")
        for version_id in version_ids:
            count = self._version_count(collection, version_id)
            if not count or (counts is not None and count != counts[version_id]):
                raise ValueError(f"Document version is missing or incomplete: {version_id}")

    def assert_model(self, kb_id: str, embeddings: BailianEmbeddings) -> None:
        """查询前核对线上集合的模型与维度，避免跨版本检索。"""

        description = self.client.describe_collection(self.collection_name(kb_id, "active")).get("description")
        if description != f"model={embeddings.model};dim={embeddings.dimension}" or embeddings.dimension != self.dimension:
            raise ValueError("Query embedding model does not match the active release")

    def search(
        self, *, kb_id: str, dense: list[float], sparse: dict[int, float],
        at: datetime, version_ids: list[str], topic: str | None = None, audience: str = "all",
        candidates: int = 30,
    ) -> list[Evidence]:
        """两路使用相同的业务过滤条件，RRF 融合后返回可引用证据。"""

        if at.tzinfo is None or len(dense) != self.dimension or not sparse or candidates < 1 or not version_ids:
            raise ValueError("Valid time, matching vectors and positive candidate count are required")
        # 参数绑定避免把主题、受众或知识库 ID 直接拼入过滤表达式。
        expr = "kb_id == {kb_id} and doc_version_id in {version_ids} and valid_from_ms <= {at_ms} and valid_to_ms > {at_ms} and (audience == 'all' or audience == {audience})"
        params: dict[str, Any] = {"kb_id": kb_id, "version_ids": sorted(set(version_ids)),
                                  "at_ms": int(at.astimezone(timezone.utc).timestamp() * 1000), "audience": audience}
        if topic is not None:
            expr += " and topic == {topic}"
            params["topic"] = topic
        # 两路各召回 candidates 个子块，Milvus 用 RRF 合并排序和去重。
        requests = [
            AnnSearchRequest(data=[dense], anns_field="dense_vector", param={"metric_type": "COSINE", "params": {"ef": 128}}, limit=candidates, expr=expr, expr_params=params),
            AnnSearchRequest(data=[sparse], anns_field="sparse_vector", param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=candidates, expr=expr, expr_params=params),
        ]
        hits = self.client.hybrid_search(
            collection_name=self.collection_name(kb_id, "active"), reqs=requests, ranker=RRFRanker(),
            limit=candidates, output_fields=_OUTPUT_FIELDS, consistency_level="Strong",
        )[0]
        # 把标题路径和无页码标记还原成对上层友好的引用结构。
        return [Evidence(
            chunk_id=hit["chunk_id"], score=float(hit["distance"]),
            heading_path=json.loads(hit["entity"]["heading_path"]),
            page_start=hit["entity"]["page_start"] or None,
            page_end=hit["entity"]["page_end"] or None,
            **{field: hit["entity"][field] for field in _OUTPUT_FIELDS if field not in ("heading_path", "page_start", "page_end")},
        ) for hit in hits]
