"""仅存取课程 collection 的向量及知识 ID。"""

from __future__ import annotations

import json

from pymilvus import MilvusClient

from config.rag import RagSettings
from domain.rag import KnowledgeChunk


class CourseMilvusStore:
    def __init__(self, settings: RagSettings, client: MilvusClient | None = None) -> None:
        self.settings = settings
        self.client = client

    def _client(self) -> MilvusClient:
        if self.client is None:
            token = self.settings.milvus_token.get_secret_value()
            kwargs = {"uri": self.settings.milvus_uri}
            if token:
                kwargs["token"] = token
            self.client = MilvusClient(**kwargs)
        return self.client

    def collection_name(self, version: str) -> str:
        return f"{self.settings.collection_prefix}_{version}"

    def publish(self, version: str, chunks: list[KnowledgeChunk], vectors: list[list[float]]) -> str:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("课程知识与向量数量不一致。")
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("课程知识向量维度不一致。")
        name = self.collection_name(version)
        client = self._client()
        if client.has_collection(name):
            dimension = client.describe_collection(name)["fields"]
            vector_field = next(field for field in dimension if field["name"] == "vector")
            if vector_field["params"]["dim"] != len(vectors[0]):
                raise RuntimeError("现有课程 collection 与 embedding 维度不一致。")
        else:
            client.create_collection(
                collection_name=name,
                dimension=len(vectors[0]),
                primary_field_name="chunk_id",
                id_type="string",
                max_length=160,
                vector_field_name="vector",
                metric_type="COSINE",
            )
        for start in range(0, len(chunks), 32):
            client.upsert(name, [
                {"chunk_id": chunk.chunk_id, "vector": vector}
                for chunk, vector in zip(chunks[start:start + 32], vectors[start:start + 32])
            ])
        client.flush(name)
        self.verify(version, {chunk.chunk_id for chunk in chunks})
        return name

    def verify(self, version: str, expected_ids: set[str]) -> None:
        name = self.collection_name(version)
        client = self._client()
        if not client.has_collection(name):
            raise RuntimeError("课程 Milvus 索引尚未构建；请执行 python -m rag.commands rebuild。")
        rows = client.query(name, ids=sorted(expected_ids), output_fields=["chunk_id"])
        if {row["chunk_id"] for row in rows} != expected_ids:
            raise RuntimeError("课程 Milvus 索引不完整；请执行 python -m rag.commands rebuild。")

    def search(self, version: str, vector: list[float], allowed_ids: set[str], limit: int) -> list[tuple[str, float]]:
        if not allowed_ids:
            return []
        expression = f"chunk_id in {json.dumps(sorted(allowed_ids), ensure_ascii=False)}"
        results = self._client().search(
            self.collection_name(version), data=[vector], filter=expression, limit=limit,
            output_fields=["chunk_id"],
        )
        return [(str(hit["chunk_id"]), float(hit["distance"])) for hit in results[0]]
