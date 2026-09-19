"""仅存取课程 collection 的向量及知识 ID。"""

from __future__ import annotations

import json

from pymilvus import MilvusClient

from config.rag import RagSettings
from domain.rag import KnowledgeChunk


class CourseMilvusStore:
    """课程知识专用的 Milvus collection 访问层。"""

    def __init__(self, settings: RagSettings, client: MilvusClient | None = None) -> None:
        """初始化存储对象，允许注入客户端以复用连接或支持测试。"""
        self.settings = settings
        self.client = client

    def _client(self) -> MilvusClient:
        """延迟创建并返回 Milvus 客户端。"""
        if self.client is None:
            token = self.settings.milvus_token.get_secret_value()
            kwargs = {"uri": self.settings.milvus_uri}
            if token:
                kwargs["token"] = token
            self.client = MilvusClient(**kwargs)
        return self.client

    def collection_name(self, version: str) -> str:
        """根据知识索引版本生成隔离的 collection 名称。"""
        return f"{self.settings.collection_prefix}_{version}"

    def publish(self, version: str, chunks: list[KnowledgeChunk], vectors: list[list[float]]) -> str:
        """将知识片段向量写入版本对应的 Milvus collection。

        Args:
            version: 当前知识索引版本。
            chunks: 待发布的课程知识片段。
            vectors: 与 ``chunks`` 一一对应的向量。

        Returns:
            发布完成的 collection 名称。

        Raises:
            ValueError: 知识片段与向量数量不匹配，或向量维度不一致。
            RuntimeError: 已有 collection 的向量维度与本次发布不兼容。
        """
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("课程知识与向量数量不一致。")
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("课程知识向量维度不一致。")
        name = self.collection_name(version)
        client = self._client()
        # 同版本 collection 可重复发布，但必须先确认向量维度兼容。
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
        # 分批 upsert 限制单次写入量，同时允许相同 chunk_id 幂等更新。
        for start in range(0, len(chunks), 32):
            client.upsert(name, [
                {"chunk_id": chunk.chunk_id, "vector": vector}
                for chunk, vector in zip(chunks[start:start + 32], vectors[start:start + 32])
            ])
        client.flush(name)
        self.verify(version, {chunk.chunk_id for chunk in chunks})
        return name

    def verify(self, version: str, expected_ids: set[str]) -> None:
        """校验目标 collection 存在且包含全部预期知识片段 ID。

        Raises:
            RuntimeError: collection 不存在或索引内容不完整。
        """
        name = self.collection_name(version)
        client = self._client()
        if not client.has_collection(name):
            raise RuntimeError("课程 Milvus 索引尚未构建；请执行 python -m rag.commands rebuild。")
        rows = client.query(name, ids=sorted(expected_ids), output_fields=["chunk_id"])
        if {row["chunk_id"] for row in rows} != expected_ids:
            raise RuntimeError("课程 Milvus 索引不完整；请执行 python -m rag.commands rebuild。")

    def search(self, version: str, vector: list[float], allowed_ids: set[str], limit: int) -> list[tuple[str, float]]:
        """在允许的知识 ID 范围内执行向量相似度检索。

        Args:
            version: 待检索的知识索引版本。
            vector: 查询向量。
            allowed_ids: 通过主题、状态和有效期过滤的知识片段 ID。
            limit: 最多返回的命中数量。

        Returns:
            按 Milvus 排序返回的 ``(chunk_id, distance)`` 列表。
        """
        if not allowed_ids:
            return []
        expression = f"chunk_id in {json.dumps(sorted(allowed_ids), ensure_ascii=False)}"
        results = self._client().search(
            self.collection_name(version), data=[vector], filter=expression, limit=limit,
            output_fields=["chunk_id"],
        )
        return [(str(hit["chunk_id"]), float(hit["distance"])) for hit in results[0]]
