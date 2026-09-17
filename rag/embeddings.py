"""通过百炼原生接口一次获取稠密向量与模型稀疏向量。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import httpx


@dataclass(frozen=True)
class DualVector:
    """稀疏分量以词表索引到权重的映射表示，并非 BM25 分数。"""

    dense: list[float]
    sparse: dict[int, float]


class BailianEmbeddings:
    """使用同一模型配置分别编码入库文档与用户查询。"""

    def __init__(
        self, *, api_key: str, model: str = "qwen3.7-text-embedding",
        dimension: int = 1024, endpoint: str = "",
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip() or dimension < 1 or not endpoint.strip():
            raise ValueError("Embedding key, model, endpoint and positive dimension are required")
        if model == "qwen3.7-text-embedding-flash":
            # 当前双路接口文档不保证 flash 返回稀疏向量，避免建出单路索引。
            raise ValueError("flash does not document dense&sparse output; use a dual-capable model")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.endpoint = endpoint
        self.http_client = http_client

    def embed(self, texts: list[str], *, text_type: Literal["document", "query"]) -> list[DualVector]:
        """请求双路向量；document 用于入库，query 用于在线检索。"""

        if text_type not in ("document", "query") or not texts or any(not text.strip() for text in texts):
            raise ValueError("Non-empty texts and valid text_type are required")
        # OpenAI 兼容接口不提供此处所需的 sparse_embedding，使用原生参数。
        request = {
            "headers": {"Authorization": f"Bearer {self.api_key}"},
            "json": {"model": self.model, "input": {"texts": texts}, "parameters": {
                "text_type": text_type, "dimension": self.dimension, "output_type": "dense&sparse",
            }},
            "timeout": 30,
        }
        try:
            response = (self.http_client or httpx).post(self.endpoint, **request)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Bailian embedding request failed") from exc
        if not isinstance(payload, dict) or payload.get("code"):
            raise RuntimeError(f"Bailian embedding error: {payload.get('code', 'invalid response') if isinstance(payload, dict) else 'invalid response'}")
        output = payload.get("output")
        entries = output.get("embeddings") if isinstance(output, dict) else None
        if not isinstance(entries, list) or len(entries) != len(texts):
            raise ValueError("Embedding response count mismatch")
        # 按 text_index 对齐输入顺序，不能假设服务端按请求顺序返回。
        vectors: list[DualVector | None] = [None] * len(texts)
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError("Invalid embedding response item")
            index, dense, sparse = item.get("text_index"), item.get("embedding"), item.get("sparse_embedding")
            if type(index) is not int or not 0 <= index < len(texts) or vectors[index] is not None:
                raise ValueError("Invalid embedding text_index")
            if not isinstance(dense, list) or len(dense) != self.dimension or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) for v in dense
            ):
                raise ValueError("Invalid dense embedding dimension or values")
            if not isinstance(sparse, list) or not sparse:
                raise ValueError("Missing sparse embedding")
            # 缺少任一路、维度错误或索引重复时拒绝入库，防止污染集合。
            sparse_vector: dict[int, float] = {}
            for entry in sparse:
                if not isinstance(entry, dict):
                    raise ValueError("Invalid sparse embedding")
                key, value = entry.get("index"), entry.get("value")
                if type(key) is not int or key < 0 or key in sparse_vector or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Invalid sparse embedding index or value")
                sparse_vector[key] = float(value)
            vectors[index] = DualVector([float(v) for v in dense], sparse_vector)
        if any(vector is None for vector in vectors):
            raise ValueError("Missing embedding text_index")
        return [vector for vector in vectors if vector is not None]
