"""使用独立的 OpenAI-compatible embedding 服务。"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import Lock

import httpx

from config.rag import RagSettings


class EmbeddingClient:
    """OpenAI-compatible embedding 客户端，带线程安全的查询向量缓存。"""

    def __init__(self, settings: RagSettings, http_client: httpx.Client | None = None) -> None:
        """初始化客户端。

        Args:
            settings: RAG 配置，包含 embedding 服务地址、模型和密钥。
            http_client: 可选的 HTTP 客户端，便于复用连接或在测试中注入替身。
        """
        self.settings = settings
        self.http_client = http_client
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序批量生成文本向量。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            与 ``texts`` 顺序一致的向量列表。

        Raises:
            RuntimeError: 密钥缺失、服务调用失败或返回的向量结构无效。
        """
        key = self.settings.embedding_api_key.get_secret_value()
        if not key or key in {"your-api-key", "YOUR_API_KEY"}:
            raise RuntimeError("缺少 COURSE_RAG_EMBEDDING_API_KEY，无法检索课程知识。")
        vectors: list[list[float]] = []
        # 限制单次请求规模，避免大批量知识发布时超过服务端输入上限。
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            kwargs = {
                "headers": {"Authorization": f"Bearer {key}"},
                "json": {"model": self.settings.embedding_model, "input": batch},
                "timeout": 60,
            }
            url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"
            try:
                response = (self.http_client.post(url, **kwargs) if self.http_client else httpx.post(url, **kwargs))
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                parsed = [[float(value) for value in item["embedding"]] for item in data]
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                raise RuntimeError("课程 embedding 服务调用失败。") from exc
            if len(parsed) != len(batch) or any(not vector or len(vector) != len(parsed[0]) for vector in parsed):
                raise RuntimeError("课程 embedding 服务返回的向量数量或维度不一致。")
            vectors.extend(parsed)
        return vectors

    def embed(self, text: str) -> list[float]:
        """生成单条文本向量，并按服务配置与文本内容缓存结果。

        Args:
            text: 待向量化的查询文本。

        Returns:
            查询文本对应的向量。
        """
        raw_key = json.dumps(
            [self.settings.embedding_base_url, self.settings.embedding_model, text],
            ensure_ascii=False,
        )
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
                return cached
        vector = self.embed_many([text])[0]
        with self._lock:
            self._query_cache[cache_key] = vector
            self._query_cache.move_to_end(cache_key)
            # 缓存采用 LRU 淘汰策略，防止查询种类持续增长占满内存。
            if len(self._query_cache) > 256:
                self._query_cache.popitem(last=False)
        return vector
