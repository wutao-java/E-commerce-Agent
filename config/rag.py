"""配置隔离的课程知识检索。"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, SecretStr

from config.settings import get_section


class RagSettings(BaseModel):
    enabled: bool = False
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: SecretStr = SecretStr("")
    collection_prefix: str = Field(default="course_rag", max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    chunk_size: int = Field(default=420, ge=100)
    chunk_overlap: int = Field(default=80, ge=0)
    candidate_k: int = Field(default=20, ge=1, le=100)
    top_k: int = Field(default=3, ge=1, le=20)
    min_vector_score: float = Field(default=0.35, ge=0, le=1)
    low_confidence_score: float = Field(default=0.62, ge=0, le=1)
    as_of_date: str = ""
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_api_key: SecretStr = SecretStr("")

    def reference_date(self) -> date:
        return date.fromisoformat(self.as_of_date) if self.as_of_date else date.today()


def get_rag_settings() -> RagSettings:
    return get_section("rag", RagSettings)
