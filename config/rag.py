"""仅在调用 RAG 能力时读取百炼、Milvus 的可选连接配置。"""

from pydantic import BaseModel, Field, SecretStr

from config.settings import get_section


class RagSettings(BaseModel):
    """配置模型、向量维度和外部端点；缺少密钥不影响普通聊天启动。"""

    # 密钥包装为 SecretStr，避免打印配置对象时泄露凭据。
    api_key: SecretStr = SecretStr("")
    # 百炼工作空间的原生 embedding 接口和 rerank 接口需分别配置。
    embedding_endpoint: str = ""
    rerank_endpoint: str = ""
    embedding_model: str = "qwen3.7-text-embedding"
    dimension: int = Field(default=1024, ge=1)
    milvus_uri: str = ""
    milvus_token: SecretStr = SecretStr("")
    prepare_on_start: bool = True


def get_rag_settings() -> RagSettings:
    """按需校验 INI 中的 rag 配置段。"""

    return get_section("rag", RagSettings)
