"""提供大模型连接配置。"""

from pydantic import BaseModel, Field, SecretStr

from config.settings import get_section


class LlmSettings(BaseModel):
    """定义 OpenAI-compatible 模型连接参数。"""

    # SecretStr 避免在配置对象的普通日志或 repr 中直接显示 API Key。
    api_key: SecretStr = SecretStr("")
    base_url: str = Field(
        default="https://api.deepseek.com",
        min_length=1,
    )
    model: str = Field(
        default="deepseek-v4-pro",
        min_length=1,
    )
    # 意图分类可单独选择模型；未指定时与回答模型使用相同默认值。
    classifier_model: str = Field(
        default="deepseek-v4-pro",
        min_length=1,
    )
    timeout_seconds: float = Field(default=30, gt=0)


def get_llm_settings() -> LlmSettings:
    """读取并校验 llm 配置段。"""

    return get_section("llm", LlmSettings)



