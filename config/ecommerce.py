"""配置电商业务后端连接。"""

from pydantic import BaseModel, Field

from config.settings import get_section


class EcommerceSettings(BaseModel):
    """定义电商业务后端的连接参数。"""

    base_url: str = Field(
        default="http://127.0.0.1:8080",
        min_length=1,
    )
    timeout_seconds: float = Field(
        default=5,
        gt=0,
        le=30,
    )


def get_ecommerce_settings() -> EcommerceSettings:
    """读取并校验 ecommerce 配置段。"""

    return get_section("ecommerce", EcommerceSettings)