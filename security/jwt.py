"""校验 Spring Boot 签发的访问 JWT。"""

# 启用延迟解析类型注解。
from __future__ import annotations

# 导入单实例缓存和通用类型能力。
from functools import lru_cache
from typing import Any, Literal

# 导入 JWT 编解码和配置模型依赖。
import jwt
from pydantic import BaseModel, Field

# 导入统一的命名配置段读取接口。
from config import get_section


class JwtSettings(BaseModel):
    """定义 Spring Boot 访问 JWT 的验签规则。"""

    # Agent 只负责验签，不持有签发私钥。
    public_key: str = ""
    algorithm: Literal["RS256"] = "RS256"
    issuer: str = "commerce-backend"
    audience: str = "ecommerce-agent"
    required_scope: str = "agent:chat"
    leeway_seconds: int = Field(default=5, ge=0, le=60)


@lru_cache(maxsize=1)
def get_jwt_settings() -> JwtSettings:
    """读取并校验 JWT 配置"""

    # 通过统一入口读取 jwt 配置段并缓存校验结果。
    return get_section("jwt", JwtSettings)


def _normalize_key(value: str, label: str) -> str:
    """还原并校验配置中的 JWT 密钥文本"""

    # 将环境变量中的转义换行还原为 PEM 所需的真实换行。
    key = value.replace("\\n", "\n").strip()
    # 空密钥无法用于签名或验签。
    if not key:
        # 使用调用方提供的密钥名称生成可诊断错误。
        raise RuntimeError(f"{label} is not configured")
    # 返回标准化后的非空密钥。
    return key


def decode_token(token: str) -> dict[str, Any]:
    """校验并解码访问 JWT。"""

    # 读取验签算法和公钥配置。
    settings = get_jwt_settings()
    # 标准化并验证用于验签的公钥。
    public_key = _normalize_key(settings.public_key, "JWT public key")
    payload = jwt.decode(
        token,
        public_key,
        algorithms=[settings.algorithm],
        issuer=settings.issuer,
        audience=settings.audience,
        leeway=settings.leeway_seconds,
        options={
            "require": [
                "exp",
                "iat",
                "nbf",
                "iss",
                "aud",
                "sub",
                "jti",
                "token_use",
                "scope",
            ]
        },
    )
    if payload.get("token_use") != "access":
        raise jwt.InvalidTokenError("invalid token_use")
    return payload
