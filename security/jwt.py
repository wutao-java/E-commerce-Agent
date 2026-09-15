"""提供 JWT 配置加载、签发和校验能力。"""

# 启用延迟解析类型注解。
from __future__ import annotations

# 导入时间计算、单实例缓存和通用类型能力。
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

# 导入 JWT 编解码和配置模型依赖。
import jwt
from pydantic import BaseModel, Field

# 导入统一的命名配置段读取接口。
from config import get_section


class JwtSettings(BaseModel):
    """定义 JWT 密钥、算法和有效期配置。"""

    # 私钥默认为空，由实际启用签发能力的环境提供。
    private_key: str = ""
    # 公钥默认为空，由实际启用校验能力的环境提供。
    public_key: str = ""
    # 默认使用非对称 RSA SHA-256 签名算法。
    algorithm: str = "RS256"
    # 限制令牌默认有效期至少为一分钟。
    expire_minutes: int = Field(default=60, ge=1)


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


def create_token(claims: dict[str, Any], expires_in: timedelta | None = None) -> str:
    """签发包含指定声明和有效期的 JWT"""

    # 读取签名算法、默认有效期和私钥配置。
    settings = get_jwt_settings()
    # 使用 UTC 时间生成跨时区一致的签发时间。
    now = datetime.now(timezone.utc)
    # 复制调用方声明，避免向原始字典写入系统字段。
    payload = dict(claims)
    # 写入签发时间和到期时间，覆盖调用方可能提供的同名字段。
    payload.update(
        {
            "iat": now,
            "exp": now + (expires_in or timedelta(minutes=settings.expire_minutes)),
        }
    )
    # 标准化并验证用于签名的私钥。
    private_key = _normalize_key(settings.private_key, "JWT private key")
    # 使用配置的算法签名并返回紧凑令牌字符串。
    return jwt.encode(payload, private_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """校验并解码 JWT"""

    # 读取验签算法和公钥配置。
    settings = get_jwt_settings()
    # 标准化并验证用于验签的公钥。
    public_key = _normalize_key(settings.public_key, "JWT public key")
    # 仅允许配置的算法，防止令牌自行降级签名算法。
    return jwt.decode(token, public_key, algorithms=[settings.algorithm])
