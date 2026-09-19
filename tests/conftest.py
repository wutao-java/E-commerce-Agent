"""测试环境统一生成并配置 Agent JWT 密钥。"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from config import clear_settings_cache
from security.jwt import get_jwt_settings


_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_KEY_PEM = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("ascii")
_PUBLIC_KEY_PEM = _PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode("ascii")


def _issue_token(**overrides: Any) -> str:
    now = datetime.now(timezone.utc)
    claims: dict[str, Any] = {
        "iss": "commerce-backend",
        "aud": ["commerce-api", "ecommerce-agent", "commerce-agent-api"],
        "sub": "1001",
        "business_user_id": "U1001",
        "nickname": "张三",
        "member_level": "gold",
        "risk_level": "low",
        "scope": "agent:chat agent:facts:read",
        "token_use": "access",
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(minutes=30),
        "jti": uuid4().hex,
    }
    claims.update(overrides)
    return jwt.encode(claims, _PRIVATE_KEY_PEM, algorithm="RS256")


@pytest.fixture(autouse=True)
def configure_agent_jwt(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("AUTH_JWT_PUBLIC_KEY", _PUBLIC_KEY_PEM)
    monkeypatch.setenv("AUTH_JWT_ISSUER", "commerce-backend")
    monkeypatch.setenv("AUTH_JWT_AGENT_AUDIENCE", "ecommerce-agent")
    monkeypatch.setenv("AUTH_JWT_REQUIRED_SCOPE", "agent:chat")
    clear_settings_cache()
    get_jwt_settings.cache_clear()
    yield
    clear_settings_cache()
    get_jwt_settings.cache_clear()


@pytest.fixture
def issue_agent_token() -> Callable[..., str]:
    return _issue_token


@pytest.fixture
def agent_auth_headers(issue_agent_token: Callable[..., str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_agent_token()}"}
