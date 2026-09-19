"""调用 Spring Boot 提供的只读业务事实接口。"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote

import httpx

from config.ecommerce import EcommerceSettings, get_ecommerce_settings


logger = logging.getLogger(__name__)


class EcommerceClientError(RuntimeError):
    """表示电商业务后端调用或响应契约异常。"""


class EcommerceClient:
    """使用当前用户 JWT 查询可信业务事实。"""

    def __init__(self,settings: EcommerceSettings | None = None,http_client: httpx.Client | None = None,) -> None:
        self._settings = settings or get_ecommerce_settings()
        self._http_client = http_client

    def get_order(self,order_no: str,access_token: str,) -> dict[str, Any] | None:
        """查询属于当前登录用户的订单及物流事实。"""

        normalized_order_no = order_no.strip()
        if not normalized_order_no:
            raise ValueError("order_no cannot be empty")

        data = self._get(
            f"/api/agent/facts/orders/{quote(normalized_order_no, safe='')}",
            access_token,
            resource="order",
        )
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.warning(
                "Ecommerce fact response rejected resource=order "
                "reason=invalid_data_type"
            )
            raise EcommerceClientError("订单事实响应格式无效")
        return data

    def list_products(self,access_token: str,) -> list[dict[str, Any]]:
        """查询当前用户可见的商品价格和库存。"""

        data = self._get(
            "/api/agent/facts/products",
            access_token,
            resource="products",
        )
        if not isinstance(data, list) or not all(
            isinstance(item, dict) for item in data
        ):
            logger.warning(
                "Ecommerce fact response rejected resource=products "
                "reason=invalid_data_type"
            )
            raise EcommerceClientError("商品事实响应格式无效")
        return data

    def _get(self,path: str,access_token: str,*,resource: str,) -> Any:
        """执行只读请求并提取统一响应中的 data。"""

        token = access_token.strip()
        if not token:
            logger.warning(
                "Ecommerce fact request rejected resource=%s "
                "reason=missing_access_token",
                resource,
            )
            raise EcommerceClientError("缺少业务事实访问凭证")

        started_at = time.perf_counter()
        logger.info(
            "Ecommerce fact request started resource=%s",
            resource,
        )

        request = (
            self._http_client.get
            if self._http_client is not None
            else httpx.get
        )
        try:
            response = request(
                f"{self._settings.base_url.rstrip('/')}{path}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=self._settings.timeout_seconds,
            )
            if response.status_code == 404:
                logger.info(
                    "Ecommerce fact request completed resource=%s "
                    "found=false duration_ms=%.2f",
                    resource,
                    (time.perf_counter() - started_at) * 1000,
                )
                return None

            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            error_response = getattr(exc, "response", None)
            logger.warning(
                "Ecommerce fact request failed resource=%s "
                "status_code=%s error_type=%s duration_ms=%.2f",
                resource,
                getattr(error_response, "status_code", None),
                type(exc).__name__,
                (time.perf_counter() - started_at) * 1000,
            )
            raise EcommerceClientError(
                "电商业务事实接口暂时不可用"
            ) from exc

        if not isinstance(payload, dict) or payload.get("code") != 0:
            logger.warning(
                "Ecommerce fact response rejected resource=%s "
                "reason=invalid_envelope",
                resource,
            )
            raise EcommerceClientError("电商业务事实响应格式无效")

        logger.info(
            "Ecommerce fact request completed resource=%s "
            "found=true duration_ms=%.2f",
            resource,
            (time.perf_counter() - started_at) * 1000,
        )
        return payload.get("data")