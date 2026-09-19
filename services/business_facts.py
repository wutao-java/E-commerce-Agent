"""查询并整理可信的实时业务事实。"""

from __future__ import annotations

import logging
import time
from typing import Any

from domain import BusinessFactNeed, BusinessFactResult
from integrations import EcommerceClient, EcommerceClientError


logger = logging.getLogger(__name__)


def _fact_text(value: Any, default: str = "待查") -> str:
    """将业务字段转换为可展示的非空文本。"""

    if value is None:
        return default
    text = str(value).strip()
    return text or default


class BusinessFactService:
    """协调事实客户端并生成确定性业务摘要。"""

    def __init__(
        self,
        ecommerce_client: EcommerceClient | None = None,
    ) -> None:
        self._ecommerce_client = ecommerce_client or EcommerceClient()

    def lookup(
        self,
        need: BusinessFactNeed,
        *,
        user_message: str,
        access_token: str | None,
    ) -> BusinessFactResult:
        """根据识别结果查询订单、物流或商品事实。"""

        started_at = time.perf_counter()
        token = (access_token or "").strip()
        logger.info(
            "Business fact lookup started kind=%s has_order_no=%s "
            "has_sku=%s",
            need.kind,
            need.order_no is not None,
            need.sku is not None,
        )

        try:
            if not need.requires_realtime:
                result = BusinessFactResult(
                    source="business_fact_service",
                    found=False,
                    summary="本轮问题不需要查询实时业务事实。",
                    failure_reason="not_realtime",
                )
            elif not token:
                result = BusinessFactResult(
                    source="business_fact_service",
                    found=False,
                    summary="当前无法取得可信登录凭证，不能查询业务事实。",
                    failure_reason="missing_access_token",
                )
            elif need.kind in {"order", "logistics"}:
                result = self._lookup_order(need, token)
            elif need.kind == "product":
                result = self._lookup_product(
                    need,
                    user_message,
                    token,
                )
            else:
                result = BusinessFactResult(
                    source="business_fact_service",
                    found=False,
                    summary="当前版本尚未接入这类实时业务事实。",
                    failure_reason="unsupported_fact_kind",
                )
        except EcommerceClientError as exc:
            logger.warning(
                "Business fact lookup unavailable kind=%s error_type=%s",
                need.kind,
                type(exc).__name__,
            )
            result = BusinessFactResult(
                found=False,
                summary="业务系统暂时不可用，无法确认实时状态。",
                failure_reason="upstream_unavailable",
            )

        logger.info(
            "Business fact lookup completed kind=%s found=%s "
            "failure_reason=%s duration_ms=%.2f",
            need.kind,
            result.found,
            result.failure_reason,
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def _lookup_order(
        self,
        need: BusinessFactNeed,
        access_token: str,
    ) -> BusinessFactResult:
        """查询订单，并按需要生成订单或物流摘要。"""

        if not need.order_no:
            return BusinessFactResult(
                source="business_fact_service",
                found=False,
                summary="请补充需要查询的订单号。",
                failure_reason="missing_order_no",
            )

        order = self._ecommerce_client.get_order(
            need.order_no,
            access_token,
        )
        if order is None:
            return BusinessFactResult(
                found=False,
                summary="没有查到该订单，或该订单不属于当前登录用户。",
                failure_reason="not_found",
            )

        summary = (
            self._summarize_logistics(order)
            if need.kind == "logistics"
            else self._summarize_order(order)
        )
        return BusinessFactResult(
            found=True,
            summary=summary,
            facts={"order": order},
        )

    def _lookup_product(
        self,
        need: BusinessFactNeed,
        user_message: str,
        access_token: str,
    ) -> BusinessFactResult:
        """查询商品列表并匹配 SKU 或商品名称。"""

        products = self._ecommerce_client.list_products(access_token)
        product = self._find_product(
            products,
            need.sku,
            user_message,
        )
        if product is None:
            return BusinessFactResult(
                found=False,
                summary="没有识别到具体商品，请补充商品 SKU 或完整商品名称。",
                failure_reason=(
                    "not_found"
                    if need.sku
                    else "missing_product_identifier"
                ),
            )

        name = _fact_text(product.get("name"), "该商品")
        sku = _fact_text(product.get("sku"), "未知 SKU")
        sale_price = _fact_text(product.get("salePrice"))
        stock = _fact_text(product.get("stock"), "未知")

        return BusinessFactResult(
            found=True,
            summary=(
                f"{name}（{sku}）当前售价 {sale_price} 元，"
                f"库存 {stock} 件。"
            ),
            facts={"product": product},
        )

    @staticmethod
    def _find_product(
        products: list[dict[str, Any]],
        sku: str | None,
        user_message: str,
    ) -> dict[str, Any] | None:
        """优先按 SKU，其次按完整名称或去除品牌后的名称匹配。"""

        if sku:
            normalized_sku = sku.casefold()
            for product in products:
                if _fact_text(
                    product.get("sku"),
                    "",
                ).casefold() == normalized_sku:
                    return product

        normalized_message = user_message.casefold()
        for product in products:
            name = _fact_text(product.get("name"), "")
            if not name:
                continue
            short_name = name.split(maxsplit=1)[-1]
            if (
                name.casefold() in normalized_message
                or short_name.casefold() in normalized_message
            ):
                return product
        return None

    @staticmethod
    def _summarize_order(order: dict[str, Any]) -> str:
        """生成不依赖模型的订单状态摘要。"""

        return (
            f"订单 {_fact_text(order.get('orderNo'))} 当前状态为 "
            f"{_fact_text(order.get('status'))}，支付状态为 "
            f"{_fact_text(order.get('paymentStatus'))}，履约状态为 "
            f"{_fact_text(order.get('fulfillmentStatus'))}。"
        )

    @staticmethod
    def _summarize_logistics(order: dict[str, Any]) -> str:
        """从订单及其物流事件生成最新物流摘要。"""

        order_no = _fact_text(order.get("orderNo"))
        status = _fact_text(order.get("status"))
        payment_status = _fact_text(order.get("paymentStatus"))
        fulfillment_status = _fact_text(
            order.get("fulfillmentStatus")
        )

        if payment_status == "UNPAID":
            return f"订单 {order_no} 尚未支付，还没有进入发货流程。"

        if fulfillment_status in {
            "UNSHIPPED",
            "PENDING_SHIPMENT",
        }:
            return (
                f"订单 {order_no} 当前状态为 {status}，"
                "尚未发货，因此暂时没有物流轨迹。"
            )

        raw_events = order.get("logisticsEvents")
        events = (
            [event for event in raw_events if isinstance(event, dict)]
            if isinstance(raw_events, list)
            else []
        )
        if events:
            latest = events[-1]
            return (
                f"订单 {order_no} 最新物流："
                f"{_fact_text(latest.get('carrier'), '未知承运方')}，"
                f"{_fact_text(latest.get('content'), '暂无轨迹说明')}，"
                f"时间 {_fact_text(latest.get('occurredAt'), '待更新')}。"
            )

        tracking_no = _fact_text(order.get("trackingNo"), "")
        if tracking_no:
            return (
                f"订单 {order_no} 已生成运单 {tracking_no}，"
                "但暂未查到物流轨迹。"
            )

        return (
            f"订单 {order_no} 当前状态为 {status}，"
            "暂未查到运单或物流轨迹。"
        )