"""识别需要访问业务系统的实时事实问题。"""

from __future__ import annotations

import logging
import re

from domain import BusinessFactNeed


logger = logging.getLogger(__name__)

ORDER_NO_PATTERN = re.compile(
    r"\b(?:SO\d{14,}-[A-Za-z]\d+|EC\d{14}[A-Fa-f0-9]{6})\b",
    re.IGNORECASE,
)
SKU_PATTERN = re.compile(
    r"\b[A-Za-z]{2,10}(?:-[A-Za-z0-9]{2,10}){2,}\b",
    re.IGNORECASE,
)

REFUND_STATUS_TERMS = (
    "退款进度",
    "退款到哪",
    "退款状态",
    "退到哪",
)
LOGISTICS_TERMS = (
    "我的物流",
    "我的快递",
    "物流到哪",
    "快递到哪",
    "物流状态",
    "物流信息",
    "发货了吗",
    "有没有发货",
    "运单",
)
PRODUCT_FACT_TERMS = (
    "库存",
    "多少钱",
    "当前价",
    "现在价格",
    "有货吗",
    "现货吗",
)
ORDER_STATUS_TERMS = (
    "我的订单",
    "订单状态",
    "订单到哪",
    "订单怎么样",
)


def extract_order_no(user_message: str) -> str | None:
    """从用户消息中提取商城订单号。"""

    match = ORDER_NO_PATTERN.search(user_message)
    return match.group(0) if match else None


def extract_sku(user_message: str) -> str | None:
    """从用户消息中提取商品 SKU。"""

    match = SKU_PATTERN.search(user_message)
    return match.group(0).upper() if match else None


def detect_business_fact_need(user_message: str) -> BusinessFactNeed:
    """判断本轮问题是否必须查询实时业务事实。"""

    order_no = extract_order_no(user_message)
    sku = extract_sku(user_message)

    if any(term in user_message for term in REFUND_STATUS_TERMS):
        need = BusinessFactNeed(
            kind="unknown",
            requires_realtime=True,
            order_no=order_no,
            reason="退款进度属于实时事实，但当前版本尚未接入退款事实接口。",
        )
    elif any(term in user_message for term in LOGISTICS_TERMS):
        need = BusinessFactNeed(
            kind="logistics",
            requires_realtime=True,
            order_no=order_no,
            reason="物流状态会持续变化，必须查询业务系统。",
        )
    elif any(term in user_message for term in PRODUCT_FACT_TERMS):
        need = BusinessFactNeed(
            kind="product",
            requires_realtime=True,
            sku=sku,
            reason="商品库存和当前价格必须查询业务系统。",
        )
    elif order_no or any(term in user_message for term in ORDER_STATUS_TERMS):
        need = BusinessFactNeed(
            kind="order",
            requires_realtime=True,
            order_no=order_no,
            reason="订单状态属于当前用户的实时业务事实。",
        )
    else:
        need = BusinessFactNeed(
            kind="unknown",
            requires_realtime=False,
            reason="没有识别到实时业务事实查询。",
        )

    logger.info(
        "Business fact need detected kind=%s requires_realtime=%s "
        "has_order_no=%s has_sku=%s",
        need.kind,
        need.requires_realtime,
        need.order_no is not None,
        need.sku is not None,
    )
    return need
