"""仅改写检索查询，不修改用户原话或业务事实。"""

from __future__ import annotations

import re

from domain import ChatCommand, Intent
from domain.rag import KnowledgeChunk, RetrievalPlan


REPLACEMENTS = {
    "耳麦": "耳机",
    "叠券": "叠加优惠券",
    "会员券": "优惠券",
    "盒子": "包装盒",
    "压了": "压坏",
}


def normalize_query(text: str) -> str:
    """统一查询中的大小写、空白和课程领域同义表达。"""
    normalized = text.strip().lower()
    for source, target in REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def is_realtime_query(query: str) -> bool:
    """判断查询是否依赖订单、物流、退款或库存等实时业务数据。"""
    text = normalize_query(query)
    if any(term in text for term in (
        "我的订单", "订单到哪", "快递到哪", "物流到哪", "退款进度",
        "退款状态", "库存还有", "有没有库存", "发货了吗", "我的快递",
    )):
        return True
    has_order_id = re.search(r"\bso\d{14,}-a\d+\b", text) is not None
    return has_order_id and any(term in text for term in ("物流", "快递", "发货", "退款", "到哪"))


def asks_for_history(query: str) -> bool:
    """判断用户是否明确请求历史或已过期的课程资料。"""
    return any(word in normalize_query(query) for word in ("历史", "复盘", "过期", "2024", "2025", "去年"))


def pre_retrieval_plan(command: ChatCommand, intent: Intent, chunks: list[KnowledgeChunk]) -> RetrievalPlan:
    """根据用户原话和意图生成检索前计划。

    查询改写仅补充检索词，不修改保存在计划中的用户原话，也不生成业务事实。

    Args:
        command: 当前聊天命令及运行时用户信息。
        intent: 上游识别出的用户意图。
        chunks: 当前索引中的全部知识片段，用于确定主题和关键词集合。

    Returns:
        包含改写查询、场景、允许主题、关键词及实时标记的检索计划。
    """
    original = command.user_message
    normalized = normalize_query(original)
    if intent == "promotion_consult" or any(term in normalized for term in ("优惠", "活动", "会员价", "优惠券")):
        scene, topics = "promotion", ["promotion", "product"]
    elif intent == "refund_request" or any(term in normalized for term in ("退货", "配件", "赠品", "包装盒")):
        scene, topics = "after_sale", ["after_sale", "order"]
    elif intent == "order_query":
        scene, topics = "shipping", ["shipping", "order"]
    elif intent == "product_consult":
        scene, topics = "product", ["product", "promotion"]
    elif intent == "complaint":
        scene, topics = "complaint", ["complaint", "after_sale"]
    else:
        scene, topics = "unknown", sorted({chunk.topic for chunk in chunks})

    # 针对课程中的固定促销场景补充召回词，提高短查询的可检索性。
    additions: list[str] = []
    if scene == "promotion" and "耳机" in normalized and "活动" in normalized:
        additions = ["会员价", "优惠券", "叠加", "结算页"]
    if command.runtime_member_level == "gold" and "会员" in normalized:
        additions.append("金卡")
    rewritten = " ".join([normalized, *(term for term in additions if term not in normalized)])
    # 仅保留实际出现在改写查询中的知识库关键词，供精确关键词通道使用。
    terms = sorted(
        {term for chunk in chunks for term in chunk.keywords if term.lower() in rewritten},
        key=lambda item: (-len(item), item),
    )
    return RetrievalPlan(
        original_query=original,
        rewritten_query=rewritten,
        scene=scene,
        allowed_topics=topics,
        keyword_terms=terms,
        realtime=is_realtime_query(original),
    )
