"""提供高置信客服意图识别规则。"""

from __future__ import annotations

import re

from domain import Intent, IntentResult


IntentRule = tuple[Intent, list[str], str]

# 规则列表只生成候选；最终仍按投诉、明确售后目标和单一核心意图的顺序裁决。
INTENT_RULES: list[IntentRule] = [
    ("complaint", ["投诉", "举报", "赔偿", "曝光", "315", "别踢皮球"],
     "用户表达了投诉、赔偿或强烈不满。"),
    ("refund_request", ["退款", "退货", "取消订单", "坏了", "无法开机", "质量问题"],
     "用户在询问退款、退货或质量问题。"),
    ("order_query", ["订单", "物流", "快递", "发货", "到哪", "运单"],
     "用户在询问订单或物流状态。"),
    ("promotion_consult", ["优惠", "活动", "会员价", "券", "满减", "折扣"],
     "用户在询问优惠或活动。"),
    ("product_consult", ["耳机", "充电器", "音箱", "推荐", "哪个好"],
     "用户在询问商品或推荐。"),
    ("general_chat", ["你好", "您好", "在吗", "谢谢"],
     "用户正在进行普通问候。"),
]

# 这些词可能只是问题的背景，不能单独作为高置信意图依据。
CONTEXT_KEYWORDS: dict[Intent, set[str]] = {
    "order_query": {"订单"},
    "product_consult": {"耳机", "充电器", "音箱"},
}

NEGATION_PATTERN = re.compile(
    r"(?:不是|并非|不想|不打算|不需要|无需|不用|不要)"
    r"(?:想要|想|要|申请|办理)?$"
)

REFUND_GOAL_PATTERN = re.compile(
    r"(?:直接|马上|立刻|帮我|给我|我要|我想|申请|办理)"
    r"[^，。！？]{0,8}(?:退款|退货|取消订单|退钱)|"
    r"(?:退款|退货|取消订单|退钱)(?:吗|么|吧|！|。|$)"
)


def is_negated_keyword(message: str, keyword: str) -> bool:
    """检查关键词的每次出现，只有全部被前置否定词修饰才视为否定。"""

    positions = [
        match.start()
        for match in re.finditer(re.escape(keyword), message)
    ]
    # 仅观察关键词前的局部文本，避免把整句话中的其他否定误套到该词上。
    return bool(positions) and all(
        NEGATION_PATTERN.search(message[max(0, position - 8):position])
        for position in positions
    )


def plan_intent_by_rules(user_message: str) -> IntentResult | None:
    """优先处理投诉和明确售后目标，其余只接受无歧义的核心意图。"""

    message = user_message.strip().lower()
    candidates: list[tuple[Intent, list[str], list[str], str]] = []
    has_negation = False

    for intent, keywords, explanation in INTENT_RULES:
        matched = [keyword for keyword in keywords if keyword in message]
        active = [
            keyword
            for keyword in matched
            if not is_negated_keyword(message, keyword)
        ]
        has_negation = has_negation or len(active) != len(matched)

        if active:
            # 区分真正表达诉求的词和仅提供商品、订单背景的词。
            core = [
                keyword
                for keyword in active
                if keyword not in CONTEXT_KEYWORDS.get(intent, set())
            ]
            candidates.append((intent, active, core, explanation))

    # 投诉优先处理；明确要求退款等动作时，再优先识别售后目标。
    complaint = next(
        (item for item in candidates if item[0] == "complaint"),
        None,
    )
    if complaint:
        selected = complaint
    else:
        refund = next(
            (item for item in candidates if item[0] == "refund_request"),
            None,
        )
        if refund and REFUND_GOAL_PATTERN.search(message):
            selected = refund
        else:
            core_candidates = [item for item in candidates if item[2]]
            # 有否定或多个核心诉求时交给分类模型，不强行按词表顺序猜测。
            if has_negation or len(core_candidates) != 1:
                return None
            selected = core_candidates[0]

    return IntentResult(
        intent=selected[0],
        source="rules",
        confidence=0.95,
        matched_keywords=selected[1],
        explanation=selected[3],
    )
