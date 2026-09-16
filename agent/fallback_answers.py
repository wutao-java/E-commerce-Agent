"""提供不依赖外部业务事实的安全兜底回答。"""

from domain import IntentResult


def build_fallback_answer(intent_result: IntentResult) -> str:
    """按粗意图提供固定话术，不推断尚未接入的订单或商品事实。"""

    if intent_result.intent == "complaint":
        return (
            "我已经先把这条消息识别为投诉类问题。"
            "当前版本还没有接入人工流转和赔偿处理，"
            "不能直接承诺处理结果。"
        )

    if intent_result.intent == "refund_request":
        return (
            "我已经先把这条消息识别为退款或售后类问题。"
            "退款、退货需要结合订单状态、商品类目和售后规则确认；"
            "当前还没有接入订单和售后工具，不能直接判断是否可退。"
        )

    if intent_result.intent == "order_query":
        return (
            "我已经先把这条消息识别为订单或物流查询。"
            "当前版本还没有接入订单工具，不能编造物流节点。"
        )

    if intent_result.intent == "promotion_consult":
        return (
            "我已经先把这条消息识别为优惠活动咨询。"
            "具体优惠需要以当前活动规则和结算页实时展示为准，"
            "不能口头承诺一定可以叠加。"
        )

    if intent_result.intent == "product_consult":
        return (
            "我已经先把这条消息识别为商品咨询。"
            "当前版本还没有接入产品知识库，不能编造商品卖点。"
        )

    if intent_result.intent == "general_chat":
        return (
            "你好，我是小哲电商客服 Agent。"
            "现在我已经能把用户问题先分到一个粗意图里。"
        )

    return "我还不能确定这条消息属于哪类客服问题，只能先标记为 unknown。"
