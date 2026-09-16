"""编排规则优先、分类模型兜底的意图识别流程。"""

from collections.abc import Callable

from domain import IntentResult
from llm import classify_intent_with_model

from .intent_rules import plan_intent_by_rules

ClassifierCall = Callable[[str], IntentResult | None]


def classify_intent(
    user_message: str,
    classifier_call: ClassifierCall = classify_intent_with_model,
) -> IntentResult:
    """识别客服消息意图，无法可靠判断时返回 unknown。"""

    rule_result = plan_intent_by_rules(user_message)
    if rule_result is not None:
        return rule_result

    model_result = classifier_call(user_message)
    if model_result is not None:
        return model_result

    return IntentResult(
        intent="unknown",
        source="rules_fallback",
        confidence=0.3,
        matched_keywords=[],
        explanation=(
            "规则没有高置信命中，分类模型也不可用或输出无效，"
            "先标记为 unknown。"
        ),
    )
