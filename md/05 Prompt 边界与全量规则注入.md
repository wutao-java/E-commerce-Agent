# 第 05 课实现讲义：Prompt 边界与全量规则注入

## 1. 本次结论

当前基线是 `lesson-04-intent-structured-output`。第 05 课不替换已有意图识别，也不接入 Java 业务接口；它在第 04 课流程之后增加四项能力：

1. 用 system prompt 固化客服身份、事实优先级和禁止承诺边界。
2. 把 5 份 current/legacy 规则文档全量注入模型上下文。
3. 用简单关键词规则记录“新旧规则可能冲突”的观察信号，但不自动裁决。
4. 在 `session_state` 中公开文档数、文档 ID、粗略 token 数和冲突线索。

本节仍然不做 RAG、引用、业务工具、退款审批、人工流转、Trace 或精确成本统计。

已从 `test` 创建分支：

```text
feat/05-prompt-boundary
```

创建分支后没有修改 `D:\E-commerce-agent` 或 `D:\E-commerce-springboot` 的源码。

## 2. 实现假设

- 继续使用当前项目的 `ChatCommand`、`ChatResult`、`IntentResult` 和 Web DTO 分层，不照搬课程快照的 `api/schemas.py`。
- 继续使用第 04 课的“规则优先、分类模型兜底”意图识别，不复制课程快照里更简单的 `classify_intent`。
- 继续保留当前线程安全的会话计数和模型失败安全回退。
- 规则文档只是本课的内存教学数据，不增加数据库表、配置中心或 Java 接口。
- `estimate_tokens()` 只用于观察趋势，不能用于账单或模型上下文硬限制。
- 冲突检测只要命中一个关键词就记录线索，可能误报；这是本课故意保留的局限，不在本节提前实现规则裁决器。

## 3. 方案取舍

### 方案 A：直接覆盖课程快照，不采用

课程快照依赖 `course_runtime`，目录也是 `backend/api`、`backend/models`。直接覆盖会丢失当前工程已有的领域模型、分类模型兜底、线程安全计数、统一配置和异常映射。

### 方案 B：把所有 Prompt 逻辑写进 Agent，不采用

文件更少，但规则数据、Prompt 拼装、模型调用和流程编排会混在一起，第 06 课继续管理 Prompt 时会很难拆。

### 方案 C：按当前架构新增 `prompts` 模块，采用

领域模型放 `domain`，规则和 Prompt 拼装放 `prompts`，模型调用留在 `llm`，Agent 只做编排。它既保持课程语义，也只修改直接相关代码。

## 4. 实现顺序

先完成步骤 1 至 7 的核心代码，再执行步骤 8 的测试，不需要先写测试。

### 步骤 1：新增 Prompt 领域模型

新建 `D:\E-commerce-agent\domain\prompt.py`：

```python
"""定义 Prompt 规则文档和上下文冲突模型。"""

from typing import Literal

from pydantic import BaseModel


class PolicyDocument(BaseModel):
    """表示本课注入模型上下文的一份规则文档。"""

    doc_id: str
    title: str
    status: Literal["current", "legacy", "draft"]
    keywords: list[str]
    body: str


class ContextConflict(BaseModel):
    """表示全量上下文中可观察到的一条冲突线索。"""

    topic: str
    newer_doc_id: str
    older_doc_id: str
    reason: str
```

然后用下面内容完整替换 `D:\E-commerce-agent\domain\__init__.py`：

```python
"""对外暴露 Agent 内部业务契约。"""

from .chat import ChatCommand, ChatResult
from .intent import Intent, IntentResult, IntentSource
from .prompt import ContextConflict, PolicyDocument

__all__ = [
    "ChatCommand",
    "ChatResult",
    "ContextConflict",
    "Intent",
    "IntentResult",
    "IntentSource",
    "PolicyDocument",
]
```

完成标志：下面命令没有报错。

```powershell
.\.venv\Scripts\python.exe -c "from domain import ContextConflict, PolicyDocument; print('domain ok')"
```

### 步骤 2：新增 Prompt 加载与组装模块

新建目录 `D:\E-commerce-agent\prompts`，再新建 `prompts\loader.py`：

```python
"""集中管理课程规则文档、Prompt 组装和上下文统计。"""

from __future__ import annotations

from domain import ChatCommand, ContextConflict, IntentResult, PolicyDocument


FULL_POLICY_DOCUMENTS: list[PolicyDocument] = [
    PolicyDocument(
        doc_id="promo-2026-audio-current",
        title="2026 春季音频节当前活动规则",
        status="current",
        keywords=["降噪耳机", "会员价", "会员券", "满减", "叠加"],
        body=(
            "当前规则：小哲电商春季音频节中，降噪耳机会员价"
            "不可再叠加会员券、满减券或店铺券。用户最终可用优惠"
            "以结算页实时展示为准，客服不得口头承诺一定可叠加。"
        ),
    ),
    PolicyDocument(
        doc_id="after-sale-2026-current",
        title="2026 售后当前口径",
        status="current",
        keywords=["退款", "退货", "质量问题", "售后", "订单状态"],
        body=(
            "当前口径：退款、退货需要结合订单状态、商品类目和售后"
            "规则确认。客服 Agent 在没有订单和售后事实时，只能说明"
            "需要核实，不能承诺马上退款或赔偿到账。"
        ),
    ),
    PolicyDocument(
        doc_id="promo-2024-double11-legacy",
        title="2024 双11耳机活动复盘旧规则",
        status="legacy",
        keywords=["降噪耳机", "会员价", "会员券", "满减", "叠加"],
        body=(
            "历史复盘：2024 双11期间，部分降噪耳机曾允许金卡会员价"
            "与一张会员券叠加。该文档只用于运营复盘，不代表当前活动口径。"
        ),
    ),
    PolicyDocument(
        doc_id="after-sale-2023-legacy",
        title="2023 旧版耳机售后口径",
        status="legacy",
        keywords=["退款", "退货", "耳机", "七天", "质量问题"],
        body=(
            "历史口径：旧版耳机售后曾写过七天内可直接退货。"
            "该口径已经被新版售后规则替换，不能单独作为当前处理依据。"
        ),
    ),
    PolicyDocument(
        doc_id="service-boundary",
        title="小哲电商客服 Agent 统一边界",
        status="current",
        keywords=["边界", "承诺", "赔偿", "物流", "人工"],
        body=(
            "客服 Agent 必须以小哲电商系统确认的事实为准。涉及优惠、"
            "退款、赔偿、发货、签收和人工处理结果时，不得在没有系统"
            "依据时直接承诺。"
        ),
    ),
]

CONFLICT_RULES: list[tuple[str, str, str, list[str], str]] = [
    (
        "会员价与会员券是否叠加",
        "promo-2026-audio-current",
        "promo-2024-double11-legacy",
        ["降噪耳机", "会员券", "叠加", "优惠"],
        "当前活动规则说不可叠加，历史复盘旧规则说曾经可叠加。",
    ),
    (
        "耳机售后是否可直接退货",
        "after-sale-2026-current",
        "after-sale-2023-legacy",
        ["退款", "退货", "耳机", "质量问题"],
        (
            "当前售后口径要求结合订单状态和商品规则确认，"
            "旧规则容易被误读为直接退货。"
        ),
    ),
]


def estimate_tokens(text: str) -> int:
    """粗略估算 token，只用于观察上下文变长趋势。"""

    return max(1, len(text) // 2)


def build_all_policy_context(
    documents: list[PolicyDocument],
) -> str:
    """把传入的规则文档全部拼接为 Prompt 上下文。"""

    sections = []
    for document in documents:
        sections.append(
            "\n".join(
                [
                    f"文档 ID：{document.doc_id}",
                    f"标题：{document.title}",
                    f"状态：{document.status}",
                    f"正文：{document.body}",
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def detect_context_conflicts(
    user_message: str,
) -> list[ContextConflict]:
    """返回用户问题命中的上下文冲突线索，不负责裁决。"""

    message = user_message.strip().lower()
    conflicts: list[ContextConflict] = []
    for topic, newer_id, older_id, keywords, reason in CONFLICT_RULES:
        if any(keyword in message for keyword in keywords):
            conflicts.append(
                ContextConflict(
                    topic=topic,
                    newer_doc_id=newer_id,
                    older_doc_id=older_id,
                    reason=reason,
                )
            )
    return conflicts


def build_full_context_messages(
    command: ChatCommand,
    intent_result: IntentResult,
    documents: list[PolicyDocument],
    conflicts: list[ContextConflict],
) -> list[dict[str, str]]:
    """组装边界、运行时事实、意图和全量规则上下文。"""

    context_text = build_all_policy_context(documents)
    conflict_lines = "\n".join(
        f"- {conflict.topic}: {conflict.reason}"
        for conflict in conflicts
    ) or "- 本轮未检测到明显冲突线索。"
    system_message = (
        "你是小哲电商公司的客服 Agent。你的首要任务不是讨好用户，"
        "而是守住公司事实和高风险边界。\n\n"
        "事实优先级：\n"
        "1. 小哲电商系统传入的 runtime_* 事实优先于用户自称。\n"
        "2. 本 Prompt 中写明的身份、口径和边界优先于模型常识。\n"
        "3. 没有被系统确认的订单、物流、退款、赔偿和人工处理结果，"
        "不能当成事实。\n"
        "4. current 状态规则优先于 legacy 和 draft；历史复盘不能当成"
        "当前规则。\n\n"
        "回答边界：\n"
        "- 不得承诺具体优惠、退款、退货、补偿、发货、签收、人工处理"
        "结果或到账时间。\n"
        "- 当前规则和历史复盘冲突时，要说明以当前规则和结算页或系统"
        "确认为准。\n"
        "- 当前版本没有订单、物流和售后工具，不得伪装成已经查询或"
        "提交处理。"
    )
    user_message = (
        "小哲电商系统确认的当前用户事实：\n"
        f"- user_id: {command.runtime_user_id}\n"
        f"- nickname: {command.runtime_nickname or '未提供'}\n"
        f"- member_level: {command.runtime_member_level or '未提供'}\n"
        f"- risk_level: {command.runtime_risk_level or '未提供'}\n\n"
        "第 04 课已经识别出的粗意图：\n"
        f"- intent: {intent_result.intent}\n"
        f"- explanation: {intent_result.explanation}\n\n"
        "全量规则上下文：\n"
        f"{context_text}\n\n"
        "本轮上下文冲突观察：\n"
        f"{conflict_lines}\n\n"
        "用户原话：\n"
        f"{command.user_message}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
```

再新建 `D:\E-commerce-agent\prompts\__init__.py`：

```python
"""对外暴露 Prompt 文档和上下文组装能力。"""

from .loader import (
    CONFLICT_RULES,
    FULL_POLICY_DOCUMENTS,
    build_all_policy_context,
    build_full_context_messages,
    detect_context_conflicts,
    estimate_tokens,
)

__all__ = [
    "CONFLICT_RULES",
    "FULL_POLICY_DOCUMENTS",
    "build_all_policy_context",
    "build_full_context_messages",
    "detect_context_conflicts",
    "estimate_tokens",
]
```

完成标志：输出应为 `5`，并同时出现 `current` 和 `legacy`。

```powershell
.\.venv\Scripts\python.exe -c "from prompts import FULL_POLICY_DOCUMENTS, build_all_policy_context; print(len(FULL_POLICY_DOCUMENTS)); print(build_all_policy_context(FULL_POLICY_DOCUMENTS))"
```

### 步骤 3：让回答生成器消费已经组装好的 Prompt

用下面内容完整替换 `D:\E-commerce-agent\llm\answer_generator.py`：

```python
"""封装基于完整 Prompt 上下文的客服回答生成。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .client import call_chat_model, extract_assistant_message

AnswerModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]


class GroundedAnswerResult(BaseModel):
    """记录最终回答是否由真实模型生成。"""

    answer: str
    used_model: bool = False
    fallback_reason: str | None = None


def compose_grounded_answer(
    *,
    messages: list[dict[str, str]],
    deterministic_answer: str,
    model_call: AnswerModelCall = call_chat_model,
) -> GroundedAnswerResult:
    """用完整 Prompt 调用模型，失败时返回安全话术。"""

    try:
        model_response = model_call(messages)
        answer = extract_assistant_message(model_response)
    except RuntimeError:
        return GroundedAnswerResult(
            answer=deterministic_answer,
            fallback_reason="model_unavailable",
        )

    return GroundedAnswerResult(answer=answer, used_model=True)
```

这里不改 `llm\client.py`，因为当前通用客户端已经支持任意 `messages`，无需复制课程快照里的模型客户端。

### 步骤 4：更新模型不可用时的安全兜底

用下面内容完整替换 `D:\E-commerce-agent\agent\fallback_answers.py`：

```python
"""提供不依赖外部业务事实的安全兜底回答。"""

from domain import IntentResult


def build_fallback_answer(intent_result: IntentResult) -> str:
    """根据粗意图生成不越过业务边界的确定性回答。"""

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
            "我可以先识别问题类型，并按当前客服边界提供说明。"
        )

    return "我还不能确定这条消息属于哪类客服问题，只能先标记为 unknown。"
```

这一步必要，因为第 04 课的活动兜底仍写着“没有接入活动规则”，会与本课已经注入静态规则的事实冲突。

### 步骤 5：更新 Agent 编排

用下面内容完整替换 `D:\E-commerce-agent\agent\customer_service_agent.py`：

```python
"""编排客服请求、意图识别、Prompt 上下文和回答生成。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from domain import ChatCommand, ChatResult
from llm import call_chat_model, classify_intent_with_model, compose_grounded_answer
from prompts import (
    FULL_POLICY_DOCUMENTS,
    build_full_context_messages,
    detect_context_conflicts,
    estimate_tokens,
)

from .fallback_answers import build_fallback_answer
from .intent_service import ClassifierCall, classify_intent

ModelCall = Callable[[list[dict[str, str]]], dict[str, Any]]


class CustomerServiceAgent:
    """协调客服能力并返回与传输协议无关的处理结果。"""

    def __init__(
        self,
        model_call: ModelCall = call_chat_model,
        classifier_call: ClassifierCall = classify_intent_with_model,
    ) -> None:
        self._model_call = model_call
        self._classifier_call = classifier_call
        self._message_count_by_session: dict[str, int] = {}
        self._session_lock = Lock()

    def chat(self, command: ChatCommand) -> ChatResult:
        """处理一次带 Prompt 边界和全量规则上下文的聊天。"""

        message_count = self._increment_message_count(command.session_id)
        intent_result = classify_intent(
            command.user_message,
            classifier_call=self._classifier_call,
        )
        conflicts = detect_context_conflicts(command.user_message)
        messages = build_full_context_messages(
            command,
            intent_result,
            FULL_POLICY_DOCUMENTS,
            conflicts,
        )
        prompt_text = "\n".join(
            message["content"] for message in messages
        )
        fallback_answer = build_fallback_answer(intent_result)
        model_answer = compose_grounded_answer(
            messages=messages,
            deterministic_answer=fallback_answer,
            model_call=self._model_call,
        )

        return ChatResult(
            session_id=command.session_id,
            answer=model_answer.answer,
            intent=intent_result.intent,
            intent_result=intent_result,
            reasoning_summary=[
                "后端沿用第 04 课的结构化粗意图识别。",
                "system prompt 写明客服身份、事实优先级和回答边界。",
                (
                    f"本轮把 {len(FULL_POLICY_DOCUMENTS)} 份规则文档"
                    "全量注入 Prompt。"
                ),
                (
                    f"本轮检测到 {len(conflicts)} 条上下文冲突线索；"
                    "这些线索只用于观察，不会自动裁决规则。"
                ),
            ],
            session_state={
                "agent_version": "lesson-05-prompt-boundary-full-context",
                "message_count": message_count,
                "runtime_context": {
                    "user_id": command.runtime_user_id,
                    "nickname": command.runtime_nickname,
                    "member_level": command.runtime_member_level,
                    "risk_level": command.runtime_risk_level,
                    "page_context": dict(command.runtime_context or {}),
                },
                "model_answer": model_answer.model_dump(),
                "prompt_boundary": {
                    "mode": "system_prompt_fact_priority_and_refusal_rules",
                    "fact_priority": [
                        "runtime_facts",
                        "current_policy_documents",
                        "legacy_documents",
                        "user_claims",
                        "model_general_knowledge",
                    ],
                    "boundary_rule_count": 4,
                },
                "prompt_context": {
                    "mode": "full_document_injection",
                    "document_count": len(FULL_POLICY_DOCUMENTS),
                    "document_ids": [
                        document.doc_id
                        for document in FULL_POLICY_DOCUMENTS
                    ],
                    "estimated_prompt_tokens": estimate_tokens(prompt_text),
                    "conflict_count": len(conflicts),
                    "conflicts": [
                        conflict.model_dump()
                        for conflict in conflicts
                    ],
                },
                "next_gap": (
                    "system prompt 能先限制乱承诺，全量 Prompt 能让规则进入"
                    "模型，但当前规则和历史规则仍会相互干扰。"
                ),
            },
        )

    def _increment_message_count(self, session_id: str) -> int:
        """安全地递增单进程会话消息计数。"""

        with self._session_lock:
            message_count = self._message_count_by_session.get(session_id, 0) + 1
            self._message_count_by_session[session_id] = message_count
            return message_count
```

这里不修改 `web\routers\chat.py`、`web\schema\chat_schema.py`、`domain\chat.py`：本节新增的观察字段都位于既有 `session_state` 中，HTTP 契约没有新增顶层字段。

### 步骤 6：更新能力声明

用下面内容完整替换 `D:\E-commerce-agent\agent_capabilities.json`：

```json
{
  "schema_version": "agent_capabilities_v1",
  "lesson": {
    "id": "lesson-05-prompt-boundary",
    "number": 5,
    "title": "Prompt 边界与长上下文冲突",
    "summary": "当前版本用 system prompt 管住客服身份、事实优先级和回答边界，并把当前与历史规则全量注入 Prompt。"
  },
  "agent": {
    "name": "小哲电商客服 Agent",
    "version": "lesson-05-prompt-boundary-full-context"
  },
  "endpoints": {
    "health": true,
    "chat": true,
    "chat_resume": false,
    "trace": false,
    "eval_run": false
  },
  "features": {
    "chat": true,
    "runtime_context": true,
    "reasoning_summary": true,
    "reasoning_content": false,
    "real_llm_answer": true,
    "grounded_model_answer": true,
    "structured_intent": true,
    "rag_citations": false,
    "tool_calls": false,
    "workflow": false,
    "human_approval": false,
    "memory": false,
    "hooks": false,
    "trace": false,
    "evaluation": false,
    "cost_summary": false
  },
  "disabled_reasons": {
    "reasoning_content": "当前版本只返回公开执行摘要，不公开模型隐藏推理。",
    "rag_citations": "当前版本是全量 Prompt 注入，尚未接入 RAG 和引用。",
    "tool_calls": "当前版本尚未接入业务工具。",
    "workflow": "当前版本尚未接入售后工作流。",
    "human_approval": "当前版本尚未接入人工审批。",
    "memory": "当前会话计数不是生产级会话记忆。",
    "hooks": "当前版本尚未接入 Hooks。",
    "trace": "当前版本尚未接入调用轨迹。",
    "evaluation": "当前版本尚未接入回归评测。",
    "cost_summary": "当前只粗略估算 Prompt token，尚未形成成本统计。"
  }
}
```

不要删除当前工程已有的 `real_llm_answer` 和 `grounded_model_answer`，课程快照没有这两个字段只是因为其目录是独立精简版本。

### 步骤 7：更新 README

只做以下定点更新，不重写无关的 MySQL、JWT、日志和启动说明：

1. 第一段将“结构化意图识别和模型回答”改为“结构化意图识别、Prompt 边界和全量规则上下文”。
2. “当前能力”增加：

```markdown
- 支持 system prompt 事实优先级和高风险回答边界
- 支持 current/legacy 规则文档全量注入和冲突线索观察
- 在 `session_state.prompt_context` 中公开文档数、粗略 token 数和冲突线索
```

3. 项目结构中增加：

```text
├── domain/
│   └── prompt.py           # 规则文档与上下文冲突领域模型
├── prompts/
│   └── loader.py           # 全量规则、冲突检测和 Prompt 组装
```

4. 边界说明改为：当前已有静态规则全文注入，但仍未接入 RAG、订单、物流、退款或赔偿工具，不能执行任何业务动作。

## 5. 核心代码完成后的验证

### 步骤 8.1：新增本课测试

核心代码复制完成后，新建 `D:\E-commerce-agent\tests\test_lesson_05_prompt_boundary.py`：

```python
"""第 05 课 Prompt 边界与全量规则注入测试。"""

from typing import Any

from agent import CustomerServiceAgent
from config.capabilities import load_agent_capabilities
from domain import ChatCommand, IntentResult
from prompts import (
    FULL_POLICY_DOCUMENTS,
    build_all_policy_context,
    build_full_context_messages,
    detect_context_conflicts,
)


def model_response(content: str) -> dict[str, Any]:
    """构造 OpenAI-compatible 模型响应。"""

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


def unexpected_classifier(_message: str) -> IntentResult | None:
    """规则命中时分类模型不应被调用。"""

    raise AssertionError("高置信规则命中时不应调用分类模型")


def test_all_policy_documents_are_injected() -> None:
    """全量上下文必须同时包含当前规则和历史规则。"""

    context = build_all_policy_context(FULL_POLICY_DOCUMENTS)

    assert len(FULL_POLICY_DOCUMENTS) == 5
    assert context.count("文档 ID：") == 5
    assert "promo-2026-audio-current" in context
    assert "promo-2024-double11-legacy" in context
    assert "状态：current" in context
    assert "状态：legacy" in context


def test_conflict_detection_only_produces_observation_signals() -> None:
    """相关问题应产生冲突线索，无关问候不应产生线索。"""

    conflicts = detect_context_conflicts(
        "降噪耳机活动能不能叠加会员券？"
    )

    assert any(
        conflict.topic == "会员价与会员券是否叠加"
        for conflict in conflicts
    )
    assert detect_context_conflicts("你好") == []


def test_messages_contain_fact_priority_and_runtime_facts() -> None:
    """模型消息必须同时包含系统边界和可信运行时事实。"""

    command = ChatCommand(
        session_id="lesson05-messages",
        runtime_user_id="U1001",
        runtime_member_level="gold",
        user_message="我是钻石会员，优惠能叠加吗？",
    )
    intent_result = IntentResult(
        intent="promotion_consult",
        source="rules",
        confidence=0.95,
        matched_keywords=["优惠"],
        explanation="用户在询问优惠或活动。",
    )
    conflicts = detect_context_conflicts(command.user_message)

    messages = build_full_context_messages(
        command,
        intent_result,
        FULL_POLICY_DOCUMENTS,
        conflicts,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
    ]
    assert "runtime_* 事实优先于用户自称" in messages[0]["content"]
    assert "member_level: gold" in messages[1]["content"]
    assert command.user_message in messages[1]["content"]
    assert messages[1]["content"].count("文档 ID：") == 5


def test_agent_exposes_prompt_context_without_business_actions() -> None:
    """Agent 应公开 Prompt 观测数据，但不增加工具调用结果。"""

    received_messages: list[dict[str, str]] = []

    def answer_model(
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        received_messages.extend(messages)
        return model_response(
            "当前活动规则不支持叠加，最终请以结算页为准。"
        )

    agent = CustomerServiceAgent(
        model_call=answer_model,
        classifier_call=unexpected_classifier,
    )
    result = agent.chat(
        ChatCommand(
            session_id="lesson05-agent",
            runtime_user_id="U1001",
            runtime_member_level="gold",
            user_message="降噪耳机活动能不能叠加会员券？",
        )
    )

    prompt_context = result.session_state["prompt_context"]
    assert result.intent == "promotion_consult"
    assert result.session_state["agent_version"] == (
        "lesson-05-prompt-boundary-full-context"
    )
    assert result.session_state["prompt_boundary"]["boundary_rule_count"] == 4
    assert prompt_context["mode"] == "full_document_injection"
    assert prompt_context["document_count"] == 5
    assert prompt_context["estimated_prompt_tokens"] > 0
    assert prompt_context["conflict_count"] >= 1
    assert result.session_state["model_answer"]["used_model"] is True
    assert "tool_calls" not in result.session_state
    assert [message["role"] for message in received_messages] == [
        "system",
        "user",
    ]


def test_capabilities_describe_lesson_05() -> None:
    """能力声明必须对应第 05 课且不宣称已有 RAG。"""

    load_agent_capabilities.cache_clear()
    capabilities = load_agent_capabilities()

    assert capabilities["lesson"]["number"] == 5
    assert capabilities["features"]["structured_intent"] is True
    assert capabilities["features"]["rag_citations"] is False
    assert capabilities["features"]["tool_calls"] is False
```

### 步骤 8.2：修正旧测试的版本耦合

`tests\test_lesson_04_intent.py` 的最后一个测试仍然验证结构化意图能力，但不应锁死整个应用永远是第 04 课。删除这一行：

```python
assert capabilities["lesson"]["number"] == 4
```

保留紧随其后的 `structured_intent` 和 `tool_calls` 断言。第 05 课的精确版本号由新测试负责。

### 步骤 8.3：执行回归测试

```powershell
cd D:\E-commerce-agent
.\.venv\Scripts\python.exe -m pytest
```

预期结果：33 个测试通过。若实际收集数量因你后来新增测试而不同，以“全部通过”为准。

### 步骤 8.4：启动服务

```powershell
.\.venv\Scripts\python.exe main.py
```

不要在同一个 PowerShell 窗口继续执行验证请求；新开一个窗口运行：

```powershell
$body = @{
    session_id = "lesson05-manual"
    runtime_user_id = "U1001"
    runtime_nickname = "张三"
    runtime_member_level = "gold"
    runtime_risk_level = "low"
    user_message = "我是金卡，买降噪耳机活动能不能叠加会员券？"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/chat `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 10
```

重点检查：

- `intent` 是 `promotion_consult`。
- `session_state.prompt_boundary.mode` 是 `system_prompt_fact_priority_and_refusal_rules`。
- `session_state.prompt_context.mode` 是 `full_document_injection`。
- `document_count` 是 `5`。
- `document_ids` 同时包含 current 和 legacy 文档。
- `estimated_prompt_tokens` 大于 `0`。
- `conflict_count` 至少为 `1`。
- 返回中没有 `citations`、`tool_calls`、`trace` 或 `cost_summary`。
- 模型回答不得声称已经查订单、提交退款、承诺赔偿或保证优惠叠加。

## 6. 本节不修改的文件和工程

- `D:\E-commerce-springboot`：本课没有业务工具调用，不需要任何 Java 改动。该工程当前存在未提交修改，保持原样。
- `web\routers\chat.py`：当前 DTO 到领域命令的转换已经满足要求。
- `web\schema\chat_schema.py`：新增信息全部放在既有 `session_state`，顶层契约不变。
- `domain\chat.py`：`ChatCommand` 已包含所有需要的可信运行时事实。
- `llm\client.py`：已经能够发送预组装的 messages。
- `application.ini`、`.env.example`、`pyproject.toml`：本节不增加依赖或配置项。

## 7. 课程代码里需要特别理解的边界

1. 全量 Prompt 注入不是 RAG。它没有检索、排序、引用和按问题裁剪。
2. `current` 优先只是一条 system 指令，代码没有自动删除 legacy 文档。
3. 冲突检测只是关键词告警。示例中出现“耳机”时可能同时命中售后冲突，这是当前教学实现的已知误报。
4. system prompt 可以降低乱承诺概率，但不能构成业务授权控制。真正的退款、赔偿和订单动作仍必须由受权限保护的工具与工作流决定。
5. 粗略 token 估算不能用于计费。不同模型 tokenizer 的结果会不同。
