"""对外暴露 Agent 内部业务契约。"""

from .chat import ChatCommand, ChatResult
from .intent import Intent, IntentResult, IntentSource

__all__ = [
    "ChatCommand",
    "ChatResult",
    "Intent",
    "IntentResult",
    "IntentSource",
]
