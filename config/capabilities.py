"""读取 Agent 能力声明。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CAPABILITIES_PATH = (Path(__file__).resolve().parents[1]/ "agent_capabilities.json")


@lru_cache(maxsize=1)
def load_agent_capabilities() -> dict[str, Any]:
    """读取并缓存 Agent 能力声明。"""

    with CAPABILITIES_PATH.open("r",encoding="utf-8",) as file:
        capabilities = json.load(file)

    if not isinstance(capabilities, dict):
        raise RuntimeError(
            "agent_capabilities.json 必须是 JSON 对象。"
        )

    return capabilities