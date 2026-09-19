"""对外暴露 Prompt Registry 的加载、选择和渲染能力。"""

from .business_facts import render_business_fact_messages
from .loader import (
    load_prompt_registry,
    render_prompt_template,
    select_prompt_fragments,
)

__all__ = [
    "load_prompt_registry",
    "render_business_fact_messages",
    "render_prompt_template",
    "select_prompt_fragments",
]
