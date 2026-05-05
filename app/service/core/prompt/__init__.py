# app/service/core/prompt/__init__.py
"""
Prompt构造模块 - 增强与上下文构造
将召回的相关文档块与原始用户问题组合成结构化Prompt
"""

from .prompt_builder import PromptBuilder, get_prompt_builder
from .context_constructor import ContextConstructor
from .templates import PROMPT_TEMPLATES, get_template

__all__ = [
    'PromptBuilder',
    'get_prompt_builder',
    'ContextConstructor',
    'PROMPT_TEMPLATES',
    'get_template'
]