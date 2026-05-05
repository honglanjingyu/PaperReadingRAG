# app/service/core/llm/__init__.py
"""
LLM 生成模块 - 将 Prompt 发送给大语言模型生成答案
"""

from .base_llm import BaseLLM
from .remote_llm import RemoteLLM
from .llm_manager import LLMManager, LLMType, get_llm_manager
from .llm_service import LLMService, get_llm_service

__all__ = [
    'BaseLLM',
    'RemoteLLM',
    'LLMManager',
    'LLMType',
    'get_llm_manager',
    'LLMService',
    'get_llm_service',
]