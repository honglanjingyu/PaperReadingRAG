# app/service/core/memory/__init__.py
"""
记忆模块 - 支持Redis持久化存储
"""

from .redis_session_memory import RedisSessionMemory, get_memory_manager, MemoryEntry
from .conversation_history import ConversationHistory
from .memory_injector import MemoryInjector

# 为了向后兼容，将 RedisSessionMemory 也导出为 SessionMemory
SessionMemory = RedisSessionMemory

__all__ = [
    'RedisSessionMemory',
    'SessionMemory',  # 向后兼容别名
    'get_memory_manager',
    'MemoryEntry',
    'ConversationHistory',
    'MemoryInjector'
]