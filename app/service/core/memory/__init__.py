# app/service/core/memory/__init__.py
"""
记忆模块 - 支持 Redis 持久化存储
"""

# 只使用 Redis 实现
from .redis_session_memory import RedisSessionMemory, get_memory_manager, MemoryEntry
from .conversation_history import ConversationHistory
from .memory_injector import MemoryInjector

__all__ = [
    'RedisSessionMemory',
    'get_memory_manager',
    'MemoryEntry',
    'ConversationHistory',
    'MemoryInjector'
]