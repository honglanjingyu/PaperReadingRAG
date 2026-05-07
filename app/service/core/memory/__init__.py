# 修改后
"""
记忆模块 - 支持 Redis 持久化存储
"""

# 默认使用 Redis
from .redis_session_memory import RedisSessionMemory, get_memory_manager
SessionMemory = RedisSessionMemory

from .conversation_history import ConversationHistory
from .memory_injector import MemoryInjector

__all__ = [
    'SessionMemory',
    'get_memory_manager',
    'MemoryEntry',
    'ConversationHistory',
    'MemoryInjector'
]