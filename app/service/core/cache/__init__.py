# app/service/core/cache/__init__.py
"""
缓存模块 - 提供 Redis 缓存支持
"""

from .cache_manager import CacheManager, get_cache_manager, cached

__all__ = ['CacheManager', 'get_cache_manager', 'cached']