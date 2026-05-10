# app/service/core/cache/__init__.py
"""
缓存模块 - 提供 Redis 缓存支持
"""

from .cache_manager import CacheManager, get_cache_manager, cached
from .document_cache import (
    DocumentCache,
    get_document_cache,
    cache_document_level,
    get_cached_document_level,
    invalidate_document_cache
)

__all__ = [
    'CacheManager',
    'get_cache_manager',
    'cached',
    'DocumentCache',
    'get_document_cache',
    'cache_document_level',
    'get_cached_document_level',
    'invalidate_document_cache'
]