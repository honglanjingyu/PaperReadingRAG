# app/service/core/graphrag/graph_cache.py
"""
知识图谱缓存模块 - 只使用 Redis 缓存（无内存缓存）
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphCache:
    """知识图谱缓存管理器 - 只使用 Redis 缓存"""

    def __init__(self):
        from app.service.core.cache import get_cache_manager
        self.cache = get_cache_manager()
        self.cache_ttl = int(os.getenv("GRAPH_CACHE_TTL", "3600"))
        self.enabled = os.getenv("ENABLE_GRAPH_CACHE", "true").lower() == "true"

        # 检查 Redis 是否可用
        self._redis_available = self.cache.redis_client is not None

        if not self._redis_available:
            logger.warning("Redis 不可用，图谱缓存将禁用。请确保 Redis 服务已启动。")

        logger.info(f"GraphCache 初始化: enabled={self.enabled and self._redis_available}, ttl={self.cache_ttl}s")

    def _get_key(self, user_level: str = None) -> str:
        """生成缓存 key"""
        level = user_level or "default"
        return f"graph:{level}"

    def get(self, user_level: str = None) -> Optional[Dict]:
        """
        获取缓存的图谱（只从 Redis 读取）

        Args:
            user_level: 用户等级

        Returns:
            缓存的图谱数据，如果没有则返回 None
        """
        if not self.enabled or not self._redis_available:
            return None

        cache_key = self._get_key(user_level)
        cached = self.cache.get("graph", cache_key)

        if cached:
            logger.debug(f"图谱缓存命中 (Redis): {cache_key}")
            return cached
        else:
            logger.debug(f"图谱缓存未命中: {cache_key}")
            return None

    def set(self, data: Dict, user_level: str = None):
        """
        缓存图谱（只存入 Redis）

        Args:
            data: 图谱数据
            user_level: 用户等级
        """
        if not self.enabled or not self._redis_available:
            return

        cache_key = self._get_key(user_level)

        # 添加缓存时间戳（但不用于内存缓存，仅用于记录）
        data["_cached_at"] = datetime.now().isoformat()
        data["_cache_ttl"] = self.cache_ttl
        data["_cache_backend"] = "redis_only"

        self.cache.set("graph", cache_key, data, self.cache_ttl)
        logger.info(f"图谱已缓存到 Redis: {cache_key}, TTL={self.cache_ttl}s")

    def invalidate(self, user_level: str = None):
        """
        使缓存失效

        Args:
            user_level: 用户等级，如果为 None 则清除所有
        """
        if not self.enabled or not self._redis_available:
            return

        if user_level:
            cache_key = self._get_key(user_level)
            self.cache.delete("graph", cache_key)
            logger.info(f"图谱缓存已失效 (Redis): {user_level}")
        else:
            self.cache.delete_pattern("graph:*")
            logger.info("所有图谱缓存已失效 (Redis)")

    def is_valid(self, user_level: str = None) -> bool:
        """检查缓存是否有效"""
        if not self.enabled or not self._redis_available:
            return False

        cached = self.get(user_level)
        return cached is not None

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "enabled": self.enabled,
            "redis_available": self._redis_available,
            "cache_ttl": self.cache_ttl,
            "cache_backend": "redis_only",
            "has_redis": self._redis_available
        }

    def warmup(self, user_level: str = None) -> bool:
        """
        预热缓存（检查缓存是否存在）

        Args:
            user_level: 用户等级

        Returns:
            缓存是否存在
        """
        return self.get(user_level) is not None


# 全局实例
_graph_cache = None


def get_graph_cache() -> GraphCache:
    """获取图谱缓存实例"""
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = GraphCache()
    return _graph_cache


__all__ = ['GraphCache', 'get_graph_cache']