# app/service/core/graphrag/graph_cache.py
"""
知识图谱缓存模块 - 基于 Redis 缓存图谱元数据
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GraphCache:
    """知识图谱缓存管理器"""

    def __init__(self):
        from app.service.core.cache import get_cache_manager
        self.cache = get_cache_manager()
        self.cache_ttl = int(os.getenv("GRAPH_CACHE_TTL", "3600"))
        self.enabled = os.getenv("ENABLE_GRAPH_CACHE", "true").lower() == "true"

        logger.info(f"GraphCache 初始化: enabled={self.enabled}, ttl={self.cache_ttl}s")

    def _get_key(self, user_level: str = None) -> str:
        """生成缓存 key"""
        level = user_level or "default"
        return f"graph:{level}"

    def get(self, user_level: str = None) -> Optional[Dict]:
        """获取缓存的图谱"""
        if not self.enabled:
            return None

        cache_key = self._get_key(user_level)
        cached = self.cache.get("graph", cache_key)

        if cached:
            logger.debug(f"图谱缓存命中: {cache_key}")
        return cached

    def set(self, data: Dict, user_level: str = None):
        """缓存图谱"""
        if not self.enabled:
            return

        cache_key = self._get_key(user_level)
        # 添加缓存时间戳
        data["_cached_at"] = datetime.now().isoformat()
        data["_cache_ttl"] = self.cache_ttl

        self.cache.set("graph", cache_key, data, self.cache_ttl)
        logger.info(f"图谱已缓存: {cache_key}, TTL={self.cache_ttl}s")

    def invalidate(self, user_level: str = None):
        """使缓存失效"""
        if user_level:
            cache_key = self._get_key(user_level)
            self.cache.delete("graph", cache_key)
            logger.info(f"图谱缓存已失效: {user_level}")
        else:
            self.cache.delete_pattern("graph:*")
            logger.info("所有图谱缓存已失效")

    def is_valid(self, user_level: str = None) -> bool:
        """检查缓存是否有效"""
        cached = self.get(user_level)
        return cached is not None


# 全局实例
_graph_cache = None


def get_graph_cache() -> GraphCache:
    """获取图谱缓存实例"""
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = GraphCache()
    return _graph_cache