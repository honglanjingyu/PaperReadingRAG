# app/service/core/rag/cached_search.py
"""搜索缓存 - 纯 Redis 实现"""

import json
import hashlib
import logging
from typing import Dict, Any, Optional

from app.service.core.cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedSearchService:
    """带缓存的搜索服务"""

    def __init__(self, cache_ttl: int = 300):
        """
        初始化缓存搜索服务

        Args:
            cache_ttl: 缓存过期时间（秒），默认300秒（5分钟）
        """
        self.cache = get_cache_manager()
        self.cache_ttl = cache_ttl
        logger.debug(f"CachedSearchService 初始化完成，cache_ttl={cache_ttl}s")

    def _get_cache_key(self, params: Dict) -> str:
        """
        生成缓存 key

        Args:
            params: 搜索参数字典

        Returns:
            缓存 key 的 MD5 哈希值（16位）
        """
        params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    def search_with_cache(self, search_func, **kwargs) -> Dict[str, Any]:
        """
        带缓存的搜索

        Args:
            search_func: 搜索函数
            **kwargs: 搜索参数（支持 use_cache 和 verbose）

        Returns:
            搜索结果字典
        """
        # 提取 verbose 参数，不传递给缓存逻辑
        verbose = kwargs.pop('verbose', False)
        use_cache = kwargs.pop('use_cache', True)

        if not use_cache:
            if verbose:
                logger.debug("缓存已禁用，直接执行搜索")
            return search_func(**kwargs)

        cache_key = self._get_cache_key(kwargs)

        # 查缓存
        cached = self.cache.get("search", cache_key)
        if cached is not None:
            if verbose:
                logger.debug(f"搜索缓存命中: key={cache_key[:8]}...")
            return cached

        if verbose:
            logger.debug(f"搜索缓存未命中: key={cache_key[:8]}...")

        # 执行搜索
        result = search_func(**kwargs)

        # 写缓存
        if result and result.get("success"):
            self.cache.set("search", cache_key, result, self.cache_ttl)
            if verbose:
                logger.debug(f"搜索结果已缓存: key={cache_key[:8]}..., ttl={self.cache_ttl}s")

        return result

    def invalidate_cache(self, pattern: str = None):
        """
        使搜索缓存失效

        Args:
            pattern: 缓存模式（可选），如果提供则删除匹配的缓存，否则删除所有搜索缓存
            
        Examples:
            >>> # 使所有搜索缓存失效
            >>> cache_service.invalidate_cache()
            
            >>> # 使包含特定文件名的缓存失效
            >>> cache_service.invalidate_cache(pattern="*test.pdf*")
            
            >>> # 使特定查询的缓存失效
            >>> cache_service.invalidate_cache(pattern="search:*")
        """
        if pattern:
            # 删除匹配模式的缓存
            # 注意：delete_pattern 会自动添加前缀，这里只需要传入模式后缀
            self.cache.delete_pattern(f"search:{pattern}")
            logger.info(f"搜索缓存已失效 (pattern={pattern})")
        else:
            # 删除所有搜索缓存
            self.cache.delete_pattern("search")
            logger.info("搜索缓存已失效 (all)")

    def invalidate(self, pattern: str = None):
        """
        使缓存失效（别名方法，兼容性）

        Args:
            pattern: 缓存模式（可选）
        """
        self.invalidate_cache(pattern)

    def invalidate_by_filename(self, filename: str):
        """
        使指定文件名的相关缓存失效

        Args:
            filename: 文件名
        """
        if not filename:
            return

        # 删除包含该文件名的搜索缓存
        self.cache.delete_pattern(f"search:*{filename}*")
        logger.info(f"搜索缓存已失效 (filename={filename})")

    def invalidate_by_question(self, question: str):
        """
        使指定问题的相关缓存失效

        Args:
            question: 问题文本
        """
        if not question:
            return
        
        # 生成问题对应的缓存 key 并删除
        params = {'question': question}
        cache_key = self._get_cache_key(params)
        self.cache.delete("search", cache_key)
        logger.debug(f"搜索缓存已失效 (question={question[:50]}...)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        return {
            "cache_ttl": self.cache_ttl,
            "cache_type": "redis" if self.cache.redis_client else "disabled",
            "redis_available": self.cache.redis_client is not None
        }

    def clear_all_cache(self):
        """
        清空所有搜索缓存
        """
        self.cache.delete_pattern("search")
        logger.info("所有搜索缓存已清空")


# 全局实例（可选）
_default_cache_service = None


def get_cached_search_service() -> CachedSearchService:
    """
    获取缓存的搜索服务实例（单例）

    Returns:
        CachedSearchService 实例
    """
    global _default_cache_service
    if _default_cache_service is None:
        _default_cache_service = CachedSearchService()
    return _default_cache_service


__all__ = [
    'CachedSearchService',
    'get_cached_search_service'
]