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
        self.cache = get_cache_manager()
        self.cache_ttl = cache_ttl

    def _get_cache_key(self, params: Dict) -> str:
        """生成缓存 key"""
        params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    @staticmethod
    def _cache_prefix(index_name: str) -> str:
        """缓存前缀带集合名（报告 2.3）。

        CacheManager 拼出的真实 key 是 ``rag:cache:{prefix}:{md5}``，
        prefix 里带 index_name 才能用 ``delete_pattern`` 按集合精准失效。
        """
        return f"search:{index_name or 'default'}"

    def search_with_cache(self, search_func, **kwargs) -> Dict[str, Any]:
        """带缓存的搜索"""
        # 提取 verbose 参数，不传递给缓存逻辑
        verbose = kwargs.pop('verbose', False)
        use_cache = kwargs.pop('use_cache', True)

        if not use_cache:
            return search_func(**kwargs)

        index_name = kwargs.get("index_name") or "default"
        cache_key = self._get_cache_key(kwargs)
        prefix = self._cache_prefix(index_name)

        # 查缓存
        cached = self.cache.get(prefix, cache_key)
        if cached is not None:
            logger.debug("搜索缓存命中: %s", index_name)
            return cached

        # 执行搜索
        result = search_func(**kwargs)

        # 写缓存
        if result and result.get("success"):
            self.cache.set(prefix, cache_key, result, self.cache_ttl)

        return result

    def invalidate(self, index_name: str = None):
        """使检索缓存失效（报告 2.3）。

        原实现用 ``*文件名*`` 去匹配 md5 缓存 key，永远匹配不上，
        文档删除后仍会命中旧结果。现在 key 按 index_name 分前缀，
        按集合粒度清空；index_name 为空时清掉所有集合的检索缓存。
        """
        if index_name:
            self.cache.delete_pattern(self._cache_prefix(index_name))
        else:
            self.cache.delete_pattern("search")
        logger.info("搜索缓存已失效: %s", index_name or "全部集合")