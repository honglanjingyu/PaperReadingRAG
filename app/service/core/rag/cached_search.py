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

    def search_with_cache(self, search_func, **kwargs) -> Dict[str, Any]:
        """带缓存的搜索"""
        # 提取 verbose 参数，不传递给缓存逻辑
        verbose = kwargs.pop('verbose', False)
        use_cache = kwargs.pop('use_cache', True)

        if not use_cache:
            return search_func(**kwargs)

        cache_key = self._get_cache_key(kwargs)

        # 查缓存
        cached = self.cache.get("search", cache_key)
        if cached is not None:
            logger.debug(f"搜索缓存命中")
            return cached

        # 执行搜索
        result = search_func(**kwargs)

        # 写缓存
        if result and result.get("success"):
            self.cache.set("search", cache_key, result, self.cache_ttl)

        return result

    def invalidate(self, pattern: str = None):
        """使缓存失效"""
        if pattern:
            self.cache.delete_pattern(f"search:{pattern}")
        else:
            self.cache.delete_pattern("search")
        logger.info("搜索缓存已失效")