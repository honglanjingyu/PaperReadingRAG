# app/service/core/cache/cache_manager.py

import os
import json
import hashlib
import logging
from typing import Optional, Any, Dict, Callable, List
from functools import wraps

logger = logging.getLogger(__name__)

# Redis 可用性标志
REDIS_AVAILABLE = False
redis_client = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("redis 模块未安装，请运行: pip install redis")


def get_redis_client():
    """获取 Redis 客户端实例"""
    global redis_client, REDIS_AVAILABLE

    if not REDIS_AVAILABLE:
        return None

    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                db=int(os.getenv("REDIS_DB", 1)),  # 使用 DB 1 用于缓存
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            redis_client.ping()
            logger.info("Redis 缓存连接成功")
        except Exception as e:
            logger.warning(f"Redis 连接失败: {e}，缓存功能将不可用")
            redis_client = None

    return redis_client


class CacheManager:
    """统一缓存管理器 - 只使用 Redis 作为缓存"""

    def __init__(self):
        self.redis_client = get_redis_client()
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))

        if self.redis_client is None:
            logger.warning("Redis 不可用，缓存功能将禁用")
        else:
            logger.info("CacheManager 初始化完成，使用 Redis 缓存")

    def _get_cache_key(self, prefix: str, key: str) -> str:
        """生成缓存 key"""
        # 使用 MD5 缩短 key
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        return f"rag:cache:{prefix}:{key_hash}"

    def get(self, prefix: str, key: str) -> Optional[Any]:
        """从 Redis 读取缓存"""
        if self.redis_client is None:
            return None

        cache_key = self._get_cache_key(prefix, key)

        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"缓存命中: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis 读取失败: {e}")

        return None

    def set(self, prefix: str, key: str, value: Any, ttl: int = None):
        """写入 Redis 缓存"""
        if self.redis_client is None:
            return

        cache_key = self._get_cache_key(prefix, key)
        ttl = ttl or self.default_ttl

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(value, ensure_ascii=False))
            logger.debug(f"缓存写入: {cache_key}, TTL={ttl}")
        except Exception as e:
            logger.warning(f"Redis 写入失败: {e}")

    def delete(self, prefix: str, key: str):
        """删除缓存"""
        if self.redis_client is None:
            return

        cache_key = self._get_cache_key(prefix, key)

        try:
            self.redis_client.delete(cache_key)
            logger.debug(f"缓存删除: {cache_key}")
        except Exception as e:
            logger.warning(f"Redis 删除失败: {e}")

    def delete_pattern(self, pattern: str):
        """批量删除匹配的缓存"""
        if self.redis_client is None:
            return

        full_pattern = f"rag:cache:{pattern}:*"

        try:
            keys = self.redis_client.keys(full_pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"批量删除缓存: {len(keys)} 条")
        except Exception as e:
            logger.warning(f"Redis 批量删除失败: {e}")

    def clear(self):
        """清空所有缓存"""
        if self.redis_client is None:
            return

        try:
            keys = self.redis_client.keys("rag:cache:*")
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"清空所有缓存: {len(keys)} 条")
        except Exception as e:
            logger.warning(f"Redis 清空失败: {e}")

    def batch_get(self, prefix: str, keys: List[str]) -> Dict[str, Optional[Any]]:
        """
        批量从 Redis 读取缓存（使用 Pipeline）

        Args:
            prefix: 缓存前缀
            keys: 原始 key 列表

        Returns:
            字典 {原始key: 缓存值}
        """
        if self.redis_client is None or not keys:
            return {key: None for key in keys}

        try:
            # 先计算所有 cache key
            cache_keys = [self._get_cache_key(prefix, key) for key in keys]

            # 使用 pipeline 批量查询
            pipe = self.redis_client.pipeline()
            for cache_key in cache_keys:
                pipe.get(cache_key)
            results = pipe.execute()

            result_dict = {}
            for original_key, cached in zip(keys, results):
                if cached:
                    result_dict[original_key] = json.loads(cached)
                else:
                    result_dict[original_key] = None

            hit_count = sum(1 for v in result_dict.values() if v is not None)
            logger.debug(f"批量查询: {len(keys)} 个key, 命中 {hit_count} 个")
            return result_dict

        except Exception as e:
            logger.warning(f"Redis 批量读取失败: {e}")
            # 回退到单条查询
            return {key: self.get(prefix, key) for key in keys}

    def batch_set(self, prefix: str, items: Dict[str, Any], ttl: int = None):
        """
        批量写入 Redis 缓存（使用 Pipeline）

        Args:
            prefix: 缓存前缀
            items: 字典 {原始key: value}
            ttl: 过期时间（秒）
        """
        if self.redis_client is None or not items:
            return

        ttl = ttl or self.default_ttl

        try:
            pipe = self.redis_client.pipeline()
            for original_key, value in items.items():
                cache_key = self._get_cache_key(prefix, original_key)
                pipe.setex(cache_key, ttl, json.dumps(value, ensure_ascii=False))
            pipe.execute()
            logger.debug(f"批量写入: {len(items)} 条, TTL={ttl}")

        except Exception as e:
            logger.warning(f"Redis 批量写入失败: {e}")
            # 回退到单条写入
            for original_key, value in items.items():
                self.set(prefix, original_key, value, ttl)


# 全局缓存实例
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(prefix: str, ttl: int = None, skip_if: Callable = None):
    """缓存装饰器 - 只使用 Redis"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否应该跳过缓存
            if skip_if and skip_if(*args, **kwargs):
                return func(*args, **kwargs)

            cache_manager = get_cache_manager()

            # 如果 Redis 不可用，直接执行函数
            if cache_manager.redis_client is None:
                return func(*args, **kwargs)

            # 生成缓存 key
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 排除不需要缓存的参数
            skip_params = {'verbose', 'callback'}
            key_data = {}
            for name, value in bound_args.arguments.items():
                if name not in skip_params:
                    try:
                        json.dumps(value)
                        key_data[name] = value
                    except (TypeError, ValueError):
                        key_data[name] = str(value)[:100]

            cache_key = json.dumps(key_data, sort_keys=True, ensure_ascii=False)

            # 查缓存
            result = cache_manager.get(prefix, cache_key)
            if result is not None:
                return result

            # 执行函数
            result = func(*args, **kwargs)

            # 缓存结果
            if result is not None:
                cache_manager.set(prefix, cache_key, result, ttl)

            return result

        return wrapper

    return decorator


__all__ = ['CacheManager', 'get_cache_manager', 'cached']