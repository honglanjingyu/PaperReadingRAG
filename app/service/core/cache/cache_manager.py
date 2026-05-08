# app/service/core/cache/cache_manager.py
"""
统一缓存管理器 - 支持多级缓存
"""

import os
import json
import hashlib
import logging
from typing import Optional, Any, Dict, Callable
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
            logger.warning(f"Redis 连接失败: {e}，将使用内存缓存")
            redis_client = None

    return redis_client


class CacheManager:
    """统一缓存管理器 - 支持多级缓存 (L1: 内存, L2: Redis)"""

    def __init__(self):
        self.redis_client = get_redis_client()
        self.local_cache: Dict[str, Any] = {}  # L1: 本地内存缓存
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))
        self.max_local_size = 1000  # 内存缓存最大条目数

    def _get_cache_key(self, prefix: str, key: str) -> str:
        """生成缓存 key"""
        # 使用 MD5 缩短 key
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        return f"rag:cache:{prefix}:{key_hash}"

    def get(self, prefix: str, key: str) -> Optional[Any]:
        """多级缓存读取"""
        cache_key = self._get_cache_key(prefix, key)

        # L1: 内存缓存
        if cache_key in self.local_cache:
            logger.debug(f"缓存命中 (L1): {cache_key}")
            return self.local_cache[cache_key]

        # L2: Redis 缓存
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    # 解码并存入 L1
                    value = json.loads(cached)
                    self._set_local(cache_key, value)
                    logger.debug(f"缓存命中 (L2): {cache_key}")
                    return value
            except Exception as e:
                logger.warning(f"Redis 读取失败: {e}")

        return None

    def set(self, prefix: str, key: str, value: Any, ttl: int = None):
        """多级缓存写入"""
        cache_key = self._get_cache_key(prefix, key)
        ttl = ttl or self.default_ttl

        # 写入 L1
        self._set_local(cache_key, value)

        # 写入 L2
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, ttl, json.dumps(value, ensure_ascii=False))
                logger.debug(f"缓存写入 (L2): {cache_key}, TTL={ttl}")
            except Exception as e:
                logger.warning(f"Redis 写入失败: {e}")

    def _set_local(self, key: str, value: Any):
        """设置内存缓存，并维护大小限制"""
        if len(self.local_cache) >= self.max_local_size:
            # 移除最早的 200 个条目
            keys_to_remove = list(self.local_cache.keys())[:200]
            for k in keys_to_remove:
                del self.local_cache[k]

        self.local_cache[key] = value

    def delete(self, prefix: str, key: str):
        """删除缓存"""
        cache_key = self._get_cache_key(prefix, key)

        # 删除 L1
        self.local_cache.pop(cache_key, None)

        # 删除 L2
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis 删除失败: {e}")

    def delete_pattern(self, pattern: str):
        """批量删除匹配的缓存"""
        full_pattern = f"rag:cache:{pattern}:*"

        # 清空 L1 中匹配的条目
        keys_to_remove = [k for k in self.local_cache.keys() if k.startswith(f"rag:cache:{pattern}:")]
        for k in keys_to_remove:
            del self.local_cache[k]

        # 删除 L2
        if self.redis_client:
            try:
                keys = self.redis_client.keys(full_pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"批量删除缓存: {len(keys)} 条")
            except Exception as e:
                logger.warning(f"Redis 批量删除失败: {e}")

    def clear(self):
        """清空所有缓存"""
        self.local_cache.clear()

        if self.redis_client:
            try:
                keys = self.redis_client.keys("rag:cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"清空所有缓存: {len(keys)} 条")
            except Exception as e:
                logger.warning(f"Redis 清空失败: {e}")


# 全局缓存实例
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(prefix: str, ttl: int = None, skip_if: Callable = None):
    """缓存装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否应该跳过缓存
            if skip_if and skip_if(*args, **kwargs):
                return func(*args, **kwargs)

            # 生成缓存 key
            cache_manager = get_cache_manager()

            # 序列化参数
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