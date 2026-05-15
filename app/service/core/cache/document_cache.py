# app/service/core/cache/document_cache.py

import logging
from typing import Dict, Optional, List, Tuple
from app.service.core.cache import get_cache_manager

logger = logging.getLogger(__name__)


class DocumentCache:
    """文档信息缓存管理器 - 纯 Redis 缓存，使用批量查询减少网络往返"""

    def __init__(self):
        self.cache = get_cache_manager()
        self.cache_ttl = 300  # 5分钟缓存

    def get_document_level(self, filename: str) -> Optional[str]:
        """获取文档等级（从 Redis 缓存）"""
        if not filename:
            return None

        cached = self.cache.get("doc_level", filename)
        if cached:
            logger.debug(f"文档等级缓存命中: {filename} -> {cached}")
            return cached
        return None

    def set_document_level(self, filename: str, level: str):
        """设置文档等级到 Redis 缓存"""
        if not filename or not level:
            return

        self.cache.set("doc_level", filename, level, self.cache_ttl)
        logger.debug(f"文档等级缓存写入: {filename} -> {level}")

    def delete_document_level(self, filename: str):
        """删除文档等级缓存"""
        if not filename:
            return

        self.cache.delete("doc_level", filename)
        logger.info(f"文档等级缓存已删除: {filename}")

    def batch_get_levels(self, filenames: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """
        批量获取文档等级（使用 Redis pipeline 批量查询）

        Returns:
            (cached_levels, missing_filenames)
        """
        if not filenames:
            return {}, []

        # 使用批量查询 API
        batch_results = self.cache.batch_get("doc_level", filenames)

        cached_levels = {}
        missing = []

        for filename in filenames:
            level = batch_results.get(filename)
            if level:
                cached_levels[filename] = level
            else:
                missing.append(filename)

        if cached_levels:
            logger.debug(f"批量获取: 缓存命中 {len(cached_levels)}/{len(filenames)} 个文档")
        return cached_levels, missing

    def batch_set_levels(self, level_map: Dict[str, str]):
        """批量设置文档等级到 Redis 缓存（使用 pipeline）"""
        if not level_map:
            return

        self.cache.batch_set("doc_level", level_map, self.cache_ttl)
        logger.debug(f"批量写入缓存: {len(level_map)} 个文档")

    def invalidate_all(self):
        """使所有文档等级缓存失效"""
        self.cache.delete_pattern("doc_level")
        logger.info("所有文档等级缓存已失效")

    def get_cache_stats(self) -> Dict[str, any]:
        """获取缓存统计信息"""
        return {
            "cache_ttl": self.cache_ttl,
            "cache_type": "redis" if self.cache.redis_client else "memory"
        }


# 全局单例
_doc_cache = None


def get_document_cache() -> DocumentCache:
    """获取文档缓存管理器实例（单例）"""
    global _doc_cache
    if _doc_cache is None:
        _doc_cache = DocumentCache()
    return _doc_cache


# ========== 便捷函数（供其他模块使用） ==========

def cache_document_level(filename: str, level: str):
    """缓存单个文档等级（便捷函数）"""
    cache = get_document_cache()
    cache.set_document_level(filename, level)


def get_cached_document_level(filename: str) -> Optional[str]:
    """获取缓存的文档等级（便捷函数）"""
    cache = get_document_cache()
    return cache.get_document_level(filename)


def invalidate_document_cache(filename: str = None):
    """使文档缓存失效（便捷函数）"""
    cache = get_document_cache()
    if filename:
        cache.delete_document_level(filename)
    else:
        cache.invalidate_all()


__all__ = [
    'DocumentCache',
    'get_document_cache',
    'cache_document_level',
    'get_cached_document_level',
    'invalidate_document_cache'
]