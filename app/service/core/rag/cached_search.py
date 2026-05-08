# app/service/core/rag/cached_search.py

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional

from ..cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedSearchService:
    """带缓存的搜索服务 - 缓存搜索结果"""

    def __init__(self, cache_ttl: int = None):
        self.cache_manager = get_cache_manager()
        self.cache_ttl = cache_ttl or int(os.getenv("CACHE_SEARCH_TTL", "300"))

    def _normalize_question(self, question: str) -> str:
        import re
        normalized = question.strip().lower()
        normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _get_cache_key(self, params: Dict) -> str:
        """生成搜索缓存 key"""
        params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"search:{content_hash}"

    def _serialize_results(self, results: Dict) -> str:
        """序列化搜索结果（只保留必要字段）"""
        if not results:
            return None

        # 只缓存必要的数据，减少存储
        cached_data = {
            "success": results.get("success"),
            "question": results.get("question"),
            "rewritten_query": results.get("rewritten_query"),
            "results": []
        }

        for r in results.get("results", [])[:10]:  # 最多缓存10条
            cached_data["results"].append({
                "rank": r.get("rank"),
                "score": r.get("score"),
                "content": r.get("content", "")[:500],  # 截断长文本
                "document_name": r.get("document_name"),
                "chunk_id": r.get("chunk_id")
            })

        cached_data["total_recalled"] = len(cached_data["results"])
        cached_data["total_returned"] = len(cached_data["results"])

        return json.dumps(cached_data, ensure_ascii=False)

    def _deserialize_results(self, data: str) -> Dict:
        """反序列化搜索结果"""
        if not data:
            return None
        return json.loads(data)

    # app/service/core/rag/cached_search.py

    def search_with_cache(
            self,
            question: str,
            search_func,
            top_k: int = 5,
            recall_k: int = 10,
            similarity_threshold: float = 0.3,
            enable_rerank: bool = True,
            enable_query_rewrite: bool = True,
            index_name: str = "rag_documents",
            **kwargs
    ) -> Dict[str, Any]:
        import time

        enabled = os.getenv("ENABLE_SEARCH_CACHE", "true").lower() == "true"

        # 生成缓存参数
        cache_params = {
            'q': question,  # 使用原始问题，不要归一化
            'top_k': top_k,
            'recall_k': recall_k,
            'sim': similarity_threshold,
            'rerank': enable_rerank,
            'rewrite': enable_query_rewrite,
            'index': index_name,
            'kw_weight': kwargs.get('keyword_weight', 0.3),
            'vec_weight': kwargs.get('vector_weight', 0.7),
            'rerank_type': kwargs.get('rerank_type', 'auto')
        }
        cache_key = self._get_cache_key(cache_params)

        # 提取 verbose 参数
        verbose = kwargs.pop('verbose', False)

        if enabled:
            start = time.time()
            cached = self.cache_manager.get("search", cache_key)
            if cached is not None:
                logger.info(f"搜索缓存命中: {question[:30]}..., 耗时: {(time.time() - start) * 1000:.2f}ms")
                # 确保返回的缓存数据完整
                if isinstance(cached, dict) and cached.get("success"):
                    return cached
                else:
                    logger.warning(f"缓存数据无效，将重新搜索")

        start = time.time()
        result = search_func(
            question=question,
            top_k=top_k,
            recall_k=recall_k,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
            enable_query_rewrite=enable_query_rewrite,
            index_name=index_name,
            verbose=verbose,
            **kwargs
        )
        search_time = (time.time() - start) * 1000
        logger.info(f"搜索执行完成: {question[:30]}..., 耗时: {search_time:.2f}ms")

        if enabled and result and result.get("success"):
            self.cache_manager.set("search", cache_key, result, self.cache_ttl)
            logger.info(f"搜索缓存写入: {question[:30]}...")

        return result

    def invalidate_cache(self, question: str = None, pattern: str = None):
        if pattern:
            self.cache_manager.delete_pattern(f"search:{pattern}")
        elif question:
            normalized = self._normalize_question(question)
            cache_key = f"search:{hashlib.md5(normalized.encode()).hexdigest()[:16]}*"
            self.cache_manager.delete_pattern(cache_key)
        else:
            self.cache_manager.delete_pattern("search")
        logger.info("搜索缓存已失效")


__all__ = ['CachedSearchService']