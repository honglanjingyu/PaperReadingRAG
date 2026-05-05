# app/service/core/vector_store/vector_search_service.py
"""向量搜索服务 - 统一接口"""

import logging
from typing import List, Dict, Any, Optional

from .factory import get_vector_store

logger = logging.getLogger(__name__)


class VectorSearchService:
    """向量搜索服务 - 负责相似度搜索和召回"""

    def __init__(self):
        self.store = get_vector_store()
        self.es_store = self.store  # 保持向后兼容

    def similarity_search(
            self,
            query_vector: List[float],
            index_name: str,
            top_k: int = 5,
            similarity_threshold: float = 0.5,
            filter_condition: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """相似度搜索"""
        if not query_vector:
            logger.warning("查询向量为空")
            return []

        # 检查存储是否可用
        if self.store is None:
            logger.error("向量存储未初始化")
            return []

        if not self.store.index_exists(index_name):
            logger.warning(f"索引不存在: {index_name}")
            return []

        try:
            results = self.store.search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                filter_condition=filter_condition,
                similarity_threshold=similarity_threshold
            )

            # 关键修复：确保结果按 _score 降序排序
            results.sort(key=lambda x: x.get("_score", 0), reverse=True)

            logger.info(f"相似度搜索完成: 召回 {len(results)} 个文档块")
            return results
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def batch_similarity_search(
            self,
            query_vectors: List[List[float]],
            index_name: str,
            top_k: int = 5,
            similarity_threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """批量相似度搜索"""
        all_results = []
        for query_vector in query_vectors:
            results = self.similarity_search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            all_results.append(results)
        return all_results


# 全局单例
_vector_search_service = None


def get_vector_search_service() -> VectorSearchService:
    """获取向量搜索服务实例"""
    global _vector_search_service
    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
    return _vector_search_service


__all__ = ['VectorSearchService', 'get_vector_search_service']