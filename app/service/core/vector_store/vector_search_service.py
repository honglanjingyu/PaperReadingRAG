# app/service/core/vector_store/vector_search_service.py

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorSearchService:
    """向量搜索服务 - 负责相似度搜索和召回"""

    def __init__(self, es_store=None):
        """
        初始化向量搜索服务

        Args:
            es_store: ESVectorStore 实例
        """
        from .es_vector_store import ESVectorStore
        self.es_store = es_store or ESVectorStore()

    def similarity_search(
        self,
        query_vector: List[float],
        index_name: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        filter_condition: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        相似度搜索：在向量数据库中召回最相关的 Top-K 个文档块

        Args:
            query_vector: 查询向量
            index_name: 索引名称
            top_k: 返回数量 (Top-K)
            similarity_threshold: 相似度阈值
            filter_condition: 过滤条件

        Returns:
            检索结果列表，每个结果包含文档内容和相似度分数
        """
        if not query_vector:
            logger.warning("查询向量为空")
            return []

        # 检查索引是否存在
        if not self.es_store.index_exists(index_name):
            logger.warning(f"索引不存在: {index_name}")
            return []

        try:
            # 执行向量相似度搜索
            results = self.es_store.search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                filter_condition=filter_condition,
                similarity_threshold=similarity_threshold
            )

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
        """
        批量相似度搜索

        Args:
            query_vectors: 查询向量列表
            index_name: 索引名称
            top_k: 每个查询返回数量
            similarity_threshold: 相似度阈值

        Returns:
            每个查询的检索结果列表
        """
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


def get_vector_search_service(es_host: str = None) -> VectorSearchService:
    """获取向量搜索服务实例"""
    global _vector_search_service
    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
    return _vector_search_service


__all__ = ['VectorSearchService', 'get_vector_search_service']