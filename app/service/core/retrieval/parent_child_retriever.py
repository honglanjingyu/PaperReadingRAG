# app/service/core/retrieval/parent_child_retriever.py
"""
父子查询检索器 - 专门负责父子分块的检索逻辑
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from app.service.core.embedding import get_embedding_service
from app.service.core.vector_store import get_vector_search_service
from app.service.core.retrieval import HybridRetriever, Reranker, QueryRewriter
from app.service.core.cache import get_cache_manager

logger = logging.getLogger(__name__)


class ParentChildRetriever:
    """
    父子查询检索器

    检索流程：
    1. 使用子块进行混合检索（向量 + BM25）
    2. 根据子块结果聚合为父块
    3. 可选的重排序
    """

    def __init__(self):
        self.vector_search = get_vector_search_service()
        self.embedding_service = get_embedding_service()
        self.hybrid_retriever = HybridRetriever()
        self.cache_manager = get_cache_manager()

        # RRF 参数
        self._rrf_k = int(os.getenv("RRF_K", "60"))

        # 检索源权重
        self._source_weights = {
            "vector": float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
            "bm25": float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")),
        }

        logger.info("ParentChildRetriever 初始化完成")

    def retrieve_parents_from_children(self, child_results: List[Dict], top_k: int = 5) -> List[Dict]:
        if not child_results:
            return []

        parent_map = {}

        for result in child_results:
            parent_id = result.get('parent_id', '')
            parent_content = result.get('parent_content', '')

            # ========== 关键修复：优先从子块结果中获取文档名 ==========
            # 子块结果中 document_name 字段是正确的，需要保留
            doc_name = (
                    result.get('document_name') or
                    result.get('docnm') or
                    result.get('docnm_kwd') or
                    result.get('source') or
                    ''
            )

            if not parent_id:
                # 没有父块信息，直接使用子块作为结果
                doc_id = result.get('_id', result.get('id', ''))
                parent_map[doc_id] = {
                    'parent_id': doc_id,
                    'content': result.get('content', result.get('content_with_weight', '')),
                    'score': result.get('_score', result.get('score', 0)),
                    'child_count': 1,
                    'docnm': doc_name,
                    'document_name': doc_name,
                    'is_parent': False
                }
            else:
                if parent_id not in parent_map:
                    parent_map[parent_id] = {
                        'parent_id': parent_id,
                        'content': parent_content or result.get('content', ''),
                        'score': 0,
                        'child_count': 0,
                        'child_scores': [],
                        'is_parent': True,
                        'children': [],
                        'docnm': doc_name,  # ← 关键：第一次创建时设置文档名
                        'document_name': doc_name
                    }
                else:
                    # 如果已有父块，确保文档名存在（如果当前有文档名而之前没有，则更新）
                    if doc_name and not parent_map[parent_id].get('docnm'):
                        parent_map[parent_id]['docnm'] = doc_name
                        parent_map[parent_id]['document_name'] = doc_name

                parent_map[parent_id]['child_count'] += 1
                parent_map[parent_id]['child_scores'].append(result.get('_score', result.get('score', 0)))
                parent_map[parent_id]['score'] = max(
                    parent_map[parent_id]['score'],
                    result.get('_score', result.get('score', 0))
                )

                if 'children' in parent_map[parent_id]:
                    parent_map[parent_id]['children'].append(result)

        # 转换为列表并按分数排序
        results = []
        for parent_id, data in parent_map.items():
            result_item = {
                'parent_id': data['parent_id'],
                'content': data['content'],
                'score': data['score'],
                'child_count': data['child_count'],
                'is_parent': data.get('is_parent', True),
                'docnm': data.get('docnm', ''),  # ← 确保输出文档名
                'document_name': data.get('document_name', '')  # ← 确保输出文档名
            }

            # 调试日志
            logger.debug(
                f"聚合结果: parent_id={parent_id}, docnm={result_item['docnm']}, document_name={result_item['document_name']}")

            results.append(result_item)

        # 按分数排序
        results.sort(key=lambda x: x.get('score', 0), reverse=True)

        return results[:top_k]

    def search_children(
            self,
            query: str,
            index_name: str,
            top_k: int = 20,
            similarity_threshold: float = 0.3,
            user_level: str = None
    ) -> List[Dict]:
        """
        检索子块（向量检索）

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            similarity_threshold: 相似度阈值
            user_level: 用户等级

        Returns:
            子块检索结果
        """
        query_vector = self.embedding_service.generate_embedding(query)
        if not query_vector:
            return []

        results = self.vector_search.similarity_search(
            query_vector=query_vector,
            index_name=index_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            user_level=user_level
        )

        return results

    def hybrid_search_children(
            self,
            query: str,
            index_name: str,
            top_k: int = 20,
            keyword_weight: float = None,
            vector_weight: float = None,
            similarity_threshold: float = 0.3,
            user_level: str = None,
            verbose: bool = False
    ) -> List[Dict]:
        """
        混合检索子块（向量 + BM25）

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            keyword_weight: 关键词权重
            vector_weight: 向量权重
            similarity_threshold: 相似度阈值
            user_level: 用户等级
            verbose: 是否打印详情

        Returns:
            子块检索结果
        """
        kw_weight = keyword_weight or self._source_weights["bm25"]
        vec_weight = vector_weight or self._source_weights["vector"]

        results = self.hybrid_retriever.hybrid_search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            keyword_weight=kw_weight,
            vector_weight=vec_weight,
            similarity_threshold=similarity_threshold,
            verbose=verbose,
            user_level=user_level
        )

        # 同时写日志
        for i, r in enumerate(results):
            logger.warning(
                f"DEBUG hybrid_search_children {i}: docnm={r.get('docnm')}, document_name={r.get('document_name')}")

        return results

    def search_parents(
            self,
            query: str,
            index_name: str,
            top_k: int = 5,
            recall_k: int = 20,
            keyword_weight: float = None,
            vector_weight: float = None,
            similarity_threshold: float = 0.3,
            user_level: str = None,
            verbose: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        父子检索 - 主要接口

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回的父块数量
            recall_k: 召回的�块数量
            keyword_weight: 关键词权重
            vector_weight: 向量权重
            similarity_threshold: 相似度阈值
            user_level: 用户等级
            verbose: 是否打印详情

        Returns:
            (parent_results, child_results): 父块结果和原始子块结果
        """
        # 1. 混合检索子块
        child_results = self.hybrid_search_children(
            query=query,
            index_name=index_name,
            top_k=recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            user_level=user_level,
            verbose=verbose
        )

        if not child_results:
            return [], []

        # 2. 聚合为父块
        parent_results = self.retrieve_parents_from_children(child_results, top_k)

        return parent_results, child_results

    def search_parents_with_rerank(
            self,
            query: str,
            index_name: str,
            top_k: int = 5,
            recall_k: int = 20,
            keyword_weight: float = None,
            vector_weight: float = None,
            similarity_threshold: float = 0.3,
            enable_rerank: bool = True,
            rerank_type: str = "auto",
            user_level: str = None,
            verbose: bool = False
    ) -> Dict[str, Any]:
        """
        父子检索 + 重排序
        """
        # 1. 执行父子检索
        parent_results, child_results = self.search_parents(
            query=query,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            user_level=user_level,
            verbose=verbose
        )

        if not parent_results:
            return {
                "success": False,
                "error": "未找到相关文档",
                "question": query,
                "results": []
            }

        # 2. 重排序
        final_results = parent_results
        if enable_rerank and parent_results:
            try:
                reranker = Reranker(api_type=rerank_type)

                docs_for_rerank = []
                for r in parent_results:
                    # ========== 关键修复：保留文档名字段 ==========
                    docs_for_rerank.append({
                        'content': r.get('content', ''),
                        'parent_id': r.get('parent_id', ''),
                        'child_count': r.get('child_count', 0),
                        'original_score': r.get('score', 0),
                        'docnm': r.get('docnm', ''),  # ← 添加
                        'document_name': r.get('document_name', '')  # ← 添加
                    })

                reranked = reranker.rerank(
                    query=query,
                    documents=docs_for_rerank,
                    top_k=top_k
                )

                if reranked:
                    final_results = reranked[:top_k]

                    # ========== 调试日志：检查重排序结果是否有文档名 ==========
                    for i, r in enumerate(final_results):
                        logger.debug(
                            f"重排序结果 {i}: parent_id={r.get('parent_id')}, docnm={r.get('docnm')}, document_name={r.get('document_name')}")

            except Exception as e:
                if verbose:
                    print(f"重排序失败: {e}")
                logger.warning(f"重排序失败: {e}")

        # 3. 格式化结果
        formatted_results = []
        for i, result in enumerate(final_results, 1):
            # ========== 优先从 result 中获取文档名 ==========
            doc_name = result.get('document_name') or result.get('docnm') or ''

            formatted_results.append({
                "rank": i,
                "score": result.get("rerank_score", result.get("score", 0)),
                "content": result.get("content", ""),
                "parent_id": result.get("parent_id", ""),
                "child_count": result.get("child_count", 0),
                "document_name": doc_name,
                "docnm": doc_name,
                "is_parent": result.get("is_parent", True)
            })

        return {
            "success": True,
            "question": query,
            "top_k": top_k,
            "recall_k": recall_k,
            "total_recalled": len(child_results),
            "total_returned": len(formatted_results),
            "results": formatted_results,
            "enable_rerank": enable_rerank
        }

    def search_with_query_rewrite(
            self,
            question: str,
            index_name: str,
            top_k: int = 5,
            recall_k: int = 20,
            enable_query_rewrite: bool = True,
            **kwargs
    ) -> Dict[str, Any]:
        """
        带查询改写的父子检索

        Args:
            question: 用户问题
            index_name: 索引名称
            top_k: 返回的父块数量
            recall_k: 召回的�块数量
            enable_query_rewrite: 是否启用查询改写
            **kwargs: 其他参数

        Returns:
            检索结果字典
        """
        rewritten_query = question

        if enable_query_rewrite:
            try:
                rewriter = QueryRewriter()
                rewritten_query = rewriter.rewrite(question, strategy='synonym')
                logger.debug(f"查询改写: {question} -> {rewritten_query}")
            except Exception as e:
                logger.warning(f"查询改写失败: {e}")

        result = self.search_parents_with_rerank(
            query=rewritten_query,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            **kwargs
        )

        result["rewritten_query"] = rewritten_query if enable_query_rewrite else None
        result["enable_query_rewrite"] = enable_query_rewrite

        return result

    def batch_search_parents(
            self,
            queries: List[str],
            index_name: str,
            top_k: int = 5,
            recall_k: int = 20,
            user_level: str = None
    ) -> List[Dict[str, Any]]:
        """
        批量父子检索

        Args:
            queries: 查询列表
            index_name: 索引名称
            top_k: 每个查询返回的父块数量
            recall_k: 每个查询召回的�块数量
            user_level: 用户等级

        Returns:
            检索结果列表
        """
        results = []
        for query in queries:
            result = self.search_parents_with_rerank(
                query=query,
                index_name=index_name,
                top_k=top_k,
                recall_k=recall_k,
                user_level=user_level
            )
            results.append(result)

        return results

    def get_cache_key(self, params: Dict) -> str:
        """生成缓存key"""
        import json
        import hashlib
        params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    def search_with_cache(
            self,
            question: str,
            index_name: str,
            top_k: int = 5,
            recall_k: int = 20,
            cache_ttl: int = 300,
            **kwargs
    ) -> Dict[str, Any]:
        """
        带缓存的父子检索

        Args:
            question: 用户问题
            index_name: 索引名称
            top_k: 返回的父块数量
            recall_k: 召回的�块数量
            cache_ttl: 缓存过期时间（秒）
            **kwargs: 其他参数

        Returns:
            检索结果字典
        """
        # 生成缓存key
        cache_params = {
            'q': question,
            'index': index_name,
            'top_k': top_k,
            'recall_k': recall_k,
            **kwargs
        }
        cache_key = self.get_cache_key(cache_params)

        # 尝试从缓存获取
        cached = self.cache_manager.get("parent_child_search", cache_key)
        if cached is not None:
            logger.debug(f"父子检索缓存命中: {question[:30]}...")
            return cached

        # 执行检索
        result = self.search_with_query_rewrite(
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            **kwargs
        )

        # 存入缓存
        if result.get("success"):
            self.cache_manager.set("parent_child_search", cache_key, result, cache_ttl)

        return result


# 全局单例
_parent_child_retriever = None


def get_parent_child_retriever() -> ParentChildRetriever:
    """获取父子查询检索器实例"""
    global _parent_child_retriever
    if _parent_child_retriever is None:
        _parent_child_retriever = ParentChildRetriever()
    return _parent_child_retriever


__all__ = [
    'ParentChildRetriever',
    'get_parent_child_retriever'
]