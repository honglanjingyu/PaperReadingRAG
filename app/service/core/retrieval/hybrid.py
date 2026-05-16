# app/service/core/retrieval/hybrid.py
"""
混合检索核心模块 - 整合基类、混合检索和父子检索
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# 基类工具函数
# ============================================================

def normalize_scores(scores: List[float], min_val: float = 0.05, max_val: float = 0.95) -> List[float]:
    """归一化分数"""
    if not scores:
        return []
    scores = [s for s in scores if s > 0]
    if not scores:
        return [min_val] * len(scores)
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        return [(min_val + max_val) / 2] * len(scores)
    return [min_val + (s - s_min) / (s_max - s_min) * (max_val - min_val) for s in scores]


def get_level_priority(level: str) -> int:
    """获取用户等级优先级"""
    priorities = {"normal": 1, "admin": 2, "owner": 3}
    return priorities.get(level, 1)


# ============================================================
# RRF 分数融合
# ============================================================

class ScoreMerger:
    """分数融合器 - 使用 RRF"""

    @staticmethod
    def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int = 60) -> Dict[str, float]:
        """RRF 融合算法"""
        scores = defaultdict(float)
        for results in results_lists:
            for rank, result in enumerate(results, 1):
                doc_id = result.get('_id', result.get('id', ''))
                if doc_id:
                    scores[doc_id] += 1.0 / (k + rank)
        return dict(scores)

    @staticmethod
    def reciprocal_rank_fusion_with_weights(
            results_lists: List[List[Dict]], weights: List[float], k: int = 60
    ) -> Dict[str, float]:
        """带权重的 RRF 融合"""
        if len(results_lists) != len(weights):
            raise ValueError("results_lists 和 weights 长度必须相等")
        scores = defaultdict(float)
        for results, weight in zip(results_lists, weights):
            for rank, result in enumerate(results, 1):
                doc_id = result.get('_id', result.get('id', ''))
                if doc_id:
                    scores[doc_id] += weight / (k + rank)
        return dict(scores)

    @staticmethod
    def rrf_normalize_results(
            vector_results: List[Dict], keyword_results: List[Dict],
            vector_weight: float = 1.0, keyword_weight: float = 1.0, k: int = 60
    ) -> List[Dict]:
        """使用 RRF 融合两个检索结果"""
        rrf_scores = defaultdict(float)

        for rank, result in enumerate(vector_results, 1):
            doc_id = result.get('_id', result.get('id', ''))
            if doc_id:
                rrf_scores[doc_id] += vector_weight / (k + rank)

        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.get('_id', result.get('id', ''))
            if doc_id:
                rrf_scores[doc_id] += keyword_weight / (k + rank)

        if not rrf_scores:
            return []

        result_map = {}
        for doc in vector_results:
            doc_id = doc.get('_id', doc.get('id', ''))
            if doc_id and doc_id in rrf_scores:
                result_map[doc_id] = {**doc, 'vector_score': doc.get('_score', 0),
                                      'keyword_score': 0, 'rrf_score': rrf_scores[doc_id],
                                      '_search_types': ['vector']}

        for doc in keyword_results:
            doc_id = doc.get('_id', doc.get('id', ''))
            if doc_id and doc_id in rrf_scores:
                if doc_id in result_map:
                    result_map[doc_id]['keyword_score'] = doc.get('_score', 0)
                    result_map[doc_id]['_search_types'].append('bm25')
                    result_map[doc_id]['rrf_score'] = rrf_scores[doc_id]
                else:
                    result_map[doc_id] = {**doc, 'vector_score': 0, 'keyword_score': doc.get('_score', 0),
                                          'rrf_score': rrf_scores[doc_id], '_search_types': ['bm25']}

        results = list(result_map.values())
        results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
        return results


# ============================================================
# 混合检索器
# ============================================================

class HybridRetriever:
    """混合检索器 - 向量 + BM25，使用 RRF 融合"""

    def __init__(self, vector_store=None, embedding_manager=None):
        from app.service.core.vector_store import get_vector_store
        from app.service.core.embedding import get_embedding_service

        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_manager or get_embedding_service()
        self.es_bm25 = None
        self.use_es_bm25 = self._check_es_available()

        self._rrf_k = int(os.getenv("RRF_K", "60"))
        self._load_weights()

    def _check_es_available(self) -> bool:
        """检查 ES 是否可用"""
        try:
            from .es_bm25 import get_es_bm25_retriever
            self.es_bm25 = get_es_bm25_retriever()
            return self.es_bm25.is_available()
        except Exception as e:
            logger.warning(f"ES BM25 不可用: {e}")
            return False

    def _load_weights(self):
        self.vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
        self.keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4"))

    def _vector_search(self, query: str, index_name: str, top_k: int,
                       user_level: str = None) -> List[Dict]:
        """向量检索"""
        query_vector = self.embedding_service.generate_embedding(query)
        if not query_vector:
            return []

        results = self.vector_store.search(
            query_vector=query_vector, index_name=index_name, top_k=top_k,
            similarity_threshold=0.1, user_level=user_level
        )
        for r in results:
            r['_search_type'] = 'vector'
            r['vector_score'] = r.get('_score', 0)
            if 'docnm' not in r or not r['docnm']:
                r['docnm'] = r.get('document_name', '')
        return results

    def _bm25_search(self, query: str, index_name: str, top_k: int,
                     user_level: str = None) -> List[Dict]:
        """BM25 检索"""
        if not self.use_es_bm25 or not self.es_bm25:
            return []

        results = self.es_bm25.search(
            query=query, index_name=index_name, top_k=top_k,
            min_score=0.05, user_level=user_level
        )
        for r in results:
            r['_search_type'] = 'bm25'
            r['keyword_score'] = r.get('_score', 0)
            if 'docnm' not in r or not r['docnm']:
                r['docnm'] = r.get('document_name', '')
        return results

    def hybrid_search(self, query: str, index_name: str, top_k: int = 5,
                      recall_k: int = 20, user_level: str = None) -> List[Dict]:
        """混合检索"""
        recall_k = max(top_k * 4, recall_k)
        vector_results = self._vector_search(query, index_name, recall_k, user_level)
        keyword_results = self._bm25_search(query, index_name, recall_k, user_level)

        if not vector_results and not keyword_results:
            return []

        results = ScoreMerger.rrf_normalize_results(
            vector_results, keyword_results, self.vector_weight, self.keyword_weight, self._rrf_k
        )
        for r in results:
            r['final_score'] = r.get('rrf_score', 0)

        min_score = float(os.getenv("RRF_MIN_SCORE", "0.01"))
        return [r for r in results if r.get('rrf_score', 0) >= min_score][:top_k]


# ============================================================
# 父子查询检索器
# ============================================================

class ParentChildRetriever:
    """父子查询检索器 - 子块检索 + 父块聚合"""

    def __init__(self):
        self.vector_search = None
        self.embedding_service = None
        self.hybrid_retriever = None
        self._init_services()

    def _init_services(self):
        from app.service.core.embedding import get_embedding_service
        from app.service.core.vector_store import get_vector_search_service
        self.vector_search = get_vector_search_service()
        self.embedding_service = get_embedding_service()
        self.hybrid_retriever = HybridRetriever()

    def retrieve_parents_from_children(self, child_results: List[Dict], top_k: int = 5) -> List[Dict]:
        """从子块结果聚合父块"""
        if not child_results:
            return []

        parent_map = {}
        for result in child_results:
            parent_id = result.get('parent_id', '')
            parent_content = result.get('parent_content', '')
            doc_name = result.get('document_name') or result.get('docnm') or ''

            if not parent_id:
                doc_id = result.get('_id', result.get('id', ''))
                parent_map[doc_id] = {
                    'parent_id': doc_id,
                    'content': result.get('content', ''),
                    'score': result.get('_score', 0),
                    'child_count': 1,
                    'child_scores': [result.get('_score', 0)],  # 添加这行
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
                        'child_scores': [],  # 添加这行
                        'children': [],
                        'docnm': doc_name,
                        'document_name': doc_name,
                        'is_parent': True
                    }
                parent_map[parent_id]['child_count'] += 1
                parent_map[parent_id]['child_scores'].append(result.get('_score', 0))  # 现在可以安全访问
                parent_map[parent_id]['score'] = max(parent_map[parent_id]['score'], result.get('_score', 0))

        results = []
        for d in parent_map.values():
            results.append({
                'parent_id': d['parent_id'],
                'content': d['content'],
                'score': d['score'],
                'child_count': d['child_count'],
                'is_parent': d.get('is_parent', True),
                'docnm': d.get('docnm', ''),
                'document_name': d.get('document_name', '')
            })

        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:top_k]

    def search_parents(self, query: str, index_name: str, top_k: int = 5,
                       recall_k: int = 20, user_level: str = None) -> Tuple[List[Dict], List[Dict]]:
        """父子检索主接口"""
        child_results = self.hybrid_retriever.hybrid_search(
            query, index_name, top_k, recall_k, user_level
        ) if self.hybrid_retriever else []
        return self.retrieve_parents_from_children(child_results, top_k), child_results

    def search_with_query_rewrite(
            self,
            question: str,
            index_name: str,
            top_k: int = 5,
            recall_k: int = 20,
            enable_query_rewrite: bool = True,
            keyword_weight: float = None,
            vector_weight: float = None,
            similarity_threshold: float = 0.3,
            enable_rerank: bool = True,
            rerank_type: str = "auto",
            user_level: str = None,
            **kwargs
    ) -> Dict[str, Any]:
        """带查询改写的父子检索"""
        from .rewriter import QueryRewriter
        from .reranker import Reranker

        rewritten_query = question

        if enable_query_rewrite:
            try:
                rewriter = QueryRewriter()
                rewritten_query = rewriter.rewrite(question, strategy='synonym')
            except Exception as e:
                logger.warning(f"查询改写失败: {e}")

        # 执行父子检索
        parent_results, child_results = self.search_parents(
            query=rewritten_query,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            user_level=user_level
        )

        if not parent_results:
            return {
                "success": False,
                "error": "未找到相关文档",
                "question": question,
                "results": []
            }

        # 重排序
        final_results = parent_results
        if enable_rerank and parent_results:
            try:
                reranker = Reranker(api_type=rerank_type)
                docs_for_rerank = []
                for r in parent_results:
                    docs_for_rerank.append({
                        'content': r.get('content', ''),
                        'parent_id': r.get('parent_id', ''),
                        'child_count': r.get('child_count', 0),
                        'original_score': r.get('score', 0),
                        'docnm': r.get('docnm', ''),
                        'document_name': r.get('document_name', '')
                    })

                reranked = reranker.rerank(
                    query=rewritten_query,
                    documents=docs_for_rerank,
                    top_k=top_k
                )

                if reranked:
                    final_results = reranked[:top_k]
            except Exception as e:
                logger.warning(f"重排序失败: {e}")

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(final_results, 1):
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
            "question": question,
            "rewritten_query": rewritten_query if enable_query_rewrite else None,
            "top_k": top_k,
            "recall_k": recall_k,
            "total_recalled": len(child_results),
            "total_returned": len(formatted_results),
            "results": formatted_results,
            "enable_rerank": enable_rerank,
            "enable_query_rewrite": enable_query_rewrite
        }

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
            user_level: str = None,  # 添加这个参数
            **kwargs
    ) -> Dict[str, Any]:
        """带缓存的父子检索"""
        from app.service.core.cache import get_cache_manager

        cache_manager = get_cache_manager()

        # 生成缓存key
        cache_params = {
            'q': question,
            'index': index_name,
            'top_k': top_k,
            'recall_k': recall_k,
            'user_level': user_level,  # 添加这行
            **kwargs
        }
        cache_key = self.get_cache_key(cache_params)

        # 尝试从缓存获取
        cached = cache_manager.get("parent_child_search", cache_key)
        if cached is not None:
            logger.debug(f"父子检索缓存命中: {question[:30]}...")
            return cached

        # 执行检索
        result = self.search_with_query_rewrite(
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            user_level=user_level,  # 添加这行
            **kwargs
        )

        # 存入缓存
        if result.get("success"):
            cache_manager.set("parent_child_search", cache_key, result, cache_ttl)

        return result


# ============================================================
# 全局实例
# ============================================================

_parent_child_retriever = None
_hybrid_retriever = None


def get_parent_child_retriever() -> ParentChildRetriever:
    global _parent_child_retriever
    if _parent_child_retriever is None:
        _parent_child_retriever = ParentChildRetriever()
    return _parent_child_retriever


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


__all__ = ['HybridRetriever', 'ParentChildRetriever', 'get_parent_child_retriever',
           'get_hybrid_retriever', 'ScoreMerger', 'normalize_scores', 'get_level_priority']