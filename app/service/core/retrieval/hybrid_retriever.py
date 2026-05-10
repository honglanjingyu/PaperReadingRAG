# app/service/core/retrieval/hybrid_retriever.py
"""混合检索器 - 融合向量检索和 BM25 关键词检索"""

import logging
import os
from typing import List, Dict, Any, Optional

from .es_bm25_retriever import get_es_bm25_retriever
from .base import BaseRetriever, ScoreMerger

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器 - 融合向量检索和 BM25 关键词检索"""

    def __init__(
            self,
            vector_store=None,
            embedding_manager=None,
            use_jieba: bool = True,
            use_synonyms: bool = True,
            use_es_bm25: bool = True
    ):
        """
        初始化混合检索器

        Args:
            vector_store: 向量存储实例
            embedding_manager: Embedding 管理器实例
            use_jieba: 是否使用 jieba 分词
            use_synonyms: 是否使用同义词扩展
            use_es_bm25: 是否使用 ES BM25
        """
        from app.service.core.vector_store import get_vector_store
        from app.service.core.embedding import get_embedding_service

        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_manager or get_embedding_service()
        self.use_jieba = use_jieba
        self.use_synonyms = use_synonyms

        # 初始化 ES BM25
        self.use_es_bm25 = use_es_bm25 and self._check_es_available()
        self.es_bm25 = get_es_bm25_retriever() if self.use_es_bm25 else None

        if not self.use_es_bm25:
            logger.warning("ES BM25 不可用，仅使用向量检索")

        # 加载权重
        self._load_weights()

    def _check_es_available(self) -> bool:
        """检查 ES 是否可用"""
        try:
            return get_es_bm25_retriever().is_available()
        except Exception as e:
            logger.warning(f"ES BM25 不可用: {e}")
            return False

    def _load_weights(self):
        """从环境变量加载混合检索权重"""
        vector_weight_str = os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")
        keyword_weight_str = os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")

        try:
            self.vector_weight = float(vector_weight_str)
        except ValueError:
            self.vector_weight = 0.6

        try:
            self.keyword_weight = float(keyword_weight_str)
        except ValueError:
            self.keyword_weight = 0.4

        # 归一化权重
        total = self.vector_weight + self.keyword_weight
        if abs(total - 1.0) > 0.01:
            self.vector_weight /= total
            self.keyword_weight /= total
            logger.info(f"权重已归一化: vector={self.vector_weight:.4f}, keyword={self.keyword_weight:.4f}")

        logger.info(f"混合检索权重: 向量={self.vector_weight}, 关键词={self.keyword_weight}")

    def set_weights(self, vector_weight: float, keyword_weight: float):
        """手动设置混合检索权重"""
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.info(f"混合检索权重已更新: 向量={self.vector_weight}, 关键词={self.keyword_weight}")

    def hybrid_search(
            self,
            query: str,
            index_name: str,
            top_k: int = 5,
            keyword_weight: float = None,
            vector_weight: float = None,
            similarity_threshold: float = 0.3,
            verbose: bool = False,
            user_level: str = None
    ) -> List[Dict[str, Any]]:
        """
        混合检索：同时进行向量检索和 BM25 关键词检索，融合结果

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            keyword_weight: 关键词权重（可选）
            vector_weight: 向量权重（可选）
            similarity_threshold: 相似度阈值
            verbose: 是否打印详细信息
            user_level: 用户等级（用于权限过滤）

        Returns:
            检索结果列表
        """
        if not query:
            return []

        kw_weight = keyword_weight or self.keyword_weight
        vec_weight = vector_weight or self.vector_weight

        # 并行执行两种检索
        vector_results = self._vector_search(query, index_name, top_k * 2, verbose, user_level)
        keyword_results = self._bm25_search(query, index_name, top_k * 2, verbose, user_level)

        if not vector_results and not keyword_results:
            return []

        # 融合结果
        results = ScoreMerger.normalize_results(
            vector_results, keyword_results, vec_weight, kw_weight
        )

        # 过滤阈值
        filtered = [r for r in results if r.get('final_score', 0) >= similarity_threshold]

        if verbose:
            print(f"混合检索完成: 召回 {len(filtered)} 个结果")

        return filtered[:top_k]

    def _vector_search(self, query: str, index_name: str, top_k: int,
                       verbose: bool = False, user_level: str = None) -> List[Dict]:
        """执行向量检索（带等级过滤）"""
        try:
            query_vector = self.embedding_service.generate_embedding(query)
            if not query_vector:
                if verbose:
                    print("  向量检索: 查询向量生成失败")
                return []

            results = self.vector_store.search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                similarity_threshold=0.1,
                user_level=user_level
            )

            if verbose and results:
                print(f"  向量检索召回: {len(results)} 个块")

            for r in results:
                r['_search_type'] = 'vector'
                r['vector_score'] = r.get('_score', 0)
                r['keyword_score'] = 0
                r['final_score'] = r.get('_score', 0)

            return results

        except Exception as e:
            if verbose:
                print(f"  向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, index_name: str, top_k: int,
                     verbose: bool = False, user_level: str = None) -> List[Dict]:
        """执行 ES BM25 关键词检索（带等级过滤）"""
        if not self.use_es_bm25 or not self.es_bm25 or not self.es_bm25.is_available():
            if verbose:
                print("  ES BM25 不可用")
            return []

        try:
            if not self.vector_store.index_exists(index_name):
                if verbose:
                    print(f"  索引不存在: {index_name}")
                return []

            doc_count = self.vector_store.get_document_count(index_name)
            if doc_count == 0:
                if verbose:
                    print("  索引为空")
                return []

            results = self.es_bm25.search(
                query=query,
                index_name=index_name,
                top_k=top_k,
                min_score=0.05,
                user_level=user_level
            )

            if verbose and results:
                print(f"  ES BM25 检索召回: {len(results)} 个块")

            for r in results:
                r['_search_type'] = 'bm25'
                r['keyword_score'] = r.get('_score', 0)
                r['keyword_score_raw'] = r.get('_score', 0)
                r['vector_score'] = 0.0

            return results

        except Exception as e:
            if verbose:
                print(f"  ES BM25 检索失败: {e}")
            return []
