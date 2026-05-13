# app/service/core/retrieval/hybrid_retriever.py
"""混合检索器 - 使用 RRF 融合向量检索和 BM25 关键词检索"""

import logging
import os
from typing import List, Dict, Any, Optional

from .es_bm25_retriever import get_es_bm25_retriever
from .base import BaseRetriever, ScoreMerger

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器 - 使用 RRF 融合向量检索和 BM25 关键词检索"""

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

        # RRF 参数
        self._rrf_k = int(os.getenv("RRF_K", "60"))
        self._load_weights()

    def _check_es_available(self) -> bool:
        """检查 ES 是否可用"""
        try:
            return get_es_bm25_retriever().is_available()
        except Exception as e:
            logger.warning(f"ES BM25 不可用: {e}")
            return False

    def _load_weights(self):
        """从环境变量加载混合检索权重（用于 RRF 的权重系数）"""
        vector_weight_str = os.getenv("HYBRID_VECTOR_WEIGHT", "1.0")
        keyword_weight_str = os.getenv("HYBRID_KEYWORD_WEIGHT", "1.0")

        try:
            self.vector_weight = float(vector_weight_str)
        except ValueError:
            self.vector_weight = 1.0

        try:
            self.keyword_weight = float(keyword_weight_str)
        except ValueError:
            self.keyword_weight = 1.0

        logger.info(f"RRF 融合权重: 向量={self.vector_weight}, 关键词={self.keyword_weight}, RRF_k={self._rrf_k}")

    def set_weights(self, vector_weight: float, keyword_weight: float):
        """手动设置 RRF 权重"""
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.info(f"RRF 权重已更新: 向量={self.vector_weight}, 关键词={self.keyword_weight}")

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
        混合检索：同时进行向量检索和 BM25 关键词检索，使用 RRF 融合结果

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            keyword_weight: 关键词权重（可选，默认使用配置值）
            vector_weight: 向量权重（可选，默认使用配置值）
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

        # 并行执行两种检索（召回更多候选，RRF 需要足够的排名信息）
        recall_k = max(top_k * 4, 20)  # RRF 需要更多候选来获得更好的排名
        vector_results = self._vector_search(query, index_name, recall_k, verbose, user_level)
        keyword_results = self._bm25_search(query, index_name, recall_k, verbose, user_level)

        if not vector_results and not keyword_results:
            logger.warning(f"混合检索: 向量和关键词都无结果")
            return []

        # 使用 RRF 融合结果
        results = ScoreMerger.rrf_normalize_results(
            vector_results, keyword_results, vec_weight, kw_weight, self._rrf_k
        )

        # 为提高兼容性，计算 final_score 为 rrf_score（或归一化值）
        for r in results:
            r['final_score'] = r.get('rrf_score', 0)
            r['_score'] = r.get('rrf_score', 0)

        # 过滤阈值（RRF 分数范围在 0~2 左右，需要调整阈值）
        # 使用更宽松的阈值，因为 RRF 分数不同于相似度分数
        min_rrf_score = float(os.getenv("RRF_MIN_SCORE", "0.01"))
        filtered = [r for r in results if r.get('rrf_score', 0) >= min_rrf_score]

        if verbose:
            print(f"RRF 混合检索完成: 召回 {len(filtered)} 个结果")
            if filtered:
                print(f"  最高 RRF 分数: {filtered[0].get('rrf_score', 0):.6f}")

        return filtered[:top_k]

    def hybrid_search_with_multiple_sources(
            self,
            query: str,
            index_name: str,
            top_k: int = 5,
            sources: List[str] = None,
            user_level: str = None,
            verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        多源混合检索：支持向量、BM25、同义词扩展等多种检索源

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            sources: 检索源列表，可选 ['vector', 'bm25', 'vector_synonym', 'bm25_synonym']
            user_level: 用户等级
            verbose: 是否打印详细信息

        Returns:
            检索结果列表
        """
        if not query:
            return []

        if sources is None:
            sources = ['vector', 'bm25']

        results_lists = []
        weights = []

        # 原始查询检索
        if 'vector' in sources:
            vec_results = self._vector_search(query, index_name, top_k * 4, verbose, user_level)
            results_lists.append(vec_results)
            weights.append(self.vector_weight)
            logger.debug(f"向量检索: {len(vec_results)} 个结果")

        if 'bm25' in sources:
            bm25_results = self._bm25_search(query, index_name, top_k * 4, verbose, user_level)
            results_lists.append(bm25_results)
            weights.append(self.keyword_weight)
            logger.debug(f"BM25 检索: {len(bm25_results)} 个结果")

        # 同义词扩展查询检索（可选）
        if 'vector_synonym' in sources or 'bm25_synonym' in sources:
            from .query_rewriter import QueryRewriter
            rewriter = QueryRewriter()
            expanded_query = rewriter._expand_with_synonyms(query)

            if expanded_query != query:
                if 'vector_synonym' in sources:
                    vec_expanded = self._vector_search(expanded_query, index_name, top_k * 3, verbose, user_level)
                    results_lists.append(vec_expanded)
                    weights.append(self.vector_weight * 0.7)  # 同义词结果权重略低
                    logger.debug(f"向量(同义词)检索: {len(vec_expanded)} 个结果")

                if 'bm25_synonym' in sources:
                    bm25_expanded = self._bm25_search(expanded_query, index_name, top_k * 3, verbose, user_level)
                    results_lists.append(bm25_expanded)
                    weights.append(self.keyword_weight * 0.7)
                    logger.debug(f"BM25(同义词)检索: {len(bm25_expanded)} 个结果")

        if not results_lists:
            return []

        # 使用带权重的 RRF 融合
        rrf_scores = ScoreMerger.reciprocal_rank_fusion_with_weights(
            results_lists, weights, self._rrf_k
        )

        # 收集文档详情
        doc_map = {}
        for results in results_lists:
            for doc in results:
                doc_id = doc.get('_id', doc.get('id', ''))
                if doc_id and doc_id in rrf_scores:
                    if doc_id not in doc_map:
                        doc_map[doc_id] = {
                            **doc,
                            'rrf_score': rrf_scores[doc_id],
                            '_search_types': []
                        }
                    if '_search_type' in doc:
                        doc_map[doc_id]['_search_types'].append(doc['_search_type'])

        results = list(doc_map.values())
        results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)

        # 计算 final_score
        for r in results:
            r['final_score'] = r.get('rrf_score', 0)
            r['_score'] = r.get('rrf_score', 0)

        min_rrf_score = float(os.getenv("RRF_MIN_SCORE", "0.005"))
        filtered = [r for r in results if r.get('rrf_score', 0) >= min_rrf_score]

        if verbose:
            print(f"多源 RRF 混合检索完成: {len(filtered)} 个结果 (来自 {len(results_lists)} 个源)")

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
                r['vector_score'] = 0.0

            return results

        except Exception as e:
            if verbose:
                print(f"  ES BM25 检索失败: {e}")
            return []