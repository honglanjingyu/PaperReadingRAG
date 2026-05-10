# app/service/core/retrieval/base.py
"""检索模块基类 - 提供公共功能"""

import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseRetriever:
    """检索器基类 - 提供分数归一化、融合等公共方法"""

    @staticmethod
    def normalize_scores(scores: List[float], min_val: float = 0.05, max_val: float = 0.95) -> List[float]:
        """
        归一化分数到 [min_val, max_val] 范围

        Args:
            scores: 原始分数列表
            min_val: 最小值
            max_val: 最大值

        Returns:
            归一化后的分数列表
        """
        if not scores:
            return []

        scores = [s for s in scores if s > 0]
        if not scores:
            return [min_val] * len(scores)

        s_min, s_max = min(scores), max(scores)
        if s_max == s_min:
            return [(min_val + max_val) / 2] * len(scores)

        return [min_val + (s - s_min) / (s_max - s_min) * (max_val - min_val) for s in scores]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的 token 数量"""
        if not text:
            return 0
        import re
        chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
        others = len(text) - chinese
        return int(chinese / 1.5 + others / 4)

    @staticmethod
    def truncate_text(text: str, max_tokens: int = 500) -> str:
        """截断文本到指定 token 数"""
        tokens = BaseRetriever.estimate_tokens(text)
        if tokens <= max_tokens:
            return text
        # 简单截断
        ratio = max_tokens / tokens
        new_len = int(len(text) * ratio)
        return text[:new_len] + "..."

    @staticmethod
    def get_level_priority(level: str) -> int:
        """获取用户等级优先级（数值越大权限越高）"""
        priorities = {"normal": 1, "admin": 2, "owner": 3}
        return priorities.get(level, 1)


class ScoreMerger:
    """
    分数融合器 - 支持多种融合策略
    从 hybrid_retriever.py 和 reranker.py 提取公共逻辑
    """

    @staticmethod
    def reciprocal_rank_fusion(
            results_lists: List[List[Dict]],
            k: int = 60
    ) -> Dict[str, float]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法

        Args:
            results_lists: 多个检索结果列表，每个列表包含文档
            k: RRF 常数

        Returns:
            文档ID到融合分数的映射
        """
        scores = {}
        for results in results_lists:
            for rank, result in enumerate(results, 1):
                doc_id = result.get('_id', result.get('id', ''))
                if doc_id:
                    scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        return scores

    @staticmethod
    def weighted_sum_fusion(
            vector_scores: Dict[str, float],
            keyword_scores: Dict[str, float],
            vector_weight: float,
            keyword_weight: float
    ) -> Dict[str, float]:
        """
        加权和融合

        Args:
            vector_scores: 向量检索分数（已归一化）
            keyword_scores: 关键词检索分数（已归一化）
            vector_weight: 向量权重
            keyword_weight: 关键词权重

        Returns:
            文档ID到融合分数的映射
        """
        all_docs = set(vector_scores.keys()) | set(keyword_scores.keys())
        fused = {}

        for doc_id in all_docs:
            v_score = vector_scores.get(doc_id, 0.0)
            k_score = keyword_scores.get(doc_id, 0.0)

            if v_score > 0.05 and k_score > 0.05:
                fused[doc_id] = v_score * vector_weight + k_score * keyword_weight
            elif v_score > 0.05:
                fused[doc_id] = v_score * 0.8
            elif k_score > 0.05:
                fused[doc_id] = k_score * 0.8
            else:
                fused[doc_id] = max(v_score, k_score) * 0.5

            fused[doc_id] = max(0.05, min(0.95, fused[doc_id]))

        return fused

    @staticmethod
    def normalize_results(
            vector_results: List[Dict],
            keyword_results: List[Dict],
            vector_weight: float,
            keyword_weight: float
    ) -> List[Dict]:
        """
        归一化并融合两个检索结果列表

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            vector_weight: 向量权重
            keyword_weight: 关键词权重

        Returns:
            融合后的结果列表（已排序）
        """
        from .base import BaseRetriever

        if not vector_results and not keyword_results:
            return []
        if not vector_results:
            return keyword_results
        if not keyword_results:
            return vector_results

        # 归一化向量分数
        vector_scores_raw = [r.get('_score', 0) for r in vector_results if r.get('_score', 0) > 0]
        norm_vector_scores = BaseRetriever.normalize_scores(vector_scores_raw)

        # 创建文档ID到归一化分数的映射
        vec_score_map = {}
        for i, result in enumerate([r for r in vector_results if r.get('_score', 0) > 0]):
            doc_id = result.get('_id', result.get('id', ''))
            if doc_id:
                vec_score_map[doc_id] = norm_vector_scores[i] if i < len(norm_vector_scores) else 0.05

        # 处理关键词结果
        kw_score_map = {}
        for result in keyword_results:
            doc_id = result.get('_id', result.get('id', ''))
            if not doc_id:
                continue

            raw_score = result.get('keyword_score', result.get('_score', 0))
            # 简单归一化：假设 BM25 分数范围 0-10
            kw_score = min(0.95, raw_score / 10.0) if raw_score > 0 else 0.05
            kw_score = max(0.05, kw_score)
            kw_score_map[doc_id] = kw_score

        # 融合分数
        fused_scores = ScoreMerger.weighted_sum_fusion(
            vec_score_map, kw_score_map, vector_weight, keyword_weight
        )

        # 构建结果
        result_map = {}
        for doc in vector_results:
            doc_id = doc.get('_id', doc.get('id', ''))
            if doc_id and doc_id in fused_scores:
                result_map[doc_id] = {
                    **doc,
                    'vector_score': vec_score_map.get(doc_id, 0),
                    'keyword_score': kw_score_map.get(doc_id, 0),
                    'final_score': fused_scores[doc_id],
                    '_search_types': ['vector']
                }

        for doc in keyword_results:
            doc_id = doc.get('_id', doc.get('id', ''))
            if doc_id:
                if doc_id in result_map:
                    result_map[doc_id]['keyword_score'] = kw_score_map.get(doc_id, 0)
                    result_map[doc_id]['_search_types'].append('bm25')
                elif doc_id in fused_scores:
                    result_map[doc_id] = {
                        **doc,
                        'vector_score': 0,
                        'keyword_score': kw_score_map.get(doc_id, 0),
                        'final_score': fused_scores[doc_id],
                        '_search_types': ['bm25']
                    }

        results = list(result_map.values())
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return results
