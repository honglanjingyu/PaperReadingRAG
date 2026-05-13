# app/service/core/retrieval/base.py

import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaseRetriever:
    """检索器基类 - 提供分数归一化、融合等公共方法"""

    @staticmethod
    def normalize_scores(scores: List[float], min_val: float = 0.05, max_val: float = 0.95) -> List[float]:
        """归一化分数到 [min_val, max_val] 范围"""
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
    主要使用 RRF (Reciprocal Rank Fusion)
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
            k: RRF 常数，通常为 60

        Returns:
            文档ID到融合分数的映射
        """
        scores = defaultdict(float)

        for results in results_lists:
            for rank, result in enumerate(results, 1):
                doc_id = result.get('_id', result.get('id', ''))
                if doc_id:
                    scores[doc_id] += 1.0 / (k + rank)

        return dict(scores)

    @staticmethod
    def reciprocal_rank_fusion_with_weights(
            results_lists: List[List[Dict]],
            weights: List[float],
            k: int = 60
    ) -> Dict[str, float]:
        """
        带权重的 RRF 融合算法

        Args:
            results_lists: 多个检索结果列表
            weights: 对应每个结果列表的权重
            k: RRF 常数

        Returns:
            文档ID到融合分数的映射
        """
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
    def weighted_sum_fusion(
            vector_scores: Dict[str, float],
            keyword_scores: Dict[str, float],
            vector_weight: float,
            keyword_weight: float
    ) -> Dict[str, float]:
        """
        加权和融合（备用方案）

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
    def rrf_normalize_results(
            vector_results: List[Dict],
            keyword_results: List[Dict],
            vector_weight: float = 1.0,
            keyword_weight: float = 1.0,
            k: int = 60
    ) -> List[Dict]:
        """
        使用 RRF 算法归一化并融合两个检索结果列表

        Args:
            vector_results: 向量检索结果（按分数降序排列）
            keyword_results: 关键词检索结果（按分数降序排列）
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            k: RRF 常数

        Returns:
            融合后的结果列表（已排序）
        """
        # 构建 RRF 分数
        rrf_scores = defaultdict(float)

        # 向量检索结果按原始顺序添加排名
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.get('_id', result.get('id', ''))
            if doc_id:
                rrf_scores[doc_id] += vector_weight / (k + rank)

        # 关键词检索结果按原始顺序添加排名
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.get('_id', result.get('id', ''))
            if doc_id:
                rrf_scores[doc_id] += keyword_weight / (k + rank)

        if not rrf_scores:
            return []

        # 构建结果映射
        result_map = {}

        # 处理向量结果
        for doc in vector_results:
            doc_id = doc.get('_id', doc.get('id', ''))
            if doc_id and doc_id in rrf_scores:
                result_map[doc_id] = {
                    **doc,
                    'vector_score': doc.get('_score', 0),
                    'keyword_score': 0,
                    'rrf_score': rrf_scores[doc_id],
                    '_search_types': ['vector']
                }

        # 处理关键词结果
        for doc in keyword_results:
            doc_id = doc.get('_id', doc.get('id', ''))
            if doc_id:
                if doc_id in result_map:
                    result_map[doc_id]['keyword_score'] = doc.get('_score', 0)
                    result_map[doc_id]['_search_types'].append('bm25')
                    # 更新 RRF 分数（已包含在之前计算中）
                    result_map[doc_id]['rrf_score'] = rrf_scores[doc_id]
                elif doc_id in rrf_scores:
                    result_map[doc_id] = {
                        **doc,
                        'vector_score': 0,
                        'keyword_score': doc.get('_score', 0),
                        'rrf_score': rrf_scores[doc_id],
                        '_search_types': ['bm25']
                    }

        # 转换为列表并按 RRF 分数排序
        results = list(result_map.values())
        results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)

        logger.debug(f"RRF 融合完成: 向量结果数={len(vector_results)}, "
                     f"关键词结果数={len(keyword_results)}, "
                     f"融合后={len(results)}")

        return results