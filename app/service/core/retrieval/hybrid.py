# app/service/core/retrieval/hybrid.py
"""
混合检索核心模块 - 整合基类、混合检索和父子检索
"""

import os
import json
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from app.service.core.vector_store import get_vector_store, get_vector_search_service
from app.service.core.embedding import get_embedding_service
from app.service.core.cache import get_cache_manager
from .rewriter import QueryRewriter
from .reranker import Reranker

logger = logging.getLogger(__name__)


# ============================================================
# 基类工具函数
# ============================================================

def normalize_scores(scores: List[float], min_val: float = 0.05, max_val: float = 0.95) -> List[float]:
    """归一化分数

    输出与输入等长、下标一一对应（调用方会按下标对齐结果）。
    非正值不参与确定 min/max 区间，但仍占位，统一落到 min_val；
    全为非正或全部相等时，整体给同一个中位/下限分。
    """
    if not scores:
        return []
    n = len(scores)
    positive = [s for s in scores if s > 0]
    if not positive:
        return [min_val] * n
    s_min, s_max = min(positive), max(positive)
    if s_max == s_min:
        return [(min_val + max_val) / 2] * n
    return [
        min_val + (s - s_min) / (s_max - s_min) * (max_val - min_val) if s > 0 else min_val
        for s in scores
    ]


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
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_manager or get_embedding_service()
        self.es_bm25 = None
        self.use_es_bm25 = self._check_es_available()

        self._rrf_k = int(os.getenv("RRF_K", "60"))
        self._load_weights()

    def _check_es_available(self) -> bool:
        """检查 ES 是否可用"""
        try:
            # 保留函数内导入：elasticsearch 缺失时需在此降级，不能提到模块顶部
            from .es_bm25 import get_es_bm25_retriever
            self.es_bm25 = get_es_bm25_retriever()
            return self.es_bm25.is_available()
        except Exception as e:
            logger.warning(f"ES BM25 不可用: {e}")
            return False

    # 报告 3.1：视觉意图 query 的识别词。只认明确的图片词，不用单字「图」，
    # 避免「图书馆」「试图」之类误判；可用 HYBRID_IMAGE_INTENT_KEYWORDS 逗号分隔覆盖。
    _DEFAULT_IMAGE_INTENT_KEYWORDS = (
        "图片", "图像", "图表", "截图", "示意图", "流程图", "架构图",
        "柱状图", "饼图", "散点图", "折线图", "曲线图", "混淆矩阵",
        "配图", "插图", "如图所示", "看图", "图中",
    )

    def _load_weights(self):
        self.vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
        self.keyword_weight = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4"))
        # 报告 3.1：视觉意图 query 的条件权重。实测视觉 query 用 0.5:1.0 时 MRR 0.7→0.8，
        # 只对命中图片意图的 query 生效，不改全局 0.6:0.4，文本组配比不受影响。
        self.image_vector_weight = float(os.getenv("HYBRID_IMAGE_VECTOR_WEIGHT", "0.5"))
        self.image_keyword_weight = float(os.getenv("HYBRID_IMAGE_KEYWORD_WEIGHT", "1.0"))
        # 报告 3.1 第 3 条：RRF 之后对 chunk_type=image 的小幅位置加权。
        # 默认 0（关闭），须与条件权重一起评测后再打开，避免把文本组 MRR 拉下来。
        self.image_boost = float(os.getenv("HYBRID_IMAGE_BOOST", "0"))
        raw_kw = os.getenv("HYBRID_IMAGE_INTENT_KEYWORDS", "")
        kws = raw_kw.split(",") if raw_kw else self._DEFAULT_IMAGE_INTENT_KEYWORDS
        self.image_intent_keywords = tuple(k.strip() for k in kws if k and k.strip())
        # 报告 3.2：余弦门限原来写死 0.1，不同 embedding 模型余弦分布差别很大
        # （VISTA 与旧 CLIP 不是一个量纲），提为按模型标定的配置项。
        self.vector_sim_threshold = float(os.getenv("HYBRID_VECTOR_SIM_THRESHOLD", "0.1"))
        # 大规模评测第 13 条：RRF 之后按 docnm 限流。同主题 40 篇干扰 vs 1 篇锚点时，
        # RRF 会放大「多者通吃」，同一文档的多块、同主题多文档挤占 top_k 名额。
        # 默认 0 = 关闭，保持既有行为；>0 时同一 docnm 最多保留 N 条候选。
        # 注：2026-10-31 大规模评测实测该项对未命中 query 零增益（top5 本就是 5 篇
        # 不同 docnm），根因不在此，仅作为可选的去冗余开关保留。
        self.per_doc_cap = int(os.getenv("HYBRID_PER_DOC_CAP", "0"))
        # 大规模评测第 13 条（真因）：RRF 奖励「两路都在榜中游」、惩罚「单路极强、
        # 另一路缺席」。实测锚点 BM25 第 1 / 向量第 50 时 RRF 后掉到第 27。
        # 打开后把两路各自 Top-1 保送到结果最前面，默认开启；设为 0 可关闭。
        self.single_route_top1 = int(os.getenv("HYBRID_SINGLE_ROUTE_TOP1", "1")) > 0

    def _is_image_intent(self, query: str) -> bool:
        """判断 query 是否在问图片内容（报告 3.1 条件权重的触发条件）。"""
        if not query or not self.image_intent_keywords:
            return False
        q = query.lower()
        return any(kw.lower() in q for kw in self.image_intent_keywords)

    def _vector_search(self, query: str, index_name: str, top_k: int,
                       user_level: str = None) -> List[Dict]:
        """向量检索"""
        query_vector = self.embedding_service.generate_embedding(query)
        if not query_vector:
            return []

        # 报告 3.2：门限由 HYBRID_VECTOR_SIM_THRESHOLD 按 embedding 模型标定，
        # 不再写死 0.1（VISTA 与旧 CLIP 的余弦分布不是一个量纲）。
        results = self.vector_store.search(
            query_vector=query_vector, index_name=index_name, top_k=top_k,
            similarity_threshold=self.vector_sim_threshold, user_level=user_level
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

    def _parallel_recall(self, query: str, index_name: str, recall_k: int,
                         user_level: str = None) -> Tuple[List[Dict], List[Dict]]:
        """报告 2.2：向量检索与 BM25 并行执行。

        两路互不依赖，并行后端到端延迟由「两者之和」降到「两者较大者」。
        任一路失败只降级该路为空，不影响另一路（与原先串行时的行为一致）。
        线程数固定为 2，按需创建，不常驻。
        """
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="recall") as pool:
            vec_future = pool.submit(
                self._vector_search, query, index_name, recall_k, user_level
            )
            kw_future = pool.submit(
                self._bm25_search, query, index_name, recall_k, user_level
            )

            def _safe(future, name: str) -> List[Dict]:
                try:
                    return future.result() or []
                except Exception as e:
                    logger.warning("%s 检索失败，已降级为空: %s", name, e)
                    return []

            return _safe(vec_future, "向量"), _safe(kw_future, "BM25")

    def hybrid_search(self, query: str, index_name: str, top_k: int = 5,
                      recall_k: int = 20, user_level: str = None) -> List[Dict]:
        """混合检索"""
        recall_k = max(top_k * 4, recall_k)
        # 报告 2.2：向量检索与 BM25 原本串行，两路互不依赖，改为并行执行，
        # 端到端延迟从「两者之和」降到「两者较大者」。
        vector_results, keyword_results = self._parallel_recall(
            query, index_name, recall_k, user_level
        )

        if not vector_results and not keyword_results:
            return []

        # 报告 3.1 第 1 条：视觉意图 query 改用条件权重（默认 0.5:1.0，实测视觉 MRR 0.7→0.8），
        # 非图片意图 query 仍用全局 0.6:0.4，文本组配比不受影响。
        if self._is_image_intent(query):
            vector_weight, keyword_weight = self.image_vector_weight, self.image_keyword_weight
        else:
            vector_weight, keyword_weight = self.vector_weight, self.keyword_weight

        results = ScoreMerger.rrf_normalize_results(
            vector_results, keyword_results, vector_weight, keyword_weight, self._rrf_k
        )
        for r in results:
            r['final_score'] = r.get('rrf_score', 0)

        # 报告 3.1 第 3 条：RRF 之后对 chunk_type=image 的块做小幅位置加权。
        # 默认 HYBRID_IMAGE_BOOST=0（关闭），须与条件权重一起评测后再打开，
        # 避免把文本组 MRR 拉下来。
        if self.image_boost > 0:
            for r in results:
                if str(r.get('chunk_type', '') or '') == 'image':
                    r['rrf_score'] = r.get('rrf_score', 0) * (1.0 + self.image_boost)
                    r['final_score'] = r['rrf_score']
            results.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)

        # RRF 单路分数上限约 weight/(k+1)（k=100 时约 0.0099、单通道 0.6/61≈0.0098），
        # 固定 0.01 门限会在 k 偏大或 BM25 降级时把整批结果截成空。默认不再做分数过滤，
        # 只过滤掉 rrf_score 为 0 的无效项；需要门限时用 RRF_MIN_SCORE 显式配置。
        min_score = float(os.getenv("RRF_MIN_SCORE", "0"))
        filtered = [r for r in results
                    if r.get('rrf_score', 0) > 0 and r.get('rrf_score', 0) >= min_score]

        # 大规模评测第 13 条：RRF 之后按 docnm 限流，抑制「同主题多者通吃」。
        # 限流作用在完整候选集（两路各 recall_k 条合并后）上，再截断 top_k ——
        # 先截断再限流会让名额空着，锚点仍然提不上来。docnm 为空的块不参与合并。
        if self.per_doc_cap > 0:
            kept, seen = [], {}
            for r in filtered:
                key = str(r.get('docnm') or r.get('document_name') or '').strip()
                if not key:
                    kept.append(r)
                    continue
                n = seen.get(key, 0)
                if n >= self.per_doc_cap:
                    continue
                seen[key] = n + 1
                kept.append(r)
            filtered = kept

        # 大规模评测第 13 条：单路 Top-1 保送。
        # RRF 的名次倒数对「单路第一、另一路缺席」极不友好：向量第 50 + BM25 第 1
        # 的锚点只有 ~1/(60+1) 分，而两路都在第 20 名的干扰文档有 ~2/(60+20) 分。
        # 这里把两路各自的 Top-1 无条件和进候选最前面（按 RRF 分高者在前），再截断 top_k。
        if self.single_route_top1 and filtered:
            by_id = {}
            for r in filtered:
                rid = r.get('_id') or r.get('id')
                if rid:
                    by_id[rid] = r
            promoted = []
            for src in (vector_results, keyword_results):
                if not src:
                    continue
                rid = src[0].get('_id') or src[0].get('id')
                if rid and rid in by_id and by_id[rid] not in promoted:
                    promoted.append(by_id[rid])
            if promoted:
                promoted.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
                head = set(id(r) for r in promoted)
                filtered = promoted + [r for r in filtered if id(r) not in head]

        return filtered[:top_k]


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