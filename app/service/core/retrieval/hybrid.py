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
        # 方案 Q2：术语归一器懒加载后缓存在这里（词表只读一次）
        self._rewriter = None

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

    def _term_expansions(self, query: str) -> List[str]:
        """术语归一（方案 Q2）：返回要追加给 BM25 的扩展词。

        词表在 QueryRewriter 里懒加载并缓存（term_map.json / TERM_MAP_PATH），
        任何异常都降级为空列表，不影响检索主链路。
        """
        try:
            if getattr(self, "_rewriter", None) is None:
                self._rewriter = QueryRewriter()
            return self._rewriter.expand_terms(query)
        except Exception as e:
            logger.warning("术语归一失败，本次不做扩展: %s: %s", type(e).__name__, e)
            return []

    def _bm25_search(self, query: str, index_name: str, top_k: int,
                     user_level: str = None,
                     expand_terms: List[str] = None) -> List[Dict]:
        """BM25 检索

        expand_terms：术语归一扩展词（方案 Q2），只在本路追加成 should 子句，
        向量路不接收、也不会用到这些词。
        """
        if not self.use_es_bm25 or not self.es_bm25:
            return []

        results = self.es_bm25.search(
            query=query, index_name=index_name, top_k=top_k,
            min_score=0.05, user_level=user_level,
            expand_terms=expand_terms
        )
        for r in results:
            r['_search_type'] = 'bm25'
            r['keyword_score'] = r.get('_score', 0)
            if 'docnm' not in r or not r['docnm']:
                r['docnm'] = r.get('document_name', '')
        return results

    def _parallel_recall(self, query: str, index_name: str, recall_k: int,
                         user_level: str = None,
                         expand_terms: List[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """报告 2.2：向量检索与 BM25 并行执行。

        两路互不依赖，并行后端到端延迟由「两者之和」降到「两者较大者」。
        任一路失败只降级该路为空，不影响另一路（与原先串行时的行为一致）。
        线程数固定为 2，按需创建，不常驻。
        """
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="recall") as pool:
            vec_future = pool.submit(
                self._vector_search, query, index_name, recall_k, user_level
            )
            # 方案 Q2：术语扩展词只喂 BM25，向量路始终用原 query。
            kw_future = pool.submit(
                self._bm25_search, query, index_name, recall_k, user_level, expand_terms
            )

            def _safe(future, name: str) -> List[Dict]:
                try:
                    return future.result() or []
                except Exception as e:
                    logger.warning("%s 检索失败，已降级为空: %s", name, e)
                    return []

            return _safe(vec_future, "向量"), _safe(kw_future, "BM25")

    def hybrid_search(self, query: str, index_name: str, top_k: int = 5,
                      recall_k: int = 20, user_level: str = None,
                      term_expand: bool = None) -> List[Dict]:
        """混合检索

        term_expand：是否做术语归一（方案 Q2）。None 时读环境变量
        ENABLE_TERM_EXPAND（默认 true）。扩展词**只喂 BM25**，向量路始终用原 query。
        """
        recall_k = max(top_k * 4, recall_k)
        if term_expand is None:
            term_expand = os.getenv("ENABLE_TERM_EXPAND", "true").lower() == "true"
        expand_terms = self._term_expansions(query) if term_expand else []
        # 报告 2.2：向量检索与 BM25 原本串行，两路互不依赖，改为并行执行，
        # 端到端延迟从「两者之和」降到「两者较大者」。
        vector_results, keyword_results = self._parallel_recall(
            query, index_name, recall_k, user_level, expand_terms
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
                    'is_parent': False,
                    # R4：图片块的语义主要在 caption / image_ref，聚合时必须保留，
                    # 否则 rerank 拿到空 content 会把整块丢掉（视觉组掉块）。
                    'chunk_type': result.get('chunk_type', '') or '',
                    'image_ref': result.get('image_ref', '') or '',
                    'caption': result.get('caption', '') or '',
                    'content_with_weight': result.get('content_with_weight', '') or '',
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
                        'is_parent': True,
                        'chunk_type': '',
                        'image_ref': '',
                        'caption': '',
                        'content_with_weight': '',
                    }
                parent_map[parent_id]['child_count'] += 1
                parent_map[parent_id]['child_scores'].append(result.get('_score', 0))  # 现在可以安全访问
                parent_map[parent_id]['score'] = max(parent_map[parent_id]['score'], result.get('_score', 0))
                # R4：父块聚合默认取父正文，但只要组内出现图片块，就要把视觉字段
                # 并到父块上，否则后面的 rerank / 原图回填都看不到这张图。
                if str(result.get('chunk_type', '') or '') == 'image':
                    parent_map[parent_id]['chunk_type'] = 'image'
                    if result.get('image_ref'):
                        parent_map[parent_id]['image_ref'] = result.get('image_ref')
                    if result.get('caption'):
                        parent_map[parent_id]['caption'] = result.get('caption')
                if not parent_map[parent_id].get('content_with_weight') and result.get('content_with_weight'):
                    parent_map[parent_id]['content_with_weight'] = result.get('content_with_weight')

        results = []
        for d in parent_map.values():
            results.append({
                'parent_id': d['parent_id'],
                'content': d['content'],
                'score': d['score'],
                'child_count': d['child_count'],
                'is_parent': d.get('is_parent', True),
                'docnm': d.get('docnm', ''),
                'document_name': d.get('document_name', ''),
                # R4：视觉字段必须带出聚合层，否则下游 rerank 的入参、原图回填
                # 都拿不到 chunk_type / image_ref（之前这里只挑 7 个字段，全丢了）。
                'chunk_type': d.get('chunk_type', '') or '',
                'image_ref': d.get('image_ref', '') or '',
                'caption': d.get('caption', '') or '',
                'content_with_weight': d.get('content_with_weight', '') or '',
            })

        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:top_k]

    def search_parents(self, query: str, index_name: str, top_k: int = 5,
                       recall_k: int = 20, user_level: str = None,
                       term_expand: bool = None) -> Tuple[List[Dict], List[Dict]]:
        """父子检索主接口

        term_expand：是否做术语归一（方案 Q2，只作用于 BM25 路）。
        """
        child_results = self.hybrid_retriever.hybrid_search(
            query, index_name, top_k, recall_k, user_level, term_expand
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
        """带查询改写的父子检索

        Q1（方案第 3 节）：原 `strategy='synonym'` 产出的 `(A OR B)` 字面串已停用。
        它对论文语料零命中，且会污染向量路；现在**改写不再产出新字符串**，
        query 原样用于向量与 BM25 的 must。

        Q2：术语归一改为「扩展词追加为 BM25 的 should 子句」，在
        hybrid_search -> _bm25_search 内部完成，向量路不受影响。
        """
        # 原 query 就是检索 query：不再做字面串改写（Q1）
        rewritten_query = question
        term_expansions: List[str] = []
        if enable_query_rewrite:
            try:
                # 仅用于观测/回传，真正的扩展在 hybrid_search 内部喂给 BM25
                term_expansions = self.hybrid_retriever._term_expansions(question)
            except Exception as e:
                logger.warning(f"术语归一失败: {type(e).__name__}: {e}")

        # 执行父子检索
        # 术语归一开关沿用 enable_query_rewrite：Q1 之后它只控制「BM25 是否做术语扩展」，
        # 不再控制任何字面串改写（向量路永远用原 query）。
        parent_results, child_results = self.search_parents(
            query=rewritten_query,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            user_level=user_level,
            term_expand=enable_query_rewrite
        )

        if not parent_results:
            return {
                "success": False,
                "error": "未找到相关文档",
                "question": question,
                "results": []
            }

        # 重排序（每次调用都记录是否真正生效，避免静默降级）
        final_results = parent_results
        rerank_meta = {
            "rerank_applied": False,
            "rerank_reason": "disabled" if not enable_rerank else "no_candidates",
            "rerank_model": None,
            "rerank_input_count": len(parent_results),
        }
        if enable_rerank and parent_results:
            reranker = None
            try:
                reranker = Reranker(api_type=rerank_type)
                docs_for_rerank = []
                for r in parent_results:
                    # R4：图片块 / 只有 caption 的块 content 可能为空，
                    # reranker 会按空串过滤掉它们（等于视觉命中被丢弃）。
                    # 这里统一给一份「可被 rerank 打分」的文本：
                    # content > content_with_weight > caption > docnm。
                    text = (r.get('content') or r.get('content_with_weight')
                            or r.get('caption') or '').strip()
                    if not text:
                        text = r.get('caption', '') or r.get('docnm', '') or ''
                    docs_for_rerank.append({
                        'content': text,
                        'content_with_weight': text,
                        'parent_id': r.get('parent_id', ''),
                        'child_count': r.get('child_count', 0),
                        'original_score': r.get('score', 0),
                        'docnm': r.get('docnm', ''),
                        'document_name': r.get('document_name', ''),
                        # 视觉字段必须跟着 rerank 结果回来，后面原图回填要用
                        'chunk_type': r.get('chunk_type', '') or '',
                        'image_ref': r.get('image_ref', '') or '',
                        'caption': r.get('caption', '') or '',
                    })

                reranked = reranker.rerank(
                    query=rewritten_query,
                    documents=docs_for_rerank,
                    top_k=top_k
                )

                if reranked:
                    final_results = reranked[:top_k]
            except Exception as e:
                logger.warning(f"重排序失败: {type(e).__name__}: {e}")

            meta = getattr(reranker, 'last_meta', None) if reranker is not None else None
            if meta:
                rerank_meta.update(meta)
            elif reranker is None:
                rerank_meta["rerank_reason"] = "init_error"
            else:
                rerank_meta["rerank_reason"] = "no_meta"

        if not rerank_meta["rerank_applied"]:
            logger.info(
                f"rerank 未生效: reason={rerank_meta.get('rerank_reason')} "
                f"候选={rerank_meta.get('rerank_input_count')} enable_rerank={enable_rerank}"
            )

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(final_results, 1):
            doc_name = result.get('document_name') or result.get('docnm') or ''
            item = {
                "rank": i,
                "score": result.get("rerank_score", result.get("score", 0)),
                "content": result.get("content", ""),
                "parent_id": result.get("parent_id", ""),
                "child_count": result.get("child_count", 0),
                "document_name": doc_name,
                "docnm": doc_name,
                "is_parent": result.get("is_parent", True),
                # R4：视觉字段透传到最终结果，供上游原图回填 / 前端展示图片块
                "chunk_type": result.get("chunk_type", "") or "",
                "image_ref": result.get("image_ref", "") or "",
                "caption": result.get("caption", "") or "",
                "content_with_weight": result.get("content_with_weight", "") or "",
            }
            # rerank 生效时带上名次变化，便于判断 rerank 到底有没有贡献
            if "original_rank" in result:
                item["original_rank"] = result.get("original_rank")
            if "new_rank" in result:
                item["new_rank"] = result.get("new_rank")
            formatted_results.append(item)

        return {
            "success": True,
            "question": question,
            # Q1 之后 query 不再被改写，原样回传便于前端/日志对照
            "rewritten_query": rewritten_query,
            # Q2 术语归一的观测字段：命中的扩展词（只喂 BM25 should 子句）
            "term_expansions": term_expansions,
            "term_expand_applied": bool(term_expansions),
            "top_k": top_k,
            "recall_k": recall_k,
            "total_recalled": len(child_results),
            "total_returned": len(formatted_results),
            "results": formatted_results,
            "enable_rerank": enable_rerank,
            "enable_query_rewrite": enable_query_rewrite,
            # rerank 可观测性：是否真正生效、未生效原因、模型名
            "rerank_applied": rerank_meta.get("rerank_applied", False),
            "rerank_reason": rerank_meta.get("rerank_reason", "unknown"),
            "rerank_model": rerank_meta.get("rerank_model") or (None if not enable_rerank else "unknown"),
            # 远程 rerank 调用耗时（毫秒）与名次变化明细（判断 rerank 有没有贡献的直接证据）
            "rerank_ms": rerank_meta.get("rerank_ms"),
            "rank_changed_count": rerank_meta.get("rank_changed_count", 0),
            "rank_changes": rerank_meta.get("rank_changes", []),
            "rerank_meta": rerank_meta
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