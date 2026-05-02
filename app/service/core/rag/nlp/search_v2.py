# app/service/core/rag/nlp/search_v2.py
"""
检索模块 - 向量检索和混合检索
支持向量相似度搜索、关键词搜索、混合检索和重排序
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 导入项目中已有的模块
from service.core.rag.nlp import rag_tokenizer
from service.core.rag.nlp.model import generate_embedding

logger = logging.getLogger(__name__)


def index_name(uid: str) -> str:
    """生成索引名称"""
    return f"{uid}"


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    total: int
    ids: list[str]
    query_vector: list[float] | None = None
    field: dict | None = None
    highlight: dict | None = None
    aggregation: list | dict | None = None
    keywords: list[str] | None = None
    group_docs: list[list] | None = None


class MatchDenseExpr:
    """向量匹配表达式"""
    def __init__(self, vector_column_name: str, embedding_data: list,
                 embedding_data_type: str, distance_type: str,
                 topn: int, extra_options: dict = None):
        self.vector_column_name = vector_column_name
        self.embedding_data = embedding_data
        self.embedding_data_type = embedding_data_type
        self.distance_type = distance_type
        self.topn = topn
        self.extra_options = extra_options or {}


class MatchTextExpr:
    """文本匹配表达式"""
    def __init__(self, fields: list, matching_text: str, topn: int, extra_options: dict = None):
        self.fields = fields
        self.matching_text = matching_text
        self.topn = topn
        self.extra_options = extra_options or {}


class FusionExpr:
    """融合表达式（用于混合检索）"""
    def __init__(self, method: str, topn: int, fusion_params: dict = None):
        self.method = method
        self.topn = topn
        self.fusion_params = fusion_params or {}


class OrderByExpr:
    """排序表达式"""
    def __init__(self):
        self.fields = []

    def asc(self, field: str):
        self.fields.append((field, 0))
        return self

    def desc(self, field: str):
        self.fields.append((field, 1))
        return self


class Dealer:
    """
    检索处理器
    提供向量检索、混合检索、重排序等功能
    """

    def __init__(self, dataStore):
        """
        初始化检索处理器

        Args:
            dataStore: 数据存储连接（如 ESConnection）
        """
        self.dataStore = dataStore
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    def get_vector(self, txt: str, emb_mdl=None, topk: int = 10, similarity: float = 0.1) -> MatchDenseExpr:
        """
        将文本转换为向量匹配表达式

        Args:
            txt: 查询文本
            emb_mdl: 嵌入模型（未使用，保留兼容性）
            topk: 返回数量
            similarity: 相似度阈值

        Returns:
            MatchDenseExpr: 向量匹配表达式
        """
        qv = generate_embedding(txt)
        if qv is None:
            raise Exception(f"无法生成查询向量: {txt[:50]}...")

        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(f"返回的数组形状 {shape} 不符合预期（应为1维）")

        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"

        return MatchDenseExpr(
            vector_column_name,
            embedding_data,
            'float',
            'cosine',
            topk,
            {"similarity": similarity}
        )

    def get_filters(self, req: dict) -> dict:
        """获取过滤条件"""
        condition = {}
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]

        for key in ["available_int", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]

        return condition

    def search(self, req: dict, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight: bool = False,
               rank_feature: dict | None = None) -> SearchResult:
        """
        执行搜索

        Args:
            req: 请求参数，包含 question, page, size, topk 等
            idx_names: 索引名称列表
            kb_ids: 知识库ID列表
            emb_mdl: 嵌入模型（未使用）
            highlight: 是否高亮
            rank_feature: 排序特征

        Returns:
            SearchResult: 搜索结果
        """
        filters = self.get_filters(req)
        order_by = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get("fields", [
            "docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks",
            "important_kwd", "position_int", "doc_id", "page_num_int",
            "top_int", "create_timestamp_flt", "question_kwd", "question_tks",
            "available_int", "content_with_weight"
        ])

        kwds = set([])
        qst = req.get("question", "")
        q_vec = []

        if not qst:
            # 无查询时直接返回
            if req.get("sort"):
                order_by.asc("page_num_int")
                order_by.asc("top_int")
                order_by.desc("create_timestamp_flt")

            res = self.dataStore.search(src, [], filters, [], order_by, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logger.debug(f"搜索完成，总计: {total}")
        else:
            # 向量检索
            highlight_fields = ["content_ltks", "title_tks"] if highlight else []
            match_dense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
            q_vec = match_dense.embedding_data

            # 添加向量字段到源字段
            src.append(f"q_{len(q_vec)}_vec")

            # 混合检索：向量 + 融合
            fusion_expr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
            match_exprs = [match_dense, fusion_expr]

            res = self.dataStore.search(
                src, highlight_fields, filters, match_exprs,
                order_by, offset, limit, idx_names, kb_ids,
                rank_feature=rank_feature
            )
            total = self.dataStore.getTotal(res)
            logger.debug(f"搜索完成，总计: {total}")

            # 如果结果为空，降低阈值重试
            if total == 0:
                logger.debug("结果为空，降低阈值重试...")
                match_dense.extra_options["similarity"] = 0.17
                res = self.dataStore.search(
                    src, highlight_fields, filters, [match_dense, fusion_expr],
                    order_by, offset, limit, idx_names, kb_ids,
                    rank_feature=rank_feature
                )
                total = self.dataStore.getTotal(res)

        ids = self.dataStore.getChunkIds(res)
        highlight_result = self.dataStore.getHighlight(res, kwds, "content_with_weight") if highlight else {}
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")

        return SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec if q_vec else None,
            aggregation=aggs,
            highlight=highlight_result,
            field=self.dataStore.getFields(res, src),
            keywords=list(kwds) if kwds else None
        )

    def token_similarity(self, query_tokens: List[str], doc_tokens_list: List[List[str]]) -> List[float]:
        """
        计算查询词与文档词列表的相似度

        Args:
            query_tokens: 查询词列表
            doc_tokens_list: 文档词列表

        Returns:
            List[float]: 相似度分数列表
        """
        def to_dict(tks):
            d = {}
            for t in tks:
                if t not in d:
                    d[t] = 0
                d[t] += 1
            return d

        query_dict = to_dict(query_tokens)
        doc_dicts = [to_dict(tks) for tks in doc_tokens_list]

        similarities = []
        for doc_dict in doc_dicts:
            s = 1e-9
            q = 1e-9
            for k, v in query_dict.items():
                if k in doc_dict:
                    s += v
                q += v
            similarities.append(s / q)

        return similarities

    def hybrid_similarity(self, query_vector: List[float], doc_vectors: List[List[float]],
                          query_tokens: List[str], doc_tokens_list: List[List[str]],
                          tkweight: float = 0.3, vtweight: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        混合相似度计算（向量相似度 + 关键词相似度）

        Args:
            query_vector: 查询向量
            doc_vectors: 文档向量列表
            query_tokens: 查询词列表
            doc_tokens_list: 文档词列表
            tkweight: 关键词权重
            vtweight: 向量权重

        Returns:
            Tuple: (综合相似度, 关键词相似度, 向量相似度)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # 计算向量相似度
        if doc_vectors and query_vector:
            query_vec = np.array(query_vector).reshape(1, -1)
            doc_vecs = np.array(doc_vectors)
            vector_sims = cosine_similarity(query_vec, doc_vecs)[0]
        else:
            vector_sims = np.zeros(len(doc_vectors)) if doc_vectors else np.array([])

        # 计算关键词相似度
        token_sims = np.array(self.token_similarity(query_tokens, doc_tokens_list))

        # 加权融合
        if len(vector_sims) > 0:
            combined_sims = tkweight * token_sims + vtweight * vector_sims
        else:
            combined_sims = token_sims

        return combined_sims, token_sims, vector_sims

    def rerank(self, sres: SearchResult, query: str,
               tkweight: float = 0.3, vtweight: float = 0.7,
               cfield: str = "content_ltks",
               rank_feature: dict | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        重排序搜索结果

        Args:
            sres: 搜索结果
            query: 查询文本
            tkweight: 关键词权重
            vtweight: 向量权重
            cfield: 内容字段名
            rank_feature: 排序特征

        Returns:
            Tuple: (综合分数, 关键词分数, 向量分数)
        """
        # 提取关键词
        query_tokens = rag_tokenizer.tokenize(query).split()

        # 准备文档向量和词列表
        vector_size = len(sres.query_vector) if sres.query_vector else 0
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size if vector_size > 0 else []

        doc_vectors = []
        doc_tokens = []

        for chunk_id in sres.ids:
            # 获取向量
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [float(v) for v in vector.split("\t")]
            doc_vectors.append(vector)

            # 获取分词
            content = sres.field[chunk_id].get(cfield, "")
            title = sres.field[chunk_id].get("title_tks", "")
            important = sres.field[chunk_id].get("important_kwd", [])

            if isinstance(important, str):
                important = [important]

            # 合并所有分词
            tokens = content.split() + title.split() * 2 + important * 5
            doc_tokens.append(tokens)

        # 计算混合相似度
        sims, tsims, vsims = self.hybrid_similarity(
            sres.query_vector or [],
            doc_vectors,
            query_tokens,
            doc_tokens,
            tkweight,
            vtweight
        )

        return sims, tsims, vsims

    def retrieval(self, question: str, embd_mdl, tenant_ids: str | list[str],
                  kb_ids: list[str], page: int, page_size: int,
                  similarity_threshold: float = 0.1,
                  vector_similarity_weight: float = 0.3,
                  top: int = 1024, doc_ids: list[str] = None,
                  aggs: bool = True, rerank_mdl=None,
                  highlight: bool = False,
                  rank_feature: dict | None = None) -> dict:
        """
        主要检索接口

        Args:
            question: 查询问题
            embd_mdl: 嵌入模型
            tenant_ids: 租户ID
            kb_ids: 知识库ID列表
            page: 页码
            page_size: 每页大小
            similarity_threshold: 相似度阈值
            vector_similarity_weight: 向量相似度权重
            top: 最大返回数量
            doc_ids: 文档ID列表过滤
            aggs: 是否返回聚合
            rerank_mdl: 重排序模型
            highlight: 是否高亮
            rank_feature: 排序特征

        Returns:
            dict: 检索结果，包含 total, chunks, doc_aggs
        """
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}

        RERANK_PAGE_LIMIT = 3
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "size": max(page_size * RERANK_PAGE_LIMIT, 128),
            "question": question,
            "vector": True,
            "topk": top,
            "similarity": similarity_threshold,
            "available_int": 1
        }

        if page > RERANK_PAGE_LIMIT:
            req["page"] = page
            req["size"] = page_size

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        idx_names = [index_name(tid) for tid in tenant_ids]

        # 执行搜索
        sres = self.search(req, idx_names, kb_ids, embd_mdl, highlight, rank_feature)
        ranks["total"] = sres.total

        if page <= RERANK_PAGE_LIMIT and sres.total > 0:
            # 执行重排序
            sim, tsim, vsim = self.rerank(
                sres, question, 1 - vector_similarity_weight,
                vector_similarity_weight, rank_feature=rank_feature
            )
            idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]
        else:
            sim = tsim = vsim = [1] * len(sres.ids)
            idx = list(range(len(sres.ids)))

        dim = len(sres.query_vector) if sres.query_vector else 0
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim if dim > 0 else []

        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break

            chunk_id = sres.ids[i]
            chunk = sres.field[chunk_id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")
            position_int = chunk.get("position_int", [])

            d = {
                "chunk_id": chunk_id,
                "content_ltks": chunk.get("content_ltks", ""),
                "content_with_weight": chunk.get("content_with_weight", ""),
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk.get("kb_id", ""),
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": float(sim[i]),
                "vector_similarity": float(vsim[i]) if i < len(vsim) else 0,
                "term_similarity": float(tsim[i]) if i < len(tsim) else 0,
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
            }

            if highlight and sres.highlight and chunk_id in sres.highlight:
                d["highlight"] = sres.highlight[chunk_id]

            ranks["chunks"].append(d)

            # 文档聚合统计
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1

        # 转换聚合结果为列表
        ranks["doc_aggs"] = [
            {"doc_name": k, "doc_id": v["doc_id"], "count": v["count"]}
            for k, v in sorted(ranks["doc_aggs"].items(), key=lambda x: x[1]["count"] * -1)
        ]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks


__all__ = ['Dealer', 'SearchResult', 'MatchDenseExpr', 'MatchTextExpr',
           'FusionExpr', 'OrderByExpr', 'index_name']