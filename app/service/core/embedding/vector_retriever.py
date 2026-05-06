"""
向量检索模块
支持相似度搜索和混合检索
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_service import get_embedding_service
import logging

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    向量检索器
    支持向量相似度搜索和混合检索
    """

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service or get_embedding_service()

    def vector_search(
            self,
            query_text: str,
            vector_store: List[Dict[str, Any]],
            top_k: int = 5,
            similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query_text: 查询文本
            vector_store: 向量存储中的文档列表（包含向量字段）
            top_k: 返回数量
            similarity_threshold: 相似度阈值

        Returns:
            检索结果列表
        """
        # 1. 生成查询向量
        query_vector = self.embedding_service.generate_embedding(query_text)
        if query_vector is None:
            logger.error("查询向量生成失败")
            return []

        # 2. 提取所有文档向量
        vector_field_name = self.embedding_service.get_vector_field_name()
        doc_vectors = []
        valid_docs = []

        for doc in vector_store:
            vec = doc.get(vector_field_name)
            if vec and len(vec) == len(query_vector):
                doc_vectors.append(vec)
                valid_docs.append(doc)

        if not doc_vectors:
            return []

        # 3. 计算余弦相似度
        query_vector = np.array(query_vector).reshape(1, -1)
        doc_vectors = np.array(doc_vectors)

        similarities = cosine_similarity(query_vector, doc_vectors)[0]

        # 4. 过滤和排序
        results = []
        for i, (doc, sim) in enumerate(zip(valid_docs, similarities)):
            if sim >= similarity_threshold:
                results.append({
                    "document": doc,
                    "similarity": float(sim),
                    "rank": len(results) + 1
                })

        # 按相似度降序排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def hybrid_search(
            self,
            query_text: str,
            vector_store: List[Dict[str, Any]],
            keyword_weight: float = 0.3,
            vector_weight: float = 0.7,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        混合检索（向量 + 关键词）

        Args:
            query_text: 查询文本
            vector_store: 向量存储中的文档列表
            keyword_weight: 关键词权重
            vector_weight: 向量权重
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 向量相似度
        query_vector = self.embedding_service.generate_embedding(query_text)
        if query_vector is None:
            return []

        vector_field_name = self.embedding_service.get_vector_field_name()
        doc_vectors = []
        valid_docs = []

        for doc in vector_store:
            vec = doc.get(vector_field_name)
            if vec:
                doc_vectors.append(vec)
                valid_docs.append(doc)

        if not doc_vectors:
            return []

        query_vec = np.array(query_vector).reshape(1, -1)
        doc_vecs = np.array(doc_vectors)

        vector_scores = cosine_similarity(query_vec, doc_vecs)[0]

        # 关键词相似度（简化版）
        keywords = set(query_text.lower().split())
        keyword_scores = []

        for doc in valid_docs:
            content = doc.get("content_with_weight", "").lower()
            matched = sum(1 for kw in keywords if kw in content)
            keyword_scores.append(matched / max(len(keywords), 1))

        # 加权融合
        final_scores = vector_weight * vector_scores + keyword_weight * np.array(keyword_scores)

        # 排序
        indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for i, idx in enumerate(indices):
            if final_scores[idx] > 0:
                results.append({
                    "document": valid_docs[idx],
                    "vector_similarity": float(vector_scores[idx]),
                    "keyword_similarity": float(keyword_scores[idx]),
                    "final_score": float(final_scores[idx]),
                    "rank": i + 1
                })

        return results

    def rerank_by_model(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            content_field: str = "content_with_weight"
    ) -> List[Dict[str, Any]]:
        """
        使用模型进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            content_field: 内容字段名

        Returns:
            重排序后的文档列表
        """
        # 这里可以实现更复杂的重排序逻辑
        # 例如使用交叉编码器模型

        # 简单的基于向量的重排序
        query_vector = self.embedding_service.generate_embedding(query)
        if query_vector is None:
            return documents

        vector_field_name = self.embedding_service.get_vector_field_name()

        for doc in documents:
            doc_vector = doc.get(vector_field_name)
            if doc_vector:
                sim = cosine_similarity([query_vector], [doc_vector])[0][0]
                doc["rerank_score"] = float(sim)

        documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return documents


def calculate_token_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """
    计算关键词相似度

    Args:
        query_tokens: 查询词列表
        doc_tokens: 文档词列表

    Returns:
        相似度分数
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    query_set = set(query_tokens)
    doc_set = set(doc_tokens)

    intersection = query_set & doc_set
    union = query_set | doc_set

    return len(intersection) / len(union) if union else 0.0