# app/service/core/retrieval/reranker.py
"""重排序器 - 支持 HTTP API、本地 Cross-Encoder、向量相似度"""

import logging
import os
import requests
import numpy as np
from typing import List, Dict, Any, Optional

from .base import BaseRetriever

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.info("sentence-transformers未安装，本地Cross-Encoder不可用")


# ========== 配置函数 ==========

def get_rerank_type() -> str:
    rerank_type = os.getenv("RERANK_TYPE", "auto").lower()
    return "remote" if rerank_type == "api" else rerank_type


def get_rerank_api_key() -> str:
    return os.getenv("RERANK_API_KEY") or os.getenv("MODEL_API_KEY")


def get_rerank_base_url() -> str:
    return os.getenv("RERANK_BASE_URL") or os.getenv("RERANK_API_URL") or os.getenv("LLM_BASE_URL")


def get_rerank_model() -> str:
    return os.getenv("RERANK_MODEL", "gte-rerank")


def get_local_rerank_path() -> str:
    return os.getenv("LOCAL_RERANK_PATH")


def get_local_rerank_model() -> str:
    return os.getenv("LOCAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class DashScopeRerankHTTP:
    """DashScope Rerank HTTP API 客户端"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or get_rerank_api_key()
        self.base_url = base_url or get_rerank_base_url()
        self.model = model or get_rerank_model()
        self._available = bool(self.api_key and self.base_url)

        if self._available:
            logger.info(f"HTTP Rerank 初始化: model={self.model}")
        else:
            logger.warning("HTTP Rerank 配置不完整")

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """调用 Rerank API"""
        if not self._available or not documents:
            return []

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": min(top_n, len(documents))}
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=body, timeout=30)
            if response.status_code == 200:
                return response.json().get("output", {}).get("results", [])
            logger.error(f"Rerank API 失败: {response.status_code}")
        except Exception as e:
            logger.error(f"Rerank API 异常: {e}")
        return []

    def rerank_documents(self, query: str, documents: List[Dict],
                         content_field: str = "content_with_weight", top_n: int = 5) -> Optional[List[Dict]]:
        """重排序文档列表"""
        if not documents:
            return None

        texts, valid_indices = [], []
        for i, doc in enumerate(documents):
            content = doc.get(content_field, doc.get('content', ''))
            if content and content.strip():
                texts.append(content)
                valid_indices.append(i)

        if not texts:
            return None

        api_results = self.rerank(query, texts, top_n)
        if not api_results:
            return None

        reranked = []
        for api_result in api_results:
            idx = api_result.get("index")
            if idx is not None and idx < len(valid_indices):
                doc = documents[valid_indices[idx]].copy()
                doc['rerank_score'] = api_result.get("relevance_score", 0)
                doc['original_score'] = doc.get('final_score', doc.get('_score', 0))
                doc['rerank_source'] = 'dashscope_http'
                reranked.append(doc)

        reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked

    def is_available(self) -> bool:
        return self._available


class Reranker(BaseRetriever):
    """重排序器 - 支持多种后端"""

    def __init__(self, model_name: str = None, api_type: str = None, **kwargs):
        self.api_type = api_type or get_rerank_type()
        self.cross_encoder = None

        # 初始化 HTTP 客户端
        self.http_reranker = DashScopeRerankHTTP()

        # 根据类型初始化
        if self.api_type == "auto":
            self._auto_select()
        elif self.api_type == "local":
            self._init_local()
        elif self.api_type == "vector":
            logger.info("使用向量相似度重排序（无需初始化）")

    def _auto_select(self):
        """自动选择最佳重排序方法"""
        if self.http_reranker.is_available():
            logger.info("自动选择: HTTP Rerank API")
        elif CROSS_ENCODER_AVAILABLE:
            logger.info("自动选择: 本地 Cross-Encoder")
            self._init_local()
        else:
            logger.info("自动选择: 向量相似度")

    def _init_local(self):
        """初始化本地 Cross-Encoder"""
        if not CROSS_ENCODER_AVAILABLE:
            return

        try:
            model_path = get_local_rerank_path() or get_local_rerank_model()
            self.cross_encoder = CrossEncoder(model_path)
            logger.info(f"本地 Cross-Encoder 加载成功: {model_path}")
        except Exception as e:
            logger.warning(f"本地 Cross-Encoder 加载失败: {e}")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5,
               content_field: str = "content_with_weight") -> List[Dict]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量
            content_field: 内容字段名

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 尝试 HTTP API
        if self.api_type in ("remote", "api") or (self.api_type == "auto" and self.http_reranker.is_available()):
            result = self._rerank_http(query, documents, top_k, content_field)
            if result:
                return result[:top_k]

        # 尝试本地 Cross-Encoder
        if self.api_type == "local" or (self.api_type == "auto" and self.cross_encoder):
            result = self._rerank_cross_encoder(query, documents, top_k, content_field)
            if result:
                return result[:top_k]

        # 降级：向量相似度
        return self._rerank_vector(query, documents, top_k, content_field)[:top_k]

    def _rerank_http(self, query: str, documents: List[Dict], top_k: int, content_field: str) -> Optional[List[Dict]]:
        """使用 HTTP API 重排序"""
        if not self.http_reranker.is_available():
            return None
        try:
            return self.http_reranker.rerank_documents(query, documents, content_field, top_k)
        except Exception as e:
            logger.error(f"HTTP 重排序失败: {e}")
            return None

    def _rerank_cross_encoder(self, query: str, documents: List[Dict], top_k: int, content_field: str) -> Optional[
        List[Dict]]:
        """使用本地 Cross-Encoder 重排序"""
        if not self.cross_encoder:
            return None

        try:
            contents = [doc.get(content_field, doc.get('content', '')) for doc in documents]
            pairs = [(query, content) for content in contents]
            scores = self.cross_encoder.predict(pairs)

            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
                doc['original_score'] = doc.get('final_score', doc.get('_score', 0))
                doc['rerank_source'] = 'cross_encoder'

            documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return documents
        except Exception as e:
            logger.error(f"Cross-Encoder 重排序失败: {e}")
            return None

    def _rerank_vector(self, query: str, documents: List[Dict], top_k: int, content_field: str) -> List[Dict]:
        """使用向量相似度重排序（备用方案）"""
        try:
            from app.service.core.embedding import get_embedding_service
            embedding_service = get_embedding_service()
            query_vector = embedding_service.generate_embedding(query)

            if not query_vector:
                return documents[:top_k]

            for doc in documents:
                doc_vector = doc.get('vector')
                if doc_vector and len(doc_vector) == len(query_vector):
                    sim = np.dot(query_vector, doc_vector) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-8
                    )
                    doc['rerank_score'] = float(sim)
                else:
                    doc['rerank_score'] = doc.get('final_score', doc.get('_score', 0))

                doc['rerank_source'] = 'vector_similarity'

            documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return documents
        except Exception as e:
            logger.error(f"向量重排序失败: {e}")
            return documents[:top_k]

    def get_available_method(self) -> str:
        """获取当前可用的重排序方法"""
        if self.http_reranker.is_available() and self.api_type != "local":
            return "http_api"
        elif self.cross_encoder:
            return "cross_encoder_local"
        else:
            return "vector_similarity"
