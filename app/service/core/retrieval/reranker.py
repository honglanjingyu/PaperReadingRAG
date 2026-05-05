# app/service/core/retrieval/reranker.py

import logging
import os
import requests
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.info("sentence-transformers未安装，本地Cross-Encoder重排序不可用")


def get_rerank_type() -> str:
    """获取重排序类型配置"""
    rerank_type = os.getenv("RERANK_TYPE", "auto").lower()
    if rerank_type == "api":
        return "remote"
    return rerank_type


def get_rerank_api_key() -> str:
    """获取 Rerank API Key：优先获取专用 key，再获取通用 key"""
    return os.getenv("RERANK_API_KEY") or os.getenv("MODEL_API_KEY")


def get_rerank_base_url() -> str:
    """获取 Rerank API Base URL"""
    return os.getenv("RERANK_BASE_URL") or os.getenv("RERANK_API_URL") or os.getenv("LLM_BASE_URL")


def get_rerank_model() -> str:
    """获取 Rerank 模型名称"""
    return os.getenv("RERANK_MODEL", "gte-rerank")


def get_local_rerank_path() -> str:
    """获取本地 Rerank 模型路径"""
    return os.getenv("LOCAL_RERANK_PATH")


def get_local_rerank_model() -> str:
    """获取本地 Rerank 模型名称"""
    return os.getenv("LOCAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class DashScopeRerankHTTP:
    """使用 HTTP API 直接调用 DashScope Rerank"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        # 优先级：参数 > 专用环境变量 > 通用环境变量
        self.api_key = api_key or get_rerank_api_key()
        self.base_url = base_url or get_rerank_base_url()
        self.model = model or get_rerank_model()

        if not self.api_key:
            logger.warning("未配置 Rerank API Key，请设置 RERANK_API_KEY 或 MODEL_API_KEY")

        if not self.base_url:
            logger.warning("未配置 Rerank API URL，请设置 RERANK_BASE_URL 或 RERANK_API_URL")
            logger.info("例如: RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")

        if self.api_key and self.base_url:
            logger.info(f"HTTP Rerank 客户端初始化成功: model={self.model}")
        else:
            logger.warning("HTTP Rerank 客户端初始化失败，配置不完整")

    def rerank(self, query: str, documents: List[str], top_n: int = 5, model: str = None) -> List[Dict]:
        if not self.api_key or not self.base_url:
            return []

        if not documents:
            return []

        model_name = model or self.model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": model_name,
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "top_n": min(top_n, len(documents))
            }
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=body, timeout=30)

            if response.status_code == 200:
                result = response.json()
                api_results = result.get("output", {}).get("results", [])
                logger.info(f"Rerank API 调用成功: 返回 {len(api_results)} 个结果")
                return api_results
            else:
                logger.error(f"Rerank API 调用失败: status={response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Rerank API 请求异常: {e}")
            return []

    def rerank_documents(self, query: str, documents: List[Dict], content_field: str = "content_with_weight", top_n: int = 5) -> Optional[List[Dict]]:
        if not documents:
            return None

        texts = []
        valid_indices = []

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
                original_idx = valid_indices[idx]
                doc = documents[original_idx].copy()
                doc['rerank_score'] = api_result.get("relevance_score", 0)
                doc['original_score'] = doc.get('final_score', doc.get('_score', 0))
                doc['rerank_source'] = 'dashscope_http'
                reranked.append(doc)

        # 关键修复：按 rerank_score 降序排序（确保顺序正确）
        reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        return reranked

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)


class Reranker:
    """重排序器 - 支持 HTTP API、本地Cross-Encoder、向量相似度"""

    def __init__(
            self,
            model_name: str = None,
            api_type: str = None,
            api_key: str = None,
            rerank_model: str = None,
            base_url: str = None
    ):
        # 优先级：参数 > 环境变量 > 默认值
        self.api_type = api_type or get_rerank_type()
        self.cross_encoder = None

        self.api_key = api_key or get_rerank_api_key()
        self.rerank_model = rerank_model or get_rerank_model()
        self.base_url = base_url or get_rerank_base_url()
        self.local_model_path = get_local_rerank_path()
        self.local_model_name = model_name or get_local_rerank_model()

        self.http_reranker = None
        if self.api_key and self.base_url:
            self.http_reranker = DashScopeRerankHTTP(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.rerank_model
            )
        else:
            logger.warning("HTTP Rerank 配置不完整")

        if self.api_type == "auto":
            self._auto_select()
        elif self.api_type == "remote" or self.api_type == "api":
            if not self.http_reranker or not self.http_reranker.is_available():
                logger.warning("HTTP API 不可用，降级到向量相似度")
        elif self.api_type == "local":
            self._init_local()
        elif self.api_type == "vector":
            logger.info("使用向量相似度重排序")

    def _auto_select(self):
        if self.http_reranker and self.http_reranker.is_available():
            logger.info("自动选择: HTTP Rerank API")
            return

        if CROSS_ENCODER_AVAILABLE:
            logger.info("自动选择: 本地Cross-Encoder模型")
            self._init_local()
            return

        logger.info("自动选择: 向量相似度重排序")

    def _init_local(self):
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence-transformers未安装，无法使用本地Cross-Encoder")
            return

        try:
            model_path = self.local_model_path or self.local_model_name
            self.cross_encoder = CrossEncoder(model_path)
            logger.info(f"本地Cross-Encoder模型加载成功: {model_path}")
        except Exception as e:
            logger.warning(f"本地Cross-Encoder模型加载失败: {e}")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5, content_field: str = "content_with_weight") -> List[Dict]:
        """重排序"""
        if not documents:
            return []

        result = None

        if self.api_type == "remote" or self.api_type == "api" or (self.api_type == "auto" and self.http_reranker and self.http_reranker.is_available()):
            result = self._rerank_with_http_api(query, documents, top_k, content_field)
            if result:
                # 确保排序
                result.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                return result[:top_k]

        if self.api_type == "local" or (self.api_type == "auto" and self.cross_encoder):
            result = self._rerank_with_cross_encoder(query, documents, top_k, content_field)
            if result:
                result.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                return result[:top_k]

        result = self._rerank_with_vector_similarity(query, documents, top_k, content_field)
        result.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return result[:top_k]

    def _rerank_with_http_api(self, query: str, documents: List[Dict], top_k: int, content_field: str) -> Optional[List[Dict]]:
        if not self.http_reranker or not self.http_reranker.is_available():
            return None

        try:
            return self.http_reranker.rerank_documents(query, documents, content_field, top_k)
        except Exception as e:
            logger.error(f"HTTP API 重排序失败: {e}")
            return None

    def _rerank_with_cross_encoder(self, query: str, documents: List[Dict], top_k: int, content_field: str) -> Optional[List[Dict]]:
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
            return documents[:top_k]

        except Exception as e:
            logger.error(f"Cross-Encoder重排序失败: {e}")
            return None

    def _rerank_with_vector_similarity(self, query: str, documents: List[Dict], top_k: int, content_field: str) -> List[Dict]:
        """使用向量相似度重排序（备用方案）"""
        try:
            from app.service.core.embedding import get_embedding_manager

            embedding_manager = get_embedding_manager()
            query_vector = embedding_manager.generate_embedding(query)

            if not query_vector:
                logger.warning("无法生成查询向量，返回原始排序")
                return documents[:top_k]

            for doc in documents:
                # 尝试从文档中获取向量
                doc_vector = None
                # 优先查找 vector 字段（Milvus）
                if 'vector' in doc and isinstance(doc['vector'], list):
                    doc_vector = doc['vector']
                # 然后查找 q_{dim}_vec 格式的字段（ES）
                for key, value in doc.items():
                    if key.endswith('_vec') and isinstance(value, list):
                        doc_vector = value
                        break

                if doc_vector and len(doc_vector) == len(query_vector):
                    sim = np.dot(query_vector, doc_vector) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-8
                    )
                    doc['rerank_score'] = float(sim)
                else:
                    doc['rerank_score'] = doc.get('final_score', doc.get('_score', 0))

                doc['original_score'] = doc.get('final_score', doc.get('_score', 0))
                doc['rerank_source'] = 'vector_similarity'

            documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"向量相似度重排序完成: {len(documents)} -> {top_k}")
            return documents[:top_k]

        except Exception as e:
            logger.error(f"向量相似度重排序失败: {e}")
            return documents[:top_k]

    def get_available_method(self) -> str:
        if self.http_reranker and self.http_reranker.is_available() and self.api_type != "local":
            return "http_api"
        elif self.cross_encoder:
            return "cross_encoder_local"
        else:
            return "vector_similarity"


__all__ = ['Reranker', 'DashScopeRerankHTTP', 'get_rerank_type', 'get_rerank_api_key', 'get_rerank_base_url', 'get_rerank_model']