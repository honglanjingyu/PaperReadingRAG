# app/service/core/retrieval/reranker.py
"""
重排序器 - 使用 HTTP API 调用 DashScope Rerank
支持 HTTP API、本地Cross-Encoder、向量相似度三种方式
"""

import logging
import os
import requests
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入本地sentence-transformers
try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.info("sentence-transformers未安装，本地Cross-Encoder重排序不可用")


class DashScopeRerankHTTP:
    """使用 HTTP API 直接调用 DashScope Rerank"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化 HTTP Rerank 客户端

        Args:
            api_key: API Key，从环境变量 LLM_API_KEY 读取
            base_url: Rerank API 地址，从环境变量 RERANK_API_URL 读取
            model: 模型名称，从环境变量 RERANK_MODEL 读取
        """
        # 从环境变量读取配置
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("RERANK_API_URL")
        self.model = model or os.getenv("RERANK_MODEL", "gte-rerank")

        # 验证配置
        if not self.api_key:
            logger.warning("未配置 LLM_API_KEY，HTTP Rerank 不可用")

        if not self.base_url:
            logger.warning("未配置 RERANK_API_URL，HTTP Rerank 不可用")
            logger.info(
                "请在 .env 中设置 RERANK_API_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")

        if self.api_key and self.base_url:
            logger.info(f"HTTP Rerank 客户端初始化成功")
            logger.info(f"  API URL: {self.base_url}")
            logger.info(f"  Model: {self.model}")
        else:
            logger.warning("HTTP Rerank 客户端初始化失败，配置不完整")

    def rerank(
            self,
            query: str,
            documents: List[str],
            top_n: int = 5,
            model: str = None
    ) -> List[Dict]:
        """
        调用 Rerank API

        Args:
            query: 查询文本
            documents: 文档文本列表
            top_n: 返回数量
            model: 模型名称（可选，覆盖默认值）

        Returns:
            排序后的结果列表，每个结果包含 index 和 relevance_score
        """
        # 检查配置
        if not self.api_key:
            logger.error("缺少 API Key (LLM_API_KEY)，无法调用 Rerank API")
            return []

        if not self.base_url:
            logger.error("缺少 Rerank API URL (RERANK_API_URL)，无法调用 Rerank API")
            return []

        if not documents:
            return []

        # 使用传入的 model 或默认的 self.model
        model_name = model or self.model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体
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
            logger.debug(
                f"调用 Rerank API: URL={self.base_url}, model={model_name}, query={query[:50]}..., docs={len(documents)}")

            response = requests.post(
                self.base_url,
                headers=headers,
                json=body,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                api_results = result.get("output", {}).get("results", [])
                logger.info(f"Rerank API 调用成功: 返回 {len(api_results)} 个结果")
                return api_results
            else:
                logger.error(f"Rerank API 调用失败: status={response.status_code}, response={response.text}")
                return []

        except requests.exceptions.Timeout:
            logger.error("Rerank API 请求超时")
            return []
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Rerank API 连接失败: {e}")
            return []
        except Exception as e:
            logger.error(f"Rerank API 请求异常: {e}")
            return []

    def rerank_documents(
            self,
            query: str,
            documents: List[Dict],
            content_field: str = "content_with_weight",
            top_n: int = 5
    ) -> Optional[List[Dict]]:
        """
        对文档列表进行重排序

        Args:
            query: 查询文本
            documents: 文档字典列表
            content_field: 内容字段名
            top_n: 返回数量

        Returns:
            重排序后的文档列表，失败返回 None
        """
        if not documents:
            return None

        # 提取文本内容，同时记录有效索引
        texts = []
        valid_indices = []

        for i, doc in enumerate(documents):
            content = doc.get(content_field, doc.get('content', ''))
            if content and content.strip():
                texts.append(content)
                valid_indices.append(i)

        if not texts:
            logger.warning("没有有效的文档内容用于重排序")
            return None

        # 调用 API
        api_results = self.rerank(query, texts, top_n)

        if not api_results:
            return None

        # 映射回原始文档
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

        logger.info(f"HTTP Rerank 完成: {len(reranked)}/{len(documents)} 个结果")
        return reranked

    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return bool(self.api_key and self.base_url)


class Reranker:
    """重排序器 - 支持 HTTP API、本地Cross-Encoder、向量相似度"""

    def __init__(
            self,
            model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            api_type: str = "auto",  # auto: 自动选择, api: HTTP API, local: 本地模型, vector: 向量相似度
            api_key: str = None,
            rerank_model: str = None,
            base_url: str = None
    ):
        """
        初始化重排序器

        Args:
            model_name: 本地Cross-Encoder模型名称
            api_type: 重排序类型 ('auto', 'api', 'local', 'vector')
            api_key: API Key（从环境变量 LLM_API_KEY 读取）
            rerank_model: Rerank 模型名称（从环境变量 RERANK_MODEL 读取）
            base_url: Rerank API 地址（从环境变量 RERANK_API_URL 读取）
        """
        self.api_type = api_type
        self.cross_encoder = None

        # 从环境变量读取配置
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.rerank_model = rerank_model or os.getenv("RERANK_MODEL", "gte-rerank")
        self.base_url = base_url or os.getenv("RERANK_API_URL")

        # 初始化 HTTP Rerank 客户端
        self.http_reranker = None
        if self.api_key and self.base_url:
            self.http_reranker = DashScopeRerankHTTP(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.rerank_model
            )
        else:
            logger.warning("HTTP Rerank 配置不完整，将使用其他重排序方式")

        # 根据 api_type 初始化
        if api_type == "auto":
            self._auto_select()
        elif api_type == "remote":
            if not self.http_reranker or not self.http_reranker.is_available():
                logger.warning("HTTP API 不可用，降级到向量相似度")
        elif api_type == "local":
            self._init_local(model_name)
        elif api_type == "vector":
            logger.info("使用向量相似度重排序")
        else:
            self._auto_select()

    def _auto_select(self):
        """自动选择最佳可用的重排序方式"""
        # 优先使用 HTTP API
        if self.http_reranker and self.http_reranker.is_available():
            logger.info("自动选择: HTTP Rerank API")
            return

        # 其次使用本地Cross-Encoder
        if CROSS_ENCODER_AVAILABLE:
            logger.info("自动选择: 本地Cross-Encoder模型")
            self._init_local()
            return

        # 最后使用向量相似度
        logger.info("自动选择: 向量相似度重排序")

    def _init_local(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """初始化本地Cross-Encoder模型"""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence-transformers未安装，无法使用本地Cross-Encoder")
            return

        try:
            self.cross_encoder = CrossEncoder(model_name)
            logger.info(f"本地Cross-Encoder模型加载成功: {model_name}")
        except Exception as e:
            logger.warning(f"本地Cross-Encoder模型加载失败: {e}")

    def rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int = 5,
            content_field: str = "content_with_weight"
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 检索结果列表
            top_k: 返回数量
            content_field: 内容字段名

        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []

        # 根据选择的类型进行重排序
        if self.api_type == "remote" or (
                self.api_type == "auto" and self.http_reranker and self.http_reranker.is_available()):
            result = self._rerank_with_http_api(query, documents, top_k, content_field)
            if result:
                return result

        if self.api_type == "local" or (self.api_type == "auto" and self.cross_encoder):
            result = self._rerank_with_cross_encoder(query, documents, top_k, content_field)
            if result:
                return result

        # 默认使用向量相似度
        return self._rerank_with_vector_similarity(query, documents, top_k, content_field)

    def _rerank_with_http_api(
            self,
            query: str,
            documents: List[Dict],
            top_k: int,
            content_field: str
    ) -> Optional[List[Dict]]:
        """使用 HTTP API 重排序"""
        if not self.http_reranker or not self.http_reranker.is_available():
            return None

        try:
            return self.http_reranker.rerank_documents(query, documents, content_field, top_k)
        except Exception as e:
            logger.error(f"HTTP API 重排序失败: {e}")
            return None

    def _rerank_with_cross_encoder(
            self,
            query: str,
            documents: List[Dict],
            top_k: int,
            content_field: str
    ) -> Optional[List[Dict]]:
        """使用本地Cross-Encoder重排序"""
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
            logger.info(f"Cross-Encoder 重排序完成: {len(documents)} -> {top_k}")
            return documents[:top_k]

        except Exception as e:
            logger.error(f"Cross-Encoder重排序失败: {e}")
            return None

    def _rerank_with_vector_similarity(
            self,
            query: str,
            documents: List[Dict],
            top_k: int,
            content_field: str
    ) -> List[Dict]:
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
        """获取当前可用的重排序方法"""
        if self.http_reranker and self.http_reranker.is_available() and self.api_type != "local":
            return "http_api"
        elif self.cross_encoder:
            return "cross_encoder_local"
        else:
            return "vector_similarity"


__all__ = ['Reranker', 'DashScopeRerankHTTP']