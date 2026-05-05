# app/service/core/embedding/remote_embedding.py

import os
import numpy as np
from typing import List, Optional
from openai import OpenAI
import logging

from .base_embedding import BaseEmbeddingModel

logger = logging.getLogger(__name__)


def get_embedding_api_key() -> str:
    """获取 Embedding API Key：优先获取专用 key，再获取通用 key"""
    return os.getenv("EMBEDDING_API_KEY") or os.getenv("MODEL_API_KEY")


def get_embedding_base_url() -> str:
    """获取 Embedding Base URL"""
    return os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")


def get_embedding_model_name() -> str:
    """获取 Embedding 模型名称"""
    return os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


class RemoteEmbeddingModel(BaseEmbeddingModel):
    """远程API Embedding模型 - 支持配置维度"""

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model_name: str = None,
            dimensions: int = None,
            encoding_format: str = "float",
            max_batch_size: int = 10
    ):
        # 优先级：参数 > 专用环境变量 > 通用环境变量 > 默认值
        self._api_key = api_key or get_embedding_api_key()
        self._base_url = base_url or get_embedding_base_url()
        self._model_name = model_name or get_embedding_model_name()
        self._dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self._encoding_format = encoding_format
        self._max_batch_size = max_batch_size

        # 验证必要配置
        if not self._api_key:
            raise ValueError("未配置 Embedding API Key，请设置 EMBEDDING_API_KEY 或 MODEL_API_KEY")
        
        if not self._base_url:
            logger.warning("未配置 Embedding Base URL，将使用默认值")
            self._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url
        )

        logger.info(f"初始化远程Embedding模型: {self._model_name}, 维度={self._dimensions}, base_url={self._base_url}")

    @property
    def dimension(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量"""
        try:
            completion = self._client.embeddings.create(
                model=self._model_name,
                input=text,
                dimensions=self._dimensions,
                encoding_format=self._encoding_format
            )
            return completion.data[0].embedding
        except Exception as e:
            logger.error(f"远程API向量生成失败: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量"""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self._max_batch_size):
            batch = texts[i:i + self._max_batch_size]

            try:
                completion = self._client.embeddings.create(
                    model=self._model_name,
                    input=batch,
                    dimensions=self._dimensions,
                    encoding_format=self._encoding_format
                )

                batch_embeddings = [item.embedding for item in completion.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"批量生成成功: batch {i // self._max_batch_size + 1}, size={len(batch)}")

            except Exception as e:
                logger.error(f"批量生成失败: {e}")
                all_embeddings.extend([None] * len(batch))

        return all_embeddings