"""
远程API Embedding模型
使用OpenAI兼容的API接口
"""

import os
import numpy as np
from typing import List, Optional
from openai import OpenAI
import logging

from .base_embedding import BaseEmbeddingModel

logger = logging.getLogger(__name__)

class RemoteEmbeddingModel(BaseEmbeddingModel):
    """远程API Embedding模型 - 支持配置维度"""

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model_name: str = "text-embedding-v3",
            dimensions: int = 1024,  # 从环境变量读取
            encoding_format: str = "float",
            max_batch_size: int = 10
    ):
        self._api_key = api_key or os.getenv("LLM_API_KEY")
        self._base_url = base_url or os.getenv("LLM_BASE_URL")
        self._model_name = model_name
        self._dimensions = dimensions  # 使用传入的维度
        self._encoding_format = encoding_format
        self._max_batch_size = max_batch_size

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url
        )

        logger.info(f"初始化远程Embedding模型: {model_name}, 维度={dimensions}")

    @property
    def dimension(self) -> int:
        return self._dimensions

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

    @property
    def dimension(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name