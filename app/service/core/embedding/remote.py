"""远程 API Embedding 模型（DashScope/OpenAI 兼容）"""

import os
import logging
from typing import List, Optional
from openai import OpenAI

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    return os.getenv("EMBEDDING_API_KEY") or os.getenv("MODEL_API_KEY")


def _get_base_url() -> str:
    return os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _get_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-v3")


class RemoteEmbeddingModel(BaseEmbeddingModel):
    """远程 API Embedding 模型"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
        dimensions: int = None,
        max_batch_size: int = 10
    ):
        self._api_key = api_key or _get_api_key()
        self._base_url = base_url or _get_base_url()
        self._model_name = model_name or _get_model_name()
        self._dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self._max_batch_size = max_batch_size

        if not self._api_key:
            raise ValueError("未配置 Embedding API Key，请设置 EMBEDDING_API_KEY 或 MODEL_API_KEY")

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        logger.info(f"远程 Embedding 初始化: {self._model_name}, 维度={self._dimensions}")

    @property
    def dimension(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        try:
            resp = self._client.embeddings.create(
                model=self._model_name,
                input=text,
                dimensions=self._dimensions,
                encoding_format="float"
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self._max_batch_size):
            batch = texts[i:i + self._max_batch_size]
            try:
                resp = self._client.embeddings.create(
                    model=self._model_name,
                    input=batch,
                    dimensions=self._dimensions,
                    encoding_format="float"
                )
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批量生成失败: {e}")
                all_embeddings.extend([None] * len(batch))
        return all_embeddings