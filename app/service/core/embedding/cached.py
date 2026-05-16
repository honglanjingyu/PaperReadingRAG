"""带 Redis 缓存的 Embedding 模型"""

import os
import hashlib
import json
import logging
from typing import List, Optional

from .base import BaseEmbeddingModel
from .remote import RemoteEmbeddingModel
from .local import LocalEmbeddingModel
from ..cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedEmbeddingModel(BaseEmbeddingModel):
    """带缓存的 Embedding 模型"""

    def __init__(self, model: BaseEmbeddingModel = None, model_type: str = None, cache_ttl: int = None):
        self.cache = get_cache_manager()
        self.cache_ttl = cache_ttl or int(os.getenv("CACHE_EMBEDDING_TTL", "604800"))  # 7天

        if model:
            self._model = model
        else:
            if model_type == "local":
                self._model = LocalEmbeddingModel()
            else:
                self._model = RemoteEmbeddingModel()

        self._dimension = self._model.dimension
        self._model_name = self._model.model_name
        self._enabled = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

        logger.info(f"缓存 Embedding: {self._model_name}, 维度={self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_cache_key(self, text: str) -> str:
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self._model_name}:{self._dimension}:{content_hash}"

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not self._enabled:
            return self._model.generate_embedding(text)

        cache_key = self._get_cache_key(text)
        cached = self.cache.get("embedding", cache_key)

        if cached is not None:
            logger.debug("Embedding 缓存命中")
            return cached

        embedding = self._model.generate_embedding(text)
        if embedding:
            self.cache.set("embedding", cache_key, embedding, self.cache_ttl)
        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts or not self._enabled:
            return self._model.generate_embeddings(texts)

        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        # 批量查询缓存
        if self.cache.redis_client:
            try:
                pipe = self.cache.redis_client.pipeline()
                for text in texts:
                    cache_key = self._get_cache_key(text)
                    pipe.get(f"rag:cache:embedding:{cache_key}")
                cached_vals = pipe.execute()

                for i, (text, cached) in enumerate(zip(texts, cached_vals)):
                    if cached:
                        results[i] = json.loads(cached)
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            except Exception as e:
                logger.warning(f"批量查询缓存失败: {e}")
                return self._model.generate_embeddings(texts)
        else:
            return self._model.generate_embeddings(texts)

        # 生成未命中的向量
        if uncached_texts:
            embeddings = self._model.generate_embeddings(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings):
                if emb:
                    results[idx] = emb
                    cache_key = self._get_cache_key(texts[idx])
                    self.cache.set("embedding", cache_key, emb, self.cache_ttl)

        return results

    def clear_cache(self):
        """清空所有 embedding 缓存"""
        self.cache.delete_pattern("embedding")
        logger.info("Embedding 缓存已清空")