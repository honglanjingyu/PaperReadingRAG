# app/service/core/embedding/cached_embedding.py

import os
import hashlib
import logging
from typing import List, Optional

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel
from ..cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedEmbeddingModel(BaseEmbeddingModel):
    """带缓存的 Embedding 模型 - 包装原有模型"""

    def __init__(
            self,
            model: BaseEmbeddingModel = None,
            model_type: str = None,
            cache_ttl: int = None,
            **kwargs
    ):
        """
        初始化缓存 Embedding 模型

        Args:
            model: 原始 Embedding 模型（可选）
            model_type: 模型类型 ('remote' 或 'local')
            cache_ttl: 缓存过期时间（秒），默认 7 天
        """
        self.cache_manager = get_cache_manager()
        self.cache_ttl = cache_ttl or int(os.getenv("CACHE_EMBEDDING_TTL", "604800"))  # 7天

        # 初始化原始模型
        if model:
            self._model = model
        else:
            if model_type == "local":
                self._model = LocalEmbeddingModel(**kwargs)
            else:
                self._model = RemoteEmbeddingModel(**kwargs)

        self._dimension = self._model.dimension
        self._model_name = self._model.model_name

        logger.info(f"缓存 Embedding 模型初始化: {self._model_name}, 维度={self._dimension}, TTL={self.cache_ttl}s")

    def _get_cache_key(self, text: str) -> str:
        """生成缓存 key"""
        # 基于文本内容和模型名称生成 key
        import hashlib
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self._model_name}:{self._dimension}:{content_hash}"

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量（带缓存）"""
        if not text:
            return None

        # 检查是否启用缓存
        enabled = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        if not enabled:
            return self._model.generate_embedding(text)

        # 1. 查缓存
        cache_key = self._get_cache_key(text)
        cached = self.cache_manager.get("embedding", cache_key)

        if cached is not None:
            logger.debug(f"Embedding 缓存命中: {text[:50]}...")
            return cached

        # 2. 调用原始模型
        embedding = self._model.generate_embedding(text)

        # 3. 存入缓存
        if embedding:
            self.cache_manager.set("embedding", cache_key, embedding, self.cache_ttl)
            logger.debug(f"Embedding 缓存写入: {text[:50]}...")

        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量（带缓存）"""
        if not texts:
            return []

        # 检查是否启用缓存
        enabled = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        if not enabled:
            return self._model.generate_embeddings(texts)

        results = []
        uncached_texts = []
        uncached_indices = []

        # 1. 批量查缓存
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached = self.cache_manager.get("embedding", cache_key)

            if cached is not None:
                results.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)

        # 2. 批量调用原始模型
        if uncached_texts:
            embeddings = self._model.generate_embeddings(uncached_texts)

            for idx, embedding in zip(uncached_indices, embeddings):
                if embedding:
                    results[idx] = embedding
                    # 存入缓存
                    cache_key = self._get_cache_key(texts[idx])
                    self.cache_manager.set("embedding", cache_key, embedding, self.cache_ttl)

        return results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def invalidate_cache(self, text: str):
        """使指定文本的缓存失效"""
        cache_key = self._get_cache_key(text)
        self.cache_manager.delete("embedding", cache_key)
        logger.info(f"使 Embedding 缓存失效: {text[:50]}...")

    def clear_cache(self):
        """清空 Embedding 缓存"""
        self.cache_manager.delete_pattern("embedding")
        logger.info("已清空 Embedding 缓存")


__all__ = ['CachedEmbeddingModel']