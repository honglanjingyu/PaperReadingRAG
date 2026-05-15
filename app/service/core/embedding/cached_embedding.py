# app/service/core/embedding/cached_embedding.py

import os
import hashlib
import logging
from typing import List, Optional
import json

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel
from ..cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedEmbeddingModel(BaseEmbeddingModel):
    """带缓存的 Embedding 模型 - 纯 Redis 缓存"""

    def __init__(self, model: BaseEmbeddingModel = None, model_type: str = None, cache_ttl: int = None, **kwargs):
        self.cache_manager = get_cache_manager()
        self.cache_ttl = cache_ttl or int(os.getenv("CACHE_EMBEDDING_TTL", "604800"))  # 7天

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
        """生成缓存 key（不含前缀）"""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self._model_name}:{self._dimension}:{content_hash}"

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量（纯 Redis 缓存）"""
        if not text:
            return None

        enabled = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        if not enabled:
            return self._model.generate_embedding(text)

        # 查 Redis 缓存
        cache_key = self._get_cache_key(text)
        cached = self.cache_manager.get("embedding", cache_key)

        if cached is not None:
            logger.debug(f"Embedding 缓存命中 (Redis): {text[:50]}...")
            return cached

        # 调用原始模型
        embedding = self._model.generate_embedding(text)

        # 存入 Redis 缓存
        if embedding:
            self.cache_manager.set("embedding", cache_key, embedding, self.cache_ttl)
            logger.debug(f"Embedding 缓存写入: {text[:50]}...")

        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量（使用 Redis pipeline）"""
        if not texts:
            return []

        enabled = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        if not enabled:
            return self._model.generate_embeddings(texts)

        results = []
        uncached_texts = []
        uncached_indices = []

        # 批量查 Redis 缓存
        if self.cache_manager.redis_client:
            try:
                pipe = self.cache_manager.redis_client.pipeline()
                for text in texts:
                    cache_key = self._get_cache_key(text)
                    pipe.get(f"rag:cache:embedding:{cache_key}")
                cached_results = pipe.execute()

                for i, (text, cached) in enumerate(zip(texts, cached_results)):
                    if cached:
                        results.append(json.loads(cached))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                        results.append(None)
            except Exception as e:
                logger.warning(f"批量查询缓存失败: {e}，回退到单条查询")
                return self._generate_embeddings_fallback(texts)
        else:
            return self._generate_embeddings_fallback(texts)

        # 批量调用原始模型
        if uncached_texts:
            embeddings = self._model.generate_embeddings(uncached_texts)

            # 批量写入 Redis 缓存
            if self.cache_manager.redis_client:
                try:
                    pipe = self.cache_manager.redis_client.pipeline()
                    for idx, embedding in zip(uncached_indices, embeddings):
                        if embedding:
                            results[idx] = embedding
                            cache_key = self._get_cache_key(texts[idx])
                            pipe.setex(f"rag:cache:embedding:{cache_key}", self.cache_ttl,
                                      json.dumps(embedding, ensure_ascii=False))
                    pipe.execute()
                except Exception as e:
                    logger.warning(f"批量写入缓存失败: {e}")
                    # 回退到单条写入
                    for idx, embedding in zip(uncached_indices, embeddings):
                        if embedding:
                            results[idx] = embedding
                            cache_key = self._get_cache_key(texts[idx])
                            self.cache_manager.set("embedding", cache_key, embedding, self.cache_ttl)
            else:
                for idx, embedding in zip(uncached_indices, embeddings):
                    if embedding:
                        results[idx] = embedding
                        cache_key = self._get_cache_key(texts[idx])
                        self.cache_manager.set("embedding", cache_key, embedding, self.cache_ttl)

        return results

    def _generate_embeddings_fallback(self, texts: List[str]) -> List[Optional[List[float]]]:
        """回退方案：单条查询"""
        return [self.generate_embedding(text) for text in texts]

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