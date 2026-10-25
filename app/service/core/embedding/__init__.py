"""
Embedding 向量化模块

提供统一的向量生成接口，通过远程 API 调用，带有 Redis 缓存。
"""

from .base import BaseEmbeddingModel
from .remote import RemoteEmbeddingModel
from .cached import CachedEmbeddingModel
from .service import (
    EmbeddingService,
    get_embedding_service,
    generate_embedding,
    generate_embeddings,
    get_vector_field_name,
    vectorize_chunks,
    VectorChunk,
)

__all__ = [
    'BaseEmbeddingModel',
    'RemoteEmbeddingModel',
    'CachedEmbeddingModel',
    'EmbeddingService',
    'get_embedding_service',
    'generate_embedding',
    'generate_embeddings',
    'get_vector_field_name',
    'vectorize_chunks',
    'VectorChunk',
]