# app/service/core/embedding/__init__.py
"""Embedding 向量化模块"""

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel
from .cached_embedding import CachedEmbeddingModel
from .embedding_service import (
    EmbeddingService, get_embedding_service,
    VectorChunk, generate_embedding, generate_embeddings,
    vectorize_chunks, get_vector_field_name
)

# 添加向后兼容别名
VectorizationService = EmbeddingService

__all__ = [
    'BaseEmbeddingModel',
    'RemoteEmbeddingModel',
    'LocalEmbeddingModel',
    'CachedEmbeddingModel',
    'EmbeddingService',
    'VectorizationService',  # 向后兼容
    'get_embedding_service',
    'VectorChunk',
    'generate_embedding',
    'generate_embeddings',
    'vectorize_chunks',
    'get_vector_field_name',
]