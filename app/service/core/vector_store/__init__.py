# app/service/core/vector_store/__init__.py

from .base import BaseVectorStore, VectorStoreType
from .factory import (
    VectorStoreFactory, get_vector_store_factory,
    get_vector_store, get_store_type, switch_vector_store
)
from .es_vector_store import ESVectorStore
from .milvus_vector_store import MilvusVectorStore
from .vector_storage_service import VectorStorageService, get_vector_storage_service
from .vector_search_service import VectorSearchService, get_vector_search_service

__all__ = [
    # 基类和类型
    'BaseVectorStore',
    'VectorStoreType',
    # 工厂
    'VectorStoreFactory',
    'get_vector_store_factory',
    'get_vector_store',
    'get_store_type',
    'switch_vector_store',
    # 实现类
    'ESVectorStore',
    'MilvusVectorStore',
    # 服务类
    'VectorStorageService',
    'get_vector_storage_service',
    'VectorSearchService',
    'get_vector_search_service'
]