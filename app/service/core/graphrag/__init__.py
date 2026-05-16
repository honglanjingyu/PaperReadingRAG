# app/service/core/graphrag/__init__.py
"""GraphRAG 模块 - 知识图谱增强 RAG"""

from .service import GraphRAGService, get_graph_rag_service
from .neo4j_store import get_neo4j_store, StoredEntity, StoredRelation
from .core import EntityExtractor, Entity, Relation
from .community import CommunityDetector, Community
from .retriever import GraphRetriever, GraphCache, get_graph_cache

# 保持向后兼容
__all__ = [
    # 主服务
    'GraphRAGService',
    'get_graph_rag_service',
    # 存储
    'get_neo4j_store',
    'StoredEntity',
    'StoredRelation',
    # 核心组件
    'EntityExtractor',
    'Entity',
    'Relation',
    'CommunityDetector',
    'Community',
    'GraphRetriever',
    'GraphCache',
    'get_graph_cache',
]