# app/service/core/graphrag/__init__.py
"""
GraphRAG 模块 - 基于知识图谱的 RAG 增强
"""

from .entity_extractor import EntityExtractor
from .community_detection import CommunityDetector
from .hierarchical_summarizer import HierarchicalSummarizer
from .graph_retriever import GraphRetriever
from .graph_rag_service import GraphRAGService, get_graph_rag_service
from .graph_cache import GraphCache, get_graph_cache

__all__ = [
    'EntityExtractor',
    'CommunityDetector',
    'HierarchicalSummarizer',
    'GraphRetriever',
    'GraphRAGService',
    'get_graph_rag_service',
    'GraphCache',
    'get_graph_cache'
]