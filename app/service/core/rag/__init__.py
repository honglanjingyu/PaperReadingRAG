# app/service/core/rag/__init__.py
"""
RAG 核心模块
整合文档处理、向量化、检索、生成等 RAG 全流程
"""

from .processor import (
    process_document,
    parse_only,
    chunk_document,
    vectorize_chunk_texts,
    get_processing_stats,
)
from .search import (
    vectorize_user_question,
    search_similar_documents,
    enhanced_search_with_hybrid_and_rerank,
    test_user_question_vectorization,
    test_similarity_search,
    test_enhanced_retrieval,
    compare_search_methods,
)
from .generation import (
    generate_answer,
    generate_answer_stream,
    test_llm_generation,
    test_llm_generation_with_rewritten_only,
)
from .utils import (
    TEST_QUESTIONS,
    run_all_tests,
    RecursiveChunker,
)

# 从 embedding 模块导入 VectorChunk 以便重新导出
from app.service.core.embedding import VectorChunk


__all__ = [
    # 处理器
    'process_document',
    'parse_only',
    'chunk_document',
    'vectorize_chunk_texts',
    'get_processing_stats',
    # 搜索
    'vectorize_user_question',
    'search_similar_documents',
    'enhanced_search_with_hybrid_and_rerank',
    'test_user_question_vectorization',
    'test_similarity_search',
    'test_enhanced_retrieval',
    'compare_search_methods',
    # 生成
    'generate_answer',
    'generate_answer_stream',
    'test_llm_generation',
    'test_llm_generation_with_rewritten_only',
    # 工具
    'TEST_QUESTIONS',
    'run_all_tests',
    'RecursiveChunker',
    # 类型
    'VectorChunk',
]