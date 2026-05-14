# app/service/core/rag/__init__.py (修复导入)

from .processor import (
    process_document,
    get_processing_stats,
)
from .search import (
    vectorize_user_question,
    search_similar_documents,
    enhanced_search_with_hybrid_and_rerank,
    # 移除不存在的导入
    # enhanced_search_with_parent_child,  # 删除这行
)
from .generation import (
    generate_answer,
    generate_answer_stream,
)

from app.service.core.embedding import VectorChunk

__all__ = [
    # 处理器
    'process_document',
    'get_processing_stats',
    # 搜索
    'vectorize_user_question',
    'search_similar_documents',
    'enhanced_search_with_hybrid_and_rerank',
    # 生成
    'generate_answer',
    'generate_answer_stream',
    # 类型
    'VectorChunk',
]