# app/service/core/rag/__init__.py

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
)
from .generation import (
    generate_answer,
    generate_answer_stream,
)

from app.service.core.embedding import VectorChunk

# 新增：导出保存报告的函数
from app.service.core.deepdoc.parser.remote_pdf_parser import save_chunked_report, RemotePDFParser

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
    # 生成
    'generate_answer',
    'generate_answer_stream',
    # 类型
    'VectorChunk',
    # 报告
    'save_chunked_report',
    'RemotePDFParser',
]