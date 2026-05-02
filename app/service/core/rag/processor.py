# app/service/core/rag/processor.py
"""
RAG 文档处理器
包含：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗 -> 智能分块 -> 向量化 -> 向量存储
"""

import os
import sys
import hashlib
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

# 导入 deepdoc 模块
from app.service.core.deepdoc import (
    TextBlock, TableBlock, PageContent, ParsedDocument, LayoutType,
    DataLoader, LayoutRecognizer, CrossPageConnector, DocumentParser,
    DataCleaner, CleaningPipeline,
    parse_document, parse_document_to_text, clean_text,
)

# 导入分块模块
from app.service.core.chunking import (
    ChunkManager, ChunkProcessor, RecursiveChunkerSimple,
    create_chunker, chunk_text_to_chunks, chunk_text_simple,
    get_chunk_statistics, Chunk, ChunkStrategy,
)

# 导入向量化模块
from app.service.core.embedding import (
    VectorChunk, VectorizationService, vectorize_chunks, vectorize_text,
    EmbeddingManager, EmbeddingType, get_embedding_manager, get_embedding_service,
)

# 导入向量存储模块
from app.service.core.vector_store import (
    ESVectorStore, VectorStorageService, get_vector_storage_service, get_vector_search_service,
)


def process_document(
    file_path: str,
    chunk_size: int = 256,
    enable_vectorization: bool = True,
    enable_storage: bool = False,
    model_type: str = None,
    from_page: int = 0,
    to_page: int = None,
    index_name: str = None,
    verbose: bool = False
) -> List[VectorChunk]:
    """
    完整的文档处理流程：
    RAG1流程：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗
    RAG2流程：智能分块 -> 向量化 -> 向量存储

    Args:
        file_path: 文件路径
        chunk_size: 分块大小（token数）
        enable_vectorization: 是否启用向量化
        enable_storage: 是否存储到向量数据库
        model_type: 模型类型 ('remote' 或 'local')
        from_page: 起始页
        to_page: 结束页
        index_name: ES索引名称
        verbose: 是否打印详细信息

    Returns:
        VectorChunk 列表
    """
    if verbose:
        print("=" * 70)
        print(f"处理文档: {os.path.basename(file_path)}")

    # ========== RAG1 完整流程 ==========
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=verbose
    )

    if not parsed.cleaned_text:
        if verbose:
            print("错误: 未能提取文本内容")
        return []

    if verbose:
        print(f"\n处理结果:")
        print(f"  文件名: {parsed.file_name}")
        print(f"  文件类型: {parsed.file_type}")
        print(f"  总页数: {parsed.total_pages}")
        print(f"  清洗后文本长度: {len(parsed.cleaned_text)} 字符")

    # ========== RAG2 功能：智能分块 ==========
    if verbose:
        print("\n[5/12] 智能分块...")

    chunks = chunk_text_to_chunks(
        parsed.cleaned_text,
        chunk_size=chunk_size,
        metadata={
            'source': parsed.file_name,
            'file_type': parsed.file_type,
            'total_pages': parsed.total_pages
        },
        strategy='recursive'
    )

    if verbose:
        stats = get_chunk_statistics(chunks)
        print(f"  生成 {stats['total_chunks']} 个块")
        print(f"  Token统计: 最小={stats['min_token_count']}, "
              f"最大={stats['max_token_count']}, "
              f"平均={stats['avg_token_count']:.1f}")

    # ========== RAG2 功能：向量化 ==========
    vector_chunks = []
    if enable_vectorization and chunks:
        if verbose:
            print("\n[6/12] 向量化处理...")

        try:
            # 转换为 VectorChunk
            vector_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{i}_{chunk.content[:100]}".encode()).hexdigest()[:16]
                vector_chunks.append(VectorChunk(
                    id=f"chunk_{i}_{chunk_id}",
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        'chunk_index': i,
                        'token_count': chunk.token_count
                    },
                    token_count=chunk.token_count,
                    chunk_index=i
                ))

            # 向量化
            vec_service = VectorizationService(model_type)
            vector_chunks = vec_service.vectorize_chunks(vector_chunks)

            if verbose:
                vectorized_count = len([c for c in vector_chunks if c.vector])
                print(f"  向量化完成: {vectorized_count}/{len(vector_chunks)} 个块")
                print(f"  向量维度: {vec_service.dimension}")

        except Exception as e:
            if verbose:
                print(f"  向量化失败: {e}")
            vector_chunks = []

    # ========== RAG2 功能：向量存储 ==========
    if enable_storage and vector_chunks:
        if verbose:
            print("\n[7/12] 存储到向量数据库...")

        try:
            storage_service = get_vector_storage_service()
            index = index_name or os.getenv("ES_INDEX_NAME", "rag_documents")
            inserted = storage_service.store_vector_chunks(vector_chunks, index, parsed.file_name)

            if verbose:
                print(f"  存储完成: {inserted}/{len(vector_chunks)} 条")

        except Exception as e:
            if verbose:
                print(f"  存储失败: {e}")

    return vector_chunks if vector_chunks else chunks


def parse_only(
    file_path: str,
    from_page: int = 0,
    to_page: int = None,
    enable_cleaning: bool = True,
    verbose: bool = False
) -> ParsedDocument:
    """仅执行 RAG1 流程：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗"""
    parser = DocumentParser()
    return parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=enable_cleaning,
        verbose=verbose
    )


def chunk_document(
    file_path: str,
    chunk_size: int = 256,
    from_page: int = 0,
    to_page: int = None,
    verbose: bool = False
) -> List[Chunk]:
    """仅执行：RAG1完整流程 + 智能分块（不含向量化和存储）"""
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=verbose
    )

    if not parsed.cleaned_text:
        return []

    return chunk_text_to_chunks(
        parsed.cleaned_text,
        chunk_size=chunk_size,
        metadata={'source': parsed.file_name, 'file_type': parsed.file_type}
    )


def vectorize_chunk_texts(
    texts: List[str],
    metadata_list: List[dict] = None,
    model_type: str = None
) -> List[VectorChunk]:
    """向量化文本列表"""
    chunks = []
    for i, text in enumerate(texts):
        chunk_id = hashlib.md5(f"{i}_{text[:100]}".encode()).hexdigest()[:16]
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
        chunks.append(VectorChunk(
            id=f"chunk_{i}_{chunk_id}",
            content=text,
            metadata=metadata
        ))

    return vectorize_chunks(chunks, model_type)


def get_processing_stats(file_path: str, from_page: int = 0, to_page: int = None) -> dict:
    """获取 RAG1 处理统计信息"""
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=False
    )

    total_text_blocks = sum(len(p.text_blocks) for p in parsed.pages)
    total_tables = sum(len(p.tables) for p in parsed.pages)

    return {
        'file_name': parsed.file_name,
        'file_type': parsed.file_type,
        'total_pages': parsed.total_pages,
        'raw_text_length': len(parsed.cleaned_text) if parsed.cleaned_text else 0,
        'cleaned_text_length': len(parsed.cleaned_text),
        'total_text_blocks': total_text_blocks,
        'total_tables': total_tables,
    }