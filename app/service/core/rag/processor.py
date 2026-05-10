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
from app.service.core.deepdoc.parser.remote_pdf_parser import save_chunked_report, RemotePDFParser,is_remote_parse_enabled
import logging
logger = logging.getLogger(__name__)

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
    VectorChunk,
    get_embedding_service,  # 使用 EmbeddingService 替代 VectorizationService
    vectorize_chunks,
)

# 导入向量存储模块 - 修复：移除 ESVectorStore 导入
from app.service.core.vector_store import (
    VectorStorageService, get_vector_storage_service, get_vector_search_service,
)


# app/service/core/rag/processor.py
# 替换原有的 process_document 函数

def process_document(
        file_path: str,
        chunk_size: int = 256,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        model_type: str = None,
        from_page: int = 0,
        to_page: int = None,
        index_name: str = None,
        verbose: bool = False,
        user_level: str = "normal"  # 新增参数：文档所属用户等级
) -> List[VectorChunk]:
    """
    完整的文档处理流程
    """
    if verbose:
        print("=" * 70)
        print(f"处理文档: {os.path.basename(file_path)}")

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
        print(f"  用户等级: {user_level}")

    if verbose:
        print("\n智能分块...")

    chunks = chunk_text_to_chunks(
        parsed.cleaned_text,
        chunk_size=chunk_size,
        metadata={
            'source': parsed.file_name,
            'file_type': parsed.file_type,
            'total_pages': parsed.total_pages,
            'user_level': user_level  # 添加等级到元数据
        },
        strategy='recursive'
    )

    if verbose:
        stats = get_chunk_statistics(chunks)
        print(f"  生成 {stats['total_chunks']} 个块")

    vector_chunks = []
    if enable_vectorization and chunks:
        if verbose:
            print("\n向量化处理...")

        try:
            vector_chunks = create_and_vectorize_chunks_with_level(
                chunks, model_type, user_level, verbose
            )

            if verbose and vector_chunks:
                vectorized_count = len([c for c in vector_chunks if c.vector])
                print(f"  完成: {vectorized_count}/{len(vector_chunks)} 个块")

        except Exception as e:
            if verbose:
                print(f"  向量化失败: {e}")
            vector_chunks = []

    if enable_storage and vector_chunks:
        if verbose:
            print("\n存储到向量数据库...")

        try:
            storage_service = get_vector_storage_service()
            index = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
            inserted = storage_service.store_vector_chunks(vector_chunks, index, parsed.file_name, user_level)

            if verbose:
                print(f"  完成: {inserted}/{len(vector_chunks)} 条")

        except Exception as e:
            if verbose:
                print(f"  存储失败: {e}")

    # ========== 新增：保存分块报告 ==========
    try:
        # 检查是否使用了远程解析
        use_remote = is_remote_parse_enabled()

        if use_remote:
            if verbose:
                print("\n保存分块报告...")

            # 获取远程解析器实例以获取原始 Markdown 内容
            remote_parser = RemotePDFParser()
            last_result = remote_parser.get_last_parse_result()

            # 保存带分块结果的报告
            report_path = save_chunked_report(
                file_name=parsed.file_name,
                chunks=vector_chunks if vector_chunks else chunks,
                sections=last_result.get("sections"),
                tables=last_result.get("tables"),
                markdown_content=last_result.get("markdown_content")
            )

            if verbose and report_path:
                print(f"  ✓ 报告已保存: {report_path}")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 保存分块报告失败: {e}")

    return vector_chunks if vector_chunks else chunks


def create_and_vectorize_chunks_with_level(chunks, model_type: str = None, user_level: str = "normal", verbose: bool = False) -> List[VectorChunk]:
    """创建 VectorChunk 并向量化，同时设置用户等级"""
    import hashlib

    vector_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{i}_{chunk.content[:100]}".encode()).hexdigest()[:16]
        vector_chunks.append(VectorChunk(
            id=f"chunk_{i}_{chunk_id}",
            content=chunk.content,
            metadata={
                **chunk.metadata,
                'chunk_index': i,
                'token_count': chunk.token_count,
                'user_level': user_level
            },
            token_count=chunk.token_count,
            chunk_index=i,
            user_level=user_level  # 设置文档等级
        ))

    embedding_service = get_embedding_service()
    if model_type:
        if model_type == 'local':
            embedding_service.switch_to_local()
        else:
            embedding_service.switch_to_remote()

    return embedding_service.vectorize_chunks(vector_chunks)


def parse_only(
    file_path: str,
    from_page: int = 0,
    to_page: int = None,
    enable_cleaning: bool = True,
    verbose: bool = False
) -> ParsedDocument:
    """仅执行 RAG1 流程：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗"""
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=enable_cleaning,
        verbose=verbose
    )
    all_text = parsed.cleaned_text or ""

    # 添加表格内容
    for page in parsed.pages:
        for table in page.tables:
            if table.data:
                table_text = _table_to_text(table.data)
                if table_text:
                    all_text += "\n\n" + table_text

    parsed.cleaned_text = all_text
    return parsed


def _table_to_text(table_data: List[List[str]]) -> str:
    """将表格数据转换为可检索的文本格式"""
    if not table_data or len(table_data) == 0:
        return ""

    lines = []

    # 方式1：Markdown 表格格式（适合阅读和检索）
    # 表头
    header = "| " + " | ".join(str(cell) if cell else "" for cell in table_data[0]) + " |"
    lines.append(header)
    # 分隔线
    separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
    lines.append(separator)
    # 数据行
    for row in table_data[1:]:
        line = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
        lines.append(line)

    return "\n".join(lines)

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