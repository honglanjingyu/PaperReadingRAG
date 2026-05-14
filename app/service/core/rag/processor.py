# app/service/core/rag/processor.py (修复版)

"""
RAG 文档处理器 - 只负责调用分块器
"""

import os
import hashlib
import logging
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# 导入 deepdoc 模块
from app.service.core.deepdoc import DocumentParser

# 导入父子分块模块
from app.service.core.chunking import (
    ParentChildDocument,
    ParentChunk,
    ChildChunk,
    chunk_document_parent_child,
    ParentChildChunker,
)

# 导入向量化模块
from app.service.core.embedding import (
    VectorChunk,
    get_embedding_service,
)

# 导入向量存储模块
from app.service.core.vector_store import get_vector_storage_service


class ParentChildVectorChunk:
    def __init__(self, child_chunk: ChildChunk, parent_chunk: ParentChunk):
        self.child = child_chunk
        self.parent = parent_chunk
        self.id = child_chunk.id
        self.content = child_chunk.content
        self.parent_content = parent_chunk.content
        self.parent_id = parent_chunk.id
        self.vector = None
        self.token_count = child_chunk.token_count
        self.metadata = {
            **child_chunk.metadata,
            'parent_id': parent_chunk.id,
            'parent_content_preview': parent_chunk.content[:200],
            'chunk_type': 'child',
            'parent_chunk_index': parent_chunk.chunk_index,
            'child_chunk_index': child_chunk.chunk_index
        }

        # ========== 关键修复：优先从 child_chunk.metadata 获取文档名 ==========
        self.docnm = ''
        if child_chunk.metadata:
            # 尝试多种可能的字段名
            for key in ['source', 'document_name', 'docnm', 'filename', 'name']:
                if key in child_chunk.metadata and child_chunk.metadata[key]:
                    self.docnm = child_chunk.metadata[key]
                    break

        # 如果还是没有，尝试从 parent_chunk.metadata 获取
        if not self.docnm and parent_chunk.metadata:
            for key in ['source', 'document_name', 'docnm', 'filename', 'name']:
                if key in parent_chunk.metadata and parent_chunk.metadata[key]:
                    self.docnm = parent_chunk.metadata[key]
                    break

        # 最终兜底：使用文件名（如果存在）
        if not self.docnm:
            self.docnm = child_chunk.metadata.get('source', '') or parent_chunk.metadata.get('source', '')

        # 调试日志
        if not self.docnm:
            logger.warning(
                f"ParentChildVectorChunk: 无法获取文档名，child_metadata={child_chunk.metadata}, parent_metadata={parent_chunk.metadata}")

    def to_dict(self, user_level: str = "normal") -> Dict:
        """转换为存储字典"""
        result = {
            "id": self.id,
            "content": self.content,
            "content_with_weight": self.content,
            "parent_id": self.parent_id,
            "parent_content": self.parent_content,
            "vector": self.vector if self.vector else [],
            "token_count": self.token_count,
            "user_level": user_level,
            "docnm": getattr(self, 'docnm', ''),  # 确保 docnm 字段存在
            **self.metadata
        }
        if not isinstance(result["vector"], list):
            result["vector"] = []
        return result


def process_document_parent_child(
        file_path: str,
        parent_chunk_size: int = 500,
        child_chunk_size: int = 150,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: int = None,
        index_name: str = None,
        verbose: bool = False,
        user_level: str = "normal"
) -> Tuple[Optional[ParentChildDocument], List[ParentChildVectorChunk]]:
    """完整的文档处理流程 - 使用父子分块"""

    # ========== 1. 解析文档 ==========
    if verbose:
        print(f"\n开始处理文档: {file_path}")
        print(f"用户等级: {user_level}")

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
            print("❌ 解析失败: 未能提取文本内容")
        return None, []

    if verbose:
        print(f"✅ 解析成功: 文本长度 {len(parsed.cleaned_text)} 字符")

    # ========== 2. 父子分块 ==========
    if verbose:
        print("\n执行父子分块...")

    chunker = ParentChildChunker(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        parent_overlap=50,
        child_overlap=20,
        min_child_size=30
    )

    parent_child_doc = chunker.chunk_document(
        text=parsed.cleaned_text,
        metadata={
            'source': parsed.file_name,  # ← 关键
            'document_name': parsed.file_name,  # ← 备用
            'docnm': parsed.file_name,  # ← 再加一个
            'file_type': parsed.file_type,
            'total_pages': parsed.total_pages,
            'user_level': user_level
        },
        document_id=hashlib.md5(file_path.encode()).hexdigest()[:16],
        document_name=parsed.file_name  # ← 确保传递
    )

    if verbose:
        stats = chunker.get_statistics(parent_child_doc)
        print(f"  父块数: {stats['parent_count']}")
        print(f"  子块数: {stats['child_count']}")
        print(f"  平均父块大小: {stats['avg_parent_tokens']:.0f} tokens")
        print(f"  平均子块大小: {stats['avg_child_tokens']:.0f} tokens")

        # 调试：打印父块和子块的ID
        if verbose and parent_child_doc.parent_chunks:
            print(f"  父块ID示例: {parent_child_doc.parent_chunks[0].id}")
        if verbose and parent_child_doc.all_children:
            print(f"  子块parent_id示例: {parent_child_doc.all_children[0].parent_id}")

    # ========== 3. 向量化子块 ==========
    vector_chunks = []

    if enable_vectorization and parent_child_doc.all_children:
        if verbose:
            print("\n向量化子块...")

        try:
            embedding_service = get_embedding_service()

            # 提取子块文本
            child_texts = [c.content for c in parent_child_doc.all_children]

            if verbose:
                print(f"  准备向量化 {len(child_texts)} 个子块")

            # 批量生成向量
            vectors = embedding_service.generate_embeddings(child_texts)

            if verbose:
                vectorized_count = sum(1 for v in vectors if v is not None)
                print(f"  向量生成完成: {vectorized_count}/{len(child_texts)}")

            # 创建父块映射 - 使用完整的父块ID
            parent_map = {p.id: p for p in parent_child_doc.parent_chunks}

            if verbose:
                print(f"  父块映射keys: {list(parent_map.keys())}")
                print(f"  子块parent_ids: {[c.parent_id for c in parent_child_doc.all_children]}")

            # 构建向量块
            for child, vector in zip(parent_child_doc.all_children, vectors):
                if vector is None:
                    if verbose:
                        print(f"    警告: 子块 {child.id[:16]} 向量化为空")
                    continue

                # 直接使用 child.parent_id 作为 key 查找
                parent = parent_map.get(child.parent_id)

                if verbose:
                    print(f"    查找: child.parent_id={child.parent_id}, found={parent is not None}")

                if parent:
                    vec_chunk = ParentChildVectorChunk(child, parent)
                    vec_chunk.vector = vector
                    vector_chunks.append(vec_chunk)
                    if verbose:
                        print(f"    已添加向量块: {child.id[:16]} -> parent {parent.id[:16]}")
                else:
                    if verbose:
                        print(f"    警告: 找不到父块 {child.parent_id[:32]}...")

            if verbose:
                print(f"  完成: {len(vector_chunks)}/{len(parent_child_doc.all_children)} 个子块已向量化")

        except Exception as e:
            if verbose:
                print(f"  向量化失败: {e}")
            import traceback
            traceback.print_exc()

    # ========== 4. 存储到向量数据库 ==========
    if enable_storage and vector_chunks:
        if verbose:
            print("\n存储到向量数据库...")

        try:
            storage_service = get_vector_storage_service()
            index = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")

            # 准备存储文档
            documents = []
            for vc in vector_chunks:
                doc = vc.to_dict(user_level)
                documents.append(doc)

            if verbose:
                print(f"  准备存储 {len(documents)} 个文档")

            if documents:
                store = storage_service.store
                if store:
                    # 确保索引存在
                    vector_dim = len(documents[0]["vector"]) if documents[0]["vector"] else 1024
                    store.create_index(index, vector_dim)

                    # 插入文档
                    inserted = store.insert(documents, index, user_level)
                    if verbose:
                        print(f"  完成: {inserted}/{len(documents)} 条")

                    # 验证存储成功
                    if inserted > 0:
                        doc_count = store.get_document_count(index)
                        if verbose:
                            print(f"  验证: 索引 {index} 现有 {doc_count} 条记录")

        except Exception as e:
            if verbose:
                print(f"  存储失败: {e}")
            import traceback
            traceback.print_exc()

    elif enable_storage and not vector_chunks:
        if verbose:
            print("\n⚠️ 跳过存储: 没有向量化的子块")

    return parent_child_doc, vector_chunks


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
        user_level: str = "normal"
) -> List:
    """
    兼容旧的 process_document 接口
    内部使用父子分块
    """
    parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", "500"))
    child_chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "256"))

    parent_child_doc, vector_chunks = process_document_parent_child(
        file_path=file_path,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        enable_vectorization=enable_vectorization,
        enable_storage=enable_storage,
        from_page=from_page,
        to_page=to_page,
        index_name=index_name,
        verbose=verbose,
        user_level=user_level
    )

    return vector_chunks


def get_processing_stats(file_path: str, from_page: int = 0, to_page: int = None) -> dict:
    """获取处理统计信息"""
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=False
    )

    # 估算父子分块统计
    chunker = ParentChildChunker()
    doc = chunker.chunk_document(parsed.cleaned_text)
    stats = chunker.get_statistics(doc)

    return {
        'file_name': parsed.file_name,
        'file_type': parsed.file_type,
        'total_pages': parsed.total_pages,
        'text_length': len(parsed.cleaned_text) if parsed.cleaned_text else 0,
        'parent_count': stats['parent_count'],
        'child_count': stats['child_count'],
        'avg_parent_tokens': stats['avg_parent_tokens'],
        'avg_child_tokens': stats['avg_child_tokens']
    }


__all__ = [
    'process_document',
    'process_document_parent_child',
    'get_processing_stats',
    'ParentChildVectorChunk',
    'ParentChildDocument'
]