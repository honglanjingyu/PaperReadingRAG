"""RAG 模块 - 文档处理、搜索、生成"""

import os
import hashlib
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ========== 导入子模块 ==========
from app.service.core.deepdoc import DocumentParser
from app.service.core.chunking import ParentChildSplitter, ParentChildDocument, ParentChunk, ChildChunk
from app.service.core.embedding import get_embedding_service, VectorChunk
from app.service.core.vector_store import get_vector_storage_service
from app.service.core.prompt import PromptBuilder
from app.service.core.llm import get_llm_service
from app.service.core.retrieval import get_parent_child_retriever
from .cached_search import CachedSearchService
from .cached_search import CachedSearchService


# ========== 文档处理 ==========

class ParentChildVectorChunk:
    """父子向量块"""
    def __init__(self, child: ChildChunk, parent: ParentChunk):
        self.child = child
        self.parent = parent
        self.id = child.id
        self.content = child.content
        self.parent_content = parent.content
        self.parent_id = parent.id
        self.vector = None
        self.token_count = child.token_count
        self.docnm = (child.metadata or {}).get('source') or (parent.metadata or {}).get('source', '')
        self.metadata = {
            **child.metadata,
            'parent_id': parent.id,
            'parent_content_preview': parent.content[:200],
            'chunk_type': 'child',
            'parent_chunk_index': parent.chunk_index,
            'child_chunk_index': child.chunk_index
        }

    def to_dict(self, user_level: str = "normal") -> Dict:
        return {
            "id": self.id, "content": self.content, "content_with_weight": self.content,
            "parent_id": self.parent_id, "parent_content": self.parent_content,
            "vector": self.vector or [], "token_count": self.token_count,
            "user_level": user_level, "docnm": self.docnm, **self.metadata
        }


def process_document(
        file_path: str,
        chunk_size: int = 256,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        from_page: int = 0,
        to_page: int = None,
        index_name: str = None,
        verbose: bool = False,
        user_level: str = "normal"
) -> List:
    """处理文档：解析 -> 分块 -> 向量化 -> 存储"""
    # 1. 解析文档
    parser = DocumentParser()
    parsed = parser.parse(file_path, from_page=from_page, to_page=to_page or 100000,
                          enable_cleaning=True, verbose=verbose)
    if not parsed.cleaned_text:
        return []

    # 2. 父子分块
    parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", "500"))
    splitter = ParentChildSplitter(parent_chunk_size=parent_chunk_size,
                                    child_chunk_size=chunk_size)
    doc = splitter.split_document(
        text=parsed.cleaned_text,
        metadata={'source': parsed.file_name, 'document_name': parsed.file_name},
        document_id=hashlib.md5(file_path.encode()).hexdigest()[:16],
        document_name=parsed.file_name
    )

    if verbose:
        logger.info(f"分块完成: {len(doc.parent_chunks)} 父块, {len(doc.all_children)} 子块")

    # 3. 向量化
    vector_chunks = []
    if enable_vectorization and doc.all_children:
        embedding = get_embedding_service()
        vectors = embedding.generate_embeddings([c.content for c in doc.all_children])
        parent_map = {p.id: p for p in doc.parent_chunks}

        for child, vec in zip(doc.all_children, vectors):
            if vec and child.parent_id in parent_map:
                vc = ParentChildVectorChunk(child, parent_map[child.parent_id])
                vc.vector = vec
                vector_chunks.append(vc)

    # 4. 存储
    if enable_storage and vector_chunks:
        storage = get_vector_storage_service()
        index = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        storage.store_vector_chunks(vector_chunks, index, parsed.file_name, user_level=user_level)

    return vector_chunks


def get_processing_stats(file_path: str) -> dict:
    """获取处理统计"""
    parser = DocumentParser()
    parsed = parser.parse(file_path, enable_cleaning=True, verbose=False)
    splitter = ParentChildSplitter()
    doc = splitter.split_document(parsed.cleaned_text or "")
    stats = splitter.get_statistics(doc)
    return {
        'file_name': parsed.file_name, 'file_type': parsed.file_type,
        'total_pages': parsed.total_pages, 'text_length': len(parsed.cleaned_text or 0),
        'parent_count': stats['parent_count'], 'child_count': stats['child_count'],
        'avg_parent_tokens': stats['avg_parent_tokens'], 'avg_child_tokens': stats['avg_child_tokens']
    }


# ========== 搜索 ==========

def enhanced_search_with_hybrid_and_rerank(
        question: str,
        index_name: str = None,
        top_k: int = 5,
        recall_k: int = 10,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
        enable_rerank: bool = True,
        enable_query_rewrite: bool = True,
        similarity_threshold: float = 0.3,
        rerank_type: str = "auto",
        user_level: str = None,
        use_cache: bool = True,
        verbose: bool = False  # 添加 verbose 参数
) -> dict:
    """增强检索 - 混合检索 + 重排序"""
    index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
    actual_recall_k = max(recall_k, top_k * 2)
    retriever = get_parent_child_retriever()

    if use_cache:
        return CachedSearchService().search_with_cache(
            search_func=retriever.search_with_query_rewrite,
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=actual_recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
            enable_query_rewrite=enable_query_rewrite,
            rerank_type=rerank_type,
            user_level=user_level,
            verbose=verbose  # 传递 verbose
        )
    return retriever.search_with_query_rewrite(
        question=question, index_name=index_name, top_k=top_k, recall_k=actual_recall_k,
        keyword_weight=keyword_weight, vector_weight=vector_weight,
        similarity_threshold=similarity_threshold, enable_rerank=enable_rerank,
        rerank_type=rerank_type, enable_query_rewrite=enable_query_rewrite,
        user_level=user_level, verbose=verbose
    )


# ========== 生成 ==========

def generate_answer(
        question: str,
        results: List[Dict],
        history: List[Dict] = None,
        template_name: str = "detailed",
        preview_answer: bool = False,
        verbose: bool = False  # 添加 verbose 参数
) -> dict:
    """生成答案（非流式）"""
    builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = builder.build_messages(question=question, results=results,
                                       history=history, template_name=template_name)
    llm = get_llm_service()
    answer = llm.generate(messages, verbose=verbose)  # 传递 verbose
    return {"success": answer is not None, "question": question, "answer": answer,
            "model_info": llm.get_model_info()}


def generate_answer_stream(question: str, results: List[Dict],
                           history: List[Dict] = None, template_name: str = "detailed",
                           verbose: bool = False):  # 添加 verbose 参数
    """生成答案（流式）"""
    builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = builder.build_messages(question=question, results=results,
                                       history=history, template_name=template_name)
    yield from get_llm_service().generate_stream(messages)


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
):
    """兼容旧接口 - 直接调用 process_document"""
    return process_document(
        file_path=file_path,
        chunk_size=child_chunk_size,
        enable_vectorization=enable_vectorization,
        enable_storage=enable_storage,
        from_page=from_page,
        to_page=to_page,
        index_name=index_name,
        verbose=verbose,
        user_level=user_level
    ), None


def process_document_with_text(
        text_content: str,
        file_name: str,
        chunk_size: int = 256,
        enable_vectorization: bool = True,
        enable_storage: bool = True,
        index_name: str = None,
        user_level: str = "normal",
        verbose: bool = False
) -> List:
    """
    处理已提取文字的内容（用于图片/音频/视频的OCR/ASR结果）

    Returns:
        List: 向量块列表，处理失败返回空列表
    """
    if not text_content or not text_content.strip():
        logger.warning(f"文件 {file_name} 没有提取到文字内容")
        return []

    logger.info(f"开始处理文档: {file_name}, 文字长度={len(text_content)}, 用户等级={user_level}")

    # 1. 父子分块
    parent_chunk_size = int(os.getenv("PARENT_CHUNK_SIZE", "500"))
    splitter = ParentChildSplitter(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=chunk_size
    )

    doc = splitter.split_document(
        text=text_content,
        metadata={'source': file_name, 'document_name': file_name},
        document_id=hashlib.md5(file_name.encode()).hexdigest()[:16],
        document_name=file_name
    )

    if verbose:
        logger.info(f"分块完成: {len(doc.parent_chunks)} 父块, {len(doc.all_children)} 子块")

    if not doc.all_children:
        logger.warning(f"文件 {file_name} 分块后没有子块")
        return []

    # 2. 向量化
    vector_chunks = []
    if enable_vectorization and doc.all_children:
        embedding = get_embedding_service()
        vectors = embedding.generate_embeddings([c.content for c in doc.all_children])
        parent_map = {p.id: p for p in doc.parent_chunks}

        for child, vec in zip(doc.all_children, vectors):
            if vec and child.parent_id in parent_map:
                vc = ParentChildVectorChunk(child, parent_map[child.parent_id])
                vc.vector = vec
                vector_chunks.append(vc)

        logger.info(f"向量化完成: {len(vector_chunks)}/{len(doc.all_children)} 个块")

    if not vector_chunks:
        logger.warning(f"文件 {file_name} 向量化后没有有效块")
        return []

    # 3. 存储
    if enable_storage and vector_chunks:
        storage = get_vector_storage_service()
        index = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        inserted = storage.store_vector_chunks(vector_chunks, index, file_name, user_level=user_level)
        logger.info(f"文档 {file_name} 存储完成，插入 {inserted} 个向量块 (期望 {len(vector_chunks)})")

        # 验证存储是否成功
        if inserted == 0:
            logger.error(f"文档 {file_name} 存储失败，插入数量为0")
            return []

    logger.info(f"✅ 文档 {file_name} 处理完成，共 {len(vector_chunks)} 个向量块")
    return vector_chunks

__all__ = [
    'process_document',
    'get_processing_stats',
    'enhanced_search_with_hybrid_and_rerank',
    'generate_answer',
    'generate_answer_stream',
    'ParentChildVectorChunk',
    'process_document_parent_child',
]