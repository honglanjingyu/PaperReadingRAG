# app/service/core/vector_store/vector_storage_service.py
"""向量存储服务 - 统一接口"""

import xxhash
import datetime
from typing import List, Dict, Any, Optional
import logging
import os

from .factory import get_vector_store

logger = logging.getLogger(__name__)


class VectorStorageService:
    """向量存储服务 - 负责将文档分块存入向量数据库"""

    def __init__(self):
        self.store = get_vector_store()
        self.es_retriever = None

    def store_vector_chunks(
            self,
            vector_chunks: List,
            index_name: str,
            file_name: str,
            kb_id: str = None,
            user_level: str = "normal"
    ) -> int:
        """存储 VectorChunk 对象到向量数据库，同时同步到 ES BM25"""
        if not vector_chunks:
            logger.warning("分块列表为空")
            return 0

        if self.store is None:
            logger.error("向量存储未初始化")
            return 0

        # 收集有向量的分块
        chunks_with_vector = []
        for c in vector_chunks:
            if hasattr(c, 'vector') and c.vector:
                chunks_with_vector.append(c)

        if not chunks_with_vector:
            logger.warning("分块中没有向量数据")
            return 0

        vector_dim = len(chunks_with_vector[0].vector)
        logger.info(f"准备存储 {len(chunks_with_vector)} 个块，向量维度: {vector_dim}")

        # 确保索引存在
        try:
            self.store.create_index(index_name, vector_dim)
            logger.info(f"索引 {index_name} 已就绪")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return 0

        now = datetime.datetime.now()
        create_timestamp = now.timestamp()
        doc_id_base = xxhash.xxh64(file_name.encode("utf-8")).hexdigest()

        # 准备文档数据
        documents = []
        for i, chunk in enumerate(chunks_with_vector):
            doc = {
                "id": chunk.id,
                "content": chunk.content,
                "content_with_weight": chunk.content,
                "kb_id": kb_id or index_name,
                "docnm": file_name,
                "docnm_kwd": file_name,
                "doc_id": doc_id_base,
                "create_timestamp_flt": create_timestamp,
                "token_count": getattr(chunk, 'token_count', 0),
                "chunk_index": getattr(chunk, 'chunk_index', i),
                "user_level": getattr(chunk, 'user_level', user_level),
                "vector": chunk.vector
            }

            # 添加父子关系字段
            if hasattr(chunk, 'parent_id'):
                doc["parent_id"] = chunk.parent_id
            if hasattr(chunk, 'parent_content'):
                doc["parent_content"] = chunk.parent_content
            if hasattr(chunk, 'chunk_type'):
                doc["chunk_type"] = chunk.metadata.get('chunk_type', 'child')
            if hasattr(chunk, 'parent_chunk_index'):
                doc["parent_chunk_index"] = chunk.metadata.get('parent_chunk_index', 0)
            if hasattr(chunk, 'child_chunk_index'):
                doc["child_chunk_index"] = chunk.metadata.get('child_chunk_index', 0)

            # 添加其他元数据
            if hasattr(chunk, 'metadata') and chunk.metadata:
                for key, value in chunk.metadata.items():
                    if key not in doc:
                        doc[key] = value

            documents.append(doc)

        # ========== 初始化 ES 检索器 ==========
        from app.service.core.retrieval.es_bm25_retriever import get_es_bm25_retriever
        if self.es_retriever is None:
            self.es_retriever = get_es_bm25_retriever()
            logger.info("ES BM25 检索器已初始化")

        # ========== 存储到 Milvus ==========
        inserted = 0
        try:
            inserted = self.store.insert(documents, index_name, user_level)
            logger.info(f"Milvus 存储完成: {inserted}/{len(documents)} 条 (user_level={user_level})")

            # 验证存储
            if inserted > 0:
                doc_count = self.store.get_document_count(index_name)
                logger.info(f"验证: 索引 {index_name} 当前共有 {doc_count} 条记录")
        except Exception as e:
            logger.error(f"Milvus 存储失败: {e}")
            import traceback
            traceback.print_exc()

        # ========== 同步存储到 ES BM25 ==========
        if documents and self.es_retriever and self.es_retriever.is_available():
            try:
                # 准备 ES 文档格式
                es_documents = []
                for doc in documents:
                    es_doc = {
                        "id": doc.get("id"),
                        "content": doc.get("content", ""),
                        "content_with_weight": doc.get("content_with_weight", ""),
                        "document_name": file_name,
                        "docnm": file_name,
                        "user_level": doc.get("user_level", user_level),
                        "kb_id": kb_id or index_name,
                        "token_count": doc.get("token_count", 0),
                        "chunk_index": doc.get("chunk_index", 0),
                        "created_at": datetime.datetime.now().isoformat()
                    }
                    # 添加父子关系字段（如果存在）
                    if doc.get("parent_id"):
                        es_doc["parent_id"] = doc.get("parent_id")
                    if doc.get("parent_content"):
                        es_doc["parent_content"] = doc.get("parent_content")
                    es_documents.append(es_doc)

                # 存储到 ES
                if es_documents:
                    es_inserted = self.es_retriever.index_documents(
                        es_documents, index_name, user_level=user_level
                    )
                    logger.info(f"ES BM25 存储完成: {es_inserted}/{len(es_documents)} 条 -> {index_name}")

                    # 验证 ES 存储
                    es_count = self.es_retriever.get_document_count(index_name)
                    logger.info(f"ES BM25 验证: 索引 rag_bm25_{index_name} 当前共有 {es_count} 条记录")
                else:
                    logger.warning("ES BM25: 没有有效的文档需要存储")

            except Exception as e:
                logger.error(f"ES BM25 存储失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"ES BM25 不可用: es_retriever={self.es_retriever is not None}, "
                           f"available={self.es_retriever.is_available() if self.es_retriever else False}")

        return inserted

    def delete_by_file(self, index_name: str, file_name: str) -> int:
        """删除指定文件的所有文档"""
        if self.store is None:
            return 0
        return self.store.delete(index_name, {"docnm": file_name})

    def delete_index(self, index_name: str):
        """删除整个索引"""
        if self.store:
            self.store.delete_index(index_name)

    def get_document_count(self, index_name: str) -> int:
        """获取索引中的文档数量"""
        if self.store is None:
            return 0
        return self.store.get_document_count(index_name)

    def index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        if self.store is None:
            return False
        return self.store.index_exists(index_name)

    def batch_delete_by_files(self, index_name: str, filenames: List[str]) -> Dict[str, int]:
        """
        批量删除指定文件的所有文档

        Args:
            index_name: 索引名称
            filenames: 文件名列表

        Returns:
            Dict[str, int]: 每个文件删除的记录数
        """
        if self.store is None:
            logger.error("向量存储未初始化")
            return {filename: 0 for filename in filenames}

        return self.store.batch_delete_by_docnm(index_name, filenames)

# 全局单例
_vector_storage_service = None


def get_vector_storage_service() -> VectorStorageService:
    """获取向量存储服务实例"""
    global _vector_storage_service
    if _vector_storage_service is None:
        _vector_storage_service = VectorStorageService()
    return _vector_storage_service


__all__ = ['VectorStorageService', 'get_vector_storage_service']