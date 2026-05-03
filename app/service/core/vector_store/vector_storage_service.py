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
        self.es_store = self.store  # 保持向后兼容

    def store_vector_chunks(
            self,
            vector_chunks: List,
            index_name: str,
            file_name: str,
            kb_id: str = None
    ) -> int:
        """存储 VectorChunk 对象到向量数据库"""
        if not vector_chunks:
            logger.warning("分块列表为空")
            return 0

        if self.store is None:
            logger.error("向量存储未初始化")
            return 0

        # 过滤出有向量的块
        chunks_with_vector = []
        for c in vector_chunks:
            if hasattr(c, 'vector') and c.vector:
                chunks_with_vector.append(c)
            elif hasattr(c, 'vector') and not c.vector:
                logger.warning(f"块 {getattr(c, 'id', 'unknown')} 没有向量数据")

        if not chunks_with_vector:
            logger.warning("分块中没有向量数据")
            return 0

        # 获取向量维度
        vector_dim = len(chunks_with_vector[0].vector)
        logger.info(f"准备存储 {len(chunks_with_vector)} 个块，向量维度: {vector_dim}")

        # 确保索引存在
        try:
            self.store.create_index(index_name, vector_dim)
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return 0

        # 获取当前时间戳
        now = datetime.datetime.now()
        create_timestamp = now.timestamp()
        doc_id_base = xxhash.xxh64(file_name.encode("utf-8")).hexdigest()

        # 构建文档
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
                f"q_{vector_dim}_vec": chunk.vector  # ES 格式
            }

            # Milvus 也支持 vector 字段
            doc["vector"] = chunk.vector

            # 添加元数据
            if hasattr(chunk, 'metadata') and chunk.metadata:
                for key, value in chunk.metadata.items():
                    if key not in doc:
                        doc[key] = value

            documents.append(doc)

        # 批量插入
        try:
            inserted = self.store.insert(documents, index_name)
            logger.info(f"存储完成: {inserted}/{len(chunks_with_vector)} 条")
            return inserted
        except Exception as e:
            logger.error(f"存储失败: {e}")
            return 0

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


# 全局单例
_vector_storage_service = None


def get_vector_storage_service() -> VectorStorageService:
    """获取向量存储服务实例"""
    global _vector_storage_service
    if _vector_storage_service is None:
        _vector_storage_service = VectorStorageService()
    return _vector_storage_service


__all__ = ['VectorStorageService', 'get_vector_storage_service']