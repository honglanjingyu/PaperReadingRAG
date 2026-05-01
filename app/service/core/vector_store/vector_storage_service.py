# app/service/core/vector_store/vector_storage_service.py

import xxhash
import datetime
from typing import List, Dict, Any, Optional
import logging
import os

from .es_vector_store import ESVectorStore

logger = logging.getLogger(__name__)


class VectorStorageService:
    """向量存储服务 - 负责将文档分块存入向量数据库"""

    def __init__(self, es_host: str = None, es_user: str = "elastic", es_password: str = "infini_rag_flow"):
        self.es_store = ESVectorStore(es_host, es_user, es_password)

    def store_vector_chunks(
            self,
            vector_chunks: List,  # VectorChunk 列表
            index_name: str,
            file_name: str,
            kb_id: str = None
    ) -> int:
        """存储 VectorChunk 对象到向量数据库"""
        if not vector_chunks:
            return 0

        # 过滤出有向量的块
        chunks_with_vector = [c for c in vector_chunks if c.vector]
        if not chunks_with_vector:
            logger.warning("分块中没有向量数据")
            return 0

        # 获取向量维度
        vector_dim = len(chunks_with_vector[0].vector)

        # 确保索引存在
        self.es_store.create_index(index_name, vector_dim)

        # 构建文档
        documents = []
        now = datetime.datetime.now()
        create_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        for chunk in chunks_with_vector:
            doc = {
                "id": chunk.id,
                "content": chunk.content,
                "content_with_weight": chunk.content,
                "kb_id": kb_id or index_name,
                "docnm": file_name,
                "docnm_kwd": file_name,
                "doc_id": xxhash.xxh64(file_name.encode("utf-8")).hexdigest(),
                "create_time": create_time_str,
                "create_timestamp_flt": now.timestamp(),
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                f"q_{vector_dim}_vec": chunk.vector
            }

            # 添加元数据
            if chunk.metadata:
                doc.update(chunk.metadata)

            documents.append(doc)

        # 批量插入
        inserted = self.es_store.insert(documents, index_name)
        logger.info(f"存储完成: {inserted}/{len(chunks_with_vector)} 条")
        return inserted

    def delete_by_file(self, index_name: str, file_name: str) -> int:
        """删除指定文件的所有文档"""
        return self.es_store.delete(index_name, {"docnm": file_name})

    def delete_index(self, index_name: str):
        """删除整个索引"""
        self.es_store.delete_index(index_name)


# 全局单例
_vector_storage_service = None


def get_vector_storage_service(es_host: str = None) -> VectorStorageService:
    """获取向量存储服务实例"""
    global _vector_storage_service
    if _vector_storage_service is None:
        _vector_storage_service = VectorStorageService(es_host)
    return _vector_storage_service