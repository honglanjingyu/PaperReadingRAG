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
            logger.warning("分块列表为空")
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
            self.es_store.create_index(index_name, vector_dim)
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return 0

        # 获取当前时间戳（使用整数时间戳，避免日期格式问题）
        now = datetime.datetime.now()
        create_timestamp = int(now.timestamp())  # 使用整数时间戳
        create_time_str = now.strftime("%Y-%m-%d %H:%M:%S")  # 作为字符串存储，不作为 date 类型

        doc_id_base = xxhash.xxh64(file_name.encode("utf-8")).hexdigest()

        # 构建文档 - 移除 create_time 字段，只使用 create_timestamp_flt
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
                "create_timestamp_flt": create_timestamp,  # 只使用数字时间戳
                "token_count": getattr(chunk, 'token_count', 0),
                "chunk_index": getattr(chunk, 'chunk_index', i),
                f"q_{vector_dim}_vec": chunk.vector
            }

            # 添加元数据
            if hasattr(chunk, 'metadata') and chunk.metadata:
                for key, value in chunk.metadata.items():
                    if key not in doc:
                        doc[key] = value

            documents.append(doc)

        # 批量插入
        try:
            inserted = self.es_store.insert_bulk(documents, index_name)
            logger.info(f"存储完成: {inserted}/{len(chunks_with_vector)} 条")
            return inserted
        except Exception as e:
            logger.error(f"存储失败: {e}")
            return 0

    def delete_by_file(self, index_name: str, file_name: str) -> int:
        """删除指定文件的所有文档"""
        return self.es_store.delete(index_name, {"docnm": file_name})

    def delete_index(self, index_name: str):
        """删除整个索引"""
        self.es_store.delete_index(index_name)

    def get_document_count(self, index_name: str) -> int:
        """获取索引中的文档数量"""
        return self.es_store.get_document_count(index_name)

    def index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        return self.es_store.index_exists(index_name)


# 全局单例
_vector_storage_service = None


def get_vector_storage_service(es_host: str = None) -> VectorStorageService:
    """获取向量存储服务实例"""
    global _vector_storage_service
    if _vector_storage_service is None:
        _vector_storage_service = VectorStorageService(es_host)
    return _vector_storage_service


__all__ = ['VectorStorageService', 'get_vector_storage_service']