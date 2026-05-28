"""
批量 Upsert 优化器 - 使用 Milvus 批量 API
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from app.service.core.vector_store import get_vector_storage_service
from app.service.core.embedding import get_embedding_service

logger = logging.getLogger(__name__)


class BatchUpsertOptimizer:
    """
    批量 Upsert 优化器

    特性：
    1. 批量缓冲：收集多个文档后批量处理
    2. 并行向量化：批量调用 embedding API
    3. 分批写入：避免单次写入过大
    """

    def __init__(
            self,
            batch_size: int = 100,
            flush_interval: float = 5.0,
            max_buffer_size: int = 1000
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size

        self._buffer: Dict[str, Dict] = {}
        self._last_flush = datetime.now()
        self._embedding_service = get_embedding_service()
        self._storage_service = get_vector_storage_service()

        logger.info(f"BatchUpsertOptimizer 初始化: batch_size={batch_size}, "
                    f"flush_interval={flush_interval}s")

    def add_document(self, filename: str, content: str, metadata: Dict = None):
        """添加文档到缓冲区"""
        self._buffer[filename] = {
            "filename": filename,
            "content": content,
            "metadata": metadata or {},
            "added_at": datetime.now()
        }

        logger.debug(f"添加文档到缓冲区: {filename}, buffer_size={len(self._buffer)}")

        # 检查是否需要自动刷新
        if len(self._buffer) >= self.max_buffer_size:
            asyncio.create_task(self.flush())

    async def flush(self, index_name: str = None) -> int:
        """刷新缓冲区，批量处理所有文档"""
        if not self._buffer:
            return 0

        index_name = index_name or os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        documents = list(self._buffer.values())
        self._buffer.clear()

        # 批量向量化
        texts = [doc["content"] for doc in documents]
        vectors = await asyncio.to_thread(
            self._embedding_service.generate_embeddings,
            texts
        )

        # 构建向量文档
        from app.service.core.rag import ParentChildVectorChunk
        from app.service.core.chunking import ParentChildSplitter

        all_chunks = []
        splitter = ParentChildSplitter()

        for doc, vector in zip(documents, vectors):
            if vector is None:
                continue

            # 分块
            split_doc = splitter.split_document(
                text=doc["content"],
                metadata={"source": doc["filename"]},
                document_name=doc["filename"]
            )

            # 构建向量块
            for child in split_doc.all_children:
                parent = split_doc.get_parent_by_child_id(child.id)
                if parent:
                    vc = self._create_vector_chunk(child, parent, vector)
                    all_chunks.append(vc)

        # 批量存储
        if all_chunks:
            inserted = await asyncio.to_thread(
                self._storage_service.store_vector_chunks,
                all_chunks,
                index_name,
                f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_level="normal"
            )
            logger.info(f"批量 Upsert 完成: {inserted} 个块")
            return inserted

        return 0

    def _create_vector_chunk(self, child, parent, vector):
        """创建向量块"""
        from app.service.core.rag import ParentChildVectorChunk

        vc = ParentChildVectorChunk(child, parent)
        vc.vector = vector
        return vc


__all__ = ['BatchUpsertOptimizer']