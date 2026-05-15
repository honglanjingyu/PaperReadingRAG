# app/api/services/document_service.py

from typing import List, Optional
import hashlib
import logging
import os

from app.service.core.rag import process_document, get_processing_stats
from app.service.core.embedding import VectorChunk
from app.service.core.cache import get_document_cache

logger = logging.getLogger(__name__)


class DocumentService:
    """文档处理服务"""

    async def process_document_task(
            self,
            process_id: str,
            file_path: str,
            chunk_size: int,
            enable_vectorization: bool,
            enable_storage: bool,
            from_page: int,
            to_page: int,
            user_level: str = "normal"
    ) -> List[VectorChunk]:
        """后台处理文档任务（处理完成后更新缓存）"""
        try:
            logger.info(f"开始处理文档: {process_id}, user_level={user_level}")

            result = process_document(
                file_path=file_path,
                chunk_size=chunk_size,
                enable_vectorization=enable_vectorization,
                enable_storage=enable_storage,
                from_page=from_page,
                to_page=to_page,
                verbose=False,
                user_level=user_level
            )

            # 处理完成后，更新文档等级缓存
            filename = os.path.basename(file_path)
            doc_cache = get_document_cache()
            doc_cache.set_document_level(filename, user_level)
            logger.info(f"文档 {filename} 等级已缓存: {user_level}")

            logger.info(f"文档处理完成: {process_id}, 生成了 {len(result) if result else 0} 个分块")

            # 使相关缓存失效
            self.invalidate_cache_for_document(filename, "rag_documents")

            return result

        except Exception as e:
            logger.error(f"文档处理失败 {process_id}: {e}")
            raise

    def get_processing_stats(self, file_path: str) -> dict:
        """获取处理统计信息"""
        return get_processing_stats(file_path)

    def invalidate_cache_for_document(self, file_name: str, index_name: str):
        """文档更新时使相关缓存失效"""
        try:
            from app.service.core.cache import get_cache_manager

            cache_manager = get_cache_manager()

            # 1. 失效该文档相关的搜索缓存
            cache_manager.delete_pattern("search")

            # 2. 失效 BM25 索引缓存
            cache_manager.delete_pattern(f"bm25:{index_name}")

            logger.info(f"文档 {file_name} 相关缓存已失效")
        except Exception as e:
            logger.warning(f"缓存失效失败: {e}")


__all__ = ['DocumentService']