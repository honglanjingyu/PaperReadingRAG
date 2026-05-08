# app/api/services/document_service.py

from typing import List, Optional
import hashlib
import logging

from app.api.config import processing_status
from app.service.core.rag import process_document, get_processing_stats
from app.service.core.embedding import VectorChunk

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
            to_page: int
    ) -> List[VectorChunk]:
        """后台处理文档任务"""
        try:
            processing_status[process_id]["message"] = "正在解析文档..."
            processing_status[process_id]["progress"] = 20

            # 处理文档
            result = process_document(
                file_path=file_path,
                chunk_size=chunk_size,
                enable_vectorization=enable_vectorization,
                enable_storage=enable_storage,
                from_page=from_page,
                to_page=to_page,
                verbose=False
            )

            processing_status[process_id]["progress"] = 100
            processing_status[process_id]["status"] = "completed"
            processing_status[process_id]["message"] = "文档处理完成"
            processing_status[process_id]["result"] = {
                "chunks_count": len(result),
                "vectorized_count": len([c for c in result if hasattr(c, 'vector') and c.vector]) if result else 0
            }

            # 文档处理完成后，使相关缓存失效
            index_name = processing_status[process_id].get("index_name", "rag_documents")
            self.invalidate_cache_for_document(processing_status[process_id]["filename"], index_name)

            return result

        except Exception as e:
            processing_status[process_id]["status"] = "failed"
            processing_status[process_id]["message"] = f"处理失败: {str(e)}"
            processing_status[process_id]["error"] = str(e)
            return []

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

            # 3. 可选：失效 Embedding 缓存（如果需要）
            # cache_manager.delete_pattern("embedding")

            logger.info(f"文档 {file_name} 相关缓存已失效")
        except Exception as e:
            logger.warning(f"缓存失效失败: {e}")


__all__ = ['DocumentService']