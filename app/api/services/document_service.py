# app/api/services/document_service.py
"""
文档处理服务
"""

from typing import List, Optional
import hashlib

from app.api.config import processing_status
from app.service.core.rag import process_document, get_processing_stats  # 修复导入路径
from app.service.core.embedding import VectorChunk  # 从 embedding 模块导入


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

            return result

        except Exception as e:
            processing_status[process_id]["status"] = "failed"
            processing_status[process_id]["message"] = f"处理失败: {str(e)}"
            processing_status[process_id]["error"] = str(e)
            return []

    def get_processing_stats(self, file_path: str) -> dict:
        """获取处理统计信息"""
        return get_processing_stats(file_path)