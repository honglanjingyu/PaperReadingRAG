# app/service/core/embedding/vectorization_service.py
"""
向量化服务 - 统一的向量化处理接口
"""

from typing import List, Optional, Dict, Any
from .embedding_manager import get_embedding_manager, EmbeddingType
from .vector_types import VectorChunk
import logging

logger = logging.getLogger(__name__)


class VectorizationService:
    """统一的向量化服务"""

    def __init__(self, model_type: str = None):
        """
        初始化向量化服务

        Args:
            model_type: 模型类型 ('remote' 或 'local')，默认从环境变量读取
        """
        self.manager = get_embedding_manager()

        if model_type:
            if model_type == 'remote':
                self.manager.switch_to_remote()
            elif model_type == 'local':
                self.manager.switch_to_local()

    def vectorize_chunks(self, chunks: List[VectorChunk]) -> List[VectorChunk]:
        """
        对分块进行向量化

        Args:
            chunks: VectorChunk 列表

        Returns:
            向量化后的 VectorChunk 列表（原地修改并返回）
        """
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        vectors = self.manager.generate_embeddings(texts)

        for chunk, vector in zip(chunks, vectors):
            if vector:
                chunk.vector = vector

        vectorized_count = len([c for c in chunks if c.vector])
        logger.info(f"向量化完成: {vectorized_count}/{len(chunks)} 个块")

        return chunks

    def vectorize_text(self, text: str, metadata: Dict = None) -> Optional[VectorChunk]:
        """向量化单个文本"""
        vector = self.manager.generate_embedding(text)
        if vector is None:
            return None

        from .vector_types import VectorChunk
        import hashlib

        chunk_id = hashlib.md5(text[:100].encode()).hexdigest()[:16]

        return VectorChunk(
            id=f"vec_{chunk_id}",
            content=text,
            vector=vector,
            metadata=metadata or {},
            token_count=self._count_tokens(text)
        )

    def _count_tokens(self, text: str) -> int:
        """估算 token 数量"""
        import re
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self.manager.dimension

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return self.manager.get_model_info()


# 便捷函数
def vectorize_chunks(chunks: List[VectorChunk], model_type: str = None) -> List[VectorChunk]:
    """快速向量化分块"""
    service = VectorizationService(model_type)
    return service.vectorize_chunks(chunks)


def vectorize_text(text: str, metadata: Dict = None, model_type: str = None) -> Optional[VectorChunk]:
    """快速向量化文本"""
    service = VectorizationService(model_type)
    return service.vectorize_text(text, metadata)


__all__ = ['VectorizationService', 'vectorize_chunks', 'vectorize_text']