"""统一的 Embedding 服务"""

import os
import hashlib
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseEmbeddingModel
from .cached import CachedEmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class VectorChunk:
    """带向量的文档分块"""
    id: str
    content: str
    vector: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0
    user_level: str = "normal"
    parent_id: str = ""
    parent_content: str = ""

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'content': self.content,
            'vector_dim': len(self.vector),
            'metadata': self.metadata,
            'token_count': self.token_count,
            'chunk_index': self.chunk_index,
            'user_level': self.user_level,
        }


def get_embedding_type() -> str:
    return os.getenv("EMBEDDING_TYPE", "remote").lower()


class EmbeddingService:
    """统一的 Embedding 服务（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model: Optional[BaseEmbeddingModel] = None
        self._init_model()

    def _init_model(self):
        """初始化模型（带缓存）"""
        model_type = get_embedding_type()
        use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

        if use_cache:
            self._model = CachedEmbeddingModel(model_type=model_type)
        else:
            from .remote import RemoteEmbeddingModel
            from .local import LocalEmbeddingModel
            if model_type == "local":
                self._model = LocalEmbeddingModel()
            else:
                self._model = RemoteEmbeddingModel()

        logger.info(f"EmbeddingService 初始化: type={model_type}, model={self._model.model_name}")

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        return self._model.generate_embedding(text) if text else None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        return self._model.generate_embeddings(texts) if texts else []

    def get_vector_field_name(self) -> str:
        return f"q_{self.dimension}_vec"

    def vectorize_chunks(self, chunks: List[VectorChunk]) -> List[VectorChunk]:
        """批量向量化分块"""
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        vectors = self.generate_embeddings(texts)

        for chunk, vec in zip(chunks, vectors):
            if vec:
                chunk.vector = vec

        logger.info(f"向量化完成: {len([c for c in chunks if c.vector])}/{len(chunks)}")
        return chunks

    def vectorize_chunk_list(self, chunks: List[dict], index_name: str,
                             file_name: str, user_level: str = "normal") -> List[dict]:
        """处理 RAG 项目的块格式"""
        import xxhash

        texts = [c.get("content_with_weight", "") for c in chunks]
        vectors = self.generate_embeddings(texts)
        vector_field = self.get_vector_field_name()

        processed = []
        for chunk, vec in zip(chunks, vectors):
            if vec is None:
                continue

            content = chunk.get("content_with_weight", "")
            chunk_id = xxhash.xxh64((content + index_name).encode()).hexdigest()

            doc = {
                "id": chunk_id,
                "content_with_weight": content,
                "create_timestamp_flt": datetime.now().timestamp(),
                "kb_id": index_name,
                "docnm": file_name,
                "doc_id": xxhash.xxh64(file_name.encode()).hexdigest(),
                vector_field: vec,
                "user_level": user_level,
            }
            processed.append(doc)

        logger.info(f"处理完成: {len(processed)} 个文档块")
        return processed

    def _count_tokens(self, text: str) -> int:
        import re
        if not text:
            return 0
        chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
        return int(chinese / 1.5 + (len(text) - chinese) / 4)

    def clear_cache(self):
        """清空 embedding 缓存"""
        if hasattr(self._model, 'clear_cache'):
            self._model.clear_cache()


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# 便捷函数
def generate_embedding(text: str) -> Optional[List[float]]:
    return get_embedding_service().generate_embedding(text)


def generate_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
    return get_embedding_service().generate_embeddings(texts)


def get_vector_field_name() -> str:
    return get_embedding_service().get_vector_field_name()


def vectorize_chunks(chunks: List[VectorChunk]) -> List[VectorChunk]:
    return get_embedding_service().vectorize_chunks(chunks)


__all__ = [
    'EmbeddingService', 'get_embedding_service', 'VectorChunk',
    'generate_embedding', 'generate_embeddings',
    'get_vector_field_name', 'vectorize_chunks'
]