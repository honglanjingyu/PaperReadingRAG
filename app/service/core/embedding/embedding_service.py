# app/service/core/embedding/embedding_service.py
"""统一的 Embedding 服务 - 整合管理器和服务接口"""

import os
import hashlib
import logging
from typing import List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel
from .cached_embedding import CachedEmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    """Embedding 模型类型"""
    REMOTE = "remote"
    LOCAL = "local"


def get_embedding_type() -> str:
    """获取 Embedding 类型配置"""
    return os.getenv("EMBEDDING_TYPE", "remote").lower()


@dataclass
class VectorChunk:
    """带向量的分块数据结构（整合 vector_types.py）"""
    id: str
    content: str
    vector: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0
    user_level: str = "normal"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'content': self.content,
            'vector': self.vector[:10] if self.vector else [],
            'vector_dim': len(self.vector),
            'metadata': self.metadata,
            'token_count': self.token_count,
            'chunk_index': self.chunk_index,
            'user_level': self.user_level
        }

    def to_es_document(self, kb_id: str = None, doc_name: str = None) -> dict:
        doc = {
            "id": self.id,
            "content_with_weight": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "created_at": datetime.now().isoformat(),
            "user_level": self.user_level
        }
        if self.vector:
            doc[f"q_{len(self.vector)}_vec"] = self.vector
        if kb_id:
            doc["kb_id"] = kb_id
        if doc_name:
            doc["docnm_kwd"] = doc_name
        return doc


class EmbeddingService:
    """
    统一的 Embedding 服务
    整合了管理器、缓存、向量化功能
    """

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
        self._models: dict = {}
        self._active_model_type: Optional[EmbeddingType] = None
        self._active_model: Optional[BaseEmbeddingModel] = None
        self._default_type = get_embedding_type()

        # 初始化默认模型
        self._init_default_model()

        logger.info(f"EmbeddingService 初始化完成，默认类型: {self._default_type}")

    def _init_default_model(self):
        """初始化默认模型"""
        if self._default_type == "local":
            self.switch_to_local()
        else:
            self.switch_to_remote()

    def _get_cached_model(self, model_type: EmbeddingType, **kwargs) -> BaseEmbeddingModel:
        """获取带缓存的模型"""
        use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

        if use_cache:
            return CachedEmbeddingModel(model_type=model_type.value, **kwargs)
        else:
            if model_type == EmbeddingType.REMOTE:
                return RemoteEmbeddingModel(**kwargs)
            else:
                return LocalEmbeddingModel(**kwargs)

    def switch_to_remote(self, **kwargs):
        """切换到远程模型"""
        self._active_model = self._get_cached_model(EmbeddingType.REMOTE, **kwargs)
        self._active_model_type = EmbeddingType.REMOTE
        logger.info("已切换到远程 Embedding 模型")

    def switch_to_local(self, **kwargs):
        """切换到本地模型"""
        self._active_model = self._get_cached_model(EmbeddingType.LOCAL, **kwargs)
        self._active_model_type = EmbeddingType.LOCAL
        logger.info("已切换到本地 Embedding 模型")

    def get_active_model(self) -> BaseEmbeddingModel:
        """获取当前激活的模型"""
        if self._active_model is None:
            self._init_default_model()
        return self._active_model

    # ========== 核心向量化方法 ==========

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量"""
        if not text:
            return None
        return self.get_active_model().generate_embedding(text)

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量"""
        if not texts:
            return []
        return self.get_active_model().generate_embeddings(texts)

    def get_vector_field_name(self) -> str:
        """获取向量字段名"""
        dim = self.dimension
        return f"q_{dim}_vec"

    # ========== 向量化服务方法（整合 vectorization_service.py） ==========

    def vectorize_chunks(self, chunks: List[VectorChunk]) -> List[VectorChunk]:
        """
        对分块进行向量化（原地修改并返回）

        Args:
            chunks: VectorChunk 列表

        Returns:
            向量化后的 VectorChunk 列表
        """
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        vectors = self.generate_embeddings(texts)

        for chunk, vector in zip(chunks, vectors):
            if vector:
                chunk.vector = vector

        vectorized_count = len([c for c in chunks if c.vector])
        logger.info(f"向量化完成: {vectorized_count}/{len(chunks)} 个块")
        return chunks

    def vectorize_text(self, text: str, metadata: dict = None) -> Optional[VectorChunk]:
        """向量化单个文本"""
        vector = self.generate_embedding(text)
        if vector is None:
            return None

        chunk_id = hashlib.md5(text[:100].encode()).hexdigest()[:16]

        return VectorChunk(
            id=f"vec_{chunk_id}",
            content=text,
            vector=vector,
            metadata=metadata or {},
            token_count=self._count_tokens(text)
        )

    def vectorize_chunk_list(self, chunks: List[dict], index_name: str, file_name: str,
                             user_level: str = "normal") -> List[dict]:
        """
        处理 RAG 项目的块格式（整合 vector_processor.py 的功能）

        Args:
            chunks: 块列表
            index_name: 索引名称
            file_name: 文件名
            user_level: 用户等级

        Returns:
            处理后的文档列表
        """
        import xxhash

        texts = [chunk.get("content_with_weight", "") for chunk in chunks]
        vectors = self.generate_embeddings(texts)

        processed = []
        vector_field_name = self.get_vector_field_name()

        for chunk, vector in zip(chunks, vectors):
            if vector is None:
                continue

            content = chunk.get("content_with_weight", "")
            chunk_id = xxhash.xxh64((content + index_name).encode("utf-8")).hexdigest()

            doc = {
                "id": chunk_id,
                "content_ltks": chunk.get("content_ltks", ""),
                "content_with_weight": content,
                "content_sm_ltks": chunk.get("content_sm_ltks", ""),
                "important_kwd": [],
                "important_tks": [],
                "question_kwd": [],
                "question_tks": [],
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "create_timestamp_flt": datetime.now().timestamp(),
                "kb_id": index_name,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "title_tks": chunk.get("title_tks", ""),
                "doc_id": xxhash.xxh64(file_name.encode("utf-8")).hexdigest(),
                "docnm": file_name,
                vector_field_name: vector,
                "user_level": user_level
            }

            processed.append(doc)

        logger.info(f"处理完成: {len(processed)} 个文档块已向量化")
        return processed

    def _count_tokens(self, text: str) -> int:
        """估算 token 数量"""
        import re
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    # ========== 属性方法 ==========

    @property
    def dimension(self) -> int:
        """获取当前模型的向量维度"""
        return self.get_active_model().dimension

    @property
    def model_name(self) -> str:
        """获取当前模型的名称"""
        return self.get_active_model().model_name

    @property
    def active_model_type(self) -> str:
        """获取当前激活的模型类型"""
        return self._active_model_type.value if self._active_model_type else None

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "type": self.active_model_type,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

    # ========== 缓存管理 ==========

    def clear_embedding_cache(self):
        """清空 Embedding 缓存"""
        model = self.get_active_model()
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
        logger.info("已清空 Embedding 缓存")

    def invalidate_cache(self, text: str):
        """使指定文本的缓存失效"""
        model = self.get_active_model()
        if hasattr(model, 'invalidate_cache'):
            model.invalidate_cache(text)


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


# 便捷函数
def generate_embedding(text: str) -> Optional[List[float]]:
    """生成单个文本的向量"""
    return get_embedding_service().generate_embedding(text)


def generate_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
    """批量生成文本向量"""
    return get_embedding_service().generate_embeddings(texts)


def vectorize_chunks(chunks: List[VectorChunk]) -> List[VectorChunk]:
    """向量化分块"""
    return get_embedding_service().vectorize_chunks(chunks)


def get_vector_field_name() -> str:
    """获取向量字段名"""
    return get_embedding_service().get_vector_field_name()


__all__ = [
    'EmbeddingService', 'get_embedding_service', 'VectorChunk',
    'generate_embedding', 'generate_embeddings',
    'vectorize_chunks', 'get_vector_field_name'
]