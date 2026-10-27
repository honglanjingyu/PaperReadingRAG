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
        """初始化模型。

        EMBEDDING_PROVIDER=multimodal（默认）走中文 CLIP：文本塔与图像塔输出同一
        向量空间、同一维度，文本块与图片块写同一个 Milvus 集合的同一个 vector 字段，
        检索时一次 ANN 同时命中两者。
        设为 vista 则走 VISTA / bge-visualized-m3（1024 维，文本塔 bge-m3 +
        视觉塔 EVA02-CLIP-L-14），同样是同空间真一次 ANN，但模型跑在独立 conda
        环境 vista 的子进程里（原因见 vista.py 文件头）。
        设为 remote 则退回原来的远程文本 embedding API（不支持图片）。
        """
        provider = os.getenv("EMBEDDING_PROVIDER", "multimodal").strip().lower()
        use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

        if provider in ("multimodal", "clip", "chinese_clip", "chinese-clip"):
            from .chinese_clip import ChineseClipEmbeddingModel
            base_model = ChineseClipEmbeddingModel()
        elif provider in ("vista", "visualized_bge", "visualized-bge", "bge_visualized"):
            from .vista import VistaEmbeddingModel
            base_model = VistaEmbeddingModel()
        else:
            from .remote import RemoteEmbeddingModel
            base_model = RemoteEmbeddingModel()

        if use_cache:
            self._model = CachedEmbeddingModel(model=base_model)
        else:
            self._model = base_model

        logger.info(
            f"EmbeddingService 初始化: provider={provider} model={self._model.model_name} dim={self._model.dimension}"
        )

    @property
    def supports_image(self) -> bool:
        """当前模型是否支持图片编码（决定能否做真一次 ANN）"""
        return hasattr(self._model, "encode_image")

    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """把图片编码成与文本同一空间的向量；模型不支持时返回 None"""
        fn = getattr(self._model, "encode_image", None)
        if fn is None:
            logger.warning("当前 embedding 模型不支持图片编码，图片将只按 caption 文本入库")
            return None
        try:
            return fn(image_path)
        except Exception as e:
            logger.error(f"图片编码失败 {image_path}: {e}")
            return None

    def encode_images(self, image_paths: List[str]) -> List[Optional[List[float]]]:
        fn = getattr(self._model, "encode_images", None)
        if fn is None:
            return [None] * len(image_paths or [])
        try:
            return fn(image_paths)
        except Exception as e:
            logger.error(f"批量图片编码失败: {e}")
            return [None] * len(image_paths or [])

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