# app/service/core/embedding/embedding_manager.py

import os
from typing import Optional, Union, List
from enum import Enum
import logging

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel
from .cached_embedding import CachedEmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    REMOTE = "remote"
    LOCAL = "local"


def get_embedding_type() -> str:
    """获取 Embedding 类型配置"""
    return os.getenv("EMBEDDING_TYPE", "remote").lower()


class EmbeddingManager:
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
        logger.info(f"EmbeddingManager初始化完成，默认类型: {self._default_type}")

    def register_model(self, model_type: Union[str, EmbeddingType], model: BaseEmbeddingModel, set_active: bool = False):
        if isinstance(model_type, str):
            model_type = EmbeddingType(model_type.lower())

        self._models[model_type] = model

        if set_active:
            self.set_active_model(model_type)

        logger.info(f"模型已注册: {model_type.value}")

    def get_remote_model(self, **kwargs) -> CachedEmbeddingModel:
        """获取远程 Embedding 模型（带缓存）"""
        if EmbeddingType.REMOTE not in self._models:
            use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

            if use_cache:
                model = CachedEmbeddingModel(model_type='remote', **kwargs)
            else:
                model = RemoteEmbeddingModel(**kwargs)
            self._models[EmbeddingType.REMOTE] = model
        return self._models[EmbeddingType.REMOTE]

    def get_local_model(self, **kwargs) -> CachedEmbeddingModel:
        """获取本地 Embedding 模型（带缓存）"""
        if EmbeddingType.LOCAL not in self._models:
            use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

            if use_cache:
                model = CachedEmbeddingModel(model_type='local', **kwargs)
            else:
                model = LocalEmbeddingModel(**kwargs)
            self._models[EmbeddingType.LOCAL] = model
        return self._models[EmbeddingType.LOCAL]

    def set_active_model(self, model_type: Union[str, EmbeddingType]):
        if isinstance(model_type, str):
            model_type = EmbeddingType(model_type.lower())

        if model_type == EmbeddingType.REMOTE:
            self._active_model = self.get_remote_model()
        elif model_type == EmbeddingType.LOCAL:
            self._active_model = self.get_local_model()
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        self._active_model_type = model_type
        logger.info(f"激活模型: {model_type.value}")

    def get_active_model(self) -> BaseEmbeddingModel:
        if self._active_model is None:
            if self._default_type == "local":
                self.set_active_model(EmbeddingType.LOCAL)
            else:
                self.set_active_model(EmbeddingType.REMOTE)

        return self._active_model

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        return self.get_active_model().generate_embedding(text)

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        return self.get_active_model().generate_embeddings(texts)

    @property
    def active_model_type(self) -> str:
        return self._active_model_type.value if self._active_model_type else None

    @property
    def dimension(self) -> int:
        return self.get_active_model().dimension

    def get_vector_field_name(self) -> str:
        dim = self.dimension
        return f"q_{dim}_vec"

    def switch_to_remote(self, **kwargs):
        """切换到远程模型"""
        if EmbeddingType.REMOTE in self._models:
            use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
            if use_cache:
                new_model = CachedEmbeddingModel(model_type='remote', **kwargs)
            else:
                new_model = RemoteEmbeddingModel(**kwargs)
            self._models[EmbeddingType.REMOTE] = new_model

        self.set_active_model(EmbeddingType.REMOTE)
        logger.info("已切换到远程Embedding模型")

    def switch_to_local(self, **kwargs):
        """切换到本地模型"""
        if EmbeddingType.LOCAL in self._models:
            use_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
            if use_cache:
                new_model = CachedEmbeddingModel(model_type='local', **kwargs)
            else:
                new_model = LocalEmbeddingModel(**kwargs)
            self._models[EmbeddingType.LOCAL] = new_model

        self.set_active_model(EmbeddingType.LOCAL)
        logger.info("已切换到本地Embedding模型")

    def get_model_info(self) -> dict:
        model = self.get_active_model()
        return {
            "type": self.active_model_type,
            "model_name": model.model_name,
            "dimension": model.dimension,
        }


_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


__all__ = ['EmbeddingManager', 'EmbeddingType', 'get_embedding_manager']