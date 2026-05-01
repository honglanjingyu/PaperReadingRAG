"""
Embedding服务管理器
统一管理远程和本地Embedding模型的切换
"""

import os
from typing import Optional, Union, List
from enum import Enum
import logging

from .base_embedding import BaseEmbeddingModel
from .remote_embedding import RemoteEmbeddingModel
from .local_embedding import LocalEmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    """Embedding模型类型"""
    REMOTE = "remote"  # 远程API
    LOCAL = "local"  # 本地模型


class EmbeddingManager:
    """
    Embedding服务管理器
    支持多模型管理和动态切换
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

        # 从环境变量读取默认配置
        self._default_type = os.getenv("EMBEDDING_TYPE", "remote").lower()

        # 远程模型配置
        self._remote_config = {
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "base_url": os.getenv("DASHSCOPE_BASE_URL"),
            "model_name": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        }

        # 本地模型配置
        self._local_config = {
            "model_name": os.getenv("LOCAL_EMBEDDING_MODEL", "bge-small-zh"),
            "model_path": os.getenv("LOCAL_EMBEDDING_PATH"),
            "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        }

        logger.info(f"EmbeddingManager初始化完成，默认类型: {self._default_type}")

    def register_model(
            self,
            model_type: Union[str, EmbeddingType],
            model: BaseEmbeddingModel,
            set_active: bool = False
    ):
        """注册模型"""
        if isinstance(model_type, str):
            model_type = EmbeddingType(model_type.lower())

        self._models[model_type] = model

        if set_active:
            self.set_active_model(model_type)

        logger.info(f"模型已注册: {model_type.value}")

    def get_remote_model(self, **kwargs) -> RemoteEmbeddingModel:
        """获取或创建远程模型"""
        if EmbeddingType.REMOTE not in self._models:
            config = {**self._remote_config, **kwargs}
            model = RemoteEmbeddingModel(**config)
            self._models[EmbeddingType.REMOTE] = model
        return self._models[EmbeddingType.REMOTE]

    def get_local_model(self, **kwargs) -> LocalEmbeddingModel:
        """获取或创建本地模型"""
        if EmbeddingType.LOCAL not in self._models:
            config = {**self._local_config, **kwargs}
            model = LocalEmbeddingModel(**config)
            self._models[EmbeddingType.LOCAL] = model
        return self._models[EmbeddingType.LOCAL]

    def set_active_model(self, model_type: Union[str, EmbeddingType]):
        """设置当前激活的模型"""
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
        """获取当前激活的模型"""
        if self._active_model is None:
            # 根据默认类型初始化
            if self._default_type == "local":
                self.set_active_model(EmbeddingType.LOCAL)
            else:
                self.set_active_model(EmbeddingType.REMOTE)

        return self._active_model

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量（使用当前激活模型）"""
        return self.get_active_model().generate_embedding(text)

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量（使用当前激活模型）"""
        return self.get_active_model().generate_embeddings(texts)

    @property
    def active_model_type(self) -> str:
        return self._active_model_type.value if self._active_model_type else None

    @property
    def dimension(self) -> int:
        """获取当前模型的向量维度"""
        return self.get_active_model().dimension

    def get_vector_field_name(self) -> str:
        """获取向量字段名"""
        dim = self.dimension
        return f"q_{dim}_vec"

    def switch_to_remote(self, **kwargs):
        """切换到远程模型"""
        if EmbeddingType.REMOTE in self._models:
            # 更新配置
            if kwargs:
                new_model = RemoteEmbeddingModel(**{**self._remote_config, **kwargs})
                self._models[EmbeddingType.REMOTE] = new_model

        self.set_active_model(EmbeddingType.REMOTE)
        logger.info("已切换到远程Embedding模型")

    def switch_to_local(self, **kwargs):
        """切换到本地模型"""
        if EmbeddingType.LOCAL in self._models:
            # 更新配置
            if kwargs:
                new_model = LocalEmbeddingModel(**{**self._local_config, **kwargs})
                self._models[EmbeddingType.LOCAL] = new_model

        self.set_active_model(EmbeddingType.LOCAL)
        logger.info("已切换到本地Embedding模型")

    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        model = self.get_active_model()
        return {
            "type": self.active_model_type,
            "model_name": model.model_name,
            "dimension": model.dimension,
        }


# 全局单例
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """获取全局EmbeddingManager实例"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager