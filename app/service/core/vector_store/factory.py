# app/service/core/vector_store/factory.py
"""向量存储工厂 - 固定使用 Milvus"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """向量存储工厂类（单例）- 固定使用 Milvus"""

    _instance = None
    _store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 读取 Milvus 配置
        self._host = os.getenv("VECTOR_STORE_HOST", "localhost")
        self._port = os.getenv("VECTOR_STORE_PORT", "19530")
        self._user = os.getenv("VECTOR_STORE_USER", "")
        self._password = os.getenv("VECTOR_STORE_PASSWORD", "")

        logger.info(f"向量存储配置: type=milvus, host={self._host}, port={self._port}")

        self._init_store()

    def _init_store(self):
        """初始化 Milvus 向量存储"""
        try:
            from .milvus_vector_store import MilvusVectorStore

            logger.info(f"初始化 Milvus: {self._host}:{self._port}")

            self._store = MilvusVectorStore(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password
            )
            logger.info("Milvus 向量存储初始化成功")

        except Exception as e:
            logger.error(f"Milvus 初始化失败: {e}")
            raise

    def get_store(self):
        """获取向量存储实例"""
        if self._store is None:
            self._init_store()
        return self._store

    def get_store_type(self) -> str:
        """获取当前使用的存储类型"""
        return "milvus"


# 全局工厂实例
_vector_store_factory = None


def get_vector_store_factory() -> VectorStoreFactory:
    """获取向量存储工厂实例"""
    global _vector_store_factory
    if _vector_store_factory is None:
        _vector_store_factory = VectorStoreFactory()
    return _vector_store_factory


def get_vector_store():
    """获取当前向量存储实例"""
    return get_vector_store_factory().get_store()


def get_store_type() -> str:
    """获取当前存储类型"""
    return "milvus"


def switch_vector_store(store_type: str) -> bool:
    """切换向量存储类型 - 固定为 milvus，不支持切换"""
    if store_type.lower() != "milvus":
        logger.warning(f"不支持切换到 {store_type}，本系统固定使用 Milvus")
    return True