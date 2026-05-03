# app/service/core/vector_store/factory.py
"""向量存储工厂 - 支持 ES 和 Milvus 动态切换"""

import os
import logging
from typing import Optional
from .base import BaseVectorStore, VectorStoreType

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """向量存储工厂类（单例）"""

    _instance = None
    _store = None
    _store_type = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 读取配置，默认使用 elasticsearch
        self._store_type = os.getenv("VECTOR_STORE_TYPE", "elasticsearch").lower()

        # 兼容旧的环境变量名
        if self._store_type == "es" or self._store_type == "elastic":
            self._store_type = "elasticsearch"
        elif self._store_type == "mv" or self._store_type == "milvus_db":
            self._store_type = "milvus"

        # 读取通用连接配置
        self._host = os.getenv("VECTOR_STORE_HOST", "localhost")
        self._port = os.getenv("VECTOR_STORE_PORT", "")
        self._user = os.getenv("VECTOR_STORE_USER", "")
        self._password = os.getenv("VECTOR_STORE_PASSWORD", "")

        logger.info(f"向量存储配置: type={self._store_type}, host={self._host}, port={self._port}")

        self._init_store()

    def _get_es_url(self) -> str:
        """构建 Elasticsearch URL"""
        host = self._host
        port = self._port or "9200"

        # 如果 host 已经包含协议，直接使用
        if host.startswith(('http://', 'https://')):
            return host

        # 否则组装 http://host:port
        return f"http://{host}:{port}"

    def _init_store(self):
        """根据配置初始化向量存储"""
        if self._store_type == VectorStoreType.MILVUS.value:
            self._init_milvus()
        else:
            # 默认使用 Elasticsearch
            self._init_elasticsearch()

    def _init_elasticsearch(self):
        """初始化 Elasticsearch"""
        try:
            from .es_vector_store import ESVectorStore

            # 构建 ES URL
            es_url = self._get_es_url()

            logger.info(f"初始化 Elasticsearch: {es_url}")

            self._store = ESVectorStore(
                es_host=es_url,
                es_user=self._user,
                es_password=self._password
            )
            self._store_type = VectorStoreType.ELASTICSEARCH.value
            logger.info(f"Elasticsearch 向量存储初始化成功")

        except Exception as e:
            logger.error(f"Elasticsearch 初始化失败: {e}")
            raise

    def _init_milvus(self):
        """初始化 Milvus"""
        try:
            from .milvus_vector_store import MilvusVectorStore

            host = self._host
            port = self._port or "19530"

            logger.info(f"初始化 Milvus: {host}:{port}")

            self._store = MilvusVectorStore(
                host=host,
                port=port,
                user=self._user,
                password=self._password
            )
            self._store_type = VectorStoreType.MILVUS.value
            logger.info(f"Milvus 向量存储初始化成功")

        except Exception as e:
            logger.error(f"Milvus 初始化失败: {e}")
            # Milvus 初始化失败，尝试回退到 Elasticsearch
            logger.warning("回退到 Elasticsearch...")
            self._init_elasticsearch()

    def get_store(self):
        """获取向量存储实例"""
        if self._store is None:
            self._init_store()
        return self._store

    def get_store_type(self) -> str:
        """获取当前使用的存储类型"""
        return self._store_type

    def switch_to(self, store_type: str) -> bool:
        """切换向量存储类型"""
        store_type = store_type.lower()
        if store_type == self._store_type:
            return True

        # 关闭当前连接
        if self._store:
            try:
                self._store.close()
            except Exception as e:
                logger.warning(f"关闭当前连接时出错: {e}")

        # 切换
        self._store_type = store_type
        self._store = None
        self._init_store()

        return self._store is not None


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
    return get_vector_store_factory().get_store_type()


def switch_vector_store(store_type: str) -> bool:
    """切换向量存储类型"""
    return get_vector_store_factory().switch_to(store_type)