# app/service/core/vector_store/base.py
"""向量存储基类 - 通用抽象接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum


class VectorStoreType(Enum):
    """向量存储类型"""
    ELASTICSEARCH = "elasticsearch"
    MILVUS = "milvus"


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def create_index(self, index_name: str, vector_dim: int, **kwargs) -> bool:
        """创建索引/集合"""
        pass

    @abstractmethod
    def insert(self, documents: List[Dict[str, Any]], index_name: str) -> int:
        """插入文档"""
        pass

    @abstractmethod
    def delete(self, index_name: str, condition: Dict[str, Any]) -> int:
        """删除文档"""
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> bool:
        """删除索引/集合"""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], index_name: str, top_k: int = 5,
               filter_condition: Optional[Dict] = None, similarity_threshold: float = 0.5) -> List[Dict]:
        """向量搜索"""
        pass

    @abstractmethod
    def index_exists(self, index_name: str) -> bool:
        """检查索引/集合是否存在"""
        pass

    @abstractmethod
    def get_document_count(self, index_name: str) -> int:
        """获取文档数量"""
        pass

    @abstractmethod
    def close(self):
        """关闭连接"""
        pass