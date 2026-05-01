# app/service/core/vector_store/base_vector_store.py
"""
向量存储基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def create_index(self, index_name: str, vector_dim: int):
        """创建索引"""
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
    def search(self, query_vector: List[float], index_name: str, top_k: int, **kwargs) -> List[Dict]:
        """向量搜索"""
        pass