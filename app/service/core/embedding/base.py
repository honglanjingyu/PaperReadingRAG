"""Embedding 模型基类"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEmbeddingModel(ABC):
    """向量化模型基类"""

    @abstractmethod
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量"""
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass