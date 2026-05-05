# app/service/core/llm/base_llm.py
"""LLM 基类 - 定义统一的生成接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator


class BaseLLM(ABC):
    """大语言模型基类"""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """生成回复（非流式）"""
        pass

    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """生成回复（流式）"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """获取模型名称"""
        pass