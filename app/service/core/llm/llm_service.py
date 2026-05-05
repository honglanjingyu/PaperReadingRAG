# app/service/core/llm/llm_service.py
"""LLM 服务 - 统一接口"""

from typing import List, Dict, Optional, Iterator

from .llm_manager import get_llm_manager


class LLMService:
    """LLM 服务"""

    def __init__(self, model_type: str = None):
        self.manager = get_llm_manager()
        if model_type:
            self.manager.set_active_model(model_type)

    def generate(self, messages: List[Dict[str, str]], verbose: bool = False, **kwargs) -> Optional[str]:
        """生成回复"""
        if verbose:
            print(f"\n调用 LLM 生成回复...")
            print(f"消息数: {len(messages)}")
        return self.manager.generate(messages, **kwargs)

    def generate_stream(self, messages: List[Dict[str, str]], verbose: bool = False, **kwargs) -> Iterator[str]:
        """流式生成回复"""
        if verbose:
            print(f"\n调用 LLM 流式生成...")
        return self.manager.generate_stream(messages, **kwargs)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return self.manager.get_model_info()


def get_llm_service(model_type: str = None) -> LLMService:
    """获取 LLM 服务实例"""
    return LLMService(model_type)


__all__ = ['LLMService', 'get_llm_service']