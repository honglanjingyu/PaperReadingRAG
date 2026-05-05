# app/service/core/llm/remote_llm.py

import os
import logging
from typing import List, Dict, Any, Optional, Iterator
from openai import OpenAI

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


def get_llm_api_key() -> str:
    """获取 LLM API Key：优先获取专用 key，再获取通用 key"""
    return os.getenv("LLM_API_KEY") or os.getenv("MODEL_API_KEY")


def get_llm_base_url() -> str:
    """获取 LLM Base URL"""
    return os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")


def get_llm_model_name() -> str:
    """获取 LLM 模型名称"""
    return os.getenv("LLM_MODEL", "qwen-turbo")


class RemoteLLM(BaseLLM):
    """远程 LLM 实现"""

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model_name: str = None,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            top_p: float = 0.9
    ):
        # 优先级：参数 > 专用环境变量 > 通用环境变量 > 默认值
        self._api_key = api_key or get_llm_api_key()
        self._base_url = base_url or get_llm_base_url()
        self._model_name = model_name or get_llm_model_name()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

        if not self._api_key:
            raise ValueError("未配置 LLM API Key，请设置 LLM_API_KEY 或 MODEL_API_KEY")

        if not self._base_url:
            logger.warning("未配置 LLM Base URL，将使用默认值")
            self._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url
        )
        logger.info(f"远程 LLM 初始化完成: {self._model_name}, base_url={self._base_url}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self._temperature),
                max_tokens=kwargs.get('max_tokens', self._max_tokens),
                top_p=kwargs.get('top_p', self._top_p)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return None

    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self._temperature),
                max_tokens=kwargs.get('max_tokens', self._max_tokens),
                top_p=kwargs.get('top_p', self._top_p),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"生成失败: {e}"