# app/service/core/llm/cached_llm.py
"""
带缓存的 LLM 模型 - 缓存相同问题的响应
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Iterator

from .base_llm import BaseLLM
from .remote_llm import RemoteLLM
from ..cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedLLM(BaseLLM):
    """带缓存的 LLM 模型 - 包装原有 LLM"""

    def __init__(
            self,
            llm: BaseLLM = None,
            cache_ttl: int = None,
            similarity_threshold: float = None,
            **kwargs
    ):
        """
        初始化缓存 LLM

        Args:
            llm: 原始 LLM 模型（可选）
            cache_ttl: 缓存过期时间（秒），默认 1 小时
            similarity_threshold: 相似问题匹配阈值
        """
        self.cache_manager = get_cache_manager()
        self.cache_ttl = cache_ttl or int(os.getenv("CACHE_LLM_TTL", "3600"))  # 1小时
        self.similarity_threshold = similarity_threshold or float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))

        # 初始化原始模型
        if llm:
            self._llm = llm
        else:
            self._llm = RemoteLLM(**kwargs)

        self._model_name = self._llm.model_name

        logger.info(f"缓存 LLM 初始化: {self._model_name}, TTL={self.cache_ttl}s")

    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """生成缓存 key"""
        # 提取用户消息
        user_messages = []
        for msg in messages:
            if msg.get('role') == 'user':
                user_messages.append(msg.get('content', ''))

        # 使用最后一条用户消息作为 key
        last_user_msg = user_messages[-1] if user_messages else ""

        # 同时考虑系统指令
        system_instruction = ""
        for msg in messages:
            if msg.get('role') == 'system':
                system_instruction = msg.get('content', '')

        content_hash = hashlib.md5(f"{system_instruction}_{last_user_msg}".encode()).hexdigest()
        return f"llm:{self._model_name}:{content_hash}"

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """生成回复（带缓存）"""
        # 检查是否启用缓存
        enabled = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
        if not enabled:
            return self._llm.generate(messages, **kwargs)

        # 1. 精确查缓存
        cache_key = self._get_cache_key(messages)
        cached = self.cache_manager.get("llm", cache_key)

        if cached is not None:
            logger.info(f"LLM 缓存命中")
            return cached

        # 2. 调用 LLM
        response = self._llm.generate(messages, **kwargs)

        # 3. 缓存结果
        if response:
            self.cache_manager.set("llm", cache_key, response, self.cache_ttl)
            logger.info(f"LLM 缓存写入")

        return response

    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式生成（不缓存流式结果）"""
        # 流式生成不缓存
        yield from self._llm.generate_stream(messages, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    def invalidate_cache(self, messages: List[Dict[str, str]] = None):
        """使 LLM 缓存失效"""
        if messages:
            cache_key = self._get_cache_key(messages)
            self.cache_manager.delete("llm", cache_key)
        else:
            self.cache_manager.delete_pattern("llm")
        logger.info("LLM 缓存已失效")


__all__ = ['CachedLLM']