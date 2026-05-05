# app/service/core/llm/llm_manager.py

import os
import logging
from typing import Optional
from enum import Enum

from .base_llm import BaseLLM
from .remote_llm import RemoteLLM

logger = logging.getLogger(__name__)


class LLMType(Enum):
    REMOTE = "remote"


def get_llm_type() -> str:
    """获取 LLM 类型配置"""
    return os.getenv("LLM_TYPE", "remote").lower()


class LLMManager:
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
        self._active_model: Optional[BaseLLM] = None
        self._default_type = get_llm_type()
        self._init_default_model()

    def _init_default_model(self):
        if self._default_type == "remote":
            self._active_model = self.get_remote_model()

    def get_remote_model(self, **kwargs) -> RemoteLLM:
        if LLMType.REMOTE not in self._models:
            model_name = os.getenv("LLM_MODEL", "qwen-turbo")
            self._models[LLMType.REMOTE] = RemoteLLM(model_name=model_name, **kwargs)
        return self._models[LLMType.REMOTE]

    def get_active_model(self) -> BaseLLM:
        return self._active_model

    def set_active_model(self, model_type: str):
        if model_type == "remote":
            self._active_model = self.get_remote_model()
        logger.info(f"激活模型: {model_type}")

    def generate(self, messages: list, **kwargs) -> Optional[str]:
        return self.get_active_model().generate(messages, **kwargs)

    def generate_stream(self, messages: list, **kwargs):
        return self.get_active_model().generate_stream(messages, **kwargs)

    def get_model_info(self) -> dict:
        model = self.get_active_model()
        return {
            "type": self._default_type,
            "model_name": model.model_name,
        }


_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager