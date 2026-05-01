# app/service/core/rag/nlp/model.py

from openai import OpenAI
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
import numpy as np
from typing import List
import os
from dotenv import load_dotenv

# 导入新的向量化服务
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from service.embedding.embedding_service import get_embedding_service, EmbeddingService

load_dotenv()

# 创建全局 embedding 服务实例
_embedding_service = None


def get_embedding_service_instance():
    """获取 EmbeddingService 实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def generate_embedding(text: str | List[str], api_key: str = None, base_url: str = None,
                       model_name: str = "text-embedding-v3", dimensions: int = 1024,
                       encoding_format: str = "float", max_batch_size: int = 10):
    """
    生成文本的向量嵌入（兼容原有接口）

    Args:
        text: 单个文本或文本列表
        api_key: API密钥（可选）
        base_url: API基础URL（可选）
        model_name: 模型名称
        dimensions: 向量维度
        encoding_format: 编码格式
        max_batch_size: 最大批量大小

    Returns:
        单个文本时返回向量，文本列表时返回向量列表
    """
    service = get_embedding_service_instance()

    if isinstance(text, str):
        return service.generate_embedding(text)
    elif isinstance(text, list):
        return service.generate_embeddings(text)
    else:
        return None