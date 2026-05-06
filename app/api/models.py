# app/api/models.py
"""
Pydantic模型定义
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str
    history: Optional[List[Dict[str, str]]] = None
    top_k: Optional[int] = None
    recall_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    enable_rerank: Optional[bool] = None
    enable_query_rewrite: Optional[bool] = None
    template_name: str = "detailed"


class GenerateRequest(BaseModel):
    """生成请求模型（已有检索结果）"""
    question: str
    results: List[Dict[str, Any]]
    history: Optional[List[Dict[str, str]]] = None
    template_name: str = "detailed"


__all__ = ['ChatRequest', 'GenerateRequest']