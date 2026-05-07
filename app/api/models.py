# app/api/models.py (更新版)
"""
Pydantic模型定义 - 添加会话记忆支持
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型 - 支持会话记忆"""
    question: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID，用于保持对话记忆")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史（传统方式）")
    top_k: Optional[int] = Field(None, description="返回结果数量")
    recall_k: Optional[int] = Field(None, description="召回数量")
    similarity_threshold: Optional[float] = Field(None, description="相似度阈值")
    enable_rerank: Optional[bool] = Field(None, description="是否启用重排序")
    enable_query_rewrite: Optional[bool] = Field(None, description="是否启用查询改写")
    template_name: str = Field("detailed", description="Prompt模板名称")
    enable_memory: bool = Field(True, description="是否启用短期记忆")


class GenerateRequest(BaseModel):
    """生成请求模型（已有检索结果）"""
    question: str = Field(..., description="用户问题")
    results: List[Dict[str, Any]] = Field(..., description="检索结果列表")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史")
    template_name: str = Field("detailed", description="Prompt模板名称")


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str
    turn_count: int
    message_count: int
    created_at: float
    last_accessed: float
    is_active: bool


__all__ = ['ChatRequest', 'GenerateRequest', 'SessionInfo']