# app/api/routes/health.py
"""
健康检查路由
"""

from fastapi import APIRouter
from typing import Dict, Any
import os

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查接口"""
    return {
        "status": "healthy",
        "message": "RAG系统运行正常"
    }


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """获取系统配置信息"""
    return {
        "embedding_type": os.getenv("EMBEDDING_TYPE", "remote"),
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "elasticsearch"),
        "index_name": os.getenv("VECTOR_INDEX_NAME", "rag_documents"),
        "llm_model": os.getenv("LLM_MODEL", "qwen-turbo"),
        "rerank_enabled": os.getenv("ENABLE_RERANK", "true").lower() == "true",
        "chunk_size": int(os.getenv("CHUNK_SIZE", "256"))
    }