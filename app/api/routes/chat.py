# app/api/routes/chat.py
"""
智能问答路由
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json

from app.api.models import ChatRequest, GenerateRequest
from app.api.config import settings
from app.api.dependencies import get_chat_service

router = APIRouter()


@router.post("/chat/ask")
async def ask_question(request: ChatRequest) -> Dict[str, Any]:
    """
    问答接口 - 完整流程
    用户问题向量化 -> 相似度搜索 -> 增强搜索 -> 上下文构造 -> 推理生成
    """
    # 使用配置或请求中的值
    top_k = request.top_k or settings.rerank_top_k
    recall_k = request.recall_k or settings.similarity_top_k
    similarity_threshold = request.similarity_threshold or settings.similarity_threshold
    enable_rerank = request.enable_rerank if request.enable_rerank is not None else settings.enable_rerank
    enable_query_rewrite = request.enable_query_rewrite if request.enable_query_rewrite is not None else settings.enable_query_rewrite

    # 确保 recall_k >= top_k
    if recall_k < top_k:
        recall_k = top_k

    try:
        chat_service = get_chat_service()
        result = await chat_service.ask(
            question=request.question,
            history=request.history,
            top_k=top_k,
            recall_k=recall_k,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
            enable_query_rewrite=enable_query_rewrite,
            template_name=request.template_name,
            keyword_weight=settings.keyword_weight,
            vector_weight=settings.vector_weight,
            rerank_type=settings.rerank_type,
            index_name=settings.index_name
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


# app/api/routes/chat.py

@router.post("/chat/ask/stream")
async def ask_question_stream(request: ChatRequest):
    """流式问答接口 - 修复版"""
    top_k = request.top_k or settings.rerank_top_k
    recall_k = request.recall_k or settings.similarity_top_k

    if recall_k < top_k:
        recall_k = top_k

    async def generate():
        try:
            chat_service = get_chat_service()

            # 先发送开始标记
            yield json.dumps({"type": "start", "content": ""}) + "\n"

            # 流式生成答案
            async for chunk in chat_service.ask_stream(
                    question=request.question,
                    history=request.history,
                    top_k=top_k,
                    recall_k=recall_k,
                    template_name=request.template_name,
                    index_name=settings.index_name
            ):
                # 确保 chunk 是字符串且非空
                if chunk:
                    yield json.dumps({"type": "answer", "content": chunk}) + "\n"

            # 发送结束标记
            yield json.dumps({"type": "end", "content": ""}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat/search")
async def search_only(request: ChatRequest) -> Dict[str, Any]:
    """仅检索接口 - 只返回相关文档，不生成答案"""
    top_k = request.top_k or settings.rerank_top_k
    recall_k = request.recall_k or settings.similarity_top_k

    if recall_k < top_k:
        recall_k = top_k

    try:
        chat_service = get_chat_service()
        result = await chat_service.search_only(
            question=request.question,
            top_k=top_k,
            recall_k=recall_k,
            index_name=settings.index_name
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/chat/generate")
async def generate_only(request: GenerateRequest) -> Dict[str, Any]:
    """仅生成接口 - 基于已有检索结果生成答案"""
    try:
        chat_service = get_chat_service()
        result = await chat_service.generate_only(
            question=request.question,
            results=request.results,
            history=request.history,
            template_name=request.template_name
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")